//! Zero-Copy Mach/NT Out-of-Line (OOL) IPC & Shared-Memory Ring Buffers.
//!
//! Provides:
//! - OOL page-descriptor remapping (`Move`, `VirtualCopy`/COW, `Share`, `PhysicalCopy`)
//! - Mach port rights transfer (`Receive`, `Send`, `SendOnce`, `DeadName`)
//! - NT ALPC section views (`AlpcSection`, `AlpcView`, `AlpcPort`)
//! - Lock-free cache-line-padded SPSC and Vyukov-style MPSC ring buffers
//! - `self_test()` for kernel boot verification

#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::fmt;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use nexus_abi::AbiError;

use crate::mm::{SpinLock, PAGE_SIZE};

/// Page size as `u64` for address arithmetic.
const PAGE_SIZE_U64: u64 = PAGE_SIZE as u64;

/// Magic number for shared-memory ring buffer headers (`"NXSR"`).
const SHM_RING_MAGIC: u32 = 0x4E58_5352;

/// Errors returned by the OOL IPC subsystem.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OolIpcError {
    InvalidDescriptor,
    InvalidPort,
    InvalidRight,
    PortDead,
    QueueFull,
    QueueEmpty,
    InvalidSection,
    InvalidView,
    AccessDenied,
    UnalignedAddress,
    SizeMismatch,
    OutOfMemory,
    AlreadyMapped,
    NotMapped,
    InvalidRingSize,
    MessageTooLarge,
    WouldBlock,
    InvalidState,
}

impl fmt::Display for OolIpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::InvalidDescriptor => "invalid OOL descriptor",
            Self::InvalidPort => "invalid Mach/ALPC port",
            Self::InvalidRight => "invalid Mach port right",
            Self::PortDead => "port is dead",
            Self::QueueFull => "IPC queue is full",
            Self::QueueEmpty => "IPC queue is empty",
            Self::InvalidSection => "invalid ALPC section",
            Self::InvalidView => "invalid ALPC section view",
            Self::AccessDenied => "access denied",
            Self::UnalignedAddress => "unaligned virtual or physical address",
            Self::SizeMismatch => "size or page count mismatch",
            Self::OutOfMemory => "out of memory",
            Self::AlreadyMapped => "virtual address range already mapped",
            Self::NotMapped => "virtual address not mapped",
            Self::InvalidRingSize => "ring buffer capacity must be a power of two >= 2",
            Self::MessageTooLarge => "message exceeds maximum size",
            Self::WouldBlock => "operation would block",
            Self::InvalidState => "invalid object state",
        };
        f.write_str(msg)
    }
}

impl From<OolIpcError> for AbiError {
    fn from(err: OolIpcError) -> Self {
        match err {
            OolIpcError::InvalidDescriptor
            | OolIpcError::UnalignedAddress
            | Self::SizeMismatch
            | Self::InvalidRingSize
            | Self::MessageTooLarge => AbiError::InvalidArg,
            OolIpcError::InvalidPort
            | OolIpcError::InvalidRight
            | Self::InvalidSection
            | Self::InvalidView => AbiError::InvalidHandle,
            OolIpcError::PortDead | Self::InvalidState => AbiError::BadState,
            OolIpcError::QueueFull | Self::QueueEmpty | Self::WouldBlock => AbiError::WouldBlock,
            OolIpcError::AccessDenied => AbiError::AccessDenied,
            OolIpcError::OutOfMemory => AbiError::NoMemory,
            OolIpcError::AlreadyMapped => AbiError::AlreadyExists,
            OolIpcError::NotMapped => AbiError::NotFound,
        }
    }
}

// ============================================================================
// 1. OOL Page-Descriptor Remapping
// ============================================================================

/// Copy/transfer semantics for an Out-of-Line (OOL) memory descriptor.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OolCopyOption {
    /// Eagerly copy physical pages into new frames for the receiver.
    PhysicalCopy,
    /// Zero-copy Copy-on-Write (COW) remapping; both sides share frames read-only until write.
    VirtualCopy,
    /// Zero-copy ownership transfer; pages are unmapped from sender and mapped into receiver.
    Move,
    /// Zero-copy shared mapping; both sender and receiver map the same physical frames.
    Share,
}

/// Protection flags for OOL and ALPC mapped pages.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OolProtection(u32);

impl OolProtection {
    pub const NONE: Self = Self(0);
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const EXECUTE: Self = Self(1 << 2);
    pub const COW: Self = Self(1 << 3);
    pub const READ_WRITE: Self = Self(Self::READ.0 | Self::WRITE.0);

    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

/// Descriptor for an Out-of-Line memory region attached to an IPC message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OolPageDescriptor {
    pub src_vaddr: u64,
    pub byte_len: usize,
    pub copy_option: OolCopyOption,
    pub prot: OolProtection,
    pub deallocate: bool,
    pub pages: Vec<u64>,
}

impl OolPageDescriptor {
    /// Create a new OOL page descriptor and validate alignment and page count.
    pub fn new(
        src_vaddr: u64,
        byte_len: usize,
        copy_option: OolCopyOption,
        prot: OolProtection,
        deallocate: bool,
        pages: Vec<u64>,
    ) -> Result<Self, OolIpcError> {
        if src_vaddr & (PAGE_SIZE_U64 - 1) != 0 {
            return Err(OolIpcError::UnalignedAddress);
        }
        if byte_len == 0 {
            return Err(OolIpcError::InvalidDescriptor);
        }
        let expected_pages = byte_len.div_ceil(PAGE_SIZE);
        if pages.len() != expected_pages {
            return Err(OolIpcError::SizeMismatch);
        }
        for &phys in &pages {
            if phys & (PAGE_SIZE_U64 - 1) != 0 {
                return Err(OolIpcError::UnalignedAddress);
            }
        }
        Ok(Self {
            src_vaddr,
            byte_len,
            copy_option,
            prot,
            deallocate,
            pages,
        })
    }

    #[inline]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
}

/// A single virtual-to-physical page mapping entry in an address space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MappedPage {
    pub phys_addr: u64,
    pub prot: OolProtection,
    pub cow: bool,
}

/// Tracks reference counts for physical frames shared across address spaces.
#[derive(Debug, Default)]
pub struct PhysFrameTracker {
    refs: BTreeMap<u64, u32>,
    next_synth_phys: u64,
}

impl PhysFrameTracker {
    pub const fn new() -> Self {
        Self {
            refs: BTreeMap::new(),
            next_synth_phys: 0x1000_0000,
        }
    }

    /// Increment reference count for `phys_addr`.
    pub fn retain(&mut self, phys_addr: u64) -> u32 {
        let count = self.refs.entry(phys_addr).or_insert(0);
        *count = count.saturating_add(1);
        *count
    }

    /// Decrement reference count for `phys_addr`, returning the new count.
    pub fn release(&mut self, phys_addr: u64) -> u32 {
        if let Some(count) = self.refs.get_mut(&phys_addr) {
            *count = count.saturating_sub(1);
            let remaining = *count;
            if remaining == 0 {
                self.refs.remove(&phys_addr);
            }
            remaining
        } else {
            0
        }
    }

    /// Return the current reference count for `phys_addr`.
    pub fn ref_count(&self, phys_addr: u64) -> u32 {
        self.refs.get(&phys_addr).copied().unwrap_or(0)
    }

    /// Allocate a synthetic physical frame address for copy-on-write or physical copy.
    pub fn alloc_synth_frame(&mut self) -> u64 {
        let phys = self.next_synth_phys;
        self.next_synth_phys = self.next_synth_phys.wrapping_add(PAGE_SIZE_U64);
        self.retain(phys);
        phys
    }
}

/// Per-task virtual address space map used for zero-copy OOL and ALPC remapping.
#[derive(Debug)]
pub struct AddressSpaceMap {
    pub asid: u64,
    mappings: BTreeMap<u64, MappedPage>,
    next_auto_vaddr: u64,
}

impl AddressSpaceMap {
    pub const fn new(asid: u64) -> Self {
        Self {
            asid,
            mappings: BTreeMap::new(),
            next_auto_vaddr: 0x0000_4000_0000_0000,
        }
    }

    /// Map a single page at `vaddr`.
    pub fn map_page(
        &mut self,
        vaddr: u64,
        phys_addr: u64,
        prot: OolProtection,
        cow: bool,
    ) -> Result<(), OolIpcError> {
        if (vaddr | phys_addr) & (PAGE_SIZE_U64 - 1) != 0 {
            return Err(OolIpcError::UnalignedAddress);
        }
        if self.mappings.contains_key(&vaddr) {
            return Err(OolIpcError::AlreadyMapped);
        }
        self.mappings.insert(
            vaddr,
            MappedPage {
                phys_addr,
                prot,
                cow,
            },
        );
        Ok(())
    }

    /// Unmap a single page at `vaddr`.
    pub fn unmap_page(&mut self, vaddr: u64) -> Result<MappedPage, OolIpcError> {
        if vaddr & (PAGE_SIZE_U64 - 1) != 0 {
            return Err(OolIpcError::UnalignedAddress);
        }
        self.mappings.remove(&vaddr).ok_or(OolIpcError::NotMapped)
    }

    /// Look up a mapped page at `vaddr`.
    pub fn lookup(&self, vaddr: u64) -> Option<&MappedPage> {
        self.mappings.get(&vaddr)
    }

    /// Look up a mutable mapped page at `vaddr`.
    pub fn lookup_mut(&mut self, vaddr: u64) -> Option<&mut MappedPage> {
        self.mappings.get_mut(&vaddr)
    }

    /// Find a contiguous unmapped virtual address range of `page_count` pages.
    pub fn alloc_va_range(&mut self, page_count: usize) -> Result<u64, OolIpcError> {
        if page_count == 0 {
            return Err(OolIpcError::InvalidDescriptor);
        }
        let mut candidate = self.next_auto_vaddr;
        let span = (page_count as u64)
            .checked_mul(PAGE_SIZE_U64)
            .ok_or(OolIpcError::OutOfMemory)?;

        loop {
            let mut conflict = false;
            for i in 0..page_count {
                let va = candidate + (i as u64) * PAGE_SIZE_U64;
                if self.mappings.contains_key(&va) {
                    conflict = true;
                    candidate = va + PAGE_SIZE_U64;
                    break;
                }
            }
            if !conflict {
                self.next_auto_vaddr = candidate.wrapping_add(span);
                return Ok(candidate);
            }
        }
    }

    /// Resolve a Copy-on-Write write fault at `vaddr`.
    pub fn resolve_cow_fault(
        &mut self,
        vaddr: u64,
        tracker: &mut PhysFrameTracker,
    ) -> Result<u64, OolIpcError> {
        let page = self
            .mappings
            .get_mut(&vaddr)
            .ok_or(OolIpcError::NotMapped)?;
        if !page.cow {
            return Err(OolIpcError::InvalidState);
        }
        let old_phys = page.phys_addr;
        if tracker.ref_count(old_phys) <= 1 {
            // Sole owner remaining: upgrade in place without copying.
            page.cow = false;
            page.prot = page
                .prot
                .without(OolProtection::COW)
                .union(OolProtection::WRITE);
            Ok(old_phys)
        } else {
            // Shared frame: allocate a private frame and drop reference to old frame.
            let new_phys = tracker.alloc_synth_frame();
            tracker.release(old_phys);
            page.phys_addr = new_phys;
            page.cow = false;
            page.prot = page
                .prot
                .without(OolProtection::COW)
                .union(OolProtection::WRITE);
            Ok(new_phys)
        }
    }
}

/// Result of remapping an OOL descriptor into a receiver address space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OolRemapResult {
    pub dst_vaddr: u64,
    pub page_count: usize,
    pub byte_len: usize,
    pub zero_copy: bool,
}

/// Remap an OOL page descriptor from `sender` into `receiver`.
pub fn remap_ool_descriptor(
    tracker: &mut PhysFrameTracker,
    sender: &mut AddressSpaceMap,
    receiver: &mut AddressSpaceMap,
    desc: &OolPageDescriptor,
    dst_vaddr_hint: Option<u64>,
) -> Result<OolRemapResult, OolIpcError> {
    let page_count = desc.page_count();
    if page_count == 0 {
        return Err(OolIpcError::InvalidDescriptor);
    }

    // Verify all source pages are mapped in sender and match descriptor frames.
    for (idx, &expected_phys) in desc.pages.iter().enumerate() {
        let src_va = desc.src_vaddr + (idx as u64) * PAGE_SIZE_U64;
        let mapped = sender.lookup(src_va).ok_or(OolIpcError::NotMapped)?;
        if mapped.phys_addr != expected_phys {
            return Err(OolIpcError::InvalidDescriptor);
        }
    }

    let dst_base = match dst_vaddr_hint {
        Some(va) => {
            if va & (PAGE_SIZE_U64 - 1) != 0 {
                return Err(OolIpcError::UnalignedAddress);
            }
            for idx in 0..page_count {
                if receiver.lookup(va + (idx as u64) * PAGE_SIZE_U64).is_some() {
                    return Err(OolIpcError::AlreadyMapped);
                }
            }
            va
        }
        None => receiver.alloc_va_range(page_count)?,
    };

    let mut zero_copy = true;

    for (idx, &phys) in desc.pages.iter().enumerate() {
        let src_va = desc.src_vaddr + (idx as u64) * PAGE_SIZE_U64;
        let dst_va = dst_base + (idx as u64) * PAGE_SIZE_U64;

        match desc.copy_option {
            OolCopyOption::Move => {
                let removed = sender.unmap_page(src_va)?;
                receiver.map_page(dst_va, removed.phys_addr, desc.prot, removed.cow)?;
            }
            OolCopyOption::VirtualCopy => {
                if desc.deallocate {
                    let removed = sender.unmap_page(src_va)?;
                    receiver.map_page(dst_va, removed.phys_addr, desc.prot, removed.cow)?;
                } else {
                    if let Some(src_page) = sender.lookup_mut(src_va) {
                        src_page.cow = true;
                        src_page.prot = src_page
                            .prot
                            .without(OolProtection::WRITE)
                            .union(OolProtection::COW);
                    }
                    tracker.retain(phys);
                    let rx_prot = desc
                        .prot
                        .without(OolProtection::WRITE)
                        .union(OolProtection::COW);
                    receiver.map_page(dst_va, phys, rx_prot, true)?;
                }
            }
            OolCopyOption::Share => {
                tracker.retain(phys);
                receiver.map_page(dst_va, phys, desc.prot, false)?;
                if desc.deallocate {
                    let _ = sender.unmap_page(src_va);
                    tracker.release(phys);
                }
            }
            OolCopyOption::PhysicalCopy => {
                zero_copy = false;
                let new_phys = tracker.alloc_synth_frame();
                receiver.map_page(dst_va, new_phys, desc.prot, false)?;
                if desc.deallocate {
                    let _ = sender.unmap_page(src_va);
                    tracker.release(phys);
                }
            }
        }
    }

    Ok(OolRemapResult {
        dst_vaddr: dst_base,
        page_count,
        byte_len: desc.byte_len,
        zero_copy,
    })
}

// ============================================================================
// 2. Mach Port Rights Transfer
// ============================================================================

/// Task-local Mach port name.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachPortName(pub u32);

impl MachPortName {
    pub const NULL: Self = Self(0);
    pub const DEAD: Self = Self(u32::MAX);
}

/// Kernel-global Mach port identifier.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachPortId(pub u64);

/// Mach port right held by a task.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MachPortRight {
    Receive,
    Send,
    SendOnce,
    PortSet,
    DeadName,
}

/// Disposition for transferring a Mach port right in a message or explicit transfer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MachMsgTypeName {
    MoveReceive,
    MoveSend,
    MoveSendOnce,
    CopySend,
    MakeSend,
    MakeSendOnce,
}

/// Port descriptor attached to a Mach OOL message.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MachPortDescriptor {
    pub name: MachPortName,
    pub disposition: MachMsgTypeName,
}

/// Mach message header.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MachMsgHeader {
    pub bits: u32,
    pub size: u32,
    pub remote_port: MachPortName,
    pub local_port: MachPortName,
    pub id: i32,
}

/// Mach message supporting inline payload, OOL page descriptors, and port right transfers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachOolMessage {
    pub header: MachMsgHeader,
    pub inline_data: Vec<u8>,
    pub ool_descriptors: Vec<OolPageDescriptor>,
    pub port_descriptors: Vec<MachPortDescriptor>,
}

#[derive(Copy, Clone, Debug)]
struct TaskPortEntry {
    port_id: MachPortId,
    right: MachPortRight,
    urefs: u32,
}

/// Per-task Mach port namespace.
#[derive(Debug)]
pub struct MachPortSpace {
    entries: BTreeMap<MachPortName, TaskPortEntry>,
    next_name: u32,
}

impl MachPortSpace {
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            next_name: 1,
        }
    }

    fn alloc_name(&mut self) -> MachPortName {
        loop {
            let candidate = MachPortName(self.next_name);
            self.next_name = self.next_name.wrapping_add(1).max(1);
            if candidate != MachPortName::DEAD && !self.entries.contains_key(&candidate) {
                return candidate;
            }
        }
    }

    fn find_by_port_and_right(
        &self,
        port_id: MachPortId,
        right: MachPortRight,
    ) -> Option<MachPortName> {
        self.entries
            .iter()
            .find(|(_, e)| e.port_id == port_id && e.right == right)
            .map(|(&name, _)| name)
    }
}

struct MachPortObject {
    receiver_task: Option<u64>,
    send_count: u32,
    send_once_count: u32,
    alive: bool,
    seqno: u64,
    queue: VecDeque<MachOolMessage>,
    max_queue_len: usize,
}

/// Kernel Mach IPC port & rights transfer manager.
pub struct MachIpcSubsystem {
    ports: BTreeMap<MachPortId, MachPortObject>,
    spaces: BTreeMap<u64, MachPortSpace>,
    next_port_id: u64,
}

impl MachIpcSubsystem {
    pub const fn new() -> Self {
        Self {
            ports: BTreeMap::new(),
            spaces: BTreeMap::new(),
            next_port_id: 1,
        }
    }

    fn space_mut(&mut self, task_id: u64) -> &mut MachPortSpace {
        self.spaces
            .entry(task_id)
            .or_insert_with(MachPortSpace::new)
    }

    /// Allocate a new Mach port with a `Receive` right in `task_id`.
    pub fn create_port(&mut self, task_id: u64, max_queue_len: usize) -> MachPortName {
        let port_id = MachPortId(self.next_port_id);
        self.next_port_id += 1;

        self.ports.insert(
            port_id,
            MachPortObject {
                receiver_task: Some(task_id),
                send_count: 0,
                send_once_count: 0,
                alive: true,
                seqno: 0,
                queue: VecDeque::new(),
                max_queue_len: max_queue_len.max(1),
            },
        );

        let space = self.space_mut(task_id);
        let name = space.alloc_name();
        space.entries.insert(
            name,
            TaskPortEntry {
                port_id,
                right: MachPortRight::Receive,
                urefs: 1,
            },
        );
        name
    }

    /// Query the right held by `task_id` for `name`.
    pub fn query_right(&self, task_id: u64, name: MachPortName) -> Option<MachPortRight> {
        self.spaces
            .get(&task_id)
            .and_then(|s| s.entries.get(&name))
            .map(|e| e.right)
    }

    /// Transfer or derive a Mach port right from `src_task` to `dst_task`.
    pub fn transfer_right(
        &mut self,
        src_task: u64,
        src_name: MachPortName,
        dst_task: u64,
        disposition: MachMsgTypeName,
    ) -> Result<MachPortName, OolIpcError> {
        if src_name == MachPortName::NULL || src_name == MachPortName::DEAD {
            return Err(OolIpcError::InvalidPort);
        }

        let src_entry = self
            .spaces
            .get(&src_task)
            .and_then(|s| s.entries.get(&src_name).copied())
            .ok_or(OolIpcError::InvalidPort)?;

        let port = self
            .ports
            .get_mut(&src_entry.port_id)
            .ok_or(OolIpcError::InvalidPort)?;
        if !port.alive {
            return Err(OolIpcError::PortDead);
        }

        let dst_right = match disposition {
            MachMsgTypeName::MoveReceive => {
                if src_entry.right != MachPortRight::Receive {
                    return Err(OolIpcError::InvalidRight);
                }
                port.receiver_task = Some(dst_task);
                MachPortRight::Receive
            }
            MachMsgTypeName::CopySend => {
                if src_entry.right != MachPortRight::Send {
                    return Err(OolIpcError::InvalidRight);
                }
                port.send_count = port.send_count.saturating_add(1);
                MachPortRight::Send
            }
            MachMsgTypeName::MakeSend => {
                if src_entry.right != MachPortRight::Receive {
                    return Err(OolIpcError::InvalidRight);
                }
                port.send_count = port.send_count.saturating_add(1);
                MachPortRight::Send
            }
            MachMsgTypeName::MoveSend => {
                if src_entry.right != MachPortRight::Send {
                    return Err(OolIpcError::InvalidRight);
                }
                MachPortRight::Send
            }
            MachMsgTypeName::MakeSendOnce => {
                if src_entry.right != MachPortRight::Receive {
                    return Err(OolIpcError::InvalidRight);
                }
                port.send_once_count = port.send_once_count.saturating_add(1);
                MachPortRight::SendOnce
            }
            MachMsgTypeName::MoveSendOnce => {
                if src_entry.right != MachPortRight::SendOnce {
                    return Err(OolIpcError::InvalidRight);
                }
                MachPortRight::SendOnce
            }
        };

        // Remove or decrement source right for Move* dispositions.
        match disposition {
            MachMsgTypeName::MoveReceive | MachMsgTypeName::MoveSendOnce => {
                if let Some(src_space) = self.spaces.get_mut(&src_task) {
                    src_space.entries.remove(&src_name);
                }
            }
            MachMsgTypeName::MoveSend => {
                if let Some(src_space) = self.spaces.get_mut(&src_task) {
                    if let Some(entry) = src_space.entries.get_mut(&src_name) {
                        if entry.urefs <= 1 {
                            src_space.entries.remove(&src_name);
                        } else {
                            entry.urefs -= 1;
                        }
                    }
                }
            }
            MachMsgTypeName::CopySend
            | MachMsgTypeName::MakeSend
            | MachMsgTypeName::MakeSendOnce => {}
        }

        // Insert or coalesce into destination task's port space.
        let dst_space = self.space_mut(dst_task);
        if dst_right == MachPortRight::Send {
            if let Some(existing) = dst_space.find_by_port_and_right(src_entry.port_id, dst_right) {
                if let Some(entry) = dst_space.entries.get_mut(&existing) {
                    entry.urefs = entry.urefs.saturating_add(1);
                }
                return Ok(existing);
            }
        }

        let dst_name = dst_space.alloc_name();
        dst_space.entries.insert(
            dst_name,
            TaskPortEntry {
                port_id: src_entry.port_id,
                right: dst_right,
                urefs: 1,
            },
        );
        Ok(dst_name)
    }

    /// Send a Mach OOL message from `src_task` to `msg.header.remote_port`.
    pub fn send_msg(&mut self, src_task: u64, mut msg: MachOolMessage) -> Result<(), OolIpcError> {
        let remote_name = msg.header.remote_port;
        let src_entry = self
            .spaces
            .get(&src_task)
            .and_then(|s| s.entries.get(&remote_name).copied())
            .ok_or(OolIpcError::InvalidPort)?;

        if src_entry.right != MachPortRight::Send && src_entry.right != MachPortRight::SendOnce {
            return Err(OolIpcError::InvalidRight);
        }

        let receiver_task = {
            let port = self
                .ports
                .get(&src_entry.port_id)
                .ok_or(OolIpcError::InvalidPort)?;
            if !port.alive {
                return Err(OolIpcError::PortDead);
            }
            if port.queue.len() >= port.max_queue_len {
                return Err(OolIpcError::QueueFull);
            }
            port.receiver_task.ok_or(OolIpcError::PortDead)?
        };

        // Transfer any attached port descriptors to the receiver task.
        let mut transferred_descs = Vec::with_capacity(msg.port_descriptors.len());
        for desc in &msg.port_descriptors {
            let new_name =
                self.transfer_right(src_task, desc.name, receiver_task, desc.disposition)?;
            transferred_descs.push(MachPortDescriptor {
                name: new_name,
                disposition: desc.disposition,
            });
        }
        msg.port_descriptors = transferred_descs;

        // Consume SendOnce right if used.
        if src_entry.right == MachPortRight::SendOnce {
            if let Some(src_space) = self.spaces.get_mut(&src_task) {
                src_space.entries.remove(&remote_name);
            }
            if let Some(port) = self.ports.get_mut(&src_entry.port_id) {
                port.send_once_count = port.send_once_count.saturating_sub(1);
            }
        }

        let port = self
            .ports
            .get_mut(&src_entry.port_id)
            .ok_or(OolIpcError::InvalidPort)?;
        port.seqno += 1;
        port.queue.push_back(msg);
        Ok(())
    }

    /// Receive the next Mach OOL message on `local_name` in `dst_task`.
    pub fn recv_msg(
        &mut self,
        dst_task: u64,
        local_name: MachPortName,
    ) -> Result<MachOolMessage, OolIpcError> {
        let entry = self
            .spaces
            .get(&dst_task)
            .and_then(|s| s.entries.get(&local_name).copied())
            .ok_or(OolIpcError::InvalidPort)?;

        if entry.right != MachPortRight::Receive {
            return Err(OolIpcError::InvalidRight);
        }

        let port = self
            .ports
            .get_mut(&entry.port_id)
            .ok_or(OolIpcError::InvalidPort)?;
        if !port.alive {
            return Err(OolIpcError::PortDead);
        }
        port.queue.pop_front().ok_or(OolIpcError::QueueEmpty)
    }

    /// Destroy a port right in `task_id`. Destroying a `Receive` right marks the port dead
    /// and converts outstanding `Send`/`SendOnce` rights into `DeadName`.
    pub fn destroy_port(&mut self, task_id: u64, name: MachPortName) -> Result<(), OolIpcError> {
        let entry = self
            .spaces
            .get_mut(&task_id)
            .and_then(|s| s.entries.remove(&name))
            .ok_or(OolIpcError::InvalidPort)?;

        if entry.right == MachPortRight::Receive {
            if let Some(port) = self.ports.get_mut(&entry.port_id) {
                port.alive = false;
                port.receiver_task = None;
                port.queue.clear();
            }
            for space in self.spaces.values_mut() {
                for e in space.entries.values_mut() {
                    if e.port_id == entry.port_id
                        && (e.right == MachPortRight::Send || e.right == MachPortRight::SendOnce)
                    {
                        e.right = MachPortRight::DeadName;
                    }
                }
            }
        } else if entry.right == MachPortRight::Send {
            if let Some(port) = self.ports.get_mut(&entry.port_id) {
                port.send_count = port.send_count.saturating_sub(1);
            }
        } else if entry.right == MachPortRight::SendOnce {
            if let Some(port) = self.ports.get_mut(&entry.port_id) {
                port.send_once_count = port.send_once_count.saturating_sub(1);
            }
        }
        Ok(())
    }
}

// ============================================================================
// 3. NT ALPC Section Views
// ============================================================================

/// NT ALPC shared memory section backed by physical page frames.
#[derive(Clone, Debug)]
pub struct AlpcSection {
    pub section_id: u64,
    pub owner_pid: u64,
    pub size: usize,
    pub pages: Vec<u64>,
    pub active_views: usize,
}

/// Mapped view of an ALPC section in a process's address space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AlpcView {
    pub view_id: u64,
    pub section_id: u64,
    pub pid: u64,
    pub base_vaddr: u64,
    pub size: usize,
    pub prot: OolProtection,
    pub auto_release: bool,
}

/// ALPC data-view message attribute (`ALPC_DATA_VIEW_ATTR`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AlpcDataViewAttr {
    pub view_id: u64,
    pub section_id: u64,
    pub view_base: u64,
    pub view_size: usize,
}

/// ALPC message carrying optional inline payload and a zero-copy section view attribute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlpcMessage {
    pub message_id: u64,
    pub sender_pid: u64,
    pub inline_payload: Vec<u8>,
    pub data_view: Option<AlpcDataViewAttr>,
}

struct AlpcPort {
    server_pid: u64,
    client_pid: Option<u64>,
    connected: bool,
    max_message_len: usize,
    queue: VecDeque<AlpcMessage>,
}

/// Kernel NT ALPC section-view and port manager.
pub struct AlpcSubsystem {
    sections: BTreeMap<u64, AlpcSection>,
    views: BTreeMap<u64, AlpcView>,
    ports: BTreeMap<u64, AlpcPort>,
    next_section_id: u64,
    next_view_id: u64,
    next_port_id: u64,
    next_msg_id: u64,
}

impl AlpcSubsystem {
    pub const fn new() -> Self {
        Self {
            sections: BTreeMap::new(),
            views: BTreeMap::new(),
            ports: BTreeMap::new(),
            next_section_id: 1,
            next_view_id: 1,
            next_port_id: 1,
            next_msg_id: 1,
        }
    }

    /// Create an ALPC section backed by `pages`.
    pub fn create_section(
        &mut self,
        owner_pid: u64,
        size: usize,
        pages: Vec<u64>,
    ) -> Result<u64, OolIpcError> {
        if size == 0 || size & (PAGE_SIZE - 1) != 0 {
            return Err(OolIpcError::SizeMismatch);
        }
        if pages.len() != size / PAGE_SIZE {
            return Err(OolIpcError::SizeMismatch);
        }
        for &phys in &pages {
            if phys & (PAGE_SIZE_U64 - 1) != 0 {
                return Err(OolIpcError::UnalignedAddress);
            }
        }
        let section_id = self.next_section_id;
        self.next_section_id += 1;
        self.sections.insert(
            section_id,
            AlpcSection {
                section_id,
                owner_pid,
                size,
                pages,
                active_views: 0,
            },
        );
        Ok(section_id)
    }

    /// Map a view of `section_id` into `as_map` for process `pid`.
    #[allow(clippy::too_many_arguments)]
    pub fn map_section_view(
        &mut self,
        section_id: u64,
        pid: u64,
        vaddr_hint: Option<u64>,
        size: usize,
        prot: OolProtection,
        auto_release: bool,
        tracker: &mut PhysFrameTracker,
        as_map: &mut AddressSpaceMap,
    ) -> Result<AlpcView, OolIpcError> {
        let section = self
            .sections
            .get_mut(&section_id)
            .ok_or(OolIpcError::InvalidSection)?;
        if size == 0 || size > section.size || size & (PAGE_SIZE - 1) != 0 {
            return Err(OolIpcError::SizeMismatch);
        }
        let page_count = size / PAGE_SIZE;
        let base_vaddr = match vaddr_hint {
            Some(va) => {
                if va & (PAGE_SIZE_U64 - 1) != 0 {
                    return Err(OolIpcError::UnalignedAddress);
                }
                for i in 0..page_count {
                    if as_map.lookup(va + (i as u64) * PAGE_SIZE_U64).is_some() {
                        return Err(OolIpcError::AlreadyMapped);
                    }
                }
                va
            }
            None => as_map.alloc_va_range(page_count)?,
        };

        for (i, &phys) in section.pages.iter().take(page_count).enumerate() {
            let va = base_vaddr + (i as u64) * PAGE_SIZE_U64;
            tracker.retain(phys);
            as_map.map_page(va, phys, prot, false)?;
        }

        section.active_views += 1;
        let view_id = self.next_view_id;
        self.next_view_id += 1;
        let view = AlpcView {
            view_id,
            section_id,
            pid,
            base_vaddr,
            size,
            prot,
            auto_release,
        };
        self.views.insert(view_id, view);
        Ok(view)
    }

    /// Unmap an ALPC section view from `as_map`.
    pub fn unmap_section_view(
        &mut self,
        view_id: u64,
        tracker: &mut PhysFrameTracker,
        as_map: &mut AddressSpaceMap,
    ) -> Result<(), OolIpcError> {
        let view = self
            .views
            .remove(&view_id)
            .ok_or(OolIpcError::InvalidView)?;
        let page_count = view.size / PAGE_SIZE;
        for i in 0..page_count {
            let va = view.base_vaddr + (i as u64) * PAGE_SIZE_U64;
            if let Ok(mapped) = as_map.unmap_page(va) {
                tracker.release(mapped.phys_addr);
            }
        }
        if let Some(section) = self.sections.get_mut(&view.section_id) {
            section.active_views = section.active_views.saturating_sub(1);
        }
        Ok(())
    }

    /// Create an ALPC connection port owned by `server_pid`.
    pub fn create_port(&mut self, server_pid: u64, max_message_len: usize) -> u64 {
        let port_id = self.next_port_id;
        self.next_port_id += 1;
        self.ports.insert(
            port_id,
            AlpcPort {
                server_pid,
                client_pid: None,
                connected: false,
                max_message_len: max_message_len.max(64),
                queue: VecDeque::new(),
            },
        );
        port_id
    }

    /// Connect `client_pid` to an ALPC port.
    pub fn connect_port(&mut self, port_id: u64, client_pid: u64) -> Result<(), OolIpcError> {
        let port = self
            .ports
            .get_mut(&port_id)
            .ok_or(OolIpcError::InvalidPort)?;
        port.client_pid = Some(client_pid);
        port.connected = true;
        Ok(())
    }

    /// Send an ALPC message with an optional zero-copy section view attribute.
    pub fn send_with_view(
        &mut self,
        port_id: u64,
        sender_pid: u64,
        inline_payload: Vec<u8>,
        data_view: Option<AlpcDataViewAttr>,
    ) -> Result<u64, OolIpcError> {
        let port = self
            .ports
            .get_mut(&port_id)
            .ok_or(OolIpcError::InvalidPort)?;
        if !port.connected {
            return Err(OolIpcError::InvalidState);
        }
        if sender_pid != port.server_pid && Some(sender_pid) != port.client_pid {
            return Err(OolIpcError::AccessDenied);
        }
        if inline_payload.len() > port.max_message_len {
            return Err(OolIpcError::MessageTooLarge);
        }
        if let Some(attr) = data_view {
            let view = self
                .views
                .get(&attr.view_id)
                .ok_or(OolIpcError::InvalidView)?;
            if view.section_id != attr.section_id || view.pid != sender_pid {
                return Err(OolIpcError::AccessDenied);
            }
        }

        let message_id = self.next_msg_id;
        self.next_msg_id += 1;
        port.queue.push_back(AlpcMessage {
            message_id,
            sender_pid,
            inline_payload,
            data_view,
        });
        Ok(message_id)
    }

    /// Receive the next ALPC message on `port_id`.
    pub fn recv_with_view(
        &mut self,
        port_id: u64,
        receiver_pid: u64,
    ) -> Result<AlpcMessage, OolIpcError> {
        let port = self
            .ports
            .get_mut(&port_id)
            .ok_or(OolIpcError::InvalidPort)?;
        if receiver_pid != port.server_pid && Some(receiver_pid) != port.client_pid {
            return Err(OolIpcError::AccessDenied);
        }
        port.queue.pop_front().ok_or(OolIpcError::QueueEmpty)
    }
}

// ============================================================================
// 4. Lock-Free SPSC & MPSC Shared-Memory Ring Buffers
// ============================================================================

/// Cache-line-aligned wrapper to prevent false sharing between producer and consumer.
#[repr(C, align(64))]
pub struct CachePadded<T> {
    value: T,
}

impl<T> CachePadded<T> {
    #[inline]
    pub const fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> core::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// Lock-free bounded Single-Producer Single-Consumer (SPSC) ring buffer.
pub struct SpscRing<T> {
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
    mask: usize,
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

unsafe impl<T: Send> Send for SpscRing<T> {}
unsafe impl<T: Send> Sync for SpscRing<T> {}

impl<T> SpscRing<T> {
    /// Create a new SPSC ring buffer with `capacity` (must be a power of two >= 2).
    pub fn new(capacity: usize) -> Result<Self, OolIpcError> {
        if capacity < 2 || !capacity.is_power_of_two() {
            return Err(OolIpcError::InvalidRingSize);
        }
        let mut vec = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            vec.push(UnsafeCell::new(MaybeUninit::uninit()));
        }
        Ok(Self {
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            mask: capacity - 1,
            buffer: vec.into_boxed_slice(),
        })
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    #[inline]
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity()
    }

    /// Push an item (single producer only). Returns `Err(val)` if full.
    pub fn try_push(&self, val: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail.wrapping_sub(head) >= self.capacity() {
            return Err(val);
        }
        let idx = tail & self.mask;
        unsafe {
            (*self.buffer[idx].get()).write(val);
        }
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Pop an item (single consumer only). Returns `None` if empty.
    pub fn try_pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let idx = head & self.mask;
        let val = unsafe { (*self.buffer[idx].get()).assume_init_read() };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(val)
    }
}

impl<T> Drop for SpscRing<T> {
    fn drop(&mut self) {
        while self.try_pop().is_some() {}
    }
}

/// Shared-memory framed SPSC byte ring buffer for variable-length zero-copy IPC packets.
pub struct ShmSpscRing {
    magic: u32,
    capacity: usize,
    mask: usize,
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
    dropped: AtomicU64,
    data: Box<[UnsafeCell<u8>]>,
}

unsafe impl Send for ShmSpscRing {}
unsafe impl Sync for ShmSpscRing {}

impl ShmSpscRing {
    /// Create a shared-memory byte ring with `capacity` bytes (power of two >= 64).
    pub fn new(capacity: usize) -> Result<Self, OolIpcError> {
        if capacity < 64 || !capacity.is_power_of_two() {
            return Err(OolIpcError::InvalidRingSize);
        }
        let mut vec = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            vec.push(UnsafeCell::new(0));
        }
        Ok(Self {
            magic: SHM_RING_MAGIC,
            capacity,
            mask: capacity - 1,
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            dropped: AtomicU64::new(0),
            data: vec.into_boxed_slice(),
        })
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.magic == SHM_RING_MAGIC
    }

    #[inline]
    pub fn available_read(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    #[inline]
    pub fn available_write(&self) -> usize {
        self.capacity - self.available_read()
    }

    #[inline]
    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    fn write_bytes_at(&self, offset: usize, src: &[u8]) {
        for (i, &b) in src.iter().enumerate() {
            let idx = (offset + i) & self.mask;
            unsafe {
                *self.data[idx].get() = b;
            }
        }
    }

    fn read_bytes_at(&self, offset: usize, dst: &mut [u8]) {
        for (i, slot) in dst.iter_mut().enumerate() {
            let idx = (offset + i) & self.mask;
            *slot = unsafe { *self.data[idx].get() };
        }
    }

    /// Write a length-prefixed frame (`u32` LE length + payload) into the ring.
    pub fn write_frame(&self, payload: &[u8]) -> Result<(), OolIpcError> {
        let total = 4usize
            .checked_add(payload.len())
            .ok_or(OolIpcError::MessageTooLarge)?;
        if total > self.capacity {
            return Err(OolIpcError::MessageTooLarge);
        }
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        let free = self.capacity - tail.wrapping_sub(head);
        if free < total {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return Err(OolIpcError::QueueFull);
        }
        let len_bytes = (payload.len() as u32).to_le_bytes();
        self.write_bytes_at(tail, &len_bytes);
        self.write_bytes_at(tail.wrapping_add(4), payload);
        self.tail.store(tail.wrapping_add(total), Ordering::Release);
        Ok(())
    }

    /// Read the next length-prefixed frame into `out`, returning the payload length.
    pub fn read_frame(&self, out: &mut [u8]) -> Result<usize, OolIpcError> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let avail = tail.wrapping_sub(head);
        if avail < 4 {
            return Err(OolIpcError::QueueEmpty);
        }
        let mut len_bytes = [0u8; 4];
        self.read_bytes_at(head, &mut len_bytes);
        let frame_len = u32::from_le_bytes(len_bytes) as usize;
        if avail < 4 + frame_len {
            return Err(OolIpcError::InvalidState);
        }
        if out.len() < frame_len {
            return Err(OolIpcError::MessageTooLarge);
        }
        self.read_bytes_at(head.wrapping_add(4), &mut out[..frame_len]);
        self.head
            .store(head.wrapping_add(4 + frame_len), Ordering::Release);
        Ok(frame_len)
    }
}

struct MpscSlot<T> {
    seq: AtomicUsize,
    data: UnsafeCell<MaybeUninit<T>>,
}

/// Lock-free bounded Multi-Producer Single-Consumer (MPSC) ring buffer (Vyukov algorithm).
pub struct MpscRing<T> {
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
    mask: usize,
    slots: Box<[MpscSlot<T>]>,
}

unsafe impl<T: Send> Send for MpscRing<T> {}
unsafe impl<T: Send> Sync for MpscRing<T> {}

impl<T> MpscRing<T> {
    /// Create a new MPSC ring buffer with `capacity` (must be a power of two >= 2).
    pub fn new(capacity: usize) -> Result<Self, OolIpcError> {
        if capacity < 2 || !capacity.is_power_of_two() {
            return Err(OolIpcError::InvalidRingSize);
        }
        let mut vec = Vec::with_capacity(capacity);
        for i in 0..capacity {
            vec.push(MpscSlot {
                seq: AtomicUsize::new(i),
                data: UnsafeCell::new(MaybeUninit::uninit()),
            });
        }
        Ok(Self {
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            mask: capacity - 1,
            slots: vec.into_boxed_slice(),
        })
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Push an item from any producer thread. Returns `Err(val)` if full.
    pub fn try_push(&self, val: T) -> Result<(), T> {
        let mut pos = self.tail.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[pos & self.mask];
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = seq as isize - pos as isize;
            if diff == 0 {
                match self.tail.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        unsafe {
                            (*slot.data.get()).write(val);
                        }
                        slot.seq.store(pos.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                return Err(val);
            } else {
                pos = self.tail.load(Ordering::Relaxed);
            }
        }
    }

    /// Pop an item (single consumer only). Returns `None` if empty.
    pub fn try_pop(&self) -> Option<T> {
        let pos = self.head.load(Ordering::Relaxed);
        let slot = &self.slots[pos & self.mask];
        let seq = slot.seq.load(Ordering::Acquire);
        let diff = seq as isize - pos.wrapping_add(1) as isize;
        if diff < 0 {
            return None;
        }
        let val = unsafe { (*slot.data.get()).assume_init_read() };
        self.head.store(pos.wrapping_add(1), Ordering::Release);
        slot.seq
            .store(pos.wrapping_add(self.capacity()), Ordering::Release);
        Some(val)
    }
}

impl<T> Drop for MpscRing<T> {
    fn drop(&mut self) {
        while self.try_pop().is_some() {}
    }
}

// ============================================================================
// 5. Global Kernel Instances & Self-Test
// ============================================================================

static GLOBAL_MACH_IPC: SpinLock<MachIpcSubsystem> = SpinLock::new(MachIpcSubsystem::new());
static GLOBAL_ALPC: SpinLock<AlpcSubsystem> = SpinLock::new(AlpcSubsystem::new());

/// Access the global Mach OOL IPC subsystem.
pub fn mach_ipc() -> &'static SpinLock<MachIpcSubsystem> {
    &GLOBAL_MACH_IPC
}

/// Access the global NT ALPC section-view subsystem.
pub fn alpc() -> &'static SpinLock<AlpcSubsystem> {
    &GLOBAL_ALPC
}

/// Run comprehensive self-tests for OOL remapping, Mach port rights, ALPC views, and ring buffers.
pub fn self_test() -> Result<(), &'static str> {
    // 1. OOL Page-Descriptor Remapping (Move, VirtualCopy/COW, Share)
    {
        let mut tracker = PhysFrameTracker::new();
        let mut sender = AddressSpaceMap::new(1);
        let mut receiver = AddressSpaceMap::new(2);

        let p0 = 0x2000_0000u64;
        let p1 = 0x2000_1000u64;
        tracker.retain(p0);
        tracker.retain(p1);
        sender
            .map_page(0x1000_0000, p0, OolProtection::READ_WRITE, false)
            .map_err(|_| "ool: map p0 failed")?;
        sender
            .map_page(0x1000_1000, p1, OolProtection::READ_WRITE, false)
            .map_err(|_| "ool: map p1 failed")?;

        // VirtualCopy (COW)
        let cow_desc = OolPageDescriptor::new(
            0x1000_0000,
            2 * PAGE_SIZE,
            OolCopyOption::VirtualCopy,
            OolProtection::READ_WRITE,
            false,
            vec![p0, p1],
        )
        .map_err(|_| "ool: cow desc failed")?;

        let res = remap_ool_descriptor(&mut tracker, &mut sender, &mut receiver, &cow_desc, None)
            .map_err(|_| "ool: cow remap failed")?;
        if !res.zero_copy || res.page_count != 2 || tracker.ref_count(p0) != 2 {
            return Err("ool: cow refcount mismatch");
        }

        // Resolve COW fault on receiver page 0
        let new_phys = receiver
            .resolve_cow_fault(res.dst_vaddr, &mut tracker)
            .map_err(|_| "ool: cow fault failed")?;
        if new_phys == p0 || tracker.ref_count(p0) != 1 {
            return Err("ool: cow fault did not split frame");
        }

        // Move semantics
        let p2 = 0x2000_2000u64;
        tracker.retain(p2);
        sender
            .map_page(0x1000_2000, p2, OolProtection::READ_WRITE, false)
            .map_err(|_| "ool: map p2 failed")?;
        let move_desc = OolPageDescriptor::new(
            0x1000_2000,
            PAGE_SIZE,
            OolCopyOption::Move,
            OolProtection::READ_WRITE,
            true,
            vec![p2],
        )
        .map_err(|_| "ool: move desc failed")?;
        let move_res =
            remap_ool_descriptor(&mut tracker, &mut sender, &mut receiver, &move_desc, None)
                .map_err(|_| "ool: move remap failed")?;
        if sender.lookup(0x1000_2000).is_some() || receiver.lookup(move_res.dst_vaddr).is_none() {
            return Err("ool: move did not transfer ownership");
        }
    }

    // 2. Mach Port Rights Transfer & OOL Message Delivery
    {
        let mut mach = MachIpcSubsystem::new();
        let task_a = 10u64;
        let task_b = 20u64;

        let rx_a = mach.create_port(task_a, 8);
        let send_b = mach
            .transfer_right(task_a, rx_a, task_b, MachMsgTypeName::MakeSend)
            .map_err(|_| "mach: MakeSend failed")?;
        let send_once_b = mach
            .transfer_right(task_a, rx_a, task_b, MachMsgTypeName::MakeSendOnce)
            .map_err(|_| "mach: MakeSendOnce failed")?;

        let msg = MachOolMessage {
            header: MachMsgHeader {
                bits: 0x1513,
                size: 64,
                remote_port: send_once_b,
                local_port: MachPortName::NULL,
                id: 42,
            },
            inline_data: vec![1, 2, 3, 4],
            ool_descriptors: Vec::new(),
            port_descriptors: vec![MachPortDescriptor {
                name: send_b,
                disposition: MachMsgTypeName::CopySend,
            }],
        };
        mach.send_msg(task_b, msg)
            .map_err(|_| "mach: send_msg failed")?;

        // SendOnce must be consumed after one send
        if mach.query_right(task_b, send_once_b).is_some() {
            return Err("mach: SendOnce not consumed");
        }

        let rx_msg = mach
            .recv_msg(task_a, rx_a)
            .map_err(|_| "mach: recv_msg failed")?;
        if rx_msg.header.id != 42 || rx_msg.inline_data != [1, 2, 3, 4] {
            return Err("mach: message payload mismatch");
        }

        // Destroying receive right turns send_b into DeadName
        mach.destroy_port(task_a, rx_a)
            .map_err(|_| "mach: destroy_port failed")?;
        if mach.query_right(task_b, send_b) != Some(MachPortRight::DeadName) {
            return Err("mach: send right did not become DeadName");
        }
    }

    // 3. NT ALPC Section Views
    {
        let mut tracker = PhysFrameTracker::new();
        let mut server_as = AddressSpaceMap::new(100);
        let mut client_as = AddressSpaceMap::new(101);
        let mut alpc_sys = AlpcSubsystem::new();

        let sec_id = alpc_sys
            .create_section(100, 2 * PAGE_SIZE, vec![0x3000_0000, 0x3000_1000])
            .map_err(|_| "alpc: create_section failed")?;
        let client_view = alpc_sys
            .map_section_view(
                sec_id,
                101,
                None,
                2 * PAGE_SIZE,
                OolProtection::READ_WRITE,
                true,
                &mut tracker,
                &mut client_as,
            )
            .map_err(|_| "alpc: map client view failed")?;
        let server_view = alpc_sys
            .map_section_view(
                sec_id,
                100,
                None,
                2 * PAGE_SIZE,
                OolProtection::READ,
                false,
                &mut tracker,
                &mut server_as,
            )
            .map_err(|_| "alpc: map server view failed")?;

        let port_id = alpc_sys.create_port(100, 256);
        alpc_sys
            .connect_port(port_id, 101)
            .map_err(|_| "alpc: connect_port failed")?;

        alpc_sys
            .send_with_view(
                port_id,
                101,
                vec![0xAA, 0xBB],
                Some(AlpcDataViewAttr {
                    view_id: client_view.view_id,
                    section_id: sec_id,
                    view_base: client_view.base_vaddr,
                    view_size: client_view.size,
                }),
            )
            .map_err(|_| "alpc: send_with_view failed")?;

        let rx = alpc_sys
            .recv_with_view(port_id, 100)
            .map_err(|_| "alpc: recv_with_view failed")?;
        if rx.inline_payload != [0xAA, 0xBB] || rx.data_view.is_none() {
            return Err("alpc: received message mismatch");
        }

        alpc_sys
            .unmap_section_view(client_view.view_id, &mut tracker, &mut client_as)
            .map_err(|_| "alpc: unmap client view failed")?;
        alpc_sys
            .unmap_section_view(server_view.view_id, &mut tracker, &mut server_as)
            .map_err(|_| "alpc: unmap server view failed")?;
    }

    // 4. Lock-Free SPSC, ShmSpscRing, and MPSC Ring Buffers
    {
        let spsc = SpscRing::<u64>::new(8).map_err(|_| "spsc: new failed")?;
        for i in 0..8u64 {
            spsc.try_push(i).map_err(|_| "spsc: push failed")?;
        }
        if spsc.try_push(99).is_ok() {
            return Err("spsc: overflow not detected");
        }
        for i in 0..8u64 {
            if spsc.try_pop() != Some(i) {
                return Err("spsc: pop mismatch");
            }
        }
        if spsc.try_pop().is_some() {
            return Err("spsc: expected empty");
        }

        let shm = ShmSpscRing::new(64).map_err(|_| "shm_spsc: new failed")?;
        if !shm.is_valid() {
            return Err("shm_spsc: invalid magic");
        }
        shm.write_frame(b"hello-ool-ipc")
            .map_err(|_| "shm_spsc: write_frame failed")?;
        let mut buf = [0u8; 32];
        let n = shm
            .read_frame(&mut buf)
            .map_err(|_| "shm_spsc: read_frame failed")?;
        if &buf[..n] != b"hello-ool-ipc" {
            return Err("shm_spsc: frame mismatch");
        }

        let mpsc = MpscRing::<u32>::new(8).map_err(|_| "mpsc: new failed")?;
        for i in 0..8u32 {
            mpsc.try_push(i * 10).map_err(|_| "mpsc: push failed")?;
        }
        if mpsc.try_push(999).is_ok() {
            return Err("mpsc: overflow not detected");
        }
        for i in 0..8u32 {
            if mpsc.try_pop() != Some(i * 10) {
                return Err("mpsc: pop mismatch");
            }
        }
    }

    Ok(())
}
