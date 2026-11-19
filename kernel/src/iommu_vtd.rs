//! Per-PCIe-BDF IOMMU (Intel VT-d / AMD-Vi) Page Tables, Interrupt Remapping,
//! and Confidential Computing (AMD SEV-SNP / Intel TDX) Probe.
//!
//! This module provides:
//! - Per-PCIe-BDF 4-level IOVA page tables (VT-d SLPT / AMD-Vi I/O page tables)
//!   with heap-allocated 4 KiB page-table nodes (keeping stack frames < 2 KiB).
//! - Hardware descriptor encodings for Intel VT-d (Root/Context/SLPT/IRTE) and
//!   AMD-Vi (DTE/IRTE), plus ACPI DMAR/IVRS table parsing.
//! - Interrupt Remapping Table Entry (IRTE) allocator, MSI/MSI-X remapped
//!   address/data encoder, and source-ID (BDF) validation.
//! - Safe CPUID/MSR probe for AMD SEV / SEV-ES / SEV-SNP and Intel TDX / TME.
//! - `self_test()` exercising all subsystems.
#![allow(dead_code)]

use alloc::alloc::{alloc_zeroed, dealloc, Layout};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::arch::asm;
use core::fmt;
use core::ptr::NonNull;

use crate::cpu::CpuVendor;
use crate::mm::SpinLock;
use crate::pci::PciDevice;
use crate::serial_println;

pub const PAGE_SIZE: usize = 4096;
const PT_ENTRIES: usize = 512;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IommuError {
    InvalidBdf,
    UnalignedAddress,
    ZeroSize,
    AddressOverflow,
    AlreadyMapped,
    NotMapped,
    DomainNotFound,
    BdfNotBound,
    OutOfMemory,
    IrteTableFull,
    InvalidIrteIndex,
    PermissionDenied,
    InvalidTable(&'static str),
}

impl IommuError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidBdf => "invalid PCIe BDF",
            Self::UnalignedAddress => "unaligned IOVA or physical address",
            Self::ZeroSize => "zero-sized DMA mapping",
            Self::AddressOverflow => "IOVA or physical address overflow",
            Self::AlreadyMapped => "IOVA page already mapped",
            Self::NotMapped => "IOVA page not mapped",
            Self::DomainNotFound => "IOMMU domain not found",
            Self::BdfNotBound => "PCIe BDF not bound to an IOMMU domain",
            Self::OutOfMemory => "out of memory allocating IOMMU page table",
            Self::IrteTableFull => "interrupt remapping table full",
            Self::InvalidIrteIndex => "invalid IRTE index",
            Self::PermissionDenied => "IOMMU source-ID or permission check failed",
            Self::InvalidTable(msg) => msg,
        }
    }
}

// ---------------------------------------------------------------------------
// PCIe BDF & Backend Types
// ---------------------------------------------------------------------------

/// PCIe Bus/Device/Function identifier (with PCI segment).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PciBdf {
    pub segment: u16,
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl PciBdf {
    pub const fn new(segment: u16, bus: u8, device: u8, function: u8) -> Self {
        Self {
            segment,
            bus,
            device,
            function,
        }
    }

    pub const fn from_pci_device(dev: &PciDevice) -> Self {
        Self {
            segment: 0,
            bus: dev.bus,
            device: dev.dev,
            function: dev.func,
        }
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.device < 32 && self.function < 8
    }

    /// 16-bit PCIe Requester / Source ID: `(bus << 8) | (device << 3) | function`.
    #[inline]
    pub const fn source_id(self) -> u16 {
        ((self.bus as u16) << 8)
            | (((self.device as u16) & 0x1F) << 3)
            | ((self.function as u16) & 0x07)
    }

    /// 8-bit devfn index within a VT-d Context Table (`(device << 3) | function`).
    #[inline]
    pub const fn devfn(self) -> u8 {
        ((self.device & 0x1F) << 3) | (self.function & 0x07)
    }
}

impl fmt::Display for PciBdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{}",
            self.segment, self.bus, self.device, self.function
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IommuBackend {
    IntelVtd,
    AmdVi,
    SoftwareEmulated,
}

// ---------------------------------------------------------------------------
// Hardware Descriptor Encodings (Intel VT-d & AMD-Vi)
// ---------------------------------------------------------------------------

/// Intel VT-d MMIO register offsets.
pub mod vtd_regs {
    pub const VER_REG: usize = 0x000;
    pub const CAP_REG: usize = 0x008;
    pub const ECAP_REG: usize = 0x010;
    pub const GCMD_REG: usize = 0x018;
    pub const GSTS_REG: usize = 0x01C;
    pub const RTADDR_REG: usize = 0x020;
    pub const CCMD_REG: usize = 0x028;
    pub const FSTS_REG: usize = 0x034;
    pub const FECTL_REG: usize = 0x038;
    pub const IRTA_REG: usize = 0x0B8;
    pub const IOTLB_REG: usize = 0x108;

    pub const GCMD_TE: u32 = 1 << 31;
    pub const GCMD_SRTP: u32 = 1 << 30;
    pub const GCMD_IRE: u32 = 1 << 25;
    pub const GCMD_SIRTP: u32 = 1 << 24;
}

/// Intel VT-d 128-bit Root Table Entry.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VtdRootEntry {
    pub lo: u64,
    pub hi: u64,
}

impl VtdRootEntry {
    pub const fn new(present: bool, context_table_phys: u64) -> Self {
        let lo = (context_table_phys & !0xFFF) | if present { 1 } else { 0 };
        Self { lo, hi: 0 }
    }

    #[inline]
    pub const fn is_present(self) -> bool {
        (self.lo & 1) != 0
    }

    #[inline]
    pub const fn context_table_addr(self) -> u64 {
        self.lo & !0xFFF
    }
}

/// Intel VT-d 128-bit Context Table Entry.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VtdContextEntry {
    pub lo: u64,
    pub hi: u64,
}

impl VtdContextEntry {
    /// Address Width encoding for 48-bit (4-level) second-level paging.
    pub const AW_48BIT_4LVL: u8 = 0b010;

    pub const fn new(
        present: bool,
        translation_type: u8,
        slpt_root_phys: u64,
        address_width: u8,
        domain_id: u16,
    ) -> Self {
        let lo = (slpt_root_phys & !0xFFF)
            | (((translation_type as u64) & 0x3) << 2)
            | if present { 1 } else { 0 };
        let hi = ((domain_id as u64) << 8) | ((address_width as u64) & 0x7);
        Self { lo, hi }
    }

    #[inline]
    pub const fn is_present(self) -> bool {
        (self.lo & 1) != 0
    }

    #[inline]
    pub const fn domain_id(self) -> u16 {
        ((self.hi >> 8) & 0xFFFF) as u16
    }

    #[inline]
    pub const fn slpt_root(self) -> u64 {
        self.lo & !0xFFF
    }
}

/// Intel VT-d 128-bit Interrupt Remapping Table Entry (IRTE).
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VtdIrte {
    pub lo: u64,
    pub hi: u64,
}

impl VtdIrte {
    #[inline]
    pub const fn is_present(self) -> bool {
        (self.lo & 1) != 0
    }

    #[inline]
    pub const fn vector(self) -> u8 {
        ((self.lo >> 16) & 0xFF) as u8
    }

    #[inline]
    pub const fn dest_id(self) -> u32 {
        ((self.lo >> 32) & 0xFFFF_FFFF) as u32
    }

    #[inline]
    pub const fn source_id(self) -> u16 {
        (self.hi & 0xFFFF) as u16
    }

    #[inline]
    pub const fn svt(self) -> u8 {
        ((self.hi >> 18) & 0x3) as u8
    }
}

/// AMD-Vi 256-bit Device Table Entry (DTE).
#[repr(C, align(32))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AmdViDte {
    pub words: [u64; 4],
}

impl AmdViDte {
    /// Mode = 4 (4-level I/O page table).
    pub const MODE_4LVL: u8 = 4;

    pub const fn new(
        valid: bool,
        translation_valid: bool,
        pt_root_phys: u64,
        domain_id: u16,
        read: bool,
        write: bool,
    ) -> Self {
        let mut w0 = (pt_root_phys & 0x000F_FFFF_FFFF_F000)
            | ((Self::MODE_4LVL as u64) << 9)
            | if translation_valid { 1 << 1 } else { 0 }
            | if valid { 1 } else { 0 };
        if read {
            w0 |= 1 << 61;
        }
        if write {
            w0 |= 1 << 62;
        }
        let w1 = domain_id as u64;
        Self {
            words: [w0, w1, 0, 0],
        }
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        (self.words[0] & 1) != 0
    }

    #[inline]
    pub const fn domain_id(self) -> u16 {
        (self.words[1] & 0xFFFF) as u16
    }

    #[inline]
    pub const fn pt_root(self) -> u64 {
        self.words[0] & 0x000F_FFFF_FFFF_F000
    }
}

/// AMD-Vi 32-bit basic Interrupt Remapping Table Entry.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AmdViIrte(pub u32);

impl AmdViIrte {
    pub const fn new(
        remap_en: bool,
        int_type: u8,
        dest_mode_logical: bool,
        dest_apic_id: u8,
        vector: u8,
    ) -> Self {
        let val = (if remap_en { 1u32 } else { 0 })
            | (((int_type as u32) & 0x7) << 2)
            | (if dest_mode_logical { 1u32 << 6 } else { 0 })
            | ((dest_apic_id as u32) << 8)
            | ((vector as u32) << 16);
        Self(val)
    }

    #[inline]
    pub const fn is_enabled(self) -> bool {
        (self.0 & 1) != 0
    }

    #[inline]
    pub const fn vector(self) -> u8 {
        ((self.0 >> 16) & 0xFF) as u8
    }
}

// ---------------------------------------------------------------------------
// ACPI DMAR / IVRS Table Parsing
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DmarUnitInfo {
    pub segment: u16,
    pub register_base: u64,
    pub include_pci_all: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DmarSummary {
    pub host_address_width: u8,
    pub intr_remap_supported: bool,
    pub unit_count: usize,
    pub first_unit: Option<DmarUnitInfo>,
}

/// Parse an Intel ACPI DMAR table from raw bytes (starting at the 36-byte SDT header).
pub fn parse_dmar_table(bytes: &[u8]) -> Result<DmarSummary, IommuError> {
    if bytes.len() < 48 {
        return Err(IommuError::InvalidTable("DMAR table too short"));
    }
    if &bytes[0..4] != b"DMAR" {
        return Err(IommuError::InvalidTable("invalid DMAR signature"));
    }
    let total_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if total_len < 48 || total_len > bytes.len() {
        return Err(IommuError::InvalidTable("invalid DMAR length"));
    }

    let host_address_width = bytes[36].saturating_add(1);
    let flags = bytes[37];
    let intr_remap_supported = (flags & 0x01) != 0;

    let mut offset = 48usize;
    let mut unit_count = 0usize;
    let mut first_unit = None;

    while offset + 4 <= total_len {
        let entry_type = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        let entry_len = u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]) as usize;
        if entry_len < 4 || offset + entry_len > total_len {
            return Err(IommuError::InvalidTable("corrupt DMAR remapping structure"));
        }
        // Type 0 = DRHD (DMA Remapping Hardware Unit Definition)
        if entry_type == 0 && entry_len >= 16 {
            let drhd_flags = bytes[offset + 4];
            let segment = u16::from_le_bytes([bytes[offset + 6], bytes[offset + 7]]);
            let reg_base = u64::from_le_bytes([
                bytes[offset + 8],
                bytes[offset + 9],
                bytes[offset + 10],
                bytes[offset + 11],
                bytes[offset + 12],
                bytes[offset + 13],
                bytes[offset + 14],
                bytes[offset + 15],
            ]);
            let info = DmarUnitInfo {
                segment,
                register_base: reg_base,
                include_pci_all: (drhd_flags & 0x01) != 0,
            };
            if first_unit.is_none() {
                first_unit = Some(info);
            }
            unit_count += 1;
        }
        offset += entry_len;
    }

    Ok(DmarSummary {
        host_address_width,
        intr_remap_supported,
        unit_count,
        first_unit,
    })
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IvrsSummary {
    pub iv_info: u32,
    pub unit_count: usize,
    pub first_mmio_base: Option<u64>,
}

/// Parse an AMD ACPI IVRS table from raw bytes (starting at the 36-byte SDT header).
pub fn parse_ivrs_table(bytes: &[u8]) -> Result<IvrsSummary, IommuError> {
    if bytes.len() < 48 {
        return Err(IommuError::InvalidTable("IVRS table too short"));
    }
    if &bytes[0..4] != b"IVRS" {
        return Err(IommuError::InvalidTable("invalid IVRS signature"));
    }
    let total_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if total_len < 48 || total_len > bytes.len() {
        return Err(IommuError::InvalidTable("invalid IVRS length"));
    }

    let iv_info = u32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
    let mut offset = 48usize;
    let mut unit_count = 0usize;
    let mut first_mmio_base = None;

    while offset + 4 <= total_len {
        let entry_type = bytes[offset];
        let entry_len = u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]) as usize;
        if entry_len < 4 || offset + entry_len > total_len {
            return Err(IommuError::InvalidTable("corrupt IVRS structure"));
        }
        // 0x10, 0x11, 0x40 = IVHD types
        if matches!(entry_type, 0x10 | 0x11 | 0x40) && entry_len >= 16 {
            let mmio_base = u64::from_le_bytes([
                bytes[offset + 8],
                bytes[offset + 9],
                bytes[offset + 10],
                bytes[offset + 11],
                bytes[offset + 12],
                bytes[offset + 13],
                bytes[offset + 14],
                bytes[offset + 15],
            ]);
            if first_mmio_base.is_none() {
                first_mmio_base = Some(mmio_base);
            }
            unit_count += 1;
        }
        offset += entry_len;
    }

    Ok(IvrsSummary {
        iv_info,
        unit_count,
        first_mmio_base,
    })
}

// ---------------------------------------------------------------------------
// Per-BDF 4-Level IOVA Page Tables (Heap-Allocated 4 KiB Nodes)
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DmaPermissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
    pub snoop: bool,
}

impl DmaPermissions {
    pub const RW: Self = Self {
        read: true,
        write: true,
        execute: false,
        snoop: true,
    };

    pub const RO: Self = Self {
        read: true,
        write: false,
        execute: false,
        snoop: true,
    };
}

const PTE_READ: u64 = 1 << 0;
const PTE_WRITE: u64 = 1 << 1;
const PTE_EXEC: u64 = 1 << 2;
const PTE_SNOOP: u64 = 1 << 11;
const PTE_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

#[inline]
const fn encode_pte(phys: u64, perms: DmaPermissions) -> u64 {
    let mut val = phys & PTE_ADDR_MASK;
    if perms.read {
        val |= PTE_READ;
    }
    if perms.write {
        val |= PTE_WRITE;
    }
    if perms.execute {
        val |= PTE_EXEC;
    }
    if perms.snoop {
        val |= PTE_SNOOP;
    }
    val
}

#[inline]
const fn decode_pte(entry: u64) -> Option<(u64, DmaPermissions)> {
    if (entry & (PTE_READ | PTE_WRITE | PTE_EXEC)) == 0 {
        return None;
    }
    let phys = entry & PTE_ADDR_MASK;
    let perms = DmaPermissions {
        read: (entry & PTE_READ) != 0,
        write: (entry & PTE_WRITE) != 0,
        execute: (entry & PTE_EXEC) != 0,
        snoop: (entry & PTE_SNOOP) != 0,
    };
    Some((phys, perms))
}

/// Heap-allocated 4 KiB-aligned 512-entry page table node.
/// Never placed on the stack, keeping all stack frames well under 2 KiB.
struct PageTableNode {
    ptr: NonNull<[u64; PT_ENTRIES]>,
}

unsafe impl Send for PageTableNode {}
unsafe impl Sync for PageTableNode {}

impl PageTableNode {
    fn new_zeroed() -> Result<Self, IommuError> {
        let layout =
            Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).map_err(|_| IommuError::OutOfMemory)?;
        let raw = unsafe { alloc_zeroed(layout) };
        let nn = NonNull::new(raw as *mut [u64; PT_ENTRIES]).ok_or(IommuError::OutOfMemory)?;
        Ok(Self { ptr: nn })
    }

    #[inline]
    fn as_slice(&self) -> &[u64; PT_ENTRIES] {
        unsafe { self.ptr.as_ref() }
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [u64; PT_ENTRIES] {
        unsafe { self.ptr.as_mut() }
    }

    #[inline]
    fn addr(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }
}

impl Drop for PageTableNode {
    fn drop(&mut self) {
        if let Ok(layout) = Layout::from_size_align(PAGE_SIZE, PAGE_SIZE) {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

/// 4-level IOVA page table for an isolated IOMMU domain.
pub struct IovaPageTable {
    domain_id: u16,
    backend: IommuBackend,
    root: PageTableNode,
    /// Owned child page-table nodes keyed by their virtual address (`node.addr()`).
    nodes: BTreeMap<u64, PageTableNode>,
    mapped_pages: usize,
    iotlb_generation: u64,
}

impl IovaPageTable {
    pub fn new(domain_id: u16, backend: IommuBackend) -> Result<Self, IommuError> {
        Ok(Self {
            domain_id,
            backend,
            root: PageTableNode::new_zeroed()?,
            nodes: BTreeMap::new(),
            mapped_pages: 0,
            iotlb_generation: 0,
        })
    }

    #[inline]
    pub const fn domain_id(&self) -> u16 {
        self.domain_id
    }

    #[inline]
    pub const fn backend(&self) -> IommuBackend {
        self.backend
    }

    #[inline]
    pub fn root_addr(&self) -> u64 {
        self.root.addr()
    }

    #[inline]
    pub const fn mapped_pages(&self) -> usize {
        self.mapped_pages
    }

    #[inline]
    pub const fn iotlb_generation(&self) -> u64 {
        self.iotlb_generation
    }

    #[inline]
    const fn indices(iova: u64) -> [usize; 4] {
        [
            ((iova >> 39) & 0x1FF) as usize,
            ((iova >> 30) & 0x1FF) as usize,
            ((iova >> 21) & 0x1FF) as usize,
            ((iova >> 12) & 0x1FF) as usize,
        ]
    }

    fn get_or_alloc_child(&mut self, parent_addr: u64, idx: usize) -> Result<u64, IommuError> {
        let entry = if parent_addr == self.root.addr() {
            self.root.as_slice()[idx]
        } else {
            let node = self
                .nodes
                .get(&parent_addr)
                .ok_or(IommuError::OutOfMemory)?;
            node.as_slice()[idx]
        };

        if (entry & (PTE_READ | PTE_WRITE)) != 0 {
            return Ok(entry & PTE_ADDR_MASK);
        }

        let child = PageTableNode::new_zeroed()?;
        let child_addr = child.addr();
        self.nodes.insert(child_addr, child);

        let new_entry = (child_addr & PTE_ADDR_MASK) | PTE_READ | PTE_WRITE;
        if parent_addr == self.root.addr() {
            self.root.as_slice_mut()[idx] = new_entry;
        } else if let Some(node) = self.nodes.get_mut(&parent_addr) {
            node.as_slice_mut()[idx] = new_entry;
        }
        Ok(child_addr)
    }

    pub fn map_page(
        &mut self,
        iova: u64,
        phys: u64,
        perms: DmaPermissions,
    ) -> Result<(), IommuError> {
        if (iova & 0xFFF) != 0 || (phys & 0xFFF) != 0 {
            return Err(IommuError::UnalignedAddress);
        }
        if !perms.read && !perms.write && !perms.execute {
            return Err(IommuError::PermissionDenied);
        }

        let [l4, l3, l2, l1] = Self::indices(iova);
        let pdpt = self.get_or_alloc_child(self.root.addr(), l4)?;
        let pd = self.get_or_alloc_child(pdpt, l3)?;
        let pt = self.get_or_alloc_child(pd, l2)?;

        let pt_node = self.nodes.get_mut(&pt).ok_or(IommuError::OutOfMemory)?;
        if decode_pte(pt_node.as_slice()[l1]).is_some() {
            return Err(IommuError::AlreadyMapped);
        }
        pt_node.as_slice_mut()[l1] = encode_pte(phys, perms);
        self.mapped_pages += 1;
        Ok(())
    }

    pub fn map_range(
        &mut self,
        iova: u64,
        phys: u64,
        size: usize,
        perms: DmaPermissions,
    ) -> Result<(), IommuError> {
        if size == 0 {
            return Err(IommuError::ZeroSize);
        }
        if (iova & 0xFFF) != 0 || (phys & 0xFFF) != 0 || (size & 0xFFF) != 0 {
            return Err(IommuError::UnalignedAddress);
        }
        let _ = iova
            .checked_add(size as u64)
            .ok_or(IommuError::AddressOverflow)?;
        let _ = phys
            .checked_add(size as u64)
            .ok_or(IommuError::AddressOverflow)?;

        let pages = size / PAGE_SIZE;
        for p in 0..pages {
            let off = (p * PAGE_SIZE) as u64;
            if let Err(e) = self.map_page(iova + off, phys + off, perms) {
                // Roll back partial range on failure.
                for r in 0..p {
                    let _ = self.unmap_page(iova + (r * PAGE_SIZE) as u64);
                }
                return Err(e);
            }
        }
        Ok(())
    }

    pub fn unmap_page(&mut self, iova: u64) -> Result<u64, IommuError> {
        if (iova & 0xFFF) != 0 {
            return Err(IommuError::UnalignedAddress);
        }
        let [l4, l3, l2, l1] = Self::indices(iova);

        let l4e = self.root.as_slice()[l4];
        if (l4e & (PTE_READ | PTE_WRITE)) == 0 {
            return Err(IommuError::NotMapped);
        }
        let pdpt = l4e & PTE_ADDR_MASK;
        let l3e = self
            .nodes
            .get(&pdpt)
            .ok_or(IommuError::NotMapped)?
            .as_slice()[l3];
        if (l3e & (PTE_READ | PTE_WRITE)) == 0 {
            return Err(IommuError::NotMapped);
        }
        let pd = l3e & PTE_ADDR_MASK;
        let l2e = self.nodes.get(&pd).ok_or(IommuError::NotMapped)?.as_slice()[l2];
        if (l2e & (PTE_READ | PTE_WRITE)) == 0 {
            return Err(IommuError::NotMapped);
        }
        let pt = l2e & PTE_ADDR_MASK;
        let pt_node = self.nodes.get_mut(&pt).ok_or(IommuError::NotMapped)?;
        let (phys, _) = decode_pte(pt_node.as_slice()[l1]).ok_or(IommuError::NotMapped)?;
        pt_node.as_slice_mut()[l1] = 0;
        self.mapped_pages = self.mapped_pages.saturating_sub(1);
        self.iotlb_generation = self.iotlb_generation.wrapping_add(1);
        Ok(phys)
    }

    pub fn unmap_range(&mut self, iova: u64, size: usize) -> Result<usize, IommuError> {
        if size == 0 {
            return Err(IommuError::ZeroSize);
        }
        if (iova & 0xFFF) != 0 || (size & 0xFFF) != 0 {
            return Err(IommuError::UnalignedAddress);
        }
        let pages = size / PAGE_SIZE;
        let mut unmapped = 0usize;
        for p in 0..pages {
            if self.unmap_page(iova + (p * PAGE_SIZE) as u64).is_ok() {
                unmapped += 1;
            }
        }
        Ok(unmapped)
    }

    pub fn translate(&self, iova: u64) -> Option<(u64, DmaPermissions)> {
        let [l4, l3, l2, l1] = Self::indices(iova);
        let l4e = self.root.as_slice()[l4];
        if (l4e & (PTE_READ | PTE_WRITE)) == 0 {
            return None;
        }
        let pdpt = self.nodes.get(&(l4e & PTE_ADDR_MASK))?;
        let l3e = pdpt.as_slice()[l3];
        if (l3e & (PTE_READ | PTE_WRITE)) == 0 {
            return None;
        }
        let pd = self.nodes.get(&(l3e & PTE_ADDR_MASK))?;
        let l2e = pd.as_slice()[l2];
        if (l2e & (PTE_READ | PTE_WRITE)) == 0 {
            return None;
        }
        let pt = self.nodes.get(&(l2e & PTE_ADDR_MASK))?;
        let (page_phys, perms) = decode_pte(pt.as_slice()[l1])?;
        Some((page_phys | (iova & 0xFFF), perms))
    }
}

// ---------------------------------------------------------------------------
// Per-PCIe-BDF IOMMU Manager
// ---------------------------------------------------------------------------

pub struct BdfIommuManager {
    backend: IommuBackend,
    next_domain_id: u16,
    bdf_to_domain: BTreeMap<PciBdf, u16>,
    domains: BTreeMap<u16, IovaPageTable>,
}

impl BdfIommuManager {
    pub const fn new(backend: IommuBackend) -> Self {
        Self {
            backend,
            next_domain_id: 1,
            bdf_to_domain: BTreeMap::new(),
            domains: BTreeMap::new(),
        }
    }

    pub fn set_backend(&mut self, backend: IommuBackend) {
        self.backend = backend;
    }

    pub const fn backend(&self) -> IommuBackend {
        self.backend
    }

    /// Bind a PCIe BDF to a newly allocated isolated IOMMU domain.
    pub fn bind_bdf(&mut self, bdf: PciBdf) -> Result<u16, IommuError> {
        if !bdf.is_valid() {
            return Err(IommuError::InvalidBdf);
        }
        if let Some(&existing) = self.bdf_to_domain.get(&bdf) {
            return Ok(existing);
        }
        let domain_id = self.next_domain_id;
        self.next_domain_id = self.next_domain_id.wrapping_add(1).max(1);
        let pt = IovaPageTable::new(domain_id, self.backend)?;
        self.domains.insert(domain_id, pt);
        self.bdf_to_domain.insert(bdf, domain_id);
        Ok(domain_id)
    }

    /// Unbind a PCIe BDF and destroy its domain if no other BDF references it.
    pub fn unbind_bdf(&mut self, bdf: PciBdf) -> Result<(), IommuError> {
        let domain_id = self
            .bdf_to_domain
            .remove(&bdf)
            .ok_or(IommuError::BdfNotBound)?;
        let still_used = self.bdf_to_domain.values().any(|&d| d == domain_id);
        if !still_used {
            self.domains.remove(&domain_id);
        }
        Ok(())
    }

    pub fn domain_for_bdf(&self, bdf: PciBdf) -> Option<u16> {
        self.bdf_to_domain.get(&bdf).copied()
    }

    pub fn map_bdf(
        &mut self,
        bdf: PciBdf,
        iova: u64,
        phys: u64,
        size: usize,
        perms: DmaPermissions,
    ) -> Result<(), IommuError> {
        let domain_id = *self
            .bdf_to_domain
            .get(&bdf)
            .ok_or(IommuError::BdfNotBound)?;
        let pt = self
            .domains
            .get_mut(&domain_id)
            .ok_or(IommuError::DomainNotFound)?;
        pt.map_range(iova, phys, size, perms)
    }

    pub fn unmap_bdf(&mut self, bdf: PciBdf, iova: u64, size: usize) -> Result<usize, IommuError> {
        let domain_id = *self
            .bdf_to_domain
            .get(&bdf)
            .ok_or(IommuError::BdfNotBound)?;
        let pt = self
            .domains
            .get_mut(&domain_id)
            .ok_or(IommuError::DomainNotFound)?;
        pt.unmap_range(iova, size)
    }

    pub fn translate_bdf(&self, bdf: PciBdf, iova: u64) -> Option<(u64, DmaPermissions)> {
        let domain_id = self.bdf_to_domain.get(&bdf)?;
        let pt = self.domains.get(domain_id)?;
        pt.translate(iova)
    }

    /// Build the hardware VT-d Context Entry for `bdf`.
    pub fn vtd_context_entry(&self, bdf: PciBdf) -> Option<VtdContextEntry> {
        let domain_id = *self.bdf_to_domain.get(&bdf)?;
        let pt = self.domains.get(&domain_id)?;
        Some(VtdContextEntry::new(
            true,
            0,
            pt.root_addr(),
            VtdContextEntry::AW_48BIT_4LVL,
            domain_id,
        ))
    }

    /// Build the hardware AMD-Vi Device Table Entry for `bdf`.
    pub fn amdvi_dte(&self, bdf: PciBdf) -> Option<AmdViDte> {
        let domain_id = *self.bdf_to_domain.get(&bdf)?;
        let pt = self.domains.get(&domain_id)?;
        Some(AmdViDte::new(
            true,
            true,
            pt.root_addr(),
            domain_id,
            true,
            true,
        ))
    }
}

// ---------------------------------------------------------------------------
// IRTE Interrupt Remapping
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IrqDeliveryMode {
    Fixed = 0,
    LowestPriority = 1,
    Smi = 2,
    Nmi = 4,
    Init = 5,
    ExtInt = 7,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IrqTriggerMode {
    Edge = 0,
    Level = 1,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IrqDestMode {
    Physical = 0,
    Logical = 1,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SourceValidation {
    None = 0,
    VerifySid = 1,
    VerifyBusRange = 2,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IrteConfig {
    pub source_bdf: PciBdf,
    pub vector: u8,
    pub dest_apic_id: u32,
    pub delivery_mode: IrqDeliveryMode,
    pub trigger_mode: IrqTriggerMode,
    pub dest_mode: IrqDestMode,
    pub source_validation: SourceValidation,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RemappedMsi {
    pub irte_index: u16,
    pub msi_addr: u64,
    pub msi_data: u32,
}

#[derive(Copy, Clone, Debug)]
struct IrteSlot {
    allocated: bool,
    cfg: IrteConfig,
}

impl IrteSlot {
    const fn empty() -> Self {
        Self {
            allocated: false,
            cfg: IrteConfig {
                source_bdf: PciBdf::new(0, 0, 0, 0),
                vector: 0,
                dest_apic_id: 0,
                delivery_mode: IrqDeliveryMode::Fixed,
                trigger_mode: IrqTriggerMode::Edge,
                dest_mode: IrqDestMode::Physical,
                source_validation: SourceValidation::VerifySid,
            },
        }
    }
}

/// Interrupt Remapping Table managing IRTE slots and MSI/MSI-X remapped encodings.
pub struct InterruptRemapTable {
    slots: Vec<IrteSlot>,
}

impl InterruptRemapTable {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.clamp(16, 4096);
        let mut slots = Vec::with_capacity(cap);
        slots.resize(cap, IrteSlot::empty());
        Self { slots }
    }

    pub fn allocate_irte(&mut self, cfg: IrteConfig) -> Result<RemappedMsi, IommuError> {
        if !cfg.source_bdf.is_valid() {
            return Err(IommuError::InvalidBdf);
        }
        let idx = self
            .slots
            .iter()
            .position(|s| !s.allocated)
            .ok_or(IommuError::IrteTableFull)?;
        self.slots[idx] = IrteSlot {
            allocated: true,
            cfg,
        };
        Ok(Self::encode_msi(idx as u16))
    }

    pub fn update_irte(&mut self, index: u16, cfg: IrteConfig) -> Result<(), IommuError> {
        if !cfg.source_bdf.is_valid() {
            return Err(IommuError::InvalidBdf);
        }
        let slot = self
            .slots
            .get_mut(index as usize)
            .ok_or(IommuError::InvalidIrteIndex)?;
        if !slot.allocated {
            return Err(IommuError::InvalidIrteIndex);
        }
        slot.cfg = cfg;
        Ok(())
    }

    pub fn free_irte(&mut self, index: u16) -> Result<(), IommuError> {
        let slot = self
            .slots
            .get_mut(index as usize)
            .ok_or(IommuError::InvalidIrteIndex)?;
        if !slot.allocated {
            return Err(IommuError::InvalidIrteIndex);
        }
        *slot = IrteSlot::empty();
        Ok(())
    }

    /// Encode x86 remappable MSI address/data for `irte_index`.
    /// Bit 4 = 1 (Interrupt Format = Remappable), Bit 3 = 0 (SHV = 0).
    #[inline]
    pub const fn encode_msi(irte_index: u16) -> RemappedMsi {
        let handle_lo = (irte_index as u64) & 0x7FFF;
        let handle_hi = ((irte_index as u64) >> 15) & 0x1;
        let msi_addr = 0xFEE0_0000u64 | (handle_lo << 5) | (1 << 4) | (handle_hi << 2);
        RemappedMsi {
            irte_index,
            msi_addr,
            msi_data: 0,
        }
    }

    /// Validate an incoming remapped interrupt against the IRTE's source-ID policy.
    pub fn validate_interrupt(
        &self,
        index: u16,
        source_bdf: PciBdf,
    ) -> Result<(u8, u32), IommuError> {
        let slot = self
            .slots
            .get(index as usize)
            .ok_or(IommuError::InvalidIrteIndex)?;
        if !slot.allocated {
            return Err(IommuError::InvalidIrteIndex);
        }
        match slot.cfg.source_validation {
            SourceValidation::None => {}
            SourceValidation::VerifySid => {
                if slot.cfg.source_bdf.source_id() != source_bdf.source_id() {
                    return Err(IommuError::PermissionDenied);
                }
            }
            SourceValidation::VerifyBusRange => {
                if slot.cfg.source_bdf.bus != source_bdf.bus {
                    return Err(IommuError::PermissionDenied);
                }
            }
        }
        Ok((slot.cfg.vector, slot.cfg.dest_apic_id))
    }

    pub fn encode_vtd_irte(&self, index: u16) -> Option<VtdIrte> {
        let slot = self.slots.get(index as usize)?;
        if !slot.allocated {
            return None;
        }
        let c = slot.cfg;
        let lo = 1u64 // Present
            | ((c.dest_mode as u64) << 2)
            | ((c.trigger_mode as u64) << 4)
            | ((c.delivery_mode as u64) << 5)
            | ((c.vector as u64) << 16)
            | ((c.dest_apic_id as u64) << 32);
        let hi = (c.source_bdf.source_id() as u64) | ((c.source_validation as u64) << 18);
        Some(VtdIrte { lo, hi })
    }

    pub fn encode_amdvi_irte(&self, index: u16) -> Option<AmdViIrte> {
        let slot = self.slots.get(index as usize)?;
        if !slot.allocated {
            return None;
        }
        let c = slot.cfg;
        Some(AmdViIrte::new(
            true,
            c.delivery_mode as u8,
            matches!(c.dest_mode, IrqDestMode::Logical),
            c.dest_apic_id as u8,
            c.vector,
        ))
    }
}

// ---------------------------------------------------------------------------
// Confidential Computing Probe (AMD SEV-SNP / Intel TDX / TME)
// ---------------------------------------------------------------------------

const MSR_AMD64_SEV: u32 = 0xC001_0131;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConfidentialTech {
    None,
    AmdSev,
    AmdSevEs,
    AmdSevSnp,
    IntelTme,
    IntelTdx,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SevSnpInfo {
    pub supported: bool,
    pub sev_active: bool,
    pub sev_es_active: bool,
    pub sev_snp_active: bool,
    pub rmp_enabled: bool,
    pub vtom_enabled: bool,
    pub c_bit_pos: u8,
    pub phys_addr_reduction: u8,
    pub min_asid: u32,
    pub num_encrypted_guests: u32,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TdxInfo {
    pub tme_supported: bool,
    pub mktme_supported: bool,
    pub tdx_guest: bool,
    pub max_keyid_bits: u8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ConfidentialProbe {
    pub vendor: CpuVendor,
    pub tech: ConfidentialTech,
    pub sev_snp: SevSnpInfo,
    pub tdx: TdxInfo,
    pub dma_requires_shared_bounce: bool,
}

impl ConfidentialProbe {
    /// Apply the AMD SEV C-bit to a physical address (for private encrypted pages).
    #[inline]
    pub const fn apply_c_bit(&self, phys: u64) -> u64 {
        if self.sev_snp.c_bit_pos > 0 && self.sev_snp.c_bit_pos < 64 {
            phys | (1u64 << self.sev_snp.c_bit_pos)
        } else {
            phys
        }
    }

    /// Clear the AMD SEV C-bit (for shared DMA bounce buffers).
    #[inline]
    pub const fn clear_c_bit(&self, phys: u64) -> u64 {
        if self.sev_snp.c_bit_pos > 0 && self.sev_snp.c_bit_pos < 64 {
            phys & !(1u64 << self.sev_snp.c_bit_pos)
        } else {
            phys
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct CpuidRegs {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

#[inline]
fn cpuid(leaf: u32, subleaf: u32) -> CpuidRegs {
    let eax: u32;
    let ebx_64: u64;
    let ecx: u32;
    let edx: u32;
    unsafe {
        asm!(
            "mov {tmp}, rbx",
            "cpuid",
            "xchg {tmp}, rbx",
            inout("eax") leaf => eax,
            inout("ecx") subleaf => ecx,
            tmp = out(reg) ebx_64,
            lateout("edx") edx,
            options(nomem, nostack, preserves_flags),
        );
    }
    CpuidRegs {
        eax,
        ebx: ebx_64 as u32,
        ecx,
        edx,
    }
}

#[inline]
unsafe fn rdmsr(msr: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    asm!(
        "rdmsr",
        in("ecx") msr,
        out("eax") lo,
        out("edx") hi,
        options(nomem, nostack, preserves_flags),
    );
    ((hi as u64) << 32) | (lo as u64)
}

/// Probe CPUID and confidential-computing MSRs safely for AMD SEV-SNP and Intel TDX/TME.
pub fn probe_confidential_computing() -> ConfidentialProbe {
    let leaf0 = cpuid(0, 0);
    let vendor = match (leaf0.ebx, leaf0.edx, leaf0.ecx) {
        (0x756e_6547, 0x4965_6e69, 0x6c65_746e) => CpuVendor::Intel,
        (0x6874_7541, 0x6974_6e65, 0x444d_4163) => CpuVendor::Amd,
        _ => CpuVendor::Unknown,
    };

    let mut sev_snp = SevSnpInfo::default();
    let mut tdx = TdxInfo::default();

    let max_ext = cpuid(0x8000_0000, 0).eax;
    if vendor == CpuVendor::Amd && max_ext >= 0x8000_001F {
        let sev_leaf = cpuid(0x8000_001F, 0);
        let sev_cap = (sev_leaf.eax & (1 << 1)) != 0;
        let snp_cap = (sev_leaf.eax & (1 << 4)) != 0;
        sev_snp.supported = sev_cap || snp_cap;
        sev_snp.c_bit_pos = (sev_leaf.ebx & 0x3F) as u8;
        sev_snp.phys_addr_reduction = ((sev_leaf.ebx >> 6) & 0x3F) as u8;
        sev_snp.num_encrypted_guests = sev_leaf.ecx;
        sev_snp.min_asid = sev_leaf.edx;

        if sev_snp.supported {
            let msr_val = unsafe { rdmsr(MSR_AMD64_SEV) };
            sev_snp.sev_active = (msr_val & (1 << 0)) != 0;
            sev_snp.sev_es_active = (msr_val & (1 << 1)) != 0;
            sev_snp.sev_snp_active = (msr_val & (1 << 2)) != 0;
            sev_snp.vtom_enabled = (msr_val & (1 << 3)) != 0;
            sev_snp.rmp_enabled = sev_snp.sev_snp_active;
        }
    }

    if vendor == CpuVendor::Intel {
        if leaf0.eax >= 0x07 {
            let leaf7 = cpuid(0x07, 0);
            tdx.tme_supported = (leaf7.ecx & (1 << 13)) != 0;
        }
        if leaf0.eax >= 0x21 {
            let leaf21 = cpuid(0x21, 0);
            // "IntelTDX    " in EBX, EDX, ECX
            if leaf21.ebx == 0x6574_6E49 && leaf21.edx == 0x5844_546C && leaf21.ecx == 0x2020_2020 {
                tdx.tdx_guest = true;
            }
        }
    }

    let tech = if sev_snp.sev_snp_active {
        ConfidentialTech::AmdSevSnp
    } else if sev_snp.sev_es_active {
        ConfidentialTech::AmdSevEs
    } else if sev_snp.sev_active {
        ConfidentialTech::AmdSev
    } else if tdx.tdx_guest {
        ConfidentialTech::IntelTdx
    } else if tdx.tme_supported {
        ConfidentialTech::IntelTme
    } else {
        ConfidentialTech::None
    };

    let dma_requires_shared_bounce = matches!(
        tech,
        ConfidentialTech::AmdSev
            | ConfidentialTech::AmdSevEs
            | ConfidentialTech::AmdSevSnp
            | ConfidentialTech::IntelTdx
    );

    ConfidentialProbe {
        vendor,
        tech,
        sev_snp,
        tdx,
        dma_requires_shared_bounce,
    }
}

// ---------------------------------------------------------------------------
// Global State
// ---------------------------------------------------------------------------

static BDF_IOMMU: SpinLock<BdfIommuManager> =
    SpinLock::new(BdfIommuManager::new(IommuBackend::SoftwareEmulated));

pub fn init(vendor: CpuVendor) -> ConfidentialProbe {
    let backend = match vendor {
        CpuVendor::Intel => IommuBackend::IntelVtd,
        CpuVendor::Amd => IommuBackend::AmdVi,
        CpuVendor::Unknown => IommuBackend::SoftwareEmulated,
    };
    BDF_IOMMU.lock().set_backend(backend);
    probe_confidential_computing()
}

pub fn bind_device(bdf: PciBdf) -> Result<u16, IommuError> {
    BDF_IOMMU.lock().bind_bdf(bdf)
}

pub fn map_device_dma(
    bdf: PciBdf,
    iova: u64,
    phys: u64,
    size: usize,
    perms: DmaPermissions,
) -> Result<(), IommuError> {
    BDF_IOMMU.lock().map_bdf(bdf, iova, phys, size, perms)
}

pub fn translate_device_dma(bdf: PciBdf, iova: u64) -> Option<(u64, DmaPermissions)> {
    BDF_IOMMU.lock().translate_bdf(bdf, iova)
}

// ---------------------------------------------------------------------------
// Self-Test
// ---------------------------------------------------------------------------

pub fn self_test() -> Result<(), &'static str> {
    serial_println!("[iommu_vtd] running self_test...");

    // 1. Per-BDF 4-level IOVA page tables & domain isolation
    let bdf_nvme = PciBdf::new(0, 0, 4, 0);
    let bdf_nic = PciBdf::new(0, 0, 5, 0);
    if !bdf_nvme.is_valid() || bdf_nvme.source_id() != 0x0020 {
        return Err("iommu_vtd self_test: BDF source_id mismatch");
    }

    let mut mgr = BdfIommuManager::new(IommuBackend::IntelVtd);
    let dom_nvme = mgr.bind_bdf(bdf_nvme).map_err(|e| e.as_str())?;
    let dom_nic = mgr.bind_bdf(bdf_nic).map_err(|e| e.as_str())?;
    if dom_nvme == dom_nic {
        return Err("iommu_vtd self_test: distinct BDFs shared domain");
    }

    // Map same IOVA in both domains to different physical pages
    mgr.map_bdf(
        bdf_nvme,
        0x1000_0000,
        0x2000_0000,
        2 * PAGE_SIZE,
        DmaPermissions::RW,
    )
    .map_err(|e| e.as_str())?;
    mgr.map_bdf(
        bdf_nic,
        0x1000_0000,
        0x3000_0000,
        PAGE_SIZE,
        DmaPermissions::RO,
    )
    .map_err(|e| e.as_str())?;

    let (p1, perm1) = mgr
        .translate_bdf(bdf_nvme, 0x1000_1234)
        .ok_or("iommu_vtd self_test: NVMe translate failed")?;
    if p1 != 0x2000_1234 || !perm1.write {
        return Err("iommu_vtd self_test: NVMe translation mismatch");
    }

    let (p2, perm2) = mgr
        .translate_bdf(bdf_nic, 0x1000_0234)
        .ok_or("iommu_vtd self_test: NIC translate failed")?;
    if p2 != 0x3000_0234 || perm2.write {
        return Err("iommu_vtd self_test: NIC translation mismatch");
    }

    if mgr.translate_bdf(bdf_nic, 0x1000_1000).is_some() {
        return Err("iommu_vtd self_test: NIC accessed unmapped page");
    }

    // Verify VT-d Context Entry & AMD-Vi DTE generation
    let ctx = mgr
        .vtd_context_entry(bdf_nvme)
        .ok_or("iommu_vtd self_test: missing VT-d context entry")?;
    if !ctx.is_present() || ctx.domain_id() != dom_nvme {
        return Err("iommu_vtd self_test: VT-d context entry mismatch");
    }
    let dte = mgr
        .amdvi_dte(bdf_nic)
        .ok_or("iommu_vtd self_test: missing AMD-Vi DTE")?;
    if !dte.is_valid() || dte.domain_id() != dom_nic {
        return Err("iommu_vtd self_test: AMD-Vi DTE mismatch");
    }

    let unmapped = mgr
        .unmap_bdf(bdf_nvme, 0x1000_0000, 2 * PAGE_SIZE)
        .map_err(|e| e.as_str())?;
    if unmapped != 2 || mgr.translate_bdf(bdf_nvme, 0x1000_0000).is_some() {
        return Err("iommu_vtd self_test: unmap failed");
    }
    mgr.unbind_bdf(bdf_nvme).map_err(|e| e.as_str())?;
    mgr.unbind_bdf(bdf_nic).map_err(|e| e.as_str())?;

    // 2. IRTE Interrupt Remapping
    let mut irt = InterruptRemapTable::new(64);
    let msi = irt
        .allocate_irte(IrteConfig {
            source_bdf: bdf_nvme,
            vector: 0x48,
            dest_apic_id: 3,
            delivery_mode: IrqDeliveryMode::Fixed,
            trigger_mode: IrqTriggerMode::Edge,
            dest_mode: IrqDestMode::Physical,
            source_validation: SourceValidation::VerifySid,
        })
        .map_err(|e| e.as_str())?;

    if (msi.msi_addr & (1 << 4)) == 0 {
        return Err("iommu_vtd self_test: remapped MSI missing format bit");
    }
    let (vec, apic) = irt
        .validate_interrupt(msi.irte_index, bdf_nvme)
        .map_err(|e| e.as_str())?;
    if vec != 0x48 || apic != 3 {
        return Err("iommu_vtd self_test: IRTE vector/apic mismatch");
    }
    if irt.validate_interrupt(msi.irte_index, bdf_nic) != Err(IommuError::PermissionDenied) {
        return Err("iommu_vtd self_test: IRTE failed to reject spoofed BDF");
    }
    let vtd_irte = irt
        .encode_vtd_irte(msi.irte_index)
        .ok_or("iommu_vtd self_test: encode_vtd_irte failed")?;
    if !vtd_irte.is_present() || vtd_irte.vector() != 0x48 || vtd_irte.source_id() != 0x0020 {
        return Err("iommu_vtd self_test: VtdIrte encoding mismatch");
    }
    irt.free_irte(msi.irte_index).map_err(|e| e.as_str())?;

    // 3. Confidential Computing Probe (SEV-SNP / TDX)
    let mut probe = probe_confidential_computing();
    probe.sev_snp.c_bit_pos = 51;
    let enc = probe.apply_c_bit(0x1000_0000);
    if enc != (0x1000_0000 | (1u64 << 51)) || probe.clear_c_bit(enc) != 0x1000_0000 {
        return Err("iommu_vtd self_test: SEV C-bit helper mismatch");
    }

    // 4. Synthetic DMAR header parse check
    let mut dmar = [0u8; 64];
    dmar[0..4].copy_from_slice(b"DMAR");
    dmar[4..8].copy_from_slice(&64u32.to_le_bytes());
    dmar[36] = 47; // HAW = 48
    dmar[37] = 1; // INTR_REMAP
    dmar[48..50].copy_from_slice(&0u16.to_le_bytes()); // DRHD
    dmar[50..52].copy_from_slice(&16u16.to_le_bytes());
    dmar[52] = 1; // INCLUDE_PCI_ALL
    dmar[56..64].copy_from_slice(&0xFED9_0000u64.to_le_bytes());
    let summary = parse_dmar_table(&dmar).map_err(|e| e.as_str())?;
    if summary.host_address_width != 48 || !summary.intr_remap_supported || summary.unit_count != 1
    {
        return Err("iommu_vtd self_test: DMAR parse mismatch");
    }

    serial_println!(
        "[iommu_vtd] self_test PASSED (tech={:?}, vendor={:?})",
        probe.tech,
        probe.vendor
    );
    Ok(())
}
