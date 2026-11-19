#![allow(dead_code)]

//! Process & Per-Process Address Space Manager.
//!
//! Provides:
//! - PID / TID table and process/thread lifecycle management
//! - Per-process 4-level x86_64 PML4 root (`AddressSpace`)
//! - Virtual Memory Areas (`VmArea` / VAD) with W^X and Copy-on-Write (`COW`) flags
//! - Demand-paging and CoW `#PF` fault handler
//! - `fork` / `exec` / `wait` / `exit` state machine
//! - Per-thread TLS (`FS_BASE` / `GS_BASE`) state and MSR programming
//! - Unified POSIX signal / Windows SEH / Darwin Mach-exception queue
//! - `self_test()` verification suite

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::arch::asm;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};
use core::ptr;

use nexus_abi::Personality;
use nexus_compat::{inspect_binary, BinaryFormat, CompatError};

use crate::cap::CapTable;
use crate::iommu::enforce_wx;
use crate::mm::{alloc_frame, dealloc_frame, stats, SpinLock, PAGE_SIZE};
use crate::sched::CpuContext;
use crate::usermode::UserContext;
use crate::vmm::{AddressSpace, PageFlags};

/// Default user stack base virtual address.
pub const USER_STACK_BASE: u64 = 0x7FFF_FFFF_0000;
/// Default user stack size in pages (16 KiB).
pub const USER_STACK_PAGES: usize = 4;
/// Default user stack top virtual address.
pub const USER_STACK_TOP: u64 = USER_STACK_BASE + (USER_STACK_PAGES as u64) * (PAGE_SIZE as u64);

/// x86_64 MSR for user-mode FS base (POSIX/ELF TLS).
pub const IA32_FS_BASE: u32 = 0xC000_0100;
/// x86_64 MSR for active GS base.
pub const IA32_GS_BASE: u32 = 0xC000_0101;
/// x86_64 MSR for swapped user GS base (Windows TEB / Darwin TLS).
pub const IA32_KERNEL_GS_BASE: u32 = 0xC000_0102;

/// x86_64 `#PF` error code bits.
pub const PF_ERR_PRESENT: u64 = 1 << 0;
pub const PF_ERR_WRITE: u64 = 1 << 1;
pub const PF_ERR_USER: u64 = 1 << 2;
pub const PF_ERR_RESERVED: u64 = 1 << 3;
pub const PF_ERR_INSTR_FETCH: u64 = 1 << 4;

/// Unique process identifier.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pid(pub u64);

impl Pid {
    pub const KERNEL: Self = Self(0);
    pub const INIT: Self = Self(1);

    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Unique thread identifier.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tid(pub u64);

impl Tid {
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Errors returned by process and address-space operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProcError {
    NotInitialized,
    ProcessNotFound,
    ThreadNotFound,
    InvalidRange,
    Overlap,
    WxViolation,
    OutOfMemory,
    Segfault,
    StackOverflow,
    ProtectionFault,
    InvalidBinary(&'static str),
    NoChild,
    WouldBlock,
    InvalidState,
}

/// Virtual Memory Area protection and behavior flags.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct VmFlags(u32);

impl VmFlags {
    pub const NONE: Self = Self(0);
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const EXEC: Self = Self(1 << 2);
    pub const USER: Self = Self(1 << 3);
    pub const COW: Self = Self(1 << 4);
    pub const DEMAND: Self = Self(1 << 5);
    pub const SHARED: Self = Self(1 << 6);
    pub const STACK: Self = Self(1 << 7);
    pub const GUARD: Self = Self(1 << 8);
    pub const LOCKED: Self = Self(1 << 9);

    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }

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

    /// Returns `true` if both `WRITE` and `EXEC` are set (violates W^X).
    #[inline]
    pub const fn violates_wx(self) -> bool {
        self.contains(Self::WRITE) && self.contains(Self::EXEC)
    }

    /// Convert `VmFlags` into hardware x86_64 `PageFlags`.
    ///
    /// If `COW` is set, `WRITABLE` is intentionally omitted so writes trigger `#PF`.
    pub fn to_page_flags(self) -> PageFlags {
        let mut pf = PageFlags::PRESENT;
        if self.contains(Self::USER) {
            pf |= PageFlags::USER;
        }
        if self.contains(Self::WRITE) && !self.contains(Self::COW) {
            pf |= PageFlags::WRITABLE;
        }
        if !self.contains(Self::EXEC) {
            pf |= PageFlags::NO_EXECUTE;
        }
        pf
    }
}

impl BitOr for VmFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for VmFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for VmFlags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for VmFlags {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl Not for VmFlags {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

/// Backing source for a `VmArea`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VmBacking {
    Anonymous,
    ZeroFill,
    FileBacked { inode: u64, offset: u64, len: usize },
    Physical { phys_base: u64 },
}

/// Virtual Address Descriptor (VAD) / Virtual Memory Area (`VmArea`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VmArea {
    pub start: u64,
    pub end: u64,
    pub flags: VmFlags,
    pub backing: VmBacking,
    pub label: String,
}

impl VmArea {
    pub fn new(
        start: u64,
        end: u64,
        flags: VmFlags,
        backing: VmBacking,
        label: &str,
    ) -> Result<Self, ProcError> {
        let page_mask = (PAGE_SIZE as u64) - 1;
        if start >= end || (start & page_mask) != 0 || (end & page_mask) != 0 {
            return Err(ProcError::InvalidRange);
        }
        if flags.violates_wx() {
            return Err(ProcError::WxViolation);
        }
        Ok(Self {
            start,
            end,
            flags,
            backing,
            label: label.to_string(),
        })
    }

    #[inline]
    pub const fn contains(&self, addr: u64) -> bool {
        addr >= self.start && addr < self.end
    }

    #[inline]
    pub const fn overlaps(&self, start: u64, end: u64) -> bool {
        self.start < end && start < self.end
    }

    #[inline]
    pub const fn size(&self) -> u64 {
        self.end - self.start
    }
}

/// Per-process virtual address space descriptor (VAD list + PML4 root).
#[derive(Clone, Debug)]
pub struct VmSpace {
    pub address_space: Option<AddressSpace>,
    pub areas: Vec<VmArea>,
    /// Tracks physical frames mapped per page virtual address for CoW/cleanup.
    pub mapped_pages: BTreeMap<u64, u64>,
}

impl VmSpace {
    pub fn new(address_space: Option<AddressSpace>) -> Self {
        Self {
            address_space,
            areas: Vec::new(),
            mapped_pages: BTreeMap::new(),
        }
    }

    /// Insert a new `VmArea` after validating alignment, non-overlap, and W^X.
    pub fn insert_vma(&mut self, vma: VmArea) -> Result<(), ProcError> {
        if vma.flags.violates_wx() {
            return Err(ProcError::WxViolation);
        }
        if self.areas.iter().any(|a| a.overlaps(vma.start, vma.end)) {
            return Err(ProcError::Overlap);
        }
        self.areas.push(vma);
        self.areas.sort_by_key(|a| a.start);
        Ok(())
    }

    /// Find the `VmArea` containing `addr`.
    pub fn find_vma(&self, addr: u64) -> Option<&VmArea> {
        self.areas.iter().find(|a| a.contains(addr))
    }

    /// Find a mutable reference to the `VmArea` containing `addr`.
    pub fn find_vma_mut(&mut self, addr: u64) -> Option<&mut VmArea> {
        self.areas.iter_mut().find(|a| a.contains(addr))
    }

    /// Remove any `VmArea` matching `[start, end)` and unmap its pages.
    pub fn remove_vma(&mut self, start: u64, end: u64) -> Result<(), ProcError> {
        let idx = self
            .areas
            .iter()
            .position(|a| a.start == start && a.end == end)
            .ok_or(ProcError::InvalidRange)?;
        self.areas.remove(idx);

        let page_size = PAGE_SIZE as u64;
        let mut vaddr = start;
        while vaddr < end {
            if let Some(phys) = self.mapped_pages.remove(&vaddr) {
                if let Some(ref mut aspace) = self.address_space {
                    let _ = aspace.unmap_page(vaddr);
                }
                release_frame_ref(phys);
            }
            vaddr += page_size;
        }
        Ok(())
    }

    /// Update protection flags on an existing `VmArea` with W^X enforcement.
    pub fn protect_vma(
        &mut self,
        start: u64,
        end: u64,
        new_flags: VmFlags,
    ) -> Result<(), ProcError> {
        if new_flags.violates_wx() {
            return Err(ProcError::WxViolation);
        }
        let vma = self
            .areas
            .iter_mut()
            .find(|a| a.start == start && a.end == end)
            .ok_or(ProcError::InvalidRange)?;
        vma.flags = new_flags;

        let page_flags = new_flags.to_page_flags();
        if !enforce_wx(page_flags) {
            return Err(ProcError::WxViolation);
        }

        if let Some(ref mut aspace) = self.address_space {
            let page_size = PAGE_SIZE as u64;
            let mut vaddr = start;
            while vaddr < end {
                if let Some(&phys) = self.mapped_pages.get(&vaddr) {
                    aspace
                        .map_page(vaddr, phys, page_flags)
                        .map_err(|_| ProcError::OutOfMemory)?;
                }
                vaddr += page_size;
            }
        }
        Ok(())
    }

    /// Release all mapped pages and clear VMAs.
    pub fn clear(&mut self) {
        let pages: Vec<(u64, u64)> = self.mapped_pages.iter().map(|(&v, &p)| (v, p)).collect();
        for (vaddr, phys) in pages {
            if let Some(ref mut aspace) = self.address_space {
                let _ = aspace.unmap_page(vaddr);
            }
            release_frame_ref(phys);
        }
        self.mapped_pages.clear();
        self.areas.clear();
    }
}

/// Global physical frame reference counts for Copy-on-Write sharing.
static FRAME_REFS: SpinLock<BTreeMap<u64, usize>> = SpinLock::new(BTreeMap::new());

fn add_frame_ref(phys: u64) {
    if phys == 0 {
        return;
    }
    let mut refs = FRAME_REFS.lock();
    let count = refs.entry(phys).or_insert(0);
    *count += 1;
}

fn get_frame_ref(phys: u64) -> usize {
    FRAME_REFS.lock().get(&phys).copied().unwrap_or(1)
}

fn release_frame_ref(phys: u64) {
    if phys == 0 {
        return;
    }
    let should_free = {
        let mut refs = FRAME_REFS.lock();
        if let Some(count) = refs.get_mut(&phys) {
            if *count > 1 {
                *count -= 1;
                false
            } else {
                refs.remove(&phys);
                true
            }
        } else {
            true
        }
    };
    if should_free {
        dealloc_frame(phys);
    }
}

/// Outcome of handling a page fault.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FaultResolution {
    DemandMapped { page_vaddr: u64, phys: u64 },
    CowCopied { page_vaddr: u64, new_phys: u64 },
    CowPromoted { page_vaddr: u64, phys: u64 },
}

/// Per-thread TLS base registers (`FS_BASE` and `GS_BASE`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TlsState {
    pub fs_base: u64,
    pub gs_base: u64,
}

impl TlsState {
    pub const fn new(fs_base: u64, gs_base: u64) -> Self {
        Self { fs_base, gs_base }
    }

    /// Program `IA32_FS_BASE` and `IA32_KERNEL_GS_BASE` MSRs for Ring 3 execution.
    ///
    /// # Safety
    /// Must be called in Ring 0 on x86_64 with valid canonical user addresses.
    pub unsafe fn apply_msrs(&self) {
        wrmsr(IA32_FS_BASE, self.fs_base);
        wrmsr(IA32_KERNEL_GS_BASE, self.gs_base);
    }
}

/// POSIX signal numbers.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PosixSignal {
    Sighup = 1,
    Sigint = 2,
    Sigquit = 3,
    Sigill = 4,
    Sigtrap = 5,
    Sigabrt = 6,
    Sigbus = 7,
    Sigfpe = 8,
    Sigkill = 9,
    Sigusr1 = 10,
    Sigsegv = 11,
    Sigusr2 = 12,
    Sigpipe = 13,
    Sigalrm = 14,
    Sigterm = 15,
    Sigchld = 17,
    Sigcont = 18,
    Sigstop = 19,
}

impl PosixSignal {
    #[inline]
    pub const fn bit(self) -> u64 {
        1u64 << ((self as u32) & 63)
    }

    #[inline]
    pub const fn is_unmaskable(self) -> bool {
        matches!(self, Self::Sigkill | Self::Sigstop)
    }
}

/// Windows NT Structured Exception Handling (SEH) exception types.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SehException {
    AccessViolation { addr: u64, write: bool },
    IllegalInstruction,
    DivideByZero,
    Breakpoint,
    StackOverflow,
    Custom(u32),
}

/// Darwin / XNU Mach exception types.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MachException {
    BadAccess { addr: u64 },
    BadInstruction,
    Arithmetic,
    Breakpoint,
    Software { code: u64 },
}

/// Unified exception/signal event queued for a process or thread.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExceptionKind {
    Posix(PosixSignal),
    Seh(SehException),
    Mach(MachException),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExceptionEvent {
    pub kind: ExceptionKind,
    pub fault_rip: u64,
    pub fault_addr: u64,
    pub target_tid: Option<Tid>,
}

/// Disposition for a POSIX signal.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SignalAction {
    Default,
    Ignore,
    Handler(u64),
}

/// Per-process signal, SEH, and Mach-exception queue.
#[derive(Clone, Debug)]
pub struct ExceptionQueue {
    pub pending: VecDeque<ExceptionEvent>,
    pub blocked_mask: u64,
    pub handlers: BTreeMap<u32, SignalAction>,
}

impl ExceptionQueue {
    pub const fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            blocked_mask: 0,
            handlers: BTreeMap::new(),
        }
    }

    pub fn enqueue(&mut self, event: ExceptionEvent) {
        if let ExceptionKind::Posix(sig) = event.kind {
            if let Some(SignalAction::Ignore) = self.handlers.get(&(sig as u32)) {
                if !sig.is_unmaskable() {
                    return;
                }
            }
        }
        self.pending.push_back(event);
    }

    /// Dequeue the next deliverable exception for `tid` (respecting `blocked_mask`).
    pub fn dequeue_for_thread(&mut self, tid: Tid) -> Option<ExceptionEvent> {
        let mut idx = 0;
        while idx < self.pending.len() {
            let ev = self.pending[idx];
            let tid_matches = ev.target_tid.is_none() || ev.target_tid == Some(tid);
            let unblocked = match ev.kind {
                ExceptionKind::Posix(sig) => {
                    sig.is_unmaskable() || (self.blocked_mask & sig.bit()) == 0
                }
                ExceptionKind::Seh(_) | ExceptionKind::Mach(_) => true,
            };
            if tid_matches && unblocked {
                return self.pending.remove(idx);
            }
            idx += 1;
        }
        None
    }

    pub fn set_action(&mut self, sig: PosixSignal, action: SignalAction) {
        if !sig.is_unmaskable() {
            self.handlers.insert(sig as u32, action);
        }
    }

    pub fn set_mask(&mut self, mask: u64) {
        let unmaskable = PosixSignal::Sigkill.bit() | PosixSignal::Sigstop.bit();
        self.blocked_mask = mask & !unmaskable;
    }
}

/// Lifecycle state of a process.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProcessState {
    Created,
    Ready,
    Running,
    Blocked,
    Zombie(i32),
    Exited(i32),
}

/// Lifecycle state of a thread.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ThreadState {
    Ready,
    Running,
    Blocked,
    Exited,
}

/// Thread Control Block (TCB) belonging to a `Process`.
#[derive(Clone, Debug)]
pub struct Thread {
    pub tid: Tid,
    pub pid: Pid,
    pub state: ThreadState,
    pub cpu_context: CpuContext,
    pub user_context: UserContext,
    pub tls: TlsState,
    pub kernel_stack_top: u64,
}

impl Thread {
    pub fn new(tid: Tid, pid: Pid, personality: Personality) -> Self {
        Self {
            tid,
            pid,
            state: ThreadState::Ready,
            cpu_context: CpuContext::default(),
            user_context: UserContext::new(0, USER_STACK_TOP).with_personality(personality),
            tls: TlsState::default(),
            kernel_stack_top: 0,
        }
    }
}

/// Process Control Block (PCB).
pub struct Process {
    pub pid: Pid,
    pub parent: Option<Pid>,
    pub children: Vec<Pid>,
    pub name: String,
    pub personality: Personality,
    pub state: ProcessState,
    pub exit_code: Option<i32>,
    pub vm: VmSpace,
    pub threads: Vec<Tid>,
    pub main_tid: Tid,
    pub cap_table: CapTable,
    pub exceptions: ExceptionQueue,
    pub entry_rip: u64,
    pub user_rsp: u64,
}

/// Target specification for `wait`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WaitTarget {
    Any,
    Pid(Pid),
}

/// Result of `wait`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WaitResult {
    Exited { pid: Pid, status: i32 },
    WouldBlock,
}

/// Metadata returned by `exec`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExecImageInfo {
    pub format: BinaryFormat,
    pub personality: Personality,
    pub entry_rip: u64,
    pub user_rsp: u64,
}

/// Global Process & Address Space Manager.
pub static PROCESS_MANAGER: SpinLock<ProcessManager> = SpinLock::new(ProcessManager::new());

pub struct ProcessManager {
    processes: BTreeMap<Pid, Process>,
    threads: BTreeMap<Tid, Thread>,
    next_pid: u64,
    next_tid: u64,
    current_pid: Option<Pid>,
    current_tid: Option<Tid>,
    initialized: bool,
}

impl ProcessManager {
    pub const fn new() -> Self {
        Self {
            processes: BTreeMap::new(),
            threads: BTreeMap::new(),
            next_pid: 1,
            next_tid: 1,
            current_pid: None,
            current_tid: None,
            initialized: false,
        }
    }

    /// Initialize the process manager and create PID 1 (`init`).
    pub fn init(&mut self) {
        if self.initialized {
            return;
        }
        self.processes.clear();
        self.threads.clear();
        self.next_pid = 2;
        self.next_tid = 2;
        self.initialized = true;

        let init_pid = Pid::INIT;
        let init_tid = Tid(1);
        let init_thread = Thread::new(init_tid, init_pid, Personality::Native);
        let init_proc = Process {
            pid: init_pid,
            parent: None,
            children: Vec::new(),
            name: "init".to_string(),
            personality: Personality::Native,
            state: ProcessState::Running,
            exit_code: None,
            vm: VmSpace::new(None),
            threads: alloc::vec![init_tid],
            main_tid: init_tid,
            cap_table: CapTable::new(),
            exceptions: ExceptionQueue::new(),
            entry_rip: 0,
            user_rsp: USER_STACK_TOP,
        };

        self.threads.insert(init_tid, init_thread);
        self.processes.insert(init_pid, init_proc);
        self.current_pid = Some(init_pid);
        self.current_tid = Some(init_tid);
    }

    fn alloc_pid(&mut self) -> Pid {
        let pid = Pid(self.next_pid);
        self.next_pid += 1;
        pid
    }

    fn alloc_tid(&mut self) -> Tid {
        let tid = Tid(self.next_tid);
        self.next_tid += 1;
        tid
    }

    pub fn current_pid(&self) -> Option<Pid> {
        self.current_pid
    }

    pub fn current_tid(&self) -> Option<Tid> {
        self.current_tid
    }

    pub fn set_current(&mut self, pid: Pid, tid: Tid) -> Result<(), ProcError> {
        if !self.processes.contains_key(&pid) {
            return Err(ProcError::ProcessNotFound);
        }
        if !self.threads.contains_key(&tid) {
            return Err(ProcError::ThreadNotFound);
        }
        self.current_pid = Some(pid);
        self.current_tid = Some(tid);
        Ok(())
    }

    pub fn process_count(&self) -> usize {
        self.processes.len()
    }

    pub fn thread_count(&self) -> usize {
        self.threads.len()
    }

    pub fn get_process(&self, pid: Pid) -> Option<&Process> {
        self.processes.get(&pid)
    }

    pub fn get_process_mut(&mut self, pid: Pid) -> Option<&mut Process> {
        self.processes.get_mut(&pid)
    }

    pub fn get_thread(&self, tid: Tid) -> Option<&Thread> {
        self.threads.get(&tid)
    }

    pub fn get_thread_mut(&mut self, tid: Tid) -> Option<&mut Thread> {
        self.threads.get_mut(&tid)
    }

    /// Create a new process with an optional parent and fresh user `AddressSpace` (if frames exist).
    pub fn create_process(
        &mut self,
        name: &str,
        personality: Personality,
        parent: Option<Pid>,
        with_address_space: bool,
    ) -> Result<Pid, ProcError> {
        if !self.initialized {
            self.init();
        }

        let aspace = if with_address_space {
            Some(AddressSpace::new_user().ok_or(ProcError::OutOfMemory)?)
        } else {
            None
        };

        let pid = self.alloc_pid();
        let tid = self.alloc_tid();
        let thread = Thread::new(tid, pid, personality);

        let proc = Process {
            pid,
            parent,
            children: Vec::new(),
            name: name.to_string(),
            personality,
            state: ProcessState::Ready,
            exit_code: None,
            vm: VmSpace::new(aspace),
            threads: alloc::vec![tid],
            main_tid: tid,
            cap_table: CapTable::new(),
            exceptions: ExceptionQueue::new(),
            entry_rip: 0,
            user_rsp: USER_STACK_TOP,
        };

        self.threads.insert(tid, thread);
        self.processes.insert(pid, proc);

        if let Some(ppid) = parent {
            if let Some(p) = self.processes.get_mut(&ppid) {
                p.children.push(pid);
            }
        }

        Ok(pid)
    }

    /// Spawn an additional thread inside an existing process.
    pub fn spawn_thread(
        &mut self,
        pid: Pid,
        entry_rip: u64,
        user_rsp: u64,
    ) -> Result<Tid, ProcError> {
        let personality = self
            .processes
            .get(&pid)
            .ok_or(ProcError::ProcessNotFound)?
            .personality;
        let tid = self.alloc_tid();
        let mut thread = Thread::new(tid, pid, personality);
        thread.user_context = UserContext::new(entry_rip, user_rsp).with_personality(personality);
        self.threads.insert(tid, thread);
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.threads.push(tid);
        }
        Ok(tid)
    }

    /// Activate the process's PML4 root in CR3 (if it has an `AddressSpace`).
    pub fn activate_address_space(&self, pid: Pid) -> Result<(), ProcError> {
        let proc = self.processes.get(&pid).ok_or(ProcError::ProcessNotFound)?;
        if let Some(ref aspace) = proc.vm.address_space {
            aspace.activate();
        }
        Ok(())
    }

    /// Handle a `#PF` page fault for `pid` at `fault_addr` with x86_64 `error_code`.
    pub fn handle_page_fault(
        &mut self,
        pid: Pid,
        fault_addr: u64,
        error_code: u64,
    ) -> Result<FaultResolution, ProcError> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ProcError::ProcessNotFound)?;

        let page_size = PAGE_SIZE as u64;
        let page_vaddr = fault_addr & !(page_size - 1);
        let is_present = (error_code & PF_ERR_PRESENT) != 0;
        let is_write = (error_code & PF_ERR_WRITE) != 0;
        let is_exec = (error_code & PF_ERR_INSTR_FETCH) != 0;

        let vma = proc
            .vm
            .find_vma(fault_addr)
            .cloned()
            .ok_or(ProcError::Segfault)?;

        if vma.flags.contains(VmFlags::GUARD) {
            return Err(ProcError::StackOverflow);
        }
        if is_exec && !vma.flags.contains(VmFlags::EXEC) {
            return Err(ProcError::ProtectionFault);
        }
        if !is_write && !is_exec && !vma.flags.contains(VmFlags::READ) {
            return Err(ProcError::ProtectionFault);
        }

        // Copy-on-Write write fault on a present page.
        if is_present && is_write && vma.flags.contains(VmFlags::COW) {
            let old_phys = proc
                .vm
                .mapped_pages
                .get(&page_vaddr)
                .copied()
                .ok_or(ProcError::ProtectionFault)?;

            let writable_flags = (vma.flags & !VmFlags::COW).to_page_flags();
            if !enforce_wx(writable_flags) {
                return Err(ProcError::WxViolation);
            }

            if get_frame_ref(old_phys) <= 1 {
                if let Some(ref mut aspace) = proc.vm.address_space {
                    aspace
                        .map_page(page_vaddr, old_phys, writable_flags)
                        .map_err(|_| ProcError::OutOfMemory)?;
                }
                return Ok(FaultResolution::CowPromoted {
                    page_vaddr,
                    phys: old_phys,
                });
            }

            let new_phys = alloc_frame().ok_or(ProcError::OutOfMemory)?;
            let hhdm = stats().hhdm_offset;
            unsafe {
                let src = hhdm.wrapping_add(old_phys) as *const u8;
                let dst = hhdm.wrapping_add(new_phys) as *mut u8;
                ptr::copy_nonoverlapping(src, dst, PAGE_SIZE);
            }

            add_frame_ref(new_phys);
            release_frame_ref(old_phys);
            proc.vm.mapped_pages.insert(page_vaddr, new_phys);

            if let Some(ref mut aspace) = proc.vm.address_space {
                aspace
                    .map_page(page_vaddr, new_phys, writable_flags)
                    .map_err(|_| ProcError::OutOfMemory)?;
            }

            return Ok(FaultResolution::CowCopied {
                page_vaddr,
                new_phys,
            });
        }

        if is_write && !vma.flags.contains(VmFlags::WRITE) {
            return Err(ProcError::ProtectionFault);
        }

        // Demand-paging fault on a non-present page.
        if !is_present && vma.flags.contains(VmFlags::DEMAND) {
            let phys = alloc_frame().ok_or(ProcError::OutOfMemory)?;
            let hhdm = stats().hhdm_offset;
            unsafe {
                let dst = hhdm.wrapping_add(phys) as *mut u8;
                ptr::write_bytes(dst, 0, PAGE_SIZE);
            }
            add_frame_ref(phys);
            proc.vm.mapped_pages.insert(page_vaddr, phys);

            let page_flags = vma.flags.to_page_flags();
            if !enforce_wx(page_flags) {
                return Err(ProcError::WxViolation);
            }
            if let Some(ref mut aspace) = proc.vm.address_space {
                aspace
                    .map_page(page_vaddr, phys, page_flags)
                    .map_err(|_| ProcError::OutOfMemory)?;
            }

            return Ok(FaultResolution::DemandMapped { page_vaddr, phys });
        }

        Err(ProcError::ProtectionFault)
    }

    /// Fork `parent_pid` into a new child process with Copy-on-Write private VMAs.
    pub fn fork(&mut self, parent_pid: Pid) -> Result<Pid, ProcError> {
        let (
            parent_name,
            personality,
            entry_rip,
            user_rsp,
            parent_main_tid,
            has_aspace,
            parent_areas,
            parent_pages,
            blocked_mask,
            handlers,
        ) = {
            let parent = self
                .processes
                .get(&parent_pid)
                .ok_or(ProcError::ProcessNotFound)?;
            if matches!(
                parent.state,
                ProcessState::Zombie(_) | ProcessState::Exited(_)
            ) {
                return Err(ProcError::InvalidState);
            }
            (
                parent.name.clone(),
                parent.personality,
                parent.entry_rip,
                parent.user_rsp,
                parent.main_tid,
                parent.vm.address_space.is_some(),
                parent.vm.areas.clone(),
                parent.vm.mapped_pages.clone(),
                parent.exceptions.blocked_mask,
                parent.exceptions.handlers.clone(),
            )
        };

        let child_aspace = if has_aspace {
            Some(AddressSpace::new_user().ok_or(ProcError::OutOfMemory)?)
        } else {
            None
        };

        let child_pid = self.alloc_pid();
        let child_tid = self.alloc_tid();

        let mut child_thread = if let Some(pt) = self.threads.get(&parent_main_tid) {
            let mut t = pt.clone();
            t.tid = child_tid;
            t.pid = child_pid;
            t.state = ThreadState::Ready;
            t
        } else {
            Thread::new(child_tid, child_pid, personality)
        };
        child_thread.state = ThreadState::Ready;

        let mut child_vm = VmSpace::new(child_aspace);
        let mut updated_parent_areas = Vec::with_capacity(parent_areas.len());

        for mut area in parent_areas {
            if area.flags.contains(VmFlags::WRITE) && !area.flags.contains(VmFlags::SHARED) {
                area.flags |= VmFlags::COW;
            }
            updated_parent_areas.push(area.clone());
            child_vm.areas.push(area);
        }

        // Share mapped physical frames with CoW read-only PTEs.
        for (&vaddr, &phys) in &parent_pages {
            add_frame_ref(phys);
            child_vm.mapped_pages.insert(vaddr, phys);

            if let Some(vma) = child_vm.find_vma(vaddr).cloned() {
                let pf = vma.flags.to_page_flags();
                if let Some(ref mut cas) = child_vm.address_space {
                    cas.map_page(vaddr, phys, pf)
                        .map_err(|_| ProcError::OutOfMemory)?;
                }
                if let Some(parent) = self.processes.get_mut(&parent_pid) {
                    if let Some(ref mut pas) = parent.vm.address_space {
                        pas.map_page(vaddr, phys, pf)
                            .map_err(|_| ProcError::OutOfMemory)?;
                    }
                }
            }
        }

        if let Some(parent) = self.processes.get_mut(&parent_pid) {
            parent.vm.areas = updated_parent_areas;
            parent.children.push(child_pid);
        }

        let child_proc = Process {
            pid: child_pid,
            parent: Some(parent_pid),
            children: Vec::new(),
            name: parent_name,
            personality,
            state: ProcessState::Ready,
            exit_code: None,
            vm: child_vm,
            threads: alloc::vec![child_tid],
            main_tid: child_tid,
            cap_table: CapTable::new(),
            exceptions: ExceptionQueue {
                pending: VecDeque::new(),
                blocked_mask,
                handlers,
            },
            entry_rip,
            user_rsp,
        };

        self.threads.insert(child_tid, child_thread);
        self.processes.insert(child_pid, child_proc);

        Ok(child_pid)
    }

    /// Replace `pid`'s address space and image with a validated ELF64/PE32+/Mach-O binary.
    pub fn exec(
        &mut self,
        pid: Pid,
        name: &str,
        binary: &[u8],
    ) -> Result<ExecImageInfo, ProcError> {
        let image =
            inspect_binary(binary).map_err(|e| ProcError::InvalidBinary(compat_err_str(e)))?;

        let mut new_aspace = AddressSpace::new_user().ok_or(ProcError::OutOfMemory)?;
        let mut new_vm = VmSpace::new(Some(new_aspace));
        let hhdm = stats().hhdm_offset;
        let page_size = PAGE_SIZE as u64;

        for seg in &image.segments {
            if seg.virtual_size == 0 {
                continue;
            }

            let mut vm_flags = VmFlags::USER;
            if seg.readable {
                vm_flags |= VmFlags::READ;
            }
            if seg.writable {
                vm_flags |= VmFlags::WRITE;
            }
            if seg.executable {
                vm_flags |= VmFlags::EXEC;
            }
            if vm_flags.violates_wx() {
                new_vm.clear();
                return Err(ProcError::WxViolation);
            }

            if seg.file_size > 0 {
                let file_end = seg
                    .file_offset
                    .checked_add(seg.file_size)
                    .ok_or(ProcError::InvalidRange)?;
                if file_end as usize > binary.len() {
                    new_vm.clear();
                    return Err(ProcError::InvalidBinary("segment file data out of bounds"));
                }
            }

            let seg_start = seg.virtual_address & !(page_size - 1);
            let seg_end = seg
                .virtual_address
                .checked_add(seg.virtual_size)
                .ok_or(ProcError::InvalidRange)?;
            let aligned_end = (seg_end + page_size - 1) & !(page_size - 1);
            let file_vaddr_end = seg
                .virtual_address
                .checked_add(seg.file_size)
                .ok_or(ProcError::InvalidRange)?;

            let vma = VmArea::new(
                seg_start,
                aligned_end,
                vm_flags,
                VmBacking::Anonymous,
                seg.name.as_deref().unwrap_or("segment"),
            )?;
            new_vm.insert_vma(vma)?;

            let page_flags = vm_flags.to_page_flags();
            let mut vaddr = seg_start;
            while vaddr < aligned_end {
                let phys = if let Some(&existing) = new_vm.mapped_pages.get(&vaddr) {
                    existing
                } else {
                    let frame = alloc_frame().ok_or(ProcError::OutOfMemory)?;
                    unsafe {
                        ptr::write_bytes(hhdm.wrapping_add(frame) as *mut u8, 0, PAGE_SIZE);
                    }
                    add_frame_ref(frame);
                    new_vm.mapped_pages.insert(vaddr, frame);
                    new_aspace
                        .map_page(vaddr, frame, page_flags)
                        .map_err(|_| ProcError::OutOfMemory)?;
                    frame
                };

                if seg.file_size > 0 {
                    let page_end = vaddr + page_size;
                    let copy_start = vaddr.max(seg.virtual_address);
                    let copy_end = page_end.min(file_vaddr_end);
                    if copy_start < copy_end {
                        let dst_off = (copy_start - vaddr) as usize;
                        let src_off =
                            (seg.file_offset + (copy_start - seg.virtual_address)) as usize;
                        let len = (copy_end - copy_start) as usize;
                        unsafe {
                            let dst = (hhdm.wrapping_add(phys) as *mut u8).add(dst_off);
                            ptr::copy_nonoverlapping(binary[src_off..].as_ptr(), dst, len);
                        }
                    }
                }
                vaddr += page_size;
            }
        }

        let stack_flags =
            VmFlags::USER | VmFlags::READ | VmFlags::WRITE | VmFlags::STACK | VmFlags::DEMAND;
        let stack_vma = VmArea::new(
            USER_STACK_BASE,
            USER_STACK_TOP,
            stack_flags,
            VmBacking::ZeroFill,
            "user-stack",
        )?;
        new_vm.insert_vma(stack_vma)?;
        new_vm.address_space = Some(new_aspace);

        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ProcError::ProcessNotFound)?;
        proc.vm.clear();
        proc.vm = new_vm;
        proc.name = name.to_string();
        proc.personality = image.personality;
        proc.entry_rip = image.entry_point;
        proc.user_rsp = USER_STACK_TOP;
        proc.exceptions.pending.clear();
        proc.exceptions.handlers.clear();

        if let Some(thread) = self.threads.get_mut(&proc.main_tid) {
            thread.user_context = UserContext::new(image.entry_point, USER_STACK_TOP)
                .with_personality(image.personality);
            thread.tls = TlsState::default();
        }

        Ok(ExecImageInfo {
            format: image.format,
            personality: image.personality,
            entry_rip: image.entry_point,
            user_rsp: USER_STACK_TOP,
        })
    }

    /// Transition `pid` to `Zombie(exit_code)`, mark its threads exited, and reparent children to `init`.
    pub fn exit(&mut self, pid: Pid, exit_code: i32) -> Result<(), ProcError> {
        if pid == Pid::INIT {
            return Err(ProcError::InvalidState);
        }

        let (parent_pid, children, thread_ids) = {
            let proc = self
                .processes
                .get_mut(&pid)
                .ok_or(ProcError::ProcessNotFound)?;
            proc.state = ProcessState::Zombie(exit_code);
            proc.exit_code = Some(exit_code);
            proc.vm.clear();
            let children = core::mem::take(&mut proc.children);
            let tids = proc.threads.clone();
            (proc.parent, children, tids)
        };

        for tid in thread_ids {
            if let Some(t) = self.threads.get_mut(&tid) {
                t.state = ThreadState::Exited;
            }
        }

        // Reparent living children to PID 1 (init).
        for child_pid in children {
            if let Some(child) = self.processes.get_mut(&child_pid) {
                child.parent = Some(Pid::INIT);
            }
            if let Some(init_proc) = self.processes.get_mut(&Pid::INIT) {
                init_proc.children.push(child_pid);
            }
        }

        // Notify parent with SIGCHLD.
        if let Some(ppid) = parent_pid {
            if let Some(parent) = self.processes.get_mut(&ppid) {
                parent.exceptions.enqueue(ExceptionEvent {
                    kind: ExceptionKind::Posix(PosixSignal::Sigchld),
                    fault_rip: 0,
                    fault_addr: 0,
                    target_tid: None,
                });
            }
        }

        Ok(())
    }

    /// Wait for a child process of `parent_pid` to exit and reap its zombie state.
    pub fn wait(&mut self, parent_pid: Pid, target: WaitTarget) -> Result<WaitResult, ProcError> {
        let children = self
            .processes
            .get(&parent_pid)
            .ok_or(ProcError::ProcessNotFound)?
            .children
            .clone();

        let matching: Vec<Pid> = match target {
            WaitTarget::Any => children,
            WaitTarget::Pid(target_pid) => {
                if children.contains(&target_pid) {
                    alloc::vec![target_pid]
                } else {
                    return Err(ProcError::NoChild);
                }
            }
        };

        if matching.is_empty() {
            return Err(ProcError::NoChild);
        }

        for child_pid in matching {
            if let Some(child) = self.processes.get(&child_pid) {
                if let ProcessState::Zombie(status) = child.state {
                    let tids = child.threads.clone();
                    for tid in tids {
                        self.threads.remove(&tid);
                    }
                    self.processes.remove(&child_pid);
                    if let Some(parent) = self.processes.get_mut(&parent_pid) {
                        parent.children.retain(|&c| c != child_pid);
                    }
                    return Ok(WaitResult::Exited {
                        pid: child_pid,
                        status,
                    });
                }
            }
        }

        Ok(WaitResult::WouldBlock)
    }

    /// Set `FS_BASE` for a thread.
    pub fn set_fs_base(&mut self, tid: Tid, fs_base: u64) -> Result<(), ProcError> {
        let thread = self
            .threads
            .get_mut(&tid)
            .ok_or(ProcError::ThreadNotFound)?;
        thread.tls.fs_base = fs_base;
        Ok(())
    }

    /// Set `GS_BASE` for a thread.
    pub fn set_gs_base(&mut self, tid: Tid, gs_base: u64) -> Result<(), ProcError> {
        let thread = self
            .threads
            .get_mut(&tid)
            .ok_or(ProcError::ThreadNotFound)?;
        thread.tls.gs_base = gs_base;
        Ok(())
    }

    /// Read TLS state for a thread.
    pub fn get_tls(&self, tid: Tid) -> Result<TlsState, ProcError> {
        self.threads
            .get(&tid)
            .map(|t| t.tls)
            .ok_or(ProcError::ThreadNotFound)
    }

    /// Enqueue a POSIX signal, SEH exception, or Mach exception on `pid`.
    pub fn enqueue_exception(&mut self, pid: Pid, event: ExceptionEvent) -> Result<(), ProcError> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ProcError::ProcessNotFound)?;
        proc.exceptions.enqueue(event);
        Ok(())
    }

    /// Dequeue the next deliverable exception for `(pid, tid)`.
    pub fn dequeue_exception(
        &mut self,
        pid: Pid,
        tid: Tid,
    ) -> Result<Option<ExceptionEvent>, ProcError> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ProcError::ProcessNotFound)?;
        Ok(proc.exceptions.dequeue_for_thread(tid))
    }
}

#[inline]
unsafe fn wrmsr(msr: u32, value: u64) {
    let low = value as u32;
    let high = (value >> 32) as u32;
    asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") low,
        in("edx") high,
        options(nomem, nostack, preserves_flags)
    );
}

const fn compat_err_str(err: CompatError) -> &'static str {
    match err {
        CompatError::TooShort => "binary too short",
        CompatError::UnknownFormat => "unknown binary format",
        CompatError::Elf(_) => "invalid ELF64 binary",
        CompatError::Pe(_) => "invalid PE32+ binary",
        CompatError::MachO(_) => "invalid Mach-O 64-bit binary",
    }
}

/// Run an in-kernel self-test covering PID/TID, VMA W^X, demand paging, CoW,
/// fork/exit/wait, TLS FS_BASE/GS_BASE, and signal/SEH/Mach-exception queues.
pub fn self_test() -> Result<(), &'static str> {
    let mut mgr = PROCESS_MANAGER.lock();
    mgr.init();

    let has_frames = stats().total_frames > 0;

    // 1. Create parent process.
    let parent_pid = mgr
        .create_process(
            "proc-selftest",
            Personality::Posix,
            Some(Pid::INIT),
            has_frames,
        )
        .map_err(|_| "failed to create test process")?;

    // 2. VMA W^X enforcement & overlap rejection.
    {
        let parent = mgr
            .get_process_mut(parent_pid)
            .ok_or("missing parent process")?;

        let wx_res = VmArea::new(
            0x40_0000,
            0x40_1000,
            VmFlags::USER | VmFlags::READ | VmFlags::WRITE | VmFlags::EXEC,
            VmBacking::Anonymous,
            "bad-wx",
        );
        if wx_res != Err(ProcError::WxViolation) {
            return Err("W^X violation was not rejected");
        }

        let demand_vma = VmArea::new(
            0x40_0000,
            0x40_2000,
            VmFlags::USER | VmFlags::READ | VmFlags::WRITE | VmFlags::DEMAND,
            VmBacking::ZeroFill,
            "test-data",
        )
        .map_err(|_| "failed to construct valid VMA")?;
        parent
            .vm
            .insert_vma(demand_vma)
            .map_err(|_| "failed to insert VMA")?;

        let overlap_vma = VmArea::new(
            0x40_1000,
            0x40_3000,
            VmFlags::USER | VmFlags::READ,
            VmBacking::Anonymous,
            "overlap",
        )
        .map_err(|_| "failed to construct overlap VMA")?;
        if parent.vm.insert_vma(overlap_vma) != Err(ProcError::Overlap) {
            return Err("overlapping VMA was not rejected");
        }
    }

    // 3. Demand-paging fault & CoW fork test (when physical frames are available).
    if has_frames {
        let res = mgr
            .handle_page_fault(parent_pid, 0x40_0100, PF_ERR_USER | PF_ERR_WRITE)
            .map_err(|_| "demand-paging fault failed")?;
        if !matches!(
            res,
            FaultResolution::DemandMapped {
                page_vaddr: 0x40_0000,
                ..
            }
        ) {
            return Err("unexpected demand-paging resolution");
        }
    }

    // 4. Fork -> Wait (WouldBlock) -> CoW write -> Exit -> Wait (Reap).
    let child_pid = mgr.fork(parent_pid).map_err(|_| "fork failed")?;

    if mgr.wait(parent_pid, WaitTarget::Pid(child_pid)) != Ok(WaitResult::WouldBlock) {
        return Err("wait on running child did not return WouldBlock");
    }

    if has_frames {
        let cow_res = mgr
            .handle_page_fault(
                child_pid,
                0x40_0100,
                PF_ERR_PRESENT | PF_ERR_WRITE | PF_ERR_USER,
            )
            .map_err(|_| "CoW fault failed")?;
        if !matches!(
            cow_res,
            FaultResolution::CowCopied {
                page_vaddr: 0x40_0000,
                ..
            }
        ) {
            return Err("expected CowCopied resolution");
        }
    }

    // 5. TLS FS_BASE / GS_BASE state.
    let child_tid = mgr
        .get_process(child_pid)
        .ok_or("missing child process")?
        .main_tid;
    mgr.set_fs_base(child_tid, 0x7000_1000)
        .map_err(|_| "set_fs_base failed")?;
    mgr.set_gs_base(child_tid, 0x7000_2000)
        .map_err(|_| "set_gs_base failed")?;
    let tls = mgr.get_tls(child_tid).map_err(|_| "get_tls failed")?;
    if tls.fs_base != 0x7000_1000 || tls.gs_base != 0x7000_2000 {
        return Err("TLS FS_BASE/GS_BASE mismatch");
    }

    // 6. Signal / SEH / Mach-exception queue.
    {
        let child = mgr
            .get_process_mut(child_pid)
            .ok_or("missing child for exception test")?;
        child.exceptions.set_mask(PosixSignal::Sigusr1.bit());
        child.exceptions.enqueue(ExceptionEvent {
            kind: ExceptionKind::Posix(PosixSignal::Sigusr1),
            fault_rip: 0,
            fault_addr: 0,
            target_tid: None,
        });
        child.exceptions.enqueue(ExceptionEvent {
            kind: ExceptionKind::Seh(SehException::AccessViolation {
                addr: 0xDEAD_BEEF,
                write: true,
            }),
            fault_rip: 0x1000,
            fault_addr: 0xDEAD_BEEF,
            target_tid: Some(child_tid),
        });
        child.exceptions.enqueue(ExceptionEvent {
            kind: ExceptionKind::Mach(MachException::BadAccess { addr: 0xCAFE_BABE }),
            fault_rip: 0x2000,
            fault_addr: 0xCAFE_BABE,
            target_tid: None,
        });
    }

    let ev1 = mgr
        .dequeue_exception(child_pid, child_tid)
        .map_err(|_| "dequeue SEH failed")?;
    if !matches!(
        ev1,
        Some(ExceptionEvent {
            kind: ExceptionKind::Seh(SehException::AccessViolation { .. }),
            ..
        })
    ) {
        return Err("expected SEH exception while SIGUSR1 is masked");
    }

    let ev2 = mgr
        .dequeue_exception(child_pid, child_tid)
        .map_err(|_| "dequeue Mach failed")?;
    if !matches!(
        ev2,
        Some(ExceptionEvent {
            kind: ExceptionKind::Mach(MachException::BadAccess { .. }),
            ..
        })
    ) {
        return Err("expected Mach exception");
    }

    // 7. Exit & Wait state machine.
    mgr.exit(child_pid, 42).map_err(|_| "child exit failed")?;
    let wait_res = mgr
        .wait(parent_pid, WaitTarget::Pid(child_pid))
        .map_err(|_| "wait failed")?;
    if wait_res
        != (WaitResult::Exited {
            pid: child_pid,
            status: 42,
        })
    {
        return Err("wait did not reap child with expected exit code");
    }

    // Clean up parent test process.
    mgr.exit(parent_pid, 0).map_err(|_| "parent exit failed")?;
    let _ = mgr.wait(Pid::INIT, WaitTarget::Pid(parent_pid));

    Ok(())
}
