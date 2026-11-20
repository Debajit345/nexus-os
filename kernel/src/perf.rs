//! Kernel Tracing, PMU Performance Counters & DTrace/eBPF Probe Engine.
//!
//! Provides:
//! - x86_64 Architectural PMU capability probing (CPUID leaf 0x0A), hardware/software
//!   performance counters, and `rdpmc`/`rdtsc` sampling
//! - Lock-protected kernel tracing ring buffer for kprobes, kretprobes, tracepoints,
//!   uprobes, and timer samples
//! - Kprobe & tracepoint registry with dynamic enable/disable and eBPF hook attachment
//! - Verified `no_std` eBPF-style register VM (11 x 64-bit registers, DAG-verified
//!   forward-only jumps, bounded stack, and key-value aggregation maps)
//! - Frame-pointer kernel stack sampler (`capture_kernel_stack`)
//! - `self_test()` verification suite
#![allow(dead_code)]

use core::arch::asm;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::mm::SpinLock;
use crate::serial_println;
use crate::time::rdtsc;

// ---------------------------------------------------------------------------
// Constants & Limits
// ---------------------------------------------------------------------------

pub const MAX_PMU_COUNTERS: usize = 8;
pub const TRACE_RING_CAP: usize = 256;
pub const STACK_RING_CAP: usize = 64;
pub const MAX_PROBES: usize = 32;
pub const MAX_BPF_PROGS: usize = 16;
pub const MAX_BPF_INSNS: usize = 64;
pub const MAX_BPF_MAPS: usize = 8;
pub const BPF_MAP_ENTRIES: usize = 32;
pub const MAX_STACK_DEPTH: usize = 16;
pub const BPF_STACK_WORDS: usize = 32; // 256 bytes

// x86_64 Architectural PMU MSRs
const MSR_IA32_PMC0: u32 = 0x00C1;
const MSR_IA32_PERFEVTSEL0: u32 = 0x0186;
const MSR_IA32_FIXED_CTR0: u32 = 0x0309;
const MSR_IA32_FIXED_CTR_CTRL: u32 = 0x038D;
const MSR_IA32_PERF_GLOBAL_STATUS: u32 = 0x038E;
const MSR_IA32_PERF_GLOBAL_CTRL: u32 = 0x038F;

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PerfError {
    NotInitialized,
    NoFreeSlot,
    InvalidId,
    AlreadyExists,
    InvalidConfig,
    VerifierEmptyProgram,
    VerifierTooLarge,
    VerifierMissingExit,
    VerifierInvalidRegister,
    VerifierReadOnlyRegister,
    VerifierInvalidJump,
    VerifierInvalidHelper,
    VerifierInvalidMemoryOffset,
    MapFull,
    NotFound,
}

// ---------------------------------------------------------------------------
// Low-level x86_64 CPUID / MSR / RDPMC Helpers
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Default)]
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
fn max_basic_cpuid_leaf() -> u32 {
    cpuid(0, 0).eax
}

#[inline]
unsafe fn rdmsr(msr: u32) -> u64 {
    let low: u32;
    let high: u32;
    asm!(
        "rdmsr",
        in("ecx") msr,
        out("eax") low,
        out("edx") high,
        options(nomem, nostack, preserves_flags),
    );
    ((high as u64) << 32) | (low as u64)
}

#[inline]
unsafe fn wrmsr(msr: u32, val: u64) {
    let low = val as u32;
    let high = (val >> 32) as u32;
    asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") low,
        in("edx") high,
        options(nomem, nostack, preserves_flags),
    );
}

#[inline]
unsafe fn rdpmc(counter: u32) -> u64 {
    let low: u32;
    let high: u32;
    asm!(
        "rdpmc",
        in("ecx") counter,
        out("eax") low,
        out("edx") high,
        options(nomem, nostack, preserves_flags),
    );
    ((high as u64) << 32) | (low as u64)
}

// ---------------------------------------------------------------------------
// PMU Performance Counters
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PmuCapabilities {
    pub supported: bool,
    pub version: u8,
    pub num_gp_counters: u8,
    pub gp_bit_width: u8,
    pub num_fixed_counters: u8,
    pub fixed_bit_width: u8,
    pub available_events_mask: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PmuEvent {
    CpuCycles,
    InstructionsRetired,
    CacheReferences,
    CacheMisses,
    BranchInstructions,
    BranchMisses,
    BusCycles,
    ContextSwitches,
    PageFaults,
    SyscallCount,
    CustomRaw(u32),
}

impl PmuEvent {
    pub const fn raw_evtsel(self) -> u32 {
        match self {
            Self::CpuCycles => 0x003C,
            Self::InstructionsRetired => 0x00C0,
            Self::CacheReferences => 0x4F2E,
            Self::CacheMisses => 0x412E,
            Self::BranchInstructions => 0x00C4,
            Self::BranchMisses => 0x00C5,
            Self::BusCycles => 0x013C,
            Self::ContextSwitches => 0x8000_0001,
            Self::PageFaults => 0x8000_0002,
            Self::SyscallCount => 0x8000_0003,
            Self::CustomRaw(raw) => raw,
        }
    }

    pub const fn is_software(self) -> bool {
        matches!(
            self,
            Self::ContextSwitches | Self::PageFaults | Self::SyscallCount
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PmuCounterConfig {
    pub event: PmuEvent,
    pub user_mode: bool,
    pub kernel_mode: bool,
    pub edge_detect: bool,
    pub sample_period: u64,
}

impl Default for PmuCounterConfig {
    fn default() -> Self {
        Self {
            event: PmuEvent::CpuCycles,
            user_mode: true,
            kernel_mode: true,
            edge_detect: false,
            sample_period: 0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PmuCounterState {
    pub allocated: bool,
    pub enabled: bool,
    pub hw_backed: bool,
    pub id: u8,
    pub config: PmuCounterConfig,
    pub value: u64,
    pub overflows: u64,
    pub last_tsc: u64,
}

impl PmuCounterState {
    const fn empty(id: u8) -> Self {
        Self {
            allocated: false,
            enabled: false,
            hw_backed: false,
            id,
            config: PmuCounterConfig {
                event: PmuEvent::CpuCycles,
                user_mode: true,
                kernel_mode: true,
                edge_detect: false,
                sample_period: 0,
            },
            value: 0,
            overflows: 0,
            last_tsc: 0,
        }
    }
}

struct PmuManager {
    caps: PmuCapabilities,
    counters: [PmuCounterState; MAX_PMU_COUNTERS],
}

impl PmuManager {
    const fn new() -> Self {
        Self {
            caps: PmuCapabilities {
                supported: false,
                version: 0,
                num_gp_counters: 0,
                gp_bit_width: 0,
                num_fixed_counters: 0,
                fixed_bit_width: 0,
                available_events_mask: 0,
            },
            counters: [
                PmuCounterState::empty(0),
                PmuCounterState::empty(1),
                PmuCounterState::empty(2),
                PmuCounterState::empty(3),
                PmuCounterState::empty(4),
                PmuCounterState::empty(5),
                PmuCounterState::empty(6),
                PmuCounterState::empty(7),
            ],
        }
    }

    fn probe(&mut self) -> PmuCapabilities {
        let max_leaf = max_basic_cpuid_leaf();
        if max_leaf < 0x0A {
            self.caps = PmuCapabilities::default();
            return self.caps;
        }
        let regs = cpuid(0x0A, 0);
        let version = (regs.eax & 0xFF) as u8;
        let num_gp = ((regs.eax >> 8) & 0xFF) as u8;
        let gp_width = ((regs.eax >> 16) & 0xFF) as u8;
        let num_fixed = (regs.edx & 0x1F) as u8;
        let fixed_width = ((regs.edx >> 5) & 0xFF) as u8;
        let supported = version >= 1 && num_gp > 0 && gp_width >= 32;

        self.caps = PmuCapabilities {
            supported,
            version,
            num_gp_counters: num_gp,
            gp_bit_width: gp_width,
            num_fixed_counters: num_fixed,
            fixed_bit_width: fixed_width,
            available_events_mask: !regs.ebx,
        };
        self.caps
    }

    fn allocate(&mut self, config: PmuCounterConfig) -> Result<u8, PerfError> {
        for slot in self.counters.iter_mut() {
            if !slot.allocated {
                let hw_backed = self.caps.supported
                    && !config.event.is_software()
                    && (slot.id as usize) < (self.caps.num_gp_counters as usize);
                slot.allocated = true;
                slot.enabled = false;
                slot.hw_backed = hw_backed;
                slot.config = config;
                slot.value = 0;
                slot.overflows = 0;
                slot.last_tsc = rdtsc();
                return Ok(slot.id);
            }
        }
        Err(PerfError::NoFreeSlot)
    }

    fn start(&mut self, id: u8) -> Result<(), PerfError> {
        let slot = self
            .counters
            .get_mut(id as usize)
            .filter(|c| c.allocated)
            .ok_or(PerfError::InvalidId)?;
        slot.enabled = true;
        slot.last_tsc = rdtsc();
        if slot.hw_backed {
            let mut evtsel = (slot.config.event.raw_evtsel() as u64) & 0xFFFF;
            if slot.config.user_mode {
                evtsel |= 1 << 16;
            }
            if slot.config.kernel_mode {
                evtsel |= 1 << 17;
            }
            if slot.config.edge_detect {
                evtsel |= 1 << 18;
            }
            evtsel |= 1 << 22; // Enable counter
            unsafe {
                wrmsr(MSR_IA32_PERFEVTSEL0 + id as u32, evtsel);
            }
        }
        Ok(())
    }

    fn stop(&mut self, id: u8) -> Result<(), PerfError> {
        let slot = self
            .counters
            .get_mut(id as usize)
            .filter(|c| c.allocated)
            .ok_or(PerfError::InvalidId)?;
        if slot.enabled {
            let now = rdtsc();
            if slot.hw_backed {
                unsafe {
                    wrmsr(MSR_IA32_PERFEVTSEL0 + id as u32, 0);
                    slot.value = rdmsr(MSR_IA32_PMC0 + id as u32);
                }
            } else if matches!(slot.config.event, PmuEvent::CpuCycles) {
                slot.value = slot.value.saturating_add(now.saturating_sub(slot.last_tsc));
            }
            slot.last_tsc = now;
            slot.enabled = false;
        }
        Ok(())
    }

    fn read(&mut self, id: u8) -> Result<u64, PerfError> {
        let slot = self
            .counters
            .get_mut(id as usize)
            .filter(|c| c.allocated)
            .ok_or(PerfError::InvalidId)?;
        if slot.enabled {
            let now = rdtsc();
            if slot.hw_backed {
                unsafe {
                    slot.value = rdmsr(MSR_IA32_PMC0 + id as u32);
                }
            } else if matches!(slot.config.event, PmuEvent::CpuCycles) {
                slot.value = slot.value.saturating_add(now.saturating_sub(slot.last_tsc));
            }
            slot.last_tsc = now;
        }
        Ok(slot.value)
    }

    fn reset(&mut self, id: u8) -> Result<(), PerfError> {
        let slot = self
            .counters
            .get_mut(id as usize)
            .filter(|c| c.allocated)
            .ok_or(PerfError::InvalidId)?;
        slot.value = 0;
        slot.overflows = 0;
        slot.last_tsc = rdtsc();
        if slot.hw_backed {
            unsafe {
                wrmsr(MSR_IA32_PMC0 + id as u32, 0);
            }
        }
        Ok(())
    }

    fn free(&mut self, id: u8) -> Result<(), PerfError> {
        self.stop(id)?;
        let slot = self
            .counters
            .get_mut(id as usize)
            .ok_or(PerfError::InvalidId)?;
        *slot = PmuCounterState::empty(id);
        Ok(())
    }

    fn record_sw_event(&mut self, event: PmuEvent, delta: u64) -> bool {
        let mut overflowed = false;
        for slot in self.counters.iter_mut() {
            if slot.allocated && slot.enabled && !slot.hw_backed && slot.config.event == event {
                slot.value = slot.value.saturating_add(delta);
                if slot.config.sample_period > 0 && slot.value >= slot.config.sample_period {
                    slot.overflows = slot.overflows.saturating_add(1);
                    slot.value %= slot.config.sample_period;
                    overflowed = true;
                }
            }
        }
        overflowed
    }
}

// ---------------------------------------------------------------------------
// Kernel Tracing Ring Buffer (Kprobes / Tracepoints / Uprobes)
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProbeKind {
    Kprobe,
    Kretprobe,
    Tracepoint,
    Uprobe,
    TimerSample,
    EbpfPrintk,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TracepointId {
    SyscallEnter,
    SyscallExit,
    SchedSwitch,
    PageFault,
    IrqEntry,
    IrqExit,
    IpcSend,
    IpcRecv,
    MmAlloc,
    MmFree,
    Custom(u16),
}

#[derive(Copy, Clone, Debug)]
pub struct TraceEvent {
    pub seq: u64,
    pub timestamp_tsc: u64,
    pub cpu_id: u16,
    pub pid: u32,
    pub tid: u32,
    pub kind: ProbeKind,
    pub probe_id: u16,
    pub ip: u64,
    pub args: [u64; 4],
    pub ret_val: i64,
}

impl TraceEvent {
    pub const fn empty() -> Self {
        Self {
            seq: 0,
            timestamp_tsc: 0,
            cpu_id: 0,
            pid: 0,
            tid: 0,
            kind: ProbeKind::Tracepoint,
            probe_id: 0,
            ip: 0,
            args: [0; 4],
            ret_val: 0,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RingStats {
    pub capacity: usize,
    pub unread: usize,
    pub total_written: u64,
    pub dropped_overwritten: u64,
}

struct TraceRingBuffer {
    buf: [TraceEvent; TRACE_RING_CAP],
    head: usize,
    tail: usize,
    count: usize,
    total_written: u64,
    dropped_overwritten: u64,
}

impl TraceRingBuffer {
    const fn new() -> Self {
        Self {
            buf: [TraceEvent::empty(); TRACE_RING_CAP],
            head: 0,
            tail: 0,
            count: 0,
            total_written: 0,
            dropped_overwritten: 0,
        }
    }

    fn push(&mut self, mut event: TraceEvent) {
        self.total_written = self.total_written.saturating_add(1);
        event.seq = self.total_written;
        if self.count == TRACE_RING_CAP {
            self.tail = (self.tail + 1) % TRACE_RING_CAP;
            self.dropped_overwritten = self.dropped_overwritten.saturating_add(1);
        } else {
            self.count += 1;
        }
        self.buf[self.head] = event;
        self.head = (self.head + 1) % TRACE_RING_CAP;
    }

    fn read_batch(&mut self, out: &mut [TraceEvent]) -> usize {
        let to_read = out.len().min(self.count);
        for slot in out.iter_mut().take(to_read) {
            *slot = self.buf[self.tail];
            self.tail = (self.tail + 1) % TRACE_RING_CAP;
            self.count -= 1;
        }
        to_read
    }

    fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.count = 0;
    }

    fn stats(&self) -> RingStats {
        RingStats {
            capacity: TRACE_RING_CAP,
            unread: self.count,
            total_written: self.total_written,
            dropped_overwritten: self.dropped_overwritten,
        }
    }
}

// ---------------------------------------------------------------------------
// Kprobe & Tracepoint Registry
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct ProbeEntry {
    pub id: u16,
    pub allocated: bool,
    pub active: bool,
    pub kind: ProbeKind,
    pub tracepoint: Option<TracepointId>,
    pub target_addr: u64,
    pub saved_opcode: u8,
    pub name: [u8; 32],
    pub name_len: u8,
    pub hit_count: u64,
    pub ebpf_prog_id: Option<u16>,
}

impl ProbeEntry {
    const fn empty(id: u16) -> Self {
        Self {
            id,
            allocated: false,
            active: false,
            kind: ProbeKind::Tracepoint,
            tracepoint: None,
            target_addr: 0,
            saved_opcode: 0,
            name: [0; 32],
            name_len: 0,
            hit_count: 0,
            ebpf_prog_id: None,
        }
    }

    pub fn name_str(&self) -> &str {
        let len = (self.name_len as usize).min(self.name.len());
        core::str::from_utf8(&self.name[..len]).unwrap_or("")
    }
}

fn copy_name(dst: &mut [u8; 32], src: &str) -> u8 {
    let bytes = src.as_bytes();
    let len = bytes.len().min(dst.len());
    dst[..len].copy_from_slice(&bytes[..len]);
    len as u8
}

struct ProbeRegistry {
    probes: [ProbeEntry; MAX_PROBES],
}

impl ProbeRegistry {
    const fn new() -> Self {
        const EMPTY: ProbeEntry = ProbeEntry::empty(0);
        let mut probes = [EMPTY; MAX_PROBES];
        let mut idx = 0;
        while idx < MAX_PROBES {
            probes[idx].id = idx as u16;
            idx += 1;
        }
        Self { probes }
    }

    fn register_kprobe(
        &mut self,
        name: &str,
        addr: u64,
        ebpf_prog_id: Option<u16>,
    ) -> Result<u16, PerfError> {
        if addr == 0 {
            return Err(PerfError::InvalidConfig);
        }
        for p in self.probes.iter() {
            if p.allocated && p.kind == ProbeKind::Kprobe && p.target_addr == addr {
                return Err(PerfError::AlreadyExists);
            }
        }
        for p in self.probes.iter_mut() {
            if !p.allocated {
                p.allocated = true;
                p.active = true;
                p.kind = ProbeKind::Kprobe;
                p.tracepoint = None;
                p.target_addr = addr;
                p.saved_opcode = 0x90; // NOP placeholder for non-destructive model
                p.name_len = copy_name(&mut p.name, name);
                p.hit_count = 0;
                p.ebpf_prog_id = ebpf_prog_id;
                return Ok(p.id);
            }
        }
        Err(PerfError::NoFreeSlot)
    }

    fn register_tracepoint(
        &mut self,
        name: &str,
        tp: TracepointId,
        ebpf_prog_id: Option<u16>,
    ) -> Result<u16, PerfError> {
        for p in self.probes.iter_mut() {
            if !p.allocated {
                p.allocated = true;
                p.active = true;
                p.kind = ProbeKind::Tracepoint;
                p.tracepoint = Some(tp);
                p.target_addr = 0;
                p.saved_opcode = 0;
                p.name_len = copy_name(&mut p.name, name);
                p.hit_count = 0;
                p.ebpf_prog_id = ebpf_prog_id;
                return Ok(p.id);
            }
        }
        Err(PerfError::NoFreeSlot)
    }

    fn set_active(&mut self, id: u16, active: bool) -> Result<(), PerfError> {
        let p = self
            .probes
            .get_mut(id as usize)
            .filter(|p| p.allocated)
            .ok_or(PerfError::InvalidId)?;
        p.active = active;
        Ok(())
    }

    fn unregister(&mut self, id: u16) -> Result<(), PerfError> {
        let p = self
            .probes
            .get_mut(id as usize)
            .filter(|p| p.allocated)
            .ok_or(PerfError::InvalidId)?;
        *p = ProbeEntry::empty(id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DTrace / eBPF-style Bytecode Probe Engine & Verifier
// ---------------------------------------------------------------------------

pub mod bpf_op {
    pub const MOV_IMM: u8 = 0xB7;
    pub const MOV_REG: u8 = 0xBF;
    pub const ADD_IMM: u8 = 0x07;
    pub const ADD_REG: u8 = 0x0F;
    pub const SUB_IMM: u8 = 0x17;
    pub const SUB_REG: u8 = 0x1F;
    pub const MUL_IMM: u8 = 0x27;
    pub const AND_IMM: u8 = 0x57;
    pub const OR_IMM: u8 = 0x47;
    pub const XOR_IMM: u8 = 0xA7;
    pub const LSH_IMM: u8 = 0x67;
    pub const RSH_IMM: u8 = 0x77;

    pub const JA: u8 = 0x05;
    pub const JEQ_IMM: u8 = 0x15;
    pub const JNE_IMM: u8 = 0x55;
    pub const JGT_IMM: u8 = 0x25;
    pub const JGE_IMM: u8 = 0x35;
    pub const JLT_IMM: u8 = 0xA5;

    pub const LDX_DW: u8 = 0x79;
    pub const STX_DW: u8 = 0x7B;

    pub const CALL: u8 = 0x85;
    pub const EXIT: u8 = 0x95;
}

pub mod bpf_helper {
    pub const KTIME_GET_NS: i32 = 1;
    pub const MAP_LOOKUP: i32 = 2;
    pub const MAP_UPDATE: i32 = 3;
    pub const TRACE_PRINTK: i32 = 4;
    pub const GET_SMP_PROC_ID: i32 = 5;
    pub const MAP_INC: i32 = 6;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EbpfInsn {
    pub opcode: u8,
    pub dst: u8,
    pub src: u8,
    pub off: i16,
    pub imm: i32,
}

impl EbpfInsn {
    pub const fn new(opcode: u8, dst: u8, src: u8, off: i16, imm: i32) -> Self {
        Self {
            opcode,
            dst,
            src,
            off,
            imm,
        }
    }

    pub const fn exit() -> Self {
        Self::new(bpf_op::EXIT, 0, 0, 0, 0)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct EbpfContext {
    pub ip: u64,
    pub args: [u64; 4],
    pub pid: u32,
    pub tid: u32,
    pub cpu_id: u16,
    pub timestamp: u64,
}

#[derive(Copy, Clone, Debug, Default)]
struct MapEntry {
    used: bool,
    key: u64,
    value: u64,
}

#[derive(Copy, Clone)]
struct EbpfMap {
    allocated: bool,
    entries: [MapEntry; BPF_MAP_ENTRIES],
}

impl EbpfMap {
    const fn empty() -> Self {
        Self {
            allocated: false,
            entries: [MapEntry {
                used: false,
                key: 0,
                value: 0,
            }; BPF_MAP_ENTRIES],
        }
    }

    fn lookup(&self, key: u64) -> Option<u64> {
        self.entries
            .iter()
            .find(|e| e.used && e.key == key)
            .map(|e| e.value)
    }

    fn update(&mut self, key: u64, value: u64) -> Result<(), PerfError> {
        for e in self.entries.iter_mut() {
            if e.used && e.key == key {
                e.value = value;
                return Ok(());
            }
        }
        for e in self.entries.iter_mut() {
            if !e.used {
                e.used = true;
                e.key = key;
                e.value = value;
                return Ok(());
            }
        }
        Err(PerfError::MapFull)
    }

    fn increment(&mut self, key: u64, delta: u64) -> Result<u64, PerfError> {
        for e in self.entries.iter_mut() {
            if e.used && e.key == key {
                e.value = e.value.saturating_add(delta);
                return Ok(e.value);
            }
        }
        for e in self.entries.iter_mut() {
            if !e.used {
                e.used = true;
                e.key = key;
                e.value = delta;
                return Ok(e.value);
            }
        }
        Err(PerfError::MapFull)
    }
}

#[derive(Copy, Clone)]
struct EbpfProgram {
    allocated: bool,
    name: [u8; 32],
    name_len: u8,
    len: usize,
    insns: [EbpfInsn; MAX_BPF_INSNS],
    run_count: u64,
}

impl EbpfProgram {
    const fn empty() -> Self {
        Self {
            allocated: false,
            name: [0; 32],
            name_len: 0,
            len: 0,
            insns: [EbpfInsn::exit(); MAX_BPF_INSNS],
            run_count: 0,
        }
    }
}

struct EbpfEngine {
    progs: [EbpfProgram; MAX_BPF_PROGS],
    maps: [EbpfMap; MAX_BPF_MAPS],
}

impl EbpfEngine {
    const fn new() -> Self {
        Self {
            progs: [EbpfProgram::empty(); MAX_BPF_PROGS],
            maps: [EbpfMap::empty(); MAX_BPF_MAPS],
        }
    }

    fn create_map(&mut self) -> Result<u16, PerfError> {
        for (idx, m) in self.maps.iter_mut().enumerate() {
            if !m.allocated {
                *m = EbpfMap::empty();
                m.allocated = true;
                return Ok(idx as u16);
            }
        }
        Err(PerfError::NoFreeSlot)
    }

    fn verify(insns: &[EbpfInsn]) -> Result<(), PerfError> {
        if insns.is_empty() {
            return Err(PerfError::VerifierEmptyProgram);
        }
        if insns.len() > MAX_BPF_INSNS {
            return Err(PerfError::VerifierTooLarge);
        }
        if insns.last().map(|i| i.opcode) != Some(bpf_op::EXIT) {
            return Err(PerfError::VerifierMissingExit);
        }

        for (pc, insn) in insns.iter().enumerate() {
            if insn.dst >= 11 || insn.src >= 11 {
                return Err(PerfError::VerifierInvalidRegister);
            }
            match insn.opcode {
                bpf_op::MOV_IMM
                | bpf_op::MOV_REG
                | bpf_op::ADD_IMM
                | bpf_op::ADD_REG
                | bpf_op::SUB_IMM
                | bpf_op::SUB_REG
                | bpf_op::MUL_IMM
                | bpf_op::AND_IMM
                | bpf_op::OR_IMM
                | bpf_op::XOR_IMM
                | bpf_op::LSH_IMM
                | bpf_op::RSH_IMM => {
                    if insn.dst == 10 {
                        return Err(PerfError::VerifierReadOnlyRegister);
                    }
                }
                bpf_op::JA
                | bpf_op::JEQ_IMM
                | bpf_op::JNE_IMM
                | bpf_op::JGT_IMM
                | bpf_op::JGE_IMM
                | bpf_op::JLT_IMM => {
                    // Strictly forward jumps guarantee termination (DAG).
                    if insn.off <= 0 {
                        return Err(PerfError::VerifierInvalidJump);
                    }
                    let target = pc as isize + 1 + insn.off as isize;
                    if target < 0 || (target as usize) >= insns.len() {
                        return Err(PerfError::VerifierInvalidJump);
                    }
                }
                bpf_op::LDX_DW => {
                    if insn.dst == 10 {
                        return Err(PerfError::VerifierReadOnlyRegister);
                    }
                    // Allow reading context fields (off 0..=48, 8-byte aligned) or stack (-256..=-8)
                    let valid_ctx = insn.off >= 0 && insn.off <= 48 && (insn.off % 8 == 0);
                    let valid_stk = insn.off < 0 && insn.off >= -256 && (insn.off % 8 == 0);
                    if !valid_ctx && !valid_stk {
                        return Err(PerfError::VerifierInvalidMemoryOffset);
                    }
                }
                bpf_op::STX_DW => {
                    // Only allow storing to stack frame (R10 + negative 8-byte aligned offset)
                    if insn.dst != 10 || insn.off >= 0 || insn.off < -256 || (insn.off % 8 != 0) {
                        return Err(PerfError::VerifierInvalidMemoryOffset);
                    }
                }
                bpf_op::CALL => {
                    if !matches!(
                        insn.imm,
                        bpf_helper::KTIME_GET_NS
                            | bpf_helper::MAP_LOOKUP
                            | bpf_helper::MAP_UPDATE
                            | bpf_helper::TRACE_PRINTK
                            | bpf_helper::GET_SMP_PROC_ID
                            | bpf_helper::MAP_INC
                    ) {
                        return Err(PerfError::VerifierInvalidHelper);
                    }
                }
                bpf_op::EXIT => {}
                _ => return Err(PerfError::InvalidConfig),
            }
        }
        Ok(())
    }

    fn load_program(&mut self, name: &str, insns: &[EbpfInsn]) -> Result<u16, PerfError> {
        Self::verify(insns)?;
        for (idx, prog) in self.progs.iter_mut().enumerate() {
            if !prog.allocated {
                prog.allocated = true;
                prog.name_len = copy_name(&mut prog.name, name);
                prog.len = insns.len();
                prog.insns[..insns.len()].copy_from_slice(insns);
                prog.run_count = 0;
                return Ok(idx as u16);
            }
        }
        Err(PerfError::NoFreeSlot)
    }

    fn run_program(&mut self, prog_id: u16, ctx: &EbpfContext) -> Result<u64, PerfError> {
        let prog = self
            .progs
            .get_mut(prog_id as usize)
            .filter(|p| p.allocated)
            .ok_or(PerfError::InvalidId)?;
        prog.run_count = prog.run_count.saturating_add(1);
        let len = prog.len;
        let insns = prog.insns;

        let mut regs = [0u64; 11];
        // R1 = context pointer tag (0x1000_0000), R10 = stack top tag (0x2000_0100)
        const CTX_BASE: u64 = 0x1000_0000;
        const STK_TOP: u64 = 0x2000_0100;
        regs[1] = CTX_BASE;
        regs[10] = STK_TOP;

        let mut stack = [0u64; BPF_STACK_WORDS];
        let mut pc: usize = 0;

        while pc < len {
            let insn = insns[pc];
            let dst = insn.dst as usize;
            let src = insn.src as usize;
            let imm64 = insn.imm as i64 as u64;

            match insn.opcode {
                bpf_op::MOV_IMM => regs[dst] = imm64,
                bpf_op::MOV_REG => regs[dst] = regs[src],
                bpf_op::ADD_IMM => regs[dst] = regs[dst].wrapping_add(imm64),
                bpf_op::ADD_REG => regs[dst] = regs[dst].wrapping_add(regs[src]),
                bpf_op::SUB_IMM => regs[dst] = regs[dst].wrapping_sub(imm64),
                bpf_op::SUB_REG => regs[dst] = regs[dst].wrapping_sub(regs[src]),
                bpf_op::MUL_IMM => regs[dst] = regs[dst].wrapping_mul(imm64),
                bpf_op::AND_IMM => regs[dst] &= imm64,
                bpf_op::OR_IMM => regs[dst] |= imm64,
                bpf_op::XOR_IMM => regs[dst] ^= imm64,
                bpf_op::LSH_IMM => regs[dst] = regs[dst].wrapping_shl((insn.imm as u32) & 63),
                bpf_op::RSH_IMM => regs[dst] = regs[dst].wrapping_shr((insn.imm as u32) & 63),
                bpf_op::JA => {
                    pc = (pc as isize + 1 + insn.off as isize) as usize;
                    continue;
                }
                bpf_op::JEQ_IMM => {
                    if regs[dst] == imm64 {
                        pc = (pc as isize + 1 + insn.off as isize) as usize;
                        continue;
                    }
                }
                bpf_op::JNE_IMM => {
                    if regs[dst] != imm64 {
                        pc = (pc as isize + 1 + insn.off as isize) as usize;
                        continue;
                    }
                }
                bpf_op::JGT_IMM => {
                    if regs[dst] > imm64 {
                        pc = (pc as isize + 1 + insn.off as isize) as usize;
                        continue;
                    }
                }
                bpf_op::JGE_IMM => {
                    if regs[dst] >= imm64 {
                        pc = (pc as isize + 1 + insn.off as isize) as usize;
                        continue;
                    }
                }
                bpf_op::JLT_IMM => {
                    if regs[dst] < imm64 {
                        pc = (pc as isize + 1 + insn.off as isize) as usize;
                        continue;
                    }
                }
                bpf_op::LDX_DW => {
                    let base = regs[src];
                    if base == CTX_BASE && insn.off >= 0 {
                        regs[dst] = match insn.off {
                            0 => ctx.ip,
                            8 => ctx.args[0],
                            16 => ctx.args[1],
                            24 => ctx.args[2],
                            32 => ctx.args[3],
                            40 => ((ctx.tid as u64) << 32) | (ctx.pid as u64),
                            48 => ctx.timestamp,
                            _ => 0,
                        };
                    } else if base == STK_TOP && insn.off < 0 && insn.off >= -256 {
                        let idx = ((-insn.off) / 8 - 1) as usize;
                        regs[dst] = stack.get(idx).copied().unwrap_or(0);
                    } else {
                        regs[dst] = 0;
                    }
                }
                bpf_op::STX_DW => {
                    if regs[dst] == STK_TOP && insn.off < 0 && insn.off >= -256 {
                        let idx = ((-insn.off) / 8 - 1) as usize;
                        if let Some(slot) = stack.get_mut(idx) {
                            *slot = regs[src];
                        }
                    }
                }
                bpf_op::CALL => {
                    regs[0] = match insn.imm {
                        bpf_helper::KTIME_GET_NS => rdtsc(),
                        bpf_helper::GET_SMP_PROC_ID => ctx.cpu_id as u64,
                        bpf_helper::MAP_LOOKUP => {
                            let map_id = regs[1] as usize;
                            let key = regs[2];
                            self.maps
                                .get(map_id)
                                .filter(|m| m.allocated)
                                .and_then(|m| m.lookup(key))
                                .unwrap_or(0)
                        }
                        bpf_helper::MAP_UPDATE => {
                            let map_id = regs[1] as usize;
                            let key = regs[2];
                            let val = regs[3];
                            if let Some(m) = self.maps.get_mut(map_id).filter(|m| m.allocated) {
                                if m.update(key, val).is_ok() {
                                    0
                                } else {
                                    u64::MAX
                                }
                            } else {
                                u64::MAX
                            }
                        }
                        bpf_helper::MAP_INC => {
                            let map_id = regs[1] as usize;
                            let key = regs[2];
                            let delta = regs[3];
                            if let Some(m) = self.maps.get_mut(map_id).filter(|m| m.allocated) {
                                m.increment(key, delta).unwrap_or(u64::MAX)
                            } else {
                                u64::MAX
                            }
                        }
                        bpf_helper::TRACE_PRINTK => {
                            // Record lightweight printk event into R0
                            regs[1]
                        }
                        _ => 0,
                    };
                }
                bpf_op::EXIT => return Ok(regs[0]),
                _ => return Err(PerfError::InvalidConfig),
            }
            pc += 1;
        }
        Ok(regs[0])
    }
}

// ---------------------------------------------------------------------------
// Kernel Stack Sampler (Frame-Pointer Unwinder)
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct StackSample {
    pub timestamp_tsc: u64,
    pub cpu_id: u16,
    pub pid: u32,
    pub tid: u32,
    pub depth: u8,
    pub frames: [u64; MAX_STACK_DEPTH],
}

impl StackSample {
    pub const fn empty() -> Self {
        Self {
            timestamp_tsc: 0,
            cpu_id: 0,
            pid: 0,
            tid: 0,
            depth: 0,
            frames: [0; MAX_STACK_DEPTH],
        }
    }
}

struct StackSampleRing {
    buf: [StackSample; STACK_RING_CAP],
    head: usize,
    tail: usize,
    count: usize,
    total_samples: u64,
}

impl StackSampleRing {
    const fn new() -> Self {
        Self {
            buf: [StackSample::empty(); STACK_RING_CAP],
            head: 0,
            tail: 0,
            count: 0,
            total_samples: 0,
        }
    }

    fn push(&mut self, sample: StackSample) {
        self.total_samples = self.total_samples.saturating_add(1);
        if self.count == STACK_RING_CAP {
            self.tail = (self.tail + 1) % STACK_RING_CAP;
        } else {
            self.count += 1;
        }
        self.buf[self.head] = sample;
        self.head = (self.head + 1) % STACK_RING_CAP;
    }

    fn read_batch(&mut self, out: &mut [StackSample]) -> usize {
        let to_read = out.len().min(self.count);
        for slot in out.iter_mut().take(to_read) {
            *slot = self.buf[self.tail];
            self.tail = (self.tail + 1) % STACK_RING_CAP;
            self.count -= 1;
        }
        to_read
    }
}

#[inline]
fn is_canonical_kernel_addr(addr: u64) -> bool {
    addr >= 0xFFFF_8000_0000_0000
}

/// Capture the current kernel call stack by walking `rbp` frame pointers.
///
/// Uses strict canonical-address, alignment, and monotonic-frame-growth checks
/// so leaf/optimized frames cannot cause a page fault.
#[inline(never)]
pub fn capture_kernel_stack(pid: u32, tid: u32, max_depth: usize) -> StackSample {
    let mut sample = StackSample::empty();
    sample.timestamp_tsc = rdtsc();
    sample.pid = pid;
    sample.tid = tid;

    let limit = max_depth.min(MAX_STACK_DEPTH);
    let mut rbp: u64;
    let mut rip: u64;
    unsafe {
        asm!(
            "mov {rbp_out}, rbp",
            "lea {rip_out}, [rip + 0]",
            rbp_out = out(reg) rbp,
            rip_out = out(reg) rip,
            options(nomem, nostack, preserves_flags),
        );
    }

    let mut depth = 0usize;
    if limit > 0 && rip != 0 {
        sample.frames[0] = rip;
        depth = 1;
    }

    while depth < limit {
        if !is_canonical_kernel_addr(rbp) || (rbp & 0x7) != 0 {
            break;
        }
        let next_rbp = unsafe { *(rbp as *const u64) };
        let ret_addr = unsafe { *((rbp + 8) as *const u64) };
        if !is_canonical_kernel_addr(ret_addr) {
            break;
        }
        sample.frames[depth] = ret_addr;
        depth += 1;

        // Frame pointer must strictly advance within a reasonable 64 KiB window.
        if next_rbp <= rbp || next_rbp.saturating_sub(rbp) > 64 * 1024 {
            break;
        }
        rbp = next_rbp;
    }

    sample.depth = depth as u8;
    sample
}

// ---------------------------------------------------------------------------
// Global Subsystem State & Public API
// ---------------------------------------------------------------------------

static INITIALIZED: AtomicBool = AtomicBool::new(false);
static TOTAL_PROBE_HITS: AtomicU64 = AtomicU64::new(0);

static PMU: SpinLock<PmuManager> = SpinLock::new(PmuManager::new());
static TRACE_RING: SpinLock<TraceRingBuffer> = SpinLock::new(TraceRingBuffer::new());
static PROBES: SpinLock<ProbeRegistry> = SpinLock::new(ProbeRegistry::new());
static EBPF: SpinLock<EbpfEngine> = SpinLock::new(EbpfEngine::new());
static STACK_RING: SpinLock<StackSampleRing> = SpinLock::new(StackSampleRing::new());

/// Initialize the kernel tracing, PMU, and eBPF probe subsystem.
pub fn init() -> PmuCapabilities {
    let caps = PMU.lock().probe();
    INITIALIZED.store(true, Ordering::Release);
    serial_println!(
        "[perf] PMU v{} (supported={}, gp_ctrs={}, width={}b, fixed_ctrs={})",
        caps.version,
        caps.supported,
        caps.num_gp_counters,
        caps.gp_bit_width,
        caps.num_fixed_counters
    );
    caps
}

pub fn pmu_capabilities() -> PmuCapabilities {
    PMU.lock().caps
}

pub fn pmu_allocate(config: PmuCounterConfig) -> Result<u8, PerfError> {
    PMU.lock().allocate(config)
}

pub fn pmu_start(id: u8) -> Result<(), PerfError> {
    PMU.lock().start(id)
}

pub fn pmu_stop(id: u8) -> Result<(), PerfError> {
    PMU.lock().stop(id)
}

pub fn pmu_read(id: u8) -> Result<u64, PerfError> {
    PMU.lock().read(id)
}

pub fn pmu_reset(id: u8) -> Result<(), PerfError> {
    PMU.lock().reset(id)
}

pub fn pmu_free(id: u8) -> Result<(), PerfError> {
    PMU.lock().free(id)
}

pub fn pmu_record_sw_event(event: PmuEvent, delta: u64) {
    if PMU.lock().record_sw_event(event, delta) {
        record_stack_sample(0, 0);
    }
}

pub fn register_kprobe(name: &str, addr: u64, ebpf_prog_id: Option<u16>) -> Result<u16, PerfError> {
    PROBES.lock().register_kprobe(name, addr, ebpf_prog_id)
}

pub fn register_tracepoint(
    name: &str,
    tp: TracepointId,
    ebpf_prog_id: Option<u16>,
) -> Result<u16, PerfError> {
    PROBES.lock().register_tracepoint(name, tp, ebpf_prog_id)
}

pub fn enable_probe(id: u16) -> Result<(), PerfError> {
    PROBES.lock().set_active(id, true)
}

pub fn disable_probe(id: u16) -> Result<(), PerfError> {
    PROBES.lock().set_active(id, false)
}

pub fn unregister_probe(id: u16) -> Result<(), PerfError> {
    PROBES.lock().unregister(id)
}

/// Fire all active tracepoints matching `tp`, running any attached eBPF programs
/// and recording events into the trace ring buffer.
pub fn fire_tracepoint(tp: TracepointId, ip: u64, pid: u32, tid: u32, args: [u64; 4]) {
    let now = rdtsc();
    let mut matched: [(u16, Option<u16>); 4] = [(0, None); 4];
    let mut match_count = 0usize;

    {
        let mut reg = PROBES.lock();
        for p in reg.probes.iter_mut() {
            if p.allocated
                && p.active
                && p.kind == ProbeKind::Tracepoint
                && p.tracepoint == Some(tp)
            {
                p.hit_count = p.hit_count.saturating_add(1);
                if match_count < matched.len() {
                    matched[match_count] = (p.id, p.ebpf_prog_id);
                    match_count += 1;
                }
            }
        }
    }

    if match_count == 0 {
        return;
    }

    TOTAL_PROBE_HITS.fetch_add(match_count as u64, Ordering::Relaxed);
    let ctx = EbpfContext {
        ip,
        args,
        pid,
        tid,
        cpu_id: 0,
        timestamp: now,
    };

    for &(probe_id, prog_opt) in matched.iter().take(match_count) {
        let ret_val = if let Some(prog_id) = prog_opt {
            EBPF.lock().run_program(prog_id, &ctx).unwrap_or(0) as i64
        } else {
            0
        };
        TRACE_RING.lock().push(TraceEvent {
            seq: 0,
            timestamp_tsc: now,
            cpu_id: 0,
            pid,
            tid,
            kind: ProbeKind::Tracepoint,
            probe_id,
            ip,
            args,
            ret_val,
        });
    }
}

/// Fire an active kprobe at `addr`, running any attached eBPF program.
pub fn fire_kprobe(addr: u64, pid: u32, tid: u32, args: [u64; 4]) -> bool {
    let now = rdtsc();
    let mut hit: Option<(u16, Option<u16>)> = None;

    {
        let mut reg = PROBES.lock();
        for p in reg.probes.iter_mut() {
            if p.allocated && p.active && p.kind == ProbeKind::Kprobe && p.target_addr == addr {
                p.hit_count = p.hit_count.saturating_add(1);
                hit = Some((p.id, p.ebpf_prog_id));
                break;
            }
        }
    }

    let Some((probe_id, prog_opt)) = hit else {
        return false;
    };

    TOTAL_PROBE_HITS.fetch_add(1, Ordering::Relaxed);
    let ctx = EbpfContext {
        ip: addr,
        args,
        pid,
        tid,
        cpu_id: 0,
        timestamp: now,
    };
    let ret_val = if let Some(prog_id) = prog_opt {
        EBPF.lock().run_program(prog_id, &ctx).unwrap_or(0) as i64
    } else {
        0
    };

    TRACE_RING.lock().push(TraceEvent {
        seq: 0,
        timestamp_tsc: now,
        cpu_id: 0,
        pid,
        tid,
        kind: ProbeKind::Kprobe,
        probe_id,
        ip: addr,
        args,
        ret_val,
    });
    true
}

pub fn read_trace_events(out: &mut [TraceEvent]) -> usize {
    TRACE_RING.lock().read_batch(out)
}

pub fn trace_ring_stats() -> RingStats {
    TRACE_RING.lock().stats()
}

pub fn create_bpf_map() -> Result<u16, PerfError> {
    EBPF.lock().create_map()
}

pub fn bpf_map_lookup(map_id: u16, key: u64) -> Option<u64> {
    EBPF.lock()
        .maps
        .get(map_id as usize)
        .filter(|m| m.allocated)
        .and_then(|m| m.lookup(key))
}

pub fn load_bpf_program(name: &str, insns: &[EbpfInsn]) -> Result<u16, PerfError> {
    EBPF.lock().load_program(name, insns)
}

pub fn run_bpf_program(prog_id: u16, ctx: &EbpfContext) -> Result<u64, PerfError> {
    EBPF.lock().run_program(prog_id, ctx)
}

pub fn record_stack_sample(pid: u32, tid: u32) {
    let sample = capture_kernel_stack(pid, tid, MAX_STACK_DEPTH);
    STACK_RING.lock().push(sample);
}

pub fn read_stack_samples(out: &mut [StackSample]) -> usize {
    STACK_RING.lock().read_batch(out)
}

// ---------------------------------------------------------------------------
// Self-Test Suite
// ---------------------------------------------------------------------------

/// Run deterministic self-tests for PMU counters, tracepoints/kprobes ring buffer,
/// eBPF verifier & VM + map aggregation, and kernel stack sampler.
pub fn self_test() -> bool {
    if !INITIALIZED.load(Ordering::Acquire) {
        init();
    }

    // 1. PMU counter allocation, start, software event, read, stop, free
    let Ok(ctr_id) = pmu_allocate(PmuCounterConfig {
        event: PmuEvent::SyscallCount,
        user_mode: true,
        kernel_mode: true,
        edge_detect: false,
        sample_period: 0,
    }) else {
        serial_println!("[perf] self_test FAIL: pmu_allocate");
        return false;
    };
    if pmu_start(ctr_id).is_err() {
        serial_println!("[perf] self_test FAIL: pmu_start");
        return false;
    }
    pmu_record_sw_event(PmuEvent::SyscallCount, 7);
    let val = pmu_read(ctr_id).unwrap_or(0);
    let _ = pmu_free(ctr_id);
    if val != 7 {
        serial_println!("[perf] self_test FAIL: pmu_read expected 7, got {}", val);
        return false;
    }

    // 2. eBPF verifier negative tests (backward jump & missing exit)
    let bad_loop = [EbpfInsn::new(bpf_op::JA, 0, 0, -1, 0), EbpfInsn::exit()];
    if EbpfEngine::verify(&bad_loop) != Err(PerfError::VerifierInvalidJump) {
        serial_println!("[perf] self_test FAIL: verifier accepted backward jump");
        return false;
    }
    let bad_no_exit = [EbpfInsn::new(bpf_op::MOV_IMM, 0, 0, 0, 1)];
    if EbpfEngine::verify(&bad_no_exit) != Err(PerfError::VerifierMissingExit) {
        serial_println!("[perf] self_test FAIL: verifier accepted missing exit");
        return false;
    }

    // 3. eBPF map + program attached to tracepoint
    let Ok(map_id) = create_bpf_map() else {
        serial_println!("[perf] self_test FAIL: create_bpf_map");
        return false;
    };
    // Program: if ctx.args[0] == 42, increment map[map_id][42] by 1 and return count; else return 0
    let prog_insns = [
        EbpfInsn::new(bpf_op::LDX_DW, 6, 1, 8, 0), // R6 = ctx.args[0]
        EbpfInsn::new(bpf_op::JNE_IMM, 6, 0, 5, 42), // if R6 != 42 -> skip to MOV R0, 0
        EbpfInsn::new(bpf_op::MOV_IMM, 1, 0, 0, map_id as i32),
        EbpfInsn::new(bpf_op::MOV_IMM, 2, 0, 0, 42),
        EbpfInsn::new(bpf_op::MOV_IMM, 3, 0, 0, 1),
        EbpfInsn::new(bpf_op::CALL, 0, 0, 0, bpf_helper::MAP_INC),
        EbpfInsn::exit(),
        EbpfInsn::new(bpf_op::MOV_IMM, 0, 0, 0, 0),
        EbpfInsn::exit(),
    ];
    let Ok(prog_id) = load_bpf_program("tp_filter_inc", &prog_insns) else {
        serial_println!("[perf] self_test FAIL: load_bpf_program");
        return false;
    };

    let Ok(tp_id) =
        register_tracepoint("sys_enter_test", TracepointId::SyscallEnter, Some(prog_id))
    else {
        serial_println!("[perf] self_test FAIL: register_tracepoint");
        return false;
    };

    TRACE_RING.lock().clear();
    fire_tracepoint(
        TracepointId::SyscallEnter,
        0xFFFF_8000_0010_0000,
        10,
        10,
        [42, 0, 0, 0],
    );
    fire_tracepoint(
        TracepointId::SyscallEnter,
        0xFFFF_8000_0010_0000,
        10,
        10,
        [99, 0, 0, 0],
    );
    fire_tracepoint(
        TracepointId::SyscallEnter,
        0xFFFF_8000_0010_0000,
        10,
        10,
        [42, 0, 0, 0],
    );
    let _ = unregister_probe(tp_id);

    let map_count = bpf_map_lookup(map_id, 42).unwrap_or(0);
    if map_count != 2 {
        serial_println!(
            "[perf] self_test FAIL: bpf map count expected 2, got {}",
            map_count
        );
        return false;
    }

    let mut events = [TraceEvent::empty(); 4];
    let read_n = read_trace_events(&mut events);
    if read_n != 3 || events[0].ret_val != 1 || events[1].ret_val != 0 || events[2].ret_val != 2 {
        serial_println!(
            "[perf] self_test FAIL: trace ring readback mismatch (n={})",
            read_n
        );
        return false;
    }

    // 4. Kprobe registration & firing
    let Ok(kp_id) = register_kprobe("kp_test", 0xFFFF_8000_0020_1234, None) else {
        serial_println!("[perf] self_test FAIL: register_kprobe");
        return false;
    };
    let kp_hit = fire_kprobe(0xFFFF_8000_0020_1234, 1, 1, [1, 2, 3, 4]);
    let _ = unregister_probe(kp_id);
    let _ = read_trace_events(&mut events);
    if !kp_hit {
        serial_println!("[perf] self_test FAIL: fire_kprobe");
        return false;
    }

    // 5. Stack sampler
    record_stack_sample(1, 1);
    let mut samples = [StackSample::empty(); 2];
    let s_n = read_stack_samples(&mut samples);
    if s_n == 0 || samples[0].depth == 0 {
        serial_println!("[perf] self_test FAIL: stack sampler captured 0 frames");
        return false;
    }

    serial_println!(
        "[perf] self_test PASS (pmu_sw=7, bpf_map=2, trace_events=3, stack_depth={})",
        samples[0].depth
    );
    true
}
