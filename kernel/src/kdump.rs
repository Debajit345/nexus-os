//! Kernel Crash Dump (kdump ELF64 vmcore), Persistent pstore Panic Log & Watchdog NMI Engine.
//!
//! Provides:
//! - ELF64 `vmcore` generator (`ET_CORE`, `EM_X86_64`) with `PT_NOTE` (`NT_PRSTATUS` per-CPU
//!   register state + `VMCOREINFO` metadata) and `PT_LOAD` physical/virtual memory segments
//! - Persistent `pstore` circular panic/oops/NMI log ring in `.bss` with CRC32 integrity checks
//! - Per-CPU Watchdog NMI engine for soft-lockup and hard-lockup detection and crash capture
//! - `self_test()` verification suite
#![allow(dead_code)]

extern crate alloc;

use alloc::vec::Vec;
use core::arch::asm;
use core::fmt::Write as _;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::mm::SpinLock;
use crate::serial_println;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KdumpError {
    InvalidElfMagic,
    InvalidElfClass,
    InvalidElfType,
    InvalidElfMachine,
    TruncatedVmcore,
    InvalidProgramHeader,
    InvalidNoteSection,
    NoCpuContext,
    NoMemoryRegions,
    RegionTooLarge,
    InvalidCpuId,
    BufferTooSmall,
}

impl KdumpError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidElfMagic => "invalid ELF64 magic",
            Self::InvalidElfClass => "invalid ELF class (expected ELFCLASS64)",
            Self::InvalidElfType => "invalid ELF type (expected ET_CORE)",
            Self::InvalidElfMachine => "invalid ELF machine (expected EM_X86_64)",
            Self::TruncatedVmcore => "truncated vmcore image",
            Self::InvalidProgramHeader => "invalid ELF64 program header",
            Self::InvalidNoteSection => "invalid PT_NOTE section",
            Self::NoCpuContext => "no CPU register context provided",
            Self::NoMemoryRegions => "no memory regions provided",
            Self::RegionTooLarge => "memory region exceeds maximum dump size",
            Self::InvalidCpuId => "CPU id out of range",
            Self::BufferTooSmall => "destination buffer too small",
        }
    }
}

// ---------------------------------------------------------------------------
// ELF64 Constants & Structures
// ---------------------------------------------------------------------------

pub const ELFMAG: [u8; 4] = [0x7F, b'E', b'L', b'F'];
pub const ELFCLASS64: u8 = 2;
pub const ELFDATA2LSB: u8 = 1;
pub const EV_CURRENT: u8 = 1;
pub const ELFOSABI_NONE: u8 = 0;

pub const ET_CORE: u16 = 4;
pub const EM_X86_64: u16 = 62;

pub const PT_LOAD: u32 = 1;
pub const PT_NOTE: u32 = 4;

pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

pub const NT_PRSTATUS: u32 = 1;
pub const NT_VMCOREINFO: u32 = 0;

pub const ELF64_EHDR_SIZE: usize = 64;
pub const ELF64_PHDR_SIZE: usize = 56;
pub const ELF64_NHDR_SIZE: usize = 12;

/// Maximum single `PT_LOAD` region size accepted by the in-memory generator (16 MiB).
pub const MAX_DUMP_REGION_BYTES: usize = 16 * 1024 * 1024;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Elf64Ehdr {
    pub e_ident: [u8; 16],
    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    pub e_entry: u64,
    pub e_phoff: u64,
    pub e_shoff: u64,
    pub e_flags: u32,
    pub e_ehsize: u16,
    pub e_phentsize: u16,
    pub e_phnum: u16,
    pub e_shentsize: u16,
    pub e_shnum: u16,
    pub e_shstrndx: u16,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Elf64Phdr {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Elf64Nhdr {
    pub n_namesz: u32,
    pub n_descsz: u32,
    pub n_type: u32,
}

/// x86_64 `user_regs_struct` layout (27 * 8 = 216 bytes) used inside `NT_PRSTATUS`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct UserRegsStruct {
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub rbp: u64,
    pub rbx: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rax: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub orig_rax: u64,
    pub rip: u64,
    pub cs: u64,
    pub eflags: u64,
    pub rsp: u64,
    pub ss: u64,
    pub fs_base: u64,
    pub gs_base: u64,
    pub ds: u64,
    pub es: u64,
    pub fs: u64,
    pub gs: u64,
}

/// Captured x86_64 CPU register snapshot (general-purpose + control registers).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuRegs {
    pub gpr: UserRegsStruct,
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
}

/// Capture a best-effort snapshot of the current CPU's registers.
#[inline(never)]
pub fn capture_current_regs() -> CpuRegs {
    let mut regs = CpuRegs::default();
    let rip: u64;
    let rsp: u64;
    let rbp: u64;
    let rflags: u64;
    let cs: u16;
    let ss: u16;
    let cr0: u64;
    let cr2: u64;
    let cr3: u64;
    let cr4: u64;

    unsafe {
        asm!(
            "lea {rip}, [rip + 0]",
            "mov {rsp}, rsp",
            "mov {rbp}, rbp",
            "pushfq",
            "pop {rflags}",
            "mov {cs:x}, cs",
            "mov {ss:x}, ss",
            rip = out(reg) rip,
            rsp = out(reg) rsp,
            rbp = out(reg) rbp,
            rflags = out(reg) rflags,
            cs = out(reg) cs,
            ss = out(reg) ss,
            options(preserves_flags),
        );
        asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack, preserves_flags));
        asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack, preserves_flags));
        asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack, preserves_flags));
        asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
    }

    regs.gpr.rip = rip;
    regs.gpr.rsp = rsp;
    regs.gpr.rbp = rbp;
    regs.gpr.eflags = rflags;
    regs.gpr.cs = cs as u64;
    regs.gpr.ss = ss as u64;
    regs.cr0 = cr0;
    regs.cr2 = cr2;
    regs.cr3 = cr3;
    regs.cr4 = cr4;
    regs
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ElfSiginfo {
    pub si_signo: i32,
    pub si_code: i32,
    pub si_errno: i32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ElfTimeval {
    pub tv_sec: i64,
    pub tv_usec: i64,
}

/// Linux/ELF64 `elf_prstatus` structure for x86_64 (`NT_PRSTATUS`, 336 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Elf64Prstatus {
    pub pr_info: ElfSiginfo,
    pub pr_cursig: i16,
    pub _pad0: u16,
    pub pr_sigpend: u64,
    pub pr_sighold: u64,
    pub pr_pid: i32,
    pub pr_ppid: i32,
    pub pr_pgrp: i32,
    pub pr_sid: i32,
    pub pr_utime: ElfTimeval,
    pub pr_stime: ElfTimeval,
    pub pr_cutime: ElfTimeval,
    pub pr_cstime: ElfTimeval,
    pub pr_reg: UserRegsStruct,
    pub pr_fpvalid: i32,
    pub _pad1: i32,
}

// ---------------------------------------------------------------------------
// Little-Endian Serialization Helpers
// ---------------------------------------------------------------------------

#[inline]
fn align_up_4(val: usize) -> usize {
    (val + 3) & !3
}

#[inline]
fn align_up_8(val: usize) -> usize {
    (val + 7) & !7
}

#[inline]
fn push_u16_le(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_i32_le(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_i16_le(out: &mut Vec<u8>, v: i16) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_u64_le(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_i64_le(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn read_u16_le(buf: &[u8], off: usize) -> Option<u16> {
    let slice = buf.get(off..off + 2)?;
    Some(u16::from_le_bytes([slice[0], slice[1]]))
}

#[inline]
fn read_u32_le(buf: &[u8], off: usize) -> Option<u32> {
    let slice = buf.get(off..off + 4)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[inline]
fn read_u64_le(buf: &[u8], off: usize) -> Option<u64> {
    let slice = buf.get(off..off + 8)?;
    Some(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn append_elf_note(out: &mut Vec<u8>, name: &[u8], note_type: u32, desc: &[u8]) {
    push_u32_le(out, name.len() as u32);
    push_u32_le(out, desc.len() as u32);
    push_u32_le(out, note_type);

    out.extend_from_slice(name);
    let name_pad = align_up_4(name.len()) - name.len();
    out.resize(out.len() + name_pad, 0);

    out.extend_from_slice(desc);
    let desc_pad = align_up_4(desc.len()) - desc.len();
    out.resize(out.len() + desc_pad, 0);
}

fn serialize_prstatus(pr: &Elf64Prstatus, out: &mut Vec<u8>) {
    push_i32_le(out, pr.pr_info.si_signo);
    push_i32_le(out, pr.pr_info.si_code);
    push_i32_le(out, pr.pr_info.si_errno);
    push_i16_le(out, pr.pr_cursig);
    push_u16_le(out, pr._pad0);
    push_u64_le(out, pr.pr_sigpend);
    push_u64_le(out, pr.pr_sighold);
    push_i32_le(out, pr.pr_pid);
    push_i32_le(out, pr.pr_ppid);
    push_i32_le(out, pr.pr_pgrp);
    push_i32_le(out, pr.pr_sid);
    push_i64_le(out, pr.pr_utime.tv_sec);
    push_i64_le(out, pr.pr_utime.tv_usec);
    push_i64_le(out, pr.pr_stime.tv_sec);
    push_i64_le(out, pr.pr_stime.tv_usec);
    push_i64_le(out, pr.pr_cutime.tv_sec);
    push_i64_le(out, pr.pr_cutime.tv_usec);
    push_i64_le(out, pr.pr_cstime.tv_sec);
    push_i64_le(out, pr.pr_cstime.tv_usec);

    let r = &pr.pr_reg;
    let regs_arr = [
        r.r15, r.r14, r.r13, r.r12, r.rbp, r.rbx, r.r11, r.r10, r.r9, r.r8, r.rax, r.rcx, r.rdx,
        r.rsi, r.rdi, r.orig_rax, r.rip, r.cs, r.eflags, r.rsp, r.ss, r.fs_base, r.gs_base, r.ds,
        r.es, r.fs, r.gs,
    ];
    for reg in regs_arr {
        push_u64_le(out, reg);
    }

    push_i32_le(out, pr.pr_fpvalid);
    push_i32_le(out, pr._pad1);
}

// ---------------------------------------------------------------------------
// Memory Region & Vmcore Generator
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum RegionSource<'a> {
    /// Explicit byte slice backing this `PT_LOAD` segment.
    Slice(&'a [u8]),
    /// Zero-filled segment of `size` bytes.
    Zeroed,
    /// Deterministic fill byte (useful for synthetic crash-dump tests).
    Pattern(u8),
}

#[derive(Clone, Debug)]
pub struct MemoryRegion<'a> {
    pub phys_addr: u64,
    pub virt_addr: u64,
    pub size: u64,
    pub flags: u32,
    pub source: RegionSource<'a>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VmcoreMetadata {
    pub crash_tsc: u64,
    pub crash_cpu: u32,
    pub kernel_base: u64,
    pub hhdm_offset: u64,
    pub page_size: u64,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VmcoreSummary {
    pub phnum: u16,
    pub pt_note_count: u16,
    pub pt_load_count: u16,
    pub prstatus_notes: u16,
    pub vmcoreinfo_notes: u16,
    pub total_bytes: usize,
    pub total_load_bytes: u64,
}

struct CpuNoteEntry {
    cpu_id: u32,
    regs: CpuRegs,
}

/// Builder for ELF64 `vmcore` crash dumps (`PT_NOTE` + `PT_LOAD`).
pub struct VmcoreGenerator<'a> {
    cpus: Vec<CpuNoteEntry>,
    regions: Vec<MemoryRegion<'a>>,
    meta: VmcoreMetadata,
    panic_reason: &'a str,
}

impl<'a> VmcoreGenerator<'a> {
    pub fn new(panic_reason: &'a str, meta: VmcoreMetadata) -> Self {
        Self {
            cpus: Vec::new(),
            regions: Vec::new(),
            meta,
            panic_reason,
        }
    }

    pub fn add_cpu_regs(&mut self, cpu_id: u32, regs: CpuRegs) {
        self.cpus.push(CpuNoteEntry { cpu_id, regs });
    }

    pub fn add_region(&mut self, region: MemoryRegion<'a>) -> Result<(), KdumpError> {
        if region.size as usize > MAX_DUMP_REGION_BYTES {
            return Err(KdumpError::RegionTooLarge);
        }
        self.regions.push(region);
        Ok(())
    }

    fn build_note_payload(&self) -> Vec<u8> {
        let mut notes = Vec::with_capacity(512);

        for entry in &self.cpus {
            let pr = Elf64Prstatus {
                pr_info: ElfSiginfo {
                    si_signo: 11, // SIGSEGV / kernel panic
                    si_code: 0,
                    si_errno: 0,
                },
                pr_cursig: 11,
                pr_pid: (entry.cpu_id as i32) + 1,
                pr_reg: entry.regs.gpr,
                pr_fpvalid: 0,
                ..Elf64Prstatus::default()
            };
            let mut desc = Vec::with_capacity(core::mem::size_of::<Elf64Prstatus>());
            serialize_prstatus(&pr, &mut desc);
            append_elf_note(&mut notes, b"CORE\0", NT_PRSTATUS, &desc);
        }

        let mut vmcoreinfo = alloc::string::String::new();
        let page_sz = if self.meta.page_size == 0 {
            4096
        } else {
            self.meta.page_size
        };
        let _ = writeln!(vmcoreinfo, "OSRELEASE=nexus-0.1.0");
        let _ = writeln!(vmcoreinfo, "PAGESIZE={}", page_sz);
        let _ = writeln!(vmcoreinfo, "CRASHTSC={}", self.meta.crash_tsc);
        let _ = writeln!(vmcoreinfo, "CRASHCPU={}", self.meta.crash_cpu);
        let _ = writeln!(vmcoreinfo, "CPUCOUNT={}", self.cpus.len());
        let _ = writeln!(vmcoreinfo, "KERNELBASE={:#x}", self.meta.kernel_base);
        let _ = writeln!(vmcoreinfo, "HHDMOFFSET={:#x}", self.meta.hhdm_offset);
        let _ = writeln!(vmcoreinfo, "PANIC={}", self.panic_reason);

        append_elf_note(
            &mut notes,
            b"VMCOREINFO\0",
            NT_VMCOREINFO,
            vmcoreinfo.as_bytes(),
        );

        notes
    }

    /// Generate a complete ELF64 `vmcore` image in memory.
    pub fn generate(&self) -> Result<Vec<u8>, KdumpError> {
        if self.cpus.is_empty() {
            return Err(KdumpError::NoCpuContext);
        }
        if self.regions.is_empty() {
            return Err(KdumpError::NoMemoryRegions);
        }

        let note_payload = self.build_note_payload();
        let phnum = 1usize + self.regions.len();
        let phdr_table_size = phnum * ELF64_PHDR_SIZE;
        let note_offset = align_up_8(ELF64_EHDR_SIZE + phdr_table_size);

        let mut cursor = align_up_8(note_offset + note_payload.len());
        let mut region_offsets = Vec::with_capacity(self.regions.len());
        for region in &self.regions {
            if region.size as usize > MAX_DUMP_REGION_BYTES {
                return Err(KdumpError::RegionTooLarge);
            }
            region_offsets.push(cursor);
            cursor = align_up_8(cursor + region.size as usize);
        }

        let mut out = Vec::with_capacity(cursor);

        // 1. ELF64 Header (64 bytes)
        out.extend_from_slice(&ELFMAG);
        out.push(ELFCLASS64);
        out.push(ELFDATA2LSB);
        out.push(EV_CURRENT);
        out.push(ELFOSABI_NONE);
        out.resize(16, 0); // e_ident padding
        push_u16_le(&mut out, ET_CORE);
        push_u16_le(&mut out, EM_X86_64);
        push_u32_le(&mut out, EV_CURRENT as u32);
        push_u64_le(&mut out, 0); // e_entry
        push_u64_le(&mut out, ELF64_EHDR_SIZE as u64); // e_phoff
        push_u64_le(&mut out, 0); // e_shoff
        push_u32_le(&mut out, 0); // e_flags
        push_u16_le(&mut out, ELF64_EHDR_SIZE as u16);
        push_u16_le(&mut out, ELF64_PHDR_SIZE as u16);
        push_u16_le(&mut out, phnum as u16);
        push_u16_le(&mut out, 0); // e_shentsize
        push_u16_le(&mut out, 0); // e_shnum
        push_u16_le(&mut out, 0); // e_shstrndx

        // 2. Program Header 0: PT_NOTE
        push_u32_le(&mut out, PT_NOTE);
        push_u32_le(&mut out, PF_R);
        push_u64_le(&mut out, note_offset as u64);
        push_u64_le(&mut out, 0);
        push_u64_le(&mut out, 0);
        push_u64_le(&mut out, note_payload.len() as u64);
        push_u64_le(&mut out, 0);
        push_u64_le(&mut out, 4);

        // 3. Program Headers 1..N: PT_LOAD
        for (region, &off) in self.regions.iter().zip(region_offsets.iter()) {
            push_u32_le(&mut out, PT_LOAD);
            push_u32_le(&mut out, region.flags);
            push_u64_le(&mut out, off as u64);
            push_u64_le(&mut out, region.virt_addr);
            push_u64_le(&mut out, region.phys_addr);
            push_u64_le(&mut out, region.size);
            push_u64_le(&mut out, region.size);
            push_u64_le(&mut out, 4096);
        }

        // 4. PT_NOTE payload
        out.resize(note_offset, 0);
        out.extend_from_slice(&note_payload);

        // 5. PT_LOAD payloads
        for (region, &off) in self.regions.iter().zip(region_offsets.iter()) {
            out.resize(off, 0);
            let sz = region.size as usize;
            match region.source {
                RegionSource::Slice(src) => {
                    let copy_len = src.len().min(sz);
                    out.extend_from_slice(&src[..copy_len]);
                    if copy_len < sz {
                        out.resize(off + sz, 0);
                    }
                }
                RegionSource::Zeroed => {
                    out.resize(off + sz, 0);
                }
                RegionSource::Pattern(byte) => {
                    out.resize(off + sz, byte);
                }
            }
        }

        Ok(out)
    }
}

/// Validate an ELF64 `vmcore` image and return a structural summary.
pub fn validate_vmcore(bytes: &[u8]) -> Result<VmcoreSummary, KdumpError> {
    if bytes.len() < ELF64_EHDR_SIZE {
        return Err(KdumpError::TruncatedVmcore);
    }
    if bytes[0..4] != ELFMAG {
        return Err(KdumpError::InvalidElfMagic);
    }
    if bytes[4] != ELFCLASS64 || bytes[5] != ELFDATA2LSB {
        return Err(KdumpError::InvalidElfClass);
    }
    let e_type = read_u16_le(bytes, 16).ok_or(KdumpError::TruncatedVmcore)?;
    if e_type != ET_CORE {
        return Err(KdumpError::InvalidElfType);
    }
    let e_machine = read_u16_le(bytes, 18).ok_or(KdumpError::TruncatedVmcore)?;
    if e_machine != EM_X86_64 {
        return Err(KdumpError::InvalidElfMachine);
    }

    let e_phoff = read_u64_le(bytes, 32).ok_or(KdumpError::TruncatedVmcore)? as usize;
    let e_phentsize = read_u16_le(bytes, 54).ok_or(KdumpError::TruncatedVmcore)? as usize;
    let e_phnum = read_u16_le(bytes, 56).ok_or(KdumpError::TruncatedVmcore)?;

    if e_phentsize != ELF64_PHDR_SIZE {
        return Err(KdumpError::InvalidProgramHeader);
    }
    let ph_end = e_phoff
        .checked_add((e_phnum as usize) * ELF64_PHDR_SIZE)
        .ok_or(KdumpError::InvalidProgramHeader)?;
    if ph_end > bytes.len() {
        return Err(KdumpError::TruncatedVmcore);
    }

    let mut summary = VmcoreSummary {
        phnum: e_phnum,
        total_bytes: bytes.len(),
        ..VmcoreSummary::default()
    };

    for idx in 0..(e_phnum as usize) {
        let off = e_phoff + idx * ELF64_PHDR_SIZE;
        let p_type = read_u32_le(bytes, off).ok_or(KdumpError::InvalidProgramHeader)?;
        let p_offset =
            read_u64_le(bytes, off + 8).ok_or(KdumpError::InvalidProgramHeader)? as usize;
        let p_filesz =
            read_u64_le(bytes, off + 32).ok_or(KdumpError::InvalidProgramHeader)? as usize;

        let seg_end = p_offset
            .checked_add(p_filesz)
            .ok_or(KdumpError::TruncatedVmcore)?;
        if seg_end > bytes.len() {
            return Err(KdumpError::TruncatedVmcore);
        }

        match p_type {
            PT_NOTE => {
                summary.pt_note_count += 1;
                let note_slice = &bytes[p_offset..seg_end];
                let mut pos = 0usize;
                while pos < note_slice.len() {
                    if pos + ELF64_NHDR_SIZE > note_slice.len() {
                        return Err(KdumpError::InvalidNoteSection);
                    }
                    let namesz = read_u32_le(note_slice, pos)
                        .ok_or(KdumpError::InvalidNoteSection)?
                        as usize;
                    let descsz = read_u32_le(note_slice, pos + 4)
                        .ok_or(KdumpError::InvalidNoteSection)?
                        as usize;
                    let ntype =
                        read_u32_le(note_slice, pos + 8).ok_or(KdumpError::InvalidNoteSection)?;
                    pos += ELF64_NHDR_SIZE;

                    let name_end = pos + align_up_4(namesz);
                    let desc_end = name_end + align_up_4(descsz);
                    if desc_end > note_slice.len() {
                        return Err(KdumpError::InvalidNoteSection);
                    }
                    let name_bytes = &note_slice[pos..pos + namesz];
                    if ntype == NT_PRSTATUS && name_bytes == b"CORE\0" {
                        summary.prstatus_notes += 1;
                    } else if ntype == NT_VMCOREINFO && name_bytes == b"VMCOREINFO\0" {
                        summary.vmcoreinfo_notes += 1;
                    }
                    pos = desc_end;
                }
            }
            PT_LOAD => {
                summary.pt_load_count += 1;
                summary.total_load_bytes += p_filesz as u64;
            }
            _ => {}
        }
    }

    Ok(summary)
}

// ---------------------------------------------------------------------------
// Persistent pstore Panic Log Ring
// ---------------------------------------------------------------------------

pub const PSTORE_MAGIC: u64 = 0x5053_5452_4E47_3031; // "PSTRNG01"
pub const PSTORE_RING_CAP: usize = 16;
pub const PSTORE_MSG_CAP: usize = 192;

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PstoreEventKind {
    Empty = 0,
    Panic = 1,
    Oops = 2,
    WatchdogSoftLockup = 3,
    WatchdogHardLockup = 4,
    Nmi = 5,
    ManualDump = 6,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PstoreRecord {
    pub seq: u64,
    pub timestamp_tsc: u64,
    pub cpu_id: u32,
    pub kind: PstoreEventKind,
    pub msg_len: u16,
    pub _pad: u8,
    pub rip: u64,
    pub rsp: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub msg: [u8; PSTORE_MSG_CAP],
    pub crc32: u32,
}

impl PstoreRecord {
    pub const fn empty() -> Self {
        Self {
            seq: 0,
            timestamp_tsc: 0,
            cpu_id: 0,
            kind: PstoreEventKind::Empty,
            msg_len: 0,
            _pad: 0,
            rip: 0,
            rsp: 0,
            cr2: 0,
            cr3: 0,
            msg: [0u8; PSTORE_MSG_CAP],
            crc32: 0,
        }
    }

    pub fn message_str(&self) -> &str {
        let len = (self.msg_len as usize).min(PSTORE_MSG_CAP);
        core::str::from_utf8(&self.msg[..len]).unwrap_or("<invalid-utf8>")
    }

    pub fn compute_crc32(&self) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        crc = crc32_update(crc, &self.seq.to_le_bytes());
        crc = crc32_update(crc, &self.timestamp_tsc.to_le_bytes());
        crc = crc32_update(crc, &self.cpu_id.to_le_bytes());
        crc = crc32_update(crc, &[self.kind as u8]);
        crc = crc32_update(crc, &self.msg_len.to_le_bytes());
        crc = crc32_update(crc, &self.rip.to_le_bytes());
        crc = crc32_update(crc, &self.rsp.to_le_bytes());
        crc = crc32_update(crc, &self.cr2.to_le_bytes());
        crc = crc32_update(crc, &self.cr3.to_le_bytes());
        let len = (self.msg_len as usize).min(PSTORE_MSG_CAP);
        crc = crc32_update(crc, &self.msg[..len]);
        !crc
    }

    pub fn is_valid(&self) -> bool {
        self.kind != PstoreEventKind::Empty && self.crc32 == self.compute_crc32()
    }
}

#[inline]
fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    crc
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PstoreStats {
    pub initialized: bool,
    pub total_written: u64,
    pub wrapped_count: u64,
    pub valid_records: usize,
    pub corrupted_records: usize,
}

struct PstoreRing {
    magic: u64,
    head: usize,
    count: usize,
    next_seq: u64,
    total_written: u64,
    wrapped_count: u64,
    records: [PstoreRecord; PSTORE_RING_CAP],
}

impl PstoreRing {
    const fn new() -> Self {
        Self {
            magic: 0,
            head: 0,
            count: 0,
            next_seq: 1,
            total_written: 0,
            wrapped_count: 0,
            records: [PstoreRecord::empty(); PSTORE_RING_CAP],
        }
    }

    fn ensure_init(&mut self) {
        if self.magic != PSTORE_MAGIC {
            self.reset();
        }
    }

    fn reset(&mut self) {
        self.magic = PSTORE_MAGIC;
        self.head = 0;
        self.count = 0;
        self.next_seq = 1;
        self.total_written = 0;
        self.wrapped_count = 0;
        for rec in &mut self.records {
            *rec = PstoreRecord::empty();
        }
    }

    fn push(
        &mut self,
        kind: PstoreEventKind,
        cpu_id: u32,
        timestamp_tsc: u64,
        regs: &CpuRegs,
        msg: &str,
    ) -> u64 {
        self.ensure_init();
        let seq = self.next_seq;
        self.next_seq += 1;

        let mut rec = PstoreRecord::empty();
        rec.seq = seq;
        rec.timestamp_tsc = timestamp_tsc;
        rec.cpu_id = cpu_id;
        rec.kind = kind;
        rec.rip = regs.gpr.rip;
        rec.rsp = regs.gpr.rsp;
        rec.cr2 = regs.cr2;
        rec.cr3 = regs.cr3;

        let bytes = msg.as_bytes();
        let copy_len = bytes.len().min(PSTORE_MSG_CAP);
        rec.msg_len = copy_len as u16;
        rec.msg[..copy_len].copy_from_slice(&bytes[..copy_len]);
        rec.crc32 = rec.compute_crc32();

        self.records[self.head] = rec;
        self.head = (self.head + 1) % PSTORE_RING_CAP;
        if self.count < PSTORE_RING_CAP {
            self.count += 1;
        } else {
            self.wrapped_count += 1;
        }
        self.total_written += 1;
        seq
    }
}

static PSTORE: SpinLock<PstoreRing> = SpinLock::new(PstoreRing::new());

pub fn pstore_init() {
    let mut ring = PSTORE.lock();
    ring.ensure_init();
}

pub fn pstore_clear() {
    let mut ring = PSTORE.lock();
    ring.reset();
}

pub fn pstore_record_event(
    kind: PstoreEventKind,
    cpu_id: u32,
    timestamp_tsc: u64,
    regs: &CpuRegs,
    msg: &str,
) -> u64 {
    let mut ring = PSTORE.lock();
    ring.push(kind, cpu_id, timestamp_tsc, regs, msg)
}

pub fn pstore_record_panic(cpu_id: u32, timestamp_tsc: u64, regs: &CpuRegs, msg: &str) -> u64 {
    pstore_record_event(PstoreEventKind::Panic, cpu_id, timestamp_tsc, regs, msg)
}

pub fn pstore_read_latest() -> Option<PstoreRecord> {
    let ring = PSTORE.lock();
    if ring.magic != PSTORE_MAGIC || ring.count == 0 {
        return None;
    }
    let idx = if ring.head == 0 {
        PSTORE_RING_CAP - 1
    } else {
        ring.head - 1
    };
    Some(ring.records[idx])
}

pub fn pstore_read_all() -> Vec<PstoreRecord> {
    let ring = PSTORE.lock();
    if ring.magic != PSTORE_MAGIC || ring.count == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(ring.count);
    let start = if ring.count < PSTORE_RING_CAP {
        0
    } else {
        ring.head
    };
    for offset in 0..ring.count {
        let idx = (start + offset) % PSTORE_RING_CAP;
        out.push(ring.records[idx]);
    }
    out
}

pub fn pstore_stats() -> PstoreStats {
    let ring = PSTORE.lock();
    if ring.magic != PSTORE_MAGIC {
        return PstoreStats::default();
    }
    let mut valid = 0usize;
    let mut corrupted = 0usize;
    let start = if ring.count < PSTORE_RING_CAP {
        0
    } else {
        ring.head
    };
    for offset in 0..ring.count {
        let idx = (start + offset) % PSTORE_RING_CAP;
        if ring.records[idx].is_valid() {
            valid += 1;
        } else {
            corrupted += 1;
        }
    }
    PstoreStats {
        initialized: true,
        total_written: ring.total_written,
        wrapped_count: ring.wrapped_count,
        valid_records: valid,
        corrupted_records: corrupted,
    }
}

// ---------------------------------------------------------------------------
// Watchdog NMI Engine
// ---------------------------------------------------------------------------

pub const MAX_WATCHDOG_CPUS: usize = 16;
pub const DEFAULT_SOFT_LOCKUP_TSC: u64 = 20_000_000_000; // ~10s @ 2 GHz
pub const DEFAULT_HARD_LOCKUP_TSC: u64 = 40_000_000_000; // ~20s @ 2 GHz

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WatchdogStatus {
    Disarmed,
    Healthy,
    SoftLockup,
    HardLockup,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct WatchdogCpuStats {
    pub cpu_id: u32,
    pub armed: bool,
    pub last_heartbeat_tsc: u64,
    pub soft_lockup_count: u64,
    pub hard_lockup_count: u64,
    pub nmi_count: u64,
    pub last_rip: u64,
    pub last_rsp: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WatchdogNmiOutcome {
    pub status: WatchdogStatus,
    pub pstore_seq: Option<u64>,
    pub nmi_count: u64,
}

struct PerCpuWatchdog {
    armed: AtomicBool,
    in_nmi: AtomicBool,
    last_heartbeat_tsc: AtomicU64,
    soft_lockup_count: AtomicU64,
    hard_lockup_count: AtomicU64,
    nmi_count: AtomicU64,
    last_rip: AtomicU64,
    last_rsp: AtomicU64,
}

impl PerCpuWatchdog {
    const fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            in_nmi: AtomicBool::new(false),
            last_heartbeat_tsc: AtomicU64::new(0),
            soft_lockup_count: AtomicU64::new(0),
            hard_lockup_count: AtomicU64::new(0),
            nmi_count: AtomicU64::new(0),
            last_rip: AtomicU64::new(0),
            last_rsp: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.armed.store(false, Ordering::Release);
        self.in_nmi.store(false, Ordering::Release);
        self.last_heartbeat_tsc.store(0, Ordering::Release);
        self.soft_lockup_count.store(0, Ordering::Release);
        self.hard_lockup_count.store(0, Ordering::Release);
        self.nmi_count.store(0, Ordering::Release);
        self.last_rip.store(0, Ordering::Release);
        self.last_rsp.store(0, Ordering::Release);
    }
}

static WATCHDOG_CPUS: [PerCpuWatchdog; MAX_WATCHDOG_CPUS] =
    [const { PerCpuWatchdog::new() }; MAX_WATCHDOG_CPUS];
static SOFT_THRESHOLD_TSC: AtomicU64 = AtomicU64::new(DEFAULT_SOFT_LOCKUP_TSC);
static HARD_THRESHOLD_TSC: AtomicU64 = AtomicU64::new(DEFAULT_HARD_LOCKUP_TSC);

pub fn watchdog_init(soft_threshold_tsc: u64, hard_threshold_tsc: u64) {
    let soft = if soft_threshold_tsc == 0 {
        DEFAULT_SOFT_LOCKUP_TSC
    } else {
        soft_threshold_tsc
    };
    let hard = if hard_threshold_tsc <= soft {
        soft.saturating_mul(2)
    } else {
        hard_threshold_tsc
    };
    SOFT_THRESHOLD_TSC.store(soft, Ordering::Release);
    HARD_THRESHOLD_TSC.store(hard, Ordering::Release);
    for cpu in &WATCHDOG_CPUS {
        cpu.reset();
    }
}

pub fn watchdog_arm_cpu(cpu_id: u32, now_tsc: u64) -> Result<(), KdumpError> {
    let slot = WATCHDOG_CPUS
        .get(cpu_id as usize)
        .ok_or(KdumpError::InvalidCpuId)?;
    slot.last_heartbeat_tsc.store(now_tsc, Ordering::Release);
    slot.armed.store(true, Ordering::Release);
    Ok(())
}

pub fn watchdog_disarm_cpu(cpu_id: u32) -> Result<(), KdumpError> {
    let slot = WATCHDOG_CPUS
        .get(cpu_id as usize)
        .ok_or(KdumpError::InvalidCpuId)?;
    slot.armed.store(false, Ordering::Release);
    Ok(())
}

pub fn watchdog_touch(cpu_id: u32, now_tsc: u64) -> Result<(), KdumpError> {
    let slot = WATCHDOG_CPUS
        .get(cpu_id as usize)
        .ok_or(KdumpError::InvalidCpuId)?;
    slot.last_heartbeat_tsc.store(now_tsc, Ordering::Release);
    Ok(())
}

pub fn watchdog_check_cpu(cpu_id: u32, now_tsc: u64) -> Result<WatchdogStatus, KdumpError> {
    let slot = WATCHDOG_CPUS
        .get(cpu_id as usize)
        .ok_or(KdumpError::InvalidCpuId)?;
    if !slot.armed.load(Ordering::Acquire) {
        return Ok(WatchdogStatus::Disarmed);
    }
    let last = slot.last_heartbeat_tsc.load(Ordering::Acquire);
    let elapsed = now_tsc.saturating_sub(last);
    let hard = HARD_THRESHOLD_TSC.load(Ordering::Acquire);
    let soft = SOFT_THRESHOLD_TSC.load(Ordering::Acquire);

    if elapsed >= hard {
        slot.hard_lockup_count.fetch_add(1, Ordering::Relaxed);
        Ok(WatchdogStatus::HardLockup)
    } else if elapsed >= soft {
        slot.soft_lockup_count.fetch_add(1, Ordering::Relaxed);
        Ok(WatchdogStatus::SoftLockup)
    } else {
        Ok(WatchdogStatus::Healthy)
    }
}

/// NMI watchdog handler: records NMI, checks lockup state, captures registers, and logs to `pstore`.
pub fn watchdog_nmi_handler(
    cpu_id: u32,
    regs: &CpuRegs,
    now_tsc: u64,
) -> Result<WatchdogNmiOutcome, KdumpError> {
    let slot = WATCHDOG_CPUS
        .get(cpu_id as usize)
        .ok_or(KdumpError::InvalidCpuId)?;

    // Guard against nested NMI re-entrancy on the same CPU.
    if slot
        .in_nmi
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Ok(WatchdogNmiOutcome {
            status: WatchdogStatus::HardLockup,
            pstore_seq: None,
            nmi_count: slot.nmi_count.load(Ordering::Relaxed),
        });
    }

    let nmi_count = slot.nmi_count.fetch_add(1, Ordering::Relaxed) + 1;
    slot.last_rip.store(regs.gpr.rip, Ordering::Release);
    slot.last_rsp.store(regs.gpr.rsp, Ordering::Release);

    let status = watchdog_check_cpu(cpu_id, now_tsc)?;
    let pstore_seq = match status {
        WatchdogStatus::HardLockup => Some(pstore_record_event(
            PstoreEventKind::WatchdogHardLockup,
            cpu_id,
            now_tsc,
            regs,
            "NMI watchdog: hard lockup detected",
        )),
        WatchdogStatus::SoftLockup => Some(pstore_record_event(
            PstoreEventKind::WatchdogSoftLockup,
            cpu_id,
            now_tsc,
            regs,
            "NMI watchdog: soft lockup detected",
        )),
        WatchdogStatus::Healthy | WatchdogStatus::Disarmed => None,
    };

    slot.in_nmi.store(false, Ordering::Release);
    Ok(WatchdogNmiOutcome {
        status,
        pstore_seq,
        nmi_count,
    })
}

pub fn watchdog_stats(cpu_id: u32) -> Option<WatchdogCpuStats> {
    let slot = WATCHDOG_CPUS.get(cpu_id as usize)?;
    Some(WatchdogCpuStats {
        cpu_id,
        armed: slot.armed.load(Ordering::Acquire),
        last_heartbeat_tsc: slot.last_heartbeat_tsc.load(Ordering::Acquire),
        soft_lockup_count: slot.soft_lockup_count.load(Ordering::Relaxed),
        hard_lockup_count: slot.hard_lockup_count.load(Ordering::Relaxed),
        nmi_count: slot.nmi_count.load(Ordering::Relaxed),
        last_rip: slot.last_rip.load(Ordering::Acquire),
        last_rsp: slot.last_rsp.load(Ordering::Acquire),
    })
}

// ---------------------------------------------------------------------------
// High-Level Crash Dump Capture
// ---------------------------------------------------------------------------

/// Record a panic in `pstore` and generate an ELF64 `vmcore` image.
pub fn capture_crash_dump<'a>(
    reason: &'a str,
    cpu_id: u32,
    regs: &CpuRegs,
    now_tsc: u64,
    regions: &[MemoryRegion<'a>],
) -> Result<Vec<u8>, KdumpError> {
    pstore_record_panic(cpu_id, now_tsc, regs, reason);

    let meta = VmcoreMetadata {
        crash_tsc: now_tsc,
        crash_cpu: cpu_id,
        kernel_base: 0xFFFF_FFFF_8000_0000,
        hhdm_offset: 0xFFFF_8000_0000_0000,
        page_size: 4096,
    };
    let mut generator = VmcoreGenerator::new(reason, meta);
    generator.add_cpu_regs(cpu_id, *regs);
    for region in regions {
        generator.add_region(region.clone())?;
    }
    generator.generate()
}

// ---------------------------------------------------------------------------
// Self-Test Suite
// ---------------------------------------------------------------------------

pub fn self_test() -> Result<(), &'static str> {
    serial_println!("[kdump] Running kdump / pstore / watchdog NMI self-test...");

    // 1. Test pstore ring wrap-around and CRC32 integrity
    pstore_clear();
    let mut test_regs = CpuRegs::default();
    test_regs.gpr.rip = 0xFFFF_FFFF_8001_2340;
    test_regs.gpr.rsp = 0xFFFF_8000_0010_FF00;
    test_regs.cr2 = 0xDEAD_BEEF_0000_1000;
    test_regs.cr3 = 0x0000_0000_0020_0000;

    for idx in 0..20u64 {
        test_regs.gpr.rax = idx;
        pstore_record_panic((idx % 4) as u32, 1000 + idx * 100, &test_regs, "test panic");
    }

    let stats = pstore_stats();
    if !stats.initialized
        || stats.total_written != 20
        || stats.wrapped_count != 4
        || stats.valid_records != PSTORE_RING_CAP
        || stats.corrupted_records != 0
    {
        return Err("pstore ring wrap-around or CRC32 stats mismatch");
    }

    let latest = pstore_read_latest().ok_or("pstore_read_latest returned None")?;
    if latest.seq != 20 || !latest.is_valid() || latest.message_str() != "test panic" {
        return Err("pstore latest record validation failed");
    }

    // Verify CRC32 detects tampering on a local copy
    let mut tampered = latest;
    tampered.msg[0] ^= 0xFF;
    if tampered.is_valid() {
        return Err("pstore CRC32 failed to detect record tampering");
    }

    // 2. Test Watchdog NMI Engine
    watchdog_init(1_000, 2_000);
    watchdog_arm_cpu(0, 10_000).map_err(|e| e.as_str())?;
    watchdog_arm_cpu(1, 10_000).map_err(|e| e.as_str())?;

    if watchdog_check_cpu(0, 10_500).map_err(|e| e.as_str())? != WatchdogStatus::Healthy {
        return Err("watchdog expected Healthy");
    }
    if watchdog_check_cpu(0, 11_200).map_err(|e| e.as_str())? != WatchdogStatus::SoftLockup {
        return Err("watchdog expected SoftLockup");
    }

    let nmi_outcome = watchdog_nmi_handler(0, &test_regs, 12_500).map_err(|e| e.as_str())?;
    if nmi_outcome.status != WatchdogStatus::HardLockup
        || nmi_outcome.pstore_seq.is_none()
        || nmi_outcome.nmi_count != 1
    {
        return Err("watchdog NMI hard-lockup outcome mismatch");
    }

    watchdog_touch(1, 12_500).map_err(|e| e.as_str())?;
    if watchdog_check_cpu(1, 12_800).map_err(|e| e.as_str())? != WatchdogStatus::Healthy {
        return Err("watchdog touch failed to keep CPU 1 healthy");
    }

    // 3. Test ELF64 vmcore generator & validator
    let sample_mem = [0xA5u8; 256];
    let regions = [
        MemoryRegion {
            phys_addr: 0x0010_0000,
            virt_addr: 0xFFFF_FFFF_8010_0000,
            size: sample_mem.len() as u64,
            flags: PF_R | PF_X,
            source: RegionSource::Slice(&sample_mem),
        },
        MemoryRegion {
            phys_addr: 0x0020_0000,
            virt_addr: 0xFFFF_8000_0020_0000,
            size: 512,
            flags: PF_R | PF_W,
            source: RegionSource::Pattern(0x5A),
        },
    ];

    let vmcore = capture_crash_dump("self_test_panic", 0, &test_regs, 12_600, &regions)
        .map_err(|e| e.as_str())?;
    let summary = validate_vmcore(&vmcore).map_err(|e| e.as_str())?;

    if summary.phnum != 3
        || summary.pt_note_count != 1
        || summary.pt_load_count != 2
        || summary.prstatus_notes != 1
        || summary.vmcoreinfo_notes != 1
        || summary.total_load_bytes != 768
    {
        return Err("vmcore summary mismatch");
    }

    // Clean up test state
    pstore_clear();
    watchdog_init(DEFAULT_SOFT_LOCKUP_TSC, DEFAULT_HARD_LOCKUP_TSC);

    serial_println!(
        "[kdump] self-test passed (vmcore={} bytes, phnum={}, prstatus={}, vmcoreinfo={})",
        summary.total_bytes,
        summary.phnum,
        summary.prstatus_notes,
        summary.vmcoreinfo_notes
    );
    Ok(())
}
