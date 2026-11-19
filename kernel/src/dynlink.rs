#![allow(dead_code)]

//! `no_std` Ring 3 Dynamic Linker & Multi-OS Process Bootstrap.
//!
//! Provides:
//! - ELF64 dynamic section (`PT_DYNAMIC`) parsing, SysV/GNU symbol hashing,
//!   and x86_64 GOT/PLT relocation (`R_X86_64_*`).
//! - ELF `AT_*` auxiliary vector (`auxv`) and SysV x86_64 initial Ring 3 stack builder.
//! - Windows x64 `TEB`, `PEB`, `PEB_LDR_DATA`, `LDR_DATA_TABLE_ENTRY`, and
//!   `RTL_USER_PROCESS_PARAMETERS` (`ntdll` loader block) builder + PE32+ base relocations.
//! - macOS/Darwin x86_64 `dyld` commpage (`_COMM_PAGE64_*`), `LC_MAIN` / `LC_LOAD_DYLINKER`
//!   parsing, and `apple[]` initial stack builder.
//! - Unified `bootstrap_process` helper for mapping bootstrap blocks into a user
//!   [`AddressSpace`] and `self_test()` for kernel boot verification.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;
use core::ptr;

use nexus_abi::Personality;
use nexus_compat::BinaryFormat;

use crate::iommu::enforce_wx;
use crate::mm::{alloc_frame, stats, PAGE_SIZE};
use crate::vmm::{AddressSpace, PageFlags};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the Ring 3 dynamic linker and process bootstrapper.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DynlinkError {
    InvalidHeader,
    OutOfBounds,
    UnsupportedRelocation,
    SymbolNotFound,
    InvalidAlignment,
    OutOfMemory,
    WxViolation,
    MapFailed,
    InvalidUtf8,
}

impl DynlinkError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidHeader => "dynlink: invalid binary or dynamic header",
            Self::OutOfBounds => "dynlink: offset or address out of bounds",
            Self::UnsupportedRelocation => "dynlink: unsupported x86_64 relocation type",
            Self::SymbolNotFound => "dynlink: unresolved dynamic symbol",
            Self::InvalidAlignment => "dynlink: invalid stack or structure alignment",
            Self::OutOfMemory => "dynlink: out of physical frames",
            Self::WxViolation => "dynlink: page flags violate W^X policy",
            Self::MapFailed => "dynlink: failed to map bootstrap page",
            Self::InvalidUtf8 => "dynlink: invalid UTF-8 string",
        }
    }
}

impl From<DynlinkError> for &'static str {
    fn from(err: DynlinkError) -> Self {
        err.as_str()
    }
}

// ---------------------------------------------------------------------------
// Memory Read/Write Abstraction (works on both flat buffers and AddressSpace)
// ---------------------------------------------------------------------------

/// Abstract user-space memory accessor for applying relocations and writing
/// bootstrap structures without large stack buffers.
pub trait UserMemory {
    fn read_bytes(&self, vaddr: u64, out: &mut [u8]) -> Result<(), DynlinkError>;
    fn write_bytes(&mut self, vaddr: u64, data: &[u8]) -> Result<(), DynlinkError>;

    fn read_u16(&self, vaddr: u64) -> Result<u16, DynlinkError> {
        let mut buf = [0u8; 2];
        self.read_bytes(vaddr, &mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_u32(&self, vaddr: u64) -> Result<u32, DynlinkError> {
        let mut buf = [0u8; 4];
        self.read_bytes(vaddr, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&self, vaddr: u64) -> Result<u64, DynlinkError> {
        let mut buf = [0u8; 8];
        self.read_bytes(vaddr, &mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn write_u16(&mut self, vaddr: u64, val: u16) -> Result<(), DynlinkError> {
        self.write_bytes(vaddr, &val.to_le_bytes())
    }

    fn write_u32(&mut self, vaddr: u64, val: u32) -> Result<(), DynlinkError> {
        self.write_bytes(vaddr, &val.to_le_bytes())
    }

    fn write_u64(&mut self, vaddr: u64, val: u64) -> Result<(), DynlinkError> {
        self.write_bytes(vaddr, &val.to_le_bytes())
    }
}

/// In-memory flat buffer backed [`UserMemory`] for testing and offline relocation.
pub struct FlatUserMemory {
    pub base_vaddr: u64,
    pub data: Vec<u8>,
}

impl FlatUserMemory {
    pub fn new(base_vaddr: u64, size: usize) -> Self {
        Self {
            base_vaddr,
            data: vec![0u8; size],
        }
    }

    fn slice_range(&self, vaddr: u64, len: usize) -> Result<core::ops::Range<usize>, DynlinkError> {
        let off = vaddr
            .checked_sub(self.base_vaddr)
            .ok_or(DynlinkError::OutOfBounds)?;
        let start = usize::try_from(off).map_err(|_| DynlinkError::OutOfBounds)?;
        let end = start.checked_add(len).ok_or(DynlinkError::OutOfBounds)?;
        if end > self.data.len() {
            return Err(DynlinkError::OutOfBounds);
        }
        Ok(start..end)
    }
}

impl UserMemory for FlatUserMemory {
    fn read_bytes(&self, vaddr: u64, out: &mut [u8]) -> Result<(), DynlinkError> {
        let range = self.slice_range(vaddr, out.len())?;
        out.copy_from_slice(&self.data[range]);
        Ok(())
    }

    fn write_bytes(&mut self, vaddr: u64, data: &[u8]) -> Result<(), DynlinkError> {
        let range = self.slice_range(vaddr, data.len())?;
        self.data[range].copy_from_slice(data);
        Ok(())
    }
}

/// [`UserMemory`] backed by a kernel [`AddressSpace`] via HHDM physical translation.
pub struct AddressSpaceMemory<'a> {
    pub space: &'a AddressSpace,
    pub hhdm_offset: u64,
}

impl<'a> AddressSpaceMemory<'a> {
    pub fn new(space: &'a AddressSpace) -> Self {
        Self {
            space,
            hhdm_offset: stats().hhdm_offset,
        }
    }
}

impl UserMemory for AddressSpaceMemory<'_> {
    fn read_bytes(&self, vaddr: u64, out: &mut [u8]) -> Result<(), DynlinkError> {
        let page_size = PAGE_SIZE as u64;
        let mut remaining = out.len();
        let mut cur_vaddr = vaddr;
        let mut dst_off = 0usize;

        while remaining > 0 {
            let phys = self
                .space
                .translate(cur_vaddr)
                .ok_or(DynlinkError::OutOfBounds)?;
            let page_off = (cur_vaddr & (page_size - 1)) as usize;
            let chunk = remaining.min(PAGE_SIZE - page_off);
            unsafe {
                let src = self.hhdm_offset.wrapping_add(phys) as *const u8;
                ptr::copy_nonoverlapping(src, out[dst_off..].as_mut_ptr(), chunk);
            }
            cur_vaddr = cur_vaddr
                .checked_add(chunk as u64)
                .ok_or(DynlinkError::OutOfBounds)?;
            dst_off += chunk;
            remaining -= chunk;
        }
        Ok(())
    }

    fn write_bytes(&mut self, vaddr: u64, data: &[u8]) -> Result<(), DynlinkError> {
        let page_size = PAGE_SIZE as u64;
        let mut remaining = data.len();
        let mut cur_vaddr = vaddr;
        let mut src_off = 0usize;

        while remaining > 0 {
            let phys = self
                .space
                .translate(cur_vaddr)
                .ok_or(DynlinkError::OutOfBounds)?;
            let page_off = (cur_vaddr & (page_size - 1)) as usize;
            let chunk = remaining.min(PAGE_SIZE - page_off);
            unsafe {
                let dst = self.hhdm_offset.wrapping_add(phys) as *mut u8;
                ptr::copy_nonoverlapping(data[src_off..].as_ptr(), dst, chunk);
            }
            cur_vaddr = cur_vaddr
                .checked_add(chunk as u64)
                .ok_or(DynlinkError::OutOfBounds)?;
            src_off += chunk;
            remaining -= chunk;
        }
        Ok(())
    }
}

/// Map `num_pages` zeroed user pages at `base_vaddr` with W^X enforcement.
pub fn map_user_pages(
    space: &mut AddressSpace,
    base_vaddr: u64,
    num_pages: usize,
    flags: PageFlags,
) -> Result<(), DynlinkError> {
    if (base_vaddr & ((PAGE_SIZE as u64) - 1)) != 0 {
        return Err(DynlinkError::InvalidAlignment);
    }
    if !enforce_wx(flags) {
        return Err(DynlinkError::WxViolation);
    }
    let hhdm = stats().hhdm_offset;
    let page_size = PAGE_SIZE as u64;
    for i in 0..num_pages {
        let vaddr = base_vaddr
            .checked_add((i as u64) * page_size)
            .ok_or(DynlinkError::OutOfBounds)?;
        if space.translate(vaddr).is_some() {
            continue;
        }
        let frame = alloc_frame().ok_or(DynlinkError::OutOfMemory)?;
        unsafe {
            let ptr = hhdm.wrapping_add(frame) as *mut u8;
            ptr::write_bytes(ptr, 0, PAGE_SIZE);
        }
        space
            .map_page(vaddr, frame, flags)
            .map_err(|_| DynlinkError::MapFailed)?;
    }
    Ok(())
}

// ===========================================================================
// 1. ELF64 Dynamic Linker (PT_DYNAMIC, SysV/GNU Hash, x86_64 Relocations)
// ===========================================================================

pub const DT_NULL: i64 = 0;
pub const DT_NEEDED: i64 = 1;
pub const DT_PLTRELSZ: i64 = 2;
pub const DT_PLTGOT: i64 = 3;
pub const DT_HASH: i64 = 4;
pub const DT_STRTAB: i64 = 5;
pub const DT_SYMTAB: i64 = 6;
pub const DT_RELA: i64 = 7;
pub const DT_RELASZ: i64 = 8;
pub const DT_RELAENT: i64 = 9;
pub const DT_STRSZ: i64 = 10;
pub const DT_SYMENT: i64 = 11;
pub const DT_INIT: i64 = 12;
pub const DT_FINI: i64 = 13;
pub const DT_SONAME: i64 = 14;
pub const DT_RPATH: i64 = 15;
pub const DT_SYMBOLIC: i64 = 16;
pub const DT_PLTREL: i64 = 20;
pub const DT_JMPREL: i64 = 23;
pub const DT_INIT_ARRAY: i64 = 25;
pub const DT_FINI_ARRAY: i64 = 26;
pub const DT_INIT_ARRAYSZ: i64 = 27;
pub const DT_FINI_ARRAYSZ: i64 = 28;
pub const DT_RUNPATH: i64 = 29;
pub const DT_FLAGS: i64 = 30;
pub const DT_GNU_HASH: i64 = 0x6FFF_FEF5;
pub const DT_FLAGS_1: i64 = 0x6FFF_FFFB;

pub const R_X86_64_NONE: u32 = 0;
pub const R_X86_64_64: u32 = 1;
pub const R_X86_64_PC32: u32 = 2;
pub const R_X86_64_GLOB_DAT: u32 = 6;
pub const R_X86_64_JUMP_SLOT: u32 = 7;
pub const R_X86_64_RELATIVE: u32 = 8;
pub const R_X86_64_DTPMOD64: u32 = 16;
pub const R_X86_64_DTPOFF64: u32 = 17;
pub const R_X86_64_TPOFF64: u32 = 18;
pub const R_X86_64_IRELATIVE: u32 = 37;

pub const SHN_UNDEF: u16 = 0;
pub const STB_LOCAL: u8 = 0;
pub const STB_GLOBAL: u8 = 1;
pub const STB_WEAK: u8 = 2;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Elf64Dyn {
    pub d_tag: i64,
    pub d_val: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Elf64Sym {
    pub st_name: u32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: u16,
    pub st_value: u64,
    pub st_size: u64,
}

impl Elf64Sym {
    #[inline]
    pub const fn bind(&self) -> u8 {
        self.st_info >> 4
    }

    #[inline]
    pub const fn sym_type(&self) -> u8 {
        self.st_info & 0x0F
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Elf64Rela {
    pub r_offset: u64,
    pub r_info: u64,
    pub r_addend: i64,
}

impl Elf64Rela {
    #[inline]
    pub const fn sym_index(&self) -> u32 {
        (self.r_info >> 32) as u32
    }

    #[inline]
    pub const fn reloc_type(&self) -> u32 {
        (self.r_info & 0xFFFF_FFFF) as u32
    }
}

/// Parsed `PT_DYNAMIC` metadata for a loaded ELF64 object.
#[derive(Clone, Debug, Default)]
pub struct DynamicInfo {
    pub load_bias: u64,
    pub strtab: u64,
    pub strsz: u64,
    pub symtab: u64,
    pub syment: u64,
    pub hash: u64,
    pub gnu_hash: u64,
    pub rela: u64,
    pub relasz: u64,
    pub relaent: u64,
    pub jmprel: u64,
    pub pltrelsz: u64,
    pub pltgot: u64,
    pub init: u64,
    pub fini: u64,
    pub init_array: u64,
    pub init_arraysz: u64,
    pub fini_array: u64,
    pub fini_arraysz: u64,
    pub needed_offsets: Vec<u64>,
    pub soname_offset: Option<u64>,
    pub symbolic: bool,
}

/// Standard ELF SysV symbol hash function.
pub fn elf_sysv_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 0;
    for &b in name {
        h = (h << 4).wrapping_add(b as u32);
        let g = h & 0xF000_0000;
        if g != 0 {
            h ^= g >> 24;
        }
        h &= !g;
    }
    h
}

/// GNU symbol hash function (`DT_GNU_HASH`).
pub fn elf_gnu_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 5381;
    for &b in name {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h
}

fn read_cstr_from_mem(
    mem: &dyn UserMemory,
    vaddr: u64,
    max_len: usize,
) -> Result<String, DynlinkError> {
    let mut bytes = Vec::new();
    let mut cur = vaddr;
    for _ in 0..max_len {
        let mut b = [0u8; 1];
        mem.read_bytes(cur, &mut b)?;
        if b[0] == 0 {
            break;
        }
        bytes.push(b[0]);
        cur = cur.checked_add(1).ok_or(DynlinkError::OutOfBounds)?;
    }
    String::from_utf8(bytes).map_err(|_| DynlinkError::InvalidUtf8)
}

fn read_sym_from_mem(
    mem: &dyn UserMemory,
    symtab: u64,
    syment: u64,
    idx: u32,
) -> Result<Elf64Sym, DynlinkError> {
    let stride = if syment == 0 {
        size_of::<Elf64Sym>() as u64
    } else {
        syment
    };
    let addr = symtab
        .checked_add((idx as u64) * stride)
        .ok_or(DynlinkError::OutOfBounds)?;
    let mut buf = [0u8; 24];
    mem.read_bytes(addr, &mut buf)?;
    Ok(Elf64Sym {
        st_name: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
        st_info: buf[4],
        st_other: buf[5],
        st_shndx: u16::from_le_bytes([buf[6], buf[7]]),
        st_value: u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]),
        st_size: u64::from_le_bytes([
            buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
        ]),
    })
}

fn read_rela_from_mem(mem: &dyn UserMemory, addr: u64) -> Result<Elf64Rela, DynlinkError> {
    let mut buf = [0u8; 24];
    mem.read_bytes(addr, &mut buf)?;
    Ok(Elf64Rela {
        r_offset: u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]),
        r_info: u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]),
        r_addend: i64::from_le_bytes([
            buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
        ]),
    })
}

impl DynamicInfo {
    /// Parse a `PT_DYNAMIC` table at `dynamic_vaddr` with `load_bias` applied to
    /// pointer-valued `DT_*` entries.
    pub fn parse(
        mem: &dyn UserMemory,
        dynamic_vaddr: u64,
        load_bias: u64,
    ) -> Result<Self, DynlinkError> {
        let mut info = Self {
            load_bias,
            syment: size_of::<Elf64Sym>() as u64,
            relaent: size_of::<Elf64Rela>() as u64,
            ..Self::default()
        };

        let mut cur = dynamic_vaddr;
        for _ in 0..4096 {
            let tag = mem.read_u64(cur)? as i64;
            let val = mem.read_u64(cur + 8)?;
            cur += 16;

            match tag {
                DT_NULL => break,
                DT_NEEDED => info.needed_offsets.push(val),
                DT_PLTRELSZ => info.pltrelsz = val,
                DT_PLTGOT => info.pltgot = load_bias.wrapping_add(val),
                DT_HASH => info.hash = load_bias.wrapping_add(val),
                DT_GNU_HASH => info.gnu_hash = load_bias.wrapping_add(val),
                DT_STRTAB => info.strtab = load_bias.wrapping_add(val),
                DT_SYMTAB => info.symtab = load_bias.wrapping_add(val),
                DT_RELA => info.rela = load_bias.wrapping_add(val),
                DT_RELASZ => info.relasz = val,
                DT_RELAENT => info.relaent = val,
                DT_STRSZ => info.strsz = val,
                DT_SYMENT => info.syment = val,
                DT_INIT => info.init = load_bias.wrapping_add(val),
                DT_FINI => info.fini = load_bias.wrapping_add(val),
                DT_SONAME => info.soname_offset = Some(val),
                DT_SYMBOLIC => info.symbolic = true,
                DT_JMPREL => info.jmprel = load_bias.wrapping_add(val),
                DT_INIT_ARRAY => info.init_array = load_bias.wrapping_add(val),
                DT_FINI_ARRAY => info.fini_array = load_bias.wrapping_add(val),
                DT_INIT_ARRAYSZ => info.init_arraysz = val,
                DT_FINI_ARRAYSZ => info.fini_arraysz = val,
                _ => {}
            }
        }

        Ok(info)
    }

    /// Resolve `DT_NEEDED` library names from `DT_STRTAB`.
    pub fn needed_libraries(&self, mem: &dyn UserMemory) -> Result<Vec<String>, DynlinkError> {
        let mut libs = Vec::with_capacity(self.needed_offsets.len());
        for &off in &self.needed_offsets {
            let addr = self
                .strtab
                .checked_add(off)
                .ok_or(DynlinkError::OutOfBounds)?;
            libs.push(read_cstr_from_mem(mem, addr, 256)?);
        }
        Ok(libs)
    }

    /// Look up an exported symbol by name using `DT_HASH` (SysV) or `DT_GNU_HASH`.
    pub fn lookup_symbol(
        &self,
        mem: &dyn UserMemory,
        name: &str,
    ) -> Result<Option<u64>, DynlinkError> {
        if self.symtab == 0 || self.strtab == 0 {
            return Ok(None);
        }

        if self.hash != 0 {
            let nbucket = mem.read_u32(self.hash)?;
            let _nchain = mem.read_u32(self.hash + 4)?;
            if nbucket > 0 {
                let h = elf_sysv_hash(name.as_bytes());
                let bucket_idx = h % nbucket;
                let buckets_base = self.hash + 8;
                let chains_base = buckets_base + (nbucket as u64) * 4;
                let mut sym_idx = mem.read_u32(buckets_base + (bucket_idx as u64) * 4)?;
                let mut guard = 0u32;
                while sym_idx != 0 && guard < 4096 {
                    guard += 1;
                    let sym = read_sym_from_mem(mem, self.symtab, self.syment, sym_idx)?;
                    if sym.st_shndx != SHN_UNDEF {
                        let sym_name = read_cstr_from_mem(
                            mem,
                            self.strtab.wrapping_add(sym.st_name as u64),
                            256,
                        )?;
                        if sym_name == name {
                            return Ok(Some(self.load_bias.wrapping_add(sym.st_value)));
                        }
                    }
                    sym_idx = mem.read_u32(chains_base + (sym_idx as u64) * 4)?;
                }
            }
        }

        if self.gnu_hash != 0 {
            let nbuckets = mem.read_u32(self.gnu_hash)?;
            let symoffset = mem.read_u32(self.gnu_hash + 4)?;
            let bloom_size = mem.read_u32(self.gnu_hash + 8)?;
            let _bloom_shift = mem.read_u32(self.gnu_hash + 12)?;
            if nbuckets > 0 {
                let h = elf_gnu_hash(name.as_bytes());
                let buckets_base = self.gnu_hash + 16 + (bloom_size as u64) * 8;
                let chains_base = buckets_base + (nbuckets as u64) * 4;
                let mut sym_idx = mem.read_u32(buckets_base + ((h % nbuckets) as u64) * 4)?;
                if sym_idx >= symoffset {
                    let mut guard = 0u32;
                    while guard < 4096 {
                        guard += 1;
                        let chain_val =
                            mem.read_u32(chains_base + ((sym_idx - symoffset) as u64) * 4)?;
                        if (chain_val & !1) == (h & !1) {
                            let sym = read_sym_from_mem(mem, self.symtab, self.syment, sym_idx)?;
                            if sym.st_shndx != SHN_UNDEF {
                                let sym_name = read_cstr_from_mem(
                                    mem,
                                    self.strtab.wrapping_add(sym.st_name as u64),
                                    256,
                                )?;
                                if sym_name == name {
                                    return Ok(Some(self.load_bias.wrapping_add(sym.st_value)));
                                }
                            }
                        }
                        if (chain_val & 1) != 0 {
                            break;
                        }
                        sym_idx += 1;
                    }
                }
            }
        }

        Ok(None)
    }

    /// Apply all `DT_RELA` and `DT_JMPREL` relocations using `resolver` for external symbols.
    pub fn apply_relocations<F>(
        &self,
        mem: &mut dyn UserMemory,
        mut resolver: F,
    ) -> Result<usize, DynlinkError>
    where
        F: FnMut(&str) -> Option<u64>,
    {
        let mut count = 0usize;
        let stride = if self.relaent == 0 {
            size_of::<Elf64Rela>() as u64
        } else {
            self.relaent
        };

        if self.rela != 0 && self.relasz > 0 {
            let n = self.relasz / stride;
            for i in 0..n {
                let rela = read_rela_from_mem(mem, self.rela + i * stride)?;
                self.apply_one_rela(mem, &rela, &mut resolver)?;
                count += 1;
            }
        }

        if self.jmprel != 0 && self.pltrelsz > 0 {
            let n = self.pltrelsz / stride;
            for i in 0..n {
                let rela = read_rela_from_mem(mem, self.jmprel + i * stride)?;
                self.apply_one_rela(mem, &rela, &mut resolver)?;
                count += 1;
            }
        }

        Ok(count)
    }

    fn apply_one_rela<F>(
        &self,
        mem: &mut dyn UserMemory,
        rela: &Elf64Rela,
        resolver: &mut F,
    ) -> Result<(), DynlinkError>
    where
        F: FnMut(&str) -> Option<u64>,
    {
        let target = self.load_bias.wrapping_add(rela.r_offset);
        let rtype = rela.reloc_type();

        match rtype {
            R_X86_64_NONE => Ok(()),
            R_X86_64_RELATIVE => {
                let val = (self.load_bias as i64).wrapping_add(rela.r_addend) as u64;
                mem.write_u64(target, val)
            }
            R_X86_64_64 | R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT | R_X86_64_PC32 => {
                let sym_addr = self.resolve_rela_symbol(mem, rela.sym_index(), resolver)?;
                let val = match rtype {
                    R_X86_64_64 => (sym_addr as i64).wrapping_add(rela.r_addend) as u64,
                    R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT => sym_addr,
                    R_X86_64_PC32 => {
                        let rel = (sym_addr as i64)
                            .wrapping_add(rela.r_addend)
                            .wrapping_sub(target as i64);
                        mem.write_u32(target, rel as i32 as u32)?;
                        return Ok(());
                    }
                    _ => unreachable!(),
                };
                mem.write_u64(target, val)
            }
            R_X86_64_IRELATIVE => {
                let val = (self.load_bias as i64).wrapping_add(rela.r_addend) as u64;
                mem.write_u64(target, val)
            }
            R_X86_64_DTPMOD64 => mem.write_u64(target, 1),
            R_X86_64_DTPOFF64 | R_X86_64_TPOFF64 => {
                let sym_addr = self.resolve_rela_symbol(mem, rela.sym_index(), resolver)?;
                let val = (sym_addr as i64).wrapping_add(rela.r_addend) as u64;
                mem.write_u64(target, val)
            }
            _ => Err(DynlinkError::UnsupportedRelocation),
        }
    }

    fn resolve_rela_symbol<F>(
        &self,
        mem: &dyn UserMemory,
        sym_idx: u32,
        resolver: &mut F,
    ) -> Result<u64, DynlinkError>
    where
        F: FnMut(&str) -> Option<u64>,
    {
        if sym_idx == 0 {
            return Ok(self.load_bias);
        }
        let sym = read_sym_from_mem(mem, self.symtab, self.syment, sym_idx)?;
        if sym.st_shndx != SHN_UNDEF && !self.symbolic {
            // Defined locally; still allow global interposition if resolver provides it.
            let name = read_cstr_from_mem(mem, self.strtab.wrapping_add(sym.st_name as u64), 256)?;
            if let Some(addr) = resolver(&name) {
                return Ok(addr);
            }
            return Ok(self.load_bias.wrapping_add(sym.st_value));
        }
        if sym.st_shndx != SHN_UNDEF {
            return Ok(self.load_bias.wrapping_add(sym.st_value));
        }
        let name = read_cstr_from_mem(mem, self.strtab.wrapping_add(sym.st_name as u64), 256)?;
        if let Some(addr) = resolver(&name) {
            return Ok(addr);
        }
        if sym.bind() == STB_WEAK {
            return Ok(0);
        }
        Err(DynlinkError::SymbolNotFound)
    }
}

// ===========================================================================
// 2. ELF AT_* Auxiliary Vector & SysV x86_64 Initial Stack Builder
// ===========================================================================

pub const AT_NULL: u64 = 0;
pub const AT_IGNORE: u64 = 1;
pub const AT_EXECFD: u64 = 2;
pub const AT_PHDR: u64 = 3;
pub const AT_PHENT: u64 = 4;
pub const AT_PHNUM: u64 = 5;
pub const AT_PAGESZ: u64 = 6;
pub const AT_BASE: u64 = 7;
pub const AT_FLAGS: u64 = 8;
pub const AT_ENTRY: u64 = 9;
pub const AT_NOTELF: u64 = 10;
pub const AT_UID: u64 = 11;
pub const AT_EUID: u64 = 12;
pub const AT_GID: u64 = 13;
pub const AT_EGID: u64 = 14;
pub const AT_PLATFORM: u64 = 15;
pub const AT_HWCAP: u64 = 16;
pub const AT_CLKTCK: u64 = 17;
pub const AT_SECURE: u64 = 23;
pub const AT_BASE_PLATFORM: u64 = 24;
pub const AT_RANDOM: u64 = 25;
pub const AT_HWCAP2: u64 = 26;
pub const AT_EXECFN: u64 = 31;
pub const AT_SYSINFO_EHDR: u64 = 33;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AuxvEntry {
    pub a_type: u64,
    pub a_val: u64,
}

/// Parameters for building an ELF64 SysV x86_64 initial Ring 3 stack.
#[derive(Clone, Debug)]
pub struct ElfAuxvConfig<'a> {
    pub phdr_vaddr: u64,
    pub phent_size: u64,
    pub phnum: u64,
    pub entry_vaddr: u64,
    pub interp_base: u64,
    pub vdso_ehdr: u64,
    pub uid: u64,
    pub gid: u64,
    pub hwcap: u64,
    pub random_seed: [u8; 16],
    pub execfn: &'a str,
    pub argv: &'a [&'a str],
    pub envp: &'a [&'a str],
}

/// Result of laying out an ELF64 initial stack.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ElfStackLayout {
    pub initial_rsp: u64,
    pub auxv_vaddr: u64,
    pub random_vaddr: u64,
    pub platform_vaddr: u64,
    pub execfn_vaddr: u64,
}

/// Build the SysV x86_64 initial stack (`argc`, `argv`, `envp`, `auxv`, strings)
/// growing downward from `stack_top` into `mem`.
pub fn build_elf_initial_stack(
    mem: &mut dyn UserMemory,
    stack_top: u64,
    cfg: &ElfAuxvConfig<'_>,
) -> Result<ElfStackLayout, DynlinkError> {
    if (stack_top & 0xF) != 0 {
        return Err(DynlinkError::InvalidAlignment);
    }

    let mut sp = stack_top;

    // 1. Push 16 random bytes for AT_RANDOM.
    sp -= 16;
    mem.write_bytes(sp, &cfg.random_seed)?;
    let random_vaddr = sp;

    // 2. Push "x86_64\0" for AT_PLATFORM.
    let platform = b"x86_64\0";
    sp -= platform.len() as u64;
    mem.write_bytes(sp, platform)?;
    let platform_vaddr = sp;

    // 3. Push execfn string.
    let execfn_bytes = cfg.execfn.as_bytes();
    sp -= (execfn_bytes.len() + 1) as u64;
    mem.write_bytes(sp, execfn_bytes)?;
    mem.write_bytes(sp + execfn_bytes.len() as u64, &[0])?;
    let execfn_vaddr = sp;

    // 4. Push envp strings.
    let mut env_ptrs = Vec::with_capacity(cfg.envp.len());
    for &env in cfg.envp.iter().rev() {
        let b = env.as_bytes();
        sp -= (b.len() + 1) as u64;
        mem.write_bytes(sp, b)?;
        mem.write_bytes(sp + b.len() as u64, &[0])?;
        env_ptrs.push(sp);
    }
    env_ptrs.reverse();

    // 5. Push argv strings.
    let mut arg_ptrs = Vec::with_capacity(cfg.argv.len());
    for &arg in cfg.argv.iter().rev() {
        let b = arg.as_bytes();
        sp -= (b.len() + 1) as u64;
        mem.write_bytes(sp, b)?;
        mem.write_bytes(sp + b.len() as u64, &[0])?;
        arg_ptrs.push(sp);
    }
    arg_ptrs.reverse();

    // Align down to 16 bytes before computing pointer table size.
    sp &= !0xFu64;

    // Build auxv entries.
    let mut auxv = Vec::with_capacity(18);
    auxv.push(AuxvEntry {
        a_type: AT_PHDR,
        a_val: cfg.phdr_vaddr,
    });
    auxv.push(AuxvEntry {
        a_type: AT_PHENT,
        a_val: cfg.phent_size,
    });
    auxv.push(AuxvEntry {
        a_type: AT_PHNUM,
        a_val: cfg.phnum,
    });
    auxv.push(AuxvEntry {
        a_type: AT_PAGESZ,
        a_val: PAGE_SIZE as u64,
    });
    auxv.push(AuxvEntry {
        a_type: AT_BASE,
        a_val: cfg.interp_base,
    });
    auxv.push(AuxvEntry {
        a_type: AT_FLAGS,
        a_val: 0,
    });
    auxv.push(AuxvEntry {
        a_type: AT_ENTRY,
        a_val: cfg.entry_vaddr,
    });
    auxv.push(AuxvEntry {
        a_type: AT_UID,
        a_val: cfg.uid,
    });
    auxv.push(AuxvEntry {
        a_type: AT_EUID,
        a_val: cfg.uid,
    });
    auxv.push(AuxvEntry {
        a_type: AT_GID,
        a_val: cfg.gid,
    });
    auxv.push(AuxvEntry {
        a_type: AT_EGID,
        a_val: cfg.gid,
    });
    auxv.push(AuxvEntry {
        a_type: AT_PLATFORM,
        a_val: platform_vaddr,
    });
    auxv.push(AuxvEntry {
        a_type: AT_HWCAP,
        a_val: cfg.hwcap,
    });
    auxv.push(AuxvEntry {
        a_type: AT_CLKTCK,
        a_val: 100,
    });
    auxv.push(AuxvEntry {
        a_type: AT_SECURE,
        a_val: 0,
    });
    auxv.push(AuxvEntry {
        a_type: AT_RANDOM,
        a_val: random_vaddr,
    });
    auxv.push(AuxvEntry {
        a_type: AT_EXECFN,
        a_val: execfn_vaddr,
    });
    if cfg.vdso_ehdr != 0 {
        auxv.push(AuxvEntry {
            a_type: AT_SYSINFO_EHDR,
            a_val: cfg.vdso_ehdr,
        });
    }
    auxv.push(AuxvEntry {
        a_type: AT_NULL,
        a_val: 0,
    });

    // Total qwords: 1 (argc) + argv.len() + 1 (NULL) + envp.len() + 1 (NULL) + auxv.len() * 2
    let total_qwords = 1 + arg_ptrs.len() + 1 + env_ptrs.len() + 1 + auxv.len() * 2;
    let table_bytes = (total_qwords as u64) * 8;
    sp -= table_bytes;
    // Ensure final RSP is 16-byte aligned.
    if (sp & 0xF) != 0 {
        sp -= 8;
    }
    let initial_rsp = sp;

    let mut cursor = initial_rsp;
    mem.write_u64(cursor, arg_ptrs.len() as u64)?;
    cursor += 8;

    for ptr_val in arg_ptrs {
        mem.write_u64(cursor, ptr_val)?;
        cursor += 8;
    }
    mem.write_u64(cursor, 0)?;
    cursor += 8;

    for ptr_val in env_ptrs {
        mem.write_u64(cursor, ptr_val)?;
        cursor += 8;
    }
    mem.write_u64(cursor, 0)?;
    cursor += 8;

    let auxv_vaddr = cursor;
    for entry in auxv {
        mem.write_u64(cursor, entry.a_type)?;
        mem.write_u64(cursor + 8, entry.a_val)?;
        cursor += 16;
    }

    Ok(ElfStackLayout {
        initial_rsp,
        auxv_vaddr,
        random_vaddr,
        platform_vaddr,
        execfn_vaddr,
    })
}

// ===========================================================================
// 3. Windows x64 PEB / TEB + ntdll Loader Block (PEB_LDR_DATA)
// ===========================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct ListEntry64 {
    pub flink: u64,
    pub blink: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct UnicodeString64 {
    pub length: u16,
    pub maximum_length: u16,
    pub _pad: u32,
    pub buffer: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct NtTib64 {
    pub exception_list: u64,
    pub stack_base: u64,
    pub stack_limit: u64,
    pub sub_system_tib: u64,
    pub fiber_data: u64,
    pub arbitrary_user_pointer: u64,
    pub self_ptr: u64,
}

/// Windows x64 Thread Environment Block (`TEB`, `gs:[0x00]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Teb64 {
    pub nt_tib: NtTib64,                       // 0x00..0x38
    pub environment_pointer: u64,              // 0x38
    pub unique_process: u64,                   // 0x40 (CLIENT_ID.UniqueProcess)
    pub unique_thread: u64,                    // 0x48 (CLIENT_ID.UniqueThread)
    pub active_rpc_handle: u64,                // 0x50
    pub thread_local_storage_pointer: u64,     // 0x58
    pub process_environment_block: u64,        // 0x60
    pub last_error_value: u32,                 // 0x68
    pub count_of_owned_critical_sections: u32, // 0x6C
}

/// Windows x64 Process Environment Block (`PEB`, `gs:[0x60]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Peb64 {
    pub inherited_address_space: u8,      // 0x00
    pub read_image_file_exec_options: u8, // 0x01
    pub being_debugged: u8,               // 0x02
    pub bit_field: u8,                    // 0x03
    pub _pad0: u32,                       // 0x04
    pub mutant: u64,                      // 0x08
    pub image_base_address: u64,          // 0x10
    pub ldr: u64,                         // 0x18 -> PEB_LDR_DATA
    pub process_parameters: u64,          // 0x20 -> RTL_USER_PROCESS_PARAMETERS
    pub sub_system_data: u64,             // 0x28
    pub process_heap: u64,                // 0x30
    pub fast_peb_lock: u64,               // 0x38
    pub _reserved0: [u64; 16],            // 0x40..0xC0
    pub os_major_version: u32,            // 0xC0 (approx compat fields)
    pub os_minor_version: u32,            // 0xC4
    pub os_build_number: u16,             // 0xC8
    pub os_csd_version: u16,              // 0xCA
    pub os_platform_id: u32,              // 0xCC
    pub number_of_processors: u32,        // 0xD0
    pub _pad1: u32,                       // 0xD4
}

/// Windows x64 `PEB_LDR_DATA`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PebLdrData64 {
    pub length: u32,
    pub initialized: u8,
    pub _pad: [u8; 3],
    pub ss_handle: u64,
    pub in_load_order_module_list: ListEntry64,
    pub in_memory_order_module_list: ListEntry64,
    pub in_initialization_order_module_list: ListEntry64,
}

/// Windows x64 `LDR_DATA_TABLE_ENTRY`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct LdrDataTableEntry64 {
    pub in_load_order_links: ListEntry64,           // +0x00
    pub in_memory_order_links: ListEntry64,         // +0x10
    pub in_initialization_order_links: ListEntry64, // +0x20
    pub dll_base: u64,                              // +0x30
    pub entry_point: u64,                           // +0x38
    pub size_of_image: u32,                         // +0x40
    pub _pad0: u32,                                 // +0x44
    pub full_dll_name: UnicodeString64,             // +0x48
    pub base_dll_name: UnicodeString64,             // +0x58
    pub flags: u32,                                 // +0x68
    pub load_count: u16,                            // +0x6C
    pub tls_index: u16,                             // +0x6E
    pub time_date_stamp: u32,                       // +0x70
    pub _pad1: u32,                                 // +0x74
}

/// Windows x64 `RTL_USER_PROCESS_PARAMETERS` (minimal subset).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct RtlUserProcessParameters64 {
    pub maximum_length: u32,
    pub length: u32,
    pub flags: u32,
    pub debug_flags: u32,
    pub console_handle: u64,
    pub console_flags: u32,
    pub _pad0: u32,
    pub standard_input: u64,
    pub standard_output: u64,
    pub standard_error: u64,
    pub current_directory_dos_path: UnicodeString64,
    pub current_directory_handle: u64,
    pub dll_path: UnicodeString64,
    pub image_path_name: UnicodeString64,
    pub command_line: UnicodeString64,
    pub environment: u64,
}

/// Module descriptor for populating `PEB_LDR_DATA`.
#[derive(Clone, Debug)]
pub struct WinModuleDesc<'a> {
    pub dll_base: u64,
    pub entry_point: u64,
    pub size_of_image: u32,
    pub full_name: &'a str,
    pub base_name: &'a str,
}

/// Configuration for building a Windows x64 process environment.
#[derive(Clone, Debug)]
pub struct WindowsBootstrapConfig<'a> {
    pub block_base: u64,
    pub pid: u64,
    pub tid: u64,
    pub stack_base: u64,
    pub stack_limit: u64,
    pub image_base: u64,
    pub process_heap: u64,
    pub image_path: &'a str,
    pub command_line: &'a str,
    pub modules: &'a [WinModuleDesc<'a>],
}

/// Addresses of the constructed Windows x64 bootstrap structures.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WindowsBootstrapLayout {
    pub teb_vaddr: u64,
    pub peb_vaddr: u64,
    pub ldr_vaddr: u64,
    pub params_vaddr: u64,
    pub first_ldr_entry_vaddr: u64,
}

unsafe fn struct_as_bytes<T: Sized>(val: &T) -> &[u8] {
    core::slice::from_raw_parts((val as *const T) as *const u8, size_of::<T>())
}

fn write_utf16le_string(
    mem: &mut dyn UserMemory,
    cursor: &mut u64,
    s: &str,
) -> Result<UnicodeString64, DynlinkError> {
    let start = *cursor;
    let mut byte_len = 0u16;
    for unit in s.encode_utf16() {
        mem.write_u16(*cursor, unit)?;
        *cursor += 2;
        byte_len = byte_len.saturating_add(2);
    }
    mem.write_u16(*cursor, 0)?;
    *cursor += 2;
    // Align cursor to 8 bytes for subsequent structs.
    *cursor = (*cursor + 7) & !7;
    Ok(UnicodeString64 {
        length: byte_len,
        maximum_length: byte_len.saturating_add(2),
        _pad: 0,
        buffer: start,
    })
}

/// Build `TEB`, `PEB`, `PEB_LDR_DATA`, `LDR_DATA_TABLE_ENTRY` circular lists,
/// and `RTL_USER_PROCESS_PARAMETERS` at `cfg.block_base`.
pub fn build_windows_bootstrap(
    mem: &mut dyn UserMemory,
    cfg: &WindowsBootstrapConfig<'_>,
) -> Result<WindowsBootstrapLayout, DynlinkError> {
    if (cfg.block_base & 0xF) != 0 {
        return Err(DynlinkError::InvalidAlignment);
    }

    let teb_vaddr = cfg.block_base;
    let peb_vaddr = teb_vaddr + ((size_of::<Teb64>() as u64 + 15) & !15);
    let ldr_vaddr = peb_vaddr + ((size_of::<Peb64>() as u64 + 15) & !15);
    let params_vaddr = ldr_vaddr + ((size_of::<PebLdrData64>() as u64 + 15) & !15);
    let entries_base = params_vaddr + ((size_of::<RtlUserProcessParameters64>() as u64 + 15) & !15);
    let entry_stride = (size_of::<LdrDataTableEntry64>() as u64 + 15) & !15;
    let mut str_cursor = entries_base + (cfg.modules.len() as u64) * entry_stride;

    // Write strings for process parameters.
    let image_path_us = write_utf16le_string(mem, &mut str_cursor, cfg.image_path)?;
    let cmd_line_us = write_utf16le_string(mem, &mut str_cursor, cfg.command_line)?;
    let cur_dir_us = write_utf16le_string(mem, &mut str_cursor, "C:\\")?;

    let mut params = Box::new(RtlUserProcessParameters64 {
        maximum_length: size_of::<RtlUserProcessParameters64>() as u32,
        length: size_of::<RtlUserProcessParameters64>() as u32,
        flags: 1, // RTL_USER_PROC_PARAMS_NORMALIZED
        current_directory_dos_path: cur_dir_us,
        image_path_name: image_path_us,
        command_line: cmd_line_us,
        ..RtlUserProcessParameters64::default()
    });
    params.standard_input = 0x10;
    params.standard_output = 0x14;
    params.standard_error = 0x18;
    mem.write_bytes(params_vaddr, unsafe { struct_as_bytes(&*params) })?;

    // Build circular doubly-linked lists for PEB_LDR_DATA.
    let load_head = ldr_vaddr + 16; // offsetof(PebLdrData64, in_load_order_module_list)
    let mem_head = ldr_vaddr + 32; // offsetof(PebLdrData64, in_memory_order_module_list)
    let init_head = ldr_vaddr + 48; // offsetof(PebLdrData64, in_initialization_order_module_list)

    let n = cfg.modules.len();
    let mut ldr = Box::new(PebLdrData64 {
        length: size_of::<PebLdrData64>() as u32,
        initialized: 1,
        _pad: [0; 3],
        ss_handle: 0,
        in_load_order_module_list: ListEntry64 {
            flink: if n > 0 { entries_base } else { load_head },
            blink: if n > 0 {
                entries_base + ((n - 1) as u64) * entry_stride
            } else {
                load_head
            },
        },
        in_memory_order_module_list: ListEntry64 {
            flink: if n > 0 { entries_base + 16 } else { mem_head },
            blink: if n > 0 {
                entries_base + ((n - 1) as u64) * entry_stride + 16
            } else {
                mem_head
            },
        },
        in_initialization_order_module_list: ListEntry64 {
            flink: if n > 0 { entries_base + 32 } else { init_head },
            blink: if n > 0 {
                entries_base + ((n - 1) as u64) * entry_stride + 32
            } else {
                init_head
            },
        },
    });
    mem.write_bytes(ldr_vaddr, unsafe { struct_as_bytes(&*ldr) })?;
    let _ = &mut ldr;

    for (i, m) in cfg.modules.iter().enumerate() {
        let cur = entries_base + (i as u64) * entry_stride;
        let prev_load = if i == 0 {
            load_head
        } else {
            entries_base + ((i - 1) as u64) * entry_stride
        };
        let next_load = if i + 1 == n {
            load_head
        } else {
            entries_base + ((i + 1) as u64) * entry_stride
        };

        let prev_mem = if i == 0 {
            mem_head
        } else {
            entries_base + ((i - 1) as u64) * entry_stride + 16
        };
        let next_mem = if i + 1 == n {
            mem_head
        } else {
            entries_base + ((i + 1) as u64) * entry_stride + 16
        };

        let prev_init = if i == 0 {
            init_head
        } else {
            entries_base + ((i - 1) as u64) * entry_stride + 32
        };
        let next_init = if i + 1 == n {
            init_head
        } else {
            entries_base + ((i + 1) as u64) * entry_stride + 32
        };

        let full_us = write_utf16le_string(mem, &mut str_cursor, m.full_name)?;
        let base_us = write_utf16le_string(mem, &mut str_cursor, m.base_name)?;

        let entry = Box::new(LdrDataTableEntry64 {
            in_load_order_links: ListEntry64 {
                flink: next_load,
                blink: prev_load,
            },
            in_memory_order_links: ListEntry64 {
                flink: next_mem,
                blink: prev_mem,
            },
            in_initialization_order_links: ListEntry64 {
                flink: next_init,
                blink: prev_init,
            },
            dll_base: m.dll_base,
            entry_point: m.entry_point,
            size_of_image: m.size_of_image,
            _pad0: 0,
            full_dll_name: full_us,
            base_dll_name: base_us,
            flags: 0x0000_4004,
            load_count: 1,
            tls_index: 0,
            time_date_stamp: 0,
            _pad1: 0,
        });
        mem.write_bytes(cur, unsafe { struct_as_bytes(&*entry) })?;
    }

    let peb = Box::new(Peb64 {
        image_base_address: cfg.image_base,
        ldr: ldr_vaddr,
        process_parameters: params_vaddr,
        process_heap: cfg.process_heap,
        os_major_version: 10,
        os_minor_version: 0,
        os_build_number: 19045,
        os_platform_id: 2, // VER_PLATFORM_WIN32_NT
        number_of_processors: 1,
        ..Peb64::default()
    });
    mem.write_bytes(peb_vaddr, unsafe { struct_as_bytes(&*peb) })?;

    let teb = Box::new(Teb64 {
        nt_tib: NtTib64 {
            exception_list: u64::MAX,
            stack_base: cfg.stack_base,
            stack_limit: cfg.stack_limit,
            sub_system_tib: 0,
            fiber_data: 0,
            arbitrary_user_pointer: 0,
            self_ptr: teb_vaddr,
        },
        environment_pointer: 0,
        unique_process: cfg.pid,
        unique_thread: cfg.tid,
        active_rpc_handle: 0,
        thread_local_storage_pointer: 0,
        process_environment_block: peb_vaddr,
        last_error_value: 0,
        count_of_owned_critical_sections: 0,
    });
    mem.write_bytes(teb_vaddr, unsafe { struct_as_bytes(&*teb) })?;

    Ok(WindowsBootstrapLayout {
        teb_vaddr,
        peb_vaddr,
        ldr_vaddr,
        params_vaddr,
        first_ldr_entry_vaddr: if n > 0 { entries_base } else { 0 },
    })
}

/// Apply PE32+ `.reloc` base relocations (`IMAGE_REL_BASED_DIR64 = 10`) when an
/// image is loaded at `actual_base` instead of `preferred_base`.
pub fn apply_pe_base_relocations(
    mem: &mut dyn UserMemory,
    reloc_vaddr: u64,
    reloc_size: u32,
    actual_base: u64,
    preferred_base: u64,
) -> Result<usize, DynlinkError> {
    let delta = actual_base.wrapping_sub(preferred_base) as i64;
    if delta == 0 || reloc_size == 0 {
        return Ok(0);
    }

    let mut offset = 0u32;
    let mut applied = 0usize;
    while offset + 8 <= reloc_size {
        let block_addr = reloc_vaddr + offset as u64;
        let page_rva = mem.read_u32(block_addr)?;
        let block_size = mem.read_u32(block_addr + 4)?;
        if block_size < 8 || offset + block_size > reloc_size {
            break;
        }
        let count = (block_size - 8) / 2;
        for i in 0..count {
            let entry = mem.read_u16(block_addr + 8 + (i as u64) * 2)?;
            let rtype = entry >> 12;
            let rva_off = (entry & 0x0FFF) as u64;
            if rtype == 10 {
                // IMAGE_REL_BASED_DIR64
                let target = actual_base + page_rva as u64 + rva_off;
                let cur = mem.read_u64(target)?;
                mem.write_u64(target, (cur as i64).wrapping_add(delta) as u64)?;
                applied += 1;
            }
        }
        offset += block_size;
    }
    Ok(applied)
}

// ===========================================================================
// 4. macOS / Darwin dyld Commpage + LC_MAIN Bootstrap
// ===========================================================================

pub const COMM_PAGE64_BASE_ADDRESS: u64 = 0x0000_7FFF_FFE0_0000;
pub const COMM_PAGE_SIGNATURE: &[u8; 16] = b"commpage 64-bit\0";
pub const COMM_PAGE_VERSION: u16 = 14;

pub const LC_MAIN: u32 = 0x8000_0028;
pub const LC_LOAD_DYLINKER: u32 = 0x0E;

/// Darwin x86_64 Commpage header (`_COMM_PAGE64_BASE_ADDRESS`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Commpage64 {
    pub signature: [u8; 16],        // +0x00: "commpage 64-bit\0"
    pub version: u16,               // +0x10: 14
    pub ncpus: u8,                  // +0x12
    pub active_cpus: u8,            // +0x13
    pub physical_cpus: u8,          // +0x14
    pub logical_cpus: u8,           // +0x15
    pub _pad0: u16,                 // +0x16
    pub cpu_capabilities: u64,      // +0x18
    pub memory_size: u64,           // +0x20
    pub timebase_numer: u32,        // +0x28
    pub timebase_denom: u32,        // +0x2C
    pub mach_absolute_time_ns: u64, // +0x30
    pub user_page_shift: u32,       // +0x38
    pub _pad1: u32,                 // +0x3C
}

impl Default for Commpage64 {
    fn default() -> Self {
        Self {
            signature: *COMM_PAGE_SIGNATURE,
            version: COMM_PAGE_VERSION,
            ncpus: 1,
            active_cpus: 1,
            physical_cpus: 1,
            logical_cpus: 1,
            _pad0: 0,
            cpu_capabilities: 0x0000_000F, // MMX | SSE | SSE2 | SSE3
            memory_size: 512 * 1024 * 1024,
            timebase_numer: 1,
            timebase_denom: 1,
            mach_absolute_time_ns: 0,
            user_page_shift: 12,
            _pad1: 0,
        }
    }
}

/// Parsed Mach-O `LC_MAIN` (`entry_point_command`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MachOLcMain {
    pub entry_offset: u64,
    pub stack_size: u64,
}

/// Parse `LC_MAIN` from a raw Mach-O 64-bit binary slice.
pub fn parse_macho_lc_main(binary: &[u8]) -> Result<Option<MachOLcMain>, DynlinkError> {
    if binary.len() < 32 {
        return Err(DynlinkError::InvalidHeader);
    }
    let magic = u32::from_le_bytes([binary[0], binary[1], binary[2], binary[3]]);
    if magic != 0xFEED_FACF {
        return Err(DynlinkError::InvalidHeader);
    }
    let ncmds = u32::from_le_bytes([binary[16], binary[17], binary[18], binary[19]]);
    let sizeofcmds = u32::from_le_bytes([binary[20], binary[21], binary[22], binary[23]]) as usize;
    let mut off = 32usize;
    let end = off
        .checked_add(sizeofcmds)
        .ok_or(DynlinkError::OutOfBounds)?;
    if end > binary.len() {
        return Err(DynlinkError::OutOfBounds);
    }

    for _ in 0..ncmds {
        if off + 8 > end {
            return Err(DynlinkError::OutOfBounds);
        }
        let cmd = u32::from_le_bytes([
            binary[off],
            binary[off + 1],
            binary[off + 2],
            binary[off + 3],
        ]);
        let cmdsize = u32::from_le_bytes([
            binary[off + 4],
            binary[off + 5],
            binary[off + 6],
            binary[off + 7],
        ]) as usize;
        if cmdsize < 8 || off + cmdsize > end {
            return Err(DynlinkError::OutOfBounds);
        }
        if cmd == LC_MAIN && cmdsize >= 24 {
            let entry_offset = u64::from_le_bytes([
                binary[off + 8],
                binary[off + 9],
                binary[off + 10],
                binary[off + 11],
                binary[off + 12],
                binary[off + 13],
                binary[off + 14],
                binary[off + 15],
            ]);
            let stack_size = u64::from_le_bytes([
                binary[off + 16],
                binary[off + 17],
                binary[off + 18],
                binary[off + 19],
                binary[off + 20],
                binary[off + 21],
                binary[off + 22],
                binary[off + 23],
            ]);
            return Ok(Some(MachOLcMain {
                entry_offset,
                stack_size,
            }));
        }
        off += cmdsize;
    }
    Ok(None)
}

/// Configuration for building a Darwin/macOS `dyld` + `LC_MAIN` initial stack and commpage.
#[derive(Clone, Debug)]
pub struct DarwinBootstrapConfig<'a> {
    pub commpage_vaddr: u64,
    pub mach_header_vaddr: u64,
    pub executable_path: &'a str,
    pub argv: &'a [&'a str],
    pub envp: &'a [&'a str],
    pub stack_guard: u64,
}

/// Result of laying out a Darwin `dyld` initial stack and commpage.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DarwinBootstrapLayout {
    pub initial_rsp: u64,
    pub commpage_vaddr: u64,
    pub apple_array_vaddr: u64,
}

/// Write the Darwin `Commpage64` and build the `dyld` initial stack:
/// `[mach_header_vaddr, argc, argv..., NULL, envp..., NULL, apple..., NULL]`.
pub fn build_darwin_bootstrap(
    mem: &mut dyn UserMemory,
    stack_top: u64,
    cfg: &DarwinBootstrapConfig<'_>,
) -> Result<DarwinBootstrapLayout, DynlinkError> {
    if (stack_top & 0xF) != 0 {
        return Err(DynlinkError::InvalidAlignment);
    }

    let commpage = Box::new(Commpage64::default());
    mem.write_bytes(cfg.commpage_vaddr, unsafe { struct_as_bytes(&*commpage) })?;

    let mut sp = stack_top;

    // Push apple[] strings: executable_path=<path> and stack_guard=0x...
    let mut apple_ptrs = Vec::with_capacity(2);

    let mut exec_kv = Vec::with_capacity(16 + cfg.executable_path.len() + 1);
    exec_kv.extend_from_slice(b"executable_path=");
    exec_kv.extend_from_slice(cfg.executable_path.as_bytes());
    exec_kv.push(0);
    sp -= exec_kv.len() as u64;
    mem.write_bytes(sp, &exec_kv)?;
    apple_ptrs.push(sp);

    let guard_str = b"stack_guard=0xdeadbeefcafebabe\0";
    sp -= guard_str.len() as u64;
    mem.write_bytes(sp, guard_str)?;
    apple_ptrs.push(sp);

    // Push envp strings.
    let mut env_ptrs = Vec::with_capacity(cfg.envp.len());
    for &env in cfg.envp.iter().rev() {
        let b = env.as_bytes();
        sp -= (b.len() + 1) as u64;
        mem.write_bytes(sp, b)?;
        mem.write_bytes(sp + b.len() as u64, &[0])?;
        env_ptrs.push(sp);
    }
    env_ptrs.reverse();

    // Push argv strings.
    let mut arg_ptrs = Vec::with_capacity(cfg.argv.len());
    for &arg in cfg.argv.iter().rev() {
        let b = arg.as_bytes();
        sp -= (b.len() + 1) as u64;
        mem.write_bytes(sp, b)?;
        mem.write_bytes(sp + b.len() as u64, &[0])?;
        arg_ptrs.push(sp);
    }
    arg_ptrs.reverse();

    sp &= !0xFu64;

    // Qwords: mach_header + argc + argv + NULL + envp + NULL + apple + NULL
    let total_qwords = 1 + 1 + arg_ptrs.len() + 1 + env_ptrs.len() + 1 + apple_ptrs.len() + 1;
    sp -= (total_qwords as u64) * 8;
    if (sp & 0xF) != 0 {
        sp -= 8;
    }
    let initial_rsp = sp;

    let mut cursor = initial_rsp;
    mem.write_u64(cursor, cfg.mach_header_vaddr)?;
    cursor += 8;
    mem.write_u64(cursor, arg_ptrs.len() as u64)?;
    cursor += 8;

    for p in arg_ptrs {
        mem.write_u64(cursor, p)?;
        cursor += 8;
    }
    mem.write_u64(cursor, 0)?;
    cursor += 8;

    for p in env_ptrs {
        mem.write_u64(cursor, p)?;
        cursor += 8;
    }
    mem.write_u64(cursor, 0)?;
    cursor += 8;

    let apple_array_vaddr = cursor;
    for p in apple_ptrs {
        mem.write_u64(cursor, p)?;
        cursor += 8;
    }
    mem.write_u64(cursor, 0)?;

    Ok(DarwinBootstrapLayout {
        initial_rsp,
        commpage_vaddr: cfg.commpage_vaddr,
        apple_array_vaddr,
    })
}

// ===========================================================================
// 5. Unified Multi-OS Process Bootstrap
// ===========================================================================

/// Default user-space virtual address for the Windows TEB/PEB/LDR bootstrap block.
pub const WIN_BOOTSTRAP_VADDR: u64 = 0x0000_7FFE_0000;
/// Default user-space virtual address for the Darwin commpage.
pub const DARWIN_COMMPAGE_VADDR: u64 = 0x0000_7FFF_FFE0_0000;

/// Result of bootstrapping a Ring 3 process for its target OS personality.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BootstrapResult {
    pub personality: Personality,
    pub entry_rip: u64,
    pub user_rsp: u64,
    pub gs_base: u64,
    pub aux_block_vaddr: u64,
}

/// Parameters for bootstrapping a Ring 3 process.
#[derive(Clone, Debug)]
pub struct BootstrapRequest<'a> {
    pub format: BinaryFormat,
    pub image_base: u64,
    pub entry_point: u64,
    pub stack_top: u64,
    pub exec_path: &'a str,
    pub argv: &'a [&'a str],
    pub envp: &'a [&'a str],
}

/// Bootstrap a mapped user [`AddressSpace`] for ELF64 (POSIX), PE32+ (Windows NT),
/// or Mach-O 64-bit (Darwin/macOS).
pub fn bootstrap_process(
    space: &mut AddressSpace,
    req: &BootstrapRequest<'_>,
) -> Result<BootstrapResult, DynlinkError> {
    let rw_flags =
        PageFlags::PRESENT | PageFlags::USER | PageFlags::WRITABLE | PageFlags::NO_EXECUTE;

    match req.format {
        BinaryFormat::Elf64 => {
            let mut mem = AddressSpaceMemory::new(space);
            let cfg = ElfAuxvConfig {
                phdr_vaddr: req.image_base + 64,
                phent_size: 56,
                phnum: 1,
                entry_vaddr: req.entry_point,
                interp_base: 0,
                vdso_ehdr: 0,
                uid: 0,
                gid: 0,
                hwcap: 0x178B_FBFB,
                random_seed: [0xA5; 16],
                execfn: req.exec_path,
                argv: req.argv,
                envp: req.envp,
            };
            let layout = build_elf_initial_stack(&mut mem, req.stack_top, &cfg)?;
            Ok(BootstrapResult {
                personality: Personality::Posix,
                entry_rip: req.entry_point,
                user_rsp: layout.initial_rsp,
                gs_base: 0,
                aux_block_vaddr: layout.auxv_vaddr,
            })
        }
        BinaryFormat::Pe32Plus => {
            map_user_pages(space, WIN_BOOTSTRAP_VADDR, 2, rw_flags)?;
            let mut mem = AddressSpaceMemory::new(space);
            let modules = [
                WinModuleDesc {
                    dll_base: req.image_base,
                    entry_point: req.entry_point,
                    size_of_image: 0x10000,
                    full_name: req.exec_path,
                    base_name: req.exec_path,
                },
                WinModuleDesc {
                    dll_base: 0x0000_7FF8_0000_0000,
                    entry_point: 0,
                    size_of_image: 0x20000,
                    full_name: "C:\\Windows\\System32\\ntdll.dll",
                    base_name: "ntdll.dll",
                },
            ];
            let cfg = WindowsBootstrapConfig {
                block_base: WIN_BOOTSTRAP_VADDR,
                pid: 1,
                tid: 1,
                stack_base: req.stack_top,
                stack_limit: req.stack_top.saturating_sub(4 * PAGE_SIZE as u64),
                image_base: req.image_base,
                process_heap: WIN_BOOTSTRAP_VADDR + PAGE_SIZE as u64,
                image_path: req.exec_path,
                command_line: req.exec_path,
                modules: &modules,
            };
            let layout = build_windows_bootstrap(&mut mem, &cfg)?;
            Ok(BootstrapResult {
                personality: Personality::Nt,
                entry_rip: req.entry_point,
                user_rsp: req.stack_top - 0x28, // x64 shadow space + return addr
                gs_base: layout.teb_vaddr,
                aux_block_vaddr: layout.peb_vaddr,
            })
        }
        BinaryFormat::MachO64 => {
            map_user_pages(space, DARWIN_COMMPAGE_VADDR, 1, rw_flags)?;
            let mut mem = AddressSpaceMemory::new(space);
            let cfg = DarwinBootstrapConfig {
                commpage_vaddr: DARWIN_COMMPAGE_VADDR,
                mach_header_vaddr: req.image_base,
                executable_path: req.exec_path,
                argv: req.argv,
                envp: req.envp,
                stack_guard: 0xDEAD_BEEF_CAFE_BABE,
            };
            let layout = build_darwin_bootstrap(&mut mem, req.stack_top, &cfg)?;
            Ok(BootstrapResult {
                personality: Personality::Darwin,
                entry_rip: req.entry_point,
                user_rsp: layout.initial_rsp,
                gs_base: 0,
                aux_block_vaddr: layout.commpage_vaddr,
            })
        }
    }
}

// ===========================================================================
// 6. Self-Test
// ===========================================================================

fn test_elf_dynlink_and_auxv() -> Result<(), &'static str> {
    // 1. Test ELF64 PT_DYNAMIC parsing, SysV hash lookup, and relocations.
    let base = 0x0040_0000u64;
    let mut mem = FlatUserMemory::new(base, 0x2000);

    // String table at +0x100: "\0puts\0"
    let strtab_off = 0x100u64;
    mem.write_bytes(base + strtab_off, b"\0puts\0")?;

    // Symbol table at +0x200: [0] = UNDEF, [1] = "puts" (st_name=1, st_shndx=1, st_value=0x800)
    let symtab_off = 0x200u64;
    mem.write_u32(base + symtab_off + 24, 1)?; // st_name = 1
    mem.write_bytes(base + symtab_off + 24 + 4, &[0x12, 0])?; // STB_GLOBAL | STT_FUNC
    mem.write_u16(base + symtab_off + 24 + 6, 1)?; // st_shndx = 1
    mem.write_u64(base + symtab_off + 24 + 8, 0x800)?; // st_value

    // SysV hash table at +0x300: nbucket=1, nchain=2, bucket[0]=1, chain[0]=0, chain[1]=0
    let hash_off = 0x300u64;
    mem.write_u32(base + hash_off, 1)?;
    mem.write_u32(base + hash_off + 4, 2)?;
    mem.write_u32(base + hash_off + 8, 1)?;

    // Rela table at +0x400:
    // [0] R_X86_64_RELATIVE at +0x900, addend=0x1234
    // [1] R_X86_64_JUMP_SLOT at +0x908, sym=1 ("puts")
    let rela_off = 0x400u64;
    mem.write_u64(base + rela_off, 0x900)?;
    mem.write_u64(base + rela_off + 8, R_X86_64_RELATIVE as u64)?;
    mem.write_u64(base + rela_off + 16, 0x1234)?;

    mem.write_u64(base + rela_off + 24, 0x908)?;
    mem.write_u64(
        base + rela_off + 32,
        (1u64 << 32) | (R_X86_64_JUMP_SLOT as u64),
    )?;
    mem.write_u64(base + rela_off + 40, 0)?;

    // PT_DYNAMIC at +0x500
    let dyn_off = 0x500u64;
    let entries: [(i64, u64); 7] = [
        (DT_STRTAB, strtab_off),
        (DT_SYMTAB, symtab_off),
        (DT_HASH, hash_off),
        (DT_RELA, rela_off),
        (DT_RELASZ, 48),
        (DT_RELAENT, 24),
        (DT_NULL, 0),
    ];
    for (i, (tag, val)) in entries.iter().enumerate() {
        let addr = base + dyn_off + (i as u64) * 16;
        mem.write_u64(addr, *tag as u64)?;
        mem.write_u64(addr + 8, *val)?;
    }

    let info = DynamicInfo::parse(&mem, base + dyn_off, base)?;
    let found = info.lookup_symbol(&mem, "puts")?;
    if found != Some(base + 0x800) {
        return Err("dynlink self_test: SysV symbol lookup mismatch");
    }

    let applied = info.apply_relocations(&mut mem, |name| {
        if name == "puts" {
            Some(0x7FFF_1234_5678)
        } else {
            None
        }
    })?;
    if applied != 2 {
        return Err("dynlink self_test: unexpected relocation count");
    }
    if mem.read_u64(base + 0x900)? != base + 0x1234 {
        return Err("dynlink self_test: R_X86_64_RELATIVE mismatch");
    }
    if mem.read_u64(base + 0x908)? != 0x7FFF_1234_5678 {
        return Err("dynlink self_test: R_X86_64_JUMP_SLOT mismatch");
    }

    // 2. Test ELF AT_* auxv stack builder.
    let stack_base = 0x7FFF_0000u64;
    let stack_top = stack_base + 0x1000;
    let mut stack_mem = FlatUserMemory::new(stack_base, 0x1000);
    let cfg = ElfAuxvConfig {
        phdr_vaddr: base + 64,
        phent_size: 56,
        phnum: 2,
        entry_vaddr: base + 0x1000,
        interp_base: 0x7F00_0000,
        vdso_ehdr: 0,
        uid: 1000,
        gid: 1000,
        hwcap: 0x1234,
        random_seed: [0x5A; 16],
        execfn: "/bin/hello",
        argv: &["/bin/hello", "--flag"],
        envp: &["PATH=/bin"],
    };
    let layout = build_elf_initial_stack(&mut stack_mem, stack_top, &cfg)?;
    if (layout.initial_rsp & 0xF) != 0 {
        return Err("dynlink self_test: ELF initial RSP not 16-byte aligned");
    }
    if stack_mem.read_u64(layout.initial_rsp)? != 2 {
        return Err("dynlink self_test: ELF argc mismatch");
    }
    if stack_mem.read_u64(layout.auxv_vaddr)? != AT_PHDR {
        return Err("dynlink self_test: ELF auxv first entry not AT_PHDR");
    }

    Ok(())
}

fn test_windows_peb_teb() -> Result<(), &'static str> {
    let block_base = 0x7FFE_0000u64;
    let mut mem = FlatUserMemory::new(block_base, 0x2000);
    let modules = [
        WinModuleDesc {
            dll_base: 0x0001_4000_0000,
            entry_point: 0x0001_4000_1000,
            size_of_image: 0x10000,
            full_name: "C:\\app.exe",
            base_name: "app.exe",
        },
        WinModuleDesc {
            dll_base: 0x7FF8_0000_0000,
            entry_point: 0x7FF8_0000_1000,
            size_of_image: 0x20000,
            full_name: "C:\\Windows\\System32\\ntdll.dll",
            base_name: "ntdll.dll",
        },
    ];
    let cfg = WindowsBootstrapConfig {
        block_base,
        pid: 42,
        tid: 43,
        stack_base: 0x7FFF_1000,
        stack_limit: 0x7FFF_0000,
        image_base: 0x0001_4000_0000,
        process_heap: block_base + 0x1000,
        image_path: "C:\\app.exe",
        command_line: "C:\\app.exe --test",
        modules: &modules,
    };
    let layout = build_windows_bootstrap(&mut mem, &cfg)?;
    // Verify TEB.Self (offset 0x30) and TEB.ProcessEnvironmentBlock (offset 0x60).
    if mem.read_u64(layout.teb_vaddr + 0x30)? != layout.teb_vaddr {
        return Err("dynlink self_test: TEB.Self mismatch");
    }
    if mem.read_u64(layout.teb_vaddr + 0x60)? != layout.peb_vaddr {
        return Err("dynlink self_test: TEB.PEB mismatch");
    }
    // Verify PEB.Ldr (offset 0x18).
    if mem.read_u64(layout.peb_vaddr + 0x18)? != layout.ldr_vaddr {
        return Err("dynlink self_test: PEB.Ldr mismatch");
    }
    // Verify first LDR_DATA_TABLE_ENTRY dll_base (offset 0x30).
    if mem.read_u64(layout.first_ldr_entry_vaddr + 0x30)? != 0x0001_4000_0000 {
        return Err("dynlink self_test: LDR entry dll_base mismatch");
    }
    Ok(())
}

fn test_darwin_commpage_and_lc_main() -> Result<(), &'static str> {
    // Minimal Mach-O 64-bit header + LC_MAIN command.
    let mut macho = vec![0u8; 64];
    macho[0..4].copy_from_slice(&0xFEED_FACFu32.to_le_bytes());
    macho[16..20].copy_from_slice(&1u32.to_le_bytes()); // ncmds = 1
    macho[20..24].copy_from_slice(&24u32.to_le_bytes()); // sizeofcmds = 24
    macho[32..36].copy_from_slice(&LC_MAIN.to_le_bytes());
    macho[36..40].copy_from_slice(&24u32.to_le_bytes());
    macho[40..48].copy_from_slice(&0x1234u64.to_le_bytes()); // entryoff
    macho[48..56].copy_from_slice(&0x8000u64.to_le_bytes()); // stacksize

    let lc_main = parse_macho_lc_main(&macho)?.ok_or("dynlink self_test: LC_MAIN not found")?;
    if lc_main.entry_offset != 0x1234 || lc_main.stack_size != 0x8000 {
        return Err("dynlink self_test: LC_MAIN fields mismatch");
    }

    let commpage_vaddr = 0x7FFF_FFE0_0000u64;
    let mut comm_mem = FlatUserMemory::new(commpage_vaddr, 0x1000);
    let stack_base = 0x7FFF_0000u64;
    let stack_top = stack_base + 0x1000;
    let mut stack_mem = FlatUserMemory::new(stack_base, 0x1000);

    // Write commpage into comm_mem and stack into stack_mem via a combined test.
    let commpage = Box::new(Commpage64::default());
    comm_mem.write_bytes(commpage_vaddr, unsafe { struct_as_bytes(&*commpage) })?;
    let mut sig = [0u8; 16];
    comm_mem.read_bytes(commpage_vaddr, &mut sig)?;
    if &sig != COMM_PAGE_SIGNATURE {
        return Err("dynlink self_test: Darwin commpage signature mismatch");
    }

    let cfg = DarwinBootstrapConfig {
        commpage_vaddr: stack_base, // place commpage at stack_base for single-buffer test
        mach_header_vaddr: 0x1_0000_0000,
        executable_path: "/usr/bin/hello",
        argv: &["/usr/bin/hello"],
        envp: &["TERM=xterm-256color"],
        stack_guard: 0xDEAD_BEEF_CAFE_BABE,
    };
    let layout = build_darwin_bootstrap(&mut stack_mem, stack_top, &cfg)?;
    if (layout.initial_rsp & 0xF) != 0 {
        return Err("dynlink self_test: Darwin initial RSP not 16-byte aligned");
    }
    if stack_mem.read_u64(layout.initial_rsp)? != 0x1_0000_0000 {
        return Err("dynlink self_test: Darwin mach_header pointer mismatch");
    }
    if stack_mem.read_u64(layout.initial_rsp + 8)? != 1 {
        return Err("dynlink self_test: Darwin argc mismatch");
    }

    Ok(())
}

/// Run all Ring 3 dynamic linker and multi-OS process bootstrap self-tests.
pub fn self_test() -> Result<(), &'static str> {
    test_elf_dynlink_and_auxv()?;
    test_windows_peb_teb()?;
    test_darwin_commpage_and_lc_main()?;
    Ok(())
}
