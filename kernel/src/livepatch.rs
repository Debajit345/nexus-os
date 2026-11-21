#![allow(dead_code)]

//! `no_std` Kernel Live-Patching (`kpatch`), Module ELF64 Relocator & Symbol Resolver.
//!
//! Provides:
//! - [`KallsymsTable`]: Kernel symbol table with forward name lookup and binary-search
//!   reverse address-to-symbol+offset resolution.
//! - [`Elf64Relocator`]: `no_std` ELF64 (`ET_REL`, `EM_X86_64`) section loader, symbol
//!   resolver, and `R_X86_64_*` relocation engine (`64`, `PC32`, `PLT32`, `32`, `32S`, `PC64`).
//! - [`TrampolinePatcher`]: Atomic x86_64 function prologue patcher supporting 5-byte
//!   near `JMP rel32` (`0xE9`) and 14-byte far `JMP [RIP+0]` (`0xFF 0x25`) trampolines,
//!   `INT3`-staged text poking, and full prologue rollback.
//! - [`PatchManager`]: Live-patch lifecycle manager (register, apply, revert, unload)
//!   with RCU grace-period synchronization.
//! - [`self_test`]: Boot-time verification of kallsyms, ELF64 relocation, and patching.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;
use core::sync::atomic::{fence, Ordering};

use crate::rcu;
use crate::sync::TicketLock;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by the kernel live-patching subsystem.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LivepatchError {
    InvalidElfHeader,
    UnsupportedArch,
    UnsupportedElfType,
    OutOfBounds,
    InvalidSection,
    InvalidSymbolTable,
    InvalidStringTable,
    SymbolNotFound,
    DuplicateSymbol,
    UnsupportedRelocation,
    RelocationOverflow,
    PatchAlreadyApplied,
    PatchNotApplied,
    PatchNotFound,
    TargetTooSmall,
    InvalidAlignment,
}

impl LivepatchError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidElfHeader => "livepatch: invalid ELF64 header",
            Self::UnsupportedArch => "livepatch: unsupported ELF architecture (expected x86_64)",
            Self::UnsupportedElfType => "livepatch: expected ET_REL relocatable object",
            Self::OutOfBounds => "livepatch: offset or slice out of bounds",
            Self::InvalidSection => "livepatch: invalid ELF section header",
            Self::InvalidSymbolTable => "livepatch: invalid or missing ELF symbol table",
            Self::InvalidStringTable => "livepatch: invalid or missing ELF string table",
            Self::SymbolNotFound => "livepatch: unresolved kernel or module symbol",
            Self::DuplicateSymbol => "livepatch: duplicate symbol or patch name",
            Self::UnsupportedRelocation => "livepatch: unsupported x86_64 relocation type",
            Self::RelocationOverflow => "livepatch: relocation value overflows target field",
            Self::PatchAlreadyApplied => "livepatch: patch is already applied",
            Self::PatchNotApplied => "livepatch: patch is not currently applied",
            Self::PatchNotFound => "livepatch: patch not found in registry",
            Self::TargetTooSmall => "livepatch: target function prologue too small for trampoline",
            Self::InvalidAlignment => "livepatch: invalid section or buffer alignment",
        }
    }
}

impl From<LivepatchError> for &'static str {
    fn from(err: LivepatchError) -> Self {
        err.as_str()
    }
}

// ---------------------------------------------------------------------------
// Little-Endian Binary Readers / Writers (zero-copy, small stack)
// ---------------------------------------------------------------------------

#[inline]
fn read_u16_le(buf: &[u8], off: usize) -> Result<u16, LivepatchError> {
    let end = off.checked_add(2).ok_or(LivepatchError::OutOfBounds)?;
    let slice = buf.get(off..end).ok_or(LivepatchError::OutOfBounds)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

#[inline]
fn read_u32_le(buf: &[u8], off: usize) -> Result<u32, LivepatchError> {
    let end = off.checked_add(4).ok_or(LivepatchError::OutOfBounds)?;
    let slice = buf.get(off..end).ok_or(LivepatchError::OutOfBounds)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[inline]
fn read_u64_le(buf: &[u8], off: usize) -> Result<u64, LivepatchError> {
    let end = off.checked_add(8).ok_or(LivepatchError::OutOfBounds)?;
    let slice = buf.get(off..end).ok_or(LivepatchError::OutOfBounds)?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

#[inline]
fn read_i64_le(buf: &[u8], off: usize) -> Result<i64, LivepatchError> {
    Ok(read_u64_le(buf, off)? as i64)
}

#[inline]
fn write_u32_le(buf: &mut [u8], off: usize, val: u32) -> Result<(), LivepatchError> {
    let end = off.checked_add(4).ok_or(LivepatchError::OutOfBounds)?;
    let slice = buf.get_mut(off..end).ok_or(LivepatchError::OutOfBounds)?;
    slice.copy_from_slice(&val.to_le_bytes());
    Ok(())
}

#[inline]
fn write_u64_le(buf: &mut [u8], off: usize, val: u64) -> Result<(), LivepatchError> {
    let end = off.checked_add(8).ok_or(LivepatchError::OutOfBounds)?;
    let slice = buf.get_mut(off..end).ok_or(LivepatchError::OutOfBounds)?;
    slice.copy_from_slice(&val.to_le_bytes());
    Ok(())
}

fn read_cstr(strtab: &[u8], off: usize) -> Result<&str, LivepatchError> {
    let tail = strtab
        .get(off..)
        .ok_or(LivepatchError::InvalidStringTable)?;
    let nul = tail
        .iter()
        .position(|&b| b == 0)
        .ok_or(LivepatchError::InvalidStringTable)?;
    core::str::from_utf8(&tail[..nul]).map_err(|_| LivepatchError::InvalidStringTable)
}

// ---------------------------------------------------------------------------
// Kallsyms Table (Kernel Symbol Resolver)
// ---------------------------------------------------------------------------

/// Kind of kernel or module symbol stored in [`KallsymsTable`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SymbolKind {
    Text,
    Data,
    Rodata,
    Bss,
    Weak,
}

/// A single entry in the kernel symbol table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KallsymEntry {
    pub name: String,
    pub addr: u64,
    pub size: u64,
    pub kind: SymbolKind,
    pub module_id: u32,
}

/// Kernel symbol table supporting name lookup and binary-search address resolution.
pub struct KallsymsTable {
    entries: Vec<KallsymEntry>,
    sorted_by_addr: bool,
}

impl KallsymsTable {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            sorted_by_addr: true,
        }
    }

    /// Register a symbol in the table.
    pub fn register(
        &mut self,
        name: &str,
        addr: u64,
        size: u64,
        kind: SymbolKind,
        module_id: u32,
    ) -> Result<(), LivepatchError> {
        if let Some(existing) = self.entries.iter().find(|e| e.name == name) {
            if existing.kind != SymbolKind::Weak && kind != SymbolKind::Weak {
                return Err(LivepatchError::DuplicateSymbol);
            }
            if existing.kind == SymbolKind::Weak && kind != SymbolKind::Weak {
                // Strong symbol overrides weak symbol.
                if let Some(slot) = self.entries.iter_mut().find(|e| e.name == name) {
                    slot.addr = addr;
                    slot.size = size;
                    slot.kind = kind;
                    slot.module_id = module_id;
                    self.sorted_by_addr = false;
                }
                return Ok(());
            }
            return Ok(());
        }

        self.entries.push(KallsymEntry {
            name: String::from(name),
            addr,
            size,
            kind,
            module_id,
        });
        self.sorted_by_addr = false;
        Ok(())
    }

    /// Remove all symbols belonging to a given module ID (never removes core kernel `module_id == 0`).
    pub fn unregister_module(&mut self, module_id: u32) {
        if module_id == 0 {
            return;
        }
        self.entries.retain(|e| e.module_id != module_id);
    }

    /// Look up a symbol's virtual address by exact name.
    pub fn lookup(&self, name: &str) -> Option<u64> {
        self.entries.iter().find(|e| e.name == name).map(|e| e.addr)
    }

    /// Look up full symbol metadata by exact name.
    pub fn lookup_entry(&self, name: &str) -> Option<&KallsymEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Ensure entries are sorted by address for binary search.
    pub fn ensure_sorted(&mut self) {
        if !self.sorted_by_addr {
            self.entries.sort_by_key(|e| e.addr);
            self.sorted_by_addr = true;
        }
    }

    /// Resolve an instruction/data address to `(symbol_name, offset_within_symbol)`.
    pub fn lookup_addr(&mut self, addr: u64) -> Option<(&str, u64)> {
        self.ensure_sorted();
        if self.entries.is_empty() {
            return None;
        }
        let idx = self.entries.partition_point(|e| e.addr <= addr);
        if idx == 0 {
            return None;
        }
        let entry = &self.entries[idx - 1];
        let offset = addr - entry.addr;
        if entry.size != 0 && offset >= entry.size {
            return None;
        }
        Some((entry.name.as_str(), offset))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for KallsymsTable {
    fn default() -> Self {
        Self::new()
    }
}

static GLOBAL_KALLSYMS: TicketLock<KallsymsTable> = TicketLock::new(KallsymsTable::new());

/// Register a core kernel symbol (`module_id = 0`) in the global kallsyms table.
pub fn register_kernel_symbol(
    name: &str,
    addr: u64,
    size: u64,
    kind: SymbolKind,
) -> Result<(), LivepatchError> {
    GLOBAL_KALLSYMS.lock().register(name, addr, size, kind, 0)
}

/// Look up a symbol address in the global kallsyms table.
pub fn kallsyms_lookup(name: &str) -> Option<u64> {
    GLOBAL_KALLSYMS.lock().lookup(name)
}

// ---------------------------------------------------------------------------
// ELF64 Constants & Structures
// ---------------------------------------------------------------------------

const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const ET_REL: u16 = 1;
const EM_X86_64: u16 = 62;

const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;

const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

const SHN_UNDEF: u16 = 0;
const SHN_ABS: u16 = 0xFFF1;
const SHN_COMMON: u16 = 0xFFF2;

const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STB_WEAK: u8 = 2;

const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const STT_SECTION: u8 = 3;

// x86_64 ELF relocation types
const R_X86_64_NONE: u32 = 0;
const R_X86_64_64: u32 = 1;
const R_X86_64_PC32: u32 = 2;
const R_X86_64_PLT32: u32 = 4;
const R_X86_64_32: u32 = 10;
const R_X86_64_32S: u32 = 11;
const R_X86_64_PC64: u32 = 24;

const ELF64_EHDR_SIZE: usize = 64;
const ELF64_SHDR_SIZE: usize = 64;
const ELF64_SYM_SIZE: usize = 24;
const ELF64_RELA_SIZE: usize = 24;

#[derive(Clone, Debug)]
struct Elf64Shdr {
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
}

#[derive(Clone, Debug)]
struct Elf64Sym {
    st_name: u32,
    st_info: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
}

impl Elf64Sym {
    #[inline]
    fn bind(&self) -> u8 {
        self.st_info >> 4
    }

    #[inline]
    fn sym_type(&self) -> u8 {
        self.st_info & 0x0F
    }
}

#[derive(Clone, Debug)]
struct Elf64Rela {
    r_offset: u64,
    r_info: u64,
    r_addend: i64,
}

impl Elf64Rela {
    #[inline]
    fn sym_index(&self) -> usize {
        (self.r_info >> 32) as usize
    }

    #[inline]
    fn reloc_type(&self) -> u32 {
        (self.r_info & 0xFFFF_FFFF) as u32
    }
}

// ---------------------------------------------------------------------------
// Loaded Module Image & ELF64 Relocator
// ---------------------------------------------------------------------------

/// Metadata for a single allocated section inside a [`ModuleImage`].
#[derive(Clone, Debug)]
pub struct LoadedSection {
    pub name: String,
    pub shndx: usize,
    pub offset: usize,
    pub size: usize,
    pub flags: u64,
    pub vaddr: u64,
}

/// Exported symbol resolved from a loaded ELF64 module.
#[derive(Clone, Debug)]
pub struct ModuleSymbol {
    pub name: String,
    pub vaddr: u64,
    pub size: u64,
    pub kind: SymbolKind,
}

/// A relocated ELF64 kernel module image residing in heap memory.
pub struct ModuleImage {
    pub module_id: u32,
    pub name: String,
    /// Contiguous backing buffer for all `SHF_ALLOC` sections.
    pub memory: Box<[u8]>,
    /// Base virtual address of `memory[0]` (equals `memory.as_ptr() as u64` unless overridden).
    pub base_vaddr: u64,
    pub sections: Vec<LoadedSection>,
    pub exports: Vec<ModuleSymbol>,
    pub relocations_applied: usize,
}

impl ModuleImage {
    /// Look up an exported symbol address within this module.
    pub fn find_symbol(&self, name: &str) -> Option<u64> {
        self.exports
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.vaddr)
    }

    /// Read bytes from a loaded section by section name.
    pub fn section_bytes(&self, name: &str) -> Option<&[u8]> {
        let sec = self.sections.iter().find(|s| s.name == name)?;
        self.memory.get(sec.offset..sec.offset + sec.size)
    }
}

/// `no_std` ELF64 (`ET_REL`, `EM_X86_64`) relocator and symbol resolver.
pub struct Elf64Relocator;

impl Elf64Relocator {
    /// Parse, lay out, resolve symbols, and relocate an ELF64 `ET_REL` object.
    ///
    /// If `base_vaddr_override` is `Some(addr)`, relocations are computed as if the
    /// module image is mapped at `addr` (useful for deterministic testing or remote
    /// address spaces); otherwise `memory.as_ptr() as u64` is used.
    pub fn load_and_relocate(
        module_id: u32,
        module_name: &str,
        elf_bytes: &[u8],
        kallsyms: &KallsymsTable,
        base_vaddr_override: Option<u64>,
    ) -> Result<ModuleImage, LivepatchError> {
        if elf_bytes.len() < ELF64_EHDR_SIZE {
            return Err(LivepatchError::InvalidElfHeader);
        }
        if elf_bytes[0..4] != ELF_MAGIC {
            return Err(LivepatchError::InvalidElfHeader);
        }
        if elf_bytes[4] != ELFCLASS64 || elf_bytes[5] != ELFDATA2LSB {
            return Err(LivepatchError::InvalidElfHeader);
        }

        let e_type = read_u16_le(elf_bytes, 16)?;
        if e_type != ET_REL {
            return Err(LivepatchError::UnsupportedElfType);
        }
        let e_machine = read_u16_le(elf_bytes, 18)?;
        if e_machine != EM_X86_64 {
            return Err(LivepatchError::UnsupportedArch);
        }

        let e_shoff = read_u64_le(elf_bytes, 40)? as usize;
        let e_shentsize = read_u16_le(elf_bytes, 58)? as usize;
        let e_shnum = read_u16_le(elf_bytes, 60)? as usize;
        let e_shstrndx = read_u16_le(elf_bytes, 62)? as usize;

        if e_shnum == 0 || e_shentsize < ELF64_SHDR_SIZE {
            return Err(LivepatchError::InvalidSection);
        }

        let shdrs = Self::parse_shdrs(elf_bytes, e_shoff, e_shentsize, e_shnum)?;
        if e_shstrndx >= shdrs.len() {
            return Err(LivepatchError::InvalidStringTable);
        }
        let shstrtab = Self::section_slice(elf_bytes, &shdrs[e_shstrndx])?;

        // Locate SHT_SYMTAB and its linked SHT_STRTAB.
        let symtab_idx = shdrs
            .iter()
            .position(|s| s.sh_type == SHT_SYMTAB)
            .ok_or(LivepatchError::InvalidSymbolTable)?;
        let symtab_shdr = &shdrs[symtab_idx];
        let strtab_idx = symtab_shdr.sh_link as usize;
        if strtab_idx >= shdrs.len() || shdrs[strtab_idx].sh_type != SHT_STRTAB {
            return Err(LivepatchError::InvalidStringTable);
        }
        let strtab = Self::section_slice(elf_bytes, &shdrs[strtab_idx])?;
        let syms = Self::parse_syms(elf_bytes, symtab_shdr)?;

        // Compute layout for all SHF_ALLOC sections.
        let mut total_size: usize = 0;
        let mut sec_offsets: Vec<Option<usize>> = vec![None; shdrs.len()];
        for (idx, shdr) in shdrs.iter().enumerate() {
            if (shdr.sh_flags & SHF_ALLOC) == 0 || shdr.sh_type == SHT_NULL {
                continue;
            }
            let align = (shdr.sh_addralign as usize).max(1);
            if !align.is_power_of_two() {
                return Err(LivepatchError::InvalidAlignment);
            }
            let aligned = (total_size + align - 1) & !(align - 1);
            let sz = shdr.sh_size as usize;
            total_size = aligned.checked_add(sz).ok_or(LivepatchError::OutOfBounds)?;
            sec_offsets[idx] = Some(aligned);
        }

        // Allocate contiguous zeroed buffer on the heap.
        let alloc_len = total_size.max(1);
        let mut memory = vec![0u8; alloc_len].into_boxed_slice();
        let base_vaddr = base_vaddr_override.unwrap_or(memory.as_ptr() as u64);

        // Copy PROGBITS sections and record LoadedSection metadata.
        let mut loaded_sections = Vec::new();
        for (idx, shdr) in shdrs.iter().enumerate() {
            let Some(off) = sec_offsets[idx] else {
                continue;
            };
            let sz = shdr.sh_size as usize;
            if shdr.sh_type != SHT_NOBITS && sz > 0 {
                let src = Self::section_slice(elf_bytes, shdr)?;
                let dst = memory
                    .get_mut(off..off + sz)
                    .ok_or(LivepatchError::OutOfBounds)?;
                dst.copy_from_slice(src);
            }
            let name = read_cstr(shstrtab, shdr.sh_name as usize)?;
            loaded_sections.push(LoadedSection {
                name: String::from(name),
                shndx: idx,
                offset: off,
                size: sz,
                flags: shdr.sh_flags,
                vaddr: base_vaddr.wrapping_add(off as u64),
            });
        }

        // Resolve all symbols to virtual addresses.
        let mut resolved_syms: Vec<u64> = vec![0; syms.len()];
        let mut exports = Vec::new();

        for (idx, sym) in syms.iter().enumerate() {
            if idx == 0 {
                continue;
            }
            let name = read_cstr(strtab, sym.st_name as usize)?;
            let addr = match sym.st_shndx {
                SHN_UNDEF => {
                    if name.is_empty() {
                        0
                    } else if let Some(kaddr) = kallsyms.lookup(name) {
                        kaddr
                    } else if sym.bind() == STB_WEAK {
                        0
                    } else {
                        return Err(LivepatchError::SymbolNotFound);
                    }
                }
                SHN_ABS => sym.st_value,
                SHN_COMMON => return Err(LivepatchError::UnsupportedRelocation),
                shndx => {
                    let sec_idx = shndx as usize;
                    let sec_off = sec_offsets
                        .get(sec_idx)
                        .and_then(|o| *o)
                        .ok_or(LivepatchError::InvalidSection)?;
                    base_vaddr
                        .wrapping_add(sec_off as u64)
                        .wrapping_add(sym.st_value)
                }
            };
            resolved_syms[idx] = addr;

            if !name.is_empty()
                && sym.st_shndx != SHN_UNDEF
                && (sym.bind() == STB_GLOBAL || sym.bind() == STB_WEAK)
                && sym.sym_type() != STT_SECTION
            {
                let kind = if sym.bind() == STB_WEAK {
                    SymbolKind::Weak
                } else if let Some(sec) = shdrs.get(sym.st_shndx as usize) {
                    if (sec.sh_flags & SHF_EXECINSTR) != 0 {
                        SymbolKind::Text
                    } else if sec.sh_type == SHT_NOBITS {
                        SymbolKind::Bss
                    } else if (sec.sh_flags & SHF_WRITE) != 0 {
                        SymbolKind::Data
                    } else {
                        SymbolKind::Rodata
                    }
                } else {
                    SymbolKind::Data
                };
                exports.push(ModuleSymbol {
                    name: String::from(name),
                    vaddr: addr,
                    size: sym.st_size,
                    kind,
                });
            }
        }

        // Apply SHT_RELA relocations.
        let mut relocations_applied = 0usize;
        for shdr in &shdrs {
            if shdr.sh_type != SHT_RELA {
                continue;
            }
            let target_sec_idx = shdr.sh_info as usize;
            let Some(target_sec_off) = sec_offsets.get(target_sec_idx).and_then(|o| *o) else {
                continue;
            };
            let target_sec_size = shdrs[target_sec_idx].sh_size as usize;
            let relas = Self::parse_relas(elf_bytes, shdr)?;

            for rela in &relas {
                let sym_idx = rela.sym_index();
                let s_val = *resolved_syms
                    .get(sym_idx)
                    .ok_or(LivepatchError::InvalidSymbolTable)?;
                let r_off = rela.r_offset as usize;
                if r_off >= target_sec_size {
                    return Err(LivepatchError::OutOfBounds);
                }
                let buf_off = target_sec_off + r_off;
                let p_val = base_vaddr.wrapping_add(buf_off as u64);
                let a_val = rela.r_addend;

                Self::apply_rela(&mut memory, buf_off, rela.reloc_type(), s_val, a_val, p_val)?;
                relocations_applied += 1;
            }
        }

        Ok(ModuleImage {
            module_id,
            name: String::from(module_name),
            memory,
            base_vaddr,
            sections: loaded_sections,
            exports,
            relocations_applied,
        })
    }

    fn parse_shdrs(
        elf: &[u8],
        shoff: usize,
        shentsize: usize,
        shnum: usize,
    ) -> Result<Vec<Elf64Shdr>, LivepatchError> {
        let mut out = Vec::with_capacity(shnum);
        for idx in 0..shnum {
            let off = shoff
                .checked_add(idx * shentsize)
                .ok_or(LivepatchError::OutOfBounds)?;
            if off + ELF64_SHDR_SIZE > elf.len() {
                return Err(LivepatchError::OutOfBounds);
            }
            out.push(Elf64Shdr {
                sh_name: read_u32_le(elf, off)?,
                sh_type: read_u32_le(elf, off + 4)?,
                sh_flags: read_u64_le(elf, off + 8)?,
                sh_offset: read_u64_le(elf, off + 24)?,
                sh_size: read_u64_le(elf, off + 32)?,
                sh_link: read_u32_le(elf, off + 40)?,
                sh_info: read_u32_le(elf, off + 44)?,
                sh_addralign: read_u64_le(elf, off + 48)?,
                sh_entsize: read_u64_le(elf, off + 56)?,
            });
        }
        Ok(out)
    }

    fn section_slice<'a>(elf: &'a [u8], shdr: &Elf64Shdr) -> Result<&'a [u8], LivepatchError> {
        let start = shdr.sh_offset as usize;
        let end = start
            .checked_add(shdr.sh_size as usize)
            .ok_or(LivepatchError::OutOfBounds)?;
        elf.get(start..end).ok_or(LivepatchError::OutOfBounds)
    }

    fn parse_syms(elf: &[u8], shdr: &Elf64Shdr) -> Result<Vec<Elf64Sym>, LivepatchError> {
        let entsize = (shdr.sh_entsize as usize).max(ELF64_SYM_SIZE);
        let size = shdr.sh_size as usize;
        if !size.is_multiple_of(entsize) {
            return Err(LivepatchError::InvalidSymbolTable);
        }
        let count = size / entsize;
        let base = shdr.sh_offset as usize;
        let mut out = Vec::with_capacity(count);
        for idx in 0..count {
            let off = base + idx * entsize;
            if off + ELF64_SYM_SIZE > elf.len() {
                return Err(LivepatchError::OutOfBounds);
            }
            out.push(Elf64Sym {
                st_name: read_u32_le(elf, off)?,
                st_info: elf[off + 4],
                st_shndx: read_u16_le(elf, off + 6)?,
                st_value: read_u64_le(elf, off + 8)?,
                st_size: read_u64_le(elf, off + 16)?,
            });
        }
        Ok(out)
    }

    fn parse_relas(elf: &[u8], shdr: &Elf64Shdr) -> Result<Vec<Elf64Rela>, LivepatchError> {
        let entsize = (shdr.sh_entsize as usize).max(ELF64_RELA_SIZE);
        let size = shdr.sh_size as usize;
        if !size.is_multiple_of(entsize) {
            return Err(LivepatchError::InvalidSection);
        }
        let count = size / entsize;
        let base = shdr.sh_offset as usize;
        let mut out = Vec::with_capacity(count);
        for idx in 0..count {
            let off = base + idx * entsize;
            if off + ELF64_RELA_SIZE > elf.len() {
                return Err(LivepatchError::OutOfBounds);
            }
            out.push(Elf64Rela {
                r_offset: read_u64_le(elf, off)?,
                r_info: read_u64_le(elf, off + 8)?,
                r_addend: read_i64_le(elf, off + 16)?,
            });
        }
        Ok(out)
    }

    fn apply_rela(
        memory: &mut [u8],
        off: usize,
        rtype: u32,
        s: u64,
        a: i64,
        p: u64,
    ) -> Result<(), LivepatchError> {
        match rtype {
            R_X86_64_NONE => Ok(()),
            R_X86_64_64 => {
                let val = (s as i128 + a as i128) as u64;
                write_u64_le(memory, off, val)
            }
            R_X86_64_PC32 | R_X86_64_PLT32 => {
                let val = (s as i128) + (a as i128) - (p as i128);
                let rel32 = i32::try_from(val).map_err(|_| LivepatchError::RelocationOverflow)?;
                write_u32_le(memory, off, rel32 as u32)
            }
            R_X86_64_32 => {
                let val = (s as i128) + (a as i128);
                let u32_val = u32::try_from(val).map_err(|_| LivepatchError::RelocationOverflow)?;
                write_u32_le(memory, off, u32_val)
            }
            R_X86_64_32S => {
                let val = (s as i128) + (a as i128);
                let i32_val = i32::try_from(val).map_err(|_| LivepatchError::RelocationOverflow)?;
                write_u32_le(memory, off, i32_val as u32)
            }
            R_X86_64_PC64 => {
                let val = ((s as i128) + (a as i128) - (p as i128)) as u64;
                write_u64_le(memory, off, val)
            }
            _ => Err(LivepatchError::UnsupportedRelocation),
        }
    }
}

// ---------------------------------------------------------------------------
// Atomic Trampoline Patcher
// ---------------------------------------------------------------------------

/// Kind of x86_64 trampoline emitted at the target function's prologue.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TrampolineKind {
    /// 5-byte `JMP rel32` (`0xE9 <i32>`), used when `|target - (src + 5)| <= 2 GiB`.
    NearJmpRel32,
    /// 14-byte `JMP [RIP+0]` (`0xFF 0x25 0x00000000 <u64>`), used for arbitrary 64-bit targets.
    FarJmpIndirect,
}

impl TrampolineKind {
    pub const fn len(self) -> usize {
        match self {
            Self::NearJmpRel32 => 5,
            Self::FarJmpIndirect => 14,
        }
    }
}

/// Abstract memory backend for atomic text patching (allows safe testing on buffers
/// as well as real kernel `.text` patching).
pub trait PatchMemory {
    fn read_bytes(&self, addr: u64, out: &mut [u8]) -> Result<(), LivepatchError>;
    fn write_bytes_atomic(&mut self, addr: u64, data: &[u8]) -> Result<(), LivepatchError>;
}

/// In-memory buffer implementation of [`PatchMemory`] for unit/self-tests.
pub struct BufferPatchMemory {
    pub base_addr: u64,
    pub data: Vec<u8>,
}

impl BufferPatchMemory {
    pub fn new(base_addr: u64, data: Vec<u8>) -> Self {
        Self { base_addr, data }
    }

    fn slice_range(
        &self,
        addr: u64,
        len: usize,
    ) -> Result<core::ops::Range<usize>, LivepatchError> {
        if addr < self.base_addr {
            return Err(LivepatchError::OutOfBounds);
        }
        let off = (addr - self.base_addr) as usize;
        let end = off.checked_add(len).ok_or(LivepatchError::OutOfBounds)?;
        if end > self.data.len() {
            return Err(LivepatchError::OutOfBounds);
        }
        Ok(off..end)
    }
}

impl PatchMemory for BufferPatchMemory {
    fn read_bytes(&self, addr: u64, out: &mut [u8]) -> Result<(), LivepatchError> {
        let range = self.slice_range(addr, out.len())?;
        out.copy_from_slice(&self.data[range]);
        Ok(())
    }

    fn write_bytes_atomic(&mut self, addr: u64, data: &[u8]) -> Result<(), LivepatchError> {
        let range = self.slice_range(addr, data.len())?;
        if data.is_empty() {
            return Ok(());
        }
        // Simulate INT3-staged text_poke_bp:
        // 1. Stage 0xCC at byte 0
        self.data[range.start] = 0xCC;
        fence(Ordering::SeqCst);
        // 2. Write tail bytes 1..N
        if data.len() > 1 {
            self.data[range.start + 1..range.end].copy_from_slice(&data[1..]);
            fence(Ordering::SeqCst);
        }
        // 3. Atomically commit byte 0
        self.data[range.start] = data[0];
        fence(Ordering::SeqCst);
        Ok(())
    }
}

/// Live kernel `.text` implementation of [`PatchMemory`] using `CR0.WP` bypass
/// and `INT3`-staged `text_poke_bp` semantics.
pub struct KernelTextMemory;

impl PatchMemory for KernelTextMemory {
    fn read_bytes(&self, addr: u64, out: &mut [u8]) -> Result<(), LivepatchError> {
        if addr == 0 {
            return Err(LivepatchError::OutOfBounds);
        }
        let src = addr as *const u8;
        for (idx, slot) in out.iter_mut().enumerate() {
            // SAFETY: Caller guarantees `addr..addr+out.len()` is mapped kernel text.
            *slot = unsafe { core::ptr::read_volatile(src.add(idx)) };
        }
        Ok(())
    }

    fn write_bytes_atomic(&mut self, addr: u64, data: &[u8]) -> Result<(), LivepatchError> {
        if addr == 0 {
            return Err(LivepatchError::OutOfBounds);
        }
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: Disables interrupts and clears CR0.WP temporarily to poke kernel text,
        // staging 0xCC at byte 0 before writing the remaining bytes and committing byte 0.
        unsafe {
            let rflags: u64;
            let cr0: u64;
            core::arch::asm!(
                "pushfq",
                "pop {rflags}",
                "cli",
                "mov {cr0}, cr0",
                rflags = out(reg) rflags,
                cr0 = out(reg) cr0,
                options(nomem, preserves_flags)
            );
            let cr0_no_wp = cr0 & !(1u64 << 16);
            core::arch::asm!("mov cr0, {0}", in(reg) cr0_no_wp, options(nostack));

            let dst = addr as *mut u8;
            core::ptr::write_volatile(dst, 0xCC);
            fence(Ordering::SeqCst);

            for (idx, &byte) in data.iter().enumerate().skip(1) {
                core::ptr::write_volatile(dst.add(idx), byte);
            }
            fence(Ordering::SeqCst);

            core::ptr::write_volatile(dst, data[0]);
            fence(Ordering::SeqCst);

            core::arch::asm!("mov cr0, {0}", in(reg) cr0, options(nostack));
            if (rflags & (1 << 9)) != 0 {
                core::arch::asm!("sti", options(nomem, nostack));
            }
        }
        Ok(())
    }
}

/// Encodes and applies/reverts x86_64 function prologue trampolines.
pub struct TrampolinePatcher;

impl TrampolinePatcher {
    /// Encode a trampoline from `src_addr` to `dst_addr`.
    /// Returns `(kind, bytes, len)`.
    pub fn encode_trampoline(src_addr: u64, dst_addr: u64) -> (TrampolineKind, [u8; 16], usize) {
        let mut buf = [0u8; 16];
        let disp = (dst_addr as i128) - ((src_addr as i128) + 5);
        if let Ok(rel32) = i32::try_from(disp) {
            buf[0] = 0xE9;
            buf[1..5].copy_from_slice(&rel32.to_le_bytes());
            (TrampolineKind::NearJmpRel32, buf, 5)
        } else {
            // FF 25 00 00 00 00 <u64 dst_addr>
            buf[0] = 0xFF;
            buf[1] = 0x25;
            buf[2..6].copy_from_slice(&0u32.to_le_bytes());
            buf[6..14].copy_from_slice(&dst_addr.to_le_bytes());
            (TrampolineKind::FarJmpIndirect, buf, 14)
        }
    }

    /// Apply a trampoline at `old_addr` redirecting execution to `new_addr`.
    /// Saves the original prologue bytes into `saved_orig` and returns `(kind, len)`.
    pub fn apply<M: PatchMemory>(
        mem: &mut M,
        old_addr: u64,
        old_size: usize,
        new_addr: u64,
        saved_orig: &mut [u8; 16],
    ) -> Result<(TrampolineKind, usize), LivepatchError> {
        let (kind, patch_bytes, patch_len) = Self::encode_trampoline(old_addr, new_addr);
        if old_size != 0 && old_size < patch_len {
            return Err(LivepatchError::TargetTooSmall);
        }
        mem.read_bytes(old_addr, &mut saved_orig[..patch_len])?;
        mem.write_bytes_atomic(old_addr, &patch_bytes[..patch_len])?;
        Ok((kind, patch_len))
    }

    /// Revert a previously applied trampoline by restoring `saved_orig[..patch_len]`.
    pub fn revert<M: PatchMemory>(
        mem: &mut M,
        old_addr: u64,
        saved_orig: &[u8; 16],
        patch_len: usize,
    ) -> Result<(), LivepatchError> {
        if patch_len == 0 || patch_len > 16 {
            return Err(LivepatchError::OutOfBounds);
        }
        mem.write_bytes_atomic(old_addr, &saved_orig[..patch_len])
    }
}

// ---------------------------------------------------------------------------
// Live-Patch Lifecycle & Registry
// ---------------------------------------------------------------------------

/// Lifecycle state of a registered live-patch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PatchState {
    Registered,
    Applied,
    Reverted,
}

/// A single function-level live-patch entry.
#[derive(Clone, Debug)]
pub struct FunctionPatch {
    pub target_symbol: String,
    pub replacement_symbol: String,
    pub old_addr: u64,
    pub old_size: usize,
    pub new_addr: u64,
    pub kind: Option<TrampolineKind>,
    pub orig_bytes: [u8; 16],
    pub patch_len: usize,
    pub state: PatchState,
}

/// A loaded live-patch module containing its relocated image and function patches.
pub struct KpatchModule {
    pub name: String,
    pub version: u32,
    pub image: ModuleImage,
    pub patches: Vec<FunctionPatch>,
    pub applied: bool,
}

/// Central live-patch manager coordinating module loading, patching, and RCU quiescence.
pub struct PatchManager {
    next_module_id: u32,
    modules: Vec<KpatchModule>,
}

impl PatchManager {
    pub const fn new() -> Self {
        Self {
            next_module_id: 1,
            modules: Vec::new(),
        }
    }

    /// Load an ELF64 live-patch module and register its function redirections.
    pub fn register_module(
        &mut self,
        name: &str,
        version: u32,
        elf_bytes: &[u8],
        kallsyms: &mut KallsymsTable,
        redirects: &[(&str, &str)],
        base_vaddr_override: Option<u64>,
    ) -> Result<u32, LivepatchError> {
        if self.modules.iter().any(|m| m.name == name) {
            return Err(LivepatchError::DuplicateSymbol);
        }
        let module_id = self.next_module_id;
        self.next_module_id = self.next_module_id.wrapping_add(1).max(1);

        let image = Elf64Relocator::load_and_relocate(
            module_id,
            name,
            elf_bytes,
            kallsyms,
            base_vaddr_override,
        )?;

        let mut patches = Vec::with_capacity(redirects.len());
        for &(target_sym, repl_sym) in redirects {
            let target_entry = kallsyms
                .lookup_entry(target_sym)
                .ok_or(LivepatchError::SymbolNotFound)?;
            let new_addr = image
                .find_symbol(repl_sym)
                .ok_or(LivepatchError::SymbolNotFound)?;
            patches.push(FunctionPatch {
                target_symbol: String::from(target_sym),
                replacement_symbol: String::from(repl_sym),
                old_addr: target_entry.addr,
                old_size: target_entry.size as usize,
                new_addr,
                kind: None,
                orig_bytes: [0u8; 16],
                patch_len: 0,
                state: PatchState::Registered,
            });
        }

        for exp in &image.exports {
            let _ = kallsyms.register(&exp.name, exp.vaddr, exp.size, exp.kind, module_id);
        }

        self.modules.push(KpatchModule {
            name: String::from(name),
            version,
            image,
            patches,
            applied: false,
        });

        Ok(module_id)
    }

    /// Atomically apply all function patches in the named module.
    pub fn apply_module<M: PatchMemory>(
        &mut self,
        name: &str,
        mem: &mut M,
    ) -> Result<usize, LivepatchError> {
        let module = self
            .modules
            .iter_mut()
            .find(|m| m.name == name)
            .ok_or(LivepatchError::PatchNotFound)?;
        if module.applied {
            return Err(LivepatchError::PatchAlreadyApplied);
        }

        let mut applied_count = 0usize;
        for patch in &mut module.patches {
            let (kind, len) = TrampolinePatcher::apply(
                mem,
                patch.old_addr,
                patch.old_size,
                patch.new_addr,
                &mut patch.orig_bytes,
            )?;
            patch.kind = Some(kind);
            patch.patch_len = len;
            patch.state = PatchState::Applied;
            applied_count += 1;
        }
        module.applied = true;
        Ok(applied_count)
    }

    /// Atomically revert all function patches in the named module and synchronize RCU.
    pub fn revert_module<M: PatchMemory>(
        &mut self,
        name: &str,
        mem: &mut M,
    ) -> Result<usize, LivepatchError> {
        let module = self
            .modules
            .iter_mut()
            .find(|m| m.name == name)
            .ok_or(LivepatchError::PatchNotFound)?;
        if !module.applied {
            return Err(LivepatchError::PatchNotApplied);
        }

        let mut reverted_count = 0usize;
        for patch in &mut module.patches {
            if patch.state == PatchState::Applied {
                TrampolinePatcher::revert(mem, patch.old_addr, &patch.orig_bytes, patch.patch_len)?;
                patch.state = PatchState::Reverted;
                reverted_count += 1;
            }
        }
        module.applied = false;
        rcu::synchronize_rcu(0);
        Ok(reverted_count)
    }

    /// Unload a reverted module and remove its symbols from `kallsyms`.
    pub fn unload_module(
        &mut self,
        name: &str,
        kallsyms: &mut KallsymsTable,
    ) -> Result<(), LivepatchError> {
        let pos = self
            .modules
            .iter()
            .position(|m| m.name == name)
            .ok_or(LivepatchError::PatchNotFound)?;
        if self.modules[pos].applied {
            return Err(LivepatchError::PatchAlreadyApplied);
        }
        let module_id = self.modules[pos].image.module_id;
        kallsyms.unregister_module(module_id);
        self.modules.remove(pos);
        Ok(())
    }

    pub fn module_count(&self) -> usize {
        self.modules.len()
    }
}

impl Default for PatchManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Self-Test (Synthetic ELF64 ET_REL Builder + End-to-End Verification)
// ---------------------------------------------------------------------------

fn push_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_shdr(
    buf: &mut Vec<u8>,
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
) {
    push_u32(buf, sh_name);
    push_u32(buf, sh_type);
    push_u64(buf, sh_flags);
    push_u64(buf, 0); // sh_addr
    push_u64(buf, sh_offset);
    push_u64(buf, sh_size);
    push_u32(buf, sh_link);
    push_u32(buf, sh_info);
    push_u64(buf, sh_addralign);
    push_u64(buf, sh_entsize);
}

fn push_sym(
    buf: &mut Vec<u8>,
    st_name: u32,
    st_info: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
) {
    push_u32(buf, st_name);
    buf.push(st_info);
    buf.push(0); // st_other
    push_u16(buf, st_shndx);
    push_u64(buf, st_value);
    push_u64(buf, st_size);
}

fn push_rela(buf: &mut Vec<u8>, r_offset: u64, sym_idx: u32, rtype: u32, r_addend: i64) {
    push_u64(buf, r_offset);
    let r_info = ((sym_idx as u64) << 32) | (rtype as u64);
    push_u64(buf, r_info);
    push_i64(buf, r_addend);
}

/// Build a minimal valid ELF64 `ET_REL` object on the heap for `self_test()`.
///
/// Sections:
/// 0: NULL
/// 1: `.text` (16 bytes: `call rel32` at offset 1, `mov rax, imm64` at offset 7, `ret` at 15)
/// 2: `.symtab` (3 symbols: NULL, `kpatch_helper` UNDEF, `patched_fn` GLOBAL in `.text`)
/// 3: `.strtab`
/// 4: `.rela.text` (`R_X86_64_PC32` at offset 1 -> `kpatch_helper`, `R_X86_64_64` at offset 7 -> `patched_fn`)
/// 5: `.shstrtab`
fn build_test_elf64() -> Vec<u8> {
    let mut shstrtab = Vec::new();
    shstrtab.push(0);
    let sh_text_name = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".text\0");
    let sh_symtab_name = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".symtab\0");
    let sh_strtab_name = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".strtab\0");
    let sh_rela_name = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".rela.text\0");
    let sh_shstrtab_name = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".shstrtab\0");

    let mut strtab = Vec::new();
    strtab.push(0);
    let sym_helper_name = strtab.len() as u32;
    strtab.extend_from_slice(b"kpatch_helper\0");
    let sym_patched_name = strtab.len() as u32;
    strtab.extend_from_slice(b"patched_fn\0");

    // .text: 16 bytes
    // [0]: 0xE8 (call rel32), [1..5]: rel32 placeholder
    // [5..7]: 0x48 0xB8 (mov rax, imm64), [7..15]: imm64 placeholder
    // [15]: 0xC3 (ret)
    let text: [u8; 16] = [
        0xE8, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xC3,
    ];

    let mut symtab = Vec::new();
    push_sym(&mut symtab, 0, 0, SHN_UNDEF, 0, 0);
    push_sym(
        &mut symtab,
        sym_helper_name,
        (STB_GLOBAL << 4) | STT_FUNC,
        SHN_UNDEF,
        0,
        0,
    );
    push_sym(
        &mut symtab,
        sym_patched_name,
        (STB_GLOBAL << 4) | STT_FUNC,
        1, // .text
        0,
        16,
    );

    let mut rela_text = Vec::new();
    // R_X86_64_PC32 at offset 1 referencing sym 1 (kpatch_helper) with addend -4
    push_rela(&mut rela_text, 1, 1, R_X86_64_PC32, -4);
    // R_X86_64_64 at offset 7 referencing sym 2 (patched_fn) with addend 0
    push_rela(&mut rela_text, 7, 2, R_X86_64_64, 0);

    let mut elf = Vec::with_capacity(512);
    // ELF64 Header (64 bytes)
    elf.extend_from_slice(&ELF_MAGIC);
    elf.push(ELFCLASS64);
    elf.push(ELFDATA2LSB);
    elf.push(1); // EV_CURRENT
    elf.extend_from_slice(&[0u8; 9]); // padding
    push_u16(&mut elf, ET_REL);
    push_u16(&mut elf, EM_X86_64);
    push_u32(&mut elf, 1); // e_version
    push_u64(&mut elf, 0); // e_entry
    push_u64(&mut elf, 0); // e_phoff
    let shoff_pos = elf.len();
    push_u64(&mut elf, 0); // e_shoff placeholder
    push_u32(&mut elf, 0); // e_flags
    push_u16(&mut elf, ELF64_EHDR_SIZE as u16);
    push_u16(&mut elf, 0); // e_phentsize
    push_u16(&mut elf, 0); // e_phnum
    push_u16(&mut elf, ELF64_SHDR_SIZE as u16);
    push_u16(&mut elf, 6); // e_shnum
    push_u16(&mut elf, 5); // e_shstrndx

    let text_off = elf.len() as u64;
    elf.extend_from_slice(&text);

    let symtab_off = elf.len() as u64;
    elf.extend_from_slice(&symtab);

    let strtab_off = elf.len() as u64;
    elf.extend_from_slice(&strtab);

    let rela_off = elf.len() as u64;
    elf.extend_from_slice(&rela_text);

    let shstrtab_off = elf.len() as u64;
    elf.extend_from_slice(&shstrtab);

    let shoff = elf.len() as u64;
    elf[shoff_pos..shoff_pos + 8].copy_from_slice(&shoff.to_le_bytes());

    // 0: NULL
    push_shdr(&mut elf, 0, SHT_NULL, 0, 0, 0, 0, 0, 0, 0);
    // 1: .text
    push_shdr(
        &mut elf,
        sh_text_name,
        SHT_PROGBITS,
        SHF_ALLOC | SHF_EXECINSTR,
        text_off,
        text.len() as u64,
        0,
        0,
        16,
        0,
    );
    // 2: .symtab
    push_shdr(
        &mut elf,
        sh_symtab_name,
        SHT_SYMTAB,
        0,
        symtab_off,
        symtab.len() as u64,
        3, // sh_link -> .strtab
        1, // sh_info
        8,
        ELF64_SYM_SIZE as u64,
    );
    // 3: .strtab
    push_shdr(
        &mut elf,
        sh_strtab_name,
        SHT_STRTAB,
        0,
        strtab_off,
        strtab.len() as u64,
        0,
        0,
        1,
        0,
    );
    // 4: .rela.text
    push_shdr(
        &mut elf,
        sh_rela_name,
        SHT_RELA,
        0,
        rela_off,
        rela_text.len() as u64,
        2, // sh_link -> .symtab
        1, // sh_info -> .text
        8,
        ELF64_RELA_SIZE as u64,
    );
    // 5: .shstrtab
    push_shdr(
        &mut elf,
        sh_shstrtab_name,
        SHT_STRTAB,
        0,
        shstrtab_off,
        shstrtab.len() as u64,
        0,
        0,
        1,
        0,
    );

    elf
}

/// Run self-tests for kallsyms, ELF64 relocation, atomic trampoline patching, and rollback.
pub fn self_test() -> Result<(), &'static str> {
    // 1. Kallsyms forward & reverse lookup
    let mut ksyms = KallsymsTable::new();
    let target_fn_addr: u64 = 0xFFFF_8000_0010_0000;
    let helper_fn_addr: u64 = 0xFFFF_8000_0010_0200;
    ksyms.register("target_fn", target_fn_addr, 32, SymbolKind::Text, 0)?;
    ksyms.register("kpatch_helper", helper_fn_addr, 64, SymbolKind::Text, 0)?;

    if ksyms.lookup("target_fn") != Some(target_fn_addr) {
        return Err("livepatch self_test: kallsyms forward lookup failed");
    }
    if ksyms.lookup_addr(target_fn_addr + 12) != Some(("target_fn", 12)) {
        return Err("livepatch self_test: kallsyms reverse lookup failed");
    }

    // 2. ELF64 ET_REL relocation
    let elf = build_test_elf64();
    let module_base: u64 = 0xFFFF_8000_0010_1000;
    let mut mgr = PatchManager::new();
    mgr.register_module(
        "kpatch_test_mod",
        1,
        &elf,
        &mut ksyms,
        &[("target_fn", "patched_fn")],
        Some(module_base),
    )?;

    if ksyms.lookup("patched_fn") != Some(module_base) {
        return Err("livepatch self_test: module export not registered in kallsyms");
    }

    // Verify relocated .text bytes inside the loaded module
    {
        let mod_ref = &mgr.modules[0];
        if mod_ref.image.relocations_applied != 2 {
            return Err("livepatch self_test: unexpected relocation count");
        }
        let text = mod_ref
            .image
            .section_bytes(".text")
            .ok_or("livepatch self_test: missing .text section")?;
        // R_X86_64_PC32 at offset 1: S(0x...0200) + A(-4) - P(0x...1001) = -0xE05
        let expected_rel32 = ((helper_fn_addr as i128) - 4 - ((module_base + 1) as i128)) as i32;
        let actual_rel32 = i32::from_le_bytes([text[1], text[2], text[3], text[4]]);
        if actual_rel32 != expected_rel32 {
            return Err("livepatch self_test: R_X86_64_PC32 mismatch");
        }
        // R_X86_64_64 at offset 7: patched_fn == module_base
        let actual_abs64 = u64::from_le_bytes([
            text[7], text[8], text[9], text[10], text[11], text[12], text[13], text[14],
        ]);
        if actual_abs64 != module_base {
            return Err("livepatch self_test: R_X86_64_64 mismatch");
        }
    }

    // 3. Atomic trampoline patch & rollback on simulated target function buffer
    let orig_prologue: [u8; 16] = [
        0x55, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x10, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
        0xC3,
    ];
    let mut mem = BufferPatchMemory::new(target_fn_addr, orig_prologue.to_vec());

    let applied = mgr.apply_module("kpatch_test_mod", &mut mem)?;
    if applied != 1 {
        return Err("livepatch self_test: expected 1 applied patch");
    }
    // Near JMP rel32: 0xE9 <rel32>
    if mem.data[0] != 0xE9 {
        return Err("livepatch self_test: expected 0xE9 near JMP opcode");
    }
    let jmp_rel = i32::from_le_bytes([mem.data[1], mem.data[2], mem.data[3], mem.data[4]]);
    let expected_jmp_rel = ((module_base as i128) - ((target_fn_addr + 5) as i128)) as i32;
    if jmp_rel != expected_jmp_rel {
        return Err("livepatch self_test: trampoline rel32 target mismatch");
    }

    // Revert and verify exact original prologue restoration
    let reverted = mgr.revert_module("kpatch_test_mod", &mut mem)?;
    if reverted != 1 || mem.data.as_slice() != orig_prologue {
        return Err("livepatch self_test: prologue rollback mismatch");
    }

    // Unload module and verify symbol cleanup
    mgr.unload_module("kpatch_test_mod", &mut ksyms)?;
    if ksyms.lookup("patched_fn").is_some() || mgr.module_count() != 0 {
        return Err("livepatch self_test: module unload cleanup failed");
    }

    // 4. Far trampoline (> 2 GiB) encoding check
    let far_dst: u64 = 0xFFFF_9000_0000_1234;
    let (far_kind, far_bytes, far_len) =
        TrampolinePatcher::encode_trampoline(target_fn_addr, far_dst);
    if far_kind != TrampolineKind::FarJmpIndirect || far_len != 14 {
        return Err("livepatch self_test: expected 14-byte far trampoline");
    }
    if far_bytes[0] != 0xFF || far_bytes[1] != 0x25 {
        return Err("livepatch self_test: invalid far JMP [RIP+0] opcode");
    }
    let encoded_dst = u64::from_le_bytes([
        far_bytes[6],
        far_bytes[7],
        far_bytes[8],
        far_bytes[9],
        far_bytes[10],
        far_bytes[11],
        far_bytes[12],
        far_bytes[13],
    ]);
    if encoded_dst != far_dst {
        return Err("livepatch self_test: far trampoline target address mismatch");
    }

    Ok(())
}
