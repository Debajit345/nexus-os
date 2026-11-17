#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Clean-room `no_std` ELF64 parser and bounds-checked validator for x86_64 Linux binaries.

use core::ops::Range;

pub const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
pub const ELFCLASS64: u8 = 2;
pub const ELFDATA2LSB: u8 = 1;
pub const EV_CURRENT: u8 = 1;

pub const ET_NONE: u16 = 0;
pub const ET_REL: u16 = 1;
pub const ET_EXEC: u16 = 2;
pub const ET_DYN: u16 = 3;
pub const ET_CORE: u16 = 4;

pub const EM_X86_64: u16 = 62;

pub const ELF64_EHDR_SIZE: u16 = 64;
pub const ELF64_PHDR_SIZE: u16 = 56;
pub const ELF64_SHDR_SIZE: u16 = 64;
pub const ELF64_DYN_SIZE: usize = 16;

pub const PT_NULL: u32 = 0;
pub const PT_LOAD: u32 = 1;
pub const PT_DYNAMIC: u32 = 2;
pub const PT_INTERP: u32 = 3;
pub const PT_NOTE: u32 = 4;
pub const PT_SHLIB: u32 = 5;
pub const PT_PHDR: u32 = 6;
pub const PT_TLS: u32 = 7;
pub const PT_GNU_EH_FRAME: u32 = 0x6474_E550;
pub const PT_GNU_STACK: u32 = 0x6474_E551;
pub const PT_GNU_RELRO: u32 = 0x6474_E552;

pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

pub const SHT_NULL: u32 = 0;
pub const SHT_PROGBITS: u32 = 1;
pub const SHT_SYMTAB: u32 = 2;
pub const SHT_STRTAB: u32 = 3;
pub const SHT_RELA: u32 = 4;
pub const SHT_HASH: u32 = 5;
pub const SHT_DYNAMIC: u32 = 6;
pub const SHT_NOTE: u32 = 7;
pub const SHT_NOBITS: u32 = 8;
pub const SHT_REL: u32 = 9;
pub const SHT_DYNSYM: u32 = 11;

pub const SHF_WRITE: u64 = 0x1;
pub const SHF_ALLOC: u64 = 0x2;
pub const SHF_EXECINSTR: u64 = 0x4;

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
pub const DT_RUNPATH: i64 = 29;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ElfError {
    TooShort,
    InvalidMagic,
    UnsupportedClass(u8),
    UnsupportedEndianness(u8),
    UnsupportedVersion(u8),
    UnsupportedType(u16),
    UnsupportedMachine(u16),
    InvalidHeaderSize(u16),
    InvalidProgramHeaderEntrySize(u16),
    InvalidSectionHeaderEntrySize(u16),
    OutOfBounds,
    InvalidSegment,
    InvalidInterp,
    InvalidDynamic,
    InvalidStringTable,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Elf64Header {
    pub elf_type: u16,
    pub machine: u16,
    pub version: u32,
    pub entry: u64,
    pub phoff: u64,
    pub shoff: u64,
    pub flags: u32,
    pub ehsize: u16,
    pub phentsize: u16,
    pub phnum: u16,
    pub shentsize: u16,
    pub shnum: u16,
    pub shstrndx: u16,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ProgramHeader {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

impl ProgramHeader {
    #[inline]
    pub const fn is_load(&self) -> bool {
        self.p_type == PT_LOAD
    }

    #[inline]
    pub const fn is_interp(&self) -> bool {
        self.p_type == PT_INTERP
    }

    #[inline]
    pub const fn is_dynamic(&self) -> bool {
        self.p_type == PT_DYNAMIC
    }

    #[inline]
    pub const fn is_readable(&self) -> bool {
        (self.p_flags & PF_R) != 0
    }

    #[inline]
    pub const fn is_writable(&self) -> bool {
        (self.p_flags & PF_W) != 0
    }

    #[inline]
    pub const fn is_executable(&self) -> bool {
        (self.p_flags & PF_X) != 0
    }

    pub fn file_range(&self) -> Result<Range<usize>, ElfError> {
        checked_range(self.p_offset, self.p_filesz)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SectionHeader {
    pub sh_name: u32,
    pub sh_type: u32,
    pub sh_flags: u64,
    pub sh_addr: u64,
    pub sh_offset: u64,
    pub sh_size: u64,
    pub sh_link: u32,
    pub sh_info: u32,
    pub sh_addralign: u64,
    pub sh_entsize: u64,
}

impl SectionHeader {
    pub fn file_range(&self) -> Result<Range<usize>, ElfError> {
        if self.sh_type == SHT_NOBITS {
            return Ok(0..0);
        }
        checked_range(self.sh_offset, self.sh_size)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DynamicEntry {
    pub d_tag: i64,
    pub d_val: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Elf64<'a> {
    data: &'a [u8],
    header: Elf64Header,
}

impl<'a> Elf64<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, ElfError> {
        if data.len() < ELF64_EHDR_SIZE as usize {
            return Err(ElfError::TooShort);
        }

        if data[0..4] != ELF_MAGIC {
            return Err(ElfError::InvalidMagic);
        }

        let class = data[4];
        if class != ELFCLASS64 {
            return Err(ElfError::UnsupportedClass(class));
        }

        let endian = data[5];
        if endian != ELFDATA2LSB {
            return Err(ElfError::UnsupportedEndianness(endian));
        }

        let ident_version = data[6];
        if ident_version != EV_CURRENT {
            return Err(ElfError::UnsupportedVersion(ident_version));
        }

        let elf_type = read_u16(data, 16)?;
        if elf_type != ET_EXEC && elf_type != ET_DYN {
            return Err(ElfError::UnsupportedType(elf_type));
        }

        let machine = read_u16(data, 18)?;
        if machine != EM_X86_64 {
            return Err(ElfError::UnsupportedMachine(machine));
        }

        let version = read_u32(data, 20)?;
        if version != EV_CURRENT as u32 {
            return Err(ElfError::UnsupportedVersion(version as u8));
        }

        let entry = read_u64(data, 24)?;
        let phoff = read_u64(data, 32)?;
        let shoff = read_u64(data, 40)?;
        let flags = read_u32(data, 48)?;
        let ehsize = read_u16(data, 52)?;
        let phentsize = read_u16(data, 54)?;
        let phnum = read_u16(data, 56)?;
        let shentsize = read_u16(data, 58)?;
        let shnum = read_u16(data, 60)?;
        let shstrndx = read_u16(data, 62)?;

        if ehsize != ELF64_EHDR_SIZE {
            return Err(ElfError::InvalidHeaderSize(ehsize));
        }

        if phnum > 0 {
            if phentsize != ELF64_PHDR_SIZE {
                return Err(ElfError::InvalidProgramHeaderEntrySize(phentsize));
            }
            let ph_table_size = (phnum as u64)
                .checked_mul(phentsize as u64)
                .ok_or(ElfError::OutOfBounds)?;
            let range = checked_range(phoff, ph_table_size)?;
            if range.end > data.len() {
                return Err(ElfError::OutOfBounds);
            }
        }

        if shnum > 0 {
            if shentsize != ELF64_SHDR_SIZE {
                return Err(ElfError::InvalidSectionHeaderEntrySize(shentsize));
            }
            let sh_table_size = (shnum as u64)
                .checked_mul(shentsize as u64)
                .ok_or(ElfError::OutOfBounds)?;
            let range = checked_range(shoff, sh_table_size)?;
            if range.end > data.len() {
                return Err(ElfError::OutOfBounds);
            }
            if shstrndx != 0xFFFF && shstrndx >= shnum {
                return Err(ElfError::OutOfBounds);
            }
        }

        let header = Elf64Header {
            elf_type,
            machine,
            version,
            entry,
            phoff,
            shoff,
            flags,
            ehsize,
            phentsize,
            phnum,
            shentsize,
            shnum,
            shstrndx,
        };

        let elf = Self { data, header };
        elf.validate_program_headers()?;
        elf.validate_section_headers()?;
        Ok(elf)
    }

    #[inline]
    pub const fn header(&self) -> &Elf64Header {
        &self.header
    }

    #[inline]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.data
    }

    pub fn program_headers(&self) -> ProgramHeaderIter<'a> {
        ProgramHeaderIter {
            data: self.data,
            offset: self.header.phoff as usize,
            remaining: self.header.phnum,
            entry_size: self.header.phentsize as usize,
        }
    }

    pub fn load_segments(&self) -> impl Iterator<Item = ProgramHeader> + 'a {
        self.program_headers().filter(|ph| ph.is_load())
    }

    pub fn section_headers(&self) -> SectionHeaderIter<'a> {
        SectionHeaderIter {
            data: self.data,
            offset: self.header.shoff as usize,
            remaining: self.header.shnum,
            entry_size: self.header.shentsize as usize,
        }
    }

    pub fn segment_data(&self, ph: &ProgramHeader) -> Result<&'a [u8], ElfError> {
        let range = ph.file_range()?;
        self.data.get(range).ok_or(ElfError::OutOfBounds)
    }

    pub fn section_data(&self, sh: &SectionHeader) -> Result<&'a [u8], ElfError> {
        if sh.sh_type == SHT_NOBITS {
            return Ok(&[]);
        }
        let range = sh.file_range()?;
        self.data.get(range).ok_or(ElfError::OutOfBounds)
    }

    pub fn interp(&self) -> Result<Option<&'a str>, ElfError> {
        for ph in self.program_headers() {
            if ph.is_interp() {
                let bytes = self.segment_data(&ph)?;
                let nul_pos = bytes
                    .iter()
                    .position(|&b| b == 0)
                    .ok_or(ElfError::InvalidInterp)?;
                if nul_pos == 0 {
                    return Err(ElfError::InvalidInterp);
                }
                let s =
                    core::str::from_utf8(&bytes[..nul_pos]).map_err(|_| ElfError::InvalidInterp)?;
                return Ok(Some(s));
            }
        }
        Ok(None)
    }

    pub fn dynamic_entries(&self) -> Result<Option<DynamicIter<'a>>, ElfError> {
        for ph in self.program_headers() {
            if ph.is_dynamic() {
                let bytes = self.segment_data(&ph)?;
                if bytes.len() % ELF64_DYN_SIZE != 0 {
                    return Err(ElfError::InvalidDynamic);
                }
                return Ok(Some(DynamicIter {
                    data: bytes,
                    offset: 0,
                    done: false,
                }));
            }
        }
        Ok(None)
    }

    pub fn section_name(&self, sh: &SectionHeader) -> Result<Option<&'a str>, ElfError> {
        if self.header.shnum == 0 || self.header.shstrndx == 0 || self.header.shstrndx == 0xFFFF {
            return Ok(None);
        }
        let strtab_sh = self
            .section_headers()
            .nth(self.header.shstrndx as usize)
            .ok_or(ElfError::OutOfBounds)?;
        if strtab_sh.sh_type != SHT_STRTAB {
            return Err(ElfError::InvalidStringTable);
        }
        let strtab = self.section_data(&strtab_sh)?;
        let start = sh.sh_name as usize;
        let tail = strtab.get(start..).ok_or(ElfError::OutOfBounds)?;
        let nul_pos = tail
            .iter()
            .position(|&b| b == 0)
            .ok_or(ElfError::InvalidStringTable)?;
        let name =
            core::str::from_utf8(&tail[..nul_pos]).map_err(|_| ElfError::InvalidStringTable)?;
        Ok(Some(name))
    }

    fn validate_program_headers(&self) -> Result<(), ElfError> {
        for ph in self.program_headers() {
            if ph.p_filesz > ph.p_memsz && ph.is_load() {
                return Err(ElfError::InvalidSegment);
            }
            let range = ph.file_range()?;
            if range.end > self.data.len() {
                return Err(ElfError::OutOfBounds);
            }
            if ph.is_load() && ph.p_align > 1 {
                if !ph.p_align.is_power_of_two() {
                    return Err(ElfError::InvalidSegment);
                }
                if (ph.p_offset % ph.p_align) != (ph.p_vaddr % ph.p_align) {
                    return Err(ElfError::InvalidSegment);
                }
            }
            if ph.is_dynamic() && (ph.p_filesz as usize) % ELF64_DYN_SIZE != 0 {
                return Err(ElfError::InvalidDynamic);
            }
        }
        Ok(())
    }

    fn validate_section_headers(&self) -> Result<(), ElfError> {
        for sh in self.section_headers() {
            if sh.sh_type != SHT_NOBITS {
                let range = sh.file_range()?;
                if range.end > self.data.len() {
                    return Err(ElfError::OutOfBounds);
                }
            }
        }
        Ok(())
    }
}

pub struct ProgramHeaderIter<'a> {
    data: &'a [u8],
    offset: usize,
    remaining: u16,
    entry_size: usize,
}

impl<'a> Iterator for ProgramHeaderIter<'a> {
    type Item = ProgramHeader;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let off = self.offset;
        let ph = ProgramHeader {
            p_type: read_u32(self.data, off).ok()?,
            p_flags: read_u32(self.data, off + 4).ok()?,
            p_offset: read_u64(self.data, off + 8).ok()?,
            p_vaddr: read_u64(self.data, off + 16).ok()?,
            p_paddr: read_u64(self.data, off + 24).ok()?,
            p_filesz: read_u64(self.data, off + 32).ok()?,
            p_memsz: read_u64(self.data, off + 40).ok()?,
            p_align: read_u64(self.data, off + 48).ok()?,
        };
        self.offset += self.entry_size;
        self.remaining -= 1;
        Some(ph)
    }
}

pub struct SectionHeaderIter<'a> {
    data: &'a [u8],
    offset: usize,
    remaining: u16,
    entry_size: usize,
}

impl<'a> Iterator for SectionHeaderIter<'a> {
    type Item = SectionHeader;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let off = self.offset;
        let sh = SectionHeader {
            sh_name: read_u32(self.data, off).ok()?,
            sh_type: read_u32(self.data, off + 4).ok()?,
            sh_flags: read_u64(self.data, off + 8).ok()?,
            sh_addr: read_u64(self.data, off + 16).ok()?,
            sh_offset: read_u64(self.data, off + 24).ok()?,
            sh_size: read_u64(self.data, off + 32).ok()?,
            sh_link: read_u32(self.data, off + 40).ok()?,
            sh_info: read_u32(self.data, off + 44).ok()?,
            sh_addralign: read_u64(self.data, off + 48).ok()?,
            sh_entsize: read_u64(self.data, off + 56).ok()?,
        };
        self.offset += self.entry_size;
        self.remaining -= 1;
        Some(sh)
    }
}

pub struct DynamicIter<'a> {
    data: &'a [u8],
    offset: usize,
    done: bool,
}

impl<'a> Iterator for DynamicIter<'a> {
    type Item = DynamicEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.offset + ELF64_DYN_SIZE > self.data.len() {
            return None;
        }
        let d_tag = read_i64(self.data, self.offset).ok()?;
        let d_val = read_u64(self.data, self.offset + 8).ok()?;
        self.offset += ELF64_DYN_SIZE;
        if d_tag == DT_NULL {
            self.done = true;
        }
        Some(DynamicEntry { d_tag, d_val })
    }
}

fn checked_range(offset: u64, size: u64) -> Result<Range<usize>, ElfError> {
    let start = usize::try_from(offset).map_err(|_| ElfError::OutOfBounds)?;
    let len = usize::try_from(size).map_err(|_| ElfError::OutOfBounds)?;
    let end = start.checked_add(len).ok_or(ElfError::OutOfBounds)?;
    Ok(start..end)
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16, ElfError> {
    let bytes: [u8; 2] = data
        .get(offset..offset + 2)
        .ok_or(ElfError::OutOfBounds)?
        .try_into()
        .map_err(|_| ElfError::OutOfBounds)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, ElfError> {
    let bytes: [u8; 4] = data
        .get(offset..offset + 4)
        .ok_or(ElfError::OutOfBounds)?
        .try_into()
        .map_err(|_| ElfError::OutOfBounds)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64, ElfError> {
    let bytes: [u8; 8] = data
        .get(offset..offset + 8)
        .ok_or(ElfError::OutOfBounds)?
        .try_into()
        .map_err(|_| ElfError::OutOfBounds)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64(data: &[u8], offset: usize) -> Result<i64, ElfError> {
    let bytes: [u8; 8] = data
        .get(offset..offset + 8)
        .ok_or(ElfError::OutOfBounds)?
        .try_into()
        .map_err(|_| ElfError::OutOfBounds)?;
    Ok(i64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u16(buf: &mut [u8], off: usize, val: u16) {
        buf[off..off + 2].copy_from_slice(&val.to_le_bytes());
    }

    fn put_u32(buf: &mut [u8], off: usize, val: u32) {
        buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }

    fn put_u64(buf: &mut [u8], off: usize, val: u64) {
        buf[off..off + 8].copy_from_slice(&val.to_le_bytes());
    }

    fn build_minimal_elf() -> Vec<u8> {
        // Layout:
        // 0x000..0x040: ELF64 header (64 bytes)
        // 0x040..0x0E8: 3 program headers (3 * 56 = 168 bytes)
        // 0x100..0x11C: PT_INTERP string "/lib64/ld-linux-x86-64.so.2\0" (28 bytes)
        // 0x120..0x150: PT_DYNAMIC (3 entries * 16 = 48 bytes)
        // 0x150..0x160: .shstrtab ("\0.text\0.shstrtab\0")
        // 0x160..0x220: 3 section headers (3 * 64 = 192 bytes)
        let mut buf = vec![0u8; 0x220];

        // e_ident
        buf[0..4].copy_from_slice(&ELF_MAGIC);
        buf[4] = ELFCLASS64;
        buf[5] = ELFDATA2LSB;
        buf[6] = EV_CURRENT;

        put_u16(&mut buf, 16, ET_DYN);
        put_u16(&mut buf, 18, EM_X86_64);
        put_u32(&mut buf, 20, EV_CURRENT as u32);
        put_u64(&mut buf, 24, 0x401000); // e_entry
        put_u64(&mut buf, 32, 0x40); // e_phoff
        put_u64(&mut buf, 40, 0x160); // e_shoff
        put_u16(&mut buf, 52, ELF64_EHDR_SIZE);
        put_u16(&mut buf, 54, ELF64_PHDR_SIZE);
        put_u16(&mut buf, 56, 3); // e_phnum
        put_u16(&mut buf, 58, ELF64_SHDR_SIZE);
        put_u16(&mut buf, 60, 3); // e_shnum
        put_u16(&mut buf, 62, 2); // e_shstrndx

        // PHDR 0: PT_LOAD
        let ph0 = 0x40;
        put_u32(&mut buf, ph0, PT_LOAD);
        put_u32(&mut buf, ph0 + 4, PF_R | PF_X);
        put_u64(&mut buf, ph0 + 8, 0);
        put_u64(&mut buf, ph0 + 16, 0x400000);
        put_u64(&mut buf, ph0 + 24, 0x400000);
        put_u64(&mut buf, ph0 + 32, 0x150);
        put_u64(&mut buf, ph0 + 40, 0x200);
        put_u64(&mut buf, ph0 + 48, 0x1000);

        // PHDR 1: PT_INTERP
        let interp = b"/lib64/ld-linux-x86-64.so.2\0";
        buf[0x100..0x100 + interp.len()].copy_from_slice(interp);
        let ph1 = 0x40 + 56;
        put_u32(&mut buf, ph1, PT_INTERP);
        put_u32(&mut buf, ph1 + 4, PF_R);
        put_u64(&mut buf, ph1 + 8, 0x100);
        put_u64(&mut buf, ph1 + 16, 0x400100);
        put_u64(&mut buf, ph1 + 24, 0x400100);
        put_u64(&mut buf, ph1 + 32, interp.len() as u64);
        put_u64(&mut buf, ph1 + 40, interp.len() as u64);
        put_u64(&mut buf, ph1 + 48, 1);

        // PHDR 2: PT_DYNAMIC
        let ph2 = 0x40 + 112;
        put_u32(&mut buf, ph2, PT_DYNAMIC);
        put_u32(&mut buf, ph2 + 4, PF_R | PF_W);
        put_u64(&mut buf, ph2 + 8, 0x120);
        put_u64(&mut buf, ph2 + 16, 0x400120);
        put_u64(&mut buf, ph2 + 24, 0x400120);
        put_u64(&mut buf, ph2 + 32, 48);
        put_u64(&mut buf, ph2 + 40, 48);
        put_u64(&mut buf, ph2 + 48, 8);

        // Dynamic entries at 0x120
        put_u64(&mut buf, 0x120, DT_NEEDED as u64);
        put_u64(&mut buf, 0x128, 1);
        put_u64(&mut buf, 0x130, DT_STRTAB as u64);
        put_u64(&mut buf, 0x138, 0x400200);
        put_u64(&mut buf, 0x140, DT_NULL as u64);
        put_u64(&mut buf, 0x148, 0);

        // .shstrtab at 0x150
        let shstrtab = b"\0.text\0.shstrtab\0";
        buf[0x150..0x150 + shstrtab.len()].copy_from_slice(shstrtab);

        // SHDR 0: SHT_NULL at 0x160
        // SHDR 1: .text at 0x1A0
        let sh1 = 0x160 + 64;
        put_u32(&mut buf, sh1, 1); // ".text"
        put_u32(&mut buf, sh1 + 4, SHT_PROGBITS);
        put_u64(&mut buf, sh1 + 8, SHF_ALLOC | SHF_EXECINSTR);
        put_u64(&mut buf, sh1 + 16, 0x401000);
        put_u64(&mut buf, sh1 + 24, 0x100);
        put_u64(&mut buf, sh1 + 32, 0x10);

        // SHDR 2: .shstrtab at 0x1E0
        let sh2 = 0x160 + 128;
        put_u32(&mut buf, sh2, 7); // ".shstrtab"
        put_u32(&mut buf, sh2 + 4, SHT_STRTAB);
        put_u64(&mut buf, sh2 + 24, 0x150);
        put_u64(&mut buf, sh2 + 32, shstrtab.len() as u64);

        buf
    }

    #[test]
    fn parse_valid_elf64() {
        let data = build_minimal_elf();
        let elf = Elf64::parse(&data).expect("valid ELF64 should parse");
        assert_eq!(elf.header().elf_type, ET_DYN);
        assert_eq!(elf.header().machine, EM_X86_64);
        assert_eq!(elf.header().entry, 0x401000);

        let loads: Vec<_> = elf.load_segments().collect();
        assert_eq!(loads.len(), 1);
        assert!(loads[0].is_readable());
        assert!(loads[0].is_executable());
        assert!(!loads[0].is_writable());

        assert_eq!(elf.interp().unwrap(), Some("/lib64/ld-linux-x86-64.so.2"));

        let dyn_entries: Vec<_> = elf.dynamic_entries().unwrap().unwrap().collect();
        assert_eq!(dyn_entries.len(), 3);
        assert_eq!(dyn_entries[0].d_tag, DT_NEEDED);
        assert_eq!(dyn_entries[0].d_val, 1);
        assert_eq!(dyn_entries[1].d_tag, DT_STRTAB);
        assert_eq!(dyn_entries[2].d_tag, DT_NULL);

        let sections: Vec<_> = elf.section_headers().collect();
        assert_eq!(sections.len(), 3);
        assert_eq!(elf.section_name(&sections[1]).unwrap(), Some(".text"));
        assert_eq!(elf.section_name(&sections[2]).unwrap(), Some(".shstrtab"));
    }

    #[test]
    fn reject_bad_magic_and_short_input() {
        assert_eq!(Elf64::parse(&[0u8; 10]), Err(ElfError::TooShort));
        let mut data = build_minimal_elf();
        data[0] = 0x00;
        assert_eq!(Elf64::parse(&data), Err(ElfError::InvalidMagic));
    }

    #[test]
    fn reject_wrong_arch_and_class() {
        let mut data = build_minimal_elf();
        data[4] = 1; // ELFCLASS32
        assert_eq!(Elf64::parse(&data), Err(ElfError::UnsupportedClass(1)));

        let mut data = build_minimal_elf();
        put_u16(&mut data, 18, 183); // EM_AARCH64
        assert_eq!(Elf64::parse(&data), Err(ElfError::UnsupportedMachine(183)));
    }

    #[test]
    fn reject_out_of_bounds_segment() {
        let mut data = build_minimal_elf();
        // Corrupt PT_LOAD p_filesz to exceed file length
        put_u64(&mut data, 0x40 + 32, 0x10_0000);
        put_u64(&mut data, 0x40 + 40, 0x10_0000);
        assert_eq!(Elf64::parse(&data), Err(ElfError::OutOfBounds));
    }

    #[test]
    fn reject_filesz_greater_than_memsz() {
        let mut data = build_minimal_elf();
        put_u64(&mut data, 0x40 + 32, 0x100);
        put_u64(&mut data, 0x40 + 40, 0x50);
        assert_eq!(Elf64::parse(&data), Err(ElfError::InvalidSegment));
    }
}
