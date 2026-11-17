#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Clean-room `no_std` PE32+ parser and bounds-checked validator for x86_64 Windows binaries.

use core::ops::Range;

pub const DOS_MAGIC: u16 = 0x5A4D; // "MZ"
pub const PE_SIGNATURE: [u8; 4] = [b'P', b'E', 0, 0];
pub const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
pub const PE32_PLUS_MAGIC: u16 = 0x020B;

pub const DOS_HEADER_SIZE: usize = 64;
pub const COFF_HEADER_SIZE: usize = 20;
pub const PE32_PLUS_OPT_HEADER_MIN_SIZE: usize = 112;
pub const DATA_DIRECTORY_SIZE: usize = 8;
pub const SECTION_HEADER_SIZE: usize = 40;

pub const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
pub const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
pub const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
pub const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
pub const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
pub const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

pub const IMAGE_DIRECTORY_ENTRY_EXPORT: usize = 0;
pub const IMAGE_DIRECTORY_ENTRY_IMPORT: usize = 1;
pub const IMAGE_DIRECTORY_ENTRY_RESOURCE: usize = 2;
pub const IMAGE_DIRECTORY_ENTRY_EXCEPTION: usize = 3;
pub const IMAGE_DIRECTORY_ENTRY_BASERELOC: usize = 5;
pub const IMAGE_DIRECTORY_ENTRY_TLS: usize = 9;
pub const IMAGE_DIRECTORY_ENTRY_IAT: usize = 12;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PeError {
    TooShort,
    InvalidDosMagic,
    InvalidPeSignature,
    UnsupportedMachine(u16),
    UnsupportedOptionalMagic(u16),
    InvalidOptionalHeaderSize(u16),
    InvalidAlignment,
    InvalidImageSize,
    OutOfBounds,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DosHeader {
    pub e_magic: u16,
    pub e_lfanew: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CoffHeader {
    pub machine: u16,
    pub number_of_sections: u16,
    pub time_date_stamp: u32,
    pub pointer_to_symbol_table: u32,
    pub number_of_symbols: u32,
    pub size_of_optional_header: u16,
    pub characteristics: u16,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DataDirectory {
    pub virtual_address: u32,
    pub size: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OptionalHeader64 {
    pub magic: u16,
    pub major_linker_version: u8,
    pub minor_linker_version: u8,
    pub size_of_code: u32,
    pub size_of_initialized_data: u32,
    pub size_of_uninitialized_data: u32,
    pub address_of_entry_point: u32,
    pub base_of_code: u32,
    pub image_base: u64,
    pub section_alignment: u32,
    pub file_alignment: u32,
    pub major_os_version: u16,
    pub minor_os_version: u16,
    pub major_subsystem_version: u16,
    pub minor_subsystem_version: u16,
    pub size_of_image: u32,
    pub size_of_headers: u32,
    pub checksum: u32,
    pub subsystem: u16,
    pub dll_characteristics: u16,
    pub size_of_stack_reserve: u64,
    pub size_of_stack_commit: u64,
    pub size_of_heap_reserve: u64,
    pub size_of_heap_commit: u64,
    pub loader_flags: u32,
    pub number_of_rva_and_sizes: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SectionHeader {
    pub name: [u8; 8],
    pub virtual_size: u32,
    pub virtual_address: u32,
    pub size_of_raw_data: u32,
    pub pointer_to_raw_data: u32,
    pub pointer_to_relocations: u32,
    pub pointer_to_linenumbers: u32,
    pub number_of_relocations: u16,
    pub number_of_linenumbers: u16,
    pub characteristics: u32,
}

impl SectionHeader {
    pub fn name_str(&self) -> Option<&str> {
        let len = self.name.iter().position(|&b| b == 0).unwrap_or(8);
        core::str::from_utf8(&self.name[..len]).ok()
    }

    #[inline]
    pub const fn is_readable(&self) -> bool {
        (self.characteristics & IMAGE_SCN_MEM_READ) != 0
    }

    #[inline]
    pub const fn is_writable(&self) -> bool {
        (self.characteristics & IMAGE_SCN_MEM_WRITE) != 0
    }

    #[inline]
    pub const fn is_executable(&self) -> bool {
        (self.characteristics & IMAGE_SCN_MEM_EXECUTE) != 0
    }

    pub fn file_range(&self) -> Result<Range<usize>, PeError> {
        if self.size_of_raw_data == 0 {
            return Ok(0..0);
        }
        checked_range(
            self.pointer_to_raw_data as usize,
            self.size_of_raw_data as usize,
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Pe32Plus<'a> {
    data: &'a [u8],
    dos: DosHeader,
    coff: CoffHeader,
    optional: OptionalHeader64,
    data_dirs_offset: usize,
    sections_offset: usize,
}

impl<'a> Pe32Plus<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, PeError> {
        if data.len() < DOS_HEADER_SIZE {
            return Err(PeError::TooShort);
        }

        let e_magic = read_u16(data, 0)?;
        if e_magic != DOS_MAGIC {
            return Err(PeError::InvalidDosMagic);
        }
        let e_lfanew = read_u32(data, 60)?;
        let pe_off = usize::try_from(e_lfanew).map_err(|_| PeError::OutOfBounds)?;
        if pe_off < DOS_HEADER_SIZE {
            return Err(PeError::OutOfBounds);
        }

        let sig_end = pe_off.checked_add(4).ok_or(PeError::OutOfBounds)?;
        let sig = data.get(pe_off..sig_end).ok_or(PeError::OutOfBounds)?;
        if sig != PE_SIGNATURE {
            return Err(PeError::InvalidPeSignature);
        }

        let coff_off = sig_end;
        let opt_off = coff_off
            .checked_add(COFF_HEADER_SIZE)
            .ok_or(PeError::OutOfBounds)?;
        if opt_off > data.len() {
            return Err(PeError::OutOfBounds);
        }

        let machine = read_u16(data, coff_off)?;
        if machine != IMAGE_FILE_MACHINE_AMD64 {
            return Err(PeError::UnsupportedMachine(machine));
        }

        let coff = CoffHeader {
            machine,
            number_of_sections: read_u16(data, coff_off + 2)?,
            time_date_stamp: read_u32(data, coff_off + 4)?,
            pointer_to_symbol_table: read_u32(data, coff_off + 8)?,
            number_of_symbols: read_u32(data, coff_off + 12)?,
            size_of_optional_header: read_u16(data, coff_off + 16)?,
            characteristics: read_u16(data, coff_off + 18)?,
        };

        let opt_size = coff.size_of_optional_header as usize;
        if opt_size < PE32_PLUS_OPT_HEADER_MIN_SIZE {
            return Err(PeError::InvalidOptionalHeaderSize(
                coff.size_of_optional_header,
            ));
        }
        let sections_offset = opt_off.checked_add(opt_size).ok_or(PeError::OutOfBounds)?;
        if sections_offset > data.len() {
            return Err(PeError::OutOfBounds);
        }

        let magic = read_u16(data, opt_off)?;
        if magic != PE32_PLUS_MAGIC {
            return Err(PeError::UnsupportedOptionalMagic(magic));
        }

        let section_alignment = read_u32(data, opt_off + 32)?;
        let file_alignment = read_u32(data, opt_off + 36)?;
        if section_alignment == 0
            || file_alignment == 0
            || !section_alignment.is_power_of_two()
            || !file_alignment.is_power_of_two()
            || section_alignment < file_alignment
        {
            return Err(PeError::InvalidAlignment);
        }

        let size_of_image = read_u32(data, opt_off + 56)?;
        let size_of_headers = read_u32(data, opt_off + 60)?;
        if size_of_headers > size_of_image {
            return Err(PeError::InvalidImageSize);
        }

        let number_of_rva_and_sizes = read_u32(data, opt_off + 108)?;
        let dirs_bytes = (number_of_rva_and_sizes as usize)
            .checked_mul(DATA_DIRECTORY_SIZE)
            .ok_or(PeError::OutOfBounds)?;
        let dirs_total = PE32_PLUS_OPT_HEADER_MIN_SIZE
            .checked_add(dirs_bytes)
            .ok_or(PeError::OutOfBounds)?;
        if dirs_total > opt_size {
            return Err(PeError::InvalidOptionalHeaderSize(
                coff.size_of_optional_header,
            ));
        }

        let optional = OptionalHeader64 {
            magic,
            major_linker_version: data[opt_off + 2],
            minor_linker_version: data[opt_off + 3],
            size_of_code: read_u32(data, opt_off + 4)?,
            size_of_initialized_data: read_u32(data, opt_off + 8)?,
            size_of_uninitialized_data: read_u32(data, opt_off + 12)?,
            address_of_entry_point: read_u32(data, opt_off + 16)?,
            base_of_code: read_u32(data, opt_off + 20)?,
            image_base: read_u64(data, opt_off + 24)?,
            section_alignment,
            file_alignment,
            major_os_version: read_u16(data, opt_off + 40)?,
            minor_os_version: read_u16(data, opt_off + 42)?,
            major_subsystem_version: read_u16(data, opt_off + 48)?,
            minor_subsystem_version: read_u16(data, opt_off + 50)?,
            size_of_image,
            size_of_headers,
            checksum: read_u32(data, opt_off + 64)?,
            subsystem: read_u16(data, opt_off + 68)?,
            dll_characteristics: read_u16(data, opt_off + 70)?,
            size_of_stack_reserve: read_u64(data, opt_off + 72)?,
            size_of_stack_commit: read_u64(data, opt_off + 80)?,
            size_of_heap_reserve: read_u64(data, opt_off + 88)?,
            size_of_heap_commit: read_u64(data, opt_off + 96)?,
            loader_flags: read_u32(data, opt_off + 104)?,
            number_of_rva_and_sizes,
        };

        let sec_table_bytes = (coff.number_of_sections as usize)
            .checked_mul(SECTION_HEADER_SIZE)
            .ok_or(PeError::OutOfBounds)?;
        let sec_table_end = sections_offset
            .checked_add(sec_table_bytes)
            .ok_or(PeError::OutOfBounds)?;
        if sec_table_end > data.len() {
            return Err(PeError::OutOfBounds);
        }

        let pe = Self {
            data,
            dos: DosHeader { e_magic, e_lfanew },
            coff,
            optional,
            data_dirs_offset: opt_off + PE32_PLUS_OPT_HEADER_MIN_SIZE,
            sections_offset,
        };

        pe.validate_sections()?;
        Ok(pe)
    }

    #[inline]
    pub const fn dos_header(&self) -> &DosHeader {
        &self.dos
    }

    #[inline]
    pub const fn coff_header(&self) -> &CoffHeader {
        &self.coff
    }

    #[inline]
    pub const fn optional_header(&self) -> &OptionalHeader64 {
        &self.optional
    }

    pub fn data_directory(&self, index: usize) -> Option<DataDirectory> {
        if index >= self.optional.number_of_rva_and_sizes as usize {
            return None;
        }
        let off = self
            .data_dirs_offset
            .checked_add(index * DATA_DIRECTORY_SIZE)?;
        Some(DataDirectory {
            virtual_address: read_u32(self.data, off).ok()?,
            size: read_u32(self.data, off + 4).ok()?,
        })
    }

    pub fn sections(&self) -> SectionIter<'a> {
        SectionIter {
            data: self.data,
            offset: self.sections_offset,
            remaining: self.coff.number_of_sections,
        }
    }

    pub fn section_data(&self, sec: &SectionHeader) -> Result<&'a [u8], PeError> {
        let range = sec.file_range()?;
        self.data.get(range).ok_or(PeError::OutOfBounds)
    }

    pub fn rva_to_offset(&self, rva: u32) -> Option<usize> {
        for sec in self.sections() {
            let span = sec.virtual_size.max(sec.size_of_raw_data);
            let end = sec.virtual_address.checked_add(span)?;
            if rva >= sec.virtual_address && rva < end {
                let delta = rva - sec.virtual_address;
                if delta < sec.size_of_raw_data {
                    return (sec.pointer_to_raw_data as usize).checked_add(delta as usize);
                }
                return None;
            }
        }
        None
    }

    fn validate_sections(&self) -> Result<(), PeError> {
        for sec in self.sections() {
            if sec.size_of_raw_data > 0 {
                let range = sec.file_range()?;
                if range.end > self.data.len() {
                    return Err(PeError::OutOfBounds);
                }
            }
            let va_end = sec
                .virtual_address
                .checked_add(sec.virtual_size)
                .ok_or(PeError::OutOfBounds)?;
            if va_end > self.optional.size_of_image {
                return Err(PeError::InvalidImageSize);
            }
        }
        Ok(())
    }
}

pub struct SectionIter<'a> {
    data: &'a [u8],
    offset: usize,
    remaining: u16,
}

impl<'a> Iterator for SectionIter<'a> {
    type Item = SectionHeader;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let off = self.offset;
        let mut name = [0u8; 8];
        name.copy_from_slice(self.data.get(off..off + 8)?);
        let sec = SectionHeader {
            name,
            virtual_size: read_u32(self.data, off + 8).ok()?,
            virtual_address: read_u32(self.data, off + 12).ok()?,
            size_of_raw_data: read_u32(self.data, off + 16).ok()?,
            pointer_to_raw_data: read_u32(self.data, off + 20).ok()?,
            pointer_to_relocations: read_u32(self.data, off + 24).ok()?,
            pointer_to_linenumbers: read_u32(self.data, off + 28).ok()?,
            number_of_relocations: read_u16(self.data, off + 32).ok()?,
            number_of_linenumbers: read_u16(self.data, off + 34).ok()?,
            characteristics: read_u32(self.data, off + 36).ok()?,
        };
        self.offset += SECTION_HEADER_SIZE;
        self.remaining -= 1;
        Some(sec)
    }
}

fn checked_range(start: usize, len: usize) -> Result<Range<usize>, PeError> {
    let end = start.checked_add(len).ok_or(PeError::OutOfBounds)?;
    Ok(start..end)
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16, PeError> {
    let bytes: [u8; 2] = data
        .get(offset..offset + 2)
        .ok_or(PeError::OutOfBounds)?
        .try_into()
        .map_err(|_| PeError::OutOfBounds)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, PeError> {
    let bytes: [u8; 4] = data
        .get(offset..offset + 4)
        .ok_or(PeError::OutOfBounds)?
        .try_into()
        .map_err(|_| PeError::OutOfBounds)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64, PeError> {
    let bytes: [u8; 8] = data
        .get(offset..offset + 8)
        .ok_or(PeError::OutOfBounds)?
        .try_into()
        .map_err(|_| PeError::OutOfBounds)?;
    Ok(u64::from_le_bytes(bytes))
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

    fn build_minimal_pe() -> Vec<u8> {
        // Layout:
        // 0x000..0x040: DOS header (e_lfanew = 0x80)
        // 0x080..0x084: PE\0\0
        // 0x084..0x098: COFF header (20 bytes)
        // 0x098..0x188: Optional header PE32+ (112 + 16*8 = 240 bytes)
        // 0x188..0x1D8: 2 section headers (2 * 40 = 80 bytes)
        // 0x200..0x220: .text raw data
        // 0x220..0x240: .rdata raw data
        let mut buf = vec![0u8; 0x240];

        put_u16(&mut buf, 0, DOS_MAGIC);
        put_u32(&mut buf, 60, 0x80);

        buf[0x80..0x84].copy_from_slice(&PE_SIGNATURE);

        let coff = 0x84;
        put_u16(&mut buf, coff, IMAGE_FILE_MACHINE_AMD64);
        put_u16(&mut buf, coff + 2, 2); // number_of_sections
        put_u16(&mut buf, coff + 16, 240); // size_of_optional_header
        put_u16(&mut buf, coff + 18, 0x0022); // executable + large address aware

        let opt = 0x98;
        put_u16(&mut buf, opt, PE32_PLUS_MAGIC);
        put_u32(&mut buf, opt + 16, 0x1000); // address_of_entry_point
        put_u64(&mut buf, opt + 24, 0x1_4000_0000); // image_base
        put_u32(&mut buf, opt + 32, 0x1000); // section_alignment
        put_u32(&mut buf, opt + 36, 0x200); // file_alignment
        put_u32(&mut buf, opt + 56, 0x3000); // size_of_image
        put_u32(&mut buf, opt + 60, 0x200); // size_of_headers
        put_u16(&mut buf, opt + 68, 3); // IMAGE_SUBSYSTEM_WINDOWS_CUI
        put_u32(&mut buf, opt + 108, 16); // number_of_rva_and_sizes

        // Import directory (index 1)
        let imp_dir = opt + 112 + IMAGE_DIRECTORY_ENTRY_IMPORT * 8;
        put_u32(&mut buf, imp_dir, 0x2000);
        put_u32(&mut buf, imp_dir + 4, 0x20);

        // Section 0: .text
        let s0 = 0x188;
        buf[s0..s0 + 5].copy_from_slice(b".text");
        put_u32(&mut buf, s0 + 8, 0x20); // virtual_size
        put_u32(&mut buf, s0 + 12, 0x1000); // virtual_address
        put_u32(&mut buf, s0 + 16, 0x20); // size_of_raw_data
        put_u32(&mut buf, s0 + 20, 0x200); // pointer_to_raw_data
        put_u32(
            &mut buf,
            s0 + 36,
            IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ,
        );
        buf[0x200] = 0xC3; // ret

        // Section 1: .rdata
        let s1 = 0x188 + 40;
        buf[s1..s1 + 6].copy_from_slice(b".rdata");
        put_u32(&mut buf, s1 + 8, 0x20);
        put_u32(&mut buf, s1 + 12, 0x2000);
        put_u32(&mut buf, s1 + 16, 0x20);
        put_u32(&mut buf, s1 + 20, 0x220);
        put_u32(
            &mut buf,
            s1 + 36,
            IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ,
        );

        buf
    }

    #[test]
    fn parse_valid_pe32_plus() {
        let data = build_minimal_pe();
        let pe = Pe32Plus::parse(&data).expect("valid PE32+ should parse");
        assert_eq!(pe.coff_header().machine, IMAGE_FILE_MACHINE_AMD64);
        assert_eq!(pe.optional_header().image_base, 0x1_4000_0000);
        assert_eq!(pe.optional_header().address_of_entry_point, 0x1000);

        let imp = pe.data_directory(IMAGE_DIRECTORY_ENTRY_IMPORT).unwrap();
        assert_eq!(imp.virtual_address, 0x2000);
        assert_eq!(imp.size, 0x20);

        let secs: Vec<_> = pe.sections().collect();
        assert_eq!(secs.len(), 2);
        assert_eq!(secs[0].name_str(), Some(".text"));
        assert!(secs[0].is_executable());
        assert!(secs[0].is_readable());
        assert!(!secs[0].is_writable());
        assert_eq!(pe.section_data(&secs[0]).unwrap()[0], 0xC3);
        assert_eq!(secs[1].name_str(), Some(".rdata"));
        assert_eq!(pe.rva_to_offset(0x1000), Some(0x200));
        assert_eq!(pe.rva_to_offset(0x2004), Some(0x224));
        assert_eq!(pe.rva_to_offset(0x9000), None);
    }

    #[test]
    fn reject_bad_dos_and_pe_signatures() {
        assert_eq!(Pe32Plus::parse(&[0u8; 16]), Err(PeError::TooShort));

        let mut data = build_minimal_pe();
        data[0] = b'X';
        assert_eq!(Pe32Plus::parse(&data), Err(PeError::InvalidDosMagic));

        let mut data = build_minimal_pe();
        data[0x80] = b'X';
        assert_eq!(Pe32Plus::parse(&data), Err(PeError::InvalidPeSignature));
    }

    #[test]
    fn reject_wrong_machine_and_magic() {
        let mut data = build_minimal_pe();
        put_u16(&mut data, 0x84, 0x014C); // x86
        assert_eq!(
            Pe32Plus::parse(&data),
            Err(PeError::UnsupportedMachine(0x014C))
        );

        let mut data = build_minimal_pe();
        put_u16(&mut data, 0x98, 0x010B); // PE32
        assert_eq!(
            Pe32Plus::parse(&data),
            Err(PeError::UnsupportedOptionalMagic(0x010B))
        );
    }

    #[test]
    fn reject_out_of_bounds_section() {
        let mut data = build_minimal_pe();
        put_u32(&mut data, 0x188 + 16, 0x10_0000); // size_of_raw_data
        assert_eq!(Pe32Plus::parse(&data), Err(PeError::OutOfBounds));
    }
}
