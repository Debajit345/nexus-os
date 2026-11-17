#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Unified ELF64, PE32+, and Mach-O 64-bit binary inspection and
//! multi-personality syscall translation for Nexus OS.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use nexus_abi::{Personality, SyscallNumber};
use nexus_elf::{Elf64, ElfError, ELF_MAGIC};
use nexus_macho::{MachO64, MachOError, MH_MAGIC_64};
use nexus_pe::{Pe32Plus, PeError, DOS_MAGIC};

/// Supported executable binary formats for x86_64 user-space personalities.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryFormat {
    Elf64,
    Pe32Plus,
    MachO64,
}

impl BinaryFormat {
    /// Returns the Nexus OS execution personality associated with this binary format.
    #[inline]
    pub const fn personality(self) -> Personality {
        match self {
            Self::Elf64 => Personality::Posix,
            Self::Pe32Plus => Personality::Nt,
            Self::MachO64 => Personality::Darwin,
        }
    }
}

/// Normalized summary of a loadable segment or section in a validated binary image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSegment {
    pub name: Option<String>,
    pub virtual_address: u64,
    pub virtual_size: u64,
    pub file_offset: u64,
    pub file_size: u64,
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
}

/// Unified metadata extracted from a validated ELF64, PE32+, or Mach-O 64-bit binary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedImage {
    pub format: BinaryFormat,
    pub personality: Personality,
    pub entry_point: u64,
    pub image_base: u64,
    pub segments: Vec<ImageSegment>,
    pub interpreter: Option<String>,
}

/// Errors returned when detecting or validating a binary image.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompatError {
    TooShort,
    UnknownFormat,
    Elf(ElfError),
    Pe(PeError),
    MachO(MachOError),
}

impl From<ElfError> for CompatError {
    fn from(err: ElfError) -> Self {
        Self::Elf(err)
    }
}

impl From<PeError> for CompatError {
    fn from(err: PeError) -> Self {
        Self::Pe(err)
    }
}

impl From<MachOError> for CompatError {
    fn from(err: MachOError) -> Self {
        Self::MachO(err)
    }
}

/// Auto-detects the binary format from the leading magic bytes.
pub fn detect_format(bytes: &[u8]) -> Result<BinaryFormat, CompatError> {
    if bytes.len() < 4 {
        return Err(CompatError::TooShort);
    }

    if bytes[0..4] == ELF_MAGIC {
        return Ok(BinaryFormat::Elf64);
    }

    let dos_magic = u16::from_le_bytes([bytes[0], bytes[1]]);
    if dos_magic == DOS_MAGIC {
        return Ok(BinaryFormat::Pe32Plus);
    }

    let macho_magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if macho_magic == MH_MAGIC_64 {
        return Ok(BinaryFormat::MachO64);
    }

    Err(CompatError::UnknownFormat)
}

/// Auto-detects and validates an ELF64, PE32+, or Mach-O 64-bit binary,
/// returning a unified [`LoadedImage`] summary.
pub fn inspect_binary(bytes: &[u8]) -> Result<LoadedImage, CompatError> {
    match detect_format(bytes)? {
        BinaryFormat::Elf64 => inspect_elf64(bytes),
        BinaryFormat::Pe32Plus => inspect_pe32_plus(bytes),
        BinaryFormat::MachO64 => inspect_macho64(bytes),
    }
}

fn inspect_elf64(bytes: &[u8]) -> Result<LoadedImage, CompatError> {
    let elf = Elf64::parse(bytes)?;
    let mut segments = Vec::new();
    let mut image_base: Option<u64> = None;

    for ph in elf.load_segments() {
        image_base = Some(match image_base {
            Some(base) => base.min(ph.p_vaddr),
            None => ph.p_vaddr,
        });
        segments.push(ImageSegment {
            name: None,
            virtual_address: ph.p_vaddr,
            virtual_size: ph.p_memsz,
            file_offset: ph.p_offset,
            file_size: ph.p_filesz,
            readable: ph.is_readable(),
            writable: ph.is_writable(),
            executable: ph.is_executable(),
        });
    }

    let interpreter = elf.interp()?.map(String::from);

    Ok(LoadedImage {
        format: BinaryFormat::Elf64,
        personality: Personality::Posix,
        entry_point: elf.header().entry,
        image_base: image_base.unwrap_or(0),
        segments,
        interpreter,
    })
}

fn inspect_pe32_plus(bytes: &[u8]) -> Result<LoadedImage, CompatError> {
    let pe = Pe32Plus::parse(bytes)?;
    let opt = pe.optional_header();
    let image_base = opt.image_base;
    let entry_point = image_base
        .checked_add(opt.address_of_entry_point as u64)
        .ok_or(CompatError::Pe(PeError::OutOfBounds))?;

    let mut segments = Vec::new();
    for sec in pe.sections() {
        let virtual_address = image_base
            .checked_add(sec.virtual_address as u64)
            .ok_or(CompatError::Pe(PeError::OutOfBounds))?;
        segments.push(ImageSegment {
            name: sec.name_str().map(String::from),
            virtual_address,
            virtual_size: sec.virtual_size as u64,
            file_offset: sec.pointer_to_raw_data as u64,
            file_size: sec.size_of_raw_data as u64,
            readable: sec.is_readable(),
            writable: sec.is_writable(),
            executable: sec.is_executable(),
        });
    }

    Ok(LoadedImage {
        format: BinaryFormat::Pe32Plus,
        personality: Personality::Nt,
        entry_point,
        image_base,
        segments,
        interpreter: None,
    })
}

fn inspect_macho64(bytes: &[u8]) -> Result<LoadedImage, CompatError> {
    let macho = MachO64::parse(bytes)?;
    let mut segments = Vec::new();
    let mut text_base: Option<u64> = None;
    let mut lowest_base: Option<u64> = None;

    for seg in macho.segments() {
        let name = seg.name_str().map(String::from);
        if name.as_deref() == Some("__TEXT") {
            text_base = Some(seg.vmaddr);
        }
        if seg.vmsize > 0 {
            lowest_base = Some(match lowest_base {
                Some(base) => base.min(seg.vmaddr),
                None => seg.vmaddr,
            });
        }
        segments.push(ImageSegment {
            name,
            virtual_address: seg.vmaddr,
            virtual_size: seg.vmsize,
            file_offset: seg.fileoff,
            file_size: seg.filesize,
            readable: seg.is_readable(),
            writable: seg.is_writable(),
            executable: seg.is_executable(),
        });
    }

    let image_base = text_base.or(lowest_base).unwrap_or(0);
    let entry_point = match macho.main_entry() {
        Some(ep) => image_base
            .checked_add(ep.entryoff)
            .ok_or(CompatError::MachO(MachOError::OutOfBounds))?,
        None => 0,
    };
    let interpreter = macho.dylinker().map(String::from);

    Ok(LoadedImage {
        format: BinaryFormat::MachO64,
        personality: Personality::Darwin,
        entry_point,
        image_base,
        segments,
        interpreter,
    })
}

/// Translates a Linux x86_64 syscall number into a Nexus [`SyscallNumber`].
pub const fn translate_linux_syscall(nr: u64) -> Option<SyscallNumber> {
    match nr {
        0 => Some(SyscallNumber::PosixRead),
        1 => Some(SyscallNumber::PosixWrite),
        3 => Some(SyscallNumber::PosixClose),
        9 => Some(SyscallNumber::PosixMmap),
        24 => Some(SyscallNumber::ThreadYield),
        57 => Some(SyscallNumber::PosixFork),
        59 => Some(SyscallNumber::PosixExecve),
        60 => Some(SyscallNumber::ThreadExit),
        231 => Some(SyscallNumber::ProcessExit),
        257 => Some(SyscallNumber::PosixOpenAt),
        _ => None,
    }
}

/// Translates a Windows NT x86_64 syscall number into a Nexus [`SyscallNumber`].
pub const fn translate_nt_syscall(nr: u64) -> Option<SyscallNumber> {
    match nr {
        // Standard Windows x64 SSDT indices
        0x0006 => Some(SyscallNumber::NtReadFile),
        0x0008 => Some(SyscallNumber::NtWriteFile),
        0x000F => Some(SyscallNumber::NtClose),
        0x0018 => Some(SyscallNumber::NtAllocateVirtualMemory),
        0x004D => Some(SyscallNumber::NtCreateProcessEx),
        0x0055 => Some(SyscallNumber::NtCreateFile),
        // Direct Nexus NT bridge indices (low 12 bits of SyscallNumber::Nt*)
        0x0001 => Some(SyscallNumber::NtCreateFile),
        0x0002 => Some(SyscallNumber::NtReadFile),
        0x0003 => Some(SyscallNumber::NtWriteFile),
        0x0004 => Some(SyscallNumber::NtClose),
        _ => None,
    }
}

/// Translates a Darwin / XNU x86_64 syscall number (BSD or Mach trap) into a Nexus [`SyscallNumber`].
pub const fn translate_darwin_syscall(nr: u64) -> Option<SyscallNumber> {
    const SYSCALL_CLASS_MASK: u64 = 0xFF00_0000;
    const SYSCALL_CLASS_MACH: u64 = 0x0100_0000;
    const SYSCALL_CLASS_BSD: u64 = 0x0200_0000;
    const SYSCALL_NUM_MASK: u64 = 0x00FF_FFFF;

    let class = nr & SYSCALL_CLASS_MASK;
    let num = nr & SYSCALL_NUM_MASK;

    match class {
        SYSCALL_CLASS_MACH => match num {
            0x1C => Some(SyscallNumber::DarwinTaskSelf),
            0x1F => Some(SyscallNumber::DarwinMachMsg),
            _ => None,
        },
        SYSCALL_CLASS_BSD | 0 => match num {
            1 => Some(SyscallNumber::ProcessExit),
            3 => Some(SyscallNumber::DarwinBsdRead),
            4 => Some(SyscallNumber::DarwinBsdWrite),
            5 => Some(SyscallNumber::DarwinBsdOpen),
            6 => Some(SyscallNumber::DarwinBsdClose),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_elf::{
        ELF64_EHDR_SIZE, ELF64_PHDR_SIZE, ELFCLASS64, ELFDATA2LSB, EM_X86_64, ET_DYN, EV_CURRENT,
        PF_R, PF_X, PT_INTERP, PT_LOAD,
    };
    use nexus_macho::{
        CPU_SUBTYPE_X86_64_ALL, CPU_TYPE_X86_64, DYLINKER_COMMAND_MIN_SIZE,
        ENTRY_POINT_COMMAND_SIZE, LC_LOAD_DYLINKER, LC_MAIN, LC_SEGMENT_64, MACH_HEADER_64_SIZE,
        MH_EXECUTE, SEGMENT_COMMAND_64_SIZE, VM_PROT_EXECUTE, VM_PROT_READ,
    };
    use nexus_pe::{
        COFF_HEADER_SIZE, DOS_HEADER_SIZE, IMAGE_FILE_MACHINE_AMD64, IMAGE_SCN_CNT_CODE,
        IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ, PE32_PLUS_MAGIC, PE32_PLUS_OPT_HEADER_MIN_SIZE,
        PE_SIGNATURE, SECTION_HEADER_SIZE,
    };

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
        let mut buf = vec![0u8; 0x140];
        buf[0..4].copy_from_slice(&ELF_MAGIC);
        buf[4] = ELFCLASS64;
        buf[5] = ELFDATA2LSB;
        buf[6] = EV_CURRENT;

        put_u16(&mut buf, 16, ET_DYN);
        put_u16(&mut buf, 18, EM_X86_64);
        put_u32(&mut buf, 20, EV_CURRENT as u32);
        put_u64(&mut buf, 24, 0x401000);
        put_u64(&mut buf, 32, 0x40);
        put_u16(&mut buf, 52, ELF64_EHDR_SIZE);
        put_u16(&mut buf, 54, ELF64_PHDR_SIZE);
        put_u16(&mut buf, 56, 2);

        // PHDR 0: PT_LOAD
        let ph0 = 0x40;
        put_u32(&mut buf, ph0, PT_LOAD);
        put_u32(&mut buf, ph0 + 4, PF_R | PF_X);
        put_u64(&mut buf, ph0 + 8, 0);
        put_u64(&mut buf, ph0 + 16, 0x400000);
        put_u64(&mut buf, ph0 + 24, 0x400000);
        put_u64(&mut buf, ph0 + 32, 0x140);
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

        buf
    }

    fn build_minimal_pe() -> Vec<u8> {
        let pe_off = DOS_HEADER_SIZE;
        let coff_off = pe_off + 4;
        let opt_off = coff_off + COFF_HEADER_SIZE;
        let opt_size = PE32_PLUS_OPT_HEADER_MIN_SIZE;
        let sec_off = opt_off + opt_size;
        let raw_off = 0x200usize;
        let mut buf = vec![0u8; raw_off + 0x200];

        put_u16(&mut buf, 0, DOS_MAGIC);
        put_u32(&mut buf, 60, pe_off as u32);
        buf[pe_off..pe_off + 4].copy_from_slice(&PE_SIGNATURE);

        put_u16(&mut buf, coff_off, IMAGE_FILE_MACHINE_AMD64);
        put_u16(&mut buf, coff_off + 2, 1);
        put_u16(&mut buf, coff_off + 16, opt_size as u16);

        put_u16(&mut buf, opt_off, PE32_PLUS_MAGIC);
        put_u32(&mut buf, opt_off + 16, 0x1000); // AddressOfEntryPoint
        put_u64(&mut buf, opt_off + 24, 0x1_4000_0000); // ImageBase
        put_u32(&mut buf, opt_off + 32, 0x1000); // SectionAlignment
        put_u32(&mut buf, opt_off + 36, 0x200); // FileAlignment
        put_u32(&mut buf, opt_off + 56, 0x2000); // SizeOfImage
        put_u32(&mut buf, opt_off + 60, 0x200); // SizeOfHeaders

        // Section 0: .text
        buf[sec_off..sec_off + 5].copy_from_slice(b".text");
        put_u32(&mut buf, sec_off + 8, 0x100); // VirtualSize
        put_u32(&mut buf, sec_off + 12, 0x1000); // VirtualAddress
        put_u32(&mut buf, sec_off + 16, 0x200); // SizeOfRawData
        put_u32(&mut buf, sec_off + 20, raw_off as u32); // PointerToRawData
        put_u32(
            &mut buf,
            sec_off + 36,
            IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_EXECUTE,
        );
        let _ = SECTION_HEADER_SIZE;

        buf
    }

    fn build_minimal_macho() -> Vec<u8> {
        let dylinker_path = b"/usr/lib/dyld\0";
        let dylinker_cmdsize = (DYLINKER_COMMAND_MIN_SIZE + dylinker_path.len() + 7) & !7;
        let sizeofcmds = SEGMENT_COMMAND_64_SIZE + ENTRY_POINT_COMMAND_SIZE + dylinker_cmdsize;
        let total_size = MACH_HEADER_64_SIZE + sizeofcmds + 0x100;
        let mut buf = vec![0u8; total_size];

        put_u32(&mut buf, 0, MH_MAGIC_64);
        put_u32(&mut buf, 4, CPU_TYPE_X86_64);
        put_u32(&mut buf, 8, CPU_SUBTYPE_X86_64_ALL);
        put_u32(&mut buf, 12, MH_EXECUTE);
        put_u32(&mut buf, 16, 3); // ncmds
        put_u32(&mut buf, 20, sizeofcmds as u32);

        // LC_SEGMENT_64 (__TEXT)
        let mut off = MACH_HEADER_64_SIZE;
        put_u32(&mut buf, off, LC_SEGMENT_64);
        put_u32(&mut buf, off + 4, SEGMENT_COMMAND_64_SIZE as u32);
        buf[off + 8..off + 14].copy_from_slice(b"__TEXT");
        put_u64(&mut buf, off + 24, 0x1_0000_0000); // vmaddr
        put_u64(&mut buf, off + 32, 0x1000); // vmsize
        put_u64(&mut buf, off + 40, 0); // fileoff
        put_u64(&mut buf, off + 48, total_size as u64); // filesize
        put_u32(&mut buf, off + 56, VM_PROT_READ | VM_PROT_EXECUTE);
        put_u32(&mut buf, off + 60, VM_PROT_READ | VM_PROT_EXECUTE);

        // LC_MAIN
        off += SEGMENT_COMMAND_64_SIZE;
        put_u32(&mut buf, off, LC_MAIN);
        put_u32(&mut buf, off + 4, ENTRY_POINT_COMMAND_SIZE as u32);
        put_u64(&mut buf, off + 8, 0x200); // entryoff

        // LC_LOAD_DYLINKER
        off += ENTRY_POINT_COMMAND_SIZE;
        put_u32(&mut buf, off, LC_LOAD_DYLINKER);
        put_u32(&mut buf, off + 4, dylinker_cmdsize as u32);
        put_u32(&mut buf, off + 8, DYLINKER_COMMAND_MIN_SIZE as u32);
        buf[off + DYLINKER_COMMAND_MIN_SIZE..off + DYLINKER_COMMAND_MIN_SIZE + dylinker_path.len()]
            .copy_from_slice(dylinker_path);

        buf
    }

    #[test]
    fn inspect_elf64_image() {
        let bytes = build_minimal_elf();
        let img = inspect_binary(&bytes).expect("ELF64 should inspect");
        assert_eq!(img.format, BinaryFormat::Elf64);
        assert_eq!(img.personality, Personality::Posix);
        assert_eq!(img.entry_point, 0x401000);
        assert_eq!(img.image_base, 0x400000);
        assert_eq!(img.segments.len(), 1);
        assert!(img.segments[0].readable);
        assert!(img.segments[0].executable);
        assert!(!img.segments[0].writable);
        assert_eq!(
            img.interpreter.as_deref(),
            Some("/lib64/ld-linux-x86-64.so.2")
        );
    }

    #[test]
    fn inspect_pe32_plus_image() {
        let bytes = build_minimal_pe();
        let img = inspect_binary(&bytes).expect("PE32+ should inspect");
        assert_eq!(img.format, BinaryFormat::Pe32Plus);
        assert_eq!(img.personality, Personality::Nt);
        assert_eq!(img.image_base, 0x1_4000_0000);
        assert_eq!(img.entry_point, 0x1_4000_1000);
        assert_eq!(img.segments.len(), 1);
        assert_eq!(img.segments[0].name.as_deref(), Some(".text"));
        assert!(img.segments[0].readable);
        assert!(img.segments[0].executable);
        assert!(!img.segments[0].writable);
        assert_eq!(img.interpreter, None);
    }

    #[test]
    fn inspect_macho64_image() {
        let bytes = build_minimal_macho();
        let img = inspect_binary(&bytes).expect("Mach-O 64 should inspect");
        assert_eq!(img.format, BinaryFormat::MachO64);
        assert_eq!(img.personality, Personality::Darwin);
        assert_eq!(img.image_base, 0x1_0000_0000);
        assert_eq!(img.entry_point, 0x1_0000_0200);
        assert_eq!(img.segments.len(), 1);
        assert_eq!(img.segments[0].name.as_deref(), Some("__TEXT"));
        assert!(img.segments[0].readable);
        assert!(img.segments[0].executable);
        assert!(!img.segments[0].writable);
        assert_eq!(img.interpreter.as_deref(), Some("/usr/lib/dyld"));
    }

    #[test]
    fn reject_unknown_and_short_binaries() {
        assert_eq!(inspect_binary(&[0x7F, b'E']), Err(CompatError::TooShort));
        assert_eq!(
            inspect_binary(&[0x00, 0x01, 0x02, 0x03]),
            Err(CompatError::UnknownFormat)
        );
    }

    #[test]
    fn linux_syscall_translation() {
        assert_eq!(translate_linux_syscall(0), Some(SyscallNumber::PosixRead));
        assert_eq!(translate_linux_syscall(1), Some(SyscallNumber::PosixWrite));
        assert_eq!(translate_linux_syscall(3), Some(SyscallNumber::PosixClose));
        assert_eq!(translate_linux_syscall(9), Some(SyscallNumber::PosixMmap));
        assert_eq!(translate_linux_syscall(57), Some(SyscallNumber::PosixFork));
        assert_eq!(
            translate_linux_syscall(59),
            Some(SyscallNumber::PosixExecve)
        );
        assert_eq!(translate_linux_syscall(60), Some(SyscallNumber::ThreadExit));
        assert_eq!(
            translate_linux_syscall(231),
            Some(SyscallNumber::ProcessExit)
        );
        assert_eq!(
            translate_linux_syscall(257),
            Some(SyscallNumber::PosixOpenAt)
        );
        assert_eq!(translate_linux_syscall(39), None);
        assert_eq!(translate_linux_syscall(63), None);
    }

    #[test]
    fn nt_syscall_translation() {
        assert_eq!(
            translate_nt_syscall(0x0055),
            Some(SyscallNumber::NtCreateFile)
        );
        assert_eq!(
            translate_nt_syscall(0x0006),
            Some(SyscallNumber::NtReadFile)
        );
        assert_eq!(
            translate_nt_syscall(0x0008),
            Some(SyscallNumber::NtWriteFile)
        );
        assert_eq!(translate_nt_syscall(0x000F), Some(SyscallNumber::NtClose));
        assert_eq!(
            translate_nt_syscall(0x0018),
            Some(SyscallNumber::NtAllocateVirtualMemory)
        );
        assert_eq!(
            translate_nt_syscall(0x004D),
            Some(SyscallNumber::NtCreateProcessEx)
        );
        assert_eq!(translate_nt_syscall(0xFFFF), None);
    }

    #[test]
    fn darwin_syscall_translation() {
        assert_eq!(
            translate_darwin_syscall(0x0200_0003),
            Some(SyscallNumber::DarwinBsdRead)
        );
        assert_eq!(
            translate_darwin_syscall(0x0200_0004),
            Some(SyscallNumber::DarwinBsdWrite)
        );
        assert_eq!(
            translate_darwin_syscall(0x0200_0005),
            Some(SyscallNumber::DarwinBsdOpen)
        );
        assert_eq!(
            translate_darwin_syscall(0x0200_0006),
            Some(SyscallNumber::DarwinBsdClose)
        );
        assert_eq!(
            translate_darwin_syscall(0x0200_0001),
            Some(SyscallNumber::ProcessExit)
        );
        assert_eq!(
            translate_darwin_syscall(0x0100_001F),
            Some(SyscallNumber::DarwinMachMsg)
        );
        assert_eq!(
            translate_darwin_syscall(0x0100_001C),
            Some(SyscallNumber::DarwinTaskSelf)
        );
        assert_eq!(
            translate_darwin_syscall(3),
            Some(SyscallNumber::DarwinBsdRead)
        );
        assert_eq!(translate_darwin_syscall(0x0200_9999), None);
    }
}
