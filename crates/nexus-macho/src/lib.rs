#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Clean-room `no_std` Mach-O 64-bit parser and bounds-checked validator for x86_64 macOS binaries.

use core::ops::Range;

pub const MH_MAGIC_64: u32 = 0xFEED_FACF;
pub const CPU_TYPE_X86_64: u32 = 0x0100_0007;
pub const CPU_SUBTYPE_X86_64_ALL: u32 = 3;

pub const MH_EXECUTE: u32 = 2;
pub const MH_DYLIB: u32 = 6;
pub const MH_DYLINKER: u32 = 7;
pub const MH_BUNDLE: u32 = 8;

pub const MACH_HEADER_64_SIZE: usize = 32;
pub const LOAD_COMMAND_HEADER_SIZE: usize = 8;
pub const SEGMENT_COMMAND_64_SIZE: usize = 72;
pub const SECTION_64_SIZE: usize = 80;
pub const ENTRY_POINT_COMMAND_SIZE: usize = 24;
pub const DYLINKER_COMMAND_MIN_SIZE: usize = 12;

pub const LC_REQ_DYLD: u32 = 0x8000_0000;
pub const LC_SYMTAB: u32 = 0x02;
pub const LC_DYSYMTAB: u32 = 0x0B;
pub const LC_LOAD_DYLIB: u32 = 0x0C;
pub const LC_LOAD_DYLINKER: u32 = 0x0E;
pub const LC_SEGMENT_64: u32 = 0x19;
pub const LC_UUID: u32 = 0x1B;
pub const LC_MAIN: u32 = 0x28 | LC_REQ_DYLD;

pub const VM_PROT_READ: u32 = 1;
pub const VM_PROT_WRITE: u32 = 2;
pub const VM_PROT_EXECUTE: u32 = 4;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MachOError {
    TooShort,
    InvalidMagic(u32),
    UnsupportedCpuType(u32),
    InvalidLoadCommand,
    InvalidSegment,
    InvalidDylinker,
    OutOfBounds,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MachHeader64 {
    pub magic: u32,
    pub cputype: u32,
    pub cpusubtype: u32,
    pub filetype: u32,
    pub ncmds: u32,
    pub sizeofcmds: u32,
    pub flags: u32,
    pub reserved: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SegmentCommand64 {
    pub segname: [u8; 16],
    pub vmaddr: u64,
    pub vmsize: u64,
    pub fileoff: u64,
    pub filesize: u64,
    pub maxprot: u32,
    pub initprot: u32,
    pub nsects: u32,
    pub flags: u32,
}

impl SegmentCommand64 {
    pub fn name_str(&self) -> Option<&str> {
        let len = self.segname.iter().position(|&b| b == 0).unwrap_or(16);
        core::str::from_utf8(&self.segname[..len]).ok()
    }

    #[inline]
    pub const fn is_readable(&self) -> bool {
        (self.initprot & VM_PROT_READ) != 0
    }

    #[inline]
    pub const fn is_writable(&self) -> bool {
        (self.initprot & VM_PROT_WRITE) != 0
    }

    #[inline]
    pub const fn is_executable(&self) -> bool {
        (self.initprot & VM_PROT_EXECUTE) != 0
    }

    pub fn file_range(&self) -> Result<Range<usize>, MachOError> {
        checked_range(self.fileoff, self.filesize)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Section64 {
    pub sectname: [u8; 16],
    pub segname: [u8; 16],
    pub addr: u64,
    pub size: u64,
    pub offset: u32,
    pub align: u32,
    pub reloff: u32,
    pub nreloc: u32,
    pub flags: u32,
}

impl Section64 {
    pub fn sect_name_str(&self) -> Option<&str> {
        let len = self.sectname.iter().position(|&b| b == 0).unwrap_or(16);
        core::str::from_utf8(&self.sectname[..len]).ok()
    }

    pub fn seg_name_str(&self) -> Option<&str> {
        let len = self.segname.iter().position(|&b| b == 0).unwrap_or(16);
        core::str::from_utf8(&self.segname[..len]).ok()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EntryPointCommand {
    pub entryoff: u64,
    pub stacksize: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoadCommand<'a> {
    Segment64 {
        segment: SegmentCommand64,
        sections_data: &'a [u8],
    },
    Main(EntryPointCommand),
    LoadDylinker(&'a str),
    Other {
        cmd: u32,
        data: &'a [u8],
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MachO64<'a> {
    data: &'a [u8],
    header: MachHeader64,
}

impl<'a> MachO64<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, MachOError> {
        if data.len() < MACH_HEADER_64_SIZE {
            return Err(MachOError::TooShort);
        }

        let magic = read_u32(data, 0)?;
        if magic != MH_MAGIC_64 {
            return Err(MachOError::InvalidMagic(magic));
        }

        let cputype = read_u32(data, 4)?;
        if cputype != CPU_TYPE_X86_64 {
            return Err(MachOError::UnsupportedCpuType(cputype));
        }

        let header = MachHeader64 {
            magic,
            cputype,
            cpusubtype: read_u32(data, 8)?,
            filetype: read_u32(data, 12)?,
            ncmds: read_u32(data, 16)?,
            sizeofcmds: read_u32(data, 20)?,
            flags: read_u32(data, 24)?,
            reserved: read_u32(data, 28)?,
        };

        let cmds_end = MACH_HEADER_64_SIZE
            .checked_add(header.sizeofcmds as usize)
            .ok_or(MachOError::OutOfBounds)?;
        if cmds_end > data.len() {
            return Err(MachOError::OutOfBounds);
        }

        let macho = Self { data, header };
        macho.validate_load_commands()?;
        Ok(macho)
    }

    #[inline]
    pub const fn header(&self) -> &MachHeader64 {
        &self.header
    }

    pub fn load_commands(&self) -> LoadCommandIter<'a> {
        let cmds_end = MACH_HEADER_64_SIZE + self.header.sizeofcmds as usize;
        LoadCommandIter {
            data: &self.data[MACH_HEADER_64_SIZE..cmds_end],
            offset: 0,
            remaining: self.header.ncmds,
        }
    }

    pub fn segments(&self) -> impl Iterator<Item = SegmentCommand64> + 'a {
        self.load_commands().filter_map(|lc| match lc {
            LoadCommand::Segment64 { segment, .. } => Some(segment),
            _ => None,
        })
    }

    pub fn main_entry(&self) -> Option<EntryPointCommand> {
        for lc in self.load_commands() {
            if let LoadCommand::Main(ep) = lc {
                return Some(ep);
            }
        }
        None
    }

    pub fn dylinker(&self) -> Option<&'a str> {
        for lc in self.load_commands() {
            if let LoadCommand::LoadDylinker(path) = lc {
                return Some(path);
            }
        }
        None
    }

    pub fn segment_data(&self, seg: &SegmentCommand64) -> Result<&'a [u8], MachOError> {
        let range = seg.file_range()?;
        self.data.get(range).ok_or(MachOError::OutOfBounds)
    }

    pub fn parse_sections(
        sections_data: &'a [u8],
        nsects: u32,
    ) -> Result<Section64Iter<'a>, MachOError> {
        let expected = (nsects as usize)
            .checked_mul(SECTION_64_SIZE)
            .ok_or(MachOError::OutOfBounds)?;
        if sections_data.len() < expected {
            return Err(MachOError::OutOfBounds);
        }
        Ok(Section64Iter {
            data: &sections_data[..expected],
            offset: 0,
        })
    }

    fn validate_load_commands(&self) -> Result<(), MachOError> {
        let cmds_end = MACH_HEADER_64_SIZE + self.header.sizeofcmds as usize;
        let cmds_slice = &self.data[MACH_HEADER_64_SIZE..cmds_end];
        let mut offset = 0usize;

        for _ in 0..self.header.ncmds {
            if offset + LOAD_COMMAND_HEADER_SIZE > cmds_slice.len() {
                return Err(MachOError::InvalidLoadCommand);
            }
            let cmd = read_u32(cmds_slice, offset)?;
            let cmdsize = read_u32(cmds_slice, offset + 4)? as usize;
            if cmdsize < LOAD_COMMAND_HEADER_SIZE || !cmdsize.is_multiple_of(8) {
                return Err(MachOError::InvalidLoadCommand);
            }
            let next_offset = offset
                .checked_add(cmdsize)
                .ok_or(MachOError::InvalidLoadCommand)?;
            if next_offset > cmds_slice.len() {
                return Err(MachOError::InvalidLoadCommand);
            }

            let lc_bytes = &cmds_slice[offset..next_offset];
            match cmd {
                LC_SEGMENT_64 => {
                    if cmdsize < SEGMENT_COMMAND_64_SIZE {
                        return Err(MachOError::InvalidSegment);
                    }
                    let seg = parse_segment_64(lc_bytes)?;
                    let sects_bytes = (seg.nsects as usize)
                        .checked_mul(SECTION_64_SIZE)
                        .ok_or(MachOError::InvalidSegment)?;
                    let min_size = SEGMENT_COMMAND_64_SIZE
                        .checked_add(sects_bytes)
                        .ok_or(MachOError::InvalidSegment)?;
                    if cmdsize < min_size {
                        return Err(MachOError::InvalidSegment);
                    }
                    if seg.filesize > seg.vmsize && seg.vmsize > 0 {
                        return Err(MachOError::InvalidSegment);
                    }
                    let range = seg.file_range()?;
                    if range.end > self.data.len() {
                        return Err(MachOError::OutOfBounds);
                    }
                }
                LC_MAIN => {
                    if cmdsize != ENTRY_POINT_COMMAND_SIZE {
                        return Err(MachOError::InvalidLoadCommand);
                    }
                }
                LC_LOAD_DYLINKER => {
                    if cmdsize < DYLINKER_COMMAND_MIN_SIZE {
                        return Err(MachOError::InvalidDylinker);
                    }
                    let str_off = read_u32(lc_bytes, 8)? as usize;
                    if str_off >= cmdsize {
                        return Err(MachOError::InvalidDylinker);
                    }
                    let tail = &lc_bytes[str_off..];
                    let nul_pos = tail
                        .iter()
                        .position(|&b| b == 0)
                        .ok_or(MachOError::InvalidDylinker)?;
                    if nul_pos == 0 {
                        return Err(MachOError::InvalidDylinker);
                    }
                    core::str::from_utf8(&tail[..nul_pos])
                        .map_err(|_| MachOError::InvalidDylinker)?;
                }
                _ => {}
            }

            offset = next_offset;
        }

        if offset != cmds_slice.len() {
            return Err(MachOError::InvalidLoadCommand);
        }

        Ok(())
    }
}

pub struct LoadCommandIter<'a> {
    data: &'a [u8],
    offset: usize,
    remaining: u32,
}

impl<'a> Iterator for LoadCommandIter<'a> {
    type Item = LoadCommand<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 || self.offset + LOAD_COMMAND_HEADER_SIZE > self.data.len() {
            return None;
        }
        let cmd = read_u32(self.data, self.offset).ok()?;
        let cmdsize = read_u32(self.data, self.offset + 4).ok()? as usize;
        let end = self.offset.checked_add(cmdsize)?;
        let lc_bytes = self.data.get(self.offset..end)?;
        self.offset = end;
        self.remaining -= 1;

        match cmd {
            LC_SEGMENT_64 => {
                let segment = parse_segment_64(lc_bytes).ok()?;
                let sections_data = lc_bytes.get(SEGMENT_COMMAND_64_SIZE..)?;
                Some(LoadCommand::Segment64 {
                    segment,
                    sections_data,
                })
            }
            LC_MAIN => {
                let entryoff = read_u64(lc_bytes, 8).ok()?;
                let stacksize = read_u64(lc_bytes, 16).ok()?;
                Some(LoadCommand::Main(EntryPointCommand {
                    entryoff,
                    stacksize,
                }))
            }
            LC_LOAD_DYLINKER => {
                let str_off = read_u32(lc_bytes, 8).ok()? as usize;
                let tail = lc_bytes.get(str_off..)?;
                let nul_pos = tail.iter().position(|&b| b == 0)?;
                let path = core::str::from_utf8(&tail[..nul_pos]).ok()?;
                Some(LoadCommand::LoadDylinker(path))
            }
            _ => Some(LoadCommand::Other {
                cmd,
                data: lc_bytes,
            }),
        }
    }
}

pub struct Section64Iter<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Iterator for Section64Iter<'a> {
    type Item = Section64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset + SECTION_64_SIZE > self.data.len() {
            return None;
        }
        let off = self.offset;
        let mut sectname = [0u8; 16];
        let mut segname = [0u8; 16];
        sectname.copy_from_slice(self.data.get(off..off + 16)?);
        segname.copy_from_slice(self.data.get(off + 16..off + 32)?);
        let sec = Section64 {
            sectname,
            segname,
            addr: read_u64(self.data, off + 32).ok()?,
            size: read_u64(self.data, off + 40).ok()?,
            offset: read_u32(self.data, off + 48).ok()?,
            align: read_u32(self.data, off + 52).ok()?,
            reloff: read_u32(self.data, off + 56).ok()?,
            nreloc: read_u32(self.data, off + 60).ok()?,
            flags: read_u32(self.data, off + 64).ok()?,
        };
        self.offset += SECTION_64_SIZE;
        Some(sec)
    }
}

fn parse_segment_64(lc_bytes: &[u8]) -> Result<SegmentCommand64, MachOError> {
    if lc_bytes.len() < SEGMENT_COMMAND_64_SIZE {
        return Err(MachOError::InvalidSegment);
    }
    let mut segname = [0u8; 16];
    segname.copy_from_slice(&lc_bytes[8..24]);
    Ok(SegmentCommand64 {
        segname,
        vmaddr: read_u64(lc_bytes, 24)?,
        vmsize: read_u64(lc_bytes, 32)?,
        fileoff: read_u64(lc_bytes, 40)?,
        filesize: read_u64(lc_bytes, 48)?,
        maxprot: read_u32(lc_bytes, 56)?,
        initprot: read_u32(lc_bytes, 60)?,
        nsects: read_u32(lc_bytes, 64)?,
        flags: read_u32(lc_bytes, 68)?,
    })
}

fn checked_range(offset: u64, size: u64) -> Result<Range<usize>, MachOError> {
    let start = usize::try_from(offset).map_err(|_| MachOError::OutOfBounds)?;
    let len = usize::try_from(size).map_err(|_| MachOError::OutOfBounds)?;
    let end = start.checked_add(len).ok_or(MachOError::OutOfBounds)?;
    Ok(start..end)
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, MachOError> {
    let bytes: [u8; 4] = data
        .get(offset..offset + 4)
        .ok_or(MachOError::OutOfBounds)?
        .try_into()
        .map_err(|_| MachOError::OutOfBounds)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64, MachOError> {
    let bytes: [u8; 8] = data
        .get(offset..offset + 8)
        .ok_or(MachOError::OutOfBounds)?
        .try_into()
        .map_err(|_| MachOError::OutOfBounds)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u32(buf: &mut [u8], off: usize, val: u32) {
        buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }

    fn put_u64(buf: &mut [u8], off: usize, val: u64) {
        buf[off..off + 8].copy_from_slice(&val.to_le_bytes());
    }

    fn build_minimal_macho() -> Vec<u8> {
        // Layout:
        // 0x000..0x020: MachHeader64 (32 bytes)
        // 0x020..0x0B8: LC_SEGMENT_64 (__TEXT, 1 section: 72 + 80 = 152 bytes)
        // 0x0B8..0x0D0: LC_MAIN (24 bytes)
        // 0x0D0..0x0F0: LC_LOAD_DYLINKER ("/usr/lib/dyld\0", 12 + 14 = 26 -> padded to 32 bytes)
        // Total sizeofcmds = 152 + 24 + 32 = 208 (0xD0)
        // 0x100..0x120: __text code
        let mut buf = vec![0u8; 0x120];

        put_u32(&mut buf, 0, MH_MAGIC_64);
        put_u32(&mut buf, 4, CPU_TYPE_X86_64);
        put_u32(&mut buf, 8, CPU_SUBTYPE_X86_64_ALL);
        put_u32(&mut buf, 12, MH_EXECUTE);
        put_u32(&mut buf, 16, 3); // ncmds
        put_u32(&mut buf, 20, 208); // sizeofcmds

        // LC_SEGMENT_64 at 0x20
        let seg = 0x20;
        put_u32(&mut buf, seg, LC_SEGMENT_64);
        put_u32(&mut buf, seg + 4, 152);
        buf[seg + 8..seg + 14].copy_from_slice(b"__TEXT");
        put_u64(&mut buf, seg + 24, 0x1_0000_0000); // vmaddr
        put_u64(&mut buf, seg + 32, 0x1000); // vmsize
        put_u64(&mut buf, seg + 40, 0); // fileoff
        put_u64(&mut buf, seg + 48, 0x120); // filesize
        put_u32(&mut buf, seg + 56, VM_PROT_READ | VM_PROT_EXECUTE);
        put_u32(&mut buf, seg + 60, VM_PROT_READ | VM_PROT_EXECUTE);
        put_u32(&mut buf, seg + 64, 1); // nsects

        // Section __text at seg + 72
        let sec = seg + 72;
        buf[sec..sec + 6].copy_from_slice(b"__text");
        buf[sec + 16..sec + 22].copy_from_slice(b"__TEXT");
        put_u64(&mut buf, sec + 32, 0x1_0000_0100); // addr
        put_u64(&mut buf, sec + 40, 0x20); // size
        put_u32(&mut buf, sec + 48, 0x100); // offset

        // LC_MAIN at 0xB8
        let main_off = 0xB8;
        put_u32(&mut buf, main_off, LC_MAIN);
        put_u32(&mut buf, main_off + 4, 24);
        put_u64(&mut buf, main_off + 8, 0x100); // entryoff
        put_u64(&mut buf, main_off + 16, 0x10_0000); // stacksize

        // LC_LOAD_DYLINKER at 0xD0
        let dyld_off = 0xD0;
        put_u32(&mut buf, dyld_off, LC_LOAD_DYLINKER);
        put_u32(&mut buf, dyld_off + 4, 32);
        put_u32(&mut buf, dyld_off + 8, 12); // str offset
        let dyld_path = b"/usr/lib/dyld\0";
        buf[dyld_off + 12..dyld_off + 12 + dyld_path.len()].copy_from_slice(dyld_path);

        buf[0x100] = 0xC3; // ret
        buf
    }

    #[test]
    fn parse_valid_macho64() {
        let data = build_minimal_macho();
        let macho = MachO64::parse(&data).expect("valid Mach-O 64 should parse");
        assert_eq!(macho.header().cputype, CPU_TYPE_X86_64);
        assert_eq!(macho.header().filetype, MH_EXECUTE);

        let segs: Vec<_> = macho.segments().collect();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].name_str(), Some("__TEXT"));
        assert!(segs[0].is_readable());
        assert!(segs[0].is_executable());
        assert!(!segs[0].is_writable());

        let ep = macho.main_entry().unwrap();
        assert_eq!(ep.entryoff, 0x100);
        assert_eq!(ep.stacksize, 0x10_0000);

        assert_eq!(macho.dylinker(), Some("/usr/lib/dyld"));

        for lc in macho.load_commands() {
            if let LoadCommand::Segment64 {
                segment,
                sections_data,
            } = lc
            {
                let secs: Vec<_> = MachO64::parse_sections(sections_data, segment.nsects)
                    .unwrap()
                    .collect();
                assert_eq!(secs.len(), 1);
                assert_eq!(secs[0].sect_name_str(), Some("__text"));
                assert_eq!(secs[0].seg_name_str(), Some("__TEXT"));
            }
        }
    }

    #[test]
    fn reject_bad_magic_and_cpu() {
        assert_eq!(MachO64::parse(&[0u8; 8]), Err(MachOError::TooShort));

        let mut data = build_minimal_macho();
        put_u32(&mut data, 0, 0xFEED_FACE); // 32-bit magic
        assert_eq!(
            MachO64::parse(&data),
            Err(MachOError::InvalidMagic(0xFEED_FACE))
        );

        let mut data = build_minimal_macho();
        put_u32(&mut data, 4, 0x0100_000C); // ARM64
        assert_eq!(
            MachO64::parse(&data),
            Err(MachOError::UnsupportedCpuType(0x0100_000C))
        );
    }

    #[test]
    fn reject_out_of_bounds_segment_and_bad_cmdsize() {
        let mut data = build_minimal_macho();
        put_u64(&mut data, 0x20 + 32, 0x20_0000); // vmsize
        put_u64(&mut data, 0x20 + 48, 0x10_0000); // filesize > data.len()
        assert_eq!(MachO64::parse(&data), Err(MachOError::OutOfBounds));

        let mut data = build_minimal_macho();
        put_u64(&mut data, 0x20 + 48, 0x2000); // filesize > vmsize (0x1000)
        assert_eq!(MachO64::parse(&data), Err(MachOError::InvalidSegment));

        let mut data = build_minimal_macho();
        put_u32(&mut data, 0xB8 + 4, 16); // LC_MAIN wrong cmdsize
        assert_eq!(MachO64::parse(&data), Err(MachOError::InvalidLoadCommand));
    }
}
