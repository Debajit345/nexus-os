//! `no_std` Clean-Room ext4 Extent-Tree & JBD2 Journal Parser/Writer.
//!
//! Provides:
//! - [`Ext4Superblock`] and [`Ext4GroupDesc`] parsers and serializers (32-bit & 64-bit).
//! - [`Ext4Inode`] parser and serializer with extent-tree root access.
//! - [`ExtentTree`] lookup (binary search across arbitrary depth) and extent insertion
//!   with coalescing and root-to-depth-1 splitting.
//! - [`Ext4DirEntry`] linear directory block parser, lookup, and entry writer.
//! - [`Ext4Fs`] high-level filesystem reader/writer over [`BlockDevice`].
//! - JBD2 journal parser and writer ([`Jbd2Header`], [`Jbd2Superblock`],
//!   [`Jbd2BlockTag`], [`Jbd2CommitHeader`], [`Jbd2RevokeBlock`], [`Jbd2Transaction`]).
//! - [`self_test`] exercising all subsystems.

#![allow(dead_code)]

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::serial_println;

/// ext4 superblock magic (`0xEF53`).
pub const EXT4_SUPER_MAGIC: u16 = 0xEF53;

/// Superblock byte offset from start of device.
pub const EXT4_SUPERBLOCK_OFFSET: usize = 1024;

/// Superblock size in bytes.
pub const EXT4_SUPERBLOCK_SIZE: usize = 1024;

/// Extent header magic (`0xF30A`).
pub const EXT4_EXT_MAGIC: u16 = 0xF30A;

/// Root directory inode number in ext4.
pub const EXT4_ROOT_INO: u32 = 2;

/// Default journal inode number in ext4.
pub const EXT4_JOURNAL_INO: u32 = 8;

/// Inode flag: inode uses extents.
pub const EXT4_EXTENTS_FL: u32 = 0x0008_0000;

/// Inode mode masks.
pub const S_IFMT: u16 = 0xF000;
pub const S_IFREG: u16 = 0x8000;
pub const S_IFDIR: u16 = 0x4000;
pub const S_IFLNK: u16 = 0xA000;

/// Superblock compat feature flags.
pub const EXT4_FEATURE_COMPAT_HAS_JOURNAL: u32 = 0x0004;
pub const EXT4_FEATURE_COMPAT_EXT_ATTR: u32 = 0x0008;
pub const EXT4_FEATURE_COMPAT_DIR_INDEX: u32 = 0x0020;

/// Superblock incompat feature flags.
pub const EXT4_FEATURE_INCOMPAT_FILETYPE: u32 = 0x0002;
pub const EXT4_FEATURE_INCOMPAT_EXTENTS: u32 = 0x0040;
pub const EXT4_FEATURE_INCOMPAT_64BIT: u32 = 0x0080;
pub const EXT4_FEATURE_INCOMPAT_FLEX_BG: u32 = 0x0200;

/// Superblock ro_compat feature flags.
pub const EXT4_FEATURE_RO_COMPAT_SPARSE_SUPER: u32 = 0x0001;
pub const EXT4_FEATURE_RO_COMPAT_LARGE_FILE: u32 = 0x0002;
pub const EXT4_FEATURE_RO_COMPAT_HUGE_FILE: u32 = 0x0008;
pub const EXT4_FEATURE_RO_COMPAT_METADATA_CSUM: u32 = 0x0400;

/// Directory entry file types.
pub const EXT4_FT_UNKNOWN: u8 = 0;
pub const EXT4_FT_REG_FILE: u8 = 1;
pub const EXT4_FT_DIR: u8 = 2;
pub const EXT4_FT_CHRDEV: u8 = 3;
pub const EXT4_FT_BLKDEV: u8 = 4;
pub const EXT4_FT_FIFO: u8 = 5;
pub const EXT4_FT_SOCK: u8 = 6;
pub const EXT4_FT_SYMLINK: u8 = 7;

/// Maximum filename length in ext4.
pub const EXT4_NAME_LEN: usize = 255;

/// JBD2 magic number (big-endian on disk: `0xC03B3998`).
pub const JBD2_MAGIC_NUMBER: u32 = 0xC03B_3998;

/// JBD2 block types.
pub const JBD2_DESCRIPTOR_BLOCK: u32 = 1;
pub const JBD2_COMMIT_BLOCK: u32 = 2;
pub const JBD2_SUPERBLOCK_V1: u32 = 3;
pub const JBD2_SUPERBLOCK_V2: u32 = 4;
pub const JBD2_REVOKE_BLOCK: u32 = 5;

/// JBD2 block tag flags.
pub const JBD2_FLAG_ESCAPE: u32 = 1;
pub const JBD2_FLAG_SAME_UUID: u32 = 2;
pub const JBD2_FLAG_DELETED: u32 = 4;
pub const JBD2_FLAG_LAST_TAG: u32 = 8;

/// JBD2 incompat feature flags.
pub const JBD2_FEATURE_INCOMPAT_REVOKE: u32 = 0x0001;
pub const JBD2_FEATURE_INCOMPAT_64BIT: u32 = 0x0002;
pub const JBD2_FEATURE_INCOMPAT_CSUM_V2: u32 = 0x0008;
pub const JBD2_FEATURE_INCOMPAT_CSUM_V3: u32 = 0x0010;

/// Reflected CRC32C (Castagnoli) polynomial.
const CRC32C_POLY: u32 = 0x82F6_3B78;

/// Errors returned by the ext4 and JBD2 subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ext4Error {
    InvalidMagic,
    UnsupportedFeature,
    InvalidBlockSize,
    InvalidInodeSize,
    CorruptSuperblock(&'static str),
    CorruptGroupDesc(&'static str),
    CorruptInode(&'static str),
    CorruptExtent(&'static str),
    CorruptDirectory(&'static str),
    CorruptJournal(&'static str),
    ChecksumMismatch,
    NotFound,
    AlreadyExists,
    NotADirectory,
    IsADirectory,
    NameTooLong,
    NoSpace,
    IoError(&'static str),
    BufferTooSmall,
}

/// Compute raw CRC32C (Castagnoli) without final inversion, matching ext4/JBD2 metadata_csum.
pub fn crc32c_raw(seed: u32, data: &[u8]) -> u32 {
    let mut crc = seed;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (CRC32C_POLY & mask);
        }
    }
    crc
}

/// Compute standard CRC32C (Castagnoli) with `0xFFFF_FFFF` initial and final XOR.
pub fn crc32c(data: &[u8]) -> u32 {
    !crc32c_raw(0xFFFF_FFFF, data)
}

#[inline]
fn read_u16_le(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

#[inline]
fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
fn write_u16_le(buf: &mut [u8], off: usize, val: u16) {
    let b = val.to_le_bytes();
    buf[off] = b[0];
    buf[off + 1] = b[1];
}

#[inline]
fn write_u32_le(buf: &mut [u8], off: usize, val: u32) {
    let b = val.to_le_bytes();
    buf[off] = b[0];
    buf[off + 1] = b[1];
    buf[off + 2] = b[2];
    buf[off + 3] = b[3];
}

#[inline]
fn read_u32_be(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
fn read_u64_be(buf: &[u8], off: usize) -> u64 {
    u64::from_be_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

#[inline]
fn write_u32_be(buf: &mut [u8], off: usize, val: u32) {
    let b = val.to_be_bytes();
    buf[off] = b[0];
    buf[off + 1] = b[1];
    buf[off + 2] = b[2];
    buf[off + 3] = b[3];
}

#[inline]
fn write_u64_be(buf: &mut [u8], off: usize, val: u64) {
    let b = val.to_be_bytes();
    buf[off..off + 8].copy_from_slice(&b);
}

/// Block device abstraction for ext4 and JBD2 I/O.
pub trait BlockDevice {
    fn block_size(&self) -> usize;
    fn total_blocks(&self) -> u64;
    fn read_block(&self, block_num: u64, buf: &mut [u8]) -> Result<(), Ext4Error>;
    fn write_block(&mut self, block_num: u64, buf: &[u8]) -> Result<(), Ext4Error>;
}

/// In-memory block device for testing and ramdisk images.
pub struct MemBlockDevice {
    block_size: usize,
    data: Vec<u8>,
}

impl MemBlockDevice {
    /// Create a zeroed in-memory block device with `num_blocks` blocks of `block_size` bytes.
    pub fn new(block_size: usize, num_blocks: u64) -> Result<Self, Ext4Error> {
        if !block_size.is_power_of_two() || !(1024..=65536).contains(&block_size) {
            return Err(Ext4Error::InvalidBlockSize);
        }
        let total_bytes = (num_blocks as usize)
            .checked_mul(block_size)
            .ok_or(Ext4Error::NoSpace)?;
        Ok(Self {
            block_size,
            data: vec![0u8; total_bytes],
        })
    }

    /// Access the underlying raw byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Access the underlying mutable raw byte slice.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl BlockDevice for MemBlockDevice {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn total_blocks(&self) -> u64 {
        (self.data.len() / self.block_size) as u64
    }

    fn read_block(&self, block_num: u64, buf: &mut [u8]) -> Result<(), Ext4Error> {
        if buf.len() < self.block_size {
            return Err(Ext4Error::BufferTooSmall);
        }
        let start = (block_num as usize)
            .checked_mul(self.block_size)
            .ok_or(Ext4Error::IoError("block offset overflow"))?;
        let end = start
            .checked_add(self.block_size)
            .ok_or(Ext4Error::IoError("block end overflow"))?;
        if end > self.data.len() {
            return Err(Ext4Error::IoError("read past end of device"));
        }
        buf[..self.block_size].copy_from_slice(&self.data[start..end]);
        Ok(())
    }

    fn write_block(&mut self, block_num: u64, buf: &[u8]) -> Result<(), Ext4Error> {
        if buf.len() < self.block_size {
            return Err(Ext4Error::BufferTooSmall);
        }
        let start = (block_num as usize)
            .checked_mul(self.block_size)
            .ok_or(Ext4Error::IoError("block offset overflow"))?;
        let end = start
            .checked_add(self.block_size)
            .ok_or(Ext4Error::IoError("block end overflow"))?;
        if end > self.data.len() {
            return Err(Ext4Error::IoError("write past end of device"));
        }
        self.data[start..end].copy_from_slice(&buf[..self.block_size]);
        Ok(())
    }
}

/// Parsed ext4 superblock (1024 bytes on disk at offset 1024).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ext4Superblock {
    pub inodes_count: u32,
    pub blocks_count: u64,
    pub r_blocks_count: u64,
    pub free_blocks_count: u64,
    pub free_inodes_count: u32,
    pub first_data_block: u32,
    pub log_block_size: u32,
    pub log_cluster_size: u32,
    pub blocks_per_group: u32,
    pub clusters_per_group: u32,
    pub inodes_per_group: u32,
    pub mtime: u32,
    pub wtime: u32,
    pub mnt_count: u16,
    pub max_mnt_count: u16,
    pub magic: u16,
    pub state: u16,
    pub errors: u16,
    pub minor_rev_level: u16,
    pub rev_level: u32,
    pub first_ino: u32,
    pub inode_size: u16,
    pub block_group_nr: u16,
    pub feature_compat: u32,
    pub feature_incompat: u32,
    pub feature_ro_compat: u32,
    pub uuid: [u8; 16],
    pub volume_name: [u8; 16],
    pub journal_inum: u32,
    pub desc_size: u16,
    pub checksum_type: u8,
    pub checksum: u32,
}

impl Ext4Superblock {
    /// Parse an ext4 superblock from a 1024-byte slice.
    pub fn parse(buf: &[u8]) -> Result<Self, Ext4Error> {
        if buf.len() < EXT4_SUPERBLOCK_SIZE {
            return Err(Ext4Error::BufferTooSmall);
        }
        let magic = read_u16_le(buf, 0x38);
        if magic != EXT4_SUPER_MAGIC {
            return Err(Ext4Error::InvalidMagic);
        }

        let log_block_size = read_u32_le(buf, 0x18);
        if log_block_size > 6 {
            return Err(Ext4Error::InvalidBlockSize);
        }

        let blocks_per_group = read_u32_le(buf, 0x20);
        let inodes_per_group = read_u32_le(buf, 0x28);
        if blocks_per_group == 0 || inodes_per_group == 0 {
            return Err(Ext4Error::CorruptSuperblock("zero blocks/inodes per group"));
        }

        let rev_level = read_u32_le(buf, 0x4C);
        let inode_size = if rev_level == 0 {
            128
        } else {
            read_u16_le(buf, 0x58)
        };
        if inode_size < 128 || !inode_size.is_power_of_two() {
            return Err(Ext4Error::InvalidInodeSize);
        }

        let feature_compat = read_u32_le(buf, 0x5C);
        let feature_incompat = read_u32_le(buf, 0x60);
        let feature_ro_compat = read_u32_le(buf, 0x64);

        let is_64bit = (feature_incompat & EXT4_FEATURE_INCOMPAT_64BIT) != 0;
        let blocks_lo = u64::from(read_u32_le(buf, 0x04));
        let r_blocks_lo = u64::from(read_u32_le(buf, 0x08));
        let free_blocks_lo = u64::from(read_u32_le(buf, 0x0C));

        let (blocks_count, r_blocks_count, free_blocks_count) = if is_64bit {
            (
                blocks_lo | (u64::from(read_u32_le(buf, 0x150)) << 32),
                r_blocks_lo | (u64::from(read_u32_le(buf, 0x154)) << 32),
                free_blocks_lo | (u64::from(read_u32_le(buf, 0x158)) << 32),
            )
        } else {
            (blocks_lo, r_blocks_lo, free_blocks_lo)
        };

        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&buf[0x68..0x78]);
        let mut volume_name = [0u8; 16];
        volume_name.copy_from_slice(&buf[0x78..0x88]);

        let raw_desc_size = read_u16_le(buf, 0xFE);
        let desc_size = if is_64bit && raw_desc_size >= 64 {
            raw_desc_size
        } else if is_64bit {
            64
        } else {
            32
        };

        let checksum_type = buf[0x175];
        let checksum = read_u32_le(buf, 0x3FC);

        if (feature_ro_compat & EXT4_FEATURE_RO_COMPAT_METADATA_CSUM) != 0 {
            let computed = crc32c_raw(0xFFFF_FFFF, &buf[..0x3FC]);
            if computed != checksum {
                return Err(Ext4Error::ChecksumMismatch);
            }
        }

        Ok(Self {
            inodes_count: read_u32_le(buf, 0x00),
            blocks_count,
            r_blocks_count,
            free_blocks_count,
            free_inodes_count: read_u32_le(buf, 0x10),
            first_data_block: read_u32_le(buf, 0x14),
            log_block_size,
            log_cluster_size: read_u32_le(buf, 0x1C),
            blocks_per_group,
            clusters_per_group: read_u32_le(buf, 0x24),
            inodes_per_group,
            mtime: read_u32_le(buf, 0x2C),
            wtime: read_u32_le(buf, 0x30),
            mnt_count: read_u16_le(buf, 0x34),
            max_mnt_count: read_u16_le(buf, 0x36),
            magic,
            state: read_u16_le(buf, 0x3A),
            errors: read_u16_le(buf, 0x3C),
            minor_rev_level: read_u16_le(buf, 0x3E),
            rev_level,
            first_ino: read_u32_le(buf, 0x54),
            inode_size,
            block_group_nr: read_u16_le(buf, 0x5A),
            feature_compat,
            feature_incompat,
            feature_ro_compat,
            uuid,
            volume_name,
            journal_inum: read_u32_le(buf, 0xE0),
            desc_size,
            checksum_type,
            checksum,
        })
    }

    /// Serialize this superblock into a 1024-byte buffer, updating `s_checksum` if `METADATA_CSUM` is set.
    pub fn serialize(&self, buf: &mut [u8]) -> Result<(), Ext4Error> {
        if buf.len() < EXT4_SUPERBLOCK_SIZE {
            return Err(Ext4Error::BufferTooSmall);
        }
        buf[..EXT4_SUPERBLOCK_SIZE].fill(0);

        write_u32_le(buf, 0x00, self.inodes_count);
        write_u32_le(buf, 0x04, self.blocks_count as u32);
        write_u32_le(buf, 0x08, self.r_blocks_count as u32);
        write_u32_le(buf, 0x0C, self.free_blocks_count as u32);
        write_u32_le(buf, 0x10, self.free_inodes_count);
        write_u32_le(buf, 0x14, self.first_data_block);
        write_u32_le(buf, 0x18, self.log_block_size);
        write_u32_le(buf, 0x1C, self.log_cluster_size);
        write_u32_le(buf, 0x20, self.blocks_per_group);
        write_u32_le(buf, 0x24, self.clusters_per_group);
        write_u32_le(buf, 0x28, self.inodes_per_group);
        write_u32_le(buf, 0x2C, self.mtime);
        write_u32_le(buf, 0x30, self.wtime);
        write_u16_le(buf, 0x34, self.mnt_count);
        write_u16_le(buf, 0x36, self.max_mnt_count);
        write_u16_le(buf, 0x38, self.magic);
        write_u16_le(buf, 0x3A, self.state);
        write_u16_le(buf, 0x3C, self.errors);
        write_u16_le(buf, 0x3E, self.minor_rev_level);
        write_u32_le(buf, 0x4C, self.rev_level);
        write_u32_le(buf, 0x54, self.first_ino);
        write_u16_le(buf, 0x58, self.inode_size);
        write_u16_le(buf, 0x5A, self.block_group_nr);
        write_u32_le(buf, 0x5C, self.feature_compat);
        write_u32_le(buf, 0x60, self.feature_incompat);
        write_u32_le(buf, 0x64, self.feature_ro_compat);
        buf[0x68..0x78].copy_from_slice(&self.uuid);
        buf[0x78..0x88].copy_from_slice(&self.volume_name);
        write_u32_le(buf, 0xE0, self.journal_inum);
        write_u16_le(buf, 0xFE, self.desc_size);

        if self.is_64bit() {
            write_u32_le(buf, 0x150, (self.blocks_count >> 32) as u32);
            write_u32_le(buf, 0x154, (self.r_blocks_count >> 32) as u32);
            write_u32_le(buf, 0x158, (self.free_blocks_count >> 32) as u32);
        }

        buf[0x175] = self.checksum_type;
        let csum = if self.has_metadata_csum() {
            crc32c_raw(0xFFFF_FFFF, &buf[..0x3FC])
        } else {
            self.checksum
        };
        write_u32_le(buf, 0x3FC, csum);
        Ok(())
    }

    /// Filesystem block size in bytes.
    pub fn block_size(&self) -> usize {
        1024usize << self.log_block_size
    }

    /// Number of block groups in the filesystem.
    pub fn group_count(&self) -> u32 {
        let usable = self
            .blocks_count
            .saturating_sub(u64::from(self.first_data_block));
        let bpg = u64::from(self.blocks_per_group);
        usable.div_ceil(bpg) as u32
    }

    /// Whether 64-bit block numbers are enabled.
    pub fn is_64bit(&self) -> bool {
        (self.feature_incompat & EXT4_FEATURE_INCOMPAT_64BIT) != 0
    }

    /// Whether extent trees are enabled.
    pub fn has_extents(&self) -> bool {
        (self.feature_incompat & EXT4_FEATURE_INCOMPAT_EXTENTS) != 0
    }

    /// Whether metadata CRC32C checksums are enabled.
    pub fn has_metadata_csum(&self) -> bool {
        (self.feature_ro_compat & EXT4_FEATURE_RO_COMPAT_METADATA_CSUM) != 0
    }

    /// Block number where the Block Group Descriptor Table starts.
    pub fn gdt_start_block(&self) -> u64 {
        if self.block_size() == 1024 {
            2
        } else {
            1
        }
    }
}

/// Parsed ext4 Block Group Descriptor (32-byte or 64-byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ext4GroupDesc {
    pub block_bitmap: u64,
    pub inode_bitmap: u64,
    pub inode_table: u64,
    pub free_blocks_count: u32,
    pub free_inodes_count: u32,
    pub used_dirs_count: u32,
    pub flags: u16,
    pub itable_unused: u32,
    pub checksum: u16,
}

impl Ext4GroupDesc {
    /// Parse a block group descriptor from `buf`.
    pub fn parse(buf: &[u8], is_64bit: bool) -> Result<Self, Ext4Error> {
        let needed = if is_64bit { 64 } else { 32 };
        if buf.len() < needed {
            return Err(Ext4Error::BufferTooSmall);
        }

        let mut block_bitmap = u64::from(read_u32_le(buf, 0x00));
        let mut inode_bitmap = u64::from(read_u32_le(buf, 0x04));
        let mut inode_table = u64::from(read_u32_le(buf, 0x08));
        let mut free_blocks_count = u32::from(read_u16_le(buf, 0x0C));
        let mut free_inodes_count = u32::from(read_u16_le(buf, 0x0E));
        let mut used_dirs_count = u32::from(read_u16_le(buf, 0x10));
        let flags = read_u16_le(buf, 0x12);
        let mut itable_unused = u32::from(read_u16_le(buf, 0x1C));
        let checksum = read_u16_le(buf, 0x1E);

        if is_64bit {
            block_bitmap |= u64::from(read_u32_le(buf, 0x20)) << 32;
            inode_bitmap |= u64::from(read_u32_le(buf, 0x24)) << 32;
            inode_table |= u64::from(read_u32_le(buf, 0x28)) << 32;
            free_blocks_count |= u32::from(read_u16_le(buf, 0x2C)) << 16;
            free_inodes_count |= u32::from(read_u16_le(buf, 0x2E)) << 16;
            used_dirs_count |= u32::from(read_u16_le(buf, 0x30)) << 16;
            itable_unused |= u32::from(read_u16_le(buf, 0x34)) << 16;
        }

        Ok(Self {
            block_bitmap,
            inode_bitmap,
            inode_table,
            free_blocks_count,
            free_inodes_count,
            used_dirs_count,
            flags,
            itable_unused,
            checksum,
        })
    }

    /// Serialize this block group descriptor into `buf`.
    pub fn serialize(&self, buf: &mut [u8], is_64bit: bool) -> Result<(), Ext4Error> {
        let needed = if is_64bit { 64 } else { 32 };
        if buf.len() < needed {
            return Err(Ext4Error::BufferTooSmall);
        }
        buf[..needed].fill(0);

        write_u32_le(buf, 0x00, self.block_bitmap as u32);
        write_u32_le(buf, 0x04, self.inode_bitmap as u32);
        write_u32_le(buf, 0x08, self.inode_table as u32);
        write_u16_le(buf, 0x0C, self.free_blocks_count as u16);
        write_u16_le(buf, 0x0E, self.free_inodes_count as u16);
        write_u16_le(buf, 0x10, self.used_dirs_count as u16);
        write_u16_le(buf, 0x12, self.flags);
        write_u16_le(buf, 0x1C, self.itable_unused as u16);
        write_u16_le(buf, 0x1E, self.checksum);

        if is_64bit {
            write_u32_le(buf, 0x20, (self.block_bitmap >> 32) as u32);
            write_u32_le(buf, 0x24, (self.inode_bitmap >> 32) as u32);
            write_u32_le(buf, 0x28, (self.inode_table >> 32) as u32);
            write_u16_le(buf, 0x2C, (self.free_blocks_count >> 16) as u16);
            write_u16_le(buf, 0x2E, (self.free_inodes_count >> 16) as u16);
            write_u16_le(buf, 0x30, (self.used_dirs_count >> 16) as u16);
            write_u16_le(buf, 0x34, (self.itable_unused >> 16) as u16);
        }
        Ok(())
    }
}

/// Parsed ext4 Inode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ext4Inode {
    pub mode: u16,
    pub uid: u32,
    pub size: u64,
    pub atime: u32,
    pub ctime: u32,
    pub mtime: u32,
    pub dtime: u32,
    pub gid: u32,
    pub links_count: u16,
    pub blocks_lo: u32,
    pub flags: u32,
    pub i_block: [u8; 60],
    pub generation: u32,
    pub extra_isize: u16,
}

impl Ext4Inode {
    /// Parse an inode from a slice of at least 128 bytes.
    pub fn parse(buf: &[u8]) -> Result<Self, Ext4Error> {
        if buf.len() < 128 {
            return Err(Ext4Error::BufferTooSmall);
        }
        let mode = read_u16_le(buf, 0x00);
        let uid = u32::from(read_u16_le(buf, 0x02)) | (u32::from(read_u16_le(buf, 0x78)) << 16);
        let size_lo = u64::from(read_u32_le(buf, 0x04));
        let size_hi = if (mode & S_IFMT) == S_IFREG {
            u64::from(read_u32_le(buf, 0x6C))
        } else {
            0
        };
        let size = size_lo | (size_hi << 32);
        let gid = u32::from(read_u16_le(buf, 0x18)) | (u32::from(read_u16_le(buf, 0x7A)) << 16);

        let mut i_block = [0u8; 60];
        i_block.copy_from_slice(&buf[0x28..0x64]);

        let extra_isize = if buf.len() >= 130 {
            read_u16_le(buf, 0x80)
        } else {
            0
        };

        Ok(Self {
            mode,
            uid,
            size,
            atime: read_u32_le(buf, 0x08),
            ctime: read_u32_le(buf, 0x0C),
            mtime: read_u32_le(buf, 0x10),
            dtime: read_u32_le(buf, 0x14),
            gid,
            links_count: read_u16_le(buf, 0x1A),
            blocks_lo: read_u32_le(buf, 0x1C),
            flags: read_u32_le(buf, 0x20),
            i_block,
            generation: read_u32_le(buf, 0x64),
            extra_isize,
        })
    }

    /// Serialize this inode into `buf` (at least 128 bytes).
    pub fn serialize(&self, buf: &mut [u8]) -> Result<(), Ext4Error> {
        if buf.len() < 128 {
            return Err(Ext4Error::BufferTooSmall);
        }
        buf.fill(0);
        write_u16_le(buf, 0x00, self.mode);
        write_u16_le(buf, 0x02, self.uid as u16);
        write_u32_le(buf, 0x04, self.size as u32);
        write_u32_le(buf, 0x08, self.atime);
        write_u32_le(buf, 0x0C, self.ctime);
        write_u32_le(buf, 0x10, self.mtime);
        write_u32_le(buf, 0x14, self.dtime);
        write_u16_le(buf, 0x18, self.gid as u16);
        write_u16_le(buf, 0x1A, self.links_count);
        write_u32_le(buf, 0x1C, self.blocks_lo);
        write_u32_le(buf, 0x20, self.flags);
        buf[0x28..0x64].copy_from_slice(&self.i_block);
        write_u32_le(buf, 0x64, self.generation);
        if self.is_regular() {
            write_u32_le(buf, 0x6C, (self.size >> 32) as u32);
        }
        write_u16_le(buf, 0x78, (self.uid >> 16) as u16);
        write_u16_le(buf, 0x7A, (self.gid >> 16) as u16);
        if buf.len() >= 130 {
            write_u16_le(buf, 0x80, self.extra_isize);
        }
        Ok(())
    }

    pub fn is_dir(&self) -> bool {
        (self.mode & S_IFMT) == S_IFDIR
    }

    pub fn is_regular(&self) -> bool {
        (self.mode & S_IFMT) == S_IFREG
    }

    pub fn is_symlink(&self) -> bool {
        (self.mode & S_IFMT) == S_IFLNK
    }

    pub fn uses_extents(&self) -> bool {
        (self.flags & EXT4_EXTENTS_FL) != 0
    }
}

/// Extent tree node header (`ext4_extent_header`, 12 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ext4ExtentHeader {
    pub magic: u16,
    pub entries: u16,
    pub max: u16,
    pub depth: u16,
    pub generation: u32,
}

impl Ext4ExtentHeader {
    pub fn parse(buf: &[u8]) -> Result<Self, Ext4Error> {
        if buf.len() < 12 {
            return Err(Ext4Error::BufferTooSmall);
        }
        let magic = read_u16_le(buf, 0);
        if magic != EXT4_EXT_MAGIC {
            return Err(Ext4Error::CorruptExtent("bad extent header magic"));
        }
        let entries = read_u16_le(buf, 2);
        let max = read_u16_le(buf, 4);
        if entries > max {
            return Err(Ext4Error::CorruptExtent("entries > max"));
        }
        Ok(Self {
            magic,
            entries,
            max,
            depth: read_u16_le(buf, 6),
            generation: read_u32_le(buf, 8),
        })
    }

    pub fn serialize(&self, buf: &mut [u8]) -> Result<(), Ext4Error> {
        if buf.len() < 12 {
            return Err(Ext4Error::BufferTooSmall);
        }
        write_u16_le(buf, 0, self.magic);
        write_u16_le(buf, 2, self.entries);
        write_u16_le(buf, 4, self.max);
        write_u16_le(buf, 6, self.depth);
        write_u32_le(buf, 8, self.generation);
        Ok(())
    }
}

/// Extent tree internal index entry (`ext4_extent_idx`, 12 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ext4ExtentIdx {
    pub block: u32,
    pub leaf_lo: u32,
    pub leaf_hi: u16,
    pub unused: u16,
}

impl Ext4ExtentIdx {
    pub fn parse(buf: &[u8]) -> Result<Self, Ext4Error> {
        if buf.len() < 12 {
            return Err(Ext4Error::BufferTooSmall);
        }
        Ok(Self {
            block: read_u32_le(buf, 0),
            leaf_lo: read_u32_le(buf, 4),
            leaf_hi: read_u16_le(buf, 8),
            unused: read_u16_le(buf, 10),
        })
    }

    pub fn serialize(&self, buf: &mut [u8]) -> Result<(), Ext4Error> {
        if buf.len() < 12 {
            return Err(Ext4Error::BufferTooSmall);
        }
        write_u32_le(buf, 0, self.block);
        write_u32_le(buf, 4, self.leaf_lo);
        write_u16_le(buf, 8, self.leaf_hi);
        write_u16_le(buf, 10, self.unused);
        Ok(())
    }

    pub fn leaf_block(&self) -> u64 {
        u64::from(self.leaf_lo) | (u64::from(self.leaf_hi) << 32)
    }

    pub fn from_parts(logical_block: u32, leaf_block: u64) -> Self {
        Self {
            block: logical_block,
            leaf_lo: leaf_block as u32,
            leaf_hi: (leaf_block >> 32) as u16,
            unused: 0,
        }
    }
}

/// Extent tree leaf entry (`ext4_extent`, 12 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ext4Extent {
    pub block: u32,
    pub len: u16,
    pub start_hi: u16,
    pub start_lo: u32,
}

impl Ext4Extent {
    pub fn parse(buf: &[u8]) -> Result<Self, Ext4Error> {
        if buf.len() < 12 {
            return Err(Ext4Error::BufferTooSmall);
        }
        Ok(Self {
            block: read_u32_le(buf, 0),
            len: read_u16_le(buf, 4),
            start_hi: read_u16_le(buf, 6),
            start_lo: read_u32_le(buf, 8),
        })
    }

    pub fn serialize(&self, buf: &mut [u8]) -> Result<(), Ext4Error> {
        if buf.len() < 12 {
            return Err(Ext4Error::BufferTooSmall);
        }
        write_u32_le(buf, 0, self.block);
        write_u16_le(buf, 4, self.len);
        write_u16_le(buf, 6, self.start_hi);
        write_u32_le(buf, 8, self.start_lo);
        Ok(())
    }

    pub fn from_parts(logical_block: u32, physical_block: u64, len: u16, unwritten: bool) -> Self {
        let encoded_len = if unwritten { len | 0x8000 } else { len };
        Self {
            block: logical_block,
            len: encoded_len,
            start_hi: (physical_block >> 32) as u16,
            start_lo: physical_block as u32,
        }
    }

    pub fn start_block(&self) -> u64 {
        u64::from(self.start_lo) | (u64::from(self.start_hi) << 32)
    }

    pub fn actual_len(&self) -> u16 {
        if self.len > 32768 {
            self.len - 32768
        } else {
            self.len
        }
    }

    pub fn is_unwritten(&self) -> bool {
        self.len > 32768
    }
}

/// Result of looking up a logical block in an extent tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtentMapping {
    pub physical_block: u64,
    pub remaining_blocks: u32,
    pub unwritten: bool,
}

/// Extent tree operations (lookup and insertion).
pub struct ExtentTree;

impl ExtentTree {
    /// Initialize an empty extent tree root inside an inode's `i_block` (60 bytes).
    pub fn init_root(i_block: &mut [u8; 60]) {
        i_block.fill(0);
        let hdr = Ext4ExtentHeader {
            magic: EXT4_EXT_MAGIC,
            entries: 0,
            max: 4,
            depth: 0,
            generation: 0,
        };
        let _ = hdr.serialize(&mut i_block[..12]);
    }

    /// Look up `logical_block` in the extent tree rooted at `root_i_block`.
    pub fn lookup(
        dev: &dyn BlockDevice,
        root_i_block: &[u8; 60],
        logical_block: u32,
    ) -> Result<Option<ExtentMapping>, Ext4Error> {
        let mut node_buf = vec![0u8; dev.block_size().max(60)];
        node_buf[..60].copy_from_slice(root_i_block);

        for _depth_guard in 0..8 {
            let hdr = Ext4ExtentHeader::parse(&node_buf[..12])?;
            let count = usize::from(hdr.entries);
            if count == 0 {
                return Ok(None);
            }

            if hdr.depth == 0 {
                // Binary search leaf extents.
                let mut lo = 0usize;
                let mut hi = count;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let off = 12 + mid * 12;
                    let ext = Ext4Extent::parse(&node_buf[off..off + 12])?;
                    if ext.block <= logical_block {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                if lo == 0 {
                    return Ok(None);
                }
                let off = 12 + (lo - 1) * 12;
                let ext = Ext4Extent::parse(&node_buf[off..off + 12])?;
                let len = u32::from(ext.actual_len());
                if logical_block >= ext.block && logical_block < ext.block.saturating_add(len) {
                    let delta = logical_block - ext.block;
                    return Ok(Some(ExtentMapping {
                        physical_block: ext.start_block() + u64::from(delta),
                        remaining_blocks: len - delta,
                        unwritten: ext.is_unwritten(),
                    }));
                }
                return Ok(None);
            }

            // Binary search internal index entries.
            let mut lo = 0usize;
            let mut hi = count;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let off = 12 + mid * 12;
                let idx = Ext4ExtentIdx::parse(&node_buf[off..off + 12])?;
                if idx.block <= logical_block {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo == 0 {
                return Ok(None);
            }
            let off = 12 + (lo - 1) * 12;
            let idx = Ext4ExtentIdx::parse(&node_buf[off..off + 12])?;
            dev.read_block(idx.leaf_block(), &mut node_buf[..dev.block_size()])?;
        }

        Err(Ext4Error::CorruptExtent("extent tree depth exceeded"))
    }

    /// Insert an extent into the extent tree rooted at `root_i_block`.
    /// Supports coalescing adjacent extents and splitting a full depth-0 root into depth-1.
    pub fn insert<F>(
        dev: &mut dyn BlockDevice,
        root_i_block: &mut [u8; 60],
        new_ext: Ext4Extent,
        mut alloc_block: F,
    ) -> Result<(), Ext4Error>
    where
        F: FnMut() -> Result<u64, Ext4Error>,
    {
        let hdr = Ext4ExtentHeader::parse(&root_i_block[..12])?;
        if hdr.depth == 0 {
            let mut extents = Vec::with_capacity(usize::from(hdr.entries) + 1);
            for i in 0..usize::from(hdr.entries) {
                let off = 12 + i * 12;
                extents.push(Ext4Extent::parse(&root_i_block[off..off + 12])?);
            }

            // Try coalescing with an existing adjacent extent.
            let extents_len = extents.len();
            if let Some(last) = extents.last_mut() {
                let last_end_logical = last.block.saturating_add(u32::from(last.actual_len()));
                let last_end_phys = last
                    .start_block()
                    .saturating_add(u64::from(last.actual_len()));
                if last_end_logical == new_ext.block
                    && last_end_phys == new_ext.start_block()
                    && last.is_unwritten() == new_ext.is_unwritten()
                    && u32::from(last.actual_len()) + u32::from(new_ext.actual_len()) <= 32768
                {
                    let combined = last.actual_len() + new_ext.actual_len();
                    *last = Ext4Extent::from_parts(
                        last.block,
                        last.start_block(),
                        combined,
                        last.is_unwritten(),
                    );
                    let off = 12 + (extents_len - 1) * 12;
                    last.serialize(&mut root_i_block[off..off + 12])?;
                    return Ok(());
                }
            }

            // Insert sorted by logical block.
            let pos = extents
                .binary_search_by_key(&new_ext.block, |e| e.block)
                .unwrap_or_else(|p| p);
            extents.insert(pos, new_ext);

            if extents.len() <= usize::from(hdr.max) {
                let new_hdr = Ext4ExtentHeader {
                    entries: extents.len() as u16,
                    ..hdr
                };
                new_hdr.serialize(&mut root_i_block[..12])?;
                for (i, ext) in extents.iter().enumerate() {
                    let off = 12 + i * 12;
                    ext.serialize(&mut root_i_block[off..off + 12])?;
                }
                return Ok(());
            }

            // Root overflow: split into a depth-1 tree with two leaf blocks.
            let mid = extents.len() / 2;
            let left_block = alloc_block()?;
            let right_block = alloc_block()?;
            let bsize = dev.block_size();
            let max_per_block = ((bsize - 12) / 12) as u16;

            let mut leaf_buf = vec![0u8; bsize];
            let left_hdr = Ext4ExtentHeader {
                magic: EXT4_EXT_MAGIC,
                entries: mid as u16,
                max: max_per_block,
                depth: 0,
                generation: hdr.generation,
            };
            left_hdr.serialize(&mut leaf_buf[..12])?;
            for (i, ext) in extents[..mid].iter().enumerate() {
                let off = 12 + i * 12;
                ext.serialize(&mut leaf_buf[off..off + 12])?;
            }
            dev.write_block(left_block, &leaf_buf)?;

            leaf_buf.fill(0);
            let right_hdr = Ext4ExtentHeader {
                magic: EXT4_EXT_MAGIC,
                entries: (extents.len() - mid) as u16,
                max: max_per_block,
                depth: 0,
                generation: hdr.generation,
            };
            right_hdr.serialize(&mut leaf_buf[..12])?;
            for (i, ext) in extents[mid..].iter().enumerate() {
                let off = 12 + i * 12;
                ext.serialize(&mut leaf_buf[off..off + 12])?;
            }
            dev.write_block(right_block, &leaf_buf)?;

            root_i_block.fill(0);
            let root_hdr = Ext4ExtentHeader {
                magic: EXT4_EXT_MAGIC,
                entries: 2,
                max: 4,
                depth: 1,
                generation: hdr.generation,
            };
            root_hdr.serialize(&mut root_i_block[..12])?;
            Ext4ExtentIdx::from_parts(extents[0].block, left_block)
                .serialize(&mut root_i_block[12..24])?;
            Ext4ExtentIdx::from_parts(extents[mid].block, right_block)
                .serialize(&mut root_i_block[24..36])?;
            return Ok(());
        }

        // Depth >= 1: descend to the appropriate leaf block and insert there.
        let count = usize::from(hdr.entries);
        if count == 0 {
            return Err(Ext4Error::CorruptExtent("empty internal extent node"));
        }
        let mut target_idx = 0usize;
        for i in 0..count {
            let off = 12 + i * 12;
            let idx = Ext4ExtentIdx::parse(&root_i_block[off..off + 12])?;
            if idx.block <= new_ext.block {
                target_idx = i;
            } else {
                break;
            }
        }
        let off = 12 + target_idx * 12;
        let idx = Ext4ExtentIdx::parse(&root_i_block[off..off + 12])?;
        let bsize = dev.block_size();
        let mut leaf_buf = vec![0u8; bsize];
        dev.read_block(idx.leaf_block(), &mut leaf_buf)?;
        let leaf_hdr = Ext4ExtentHeader::parse(&leaf_buf[..12])?;
        if leaf_hdr.depth != 0 {
            return Err(Ext4Error::UnsupportedFeature);
        }
        if leaf_hdr.entries >= leaf_hdr.max {
            return Err(Ext4Error::NoSpace);
        }
        let mut extents = Vec::with_capacity(usize::from(leaf_hdr.entries) + 1);
        for i in 0..usize::from(leaf_hdr.entries) {
            let eoff = 12 + i * 12;
            extents.push(Ext4Extent::parse(&leaf_buf[eoff..eoff + 12])?);
        }
        let pos = extents
            .binary_search_by_key(&new_ext.block, |e| e.block)
            .unwrap_or_else(|p| p);
        extents.insert(pos, new_ext);
        let updated_hdr = Ext4ExtentHeader {
            entries: extents.len() as u16,
            ..leaf_hdr
        };
        updated_hdr.serialize(&mut leaf_buf[..12])?;
        for (i, ext) in extents.iter().enumerate() {
            let eoff = 12 + i * 12;
            ext.serialize(&mut leaf_buf[eoff..eoff + 12])?;
        }
        dev.write_block(idx.leaf_block(), &leaf_buf)?;
        Ok(())
    }
}

/// Parsed ext4 directory entry (`ext4_dir_entry_2`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ext4DirEntry {
    pub inode: u32,
    pub rec_len: u16,
    pub file_type: u8,
    pub name: String,
}

/// Parse all valid directory entries in a single directory block.
pub fn parse_dir_block(block: &[u8]) -> Result<Vec<Ext4DirEntry>, Ext4Error> {
    let mut entries = Vec::new();
    let mut offset = 0usize;
    while offset + 8 <= block.len() {
        let inode = read_u32_le(block, offset);
        let rec_len = read_u16_le(block, offset + 4);
        if rec_len < 8 || !rec_len.is_multiple_of(4) || offset + usize::from(rec_len) > block.len()
        {
            return Err(Ext4Error::CorruptDirectory("invalid rec_len"));
        }
        let name_len = usize::from(block[offset + 6]);
        let file_type = block[offset + 7];
        if inode != 0 {
            if 8 + name_len > usize::from(rec_len) {
                return Err(Ext4Error::CorruptDirectory("name_len exceeds rec_len"));
            }
            let name_bytes = &block[offset + 8..offset + 8 + name_len];
            let name = core::str::from_utf8(name_bytes)
                .map_err(|_| Ext4Error::CorruptDirectory("non-utf8 filename"))?
                .to_string();
            entries.push(Ext4DirEntry {
                inode,
                rec_len,
                file_type,
                name,
            });
        }
        offset += usize::from(rec_len);
    }
    Ok(entries)
}

/// Look up a single filename in a directory block.
pub fn lookup_in_dir_block(block: &[u8], target: &str) -> Result<Option<Ext4DirEntry>, Ext4Error> {
    let mut offset = 0usize;
    while offset + 8 <= block.len() {
        let inode = read_u32_le(block, offset);
        let rec_len = read_u16_le(block, offset + 4);
        if rec_len < 8 || !rec_len.is_multiple_of(4) || offset + usize::from(rec_len) > block.len()
        {
            return Err(Ext4Error::CorruptDirectory("invalid rec_len"));
        }
        let name_len = usize::from(block[offset + 6]);
        let file_type = block[offset + 7];
        if inode != 0 && name_len == target.len() && 8 + name_len <= usize::from(rec_len) {
            let name_bytes = &block[offset + 8..offset + 8 + name_len];
            if name_bytes == target.as_bytes() {
                return Ok(Some(Ext4DirEntry {
                    inode,
                    rec_len,
                    file_type,
                    name: target.to_string(),
                }));
            }
        }
        offset += usize::from(rec_len);
    }
    Ok(None)
}

/// Initialize an empty directory block with `.` and `..` entries.
pub fn init_dir_block(block: &mut [u8], self_ino: u32, parent_ino: u32) -> Result<(), Ext4Error> {
    if block.len() < 32 || !block.len().is_multiple_of(4) {
        return Err(Ext4Error::BufferTooSmall);
    }
    block.fill(0);
    // "." entry: 12 bytes
    write_u32_le(block, 0, self_ino);
    write_u16_le(block, 4, 12);
    block[6] = 1;
    block[7] = EXT4_FT_DIR;
    block[8] = b'.';

    // ".." entry: spans rest of block
    let rest = (block.len() - 12) as u16;
    write_u32_le(block, 12, parent_ino);
    write_u16_le(block, 16, rest);
    block[18] = 2;
    block[19] = EXT4_FT_DIR;
    block[20] = b'.';
    block[21] = b'.';
    Ok(())
}

/// Append a directory entry into an existing directory block by splitting slack space.
pub fn append_dir_entry(
    block: &mut [u8],
    inode: u32,
    file_type: u8,
    name: &str,
) -> Result<(), Ext4Error> {
    if name.is_empty() || name.len() > EXT4_NAME_LEN {
        return Err(Ext4Error::NameTooLong);
    }
    let needed = (8 + name.len()).div_ceil(4) * 4;
    let mut offset = 0usize;
    while offset + 8 <= block.len() {
        let cur_ino = read_u32_le(block, offset);
        let cur_rec_len = usize::from(read_u16_le(block, offset + 4));
        if cur_rec_len < 8 || !cur_rec_len.is_multiple_of(4) || offset + cur_rec_len > block.len() {
            return Err(Ext4Error::CorruptDirectory("invalid rec_len"));
        }
        if cur_ino == 0 && cur_rec_len >= needed {
            write_u32_le(block, offset, inode);
            block[offset + 6] = name.len() as u8;
            block[offset + 7] = file_type;
            block[offset + 8..offset + 8 + name.len()].copy_from_slice(name.as_bytes());
            return Ok(());
        }
        let cur_name_len = usize::from(block[offset + 6]);
        let actual_len = (8 + cur_name_len).div_ceil(4) * 4;
        let slack = cur_rec_len.saturating_sub(actual_len);
        if cur_ino != 0 && slack >= needed {
            write_u16_le(block, offset + 4, actual_len as u16);
            let new_off = offset + actual_len;
            write_u32_le(block, new_off, inode);
            write_u16_le(block, new_off + 4, slack as u16);
            block[new_off + 6] = name.len() as u8;
            block[new_off + 7] = file_type;
            block[new_off + 8..new_off + 8 + name.len()].copy_from_slice(name.as_bytes());
            return Ok(());
        }
        offset += cur_rec_len;
    }
    Err(Ext4Error::NoSpace)
}

/// High-level ext4 filesystem reader/writer over a [`BlockDevice`].
pub struct Ext4Fs<D: BlockDevice> {
    pub dev: D,
    pub sb: Ext4Superblock,
    pub groups: Vec<Ext4GroupDesc>,
}

impl<D: BlockDevice> Ext4Fs<D> {
    /// Mount an ext4 filesystem from `dev`, parsing the superblock and group descriptors.
    pub fn mount(dev: D) -> Result<Self, Ext4Error> {
        let bsize = dev.block_size();
        let mut buf = vec![0u8; bsize];
        let sb = if bsize == 1024 {
            dev.read_block(1, &mut buf)?;
            Ext4Superblock::parse(&buf)?
        } else {
            dev.read_block(0, &mut buf)?;
            Ext4Superblock::parse(&buf[EXT4_SUPERBLOCK_OFFSET..])?
        };

        if sb.block_size() != bsize {
            return Err(Ext4Error::InvalidBlockSize);
        }

        let group_count = sb.group_count() as usize;
        let desc_size = usize::from(sb.desc_size);
        let is_64bit = sb.is_64bit();
        let gdt_start = sb.gdt_start_block();
        let descs_per_block = bsize / desc_size;

        let mut groups = Vec::with_capacity(group_count);
        let mut cur_block = u64::MAX;
        for g in 0..group_count {
            let blk = gdt_start + (g / descs_per_block) as u64;
            if blk != cur_block {
                dev.read_block(blk, &mut buf)?;
                cur_block = blk;
            }
            let off = (g % descs_per_block) * desc_size;
            groups.push(Ext4GroupDesc::parse(&buf[off..off + desc_size], is_64bit)?);
        }

        Ok(Self { dev, sb, groups })
    }

    /// Read an inode by 1-based inode number.
    pub fn read_inode(&self, ino: u32) -> Result<Ext4Inode, Ext4Error> {
        if ino == 0 || ino > self.sb.inodes_count {
            return Err(Ext4Error::NotFound);
        }
        let idx = ino - 1;
        let group = (idx / self.sb.inodes_per_group) as usize;
        let index_in_group = idx % self.sb.inodes_per_group;
        let gd = self
            .groups
            .get(group)
            .ok_or(Ext4Error::CorruptGroupDesc("group out of range"))?;

        let inode_size = usize::from(self.sb.inode_size);
        let bsize = self.dev.block_size();
        let byte_offset = usize::try_from(index_in_group)
            .map_err(|_| Ext4Error::CorruptInode("inode offset overflow"))?
            * inode_size;
        let block_num = gd.inode_table + (byte_offset / bsize) as u64;
        let off_in_block = byte_offset % bsize;

        let mut buf = vec![0u8; bsize];
        self.dev.read_block(block_num, &mut buf)?;
        Ext4Inode::parse(&buf[off_in_block..off_in_block + inode_size])
    }

    /// Write an inode back to its inode table block.
    pub fn write_inode(&mut self, ino: u32, inode: &Ext4Inode) -> Result<(), Ext4Error> {
        if ino == 0 || ino > self.sb.inodes_count {
            return Err(Ext4Error::NotFound);
        }
        let idx = ino - 1;
        let group = (idx / self.sb.inodes_per_group) as usize;
        let index_in_group = idx % self.sb.inodes_per_group;
        let gd = self
            .groups
            .get(group)
            .ok_or(Ext4Error::CorruptGroupDesc("group out of range"))?;

        let inode_size = usize::from(self.sb.inode_size);
        let bsize = self.dev.block_size();
        let byte_offset = (index_in_group as usize) * inode_size;
        let block_num = gd.inode_table + (byte_offset / bsize) as u64;
        let off_in_block = byte_offset % bsize;

        let mut buf = vec![0u8; bsize];
        self.dev.read_block(block_num, &mut buf)?;
        inode.serialize(&mut buf[off_in_block..off_in_block + inode_size])?;
        self.dev.write_block(block_num, &buf)?;
        Ok(())
    }

    /// Look up a child entry by name inside a directory inode.
    pub fn lookup_in_dir(
        &self,
        dir_inode: &Ext4Inode,
        name: &str,
    ) -> Result<Option<Ext4DirEntry>, Ext4Error> {
        if !dir_inode.is_dir() {
            return Err(Ext4Error::NotADirectory);
        }
        if !dir_inode.uses_extents() {
            return Err(Ext4Error::UnsupportedFeature);
        }
        let bsize = self.dev.block_size();
        let num_blocks = dir_inode.size.div_ceil(bsize as u64) as u32;
        let mut buf = vec![0u8; bsize];
        for lblock in 0..num_blocks {
            if let Some(mapping) = ExtentTree::lookup(&self.dev, &dir_inode.i_block, lblock)? {
                self.dev.read_block(mapping.physical_block, &mut buf)?;
                if let Some(entry) = lookup_in_dir_block(&buf, name)? {
                    return Ok(Some(entry));
                }
            }
        }
        Ok(None)
    }

    /// Resolve an absolute or relative path starting from `EXT4_ROOT_INO`.
    pub fn lookup_path(&self, path: &str) -> Result<(u32, Ext4Inode), Ext4Error> {
        let mut cur_ino = EXT4_ROOT_INO;
        let mut cur_inode = self.read_inode(cur_ino)?;
        for component in path.split('/').filter(|s| !s.is_empty()) {
            if !cur_inode.is_dir() {
                return Err(Ext4Error::NotADirectory);
            }
            let entry = self
                .lookup_in_dir(&cur_inode, component)?
                .ok_or(Ext4Error::NotFound)?;
            cur_ino = entry.inode;
            cur_inode = self.read_inode(cur_ino)?;
        }
        Ok((cur_ino, cur_inode))
    }

    /// Read file data from `inode` starting at `offset` into `out`.
    pub fn read_file(
        &self,
        inode: &Ext4Inode,
        offset: u64,
        out: &mut [u8],
    ) -> Result<usize, Ext4Error> {
        if inode.is_dir() {
            return Err(Ext4Error::IsADirectory);
        }
        if !inode.uses_extents() {
            return Err(Ext4Error::UnsupportedFeature);
        }
        if offset >= inode.size || out.is_empty() {
            return Ok(0);
        }
        let bsize = self.dev.block_size();
        let to_read = ((inode.size - offset) as usize).min(out.len());
        let mut read_total = 0usize;
        let mut blk_buf = vec![0u8; bsize];

        while read_total < to_read {
            let cur_pos = offset + read_total as u64;
            let lblock = (cur_pos / bsize as u64) as u32;
            let off_in_blk = (cur_pos % bsize as u64) as usize;
            let chunk = (bsize - off_in_blk).min(to_read - read_total);

            match ExtentTree::lookup(&self.dev, &inode.i_block, lblock)? {
                Some(mapping) if !mapping.unwritten => {
                    self.dev.read_block(mapping.physical_block, &mut blk_buf)?;
                    out[read_total..read_total + chunk]
                        .copy_from_slice(&blk_buf[off_in_blk..off_in_blk + chunk]);
                }
                _ => {
                    // Sparse hole or unwritten extent reads as zeroes.
                    out[read_total..read_total + chunk].fill(0);
                }
            }
            read_total += chunk;
        }
        Ok(read_total)
    }
}

// ============================================================================
// JBD2 Journal Parser & Writer (Big-Endian On-Disk Format)
// ============================================================================

/// JBD2 standard 12-byte block header (`journal_header_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Jbd2Header {
    pub magic: u32,
    pub blocktype: u32,
    pub sequence: u32,
}

impl Jbd2Header {
    pub fn parse(buf: &[u8]) -> Result<Self, Ext4Error> {
        if buf.len() < 12 {
            return Err(Ext4Error::BufferTooSmall);
        }
        let magic = read_u32_be(buf, 0);
        if magic != JBD2_MAGIC_NUMBER {
            return Err(Ext4Error::InvalidMagic);
        }
        Ok(Self {
            magic,
            blocktype: read_u32_be(buf, 4),
            sequence: read_u32_be(buf, 8),
        })
    }

    pub fn serialize(&self, buf: &mut [u8]) -> Result<(), Ext4Error> {
        if buf.len() < 12 {
            return Err(Ext4Error::BufferTooSmall);
        }
        write_u32_be(buf, 0, self.magic);
        write_u32_be(buf, 4, self.blocktype);
        write_u32_be(buf, 8, self.sequence);
        Ok(())
    }
}

/// JBD2 journal superblock (`journal_superblock_t`, 1024 bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jbd2Superblock {
    pub header: Jbd2Header,
    pub blocksize: u32,
    pub maxlen: u32,
    pub first: u32,
    pub sequence: u32,
    pub start: u32,
    pub errno: i32,
    pub feature_compat: u32,
    pub feature_incompat: u32,
    pub feature_ro_compat: u32,
    pub uuid: [u8; 16],
    pub nr_users: u32,
    pub dynsuper: u32,
    pub max_transaction: u32,
    pub max_trans_data: u32,
    pub checksum_type: u8,
    pub checksum: u32,
}

impl Jbd2Superblock {
    pub fn parse(buf: &[u8]) -> Result<Self, Ext4Error> {
        if buf.len() < 1024 {
            return Err(Ext4Error::BufferTooSmall);
        }
        let header = Jbd2Header::parse(&buf[..12])?;
        if header.blocktype != JBD2_SUPERBLOCK_V1 && header.blocktype != JBD2_SUPERBLOCK_V2 {
            return Err(Ext4Error::CorruptJournal("not a journal superblock"));
        }
        let blocksize = read_u32_be(buf, 0x0C);
        if !blocksize.is_power_of_two() || !(1024..=65536).contains(&blocksize) {
            return Err(Ext4Error::InvalidBlockSize);
        }
        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&buf[0x30..0x40]);

        Ok(Self {
            header,
            blocksize,
            maxlen: read_u32_be(buf, 0x10),
            first: read_u32_be(buf, 0x14),
            sequence: read_u32_be(buf, 0x18),
            start: read_u32_be(buf, 0x1C),
            errno: read_u32_be(buf, 0x20) as i32,
            feature_compat: read_u32_be(buf, 0x24),
            feature_incompat: read_u32_be(buf, 0x28),
            feature_ro_compat: read_u32_be(buf, 0x2C),
            uuid,
            nr_users: read_u32_be(buf, 0x40),
            dynsuper: read_u32_be(buf, 0x44),
            max_transaction: read_u32_be(buf, 0x48),
            max_trans_data: read_u32_be(buf, 0x4C),
            checksum_type: buf[0x50],
            checksum: read_u32_be(buf, 0xFC),
        })
    }

    pub fn serialize(&self, buf: &mut [u8]) -> Result<(), Ext4Error> {
        if buf.len() < 1024 {
            return Err(Ext4Error::BufferTooSmall);
        }
        buf[..1024].fill(0);
        self.header.serialize(&mut buf[..12])?;
        write_u32_be(buf, 0x0C, self.blocksize);
        write_u32_be(buf, 0x10, self.maxlen);
        write_u32_be(buf, 0x14, self.first);
        write_u32_be(buf, 0x18, self.sequence);
        write_u32_be(buf, 0x1C, self.start);
        write_u32_be(buf, 0x20, self.errno as u32);
        write_u32_be(buf, 0x24, self.feature_compat);
        write_u32_be(buf, 0x28, self.feature_incompat);
        write_u32_be(buf, 0x2C, self.feature_ro_compat);
        buf[0x30..0x40].copy_from_slice(&self.uuid);
        write_u32_be(buf, 0x40, self.nr_users);
        write_u32_be(buf, 0x44, self.dynsuper);
        write_u32_be(buf, 0x48, self.max_transaction);
        write_u32_be(buf, 0x4C, self.max_trans_data);
        buf[0x50] = self.checksum_type;
        write_u32_be(buf, 0xFC, self.checksum);
        Ok(())
    }

    pub fn is_64bit(&self) -> bool {
        (self.feature_incompat & JBD2_FEATURE_INCOMPAT_64BIT) != 0
    }

    pub fn has_csum_v3(&self) -> bool {
        (self.feature_incompat & JBD2_FEATURE_INCOMPAT_CSUM_V3) != 0
    }
}

/// JBD2 descriptor block tag (`journal_block_tag_t` / `journal_block_tag3_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Jbd2BlockTag {
    pub blocknr: u64,
    pub flags: u32,
    pub checksum: u32,
}

impl Jbd2BlockTag {
    /// Parse all tags from a JBD2 descriptor block.
    pub fn parse_descriptor_block(
        block: &[u8],
        is_64bit: bool,
        csum_v3: bool,
    ) -> Result<(Jbd2Header, Vec<Self>), Ext4Error> {
        let header = Jbd2Header::parse(&block[..12])?;
        if header.blocktype != JBD2_DESCRIPTOR_BLOCK {
            return Err(Ext4Error::CorruptJournal("not a descriptor block"));
        }
        let mut tags = Vec::new();
        let mut off = 12usize;
        let tag_size = if csum_v3 {
            16
        } else if is_64bit {
            12
        } else {
            8
        };

        while off + tag_size <= block.len() {
            let (blocknr, flags, checksum) = if csum_v3 {
                let lo = u64::from(read_u32_be(block, off));
                let flags = read_u32_be(block, off + 4);
                let hi = u64::from(read_u32_be(block, off + 8));
                let csum = read_u32_be(block, off + 12);
                (lo | (hi << 32), flags, csum)
            } else if is_64bit {
                let lo = u64::from(read_u32_be(block, off));
                let csum_flags = read_u32_be(block, off + 4);
                let hi = u64::from(read_u32_be(block, off + 8));
                (lo | (hi << 32), csum_flags & 0xFFFF, csum_flags >> 16)
            } else {
                let lo = u64::from(read_u32_be(block, off));
                let csum_flags = read_u32_be(block, off + 4);
                (lo, csum_flags & 0xFFFF, csum_flags >> 16)
            };

            tags.push(Self {
                blocknr,
                flags,
                checksum,
            });
            off += tag_size;
            if (flags & JBD2_FLAG_SAME_UUID) == 0 {
                off += 16;
            }
            if (flags & JBD2_FLAG_LAST_TAG) != 0 {
                break;
            }
        }
        Ok((header, tags))
    }

    /// Serialize a descriptor block with `sequence` and `tags`.
    pub fn write_descriptor_block(
        block: &mut [u8],
        sequence: u32,
        tags: &[Self],
        is_64bit: bool,
        csum_v3: bool,
    ) -> Result<(), Ext4Error> {
        if block.len() < 64 {
            return Err(Ext4Error::BufferTooSmall);
        }
        block.fill(0);
        let hdr = Jbd2Header {
            magic: JBD2_MAGIC_NUMBER,
            blocktype: JBD2_DESCRIPTOR_BLOCK,
            sequence,
        };
        hdr.serialize(&mut block[..12])?;

        let tag_size = if csum_v3 {
            16
        } else if is_64bit {
            12
        } else {
            8
        };
        let mut off = 12usize;
        for (i, tag) in tags.iter().enumerate() {
            if off + tag_size > block.len() {
                return Err(Ext4Error::NoSpace);
            }
            let mut flags = tag.flags | JBD2_FLAG_SAME_UUID;
            if i + 1 == tags.len() {
                flags |= JBD2_FLAG_LAST_TAG;
            } else {
                flags &= !JBD2_FLAG_LAST_TAG;
            }

            if csum_v3 {
                write_u32_be(block, off, tag.blocknr as u32);
                write_u32_be(block, off + 4, flags);
                write_u32_be(block, off + 8, (tag.blocknr >> 32) as u32);
                write_u32_be(block, off + 12, tag.checksum);
            } else if is_64bit {
                write_u32_be(block, off, tag.blocknr as u32);
                write_u32_be(block, off + 4, (tag.checksum << 16) | (flags & 0xFFFF));
                write_u32_be(block, off + 8, (tag.blocknr >> 32) as u32);
            } else {
                write_u32_be(block, off, tag.blocknr as u32);
                write_u32_be(block, off + 4, (tag.checksum << 16) | (flags & 0xFFFF));
            }
            off += tag_size;
        }
        Ok(())
    }
}

/// JBD2 commit block header (`commit_header`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Jbd2CommitHeader {
    pub header: Jbd2Header,
    pub chksum_type: u8,
    pub chksum_size: u8,
    pub chksum: [u32; 8],
    pub commit_sec: u64,
    pub commit_nsec: u32,
}

impl Jbd2CommitHeader {
    pub fn parse(buf: &[u8]) -> Result<Self, Ext4Error> {
        if buf.len() < 64 {
            return Err(Ext4Error::BufferTooSmall);
        }
        let header = Jbd2Header::parse(&buf[..12])?;
        if header.blocktype != JBD2_COMMIT_BLOCK {
            return Err(Ext4Error::CorruptJournal("not a commit block"));
        }
        let mut chksum = [0u32; 8];
        for (i, slot) in chksum.iter_mut().enumerate() {
            *slot = read_u32_be(buf, 0x10 + i * 4);
        }
        Ok(Self {
            header,
            chksum_type: buf[0x0C],
            chksum_size: buf[0x0D],
            chksum,
            commit_sec: read_u64_be(buf, 0x30),
            commit_nsec: read_u32_be(buf, 0x38),
        })
    }

    pub fn serialize(&self, buf: &mut [u8]) -> Result<(), Ext4Error> {
        if buf.len() < 64 {
            return Err(Ext4Error::BufferTooSmall);
        }
        buf.fill(0);
        self.header.serialize(&mut buf[..12])?;
        buf[0x0C] = self.chksum_type;
        buf[0x0D] = self.chksum_size;
        for (i, &val) in self.chksum.iter().enumerate() {
            write_u32_be(buf, 0x10 + i * 4, val);
        }
        write_u64_be(buf, 0x30, self.commit_sec);
        write_u32_be(buf, 0x38, self.commit_nsec);
        Ok(())
    }
}

/// JBD2 revoke block (`jbd2_journal_revoke_header_t` + block numbers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jbd2RevokeBlock {
    pub header: Jbd2Header,
    pub revoked_blocks: Vec<u64>,
}

impl Jbd2RevokeBlock {
    pub fn parse(buf: &[u8], is_64bit: bool) -> Result<Self, Ext4Error> {
        if buf.len() < 16 {
            return Err(Ext4Error::BufferTooSmall);
        }
        let header = Jbd2Header::parse(&buf[..12])?;
        if header.blocktype != JBD2_REVOKE_BLOCK {
            return Err(Ext4Error::CorruptJournal("not a revoke block"));
        }
        let r_count = read_u32_be(buf, 12) as usize;
        if r_count < 16 || r_count > buf.len() {
            return Err(Ext4Error::CorruptJournal("invalid revoke r_count"));
        }
        let entry_size = if is_64bit { 8 } else { 4 };
        let mut revoked_blocks = Vec::new();
        let mut off = 16usize;
        while off + entry_size <= r_count {
            let blk = if is_64bit {
                read_u64_be(buf, off)
            } else {
                u64::from(read_u32_be(buf, off))
            };
            revoked_blocks.push(blk);
            off += entry_size;
        }
        Ok(Self {
            header,
            revoked_blocks,
        })
    }

    pub fn serialize(&self, buf: &mut [u8], is_64bit: bool) -> Result<(), Ext4Error> {
        let entry_size = if is_64bit { 8 } else { 4 };
        let r_count = 16 + self.revoked_blocks.len() * entry_size;
        if buf.len() < r_count {
            return Err(Ext4Error::BufferTooSmall);
        }
        buf.fill(0);
        self.header.serialize(&mut buf[..12])?;
        write_u32_be(buf, 12, r_count as u32);
        let mut off = 16usize;
        for &blk in &self.revoked_blocks {
            if is_64bit {
                write_u64_be(buf, off, blk);
            } else {
                write_u32_be(buf, off, blk as u32);
            }
            off += entry_size;
        }
        Ok(())
    }
}

/// Parsed JBD2 transaction (descriptor tags + data blocks + optional commit).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jbd2Transaction {
    pub sequence: u32,
    pub tags: Vec<Jbd2BlockTag>,
    pub data_blocks: Vec<Vec<u8>>,
    pub revoked: Vec<u64>,
    pub committed: bool,
    pub commit_sec: u64,
}

/// Parse a JBD2 transaction starting at `start_block` in `dev`.
pub fn parse_jbd2_transaction(
    dev: &dyn BlockDevice,
    jsb: &Jbd2Superblock,
    start_block: u64,
) -> Result<(Jbd2Transaction, u64), Ext4Error> {
    let bsize = dev.block_size();
    let mut buf = vec![0u8; bsize];
    let mut cur = start_block;
    let mut tx = Jbd2Transaction {
        sequence: 0,
        tags: Vec::new(),
        data_blocks: Vec::new(),
        revoked: Vec::new(),
        committed: false,
        commit_sec: 0,
    };

    let mut first = true;
    loop {
        if cur >= u64::from(jsb.maxlen) {
            cur = u64::from(jsb.first);
        }
        dev.read_block(cur, &mut buf)?;
        let hdr = Jbd2Header::parse(&buf[..12])?;
        if first {
            tx.sequence = hdr.sequence;
            first = false;
        } else if hdr.sequence != tx.sequence {
            break;
        }
        cur += 1;

        match hdr.blocktype {
            JBD2_DESCRIPTOR_BLOCK => {
                let (_, tags) =
                    Jbd2BlockTag::parse_descriptor_block(&buf, jsb.is_64bit(), jsb.has_csum_v3())?;
                for _tag in &tags {
                    if cur >= u64::from(jsb.maxlen) {
                        cur = u64::from(jsb.first);
                    }
                    let mut data = vec![0u8; bsize];
                    dev.read_block(cur, &mut data)?;
                    cur += 1;
                    tx.data_blocks.push(data);
                }
                tx.tags.extend(tags);
            }
            JBD2_REVOKE_BLOCK => {
                let rev = Jbd2RevokeBlock::parse(&buf, jsb.is_64bit())?;
                tx.revoked.extend(rev.revoked_blocks);
            }
            JBD2_COMMIT_BLOCK => {
                let commit = Jbd2CommitHeader::parse(&buf)?;
                tx.committed = true;
                tx.commit_sec = commit.commit_sec;
                break;
            }
            _ => return Err(Ext4Error::CorruptJournal("unexpected journal block type")),
        }
    }

    Ok((tx, cur))
}

/// Write a complete JBD2 transaction (descriptor + data blocks + commit) into `dev`.
pub fn write_jbd2_transaction(
    dev: &mut dyn BlockDevice,
    jsb: &Jbd2Superblock,
    start_block: u64,
    sequence: u32,
    blocks: &[(u64, &[u8])],
    commit_sec: u64,
) -> Result<u64, Ext4Error> {
    let bsize = dev.block_size();
    let mut cur = start_block;
    let mut tags = Vec::with_capacity(blocks.len());
    for &(fs_blk, data) in blocks {
        if data.len() < bsize {
            return Err(Ext4Error::BufferTooSmall);
        }
        let csum = crc32c_raw(0xFFFF_FFFF, &data[..bsize]);
        tags.push(Jbd2BlockTag {
            blocknr: fs_blk,
            flags: JBD2_FLAG_SAME_UUID,
            checksum: csum,
        });
    }

    let mut desc_buf = vec![0u8; bsize];
    Jbd2BlockTag::write_descriptor_block(
        &mut desc_buf,
        sequence,
        &tags,
        jsb.is_64bit(),
        jsb.has_csum_v3(),
    )?;
    if cur >= u64::from(jsb.maxlen) {
        cur = u64::from(jsb.first);
    }
    dev.write_block(cur, &desc_buf)?;
    cur += 1;

    for &(_, data) in blocks {
        if cur >= u64::from(jsb.maxlen) {
            cur = u64::from(jsb.first);
        }
        dev.write_block(cur, &data[..bsize])?;
        cur += 1;
    }

    let commit = Jbd2CommitHeader {
        header: Jbd2Header {
            magic: JBD2_MAGIC_NUMBER,
            blocktype: JBD2_COMMIT_BLOCK,
            sequence,
        },
        chksum_type: 4,
        chksum_size: 4,
        chksum: [0; 8],
        commit_sec,
        commit_nsec: 0,
    };
    let mut commit_buf = vec![0u8; bsize];
    commit.serialize(&mut commit_buf)?;
    if cur >= u64::from(jsb.maxlen) {
        cur = u64::from(jsb.first);
    }
    dev.write_block(cur, &commit_buf)?;
    cur += 1;

    Ok(cur)
}

/// Self-test exercising ext4 superblock, group descriptors, extent trees,
/// directory lookup/writing, and JBD2 journal parsing/writing.
pub fn self_test() -> Result<(), Ext4Error> {
    serial_println!("[ext4] Running clean-room ext4 & JBD2 self-test...");

    let bsize = 4096usize;
    let mut dev = MemBlockDevice::new(bsize, 64)?;

    // 1. Build and serialize a 64-bit ext4 superblock with metadata_csum & extents.
    let sb = Ext4Superblock {
        inodes_count: 128,
        blocks_count: 64,
        r_blocks_count: 0,
        free_blocks_count: 48,
        free_inodes_count: 120,
        first_data_block: 0,
        log_block_size: 2, // 4096
        log_cluster_size: 2,
        blocks_per_group: 64,
        clusters_per_group: 64,
        inodes_per_group: 128,
        mtime: 0,
        wtime: 0,
        mnt_count: 1,
        max_mnt_count: 20,
        magic: EXT4_SUPER_MAGIC,
        state: 1,
        errors: 1,
        minor_rev_level: 0,
        rev_level: 1,
        first_ino: 11,
        inode_size: 256,
        block_group_nr: 0,
        feature_compat: EXT4_FEATURE_COMPAT_HAS_JOURNAL,
        feature_incompat: EXT4_FEATURE_INCOMPAT_FILETYPE
            | EXT4_FEATURE_INCOMPAT_EXTENTS
            | EXT4_FEATURE_INCOMPAT_64BIT,
        feature_ro_compat: EXT4_FEATURE_RO_COMPAT_METADATA_CSUM,
        uuid: [0x42; 16],
        volume_name: *b"nexus-ext4\0\0\0\0\0\0",
        journal_inum: EXT4_JOURNAL_INO,
        desc_size: 64,
        checksum_type: 1,
        checksum: 0,
    };

    let mut blk0 = vec![0u8; bsize];
    sb.serialize(&mut blk0[EXT4_SUPERBLOCK_OFFSET..])?;
    dev.write_block(0, &blk0)?;

    // 2. Write Block Group 0 Descriptor at block 1 (inode table at block 4).
    let gd = Ext4GroupDesc {
        block_bitmap: 2,
        inode_bitmap: 3,
        inode_table: 4,
        free_blocks_count: 48,
        free_inodes_count: 120,
        used_dirs_count: 1,
        flags: 0,
        itable_unused: 118,
        checksum: 0,
    };
    let mut blk1 = vec![0u8; bsize];
    gd.serialize(&mut blk1, true)?;
    dev.write_block(1, &blk1)?;

    // 3. Create root directory (inode 2) at data block 12, and file (inode 12) at data blocks 13..18.
    let mut root_dir_blk = vec![0u8; bsize];
    init_dir_block(&mut root_dir_blk, EXT4_ROOT_INO, EXT4_ROOT_INO)?;
    append_dir_entry(&mut root_dir_blk, 12, EXT4_FT_REG_FILE, "hello.txt")?;
    dev.write_block(12, &root_dir_blk)?;

    let mut root_inode = Ext4Inode {
        mode: S_IFDIR | 0o755,
        uid: 0,
        size: bsize as u64,
        atime: 0,
        ctime: 0,
        mtime: 0,
        dtime: 0,
        gid: 0,
        links_count: 2,
        blocks_lo: 8,
        flags: EXT4_EXTENTS_FL,
        i_block: [0u8; 60],
        generation: 1,
        extra_isize: 32,
    };
    ExtentTree::init_root(&mut root_inode.i_block);
    let mut next_free_blk = 20u64;
    ExtentTree::insert(
        &mut dev,
        &mut root_inode.i_block,
        Ext4Extent::from_parts(0, 12, 1, false),
        || {
            let b = next_free_blk;
            next_free_blk += 1;
            Ok(b)
        },
    )?;

    // Write file data to block 13 and test extent tree split (>4 non-contiguous extents).
    let payload = b"Nexus OS clean-room ext4 extent tree!";
    let mut file_blk = vec![0u8; bsize];
    file_blk[..payload.len()].copy_from_slice(payload);
    dev.write_block(13, &file_blk)?;

    let mut file_inode = Ext4Inode {
        mode: S_IFREG | 0o644,
        uid: 0,
        size: payload.len() as u64,
        atime: 0,
        ctime: 0,
        mtime: 0,
        dtime: 0,
        gid: 0,
        links_count: 1,
        blocks_lo: 8,
        flags: EXT4_EXTENTS_FL,
        i_block: [0u8; 60],
        generation: 1,
        extra_isize: 32,
    };
    ExtentTree::init_root(&mut file_inode.i_block);
    // Insert 5 non-contiguous extents to trigger root-to-depth-1 split.
    let test_extents = [
        Ext4Extent::from_parts(0, 13, 1, false),
        Ext4Extent::from_parts(2, 15, 1, false),
        Ext4Extent::from_parts(4, 17, 1, false),
        Ext4Extent::from_parts(6, 19, 1, false),
        Ext4Extent::from_parts(8, 25, 1, false),
    ];
    for ext in test_extents {
        ExtentTree::insert(&mut dev, &mut file_inode.i_block, ext, || {
            let b = next_free_blk;
            next_free_blk += 1;
            Ok(b)
        })?;
    }

    // Mount via Ext4Fs, write inodes, and verify path lookup + file read.
    let mut fs = Ext4Fs::mount(dev)?;
    fs.write_inode(EXT4_ROOT_INO, &root_inode)?;
    fs.write_inode(12, &file_inode)?;

    let (found_ino, found_inode) = fs.lookup_path("/hello.txt")?;
    if found_ino != 12 {
        return Err(Ext4Error::NotFound);
    }
    let mut read_back = vec![0u8; payload.len()];
    let n = fs.read_file(&found_inode, 0, &mut read_back)?;
    if n != payload.len() || read_back.as_slice() != payload {
        return Err(Ext4Error::CorruptExtent("file read mismatch"));
    }
    // Verify depth-1 extent lookup for logical block 8 -> physical block 25.
    let map8 = ExtentTree::lookup(&fs.dev, &found_inode.i_block, 8)?
        .ok_or(Ext4Error::CorruptExtent("missing split extent"))?;
    if map8.physical_block != 25 {
        return Err(Ext4Error::CorruptExtent("wrong split extent target"));
    }

    // 4. JBD2 Journal Superblock, Transaction Write & Parse, and Revoke Block test.
    let mut jdev = MemBlockDevice::new(bsize, 16)?;
    let jsb = Jbd2Superblock {
        header: Jbd2Header {
            magic: JBD2_MAGIC_NUMBER,
            blocktype: JBD2_SUPERBLOCK_V2,
            sequence: 1,
        },
        blocksize: bsize as u32,
        maxlen: 16,
        first: 1,
        sequence: 1,
        start: 1,
        errno: 0,
        feature_compat: 0,
        feature_incompat: JBD2_FEATURE_INCOMPAT_REVOKE
            | JBD2_FEATURE_INCOMPAT_64BIT
            | JBD2_FEATURE_INCOMPAT_CSUM_V3,
        feature_ro_compat: 0,
        uuid: [0x77; 16],
        nr_users: 1,
        dynsuper: 0,
        max_transaction: 4,
        max_trans_data: 4,
        checksum_type: 4,
        checksum: 0,
    };
    let mut jsb_buf = vec![0u8; bsize];
    jsb.serialize(&mut jsb_buf)?;
    jdev.write_block(0, &jsb_buf)?;

    let mut j_read_buf = vec![0u8; bsize];
    jdev.read_block(0, &mut j_read_buf)?;
    let parsed_jsb = Jbd2Superblock::parse(&j_read_buf)?;
    if parsed_jsb.maxlen != 16 || !parsed_jsb.has_csum_v3() {
        return Err(Ext4Error::CorruptJournal("jsb roundtrip mismatch"));
    }

    let mut tx_data = vec![0xABu8; bsize];
    tx_data[0..4].copy_from_slice(b"JBD2");
    let next_jblk = write_jbd2_transaction(
        &mut jdev,
        &parsed_jsb,
        1,
        42,
        &[(13, &tx_data)],
        1_700_000_000,
    )?;
    if next_jblk != 4 {
        return Err(Ext4Error::CorruptJournal("unexpected journal cursor"));
    }

    let (parsed_tx, _) = parse_jbd2_transaction(&jdev, &parsed_jsb, 1)?;
    if !parsed_tx.committed
        || parsed_tx.sequence != 42
        || parsed_tx.tags.len() != 1
        || parsed_tx.tags[0].blocknr != 13
        || parsed_tx.data_blocks[0][0..4] != *b"JBD2"
    {
        return Err(Ext4Error::CorruptJournal("transaction roundtrip mismatch"));
    }

    let rev = Jbd2RevokeBlock {
        header: Jbd2Header {
            magic: JBD2_MAGIC_NUMBER,
            blocktype: JBD2_REVOKE_BLOCK,
            sequence: 43,
        },
        revoked_blocks: vec![13, 99],
    };
    let mut rev_buf = vec![0u8; bsize];
    rev.serialize(&mut rev_buf, true)?;
    let parsed_rev = Jbd2RevokeBlock::parse(&rev_buf, true)?;
    if parsed_rev.revoked_blocks != [13, 99] {
        return Err(Ext4Error::CorruptJournal("revoke roundtrip mismatch"));
    }

    serial_println!("[ext4] Self-test PASSED (superblock, GDT, extent split, dir, JBD2).");
    Ok(())
}
