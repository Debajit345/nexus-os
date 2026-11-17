use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

/// Standard 512-byte logical sector size.
pub const SECTOR_SIZE: usize = 512;

/// Errors returned by block I/O, FAT32, and NexusFS operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsError {
    /// Sector or byte offset is outside the device or buffer bounds.
    OutOfBounds,
    /// Underlying block device I/O failure.
    IoError,
    /// Missing or invalid 0x55AA boot sector signature.
    InvalidBootSignature,
    /// Malformed BIOS Parameter Block.
    InvalidBpb,
    /// Sector size is not 512 bytes.
    UnsupportedSectorSize,
    /// Cluster size is zero or not a power of two.
    UnsupportedClusterSize,
    /// Volume is not FAT32 (e.g. FAT12/FAT16).
    NotFat32,
    /// Encountered a free, reserved, or bad cluster in a chain.
    InvalidCluster,
    /// Cluster chain exceeded total cluster count (cycle detected).
    ClusterChainLoop,
    /// Requested file or directory was not found.
    NotFound,
    /// Expected a regular file, found a directory.
    IsADirectory,
    /// Expected a directory, found a regular file.
    NotADirectory,
    /// Invalid path syntax.
    InvalidPath,
    /// NexusFS superblock magic mismatch.
    InvalidSuperblockMagic,
    /// Unsupported NexusFS format version.
    UnsupportedVersion,
    /// Invalid NexusFS block size.
    InvalidBlockSize,
    /// Metadata checksum verification failed.
    ChecksumMismatch,
    /// Metadata fields failed structural validation.
    CorruptMetadata,
}

impl fmt::Display for FsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::OutOfBounds => "block or buffer access out of bounds",
            Self::IoError => "block device I/O error",
            Self::InvalidBootSignature => "invalid boot sector signature",
            Self::InvalidBpb => "invalid BIOS parameter block",
            Self::UnsupportedSectorSize => "unsupported sector size",
            Self::UnsupportedClusterSize => "unsupported cluster size",
            Self::NotFat32 => "volume is not FAT32",
            Self::InvalidCluster => "invalid cluster number in FAT chain",
            Self::ClusterChainLoop => "cycle detected in FAT cluster chain",
            Self::NotFound => "file or directory not found",
            Self::IsADirectory => "target is a directory",
            Self::NotADirectory => "target is not a directory",
            Self::InvalidPath => "invalid path",
            Self::InvalidSuperblockMagic => "invalid NexusFS superblock magic",
            Self::UnsupportedVersion => "unsupported NexusFS version",
            Self::InvalidBlockSize => "invalid NexusFS block size",
            Self::ChecksumMismatch => "NexusFS metadata checksum mismatch",
            Self::CorruptMetadata => "corrupt NexusFS metadata",
        };
        f.write_str(msg)
    }
}

/// Sector-level block device abstraction.
pub trait BlockIo {
    /// Read a single 512-byte sector at `lba` into `buf`.
    fn read_sector(&mut self, lba: u64, buf: &mut [u8; 512]) -> Result<(), FsError>;

    /// Write a single 512-byte sector from `buf` to `lba`.
    fn write_sector(&mut self, lba: u64, buf: &[u8; 512]) -> Result<(), FsError>;
}

/// In-memory 512-byte sector block device for testing and ramdisks.
#[derive(Debug, Clone)]
pub struct MemBlockIo {
    data: Vec<u8>,
}

impl MemBlockIo {
    /// Create a zero-initialized in-memory block device with `num_sectors` 512-byte sectors.
    pub fn new(num_sectors: usize) -> Self {
        Self {
            data: vec![0u8; num_sectors.saturating_mul(SECTOR_SIZE)],
        }
    }

    /// Wrap an existing byte buffer whose length is a multiple of 512 bytes.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, FsError> {
        if !data.len().is_multiple_of(SECTOR_SIZE) {
            return Err(FsError::OutOfBounds);
        }
        Ok(Self { data })
    }

    /// Total number of 512-byte sectors in this device.
    pub fn sector_count(&self) -> u64 {
        (self.data.len() / SECTOR_SIZE) as u64
    }

    /// Immutable view of the backing byte buffer.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Mutable view of the backing byte buffer.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl BlockIo for MemBlockIo {
    fn read_sector(&mut self, lba: u64, buf: &mut [u8; 512]) -> Result<(), FsError> {
        let offset = usize::try_from(lba)
            .ok()
            .and_then(|s| s.checked_mul(SECTOR_SIZE))
            .ok_or(FsError::OutOfBounds)?;
        let end = offset
            .checked_add(SECTOR_SIZE)
            .ok_or(FsError::OutOfBounds)?;
        let slice = self.data.get(offset..end).ok_or(FsError::OutOfBounds)?;
        buf.copy_from_slice(slice);
        Ok(())
    }

    fn write_sector(&mut self, lba: u64, buf: &[u8; 512]) -> Result<(), FsError> {
        let offset = usize::try_from(lba)
            .ok()
            .and_then(|s| s.checked_mul(SECTOR_SIZE))
            .ok_or(FsError::OutOfBounds)?;
        let end = offset
            .checked_add(SECTOR_SIZE)
            .ok_or(FsError::OutOfBounds)?;
        let slice = self.data.get_mut(offset..end).ok_or(FsError::OutOfBounds)?;
        slice.copy_from_slice(buf);
        Ok(())
    }
}
