#![allow(dead_code)]

//! Filesystem mount bridge between `nexus-fs` and `crate::blk::BlockDevice`.

use nexus_fs::{
    validate_superblock, BlockIo, FsError, MemBlockIo, NexusSuperblock, NEXUSFS_DEFAULT_BLOCK_SIZE,
    SECTOR_SIZE,
};

use crate::blk::BlockDevice;
use crate::serial_println;

/// Adapter exposing any kernel [`BlockDevice`] (NVMe, VirtIO-Blk, etc.) as a
/// `nexus_fs::BlockIo` 512-byte sector device.
pub struct BlkIoAdapter<'a, D: BlockDevice> {
    device: &'a D,
}

impl<'a, D: BlockDevice> BlkIoAdapter<'a, D> {
    /// Wrap a reference to a kernel [`BlockDevice`].
    pub fn new(device: &'a D) -> Self {
        Self { device }
    }

    /// Return the underlying [`BlockDevice`] reference.
    pub fn device(&self) -> &'a D {
        self.device
    }
}

impl<'a, D: BlockDevice> BlockIo for BlkIoAdapter<'a, D> {
    fn read_sector(&mut self, lba: u64, buf: &mut [u8; 512]) -> Result<(), FsError> {
        if self.device.block_size() != SECTOR_SIZE {
            return Err(FsError::UnsupportedSectorSize);
        }
        if lba >= self.device.block_count() {
            return Err(FsError::OutOfBounds);
        }
        self.device
            .read_blocks(lba, buf)
            .map_err(|_| FsError::IoError)
    }

    fn write_sector(&mut self, lba: u64, buf: &[u8; 512]) -> Result<(), FsError> {
        if self.device.block_size() != SECTOR_SIZE {
            return Err(FsError::UnsupportedSectorSize);
        }
        if lba >= self.device.block_count() {
            return Err(FsError::OutOfBounds);
        }
        self.device
            .write_blocks(lba, buf)
            .map_err(|_| FsError::IoError)
    }
}

/// Run an early-boot self-test for `nexus-fs` superblock creation, sector
/// round-trip via [`MemBlockIo`], and [`validate_superblock`].
pub fn run_fs_selftest() {
    let uuid = [
        0x4E, 0x45, 0x58, 0x55, 0x53, 0x4F, 0x53, 0x21, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
        0x07,
    ];
    let sb = NexusSuperblock::new(1024, NEXUSFS_DEFAULT_BLOCK_SIZE, uuid);

    if let Err(err) = validate_superblock(&sb) {
        serial_println!("FS: NexusFS superblock validation failed: {}", err);
        return;
    }

    let mut mem = MemBlockIo::new(4);
    let raw = sb.to_bytes();
    if let Err(err) = mem.write_sector(0, &raw) {
        serial_println!("FS: MemBlockIo write_sector failed: {}", err);
        return;
    }

    let mut read_buf = [0u8; SECTOR_SIZE];
    if let Err(err) = mem.read_sector(0, &mut read_buf) {
        serial_println!("FS: MemBlockIo read_sector failed: {}", err);
        return;
    }

    match NexusSuperblock::from_bytes(&read_buf) {
        Ok(parsed) => {
            if let Err(err) = validate_superblock(&parsed) {
                serial_println!("FS: NexusFS round-trip validation failed: {}", err);
            } else {
                serial_println!(
                    "FS: NexusFS selftest passed (version={}, block_size={}, total_blocks={}, crc32={:#010x})",
                    parsed.version,
                    parsed.block_size,
                    parsed.total_blocks,
                    parsed.checksum
                );
            }
        }
        Err(err) => {
            serial_println!("FS: NexusFS superblock decode failed: {}", err);
        }
    }
}
