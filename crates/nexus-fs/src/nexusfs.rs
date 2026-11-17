use crate::block::{FsError, SECTOR_SIZE};

/// Magic bytes identifying a NexusFS superblock (`"NEXUSFS\0"`).
pub const NEXUSFS_MAGIC: [u8; 8] = *b"NEXUSFS\0";

/// Current NexusFS on-disk format version.
pub const NEXUSFS_VERSION: u32 = 1;

/// Default NexusFS logical block size (4 KiB).
pub const NEXUSFS_DEFAULT_BLOCK_SIZE: u32 = 4096;

/// Serialized size of a NexusFS superblock header (fits in one 512-byte sector).
pub const NEXUSFS_SUPERBLOCK_SIZE: usize = SECTOR_SIZE;

/// Serialized size of a NexusFS inode record.
pub const NEXUSFS_INODE_SIZE: usize = 128;

/// Copy-on-write superblock for NexusFS native storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NexusSuperblock {
    pub magic: [u8; 8],
    pub version: u32,
    pub block_size: u32,
    pub total_blocks: u64,
    pub generation: u64,
    pub root_inode: u64,
    pub inode_tree_root: u64,
    pub free_tree_root: u64,
    pub uuid: [u8; 16],
    pub checksum: u32,
}

impl NexusSuperblock {
    /// Construct a new valid NexusFS superblock with a computed checksum.
    pub fn new(total_blocks: u64, block_size: u32, uuid: [u8; 16]) -> Self {
        let mut sb = Self {
            magic: NEXUSFS_MAGIC,
            version: NEXUSFS_VERSION,
            block_size,
            total_blocks,
            generation: 1,
            root_inode: 1,
            inode_tree_root: 1,
            free_tree_root: 2,
            uuid,
            checksum: 0,
        };
        sb.checksum = sb.compute_checksum();
        sb
    }

    /// Compute the CRC32 checksum over the canonical superblock fields.
    pub fn compute_checksum(&self) -> u32 {
        let mut buf = [0u8; 76];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.block_size.to_le_bytes());
        buf[16..24].copy_from_slice(&self.total_blocks.to_le_bytes());
        buf[24..32].copy_from_slice(&self.generation.to_le_bytes());
        buf[32..40].copy_from_slice(&self.root_inode.to_le_bytes());
        buf[40..48].copy_from_slice(&self.inode_tree_root.to_le_bytes());
        buf[48..56].copy_from_slice(&self.free_tree_root.to_le_bytes());
        buf[56..72].copy_from_slice(&self.uuid);
        // bytes 72..76 are reserved/zeroed during checksum calculation
        crc32_ieee(&buf)
    }

    /// Serialize the superblock into a 512-byte sector buffer.
    pub fn to_bytes(&self) -> [u8; NEXUSFS_SUPERBLOCK_SIZE] {
        let mut out = [0u8; NEXUSFS_SUPERBLOCK_SIZE];
        out[0..8].copy_from_slice(&self.magic);
        out[8..12].copy_from_slice(&self.version.to_le_bytes());
        out[12..16].copy_from_slice(&self.block_size.to_le_bytes());
        out[16..24].copy_from_slice(&self.total_blocks.to_le_bytes());
        out[24..32].copy_from_slice(&self.generation.to_le_bytes());
        out[32..40].copy_from_slice(&self.root_inode.to_le_bytes());
        out[40..48].copy_from_slice(&self.inode_tree_root.to_le_bytes());
        out[48..56].copy_from_slice(&self.free_tree_root.to_le_bytes());
        out[56..72].copy_from_slice(&self.uuid);
        out[72..76].copy_from_slice(&self.checksum.to_le_bytes());
        out
    }

    /// Deserialize a superblock from a 512-byte sector buffer and validate it.
    pub fn from_bytes(buf: &[u8; NEXUSFS_SUPERBLOCK_SIZE]) -> Result<Self, FsError> {
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&buf[0..8]);
        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&buf[56..72]);

        let sb = Self {
            magic,
            version: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            block_size: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            total_blocks: u64::from_le_bytes([
                buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
            ]),
            generation: u64::from_le_bytes([
                buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
            ]),
            root_inode: u64::from_le_bytes([
                buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
            ]),
            inode_tree_root: u64::from_le_bytes([
                buf[40], buf[41], buf[42], buf[43], buf[44], buf[45], buf[46], buf[47],
            ]),
            free_tree_root: u64::from_le_bytes([
                buf[48], buf[49], buf[50], buf[51], buf[52], buf[53], buf[54], buf[55],
            ]),
            uuid,
            checksum: u32::from_le_bytes([buf[72], buf[73], buf[74], buf[75]]),
        };

        validate_superblock(&sb)?;
        Ok(sb)
    }
}

/// Validate a [`NexusSuperblock`] for magic, version, geometry, and checksum integrity.
pub fn validate_superblock(sb: &NexusSuperblock) -> Result<(), FsError> {
    if sb.magic != NEXUSFS_MAGIC {
        return Err(FsError::InvalidSuperblockMagic);
    }
    if sb.version != NEXUSFS_VERSION {
        return Err(FsError::UnsupportedVersion);
    }
    if !sb.block_size.is_power_of_two() || !(512..=65536).contains(&sb.block_size) {
        return Err(FsError::InvalidBlockSize);
    }
    if sb.total_blocks < 4 || sb.generation == 0 || sb.root_inode == 0 {
        return Err(FsError::CorruptMetadata);
    }
    if sb.inode_tree_root >= sb.total_blocks || sb.free_tree_root >= sb.total_blocks {
        return Err(FsError::CorruptMetadata);
    }
    if sb.checksum != sb.compute_checksum() {
        return Err(FsError::ChecksumMismatch);
    }
    Ok(())
}

/// Kind of object represented by a [`NexusInode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NexusInodeKind {
    File = 1,
    Directory = 2,
    Symlink = 3,
}

impl NexusInodeKind {
    fn from_u8(val: u8) -> Result<Self, FsError> {
        match val {
            1 => Ok(Self::File),
            2 => Ok(Self::Directory),
            3 => Ok(Self::Symlink),
            _ => Err(FsError::CorruptMetadata),
        }
    }
}

/// Copy-on-write inode metadata record for NexusFS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NexusInode {
    pub inode_id: u64,
    pub kind: NexusInodeKind,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub generation: u64,
    pub extent_root: u64,
    pub created_ns: u64,
    pub modified_ns: u64,
    pub links: u32,
    pub flags: u32,
}

impl NexusInode {
    /// Serialize the inode into a fixed 128-byte record.
    pub fn to_bytes(&self) -> [u8; NEXUSFS_INODE_SIZE] {
        let mut out = [0u8; NEXUSFS_INODE_SIZE];
        out[0..8].copy_from_slice(&self.inode_id.to_le_bytes());
        out[8] = self.kind as u8;
        out[12..16].copy_from_slice(&self.mode.to_le_bytes());
        out[16..20].copy_from_slice(&self.uid.to_le_bytes());
        out[20..24].copy_from_slice(&self.gid.to_le_bytes());
        out[24..32].copy_from_slice(&self.size.to_le_bytes());
        out[32..40].copy_from_slice(&self.generation.to_le_bytes());
        out[40..48].copy_from_slice(&self.extent_root.to_le_bytes());
        out[48..56].copy_from_slice(&self.created_ns.to_le_bytes());
        out[56..64].copy_from_slice(&self.modified_ns.to_le_bytes());
        out[64..68].copy_from_slice(&self.links.to_le_bytes());
        out[68..72].copy_from_slice(&self.flags.to_le_bytes());
        out
    }

    /// Deserialize an inode from a 128-byte record.
    pub fn from_bytes(buf: &[u8; NEXUSFS_INODE_SIZE]) -> Result<Self, FsError> {
        let inode_id = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);
        if inode_id == 0 {
            return Err(FsError::CorruptMetadata);
        }
        let kind = NexusInodeKind::from_u8(buf[8])?;
        Ok(Self {
            inode_id,
            kind,
            mode: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            uid: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
            gid: u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]),
            size: u64::from_le_bytes([
                buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
            ]),
            generation: u64::from_le_bytes([
                buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
            ]),
            extent_root: u64::from_le_bytes([
                buf[40], buf[41], buf[42], buf[43], buf[44], buf[45], buf[46], buf[47],
            ]),
            created_ns: u64::from_le_bytes([
                buf[48], buf[49], buf[50], buf[51], buf[52], buf[53], buf[54], buf[55],
            ]),
            modified_ns: u64::from_le_bytes([
                buf[56], buf[57], buf[58], buf[59], buf[60], buf[61], buf[62], buf[63],
            ]),
            links: u32::from_le_bytes([buf[64], buf[65], buf[66], buf[67]]),
            flags: u32::from_le_bytes([buf[68], buf[69], buf[70], buf[71]]),
        })
    }
}

fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}
