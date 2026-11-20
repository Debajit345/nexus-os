//! Clean-room APFS Container/Volume Superblock & B-Tree Object Map Parser.
//!
//! Implements `no_std` parsing and verification for:
//! - `obj_phys_t` headers and APFS Fletcher-64 checksums
//! - `nx_superblock_t` container superblocks
//! - `omap_phys_t` object maps and `btree_node_phys_t` B-tree lookups
//! - `apfs_superblock_t` volume superblocks and metadata
//! - `self_test()` with synthetic in-memory APFS container verification

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// APFS container superblock magic (`"NXSB"` in little-endian).
pub const NX_MAGIC: u32 = 0x4253_584E;
/// APFS volume superblock magic (`"APSB"` in little-endian).
pub const APFS_MAGIC: u32 = 0x4253_5041;

/// Default APFS block size (4 KiB).
pub const APFS_DEFAULT_BLOCK_SIZE: usize = 4096;
/// Maximum number of file systems (volumes) in an APFS container.
pub const NX_MAX_FILE_SYSTEMS: usize = 100;
/// Maximum B-tree traversal depth to guard against corrupt/cyclic trees.
pub const APFS_MAX_BTREE_DEPTH: usize = 16;

/// Object type mask (lower 16 bits of `o_type`).
pub const OBJECT_TYPE_MASK: u32 = 0x0000_FFFF;
/// Container superblock object type.
pub const OBJECT_TYPE_NX_SUPERBLOCK: u32 = 0x0001;
/// B-tree root node object type.
pub const OBJECT_TYPE_BTREE: u32 = 0x0002;
/// B-tree non-root node object type.
pub const OBJECT_TYPE_BTREE_NODE: u32 = 0x0003;
/// Object map (`omap_phys_t`) object type.
pub const OBJECT_TYPE_OMAP: u32 = 0x000B;
/// Volume superblock (`apfs_superblock_t`) object type.
pub const OBJECT_TYPE_FS: u32 = 0x000D;

/// B-tree node flags.
pub const BTNODE_ROOT: u16 = 0x0001;
pub const BTNODE_LEAF: u16 = 0x0002;
pub const BTNODE_FIXED_KV_SIZE: u16 = 0x0004;

/// Size of `obj_phys_t` header in bytes.
const OBJ_PHYS_SIZE: usize = 32;
/// Size of `btree_node_phys_t` header in bytes.
const BTREE_NODE_HEADER_SIZE: usize = 56;
/// Size of `btree_info_t` trailer at the end of a root B-tree node.
const BTREE_INFO_SIZE: usize = 40;

/// Errors returned by the APFS parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApfsError {
    BufferTooSmall,
    InvalidBlockSize,
    InvalidMagic,
    ChecksumMismatch,
    InvalidObjectType,
    InvalidBTreeNode,
    BTreeDepthExceeded,
    OidNotFound,
    IoError,
}

/// Block-level reader abstraction for APFS containers.
pub trait ApfsBlockReader {
    fn block_size(&self) -> usize;
    fn read_block(&self, paddr: u64, buf: &mut [u8]) -> Result<(), ApfsError>;
}

/// In-memory slice-backed APFS block reader.
pub struct SliceBlockReader<'a> {
    data: &'a [u8],
    block_size: usize,
}

impl<'a> SliceBlockReader<'a> {
    pub fn new(data: &'a [u8], block_size: usize) -> Result<Self, ApfsError> {
        if block_size < 512 || !block_size.is_power_of_two() {
            return Err(ApfsError::InvalidBlockSize);
        }
        Ok(Self { data, block_size })
    }
}

impl ApfsBlockReader for SliceBlockReader<'_> {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn read_block(&self, paddr: u64, buf: &mut [u8]) -> Result<(), ApfsError> {
        if buf.len() < self.block_size {
            return Err(ApfsError::BufferTooSmall);
        }
        let offset = usize::try_from(paddr)
            .ok()
            .and_then(|idx| idx.checked_mul(self.block_size))
            .ok_or(ApfsError::IoError)?;
        let end = offset
            .checked_add(self.block_size)
            .ok_or(ApfsError::IoError)?;
        if end > self.data.len() {
            return Err(ApfsError::IoError);
        }
        buf[..self.block_size].copy_from_slice(&self.data[offset..end]);
        Ok(())
    }
}

/// Common APFS physical object header (`obj_phys_t`, 32 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjPhys {
    pub o_cksum: u64,
    pub o_oid: u64,
    pub o_xid: u64,
    pub o_type: u32,
    pub o_subtype: u32,
}

impl ObjPhys {
    pub fn parse(block: &[u8]) -> Result<Self, ApfsError> {
        if block.len() < OBJ_PHYS_SIZE {
            return Err(ApfsError::BufferTooSmall);
        }
        Ok(Self {
            o_cksum: read_le_u64(block, 0),
            o_oid: read_le_u64(block, 8),
            o_xid: read_le_u64(block, 16),
            o_type: read_le_u32(block, 24),
            o_subtype: read_le_u32(block, 28),
        })
    }

    pub const fn object_type(&self) -> u32 {
        self.o_type & OBJECT_TYPE_MASK
    }
}

/// Compute the APFS Fletcher-64 checksum over `block[8..]`.
pub fn compute_fletcher64(block: &[u8]) -> Result<u64, ApfsError> {
    if block.len() < OBJ_PHYS_SIZE || !block.len().is_multiple_of(4) {
        return Err(ApfsError::BufferTooSmall);
    }
    let mut sum1: u64 = 0;
    let mut sum2: u64 = 0;
    let (chunks, _) = block[8..].as_chunks::<4>();
    for chunk in chunks {
        let word = u64::from(u32::from_le_bytes(*chunk));
        sum1 = (sum1 + word) % 0xFFFF_FFFF;
        sum2 = (sum2 + sum1) % 0xFFFF_FFFF;
    }
    let check1 = 0xFFFF_FFFF - ((sum1 + sum2) % 0xFFFF_FFFF);
    let check2 = 0xFFFF_FFFF - ((sum1 + check1) % 0xFFFF_FFFF);
    Ok((check2 << 32) | check1)
}

/// Verify the Fletcher-64 checksum stored in `obj_phys_t.o_cksum`.
pub fn verify_fletcher64(block: &[u8]) -> Result<(), ApfsError> {
    let expected = compute_fletcher64(block)?;
    let stored = read_le_u64(block, 0);
    if stored == expected {
        Ok(())
    } else {
        Err(ApfsError::ChecksumMismatch)
    }
}

/// Write the computed Fletcher-64 checksum into `block[0..8]`.
pub fn stamp_fletcher64(block: &mut [u8]) -> Result<u64, ApfsError> {
    let cksum = compute_fletcher64(block)?;
    block[0..8].copy_from_slice(&cksum.to_le_bytes());
    Ok(cksum)
}

/// Parsed APFS Container Superblock (`nx_superblock_t`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NxSuperblock {
    pub header: ObjPhys,
    pub nx_magic: u32,
    pub nx_block_size: u32,
    pub nx_block_count: u64,
    pub nx_features: u64,
    pub nx_readonly_compatible_features: u64,
    pub nx_incompatible_features: u64,
    pub nx_uuid: [u8; 16],
    pub nx_next_oid: u64,
    pub nx_next_xid: u64,
    pub nx_xp_desc_blocks: u32,
    pub nx_xp_data_blocks: u32,
    pub nx_xp_desc_base: u64,
    pub nx_xp_data_base: u64,
    pub nx_spaceman_oid: u64,
    pub nx_omap_oid: u64,
    pub nx_reaper_oid: u64,
    pub nx_max_file_systems: u32,
    pub nx_fs_oids: Vec<u64>,
}

impl NxSuperblock {
    /// Parse and checksum-verify an `nx_superblock_t` from a container block.
    pub fn parse(block: &[u8]) -> Result<Self, ApfsError> {
        if block.len() < 0x3D8 {
            return Err(ApfsError::BufferTooSmall);
        }
        verify_fletcher64(block)?;
        let header = ObjPhys::parse(block)?;
        if header.object_type() != OBJECT_TYPE_NX_SUPERBLOCK {
            return Err(ApfsError::InvalidObjectType);
        }
        let nx_magic = read_le_u32(block, 0x20);
        if nx_magic != NX_MAGIC {
            return Err(ApfsError::InvalidMagic);
        }
        let nx_block_size = read_le_u32(block, 0x24);
        if nx_block_size < 4096 || !nx_block_size.is_power_of_two() {
            return Err(ApfsError::InvalidBlockSize);
        }

        let mut nx_uuid = [0u8; 16];
        nx_uuid.copy_from_slice(&block[0x48..0x58]);

        let nx_max_file_systems = read_le_u32(block, 0xB4);
        let fs_slots = (nx_max_file_systems as usize).min(NX_MAX_FILE_SYSTEMS);
        let mut nx_fs_oids = Vec::new();
        let (oid_chunks, _) = block[0xB8..0xB8 + fs_slots * 8].as_chunks::<8>();
        for chunk in oid_chunks {
            let oid = u64::from_le_bytes(*chunk);
            if oid != 0 {
                nx_fs_oids.push(oid);
            }
        }

        Ok(Self {
            header,
            nx_magic,
            nx_block_size,
            nx_block_count: read_le_u64(block, 0x28),
            nx_features: read_le_u64(block, 0x30),
            nx_readonly_compatible_features: read_le_u64(block, 0x38),
            nx_incompatible_features: read_le_u64(block, 0x40),
            nx_uuid,
            nx_next_oid: read_le_u64(block, 0x58),
            nx_next_xid: read_le_u64(block, 0x60),
            nx_xp_desc_blocks: read_le_u32(block, 0x68),
            nx_xp_data_blocks: read_le_u32(block, 0x6C),
            nx_xp_desc_base: read_le_u64(block, 0x70),
            nx_xp_data_base: read_le_u64(block, 0x78),
            nx_spaceman_oid: read_le_u64(block, 0x98),
            nx_omap_oid: read_le_u64(block, 0xA0),
            nx_reaper_oid: read_le_u64(block, 0xA8),
            nx_max_file_systems,
            nx_fs_oids,
        })
    }
}

/// Parsed APFS Object Map (`omap_phys_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OmapPhys {
    pub header: ObjPhys,
    pub om_flags: u32,
    pub om_snap_count: u32,
    pub om_tree_type: u32,
    pub om_snapshot_tree_type: u32,
    pub om_tree_oid: u64,
    pub om_snapshot_tree_oid: u64,
    pub om_most_recent_snap: u64,
}

impl OmapPhys {
    /// Parse and checksum-verify an `omap_phys_t` block.
    pub fn parse(block: &[u8]) -> Result<Self, ApfsError> {
        if block.len() < 88 {
            return Err(ApfsError::BufferTooSmall);
        }
        verify_fletcher64(block)?;
        let header = ObjPhys::parse(block)?;
        if header.object_type() != OBJECT_TYPE_OMAP {
            return Err(ApfsError::InvalidObjectType);
        }
        Ok(Self {
            header,
            om_flags: read_le_u32(block, 32),
            om_snap_count: read_le_u32(block, 36),
            om_tree_type: read_le_u32(block, 40),
            om_snapshot_tree_type: read_le_u32(block, 44),
            om_tree_oid: read_le_u64(block, 48),
            om_snapshot_tree_oid: read_le_u64(block, 56),
            om_most_recent_snap: read_le_u64(block, 64),
        })
    }

    /// Look up a virtual `oid` at or below `max_xid` in the object map B-tree.
    pub fn lookup_oid(
        &self,
        reader: &dyn ApfsBlockReader,
        oid: u64,
        max_xid: u64,
    ) -> Result<OmapVal, ApfsError> {
        let block_size = reader.block_size();
        let mut node_buf = vec![0u8; block_size];
        let mut current_paddr = self.om_tree_oid;

        for _depth in 0..APFS_MAX_BTREE_DEPTH {
            reader.read_block(current_paddr, &mut node_buf)?;
            verify_fletcher64(&node_buf)?;
            let node = BTreeNodeHeader::parse(&node_buf)?;
            if (node.btn_flags & BTNODE_FIXED_KV_SIZE) == 0 {
                return Err(ApfsError::InvalidBTreeNode);
            }

            let is_root = (node.btn_flags & BTNODE_ROOT) != 0;
            let is_leaf = (node.btn_flags & BTNODE_LEAF) != 0;
            let toc_start = BTREE_NODE_HEADER_SIZE + usize::from(node.table_off);
            let key_area_start = toc_start + usize::from(node.table_len);
            let val_area_end = if is_root {
                block_size
                    .checked_sub(BTREE_INFO_SIZE)
                    .ok_or(ApfsError::InvalidBTreeNode)?
            } else {
                block_size
            };

            let nkeys = node.btn_nkeys as usize;
            let toc_needed = nkeys.checked_mul(4).ok_or(ApfsError::InvalidBTreeNode)?;
            if toc_start + toc_needed > key_area_start || key_area_start > val_area_end {
                return Err(ApfsError::InvalidBTreeNode);
            }

            let mut best_idx: Option<usize> = None;
            let (toc_chunks, _) = node_buf[toc_start..toc_start + toc_needed].as_chunks::<4>();
            for (idx, entry) in toc_chunks.iter().enumerate() {
                let k_off = u16::from_le_bytes([entry[0], entry[1]]) as usize;
                let key_pos = key_area_start + k_off;
                if key_pos + 16 > val_area_end {
                    return Err(ApfsError::InvalidBTreeNode);
                }
                let ok_oid = read_le_u64(&node_buf, key_pos);
                let ok_xid = read_le_u64(&node_buf, key_pos + 8);
                if (ok_oid, ok_xid) <= (oid, max_xid) {
                    best_idx = Some(idx);
                } else {
                    break;
                }
            }

            let idx = best_idx.ok_or(ApfsError::OidNotFound)?;
            let entry = toc_chunks[idx];
            let k_off = u16::from_le_bytes([entry[0], entry[1]]) as usize;
            let v_off = u16::from_le_bytes([entry[2], entry[3]]) as usize;
            let key_pos = key_area_start + k_off;
            let ok_oid = read_le_u64(&node_buf, key_pos);

            if is_leaf {
                if ok_oid != oid {
                    return Err(ApfsError::OidNotFound);
                }
                let val_pos = val_area_end
                    .checked_sub(v_off)
                    .ok_or(ApfsError::InvalidBTreeNode)?;
                if val_pos + 16 > val_area_end {
                    return Err(ApfsError::InvalidBTreeNode);
                }
                return Ok(OmapVal {
                    ov_flags: read_le_u32(&node_buf, val_pos),
                    ov_size: read_le_u32(&node_buf, val_pos + 4),
                    ov_paddr: read_le_u64(&node_buf, val_pos + 8),
                });
            }

            let val_pos = val_area_end
                .checked_sub(v_off)
                .ok_or(ApfsError::InvalidBTreeNode)?;
            if val_pos + 8 > val_area_end {
                return Err(ApfsError::InvalidBTreeNode);
            }
            current_paddr = read_le_u64(&node_buf, val_pos);
        }

        Err(ApfsError::BTreeDepthExceeded)
    }
}

/// Object map leaf value (`omap_val_t`, 16 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OmapVal {
    pub ov_flags: u32,
    pub ov_size: u32,
    pub ov_paddr: u64,
}

/// Parsed B-tree node header (`btree_node_phys_t`, 56 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BTreeNodeHeader {
    header: ObjPhys,
    btn_flags: u16,
    btn_level: u16,
    btn_nkeys: u32,
    table_off: u16,
    table_len: u16,
}

impl BTreeNodeHeader {
    fn parse(block: &[u8]) -> Result<Self, ApfsError> {
        if block.len() < BTREE_NODE_HEADER_SIZE {
            return Err(ApfsError::BufferTooSmall);
        }
        let header = ObjPhys::parse(block)?;
        let obj_type = header.object_type();
        if obj_type != OBJECT_TYPE_BTREE && obj_type != OBJECT_TYPE_BTREE_NODE {
            return Err(ApfsError::InvalidObjectType);
        }
        Ok(Self {
            header,
            btn_flags: read_le_u16(block, 32),
            btn_level: read_le_u16(block, 34),
            btn_nkeys: read_le_u32(block, 36),
            table_off: read_le_u16(block, 40),
            table_len: read_le_u16(block, 42),
        })
    }
}

/// Parsed APFS Volume Superblock (`apfs_superblock_t`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApfsSuperblock {
    pub header: ObjPhys,
    pub apfs_magic: u32,
    pub apfs_fs_index: u32,
    pub apfs_features: u64,
    pub apfs_readonly_compatible_features: u64,
    pub apfs_incompatible_features: u64,
    pub apfs_unmount_time: u64,
    pub apfs_fs_reserve_block_count: u64,
    pub apfs_fs_quota_block_count: u64,
    pub apfs_fs_alloc_count: u64,
    pub apfs_omap_oid: u64,
    pub apfs_root_tree_oid: u64,
    pub apfs_extentref_tree_oid: u64,
    pub apfs_snap_meta_tree_oid: u64,
    pub apfs_next_obj_id: u64,
    pub apfs_num_files: u64,
    pub apfs_num_directories: u64,
    pub apfs_num_symlinks: u64,
    pub apfs_num_other_fsobjects: u64,
    pub apfs_num_snapshots: u64,
    pub apfs_total_blocks_alloced: u64,
    pub apfs_total_blocks_freed: u64,
    pub apfs_vol_uuid: [u8; 16],
    pub apfs_last_mod_time: u64,
    pub apfs_fs_flags: u64,
    pub apfs_volname: String,
    pub apfs_next_doc_id: u32,
    pub apfs_role: u16,
}

impl ApfsSuperblock {
    /// Parse and checksum-verify an `apfs_superblock_t` from a volume block.
    pub fn parse(block: &[u8]) -> Result<Self, ApfsError> {
        if block.len() < 0x3C6 {
            return Err(ApfsError::BufferTooSmall);
        }
        verify_fletcher64(block)?;
        let header = ObjPhys::parse(block)?;
        if header.object_type() != OBJECT_TYPE_FS {
            return Err(ApfsError::InvalidObjectType);
        }
        let apfs_magic = read_le_u32(block, 0x20);
        if apfs_magic != APFS_MAGIC {
            return Err(ApfsError::InvalidMagic);
        }

        let mut apfs_vol_uuid = [0u8; 16];
        apfs_vol_uuid.copy_from_slice(&block[0xF0..0x100]);

        let name_bytes = &block[0x2C0..0x3C0];
        let nul_pos = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        let apfs_volname = String::from_utf8_lossy(&name_bytes[..nul_pos]).into_owned();

        Ok(Self {
            header,
            apfs_magic,
            apfs_fs_index: read_le_u32(block, 0x24),
            apfs_features: read_le_u64(block, 0x28),
            apfs_readonly_compatible_features: read_le_u64(block, 0x30),
            apfs_incompatible_features: read_le_u64(block, 0x38),
            apfs_unmount_time: read_le_u64(block, 0x40),
            apfs_fs_reserve_block_count: read_le_u64(block, 0x48),
            apfs_fs_quota_block_count: read_le_u64(block, 0x50),
            apfs_fs_alloc_count: read_le_u64(block, 0x58),
            apfs_omap_oid: read_le_u64(block, 0x80),
            apfs_root_tree_oid: read_le_u64(block, 0x88),
            apfs_extentref_tree_oid: read_le_u64(block, 0x90),
            apfs_snap_meta_tree_oid: read_le_u64(block, 0x98),
            apfs_next_obj_id: read_le_u64(block, 0xB0),
            apfs_num_files: read_le_u64(block, 0xB8),
            apfs_num_directories: read_le_u64(block, 0xC0),
            apfs_num_symlinks: read_le_u64(block, 0xC8),
            apfs_num_other_fsobjects: read_le_u64(block, 0xD0),
            apfs_num_snapshots: read_le_u64(block, 0xD8),
            apfs_total_blocks_alloced: read_le_u64(block, 0xE0),
            apfs_total_blocks_freed: read_le_u64(block, 0xE8),
            apfs_vol_uuid,
            apfs_last_mod_time: read_le_u64(block, 0x100),
            apfs_fs_flags: read_le_u64(block, 0x108),
            apfs_volname,
            apfs_next_doc_id: read_le_u32(block, 0x3C0),
            apfs_role: read_le_u16(block, 0x3C4),
        })
    }
}

/// High-level parsed APFS container handle.
pub struct ApfsContainer {
    pub superblock: NxSuperblock,
    pub omap: OmapPhys,
}

impl ApfsContainer {
    /// Mount and verify an APFS container from block 0 (`nx_superblock_t`) and its `omap_phys_t`.
    pub fn open(reader: &dyn ApfsBlockReader) -> Result<Self, ApfsError> {
        let block_size = reader.block_size();
        let mut buf = vec![0u8; block_size];
        reader.read_block(0, &mut buf)?;
        let superblock = NxSuperblock::parse(&buf)?;

        reader.read_block(superblock.nx_omap_oid, &mut buf)?;
        let omap = OmapPhys::parse(&buf)?;

        Ok(Self { superblock, omap })
    }

    /// Resolve a volume virtual OID via the container object map and parse its `apfs_superblock_t`.
    pub fn open_volume(
        &self,
        reader: &dyn ApfsBlockReader,
        vol_oid: u64,
    ) -> Result<ApfsSuperblock, ApfsError> {
        let val = self
            .omap
            .lookup_oid(reader, vol_oid, self.superblock.header.o_xid)?;
        let mut buf = vec![0u8; reader.block_size()];
        reader.read_block(val.ov_paddr, &mut buf)?;
        ApfsSuperblock::parse(&buf)
    }
}

/// Run clean-room self-tests for APFS container, omap B-tree, and volume superblock parsing.
pub fn self_test() -> Result<(), &'static str> {
    let bs = APFS_DEFAULT_BLOCK_SIZE;
    // 5 blocks:
    // 0: nx_superblock_t
    // 1: omap_phys_t
    // 2: omap B-tree root (internal node)
    // 3: omap B-tree leaf node
    // 4: apfs_superblock_t (volume)
    let mut disk = vec![0u8; bs * 5];

    // Block 0: nx_superblock_t
    {
        let b0 = &mut disk[0..bs];
        write_obj_phys(b0, 1, 10, OBJECT_TYPE_NX_SUPERBLOCK, 0);
        write_le_u32(b0, 0x20, NX_MAGIC);
        write_le_u32(b0, 0x24, bs as u32);
        write_le_u64(b0, 0x28, 5);
        write_le_u64(b0, 0x58, 1024);
        write_le_u64(b0, 0x60, 11);
        write_le_u64(b0, 0xA0, 1); // nx_omap_oid -> block 1
        write_le_u32(b0, 0xB4, NX_MAX_FILE_SYSTEMS as u32);
        write_le_u64(b0, 0xB8, 0x400); // volume 0 virtual OID
        stamp_fletcher64(b0).map_err(|_| "nx_superblock checksum failed")?;
    }

    // Block 1: omap_phys_t
    {
        let b1 = &mut disk[bs..bs * 2];
        write_obj_phys(b1, 1, 10, OBJECT_TYPE_OMAP, 0);
        write_le_u64(b1, 48, 2); // om_tree_oid -> block 2
        stamp_fletcher64(b1).map_err(|_| "omap_phys checksum failed")?;
    }

    // Block 2: omap B-tree root (internal node pointing to leaf block 3)
    {
        let b2 = &mut disk[bs * 2..bs * 3];
        write_obj_phys(b2, 2, 10, OBJECT_TYPE_BTREE, OBJECT_TYPE_OMAP);
        write_le_u16(b2, 32, BTNODE_ROOT | BTNODE_FIXED_KV_SIZE);
        write_le_u16(b2, 34, 1); // level 1
        write_le_u32(b2, 36, 1); // 1 key
        write_le_u16(b2, 40, 0); // table_off
        write_le_u16(b2, 42, 4); // table_len (1 * 4 bytes)

        let key_area = BTREE_NODE_HEADER_SIZE + 4;
        let val_end = bs - BTREE_INFO_SIZE;
        // TOC entry 0: k_off = 0, v_off = 8
        write_le_u16(b2, BTREE_NODE_HEADER_SIZE, 0);
        write_le_u16(b2, BTREE_NODE_HEADER_SIZE + 2, 8);
        // Key: (oid=0x400, xid=10)
        write_le_u64(b2, key_area, 0x400);
        write_le_u64(b2, key_area + 8, 10);
        // Val: child paddr = 3
        write_le_u64(b2, val_end - 8, 3);
        stamp_fletcher64(b2).map_err(|_| "btree root checksum failed")?;
    }

    // Block 3: omap B-tree leaf node mapping (0x400, 10) -> paddr 4
    {
        let b3 = &mut disk[bs * 3..bs * 4];
        write_obj_phys(b3, 3, 10, OBJECT_TYPE_BTREE_NODE, OBJECT_TYPE_OMAP);
        write_le_u16(b3, 32, BTNODE_LEAF | BTNODE_FIXED_KV_SIZE);
        write_le_u16(b3, 34, 0); // level 0
        write_le_u32(b3, 36, 1); // 1 key
        write_le_u16(b3, 40, 0); // table_off
        write_le_u16(b3, 42, 4); // table_len

        let key_area = BTREE_NODE_HEADER_SIZE + 4;
        let val_end = bs;
        // TOC entry 0: k_off = 0, v_off = 16
        write_le_u16(b3, BTREE_NODE_HEADER_SIZE, 0);
        write_le_u16(b3, BTREE_NODE_HEADER_SIZE + 2, 16);
        // Key: (oid=0x400, xid=10)
        write_le_u64(b3, key_area, 0x400);
        write_le_u64(b3, key_area + 8, 10);
        // Val: omap_val_t { flags=0, size=4096, paddr=4 }
        write_le_u32(b3, val_end - 16, 0);
        write_le_u32(b3, val_end - 12, bs as u32);
        write_le_u64(b3, val_end - 8, 4);
        stamp_fletcher64(b3).map_err(|_| "btree leaf checksum failed")?;
    }

    // Block 4: apfs_superblock_t
    {
        let b4 = &mut disk[bs * 4..bs * 5];
        write_obj_phys(b4, 0x400, 10, OBJECT_TYPE_FS, 0);
        write_le_u32(b4, 0x20, APFS_MAGIC);
        write_le_u32(b4, 0x24, 0);
        write_le_u64(b4, 0x80, 0x401); // apfs_omap_oid
        write_le_u64(b4, 0x88, 0x402); // apfs_root_tree_oid
        write_le_u64(b4, 0xB0, 16); // apfs_next_obj_id
        write_le_u64(b4, 0xB8, 42); // apfs_num_files
        write_le_u64(b4, 0xC0, 7); // apfs_num_directories
        let vol_name = b"NexusAPFS";
        b4[0x2C0..0x2C0 + vol_name.len()].copy_from_slice(vol_name);
        stamp_fletcher64(b4).map_err(|_| "apfs_superblock checksum failed")?;
    }

    let reader = SliceBlockReader::new(&disk, bs).map_err(|_| "SliceBlockReader::new failed")?;
    let container = ApfsContainer::open(&reader).map_err(|_| "ApfsContainer::open failed")?;
    if container.superblock.nx_fs_oids.as_slice() != [0x400] {
        return Err("unexpected nx_fs_oids");
    }

    let vol = container
        .open_volume(&reader, 0x400)
        .map_err(|_| "ApfsContainer::open_volume failed")?;
    if vol.apfs_volname != "NexusAPFS" || vol.apfs_num_files != 42 || vol.apfs_num_directories != 7
    {
        return Err("unexpected volume metadata");
    }

    // Negative test: missing OID
    if container.omap.lookup_oid(&reader, 0x999, 10) != Err(ApfsError::OidNotFound) {
        return Err("expected OidNotFound for missing OID");
    }

    // Negative test: corrupted checksum
    disk[0x20] ^= 0xFF;
    if NxSuperblock::parse(&disk[0..bs]) != Err(ApfsError::ChecksumMismatch) {
        return Err("expected ChecksumMismatch on corrupted superblock");
    }

    Ok(())
}

fn write_obj_phys(buf: &mut [u8], oid: u64, xid: u64, o_type: u32, o_subtype: u32) {
    write_le_u64(buf, 8, oid);
    write_le_u64(buf, 16, xid);
    write_le_u32(buf, 24, o_type);
    write_le_u32(buf, 28, o_subtype);
}

#[inline]
fn read_le_u16(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([buf[offset], buf[offset + 1]])
}

#[inline]
fn read_le_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}

#[inline]
fn read_le_u64(buf: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
        buf[offset + 4],
        buf[offset + 5],
        buf[offset + 6],
        buf[offset + 7],
    ])
}

#[inline]
fn write_le_u16(buf: &mut [u8], offset: usize, val: u16) {
    buf[offset..offset + 2].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn write_le_u32(buf: &mut [u8], offset: usize, val: u32) {
    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn write_le_u64(buf: &mut [u8], offset: usize, val: u64) {
    buf[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
}
