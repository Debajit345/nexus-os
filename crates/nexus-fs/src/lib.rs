#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Clean-room `no_std` + `alloc` filesystem crate for Nexus OS.
//!
//! Provides:
//! - [`BlockIo`] sector I/O trait and [`MemBlockIo`] in-memory block device.
//! - Clean-room FAT32 reader ([`Fat32Volume`]) supporting BPB parsing, cluster
//!   chain traversal, 8.3 directory listing, and file reading.
//! - Clean-room NexusFS copy-on-write superblock and inode metadata
//!   ([`NexusSuperblock`], [`NexusInode`], [`validate_superblock`]).

extern crate alloc;

pub mod block;
pub mod fat32;
pub mod nexusfs;

pub use block::{BlockIo, FsError, MemBlockIo, SECTOR_SIZE};
pub use fat32::{Fat32Bpb, Fat32DirEntry, Fat32Volume};
pub use nexusfs::{
    validate_superblock, NexusInode, NexusInodeKind, NexusSuperblock, NEXUSFS_DEFAULT_BLOCK_SIZE,
    NEXUSFS_INODE_SIZE, NEXUSFS_MAGIC, NEXUSFS_SUPERBLOCK_SIZE, NEXUSFS_VERSION,
};

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn write_fat32_entry(fat: &mut [u8], cluster: u32, value: u32) {
        let off = (cluster as usize) * 4;
        fat[off..off + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn make_dir_entry(raw_name: &[u8; 11], attr: u8, first_cluster: u32, size: u32) -> [u8; 32] {
        let mut entry = [0u8; 32];
        entry[0..11].copy_from_slice(raw_name);
        entry[11] = attr;
        let hi = ((first_cluster >> 16) & 0xFFFF) as u16;
        let lo = (first_cluster & 0xFFFF) as u16;
        entry[20..22].copy_from_slice(&hi.to_le_bytes());
        entry[26..28].copy_from_slice(&lo.to_le_bytes());
        entry[28..32].copy_from_slice(&size.to_le_bytes());
        entry
    }

    fn build_synthetic_fat32() -> MemBlockIo {
        // Layout:
        // - 1 sector/cluster (512 bytes/cluster)
        // - reserved_sectors = 2 (sector 0 = BPB, sector 1 = FSInfo)
        // - num_fats = 2, fat_size_sectors = 2 (FAT1: sectors 2..4, FAT2: sectors 4..6)
        // - data_start = sector 6
        // - total_sectors = 64 (58 data clusters: clusters 2..60)
        let total_sectors = 64usize;
        let reserved_sectors = 2u16;
        let num_fats = 2u8;
        let fat_size_sectors = 2u32;
        let sectors_per_cluster = 1u8;
        let root_cluster = 2u32;

        let mut io = MemBlockIo::new(total_sectors);

        // Sector 0: BPB
        let mut bpb = [0u8; SECTOR_SIZE];
        bpb[0..3].copy_from_slice(&[0xEB, 0x58, 0x90]);
        bpb[3..11].copy_from_slice(b"NEXUSFAT");
        bpb[11..13].copy_from_slice(&(SECTOR_SIZE as u16).to_le_bytes());
        bpb[13] = sectors_per_cluster;
        bpb[14..16].copy_from_slice(&reserved_sectors.to_le_bytes());
        bpb[16] = num_fats;
        bpb[17..19].copy_from_slice(&0u16.to_le_bytes()); // root_entry_count = 0
        bpb[19..21].copy_from_slice(&0u16.to_le_bytes()); // total_sectors_16 = 0
        bpb[21] = 0xF8; // media
        bpb[22..24].copy_from_slice(&0u16.to_le_bytes()); // fat_size_16 = 0
        bpb[32..36].copy_from_slice(&(total_sectors as u32).to_le_bytes());
        bpb[36..40].copy_from_slice(&fat_size_sectors.to_le_bytes());
        bpb[44..48].copy_from_slice(&root_cluster.to_le_bytes());
        bpb[48..50].copy_from_slice(&1u16.to_le_bytes()); // fs_info_sector
        bpb[67..71].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        bpb[71..82].copy_from_slice(b"NEXUSOS    ");
        bpb[510] = 0x55;
        bpb[511] = 0xAA;
        io.write_sector(0, &bpb).unwrap();

        // FAT1 (sectors 2..4)
        let mut fat = vec![0u8; (fat_size_sectors as usize) * SECTOR_SIZE];
        write_fat32_entry(&mut fat, 0, 0x0FFF_FFF8);
        write_fat32_entry(&mut fat, 1, 0x0FFF_FFFF);
        // Cluster 2: root directory (EOC)
        write_fat32_entry(&mut fat, 2, 0x0FFF_FFFF);
        // Cluster 3 -> Cluster 4 -> EOC: HELLO.TXT (600 bytes, spans 2 clusters)
        write_fat32_entry(&mut fat, 3, 4);
        write_fat32_entry(&mut fat, 4, 0x0FFF_FFFF);
        // Cluster 5: SUBDIR (EOC)
        write_fat32_entry(&mut fat, 5, 0x0FFF_FFFF);
        // Cluster 6: SUBDIR/NOTE.MD (EOC)
        write_fat32_entry(&mut fat, 6, 0x0FFF_FFFF);

        for s in 0..fat_size_sectors as usize {
            let mut sec = [0u8; SECTOR_SIZE];
            sec.copy_from_slice(&fat[s * SECTOR_SIZE..(s + 1) * SECTOR_SIZE]);
            io.write_sector(u64::from(reserved_sectors) + s as u64, &sec)
                .unwrap();
            io.write_sector(
                u64::from(reserved_sectors) + u64::from(fat_size_sectors) + s as u64,
                &sec,
            )
            .unwrap();
        }

        let data_start =
            u64::from(reserved_sectors) + u64::from(num_fats) * u64::from(fat_size_sectors);

        // Cluster 2 (data_start + 0): Root directory
        let mut root_dir = [0u8; SECTOR_SIZE];
        // Entry 0: Volume label (should be skipped)
        root_dir[0..32].copy_from_slice(&make_dir_entry(b"NEXUSOS    ", 0x08, 0, 0));
        // Entry 1: Deleted entry (should be skipped)
        let mut deleted = make_dir_entry(b"OLD     TXT", 0x20, 0, 10);
        deleted[0] = 0xE5;
        root_dir[32..64].copy_from_slice(&deleted);
        // Entry 2: LFN entry (should be skipped)
        let mut lfn = [0u8; 32];
        lfn[0] = 0x41;
        lfn[11] = 0x0F;
        root_dir[64..96].copy_from_slice(&lfn);
        // Entry 3: HELLO.TXT (600 bytes at cluster 3)
        root_dir[96..128].copy_from_slice(&make_dir_entry(b"HELLO   TXT", 0x20, 3, 600));
        // Entry 4: SUBDIR (directory at cluster 5)
        root_dir[128..160].copy_from_slice(&make_dir_entry(b"SUBDIR     ", 0x10, 5, 0));
        io.write_sector(data_start, &root_dir).unwrap();

        // Cluster 3 & 4 (data_start + 1, + 2): HELLO.TXT payload (600 bytes)
        let mut hello_payload = Vec::with_capacity(600);
        for i in 0..600usize {
            hello_payload.push(b'A' + (i % 26) as u8);
        }
        let mut c3 = [0u8; SECTOR_SIZE];
        c3.copy_from_slice(&hello_payload[0..512]);
        io.write_sector(data_start + 1, &c3).unwrap();

        let mut c4 = [0u8; SECTOR_SIZE];
        c4[0..88].copy_from_slice(&hello_payload[512..600]);
        io.write_sector(data_start + 2, &c4).unwrap();

        // Cluster 5 (data_start + 3): SUBDIR
        let mut subdir = [0u8; SECTOR_SIZE];
        subdir[0..32].copy_from_slice(&make_dir_entry(b"NOTE    MD ", 0x20, 6, 17));
        io.write_sector(data_start + 3, &subdir).unwrap();

        // Cluster 6 (data_start + 4): SUBDIR/NOTE.MD
        let mut note = [0u8; SECTOR_SIZE];
        note[0..17].copy_from_slice(b"NexusFS + FAT32!\n");
        io.write_sector(data_start + 4, &note).unwrap();

        io
    }

    #[test]
    fn mem_block_io_read_write_and_bounds() {
        let mut io = MemBlockIo::new(4);
        assert_eq!(io.sector_count(), 4);

        let mut sector = [0u8; SECTOR_SIZE];
        sector[0] = 0x42;
        sector[511] = 0x99;
        io.write_sector(2, &sector).unwrap();

        let mut read_back = [0u8; SECTOR_SIZE];
        io.read_sector(2, &mut read_back).unwrap();
        assert_eq!(read_back, sector);

        assert_eq!(io.read_sector(4, &mut read_back), Err(FsError::OutOfBounds));
        assert_eq!(io.write_sector(4, &sector), Err(FsError::OutOfBounds));
    }

    #[test]
    fn fat32_mount_list_root_and_read_file() {
        let mut io = build_synthetic_fat32();
        let vol = Fat32Volume::mount(&mut io).expect("mount synthetic FAT32");

        assert_eq!(vol.bpb().bytes_per_sector, 512);
        assert_eq!(vol.bpb().sectors_per_cluster, 1);
        assert_eq!(vol.bpb().root_cluster, 2);

        let root_entries = vol.list_root(&mut io).expect("list root");
        assert_eq!(root_entries.len(), 2);
        assert_eq!(root_entries[0].name, "HELLO.TXT");
        assert!(!root_entries[0].is_dir());
        assert_eq!(root_entries[0].size, 600);
        assert_eq!(root_entries[1].name, "SUBDIR");
        assert!(root_entries[1].is_dir());

        let hello = vol.read_file(&mut io, "hello.txt").expect("read HELLO.TXT");
        assert_eq!(hello.len(), 600);
        for (i, &b) in hello.iter().enumerate() {
            assert_eq!(b, b'A' + (i % 26) as u8);
        }

        let nested = vol
            .read_file(&mut io, "/SUBDIR/NOTE.MD")
            .expect("read nested file");
        assert_eq!(nested, b"NexusFS + FAT32!\n");

        assert_eq!(
            vol.read_file(&mut io, "MISSING.TXT"),
            Err(FsError::NotFound)
        );
        assert_eq!(vol.read_file(&mut io, "SUBDIR"), Err(FsError::IsADirectory));
    }

    #[test]
    fn nexusfs_superblock_validation_and_inode_roundtrip() {
        let uuid = [0x11u8; 16];
        let sb = NexusSuperblock::new(1024, NEXUSFS_DEFAULT_BLOCK_SIZE, uuid);
        assert!(validate_superblock(&sb).is_ok());

        let raw = sb.to_bytes();
        let parsed = NexusSuperblock::from_bytes(&raw).expect("parse valid superblock");
        assert_eq!(parsed, sb);

        // Bad magic
        let mut bad_magic = sb.clone();
        bad_magic.magic = *b"BADMAGIC";
        assert_eq!(
            validate_superblock(&bad_magic),
            Err(FsError::InvalidSuperblockMagic)
        );

        // Unsupported version
        let mut bad_ver = sb.clone();
        bad_ver.version = 99;
        assert_eq!(
            validate_superblock(&bad_ver),
            Err(FsError::UnsupportedVersion)
        );

        // Invalid block size
        let mut bad_bs = sb.clone();
        bad_bs.block_size = 3000;
        assert_eq!(validate_superblock(&bad_bs), Err(FsError::InvalidBlockSize));

        // Checksum mismatch
        let mut bad_crc = sb.clone();
        bad_crc.checksum ^= 1;
        assert_eq!(
            validate_superblock(&bad_crc),
            Err(FsError::ChecksumMismatch)
        );

        // Inode round-trip
        let inode = NexusInode {
            inode_id: 42,
            kind: NexusInodeKind::File,
            mode: 0o644,
            uid: 1000,
            gid: 1000,
            size: 8192,
            generation: 7,
            extent_root: 128,
            created_ns: 1_700_000_000,
            modified_ns: 1_700_000_123,
            links: 1,
            flags: 0,
        };
        let inode_bytes = inode.to_bytes();
        let inode_parsed = NexusInode::from_bytes(&inode_bytes).expect("parse inode");
        assert_eq!(inode_parsed, inode);
    }
}
