use alloc::string::String;
use alloc::vec::Vec;

use crate::block::{BlockIo, FsError, SECTOR_SIZE};

const FAT32_EOC_MIN: u32 = 0x0FFF_FFF8;
const FAT32_BAD_CLUSTER: u32 = 0x0FFF_FFF7;
const FAT32_ENTRY_MASK: u32 = 0x0FFF_FFFF;

const ATTR_VOLUME_ID: u8 = 0x08;
const ATTR_DIRECTORY: u8 = 0x10;
const ATTR_LFN: u8 = 0x0F;

/// Parsed FAT32 BIOS Parameter Block (BPB).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fat32Bpb {
    pub bytes_per_sector: u16,
    pub sectors_per_cluster: u8,
    pub reserved_sectors: u16,
    pub num_fats: u8,
    pub total_sectors: u32,
    pub fat_size_sectors: u32,
    pub root_cluster: u32,
    pub fs_info_sector: u16,
    pub volume_id: u32,
    pub volume_label: [u8; 11],
}

impl Fat32Bpb {
    /// Parse and validate a FAT32 BPB from sector 0.
    pub fn parse(sector0: &[u8; SECTOR_SIZE]) -> Result<Self, FsError> {
        if sector0[510] != 0x55 || sector0[511] != 0xAA {
            return Err(FsError::InvalidBootSignature);
        }

        let bytes_per_sector = read_u16_le(sector0, 11);
        if bytes_per_sector != SECTOR_SIZE as u16 {
            return Err(FsError::UnsupportedSectorSize);
        }

        let sectors_per_cluster = sector0[13];
        if sectors_per_cluster == 0 || !sectors_per_cluster.is_power_of_two() {
            return Err(FsError::UnsupportedClusterSize);
        }

        let reserved_sectors = read_u16_le(sector0, 14);
        if reserved_sectors == 0 {
            return Err(FsError::InvalidBpb);
        }

        let num_fats = sector0[16];
        if num_fats == 0 {
            return Err(FsError::InvalidBpb);
        }

        let root_entry_count = read_u16_le(sector0, 17);
        let fat_size_16 = read_u16_le(sector0, 22);
        if root_entry_count != 0 || fat_size_16 != 0 {
            return Err(FsError::NotFat32);
        }

        let total_sectors_16 = read_u16_le(sector0, 19);
        let total_sectors_32 = read_u32_le(sector0, 32);
        let total_sectors = if total_sectors_16 != 0 {
            u32::from(total_sectors_16)
        } else {
            total_sectors_32
        };
        if total_sectors == 0 {
            return Err(FsError::InvalidBpb);
        }

        let fat_size_sectors = read_u32_le(sector0, 36);
        if fat_size_sectors == 0 {
            return Err(FsError::NotFat32);
        }

        let root_cluster = read_u32_le(sector0, 44) & FAT32_ENTRY_MASK;
        if root_cluster < 2 {
            return Err(FsError::InvalidBpb);
        }

        let fs_info_sector = read_u16_le(sector0, 48);
        let volume_id = read_u32_le(sector0, 67);
        let mut volume_label = [0u8; 11];
        volume_label.copy_from_slice(&sector0[71..82]);

        let fats_total = u32::from(num_fats)
            .checked_mul(fat_size_sectors)
            .ok_or(FsError::InvalidBpb)?;
        let data_start = u32::from(reserved_sectors)
            .checked_add(fats_total)
            .ok_or(FsError::InvalidBpb)?;
        if data_start >= total_sectors {
            return Err(FsError::InvalidBpb);
        }

        Ok(Self {
            bytes_per_sector,
            sectors_per_cluster,
            reserved_sectors,
            num_fats,
            total_sectors,
            fat_size_sectors,
            root_cluster,
            fs_info_sector,
            volume_id,
            volume_label,
        })
    }
}

/// A parsed 8.3 FAT32 directory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fat32DirEntry {
    pub name: String,
    pub raw_name: [u8; 11],
    pub attr: u8,
    pub first_cluster: u32,
    pub size: u32,
}

impl Fat32DirEntry {
    /// Returns `true` if this entry represents a subdirectory.
    pub fn is_dir(&self) -> bool {
        (self.attr & ATTR_DIRECTORY) != 0
    }
}

/// Clean-room read-only FAT32 volume reader.
#[derive(Debug, Clone)]
pub struct Fat32Volume {
    bpb: Fat32Bpb,
    fat_start_lba: u64,
    data_start_lba: u64,
    total_clusters: u32,
}

impl Fat32Volume {
    /// Mount a FAT32 volume by reading and validating sector 0.
    pub fn mount<B: BlockIo>(io: &mut B) -> Result<Self, FsError> {
        let mut sector0 = [0u8; SECTOR_SIZE];
        io.read_sector(0, &mut sector0)?;
        let bpb = Fat32Bpb::parse(&sector0)?;

        let fat_start_lba = u64::from(bpb.reserved_sectors);
        let fats_sectors = u64::from(bpb.num_fats) * u64::from(bpb.fat_size_sectors);
        let data_start_lba = fat_start_lba + fats_sectors;
        let data_sectors = u64::from(bpb.total_sectors) - data_start_lba;
        let total_clusters = u32::try_from(data_sectors / u64::from(bpb.sectors_per_cluster))
            .map_err(|_| FsError::InvalidBpb)?;

        if total_clusters == 0 || bpb.root_cluster >= total_clusters + 2 {
            return Err(FsError::InvalidBpb);
        }

        Ok(Self {
            bpb,
            fat_start_lba,
            data_start_lba,
            total_clusters,
        })
    }

    /// Access the parsed BPB.
    pub fn bpb(&self) -> &Fat32Bpb {
        &self.bpb
    }

    /// Convert a 2-based cluster number to its starting LBA.
    pub fn cluster_to_lba(&self, cluster: u32) -> Result<u64, FsError> {
        if cluster < 2 || cluster >= self.total_clusters + 2 {
            return Err(FsError::InvalidCluster);
        }
        let offset = u64::from(cluster - 2) * u64::from(self.bpb.sectors_per_cluster);
        Ok(self.data_start_lba + offset)
    }

    /// Look up the next cluster in the FAT32 allocation table.
    /// Returns `Ok(None)` on end-of-chain.
    pub fn next_cluster<B: BlockIo>(
        &self,
        io: &mut B,
        cluster: u32,
    ) -> Result<Option<u32>, FsError> {
        if cluster < 2 || cluster >= self.total_clusters + 2 {
            return Err(FsError::InvalidCluster);
        }

        let fat_byte_offset = u64::from(cluster) * 4;
        let fat_sector = self.fat_start_lba + (fat_byte_offset / SECTOR_SIZE as u64);
        let offset_in_sector = (fat_byte_offset % SECTOR_SIZE as u64) as usize;

        let mut sector = [0u8; SECTOR_SIZE];
        io.read_sector(fat_sector, &mut sector)?;

        let raw = read_u32_le(&sector, offset_in_sector) & FAT32_ENTRY_MASK;
        if raw == 0 || raw == FAT32_BAD_CLUSTER {
            return Err(FsError::InvalidCluster);
        }
        if raw >= FAT32_EOC_MIN {
            return Ok(None);
        }
        if raw < 2 || raw >= self.total_clusters + 2 {
            return Err(FsError::InvalidCluster);
        }
        Ok(Some(raw))
    }

    /// Read all bytes of a cluster chain starting at `start_cluster`.
    pub fn read_cluster_chain<B: BlockIo>(
        &self,
        io: &mut B,
        start_cluster: u32,
    ) -> Result<Vec<u8>, FsError> {
        let spc = usize::from(self.bpb.sectors_per_cluster);
        let cluster_bytes = spc * SECTOR_SIZE;
        let mut out = Vec::new();
        let mut current = Some(start_cluster);
        let mut visited = 0u32;

        while let Some(cluster) = current {
            if visited >= self.total_clusters {
                return Err(FsError::ClusterChainLoop);
            }
            visited += 1;

            let base_lba = self.cluster_to_lba(cluster)?;
            let start_len = out.len();
            out.resize(start_len + cluster_bytes, 0);
            for s in 0..spc {
                let mut buf = [0u8; SECTOR_SIZE];
                io.read_sector(base_lba + s as u64, &mut buf)?;
                let dst_start = start_len + s * SECTOR_SIZE;
                out[dst_start..dst_start + SECTOR_SIZE].copy_from_slice(&buf);
            }

            current = self.next_cluster(io, cluster)?;
        }

        Ok(out)
    }

    /// List all valid 8.3 entries in the directory starting at `dir_cluster`.
    pub fn list_dir<B: BlockIo>(
        &self,
        io: &mut B,
        dir_cluster: u32,
    ) -> Result<Vec<Fat32DirEntry>, FsError> {
        let data = self.read_cluster_chain(io, dir_cluster)?;
        let mut entries = Vec::new();

        for chunk in data.as_chunks::<32>().0 {
            let first = chunk[0];
            if first == 0x00 {
                break;
            }
            if first == 0xE5 {
                continue;
            }

            let attr = chunk[11];
            if (attr & ATTR_LFN) == ATTR_LFN || (attr & ATTR_VOLUME_ID) != 0 {
                continue;
            }

            let mut raw_name = [0u8; 11];
            raw_name.copy_from_slice(&chunk[0..11]);
            if raw_name[0] == 0x05 {
                raw_name[0] = 0xE5;
            }

            let name = format_8_3_name(&raw_name);
            let cluster_hi = u32::from(read_u16_le(chunk, 20));
            let cluster_lo = u32::from(read_u16_le(chunk, 26));
            let first_cluster = ((cluster_hi << 16) | cluster_lo) & FAT32_ENTRY_MASK;
            let size = read_u32_le(chunk, 28);

            entries.push(Fat32DirEntry {
                name,
                raw_name,
                attr,
                first_cluster,
                size,
            });
        }

        Ok(entries)
    }

    /// List all valid 8.3 entries in the root directory.
    pub fn list_root<B: BlockIo>(&self, io: &mut B) -> Result<Vec<Fat32DirEntry>, FsError> {
        self.list_dir(io, self.bpb.root_cluster)
    }

    /// Read a file by path (e.g. `"HELLO.TXT"` or `"/SUBDIR/DATA.BIN"`).
    /// Matching is case-insensitive against 8.3 names.
    pub fn read_file<B: BlockIo>(&self, io: &mut B, path: &str) -> Result<Vec<u8>, FsError> {
        let components: Vec<&str> = path
            .split(['/', '\\'])
            .filter(|part| !part.is_empty())
            .collect();
        if components.is_empty() {
            return Err(FsError::InvalidPath);
        }

        let mut current_cluster = self.bpb.root_cluster;
        for (idx, component) in components.iter().enumerate() {
            let entries = self.list_dir(io, current_cluster)?;
            let entry = entries
                .into_iter()
                .find(|e| e.name.eq_ignore_ascii_case(component))
                .ok_or(FsError::NotFound)?;

            let is_last = idx + 1 == components.len();
            if is_last {
                if entry.is_dir() {
                    return Err(FsError::IsADirectory);
                }
                if entry.size == 0 || entry.first_cluster == 0 {
                    return Ok(Vec::new());
                }
                let mut data = self.read_cluster_chain(io, entry.first_cluster)?;
                let target_len = entry.size as usize;
                if data.len() < target_len {
                    return Err(FsError::CorruptMetadata);
                }
                data.truncate(target_len);
                return Ok(data);
            } else {
                if !entry.is_dir() {
                    return Err(FsError::NotADirectory);
                }
                if entry.first_cluster < 2 {
                    return Err(FsError::InvalidCluster);
                }
                current_cluster = entry.first_cluster;
            }
        }

        Err(FsError::NotFound)
    }
}

fn format_8_3_name(raw: &[u8; 11]) -> String {
    let base = trim_ascii_spaces(&raw[0..8]);
    let ext = trim_ascii_spaces(&raw[8..11]);

    let mut out = String::new();
    for &b in base {
        out.push(b as char);
    }
    if !ext.is_empty() {
        out.push('.');
        for &b in ext {
            out.push(b as char);
        }
    }
    out
}

fn trim_ascii_spaces(mut slice: &[u8]) -> &[u8] {
    while let Some(&last) = slice.last() {
        if last == b' ' {
            slice = &slice[..slice.len() - 1];
        } else {
            break;
        }
    }
    slice
}

fn read_u16_le(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([buf[offset], buf[offset + 1]])
}

fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}
