//! `no_std` Clean-Room NTFS MFT Record, Data-Run & `$FILE_NAME` Parser.
//!
//! Provides:
//! - [`NtfsBootSector`]: BIOS Parameter Block (BPB) boot sector parser and geometry helpers.
//! - [`MftRecordHeader`] & [`apply_fixup`]: MFT `FILE` record header parser and Update Sequence
//!   Array (USA) sector fixup verification/restoration.
//! - [`AttributeIter`] & [`AttributeRecord`]: Resident and non-resident MFT attribute walker.
//! - [`decode_data_runs`] & [`NtfsData`]: `$DATA` resident/non-resident mapping-pair decoder
//!   (including signed LCN deltas and sparse runs).
//! - [`FileNameAttr`] & [`decode_utf16le`]: `$FILE_NAME` (`0x30`) parser and UTF-16LE decoder.
//! - [`self_test`]: Self-contained verification suite with small stack frames.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

/// NTFS OEM identifier in the boot sector (`"NTFS    "`).
pub const NTFS_OEM_ID: [u8; 8] = *b"NTFS    ";

/// Standard 512-byte sector size used for boot sector and MFT fixup strides.
pub const NTFS_SECTOR_SIZE: usize = 512;

/// MFT record magic (`"FILE"`).
pub const MFT_RECORD_MAGIC: [u8; 4] = *b"FILE";

/// MFT record flag: record is currently in use.
pub const MFT_FLAG_IN_USE: u16 = 0x0001;

/// MFT record flag: record represents a directory.
pub const MFT_FLAG_DIRECTORY: u16 = 0x0002;

/// NTFS attribute type identifiers.
pub const ATTR_STANDARD_INFORMATION: u32 = 0x10;
pub const ATTR_ATTRIBUTE_LIST: u32 = 0x20;
pub const ATTR_FILE_NAME: u32 = 0x30;
pub const ATTR_OBJECT_ID: u32 = 0x40;
pub const ATTR_SECURITY_DESCRIPTOR: u32 = 0x50;
pub const ATTR_VOLUME_NAME: u32 = 0x60;
pub const ATTR_VOLUME_INFORMATION: u32 = 0x70;
pub const ATTR_DATA: u32 = 0x80;
pub const ATTR_INDEX_ROOT: u32 = 0x90;
pub const ATTR_INDEX_ALLOCATION: u32 = 0xA0;
pub const ATTR_BITMAP: u32 = 0xB0;
pub const ATTR_REPARSE_POINT: u32 = 0xC0;
pub const ATTR_END: u32 = 0xFFFF_FFFF;

/// Errors returned by the NTFS parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NtfsError {
    BufferTooSmall,
    InvalidBootSignature,
    InvalidOemId,
    InvalidSectorSize,
    InvalidClusterSize,
    InvalidMftRecordSize,
    InvalidIndexRecordSize,
    InvalidMftMagic,
    InvalidFixupArray,
    FixupMismatch,
    InvalidAttributeHeader,
    AttributeOutOfBounds,
    InvalidDataRun,
    DataRunOverflow,
    InvalidFileName,
    InvalidUtf16,
    NotResident,
    NotNonResident,
    AttributeNotFound,
}

impl NtfsError {
    /// Return a static human-readable description of the error.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BufferTooSmall => "ntfs: buffer too small",
            Self::InvalidBootSignature => "ntfs: invalid boot sector signature",
            Self::InvalidOemId => "ntfs: invalid OEM ID (expected NTFS)",
            Self::InvalidSectorSize => "ntfs: invalid bytes per sector",
            Self::InvalidClusterSize => "ntfs: invalid sectors per cluster",
            Self::InvalidMftRecordSize => "ntfs: invalid MFT record size",
            Self::InvalidIndexRecordSize => "ntfs: invalid index record size",
            Self::InvalidMftMagic => "ntfs: invalid MFT FILE magic",
            Self::InvalidFixupArray => "ntfs: invalid update sequence array",
            Self::FixupMismatch => "ntfs: sector fixup USN mismatch",
            Self::InvalidAttributeHeader => "ntfs: invalid attribute header",
            Self::AttributeOutOfBounds => "ntfs: attribute extends past record bounds",
            Self::InvalidDataRun => "ntfs: invalid data run encoding",
            Self::DataRunOverflow => "ntfs: data run LCN/VCN overflow",
            Self::InvalidFileName => "ntfs: invalid $FILE_NAME attribute",
            Self::InvalidUtf16 => "ntfs: invalid UTF-16LE sequence",
            Self::NotResident => "ntfs: expected resident attribute",
            Self::NotNonResident => "ntfs: expected non-resident attribute",
            Self::AttributeNotFound => "ntfs: requested attribute not found",
        }
    }
}

impl fmt::Display for NtfsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<NtfsError> for &'static str {
    fn from(err: NtfsError) -> Self {
        err.as_str()
    }
}

#[inline]
fn read_u16_le(buf: &[u8], offset: usize) -> Result<u16, NtfsError> {
    let end = offset.checked_add(2).ok_or(NtfsError::BufferTooSmall)?;
    let slice = buf.get(offset..end).ok_or(NtfsError::BufferTooSmall)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

#[inline]
fn read_u32_le(buf: &[u8], offset: usize) -> Result<u32, NtfsError> {
    let end = offset.checked_add(4).ok_or(NtfsError::BufferTooSmall)?;
    let slice = buf.get(offset..end).ok_or(NtfsError::BufferTooSmall)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[inline]
fn read_u64_le(buf: &[u8], offset: usize) -> Result<u64, NtfsError> {
    let end = offset.checked_add(8).ok_or(NtfsError::BufferTooSmall)?;
    let slice = buf.get(offset..end).ok_or(NtfsError::BufferTooSmall)?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn decode_record_size(raw: i8, cluster_size: u32) -> Option<u32> {
    if raw > 0 {
        (raw as u32).checked_mul(cluster_size)
    } else if raw < 0 {
        let shift = (-raw) as u32;
        if !(9..=16).contains(&shift) {
            return None;
        }
        1u32.checked_shl(shift)
    } else {
        None
    }
}

/// Parsed NTFS BIOS Parameter Block (BPB) boot sector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NtfsBootSector {
    pub bytes_per_sector: u16,
    pub sectors_per_cluster: u8,
    pub total_sectors: u64,
    pub mft_lcn: u64,
    pub mft_mirr_lcn: u64,
    pub clusters_per_mft_record: i8,
    pub clusters_per_index_record: i8,
    pub volume_serial: u64,
    mft_record_size: u32,
    index_record_size: u32,
}

impl NtfsBootSector {
    /// Parse and validate a 512-byte NTFS boot sector.
    pub fn parse(sector: &[u8]) -> Result<Self, NtfsError> {
        if sector.len() < NTFS_SECTOR_SIZE {
            return Err(NtfsError::BufferTooSmall);
        }
        if sector[0x03..0x0B] != NTFS_OEM_ID {
            return Err(NtfsError::InvalidOemId);
        }
        if sector[0x1FE] != 0x55 || sector[0x1FF] != 0xAA {
            return Err(NtfsError::InvalidBootSignature);
        }

        let bytes_per_sector = read_u16_le(sector, 0x0B)?;
        if !(512..=4096).contains(&bytes_per_sector) || !bytes_per_sector.is_power_of_two() {
            return Err(NtfsError::InvalidSectorSize);
        }

        let sectors_per_cluster = sector[0x0D];
        if sectors_per_cluster == 0
            || sectors_per_cluster > 128
            || !sectors_per_cluster.is_power_of_two()
        {
            return Err(NtfsError::InvalidClusterSize);
        }

        let cluster_size = (bytes_per_sector as u32) * (sectors_per_cluster as u32);
        let total_sectors = read_u64_le(sector, 0x28)?;
        let mft_lcn = read_u64_le(sector, 0x30)?;
        let mft_mirr_lcn = read_u64_le(sector, 0x38)?;
        let clusters_per_mft_record = sector[0x40] as i8;
        let clusters_per_index_record = sector[0x44] as i8;
        let volume_serial = read_u64_le(sector, 0x48)?;

        let mft_record_size = decode_record_size(clusters_per_mft_record, cluster_size)
            .filter(|&sz| (512..=65536).contains(&sz))
            .ok_or(NtfsError::InvalidMftRecordSize)?;

        let index_record_size = decode_record_size(clusters_per_index_record, cluster_size)
            .filter(|&sz| (512..=65536).contains(&sz))
            .ok_or(NtfsError::InvalidIndexRecordSize)?;

        Ok(Self {
            bytes_per_sector,
            sectors_per_cluster,
            total_sectors,
            mft_lcn,
            mft_mirr_lcn,
            clusters_per_mft_record,
            clusters_per_index_record,
            volume_serial,
            mft_record_size,
            index_record_size,
        })
    }

    /// Cluster size in bytes.
    #[inline]
    pub const fn cluster_size(&self) -> u32 {
        (self.bytes_per_sector as u32) * (self.sectors_per_cluster as u32)
    }

    /// MFT record size in bytes (typically 1024).
    #[inline]
    pub const fn mft_record_size(&self) -> u32 {
        self.mft_record_size
    }

    /// Index allocation record size in bytes (typically 4096).
    #[inline]
    pub const fn index_record_size(&self) -> u32 {
        self.index_record_size
    }

    /// Convert a Logical Cluster Number (LCN) to a byte offset from volume start.
    #[inline]
    pub fn lcn_to_byte_offset(&self, lcn: u64) -> Option<u64> {
        lcn.checked_mul(self.cluster_size() as u64)
    }

    /// Byte offset of `$MFT` from volume start.
    #[inline]
    pub fn mft_byte_offset(&self) -> Option<u64> {
        self.lcn_to_byte_offset(self.mft_lcn)
    }
}

/// Parsed MFT `FILE` record header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MftRecordHeader {
    pub usa_offset: u16,
    pub usa_count: u16,
    pub lsn: u64,
    pub sequence_number: u16,
    pub hard_link_count: u16,
    pub first_attr_offset: u16,
    pub flags: u16,
    pub used_size: u32,
    pub allocated_size: u32,
    pub base_record_ref: u64,
    pub next_attr_id: u16,
    pub record_number: u32,
}

impl MftRecordHeader {
    /// Parse and validate an MFT `FILE` record header.
    pub fn parse(buf: &[u8]) -> Result<Self, NtfsError> {
        if buf.len() < 42 {
            return Err(NtfsError::BufferTooSmall);
        }
        if buf[0..4] != MFT_RECORD_MAGIC {
            return Err(NtfsError::InvalidMftMagic);
        }

        let usa_offset = read_u16_le(buf, 0x04)?;
        let usa_count = read_u16_le(buf, 0x06)?;
        let lsn = read_u64_le(buf, 0x08)?;
        let sequence_number = read_u16_le(buf, 0x10)?;
        let hard_link_count = read_u16_le(buf, 0x12)?;
        let first_attr_offset = read_u16_le(buf, 0x14)?;
        let flags = read_u16_le(buf, 0x16)?;
        let used_size = read_u32_le(buf, 0x18)?;
        let allocated_size = read_u32_le(buf, 0x1C)?;
        let base_record_ref = read_u64_le(buf, 0x20)?;
        let next_attr_id = read_u16_le(buf, 0x28)?;
        let record_number = if first_attr_offset >= 48 && buf.len() >= 48 {
            read_u32_le(buf, 0x2C)?
        } else {
            0
        };

        let first_attr = first_attr_offset as usize;
        let used = used_size as usize;
        if first_attr < 42 || used > buf.len() || first_attr > used {
            return Err(NtfsError::InvalidAttributeHeader);
        }

        Ok(Self {
            usa_offset,
            usa_count,
            lsn,
            sequence_number,
            hard_link_count,
            first_attr_offset,
            flags,
            used_size,
            allocated_size,
            base_record_ref,
            next_attr_id,
            record_number,
        })
    }

    /// Returns `true` if the MFT record is marked in use.
    #[inline]
    pub const fn is_in_use(&self) -> bool {
        (self.flags & MFT_FLAG_IN_USE) != 0
    }

    /// Returns `true` if the MFT record represents a directory.
    #[inline]
    pub const fn is_directory(&self) -> bool {
        (self.flags & MFT_FLAG_DIRECTORY) != 0
    }

    /// Return an iterator over the attributes in this MFT record.
    ///
    /// Call [`apply_fixup`] on the mutable record buffer before walking attributes.
    pub fn attributes<'a>(&self, record: &'a [u8]) -> Result<AttributeIter<'a>, NtfsError> {
        let start = self.first_attr_offset as usize;
        let end = self.used_size as usize;
        if end > record.len() || start > end {
            return Err(NtfsError::AttributeOutOfBounds);
        }
        Ok(AttributeIter {
            slice: &record[start..end],
            offset: 0,
            done: false,
        })
    }
}

/// Verify and apply the Update Sequence Array (USA) fixup to an MFT record buffer.
pub fn apply_fixup(record: &mut [u8]) -> Result<(), NtfsError> {
    if record.len() < 42 {
        return Err(NtfsError::BufferTooSmall);
    }
    if record[0..4] != MFT_RECORD_MAGIC {
        return Err(NtfsError::InvalidMftMagic);
    }

    let usa_offset = read_u16_le(record, 0x04)? as usize;
    let usa_count = read_u16_le(record, 0x06)? as usize;
    if usa_count < 2 || usa_offset < 42 {
        return Err(NtfsError::InvalidFixupArray);
    }

    let usa_bytes = usa_count
        .checked_mul(2)
        .ok_or(NtfsError::InvalidFixupArray)?;
    let usa_end = usa_offset
        .checked_add(usa_bytes)
        .ok_or(NtfsError::InvalidFixupArray)?;
    if usa_end > record.len() {
        return Err(NtfsError::InvalidFixupArray);
    }

    let sectors_covered = usa_count - 1;
    let covered_bytes = sectors_covered
        .checked_mul(NTFS_SECTOR_SIZE)
        .ok_or(NtfsError::InvalidFixupArray)?;
    if covered_bytes > record.len() || usa_end > NTFS_SECTOR_SIZE - 2 {
        return Err(NtfsError::InvalidFixupArray);
    }

    let usn0 = record[usa_offset];
    let usn1 = record[usa_offset + 1];

    for idx in 1..usa_count {
        let entry_off = usa_offset + idx * 2;
        let tail_off = idx * NTFS_SECTOR_SIZE - 2;
        if record[tail_off] != usn0 || record[tail_off + 1] != usn1 {
            return Err(NtfsError::FixupMismatch);
        }
        let orig0 = record[entry_off];
        let orig1 = record[entry_off + 1];
        record[tail_off] = orig0;
        record[tail_off + 1] = orig1;
    }

    Ok(())
}

/// Resident or non-resident payload of an MFT attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeBody<'a> {
    Resident {
        value: &'a [u8],
        resident_flags: u8,
    },
    NonResident {
        lowest_vcn: u64,
        highest_vcn: u64,
        compression_unit: u16,
        allocated_size: u64,
        data_size: u64,
        initialized_size: u64,
        data_runs: &'a [u8],
    },
}

/// Parsed MFT attribute record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttributeRecord<'a> {
    pub attr_type: u32,
    pub flags: u16,
    pub attr_id: u16,
    pub name_utf16le: &'a [u8],
    pub body: AttributeBody<'a>,
}

impl<'a> AttributeRecord<'a> {
    /// Decode the attribute's optional UTF-16LE name.
    pub fn name(&self) -> Result<String, NtfsError> {
        decode_utf16le(self.name_utf16le)
    }
}

/// Iterator over attributes inside an MFT record.
pub struct AttributeIter<'a> {
    slice: &'a [u8],
    offset: usize,
    done: bool,
}

impl<'a> Iterator for AttributeIter<'a> {
    type Item = Result<AttributeRecord<'a>, NtfsError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.offset >= self.slice.len() {
            return None;
        }

        let rem = &self.slice[self.offset..];
        if rem.len() < 4 {
            self.done = true;
            return Some(Err(NtfsError::AttributeOutOfBounds));
        }

        let attr_type = match read_u32_le(rem, 0x00) {
            Ok(v) => v,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };

        if attr_type == ATTR_END {
            self.done = true;
            return None;
        }

        if rem.len() < 16 {
            self.done = true;
            return Some(Err(NtfsError::InvalidAttributeHeader));
        }

        let length = match read_u32_le(rem, 0x04) {
            Ok(v) => v as usize,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };

        if length < 16 || !length.is_multiple_of(8) || length > rem.len() {
            self.done = true;
            return Some(Err(NtfsError::InvalidAttributeHeader));
        }

        let attr_slice = &rem[..length];
        self.offset += length;

        let non_resident = attr_slice[0x08];
        let name_len = attr_slice[0x09] as usize;
        let name_offset = match read_u16_le(attr_slice, 0x0A) {
            Ok(v) => v as usize,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        let flags = match read_u16_le(attr_slice, 0x0C) {
            Ok(v) => v,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        let attr_id = match read_u16_le(attr_slice, 0x0E) {
            Ok(v) => v,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };

        let name_utf16le = if name_len == 0 {
            &attr_slice[0..0]
        } else {
            let name_bytes = name_len * 2;
            let Some(name_end) = name_offset.checked_add(name_bytes) else {
                self.done = true;
                return Some(Err(NtfsError::InvalidAttributeHeader));
            };
            if name_end > attr_slice.len() {
                self.done = true;
                return Some(Err(NtfsError::AttributeOutOfBounds));
            }
            &attr_slice[name_offset..name_end]
        };

        let body = if non_resident == 0 {
            if attr_slice.len() < 24 {
                self.done = true;
                return Some(Err(NtfsError::InvalidAttributeHeader));
            }
            let value_len = match read_u32_le(attr_slice, 0x10) {
                Ok(v) => v as usize,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            let value_off = match read_u16_le(attr_slice, 0x14) {
                Ok(v) => v as usize,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            let resident_flags = attr_slice[0x16];
            let Some(value_end) = value_off.checked_add(value_len) else {
                self.done = true;
                return Some(Err(NtfsError::AttributeOutOfBounds));
            };
            if value_end > attr_slice.len() {
                self.done = true;
                return Some(Err(NtfsError::AttributeOutOfBounds));
            }
            AttributeBody::Resident {
                value: &attr_slice[value_off..value_end],
                resident_flags,
            }
        } else {
            if attr_slice.len() < 64 {
                self.done = true;
                return Some(Err(NtfsError::InvalidAttributeHeader));
            }
            let lowest_vcn = match read_u64_le(attr_slice, 0x10) {
                Ok(v) => v,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            let highest_vcn = match read_u64_le(attr_slice, 0x18) {
                Ok(v) => v,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            let data_runs_off = match read_u16_le(attr_slice, 0x20) {
                Ok(v) => v as usize,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            let compression_unit = match read_u16_le(attr_slice, 0x22) {
                Ok(v) => v,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            let allocated_size = match read_u64_le(attr_slice, 0x28) {
                Ok(v) => v,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            let data_size = match read_u64_le(attr_slice, 0x30) {
                Ok(v) => v,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            let initialized_size = match read_u64_le(attr_slice, 0x38) {
                Ok(v) => v,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            if data_runs_off > attr_slice.len() {
                self.done = true;
                return Some(Err(NtfsError::AttributeOutOfBounds));
            }
            AttributeBody::NonResident {
                lowest_vcn,
                highest_vcn,
                compression_unit,
                allocated_size,
                data_size,
                initialized_size,
                data_runs: &attr_slice[data_runs_off..],
            }
        };

        Some(Ok(AttributeRecord {
            attr_type,
            flags,
            attr_id,
            name_utf16le,
            body,
        }))
    }
}

/// A single decoded NTFS data run (extent mapping VCNs to LCNs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataRun {
    pub vcn_start: u64,
    pub cluster_count: u64,
    /// `Some(lcn)` for allocated clusters, `None` for sparse runs.
    pub lcn: Option<u64>,
}

/// Decode NTFS mapping-pair data runs into a list of [`DataRun`] extents.
pub fn decode_data_runs(runs: &[u8]) -> Result<Vec<DataRun>, NtfsError> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut current_lcn: i64 = 0;
    let mut vcn_start: u64 = 0;

    while pos < runs.len() {
        let header = runs[pos];
        pos += 1;
        if header == 0 {
            break;
        }

        let len_bytes = (header & 0x0F) as usize;
        let off_bytes = ((header >> 4) & 0x0F) as usize;
        if len_bytes == 0 || len_bytes > 8 || off_bytes > 8 {
            return Err(NtfsError::InvalidDataRun);
        }

        let needed = len_bytes + off_bytes;
        if pos + needed > runs.len() {
            return Err(NtfsError::InvalidDataRun);
        }

        let mut len_buf = [0u8; 8];
        len_buf[..len_bytes].copy_from_slice(&runs[pos..pos + len_bytes]);
        let cluster_count = u64::from_le_bytes(len_buf);
        if cluster_count == 0 {
            return Err(NtfsError::InvalidDataRun);
        }
        pos += len_bytes;

        let lcn = if off_bytes == 0 {
            None
        } else {
            let off_slice = &runs[pos..pos + off_bytes];
            let sign_fill = if (off_slice[off_bytes - 1] & 0x80) != 0 {
                0xFF
            } else {
                0x00
            };
            let mut off_buf = [sign_fill; 8];
            off_buf[..off_bytes].copy_from_slice(off_slice);
            let delta = i64::from_le_bytes(off_buf);
            pos += off_bytes;

            current_lcn = current_lcn
                .checked_add(delta)
                .ok_or(NtfsError::DataRunOverflow)?;
            if current_lcn < 0 {
                return Err(NtfsError::DataRunOverflow);
            }
            Some(current_lcn as u64)
        };

        out.push(DataRun {
            vcn_start,
            cluster_count,
            lcn,
        });

        vcn_start = vcn_start
            .checked_add(cluster_count)
            .ok_or(NtfsError::DataRunOverflow)?;
    }

    Ok(out)
}

/// Decoded `$DATA` (`0x80`) attribute payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NtfsData<'a> {
    Resident(&'a [u8]),
    NonResident {
        data_size: u64,
        allocated_size: u64,
        initialized_size: u64,
        runs: Vec<DataRun>,
    },
}

/// Parse a `$DATA` attribute record into [`NtfsData`].
pub fn parse_data_attribute<'a>(attr: &AttributeRecord<'a>) -> Result<NtfsData<'a>, NtfsError> {
    match attr.body {
        AttributeBody::Resident { value, .. } => Ok(NtfsData::Resident(value)),
        AttributeBody::NonResident {
            allocated_size,
            data_size,
            initialized_size,
            data_runs,
            ..
        } => {
            let runs = decode_data_runs(data_runs)?;
            Ok(NtfsData::NonResident {
                data_size,
                allocated_size,
                initialized_size,
                runs,
            })
        }
    }
}

/// `$FILE_NAME` namespace classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileNamespace {
    Posix,
    Win32,
    Dos,
    Win32AndDos,
    Unknown(u8),
}

impl FileNamespace {
    #[inline]
    pub const fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Posix,
            1 => Self::Win32,
            2 => Self::Dos,
            3 => Self::Win32AndDos,
            other => Self::Unknown(other),
        }
    }
}

/// Decode a UTF-16LE byte slice into a UTF-8 [`String`].
pub fn decode_utf16le(bytes: &[u8]) -> Result<String, NtfsError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(NtfsError::InvalidUtf16);
    }
    let (pairs, _) = bytes.as_chunks::<2>();
    let mut out = String::with_capacity(pairs.len());
    for decoded in char::decode_utf16(pairs.iter().copied().map(u16::from_le_bytes)) {
        let ch = decoded.map_err(|_| NtfsError::InvalidUtf16)?;
        out.push(ch);
    }
    Ok(out)
}

/// Parsed `$FILE_NAME` (`0x30`) attribute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileNameAttr {
    pub parent_mft_record: u64,
    pub parent_sequence: u16,
    pub creation_time: u64,
    pub modification_time: u64,
    pub mft_modification_time: u64,
    pub access_time: u64,
    pub allocated_size: u64,
    pub real_size: u64,
    pub flags: u32,
    pub reparse_value: u32,
    pub namespace: FileNamespace,
    pub name: String,
}

impl FileNameAttr {
    /// Parse a resident `$FILE_NAME` attribute value.
    pub fn parse(value: &[u8]) -> Result<Self, NtfsError> {
        if value.len() < 0x42 {
            return Err(NtfsError::BufferTooSmall);
        }

        let parent_ref = read_u64_le(value, 0x00)?;
        let parent_mft_record = parent_ref & 0x0000_FFFF_FFFF_FFFF;
        let parent_sequence = (parent_ref >> 48) as u16;

        let creation_time = read_u64_le(value, 0x08)?;
        let modification_time = read_u64_le(value, 0x10)?;
        let mft_modification_time = read_u64_le(value, 0x18)?;
        let access_time = read_u64_le(value, 0x20)?;
        let allocated_size = read_u64_le(value, 0x28)?;
        let real_size = read_u64_le(value, 0x30)?;
        let flags = read_u32_le(value, 0x38)?;
        let reparse_value = read_u32_le(value, 0x3C)?;

        let name_len = value[0x40] as usize;
        let namespace = FileNamespace::from_raw(value[0x41]);
        let name_bytes = name_len * 2;
        let name_end = 0x42usize
            .checked_add(name_bytes)
            .ok_or(NtfsError::InvalidFileName)?;
        if name_end > value.len() {
            return Err(NtfsError::InvalidFileName);
        }

        let name = decode_utf16le(&value[0x42..name_end])?;

        Ok(Self {
            parent_mft_record,
            parent_sequence,
            creation_time,
            modification_time,
            mft_modification_time,
            access_time,
            allocated_size,
            real_size,
            flags,
            reparse_value,
            namespace,
            name,
        })
    }
}

fn write_u16_le(buf: &mut [u8], offset: usize, val: u16) {
    buf[offset..offset + 2].copy_from_slice(&val.to_le_bytes());
}

fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

fn write_u64_le(buf: &mut [u8], offset: usize, val: u64) {
    buf[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
}

/// Run self-contained NTFS parser tests (BPB, MFT fixup, attribute walker,
/// resident/non-resident `$DATA` runs, `$FILE_NAME`, and UTF-16LE decoding).
pub fn self_test() -> Result<(), &'static str> {
    // 1. BPB boot sector test (heap-allocated to keep stack frame < 2 KiB).
    let mut boot = vec![0u8; NTFS_SECTOR_SIZE];
    boot[0x03..0x0B].copy_from_slice(&NTFS_OEM_ID);
    write_u16_le(&mut boot, 0x0B, 512);
    boot[0x0D] = 8; // 4096-byte clusters
    write_u64_le(&mut boot, 0x28, 204_800);
    write_u64_le(&mut boot, 0x30, 4); // $MFT at LCN 4
    write_u64_le(&mut boot, 0x38, 1024);
    boot[0x40] = (-10i8) as u8; // 1024-byte MFT records
    boot[0x44] = 1; // 1 cluster (4096-byte) index records
    write_u64_le(&mut boot, 0x48, 0xDEAD_BEEF_CAFE_BABE);
    boot[0x1FE] = 0x55;
    boot[0x1FF] = 0xAA;

    let bpb = NtfsBootSector::parse(&boot)?;
    if bpb.cluster_size() != 4096
        || bpb.mft_record_size() != 1024
        || bpb.index_record_size() != 4096
        || bpb.mft_byte_offset() != Some(16_384)
    {
        return Err("ntfs: BPB geometry mismatch");
    }

    // 2. MFT FILE record + USA fixup + $FILE_NAME + resident $DATA + non-resident $DATA.
    let mut mft = vec![0u8; 1024];
    mft[0..4].copy_from_slice(&MFT_RECORD_MAGIC);
    write_u16_le(&mut mft, 0x04, 0x30); // usa_offset
    write_u16_le(&mut mft, 0x06, 3); // usa_count (USN + 2 sectors)
    write_u64_le(&mut mft, 0x08, 42); // lsn
    write_u16_le(&mut mft, 0x10, 1); // sequence_number
    write_u16_le(&mut mft, 0x12, 1); // hard_link_count
    write_u16_le(&mut mft, 0x14, 0x38); // first_attr_offset
    write_u16_le(&mut mft, 0x16, MFT_FLAG_IN_USE);
    write_u32_le(&mut mft, 0x1C, 1024); // allocated_size
    write_u32_le(&mut mft, 0x2C, 64); // record_number

    // USA: USN = 0x0007, sector 0 original tail = 0xAA55, sector 1 original tail = 0x1234.
    write_u16_le(&mut mft, 0x30, 0x0007);
    write_u16_le(&mut mft, 0x32, 0xAA55);
    write_u16_le(&mut mft, 0x34, 0x1234);
    write_u16_le(&mut mft, 510, 0x0007);
    write_u16_le(&mut mft, 1022, 0x0007);

    // Attribute 1: $FILE_NAME (0x30), resident, name = "hello.txt" (9 UTF-16LE chars = 18 bytes).
    let mut off = 0x38usize;
    let fn_name = "hello.txt";
    let fn_val_len = 0x42 + fn_name.len() * 2; // 84 bytes
    let fn_attr_len = (24 + fn_val_len + 7) & !7; // 112 bytes (8-byte aligned)
    write_u32_le(&mut mft, off, ATTR_FILE_NAME);
    write_u32_le(&mut mft, off + 0x04, fn_attr_len as u32);
    mft[off + 0x08] = 0; // resident
    write_u32_le(&mut mft, off + 0x10, fn_val_len as u32);
    write_u16_le(&mut mft, off + 0x14, 24);
    let fn_val = off + 24;
    write_u64_le(&mut mft, fn_val, (1u64 << 48) | 5); // parent MFT record 5, seq 1
    write_u64_le(&mut mft, fn_val + 0x30, 13); // real_size
    mft[fn_val + 0x40] = fn_name.len() as u8;
    mft[fn_val + 0x41] = 1; // Win32
    for (idx, byte) in fn_name.bytes().enumerate() {
        write_u16_le(&mut mft, fn_val + 0x42 + idx * 2, byte as u16);
    }
    off += fn_attr_len;

    // Attribute 2: $DATA (0x80), resident, payload = b"nexus-ntfs-ok".
    let payload = b"nexus-ntfs-ok";
    let res_attr_len = (24 + payload.len() + 7) & !7; // 40 bytes
    write_u32_le(&mut mft, off, ATTR_DATA);
    write_u32_le(&mut mft, off + 0x04, res_attr_len as u32);
    mft[off + 0x08] = 0;
    write_u32_le(&mut mft, off + 0x10, payload.len() as u32);
    write_u16_le(&mut mft, off + 0x14, 24);
    mft[off + 24..off + 24 + payload.len()].copy_from_slice(payload);
    off += res_attr_len;

    // Attribute 3: $DATA (0x80), non-resident, 3 runs:
    //   Run 1: header 0x21 (len=1, off=2) -> count=4, delta=+0x0100 (LCN 256)
    //   Run 2: header 0x01 (len=1, off=0) -> count=2, sparse (LCN None)
    //   Run 3: header 0x11 (len=1, off=1) -> count=3, delta=-16 (0xF0 -> LCN 240)
    //   Terminator: 0x00
    let runs_bytes: [u8; 8] = [0x21, 0x04, 0x00, 0x01, 0x01, 0x02, 0x11, 0x03];
    let runs_tail: [u8; 2] = [0xF0, 0x00];
    let nr_attr_len = (64 + runs_bytes.len() + runs_tail.len() + 7) & !7; // 80 bytes
    write_u32_le(&mut mft, off, ATTR_DATA);
    write_u32_le(&mut mft, off + 0x04, nr_attr_len as u32);
    mft[off + 0x08] = 1; // non-resident
    write_u64_le(&mut mft, off + 0x10, 0); // lowest_vcn
    write_u64_le(&mut mft, off + 0x18, 8); // highest_vcn
    write_u16_le(&mut mft, off + 0x20, 64); // data_runs_offset
    write_u64_le(&mut mft, off + 0x28, 9 * 4096); // allocated_size
    write_u64_le(&mut mft, off + 0x30, 35_000); // data_size
    write_u64_le(&mut mft, off + 0x38, 35_000); // initialized_size
    mft[off + 64..off + 64 + runs_bytes.len()].copy_from_slice(&runs_bytes);
    mft[off + 64 + runs_bytes.len()..off + 64 + runs_bytes.len() + runs_tail.len()]
        .copy_from_slice(&runs_tail);
    off += nr_attr_len;

    // End marker.
    write_u32_le(&mut mft, off, ATTR_END);
    write_u32_le(&mut mft, 0x18, (off + 4) as u32); // used_size

    // Verify fixup mismatch detection on a corrupted copy, then apply fixup on real buffer.
    let mut bad_mft = mft.clone();
    bad_mft[1022] = 0x99;
    if apply_fixup(&mut bad_mft) != Err(NtfsError::FixupMismatch) {
        return Err("ntfs: expected FixupMismatch on corrupted sector tail");
    }

    apply_fixup(&mut mft)?;
    if mft[510..512] != 0xAA55u16.to_le_bytes() || mft[1022..1024] != 0x1234u16.to_le_bytes() {
        return Err("ntfs: fixup did not restore sector tails");
    }

    let hdr = MftRecordHeader::parse(&mft)?;
    if !hdr.is_in_use() || hdr.is_directory() || hdr.record_number != 64 {
        return Err("ntfs: MFT header fields mismatch");
    }

    let mut saw_fn = false;
    let mut saw_res_data = false;
    let mut saw_nr_data = false;

    for attr_res in hdr.attributes(&mft)? {
        let attr = attr_res?;
        match attr.attr_type {
            ATTR_FILE_NAME => {
                let AttributeBody::Resident { value, .. } = attr.body else {
                    return Err("ntfs: expected resident $FILE_NAME");
                };
                let parsed = FileNameAttr::parse(value)?;
                if parsed.name != "hello.txt"
                    || parsed.parent_mft_record != 5
                    || parsed.parent_sequence != 1
                    || parsed.namespace != FileNamespace::Win32
                {
                    return Err("ntfs: $FILE_NAME parse mismatch");
                }
                saw_fn = true;
            }
            ATTR_DATA => match parse_data_attribute(&attr)? {
                NtfsData::Resident(bytes) => {
                    if bytes != payload {
                        return Err("ntfs: resident $DATA mismatch");
                    }
                    saw_res_data = true;
                }
                NtfsData::NonResident {
                    data_size, runs, ..
                } => {
                    if data_size != 35_000 || runs.len() != 3 {
                        return Err("ntfs: non-resident $DATA count mismatch");
                    }
                    if runs[0]
                        != (DataRun {
                            vcn_start: 0,
                            cluster_count: 4,
                            lcn: Some(256),
                        })
                        || runs[1]
                            != (DataRun {
                                vcn_start: 4,
                                cluster_count: 2,
                                lcn: None,
                            })
                        || runs[2]
                            != (DataRun {
                                vcn_start: 6,
                                cluster_count: 3,
                                lcn: Some(240),
                            })
                    {
                        return Err("ntfs: decoded data runs mismatch");
                    }
                    saw_nr_data = true;
                }
            },
            _ => {}
        }
    }

    if !saw_fn || !saw_res_data || !saw_nr_data {
        return Err("ntfs: missing expected attributes in self_test");
    }

    // 3. UTF-16LE surrogate pair & invalid surrogate test.
    let rocket_utf16le: [u8; 4] = [0x3D, 0xD8, 0x80, 0xDE]; // U+1F680
    let rocket = decode_utf16le(&rocket_utf16le)?;
    if rocket != "\u{1F680}" {
        return Err("ntfs: UTF-16LE surrogate pair decode mismatch");
    }
    let bad_surrogate: [u8; 2] = [0x3D, 0xD8];
    if decode_utf16le(&bad_surrogate) != Err(NtfsError::InvalidUtf16) {
        return Err("ntfs: expected InvalidUtf16 on unpaired surrogate");
    }

    crate::serial_println!(
        "NTFS: self_test passed (BPB, MFT fixup, $FILE_NAME, $DATA runs, UTF-16LE)"
    );
    Ok(())
}
