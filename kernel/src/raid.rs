//! Software RAID (0/1/5/6/10) & LVM/Device-Mapper Volume Manager.
//!
//! Implements `no_std` block-level storage virtualization:
//! - Software RAID levels:
//!   - RAID 0 (Striping)
//!   - RAID 1 (Mirroring with degraded read/write & rebuild)
//!   - RAID 5 (Left-symmetric distributed XOR parity, single-disk fault tolerance & rebuild)
//!   - RAID 6 (Left-symmetric distributed P+Q Reed-Solomon GF(2^8) dual parity,
//!     dual-disk fault tolerance & rebuild)
//!   - RAID 10 (Striped mirror pairs, 1-disk-per-pair fault tolerance & rebuild)
//! - LVM / Device-Mapper Volume Manager:
//!   - Physical Volumes (PV), Volume Groups (VG), Logical Volumes (LV)
//!   - Linear, Striped, and Mirror DM targets
//!   - Thin-Provisioning Pool & Thin Volume extent mapper (on-demand allocation,
//!     zero-on-read for unmapped extents, discard/unmap)
//!   - Snapshot Copy-on-Write (COW) target & Origin-with-Snapshots coordinator
//! - `self_test()` with comprehensive in-memory verification

#![allow(dead_code)]

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::blk::{BlockDevice, SECTOR_SIZE};
use crate::mm::SpinLock;

/// Default stripe chunk size in 512-byte blocks (4 KiB).
pub const DEFAULT_CHUNK_BLOCKS: u64 = 8;

/// Errors returned by RAID and LVM/Device-Mapper operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaidError {
    InvalidConfig,
    InvalidDiskIndex,
    DiskFailed,
    TooManyFailedDisks,
    OutOfBounds,
    UnalignedBuffer,
    BlockSizeMismatch,
    NoSpace,
    NotFound,
    IoError,
}

impl RaidError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidConfig => "raid: invalid configuration",
            Self::InvalidDiskIndex => "raid: invalid disk index",
            Self::DiskFailed => "raid: disk is failed",
            Self::TooManyFailedDisks => "raid: too many failed disks",
            Self::OutOfBounds => "raid: LBA out of bounds",
            Self::UnalignedBuffer => "raid: buffer length not a multiple of block size",
            Self::BlockSizeMismatch => "raid: member block size mismatch",
            Self::NoSpace => "lvm: insufficient free extents or pool chunks",
            Self::NotFound => "lvm: volume or extent not found",
            Self::IoError => "raid: underlying block I/O error",
        }
    }
}

impl fmt::Display for RaidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<RaidError> for &'static str {
    fn from(err: RaidError) -> Self {
        err.as_str()
    }
}

// ---------------------------------------------------------------------------
// In-Memory Block Device (for testing & virtual backing stores)
// ---------------------------------------------------------------------------

struct MemDiskInner {
    data: Vec<u8>,
    online: bool,
}

/// Thread-safe in-memory `BlockDevice` with fault-injection support.
pub struct MemBlockDevice {
    block_size: usize,
    block_count: u64,
    inner: SpinLock<MemDiskInner>,
}

impl MemBlockDevice {
    /// Create a zero-initialized in-memory block device.
    pub fn new(block_count: u64, block_size: usize) -> Result<Self, RaidError> {
        if block_size == 0 || block_count == 0 {
            return Err(RaidError::InvalidConfig);
        }
        let total_bytes = (block_count as usize)
            .checked_mul(block_size)
            .ok_or(RaidError::InvalidConfig)?;
        Ok(Self {
            block_size,
            block_count,
            inner: SpinLock::new(MemDiskInner {
                data: vec![0u8; total_bytes],
                online: true,
            }),
        })
    }

    /// Simulate disk failure or recovery at the hardware level.
    pub fn set_online(&self, online: bool) {
        self.inner.lock().online = online;
    }

    /// Check whether the simulated disk is online.
    pub fn is_online(&self) -> bool {
        self.inner.lock().online
    }
}

impl BlockDevice for MemBlockDevice {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn block_count(&self) -> u64 {
        self.block_count
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(self.block_size) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / self.block_size) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.block_count {
            return Err(RaidError::OutOfBounds.as_str());
        }
        let guard = self.inner.lock();
        if !guard.online {
            return Err(RaidError::IoError.as_str());
        }
        let start = (lba as usize) * self.block_size;
        let end = start + buf.len();
        buf.copy_from_slice(&guard.data[start..end]);
        Ok(())
    }

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(self.block_size) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / self.block_size) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.block_count {
            return Err(RaidError::OutOfBounds.as_str());
        }
        let mut guard = self.inner.lock();
        if !guard.online {
            return Err(RaidError::IoError.as_str());
        }
        let start = (lba as usize) * self.block_size;
        let end = start + buf.len();
        guard.data[start..end].copy_from_slice(buf);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GF(2^8) Arithmetic for RAID 6 P+Q Syndrome (polynomial 0x11D, g = 0x02)
// ---------------------------------------------------------------------------

const GF_POLY: u16 = 0x11D;

const fn build_gf_tables() -> ([u8; 256], [u8; 256]) {
    let mut exp = [0u8; 256];
    let mut log = [0u8; 256];
    let mut x: u16 = 1;
    let mut i = 0usize;
    while i < 255 {
        exp[i] = x as u8;
        log[x as usize] = i as u8;
        x <<= 1;
        if (x & 0x100) != 0 {
            x ^= GF_POLY;
        }
        i += 1;
    }
    exp[255] = exp[0];
    (exp, log)
}

const GF_TABLES: ([u8; 256], [u8; 256]) = build_gf_tables();
const GF_EXP: [u8; 256] = GF_TABLES.0;
const GF_LOG: [u8; 256] = GF_TABLES.1;

#[inline]
fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let sum = (GF_LOG[a as usize] as usize) + (GF_LOG[b as usize] as usize);
    GF_EXP[sum % 255]
}

#[inline]
fn gf_inv(a: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    let idx = 255 - (GF_LOG[a as usize] as usize);
    GF_EXP[idx % 255]
}

#[inline]
fn gf_pow2(exp: usize) -> u8 {
    GF_EXP[exp % 255]
}

/// XOR `src` into `dst` (`dst[i] ^= src[i]`).
#[inline]
fn xor_into(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= *s;
    }
}

/// Multiply `src` by `coeff` in GF(2^8) and XOR into `dst`.
#[inline]
fn gf_mul_xor_into(dst: &mut [u8], src: &[u8], coeff: u8) {
    if coeff == 0 {
        return;
    }
    if coeff == 1 {
        xor_into(dst, src);
        return;
    }
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d ^= gf_mul(s, coeff);
    }
}

// ---------------------------------------------------------------------------
// Software RAID (0 / 1 / 5 / 6 / 10)
// ---------------------------------------------------------------------------

/// Supported Software RAID levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaidLevel {
    /// Striping across N >= 1 disks.
    Raid0,
    /// Mirroring across N >= 2 disks.
    Raid1,
    /// Distributed XOR parity across N >= 3 disks (left-symmetric).
    Raid5,
    /// Distributed P+Q dual parity across N >= 4 disks (left-symmetric).
    Raid6,
    /// Striped mirror pairs across N >= 4 (even) disks.
    Raid10,
}

/// Operational state of a member disk in a RAID array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskState {
    Online,
    Failed,
    Rebuilding,
}

struct RaidMember {
    dev: Arc<dyn BlockDevice>,
    state: DiskState,
}

struct RaidState {
    members: Vec<RaidMember>,
    read_rr: usize,
}

/// Software RAID array implementing `BlockDevice`.
pub struct RaidArray {
    level: RaidLevel,
    block_size: usize,
    chunk_blocks: u64,
    usable_blocks_per_disk: u64,
    total_blocks: u64,
    state: SpinLock<RaidState>,
}

impl RaidArray {
    /// Create a new Software RAID array from member block devices.
    pub fn new(
        level: RaidLevel,
        devices: Vec<Arc<dyn BlockDevice>>,
        chunk_blocks: u64,
    ) -> Result<Self, RaidError> {
        let n = devices.len();
        let min_disks = match level {
            RaidLevel::Raid0 => 1,
            RaidLevel::Raid1 => 2,
            RaidLevel::Raid5 => 3,
            RaidLevel::Raid6 => 4,
            RaidLevel::Raid10 => 4,
        };
        if n < min_disks {
            return Err(RaidError::InvalidConfig);
        }
        if level == RaidLevel::Raid10 && !n.is_multiple_of(2) {
            return Err(RaidError::InvalidConfig);
        }
        if chunk_blocks == 0 {
            return Err(RaidError::InvalidConfig);
        }

        let block_size = devices[0].block_size();
        if block_size == 0 {
            return Err(RaidError::InvalidConfig);
        }
        let mut min_blocks = u64::MAX;
        for dev in &devices {
            if dev.block_size() != block_size {
                return Err(RaidError::BlockSizeMismatch);
            }
            let cnt = dev.block_count();
            if cnt < min_blocks {
                min_blocks = cnt;
            }
        }

        let usable_blocks_per_disk = match level {
            RaidLevel::Raid1 => min_blocks,
            _ => (min_blocks / chunk_blocks) * chunk_blocks,
        };
        if usable_blocks_per_disk == 0 {
            return Err(RaidError::InvalidConfig);
        }

        let total_blocks = match level {
            RaidLevel::Raid0 => usable_blocks_per_disk * (n as u64),
            RaidLevel::Raid1 => usable_blocks_per_disk,
            RaidLevel::Raid5 => usable_blocks_per_disk * ((n - 1) as u64),
            RaidLevel::Raid6 => usable_blocks_per_disk * ((n - 2) as u64),
            RaidLevel::Raid10 => usable_blocks_per_disk * ((n / 2) as u64),
        };

        let members = devices
            .into_iter()
            .map(|dev| RaidMember {
                dev,
                state: DiskState::Online,
            })
            .collect();

        Ok(Self {
            level,
            block_size,
            chunk_blocks,
            usable_blocks_per_disk,
            total_blocks,
            state: SpinLock::new(RaidState {
                members,
                read_rr: 0,
            }),
        })
    }

    pub fn level(&self) -> RaidLevel {
        self.level
    }

    pub fn disk_count(&self) -> usize {
        self.state.lock().members.len()
    }

    pub fn disk_state(&self, idx: usize) -> Result<DiskState, RaidError> {
        let guard = self.state.lock();
        guard
            .members
            .get(idx)
            .map(|m| m.state)
            .ok_or(RaidError::InvalidDiskIndex)
    }

    /// Mark a member disk as failed.
    pub fn mark_disk_failed(&self, idx: usize) -> Result<(), RaidError> {
        let mut guard = self.state.lock();
        let member = guard
            .members
            .get_mut(idx)
            .ok_or(RaidError::InvalidDiskIndex)?;
        member.state = DiskState::Failed;
        Ok(())
    }

    /// Mark a member disk as online without rebuilding.
    pub fn mark_disk_online(&self, idx: usize) -> Result<(), RaidError> {
        let mut guard = self.state.lock();
        let member = guard
            .members
            .get_mut(idx)
            .ok_or(RaidError::InvalidDiskIndex)?;
        member.state = DiskState::Online;
        Ok(())
    }

    /// Rebuild a failed/replaced disk from surviving members.
    pub fn rebuild_disk(&self, target_idx: usize) -> Result<(), RaidError> {
        let mut guard = self.state.lock();
        let n = guard.members.len();
        if target_idx >= n {
            return Err(RaidError::InvalidDiskIndex);
        }
        if self.level == RaidLevel::Raid0 {
            return Err(RaidError::TooManyFailedDisks);
        }

        guard.members[target_idx].state = DiskState::Rebuilding;

        let res = match self.level {
            RaidLevel::Raid0 => Err(RaidError::TooManyFailedDisks),
            RaidLevel::Raid1 => self.rebuild_raid1(&guard.members, target_idx),
            RaidLevel::Raid5 => self.rebuild_raid5(&guard.members, target_idx),
            RaidLevel::Raid6 => self.rebuild_raid6(&guard.members, target_idx),
            RaidLevel::Raid10 => self.rebuild_raid10(&guard.members, target_idx),
        };

        if res.is_ok() {
            guard.members[target_idx].state = DiskState::Online;
        } else {
            guard.members[target_idx].state = DiskState::Failed;
        }
        res
    }

    fn rebuild_raid1(&self, members: &[RaidMember], target_idx: usize) -> Result<(), RaidError> {
        let src_idx = members
            .iter()
            .enumerate()
            .find(|&(i, m)| i != target_idx && m.state == DiskState::Online)
            .map(|(i, _)| i)
            .ok_or(RaidError::TooManyFailedDisks)?;

        let mut buf = vec![0u8; self.block_size];
        for lba in 0..self.usable_blocks_per_disk {
            members[src_idx]
                .dev
                .read_blocks(lba, &mut buf)
                .map_err(|_| RaidError::IoError)?;
            members[target_idx]
                .dev
                .write_blocks(lba, &buf)
                .map_err(|_| RaidError::IoError)?;
        }
        Ok(())
    }

    fn rebuild_raid10(&self, members: &[RaidMember], target_idx: usize) -> Result<(), RaidError> {
        let peer_idx = target_idx ^ 1;
        if members[peer_idx].state != DiskState::Online {
            return Err(RaidError::TooManyFailedDisks);
        }
        let mut buf = vec![0u8; self.block_size];
        for lba in 0..self.usable_blocks_per_disk {
            members[peer_idx]
                .dev
                .read_blocks(lba, &mut buf)
                .map_err(|_| RaidError::IoError)?;
            members[target_idx]
                .dev
                .write_blocks(lba, &buf)
                .map_err(|_| RaidError::IoError)?;
        }
        Ok(())
    }

    fn rebuild_raid5(&self, members: &[RaidMember], target_idx: usize) -> Result<(), RaidError> {
        let n = members.len();
        for (i, m) in members.iter().enumerate() {
            if i != target_idx && m.state != DiskState::Online {
                return Err(RaidError::TooManyFailedDisks);
            }
        }
        let mut acc = vec![0u8; self.block_size];
        let mut tmp = vec![0u8; self.block_size];
        for lba in 0..self.usable_blocks_per_disk {
            acc.fill(0);
            for (i, m) in members.iter().enumerate() {
                if i == target_idx {
                    continue;
                }
                m.dev
                    .read_blocks(lba, &mut tmp)
                    .map_err(|_| RaidError::IoError)?;
                xor_into(&mut acc, &tmp);
            }
            members[target_idx]
                .dev
                .write_blocks(lba, &acc)
                .map_err(|_| RaidError::IoError)?;
        }
        let _ = n;
        Ok(())
    }

    fn rebuild_raid6(&self, members: &[RaidMember], target_idx: usize) -> Result<(), RaidError> {
        let n = members.len();
        let num_stripes = self.usable_blocks_per_disk / self.chunk_blocks;
        let mut buf = vec![0u8; self.block_size];

        for stripe in 0..num_stripes {
            let (p_disk, q_disk, data_disks) = raid6_layout(n, stripe);
            for off in 0..self.chunk_blocks {
                let disk_lba = stripe * self.chunk_blocks + off;
                // Reconstruct all data blocks for this stripe offset, then write target_idx.
                let data_blocks = self.raid6_read_stripe_row(
                    members,
                    p_disk,
                    q_disk,
                    &data_disks,
                    disk_lba,
                    Some(target_idx),
                )?;

                if target_idx == p_disk {
                    buf.fill(0);
                    for d in &data_blocks {
                        xor_into(&mut buf, d);
                    }
                } else if target_idx == q_disk {
                    buf.fill(0);
                    for (k, d) in data_blocks.iter().enumerate() {
                        gf_mul_xor_into(&mut buf, d, gf_pow2(k));
                    }
                } else {
                    let pos = data_disks
                        .iter()
                        .position(|&d| d == target_idx)
                        .ok_or(RaidError::InvalidDiskIndex)?;
                    buf.copy_from_slice(&data_blocks[pos]);
                }

                members[target_idx]
                    .dev
                    .write_blocks(disk_lba, &buf)
                    .map_err(|_| RaidError::IoError)?;
            }
        }
        Ok(())
    }

    /// Read a single block at `lba` into `out` (length == `self.block_size`).
    fn read_single_block(
        &self,
        guard: &mut RaidState,
        lba: u64,
        out: &mut [u8],
    ) -> Result<(), RaidError> {
        let n = guard.members.len();
        match self.level {
            RaidLevel::Raid0 => {
                let chunk_idx = lba / self.chunk_blocks;
                let chunk_off = lba % self.chunk_blocks;
                let stripe_idx = chunk_idx / (n as u64);
                let disk_idx = (chunk_idx % (n as u64)) as usize;
                if guard.members[disk_idx].state != DiskState::Online {
                    return Err(RaidError::DiskFailed);
                }
                let disk_lba = stripe_idx * self.chunk_blocks + chunk_off;
                guard.members[disk_idx]
                    .dev
                    .read_blocks(disk_lba, out)
                    .map_err(|_| RaidError::IoError)
            }
            RaidLevel::Raid1 => {
                let start = guard.read_rr % n;
                guard.read_rr = guard.read_rr.wrapping_add(1);
                for offset in 0..n {
                    let idx = (start + offset) % n;
                    if guard.members[idx].state == DiskState::Online
                        && guard.members[idx].dev.read_blocks(lba, out).is_ok()
                    {
                        return Ok(());
                    }
                }
                Err(RaidError::TooManyFailedDisks)
            }
            RaidLevel::Raid10 => {
                let pairs = n / 2;
                let chunk_idx = lba / self.chunk_blocks;
                let chunk_off = lba % self.chunk_blocks;
                let stripe_idx = chunk_idx / (pairs as u64);
                let pair_idx = (chunk_idx % (pairs as u64)) as usize;
                let disk_lba = stripe_idx * self.chunk_blocks + chunk_off;
                let d0 = pair_idx * 2;
                let d1 = d0 + 1;
                if guard.members[d0].state == DiskState::Online
                    && guard.members[d0].dev.read_blocks(disk_lba, out).is_ok()
                {
                    return Ok(());
                }
                if guard.members[d1].state == DiskState::Online
                    && guard.members[d1].dev.read_blocks(disk_lba, out).is_ok()
                {
                    return Ok(());
                }
                Err(RaidError::TooManyFailedDisks)
            }
            RaidLevel::Raid5 => {
                let data_disks_per_stripe = n - 1;
                let chunk_idx = lba / self.chunk_blocks;
                let chunk_off = lba % self.chunk_blocks;
                let stripe_idx = chunk_idx / (data_disks_per_stripe as u64);
                let data_slot = (chunk_idx % (data_disks_per_stripe as u64)) as usize;
                let (parity_disk, data_disks) = raid5_layout(n, stripe_idx);
                let target_disk = data_disks[data_slot];
                let disk_lba = stripe_idx * self.chunk_blocks + chunk_off;

                if guard.members[target_disk].state == DiskState::Online
                    && guard.members[target_disk]
                        .dev
                        .read_blocks(disk_lba, out)
                        .is_ok()
                {
                    return Ok(());
                }

                // Degraded read: XOR all other disks (including parity_disk).
                out.fill(0);
                let mut tmp = vec![0u8; self.block_size];
                for (i, m) in guard.members.iter().enumerate() {
                    if i == target_disk {
                        continue;
                    }
                    if m.state != DiskState::Online {
                        return Err(RaidError::TooManyFailedDisks);
                    }
                    m.dev
                        .read_blocks(disk_lba, &mut tmp)
                        .map_err(|_| RaidError::IoError)?;
                    xor_into(out, &tmp);
                }
                let _ = parity_disk;
                Ok(())
            }
            RaidLevel::Raid6 => {
                let data_disks_per_stripe = n - 2;
                let chunk_idx = lba / self.chunk_blocks;
                let chunk_off = lba % self.chunk_blocks;
                let stripe_idx = chunk_idx / (data_disks_per_stripe as u64);
                let data_slot = (chunk_idx % (data_disks_per_stripe as u64)) as usize;
                let (p_disk, q_disk, data_disks) = raid6_layout(n, stripe_idx);
                let target_disk = data_disks[data_slot];
                let disk_lba = stripe_idx * self.chunk_blocks + chunk_off;

                if guard.members[target_disk].state == DiskState::Online
                    && guard.members[target_disk]
                        .dev
                        .read_blocks(disk_lba, out)
                        .is_ok()
                {
                    return Ok(());
                }

                let row = self.raid6_read_stripe_row(
                    &guard.members,
                    p_disk,
                    q_disk,
                    &data_disks,
                    disk_lba,
                    None,
                )?;
                out.copy_from_slice(&row[data_slot]);
                Ok(())
            }
        }
    }

    /// Write a single block at `lba` from `data` (length == `self.block_size`).
    fn write_single_block(
        &self,
        guard: &mut RaidState,
        lba: u64,
        data: &[u8],
    ) -> Result<(), RaidError> {
        let n = guard.members.len();
        match self.level {
            RaidLevel::Raid0 => {
                let chunk_idx = lba / self.chunk_blocks;
                let chunk_off = lba % self.chunk_blocks;
                let stripe_idx = chunk_idx / (n as u64);
                let disk_idx = (chunk_idx % (n as u64)) as usize;
                if guard.members[disk_idx].state != DiskState::Online {
                    return Err(RaidError::DiskFailed);
                }
                let disk_lba = stripe_idx * self.chunk_blocks + chunk_off;
                guard.members[disk_idx]
                    .dev
                    .write_blocks(disk_lba, data)
                    .map_err(|_| RaidError::IoError)
            }
            RaidLevel::Raid1 => {
                let mut wrote_any = false;
                for m in &guard.members {
                    if m.state == DiskState::Online && m.dev.write_blocks(lba, data).is_ok() {
                        wrote_any = true;
                    }
                }
                if wrote_any {
                    Ok(())
                } else {
                    Err(RaidError::TooManyFailedDisks)
                }
            }
            RaidLevel::Raid10 => {
                let pairs = n / 2;
                let chunk_idx = lba / self.chunk_blocks;
                let chunk_off = lba % self.chunk_blocks;
                let stripe_idx = chunk_idx / (pairs as u64);
                let pair_idx = (chunk_idx % (pairs as u64)) as usize;
                let disk_lba = stripe_idx * self.chunk_blocks + chunk_off;
                let d0 = pair_idx * 2;
                let d1 = d0 + 1;
                let mut wrote_any = false;
                if guard.members[d0].state == DiskState::Online
                    && guard.members[d0].dev.write_blocks(disk_lba, data).is_ok()
                {
                    wrote_any = true;
                }
                if guard.members[d1].state == DiskState::Online
                    && guard.members[d1].dev.write_blocks(disk_lba, data).is_ok()
                {
                    wrote_any = true;
                }
                if wrote_any {
                    Ok(())
                } else {
                    Err(RaidError::TooManyFailedDisks)
                }
            }
            RaidLevel::Raid5 => {
                let data_disks_per_stripe = n - 1;
                let chunk_idx = lba / self.chunk_blocks;
                let chunk_off = lba % self.chunk_blocks;
                let stripe_idx = chunk_idx / (data_disks_per_stripe as u64);
                let data_slot = (chunk_idx % (data_disks_per_stripe as u64)) as usize;
                let (parity_disk, data_disks) = raid5_layout(n, stripe_idx);
                let target_disk = data_disks[data_slot];
                let disk_lba = stripe_idx * self.chunk_blocks + chunk_off;

                let failed_count = guard
                    .members
                    .iter()
                    .filter(|m| m.state != DiskState::Online)
                    .count();
                if failed_count > 1 {
                    return Err(RaidError::TooManyFailedDisks);
                }

                // Reconstruct-write: read all other data disks, compute new parity.
                let mut parity = data.to_vec();
                let mut tmp = vec![0u8; self.block_size];
                for &d_idx in &data_disks {
                    if d_idx == target_disk {
                        continue;
                    }
                    if guard.members[d_idx].state == DiskState::Online {
                        guard.members[d_idx]
                            .dev
                            .read_blocks(disk_lba, &mut tmp)
                            .map_err(|_| RaidError::IoError)?;
                    } else {
                        // Reconstruct missing sibling data disk from old target + old parity + others.
                        self.raid5_reconstruct_disk_block(
                            &guard.members,
                            d_idx,
                            disk_lba,
                            &mut tmp,
                        )?;
                    }
                    xor_into(&mut parity, &tmp);
                }

                if guard.members[target_disk].state == DiskState::Online {
                    guard.members[target_disk]
                        .dev
                        .write_blocks(disk_lba, data)
                        .map_err(|_| RaidError::IoError)?;
                }
                if guard.members[parity_disk].state == DiskState::Online {
                    guard.members[parity_disk]
                        .dev
                        .write_blocks(disk_lba, &parity)
                        .map_err(|_| RaidError::IoError)?;
                }
                Ok(())
            }
            RaidLevel::Raid6 => {
                let data_disks_per_stripe = n - 2;
                let chunk_idx = lba / self.chunk_blocks;
                let chunk_off = lba % self.chunk_blocks;
                let stripe_idx = chunk_idx / (data_disks_per_stripe as u64);
                let data_slot = (chunk_idx % (data_disks_per_stripe as u64)) as usize;
                let (p_disk, q_disk, data_disks) = raid6_layout(n, stripe_idx);
                let target_disk = data_disks[data_slot];
                let disk_lba = stripe_idx * self.chunk_blocks + chunk_off;

                let mut row = self.raid6_read_stripe_row(
                    &guard.members,
                    p_disk,
                    q_disk,
                    &data_disks,
                    disk_lba,
                    None,
                )?;
                row[data_slot].copy_from_slice(data);

                let mut p_buf = vec![0u8; self.block_size];
                let mut q_buf = vec![0u8; self.block_size];
                for (k, d) in row.iter().enumerate() {
                    xor_into(&mut p_buf, d);
                    gf_mul_xor_into(&mut q_buf, d, gf_pow2(k));
                }

                if guard.members[target_disk].state == DiskState::Online {
                    guard.members[target_disk]
                        .dev
                        .write_blocks(disk_lba, data)
                        .map_err(|_| RaidError::IoError)?;
                }
                if guard.members[p_disk].state == DiskState::Online {
                    guard.members[p_disk]
                        .dev
                        .write_blocks(disk_lba, &p_buf)
                        .map_err(|_| RaidError::IoError)?;
                }
                if guard.members[q_disk].state == DiskState::Online {
                    guard.members[q_disk]
                        .dev
                        .write_blocks(disk_lba, &q_buf)
                        .map_err(|_| RaidError::IoError)?;
                }
                Ok(())
            }
        }
    }

    fn raid5_reconstruct_disk_block(
        &self,
        members: &[RaidMember],
        missing_idx: usize,
        disk_lba: u64,
        out: &mut [u8],
    ) -> Result<(), RaidError> {
        out.fill(0);
        let mut tmp = vec![0u8; self.block_size];
        for (i, m) in members.iter().enumerate() {
            if i == missing_idx {
                continue;
            }
            if m.state != DiskState::Online {
                return Err(RaidError::TooManyFailedDisks);
            }
            m.dev
                .read_blocks(disk_lba, &mut tmp)
                .map_err(|_| RaidError::IoError)?;
            xor_into(out, &tmp);
        }
        Ok(())
    }

    /// Read and (if needed) reconstruct all data blocks in a RAID 6 stripe row.
    fn raid6_read_stripe_row(
        &self,
        members: &[RaidMember],
        p_disk: usize,
        q_disk: usize,
        data_disks: &[usize],
        disk_lba: u64,
        force_missing: Option<usize>,
    ) -> Result<Vec<Vec<u8>>, RaidError> {
        let k = data_disks.len();
        let mut data_rows: Vec<Option<Vec<u8>>> = vec![None; k];
        let mut missing_data_slots: Vec<usize> = Vec::new();

        for (slot, &d_idx) in data_disks.iter().enumerate() {
            let is_forced = force_missing == Some(d_idx);
            if !is_forced && members[d_idx].state == DiskState::Online {
                let mut buf = vec![0u8; self.block_size];
                if members[d_idx].dev.read_blocks(disk_lba, &mut buf).is_ok() {
                    data_rows[slot] = Some(buf);
                    continue;
                }
            }
            missing_data_slots.push(slot);
        }

        if missing_data_slots.is_empty() {
            return Ok(data_rows.into_iter().map(|r| r.unwrap()).collect());
        }

        let p_ok = force_missing != Some(p_disk) && members[p_disk].state == DiskState::Online;
        let q_ok = force_missing != Some(q_disk) && members[q_disk].state == DiskState::Online;

        let mut p_buf = vec![0u8; self.block_size];
        let p_avail = p_ok
            && members[p_disk]
                .dev
                .read_blocks(disk_lba, &mut p_buf)
                .is_ok();

        let mut q_buf = vec![0u8; self.block_size];
        let q_avail = q_ok
            && members[q_disk]
                .dev
                .read_blocks(disk_lba, &mut q_buf)
                .is_ok();

        match missing_data_slots.len() {
            1 => {
                let x = missing_data_slots[0];
                if p_avail {
                    let mut rec = p_buf;
                    for (slot, row) in data_rows.iter().enumerate() {
                        if slot != x {
                            xor_into(&mut rec, row.as_ref().unwrap());
                        }
                    }
                    data_rows[x] = Some(rec);
                } else if q_avail {
                    let mut q_partial = q_buf;
                    for (slot, row) in data_rows.iter().enumerate() {
                        if slot != x {
                            gf_mul_xor_into(&mut q_partial, row.as_ref().unwrap(), gf_pow2(slot));
                        }
                    }
                    let inv_gx = gf_inv(gf_pow2(x));
                    let mut rec = vec![0u8; self.block_size];
                    gf_mul_xor_into(&mut rec, &q_partial, inv_gx);
                    data_rows[x] = Some(rec);
                } else {
                    return Err(RaidError::TooManyFailedDisks);
                }
            }
            2 => {
                if !p_avail || !q_avail {
                    return Err(RaidError::TooManyFailedDisks);
                }
                let x = missing_data_slots[0];
                let y = missing_data_slots[1];

                let mut p_xy = p_buf;
                let mut q_xy = q_buf;
                for (slot, row) in data_rows.iter().enumerate() {
                    if slot != x && slot != y {
                        let d = row.as_ref().unwrap();
                        xor_into(&mut p_xy, d);
                        gf_mul_xor_into(&mut q_xy, d, gf_pow2(slot));
                    }
                }

                let gx = gf_pow2(x);
                let gy = gf_pow2(y);
                let inv_diff = gf_inv(gx ^ gy);

                // D_y = inv(g^x ^ g^y) * (g^x * P_xy ^ Q_xy)
                let mut rhs = q_xy;
                gf_mul_xor_into(&mut rhs, &p_xy, gx);
                let mut d_y = vec![0u8; self.block_size];
                gf_mul_xor_into(&mut d_y, &rhs, inv_diff);

                // D_x = P_xy ^ D_y
                let mut d_x = p_xy;
                xor_into(&mut d_x, &d_y);

                data_rows[x] = Some(d_x);
                data_rows[y] = Some(d_y);
            }
            _ => return Err(RaidError::TooManyFailedDisks),
        }

        Ok(data_rows.into_iter().map(|r| r.unwrap()).collect())
    }
}

fn raid5_layout(n: usize, stripe_idx: u64) -> (usize, Vec<usize>) {
    let parity_disk = (n - 1) - ((stripe_idx as usize) % n);
    let mut data_disks = Vec::with_capacity(n - 1);
    for offset in 1..n {
        data_disks.push((parity_disk + offset) % n);
    }
    (parity_disk, data_disks)
}

fn raid6_layout(n: usize, stripe_idx: u64) -> (usize, usize, Vec<usize>) {
    let p_disk = (n - 1) - ((stripe_idx as usize) % n);
    let q_disk = (p_disk + 1) % n;
    let mut data_disks = Vec::with_capacity(n - 2);
    for offset in 2..n {
        data_disks.push((p_disk + offset) % n);
    }
    (p_disk, q_disk, data_disks)
}

impl BlockDevice for RaidArray {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn block_count(&self) -> u64 {
        self.total_blocks
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(self.block_size) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / self.block_size) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.total_blocks {
            return Err(RaidError::OutOfBounds.as_str());
        }
        let mut guard = self.state.lock();
        for (i, chunk) in buf.chunks_mut(self.block_size).enumerate() {
            self.read_single_block(&mut guard, lba + (i as u64), chunk)
                .map_err(Into::<&'static str>::into)?;
        }
        Ok(())
    }

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(self.block_size) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / self.block_size) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.total_blocks {
            return Err(RaidError::OutOfBounds.as_str());
        }
        let mut guard = self.state.lock();
        for (i, chunk) in buf.chunks(self.block_size).enumerate() {
            self.write_single_block(&mut guard, lba + (i as u64), chunk)
                .map_err(Into::<&'static str>::into)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LVM / Device-Mapper Volume Manager
// ---------------------------------------------------------------------------

/// Physical Extent address `(pv_index, pe_index)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalExtent {
    pub pv_idx: usize,
    pub pe_idx: u64,
}

/// Physical Volume (PV) wrapping a `BlockDevice` divided into fixed-size extents.
pub struct PhysicalVolume {
    pub name: String,
    pub dev: Arc<dyn BlockDevice>,
    pub extent_size_blocks: u64,
    pub total_extents: u64,
    allocated: Vec<bool>,
}

impl PhysicalVolume {
    pub fn new(
        name: impl Into<String>,
        dev: Arc<dyn BlockDevice>,
        extent_size_blocks: u64,
    ) -> Result<Self, RaidError> {
        if extent_size_blocks == 0 || dev.block_size() == 0 {
            return Err(RaidError::InvalidConfig);
        }
        let total_extents = dev.block_count() / extent_size_blocks;
        if total_extents == 0 {
            return Err(RaidError::InvalidConfig);
        }
        Ok(Self {
            name: name.into(),
            dev,
            extent_size_blocks,
            total_extents,
            allocated: vec![false; total_extents as usize],
        })
    }

    pub fn free_extents(&self) -> u64 {
        self.allocated.iter().filter(|&&used| !used).count() as u64
    }

    fn alloc_extent(&mut self) -> Option<u64> {
        for (idx, used) in self.allocated.iter_mut().enumerate() {
            if !*used {
                *used = true;
                return Some(idx as u64);
            }
        }
        None
    }

    fn free_extent(&mut self, pe_idx: u64) {
        if let Some(slot) = self.allocated.get_mut(pe_idx as usize) {
            *slot = false;
        }
    }
}

/// Device-Mapper target mapping for a Logical Volume.
#[derive(Debug, Clone)]
pub enum DmTarget {
    /// Linear mapping of logical extents to physical extents.
    Linear { extents: Vec<PhysicalExtent> },
    /// Striped mapping across `stripe_count` legs with `stripe_blocks` chunk size.
    Striped {
        stripe_count: usize,
        stripe_blocks: u64,
        legs: Vec<Vec<PhysicalExtent>>,
    },
    /// Mirrored mapping across multiple legs.
    Mirror { legs: Vec<Vec<PhysicalExtent>> },
}

/// Logical Volume (LV) backed by a `VolumeGroup`'s Physical Volumes.
pub struct LogicalVolume {
    pub name: String,
    block_size: usize,
    extent_size_blocks: u64,
    total_blocks: u64,
    pvs: Vec<Arc<dyn BlockDevice>>,
    target: DmTarget,
}

impl LogicalVolume {
    fn read_block_at(&self, lba: u64, out: &mut [u8]) -> Result<(), RaidError> {
        match &self.target {
            DmTarget::Linear { extents } => {
                let le = (lba / self.extent_size_blocks) as usize;
                let off = lba % self.extent_size_blocks;
                let pe = extents.get(le).ok_or(RaidError::OutOfBounds)?;
                let phys_lba = pe.pe_idx * self.extent_size_blocks + off;
                self.pvs[pe.pv_idx]
                    .read_blocks(phys_lba, out)
                    .map_err(|_| RaidError::IoError)
            }
            DmTarget::Striped {
                stripe_count,
                stripe_blocks,
                legs,
            } => {
                let chunk_idx = lba / *stripe_blocks;
                let chunk_off = lba % *stripe_blocks;
                let leg_idx = (chunk_idx % (*stripe_count as u64)) as usize;
                let leg_lba = (chunk_idx / (*stripe_count as u64)) * *stripe_blocks + chunk_off;
                let le = (leg_lba / self.extent_size_blocks) as usize;
                let off = leg_lba % self.extent_size_blocks;
                let pe = legs[leg_idx].get(le).ok_or(RaidError::OutOfBounds)?;
                let phys_lba = pe.pe_idx * self.extent_size_blocks + off;
                self.pvs[pe.pv_idx]
                    .read_blocks(phys_lba, out)
                    .map_err(|_| RaidError::IoError)
            }
            DmTarget::Mirror { legs } => {
                let le = (lba / self.extent_size_blocks) as usize;
                let off = lba % self.extent_size_blocks;
                for leg in legs {
                    if let Some(pe) = leg.get(le) {
                        let phys_lba = pe.pe_idx * self.extent_size_blocks + off;
                        if self.pvs[pe.pv_idx].read_blocks(phys_lba, out).is_ok() {
                            return Ok(());
                        }
                    }
                }
                Err(RaidError::IoError)
            }
        }
    }

    fn write_block_at(&self, lba: u64, data: &[u8]) -> Result<(), RaidError> {
        match &self.target {
            DmTarget::Linear { extents } => {
                let le = (lba / self.extent_size_blocks) as usize;
                let off = lba % self.extent_size_blocks;
                let pe = extents.get(le).ok_or(RaidError::OutOfBounds)?;
                let phys_lba = pe.pe_idx * self.extent_size_blocks + off;
                self.pvs[pe.pv_idx]
                    .write_blocks(phys_lba, data)
                    .map_err(|_| RaidError::IoError)
            }
            DmTarget::Striped {
                stripe_count,
                stripe_blocks,
                legs,
            } => {
                let chunk_idx = lba / *stripe_blocks;
                let chunk_off = lba % *stripe_blocks;
                let leg_idx = (chunk_idx % (*stripe_count as u64)) as usize;
                let leg_lba = (chunk_idx / (*stripe_count as u64)) * *stripe_blocks + chunk_off;
                let le = (leg_lba / self.extent_size_blocks) as usize;
                let off = leg_lba % self.extent_size_blocks;
                let pe = legs[leg_idx].get(le).ok_or(RaidError::OutOfBounds)?;
                let phys_lba = pe.pe_idx * self.extent_size_blocks + off;
                self.pvs[pe.pv_idx]
                    .write_blocks(phys_lba, data)
                    .map_err(|_| RaidError::IoError)
            }
            DmTarget::Mirror { legs } => {
                let le = (lba / self.extent_size_blocks) as usize;
                let off = lba % self.extent_size_blocks;
                let mut wrote_any = false;
                for leg in legs {
                    if let Some(pe) = leg.get(le) {
                        let phys_lba = pe.pe_idx * self.extent_size_blocks + off;
                        if self.pvs[pe.pv_idx].write_blocks(phys_lba, data).is_ok() {
                            wrote_any = true;
                        }
                    }
                }
                if wrote_any {
                    Ok(())
                } else {
                    Err(RaidError::IoError)
                }
            }
        }
    }
}

impl BlockDevice for LogicalVolume {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn block_count(&self) -> u64 {
        self.total_blocks
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(self.block_size) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / self.block_size) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.total_blocks {
            return Err(RaidError::OutOfBounds.as_str());
        }
        for (i, chunk) in buf.chunks_mut(self.block_size).enumerate() {
            self.read_block_at(lba + (i as u64), chunk)
                .map_err(Into::<&'static str>::into)?;
        }
        Ok(())
    }

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(self.block_size) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / self.block_size) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.total_blocks {
            return Err(RaidError::OutOfBounds.as_str());
        }
        for (i, chunk) in buf.chunks(self.block_size).enumerate() {
            self.write_block_at(lba + (i as u64), chunk)
                .map_err(Into::<&'static str>::into)?;
        }
        Ok(())
    }
}

/// LVM Volume Group (VG) managing Physical Volumes and allocating Logical Volumes.
pub struct VolumeGroup {
    pub name: String,
    block_size: usize,
    extent_size_blocks: u64,
    pvs: Vec<PhysicalVolume>,
}

impl VolumeGroup {
    pub fn new(
        name: impl Into<String>,
        block_size: usize,
        extent_size_blocks: u64,
    ) -> Result<Self, RaidError> {
        if block_size == 0 || extent_size_blocks == 0 {
            return Err(RaidError::InvalidConfig);
        }
        Ok(Self {
            name: name.into(),
            block_size,
            extent_size_blocks,
            pvs: Vec::new(),
        })
    }

    pub fn add_pv(
        &mut self,
        name: impl Into<String>,
        dev: Arc<dyn BlockDevice>,
    ) -> Result<(), RaidError> {
        if dev.block_size() != self.block_size {
            return Err(RaidError::BlockSizeMismatch);
        }
        let pv = PhysicalVolume::new(name, dev, self.extent_size_blocks)?;
        self.pvs.push(pv);
        Ok(())
    }

    pub fn total_free_extents(&self) -> u64 {
        self.pvs.iter().map(PhysicalVolume::free_extents).sum()
    }

    fn pv_devices(&self) -> Vec<Arc<dyn BlockDevice>> {
        self.pvs.iter().map(|pv| Arc::clone(&pv.dev)).collect()
    }

    /// Create a Linear Logical Volume of `extent_count` extents.
    pub fn create_linear_lv(
        &mut self,
        name: impl Into<String>,
        extent_count: u64,
    ) -> Result<LogicalVolume, RaidError> {
        if extent_count == 0 || self.total_free_extents() < extent_count {
            return Err(RaidError::NoSpace);
        }
        let mut extents = Vec::with_capacity(extent_count as usize);
        for pv_idx in 0..self.pvs.len() {
            while (extents.len() as u64) < extent_count {
                if let Some(pe_idx) = self.pvs[pv_idx].alloc_extent() {
                    extents.push(PhysicalExtent { pv_idx, pe_idx });
                } else {
                    break;
                }
            }
        }
        Ok(LogicalVolume {
            name: name.into(),
            block_size: self.block_size,
            extent_size_blocks: self.extent_size_blocks,
            total_blocks: extent_count * self.extent_size_blocks,
            pvs: self.pv_devices(),
            target: DmTarget::Linear { extents },
        })
    }

    /// Create a Striped Logical Volume across `stripe_count` PVs.
    pub fn create_striped_lv(
        &mut self,
        name: impl Into<String>,
        extents_per_leg: u64,
        stripe_count: usize,
        stripe_blocks: u64,
    ) -> Result<LogicalVolume, RaidError> {
        if stripe_count == 0
            || stripe_count > self.pvs.len()
            || extents_per_leg == 0
            || stripe_blocks == 0
            || !self.extent_size_blocks.is_multiple_of(stripe_blocks)
        {
            return Err(RaidError::InvalidConfig);
        }
        for pv in self.pvs.iter().take(stripe_count) {
            if pv.free_extents() < extents_per_leg {
                return Err(RaidError::NoSpace);
            }
        }
        let mut legs = Vec::with_capacity(stripe_count);
        for (pv_idx, pv) in self.pvs.iter_mut().enumerate().take(stripe_count) {
            let mut leg = Vec::with_capacity(extents_per_leg as usize);
            for _ in 0..extents_per_leg {
                let pe_idx = pv.alloc_extent().ok_or(RaidError::NoSpace)?;
                leg.push(PhysicalExtent { pv_idx, pe_idx });
            }
            legs.push(leg);
        }
        Ok(LogicalVolume {
            name: name.into(),
            block_size: self.block_size,
            extent_size_blocks: self.extent_size_blocks,
            total_blocks: extents_per_leg * (stripe_count as u64) * self.extent_size_blocks,
            pvs: self.pv_devices(),
            target: DmTarget::Striped {
                stripe_count,
                stripe_blocks,
                legs,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Thin-Provisioning Extent Mapper (ThinPool + ThinVolume)
// ---------------------------------------------------------------------------

/// Statistics for a `ThinPool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinPoolStats {
    pub total_chunks: u64,
    pub allocated_chunks: u64,
    pub free_chunks: u64,
    pub chunk_blocks: u64,
}

struct ThinPoolInner {
    free_bitmap: Vec<bool>,
    allocated_count: u64,
}

/// Thin-provisioning storage pool managing physical chunks on a backing `BlockDevice`.
pub struct ThinPool {
    backing: Arc<dyn BlockDevice>,
    block_size: usize,
    chunk_blocks: u64,
    total_chunks: u64,
    inner: SpinLock<ThinPoolInner>,
}

impl ThinPool {
    pub fn new(backing: Arc<dyn BlockDevice>, chunk_blocks: u64) -> Result<Self, RaidError> {
        let block_size = backing.block_size();
        if block_size == 0 || chunk_blocks == 0 {
            return Err(RaidError::InvalidConfig);
        }
        let total_chunks = backing.block_count() / chunk_blocks;
        if total_chunks == 0 {
            return Err(RaidError::InvalidConfig);
        }
        Ok(Self {
            backing,
            block_size,
            chunk_blocks,
            total_chunks,
            inner: SpinLock::new(ThinPoolInner {
                free_bitmap: vec![true; total_chunks as usize],
                allocated_count: 0,
            }),
        })
    }

    pub fn stats(&self) -> ThinPoolStats {
        let guard = self.inner.lock();
        ThinPoolStats {
            total_chunks: self.total_chunks,
            allocated_chunks: guard.allocated_count,
            free_chunks: self.total_chunks - guard.allocated_count,
            chunk_blocks: self.chunk_blocks,
        }
    }

    fn alloc_chunk(&self) -> Result<u64, RaidError> {
        let mut guard = self.inner.lock();
        for (idx, is_free) in guard.free_bitmap.iter_mut().enumerate() {
            if *is_free {
                *is_free = false;
                guard.allocated_count += 1;
                return Ok(idx as u64);
            }
        }
        Err(RaidError::NoSpace)
    }

    fn free_chunk(&self, phys_chunk: u64) {
        let mut guard = self.inner.lock();
        if let Some(slot) = guard.free_bitmap.get_mut(phys_chunk as usize) {
            if !*slot {
                *slot = true;
                guard.allocated_count = guard.allocated_count.saturating_sub(1);
            }
        }
    }
}

/// Thinly-provisioned Logical Volume backed by a `ThinPool`.
pub struct ThinVolume {
    pool: Arc<ThinPool>,
    virtual_blocks: u64,
    /// Maps `virtual_chunk -> physical_chunk`.
    extent_map: SpinLock<BTreeMap<u64, u64>>,
}

impl ThinVolume {
    pub fn new(pool: Arc<ThinPool>, virtual_blocks: u64) -> Result<Self, RaidError> {
        if virtual_blocks == 0 {
            return Err(RaidError::InvalidConfig);
        }
        Ok(Self {
            pool,
            virtual_blocks,
            extent_map: SpinLock::new(BTreeMap::new()),
        })
    }

    pub fn mapped_chunks(&self) -> usize {
        self.extent_map.lock().len()
    }

    /// Discard (TRIM/UNMAP) a range of blocks, freeing any fully covered chunks.
    pub fn discard(&self, lba: u64, block_count: u64) -> Result<(), RaidError> {
        let end_lba = lba.checked_add(block_count).ok_or(RaidError::OutOfBounds)?;
        if end_lba > self.virtual_blocks {
            return Err(RaidError::OutOfBounds);
        }
        let cb = self.pool.chunk_blocks;
        let first_full = lba.div_ceil(cb);
        let last_full = end_lba / cb;
        if first_full >= last_full {
            return Ok(());
        }
        let mut map = self.extent_map.lock();
        for vchunk in first_full..last_full {
            if let Some(pchunk) = map.remove(&vchunk) {
                self.pool.free_chunk(pchunk);
            }
        }
        Ok(())
    }
}

impl BlockDevice for ThinVolume {
    fn block_size(&self) -> usize {
        self.pool.block_size
    }

    fn block_count(&self) -> u64 {
        self.virtual_blocks
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        let bs = self.pool.block_size;
        if !buf.len().is_multiple_of(bs) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / bs) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.virtual_blocks {
            return Err(RaidError::OutOfBounds.as_str());
        }

        let cb = self.pool.chunk_blocks;
        let map = self.extent_map.lock();
        for (i, block_buf) in buf.chunks_mut(bs).enumerate() {
            let cur_lba = lba + (i as u64);
            let vchunk = cur_lba / cb;
            let off = cur_lba % cb;
            if let Some(&pchunk) = map.get(&vchunk) {
                let phys_lba = pchunk * cb + off;
                self.pool.backing.read_blocks(phys_lba, block_buf)?;
            } else {
                block_buf.fill(0);
            }
        }
        Ok(())
    }

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        let bs = self.pool.block_size;
        if !buf.len().is_multiple_of(bs) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / bs) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.virtual_blocks {
            return Err(RaidError::OutOfBounds.as_str());
        }

        let cb = self.pool.chunk_blocks;
        let mut map = self.extent_map.lock();
        let zero_block = vec![0u8; bs];

        for (i, block_buf) in buf.chunks(bs).enumerate() {
            let cur_lba = lba + (i as u64);
            let vchunk = cur_lba / cb;
            let off = cur_lba % cb;
            let pchunk = if let Some(&existing) = map.get(&vchunk) {
                existing
            } else {
                let new_chunk = self
                    .pool
                    .alloc_chunk()
                    .map_err(Into::<&'static str>::into)?;
                // Zero-initialize newly allocated physical chunk before partial write.
                let base = new_chunk * cb;
                for z in 0..cb {
                    self.pool.backing.write_blocks(base + z, &zero_block)?;
                }
                map.insert(vchunk, new_chunk);
                new_chunk
            };
            let phys_lba = pchunk * cb + off;
            self.pool.backing.write_blocks(phys_lba, block_buf)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Snapshot Copy-on-Write (COW) Target
// ---------------------------------------------------------------------------

struct SnapshotInner {
    /// Maps `origin_chunk -> cow_chunk`.
    exceptions: BTreeMap<u64, u64>,
    next_cow_chunk: u64,
}

/// Device-Mapper Snapshot COW target.
pub struct SnapshotVolume {
    origin: Arc<dyn BlockDevice>,
    cow_store: Arc<dyn BlockDevice>,
    block_size: usize,
    chunk_blocks: u64,
    max_cow_chunks: u64,
    inner: SpinLock<SnapshotInner>,
}

impl SnapshotVolume {
    pub fn new(
        origin: Arc<dyn BlockDevice>,
        cow_store: Arc<dyn BlockDevice>,
        chunk_blocks: u64,
    ) -> Result<Self, RaidError> {
        let block_size = origin.block_size();
        if block_size == 0 || chunk_blocks == 0 {
            return Err(RaidError::InvalidConfig);
        }
        if cow_store.block_size() != block_size {
            return Err(RaidError::BlockSizeMismatch);
        }
        let max_cow_chunks = cow_store.block_count() / chunk_blocks;
        if max_cow_chunks == 0 {
            return Err(RaidError::InvalidConfig);
        }
        Ok(Self {
            origin,
            cow_store,
            block_size,
            chunk_blocks,
            max_cow_chunks,
            inner: SpinLock::new(SnapshotInner {
                exceptions: BTreeMap::new(),
                next_cow_chunk: 0,
            }),
        })
    }

    pub fn exception_count(&self) -> usize {
        self.inner.lock().exceptions.len()
    }

    /// Ensure `origin_chunk` is preserved in the COW store before the origin is modified.
    pub fn cow_preserve_origin_chunk(&self, origin_chunk: u64) -> Result<(), RaidError> {
        let mut guard = self.inner.lock();
        if guard.exceptions.contains_key(&origin_chunk) {
            return Ok(());
        }
        if guard.next_cow_chunk >= self.max_cow_chunks {
            return Err(RaidError::NoSpace);
        }
        let cow_chunk = guard.next_cow_chunk;
        guard.next_cow_chunk += 1;

        let mut buf = vec![0u8; self.block_size];
        let orig_base = origin_chunk * self.chunk_blocks;
        let cow_base = cow_chunk * self.chunk_blocks;
        for off in 0..self.chunk_blocks {
            let orig_lba = orig_base + off;
            if orig_lba < self.origin.block_count() {
                self.origin
                    .read_blocks(orig_lba, &mut buf)
                    .map_err(|_| RaidError::IoError)?;
            } else {
                buf.fill(0);
            }
            self.cow_store
                .write_blocks(cow_base + off, &buf)
                .map_err(|_| RaidError::IoError)?;
        }
        guard.exceptions.insert(origin_chunk, cow_chunk);
        Ok(())
    }
}

impl BlockDevice for SnapshotVolume {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn block_count(&self) -> u64 {
        self.origin.block_count()
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(self.block_size) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / self.block_size) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.origin.block_count() {
            return Err(RaidError::OutOfBounds.as_str());
        }

        let guard = self.inner.lock();
        for (i, block_buf) in buf.chunks_mut(self.block_size).enumerate() {
            let cur_lba = lba + (i as u64);
            let chunk = cur_lba / self.chunk_blocks;
            let off = cur_lba % self.chunk_blocks;
            if let Some(&cow_chunk) = guard.exceptions.get(&chunk) {
                let cow_lba = cow_chunk * self.chunk_blocks + off;
                self.cow_store.read_blocks(cow_lba, block_buf)?;
            } else {
                self.origin.read_blocks(cur_lba, block_buf)?;
            }
        }
        Ok(())
    }

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(self.block_size) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / self.block_size) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.origin.block_count() {
            return Err(RaidError::OutOfBounds.as_str());
        }

        for (i, block_buf) in buf.chunks(self.block_size).enumerate() {
            let cur_lba = lba + (i as u64);
            let chunk = cur_lba / self.chunk_blocks;
            let off = cur_lba % self.chunk_blocks;
            self.cow_preserve_origin_chunk(chunk)
                .map_err(Into::<&'static str>::into)?;
            let cow_chunk = {
                let guard = self.inner.lock();
                *guard
                    .exceptions
                    .get(&chunk)
                    .ok_or(RaidError::NotFound.as_str())?
            };
            let cow_lba = cow_chunk * self.chunk_blocks + off;
            self.cow_store.write_blocks(cow_lba, block_buf)?;
        }
        Ok(())
    }
}

/// Origin device wrapper that triggers COW preservation on attached `SnapshotVolume`s
/// before overwriting origin blocks.
pub struct OriginWithSnapshots {
    origin: Arc<dyn BlockDevice>,
    snapshots: SpinLock<Vec<Arc<SnapshotVolume>>>,
}

impl OriginWithSnapshots {
    pub fn new(origin: Arc<dyn BlockDevice>) -> Self {
        Self {
            origin,
            snapshots: SpinLock::new(Vec::new()),
        }
    }

    pub fn attach_snapshot(&self, snap: Arc<SnapshotVolume>) {
        self.snapshots.lock().push(snap);
    }
}

impl BlockDevice for OriginWithSnapshots {
    fn block_size(&self) -> usize {
        self.origin.block_size()
    }

    fn block_count(&self) -> u64 {
        self.origin.block_count()
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        self.origin.read_blocks(lba, buf)
    }

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        let bs = self.origin.block_size();
        if !buf.len().is_multiple_of(bs) {
            return Err(RaidError::UnalignedBuffer.as_str());
        }
        let blocks = (buf.len() / bs) as u64;
        let end_lba = lba
            .checked_add(blocks)
            .ok_or(RaidError::OutOfBounds.as_str())?;
        if end_lba > self.origin.block_count() {
            return Err(RaidError::OutOfBounds.as_str());
        }

        let snaps = self.snapshots.lock();
        for snap in snaps.iter() {
            let first_chunk = lba / snap.chunk_blocks;
            let last_chunk = (end_lba.saturating_sub(1)) / snap.chunk_blocks;
            for chunk in first_chunk..=last_chunk {
                snap.cow_preserve_origin_chunk(chunk)
                    .map_err(Into::<&'static str>::into)?;
            }
        }
        drop(snaps);

        self.origin.write_blocks(lba, buf)
    }
}

// ---------------------------------------------------------------------------
// Self-Test Suite
// ---------------------------------------------------------------------------

fn make_pattern(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_add((i as u8).wrapping_mul(31)))
        .collect()
}

fn make_disks(count: usize, blocks: u64) -> Result<Vec<Arc<dyn BlockDevice>>, &'static str> {
    let mut out: Vec<Arc<dyn BlockDevice>> = Vec::with_capacity(count);
    for _ in 0..count {
        let dev = MemBlockDevice::new(blocks, SECTOR_SIZE).map_err(Into::<&'static str>::into)?;
        out.push(Arc::new(dev));
    }
    Ok(out)
}

/// Run comprehensive in-memory self-tests for RAID 0/1/5/6/10, LVM, Thin-Provisioning, and Snapshot COW.
pub fn self_test() -> Result<(), &'static str> {
    // 1. RAID 0 (3 disks, chunk=4 blocks)
    {
        let disks = make_disks(3, 32)?;
        let raid0 = RaidArray::new(RaidLevel::Raid0, disks, 4)?;
        let wbuf = make_pattern(0x11, 12 * SECTOR_SIZE);
        raid0.write_blocks(0, &wbuf)?;
        let mut rbuf = vec![0u8; wbuf.len()];
        raid0.read_blocks(0, &mut rbuf)?;
        if rbuf != wbuf {
            return Err("raid0: readback mismatch");
        }
    }

    // 2. RAID 1 (2 disks, fail disk 0, degraded read, rebuild disk 0)
    {
        let disks = make_disks(2, 16)?;
        let raid1 = RaidArray::new(RaidLevel::Raid1, disks, 4)?;
        let wbuf = make_pattern(0x22, 8 * SECTOR_SIZE);
        raid1.write_blocks(2, &wbuf)?;
        raid1.mark_disk_failed(0)?;
        let mut rbuf = vec![0u8; wbuf.len()];
        raid1.read_blocks(2, &mut rbuf)?;
        if rbuf != wbuf {
            return Err("raid1: degraded read mismatch");
        }
        raid1.rebuild_disk(0)?;
        raid1.mark_disk_failed(1)?;
        rbuf.fill(0);
        raid1.read_blocks(2, &mut rbuf)?;
        if rbuf != wbuf {
            return Err("raid1: rebuilt disk read mismatch");
        }
    }

    // 3. RAID 5 (4 disks, chunk=2, fail disk 2, degraded read/write, rebuild)
    {
        let disks = make_disks(4, 16)?;
        let raid5 = RaidArray::new(RaidLevel::Raid5, disks, 2)?;
        let wbuf = make_pattern(0x33, 12 * SECTOR_SIZE);
        raid5.write_blocks(0, &wbuf)?;
        raid5.mark_disk_failed(2)?;
        let mut rbuf = vec![0u8; wbuf.len()];
        raid5.read_blocks(0, &mut rbuf)?;
        if rbuf != wbuf {
            return Err("raid5: degraded read mismatch");
        }
        let wbuf2 = make_pattern(0x44, 4 * SECTOR_SIZE);
        raid5.write_blocks(4, &wbuf2)?;
        raid5.rebuild_disk(2)?;
        let mut rbuf2 = vec![0u8; wbuf2.len()];
        raid5.read_blocks(4, &mut rbuf2)?;
        if rbuf2 != wbuf2 {
            return Err("raid5: post-rebuild read mismatch");
        }
    }

    // 4. RAID 6 (5 disks, chunk=2, dual-disk P+Q recovery & rebuild)
    {
        let disks = make_disks(5, 16)?;
        let raid6 = RaidArray::new(RaidLevel::Raid6, disks, 2)?;
        let wbuf = make_pattern(0x55, 12 * SECTOR_SIZE);
        raid6.write_blocks(0, &wbuf)?;

        // Fail 2 disks (0 and 1) -> dual-disk recovery
        raid6.mark_disk_failed(0)?;
        raid6.mark_disk_failed(1)?;
        let mut rbuf = vec![0u8; wbuf.len()];
        raid6.read_blocks(0, &mut rbuf)?;
        if rbuf != wbuf {
            return Err("raid6: dual-disk degraded read mismatch");
        }

        // Rebuild disk 0, then fail disk 4 (parity/data mix) and verify
        raid6.rebuild_disk(0)?;
        raid6.mark_disk_failed(4)?;
        rbuf.fill(0);
        raid6.read_blocks(0, &mut rbuf)?;
        if rbuf != wbuf {
            return Err("raid6: second dual-disk degraded read mismatch");
        }
        raid6.rebuild_disk(1)?;
        raid6.rebuild_disk(4)?;
    }

    // 5. RAID 10 (4 disks = 2 mirror pairs, fail 1 disk per pair)
    {
        let disks = make_disks(4, 16)?;
        let raid10 = RaidArray::new(RaidLevel::Raid10, disks, 2)?;
        let wbuf = make_pattern(0x66, 8 * SECTOR_SIZE);
        raid10.write_blocks(0, &wbuf)?;
        raid10.mark_disk_failed(0)?;
        raid10.mark_disk_failed(3)?;
        let mut rbuf = vec![0u8; wbuf.len()];
        raid10.read_blocks(0, &mut rbuf)?;
        if rbuf != wbuf {
            return Err("raid10: degraded read mismatch");
        }
        raid10.rebuild_disk(0)?;
        raid10.rebuild_disk(3)?;
    }

    // 6. LVM PV / VG / Linear & Striped LV
    {
        let pv0 = Arc::new(MemBlockDevice::new(32, SECTOR_SIZE)?);
        let pv1 = Arc::new(MemBlockDevice::new(32, SECTOR_SIZE)?);
        let mut vg = VolumeGroup::new("vg0", SECTOR_SIZE, 4)?;
        vg.add_pv("pv0", pv0)?;
        vg.add_pv("pv1", pv1)?;

        let linear = vg.create_linear_lv("lv_lin", 4)?;
        let wbuf = make_pattern(0x77, 6 * SECTOR_SIZE);
        linear.write_blocks(2, &wbuf)?;
        let mut rbuf = vec![0u8; wbuf.len()];
        linear.read_blocks(2, &mut rbuf)?;
        if rbuf != wbuf {
            return Err("lvm: linear LV readback mismatch");
        }

        let striped = vg.create_striped_lv("lv_str", 2, 2, 2)?;
        let wbuf2 = make_pattern(0x88, 8 * SECTOR_SIZE);
        striped.write_blocks(0, &wbuf2)?;
        let mut rbuf2 = vec![0u8; wbuf2.len()];
        striped.read_blocks(0, &mut rbuf2)?;
        if rbuf2 != wbuf2 {
            return Err("lvm: striped LV readback mismatch");
        }
    }

    // 7. Thin-Provisioning Pool & ThinVolume
    {
        let backing = Arc::new(MemBlockDevice::new(64, SECTOR_SIZE)?);
        let pool = Arc::new(ThinPool::new(backing, 4)?);
        let thin = ThinVolume::new(Arc::clone(&pool), 128)?;

        let mut rbuf = vec![0xAAu8; 4 * SECTOR_SIZE];
        thin.read_blocks(16, &mut rbuf)?;
        if rbuf.iter().any(|&b| b != 0) || pool.stats().allocated_chunks != 0 {
            return Err("thin: unmapped read should return zeros without allocating");
        }

        let wbuf = make_pattern(0x99, 4 * SECTOR_SIZE);
        thin.write_blocks(16, &wbuf)?;
        if pool.stats().allocated_chunks != 1 {
            return Err("thin: expected 1 allocated chunk");
        }
        thin.read_blocks(16, &mut rbuf)?;
        if rbuf != wbuf {
            return Err("thin: readback mismatch");
        }

        thin.discard(16, 4)?;
        if pool.stats().allocated_chunks != 0 {
            return Err("thin: discard did not free chunk");
        }
    }

    // 8. Snapshot COW Target
    {
        let raw_origin = Arc::new(MemBlockDevice::new(32, SECTOR_SIZE)?);
        let origin = Arc::new(OriginWithSnapshots::new(raw_origin));
        let cow_store = Arc::new(MemBlockDevice::new(32, SECTOR_SIZE)?);

        let orig_v1 = make_pattern(0xA1, 4 * SECTOR_SIZE);
        origin.write_blocks(0, &orig_v1)?;

        let snap = Arc::new(SnapshotVolume::new(
            Arc::clone(&origin) as Arc<dyn BlockDevice>,
            cow_store,
            4,
        )?);
        origin.attach_snapshot(Arc::clone(&snap));

        // Overwrite origin -> COW must preserve orig_v1 for snapshot
        let orig_v2 = make_pattern(0xB2, 4 * SECTOR_SIZE);
        origin.write_blocks(0, &orig_v2)?;

        let mut snap_read = vec![0u8; 4 * SECTOR_SIZE];
        snap.read_blocks(0, &mut snap_read)?;
        if snap_read != orig_v1 {
            return Err("snapshot: COW failed to preserve origin data");
        }

        // Write to snapshot -> origin must remain orig_v2
        let snap_v3 = make_pattern(0xC3, 4 * SECTOR_SIZE);
        snap.write_blocks(0, &snap_v3)?;
        let mut orig_read = vec![0u8; 4 * SECTOR_SIZE];
        origin.read_blocks(0, &mut orig_read)?;
        if orig_read != orig_v2 {
            return Err("snapshot: write leaked to origin");
        }
        snap.read_blocks(0, &mut snap_read)?;
        if snap_read != snap_v3 {
            return Err("snapshot: readback mismatch after snapshot write");
        }
    }

    Ok(())
}
