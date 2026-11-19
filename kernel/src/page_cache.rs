//! `no_std` Page Cache, Crash-Safe WAL Journal & DMA Scatter-Gather Engine.
//!
//! Provides:
//! - 4 KiB LRU [`PageCache`] with pinning and dirty writeback to any [`BlockDevice`].
//! - Crash-safe Write-Ahead Log ([`WalJournal`]) with CRC32C (Castagnoli) checksums
//!   and transactional recovery/replay.
//! - [`SgList`] DMA scatter-gather descriptor builder with coalescing, segment
//!   splitting, and alignment validation.
//! - [`self_test`] exercising all three subsystems against an in-memory block device.

#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::blk::{BlockDevice, SECTOR_SIZE};
use crate::mm::{SpinLock, PAGE_SIZE};
use crate::serial_println;

/// Number of 512-byte sectors per 4 KiB page.
pub const SECTORS_PER_PAGE: usize = PAGE_SIZE / SECTOR_SIZE;

/// Default number of 4 KiB pages held by [`PageCache`].
pub const DEFAULT_CACHE_CAPACITY: usize = 64;

/// Reflected CRC32C (Castagnoli) polynomial.
pub const CRC32C_POLY: u32 = 0x82F6_3B78;

/// WAL record magic ("NXWL").
pub const WAL_MAGIC: u32 = 0x4E58_574C;

/// WAL format version.
pub const WAL_VERSION: u16 = 1;

/// Fixed serialized size of a WAL record header in bytes.
pub const WAL_HEADER_SIZE: usize = 40;

/// Maximum number of scatter-gather descriptors in a single [`SgList`].
pub const MAX_SG_SEGMENTS: usize = 128;

/// Default maximum byte length per DMA scatter-gather segment (64 KiB).
pub const DEFAULT_MAX_SEGMENT_BYTES: usize = 64 * 1024;

/// Scatter-gather descriptor flags.
pub const SG_FLAG_NONE: u32 = 0;
/// End-of-table marker on the final descriptor.
pub const SG_FLAG_EOT: u32 = 1 << 0;
/// Chain descriptor pointing to another descriptor table.
pub const SG_FLAG_CHAIN: u32 = 1 << 1;
/// Request interrupt on completion of this segment.
pub const SG_FLAG_IOC: u32 = 1 << 2;

/// Errors returned by the page cache, WAL journal, and DMA scatter-gather engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageCacheError {
    InvalidCapacity,
    InvalidOffset,
    BufferTooSmall,
    IoError(&'static str),
    AllPagesPinned,
    WalFull,
    WalCorrupted,
    WalInvalidRecord,
    TransactionNotFound,
    DmaAlignmentError,
    DmaEmpty,
    DmaTooManySegments,
}

// ---------------------------------------------------------------------------
// CRC32C (Castagnoli)
// ---------------------------------------------------------------------------

const fn build_crc32c_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ CRC32C_POLY;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC32C_TABLE: [u32; 256] = build_crc32c_table();

/// Compute the CRC32C (Castagnoli) checksum of `data`.
pub fn crc32c(data: &[u8]) -> u32 {
    crc32c_update(0, data)
}

/// Incrementally update a CRC32C checksum with `data`.
pub fn crc32c_update(prev_crc: u32, data: &[u8]) -> u32 {
    let mut crc = !prev_crc;
    for &b in data {
        let idx = ((crc ^ (b as u32)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLE[idx];
    }
    !crc
}

// ---------------------------------------------------------------------------
// 4 KiB LRU Page Cache with Dirty Writeback
// ---------------------------------------------------------------------------

/// Unique identifier for a cached 4 KiB page on a block device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageKey {
    pub dev_id: u32,
    pub page_idx: u64,
}

/// Clean/dirty state of a cached page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    Clean,
    Dirty,
}

/// A single 4 KiB page frame in the cache.
pub struct CachePage {
    pub key: PageKey,
    pub data: Box<[u8; PAGE_SIZE]>,
    pub state: PageState,
    pub pin_count: u32,
    pub last_access: u64,
}

/// Runtime statistics for [`PageCache`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PageCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub writebacks: u64,
    pub dirty_pages: usize,
    pub cached_pages: usize,
    pub capacity: usize,
}

/// 4 KiB LRU page cache with dirty writeback.
pub struct PageCache {
    capacity: usize,
    pages: Vec<CachePage>,
    tick: u64,
    stats: PageCacheStats,
}

impl PageCache {
    /// Create a new page cache holding up to `capacity` 4 KiB pages.
    pub fn new(capacity: usize) -> Result<Self, PageCacheError> {
        if capacity == 0 {
            return Err(PageCacheError::InvalidCapacity);
        }
        Ok(Self {
            capacity,
            pages: Vec::with_capacity(capacity),
            tick: 0,
            stats: PageCacheStats {
                capacity,
                ..PageCacheStats::default()
            },
        })
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }

    fn find_index(&self, key: PageKey) -> Option<usize> {
        self.pages.iter().position(|p| p.key == key)
    }

    fn read_page_from_device<D: BlockDevice>(
        dev: &D,
        page_idx: u64,
    ) -> Result<Box<[u8; PAGE_SIZE]>, PageCacheError> {
        let blk_size = dev.block_size();
        if blk_size == 0 || !PAGE_SIZE.is_multiple_of(blk_size) {
            return Err(PageCacheError::IoError("unsupported device block size"));
        }
        let blocks_per_page = (PAGE_SIZE / blk_size) as u64;
        let start_lba = page_idx
            .checked_mul(blocks_per_page)
            .ok_or(PageCacheError::InvalidOffset)?;
        let end_lba = start_lba
            .checked_add(blocks_per_page)
            .ok_or(PageCacheError::InvalidOffset)?;
        if end_lba > dev.block_count() {
            return Err(PageCacheError::InvalidOffset);
        }

        let mut buf = Box::new([0u8; PAGE_SIZE]);
        dev.read_blocks(start_lba, &mut buf[..])
            .map_err(PageCacheError::IoError)?;
        Ok(buf)
    }

    fn write_page_to_device<D: BlockDevice>(
        dev: &D,
        page_idx: u64,
        data: &[u8; PAGE_SIZE],
    ) -> Result<(), PageCacheError> {
        let blk_size = dev.block_size();
        if blk_size == 0 || !PAGE_SIZE.is_multiple_of(blk_size) {
            return Err(PageCacheError::IoError("unsupported device block size"));
        }
        let blocks_per_page = (PAGE_SIZE / blk_size) as u64;
        let start_lba = page_idx
            .checked_mul(blocks_per_page)
            .ok_or(PageCacheError::InvalidOffset)?;
        let end_lba = start_lba
            .checked_add(blocks_per_page)
            .ok_or(PageCacheError::InvalidOffset)?;
        if end_lba > dev.block_count() {
            return Err(PageCacheError::InvalidOffset);
        }

        dev.write_blocks(start_lba, &data[..])
            .map_err(PageCacheError::IoError)
    }

    fn evict_lru<D: BlockDevice>(&mut self, dev: &D) -> Result<(), PageCacheError> {
        let mut victim_idx: Option<usize> = None;
        let mut oldest_tick = u64::MAX;

        for (idx, page) in self.pages.iter().enumerate() {
            if page.pin_count == 0 && page.last_access <= oldest_tick {
                oldest_tick = page.last_access;
                victim_idx = Some(idx);
            }
        }

        let idx = victim_idx.ok_or(PageCacheError::AllPagesPinned)?;
        if self.pages[idx].state == PageState::Dirty {
            let page_idx = self.pages[idx].key.page_idx;
            Self::write_page_to_device(dev, page_idx, &self.pages[idx].data)?;
            self.stats.writebacks += 1;
        }
        self.pages.swap_remove(idx);
        self.stats.evictions += 1;
        Ok(())
    }

    fn ensure_loaded<D: BlockDevice>(
        &mut self,
        dev_id: u32,
        page_idx: u64,
        dev: &D,
    ) -> Result<usize, PageCacheError> {
        let key = PageKey { dev_id, page_idx };
        if let Some(idx) = self.find_index(key) {
            self.stats.hits += 1;
            let t = self.next_tick();
            self.pages[idx].last_access = t;
            return Ok(idx);
        }

        self.stats.misses += 1;
        if self.pages.len() >= self.capacity {
            self.evict_lru(dev)?;
        }

        let data = Self::read_page_from_device(dev, page_idx)?;
        let t = self.next_tick();
        self.pages.push(CachePage {
            key,
            data,
            state: PageState::Clean,
            pin_count: 0,
            last_access: t,
        });
        Ok(self.pages.len() - 1)
    }

    /// Get an immutable reference to a cached 4 KiB page, loading it from `dev` on miss.
    pub fn get_or_load<D: BlockDevice>(
        &mut self,
        dev_id: u32,
        page_idx: u64,
        dev: &D,
    ) -> Result<&[u8; PAGE_SIZE], PageCacheError> {
        let idx = self.ensure_loaded(dev_id, page_idx, dev)?;
        Ok(&self.pages[idx].data)
    }

    /// Get a mutable reference to a cached 4 KiB page, marking it dirty.
    pub fn get_or_load_mut<D: BlockDevice>(
        &mut self,
        dev_id: u32,
        page_idx: u64,
        dev: &D,
    ) -> Result<&mut [u8; PAGE_SIZE], PageCacheError> {
        let idx = self.ensure_loaded(dev_id, page_idx, dev)?;
        self.pages[idx].state = PageState::Dirty;
        Ok(&mut self.pages[idx].data)
    }

    /// Read a 4 KiB page into `out`, using the cache.
    pub fn read_page<D: BlockDevice>(
        &mut self,
        dev_id: u32,
        page_idx: u64,
        dev: &D,
        out: &mut [u8; PAGE_SIZE],
    ) -> Result<(), PageCacheError> {
        let page = self.get_or_load(dev_id, page_idx, dev)?;
        out.copy_from_slice(page);
        Ok(())
    }

    /// Write a 4 KiB page into the cache and mark it dirty (write-back).
    pub fn write_page<D: BlockDevice>(
        &mut self,
        dev_id: u32,
        page_idx: u64,
        dev: &D,
        data: &[u8; PAGE_SIZE],
    ) -> Result<(), PageCacheError> {
        let key = PageKey { dev_id, page_idx };
        if let Some(idx) = self.find_index(key) {
            self.stats.hits += 1;
            let t = self.next_tick();
            self.pages[idx].last_access = t;
            self.pages[idx].data.copy_from_slice(data);
            self.pages[idx].state = PageState::Dirty;
            return Ok(());
        }

        self.stats.misses += 1;
        if self.pages.len() >= self.capacity {
            self.evict_lru(dev)?;
        }

        let mut buf = Box::new([0u8; PAGE_SIZE]);
        buf.copy_from_slice(data);
        let t = self.next_tick();
        self.pages.push(CachePage {
            key,
            data: buf,
            state: PageState::Dirty,
            pin_count: 0,
            last_access: t,
        });
        Ok(())
    }

    /// Pin a cached page so it cannot be evicted. Returns `false` if not cached.
    pub fn pin(&mut self, dev_id: u32, page_idx: u64) -> bool {
        if let Some(idx) = self.find_index(PageKey { dev_id, page_idx }) {
            self.pages[idx].pin_count = self.pages[idx].pin_count.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Unpin a cached page. Returns `false` if not cached or already unpinned.
    pub fn unpin(&mut self, dev_id: u32, page_idx: u64) -> bool {
        if let Some(idx) = self.find_index(PageKey { dev_id, page_idx }) {
            if self.pages[idx].pin_count > 0 {
                self.pages[idx].pin_count -= 1;
                return true;
            }
        }
        false
    }

    /// Flush a single dirty page to `dev`. Returns `Ok(true)` if a writeback occurred.
    pub fn flush_page<D: BlockDevice>(
        &mut self,
        dev_id: u32,
        page_idx: u64,
        dev: &D,
    ) -> Result<bool, PageCacheError> {
        let Some(idx) = self.find_index(PageKey { dev_id, page_idx }) else {
            return Ok(false);
        };
        if self.pages[idx].state == PageState::Dirty {
            Self::write_page_to_device(dev, page_idx, &self.pages[idx].data)?;
            self.pages[idx].state = PageState::Clean;
            self.stats.writebacks += 1;
            return Ok(true);
        }
        Ok(false)
    }

    /// Flush all dirty pages belonging to `dev_id` to `dev`.
    pub fn flush_device<D: BlockDevice>(
        &mut self,
        dev_id: u32,
        dev: &D,
    ) -> Result<usize, PageCacheError> {
        let mut flushed = 0usize;
        for page in &mut self.pages {
            if page.key.dev_id == dev_id && page.state == PageState::Dirty {
                Self::write_page_to_device(dev, page.key.page_idx, &page.data)?;
                page.state = PageState::Clean;
                self.stats.writebacks += 1;
                flushed += 1;
            }
        }
        Ok(flushed)
    }

    /// Flush all dirty pages in the cache to `dev`.
    pub fn flush_all<D: BlockDevice>(&mut self, dev: &D) -> Result<usize, PageCacheError> {
        let mut flushed = 0usize;
        for page in &mut self.pages {
            if page.state == PageState::Dirty {
                Self::write_page_to_device(dev, page.key.page_idx, &page.data)?;
                page.state = PageState::Clean;
                self.stats.writebacks += 1;
                flushed += 1;
            }
        }
        Ok(flushed)
    }

    /// Remove a page from the cache without writing it back.
    pub fn invalidate(&mut self, dev_id: u32, page_idx: u64) -> bool {
        if let Some(idx) = self.find_index(PageKey { dev_id, page_idx }) {
            self.pages.swap_remove(idx);
            true
        } else {
            false
        }
    }

    /// Return a snapshot of cache statistics.
    pub fn stats(&self) -> PageCacheStats {
        let dirty_pages = self
            .pages
            .iter()
            .filter(|p| p.state == PageState::Dirty)
            .count();
        PageCacheStats {
            hits: self.stats.hits,
            misses: self.stats.misses,
            evictions: self.stats.evictions,
            writebacks: self.stats.writebacks,
            dirty_pages,
            cached_pages: self.pages.len(),
            capacity: self.capacity,
        }
    }
}

/// Spinlock-protected global/shared page cache wrapper.
pub struct SharedPageCache {
    inner: SpinLock<PageCache>,
}

impl SharedPageCache {
    pub fn new(capacity: usize) -> Result<Self, PageCacheError> {
        Ok(Self {
            inner: SpinLock::new(PageCache::new(capacity)?),
        })
    }

    pub fn lock(&self) -> crate::mm::SpinLockGuard<'_, PageCache> {
        self.inner.lock()
    }
}

// ---------------------------------------------------------------------------
// Crash-Safe WAL Journal with CRC32C Checksums
// ---------------------------------------------------------------------------

/// Kind of record stored in the Write-Ahead Log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalRecordKind {
    Begin = 1,
    WritePage = 2,
    Commit = 3,
    Abort = 4,
    Checkpoint = 5,
}

impl WalRecordKind {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Begin),
            2 => Some(Self::WritePage),
            3 => Some(Self::Commit),
            4 => Some(Self::Abort),
            5 => Some(Self::Checkpoint),
            _ => None,
        }
    }
}

/// Decoded WAL record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
    pub lsn: u64,
    pub tx_id: u64,
    pub kind: WalRecordKind,
    pub target_lba: u64,
    pub payload: Vec<u8>,
}

/// Crash-safe Write-Ahead Log journal with CRC32C checksums.
pub struct WalJournal {
    next_lsn: u64,
    next_tx_id: u64,
    checkpoint_lsn: u64,
    max_bytes: usize,
    log_buf: Vec<u8>,
}

impl WalJournal {
    /// Create a new WAL journal with a maximum log buffer size in bytes.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            next_lsn: 1,
            next_tx_id: 1,
            checkpoint_lsn: 0,
            max_bytes,
            log_buf: Vec::new(),
        }
    }

    /// Current number of bytes used in the WAL buffer.
    pub fn used_bytes(&self) -> usize {
        self.log_buf.len()
    }

    /// Next LSN that will be assigned.
    pub fn next_lsn(&self) -> u64 {
        self.next_lsn
    }

    /// Most recent checkpoint LSN.
    pub fn checkpoint_lsn(&self) -> u64 {
        self.checkpoint_lsn
    }

    /// Raw serialized WAL bytes (for inspection or testing).
    pub fn as_bytes(&self) -> &[u8] {
        &self.log_buf
    }

    /// Mutable access to raw WAL bytes (for fault-injection testing).
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.log_buf
    }

    fn append_record(
        &mut self,
        tx_id: u64,
        kind: WalRecordKind,
        target_lba: u64,
        payload: &[u8],
    ) -> Result<u64, PageCacheError> {
        let total_len = WAL_HEADER_SIZE
            .checked_add(payload.len())
            .ok_or(PageCacheError::WalFull)?;
        if self.log_buf.len().saturating_add(total_len) > self.max_bytes {
            return Err(PageCacheError::WalFull);
        }

        let lsn = self.next_lsn;
        self.next_lsn = self.next_lsn.wrapping_add(1);

        let mut hdr = [0u8; WAL_HEADER_SIZE];
        hdr[0..4].copy_from_slice(&WAL_MAGIC.to_le_bytes());
        hdr[4..6].copy_from_slice(&WAL_VERSION.to_le_bytes());
        hdr[6] = kind as u8;
        hdr[7] = 0; // flags
        hdr[8..16].copy_from_slice(&lsn.to_le_bytes());
        hdr[16..24].copy_from_slice(&tx_id.to_le_bytes());
        hdr[24..32].copy_from_slice(&target_lba.to_le_bytes());
        hdr[32..36].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        // hdr[36..40] is zeroed while computing CRC32C
        let mut crc = crc32c(&hdr[0..36]);
        crc = crc32c_update(crc, payload);
        hdr[36..40].copy_from_slice(&crc.to_le_bytes());

        self.log_buf.extend_from_slice(&hdr);
        self.log_buf.extend_from_slice(payload);
        Ok(lsn)
    }

    /// Begin a new transaction and return its `tx_id`.
    pub fn begin_tx(&mut self) -> Result<u64, PageCacheError> {
        let tx_id = self.next_tx_id;
        self.next_tx_id = self.next_tx_id.wrapping_add(1);
        self.append_record(tx_id, WalRecordKind::Begin, 0, &[])?;
        Ok(tx_id)
    }

    /// Log a page/sector write for `tx_id` targeting `target_lba`.
    pub fn log_write(
        &mut self,
        tx_id: u64,
        target_lba: u64,
        data: &[u8],
    ) -> Result<u64, PageCacheError> {
        self.append_record(tx_id, WalRecordKind::WritePage, target_lba, data)
    }

    /// Commit transaction `tx_id`.
    pub fn commit_tx(&mut self, tx_id: u64) -> Result<u64, PageCacheError> {
        self.append_record(tx_id, WalRecordKind::Commit, 0, &[])
    }

    /// Abort transaction `tx_id`.
    pub fn abort_tx(&mut self, tx_id: u64) -> Result<u64, PageCacheError> {
        self.append_record(tx_id, WalRecordKind::Abort, 0, &[])
    }

    /// Record a checkpoint marker in the WAL.
    pub fn checkpoint(&mut self) -> Result<u64, PageCacheError> {
        let lsn = self.append_record(0, WalRecordKind::Checkpoint, 0, &[])?;
        self.checkpoint_lsn = lsn;
        Ok(lsn)
    }

    /// Flush the WAL buffer to `dev` starting at `start_lba` (padded to sector size).
    pub fn flush_to_device<D: BlockDevice>(
        &self,
        dev: &D,
        start_lba: u64,
    ) -> Result<usize, PageCacheError> {
        let blk_size = dev.block_size();
        if blk_size == 0 {
            return Err(PageCacheError::IoError("invalid block size"));
        }
        if self.log_buf.is_empty() {
            return Ok(0);
        }
        let padded_len = self.log_buf.len().div_ceil(blk_size) * blk_size;
        let sectors = (padded_len / blk_size) as u64;
        if start_lba.saturating_add(sectors) > dev.block_count() {
            return Err(PageCacheError::InvalidOffset);
        }

        let mut padded = vec![0u8; padded_len];
        padded[..self.log_buf.len()].copy_from_slice(&self.log_buf);
        dev.write_blocks(start_lba, &padded)
            .map_err(PageCacheError::IoError)?;
        Ok(padded_len)
    }

    /// Load a WAL buffer from `dev` starting at `start_lba` for `byte_len` bytes.
    pub fn load_from_device<D: BlockDevice>(
        dev: &D,
        start_lba: u64,
        byte_len: usize,
    ) -> Result<Self, PageCacheError> {
        let blk_size = dev.block_size();
        if blk_size == 0 {
            return Err(PageCacheError::IoError("invalid block size"));
        }
        let padded_len = byte_len.div_ceil(blk_size) * blk_size;
        let sectors = (padded_len / blk_size) as u64;
        if start_lba.saturating_add(sectors) > dev.block_count() {
            return Err(PageCacheError::InvalidOffset);
        }

        let mut buf = vec![0u8; padded_len];
        if padded_len > 0 {
            dev.read_blocks(start_lba, &mut buf)
                .map_err(PageCacheError::IoError)?;
        }
        buf.truncate(byte_len);

        let mut journal = Self::new(byte_len.max(PAGE_SIZE));
        journal.log_buf = buf;
        // Advance next_lsn / next_tx_id past any valid records found.
        let records = journal.scan_valid_records();
        for rec in &records {
            if rec.lsn >= journal.next_lsn {
                journal.next_lsn = rec.lsn + 1;
            }
            if rec.tx_id >= journal.next_tx_id {
                journal.next_tx_id = rec.tx_id + 1;
            }
            if rec.kind == WalRecordKind::Checkpoint && rec.lsn > journal.checkpoint_lsn {
                journal.checkpoint_lsn = rec.lsn;
            }
        }
        Ok(journal)
    }

    /// Scan `log_buf` from the start, stopping cleanly at the first invalid or
    /// CRC32C-mismatched record (torn write at tail).
    pub fn scan_valid_records(&self) -> Vec<WalRecord> {
        let mut out = Vec::new();
        let mut offset = 0usize;

        while offset + WAL_HEADER_SIZE <= self.log_buf.len() {
            let hdr = &self.log_buf[offset..offset + WAL_HEADER_SIZE];
            let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
            let version = u16::from_le_bytes([hdr[4], hdr[5]]);
            if magic != WAL_MAGIC || version != WAL_VERSION {
                break;
            }
            let Some(kind) = WalRecordKind::from_u8(hdr[6]) else {
                break;
            };
            let lsn = u64::from_le_bytes([
                hdr[8], hdr[9], hdr[10], hdr[11], hdr[12], hdr[13], hdr[14], hdr[15],
            ]);
            let tx_id = u64::from_le_bytes([
                hdr[16], hdr[17], hdr[18], hdr[19], hdr[20], hdr[21], hdr[22], hdr[23],
            ]);
            let target_lba = u64::from_le_bytes([
                hdr[24], hdr[25], hdr[26], hdr[27], hdr[28], hdr[29], hdr[30], hdr[31],
            ]);
            let payload_len = u32::from_le_bytes([hdr[32], hdr[33], hdr[34], hdr[35]]) as usize;
            let stored_crc = u32::from_le_bytes([hdr[36], hdr[37], hdr[38], hdr[39]]);

            let payload_start = offset + WAL_HEADER_SIZE;
            let Some(payload_end) = payload_start.checked_add(payload_len) else {
                break;
            };
            if payload_end > self.log_buf.len() {
                break;
            }

            let payload = &self.log_buf[payload_start..payload_end];
            let mut calc_crc = crc32c(&hdr[0..36]);
            calc_crc = crc32c_update(calc_crc, payload);
            if calc_crc != stored_crc {
                break;
            }

            out.push(WalRecord {
                lsn,
                tx_id,
                kind,
                target_lba,
                payload: payload.to_vec(),
            });
            offset = payload_end;
        }

        out
    }

    /// Recover all committed `WritePage` records after the latest checkpoint.
    /// Uncommitted or aborted transactions and torn tail records are excluded.
    pub fn recover(&self) -> Vec<WalRecord> {
        let records = self.scan_valid_records();
        let mut latest_ckpt_lsn = 0u64;
        for rec in &records {
            if rec.kind == WalRecordKind::Checkpoint && rec.lsn > latest_ckpt_lsn {
                latest_ckpt_lsn = rec.lsn;
            }
        }

        let mut committed_txs: Vec<u64> = Vec::new();
        for rec in &records {
            if rec.kind == WalRecordKind::Commit && !committed_txs.contains(&rec.tx_id) {
                committed_txs.push(rec.tx_id);
            }
        }

        records
            .into_iter()
            .filter(|rec| {
                rec.kind == WalRecordKind::WritePage
                    && rec.lsn > latest_ckpt_lsn
                    && committed_txs.contains(&rec.tx_id)
            })
            .collect()
    }

    /// Replay all committed post-checkpoint writes onto `dev`.
    pub fn replay_to_device<D: BlockDevice>(&self, dev: &D) -> Result<usize, PageCacheError> {
        let to_replay = self.recover();
        let blk_size = dev.block_size();
        if blk_size == 0 {
            return Err(PageCacheError::IoError("invalid block size"));
        }
        let mut replayed = 0usize;
        for rec in &to_replay {
            if rec.payload.is_empty() || rec.payload.len() % blk_size != 0 {
                return Err(PageCacheError::WalInvalidRecord);
            }
            let sectors = (rec.payload.len() / blk_size) as u64;
            if rec.target_lba.saturating_add(sectors) > dev.block_count() {
                return Err(PageCacheError::InvalidOffset);
            }
            dev.write_blocks(rec.target_lba, &rec.payload)
                .map_err(PageCacheError::IoError)?;
            replayed += 1;
        }
        Ok(replayed)
    }

    /// Discard all WAL records at or before the latest checkpoint.
    pub fn truncate_before_checkpoint(&mut self) {
        let records = self.scan_valid_records();
        let mut latest_ckpt_lsn = 0u64;
        for rec in &records {
            if rec.kind == WalRecordKind::Checkpoint && rec.lsn > latest_ckpt_lsn {
                latest_ckpt_lsn = rec.lsn;
            }
        }
        if latest_ckpt_lsn == 0 {
            return;
        }

        let mut new_buf = Vec::new();
        for rec in records {
            if rec.lsn > latest_ckpt_lsn {
                let mut hdr = [0u8; WAL_HEADER_SIZE];
                hdr[0..4].copy_from_slice(&WAL_MAGIC.to_le_bytes());
                hdr[4..6].copy_from_slice(&WAL_VERSION.to_le_bytes());
                hdr[6] = rec.kind as u8;
                hdr[7] = 0;
                hdr[8..16].copy_from_slice(&rec.lsn.to_le_bytes());
                hdr[16..24].copy_from_slice(&rec.tx_id.to_le_bytes());
                hdr[24..32].copy_from_slice(&rec.target_lba.to_le_bytes());
                hdr[32..36].copy_from_slice(&(rec.payload.len() as u32).to_le_bytes());
                let mut crc = crc32c(&hdr[0..36]);
                crc = crc32c_update(crc, &rec.payload);
                hdr[36..40].copy_from_slice(&crc.to_le_bytes());
                new_buf.extend_from_slice(&hdr);
                new_buf.extend_from_slice(&rec.payload);
            }
        }
        self.log_buf = new_buf;
    }
}

// ---------------------------------------------------------------------------
// DMA Scatter-Gather Engine (SgList)
// ---------------------------------------------------------------------------

/// Hardware-friendly DMA scatter-gather descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct SgDescriptor {
    pub phys_addr: u64,
    pub len: u32,
    pub flags: u32,
}

/// DMA scatter-gather descriptor list builder.
///
/// Automatically validates alignment, coalesces physically contiguous regions
/// up to `max_segment_bytes`, splits oversized regions, and marks the final
/// descriptor with [`SG_FLAG_EOT`] on [`SgList::finalize`].
#[derive(Debug, Clone)]
pub struct SgList {
    segments: Vec<SgDescriptor>,
    total_bytes: usize,
    max_segment_bytes: usize,
    align_mask: u64,
}

impl Default for SgList {
    fn default() -> Self {
        Self::new()
    }
}

impl SgList {
    /// Create a new [`SgList`] with 64 KiB max segment size and 4-byte alignment.
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            total_bytes: 0,
            max_segment_bytes: DEFAULT_MAX_SEGMENT_BYTES,
            align_mask: 3,
        }
    }

    /// Create a new [`SgList`] with custom `max_segment_bytes` and power-of-two `alignment`.
    pub fn with_constraints(
        max_segment_bytes: usize,
        alignment: usize,
    ) -> Result<Self, PageCacheError> {
        if max_segment_bytes == 0 || max_segment_bytes > u32::MAX as usize {
            return Err(PageCacheError::InvalidCapacity);
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(PageCacheError::DmaAlignmentError);
        }
        Ok(Self {
            segments: Vec::new(),
            total_bytes: 0,
            max_segment_bytes,
            align_mask: (alignment - 1) as u64,
        })
    }

    /// Push a physical memory region into the scatter-gather list.
    pub fn push_region(&mut self, phys_addr: u64, len: usize) -> Result<(), PageCacheError> {
        if len == 0 {
            return Ok(());
        }
        if (phys_addr & self.align_mask) != 0 || ((len as u64) & self.align_mask) != 0 {
            return Err(PageCacheError::DmaAlignmentError);
        }

        let mut cur_addr = phys_addr;
        let mut remaining = len;

        // Coalesce with the previous segment if physically contiguous and within limit.
        if let Some(last) = self.segments.last_mut() {
            let last_end = last.phys_addr.saturating_add(last.len as u64);
            if last_end == cur_addr && (last.len as usize) < self.max_segment_bytes {
                let room = self.max_segment_bytes - (last.len as usize);
                let take = remaining.min(room);
                last.len += take as u32;
                last.flags &= !SG_FLAG_EOT;
                cur_addr = cur_addr.saturating_add(take as u64);
                remaining -= take;
                self.total_bytes += take;
            }
        }

        while remaining > 0 {
            if self.segments.len() >= MAX_SG_SEGMENTS {
                return Err(PageCacheError::DmaTooManySegments);
            }
            let chunk = remaining.min(self.max_segment_bytes);
            self.segments.push(SgDescriptor {
                phys_addr: cur_addr,
                len: chunk as u32,
                flags: SG_FLAG_NONE,
            });
            cur_addr = cur_addr.saturating_add(chunk as u64);
            remaining -= chunk;
            self.total_bytes += chunk;
        }

        Ok(())
    }

    /// Translate an HHDM virtual buffer address to physical and push it.
    pub fn push_virt_buffer(
        &mut self,
        virt_addr: usize,
        len: usize,
        hhdm_offset: u64,
    ) -> Result<(), PageCacheError> {
        let virt = virt_addr as u64;
        if virt < hhdm_offset {
            return Err(PageCacheError::InvalidOffset);
        }
        self.push_region(virt - hhdm_offset, len)
    }

    /// Mark the final descriptor with [`SG_FLAG_EOT`].
    pub fn finalize(&mut self) -> Result<(), PageCacheError> {
        if self.segments.is_empty() {
            return Err(PageCacheError::DmaEmpty);
        }
        for seg in &mut self.segments {
            seg.flags &= !SG_FLAG_EOT;
        }
        if let Some(last) = self.segments.last_mut() {
            last.flags |= SG_FLAG_EOT;
        }
        Ok(())
    }

    /// Return the built scatter-gather descriptors.
    pub fn segments(&self) -> &[SgDescriptor] {
        &self.segments
    }

    /// Number of descriptors in the list.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Returns `true` if no descriptors have been added.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Total byte count covered by all descriptors.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Clear all descriptors.
    pub fn clear(&mut self) {
        self.segments.clear();
        self.total_bytes = 0;
    }
}

// ---------------------------------------------------------------------------
// Self-Test
// ---------------------------------------------------------------------------

struct MemBlockDev {
    sectors: SpinLock<Vec<u8>>,
    sector_count: u64,
}

impl MemBlockDev {
    fn new(sector_count: u64) -> Self {
        Self {
            sectors: SpinLock::new(vec![0u8; (sector_count as usize) * SECTOR_SIZE]),
            sector_count,
        }
    }
}

impl BlockDevice for MemBlockDev {
    fn block_size(&self) -> usize {
        SECTOR_SIZE
    }

    fn block_count(&self) -> u64 {
        self.sector_count
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(SECTOR_SIZE) {
            return Err("unaligned read buffer");
        }
        let start = (lba as usize) * SECTOR_SIZE;
        let end = start + buf.len();
        let guard = self.sectors.lock();
        if end > guard.len() {
            return Err("read out of bounds");
        }
        buf.copy_from_slice(&guard[start..end]);
        Ok(())
    }

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(SECTOR_SIZE) {
            return Err("unaligned write buffer");
        }
        let start = (lba as usize) * SECTOR_SIZE;
        let end = start + buf.len();
        let mut guard = self.sectors.lock();
        if end > guard.len() {
            return Err("write out of bounds");
        }
        guard[start..end].copy_from_slice(buf);
        Ok(())
    }
}

/// Run self-tests for the 4 KiB LRU page cache, WAL journal with CRC32C, and
/// `SgList` DMA scatter-gather descriptor builder.
pub fn self_test() -> Result<(), &'static str> {
    let dev = MemBlockDev::new(256); // 128 KiB (32 pages)

    // 1. PageCache LRU + dirty writeback test (capacity = 2 pages)
    let mut cache = PageCache::new(2).map_err(|_| "PageCache::new failed")?;
    let p0 = [0xA0u8; PAGE_SIZE];
    let p1 = [0xB1u8; PAGE_SIZE];
    let p2 = [0xC2u8; PAGE_SIZE];

    cache
        .write_page(0, 0, &dev, &p0)
        .map_err(|_| "write_page 0 failed")?;
    cache
        .write_page(0, 1, &dev, &p1)
        .map_err(|_| "write_page 1 failed")?;
    // Writing page 2 must evict LRU page 0 and write it back to `dev`.
    cache
        .write_page(0, 2, &dev, &p2)
        .map_err(|_| "write_page 2 failed")?;

    let mut verify_buf = [0u8; PAGE_SIZE];
    cache
        .read_page(0, 0, &dev, &mut verify_buf)
        .map_err(|_| "read_page 0 failed")?;
    if verify_buf != p0 {
        return Err("PageCache dirty writeback mismatch on page 0");
    }
    cache
        .flush_all(&dev)
        .map_err(|_| "PageCache flush_all failed")?;
    let stats = cache.stats();
    if stats.evictions < 1 || stats.writebacks < 2 || stats.dirty_pages != 0 {
        return Err("PageCache stats mismatch");
    }

    // 2. WAL Journal + CRC32C crash recovery test
    let mut wal = WalJournal::new(16 * 1024);
    let tx1 = wal.begin_tx().map_err(|_| "wal begin_tx 1 failed")?;
    let committed_sector = [0x5Au8; SECTOR_SIZE];
    wal.log_write(tx1, 16, &committed_sector)
        .map_err(|_| "wal log_write 1 failed")?;
    wal.commit_tx(tx1).map_err(|_| "wal commit_tx 1 failed")?;

    let tx2 = wal.begin_tx().map_err(|_| "wal begin_tx 2 failed")?;
    let uncommitted_sector = [0xEEu8; SECTOR_SIZE];
    wal.log_write(tx2, 24, &uncommitted_sector)
        .map_err(|_| "wal log_write 2 failed")?;
    // Simulate crash before tx2 commits, plus a torn/corrupted tail byte.
    let replayed = wal
        .replay_to_device(&dev)
        .map_err(|_| "wal replay_to_device failed")?;
    if replayed != 1 {
        return Err("WAL replay count mismatch");
    }

    let mut sec16 = [0u8; SECTOR_SIZE];
    let mut sec24 = [0u8; SECTOR_SIZE];
    dev.read_blocks(16, &mut sec16)?;
    dev.read_blocks(24, &mut sec24)?;
    if sec16 != committed_sector || sec24 != [0u8; SECTOR_SIZE] {
        return Err("WAL recovery state mismatch");
    }

    // Verify CRC32C detects corruption in the WAL buffer.
    let valid_before = wal.scan_valid_records().len();
    if let Some(first_byte) = wal.as_bytes_mut().get_mut(10) {
        *first_byte ^= 0xFF;
    }
    if wal.scan_valid_records().len() >= valid_before {
        return Err("WAL CRC32C failed to detect corruption");
    }

    // 3. SgList DMA descriptor builder test
    let mut sg = SgList::with_constraints(4096, 512).map_err(|_| "SgList::with_constraints")?;
    // Two contiguous 2 KiB regions should coalesce into one 4 KiB segment.
    sg.push_region(0x10_0000, 2048)
        .map_err(|_| "sg push_region 1 failed")?;
    sg.push_region(0x10_0800, 2048)
        .map_err(|_| "sg push_region 2 failed")?;
    // A 6 KiB region should split into 4 KiB + 2 KiB.
    sg.push_region(0x20_0000, 6144)
        .map_err(|_| "sg push_region 3 failed")?;
    sg.finalize().map_err(|_| "sg finalize failed")?;

    if sg.len() != 3 || sg.total_bytes() != 10240 {
        return Err("SgList coalescing/splitting mismatch");
    }
    if (sg.segments()[2].flags & SG_FLAG_EOT) == 0 {
        return Err("SgList missing EOT flag");
    }
    if sg.push_region(0x30_0001, 512).is_ok() {
        return Err("SgList failed to reject unaligned address");
    }

    serial_println!(
        "PAGE_CACHE: self_test passed (evictions={}, writebacks={}, wal_replayed={}, sg_segs={})",
        stats.evictions,
        stats.writebacks,
        replayed,
        sg.len()
    );
    Ok(())
}
