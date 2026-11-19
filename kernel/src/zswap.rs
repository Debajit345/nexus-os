//! `no_std` Demand Paging, Compressed ZRAM/Zswap Pool & Page-Reclaim Engine.
//!
//! Provides:
//! - Fast LZ-style block compressor and decompressor with zero-page / same-fill
//!   fast paths and incompressible-page raw fallback.
//! - [`SwapSlotAllocator`] bitmap allocator for virtual swap slots.
//! - [`ZswapPool`] compressed in-RAM swap pool with CRC32C integrity checks and
//!   memory budget enforcement.
//! - [`PageReclaimEngine`] two-list (Active / Inactive) LRU page-reclaim engine
//!   with second-chance promotion, pinning, and demand-paging fault-in.
//! - [`OomEvaluator`] process OOM badness scorer and victim selector.
//! - [`DemandPager`] unified demand-paging and memory-pressure controller.
//! - [`self_test`] exercising all subsystems with small stack frames (< 2 KiB).

#![allow(dead_code)]

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::mm::{SpinLock, PAGE_SIZE};
use crate::page_cache::crc32c;
use crate::serial_println;

/// Magic identifier for zswap metadata ("ZSWP").
pub const ZSWAP_MAGIC: u32 = 0x5A53_5750;

/// Default number of swap slots managed by [`SwapSlotAllocator`].
pub const DEFAULT_SWAP_SLOTS: usize = 4096;

/// Default compressed pool budget in bytes (4 MiB).
pub const DEFAULT_POOL_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Minimum `oom_score_adj` value that makes a process immune to OOM killing.
pub const OOM_SCORE_ADJ_MIN: i16 = -1000;

/// Maximum `oom_score_adj` value.
pub const OOM_SCORE_ADJ_MAX: i16 = 1000;

/// Hash table size for the LZ-style compressor (power of two).
const LZ_HASH_BITS: usize = 11;
const LZ_HASH_SIZE: usize = 1 << LZ_HASH_BITS;
const LZ_MIN_MATCH: usize = 4;
const LZ_MAX_OFFSET: usize = u16::MAX as usize;

/// Errors returned by the zswap, demand-paging, and reclaim subsystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZswapError {
    InvalidPageSize,
    CorruptedStream,
    OutputTooSmall,
    PoolFull,
    SwapFull,
    SlotNotFound,
    SlotAlreadyFree,
    PageNotFound,
    PagePinned,
    NoReclaimablePages,
    NoKillableProcess,
    InvalidPid,
}

impl fmt::Display for ZswapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPageSize => write!(f, "invalid page size"),
            Self::CorruptedStream => write!(f, "corrupted compressed stream"),
            Self::OutputTooSmall => write!(f, "output buffer too small"),
            Self::PoolFull => write!(f, "zswap compressed pool full"),
            Self::SwapFull => write!(f, "swap slot space exhausted"),
            Self::SlotNotFound => write!(f, "swap slot not found"),
            Self::SlotAlreadyFree => write!(f, "swap slot already free"),
            Self::PageNotFound => write!(f, "page not tracked"),
            Self::PagePinned => write!(f, "page is pinned"),
            Self::NoReclaimablePages => write!(f, "no reclaimable pages available"),
            Self::NoKillableProcess => write!(f, "no killable process for OOM"),
            Self::InvalidPid => write!(f, "invalid process id"),
        }
    }
}

// ---------------------------------------------------------------------------
// Fast LZ-Style Block Compressor / Decompressor
// ---------------------------------------------------------------------------

/// Check whether a slice consists entirely of a single repeated byte.
/// Returns `Some(byte)` if uniform (including all-zero), or `None` otherwise.
pub fn detect_same_fill(data: &[u8]) -> Option<u8> {
    let first = *data.first()?;
    if data.iter().all(|&b| b == first) {
        Some(first)
    } else {
        None
    }
}

#[inline]
fn lz_hash4(data: &[u8], pos: usize) -> usize {
    let v = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    ((v.wrapping_mul(2_654_435_761)) >> (32 - LZ_HASH_BITS)) as usize
}

fn write_ext_len(out: &mut Vec<u8>, mut len: usize) {
    while len >= 255 {
        out.push(255);
        len -= 255;
    }
    out.push(len as u8);
}

fn read_ext_len(src: &[u8], pos: &mut usize) -> Result<usize, ZswapError> {
    let mut total = 0usize;
    loop {
        if *pos >= src.len() {
            return Err(ZswapError::CorruptedStream);
        }
        let b = src[*pos] as usize;
        *pos += 1;
        total = total.checked_add(b).ok_or(ZswapError::CorruptedStream)?;
        if b < 255 {
            return Ok(total);
        }
    }
}

/// Compress `src` using a fast LZ4-style byte-oriented block format.
///
/// The hash table is heap-allocated so the stack frame remains tiny (< 256 B).
pub fn lz_compress(src: &[u8]) -> Vec<u8> {
    if src.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(src.len() / 2 + 16);
    if src.len() < LZ_MIN_MATCH + 1 {
        let lit_len = src.len();
        let token_lit = lit_len.min(15) as u8;
        out.push(token_lit << 4);
        if lit_len >= 15 {
            write_ext_len(&mut out, lit_len - 15);
        }
        out.extend_from_slice(src);
        return out;
    }

    let mut hash_table = vec![u32::MAX; LZ_HASH_SIZE];
    let mut anchor = 0usize;
    let mut pos = 0usize;
    let limit = src.len().saturating_sub(LZ_MIN_MATCH);

    while pos <= limit {
        let h = lz_hash4(src, pos);
        let candidate = hash_table[h] as usize;
        hash_table[h] = pos as u32;

        let valid_match = candidate != u32::MAX as usize
            && pos > candidate
            && (pos - candidate) <= LZ_MAX_OFFSET
            && src[candidate..candidate + 4] == src[pos..pos + 4];

        if !valid_match {
            pos += 1;
            continue;
        }

        // Extend match forward.
        let mut match_len = LZ_MIN_MATCH;
        while pos + match_len < src.len() && src[candidate + match_len] == src[pos + match_len] {
            match_len += 1;
        }

        let lit_len = pos - anchor;
        let enc_match = match_len - LZ_MIN_MATCH;
        let token = ((lit_len.min(15) as u8) << 4) | (enc_match.min(15) as u8);
        out.push(token);

        if lit_len >= 15 {
            write_ext_len(&mut out, lit_len - 15);
        }
        out.extend_from_slice(&src[anchor..pos]);

        let offset = (pos - candidate) as u16;
        out.extend_from_slice(&offset.to_le_bytes());

        if enc_match >= 15 {
            write_ext_len(&mut out, enc_match - 15);
        }

        pos += match_len;
        anchor = pos;
    }

    // Emit trailing literals if any remain.
    if anchor < src.len() {
        let lit_len = src.len() - anchor;
        let token = (lit_len.min(15) as u8) << 4;
        out.push(token);
        if lit_len >= 15 {
            write_ext_len(&mut out, lit_len - 15);
        }
        out.extend_from_slice(&src[anchor..]);
    }

    out
}

/// Decompress an LZ4-style block produced by [`lz_compress`] into `dst`.
///
/// Returns the number of bytes written to `dst`.
pub fn lz_decompress(src: &[u8], dst: &mut [u8]) -> Result<usize, ZswapError> {
    if src.is_empty() {
        return Ok(0);
    }

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;

    while in_pos < src.len() {
        let token = src[in_pos];
        in_pos += 1;

        let mut lit_len = (token >> 4) as usize;
        if lit_len == 15 {
            lit_len = lit_len
                .checked_add(read_ext_len(src, &mut in_pos)?)
                .ok_or(ZswapError::CorruptedStream)?;
        }

        if in_pos
            .checked_add(lit_len)
            .is_none_or(|end| end > src.len())
        {
            return Err(ZswapError::CorruptedStream);
        }
        if out_pos
            .checked_add(lit_len)
            .is_none_or(|end| end > dst.len())
        {
            return Err(ZswapError::OutputTooSmall);
        }

        dst[out_pos..out_pos + lit_len].copy_from_slice(&src[in_pos..in_pos + lit_len]);
        in_pos += lit_len;
        out_pos += lit_len;

        // End of stream after trailing literals.
        if in_pos >= src.len() {
            break;
        }

        if in_pos + 2 > src.len() {
            return Err(ZswapError::CorruptedStream);
        }
        let offset = u16::from_le_bytes([src[in_pos], src[in_pos + 1]]) as usize;
        in_pos += 2;

        if offset == 0 || offset > out_pos {
            return Err(ZswapError::CorruptedStream);
        }

        let mut match_len = (token & 0x0F) as usize;
        if match_len == 15 {
            match_len = match_len
                .checked_add(read_ext_len(src, &mut in_pos)?)
                .ok_or(ZswapError::CorruptedStream)?;
        }
        match_len = match_len
            .checked_add(LZ_MIN_MATCH)
            .ok_or(ZswapError::CorruptedStream)?;

        if out_pos
            .checked_add(match_len)
            .is_none_or(|end| end > dst.len())
        {
            return Err(ZswapError::OutputTooSmall);
        }

        // Byte-by-byte copy handles overlapping RLE matches (offset < match_len).
        let mut copy_src = out_pos - offset;
        for _ in 0..match_len {
            dst[out_pos] = dst[copy_src];
            out_pos += 1;
            copy_src += 1;
        }
    }

    Ok(out_pos)
}

// ---------------------------------------------------------------------------
// Swap-Slot Allocator
// ---------------------------------------------------------------------------

/// Opaque handle to an allocated swap slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SwapSlot(pub u32);

/// Bitmap-backed swap-slot allocator.
pub struct SwapSlotAllocator {
    bitmap: Vec<u64>,
    capacity: usize,
    used: usize,
    next_hint: usize,
}

impl SwapSlotAllocator {
    /// Create a new swap-slot allocator with `capacity` slots.
    pub fn new(capacity: usize) -> Self {
        let words = capacity.div_ceil(64);
        Self {
            bitmap: vec![0u64; words],
            capacity,
            used: 0,
            next_hint: 0,
        }
    }

    /// Allocate a free swap slot.
    pub fn alloc(&mut self) -> Result<SwapSlot, ZswapError> {
        if self.used >= self.capacity || self.bitmap.is_empty() {
            return Err(ZswapError::SwapFull);
        }

        let words = self.bitmap.len();
        let start_word = self.next_hint / 64;

        for offset in 0..words {
            let word_idx = (start_word + offset) % words;
            let word = self.bitmap[word_idx];
            if word != u64::MAX {
                let bit = (!word).trailing_zeros() as usize;
                let slot_idx = word_idx * 64 + bit;
                if slot_idx < self.capacity {
                    self.bitmap[word_idx] |= 1u64 << bit;
                    self.used += 1;
                    self.next_hint = (slot_idx + 1) % self.capacity;
                    return Ok(SwapSlot(slot_idx as u32));
                }
            }
        }

        Err(ZswapError::SwapFull)
    }

    /// Free a previously allocated swap slot.
    pub fn free(&mut self, slot: SwapSlot) -> Result<(), ZswapError> {
        let idx = slot.0 as usize;
        if idx >= self.capacity {
            return Err(ZswapError::SlotNotFound);
        }
        let word_idx = idx / 64;
        let bit = idx % 64;
        let mask = 1u64 << bit;
        if (self.bitmap[word_idx] & mask) == 0 {
            return Err(ZswapError::SlotAlreadyFree);
        }
        self.bitmap[word_idx] &= !mask;
        self.used -= 1;
        if idx < self.next_hint {
            self.next_hint = idx;
        }
        Ok(())
    }

    /// Check whether `slot` is currently allocated.
    pub fn is_allocated(&self, slot: SwapSlot) -> bool {
        let idx = slot.0 as usize;
        if idx >= self.capacity {
            return false;
        }
        (self.bitmap[idx / 64] & (1u64 << (idx % 64))) != 0
    }

    pub fn used_slots(&self) -> usize {
        self.used
    }

    pub fn free_slots(&self) -> usize {
        self.capacity.saturating_sub(self.used)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// Compressed ZRAM / Zswap Pool
// ---------------------------------------------------------------------------

/// Storage encoding for a page inside [`ZswapPool`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageEncoding {
    /// All-zero page (0 bytes stored).
    Zero,
    /// Uniform byte-filled page (1 byte stored).
    SameFill(u8),
    /// LZ-compressed payload.
    Compressed(Vec<u8>),
    /// Incompressible page stored verbatim.
    Raw(Vec<u8>),
}

impl PageEncoding {
    fn stored_bytes(&self) -> usize {
        match self {
            Self::Zero => 0,
            Self::SameFill(_) => 1,
            Self::Compressed(v) | Self::Raw(v) => v.len(),
        }
    }
}

#[derive(Debug, Clone)]
struct ZswapEntry {
    encoding: PageEncoding,
    crc32: u32,
}

/// Snapshot of [`ZswapPool`] statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZswapPoolStats {
    pub max_pool_bytes: usize,
    pub used_pool_bytes: usize,
    pub stored_pages: usize,
    pub zero_pages: usize,
    pub same_fill_pages: usize,
    pub compressed_pages: usize,
    pub raw_pages: usize,
    pub total_uncompressed_bytes: u64,
    pub total_compressed_bytes: u64,
}

/// Compressed in-memory swap pool (ZRAM / Zswap).
pub struct ZswapPool {
    entries: BTreeMap<SwapSlot, ZswapEntry>,
    max_pool_bytes: usize,
    used_pool_bytes: usize,
    zero_pages: usize,
    same_fill_pages: usize,
    compressed_pages: usize,
    raw_pages: usize,
    total_uncompressed_bytes: u64,
    total_compressed_bytes: u64,
}

impl ZswapPool {
    /// Create a new compressed pool with a maximum byte budget.
    pub fn new(max_pool_bytes: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            max_pool_bytes,
            used_pool_bytes: 0,
            zero_pages: 0,
            same_fill_pages: 0,
            compressed_pages: 0,
            raw_pages: 0,
            total_uncompressed_bytes: 0,
            total_compressed_bytes: 0,
        }
    }

    /// Compress and store a 4 KiB page under `slot`.
    ///
    /// Returns the number of compressed bytes consumed in the pool.
    pub fn store(&mut self, slot: SwapSlot, page: &[u8]) -> Result<usize, ZswapError> {
        if page.len() != PAGE_SIZE {
            return Err(ZswapError::InvalidPageSize);
        }

        let checksum = crc32c(page);
        let encoding = match detect_same_fill(page) {
            Some(0) => PageEncoding::Zero,
            Some(b) => PageEncoding::SameFill(b),
            None => {
                let compressed = lz_compress(page);
                if compressed.len() < page.len() {
                    PageEncoding::Compressed(compressed)
                } else {
                    PageEncoding::Raw(page.to_vec())
                }
            }
        };

        let new_bytes = encoding.stored_bytes();
        let old_bytes = self
            .entries
            .get(&slot)
            .map_or(0, |e| e.encoding.stored_bytes());

        let projected = self
            .used_pool_bytes
            .saturating_sub(old_bytes)
            .saturating_add(new_bytes);
        if projected > self.max_pool_bytes {
            return Err(ZswapError::PoolFull);
        }

        if self.entries.contains_key(&slot) {
            self.remove(slot)?;
        }

        match &encoding {
            PageEncoding::Zero => self.zero_pages += 1,
            PageEncoding::SameFill(_) => self.same_fill_pages += 1,
            PageEncoding::Compressed(_) => self.compressed_pages += 1,
            PageEncoding::Raw(_) => self.raw_pages += 1,
        }

        self.used_pool_bytes += new_bytes;
        self.total_uncompressed_bytes += PAGE_SIZE as u64;
        self.total_compressed_bytes += new_bytes as u64;
        self.entries.insert(
            slot,
            ZswapEntry {
                encoding,
                crc32: checksum,
            },
        );

        Ok(new_bytes)
    }

    /// Decompress the page stored at `slot` into `out` and verify its CRC32C.
    pub fn load(&self, slot: SwapSlot, out: &mut [u8]) -> Result<(), ZswapError> {
        if out.len() != PAGE_SIZE {
            return Err(ZswapError::InvalidPageSize);
        }
        let entry = self.entries.get(&slot).ok_or(ZswapError::SlotNotFound)?;

        match &entry.encoding {
            PageEncoding::Zero => out.fill(0),
            PageEncoding::SameFill(b) => out.fill(*b),
            PageEncoding::Compressed(buf) => {
                let written = lz_decompress(buf, out)?;
                if written != PAGE_SIZE {
                    return Err(ZswapError::CorruptedStream);
                }
            }
            PageEncoding::Raw(buf) => {
                if buf.len() != PAGE_SIZE {
                    return Err(ZswapError::CorruptedStream);
                }
                out.copy_from_slice(buf);
            }
        }

        if crc32c(out) != entry.crc32 {
            return Err(ZswapError::CorruptedStream);
        }
        Ok(())
    }

    /// Remove a stored page from the pool and reclaim its memory budget.
    pub fn remove(&mut self, slot: SwapSlot) -> Result<(), ZswapError> {
        let entry = self.entries.remove(&slot).ok_or(ZswapError::SlotNotFound)?;
        let bytes = entry.encoding.stored_bytes();
        self.used_pool_bytes = self.used_pool_bytes.saturating_sub(bytes);
        match entry.encoding {
            PageEncoding::Zero => self.zero_pages = self.zero_pages.saturating_sub(1),
            PageEncoding::SameFill(_) => {
                self.same_fill_pages = self.same_fill_pages.saturating_sub(1)
            }
            PageEncoding::Compressed(_) => {
                self.compressed_pages = self.compressed_pages.saturating_sub(1)
            }
            PageEncoding::Raw(_) => self.raw_pages = self.raw_pages.saturating_sub(1),
        }
        Ok(())
    }

    /// Compression ratio in permille (e.g. 2500 = 2.50x).
    pub fn compression_ratio_permille(&self) -> u32 {
        if self.total_compressed_bytes == 0 {
            return if self.total_uncompressed_bytes > 0 {
                u32::MAX
            } else {
                1000
            };
        }
        ((self.total_uncompressed_bytes * 1000) / self.total_compressed_bytes) as u32
    }

    pub fn stats(&self) -> ZswapPoolStats {
        ZswapPoolStats {
            max_pool_bytes: self.max_pool_bytes,
            used_pool_bytes: self.used_pool_bytes,
            stored_pages: self.entries.len(),
            zero_pages: self.zero_pages,
            same_fill_pages: self.same_fill_pages,
            compressed_pages: self.compressed_pages,
            raw_pages: self.raw_pages,
            total_uncompressed_bytes: self.total_uncompressed_bytes,
            total_compressed_bytes: self.total_compressed_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// Active / Inactive Two-List LRU Page-Reclaim Engine
// ---------------------------------------------------------------------------

/// Unique identifier for a virtual page belonging to a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageId {
    pub pid: u32,
    pub vaddr: u64,
}

/// Residency state of a tracked page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    Active,
    Inactive,
    Swapped(SwapSlot),
}

/// Metadata tracked per page by [`PageReclaimEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageMeta {
    pub id: PageId,
    pub state: PageState,
    pub referenced: bool,
    pub dirty: bool,
    pub pinned: bool,
    pub access_count: u32,
}

/// Summary of a reclaim pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReclaimResult {
    pub scanned: usize,
    pub reclaimed: usize,
    pub promoted: usize,
    pub skipped_pinned: usize,
    pub failed: usize,
}

/// Snapshot of [`PageReclaimEngine`] statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReclaimStats {
    pub active_pages: usize,
    pub inactive_pages: usize,
    pub swapped_pages: usize,
    pub total_reclaimed: u64,
    pub total_faulted_in: u64,
}

/// Two-list (Active / Inactive) LRU page-reclaim engine backed by [`ZswapPool`].
pub struct PageReclaimEngine {
    pages: BTreeMap<PageId, PageMeta>,
    active_lru: VecDeque<PageId>,
    inactive_lru: VecDeque<PageId>,
    pub pool: ZswapPool,
    pub slots: SwapSlotAllocator,
    total_reclaimed: u64,
    total_faulted_in: u64,
}

impl PageReclaimEngine {
    /// Create a new reclaim engine with the given swap-slot capacity and pool budget.
    pub fn new(swap_slots: usize, max_pool_bytes: usize) -> Self {
        Self {
            pages: BTreeMap::new(),
            active_lru: VecDeque::new(),
            inactive_lru: VecDeque::new(),
            pool: ZswapPool::new(max_pool_bytes),
            slots: SwapSlotAllocator::new(swap_slots),
            total_reclaimed: 0,
            total_faulted_in: 0,
        }
    }

    /// Start tracking a resident page (placed at the tail of the Inactive list).
    pub fn track_page(&mut self, id: PageId, dirty: bool, pinned: bool) {
        if let Some(meta) = self.pages.get_mut(&id) {
            meta.dirty |= dirty;
            meta.pinned = pinned;
            return;
        }
        let meta = PageMeta {
            id,
            state: PageState::Inactive,
            referenced: false,
            dirty,
            pinned,
            access_count: 1,
        };
        self.pages.insert(id, meta);
        self.inactive_lru.push_back(id);
    }

    /// Record a page access; promotes Inactive pages with a prior reference to Active.
    pub fn mark_accessed(&mut self, id: PageId) -> Result<(), ZswapError> {
        let meta = self.pages.get_mut(&id).ok_or(ZswapError::PageNotFound)?;
        meta.access_count = meta.access_count.saturating_add(1);
        if meta.state == PageState::Inactive && meta.referenced {
            meta.state = PageState::Active;
            meta.referenced = false;
            Self::remove_from_queue(&mut self.inactive_lru, id);
            self.active_lru.push_back(id);
        } else {
            meta.referenced = true;
        }
        Ok(())
    }

    /// Pin or unpin a tracked page.
    pub fn set_pinned(&mut self, id: PageId, pinned: bool) -> Result<(), ZswapError> {
        let meta = self.pages.get_mut(&id).ok_or(ZswapError::PageNotFound)?;
        meta.pinned = pinned;
        Ok(())
    }

    /// Stop tracking a page and free any associated swap slot / pool entry.
    pub fn untrack_page(&mut self, id: PageId) -> Option<PageMeta> {
        let meta = self.pages.remove(&id)?;
        match meta.state {
            PageState::Active => Self::remove_from_queue(&mut self.active_lru, id),
            PageState::Inactive => Self::remove_from_queue(&mut self.inactive_lru, id),
            PageState::Swapped(slot) => {
                let _ = self.pool.remove(slot);
                let _ = self.slots.free(slot);
            }
        }
        Some(meta)
    }

    /// Demote cold pages from Active to Inactive to maintain `target_inactive_pct`.
    pub fn balance_lists(&mut self, target_inactive_pct: usize) {
        let resident = self.active_lru.len() + self.inactive_lru.len();
        if resident == 0 {
            return;
        }
        let target_inactive = (resident * target_inactive_pct.min(100)) / 100;
        let mut budget = self.active_lru.len();

        while self.inactive_lru.len() < target_inactive && budget > 0 {
            budget -= 1;
            let Some(id) = self.active_lru.pop_front() else {
                break;
            };
            if let Some(meta) = self.pages.get_mut(&id) {
                if meta.state != PageState::Active {
                    continue;
                }
                if meta.referenced {
                    meta.referenced = false;
                    self.active_lru.push_back(id);
                } else {
                    meta.state = PageState::Inactive;
                    self.inactive_lru.push_back(id);
                }
            }
        }
    }

    /// Reclaim up to `count` pages from the Inactive list into [`ZswapPool`].
    ///
    /// Uses a single heap-allocated 4 KiB scratch buffer so the stack frame stays small.
    pub fn reclaim_pages(
        &mut self,
        count: usize,
        read_page: &mut dyn FnMut(PageId, &mut [u8]) -> bool,
    ) -> ReclaimResult {
        let mut result = ReclaimResult::default();
        if count == 0 {
            return result;
        }

        // Ensure Inactive list has candidates if Active is populated.
        if self.inactive_lru.is_empty() && !self.active_lru.is_empty() {
            self.balance_lists(50);
        }

        let mut scratch = vec![0u8; PAGE_SIZE];
        let max_scan = self.inactive_lru.len().saturating_mul(2).max(count);

        while result.reclaimed < count && result.scanned < max_scan {
            let Some(id) = self.inactive_lru.pop_front() else {
                break;
            };
            result.scanned += 1;

            let Some(meta) = self.pages.get_mut(&id) else {
                continue;
            };
            if meta.state != PageState::Inactive {
                continue;
            }
            if meta.pinned {
                result.skipped_pinned += 1;
                self.inactive_lru.push_back(id);
                continue;
            }
            if meta.referenced {
                meta.referenced = false;
                meta.state = PageState::Active;
                self.active_lru.push_back(id);
                result.promoted += 1;
                continue;
            }

            if !read_page(id, &mut scratch) {
                result.failed += 1;
                self.inactive_lru.push_back(id);
                continue;
            }

            let Ok(slot) = self.slots.alloc() else {
                result.failed += 1;
                self.inactive_lru.push_back(id);
                break;
            };

            if self.pool.store(slot, &scratch).is_err() {
                let _ = self.slots.free(slot);
                result.failed += 1;
                self.inactive_lru.push_back(id);
                break;
            }

            meta.state = PageState::Swapped(slot);
            meta.dirty = false;
            result.reclaimed += 1;
            self.total_reclaimed += 1;
        }

        result
    }

    /// Demand-page a swapped-out page back into `out_page` and move it to Active.
    pub fn fault_in(&mut self, id: PageId, out_page: &mut [u8]) -> Result<(), ZswapError> {
        if out_page.len() != PAGE_SIZE {
            return Err(ZswapError::InvalidPageSize);
        }
        let meta = self.pages.get_mut(&id).ok_or(ZswapError::PageNotFound)?;
        let slot = match meta.state {
            PageState::Swapped(s) => s,
            PageState::Active | PageState::Inactive => return Ok(()),
        };

        self.pool.load(slot, out_page)?;
        let _ = self.pool.remove(slot);
        let _ = self.slots.free(slot);

        meta.state = PageState::Active;
        meta.referenced = true;
        meta.access_count = meta.access_count.saturating_add(1);
        self.active_lru.push_back(id);
        self.total_faulted_in += 1;
        Ok(())
    }

    pub fn get_meta(&self, id: PageId) -> Option<PageMeta> {
        self.pages.get(&id).copied()
    }

    pub fn stats(&self) -> ReclaimStats {
        let swapped = self
            .pages
            .values()
            .filter(|m| matches!(m.state, PageState::Swapped(_)))
            .count();
        ReclaimStats {
            active_pages: self.active_lru.len(),
            inactive_pages: self.inactive_lru.len(),
            swapped_pages: swapped,
            total_reclaimed: self.total_reclaimed,
            total_faulted_in: self.total_faulted_in,
        }
    }

    fn remove_from_queue(queue: &mut VecDeque<PageId>, target: PageId) {
        if let Some(pos) = queue.iter().position(|&x| x == target) {
            queue.remove(pos);
        }
    }
}

// ---------------------------------------------------------------------------
// OOM Score Evaluator
// ---------------------------------------------------------------------------

/// Process memory profile used by [`OomEvaluator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OomCandidate {
    pub pid: u32,
    pub rss_pages: usize,
    pub swap_pages: usize,
    pub oom_score_adj: i16,
    pub is_kernel_thread: bool,
    pub is_init: bool,
    pub age_ticks: u64,
}

/// Selected OOM victim and its computed badness score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OomVictim {
    pub pid: u32,
    pub score: u32,
    pub total_pages: usize,
}

/// Linux-style OOM badness evaluator.
pub struct OomEvaluator;

impl OomEvaluator {
    /// Compute the OOM badness score (`1..=1000`) for `candidate`.
    ///
    /// Returns `None` if the process is immune (init, kernel thread, or
    /// `oom_score_adj <= OOM_SCORE_ADJ_MIN`).
    pub fn compute_score(candidate: &OomCandidate, total_ram_pages: usize) -> Option<u32> {
        if candidate.pid == 0
            || candidate.is_init
            || candidate.is_kernel_thread
            || candidate.oom_score_adj <= OOM_SCORE_ADJ_MIN
        {
            return None;
        }

        let total_pages = candidate.rss_pages.saturating_add(candidate.swap_pages);
        let denom = total_ram_pages.max(1);
        let base = ((total_pages as u64).saturating_mul(1000) / (denom as u64)).min(1000) as i32;
        let adj = (candidate.oom_score_adj as i32)
            .clamp(OOM_SCORE_ADJ_MIN as i32, OOM_SCORE_ADJ_MAX as i32);
        let score = (base + adj).clamp(1, 1000) as u32;
        Some(score)
    }

    /// Select the highest-scoring killable process from `candidates`.
    pub fn select_victim(
        candidates: &[OomCandidate],
        total_ram_pages: usize,
    ) -> Result<OomVictim, ZswapError> {
        let mut best: Option<(OomVictim, u64)> = None;

        for c in candidates {
            let Some(score) = Self::compute_score(c, total_ram_pages) else {
                continue;
            };
            let total_pages = c.rss_pages.saturating_add(c.swap_pages);
            let candidate_victim = OomVictim {
                pid: c.pid,
                score,
                total_pages,
            };

            let replace = match best {
                None => true,
                Some((prev, prev_age)) => {
                    score > prev.score
                        || (score == prev.score && total_pages > prev.total_pages)
                        || (score == prev.score
                            && total_pages == prev.total_pages
                            && c.age_ticks < prev_age)
                }
            };

            if replace {
                best = Some((candidate_victim, c.age_ticks));
            }
        }

        best.map(|(v, _)| v).ok_or(ZswapError::NoKillableProcess)
    }
}

// ---------------------------------------------------------------------------
// Demand Paging Controller
// ---------------------------------------------------------------------------

/// Outcome of handling a page fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultOutcome {
    SwappedIn,
    ZeroPageAllocated,
    AlreadyResident,
}

/// Action taken by [`DemandPager::reclaim_under_pressure`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureAction {
    None,
    Reclaimed(usize),
    OomKillRecommended(OomVictim),
}

/// Unified demand-paging, zswap, and memory-pressure controller.
pub struct DemandPager {
    pub reclaim: PageReclaimEngine,
    pub total_ram_pages: usize,
}

impl DemandPager {
    pub fn new(total_ram_pages: usize, swap_slots: usize, max_pool_bytes: usize) -> Self {
        Self {
            reclaim: PageReclaimEngine::new(swap_slots, max_pool_bytes),
            total_ram_pages,
        }
    }

    /// Handle a page fault for `id`.
    ///
    /// - If swapped out in zswap, decompresses into `out_page` and promotes to Active.
    /// - If untracked, allocates a demand-zero page and tracks it.
    /// - If already resident, updates access/dirty state.
    pub fn handle_page_fault(
        &mut self,
        id: PageId,
        write: bool,
        out_page: &mut [u8],
    ) -> Result<FaultOutcome, ZswapError> {
        if out_page.len() != PAGE_SIZE {
            return Err(ZswapError::InvalidPageSize);
        }

        match self.reclaim.get_meta(id) {
            Some(meta) => match meta.state {
                PageState::Swapped(_) => {
                    self.reclaim.fault_in(id, out_page)?;
                    if write {
                        if let Some(m) = self.reclaim.pages.get_mut(&id) {
                            m.dirty = true;
                        }
                    }
                    Ok(FaultOutcome::SwappedIn)
                }
                PageState::Active | PageState::Inactive => {
                    let _ = self.reclaim.mark_accessed(id);
                    if write {
                        if let Some(m) = self.reclaim.pages.get_mut(&id) {
                            m.dirty = true;
                        }
                    }
                    Ok(FaultOutcome::AlreadyResident)
                }
            },
            None => {
                out_page.fill(0);
                self.reclaim.track_page(id, write, false);
                Ok(FaultOutcome::ZeroPageAllocated)
            }
        }
    }

    /// Respond to memory pressure based on `free_frames` and watermarks.
    pub fn reclaim_under_pressure(
        &mut self,
        free_frames: usize,
        low_watermark: usize,
        min_watermark: usize,
        candidates: &[OomCandidate],
        read_page: &mut dyn FnMut(PageId, &mut [u8]) -> bool,
    ) -> Result<PressureAction, ZswapError> {
        if free_frames >= low_watermark {
            return Ok(PressureAction::None);
        }

        let deficit = low_watermark.saturating_sub(free_frames).max(1);
        let res = self.reclaim.reclaim_pages(deficit, read_page);
        if free_frames + res.reclaimed >= min_watermark || res.reclaimed > 0 {
            return Ok(PressureAction::Reclaimed(res.reclaimed));
        }

        let victim = OomEvaluator::select_victim(candidates, self.total_ram_pages)?;
        Ok(PressureAction::OomKillRecommended(victim))
    }
}

/// Global kernel demand-paging / zswap controller.
pub static ZSWAP_PAGER: SpinLock<Option<DemandPager>> = SpinLock::new(None);

/// Initialize the global [`ZSWAP_PAGER`] instance.
pub fn init(total_ram_pages: usize) {
    let mut guard = ZSWAP_PAGER.lock();
    *guard = Some(DemandPager::new(
        total_ram_pages,
        DEFAULT_SWAP_SLOTS,
        DEFAULT_POOL_MAX_BYTES,
    ));
}

// ---------------------------------------------------------------------------
// Self-Test
// ---------------------------------------------------------------------------

/// Exercise LZ compression, swap-slot allocation, zswap pool, LRU reclaim,
/// demand paging, and OOM scoring.
pub fn self_test() -> Result<(), &'static str> {
    serial_println!("[zswap] running self_test...");

    // 1. LZ compressor / decompressor round-trip on multiple page patterns.
    let mut page = vec![0u8; PAGE_SIZE];
    let mut restored = vec![0u8; PAGE_SIZE];

    // 1a. Zero page detection.
    if detect_same_fill(&page) != Some(0) {
        return Err("zswap: zero page detection failed");
    }

    // 1b. Repetitive structured data (highly compressible).
    for (i, b) in page.iter_mut().enumerate() {
        *b = ((i / 16) & 0xFF) as u8;
    }
    let comp = lz_compress(&page);
    if comp.len() >= PAGE_SIZE / 2 {
        return Err("zswap: repetitive page did not compress well");
    }
    let written = lz_decompress(&comp, &mut restored).map_err(|_| "zswap: decompress failed")?;
    if written != PAGE_SIZE || restored != page {
        return Err("zswap: repetitive page round-trip mismatch");
    }

    // 1c. Pseudo-random incompressible data (raw fallback in pool).
    let mut rng: u32 = 0x1234_5678;
    for b in page.iter_mut() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        *b = (rng & 0xFF) as u8;
    }
    let comp_rand = lz_compress(&page);
    restored.fill(0);
    let written_rand =
        lz_decompress(&comp_rand, &mut restored).map_err(|_| "zswap: random decompress failed")?;
    if written_rand != PAGE_SIZE || restored != page {
        return Err("zswap: random page round-trip mismatch");
    }

    // 2. SwapSlotAllocator alloc / free / reuse.
    let mut alloc = SwapSlotAllocator::new(4);
    let s0 = alloc.alloc().map_err(|_| "zswap: slot alloc 0 failed")?;
    let s1 = alloc.alloc().map_err(|_| "zswap: slot alloc 1 failed")?;
    let _s2 = alloc.alloc().map_err(|_| "zswap: slot alloc 2 failed")?;
    let _s3 = alloc.alloc().map_err(|_| "zswap: slot alloc 3 failed")?;
    if alloc.alloc() != Err(ZswapError::SwapFull) {
        return Err("zswap: expected SwapFull");
    }
    alloc.free(s1).map_err(|_| "zswap: slot free failed")?;
    let s1_reused = alloc.alloc().map_err(|_| "zswap: slot reuse failed")?;
    if s1_reused != s1 || s0.0 != 0 {
        return Err("zswap: unexpected slot reuse index");
    }

    // 3. DemandPager + PageReclaimEngine + ZswapPool end-to-end.
    let mut pager = DemandPager::new(1024, 64, 256 * 1024);
    let p1 = PageId {
        pid: 10,
        vaddr: 0x1000,
    };
    let p2 = PageId {
        pid: 10,
        vaddr: 0x2000,
    };
    let p3 = PageId {
        pid: 10,
        vaddr: 0x3000,
    };

    // Demand-zero fault for p1.
    let outcome = pager
        .handle_page_fault(p1, true, &mut restored)
        .map_err(|_| "zswap: demand-zero fault failed")?;
    if outcome != FaultOutcome::ZeroPageAllocated {
        return Err("zswap: expected ZeroPageAllocated");
    }

    // Track p2 (pinned) and p3 (reclaimable).
    pager.reclaim.track_page(p2, false, true);
    pager.reclaim.track_page(p3, true, false);

    // Reclaim 2 pages: p1 and p3 should be swapped out, p2 skipped (pinned).
    let reclaim_res = pager.reclaim.reclaim_pages(2, &mut |id, buf| {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((id.vaddr as usize + i) & 0xFF) as u8;
        }
        true
    });
    if reclaim_res.reclaimed != 2 || reclaim_res.skipped_pinned == 0 {
        return Err("zswap: unexpected reclaim counts");
    }

    // Fault p3 back in and verify contents.
    restored.fill(0);
    let fault_res = pager
        .handle_page_fault(p3, false, &mut restored)
        .map_err(|_| "zswap: fault_in failed")?;
    if fault_res != FaultOutcome::SwappedIn {
        return Err("zswap: expected SwappedIn");
    }
    for (i, &b) in restored.iter().enumerate() {
        if b != ((p3.vaddr as usize + i) & 0xFF) as u8 {
            return Err("zswap: faulted-in page data mismatch");
        }
    }

    // 4. OOM Evaluator.
    let candidates = [
        OomCandidate {
            pid: 1,
            rss_pages: 500,
            swap_pages: 0,
            oom_score_adj: 0,
            is_kernel_thread: false,
            is_init: true,
            age_ticks: 10,
        },
        OomCandidate {
            pid: 2,
            rss_pages: 900,
            swap_pages: 0,
            oom_score_adj: OOM_SCORE_ADJ_MIN,
            is_kernel_thread: false,
            is_init: false,
            age_ticks: 20,
        },
        OomCandidate {
            pid: 42,
            rss_pages: 400,
            swap_pages: 100,
            oom_score_adj: 50,
            is_kernel_thread: false,
            is_init: false,
            age_ticks: 100,
        },
        OomCandidate {
            pid: 99,
            rss_pages: 200,
            swap_pages: 0,
            oom_score_adj: 0,
            is_kernel_thread: false,
            is_init: false,
            age_ticks: 200,
        },
    ];
    let victim = OomEvaluator::select_victim(&candidates, 1024)
        .map_err(|_| "zswap: OOM victim selection failed")?;
    if victim.pid != 42 {
        return Err("zswap: wrong OOM victim selected");
    }

    serial_println!(
        "[zswap] self_test passed (ratio={} permille, victim_pid={})",
        pager.reclaim.pool.compression_ratio_permille(),
        victim.pid
    );
    Ok(())
}
