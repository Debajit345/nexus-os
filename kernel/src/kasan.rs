//! Kernel Address Sanitizer (KASAN 8:1 shadow memory) & Deterministic Syscall Fuzzer.
//!
//! Provides:
//! - 8:1 shadow memory redzone tracker (`0x00` accessible, `1..=7` partial granule,
//!   `0xF1..=0xFF` poisoned redzones / freed / UAF / scope / generic)
//! - Quarantine-backed KASAN arena allocator with left/right redzones
//! - UAF, OOB (left/right/partial-granule), double-free, invalid-free, and wild-pointer detection
//! - Deterministic multi-personality syscall fuzzer harness across Native, POSIX, NT, and Darwin
//! - `self_test()` verification suite
#![allow(dead_code)]

use core::sync::atomic::{AtomicBool, Ordering};

use nexus_abi::{AbiError, Personality, SyscallNumber};

use crate::mm::SpinLock;
use crate::{serial_println, syscall};

// ---------------------------------------------------------------------------
// KASAN 8:1 Shadow Memory Constants
// ---------------------------------------------------------------------------

pub const SHADOW_SCALE: usize = 8;
pub const SHADOW_SHIFT: usize = 3;

/// 64 KiB tracked kernel sanitizer arena in `.bss` (8 KiB shadow map).
pub const ARENA_SIZE: usize = 64 * 1024;
pub const SHADOW_SIZE: usize = ARENA_SIZE / SHADOW_SCALE;
pub const DEFAULT_REDZONE_SIZE: usize = 16;
pub const MAX_ALLOC_RECORDS: usize = 128;
pub const QUARANTINE_CAP: usize = 32;

/// Shadow byte encodings (standard KASAN 8:1 convention).
pub const SHADOW_ACCESSIBLE: u8 = 0x00;
pub const KASAN_STACK_LEFT: u8 = 0xF1;
pub const KASAN_STACK_MID: u8 = 0xF2;
pub const KASAN_STACK_RIGHT: u8 = 0xF3;
pub const KASAN_USE_AFTER_SCOPE: u8 = 0xF8;
pub const KASAN_GLOBAL_REDZONE: u8 = 0xF9;
pub const KASAN_HEAP_REDZONE: u8 = 0xFA;
pub const KASAN_HEAP_FREE: u8 = 0xFB;
pub const KASAN_SLAB_REDZONE: u8 = 0xFC;
pub const KASAN_SLAB_FREE: u8 = 0xFD;
pub const KASAN_POISON_GENERIC: u8 = 0xFF;

// ---------------------------------------------------------------------------
// Violation & Diagnostic Types
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ViolationKind {
    OutOfBoundsLeft,
    OutOfBoundsRight,
    PartialGranuleOob,
    UseAfterFree,
    UseAfterScope,
    DoubleFree,
    InvalidFree,
    WildPointer,
    PoisonedAccess,
    OutOfMemory,
    InvalidSize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct KasanReport {
    pub kind: ViolationKind,
    pub addr: usize,
    pub size: usize,
    pub is_write: bool,
    pub shadow_val: u8,
    pub alloc_base: usize,
    pub alloc_size: usize,
    pub alloc_seq: u64,
    pub free_seq: u64,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct KasanStats {
    pub checks_passed: u64,
    pub violations_detected: u64,
    pub uaf_detected: u64,
    pub oob_detected: u64,
    pub double_free_detected: u64,
    pub invalid_free_detected: u64,
    pub wild_ptr_detected: u64,
    pub allocs: u64,
    pub frees: u64,
    pub quarantine_evictions: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum AllocState {
    Empty,
    Active,
    Quarantined,
    Freed,
}

#[derive(Copy, Clone, Debug)]
struct AllocRecord {
    state: AllocState,
    block_start: usize,
    block_len: usize,
    user_addr: usize,
    user_size: usize,
    alloc_seq: u64,
    free_seq: u64,
}

impl AllocRecord {
    const fn empty() -> Self {
        Self {
            state: AllocState::Empty,
            block_start: 0,
            block_len: 0,
            user_addr: 0,
            user_size: 0,
            alloc_seq: 0,
            free_seq: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// KASAN Tracker State (stored in .bss via SpinLock, zero large stack frames)
// ---------------------------------------------------------------------------

pub struct KasanTracker {
    arena: [u8; ARENA_SIZE],
    shadow: [u8; SHADOW_SIZE],
    records: [AllocRecord; MAX_ALLOC_RECORDS],
    quarantine: [usize; QUARANTINE_CAP],
    quarantine_head: usize,
    quarantine_len: usize,
    bump_offset: usize,
    seq_counter: u64,
    stats: KasanStats,
    initialized: bool,
}

impl KasanTracker {
    const fn new() -> Self {
        Self {
            arena: [0; ARENA_SIZE],
            shadow: [KASAN_POISON_GENERIC; SHADOW_SIZE],
            records: [AllocRecord::empty(); MAX_ALLOC_RECORDS],
            quarantine: [0; QUARANTINE_CAP],
            quarantine_head: 0,
            quarantine_len: 0,
            bump_offset: 0,
            seq_counter: 0,
            stats: KasanStats {
                checks_passed: 0,
                violations_detected: 0,
                uaf_detected: 0,
                oob_detected: 0,
                double_free_detected: 0,
                invalid_free_detected: 0,
                wild_ptr_detected: 0,
                allocs: 0,
                frees: 0,
                quarantine_evictions: 0,
            },
            initialized: false,
        }
    }

    #[inline]
    fn base_addr(&self) -> usize {
        self.arena.as_ptr() as usize
    }

    #[inline]
    fn end_addr(&self) -> usize {
        self.base_addr() + ARENA_SIZE
    }

    fn reset(&mut self) {
        self.shadow.fill(KASAN_POISON_GENERIC);
        for rec in &mut self.records {
            *rec = AllocRecord::empty();
        }
        self.quarantine.fill(0);
        self.quarantine_head = 0;
        self.quarantine_len = 0;
        self.bump_offset = 0;
        self.seq_counter = 0;
        self.stats = KasanStats::default();
        self.initialized = true;
    }

    #[inline]
    fn ensure_init(&mut self) {
        if !self.initialized {
            self.reset();
        }
    }

    #[inline]
    fn addr_to_offset(&self, addr: usize) -> Option<usize> {
        let base = self.base_addr();
        if addr >= base && addr < base + ARENA_SIZE {
            Some(addr - base)
        } else {
            None
        }
    }

    /// Poison `[addr, addr + size)` with `tag` in the 8:1 shadow map.
    pub fn poison_range(&mut self, addr: usize, size: usize, tag: u8) -> Result<(), ViolationKind> {
        self.ensure_init();
        if size == 0 {
            return Ok(());
        }
        let start_off = self
            .addr_to_offset(addr)
            .ok_or(ViolationKind::WildPointer)?;
        let end_off = start_off
            .checked_add(size)
            .filter(|&e| e <= ARENA_SIZE)
            .ok_or(ViolationKind::WildPointer)?;

        let first_granule = start_off >> SHADOW_SHIFT;
        let last_granule = (end_off - 1) >> SHADOW_SHIFT;

        for (g, slot) in self
            .shadow
            .iter_mut()
            .enumerate()
            .take(last_granule + 1)
            .skip(first_granule)
        {
            let g_start = g << SHADOW_SHIFT;
            let g_end = g_start + SHADOW_SCALE;
            if start_off <= g_start && end_off >= g_end {
                *slot = tag;
            } else if start_off > g_start && end_off >= g_end {
                // Prefix of granule remains accessible if it was previously accessible.
                let prefix = (start_off - g_start) as u8;
                let cur = *slot;
                if cur == SHADOW_ACCESSIBLE || (cur < 8 && cur > prefix) {
                    *slot = prefix;
                } else if cur >= 0x80 {
                    *slot = tag;
                }
            } else {
                *slot = tag;
            }
        }
        Ok(())
    }

    /// Unpoison `[addr, addr + size)` in the 8:1 shadow map, encoding partial
    /// trailing granules (`1..=7`) accurately.
    pub fn unpoison_range(&mut self, addr: usize, size: usize) -> Result<(), ViolationKind> {
        self.ensure_init();
        if size == 0 {
            return Ok(());
        }
        let start_off = self
            .addr_to_offset(addr)
            .ok_or(ViolationKind::WildPointer)?;
        let end_off = start_off
            .checked_add(size)
            .filter(|&e| e <= ARENA_SIZE)
            .ok_or(ViolationKind::WildPointer)?;

        let mut cur = start_off;
        while cur < end_off {
            let g = cur >> SHADOW_SHIFT;
            let g_end = (g + 1) << SHADOW_SHIFT;
            let chunk_end = end_off.min(g_end);
            let g_start = g << SHADOW_SHIFT;

            if cur == g_start && chunk_end == g_end {
                self.shadow[g] = SHADOW_ACCESSIBLE;
            } else if cur == g_start {
                let accessible_bytes = (chunk_end - g_start) as u8;
                self.shadow[g] = accessible_bytes;
            } else {
                // Unaligned start within granule: mark full granule accessible
                // only if it covers to the end, otherwise partial.
                let accessible_bytes = (chunk_end - g_start) as u8;
                self.shadow[g] = if accessible_bytes >= SHADOW_SCALE as u8 {
                    SHADOW_ACCESSIBLE
                } else {
                    accessible_bytes
                };
            }
            cur = chunk_end;
        }
        Ok(())
    }

    fn find_nearest_record(&self, addr: usize) -> Option<&AllocRecord> {
        let mut best: Option<&AllocRecord> = None;
        let mut best_dist = usize::MAX;

        for rec in &self.records {
            if rec.state == AllocState::Empty {
                continue;
            }
            let block_end = rec.block_start + rec.block_len;
            if addr >= rec.block_start && addr < block_end {
                return Some(rec);
            }
            let dist = if addr < rec.block_start {
                rec.block_start - addr
            } else {
                addr - block_end
            };
            if dist < best_dist {
                best_dist = dist;
                best = Some(rec);
            }
        }
        best
    }

    fn classify_violation(
        &self,
        addr: usize,
        shadow_val: u8,
    ) -> (ViolationKind, usize, usize, u64, u64) {
        let rec = self.find_nearest_record(addr);
        let (alloc_base, alloc_size, alloc_seq, free_seq) = match rec {
            Some(r) => (r.user_addr, r.user_size, r.alloc_seq, r.free_seq),
            None => (0, 0, 0, 0),
        };

        let kind = match shadow_val {
            KASAN_HEAP_FREE | KASAN_SLAB_FREE => ViolationKind::UseAfterFree,
            KASAN_USE_AFTER_SCOPE => ViolationKind::UseAfterScope,
            1..=7 => ViolationKind::PartialGranuleOob,
            KASAN_HEAP_REDZONE | KASAN_SLAB_REDZONE | KASAN_STACK_LEFT | KASAN_STACK_RIGHT => {
                if let Some(r) = rec {
                    if addr < r.user_addr {
                        ViolationKind::OutOfBoundsLeft
                    } else {
                        ViolationKind::OutOfBoundsRight
                    }
                } else {
                    ViolationKind::PoisonedAccess
                }
            }
            _ => ViolationKind::PoisonedAccess,
        };

        (kind, alloc_base, alloc_size, alloc_seq, free_seq)
    }

    fn record_violation(&mut self, kind: ViolationKind) {
        self.stats.violations_detected += 1;
        match kind {
            ViolationKind::UseAfterFree | ViolationKind::UseAfterScope => {
                self.stats.uaf_detected += 1;
            }
            ViolationKind::OutOfBoundsLeft
            | ViolationKind::OutOfBoundsRight
            | ViolationKind::PartialGranuleOob => {
                self.stats.oob_detected += 1;
            }
            ViolationKind::DoubleFree => {
                self.stats.double_free_detected += 1;
            }
            ViolationKind::InvalidFree => {
                self.stats.invalid_free_detected += 1;
            }
            ViolationKind::WildPointer => {
                self.stats.wild_ptr_detected += 1;
            }
            _ => {}
        }
    }

    /// Validate `[addr, addr + size)` against the 8:1 shadow map.
    pub fn check_access(
        &mut self,
        addr: usize,
        size: usize,
        is_write: bool,
    ) -> Result<(), KasanReport> {
        self.ensure_init();
        if size == 0 {
            self.stats.checks_passed += 1;
            return Ok(());
        }

        let start_off = match self.addr_to_offset(addr) {
            Some(off) => off,
            None => {
                self.record_violation(ViolationKind::WildPointer);
                return Err(KasanReport {
                    kind: ViolationKind::WildPointer,
                    addr,
                    size,
                    is_write,
                    shadow_val: KASAN_POISON_GENERIC,
                    alloc_base: 0,
                    alloc_size: 0,
                    alloc_seq: 0,
                    free_seq: 0,
                });
            }
        };

        let end_off = match start_off.checked_add(size) {
            Some(e) if e <= ARENA_SIZE => e,
            _ => {
                self.record_violation(ViolationKind::WildPointer);
                return Err(KasanReport {
                    kind: ViolationKind::WildPointer,
                    addr,
                    size,
                    is_write,
                    shadow_val: KASAN_POISON_GENERIC,
                    alloc_base: 0,
                    alloc_size: 0,
                    alloc_seq: 0,
                    free_seq: 0,
                });
            }
        };

        let base = self.base_addr();
        let mut cur = start_off;
        while cur < end_off {
            let g = cur >> SHADOW_SHIFT;
            let off_in_g = cur & (SHADOW_SCALE - 1);
            let s = self.shadow[g];

            if s == SHADOW_ACCESSIBLE {
                let next_g = (g + 1) << SHADOW_SHIFT;
                cur = end_off.min(next_g);
                continue;
            }

            if s < SHADOW_SCALE as u8 {
                let g_end = (g + 1) << SHADOW_SHIFT;
                let access_end_in_g = end_off.min(g_end) - (g << SHADOW_SHIFT);
                if access_end_in_g <= s as usize {
                    cur = end_off.min(g_end);
                    continue;
                }
                let fault_off = (g << SHADOW_SHIFT) + (s as usize).max(off_in_g);
                let fault_addr = base + fault_off;
                let (kind, alloc_base, alloc_size, alloc_seq, free_seq) =
                    self.classify_violation(fault_addr, s);
                self.record_violation(kind);
                return Err(KasanReport {
                    kind,
                    addr: fault_addr,
                    size,
                    is_write,
                    shadow_val: s,
                    alloc_base,
                    alloc_size,
                    alloc_seq,
                    free_seq,
                });
            }

            // Poisoned granule (s >= 0x80)
            let fault_addr = base + cur;
            let (kind, alloc_base, alloc_size, alloc_seq, free_seq) =
                self.classify_violation(fault_addr, s);
            self.record_violation(kind);
            return Err(KasanReport {
                kind,
                addr: fault_addr,
                size,
                is_write,
                shadow_val: s,
                alloc_base,
                alloc_size,
                alloc_seq,
                free_seq,
            });
        }

        self.stats.checks_passed += 1;
        Ok(())
    }

    fn evict_quarantine_one(&mut self) {
        if self.quarantine_len == 0 {
            return;
        }
        let idx = self.quarantine[self.quarantine_head];
        self.quarantine_head = (self.quarantine_head + 1) % QUARANTINE_CAP;
        self.quarantine_len -= 1;
        if idx < MAX_ALLOC_RECORDS && self.records[idx].state == AllocState::Quarantined {
            self.records[idx].state = AllocState::Freed;
            self.stats.quarantine_evictions += 1;
        }
    }

    /// Allocate `size` bytes with `align` alignment and 16-byte left/right redzones.
    pub fn alloc(&mut self, size: usize, align: usize) -> Result<usize, ViolationKind> {
        self.ensure_init();
        if size == 0 {
            return Err(ViolationKind::InvalidSize);
        }
        let align = align.max(SHADOW_SCALE).next_power_of_two();
        let padded_size = (size + SHADOW_SCALE - 1) & !(SHADOW_SCALE - 1);
        let total_needed = DEFAULT_REDZONE_SIZE + padded_size + DEFAULT_REDZONE_SIZE;

        // Find a free record slot
        let mut slot_idx = None;
        for (i, rec) in self.records.iter().enumerate() {
            if rec.state == AllocState::Empty {
                slot_idx = Some(i);
                break;
            }
        }

        // Try bump allocation first
        let base = self.base_addr();
        let aligned_user_off = {
            let raw_user = self.bump_offset + DEFAULT_REDZONE_SIZE;
            (raw_user + align - 1) & !(align - 1)
        };
        let block_start_off = aligned_user_off - DEFAULT_REDZONE_SIZE;
        let block_end_off = aligned_user_off + padded_size + DEFAULT_REDZONE_SIZE;

        if let Some(idx) = slot_idx {
            if block_end_off <= ARENA_SIZE {
                self.bump_offset = block_end_off;
                let block_start = base + block_start_off;
                let user_addr = base + aligned_user_off;
                let right_rz = user_addr + padded_size;

                self.poison_range(block_start, DEFAULT_REDZONE_SIZE, KASAN_HEAP_REDZONE)?;
                self.unpoison_range(user_addr, size)?;
                if padded_size > size {
                    // Partial granule already encoded by unpoison_range; ensure trailing bytes stay poisoned.
                }
                self.poison_range(right_rz, DEFAULT_REDZONE_SIZE, KASAN_HEAP_REDZONE)?;

                // Zero user payload
                let user_off = aligned_user_off;
                self.arena[user_off..user_off + size].fill(0);

                self.seq_counter += 1;
                self.records[idx] = AllocRecord {
                    state: AllocState::Active,
                    block_start,
                    block_len: block_end_off - block_start_off,
                    user_addr,
                    user_size: size,
                    alloc_seq: self.seq_counter,
                    free_seq: 0,
                };
                self.stats.allocs += 1;
                return Ok(user_addr);
            }
        }

        // Fallback: reuse an evicted Freed record if large enough
        if self.quarantine_len == QUARANTINE_CAP {
            self.evict_quarantine_one();
        }
        for i in 0..MAX_ALLOC_RECORDS {
            let rec = self.records[i];
            if rec.state == AllocState::Freed && rec.block_len >= total_needed {
                let user_addr = rec.block_start + DEFAULT_REDZONE_SIZE;
                if user_addr & (align - 1) == 0 {
                    let right_rz = user_addr + padded_size;
                    self.poison_range(rec.block_start, DEFAULT_REDZONE_SIZE, KASAN_HEAP_REDZONE)?;
                    self.unpoison_range(user_addr, size)?;
                    self.poison_range(right_rz, DEFAULT_REDZONE_SIZE, KASAN_HEAP_REDZONE)?;

                    let user_off = user_addr - base;
                    self.arena[user_off..user_off + size].fill(0);

                    self.seq_counter += 1;
                    self.records[i].state = AllocState::Active;
                    self.records[i].user_addr = user_addr;
                    self.records[i].user_size = size;
                    self.records[i].alloc_seq = self.seq_counter;
                    self.records[i].free_seq = 0;
                    self.stats.allocs += 1;
                    return Ok(user_addr);
                }
            }
        }

        Err(ViolationKind::OutOfMemory)
    }

    /// Free a KASAN-tracked allocation, poisoning its payload with `KASAN_HEAP_FREE`
    /// and placing it into the FIFO quarantine.
    pub fn free(&mut self, addr: usize) -> Result<(), KasanReport> {
        self.ensure_init();
        let mut found_idx = None;
        for (i, rec) in self.records.iter().enumerate() {
            if rec.state != AllocState::Empty && rec.user_addr == addr {
                found_idx = Some(i);
                break;
            }
        }

        let idx = match found_idx {
            Some(i) => i,
            None => {
                self.record_violation(ViolationKind::InvalidFree);
                return Err(KasanReport {
                    kind: ViolationKind::InvalidFree,
                    addr,
                    size: 0,
                    is_write: true,
                    shadow_val: KASAN_POISON_GENERIC,
                    alloc_base: 0,
                    alloc_size: 0,
                    alloc_seq: 0,
                    free_seq: 0,
                });
            }
        };

        let rec = self.records[idx];
        if rec.state == AllocState::Quarantined || rec.state == AllocState::Freed {
            self.record_violation(ViolationKind::DoubleFree);
            return Err(KasanReport {
                kind: ViolationKind::DoubleFree,
                addr,
                size: rec.user_size,
                is_write: true,
                shadow_val: KASAN_HEAP_FREE,
                alloc_base: rec.user_addr,
                alloc_size: rec.user_size,
                alloc_seq: rec.alloc_seq,
                free_seq: rec.free_seq,
            });
        }

        let padded_size = (rec.user_size + SHADOW_SCALE - 1) & !(SHADOW_SCALE - 1);
        let _ = self.poison_range(rec.user_addr, padded_size, KASAN_HEAP_FREE);

        self.seq_counter += 1;
        self.records[idx].state = AllocState::Quarantined;
        self.records[idx].free_seq = self.seq_counter;
        self.stats.frees += 1;

        if self.quarantine_len == QUARANTINE_CAP {
            self.evict_quarantine_one();
        }
        let tail = (self.quarantine_head + self.quarantine_len) % QUARANTINE_CAP;
        self.quarantine[tail] = idx;
        self.quarantine_len += 1;

        Ok(())
    }

    pub fn read_u8(&mut self, addr: usize) -> Result<u8, KasanReport> {
        self.check_access(addr, 1, false)?;
        let off = addr - self.base_addr();
        Ok(self.arena[off])
    }

    pub fn write_u8(&mut self, addr: usize, val: u8) -> Result<(), KasanReport> {
        self.check_access(addr, 1, true)?;
        let off = addr - self.base_addr();
        self.arena[off] = val;
        Ok(())
    }

    pub fn write_bytes(&mut self, addr: usize, data: &[u8]) -> Result<(), KasanReport> {
        self.check_access(addr, data.len(), true)?;
        let off = addr - self.base_addr();
        self.arena[off..off + data.len()].copy_from_slice(data);
        Ok(())
    }

    pub fn read_bytes(&mut self, addr: usize, out: &mut [u8]) -> Result<(), KasanReport> {
        self.check_access(addr, out.len(), false)?;
        let off = addr - self.base_addr();
        out.copy_from_slice(&self.arena[off..off + out.len()]);
        Ok(())
    }
}

static KASAN: SpinLock<KasanTracker> = SpinLock::new(KasanTracker::new());
static KASAN_ENABLED: AtomicBool = AtomicBool::new(true);

pub fn init() {
    KASAN.lock().reset();
    KASAN_ENABLED.store(true, Ordering::Release);
}

pub fn is_enabled() -> bool {
    KASAN_ENABLED.load(Ordering::Acquire)
}

pub fn set_enabled(enabled: bool) {
    KASAN_ENABLED.store(enabled, Ordering::Release);
}

pub fn arena_bounds() -> (usize, usize) {
    let guard = KASAN.lock();
    (guard.base_addr(), guard.end_addr())
}

pub fn alloc(size: usize, align: usize) -> Result<usize, ViolationKind> {
    KASAN.lock().alloc(size, align)
}

pub fn free(addr: usize) -> Result<(), KasanReport> {
    KASAN.lock().free(addr)
}

pub fn poison_range(addr: usize, size: usize, tag: u8) -> Result<(), ViolationKind> {
    KASAN.lock().poison_range(addr, size, tag)
}

pub fn unpoison_range(addr: usize, size: usize) -> Result<(), ViolationKind> {
    KASAN.lock().unpoison_range(addr, size)
}

pub fn check_read(addr: usize, size: usize) -> Result<(), KasanReport> {
    if !is_enabled() {
        return Ok(());
    }
    KASAN.lock().check_access(addr, size, false)
}

pub fn check_write(addr: usize, size: usize) -> Result<(), KasanReport> {
    if !is_enabled() {
        return Ok(());
    }
    KASAN.lock().check_access(addr, size, true)
}

pub fn read_u8(addr: usize) -> Result<u8, KasanReport> {
    KASAN.lock().read_u8(addr)
}

pub fn write_u8(addr: usize, val: u8) -> Result<(), KasanReport> {
    KASAN.lock().write_u8(addr, val)
}

pub fn write_bytes(addr: usize, data: &[u8]) -> Result<(), KasanReport> {
    KASAN.lock().write_bytes(addr, data)
}

pub fn read_bytes(addr: usize, out: &mut [u8]) -> Result<(), KasanReport> {
    KASAN.lock().read_bytes(addr, out)
}

pub fn stats() -> KasanStats {
    KASAN.lock().stats
}

// ---------------------------------------------------------------------------
// Deterministic Syscall Fuzzer Harness (Native / POSIX / NT / Darwin)
// ---------------------------------------------------------------------------

/// Deterministic SplitMix64 PRNG for reproducible kernel fuzzing.
#[derive(Copy, Clone, Debug)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % (bound as u64)) as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FuzzConfig {
    pub seed: u64,
    pub iterations: usize,
    pub include_cross_personality: bool,
    pub include_invalid_syscalls: bool,
    pub include_kasan_poisoned_ptrs: bool,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            seed: 0x4E45_5855_535F_4B41,
            iterations: 128,
            include_cross_personality: true,
            include_invalid_syscalls: true,
            include_kasan_poisoned_ptrs: true,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PersonalityStats {
    pub dispatched: u64,
    pub succeeded: u64,
    pub rejected: u64,
    pub cross_personality_denied: u64,
    pub kasan_caught: u64,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FuzzReport {
    pub seed: u64,
    pub iterations: usize,
    pub total_dispatched: u64,
    pub total_succeeded: u64,
    pub total_rejected: u64,
    pub unknown_syscall_rejected: u64,
    pub cross_personality_denied: u64,
    pub kasan_violations_caught: u64,
    pub per_personality: [PersonalityStats; 4],
}

const NATIVE_SYSCALLS: &[SyscallNumber] = &[
    SyscallNumber::CapClose,
    SyscallNumber::CapDuplicate,
    SyscallNumber::CapTransfer,
    SyscallNumber::ChannelCreate,
    SyscallNumber::ChannelSend,
    SyscallNumber::ChannelRecv,
    SyscallNumber::VmoCreate,
    SyscallNumber::VmoMap,
    SyscallNumber::VmoUnmap,
    SyscallNumber::ThreadSpawn,
    SyscallNumber::ThreadExit,
    SyscallNumber::ThreadYield,
    SyscallNumber::ProcessSpawn,
    SyscallNumber::ProcessExit,
    SyscallNumber::WaitOne,
    SyscallNumber::Signal,
    SyscallNumber::DebugLog,
];

const POSIX_SYSCALLS: &[SyscallNumber] = &[
    SyscallNumber::PosixRead,
    SyscallNumber::PosixWrite,
    SyscallNumber::PosixOpenAt,
    SyscallNumber::PosixClose,
    SyscallNumber::PosixMmap,
    SyscallNumber::PosixFork,
    SyscallNumber::PosixExecve,
];

const NT_SYSCALLS: &[SyscallNumber] = &[
    SyscallNumber::NtCreateFile,
    SyscallNumber::NtReadFile,
    SyscallNumber::NtWriteFile,
    SyscallNumber::NtClose,
    SyscallNumber::NtAllocateVirtualMemory,
    SyscallNumber::NtCreateProcessEx,
];

const DARWIN_SYSCALLS: &[SyscallNumber] = &[
    SyscallNumber::DarwinMachMsg,
    SyscallNumber::DarwinBsdRead,
    SyscallNumber::DarwinBsdWrite,
    SyscallNumber::DarwinBsdOpen,
    SyscallNumber::DarwinBsdClose,
    SyscallNumber::DarwinTaskSelf,
];

fn syscalls_for(p: Personality) -> &'static [SyscallNumber] {
    match p {
        Personality::Native => NATIVE_SYSCALLS,
        Personality::Posix => POSIX_SYSCALLS,
        Personality::Nt => NT_SYSCALLS,
        Personality::Darwin => DARWIN_SYSCALLS,
    }
}

#[inline]
fn personality_index(p: Personality) -> usize {
    match p {
        Personality::Native => 0,
        Personality::Posix => 1,
        Personality::Nt => 2,
        Personality::Darwin => 3,
    }
}

/// Check whether a syscall takes a user pointer at `(ptr_arg_idx, len_arg_idx, is_write)`.
fn pointer_arg_spec(sc: SyscallNumber) -> Option<(usize, usize, bool)> {
    match sc {
        SyscallNumber::DebugLog => Some((0, 1, false)),
        SyscallNumber::PosixOpenAt => Some((1, 2, false)),
        SyscallNumber::PosixRead => Some((1, 2, true)),
        SyscallNumber::PosixWrite => Some((1, 2, false)),
        SyscallNumber::NtCreateFile => Some((0, 1, false)),
        SyscallNumber::NtReadFile => Some((1, 2, true)),
        SyscallNumber::NtWriteFile => Some((1, 2, false)),
        SyscallNumber::DarwinBsdOpen => Some((0, 1, false)),
        SyscallNumber::DarwinBsdRead => Some((1, 2, true)),
        SyscallNumber::DarwinBsdWrite => Some((1, 2, false)),
        _ => None,
    }
}

/// Run the deterministic multi-personality syscall fuzzer harness.
///
/// Uses KASAN-tracked buffers for pointer arguments so poisoned/OOB/UAF/wild
/// pointers are caught cleanly without causing unhandled kernel page faults.
pub fn run_syscall_fuzzer(config: FuzzConfig) -> FuzzReport {
    let mut rng = DeterministicRng::new(config.seed);
    let mut report = FuzzReport {
        seed: config.seed,
        iterations: config.iterations,
        ..FuzzReport::default()
    };

    // Allocate KASAN-tracked scratch buffers (valid + freed UAF)
    let valid_buf = alloc(64, 8).unwrap_or(0);
    let path_buf = alloc(32, 8).unwrap_or(0);
    let uaf_buf = alloc(32, 8).unwrap_or(0);
    if valid_buf != 0 {
        let _ = write_bytes(valid_buf, b"kasan-fuzz-payload");
    }
    if path_buf != 0 {
        let _ = write_bytes(path_buf, b"/etc/nexus.conf");
    }
    if uaf_buf != 0 {
        let _ = free(uaf_buf);
    }

    let mut open_fds = [0u64; 8];
    let mut open_fd_count = 0usize;

    let personalities = [
        Personality::Native,
        Personality::Posix,
        Personality::Nt,
        Personality::Darwin,
    ];

    for _ in 0..config.iterations {
        let p = personalities[rng.next_usize(personalities.len())];
        let p_idx = personality_index(p);

        let mode = rng.next_usize(10);
        let raw_num: u64 = if config.include_invalid_syscalls && mode == 0 {
            // Unknown syscall number
            match rng.next_usize(3) {
                0 => 0xDEAD_BEEF,
                1 => 0xFFFF_FFFF,
                _ => 0x4000 | (rng.next_u64() & 0x0FFF),
            }
        } else if config.include_cross_personality && mode == 1 {
            // Cross-personality syscall mismatch
            let other = personalities[(p_idx + 1 + rng.next_usize(3)) % 4];
            let list = syscalls_for(other);
            list[rng.next_usize(list.len())] as u64
        } else {
            let list = syscalls_for(p);
            list[rng.next_usize(list.len())] as u64
        };

        let mut args = [0u64; 6];
        for arg in &mut args {
            *arg = match rng.next_usize(6) {
                0 => 0,
                1 => 1,
                2 => 16,
                3 => 4097,
                4 => 65537,
                _ => rng.next_u64() & 0xFFFF,
            };
        }

        // If this is a known syscall, shape pointer/fd arguments safely
        if let Some(sc) = SyscallNumber::from_u64(raw_num) {
            if sc.personality() == p {
                if let Some((ptr_idx, len_idx, is_write)) = pointer_arg_spec(sc) {
                    let ptr_choice = rng.next_usize(5);
                    match ptr_choice {
                        0 if valid_buf != 0 && path_buf != 0 => {
                            if matches!(
                                sc,
                                SyscallNumber::PosixOpenAt
                                    | SyscallNumber::NtCreateFile
                                    | SyscallNumber::DarwinBsdOpen
                            ) {
                                args[ptr_idx] = path_buf as u64;
                                args[len_idx] = 15; // "/etc/nexus.conf"
                            } else {
                                args[ptr_idx] = valid_buf as u64;
                                args[len_idx] = 18;
                            }
                        }
                        1 if config.include_kasan_poisoned_ptrs && uaf_buf != 0 => {
                            // UAF pointer
                            args[ptr_idx] = uaf_buf as u64;
                            args[len_idx] = 8;
                        }
                        2 if config.include_kasan_poisoned_ptrs && valid_buf != 0 => {
                            // OOB right redzone pointer
                            args[ptr_idx] = (valid_buf + 64) as u64;
                            args[len_idx] = 4;
                        }
                        3 => {
                            // Null pointer (must be rejected by syscall::dispatch)
                            args[ptr_idx] = 0;
                            args[len_idx] = 16;
                        }
                        _ => {
                            // Wild pointer (caught by KASAN pre-check)
                            args[ptr_idx] = 0xDEAD_BEEF_0000;
                            args[len_idx] = 8;
                        }
                    }

                    // KASAN pre-check for non-null pointers before kernel dereference
                    if args[ptr_idx] != 0 && args[len_idx] > 0 && args[len_idx] <= 65536 {
                        let check_res = if is_write {
                            check_write(args[ptr_idx] as usize, args[len_idx] as usize)
                        } else {
                            check_read(args[ptr_idx] as usize, args[len_idx] as usize)
                        };
                        if check_res.is_err() {
                            report.kasan_violations_caught += 1;
                            report.per_personality[p_idx].kasan_caught += 1;
                            continue;
                        }
                    }
                }

                // Supply a valid open FD occasionally for read/write/close
                if matches!(
                    sc,
                    SyscallNumber::PosixRead
                        | SyscallNumber::PosixWrite
                        | SyscallNumber::PosixClose
                        | SyscallNumber::NtReadFile
                        | SyscallNumber::NtWriteFile
                        | SyscallNumber::NtClose
                        | SyscallNumber::DarwinBsdRead
                        | SyscallNumber::DarwinBsdWrite
                        | SyscallNumber::DarwinBsdClose
                ) && open_fd_count > 0
                    && rng.next_usize(2) == 0
                {
                    args[0] = open_fds[rng.next_usize(open_fd_count)];
                }

                // Keep VmoCreate size bounded so we don't exhaust physical frames
                if sc == SyscallNumber::VmoCreate {
                    args[0] = if rng.next_usize(2) == 0 { 0 } else { 4097 };
                }
            }
        }

        report.total_dispatched += 1;
        report.per_personality[p_idx].dispatched += 1;

        match syscall::dispatch(p, raw_num, args) {
            Ok(val) => {
                report.total_succeeded += 1;
                report.per_personality[p_idx].succeeded += 1;
                if let Some(sc) = SyscallNumber::from_u64(raw_num) {
                    if matches!(
                        sc,
                        SyscallNumber::PosixOpenAt
                            | SyscallNumber::NtCreateFile
                            | SyscallNumber::DarwinBsdOpen
                    ) && open_fd_count < open_fds.len()
                    {
                        open_fds[open_fd_count] = val;
                        open_fd_count += 1;
                    }
                }
            }
            Err(err) => {
                report.total_rejected += 1;
                report.per_personality[p_idx].rejected += 1;
                if err == AbiError::NotSupported && SyscallNumber::from_u64(raw_num).is_none() {
                    report.unknown_syscall_rejected += 1;
                } else if err == AbiError::AccessDenied {
                    report.cross_personality_denied += 1;
                    report.per_personality[p_idx].cross_personality_denied += 1;
                }
            }
        }
    }

    // Clean up any FDs opened during fuzzing
    for &fd in &open_fds[..open_fd_count] {
        let _ = syscall::dispatch(
            Personality::Posix,
            SyscallNumber::PosixClose as u64,
            [fd, 0, 0, 0, 0, 0],
        );
    }
    if valid_buf != 0 {
        let _ = free(valid_buf);
    }
    if path_buf != 0 {
        let _ = free(path_buf);
    }

    report
}

// ---------------------------------------------------------------------------
// Self-Test Suite
// ---------------------------------------------------------------------------

pub fn self_test() -> Result<(), &'static str> {
    init();

    // 1. Partial-granule 8:1 shadow tracking (13 bytes -> 8 + 5)
    let ptr = alloc(13, 8).map_err(|_| "kasan: alloc(13) failed")?;
    write_bytes(ptr, b"hello_kasan13").map_err(|_| "kasan: valid 13-byte write failed")?;
    let mut tmp = [0u8; 13];
    read_bytes(ptr, &mut tmp).map_err(|_| "kasan: valid 13-byte read failed")?;
    if &tmp != b"hello_kasan13" {
        return Err("kasan: readback mismatch");
    }

    // Byte 13 (14th byte) is in the partial granule's poisoned tail
    match check_read(ptr + 13, 1) {
        Err(r) if r.kind == ViolationKind::PartialGranuleOob => {}
        _ => return Err("kasan: failed to detect partial-granule OOB"),
    }

    // 2. Left redzone OOB
    match check_write(ptr - 1, 1) {
        Err(r) if r.kind == ViolationKind::OutOfBoundsLeft => {}
        _ => return Err("kasan: failed to detect left redzone OOB"),
    }

    // 3. Right redzone OOB (past padded 16-byte boundary)
    match check_read(ptr + 16, 1) {
        Err(r) if r.kind == ViolationKind::OutOfBoundsRight => {}
        _ => return Err("kasan: failed to detect right redzone OOB"),
    }

    // 4. Invalid free (interior pointer)
    match free(ptr + 4) {
        Err(r) if r.kind == ViolationKind::InvalidFree => {}
        _ => return Err("kasan: failed to detect invalid free"),
    }

    // 5. Free + Use-After-Free + Double-Free
    free(ptr).map_err(|_| "kasan: valid free failed")?;
    match read_u8(ptr) {
        Err(r) if r.kind == ViolationKind::UseAfterFree => {}
        _ => return Err("kasan: failed to detect use-after-free"),
    }
    match free(ptr) {
        Err(r) if r.kind == ViolationKind::DoubleFree => {}
        _ => return Err("kasan: failed to detect double-free"),
    }

    // 6. Wild pointer detection
    match check_read(0xDEAD_BEEF_CAFE, 4) {
        Err(r) if r.kind == ViolationKind::WildPointer => {}
        _ => return Err("kasan: failed to detect wild pointer"),
    }

    // 7. Quarantine eviction stress
    for _ in 0..(QUARANTINE_CAP + 4) {
        let p = alloc(16, 8).map_err(|_| "kasan: quarantine alloc failed")?;
        free(p).map_err(|_| "kasan: quarantine free failed")?;
    }

    // 8. Deterministic Syscall Fuzzer Harness across all 4 personalities
    let cfg = FuzzConfig {
        seed: 0x4E45_5855_535F_4B41,
        iterations: 96,
        include_cross_personality: true,
        include_invalid_syscalls: true,
        include_kasan_poisoned_ptrs: true,
    };
    let r1 = run_syscall_fuzzer(cfg);
    let r2 = run_syscall_fuzzer(cfg);
    if r1 != r2 {
        return Err("kasan: syscall fuzzer is not deterministic");
    }
    if r1.total_dispatched == 0 || r1.kasan_violations_caught == 0 {
        return Err("kasan: syscall fuzzer did not exercise dispatch/kasan");
    }
    if r1.cross_personality_denied == 0 || r1.unknown_syscall_rejected == 0 {
        return Err("kasan: syscall fuzzer did not verify cross-personality/unknown rejection");
    }

    let st = stats();
    serial_println!(
        "[kasan] self_test OK: checks={} violations={} (uaf={} oob={} df={}) fuzz_dispatched={} kasan_caught={}",
        st.checks_passed,
        st.violations_detected,
        st.uaf_detected,
        st.oob_detected,
        st.double_free_detected,
        r1.total_dispatched,
        r1.kasan_violations_caught
    );

    Ok(())
}
