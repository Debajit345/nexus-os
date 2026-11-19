//! `no_std` Slab/Magazine Kernel Allocator & NUMA/Huge-Page Manager.
//!
//! Provides:
//! - Power-of-two size-class slabs (`16 B .. 4 KiB`) with out-of-line
//!   [`SlabDescriptor`]s and 256-bit allocation bitmaps (100% page utilization,
//!   O(1) double-free and bounds validation).
//! - Bonwick-style per-CPU [`Magazine`] caches (`loaded` + `previous` spare)
//!   backed by a global [`MagazineDepot`] to minimize lock contention.
//! - NUMA node physical frame pools ([`NumaManager`]) with ACPI SLIT-style
//!   distance matrices and `LocalOnly`, `NearestFirst`, and `Interleave`
//!   allocation policies.
//! - 2 MiB and 1 GiB huge-page descriptors ([`HugePageManager`]) with
//!   reference counting, alignment enforcement, `1 GiB -> 512 x 2 MiB`
//!   splitting, and `512 x 2 MiB -> 1 GiB` coalescing.
//! - [`self_test`] exercising all subsystems with small stack frames (<2 KiB).

#![allow(dead_code)]

use alloc::alloc::{alloc_zeroed, dealloc, Layout};
use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::cpu::MAX_CPUS;
use crate::mm::{SpinLock, PAGE_SIZE};
use crate::serial_println;

/// 2 MiB huge-page size in bytes.
pub const HUGE_PAGE_2M: usize = 2 * 1024 * 1024;

/// 1 GiB huge-page size in bytes.
pub const HUGE_PAGE_1G: usize = 1024 * 1024 * 1024;

/// Number of power-of-two slab size classes (`16 B .. 4 KiB`).
pub const NUM_SIZE_CLASSES: usize = 9;

/// Supported slab object size classes in bytes.
pub const SIZE_CLASSES: [usize; NUM_SIZE_CLASSES] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

/// Number of object pointers held by a single per-CPU [`Magazine`].
pub const MAGAZINE_CAPACITY: usize = 16;

/// Maximum number of NUMA nodes tracked by [`NumaManager`].
pub const MAX_NUMA_NODES: usize = 8;

/// Default local NUMA distance (ACPI SLIT convention).
pub const NUMA_LOCAL_DISTANCE: u8 = 10;

/// Default remote NUMA distance.
pub const NUMA_REMOTE_DISTANCE: u8 = 20;

/// Unreachable NUMA distance sentinel.
pub const NUMA_UNREACHABLE_DISTANCE: u8 = 255;

/// Huge-page descriptor flag: backing memory is zeroed.
pub const HUGE_FLAG_ZEROED: u32 = 1 << 0;
/// Huge-page descriptor flag: pinned in physical memory.
pub const HUGE_FLAG_PINNED: u32 = 1 << 1;
/// Huge-page descriptor flag: suitable for DMA.
pub const HUGE_FLAG_DMA: u32 = 1 << 2;

/// Errors returned by the slab allocator, NUMA manager, and huge-page manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlabError {
    SizeTooLarge,
    InvalidAlignment,
    InvalidPointer,
    DoubleFree,
    OutOfMemory,
    InvalidCpuId,
    InvalidNodeId,
    NodeAlreadyRegistered,
    NodeNotOnline,
    InvalidRange,
    InvalidHugePageAlignment,
    HugePageNotFound,
    HugePageBusy,
    HugePageNotSplit,
    HugePageAlreadySplit,
}

/// Map an allocation `(size, align)` pair to a slab size-class index (`0..9`).
///
/// Returns `None` if `max(size, align) > 4096`.
pub const fn size_class_index(size: usize, align: usize) -> Option<usize> {
    let effective = if size > align { size } else { align };
    let effective = if effective == 0 { 1 } else { effective };
    let mut i = 0;
    while i < NUM_SIZE_CLASSES {
        if effective <= SIZE_CLASSES[i] {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Return the object size in bytes for a given size-class index.
pub const fn class_size(index: usize) -> usize {
    SIZE_CLASSES[index]
}

// ---------------------------------------------------------------------------
// Slab Descriptor & Slab Cache
// ---------------------------------------------------------------------------

/// Occupancy state of a single 4 KiB slab page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlabState {
    Empty,
    Partial,
    Full,
}

/// Out-of-line descriptor for a single 4 KiB slab page.
///
/// Using an out-of-line descriptor gives 100% page utilization for all size
/// classes (`16 B .. 4 KiB`), including `2048 B` (2 objects/page) and `4096 B`
/// (1 object/page), while keeping double-free and bounds checks O(1).
#[derive(Debug)]
pub struct SlabDescriptor {
    pub page_virt: usize,
    pub page_phys: u64,
    pub class_idx: u8,
    pub numa_node: u8,
    pub capacity: u16,
    pub in_use: u16,
    /// Bit = 1 means slot is FREE; bit = 0 means slot is ALLOCATED.
    free_bitmap: [u64; 4],
    owned_backing: bool,
}

impl SlabDescriptor {
    /// Create a new slab descriptor over an existing 4 KiB-aligned page.
    pub fn new(
        page_virt: usize,
        page_phys: u64,
        class_idx: usize,
        numa_node: u8,
        owned_backing: bool,
    ) -> Result<Self, SlabError> {
        if page_virt == 0 || (page_virt & (PAGE_SIZE - 1)) != 0 {
            return Err(SlabError::InvalidAlignment);
        }
        if class_idx >= NUM_SIZE_CLASSES {
            return Err(SlabError::SizeTooLarge);
        }

        let obj_size = SIZE_CLASSES[class_idx];
        let capacity = PAGE_SIZE / obj_size;
        let mut free_bitmap = [0u64; 4];

        let mut remaining = capacity;
        let mut word = 0usize;
        while remaining > 0 && word < 4 {
            if remaining >= 64 {
                free_bitmap[word] = u64::MAX;
                remaining -= 64;
            } else {
                free_bitmap[word] = (1u64 << remaining) - 1;
                remaining = 0;
            }
            word += 1;
        }

        Ok(Self {
            page_virt,
            page_phys,
            class_idx: class_idx as u8,
            numa_node,
            capacity: capacity as u16,
            in_use: 0,
            free_bitmap,
            owned_backing,
        })
    }

    /// Current occupancy state of this slab.
    pub fn state(&self) -> SlabState {
        if self.in_use == 0 {
            SlabState::Empty
        } else if self.in_use >= self.capacity {
            SlabState::Full
        } else {
            SlabState::Partial
        }
    }

    /// Allocate one object slot from this slab.
    pub fn alloc_slot(&mut self) -> Option<usize> {
        let obj_size = SIZE_CLASSES[self.class_idx as usize];
        for (word_idx, word) in self.free_bitmap.iter_mut().enumerate() {
            if *word != 0 {
                let bit = word.trailing_zeros() as usize;
                let slot_idx = word_idx * 64 + bit;
                if slot_idx >= self.capacity as usize {
                    return None;
                }
                *word &= !(1u64 << bit);
                self.in_use += 1;
                return Some(self.page_virt + slot_idx * obj_size);
            }
        }
        None
    }

    /// Return an object pointer to this slab, validating bounds and double-free.
    pub fn free_slot(&mut self, ptr: usize) -> Result<(), SlabError> {
        if ptr < self.page_virt || ptr >= self.page_virt + PAGE_SIZE {
            return Err(SlabError::InvalidPointer);
        }
        let obj_size = SIZE_CLASSES[self.class_idx as usize];
        let offset = ptr - self.page_virt;
        if !offset.is_multiple_of(obj_size) {
            return Err(SlabError::InvalidPointer);
        }
        let slot_idx = offset / obj_size;
        if slot_idx >= self.capacity as usize {
            return Err(SlabError::InvalidPointer);
        }

        let word_idx = slot_idx / 64;
        let bit = slot_idx % 64;
        let mask = 1u64 << bit;
        if (self.free_bitmap[word_idx] & mask) != 0 {
            return Err(SlabError::DoubleFree);
        }

        self.free_bitmap[word_idx] |= mask;
        self.in_use -= 1;
        Ok(())
    }

    /// Check whether `ptr` falls within this slab's 4 KiB page.
    #[inline]
    pub fn contains(&self, ptr: usize) -> bool {
        ptr >= self.page_virt && ptr < self.page_virt + PAGE_SIZE
    }
}

impl Drop for SlabDescriptor {
    fn drop(&mut self) {
        if self.owned_backing && self.page_virt != 0 {
            if let Ok(layout) = Layout::from_size_align(PAGE_SIZE, PAGE_SIZE) {
                unsafe {
                    dealloc(self.page_virt as *mut u8, layout);
                }
            }
            self.page_virt = 0;
        }
    }
}

/// Per-size-class slab cache managing `SlabDescriptor`s.
pub struct SlabCache {
    pub class_idx: usize,
    pub slabs: Vec<SlabDescriptor>,
    pub total_allocs: u64,
    pub total_frees: u64,
}

impl SlabCache {
    pub fn new(class_idx: usize) -> Self {
        Self {
            class_idx,
            slabs: Vec::new(),
            total_allocs: 0,
            total_frees: 0,
        }
    }

    /// Grow the cache by allocating a new 4 KiB backing page from the kernel heap.
    pub fn grow(&mut self, numa_node: u8) -> Result<usize, SlabError> {
        let layout = Layout::from_size_align(PAGE_SIZE, PAGE_SIZE)
            .map_err(|_| SlabError::InvalidAlignment)?;
        let raw = unsafe { alloc_zeroed(layout) };
        if raw.is_null() {
            return Err(SlabError::OutOfMemory);
        }
        let page_virt = raw as usize;
        let desc =
            SlabDescriptor::new(page_virt, page_virt as u64, self.class_idx, numa_node, true)?;
        self.slabs.push(desc);
        Ok(self.slabs.len() - 1)
    }

    /// Allocate one object directly from the slab cache (preferring partial slabs).
    pub fn alloc_object(&mut self, numa_node: u8) -> Result<usize, SlabError> {
        // Prefer partial slabs on the requested NUMA node, then any partial/empty slab.
        for slab in self.slabs.iter_mut() {
            if slab.numa_node == numa_node && slab.state() == SlabState::Partial {
                if let Some(ptr) = slab.alloc_slot() {
                    self.total_allocs += 1;
                    return Ok(ptr);
                }
            }
        }
        for slab in self.slabs.iter_mut() {
            if slab.state() != SlabState::Full {
                if let Some(ptr) = slab.alloc_slot() {
                    self.total_allocs += 1;
                    return Ok(ptr);
                }
            }
        }

        let idx = self.grow(numa_node)?;
        let ptr = self.slabs[idx].alloc_slot().ok_or(SlabError::OutOfMemory)?;
        self.total_allocs += 1;
        Ok(ptr)
    }

    /// Free an object back to its owning slab.
    pub fn free_object(&mut self, ptr: usize) -> Result<(), SlabError> {
        for slab in self.slabs.iter_mut() {
            if slab.contains(ptr) {
                slab.free_slot(ptr)?;
                self.total_frees += 1;
                return Ok(());
            }
        }
        Err(SlabError::InvalidPointer)
    }

    /// Reclaim all completely empty slabs, returning the number of pages freed.
    pub fn reclaim_empty(&mut self) -> usize {
        let before = self.slabs.len();
        self.slabs.retain(|s| s.state() != SlabState::Empty);
        before - self.slabs.len()
    }

    /// Count `(empty, partial, full)` slabs in this cache.
    pub fn slab_counts(&self) -> (usize, usize, usize) {
        let mut empty = 0;
        let mut partial = 0;
        let mut full = 0;
        for s in &self.slabs {
            match s.state() {
                SlabState::Empty => empty += 1,
                SlabState::Partial => partial += 1,
                SlabState::Full => full += 1,
            }
        }
        (empty, partial, full)
    }
}

// ---------------------------------------------------------------------------
// Per-CPU Magazines & Global Depot
// ---------------------------------------------------------------------------

/// Fixed-capacity magazine of cached object pointers for a single size class.
#[derive(Debug)]
pub struct Magazine {
    objects: [usize; MAGAZINE_CAPACITY],
    count: usize,
}

impl Magazine {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            objects: [0; MAGAZINE_CAPACITY],
            count: 0,
        })
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.count >= MAGAZINE_CAPACITY
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn push(&mut self, ptr: usize) -> bool {
        if self.count >= MAGAZINE_CAPACITY {
            return false;
        }
        self.objects[self.count] = ptr;
        self.count += 1;
        true
    }

    #[inline]
    pub fn pop(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let ptr = self.objects[self.count];
        self.objects[self.count] = 0;
        Some(ptr)
    }
}

/// Per-CPU, per-size-class magazine pair (`loaded` + `previous` spare).
pub struct PerCpuClassCache {
    pub loaded: Option<Box<Magazine>>,
    pub previous: Option<Box<Magazine>>,
    pub alloc_hits: u64,
    pub free_hits: u64,
    pub reloads: u64,
    pub flushes: u64,
}

impl PerCpuClassCache {
    pub fn new() -> Self {
        Self {
            loaded: None,
            previous: None,
            alloc_hits: 0,
            free_hits: 0,
            reloads: 0,
            flushes: 0,
        }
    }
}

/// Global depot of full and empty magazines for a single size class.
#[allow(clippy::vec_box)]
pub struct MagazineDepot {
    pub full: Vec<Box<Magazine>>,
    pub empty: Vec<Box<Magazine>>,
}

impl MagazineDepot {
    pub fn new() -> Self {
        Self {
            full: Vec::new(),
            empty: Vec::new(),
        }
    }
}

/// Snapshot of slab allocator statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct SlabStats {
    pub total_slabs: usize,
    pub empty_slabs: usize,
    pub partial_slabs: usize,
    pub full_slabs: usize,
    pub magazine_alloc_hits: u64,
    pub magazine_free_hits: u64,
    pub slab_allocs: u64,
    pub slab_frees: u64,
    pub depot_full_magazines: usize,
    pub depot_empty_magazines: usize,
}

/// Complete `no_std` Slab + Per-CPU Magazine Kernel Allocator.
pub struct SlabAllocator {
    caches: Vec<SlabCache>,
    depots: Vec<MagazineDepot>,
    /// Indexed as `per_cpu[cpu_id * NUM_SIZE_CLASSES + class_idx]`.
    per_cpu: Vec<PerCpuClassCache>,
    cpu_count: usize,
}

impl SlabAllocator {
    /// Create a new slab allocator supporting `cpu_count` CPUs (clamped to `1..=MAX_CPUS`).
    pub fn new(cpu_count: usize) -> Box<Self> {
        let cpus = cpu_count.clamp(1, MAX_CPUS);
        let mut caches = Vec::with_capacity(NUM_SIZE_CLASSES);
        let mut depots = Vec::with_capacity(NUM_SIZE_CLASSES);
        for i in 0..NUM_SIZE_CLASSES {
            caches.push(SlabCache::new(i));
            depots.push(MagazineDepot::new());
        }

        let total_slots = cpus * NUM_SIZE_CLASSES;
        let mut per_cpu = Vec::with_capacity(total_slots);
        for _ in 0..total_slots {
            per_cpu.push(PerCpuClassCache::new());
        }

        Box::new(Self {
            caches,
            depots,
            per_cpu,
            cpu_count: cpus,
        })
    }

    #[inline]
    fn cpu_slot_mut(
        &mut self,
        cpu_id: usize,
        class_idx: usize,
    ) -> Result<&mut PerCpuClassCache, SlabError> {
        if cpu_id >= self.cpu_count {
            return Err(SlabError::InvalidCpuId);
        }
        Ok(&mut self.per_cpu[cpu_id * NUM_SIZE_CLASSES + class_idx])
    }

    /// Allocate an object of `(size, align)` on `cpu_id` with preferred `numa_node`.
    pub fn alloc(
        &mut self,
        cpu_id: usize,
        size: usize,
        align: usize,
        numa_node: u8,
    ) -> Result<usize, SlabError> {
        let class_idx = size_class_index(size, align).ok_or(SlabError::SizeTooLarge)?;
        if cpu_id >= self.cpu_count {
            return Err(SlabError::InvalidCpuId);
        }

        let slot_idx = cpu_id * NUM_SIZE_CLASSES + class_idx;

        // 1. Fast path: pop from loaded magazine.
        if let Some(mag) = self.per_cpu[slot_idx].loaded.as_mut() {
            if let Some(ptr) = mag.pop() {
                self.per_cpu[slot_idx].alloc_hits += 1;
                return Ok(ptr);
            }
        }

        // 2. Swap with previous magazine if it has objects.
        let prev_has_items = self.per_cpu[slot_idx]
            .previous
            .as_ref()
            .is_some_and(|m| !m.is_empty());
        if prev_has_items {
            let slot = &mut self.per_cpu[slot_idx];
            core::mem::swap(&mut slot.loaded, &mut slot.previous);
            if let Some(mag) = slot.loaded.as_mut() {
                if let Some(ptr) = mag.pop() {
                    slot.alloc_hits += 1;
                    return Ok(ptr);
                }
            }
        }

        // 3. Try to acquire a full magazine from the global depot.
        if let Some(mut full_mag) = self.depots[class_idx].full.pop() {
            if let Some(old_loaded) = self.per_cpu[slot_idx].loaded.take() {
                self.depots[class_idx].empty.push(old_loaded);
            }
            let ptr = full_mag.pop().ok_or(SlabError::OutOfMemory)?;
            self.per_cpu[slot_idx].loaded = Some(full_mag);
            self.per_cpu[slot_idx].reloads += 1;
            return Ok(ptr);
        }

        // 4. Refill a magazine from the underlying SlabCache.
        let mut mag = self.depots[class_idx]
            .empty
            .pop()
            .unwrap_or_else(Magazine::new);
        for _ in 0..MAGAZINE_CAPACITY {
            match self.caches[class_idx].alloc_object(numa_node) {
                Ok(ptr) => {
                    let _ = mag.push(ptr);
                }
                Err(_) if !mag.is_empty() => break,
                Err(e) => {
                    self.depots[class_idx].empty.push(mag);
                    return Err(e);
                }
            }
        }

        let ptr = mag.pop().ok_or(SlabError::OutOfMemory)?;
        if let Some(old_loaded) = self.per_cpu[slot_idx].loaded.take() {
            self.depots[class_idx].empty.push(old_loaded);
        }
        self.per_cpu[slot_idx].loaded = Some(mag);
        self.per_cpu[slot_idx].reloads += 1;
        Ok(ptr)
    }

    /// Free an object of `(size, align)` on `cpu_id`.
    pub fn free(
        &mut self,
        cpu_id: usize,
        ptr: usize,
        size: usize,
        align: usize,
    ) -> Result<(), SlabError> {
        if ptr == 0 {
            return Err(SlabError::InvalidPointer);
        }
        let class_idx = size_class_index(size, align).ok_or(SlabError::SizeTooLarge)?;
        if cpu_id >= self.cpu_count {
            return Err(SlabError::InvalidCpuId);
        }

        let slot_idx = cpu_id * NUM_SIZE_CLASSES + class_idx;

        // 1. Fast path: push into loaded magazine if it has room.
        if self.per_cpu[slot_idx].loaded.is_none() {
            let mag = self.depots[class_idx]
                .empty
                .pop()
                .unwrap_or_else(Magazine::new);
            self.per_cpu[slot_idx].loaded = Some(mag);
        }
        if let Some(mag) = self.per_cpu[slot_idx].loaded.as_mut() {
            if mag.push(ptr) {
                self.per_cpu[slot_idx].free_hits += 1;
                return Ok(());
            }
        }

        // 2. Swap with previous magazine if it is missing or not full.
        let prev_can_accept = self.per_cpu[slot_idx]
            .previous
            .as_ref()
            .is_none_or(|m| !m.is_full());
        if prev_can_accept {
            if self.per_cpu[slot_idx].previous.is_none() {
                let mag = self.depots[class_idx]
                    .empty
                    .pop()
                    .unwrap_or_else(Magazine::new);
                self.per_cpu[slot_idx].previous = Some(mag);
            }
            let slot = &mut self.per_cpu[slot_idx];
            core::mem::swap(&mut slot.loaded, &mut slot.previous);
            if let Some(mag) = slot.loaded.as_mut() {
                if mag.push(ptr) {
                    slot.free_hits += 1;
                    return Ok(());
                }
            }
        }

        // 3. Loaded magazine is full: move it to the depot and grab an empty one.
        if let Some(full_mag) = self.per_cpu[slot_idx].loaded.take() {
            self.depots[class_idx].full.push(full_mag);
            self.per_cpu[slot_idx].flushes += 1;
        }
        let mut fresh = self.depots[class_idx]
            .empty
            .pop()
            .unwrap_or_else(Magazine::new);
        let _ = fresh.push(ptr);
        self.per_cpu[slot_idx].loaded = Some(fresh);
        self.per_cpu[slot_idx].free_hits += 1;
        Ok(())
    }

    /// Flush all magazines for `cpu_id` back to the underlying slab caches.
    pub fn drain_cpu(&mut self, cpu_id: usize) -> Result<usize, SlabError> {
        if cpu_id >= self.cpu_count {
            return Err(SlabError::InvalidCpuId);
        }
        let mut drained = 0usize;
        for class_idx in 0..NUM_SIZE_CLASSES {
            let slot_idx = cpu_id * NUM_SIZE_CLASSES + class_idx;
            if let Some(mut mag) = self.per_cpu[slot_idx].loaded.take() {
                while let Some(ptr) = mag.pop() {
                    self.caches[class_idx].free_object(ptr)?;
                    drained += 1;
                }
                self.depots[class_idx].empty.push(mag);
            }
            if let Some(mut mag) = self.per_cpu[slot_idx].previous.take() {
                while let Some(ptr) = mag.pop() {
                    self.caches[class_idx].free_object(ptr)?;
                    drained += 1;
                }
                self.depots[class_idx].empty.push(mag);
            }
        }
        Ok(drained)
    }

    /// Flush all depot full magazines and reclaim empty slabs across all classes.
    pub fn reclaim_all(&mut self) -> Result<usize, SlabError> {
        for cpu_id in 0..self.cpu_count {
            let _ = self.drain_cpu(cpu_id)?;
        }
        for class_idx in 0..NUM_SIZE_CLASSES {
            while let Some(mut mag) = self.depots[class_idx].full.pop() {
                while let Some(ptr) = mag.pop() {
                    self.caches[class_idx].free_object(ptr)?;
                }
                self.depots[class_idx].empty.push(mag);
            }
        }
        let mut reclaimed_pages = 0usize;
        for cache in &mut self.caches {
            reclaimed_pages += cache.reclaim_empty();
        }
        Ok(reclaimed_pages)
    }

    /// Collect aggregate statistics across all size classes and CPUs.
    pub fn stats(&self) -> SlabStats {
        let mut st = SlabStats::default();
        for (idx, cache) in self.caches.iter().enumerate() {
            let (e, p, f) = cache.slab_counts();
            st.empty_slabs += e;
            st.partial_slabs += p;
            st.full_slabs += f;
            st.total_slabs += e + p + f;
            st.slab_allocs += cache.total_allocs;
            st.slab_frees += cache.total_frees;
            st.depot_full_magazines += self.depots[idx].full.len();
            st.depot_empty_magazines += self.depots[idx].empty.len();
        }
        for slot in &self.per_cpu {
            st.magazine_alloc_hits += slot.alloc_hits;
            st.magazine_free_hits += slot.free_hits;
        }
        st
    }
}

// ---------------------------------------------------------------------------
// NUMA Node Frame Pools
// ---------------------------------------------------------------------------

/// NUMA physical frame allocation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaPolicy {
    /// Allocate strictly from the requested node.
    LocalOnly,
    /// Try the requested node first, then fall back in ascending distance order.
    NearestFirst,
    /// Round-robin across all online nodes.
    Interleave,
}

/// Summary statistics for a single NUMA node.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumaNodeInfo {
    pub node_id: u8,
    pub online: bool,
    pub cpu_mask: u64,
    pub total_frames: usize,
    pub free_frames: usize,
    pub allocated_frames: usize,
}

#[derive(Debug, Clone, Copy)]
struct NumaPhysRange {
    next: u64,
    end: u64,
    base: u64,
}

struct NumaNodePool {
    info: NumaNodeInfo,
    ranges: Vec<NumaPhysRange>,
    free_stack: Vec<u64>,
    distances: [u8; MAX_NUMA_NODES],
}

impl NumaNodePool {
    fn new(node_id: u8) -> Self {
        let mut distances = [NUMA_REMOTE_DISTANCE; MAX_NUMA_NODES];
        distances[node_id as usize] = NUMA_LOCAL_DISTANCE;
        Self {
            info: NumaNodeInfo {
                node_id,
                online: false,
                cpu_mask: 0,
                total_frames: 0,
                free_frames: 0,
                allocated_frames: 0,
            },
            ranges: Vec::new(),
            free_stack: Vec::new(),
            distances,
        }
    }

    fn add_range(&mut self, base_phys: u64, size_bytes: u64) -> Result<usize, SlabError> {
        let page = PAGE_SIZE as u64;
        let start = (base_phys + page - 1) & !(page - 1);
        let end = base_phys.saturating_add(size_bytes) & !(page - 1);
        if end <= start {
            return Err(SlabError::InvalidRange);
        }
        let frames = ((end - start) / page) as usize;
        self.ranges.push(NumaPhysRange {
            next: start,
            end,
            base: start,
        });
        self.info.total_frames += frames;
        self.info.free_frames += frames;
        Ok(frames)
    }

    fn alloc_frame(&mut self) -> Option<u64> {
        if !self.info.online || self.info.free_frames == 0 {
            return None;
        }
        if let Some(phys) = self.free_stack.pop() {
            self.info.free_frames -= 1;
            self.info.allocated_frames += 1;
            return Some(phys);
        }
        for r in &mut self.ranges {
            if r.next < r.end {
                let phys = r.next;
                r.next += PAGE_SIZE as u64;
                self.info.free_frames -= 1;
                self.info.allocated_frames += 1;
                return Some(phys);
            }
        }
        None
    }

    fn owns_phys(&self, phys_addr: u64) -> bool {
        self.ranges
            .iter()
            .any(|r| phys_addr >= r.base && phys_addr < r.end)
    }

    fn free_frame(&mut self, phys_addr: u64) -> Result<(), SlabError> {
        if (phys_addr & (PAGE_SIZE as u64 - 1)) != 0 {
            return Err(SlabError::InvalidAlignment);
        }
        if !self.owns_phys(phys_addr) {
            return Err(SlabError::InvalidPointer);
        }
        self.free_stack.push(phys_addr);
        self.info.free_frames += 1;
        self.info.allocated_frames = self.info.allocated_frames.saturating_sub(1);
        Ok(())
    }
}

/// Multi-node NUMA physical frame pool manager.
pub struct NumaManager {
    nodes: Vec<NumaNodePool>,
    interleave_cursor: usize,
}

impl NumaManager {
    pub fn new() -> Box<Self> {
        let mut nodes = Vec::with_capacity(MAX_NUMA_NODES);
        for id in 0..MAX_NUMA_NODES {
            nodes.push(NumaNodePool::new(id as u8));
        }
        Box::new(Self {
            nodes,
            interleave_cursor: 0,
        })
    }

    /// Register and bring a NUMA node online with its associated CPU bitmask.
    pub fn register_node(&mut self, node_id: u8, cpu_mask: u64) -> Result<(), SlabError> {
        let idx = node_id as usize;
        if idx >= MAX_NUMA_NODES {
            return Err(SlabError::InvalidNodeId);
        }
        if self.nodes[idx].info.online {
            return Err(SlabError::NodeAlreadyRegistered);
        }
        self.nodes[idx].info.online = true;
        self.nodes[idx].info.cpu_mask = cpu_mask;
        Ok(())
    }

    /// Add a physical memory range to an online NUMA node.
    pub fn add_memory_range(
        &mut self,
        node_id: u8,
        base_phys: u64,
        size_bytes: u64,
    ) -> Result<usize, SlabError> {
        let idx = node_id as usize;
        if idx >= MAX_NUMA_NODES {
            return Err(SlabError::InvalidNodeId);
        }
        if !self.nodes[idx].info.online {
            return Err(SlabError::NodeNotOnline);
        }
        self.nodes[idx].add_range(base_phys, size_bytes)
    }

    /// Set the ACPI SLIT-style distance between two NUMA nodes.
    pub fn set_distance(
        &mut self,
        from_node: u8,
        to_node: u8,
        distance: u8,
    ) -> Result<(), SlabError> {
        let f = from_node as usize;
        let t = to_node as usize;
        if f >= MAX_NUMA_NODES || t >= MAX_NUMA_NODES {
            return Err(SlabError::InvalidNodeId);
        }
        self.nodes[f].distances[t] = distance;
        Ok(())
    }

    /// Return the NUMA node owning `cpu_id`, defaulting to node `0`.
    pub fn node_for_cpu(&self, cpu_id: usize) -> u8 {
        if cpu_id < 64 {
            let bit = 1u64 << cpu_id;
            for node in &self.nodes {
                if node.info.online && (node.info.cpu_mask & bit) != 0 {
                    return node.info.node_id;
                }
            }
        }
        0
    }

    /// Allocate a 4 KiB physical frame according to `policy`, returning `(phys_addr, actual_node)`.
    pub fn alloc_frame(&mut self, preferred_node: u8, policy: NumaPolicy) -> Option<(u64, u8)> {
        let pref = preferred_node as usize;
        if pref >= MAX_NUMA_NODES {
            return None;
        }

        match policy {
            NumaPolicy::LocalOnly => self.nodes[pref]
                .alloc_frame()
                .map(|phys| (phys, preferred_node)),
            NumaPolicy::NearestFirst => {
                if let Some(phys) = self.nodes[pref].alloc_frame() {
                    return Some((phys, preferred_node));
                }
                // Build a small sorted candidate list on the stack (8 bytes).
                let mut order = [0u8; MAX_NUMA_NODES];
                for (i, slot) in order.iter_mut().enumerate() {
                    *slot = i as u8;
                }
                let distances = self.nodes[pref].distances;
                order.sort_unstable_by_key(|&n| distances[n as usize]);

                for &cand in &order {
                    let c = cand as usize;
                    if c == pref || distances[c] == NUMA_UNREACHABLE_DISTANCE {
                        continue;
                    }
                    if let Some(phys) = self.nodes[c].alloc_frame() {
                        return Some((phys, cand));
                    }
                }
                None
            }
            NumaPolicy::Interleave => {
                for offset in 0..MAX_NUMA_NODES {
                    let idx = (self.interleave_cursor + offset) % MAX_NUMA_NODES;
                    if let Some(phys) = self.nodes[idx].alloc_frame() {
                        self.interleave_cursor = (idx + 1) % MAX_NUMA_NODES;
                        return Some((phys, idx as u8));
                    }
                }
                None
            }
        }
    }

    /// Free a 4 KiB physical frame back to its owning NUMA node.
    pub fn free_frame(&mut self, phys_addr: u64) -> Result<(), SlabError> {
        for node in &mut self.nodes {
            if node.info.online && node.owns_phys(phys_addr) {
                return node.free_frame(phys_addr);
            }
        }
        Err(SlabError::InvalidPointer)
    }

    /// Query statistics for a NUMA node.
    pub fn node_stats(&self, node_id: u8) -> Option<NumaNodeInfo> {
        let idx = node_id as usize;
        if idx >= MAX_NUMA_NODES {
            return None;
        }
        Some(self.nodes[idx].info)
    }
}

// ---------------------------------------------------------------------------
// 2 MiB / 1 GiB Huge-Page Descriptors
// ---------------------------------------------------------------------------

/// Supported x86_64 huge-page sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HugePageSize {
    Size2M,
    Size1G,
}

impl HugePageSize {
    pub const fn bytes(self) -> usize {
        match self {
            Self::Size2M => HUGE_PAGE_2M,
            Self::Size1G => HUGE_PAGE_1G,
        }
    }

    pub const fn sub_4k_count(self) -> usize {
        self.bytes() / PAGE_SIZE
    }
}

/// Lifecycle state of a huge-page descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HugePageState {
    Free,
    Allocated,
    Split,
}

/// Descriptor for a single 2 MiB or 1 GiB physical huge page.
#[derive(Debug, Clone, Copy)]
pub struct HugePageDescriptor {
    pub phys_addr: u64,
    pub size: HugePageSize,
    pub numa_node: u8,
    pub state: HugePageState,
    pub ref_count: u32,
    pub flags: u32,
}

/// Aggregate statistics for [`HugePageManager`].
#[derive(Debug, Clone, Copy, Default)]
pub struct HugePageStats {
    pub total_2m: usize,
    pub free_2m: usize,
    pub allocated_2m: usize,
    pub split_2m: usize,
    pub total_1g: usize,
    pub free_1g: usize,
    pub allocated_1g: usize,
    pub split_1g: usize,
}

/// Manager for 2 MiB and 1 GiB huge-page descriptors, supporting NUMA-aware
/// allocation, reference counting, and `1 GiB <-> 512 x 2 MiB` split/coalesce.
pub struct HugePageManager {
    descriptors: Vec<HugePageDescriptor>,
}

impl HugePageManager {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            descriptors: Vec::new(),
        })
    }

    /// Register a physical region as huge-page descriptors of `page_size`.
    pub fn register_region(
        &mut self,
        base_phys: u64,
        size_bytes: u64,
        page_size: HugePageSize,
        numa_node: u8,
    ) -> Result<usize, SlabError> {
        if (numa_node as usize) >= MAX_NUMA_NODES {
            return Err(SlabError::InvalidNodeId);
        }
        let align = page_size.bytes() as u64;
        let start = (base_phys + align - 1) & !(align - 1);
        let end = base_phys.saturating_add(size_bytes) & !(align - 1);
        if end <= start {
            return Err(SlabError::InvalidHugePageAlignment);
        }

        let mut count = 0usize;
        let mut addr = start;
        while addr < end {
            self.descriptors.push(HugePageDescriptor {
                phys_addr: addr,
                size: page_size,
                numa_node,
                state: HugePageState::Free,
                ref_count: 0,
                flags: 0,
            });
            addr += align;
            count += 1;
        }
        Ok(count)
    }

    /// Allocate a huge page of `page_size` with NUMA preference.
    pub fn alloc_huge_page(
        &mut self,
        page_size: HugePageSize,
        preferred_node: u8,
        policy: NumaPolicy,
        flags: u32,
    ) -> Option<HugePageDescriptor> {
        // First pass: exact node match.
        for desc in &mut self.descriptors {
            if desc.size == page_size
                && desc.state == HugePageState::Free
                && desc.numa_node == preferred_node
            {
                desc.state = HugePageState::Allocated;
                desc.ref_count = 1;
                desc.flags = flags;
                return Some(*desc);
            }
        }

        if policy == NumaPolicy::LocalOnly {
            return None;
        }

        // Fallback pass: any free huge page of the requested size.
        for desc in &mut self.descriptors {
            if desc.size == page_size && desc.state == HugePageState::Free {
                desc.state = HugePageState::Allocated;
                desc.ref_count = 1;
                desc.flags = flags;
                return Some(*desc);
            }
        }
        None
    }

    /// Increment the reference count of an allocated huge page.
    pub fn retain(&mut self, phys_addr: u64) -> Result<u32, SlabError> {
        for desc in &mut self.descriptors {
            if desc.phys_addr == phys_addr && desc.state == HugePageState::Allocated {
                desc.ref_count = desc.ref_count.saturating_add(1);
                return Ok(desc.ref_count);
            }
        }
        Err(SlabError::HugePageNotFound)
    }

    /// Decrement the reference count of a huge page, freeing it when it reaches 0.
    pub fn free_huge_page(&mut self, phys_addr: u64) -> Result<(), SlabError> {
        for desc in &mut self.descriptors {
            if desc.phys_addr == phys_addr && desc.state == HugePageState::Allocated {
                if desc.ref_count > 1 {
                    desc.ref_count -= 1;
                } else {
                    desc.ref_count = 0;
                    desc.state = HugePageState::Free;
                    desc.flags = 0;
                }
                return Ok(());
            }
        }
        Err(SlabError::HugePageNotFound)
    }

    /// Split a free 1 GiB huge page into 512 free 2 MiB huge-page descriptors.
    pub fn split_1g_to_2m(&mut self, phys_addr: u64) -> Result<usize, SlabError> {
        if (phys_addr & (HUGE_PAGE_1G as u64 - 1)) != 0 {
            return Err(SlabError::InvalidHugePageAlignment);
        }
        let mut found_node = None;
        for desc in &mut self.descriptors {
            if desc.phys_addr == phys_addr && desc.size == HugePageSize::Size1G {
                if desc.state == HugePageState::Split {
                    return Err(SlabError::HugePageAlreadySplit);
                }
                if desc.state != HugePageState::Free {
                    return Err(SlabError::HugePageBusy);
                }
                desc.state = HugePageState::Split;
                found_node = Some(desc.numa_node);
                break;
            }
        }

        let numa_node = found_node.ok_or(SlabError::HugePageNotFound)?;
        let sub_count = HUGE_PAGE_1G / HUGE_PAGE_2M; // 512
        self.descriptors.reserve(sub_count);
        for i in 0..sub_count {
            self.descriptors.push(HugePageDescriptor {
                phys_addr: phys_addr + (i as u64) * (HUGE_PAGE_2M as u64),
                size: HugePageSize::Size2M,
                numa_node,
                state: HugePageState::Free,
                ref_count: 0,
                flags: 0,
            });
        }
        Ok(sub_count)
    }

    /// Coalesce 512 free 2 MiB descriptors back into their parent 1 GiB descriptor.
    pub fn coalesce_2m_to_1g(&mut self, phys_addr: u64) -> Result<(), SlabError> {
        if (phys_addr & (HUGE_PAGE_1G as u64 - 1)) != 0 {
            return Err(SlabError::InvalidHugePageAlignment);
        }

        let parent_idx = self
            .descriptors
            .iter()
            .position(|d| d.phys_addr == phys_addr && d.size == HugePageSize::Size1G)
            .ok_or(SlabError::HugePageNotFound)?;

        if self.descriptors[parent_idx].state != HugePageState::Split {
            return Err(SlabError::HugePageNotSplit);
        }

        let end_addr = phys_addr + HUGE_PAGE_1G as u64;
        let mut free_children = 0usize;
        for d in &self.descriptors {
            if d.size == HugePageSize::Size2M && d.phys_addr >= phys_addr && d.phys_addr < end_addr
            {
                if d.state != HugePageState::Free || d.ref_count != 0 {
                    return Err(SlabError::HugePageBusy);
                }
                free_children += 1;
            }
        }

        let expected = HUGE_PAGE_1G / HUGE_PAGE_2M;
        if free_children != expected {
            return Err(SlabError::HugePageBusy);
        }

        self.descriptors.retain(|d| {
            !(d.size == HugePageSize::Size2M && d.phys_addr >= phys_addr && d.phys_addr < end_addr)
        });

        // Re-locate parent after retain.
        if let Some(parent) = self
            .descriptors
            .iter_mut()
            .find(|d| d.phys_addr == phys_addr && d.size == HugePageSize::Size1G)
        {
            parent.state = HugePageState::Free;
        }
        Ok(())
    }

    /// Mark a free 2 MiB huge page as split into 512 x 4 KiB frames.
    pub fn split_2m_to_4k(&mut self, phys_addr: u64) -> Result<usize, SlabError> {
        if (phys_addr & (HUGE_PAGE_2M as u64 - 1)) != 0 {
            return Err(SlabError::InvalidHugePageAlignment);
        }
        for desc in &mut self.descriptors {
            if desc.phys_addr == phys_addr && desc.size == HugePageSize::Size2M {
                if desc.state == HugePageState::Split {
                    return Err(SlabError::HugePageAlreadySplit);
                }
                if desc.state != HugePageState::Free {
                    return Err(SlabError::HugePageBusy);
                }
                desc.state = HugePageState::Split;
                return Ok(HugePageSize::Size2M.sub_4k_count());
            }
        }
        Err(SlabError::HugePageNotFound)
    }

    /// Collect aggregate huge-page statistics.
    pub fn stats(&self) -> HugePageStats {
        let mut st = HugePageStats::default();
        for d in &self.descriptors {
            match (d.size, d.state) {
                (HugePageSize::Size2M, HugePageState::Free) => {
                    st.total_2m += 1;
                    st.free_2m += 1;
                }
                (HugePageSize::Size2M, HugePageState::Allocated) => {
                    st.total_2m += 1;
                    st.allocated_2m += 1;
                }
                (HugePageSize::Size2M, HugePageState::Split) => {
                    st.total_2m += 1;
                    st.split_2m += 1;
                }
                (HugePageSize::Size1G, HugePageState::Free) => {
                    st.total_1g += 1;
                    st.free_1g += 1;
                }
                (HugePageSize::Size1G, HugePageState::Allocated) => {
                    st.total_1g += 1;
                    st.allocated_1g += 1;
                }
                (HugePageSize::Size1G, HugePageState::Split) => {
                    st.total_1g += 1;
                    st.split_1g += 1;
                }
            }
        }
        st
    }
}

// ---------------------------------------------------------------------------
// Global Subsystem Handle
// ---------------------------------------------------------------------------

/// Combined global state for the slab allocator, NUMA pools, and huge pages.
pub struct SlabSubsystem {
    pub slab: Box<SlabAllocator>,
    pub numa: Box<NumaManager>,
    pub huge: Box<HugePageManager>,
}

pub static SLAB_SUBSYSTEM: SpinLock<Option<SlabSubsystem>> = SpinLock::new(None);

/// Initialize the global [`SLAB_SUBSYSTEM`] for `cpu_count` CPUs.
pub fn init(cpu_count: usize) {
    let mut guard = SLAB_SUBSYSTEM.lock();
    if guard.is_none() {
        *guard = Some(SlabSubsystem {
            slab: SlabAllocator::new(cpu_count),
            numa: NumaManager::new(),
            huge: HugePageManager::new(),
        });
    }
}

// ---------------------------------------------------------------------------
// Self-Test
// ---------------------------------------------------------------------------

/// Exercise size-class slabs (`16 B .. 4 KiB`), per-CPU magazines, NUMA node
/// frame pools, and 2 MiB / 1 GiB huge-page descriptors.
pub fn self_test() -> Result<(), &'static str> {
    serial_println!("[slab] running self_test...");

    // 1. Size-class mapping checks.
    if size_class_index(1, 1) != Some(0)
        || size_class_index(16, 16) != Some(0)
        || size_class_index(17, 1) != Some(1)
        || size_class_index(4096, 4096) != Some(8)
        || size_class_index(4097, 1).is_some()
    {
        return Err("size_class_index mapping mismatch");
    }

    // 2. Slab + per-CPU magazine allocation/free across all 9 size classes.
    let mut alloc = SlabAllocator::new(4);
    let mut ptrs: Vec<(usize, usize)> = Vec::new();

    for &sz in &SIZE_CLASSES {
        for _ in 0..20 {
            let p = alloc.alloc(0, sz, sz, 0).map_err(|_| "slab alloc failed")?;
            if (p & (sz - 1)) != 0 {
                return Err("slab allocation alignment violation");
            }
            ptrs.push((p, sz));
        }
    }

    // Free on CPU 1 to exercise cross-CPU magazine caching, then drain & reclaim.
    for &(p, sz) in &ptrs {
        alloc
            .free(1, p, sz, sz)
            .map_err(|_| "slab cross-cpu free failed")?;
    }

    let st_before = alloc.stats();
    if st_before.magazine_alloc_hits == 0 || st_before.magazine_free_hits == 0 {
        return Err("expected per-CPU magazine hits");
    }

    let reclaimed = alloc.reclaim_all().map_err(|_| "slab reclaim_all failed")?;
    if reclaimed == 0 {
        return Err("expected empty slabs to be reclaimed");
    }
    let st_after = alloc.stats();
    if st_after.total_slabs != 0 {
        return Err("expected all slabs reclaimed after full free");
    }

    // 3. NUMA node frame pools & distance-based fallback.
    let mut numa = NumaManager::new();
    numa.register_node(0, 0b0011)
        .map_err(|_| "numa register node 0 failed")?;
    numa.register_node(1, 0b1100)
        .map_err(|_| "numa register node 1 failed")?;
    numa.set_distance(0, 1, 25)
        .map_err(|_| "numa set_distance failed")?;

    if numa.node_for_cpu(2) != 1 || numa.node_for_cpu(0) != 0 {
        return Err("numa node_for_cpu mismatch");
    }

    // Node 0 gets 2 frames, Node 1 gets 4 frames.
    numa.add_memory_range(0, 0x1000_0000, 2 * PAGE_SIZE as u64)
        .map_err(|_| "numa add_memory_range node 0 failed")?;
    numa.add_memory_range(1, 0x2000_0000, 4 * PAGE_SIZE as u64)
        .map_err(|_| "numa add_memory_range node 1 failed")?;

    let (f0, n0) = numa
        .alloc_frame(0, NumaPolicy::LocalOnly)
        .ok_or("numa local alloc 0 failed")?;
    let (f1, n1) = numa
        .alloc_frame(0, NumaPolicy::LocalOnly)
        .ok_or("numa local alloc 1 failed")?;
    if n0 != 0 || n1 != 0 {
        return Err("numa local alloc returned wrong node");
    }
    if numa.alloc_frame(0, NumaPolicy::LocalOnly).is_some() {
        return Err("numa LocalOnly should fail when node 0 is exhausted");
    }

    // NearestFirst should fall back to node 1.
    let (f2, n2) = numa
        .alloc_frame(0, NumaPolicy::NearestFirst)
        .ok_or("numa NearestFirst fallback failed")?;
    if n2 != 1 {
        return Err("numa NearestFirst did not fall back to node 1");
    }

    numa.free_frame(f0).map_err(|_| "numa free f0 failed")?;
    numa.free_frame(f1).map_err(|_| "numa free f1 failed")?;
    numa.free_frame(f2).map_err(|_| "numa free f2 failed")?;

    // 4. 2 MiB / 1 GiB huge-page descriptors, refcounting, split & coalesce.
    let mut huge = HugePageManager::new();
    let base_1g = 0x4000_0000u64; // 1 GiB aligned
    let reg_1g = huge
        .register_region(base_1g, HUGE_PAGE_1G as u64, HugePageSize::Size1G, 0)
        .map_err(|_| "huge register 1G failed")?;
    if reg_1g != 1 {
        return Err("expected 1 x 1GiB descriptor");
    }

    // Allocate, retain, and release the 1 GiB page.
    let desc = huge
        .alloc_huge_page(
            HugePageSize::Size1G,
            0,
            NumaPolicy::LocalOnly,
            HUGE_FLAG_ZEROED,
        )
        .ok_or("huge alloc 1G failed")?;
    if huge
        .retain(desc.phys_addr)
        .map_err(|_| "huge retain failed")?
        != 2
    {
        return Err("huge refcount expected 2");
    }
    huge.free_huge_page(desc.phys_addr)
        .map_err(|_| "huge free 1 failed")?;
    huge.free_huge_page(desc.phys_addr)
        .map_err(|_| "huge free 2 failed")?;

    // Split 1 GiB -> 512 x 2 MiB, allocate one 2 MiB, free it, then coalesce back.
    let split_cnt = huge
        .split_1g_to_2m(base_1g)
        .map_err(|_| "huge split_1g_to_2m failed")?;
    if split_cnt != 512 {
        return Err("expected 512 x 2MiB descriptors from 1GiB split");
    }

    let sub_2m = huge
        .alloc_huge_page(
            HugePageSize::Size2M,
            0,
            NumaPolicy::LocalOnly,
            HUGE_FLAG_PINNED,
        )
        .ok_or("huge alloc 2M after split failed")?;
    if huge.coalesce_2m_to_1g(base_1g).is_ok() {
        return Err("coalesce should fail while a 2MiB child is allocated");
    }
    huge.free_huge_page(sub_2m.phys_addr)
        .map_err(|_| "huge free 2M failed")?;
    huge.coalesce_2m_to_1g(base_1g)
        .map_err(|_| "huge coalesce_2m_to_1g failed")?;

    let hst = huge.stats();
    if hst.total_1g != 1 || hst.free_1g != 1 || hst.total_2m != 0 {
        return Err("huge stats mismatch after coalesce");
    }

    serial_println!("[slab] self_test passed");
    Ok(())
}
