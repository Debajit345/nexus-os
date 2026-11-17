//! Physical frame allocator and kernel heap allocator (`#[global_allocator]`).

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicBool, Ordering};

use crate::limine::{MemmapResponse, MEMMAP_USABLE};

pub const PAGE_SIZE: usize = 4096;
const MAX_REGIONS: usize = 64;
const FREE_STACK_CAP: usize = 512;
const DEFAULT_HEAP_FRAMES: usize = 256; // 1 MiB initial kernel heap

/// Minimal spinlock for `no_std` kernel synchronization.
pub struct SpinLock<T> {
    locked: AtomicBool,
    inner: core::cell::UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for SpinLock<T> {}
unsafe impl<T: Send> Send for SpinLock<T> {}

impl<T> SpinLock<T> {
    pub const fn new(val: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            inner: core::cell::UnsafeCell::new(val),
        }
    }

    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        SpinLockGuard { lock: self }
    }
}

pub struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> core::ops::Deref for SpinLockGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.inner.get() }
    }
}

impl<T> core::ops::DerefMut for SpinLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.inner.get() }
    }
}

impl<T> Drop for SpinLockGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct PhysRegion {
    next: u64,
    end: u64,
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Default)]
pub struct MemoryStats {
    pub total_frames: usize,
    pub free_frames: usize,
    pub allocated_frames: usize,
    pub hhdm_offset: u64,
    pub heap_size: usize,
}

pub struct FrameAllocator {
    regions: [PhysRegion; MAX_REGIONS],
    region_count: usize,
    current_region: usize,
    free_stack: [u64; FREE_STACK_CAP],
    free_stack_len: usize,
    total_frames: usize,
    allocated_frames: usize,
    hhdm_offset: u64,
    initialized: bool,
}

impl FrameAllocator {
    const fn new() -> Self {
        Self {
            regions: [PhysRegion { next: 0, end: 0 }; MAX_REGIONS],
            region_count: 0,
            current_region: 0,
            free_stack: [0; FREE_STACK_CAP],
            free_stack_len: 0,
            total_frames: 0,
            allocated_frames: 0,
            hhdm_offset: 0,
            initialized: false,
        }
    }

    fn init(&mut self, memmap: &MemmapResponse, hhdm_offset: u64) {
        self.region_count = 0;
        self.current_region = 0;
        self.free_stack_len = 0;
        self.total_frames = 0;
        self.allocated_frames = 0;
        self.hhdm_offset = hhdm_offset;

        for entry in memmap.entries() {
            if entry.entry_type != MEMMAP_USABLE || entry.length == 0 {
                continue;
            }
            if self.region_count >= MAX_REGIONS {
                break;
            }

            let mut start = align_up(entry.base, PAGE_SIZE as u64);
            let end = align_down(entry.base.saturating_add(entry.length), PAGE_SIZE as u64);

            // Never hand out physical frame 0 (null page).
            if start < PAGE_SIZE as u64 {
                start = PAGE_SIZE as u64;
            }

            if end > start {
                let frames = ((end - start) / PAGE_SIZE as u64) as usize;
                self.regions[self.region_count] = PhysRegion { next: start, end };
                self.region_count += 1;
                self.total_frames += frames;
            }
        }

        self.initialized = true;
    }

    pub fn alloc_frame(&mut self) -> Option<u64> {
        if !self.initialized {
            return None;
        }

        if self.free_stack_len > 0 {
            self.free_stack_len -= 1;
            let addr = self.free_stack[self.free_stack_len];
            self.allocated_frames += 1;
            return Some(addr);
        }

        while self.current_region < self.region_count {
            let region = &mut self.regions[self.current_region];
            if region.next + PAGE_SIZE as u64 <= region.end {
                let addr = region.next;
                region.next += PAGE_SIZE as u64;
                self.allocated_frames += 1;
                return Some(addr);
            }
            self.current_region += 1;
        }

        None
    }

    pub fn alloc_contiguous(&mut self, num_frames: usize) -> Option<u64> {
        if !self.initialized || num_frames == 0 {
            return None;
        }
        let bytes = (num_frames as u64).checked_mul(PAGE_SIZE as u64)?;
        for idx in self.current_region..self.region_count {
            let region = &mut self.regions[idx];
            if region.end.saturating_sub(region.next) >= bytes {
                let addr = region.next;
                region.next += bytes;
                self.allocated_frames += num_frames;
                return Some(addr);
            }
        }
        None
    }

    #[allow(dead_code)]
    pub fn dealloc_frame(&mut self, phys_addr: u64) {
        if !self.initialized || phys_addr == 0 || (phys_addr % PAGE_SIZE as u64) != 0 {
            return;
        }
        if self.free_stack_len < FREE_STACK_CAP {
            self.free_stack[self.free_stack_len] = phys_addr;
            self.free_stack_len += 1;
            self.allocated_frames = self.allocated_frames.saturating_sub(1);
        }
    }
}

static FRAME_ALLOCATOR: SpinLock<FrameAllocator> = SpinLock::new(FrameAllocator::new());

#[inline]
const fn align_up(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

#[inline]
const fn align_down(val: u64, align: u64) -> u64 {
    val & !(align - 1)
}

#[inline]
const fn align_up_usize(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// Intrusive free-list node stored inside free heap blocks.
#[repr(C)]
struct FreeBlock {
    size: usize,
    next: Option<NonNull<FreeBlock>>,
}

struct FreeListHeap {
    head: Option<NonNull<FreeBlock>>,
    total_size: usize,
    used_size: usize,
}

unsafe impl Send for FreeListHeap {}

impl FreeListHeap {
    const fn empty() -> Self {
        Self {
            head: None,
            total_size: 0,
            used_size: 0,
        }
    }

    unsafe fn init(&mut self, start: usize, size: usize) {
        let min_block = core::mem::size_of::<FreeBlock>();
        let aligned_start = align_up_usize(start, core::mem::align_of::<FreeBlock>());
        let diff = aligned_start.saturating_sub(start);
        if size <= diff + min_block {
            return;
        }
        let usable_size = size - diff;
        let block_ptr = aligned_start as *mut FreeBlock;
        ptr::write(
            block_ptr,
            FreeBlock {
                size: usable_size,
                next: None,
            },
        );
        self.head = NonNull::new(block_ptr);
        self.total_size = usable_size;
        self.used_size = 0;
    }

    unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let min_block = core::mem::size_of::<FreeBlock>();
        let align = layout.align().max(core::mem::align_of::<FreeBlock>());
        let req_size = align_up_usize(
            layout.size().max(min_block),
            core::mem::align_of::<FreeBlock>(),
        );

        let mut prev: Option<NonNull<FreeBlock>> = None;
        let mut curr = self.head;

        while let Some(mut block_nn) = curr {
            let block = block_nn.as_mut();
            let block_addr = block_nn.as_ptr() as usize;
            let block_end = block_addr + block.size;

            let alloc_start = align_up_usize(block_addr, align);
            let front_pad = alloc_start - block_addr;

            // If there is front padding, it must either be 0 or large enough to hold a FreeBlock.
            let effective_start = if front_pad == 0 || front_pad >= min_block {
                alloc_start
            } else {
                align_up_usize(block_addr + min_block, align)
            };

            let effective_front = effective_start - block_addr;
            if let Some(alloc_end) = effective_start.checked_add(req_size) {
                if alloc_end <= block_end {
                    let back_pad = block_end - alloc_end;
                    let next_block = block.next;

                    // Build replacement chain for this block.
                    let mut replacement: Option<NonNull<FreeBlock>> = next_block;

                    if back_pad >= min_block {
                        let back_ptr = alloc_end as *mut FreeBlock;
                        ptr::write(
                            back_ptr,
                            FreeBlock {
                                size: back_pad,
                                next: replacement,
                            },
                        );
                        replacement = NonNull::new(back_ptr);
                    }

                    if effective_front >= min_block {
                        block.size = effective_front;
                        block.next = replacement;
                        replacement = Some(block_nn);
                    }

                    match prev {
                        Some(mut p) => p.as_mut().next = replacement,
                        None => self.head = replacement,
                    }

                    let consumed = block_end
                        - block_addr
                        - if back_pad >= min_block { back_pad } else { 0 }
                        - if effective_front >= min_block {
                            effective_front
                        } else {
                            0
                        };
                    self.used_size += consumed;
                    return effective_start as *mut u8;
                }
            }

            prev = curr;
            curr = block.next;
        }

        ptr::null_mut()
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        if ptr.is_null() {
            return;
        }
        let min_block = core::mem::size_of::<FreeBlock>();
        let size = align_up_usize(
            layout.size().max(min_block),
            core::mem::align_of::<FreeBlock>(),
        );
        let addr = ptr as usize;

        self.used_size = self.used_size.saturating_sub(size);

        let new_block_ptr = addr as *mut FreeBlock;
        ptr::write(new_block_ptr, FreeBlock { size, next: None });
        let mut new_nn = NonNull::new_unchecked(new_block_ptr);

        // Insert sorted by address and coalesce adjacent blocks.
        let mut prev: Option<NonNull<FreeBlock>> = None;
        let mut curr = self.head;

        while let Some(c) = curr {
            if (c.as_ptr() as usize) > addr {
                break;
            }
            prev = curr;
            curr = c.as_ref().next;
        }

        new_nn.as_mut().next = curr;
        match prev {
            Some(mut p) => p.as_mut().next = Some(new_nn),
            None => self.head = Some(new_nn),
        }

        // Coalesce with next block if contiguous.
        if let Some(next_nn) = new_nn.as_ref().next {
            let new_end = (new_nn.as_ptr() as usize) + new_nn.as_ref().size;
            if new_end == next_nn.as_ptr() as usize {
                new_nn.as_mut().size += next_nn.as_ref().size;
                new_nn.as_mut().next = next_nn.as_ref().next;
            }
        }

        // Coalesce with previous block if contiguous.
        if let Some(mut prev_nn) = prev {
            let prev_end = (prev_nn.as_ptr() as usize) + prev_nn.as_ref().size;
            if prev_end == new_nn.as_ptr() as usize {
                prev_nn.as_mut().size += new_nn.as_ref().size;
                prev_nn.as_mut().next = new_nn.as_ref().next;
            }
        }
    }
}

pub struct KernelAllocator {
    heap: SpinLock<FreeListHeap>,
}

impl KernelAllocator {
    const fn new() -> Self {
        Self {
            heap: SpinLock::new(FreeListHeap::empty()),
        }
    }

    unsafe fn init(&self, start: usize, size: usize) {
        self.heap.lock().init(start, size);
    }

    fn total_size(&self) -> usize {
        self.heap.lock().total_size
    }
}

unsafe impl GlobalAlloc for KernelAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.heap.lock().alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.heap.lock().dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: KernelAllocator = KernelAllocator::new();

/// Initialize the physical frame allocator and kernel heap from Limine's memory map.
pub fn init(memmap: &MemmapResponse, hhdm_offset: u64) -> MemoryStats {
    let heap_alloc = {
        let mut fa = FRAME_ALLOCATOR.lock();
        fa.init(memmap, hhdm_offset);
        fa.alloc_contiguous(DEFAULT_HEAP_FRAMES)
            .map(|phys| (phys, DEFAULT_HEAP_FRAMES))
            .or_else(|| fa.alloc_contiguous(32).map(|phys| (phys, 32)))
    };

    if let Some((phys, frames)) = heap_alloc {
        let heap_virt = (hhdm_offset + phys) as usize;
        let heap_bytes = frames * PAGE_SIZE;
        unsafe {
            GLOBAL_ALLOCATOR.init(heap_virt, heap_bytes);
        }
    }

    stats()
}

pub fn alloc_frame() -> Option<u64> {
    FRAME_ALLOCATOR.lock().alloc_frame()
}

#[allow(dead_code)]
pub fn dealloc_frame(phys_addr: u64) {
    FRAME_ALLOCATOR.lock().dealloc_frame(phys_addr);
}

pub fn stats() -> MemoryStats {
    let fa = FRAME_ALLOCATOR.lock();
    let free_frames = fa.total_frames.saturating_sub(fa.allocated_frames);
    MemoryStats {
        total_frames: fa.total_frames,
        free_frames,
        allocated_frames: fa.allocated_frames,
        hhdm_offset: fa.hhdm_offset,
        heap_size: GLOBAL_ALLOCATOR.total_size(),
    }
}
