//! `no_std` Multi-Queue Block Layer (`blk-mq`) & I/O Scheduler.
//!
//! Provides:
//! - Per-CPU software staging queues ([`SwQueueInner`]) with local request merging.
//! - Hardware dispatch queues ([`HwQueueInner`]) with bitmap [`TagAllocator`].
//! - Request merging (front/back contiguous LBA coalescing) and elevator sorting
//!   ([`SchedulerPolicy::None`], [`SchedulerPolicy::Elevator`], [`SchedulerPolicy::Deadline`]).
//! - [`BlkMqQueue`] coordinator supporting both synchronous [`BlockDevice`] execution
//!   and asynchronous tag-based dispatch/completion.
//! - [`self_test`] exercising tag allocation, merging, elevator ordering, and end-to-end I/O.

#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::blk::{BlockDevice, SECTOR_SIZE};
use crate::mm::SpinLock;
use crate::serial_println;

/// Default number of hardware tags per dispatch queue.
pub const DEFAULT_QUEUE_DEPTH: usize = 64;

/// Maximum supported queue depth per hardware queue.
pub const MAX_QUEUE_DEPTH: usize = 4096;

/// Default maximum sectors in a single merged request (128 KiB at 512 B/sector).
pub const DEFAULT_MAX_MERGE_SECTORS: u32 = 256;

/// Errors returned by the multi-queue block layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlkMqError {
    InvalidConfig,
    QueueFull,
    NoTagsAvailable,
    InvalidTag,
    InvalidRequest,
    QueueStopped,
    DeviceError(&'static str),
}

/// Block I/O operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReqOp {
    Read,
    Write,
    Flush,
    Discard,
}

/// I/O scheduling priority class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IoPriority {
    RealTime = 0,
    BestEffort = 1,
    Idle = 2,
}

/// Lifecycle state of a block request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReqState {
    Staged,
    Queued,
    Dispatched,
    Completed,
    Failed,
}

/// I/O scheduler policy for hardware dispatch queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerPolicy {
    /// FIFO order (still performs contiguous request merging).
    None,
    /// C-SCAN / LOOK elevator sorting by LBA ascending from the last dispatched LBA.
    Elevator,
    /// Priority-aware elevator (RealTime > BestEffort > Idle, LBA-sorted within class).
    Deadline,
}

/// Result of attempting to insert a request into a queue with merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeResult {
    MergedBack,
    MergedFront,
    Inserted,
}

/// Incoming block I/O descriptor submitted to `blk-mq`.
#[derive(Debug, Clone)]
pub struct Bio {
    pub op: ReqOp,
    pub lba: u64,
    pub sector_count: u32,
    pub priority: IoPriority,
    pub data: Vec<u8>,
}

impl Bio {
    /// Create a new read `Bio` for `sector_count` sectors starting at `lba`.
    pub fn read(lba: u64, sector_count: u32) -> Self {
        Self {
            op: ReqOp::Read,
            lba,
            sector_count,
            priority: IoPriority::BestEffort,
            data: Vec::new(),
        }
    }

    /// Create a new write `Bio` with payload `data` starting at `lba`.
    pub fn write(lba: u64, data: Vec<u8>) -> Self {
        let sector_count = (data.len() / SECTOR_SIZE) as u32;
        Self {
            op: ReqOp::Write,
            lba,
            sector_count,
            priority: IoPriority::BestEffort,
            data,
        }
    }

    /// Create a flush barrier `Bio`.
    pub fn flush() -> Self {
        Self {
            op: ReqOp::Flush,
            lba: 0,
            sector_count: 0,
            priority: IoPriority::RealTime,
            data: Vec::new(),
        }
    }

    /// Set the I/O priority class.
    pub fn with_priority(mut self, priority: IoPriority) -> Self {
        self.priority = priority;
        self
    }
}

/// A block layer request, potentially coalesced from multiple contiguous [`Bio`]s.
#[derive(Debug, Clone)]
pub struct BlkRequest {
    pub id: u64,
    pub tag: Option<u16>,
    pub op: ReqOp,
    pub lba: u64,
    pub sector_count: u32,
    pub priority: IoPriority,
    pub state: ReqState,
    pub cpu_id: usize,
    pub hw_queue_id: usize,
    pub merge_count: u32,
    pub data: Vec<u8>,
    pub status: Result<(), &'static str>,
}

impl BlkRequest {
    /// Construct a new staged request from a [`Bio`].
    pub fn from_bio(id: u64, cpu_id: usize, hw_queue_id: usize, bio: Bio) -> Self {
        Self {
            id,
            tag: None,
            op: bio.op,
            lba: bio.lba,
            sector_count: bio.sector_count,
            priority: bio.priority,
            state: ReqState::Staged,
            cpu_id,
            hw_queue_id,
            merge_count: 1,
            data: bio.data,
            status: Ok(()),
        }
    }

    /// Exclusive end LBA (`lba + sector_count`).
    #[inline]
    pub fn end_lba(&self) -> u64 {
        self.lba.saturating_add(self.sector_count as u64)
    }

    /// Whether `other` can be merged onto the back (tail) of `self`.
    pub fn can_merge_back(&self, other: &Self, max_sectors: u32) -> bool {
        if !self.is_mergeable_with(other) {
            return false;
        }
        let Some(total) = self.sector_count.checked_add(other.sector_count) else {
            return false;
        };
        total <= max_sectors && self.end_lba() == other.lba
    }

    /// Whether `other` can be merged onto the front (head) of `self`.
    pub fn can_merge_front(&self, other: &Self, max_sectors: u32) -> bool {
        if !self.is_mergeable_with(other) {
            return false;
        }
        let Some(total) = self.sector_count.checked_add(other.sector_count) else {
            return false;
        };
        total <= max_sectors && other.end_lba() == self.lba
    }

    #[inline]
    fn is_mergeable_with(&self, other: &Self) -> bool {
        matches!(self.op, ReqOp::Read | ReqOp::Write)
            && self.op == other.op
            && self.priority == other.priority
            && self.sector_count > 0
            && other.sector_count > 0
    }

    /// Merge `other` onto the back of `self`.
    pub fn merge_back(&mut self, mut other: Self) {
        self.sector_count = self.sector_count.saturating_add(other.sector_count);
        self.merge_count = self.merge_count.saturating_add(other.merge_count);
        if self.op == ReqOp::Write {
            self.data.append(&mut other.data);
        }
    }

    /// Merge `other` onto the front of `self`.
    pub fn merge_front(&mut self, mut other: Self) {
        self.lba = other.lba;
        self.sector_count = self.sector_count.saturating_add(other.sector_count);
        self.merge_count = self.merge_count.saturating_add(other.merge_count);
        if self.op == ReqOp::Write {
            other.data.append(&mut self.data);
            self.data = other.data;
        }
    }
}

/// Bitmap-based hardware tag allocator.
#[derive(Debug, Clone)]
pub struct TagAllocator {
    depth: usize,
    bitmap: Vec<u64>,
    allocated: usize,
    next_hint: usize,
}

impl TagAllocator {
    /// Create a new tag allocator with `depth` tags (`0..depth`).
    pub fn new(depth: usize) -> Result<Self, BlkMqError> {
        if depth == 0 || depth > MAX_QUEUE_DEPTH {
            return Err(BlkMqError::InvalidConfig);
        }
        let words = depth.div_ceil(64);
        Ok(Self {
            depth,
            bitmap: vec![0u64; words],
            allocated: 0,
            next_hint: 0,
        })
    }

    /// Allocate the next free tag, or `None` if all tags are in use.
    pub fn alloc(&mut self) -> Option<u16> {
        if self.allocated >= self.depth {
            return None;
        }
        for offset in 0..self.depth {
            let idx = (self.next_hint + offset) % self.depth;
            let word = idx / 64;
            let bit = idx % 64;
            if (self.bitmap[word] & (1u64 << bit)) == 0 {
                self.bitmap[word] |= 1u64 << bit;
                self.allocated += 1;
                self.next_hint = (idx + 1) % self.depth;
                return Some(idx as u16);
            }
        }
        None
    }

    /// Reserve a specific tag if it is currently free.
    pub fn alloc_specific(&mut self, tag: u16) -> Result<(), BlkMqError> {
        let idx = tag as usize;
        if idx >= self.depth {
            return Err(BlkMqError::InvalidTag);
        }
        let word = idx / 64;
        let bit = idx % 64;
        if (self.bitmap[word] & (1u64 << bit)) != 0 {
            return Err(BlkMqError::NoTagsAvailable);
        }
        self.bitmap[word] |= 1u64 << bit;
        self.allocated += 1;
        Ok(())
    }

    /// Release a previously allocated tag.
    pub fn free(&mut self, tag: u16) -> Result<(), BlkMqError> {
        let idx = tag as usize;
        if idx >= self.depth {
            return Err(BlkMqError::InvalidTag);
        }
        let word = idx / 64;
        let bit = idx % 64;
        if (self.bitmap[word] & (1u64 << bit)) == 0 {
            return Err(BlkMqError::InvalidTag);
        }
        self.bitmap[word] &= !(1u64 << bit);
        self.allocated -= 1;
        self.next_hint = idx;
        Ok(())
    }

    /// Check whether `tag` is currently allocated.
    pub fn is_allocated(&self, tag: u16) -> bool {
        let idx = tag as usize;
        if idx >= self.depth {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        (self.bitmap[word] & (1u64 << bit)) != 0
    }

    /// Number of tags currently in flight.
    #[inline]
    pub fn in_flight(&self) -> usize {
        self.allocated
    }

    /// Number of free tags remaining.
    #[inline]
    pub fn available(&self) -> usize {
        self.depth - self.allocated
    }

    /// Total tag capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.depth
    }
}

/// Request queue with contiguous LBA merging and elevator selection.
#[derive(Debug, Default)]
pub struct ElevatorQueue {
    pending: Vec<BlkRequest>,
}

impl ElevatorQueue {
    /// Create an empty elevator queue.
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Insert `req`, merging with an existing contiguous request if possible.
    pub fn insert_or_merge(&mut self, req: BlkRequest, max_merge_sectors: u32) -> MergeResult {
        for existing in &mut self.pending {
            if existing.can_merge_back(&req, max_merge_sectors) {
                existing.merge_back(req);
                return MergeResult::MergedBack;
            }
            if existing.can_merge_front(&req, max_merge_sectors) {
                existing.merge_front(req);
                return MergeResult::MergedFront;
            }
        }
        self.pending.push(req);
        MergeResult::Inserted
    }

    /// Pop the next request according to `policy` and `last_lba`.
    pub fn pop_next(&mut self, policy: SchedulerPolicy, last_lba: u64) -> Option<BlkRequest> {
        if self.pending.is_empty() {
            return None;
        }
        let idx = match policy {
            SchedulerPolicy::None => 0,
            SchedulerPolicy::Elevator => self.pick_elevator_idx(None, last_lba),
            SchedulerPolicy::Deadline => {
                let best_prio = self
                    .pending
                    .iter()
                    .map(|r| r.priority)
                    .min()
                    .unwrap_or(IoPriority::BestEffort);
                self.pick_elevator_idx(Some(best_prio), last_lba)
            }
        };
        Some(self.pending.remove(idx))
    }

    fn pick_elevator_idx(&self, prio_filter: Option<IoPriority>, last_lba: u64) -> usize {
        let mut best_above: Option<(usize, u64)> = None;
        let mut best_wrap: Option<(usize, u64)> = None;

        for (i, req) in self.pending.iter().enumerate() {
            if let Some(prio) = prio_filter {
                if req.priority != prio {
                    continue;
                }
            }
            if req.lba >= last_lba {
                if best_above.is_none_or(|(_, lba)| req.lba < lba) {
                    best_above = Some((i, req.lba));
                }
            } else if best_wrap.is_none_or(|(_, lba)| req.lba < lba) {
                best_wrap = Some((i, req.lba));
            }
        }

        best_above.or(best_wrap).map(|(idx, _)| idx).unwrap_or(0)
    }

    /// Number of pending requests in the queue.
    #[inline]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether the queue has no pending requests.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

/// Per-CPU software staging queue state.
#[derive(Debug)]
pub struct SwQueueInner {
    pub cpu_id: usize,
    pub hw_queue_id: usize,
    pub queue: ElevatorQueue,
    pub submitted: u64,
    pub merged: u64,
    pub flushed: u64,
}

impl SwQueueInner {
    fn new(cpu_id: usize, hw_queue_id: usize) -> Self {
        Self {
            cpu_id,
            hw_queue_id,
            queue: ElevatorQueue::new(),
            submitted: 0,
            merged: 0,
            flushed: 0,
        }
    }
}

/// Hardware dispatch queue state.
#[derive(Debug)]
pub struct HwQueueInner {
    pub hw_id: usize,
    pub tags: TagAllocator,
    pub scheduler: ElevatorQueue,
    pub in_flight: Vec<Option<BlkRequest>>,
    pub last_dispatched_lba: u64,
    pub policy: SchedulerPolicy,
    pub stopped: bool,
    pub dispatched: u64,
    pub completed: u64,
    pub merged: u64,
    pub errors: u64,
}

impl HwQueueInner {
    fn new(hw_id: usize, depth: usize, policy: SchedulerPolicy) -> Result<Self, BlkMqError> {
        let tags = TagAllocator::new(depth)?;
        let mut in_flight = Vec::with_capacity(depth);
        in_flight.resize_with(depth, || None);
        Ok(Self {
            hw_id,
            tags,
            scheduler: ElevatorQueue::new(),
            in_flight,
            last_dispatched_lba: 0,
            policy,
            stopped: false,
            dispatched: 0,
            completed: 0,
            merged: 0,
            errors: 0,
        })
    }

    /// Enqueue a request into the hardware scheduler, merging if contiguous.
    pub fn enqueue(&mut self, mut req: BlkRequest, max_merge_sectors: u32) -> MergeResult {
        req.state = ReqState::Queued;
        req.hw_queue_id = self.hw_id;
        let res = self.scheduler.insert_or_merge(req, max_merge_sectors);
        if res != MergeResult::Inserted {
            self.merged += 1;
        }
        res
    }

    /// Allocate a hardware tag and pop the next scheduled request for dispatch.
    pub fn dispatch_next(&mut self) -> Option<BlkRequest> {
        if self.stopped || self.scheduler.is_empty() {
            return None;
        }
        let tag = self.tags.alloc()?;
        let mut req = self
            .scheduler
            .pop_next(self.policy, self.last_dispatched_lba)?;
        req.tag = Some(tag);
        req.state = ReqState::Dispatched;
        self.last_dispatched_lba = req.end_lba();
        self.dispatched += 1;
        self.in_flight[tag as usize] = Some(req.clone());
        Some(req)
    }

    /// Complete an in-flight request by its hardware tag and release the tag.
    pub fn complete(
        &mut self,
        tag: u16,
        status: Result<(), &'static str>,
    ) -> Result<BlkRequest, BlkMqError> {
        self.tags.free(tag)?;
        let mut req = self.in_flight[tag as usize]
            .take()
            .ok_or(BlkMqError::InvalidTag)?;
        req.status = status;
        if status.is_ok() {
            req.state = ReqState::Completed;
            self.completed += 1;
        } else {
            req.state = ReqState::Failed;
            self.errors += 1;
        }
        Ok(req)
    }
}

/// Configuration for [`BlkMqQueue`].
#[derive(Debug, Clone, Copy)]
pub struct BlkMqConfig {
    pub num_cpus: usize,
    pub num_hw_queues: usize,
    pub queue_depth: usize,
    pub max_merge_sectors: u32,
    pub policy: SchedulerPolicy,
}

impl Default for BlkMqConfig {
    fn default() -> Self {
        Self {
            num_cpus: 4,
            num_hw_queues: 2,
            queue_depth: DEFAULT_QUEUE_DEPTH,
            max_merge_sectors: DEFAULT_MAX_MERGE_SECTORS,
            policy: SchedulerPolicy::Elevator,
        }
    }
}

/// Aggregate statistics across all software and hardware queues.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlkMqStats {
    pub submitted: u64,
    pub merged: u64,
    pub dispatched: u64,
    pub completed: u64,
    pub errors: u64,
    pub in_flight: usize,
    pub staged: usize,
    pub queued: usize,
}

/// Multi-Queue Block Layer (`blk-mq`) coordinator.
pub struct BlkMqQueue {
    config: BlkMqConfig,
    next_req_id: AtomicU64,
    sw_queues: Vec<SpinLock<SwQueueInner>>,
    hw_queues: Vec<SpinLock<HwQueueInner>>,
}

impl BlkMqQueue {
    /// Create a new multi-queue block layer instance.
    pub fn new(config: BlkMqConfig) -> Result<Self, BlkMqError> {
        if config.num_cpus == 0
            || config.num_hw_queues == 0
            || config.queue_depth == 0
            || config.queue_depth > MAX_QUEUE_DEPTH
            || config.max_merge_sectors == 0
        {
            return Err(BlkMqError::InvalidConfig);
        }

        let mut sw_queues = Vec::with_capacity(config.num_cpus);
        for cpu in 0..config.num_cpus {
            let hw_id = cpu % config.num_hw_queues;
            sw_queues.push(SpinLock::new(SwQueueInner::new(cpu, hw_id)));
        }

        let mut hw_queues = Vec::with_capacity(config.num_hw_queues);
        for hw_id in 0..config.num_hw_queues {
            hw_queues.push(SpinLock::new(HwQueueInner::new(
                hw_id,
                config.queue_depth,
                config.policy,
            )?));
        }

        Ok(Self {
            config,
            next_req_id: AtomicU64::new(1),
            sw_queues,
            hw_queues,
        })
    }

    /// Submit a [`Bio`] to the staging queue for `cpu_id`.
    pub fn submit_bio(&self, cpu_id: usize, bio: Bio) -> Result<u64, BlkMqError> {
        if matches!(bio.op, ReqOp::Read | ReqOp::Write) && bio.sector_count == 0 {
            return Err(BlkMqError::InvalidRequest);
        }
        if bio.op == ReqOp::Write && bio.data.len() != (bio.sector_count as usize) * SECTOR_SIZE {
            return Err(BlkMqError::InvalidRequest);
        }

        let sw_idx = cpu_id % self.config.num_cpus;
        let hw_idx = sw_idx % self.config.num_hw_queues;
        let req_id = self.next_req_id.fetch_add(1, Ordering::Relaxed);
        let req = BlkRequest::from_bio(req_id, sw_idx, hw_idx, bio);

        let mut sw = self.sw_queues[sw_idx].lock();
        sw.submitted += 1;
        let res = sw.queue.insert_or_merge(req, self.config.max_merge_sectors);
        if res != MergeResult::Inserted {
            sw.merged += 1;
        }
        Ok(req_id)
    }

    /// Flush all staged requests from `cpu_id`'s software queue to its hardware queue.
    pub fn flush_sw_queue(&self, cpu_id: usize) -> usize {
        let sw_idx = cpu_id % self.config.num_cpus;
        let (hw_idx, drained) = {
            let mut sw = self.sw_queues[sw_idx].lock();
            let hw_idx = sw.hw_queue_id;
            let mut drained = Vec::with_capacity(sw.queue.len());
            while let Some(req) = sw.queue.pop_next(SchedulerPolicy::None, 0) {
                drained.push(req);
            }
            sw.flushed += drained.len() as u64;
            (hw_idx, drained)
        };

        let count = drained.len();
        if count > 0 {
            let mut hw = self.hw_queues[hw_idx].lock();
            for req in drained {
                hw.enqueue(req, self.config.max_merge_sectors);
            }
        }
        count
    }

    /// Flush all per-CPU software staging queues into their mapped hardware queues.
    pub fn flush_all_sw_queues(&self) -> usize {
        let mut total = 0;
        for cpu in 0..self.config.num_cpus {
            total += self.flush_sw_queue(cpu);
        }
        total
    }

    /// Pop the next tagged request from hardware queue `hw_id` for async driver submission.
    pub fn pop_dispatch(&self, hw_id: usize) -> Option<BlkRequest> {
        let hw_idx = hw_id % self.config.num_hw_queues;
        let mut hw = self.hw_queues[hw_idx].lock();
        hw.dispatch_next()
    }

    /// Complete an async in-flight request on hardware queue `hw_id` by `tag`.
    pub fn complete_tag(
        &self,
        hw_id: usize,
        tag: u16,
        status: Result<(), &'static str>,
    ) -> Result<BlkRequest, BlkMqError> {
        let hw_idx = hw_id % self.config.num_hw_queues;
        let mut hw = self.hw_queues[hw_idx].lock();
        hw.complete(tag, status)
    }

    /// Dispatch and synchronously execute all queued requests on `hw_id` against `dev`.
    pub fn dispatch_hw_queue(
        &self,
        hw_id: usize,
        dev: &dyn BlockDevice,
    ) -> Result<Vec<BlkRequest>, BlkMqError> {
        let hw_idx = hw_id % self.config.num_hw_queues;
        let mut completed = Vec::new();

        loop {
            let mut req = {
                let mut hw = self.hw_queues[hw_idx].lock();
                match hw.dispatch_next() {
                    Some(r) => r,
                    None => break,
                }
            };

            let tag = req.tag.ok_or(BlkMqError::InvalidTag)?;
            let io_res = execute_on_device(dev, &mut req);

            let mut hw = self.hw_queues[hw_idx].lock();
            let mut done = hw.complete(tag, io_res)?;
            done.data = req.data;
            completed.push(done);
        }

        Ok(completed)
    }

    /// Flush all software queues and execute all hardware queues against `dev`.
    pub fn run_all(&self, dev: &dyn BlockDevice) -> Result<Vec<BlkRequest>, BlkMqError> {
        self.flush_all_sw_queues();
        let mut all_completed = Vec::new();
        for hw_id in 0..self.config.num_hw_queues {
            let mut done = self.dispatch_hw_queue(hw_id, dev)?;
            all_completed.append(&mut done);
        }
        Ok(all_completed)
    }

    /// Update the scheduler policy on all hardware queues.
    pub fn set_policy(&self, policy: SchedulerPolicy) {
        for hw_lock in &self.hw_queues {
            hw_lock.lock().policy = policy;
        }
    }

    /// Collect aggregate statistics across all queues.
    pub fn stats(&self) -> BlkMqStats {
        let mut stats = BlkMqStats::default();
        for sw_lock in &self.sw_queues {
            let sw = sw_lock.lock();
            stats.submitted += sw.submitted;
            stats.merged += sw.merged;
            stats.staged += sw.queue.len();
        }
        for hw_lock in &self.hw_queues {
            let hw = hw_lock.lock();
            stats.merged += hw.merged;
            stats.dispatched += hw.dispatched;
            stats.completed += hw.completed;
            stats.errors += hw.errors;
            stats.in_flight += hw.tags.in_flight();
            stats.queued += hw.scheduler.len();
        }
        stats
    }
}

fn execute_on_device(dev: &dyn BlockDevice, req: &mut BlkRequest) -> Result<(), &'static str> {
    match req.op {
        ReqOp::Read => {
            let byte_len = (req.sector_count as usize) * SECTOR_SIZE;
            req.data.resize(byte_len, 0);
            dev.read_blocks(req.lba, &mut req.data)
        }
        ReqOp::Write => dev.write_blocks(req.lba, &req.data),
        ReqOp::Flush | ReqOp::Discard => Ok(()),
    }
}

/// In-memory [`BlockDevice`] used by [`self_test`].
struct MemBlockDevice {
    sectors: SpinLock<Vec<u8>>,
    block_count: u64,
}

impl MemBlockDevice {
    fn new(block_count: u64) -> Self {
        let total_bytes = (block_count as usize) * SECTOR_SIZE;
        Self {
            sectors: SpinLock::new(vec![0u8; total_bytes]),
            block_count,
        }
    }
}

impl BlockDevice for MemBlockDevice {
    fn block_size(&self) -> usize {
        SECTOR_SIZE
    }

    fn block_count(&self) -> u64 {
        self.block_count
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(SECTOR_SIZE) {
            return Err("unaligned read buffer");
        }
        let count = (buf.len() / SECTOR_SIZE) as u64;
        if lba.saturating_add(count) > self.block_count {
            return Err("read out of bounds");
        }
        let start = (lba as usize) * SECTOR_SIZE;
        let data = self.sectors.lock();
        buf.copy_from_slice(&data[start..start + buf.len()]);
        Ok(())
    }

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        if !buf.len().is_multiple_of(SECTOR_SIZE) {
            return Err("unaligned write buffer");
        }
        let count = (buf.len() / SECTOR_SIZE) as u64;
        if lba.saturating_add(count) > self.block_count {
            return Err("write out of bounds");
        }
        let start = (lba as usize) * SECTOR_SIZE;
        let mut data = self.sectors.lock();
        data[start..start + buf.len()].copy_from_slice(buf);
        Ok(())
    }
}

/// Run self-tests for the multi-queue block layer and I/O scheduler.
pub fn self_test() -> Result<(), &'static str> {
    // 1. TagAllocator test
    let mut tags = TagAllocator::new(4).map_err(|_| "tag allocator init failed")?;
    let mut allocated = Vec::new();
    for _ in 0..4 {
        let t = tags.alloc().ok_or("expected free tag")?;
        allocated.push(t);
    }
    if tags.alloc().is_some() {
        return Err("tag allocator exceeded capacity");
    }
    if tags.in_flight() != 4 || tags.available() != 0 {
        return Err("tag allocator counts mismatch");
    }
    tags.free(allocated[1]).map_err(|_| "tag free failed")?;
    if tags.free(allocated[1]).is_ok() {
        return Err("double free should fail");
    }
    let reused = tags.alloc().ok_or("expected reused tag")?;
    if reused != allocated[1] {
        return Err("expected freed tag to be reused");
    }

    // 2. Request merging (back + front) and non-mergeable Flush
    let mut eq = ElevatorQueue::new();
    let r1 = BlkRequest::from_bio(1, 0, 0, Bio::write(10, vec![0xAA; SECTOR_SIZE]));
    let r2 = BlkRequest::from_bio(2, 0, 0, Bio::write(11, vec![0xBB; SECTOR_SIZE]));
    let r3 = BlkRequest::from_bio(3, 0, 0, Bio::write(9, vec![0x99; SECTOR_SIZE]));
    let r_flush = BlkRequest::from_bio(4, 0, 0, Bio::flush());

    if eq.insert_or_merge(r1, 8) != MergeResult::Inserted {
        return Err("first insert should not merge");
    }
    if eq.insert_or_merge(r2, 8) != MergeResult::MergedBack {
        return Err("contiguous tail should back-merge");
    }
    if eq.insert_or_merge(r3, 8) != MergeResult::MergedFront {
        return Err("contiguous head should front-merge");
    }
    if eq.insert_or_merge(r_flush, 8) != MergeResult::Inserted {
        return Err("flush should not merge");
    }
    if eq.len() != 2 {
        return Err("expected 2 requests after merging");
    }
    let merged = eq
        .pop_next(SchedulerPolicy::None, 0)
        .ok_or("missing merged request")?;
    if merged.lba != 9 || merged.sector_count != 3 || merged.merge_count != 3 {
        return Err("merged request metadata mismatch");
    }
    if merged.data[0] != 0x99
        || merged.data[SECTOR_SIZE] != 0xAA
        || merged.data[2 * SECTOR_SIZE] != 0xBB
    {
        return Err("merged write payload order mismatch");
    }

    // 3. Elevator C-SCAN ordering
    let mut elev = ElevatorQueue::new();
    for (id, lba) in [(1, 100), (2, 20), (3, 60), (4, 10), (5, 80)] {
        elev.insert_or_merge(BlkRequest::from_bio(id, 0, 0, Bio::read(lba, 1)), 8);
    }
    let mut order = Vec::new();
    let mut cursor = 50u64;
    while let Some(req) = elev.pop_next(SchedulerPolicy::Elevator, cursor) {
        order.push(req.lba);
        cursor = req.end_lba();
    }
    if order != [60, 80, 100, 10, 20] {
        return Err("elevator C-SCAN order mismatch");
    }

    // 4. Deadline priority ordering
    let mut dl = ElevatorQueue::new();
    dl.insert_or_merge(
        BlkRequest::from_bio(1, 0, 0, Bio::read(10, 1).with_priority(IoPriority::Idle)),
        8,
    );
    dl.insert_or_merge(
        BlkRequest::from_bio(
            2,
            0,
            0,
            Bio::read(30, 1).with_priority(IoPriority::BestEffort),
        ),
        8,
    );
    dl.insert_or_merge(
        BlkRequest::from_bio(
            3,
            0,
            0,
            Bio::read(50, 1).with_priority(IoPriority::RealTime),
        ),
        8,
    );
    let p0 = dl
        .pop_next(SchedulerPolicy::Deadline, 0)
        .ok_or("missing rt")?;
    let p1 = dl
        .pop_next(SchedulerPolicy::Deadline, 0)
        .ok_or("missing be")?;
    let p2 = dl
        .pop_next(SchedulerPolicy::Deadline, 0)
        .ok_or("missing idle")?;
    if p0.priority != IoPriority::RealTime
        || p1.priority != IoPriority::BestEffort
        || p2.priority != IoPriority::Idle
    {
        return Err("deadline priority order mismatch");
    }

    // 5. End-to-end BlkMqQueue with MemBlockDevice
    let dev = MemBlockDevice::new(128);
    let mq = BlkMqQueue::new(BlkMqConfig {
        num_cpus: 4,
        num_hw_queues: 2,
        queue_depth: 16,
        max_merge_sectors: 16,
        policy: SchedulerPolicy::Elevator,
    })
    .map_err(|_| "BlkMqQueue init failed")?;

    mq.submit_bio(0, Bio::write(4, vec![0x11; SECTOR_SIZE]))
        .map_err(|_| "submit write 1 failed")?;
    mq.submit_bio(0, Bio::write(5, vec![0x22; SECTOR_SIZE]))
        .map_err(|_| "submit write 2 failed")?;
    mq.submit_bio(1, Bio::write(20, vec![0x33; SECTOR_SIZE]))
        .map_err(|_| "submit write 3 failed")?;

    let write_done = mq.run_all(&dev).map_err(|_| "run_all writes failed")?;
    if write_done.len() != 2 {
        return Err("expected 2 write requests after coalescing");
    }

    mq.submit_bio(2, Bio::read(4, 2))
        .map_err(|_| "submit read 1 failed")?;
    mq.submit_bio(3, Bio::read(20, 1))
        .map_err(|_| "submit read 2 failed")?;

    let read_done = mq.run_all(&dev).map_err(|_| "run_all reads failed")?;
    if read_done.len() != 2 {
        return Err("expected 2 read completions");
    }
    for req in &read_done {
        if req.lba == 4 {
            if req.data.len() != 2 * SECTOR_SIZE
                || req.data[0] != 0x11
                || req.data[SECTOR_SIZE] != 0x22
            {
                return Err("read-after-write merged data mismatch");
            }
        } else if req.lba == 20 {
            if req.data.len() != SECTOR_SIZE || req.data[0] != 0x33 {
                return Err("read-after-write single data mismatch");
            }
        } else {
            return Err("unexpected LBA in read completions");
        }
    }

    // 6. Async pop_dispatch + complete_tag lifecycle
    mq.submit_bio(0, Bio::read(0, 1))
        .map_err(|_| "submit async read failed")?;
    mq.flush_sw_queue(0);
    let dispatched = mq.pop_dispatch(0).ok_or("expected async dispatch")?;
    let tag = dispatched.tag.ok_or("expected assigned tag")?;
    if mq.stats().in_flight != 1 {
        return Err("expected 1 in-flight request");
    }
    mq.complete_tag(0, tag, Ok(()))
        .map_err(|_| "complete_tag failed")?;

    let final_stats = mq.stats();
    if final_stats.in_flight != 0
        || final_stats.staged != 0
        || final_stats.queued != 0
        || final_stats.errors != 0
    {
        return Err("final blk-mq stats not clean");
    }

    serial_println!(
        "[blk-mq] self_test passed (submitted={}, merged={}, completed={})",
        final_stats.submitted,
        final_stats.merged,
        final_stats.completed
    );
    Ok(())
}
