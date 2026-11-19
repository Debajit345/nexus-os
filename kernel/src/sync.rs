//! Kernel synchronization primitives, futex wait-queue table, and unified event multiplexer.
//!
//! Provides:
//! - [`TicketLock`]: Fair FIFO ticket spinlock.
//! - [`RwLock`]: Writer-preferring reader-writer spinlock.
//! - [`FutexTable`] / [`FUTEX_TABLE`]: Hash-bucketed wait-queue table backing Linux `futex`,
//!   Win32 `WaitOnAddress`, and Mach `ulock`.
//! - [`EventMux`] / [`EventMuxTable`] / [`EVENT_MUX_TABLE`]: Unified event multiplexer backing
//!   Linux `epoll`, Windows `IOCP`, and BSD/macOS `kqueue`.
#![allow(dead_code)]

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use nexus_abi::AbiError;

use crate::sched::{TaskId, SCHEDULER};

// ---------------------------------------------------------------------------
// TicketLock (Fair FIFO Spinlock)
// ---------------------------------------------------------------------------

/// Fair FIFO ticket spinlock for `no_std` kernel synchronization.
pub struct TicketLock<T> {
    next_ticket: AtomicU64,
    now_serving: AtomicU64,
    inner: UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for TicketLock<T> {}
unsafe impl<T: Send> Send for TicketLock<T> {}

impl<T> TicketLock<T> {
    /// Create a new unlocked `TicketLock` wrapping `val`.
    pub const fn new(val: T) -> Self {
        Self {
            next_ticket: AtomicU64::new(0),
            now_serving: AtomicU64::new(0),
            inner: UnsafeCell::new(val),
        }
    }

    /// Acquire the lock, spinning fairly in FIFO ticket order.
    pub fn lock(&self) -> TicketLockGuard<'_, T> {
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        while self.now_serving.load(Ordering::Acquire) != ticket {
            core::hint::spin_loop();
        }
        TicketLockGuard { lock: self }
    }

    /// Attempt to acquire the lock without spinning.
    pub fn try_lock(&self) -> Option<TicketLockGuard<'_, T>> {
        let current = self.now_serving.load(Ordering::Acquire);
        if self
            .next_ticket
            .compare_exchange(current, current + 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(TicketLockGuard { lock: self })
        } else {
            None
        }
    }

    /// Return `true` if the lock is currently held.
    #[inline]
    pub fn is_locked(&self) -> bool {
        self.next_ticket.load(Ordering::Relaxed) != self.now_serving.load(Ordering::Acquire)
    }
}

/// RAII guard for [`TicketLock`].
pub struct TicketLockGuard<'a, T> {
    lock: &'a TicketLock<T>,
}

impl<T> Deref for TicketLockGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.inner.get() }
    }
}

impl<T> DerefMut for TicketLockGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.inner.get() }
    }
}

impl<T> Drop for TicketLockGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock.now_serving.fetch_add(1, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// RwLock (Writer-Preferring Reader-Writer Spinlock)
// ---------------------------------------------------------------------------

const RW_WRITER_ACTIVE: usize = 1 << (usize::BITS - 1);
const RW_WRITER_PENDING: usize = 1 << (usize::BITS - 2);
const RW_READER_MASK: usize = !(RW_WRITER_ACTIVE | RW_WRITER_PENDING);

/// Writer-preferring reader-writer spinlock for `no_std` kernel data structures.
pub struct RwLock<T> {
    state: AtomicUsize,
    inner: UnsafeCell<T>,
}

unsafe impl<T: Send + Sync> Sync for RwLock<T> {}
unsafe impl<T: Send> Send for RwLock<T> {}

impl<T> RwLock<T> {
    /// Create a new unlocked `RwLock` wrapping `val`.
    pub const fn new(val: T) -> Self {
        Self {
            state: AtomicUsize::new(0),
            inner: UnsafeCell::new(val),
        }
    }

    /// Acquire a shared read lock, spinning until no writer is active or pending.
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        loop {
            if let Some(guard) = self.try_read() {
                return guard;
            }
            core::hint::spin_loop();
        }
    }

    /// Attempt to acquire a shared read lock without spinning.
    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, T>> {
        let cur = self.state.load(Ordering::Relaxed);
        if (cur & (RW_WRITER_ACTIVE | RW_WRITER_PENDING)) != 0 {
            return None;
        }
        let readers = cur & RW_READER_MASK;
        if readers == RW_READER_MASK {
            return None;
        }
        if self
            .state
            .compare_exchange_weak(cur, cur + 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(RwLockReadGuard { lock: self })
        } else {
            None
        }
    }

    /// Acquire an exclusive write lock, setting the writer-pending bit to prevent starvation.
    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        loop {
            let cur = self.state.load(Ordering::Relaxed);
            if (cur & (RW_WRITER_ACTIVE | RW_READER_MASK)) == 0 {
                let next = (cur & !RW_WRITER_PENDING) | RW_WRITER_ACTIVE;
                if self
                    .state
                    .compare_exchange_weak(cur, next, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    return RwLockWriteGuard { lock: self };
                }
            } else if (cur & RW_WRITER_PENDING) == 0 {
                let _ = self.state.compare_exchange_weak(
                    cur,
                    cur | RW_WRITER_PENDING,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
            core::hint::spin_loop();
        }
    }

    /// Attempt to acquire an exclusive write lock without spinning.
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, T>> {
        let cur = self.state.load(Ordering::Relaxed);
        if (cur & (RW_WRITER_ACTIVE | RW_READER_MASK)) != 0 {
            return None;
        }
        let next = (cur & !RW_WRITER_PENDING) | RW_WRITER_ACTIVE;
        if self
            .state
            .compare_exchange(cur, next, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(RwLockWriteGuard { lock: self })
        } else {
            None
        }
    }

    /// Return the number of active readers.
    #[inline]
    pub fn reader_count(&self) -> usize {
        self.state.load(Ordering::Relaxed) & RW_READER_MASK
    }

    /// Return `true` if a writer currently holds the lock.
    #[inline]
    pub fn is_write_locked(&self) -> bool {
        (self.state.load(Ordering::Relaxed) & RW_WRITER_ACTIVE) != 0
    }
}

/// RAII shared read guard for [`RwLock`].
pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockReadGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.inner.get() }
    }
}

impl<T> Drop for RwLockReadGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock.state.fetch_sub(1, Ordering::Release);
    }
}

/// RAII exclusive write guard for [`RwLock`].
pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.inner.get() }
    }
}

impl<T> DerefMut for RwLockWriteGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.inner.get() }
    }
}

impl<T> Drop for RwLockWriteGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock
            .state
            .fetch_and(!RW_WRITER_ACTIVE, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Futex Wait-Queue Table (Linux futex / Win32 WaitOnAddress / Mach ulock)
// ---------------------------------------------------------------------------

const FUTEX_BUCKETS: usize = 64;

/// Key identifying a futex wait address (process-private or shared physical).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FutexKey {
    /// Address-space / process ID (`0` for shared physical futexes).
    pub space_id: u64,
    /// Virtual (private) or physical (shared) word address.
    pub addr: u64,
}

impl FutexKey {
    /// Create a process-private futex key.
    #[inline]
    pub const fn private(space_id: u64, addr: u64) -> Self {
        Self { space_id, addr }
    }

    /// Create a shared (cross-process) futex key.
    #[inline]
    pub const fn shared(phys_addr: u64) -> Self {
        Self {
            space_id: 0,
            addr: phys_addr,
        }
    }

    #[inline]
    const fn bucket_index(self) -> usize {
        // SplitMix64-style mixing for uniform bucket distribution.
        let mut h = self.space_id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.addr;
        h ^= h >> 30;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 27;
        (h as usize) & (FUTEX_BUCKETS - 1)
    }
}

/// Outcome of a futex wait registration attempt.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FutexWaitResult {
    /// Task was enqueued on the wait queue (caller should block the task).
    Enqueued,
    /// Observed value did not match expected value (`EAGAIN` / immediate return).
    ValueMismatch,
}

/// A single task waiting on a futex key.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FutexWaiter {
    pub task_id: TaskId,
    pub key: FutexKey,
    pub bitset: u32,
    pub deadline_ns: Option<u64>,
}

struct FutexBucket {
    waiters: VecDeque<FutexWaiter>,
}

impl FutexBucket {
    const fn new() -> Self {
        Self {
            waiters: VecDeque::new(),
        }
    }
}

/// Hash-bucketed futex wait-queue table supporting Linux `futex`, Win32 `WaitOnAddress`,
/// and Mach `ulock`.
pub struct FutexTable {
    buckets: [TicketLock<FutexBucket>; FUTEX_BUCKETS],
}

impl FutexTable {
    /// Create an empty `FutexTable`.
    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const EMPTY_BUCKET: TicketLock<FutexBucket> = TicketLock::new(FutexBucket::new());
        Self {
            buckets: [EMPTY_BUCKET; FUTEX_BUCKETS],
        }
    }

    /// Enqueue `task_id` on `key` if `current_val == expected` (32-bit Linux `futex` / Mach `ulock`).
    pub fn wait_u32(
        &self,
        key: FutexKey,
        expected: u32,
        current_val: u32,
        bitset: u32,
        task_id: TaskId,
        deadline_ns: Option<u64>,
    ) -> FutexWaitResult {
        if current_val != expected {
            return FutexWaitResult::ValueMismatch;
        }
        let idx = key.bucket_index();
        let mut bucket = self.buckets[idx].lock();
        bucket.waiters.push_back(FutexWaiter {
            task_id,
            key,
            bitset: if bitset == 0 { u32::MAX } else { bitset },
            deadline_ns,
        });
        FutexWaitResult::Enqueued
    }

    /// Enqueue `task_id` on `key` if `current_val == expected` (64-bit Win32 `WaitOnAddress` / `ulock64`).
    pub fn wait_u64(
        &self,
        key: FutexKey,
        expected: u64,
        current_val: u64,
        task_id: TaskId,
        deadline_ns: Option<u64>,
    ) -> FutexWaitResult {
        if current_val != expected {
            return FutexWaitResult::ValueMismatch;
        }
        let idx = key.bucket_index();
        let mut bucket = self.buckets[idx].lock();
        bucket.waiters.push_back(FutexWaiter {
            task_id,
            key,
            bitset: u32::MAX,
            deadline_ns,
        });
        FutexWaitResult::Enqueued
    }

    /// Wake up to `max_wake` waiters on `key` whose bitset intersects `bitset`.
    /// Returns the woken `TaskId`s in FIFO order.
    pub fn wake(&self, key: FutexKey, max_wake: usize, bitset: u32) -> Vec<TaskId> {
        if max_wake == 0 {
            return Vec::new();
        }
        let mask = if bitset == 0 { u32::MAX } else { bitset };
        let idx = key.bucket_index();
        let mut bucket = self.buckets[idx].lock();
        let mut woken = Vec::new();
        let mut remaining = VecDeque::with_capacity(bucket.waiters.len());

        while let Some(waiter) = bucket.waiters.pop_front() {
            if woken.len() < max_wake && waiter.key == key && (waiter.bitset & mask) != 0 {
                woken.push(waiter.task_id);
            } else {
                remaining.push_back(waiter);
            }
        }
        bucket.waiters = remaining;
        woken
    }

    /// Wake up to `max_wake` waiters on `key` and unblock them in the global [`SCHEDULER`].
    pub fn wake_and_unblock(&self, key: FutexKey, max_wake: usize, bitset: u32) -> usize {
        let woken = self.wake(key, max_wake, bitset);
        let count = woken.len();
        if count > 0 {
            let mut sched = SCHEDULER.lock();
            for tid in woken {
                sched.unblock(tid);
            }
        }
        count
    }

    /// Wake up to `max_wake` waiters on `src`, then requeue up to `max_requeue` remaining
    /// waiters from `src` to `dst` (`FUTEX_REQUEUE` / `FUTEX_CMP_REQUEUE`).
    /// Returns `(woken_tasks, requeued_count)`.
    pub fn requeue(
        &self,
        src: FutexKey,
        dst: FutexKey,
        max_wake: usize,
        max_requeue: usize,
    ) -> (Vec<TaskId>, usize) {
        if src == dst {
            let woken = self.wake(src, max_wake, u32::MAX);
            return (woken, 0);
        }

        let src_idx = src.bucket_index();
        let dst_idx = dst.bucket_index();

        if src_idx == dst_idx {
            let mut bucket = self.buckets[src_idx].lock();
            let mut woken = Vec::new();
            let mut requeued = 0usize;
            let mut remaining = VecDeque::with_capacity(bucket.waiters.len());

            while let Some(mut waiter) = bucket.waiters.pop_front() {
                if waiter.key == src {
                    if woken.len() < max_wake {
                        woken.push(waiter.task_id);
                    } else if requeued < max_requeue {
                        waiter.key = dst;
                        requeued += 1;
                        remaining.push_back(waiter);
                    } else {
                        remaining.push_back(waiter);
                    }
                } else {
                    remaining.push_back(waiter);
                }
            }
            bucket.waiters = remaining;
            return (woken, requeued);
        }

        // Lock buckets in ascending index order to prevent AB-BA deadlock.
        let (first_idx, second_idx) = if src_idx < dst_idx {
            (src_idx, dst_idx)
        } else {
            (dst_idx, src_idx)
        };
        let mut first_guard = self.buckets[first_idx].lock();
        let mut second_guard = self.buckets[second_idx].lock();

        let (src_bucket, dst_bucket) = if src_idx < dst_idx {
            (&mut *first_guard, &mut *second_guard)
        } else {
            (&mut *second_guard, &mut *first_guard)
        };

        let mut woken = Vec::new();
        let mut requeued = 0usize;
        let mut remaining = VecDeque::with_capacity(src_bucket.waiters.len());

        while let Some(mut waiter) = src_bucket.waiters.pop_front() {
            if waiter.key == src {
                if woken.len() < max_wake {
                    woken.push(waiter.task_id);
                } else if requeued < max_requeue {
                    waiter.key = dst;
                    requeued += 1;
                    dst_bucket.waiters.push_back(waiter);
                } else {
                    remaining.push_back(waiter);
                }
            } else {
                remaining.push_back(waiter);
            }
        }
        src_bucket.waiters = remaining;
        (woken, requeued)
    }

    /// Remove and return all waiters whose `deadline_ns <= now_ns`.
    pub fn expire_timeouts(&self, now_ns: u64) -> Vec<TaskId> {
        let mut expired = Vec::new();
        for bucket_lock in &self.buckets {
            let mut bucket = bucket_lock.lock();
            if bucket.waiters.is_empty() {
                continue;
            }
            let mut remaining = VecDeque::with_capacity(bucket.waiters.len());
            while let Some(waiter) = bucket.waiters.pop_front() {
                if let Some(deadline) = waiter.deadline_ns {
                    if deadline <= now_ns {
                        expired.push(waiter.task_id);
                        continue;
                    }
                }
                remaining.push_back(waiter);
            }
            bucket.waiters = remaining;
        }
        expired
    }

    /// Remove all pending futex waits for `task_id` (e.g. on task exit or signal interrupt).
    pub fn cancel_task(&self, task_id: TaskId) -> usize {
        let mut removed = 0usize;
        for bucket_lock in &self.buckets {
            let mut bucket = bucket_lock.lock();
            if bucket.waiters.is_empty() {
                continue;
            }
            let before = bucket.waiters.len();
            bucket.waiters.retain(|w| w.task_id != task_id);
            removed += before - bucket.waiters.len();
        }
        removed
    }

    /// Count waiters currently queued on `key`.
    pub fn waiter_count(&self, key: FutexKey) -> usize {
        let bucket = self.buckets[key.bucket_index()].lock();
        bucket.waiters.iter().filter(|w| w.key == key).count()
    }

    /// Count total waiters across all buckets.
    pub fn total_waiters(&self) -> usize {
        self.buckets.iter().map(|b| b.lock().waiters.len()).sum()
    }
}

/// Global kernel futex wait-queue table.
pub static FUTEX_TABLE: FutexTable = FutexTable::new();

// ---------------------------------------------------------------------------
// Unified Event Multiplexer (Epoll / IOCP / Kqueue)
// ---------------------------------------------------------------------------

/// Personality backend flavor for an [`EventMux`] instance.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MuxBackend {
    Native,
    Epoll,
    Iocp,
    Kqueue,
}

/// Bitmask of event interests and readiness flags.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EventFlags(u32);

impl EventFlags {
    pub const NONE: Self = Self(0);
    pub const READABLE: Self = Self(1 << 0);
    pub const WRITABLE: Self = Self(1 << 1);
    pub const ERROR: Self = Self(1 << 2);
    pub const HANGUP: Self = Self(1 << 3);
    pub const EDGE_TRIGGERED: Self = Self(1 << 4);
    pub const ONESHOT: Self = Self(1 << 5);
    pub const IOCP_COMPLETION: Self = Self(1 << 6);
    pub const TIMER: Self = Self(1 << 7);
    pub const USER: Self = Self(1 << 8);

    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

/// Event returned by [`EventMux::poll`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MuxEvent {
    pub handle: u64,
    pub events: EventFlags,
    pub user_data: u64,
    pub bytes_transferred: usize,
    pub completion_key: u64,
}

#[derive(Copy, Clone, Debug)]
struct MuxRegistration {
    handle: u64,
    interest: EventFlags,
    pending: EventFlags,
    user_data: u64,
    armed: bool,
}

#[derive(Copy, Clone, Debug)]
struct IocpPacket {
    completion_key: u64,
    bytes_transferred: usize,
    user_data: u64,
}

/// Unified event multiplexer supporting Linux `epoll`, Windows `IOCP`, and BSD/macOS `kqueue`.
pub struct EventMux {
    backend: MuxBackend,
    registrations: BTreeMap<u64, MuxRegistration>,
    iocp_queue: VecDeque<IocpPacket>,
    waiters: VecDeque<TaskId>,
}

impl EventMux {
    /// Create a new `EventMux` configured for `backend`.
    pub const fn new(backend: MuxBackend) -> Self {
        Self {
            backend,
            registrations: BTreeMap::new(),
            iocp_queue: VecDeque::new(),
            waiters: VecDeque::new(),
        }
    }

    /// Return the personality backend of this multiplexer.
    #[inline]
    pub const fn backend(&self) -> MuxBackend {
        self.backend
    }

    /// Register a handle/fd with interest flags and opaque `user_data` (`EPOLL_CTL_ADD` / `EV_ADD`).
    pub fn ctl_add(
        &mut self,
        handle: u64,
        interest: EventFlags,
        user_data: u64,
    ) -> Result<(), AbiError> {
        if self.registrations.contains_key(&handle) {
            return Err(AbiError::AlreadyExists);
        }
        self.registrations.insert(
            handle,
            MuxRegistration {
                handle,
                interest,
                pending: EventFlags::NONE,
                user_data,
                armed: true,
            },
        );
        Ok(())
    }

    /// Modify an existing registration (`EPOLL_CTL_MOD` / re-arm `ONESHOT`).
    pub fn ctl_mod(
        &mut self,
        handle: u64,
        interest: EventFlags,
        user_data: u64,
    ) -> Result<(), AbiError> {
        let reg = self
            .registrations
            .get_mut(&handle)
            .ok_or(AbiError::NotFound)?;
        reg.interest = interest;
        reg.user_data = user_data;
        reg.armed = true;
        Ok(())
    }

    /// Remove a registered handle (`EPOLL_CTL_DEL` / `EV_DELETE`).
    pub fn ctl_del(&mut self, handle: u64) -> Result<(), AbiError> {
        self.registrations
            .remove(&handle)
            .map(|_| ())
            .ok_or(AbiError::NotFound)
    }

    /// Signal readiness on `handle` and return any blocked waiter `TaskId`s to wake.
    pub fn notify(&mut self, handle: u64, ready: EventFlags) -> Vec<TaskId> {
        let Some(reg) = self.registrations.get_mut(&handle) else {
            return Vec::new();
        };
        if !reg.armed {
            return Vec::new();
        }
        // ERROR and HANGUP are always reported even if not explicitly requested (epoll semantics).
        let effective_mask = reg
            .interest
            .union(EventFlags::ERROR)
            .union(EventFlags::HANGUP);
        let matched = ready.intersect(effective_mask);
        if matched.is_empty() {
            return Vec::new();
        }
        reg.pending = reg.pending.union(matched);
        self.take_waiters()
    }

    /// Clear level-triggered readiness bits on `handle` (e.g. when a socket/pipe buffer drains).
    pub fn clear_ready(&mut self, handle: u64, mask: EventFlags) {
        if let Some(reg) = self.registrations.get_mut(&handle) {
            reg.pending = reg.pending.difference(mask);
        }
    }

    /// Post an asynchronous completion packet (Windows IOCP `PostQueuedCompletionStatus`).
    pub fn post_completion(
        &mut self,
        completion_key: u64,
        bytes_transferred: usize,
        user_data: u64,
    ) -> Vec<TaskId> {
        self.iocp_queue.push_back(IocpPacket {
            completion_key,
            bytes_transferred,
            user_data,
        });
        self.take_waiters()
    }

    /// Poll up to `max_events` ready events or IOCP completion packets.
    pub fn poll(&mut self, max_events: usize) -> Vec<MuxEvent> {
        if max_events == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();

        // 1. Drain IOCP completion packets first.
        while out.len() < max_events {
            let Some(pkt) = self.iocp_queue.pop_front() else {
                break;
            };
            out.push(MuxEvent {
                handle: pkt.completion_key,
                events: EventFlags::IOCP_COMPLETION,
                user_data: pkt.user_data,
                bytes_transferred: pkt.bytes_transferred,
                completion_key: pkt.completion_key,
            });
        }

        // 2. Collect ready registrations.
        for reg in self.registrations.values_mut() {
            if out.len() >= max_events {
                break;
            }
            if !reg.armed || reg.pending.is_empty() {
                continue;
            }
            out.push(MuxEvent {
                handle: reg.handle,
                events: reg.pending,
                user_data: reg.user_data,
                bytes_transferred: 0,
                completion_key: 0,
            });

            if reg.interest.contains(EventFlags::ONESHOT) {
                reg.armed = false;
                reg.pending = EventFlags::NONE;
            } else if reg.interest.contains(EventFlags::EDGE_TRIGGERED) {
                reg.pending = EventFlags::NONE;
            }
        }

        out
    }

    /// Register a task to be woken when new events or completions arrive.
    pub fn register_waiter(&mut self, task_id: TaskId) {
        if !self.waiters.contains(&task_id) {
            self.waiters.push_back(task_id);
        }
    }

    /// Drain all registered waiter `TaskId`s.
    pub fn take_waiters(&mut self) -> Vec<TaskId> {
        self.waiters.drain(..).collect()
    }

    /// Return the number of registered handles.
    #[inline]
    pub fn registration_count(&self) -> usize {
        self.registrations.len()
    }

    /// Return the number of queued IOCP completion packets.
    #[inline]
    pub fn pending_completion_count(&self) -> usize {
        self.iocp_queue.len()
    }
}

/// Registry of kernel [`EventMux`] instances indexed by handle ID.
pub struct EventMuxTable {
    muxes: BTreeMap<u64, EventMux>,
    next_id: u64,
}

impl EventMuxTable {
    /// Create an empty `EventMuxTable`.
    pub const fn new() -> Self {
        Self {
            muxes: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Allocate a new [`EventMux`] and return its ID.
    pub fn create(&mut self, backend: MuxBackend) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.muxes.insert(id, EventMux::new(backend));
        id
    }

    /// Look up a mutable reference to an [`EventMux`] by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut EventMux> {
        self.muxes.get_mut(&id)
    }

    /// Destroy an [`EventMux`] by ID.
    pub fn destroy(&mut self, id: u64) -> Result<(), AbiError> {
        self.muxes
            .remove(&id)
            .map(|_| ())
            .ok_or(AbiError::InvalidHandle)
    }

    /// Return the number of active [`EventMux`] instances.
    #[inline]
    pub fn count(&self) -> usize {
        self.muxes.len()
    }
}

/// Global kernel event multiplexer registry.
pub static EVENT_MUX_TABLE: TicketLock<EventMuxTable> = TicketLock::new(EventMuxTable::new());

// ---------------------------------------------------------------------------
// Self-Test
// ---------------------------------------------------------------------------

/// Run kernel synchronization, futex, and event multiplexer self-tests.
pub fn self_test() -> bool {
    // 1. TicketLock
    let tlock = TicketLock::new(10u64);
    if tlock.is_locked() {
        crate::serial_println!("sync::self_test: TicketLock initially locked");
        return false;
    }
    {
        let mut g = tlock.lock();
        if !tlock.is_locked() || tlock.try_lock().is_some() {
            crate::serial_println!("sync::self_test: TicketLock contention check failed");
            return false;
        }
        *g += 32;
    }
    if tlock.is_locked() || *tlock.lock() != 42 {
        crate::serial_println!("sync::self_test: TicketLock value/unlock check failed");
        return false;
    }

    // 2. RwLock
    let rw = RwLock::new(100u32);
    {
        let r1 = rw.read();
        let r2 = rw.read();
        if *r1 != 100 || *r2 != 100 || rw.reader_count() != 2 || rw.try_write().is_some() {
            crate::serial_println!("sync::self_test: RwLock shared read check failed");
            return false;
        }
    }
    {
        let mut w = rw.write();
        if !rw.is_write_locked() || rw.try_read().is_some() {
            crate::serial_println!("sync::self_test: RwLock exclusive write check failed");
            return false;
        }
        *w = 250;
    }
    if *rw.read() != 250 {
        crate::serial_println!("sync::self_test: RwLock post-write read check failed");
        return false;
    }

    // 3. FutexTable (Linux futex / Win32 WaitOnAddress / Mach ulock)
    let ft = FutexTable::new();
    let k1 = FutexKey::private(1, 0x1000);
    let k2 = FutexKey::shared(0x2000);

    if ft.wait_u32(k1, 1, 2, u32::MAX, TaskId(10), None) != FutexWaitResult::ValueMismatch {
        crate::serial_println!("sync::self_test: Futex value mismatch check failed");
        return false;
    }
    if ft.wait_u32(k1, 1, 1, 0x01, TaskId(11), Some(500)) != FutexWaitResult::Enqueued
        || ft.wait_u32(k1, 1, 1, 0x02, TaskId(12), Some(1000)) != FutexWaitResult::Enqueued
        || ft.wait_u64(k2, 99, 99, TaskId(13), None) != FutexWaitResult::Enqueued
    {
        crate::serial_println!("sync::self_test: Futex enqueue check failed");
        return false;
    }

    let woken_bitset = ft.wake(k1, 4, 0x02);
    if woken_bitset != [TaskId(12)] {
        crate::serial_println!("sync::self_test: Futex bitset wake check failed");
        return false;
    }

    let (woken_req, req_count) = ft.requeue(k1, k2, 0, 1);
    if !woken_req.is_empty() || req_count != 1 || ft.waiter_count(k2) != 2 {
        crate::serial_println!("sync::self_test: Futex requeue check failed");
        return false;
    }

    let expired = ft.expire_timeouts(600);
    if expired != [TaskId(11)] {
        crate::serial_println!("sync::self_test: Futex timeout expiration check failed");
        return false;
    }
    if ft.cancel_task(TaskId(13)) != 1 || ft.total_waiters() != 0 {
        crate::serial_println!("sync::self_test: Futex cancel/cleanup check failed");
        return false;
    }

    // 4. EventMux (Epoll / IOCP / Kqueue)
    let mut table = EventMuxTable::new();
    let epoll_id = table.create(MuxBackend::Epoll);
    let iocp_id = table.create(MuxBackend::Iocp);

    if let Some(ep) = table.get_mut(epoll_id) {
        if ep.ctl_add(3, EventFlags::READABLE, 0xAA).is_err()
            || ep
                .ctl_add(
                    4,
                    EventFlags::READABLE.union(EventFlags::EDGE_TRIGGERED),
                    0xBB,
                )
                .is_err()
            || ep
                .ctl_add(5, EventFlags::WRITABLE.union(EventFlags::ONESHOT), 0xCC)
                .is_err()
        {
            crate::serial_println!("sync::self_test: EventMux ctl_add failed");
            return false;
        }
        ep.register_waiter(TaskId(20));
        let w1 = ep.notify(3, EventFlags::READABLE);
        let w2 = ep.notify(4, EventFlags::READABLE);
        let _ = ep.notify(5, EventFlags::WRITABLE);
        if w1 != [TaskId(20)] || !w2.is_empty() {
            crate::serial_println!("sync::self_test: EventMux waiter wake failed");
            return false;
        }

        let ev1 = ep.poll(8);
        if ev1.len() != 3 {
            crate::serial_println!("sync::self_test: EventMux first poll count mismatch");
            return false;
        }
        // Second poll: fd 3 (level-triggered) remains ready; fd 4 (ET) and fd 5 (ONESHOT) cleared.
        let ev2 = ep.poll(8);
        if ev2.len() != 1 || ev2[0].handle != 3 {
            crate::serial_println!("sync::self_test: EventMux level/ET/oneshot semantics failed");
            return false;
        }
        ep.clear_ready(3, EventFlags::READABLE);
        if !ep.poll(8).is_empty() {
            crate::serial_println!("sync::self_test: EventMux clear_ready failed");
            return false;
        }
    } else {
        return false;
    }

    if let Some(iocp) = table.get_mut(iocp_id) {
        iocp.post_completion(0x1234, 512, 0xDEAD);
        let ev = iocp.poll(4);
        if ev.len() != 1
            || ev[0].completion_key != 0x1234
            || ev[0].bytes_transferred != 512
            || ev[0].user_data != 0xDEAD
            || !ev[0].events.contains(EventFlags::IOCP_COMPLETION)
        {
            crate::serial_println!("sync::self_test: EventMux IOCP completion check failed");
            return false;
        }
    } else {
        return false;
    }

    if table.destroy(epoll_id).is_err() || table.destroy(iocp_id).is_err() || table.count() != 0 {
        crate::serial_println!("sync::self_test: EventMuxTable destroy check failed");
        return false;
    }

    crate::serial_println!("sync::self_test: all checks passed");
    true
}
