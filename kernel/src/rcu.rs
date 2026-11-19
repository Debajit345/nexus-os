//! RCU Grace-Period Reclamation & Lockdep Deadlock Cycle Detector.
//!
//! Provides:
//! - Epoch-based quiescent-state Read-Copy-Update (RCU) reclamation:
//!   - Per-CPU read-side critical section tracking (`rcu_read_lock` / `rcu_read_unlock`)
//!   - Quiescent-state reporting (`quiescent_state`) and grace-period advancement
//!   - Deferred callback queue (`call_rcu`, `call_rcu_boxed`, `poll_callbacks`, `synchronize_rcu`)
//!   - Atomic RCU pointer wrapper (`RcuPtr<T>`)
//! - Lockdep lock-order graph cycle detector:
//!   - Lock class registration (`register_lock_class`)
//!   - Per-CPU held-lock stack tracking (`lock_acquire`, `lock_release`)
//!   - Directed lock-order graph with iterative BFS cycle detection (catches AB-BA,
//!     transitive A->B->C->A cycles, and recursive self-deadlocks)
//! - `self_test()` verification suite
#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use crate::cpu::MAX_CPUS;
use crate::mm::SpinLock;

// ---------------------------------------------------------------------------
// RCU Grace-Period Reclamation
// ---------------------------------------------------------------------------

/// Maximum number of callbacks drained per batch to keep stack frames small.
const CALLBACK_BATCH_SIZE: usize = 16;

/// Per-CPU RCU tracking state.
struct PerCpuRcu {
    online: AtomicBool,
    nesting: AtomicU32,
    quiescent_epoch: AtomicU64,
    qs_count: AtomicU64,
}

impl PerCpuRcu {
    const fn new(online: bool) -> Self {
        Self {
            online: AtomicBool::new(online),
            nesting: AtomicU32::new(0),
            quiescent_epoch: AtomicU64::new(0),
            qs_count: AtomicU64::new(0),
        }
    }
}

/// A deferred RCU reclamation callback.
#[derive(Clone, Copy)]
struct RcuCallback {
    func: fn(*mut u8),
    arg: usize,
    target_epoch: u64,
}

struct RcuQueue {
    pending: VecDeque<RcuCallback>,
    reclaimed_total: u64,
}

impl RcuQueue {
    const fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            reclaimed_total: 0,
        }
    }
}

/// Snapshot of RCU subsystem statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RcuStats {
    pub global_epoch: u64,
    pub completed_epoch: u64,
    pub pending_callbacks: usize,
    pub reclaimed_callbacks: u64,
    pub online_cpus: usize,
}

static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(1);
static COMPLETED_EPOCH: AtomicU64 = AtomicU64::new(0);
static RCU_QUEUE: SpinLock<RcuQueue> = SpinLock::new(RcuQueue::new());

#[allow(clippy::declare_interior_mutable_const)]
const OFFLINE_CPU_RCU: PerCpuRcu = PerCpuRcu::new(false);

static PER_CPU_RCU: [PerCpuRcu; MAX_CPUS] = {
    let mut arr = [OFFLINE_CPU_RCU; MAX_CPUS];
    arr[0] = PerCpuRcu::new(true);
    arr
};

/// Mark a CPU as online for RCU quiescent-state tracking.
pub fn register_cpu(cpu_id: usize) {
    if cpu_id < MAX_CPUS {
        let epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
        PER_CPU_RCU[cpu_id]
            .quiescent_epoch
            .store(epoch, Ordering::Release);
        PER_CPU_RCU[cpu_id].nesting.store(0, Ordering::Release);
        PER_CPU_RCU[cpu_id].online.store(true, Ordering::Release);
    }
}

/// Mark a CPU as offline for RCU quiescent-state tracking and try advancing the grace period.
pub fn unregister_cpu(cpu_id: usize) {
    if cpu_id < MAX_CPUS && cpu_id != 0 {
        PER_CPU_RCU[cpu_id].online.store(false, Ordering::Release);
        try_advance_grace_period();
    }
}

/// RAII guard representing an active RCU read-side critical section on a CPU.
pub struct RcuReadGuard {
    cpu_id: usize,
}

impl Drop for RcuReadGuard {
    #[inline]
    fn drop(&mut self) {
        rcu_read_unlock(self.cpu_id);
    }
}

/// Enter an RCU read-side critical section on `cpu_id`.
#[inline]
pub fn rcu_read_lock(cpu_id: usize) -> RcuReadGuard {
    let id = if cpu_id < MAX_CPUS { cpu_id } else { 0 };
    PER_CPU_RCU[id].nesting.fetch_add(1, Ordering::Acquire);
    RcuReadGuard { cpu_id: id }
}

/// Exit an RCU read-side critical section on `cpu_id`.
#[inline]
pub fn rcu_read_unlock(cpu_id: usize) {
    let id = if cpu_id < MAX_CPUS { cpu_id } else { 0 };
    let prev = PER_CPU_RCU[id].nesting.fetch_sub(1, Ordering::Release);
    debug_assert!(prev > 0, "rcu_read_unlock underflow");
}

/// Report a quiescent state on `cpu_id` (e.g. context switch, idle loop, syscall boundary).
///
/// If the CPU is currently inside an RCU read-side critical section (`nesting > 0`),
/// the quiescent epoch is not updated.
pub fn quiescent_state(cpu_id: usize) {
    if cpu_id >= MAX_CPUS {
        return;
    }
    let cpu = &PER_CPU_RCU[cpu_id];
    if !cpu.online.load(Ordering::Acquire) {
        return;
    }
    if cpu.nesting.load(Ordering::Acquire) == 0 {
        let epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
        cpu.quiescent_epoch.store(epoch, Ordering::Release);
        cpu.qs_count.fetch_add(1, Ordering::Relaxed);
        try_advance_grace_period();
    }
}

/// Start a new RCU grace period by advancing the global epoch and returning the target epoch.
pub fn start_grace_period() -> u64 {
    GLOBAL_EPOCH.fetch_add(1, Ordering::AcqRel) + 1
}

/// Attempt to advance `COMPLETED_EPOCH` based on the minimum quiescent epoch of all online CPUs.
///
/// Returns the updated `completed_epoch`.
pub fn try_advance_grace_period() -> u64 {
    let global = GLOBAL_EPOCH.load(Ordering::Acquire);
    let mut min_qs_epoch = global;

    for cpu in &PER_CPU_RCU {
        if !cpu.online.load(Ordering::Acquire) {
            continue;
        }
        if cpu.nesting.load(Ordering::Acquire) > 0 {
            // CPU is actively in a read-side critical section; cannot advance past its last QS.
            let qs = cpu.quiescent_epoch.load(Ordering::Acquire);
            if qs < min_qs_epoch {
                min_qs_epoch = qs;
            }
        } else {
            let qs = cpu.quiescent_epoch.load(Ordering::Acquire);
            if qs < min_qs_epoch {
                min_qs_epoch = qs;
            }
        }
    }

    let mut completed = COMPLETED_EPOCH.load(Ordering::Acquire);
    while min_qs_epoch > completed {
        match COMPLETED_EPOCH.compare_exchange_weak(
            completed,
            min_qs_epoch,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return min_qs_epoch,
            Err(actual) => completed = actual,
        }
    }
    completed
}

/// Enqueue a callback to be invoked after a full RCU grace period has elapsed.
pub fn call_rcu(func: fn(*mut u8), arg: *mut u8) {
    let target_epoch = GLOBAL_EPOCH.fetch_add(1, Ordering::AcqRel);
    let cb = RcuCallback {
        func,
        arg: arg as usize,
        target_epoch,
    };
    let mut queue = RCU_QUEUE.lock();
    queue.pending.push_back(cb);
}

/// Enqueue a boxed value to be dropped after a full RCU grace period has elapsed.
pub fn call_rcu_boxed<T: 'static>(val: Box<T>) {
    fn drop_boxed<T>(ptr: *mut u8) {
        if !ptr.is_null() {
            unsafe {
                drop(Box::from_raw(ptr.cast::<T>()));
            }
        }
    }
    let raw = Box::into_raw(val).cast::<u8>();
    call_rcu(drop_boxed::<T>, raw);
}

/// Drain and execute all RCU callbacks whose target epoch has completed.
///
/// Returns the number of callbacks executed.
pub fn poll_callbacks() -> usize {
    try_advance_grace_period();
    let completed = COMPLETED_EPOCH.load(Ordering::Acquire);
    let mut executed = 0usize;

    loop {
        let mut batch: [Option<RcuCallback>; CALLBACK_BATCH_SIZE] = [None; CALLBACK_BATCH_SIZE];
        let mut count = 0usize;

        {
            let mut queue = RCU_QUEUE.lock();
            while count < CALLBACK_BATCH_SIZE {
                match queue.pending.front() {
                    Some(front) if front.target_epoch <= completed => {
                        batch[count] = queue.pending.pop_front();
                        count += 1;
                    }
                    _ => break,
                }
            }
            queue.reclaimed_total += count as u64;
        }

        if count == 0 {
            break;
        }

        for cb in batch.iter().take(count).flatten() {
            (cb.func)(cb.arg as *mut u8);
        }
        executed += count;
    }

    executed
}

/// Block (spin) until a full grace period completes for `cpu_id` and drain ready callbacks.
///
/// Note: `cpu_id` must not be inside an RCU read-side critical section.
pub fn synchronize_rcu(cpu_id: usize) {
    let target = GLOBAL_EPOCH.fetch_add(1, Ordering::AcqRel);
    for _ in 0..1_000_000u32 {
        quiescent_state(cpu_id);
        if COMPLETED_EPOCH.load(Ordering::Acquire) >= target {
            break;
        }
        core::hint::spin_loop();
    }
    poll_callbacks();
}

/// Return a snapshot of current RCU statistics.
pub fn rcu_stats() -> RcuStats {
    let online_cpus = PER_CPU_RCU
        .iter()
        .filter(|c| c.online.load(Ordering::Acquire))
        .count();
    let pending_callbacks = RCU_QUEUE.lock().pending.len();
    let reclaimed_callbacks = RCU_QUEUE.lock().reclaimed_total;
    RcuStats {
        global_epoch: GLOBAL_EPOCH.load(Ordering::Acquire),
        completed_epoch: COMPLETED_EPOCH.load(Ordering::Acquire),
        pending_callbacks,
        reclaimed_callbacks,
        online_cpus,
    }
}

/// Atomic pointer wrapper for RCU-protected data publication and dereferencing.
pub struct RcuPtr<T> {
    ptr: AtomicPtr<T>,
}

impl<T> RcuPtr<T> {
    /// Create a new `RcuPtr` initialized with `ptr`.
    pub const fn new(ptr: *mut T) -> Self {
        Self {
            ptr: AtomicPtr::new(ptr),
        }
    }

    /// Dereference the pointer within an active RCU read-side critical section.
    ///
    /// # Safety
    /// The caller must ensure any non-null pointer stored in `self` remains valid
    /// for the lifetime of `_guard` (i.e. reclaimed only after an RCU grace period).
    #[inline]
    pub unsafe fn dereference<'a>(&self, _guard: &'a RcuReadGuard) -> Option<&'a T> {
        let p = self.ptr.load(Ordering::Acquire);
        if p.is_null() {
            None
        } else {
            Some(unsafe { &*p })
        }
    }

    /// Publish a new pointer with Release ordering and return the previous pointer.
    #[inline]
    pub fn assign(&self, new_ptr: *mut T) -> *mut T {
        self.ptr.swap(new_ptr, Ordering::AcqRel)
    }

    /// Load the raw pointer with Acquire ordering.
    #[inline]
    pub fn load_raw(&self) -> *mut T {
        self.ptr.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Lockdep Deadlock Cycle Detector
// ---------------------------------------------------------------------------

/// Maximum number of distinct lock classes tracked by Lockdep.
pub const MAX_LOCK_CLASSES: usize = 64;

/// Maximum lock nesting depth per CPU.
pub const MAX_HELD_LOCKS: usize = 16;

/// Identifier for a registered lock class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LockClassId(pub u16);

/// Errors reported by the Lockdep cycle detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockdepError {
    /// Too many lock classes registered (exceeds `MAX_LOCK_CLASSES`).
    TooManyClasses,
    /// Invalid or unregistered lock class ID.
    InvalidClass,
    /// Invalid CPU ID.
    InvalidCpu,
    /// Per-CPU held-lock stack overflow (exceeds `MAX_HELD_LOCKS`).
    StackOverflow,
    /// Attempted to acquire a lock class already held on the same CPU.
    RecursiveLock(LockClassId),
    /// Acquiring `to` while holding `from` would create a cycle in the lock-order graph.
    CycleDetected { from: LockClassId, to: LockClassId },
    /// Attempted to release a lock class that is not currently held on this CPU.
    NotHeld(LockClassId),
}

/// Directed lock-order graph and per-CPU held-lock stack tracker.
pub struct LockdepGraph {
    num_classes: usize,
    names: [&'static str; MAX_LOCK_CLASSES],
    /// Adjacency bitmask: bit `j` in `edges[i]` means lock `i` was acquired before lock `j`.
    edges: [u64; MAX_LOCK_CLASSES],
    /// Per-CPU stack of currently held lock class IDs.
    held: [[u16; MAX_HELD_LOCKS]; MAX_CPUS],
    held_len: [u8; MAX_CPUS],
    violations: u64,
}

impl LockdepGraph {
    /// Create an empty `LockdepGraph`.
    pub const fn new() -> Self {
        Self {
            num_classes: 0,
            names: [""; MAX_LOCK_CLASSES],
            edges: [0; MAX_LOCK_CLASSES],
            held: [[0; MAX_HELD_LOCKS]; MAX_CPUS],
            held_len: [0; MAX_CPUS],
            violations: 0,
        }
    }

    /// Reset all classes, edges, and held stacks.
    pub fn reset(&mut self) {
        self.num_classes = 0;
        self.names = [""; MAX_LOCK_CLASSES];
        self.edges = [0; MAX_LOCK_CLASSES];
        self.held = [[0; MAX_HELD_LOCKS]; MAX_CPUS];
        self.held_len = [0; MAX_CPUS];
        self.violations = 0;
    }

    /// Register a new lock class and return its `LockClassId`.
    pub fn register_class(&mut self, name: &'static str) -> Result<LockClassId, LockdepError> {
        if self.num_classes >= MAX_LOCK_CLASSES {
            return Err(LockdepError::TooManyClasses);
        }
        let id = self.num_classes as u16;
        self.names[self.num_classes] = name;
        self.num_classes += 1;
        Ok(LockClassId(id))
    }

    /// Return the name of a registered lock class, if valid.
    pub fn class_name(&self, class: LockClassId) -> Option<&'static str> {
        let idx = class.0 as usize;
        if idx < self.num_classes {
            Some(self.names[idx])
        } else {
            None
        }
    }

    /// Check whether a directed path `from ->* to` exists in the lock-order graph
    /// using an iterative BFS with a 64-bit visited bitmask.
    pub fn has_path(&self, from: LockClassId, to: LockClassId) -> bool {
        let start = from.0 as usize;
        let target = to.0 as usize;
        if start >= self.num_classes || target >= self.num_classes {
            return false;
        }
        if start == target {
            return true;
        }

        let mut visited: u64 = 1u64 << start;
        let mut queue = [0u8; MAX_LOCK_CLASSES];
        let mut head = 0usize;
        let mut tail = 0usize;

        queue[tail] = start as u8;
        tail += 1;

        while head < tail {
            let curr = queue[head] as usize;
            head += 1;

            let mut neighbors = self.edges[curr];
            while neighbors != 0 {
                let next = neighbors.trailing_zeros() as usize;
                neighbors &= neighbors - 1;

                if next == target {
                    return true;
                }
                let bit = 1u64 << next;
                if (visited & bit) == 0 {
                    visited |= bit;
                    if tail < MAX_LOCK_CLASSES {
                        queue[tail] = next as u8;
                        tail += 1;
                    }
                }
            }
        }

        false
    }

    /// Record acquisition of `class` on `cpu_id`, checking for recursive locks and cycles.
    pub fn acquire(&mut self, cpu_id: usize, class: LockClassId) -> Result<(), LockdepError> {
        if cpu_id >= MAX_CPUS {
            return Err(LockdepError::InvalidCpu);
        }
        let cls_idx = class.0 as usize;
        if cls_idx >= self.num_classes {
            return Err(LockdepError::InvalidClass);
        }

        let depth = self.held_len[cpu_id] as usize;
        for i in 0..depth {
            if self.held[cpu_id][i] == class.0 {
                self.violations += 1;
                return Err(LockdepError::RecursiveLock(class));
            }
        }

        if depth >= MAX_HELD_LOCKS {
            return Err(LockdepError::StackOverflow);
        }

        // Check if adding edges `held_lock -> class` would create a cycle (`class ->* held_lock`).
        for i in 0..depth {
            let held_id = LockClassId(self.held[cpu_id][i]);
            if self.has_path(class, held_id) {
                self.violations += 1;
                return Err(LockdepError::CycleDetected {
                    from: held_id,
                    to: class,
                });
            }
        }

        // No cycle: record directed edges `held_lock -> class`.
        let to_bit = 1u64 << cls_idx;
        for i in 0..depth {
            let from_idx = self.held[cpu_id][i] as usize;
            self.edges[from_idx] |= to_bit;
        }

        self.held[cpu_id][depth] = class.0;
        self.held_len[cpu_id] = (depth + 1) as u8;
        Ok(())
    }

    /// Record release of `class` on `cpu_id`.
    ///
    /// Supports LIFO release as well as out-of-order release of a held lock.
    pub fn release(&mut self, cpu_id: usize, class: LockClassId) -> Result<(), LockdepError> {
        if cpu_id >= MAX_CPUS {
            return Err(LockdepError::InvalidCpu);
        }
        let depth = self.held_len[cpu_id] as usize;
        if depth == 0 {
            return Err(LockdepError::NotHeld(class));
        }

        // Fast path: LIFO top of stack.
        if self.held[cpu_id][depth - 1] == class.0 {
            self.held_len[cpu_id] = (depth - 1) as u8;
            return Ok(());
        }

        // Search stack for out-of-order release.
        for i in (0..depth).rev() {
            if self.held[cpu_id][i] == class.0 {
                for j in i..(depth - 1) {
                    self.held[cpu_id][j] = self.held[cpu_id][j + 1];
                }
                self.held_len[cpu_id] = (depth - 1) as u8;
                return Ok(());
            }
        }

        Err(LockdepError::NotHeld(class))
    }

    /// Return the number of detected lockdep violations.
    pub fn violation_count(&self) -> u64 {
        self.violations
    }

    /// Return the number of registered lock classes.
    pub fn class_count(&self) -> usize {
        self.num_classes
    }
}

static LOCKDEP: SpinLock<LockdepGraph> = SpinLock::new(LockdepGraph::new());

/// Register a lock class in the global Lockdep graph.
pub fn register_lock_class(name: &'static str) -> Result<LockClassId, LockdepError> {
    LOCKDEP.lock().register_class(name)
}

/// Record lock acquisition in the global Lockdep graph.
pub fn lock_acquire(cpu_id: usize, class: LockClassId) -> Result<(), LockdepError> {
    LOCKDEP.lock().acquire(cpu_id, class)
}

/// Record lock release in the global Lockdep graph.
pub fn lock_release(cpu_id: usize, class: LockClassId) -> Result<(), LockdepError> {
    LOCKDEP.lock().release(cpu_id, class)
}

/// Check if a directed lock-order path exists in the global Lockdep graph.
pub fn lockdep_has_path(from: LockClassId, to: LockClassId) -> bool {
    LOCKDEP.lock().has_path(from, to)
}

/// Return the total number of Lockdep violations recorded globally.
pub fn lockdep_violations() -> u64 {
    LOCKDEP.lock().violation_count()
}

// ---------------------------------------------------------------------------
// Self-Test Suite
// ---------------------------------------------------------------------------

static SELF_TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn test_rcu_callback(arg: *mut u8) {
    let val = arg as usize;
    SELF_TEST_COUNTER.fetch_add(val, Ordering::AcqRel);
}

/// Run self-tests for RCU grace-period reclamation and Lockdep cycle detection.
pub fn self_test() -> Result<(), &'static str> {
    // 1. RCU Grace-Period & Quiescent-State Reclamation Test
    SELF_TEST_COUNTER.store(0, Ordering::Release);
    register_cpu(0);
    register_cpu(1);

    // CPU 1 enters a read-side critical section.
    let guard1 = rcu_read_lock(1);

    // Queue a callback with argument 42.
    call_rcu(test_rcu_callback, 42usize as *mut u8);

    // CPU 0 reports a quiescent state, but CPU 1 is still in a read-side critical section.
    quiescent_state(0);
    let ran_early = poll_callbacks();
    if ran_early != 0 || SELF_TEST_COUNTER.load(Ordering::Acquire) != 0 {
        drop(guard1);
        unregister_cpu(1);
        return Err("rcu: callback executed before all CPUs exited read-side critical section");
    }

    // CPU 1 exits read-side critical section and reports quiescent state.
    drop(guard1);
    quiescent_state(1);
    let ran_after = poll_callbacks();
    if ran_after == 0 || SELF_TEST_COUNTER.load(Ordering::Acquire) != 42 {
        unregister_cpu(1);
        return Err("rcu: callback did not execute after grace period completed");
    }

    // Test RcuPtr publish & dereference.
    let val_a: u64 = 0xDEAD_BEEF;
    let val_b: u64 = 0xCAFE_BABE;
    let rcu_ptr = RcuPtr::new(&val_a as *const u64 as *mut u64);
    {
        let g = rcu_read_lock(0);
        let deref = unsafe { rcu_ptr.dereference(&g) };
        if deref != Some(&0xDEAD_BEEF) {
            unregister_cpu(1);
            return Err("rcu: RcuPtr initial dereference mismatch");
        }
    }
    let old = rcu_ptr.assign(&val_b as *const u64 as *mut u64);
    if !core::ptr::eq(old, &val_a) {
        unregister_cpu(1);
        return Err("rcu: RcuPtr assign did not return previous pointer");
    }
    {
        let g = rcu_read_lock(0);
        let deref = unsafe { rcu_ptr.dereference(&g) };
        if deref != Some(&0xCAFE_BABE) {
            unregister_cpu(1);
            return Err("rcu: RcuPtr updated dereference mismatch");
        }
    }

    unregister_cpu(1);

    // 2. Lockdep Deadlock Cycle Detector Test (heap-allocated to keep stack frame < 2 KiB)
    let mut graph = Box::new(LockdepGraph::new());
    let a = graph
        .register_class("lock_a")
        .map_err(|_| "lockdep: failed to register lock_a")?;
    let b = graph
        .register_class("lock_b")
        .map_err(|_| "lockdep: failed to register lock_b")?;
    let c = graph
        .register_class("lock_c")
        .map_err(|_| "lockdep: failed to register lock_c")?;
    let d = graph
        .register_class("lock_d")
        .map_err(|_| "lockdep: failed to register lock_d")?;

    // Establish A -> B
    graph
        .acquire(0, a)
        .map_err(|_| "lockdep: acquire A failed")?;
    graph
        .acquire(0, b)
        .map_err(|_| "lockdep: acquire B under A failed")?;
    graph
        .release(0, b)
        .map_err(|_| "lockdep: release B failed")?;
    graph
        .release(0, a)
        .map_err(|_| "lockdep: release A failed")?;

    // Establish B -> C
    graph
        .acquire(0, b)
        .map_err(|_| "lockdep: acquire B failed")?;
    graph
        .acquire(0, c)
        .map_err(|_| "lockdep: acquire C under B failed")?;
    graph
        .release(0, c)
        .map_err(|_| "lockdep: release C failed")?;
    graph
        .release(0, b)
        .map_err(|_| "lockdep: release B failed")?;

    if !graph.has_path(a, c) {
        return Err("lockdep: transitive path A -> C missing");
    }

    // Test 2-lock AB-BA cycle: hold B, try A
    graph
        .acquire(0, b)
        .map_err(|_| "lockdep: acquire B for AB-BA test failed")?;
    match graph.acquire(0, a) {
        Err(LockdepError::CycleDetected { from, to }) if from == b && to == a => {}
        _ => return Err("lockdep: failed to detect 2-lock AB-BA cycle"),
    }
    graph
        .release(0, b)
        .map_err(|_| "lockdep: release B after AB-BA test failed")?;

    // Test 3-lock A -> B -> C -> A cycle: hold C, try A
    graph
        .acquire(0, c)
        .map_err(|_| "lockdep: acquire C for 3-lock cycle test failed")?;
    match graph.acquire(0, a) {
        Err(LockdepError::CycleDetected { from, to }) if from == c && to == a => {}
        _ => return Err("lockdep: failed to detect 3-lock A->B->C->A cycle"),
    }
    graph
        .release(0, c)
        .map_err(|_| "lockdep: release C after 3-lock cycle test failed")?;

    // Test recursive self-deadlock: hold D, try D
    graph
        .acquire(0, d)
        .map_err(|_| "lockdep: acquire D failed")?;
    match graph.acquire(0, d) {
        Err(LockdepError::RecursiveLock(cls)) if cls == d => {}
        _ => return Err("lockdep: failed to detect recursive lock"),
    }
    graph
        .release(0, d)
        .map_err(|_| "lockdep: release D failed")?;

    // Test releasing unheld lock
    if graph.release(0, d) != Err(LockdepError::NotHeld(d)) {
        return Err("lockdep: releasing unheld lock did not return NotHeld");
    }

    if graph.violation_count() != 3 {
        return Err("lockdep: unexpected violation count");
    }

    Ok(())
}
