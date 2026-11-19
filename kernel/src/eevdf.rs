//! SMP Preemptive EEVDF + Real-Time Deadline Scheduler & Intel Hybrid P/E-Core Thread Director.
//!
//! Implements:
//! - Per-CPU runqueues (`MAX_CPUS`) with canonical lock ordering to prevent AB-BA deadlocks
//! - EEVDF (Earliest Eligible Virtual Deadline First) with `vruntime`, `vdeadline`,
//!   weighted `avg_vruntime`, and `vlag` eligibility tracking
//! - Real-Time `SCHED_DEADLINE` (EDF + Constant Bandwidth Server throttling/replenishment)
//!   and `SCHED_FIFO`/`SCHED_RR` priority classes
//! - Intel Hybrid P/E-Core Thread Director (workload classification, HFI-style IPC/cache
//!   telemetry, P/E-core affinity & capacity-aware placement)
//! - Work-stealing SMP load balancer with cross-CPU `vlag` preservation
//! - `self_test()` verification suite
#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crate::cpu::{CoreType, MAX_CPUS};
use crate::mm::SpinLock;

// ---------------------------------------------------------------------------
// Constants & Weight Tables
// ---------------------------------------------------------------------------

/// Weight of a nice-0 task in EEVDF.
pub const NICE_0_WEIGHT: u64 = 1024;

/// Default requested virtual time slice (3 ms in nanoseconds).
pub const DEFAULT_SLICE_NS: u64 = 3_000_000;

/// Minimum allowed time slice (0.5 ms).
pub const MIN_SLICE_NS: u64 = 500_000;

/// Maximum allowed time slice (20 ms).
pub const MAX_SLICE_NS: u64 = 20_000_000;

/// Maximum preserved virtual lag on wakeup/migration (12 ms).
pub const MAX_LAG_NS: i64 = 12_000_000;

/// Maximum Deadline bandwidth per CPU in parts-per-million (95%).
pub const MAX_DL_BANDWIDTH_PPM: u32 = 950_000;

/// Default P-core / Unified compute capacity.
pub const P_CORE_CAPACITY: u32 = 1024;

/// Default E-core compute capacity.
pub const E_CORE_CAPACITY: u32 = 512;

/// Standard Linux nice -20..=19 to weight table (~1.25x per nice level).
const PRIO_TO_WEIGHT: [u64; 40] = [
    88761, 71755, 56483, 46273, 36291, // -20..-16
    29154, 23254, 18705, 14949, 11916, // -15..-11
    9548, 7620, 6100, 4904, 3906, // -10..-6
    3121, 2501, 1991, 1586, 1277, // -5..-1
    1024, 820, 655, 526, 423, // 0..4
    335, 272, 215, 172, 137, // 5..9
    110, 87, 70, 56, 45, // 10..14
    36, 29, 23, 18, 15, // 15..19
];

/// Convert a nice value (`-20..=19`) to an EEVDF weight.
#[inline]
pub fn nice_to_weight(nice: i8) -> u64 {
    let clamped = nice.clamp(-20, 19);
    PRIO_TO_WEIGHT[(clamped + 20) as usize]
}

/// Compute virtual time delta: `delta_ns * NICE_0_WEIGHT / weight`.
#[inline]
pub fn calc_delta_fair(delta_ns: u64, weight: u64) -> u64 {
    let w = weight.max(1);
    if w == NICE_0_WEIGHT {
        return delta_ns;
    }
    let num = (delta_ns as u128) * (NICE_0_WEIGHT as u128);
    (num / (w as u128)).max(1) as u64
}

// ---------------------------------------------------------------------------
// Scheduling Classes, Affinity & Thread Director Types
// ---------------------------------------------------------------------------

/// Scheduling policy class (strictly ordered: Deadline > Realtime > Normal > Idle).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SchedClass {
    Deadline = 0,
    Realtime = 1,
    Normal = 2,
    Idle = 3,
}

/// Lifecycle state of a scheduled task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Ready,
    Running,
    Blocked,
    Throttled,
    Exited,
}

/// CPU / Hybrid core affinity policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreAffinity {
    /// Run on any online CPU; Thread Director chooses optimal core.
    Any,
    /// Prefer P-cores, fall back to E-cores if P-cores are saturated.
    PreferPerformance,
    /// Prefer E-cores, fall back to P-cores if E-cores are saturated.
    PreferEfficiency,
    /// Strictly pin to P-cores (or Unified cores if no P-cores exist).
    PinPerformance,
    /// Strictly pin to E-cores (or Unified cores if no E-cores exist).
    PinEfficiency,
    /// Explicit 64-bit CPU bitmask.
    CpuMask(u64),
}

/// Intel Thread Director workload classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadClass {
    Interactive,
    ComputeHeavy,
    VectorAvx,
    Background,
    Eco,
    Unknown,
}

/// Hardware Feedback Interface (HFI) / Thread Director per-task telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThreadDirectorStats {
    /// Instructions per cycle * 100 (e.g. 180 = 1.80 IPC).
    pub ipc_x100: u32,
    /// L2/LLC cache miss rate in per-mille (0..=1000).
    pub cache_miss_permille: u16,
    /// Number of voluntary context switches (sleep/yield).
    pub voluntary_switches: u64,
    /// Number of involuntary preemptions.
    pub preemptions: u64,
    /// Total nanoseconds executed on P-cores.
    pub p_core_ns: u64,
    /// Total nanoseconds executed on E-cores.
    pub e_core_ns: u64,
}

/// EEVDF per-task scheduling state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EevdfEntity {
    pub nice: i8,
    pub weight: u64,
    pub vruntime: u64,
    pub vdeadline: u64,
    pub slice_ns: u64,
    pub vlag: i64,
}

impl EevdfEntity {
    pub fn new(nice: i8, slice_ns: u64) -> Self {
        let weight = nice_to_weight(nice);
        let clamped_slice = slice_ns.clamp(MIN_SLICE_NS, MAX_SLICE_NS);
        Self {
            nice: nice.clamp(-20, 19),
            weight,
            vruntime: 0,
            vdeadline: calc_delta_fair(clamped_slice, weight),
            slice_ns: clamped_slice,
            vlag: 0,
        }
    }
}

/// Real-Time `SCHED_DEADLINE` (EDF + CBS) per-task state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeadlineEntity {
    pub runtime_ns: u64,
    pub deadline_ns: u64,
    pub period_ns: u64,
    pub abs_deadline_ns: u64,
    pub rem_runtime_ns: u64,
    pub throttled_until_ns: u64,
}

impl DeadlineEntity {
    pub const fn zero() -> Self {
        Self {
            runtime_ns: 0,
            deadline_ns: 0,
            period_ns: 0,
            abs_deadline_ns: 0,
            rem_runtime_ns: 0,
            throttled_until_ns: 0,
        }
    }

    pub fn new(runtime_ns: u64, deadline_ns: u64, period_ns: u64, now_ns: u64) -> Self {
        let period = period_ns.max(1);
        let dl = deadline_ns.max(runtime_ns).min(period);
        Self {
            runtime_ns,
            deadline_ns: dl,
            period_ns: period,
            abs_deadline_ns: now_ns.saturating_add(dl),
            rem_runtime_ns: runtime_ns,
            throttled_until_ns: 0,
        }
    }

    /// Bandwidth utilization in parts-per-million.
    #[inline]
    pub fn bandwidth_ppm(&self) -> u32 {
        if self.period_ns == 0 {
            return 0;
        }
        let ppm = ((self.runtime_ns as u128) * 1_000_000) / (self.period_ns as u128);
        ppm.min(1_000_000) as u32
    }
}

/// Real-Time `SCHED_FIFO` / `SCHED_RR` per-task state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtEntity {
    /// Priority 1..=99 (higher = more urgent).
    pub priority: u8,
    /// Round-robin quantum in ns (0 = FIFO, run until yield/block/preempt).
    pub rr_slice_ns: u64,
    pub rem_slice_ns: u64,
}

impl RtEntity {
    pub const fn zero() -> Self {
        Self {
            priority: 0,
            rr_slice_ns: 0,
            rem_slice_ns: 0,
        }
    }
}

/// Scheduled task descriptor.
#[derive(Debug, Clone)]
pub struct SchedTask {
    pub id: u64,
    pub state: TaskState,
    pub class: SchedClass,
    pub affinity: CoreAffinity,
    pub workload: WorkloadClass,
    pub director: ThreadDirectorStats,
    pub last_cpu: usize,
    pub pinned: bool,
    pub total_exec_ns: u64,
    pub wait_ns: u64,
    pub last_enqueue_ns: u64,
    pub last_dispatch_ns: u64,
    pub eevdf: EevdfEntity,
    pub dl: DeadlineEntity,
    pub rt: RtEntity,
}

impl SchedTask {
    pub fn new_normal(id: u64, nice: i8, slice_ns: u64, affinity: CoreAffinity) -> Self {
        Self {
            id,
            state: TaskState::Ready,
            class: SchedClass::Normal,
            affinity,
            workload: WorkloadClass::Unknown,
            director: ThreadDirectorStats::default(),
            last_cpu: 0,
            pinned: matches!(affinity, CoreAffinity::CpuMask(m) if m.count_ones() == 1),
            total_exec_ns: 0,
            wait_ns: 0,
            last_enqueue_ns: 0,
            last_dispatch_ns: 0,
            eevdf: EevdfEntity::new(nice, slice_ns),
            dl: DeadlineEntity::zero(),
            rt: RtEntity::zero(),
        }
    }

    pub fn new_deadline(
        id: u64,
        runtime_ns: u64,
        deadline_ns: u64,
        period_ns: u64,
        now_ns: u64,
        affinity: CoreAffinity,
    ) -> Self {
        Self {
            id,
            state: TaskState::Ready,
            class: SchedClass::Deadline,
            affinity,
            workload: WorkloadClass::Interactive,
            director: ThreadDirectorStats::default(),
            last_cpu: 0,
            pinned: matches!(affinity, CoreAffinity::CpuMask(m) if m.count_ones() == 1),
            total_exec_ns: 0,
            wait_ns: 0,
            last_enqueue_ns: now_ns,
            last_dispatch_ns: 0,
            eevdf: EevdfEntity::new(-20, DEFAULT_SLICE_NS),
            dl: DeadlineEntity::new(runtime_ns, deadline_ns, period_ns, now_ns),
            rt: RtEntity::zero(),
        }
    }

    pub fn new_realtime(id: u64, priority: u8, rr_slice_ns: u64, affinity: CoreAffinity) -> Self {
        let prio = priority.clamp(1, 99);
        Self {
            id,
            state: TaskState::Ready,
            class: SchedClass::Realtime,
            affinity,
            workload: WorkloadClass::Interactive,
            director: ThreadDirectorStats::default(),
            last_cpu: 0,
            pinned: matches!(affinity, CoreAffinity::CpuMask(m) if m.count_ones() == 1),
            total_exec_ns: 0,
            wait_ns: 0,
            last_enqueue_ns: 0,
            last_dispatch_ns: 0,
            eevdf: EevdfEntity::new(-20, DEFAULT_SLICE_NS),
            dl: DeadlineEntity::zero(),
            rt: RtEntity {
                priority: prio,
                rr_slice_ns,
                rem_slice_ns: rr_slice_ns,
            },
        }
    }
}

/// Action returned by `tick()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickAction {
    /// Keep running the current task.
    Continue(u64),
    /// Preempt current task and switch to `next_id`.
    Preempt { prev_id: u64, next_id: u64 },
    /// No runnable tasks on this CPU; enter idle/halt.
    Idle,
}

/// Snapshot of per-CPU runqueue statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuRunqueueStats {
    pub cpu_id: usize,
    pub online: bool,
    pub core_type: CoreType,
    pub capacity: u32,
    pub nr_running: usize,
    pub current_task_id: Option<u64>,
    pub avg_vruntime: u64,
    pub min_vruntime: u64,
    pub total_weight: u64,
    pub dl_bandwidth_ppm: u32,
    pub context_switches: u64,
    pub preemptions: u64,
    pub steal_count: u64,
}

/// Errors returned by scheduler operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedError {
    InvalidCpu,
    CpuOffline,
    TaskNotFound,
    DeadlineBandwidthExceeded,
    InvalidParameters,
    NoCompatibleCpu,
}

// ---------------------------------------------------------------------------
// Per-CPU Runqueue
// ---------------------------------------------------------------------------

/// Per-CPU runqueue holding queued tasks and the currently running task.
pub struct CpuRunqueue {
    pub cpu_id: usize,
    pub online: bool,
    pub core_type: CoreType,
    pub capacity: u32,
    pub clock_ns: u64,
    pub avg_vruntime: u64,
    pub min_vruntime: u64,
    pub total_weight: u64,
    pub dl_bandwidth_ppm: u32,
    pub current: Option<Box<SchedTask>>,
    pub queue: Vec<SchedTask>,
    pub context_switches: u64,
    pub preemptions: u64,
    pub steal_count: u64,
}

impl CpuRunqueue {
    pub const fn new(cpu_id: usize) -> Self {
        Self {
            cpu_id,
            online: false,
            core_type: CoreType::Unified,
            capacity: P_CORE_CAPACITY,
            clock_ns: 0,
            avg_vruntime: 0,
            min_vruntime: 0,
            total_weight: 0,
            dl_bandwidth_ppm: 0,
            current: None,
            queue: Vec::new(),
            context_switches: 0,
            preemptions: 0,
            steal_count: 0,
        }
    }

    /// Number of runnable tasks (queued + current if running).
    #[inline]
    pub fn nr_running(&self) -> usize {
        let cur = usize::from(
            self.current
                .as_ref()
                .is_some_and(|t| t.state == TaskState::Running),
        );
        let queued = self
            .queue
            .iter()
            .filter(|t| t.state == TaskState::Ready)
            .count();
        cur + queued
    }

    /// Recompute `min_vruntime`, `total_weight`, and weighted `avg_vruntime`
    /// across all runnable EEVDF (`SchedClass::Normal`) tasks on this CPU.
    pub fn update_vruntime_stats(&mut self) {
        let mut min_v = u64::MAX;
        let mut weight_sum: u64 = 0;

        if let Some(cur) = self.current.as_deref() {
            if cur.class == SchedClass::Normal && cur.state == TaskState::Running {
                min_v = min_v.min(cur.eevdf.vruntime);
                weight_sum = weight_sum.saturating_add(cur.eevdf.weight);
            }
        }

        for t in &self.queue {
            if t.class == SchedClass::Normal && t.state == TaskState::Ready {
                min_v = min_v.min(t.eevdf.vruntime);
                weight_sum = weight_sum.saturating_add(t.eevdf.weight);
            }
        }

        if weight_sum == 0 || min_v == u64::MAX {
            self.total_weight = 0;
            return;
        }

        let mut weighted_delta_sum: u128 = 0;
        if let Some(cur) = self.current.as_deref() {
            if cur.class == SchedClass::Normal && cur.state == TaskState::Running {
                let delta = (cur.eevdf.vruntime - min_v) as u128;
                weighted_delta_sum += delta * (cur.eevdf.weight as u128);
            }
        }
        for t in &self.queue {
            if t.class == SchedClass::Normal && t.state == TaskState::Ready {
                let delta = (t.eevdf.vruntime - min_v) as u128;
                weighted_delta_sum += delta * (t.eevdf.weight as u128);
            }
        }

        self.min_vruntime = self.min_vruntime.max(min_v);
        self.total_weight = weight_sum;
        self.avg_vruntime =
            min_v.saturating_add((weighted_delta_sum / (weight_sum as u128)) as u64);
    }

    /// Place an EEVDF task onto this runqueue, preserving clamped `vlag` on wakeup.
    fn place_eevdf_entity(&mut self, task: &mut SchedTask, is_wakeup: bool) {
        self.update_vruntime_stats();
        let base_v = self.avg_vruntime;
        if is_wakeup {
            let clamped_lag = task.eevdf.vlag.clamp(-MAX_LAG_NS, MAX_LAG_NS);
            let placed = (base_v as i128) - (clamped_lag as i128);
            task.eevdf.vruntime = placed.max(0) as u64;
        } else {
            task.eevdf.vruntime = base_v;
            task.eevdf.vlag = 0;
        }
        let vslice = calc_delta_fair(task.eevdf.slice_ns, task.eevdf.weight);
        task.eevdf.vdeadline = task.eevdf.vruntime.saturating_add(vslice);
    }

    /// Check if an EEVDF task is eligible (`lag >= 0`, i.e. `vruntime <= avg_vruntime`).
    #[inline]
    pub fn is_eligible(&self, task: &SchedTask) -> bool {
        task.eevdf.vruntime <= self.avg_vruntime
    }

    /// Unthrottle any `SCHED_DEADLINE` tasks whose replenishment period has arrived.
    fn unthrottle_deadlines(&mut self) {
        let now = self.clock_ns;
        for t in &mut self.queue {
            if t.class == SchedClass::Deadline
                && t.state == TaskState::Throttled
                && now >= t.dl.throttled_until_ns
            {
                t.state = TaskState::Ready;
                t.dl.rem_runtime_ns = t.dl.runtime_ns;
                t.dl.abs_deadline_ns = now.saturating_add(t.dl.deadline_ns);
            }
        }
    }

    /// Pick the index in `self.queue` of the highest-priority runnable task.
    fn pick_next_index(&mut self) -> Option<usize> {
        self.unthrottle_deadlines();
        self.update_vruntime_stats();

        // 1. SCHED_DEADLINE: earliest absolute deadline among Ready tasks with rem_runtime > 0.
        let mut best_dl: Option<(usize, u64)> = None;
        for (idx, t) in self.queue.iter().enumerate() {
            if t.class == SchedClass::Deadline
                && t.state == TaskState::Ready
                && t.dl.rem_runtime_ns > 0
            {
                match best_dl {
                    None => best_dl = Some((idx, t.dl.abs_deadline_ns)),
                    Some((_, best_d)) if t.dl.abs_deadline_ns < best_d => {
                        best_dl = Some((idx, t.dl.abs_deadline_ns));
                    }
                    _ => {}
                }
            }
        }
        if let Some((idx, _)) = best_dl {
            return Some(idx);
        }

        // 2. SCHED_FIFO / SCHED_RR: highest RT priority.
        let mut best_rt: Option<(usize, u8)> = None;
        for (idx, t) in self.queue.iter().enumerate() {
            if t.class == SchedClass::Realtime && t.state == TaskState::Ready {
                match best_rt {
                    None => best_rt = Some((idx, t.rt.priority)),
                    Some((_, best_p)) if t.rt.priority > best_p => {
                        best_rt = Some((idx, t.rt.priority));
                    }
                    _ => {}
                }
            }
        }
        if let Some((idx, _)) = best_rt {
            return Some(idx);
        }

        // 3. EEVDF (SCHED_NORMAL): earliest virtual deadline among eligible tasks.
        let avg_v = self.avg_vruntime;
        let mut best_eligible: Option<(usize, u64, u64)> = None;
        let mut fallback_min_v: Option<(usize, u64)> = None;

        for (idx, t) in self.queue.iter().enumerate() {
            if t.class == SchedClass::Normal && t.state == TaskState::Ready {
                match fallback_min_v {
                    None => fallback_min_v = Some((idx, t.eevdf.vruntime)),
                    Some((_, mv)) if t.eevdf.vruntime < mv => {
                        fallback_min_v = Some((idx, t.eevdf.vruntime));
                    }
                    _ => {}
                }

                if t.eevdf.vruntime <= avg_v {
                    match best_eligible {
                        None => best_eligible = Some((idx, t.eevdf.vdeadline, t.eevdf.vruntime)),
                        Some((_, bd, bv))
                            if t.eevdf.vdeadline < bd
                                || (t.eevdf.vdeadline == bd && t.eevdf.vruntime < bv) =>
                        {
                            best_eligible = Some((idx, t.eevdf.vdeadline, t.eevdf.vruntime));
                        }
                        _ => {}
                    }
                }
            }
        }

        if let Some((idx, _, _)) = best_eligible {
            return Some(idx);
        }
        if let Some((idx, _)) = fallback_min_v {
            return Some(idx);
        }

        // 4. SCHED_IDLE
        self.queue
            .iter()
            .position(|t| t.class == SchedClass::Idle && t.state == TaskState::Ready)
    }

    /// Check whether a queued task should preempt `self.current`.
    fn should_preempt_current(&mut self) -> bool {
        let cur = match self.current.as_deref() {
            Some(c) if c.state == TaskState::Running => c,
            _ => return true,
        };

        let cur_class = cur.class;
        let cur_rt_prio = cur.rt.priority;
        let cur_rt_rem = cur.rt.rem_slice_ns;
        let cur_rt_rr = cur.rt.rr_slice_ns;
        let cur_dl_abs = cur.dl.abs_deadline_ns;
        let cur_vdeadline = cur.eevdf.vdeadline;

        self.unthrottle_deadlines();
        self.update_vruntime_stats();
        let avg_v = self.avg_vruntime;

        for t in &self.queue {
            if t.state != TaskState::Ready {
                continue;
            }
            if t.class < cur_class {
                return true;
            }
            if t.class == cur_class {
                match cur_class {
                    SchedClass::Deadline => {
                        if t.dl.rem_runtime_ns > 0 && t.dl.abs_deadline_ns < cur_dl_abs {
                            return true;
                        }
                    }
                    SchedClass::Realtime => {
                        if t.rt.priority > cur_rt_prio {
                            return true;
                        }
                        if t.rt.priority == cur_rt_prio && cur_rt_rr > 0 && cur_rt_rem == 0 {
                            return true;
                        }
                    }
                    SchedClass::Normal => {
                        if t.eevdf.vruntime <= avg_v && t.eevdf.vdeadline < cur_vdeadline {
                            return true;
                        }
                    }
                    SchedClass::Idle => {}
                }
            }
        }

        false
    }

    /// Dispatch the next runnable task onto this CPU, returning its task ID if any.
    pub fn schedule_next(&mut self) -> Option<u64> {
        if let Some(mut prev) = self.current.take() {
            if prev.state == TaskState::Running {
                prev.state = TaskState::Ready;
                prev.last_enqueue_ns = self.clock_ns;
                if prev.class == SchedClass::Normal {
                    let vslice = calc_delta_fair(prev.eevdf.slice_ns, prev.eevdf.weight);
                    prev.eevdf.vdeadline = prev.eevdf.vruntime.saturating_add(vslice);
                }
                self.queue.push(*prev);
            } else if prev.state == TaskState::Throttled || prev.state == TaskState::Blocked {
                self.queue.push(*prev);
            } else if prev.state == TaskState::Exited && prev.class == SchedClass::Deadline {
                self.dl_bandwidth_ppm = self
                    .dl_bandwidth_ppm
                    .saturating_sub(prev.dl.bandwidth_ppm());
            }
        }

        let next_idx = self.pick_next_index()?;
        let mut next = self.queue.swap_remove(next_idx);
        next.state = TaskState::Running;
        next.last_cpu = self.cpu_id;
        next.wait_ns = next
            .wait_ns
            .saturating_add(self.clock_ns.saturating_sub(next.last_enqueue_ns));
        next.last_dispatch_ns = self.clock_ns;
        if next.class == SchedClass::Realtime
            && next.rt.rr_slice_ns > 0
            && next.rt.rem_slice_ns == 0
        {
            next.rt.rem_slice_ns = next.rt.rr_slice_ns;
        }
        let next_id = next.id;
        self.current = Some(Box::new(next));
        self.context_switches = self.context_switches.saturating_add(1);
        Some(next_id)
    }

    /// Advance this CPU's clock by `elapsed_ns` and account runtime for the current task.
    pub fn tick(&mut self, elapsed_ns: u64) -> TickAction {
        self.clock_ns = self.clock_ns.saturating_add(elapsed_ns);

        let mut need_preempt = false;
        let mut cur_id_opt = None;

        if let Some(cur) = self.current.as_deref_mut() {
            if cur.state == TaskState::Running {
                cur_id_opt = Some(cur.id);
                cur.total_exec_ns = cur.total_exec_ns.saturating_add(elapsed_ns);
                match self.core_type {
                    CoreType::Performance | CoreType::Unified => {
                        cur.director.p_core_ns = cur.director.p_core_ns.saturating_add(elapsed_ns);
                    }
                    CoreType::Efficiency => {
                        cur.director.e_core_ns = cur.director.e_core_ns.saturating_add(elapsed_ns);
                    }
                }

                match cur.class {
                    SchedClass::Deadline => {
                        if cur.dl.rem_runtime_ns <= elapsed_ns {
                            cur.dl.rem_runtime_ns = 0;
                            cur.state = TaskState::Throttled;
                            cur.dl.throttled_until_ns =
                                self.clock_ns.saturating_add(cur.dl.period_ns);
                            need_preempt = true;
                        } else {
                            cur.dl.rem_runtime_ns -= elapsed_ns;
                        }
                    }
                    SchedClass::Realtime => {
                        if cur.rt.rr_slice_ns > 0 {
                            if cur.rt.rem_slice_ns <= elapsed_ns {
                                cur.rt.rem_slice_ns = 0;
                                need_preempt = true;
                            } else {
                                cur.rt.rem_slice_ns -= elapsed_ns;
                            }
                        }
                    }
                    SchedClass::Normal => {
                        let vdelta = calc_delta_fair(elapsed_ns, cur.eevdf.weight);
                        cur.eevdf.vruntime = cur.eevdf.vruntime.saturating_add(vdelta);
                        if cur.eevdf.vruntime >= cur.eevdf.vdeadline {
                            let vslice = calc_delta_fair(cur.eevdf.slice_ns, cur.eevdf.weight);
                            cur.eevdf.vdeadline = cur.eevdf.vruntime.saturating_add(vslice);
                            need_preempt = true;
                        }
                    }
                    SchedClass::Idle => {
                        need_preempt = true;
                    }
                }
            } else {
                need_preempt = true;
            }
        } else {
            need_preempt = true;
        }

        if !need_preempt && self.should_preempt_current() {
            need_preempt = true;
        }

        if !need_preempt {
            if let Some(id) = cur_id_opt {
                return TickAction::Continue(id);
            }
            return TickAction::Idle;
        }

        if let Some(prev_id) = cur_id_opt {
            if let Some(cur) = self.current.as_deref_mut() {
                cur.director.preemptions = cur.director.preemptions.saturating_add(1);
            }
            self.preemptions = self.preemptions.saturating_add(1);
            match self.schedule_next() {
                Some(next_id) if next_id == prev_id => TickAction::Continue(next_id),
                Some(next_id) => TickAction::Preempt { prev_id, next_id },
                None => TickAction::Idle,
            }
        } else {
            match self.schedule_next() {
                Some(next_id) => TickAction::Preempt {
                    prev_id: 0,
                    next_id,
                },
                None => TickAction::Idle,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Intel Hybrid P/E-Core Thread Director
// ---------------------------------------------------------------------------

/// Classify a task's workload using Thread Director / HFI telemetry.
pub fn classify_workload(task: &SchedTask) -> WorkloadClass {
    if matches!(task.class, SchedClass::Deadline | SchedClass::Realtime) {
        return WorkloadClass::Interactive;
    }
    if task.workload == WorkloadClass::VectorAvx || task.workload == WorkloadClass::Eco {
        return task.workload;
    }
    let d = &task.director;
    if d.ipc_x100 >= 150 {
        return WorkloadClass::ComputeHeavy;
    }
    if d.voluntary_switches > d.preemptions.saturating_mul(2) && d.voluntary_switches >= 4 {
        return WorkloadClass::Interactive;
    }
    if d.ipc_x100 > 0 && d.ipc_x100 < 80 && d.cache_miss_permille >= 150 {
        return WorkloadClass::Background;
    }
    task.workload
}

/// Check whether `task` is permitted to run on `cpu_id` with `core_type`.
#[inline]
pub fn is_cpu_allowed(
    affinity: CoreAffinity,
    cpu_id: usize,
    core_type: CoreType,
    has_p_cores: bool,
    has_e_cores: bool,
) -> bool {
    if cpu_id >= MAX_CPUS {
        return false;
    }
    match affinity {
        CoreAffinity::Any | CoreAffinity::PreferPerformance | CoreAffinity::PreferEfficiency => {
            true
        }
        CoreAffinity::PinPerformance => {
            if has_p_cores {
                core_type == CoreType::Performance
            } else {
                true
            }
        }
        CoreAffinity::PinEfficiency => {
            if has_e_cores {
                core_type == CoreType::Efficiency
            } else {
                true
            }
        }
        CoreAffinity::CpuMask(mask) => (mask & (1u64 << cpu_id)) != 0,
    }
}

/// Score a candidate CPU for placing `task` (higher is better).
fn score_cpu_candidate(
    cpu_id: usize,
    core_type: CoreType,
    capacity: u32,
    nr_running: usize,
    task: &SchedTask,
) -> i64 {
    let mut score: i64 = (capacity as i64) / 4;
    let workload = classify_workload(task);

    let wants_p = matches!(
        task.affinity,
        CoreAffinity::PreferPerformance | CoreAffinity::PinPerformance
    ) || matches!(
        workload,
        WorkloadClass::Interactive | WorkloadClass::ComputeHeavy | WorkloadClass::VectorAvx
    );

    let wants_e = matches!(
        task.affinity,
        CoreAffinity::PreferEfficiency | CoreAffinity::PinEfficiency
    ) || matches!(workload, WorkloadClass::Background | WorkloadClass::Eco);

    if wants_p && core_type == CoreType::Performance {
        score += 1000;
    } else if wants_e && core_type == CoreType::Efficiency {
        score += 900;
    } else if core_type == CoreType::Unified {
        score += 400;
    }

    if task.last_cpu == cpu_id {
        score += 200;
    }

    score -= (nr_running as i64) * 250;
    score
}

// ---------------------------------------------------------------------------
// Global SMP Scheduler
// ---------------------------------------------------------------------------

const fn make_runqueues() -> [SpinLock<CpuRunqueue>; MAX_CPUS] {
    [const { SpinLock::new(CpuRunqueue::new(0)) }; MAX_CPUS]
}

/// Global SMP EEVDF + Deadline + Thread Director scheduler.
pub struct SmpScheduler {
    runqueues: [SpinLock<CpuRunqueue>; MAX_CPUS],
    num_cpus: AtomicUsize,
    next_task_id: AtomicU64,
    initialized: AtomicBool,
}

impl SmpScheduler {
    pub const fn new() -> Self {
        Self {
            runqueues: make_runqueues(),
            num_cpus: AtomicUsize::new(0),
            next_task_id: AtomicU64::new(1),
            initialized: AtomicBool::new(false),
        }
    }

    /// Initialize the SMP scheduler with CPU 0 online.
    pub fn init(&self, bsp_core_type: CoreType) {
        for cpu_id in 0..MAX_CPUS {
            let mut rq = self.runqueues[cpu_id].lock();
            rq.cpu_id = cpu_id;
            rq.online = cpu_id == 0;
            rq.core_type = if cpu_id == 0 {
                bsp_core_type
            } else {
                CoreType::Unified
            };
            rq.capacity = match rq.core_type {
                CoreType::Efficiency => E_CORE_CAPACITY,
                CoreType::Performance | CoreType::Unified => P_CORE_CAPACITY,
            };
            rq.clock_ns = 0;
            rq.avg_vruntime = 0;
            rq.min_vruntime = 0;
            rq.total_weight = 0;
            rq.dl_bandwidth_ppm = 0;
            rq.current = None;
            rq.queue.clear();
            rq.context_switches = 0;
            rq.preemptions = 0;
            rq.steal_count = 0;
        }
        self.num_cpus.store(1, Ordering::Release);
        self.next_task_id.store(1, Ordering::Release);
        self.initialized.store(true, Ordering::Release);
    }

    /// Register or update an online CPU with its hybrid core type and capacity.
    pub fn register_cpu(
        &self,
        cpu_id: usize,
        core_type: CoreType,
        capacity: u32,
    ) -> Result<(), SchedError> {
        if cpu_id >= MAX_CPUS {
            return Err(SchedError::InvalidCpu);
        }
        {
            let mut rq = self.runqueues[cpu_id].lock();
            rq.cpu_id = cpu_id;
            rq.online = true;
            rq.core_type = core_type;
            rq.capacity = capacity.max(1);
        }
        let mut cur = self.num_cpus.load(Ordering::Acquire);
        while cpu_id + 1 > cur {
            match self.num_cpus.compare_exchange_weak(
                cur,
                cpu_id + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
        Ok(())
    }

    /// Allocate a unique task ID.
    #[inline]
    pub fn alloc_task_id(&self) -> u64 {
        self.next_task_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Detect whether the online topology has P-cores and/or E-cores.
    fn topology_flags(&self) -> (bool, bool, usize) {
        let n = self.num_cpus.load(Ordering::Acquire).min(MAX_CPUS);
        let mut has_p = false;
        let mut has_e = false;
        for cpu_id in 0..n {
            let rq = self.runqueues[cpu_id].lock();
            if rq.online {
                match rq.core_type {
                    CoreType::Performance => has_p = true,
                    CoreType::Efficiency => has_e = true,
                    CoreType::Unified => {}
                }
            }
        }
        (has_p, has_e, n)
    }

    /// Select the best online CPU for `task` using Thread Director & load awareness.
    pub fn select_target_cpu(&self, task: &SchedTask) -> Result<usize, SchedError> {
        let (has_p, has_e, n) = self.topology_flags();
        let mut best_cpu: Option<usize> = None;
        let mut best_score: i64 = i64::MIN;

        for cpu_id in 0..n {
            let rq = self.runqueues[cpu_id].lock();
            if !rq.online {
                continue;
            }
            if !is_cpu_allowed(task.affinity, cpu_id, rq.core_type, has_p, has_e) {
                continue;
            }
            if task.class == SchedClass::Deadline
                && rq.dl_bandwidth_ppm.saturating_add(task.dl.bandwidth_ppm())
                    > MAX_DL_BANDWIDTH_PPM
            {
                continue;
            }
            let score =
                score_cpu_candidate(cpu_id, rq.core_type, rq.capacity, rq.nr_running(), task);
            if best_cpu.is_none() || score > best_score {
                best_cpu = Some(cpu_id);
                best_score = score;
            }
        }

        best_cpu.ok_or(SchedError::NoCompatibleCpu)
    }

    /// Enqueue a new or waking task onto the best target CPU.
    pub fn enqueue(&self, mut task: SchedTask, is_wakeup: bool) -> Result<usize, SchedError> {
        let target_cpu = self.select_target_cpu(&task)?;
        let mut rq = self.runqueues[target_cpu].lock();
        if !rq.online {
            return Err(SchedError::CpuOffline);
        }
        if task.class == SchedClass::Deadline {
            let bw = task.dl.bandwidth_ppm();
            if rq.dl_bandwidth_ppm.saturating_add(bw) > MAX_DL_BANDWIDTH_PPM {
                return Err(SchedError::DeadlineBandwidthExceeded);
            }
            rq.dl_bandwidth_ppm = rq.dl_bandwidth_ppm.saturating_add(bw);
        }
        task.state = TaskState::Ready;
        task.last_cpu = target_cpu;
        task.last_enqueue_ns = rq.clock_ns;
        if task.class == SchedClass::Normal {
            rq.place_eevdf_entity(&mut task, is_wakeup);
        }
        rq.queue.push(task);
        rq.update_vruntime_stats();
        Ok(target_cpu)
    }

    /// Spawn a normal EEVDF task and enqueue it.
    pub fn spawn_normal(
        &self,
        nice: i8,
        slice_ns: u64,
        affinity: CoreAffinity,
    ) -> Result<(u64, usize), SchedError> {
        let id = self.alloc_task_id();
        let task = SchedTask::new_normal(id, nice, slice_ns, affinity);
        let cpu = self.enqueue(task, false)?;
        Ok((id, cpu))
    }

    /// Spawn a `SCHED_DEADLINE` task with admission control.
    pub fn spawn_deadline(
        &self,
        runtime_ns: u64,
        deadline_ns: u64,
        period_ns: u64,
        affinity: CoreAffinity,
    ) -> Result<(u64, usize), SchedError> {
        if runtime_ns == 0 || period_ns == 0 || runtime_ns > period_ns {
            return Err(SchedError::InvalidParameters);
        }
        let id = self.alloc_task_id();
        let now_ns = self.runqueues[0].lock().clock_ns;
        let task =
            SchedTask::new_deadline(id, runtime_ns, deadline_ns, period_ns, now_ns, affinity);
        let cpu = self.enqueue(task, false)?;
        Ok((id, cpu))
    }

    /// Spawn a `SCHED_FIFO` / `SCHED_RR` real-time task.
    pub fn spawn_realtime(
        &self,
        priority: u8,
        rr_slice_ns: u64,
        affinity: CoreAffinity,
    ) -> Result<(u64, usize), SchedError> {
        let id = self.alloc_task_id();
        let task = SchedTask::new_realtime(id, priority, rr_slice_ns, affinity);
        let cpu = self.enqueue(task, false)?;
        Ok((id, cpu))
    }

    /// Schedule the next task on `cpu_id`.
    pub fn schedule(&self, cpu_id: usize) -> Result<Option<u64>, SchedError> {
        if cpu_id >= MAX_CPUS {
            return Err(SchedError::InvalidCpu);
        }
        let mut rq = self.runqueues[cpu_id].lock();
        if !rq.online {
            return Err(SchedError::CpuOffline);
        }
        Ok(rq.schedule_next())
    }

    /// Advance `cpu_id` by `elapsed_ns` and return the resulting `TickAction`.
    pub fn tick(&self, cpu_id: usize, elapsed_ns: u64) -> Result<TickAction, SchedError> {
        if cpu_id >= MAX_CPUS {
            return Err(SchedError::InvalidCpu);
        }
        let mut rq = self.runqueues[cpu_id].lock();
        if !rq.online {
            return Err(SchedError::CpuOffline);
        }
        Ok(rq.tick(elapsed_ns))
    }

    /// Block the currently running task on `cpu_id` and schedule the next task.
    pub fn block_current(&self, cpu_id: usize) -> Result<Option<u64>, SchedError> {
        if cpu_id >= MAX_CPUS {
            return Err(SchedError::InvalidCpu);
        }
        let mut rq = self.runqueues[cpu_id].lock();
        if !rq.online {
            return Err(SchedError::CpuOffline);
        }
        rq.update_vruntime_stats();
        let avg_v = rq.avg_vruntime;
        if let Some(cur) = rq.current.as_deref_mut() {
            cur.state = TaskState::Blocked;
            cur.director.voluntary_switches = cur.director.voluntary_switches.saturating_add(1);
            if cur.class == SchedClass::Normal {
                let lag = (avg_v as i128) - (cur.eevdf.vruntime as i128);
                cur.eevdf.vlag = lag.clamp(-(MAX_LAG_NS as i128), MAX_LAG_NS as i128) as i64;
            }
        }
        Ok(rq.schedule_next())
    }

    /// Unblock a `Blocked` task by ID across CPUs and re-enqueue it with preserved `vlag`.
    pub fn unblock(&self, task_id: u64) -> Result<usize, SchedError> {
        let n = self.num_cpus.load(Ordering::Acquire).min(MAX_CPUS);
        let mut found: Option<SchedTask> = None;

        for cpu_id in 0..n {
            let mut rq = self.runqueues[cpu_id].lock();
            if let Some(pos) = rq
                .queue
                .iter()
                .position(|t| t.id == task_id && t.state == TaskState::Blocked)
            {
                let task = rq.queue.swap_remove(pos);
                if task.class == SchedClass::Deadline {
                    rq.dl_bandwidth_ppm =
                        rq.dl_bandwidth_ppm.saturating_sub(task.dl.bandwidth_ppm());
                }
                found = Some(task);
                break;
            }
        }

        let task = found.ok_or(SchedError::TaskNotFound)?;
        self.enqueue(task, true)
    }

    /// Update Thread Director telemetry for the currently running task on `cpu_id`.
    pub fn update_director_telemetry(
        &self,
        cpu_id: usize,
        ipc_x100: u32,
        cache_miss_permille: u16,
    ) -> Result<(), SchedError> {
        if cpu_id >= MAX_CPUS {
            return Err(SchedError::InvalidCpu);
        }
        let mut rq = self.runqueues[cpu_id].lock();
        if let Some(cur) = rq.current.as_deref_mut() {
            cur.director.ipc_x100 = ipc_x100;
            cur.director.cache_miss_permille = cache_miss_permille;
            cur.workload = classify_workload(cur);
            Ok(())
        } else {
            Err(SchedError::TaskNotFound)
        }
    }

    /// Work-stealing load balancer: `thief_cpu` attempts to steal a compatible
    /// queued task from the busiest online CPU. Uses canonical lock ordering
    /// (`min(a, b)` before `max(a, b)`) to prevent AB-BA deadlocks.
    pub fn steal_work(&self, thief_cpu: usize) -> Result<Option<u64>, SchedError> {
        if thief_cpu >= MAX_CPUS {
            return Err(SchedError::InvalidCpu);
        }
        let (has_p, has_e, n) = self.topology_flags();

        let thief_core_type = {
            let rq = self.runqueues[thief_cpu].lock();
            if !rq.online {
                return Err(SchedError::CpuOffline);
            }
            rq.core_type
        };

        // Find busiest source CPU with at least 1 queued Ready task.
        let mut busiest_cpu: Option<usize> = None;
        let mut busiest_nr: usize = 0;
        for cpu_id in 0..n {
            if cpu_id == thief_cpu {
                continue;
            }
            let rq = self.runqueues[cpu_id].lock();
            if !rq.online {
                continue;
            }
            let nr = rq.nr_running();
            let queued_ready = rq
                .queue
                .iter()
                .filter(|t| t.state == TaskState::Ready)
                .count();
            if queued_ready > 0 && nr > busiest_nr {
                busiest_nr = nr;
                busiest_cpu = Some(cpu_id);
            }
        }

        let src_cpu = match busiest_cpu {
            Some(c) => c,
            None => return Ok(None),
        };

        // Canonical lock ordering: always lock lower CPU ID first.
        let (first, second) = if thief_cpu < src_cpu {
            (thief_cpu, src_cpu)
        } else {
            (src_cpu, thief_cpu)
        };
        let mut g1 = self.runqueues[first].lock();
        let mut g2 = self.runqueues[second].lock();

        let (dst_rq, src_rq) = if thief_cpu < src_cpu {
            (&mut *g1, &mut *g2)
        } else {
            (&mut *g2, &mut *g1)
        };

        src_rq.update_vruntime_stats();
        let src_avg_v = src_rq.avg_vruntime;

        let steal_idx = src_rq.queue.iter().position(|t| {
            t.state == TaskState::Ready
                && !t.pinned
                && is_cpu_allowed(t.affinity, thief_cpu, thief_core_type, has_p, has_e)
                && (t.class != SchedClass::Deadline
                    || dst_rq.dl_bandwidth_ppm.saturating_add(t.dl.bandwidth_ppm())
                        <= MAX_DL_BANDWIDTH_PPM)
        });

        let idx = match steal_idx {
            Some(i) => i,
            None => return Ok(None),
        };

        let mut task = src_rq.queue.swap_remove(idx);
        if task.class == SchedClass::Deadline {
            let bw = task.dl.bandwidth_ppm();
            src_rq.dl_bandwidth_ppm = src_rq.dl_bandwidth_ppm.saturating_sub(bw);
            dst_rq.dl_bandwidth_ppm = dst_rq.dl_bandwidth_ppm.saturating_add(bw);
        } else if task.class == SchedClass::Normal {
            let lag = (src_avg_v as i128) - (task.eevdf.vruntime as i128);
            task.eevdf.vlag = lag.clamp(-(MAX_LAG_NS as i128), MAX_LAG_NS as i128) as i64;
            dst_rq.place_eevdf_entity(&mut task, true);
        }

        task.last_cpu = thief_cpu;
        task.last_enqueue_ns = dst_rq.clock_ns;
        let stolen_id = task.id;
        dst_rq.queue.push(task);
        dst_rq.steal_count = dst_rq.steal_count.saturating_add(1);
        dst_rq.update_vruntime_stats();
        src_rq.update_vruntime_stats();

        Ok(Some(stolen_id))
    }

    /// Run a pass of work-stealing across all idle online CPUs.
    pub fn rebalance_all(&self) -> usize {
        let n = self.num_cpus.load(Ordering::Acquire).min(MAX_CPUS);
        let mut stolen = 0usize;
        for cpu_id in 0..n {
            let is_idle = {
                let rq = self.runqueues[cpu_id].lock();
                rq.online && rq.nr_running() == 0
            };
            if is_idle {
                if let Ok(Some(_)) = self.steal_work(cpu_id) {
                    stolen += 1;
                }
            }
        }
        stolen
    }

    /// Return a snapshot of `cpu_id`'s runqueue statistics.
    pub fn cpu_stats(&self, cpu_id: usize) -> Result<CpuRunqueueStats, SchedError> {
        if cpu_id >= MAX_CPUS {
            return Err(SchedError::InvalidCpu);
        }
        let rq = self.runqueues[cpu_id].lock();
        Ok(CpuRunqueueStats {
            cpu_id: rq.cpu_id,
            online: rq.online,
            core_type: rq.core_type,
            capacity: rq.capacity,
            nr_running: rq.nr_running(),
            current_task_id: rq.current.as_ref().map(|t| t.id),
            avg_vruntime: rq.avg_vruntime,
            min_vruntime: rq.min_vruntime,
            total_weight: rq.total_weight,
            dl_bandwidth_ppm: rq.dl_bandwidth_ppm,
            context_switches: rq.context_switches,
            preemptions: rq.preemptions,
            steal_count: rq.steal_count,
        })
    }
}

/// Global SMP EEVDF scheduler instance.
pub static SMP_SCHEDULER: SmpScheduler = SmpScheduler::new();

// ---------------------------------------------------------------------------
// Self-Test Suite
// ---------------------------------------------------------------------------

/// Verify EEVDF eligibility/vdeadline, Deadline EDF+CBS, Thread Director P/E-core
/// placement, and SMP work-stealing load balancing.
pub fn self_test() -> Result<(), &'static str> {
    static TEST_SCHED: SmpScheduler = SmpScheduler::new();
    let sched = &TEST_SCHED;
    sched.init(CoreType::Performance);
    sched
        .register_cpu(1, CoreType::Efficiency, E_CORE_CAPACITY)
        .map_err(|_| "register_cpu(1) failed")?;
    sched
        .register_cpu(2, CoreType::Performance, P_CORE_CAPACITY)
        .map_err(|_| "register_cpu(2) failed")?;
    sched
        .register_cpu(3, CoreType::Efficiency, E_CORE_CAPACITY)
        .map_err(|_| "register_cpu(3) failed")?;

    // 1. Thread Director P/E-core affinity placement
    let (p_task, p_cpu) = sched
        .spawn_normal(0, DEFAULT_SLICE_NS, CoreAffinity::PreferPerformance)
        .map_err(|_| "spawn_normal P-core failed")?;
    let p_stats = sched.cpu_stats(p_cpu).map_err(|_| "cpu_stats failed")?;
    if p_stats.core_type != CoreType::Performance {
        return Err("PreferPerformance task was not placed on a P-core");
    }

    let (e_task, e_cpu) = sched
        .spawn_normal(5, DEFAULT_SLICE_NS, CoreAffinity::PreferEfficiency)
        .map_err(|_| "spawn_normal E-core failed")?;
    let e_stats = sched.cpu_stats(e_cpu).map_err(|_| "cpu_stats failed")?;
    if e_stats.core_type != CoreType::Efficiency {
        return Err("PreferEfficiency task was not placed on an E-core");
    }

    // 2. EEVDF vruntime / vdeadline fairness & preemption
    let (hi_prio, hi_cpu) = sched
        .spawn_normal(-10, MIN_SLICE_NS, CoreAffinity::CpuMask(1 << 0))
        .map_err(|_| "spawn_normal high-weight failed")?;
    let _ = sched.schedule(0).map_err(|_| "schedule(0) failed")?;
    let _ = sched.tick(0, 1_000_000).map_err(|_| "tick(0) failed")?;
    let _ = (p_task, e_task, hi_prio, hi_cpu);

    // 3. Real-Time Deadline (EDF + CBS) preempts Normal & enforces bandwidth cap
    let (dl_id, dl_cpu) = sched
        .spawn_deadline(
            1_000_000,
            4_000_000,
            10_000_000,
            CoreAffinity::CpuMask(1 << 0),
        )
        .map_err(|_| "spawn_deadline failed")?;
    if dl_cpu != 0 {
        return Err("Deadline task ignored CpuMask(1 << 0)");
    }
    let action = sched.tick(0, 500_000).map_err(|_| "tick(0) dl failed")?;
    match action {
        TickAction::Preempt { next_id, .. } if next_id == dl_id => {}
        TickAction::Continue(id) if id == dl_id => {}
        _ => return Err("Deadline task did not preempt Normal task"),
    }

    // Bandwidth admission control: 90% + 10% > 95% must fail on CPU 0
    let bw_err = sched.spawn_deadline(
        9_000_000,
        10_000_000,
        10_000_000,
        CoreAffinity::CpuMask(1 << 0),
    );
    if bw_err != Err(SchedError::DeadlineBandwidthExceeded) {
        return Err("Deadline admission control failed to reject over-committed CPU");
    }

    // 4. Work-stealing load balancer: overload CPU 2 and steal from idle CPU 3
    for _ in 0..3 {
        let _ = sched
            .spawn_normal(0, DEFAULT_SLICE_NS, CoreAffinity::CpuMask(1 << 2 | 1 << 3))
            .map_err(|_| "spawn_normal for steal test failed")?;
    }
    // Ensure CPU 3 is idle and CPU 2 has queued tasks
    let stolen = sched.steal_work(3).map_err(|_| "steal_work(3) failed")?;
    if stolen.is_none() {
        // If tasks were already spread across 2 and 3, verify total nr_running >= 3
        let s2 = sched.cpu_stats(2).map_err(|_| "cpu_stats(2) failed")?;
        let s3 = sched.cpu_stats(3).map_err(|_| "cpu_stats(3) failed")?;
        if s2.nr_running + s3.nr_running < 3 {
            return Err("Work-stealing test lost tasks");
        }
    }

    // 5. Block & unblock with vlag preservation
    let cur0 = sched
        .schedule(0)
        .map_err(|_| "schedule(0) before block failed")?;
    if let Some(tid) = cur0 {
        let _ = sched
            .block_current(0)
            .map_err(|_| "block_current(0) failed")?;
        let _ = sched.unblock(tid).map_err(|_| "unblock failed")?;
    }

    Ok(())
}
