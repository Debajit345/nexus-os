//! Preemptive priority + round-robin task scheduler.
#![allow(dead_code)]

use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use nexus_abi::Personality;

use crate::cap::CapTable;
use crate::mm::SpinLock;

/// Default time slice in timer ticks for a task.
const BASE_TIME_SLICE: u64 = 10;

/// Unique task identifier.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TaskId(pub u64);

impl TaskId {
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Lifecycle state of a scheduled task.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TaskState {
    Ready,
    Running,
    Blocked,
    Exited,
}

/// Saved x86_64 CPU context (callee-saved registers + instruction/stack/flags).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuContext {
    pub rbx: u64,
    pub rbp: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rsp: u64,
    pub rip: u64,
    pub rflags: u64,
}

/// Kernel task control block (TCB).
pub struct Task {
    pub id: TaskId,
    pub name: String,
    pub personality: Personality,
    pub state: TaskState,
    pub priority: u8,
    pub ticks_remaining: u64,
    pub context: CpuContext,
    pub cap_table: CapTable,
}

impl Task {
    fn new(id: TaskId, name: &str, personality: Personality, priority: u8) -> Self {
        Self {
            id,
            name: name.to_string(),
            personality,
            state: TaskState::Ready,
            priority,
            ticks_remaining: time_slice_for_priority(priority),
            context: CpuContext::default(),
            cap_table: CapTable::new(),
        }
    }
}

#[inline]
const fn time_slice_for_priority(priority: u8) -> u64 {
    BASE_TIME_SLICE + (priority as u64 / 32)
}

/// Global kernel scheduler protected by a spinlock.
pub static SCHEDULER: SpinLock<Scheduler> = SpinLock::new(Scheduler::new());

/// Priority-aware round-robin task scheduler.
pub struct Scheduler {
    tasks: Vec<Task>,
    ready_queue: VecDeque<TaskId>,
    current: Option<TaskId>,
    next_id: u64,
    initialized: bool,
}

impl Scheduler {
    pub const fn new() -> Self {
        Self {
            tasks: Vec::new(),
            ready_queue: VecDeque::new(),
            current: None,
            next_id: 1,
            initialized: false,
        }
    }

    /// Initialize the scheduler and create the initial kernel idle/boot task.
    pub fn init(&mut self) {
        if self.initialized {
            return;
        }
        self.tasks.clear();
        self.ready_queue.clear();
        self.current = None;
        self.next_id = 1;
        self.initialized = true;

        let boot_id = self.alloc_task("kernel-boot", Personality::Native, 0);
        if let Some(task) = self.find_task_mut(boot_id) {
            task.state = TaskState::Running;
        }
        self.current = Some(boot_id);
    }

    /// Spawn a new task in the `Ready` state and enqueue it.
    pub fn spawn(&mut self, name: &str, personality: Personality, priority: u8) -> TaskId {
        let id = self.alloc_task(name, personality, priority);
        self.ready_queue.push_back(id);
        if self.current.is_none() {
            self.schedule_next();
        }
        id
    }

    /// Advance the scheduler by one timer tick, preempting if the time slice expires.
    /// Returns the currently running `TaskId` after the tick.
    pub fn tick(&mut self) -> Option<TaskId> {
        let cur_id = match self.current {
            Some(id) => id,
            None => return self.schedule_next(),
        };

        let expired = if let Some(task) = self.find_task_mut(cur_id) {
            if task.state != TaskState::Running {
                true
            } else if task.ticks_remaining > 1 {
                task.ticks_remaining -= 1;
                false
            } else {
                task.ticks_remaining = time_slice_for_priority(task.priority);
                true
            }
        } else {
            true
        };

        if expired {
            self.preempt_current();
        }

        self.current
    }

    /// Voluntarily yield the CPU from the current task to the next ready task.
    pub fn yield_now(&mut self) {
        self.preempt_current();
    }

    /// Return the `TaskId` of the currently running task, if any.
    pub fn current_task_id(&self) -> Option<TaskId> {
        self.current
    }

    /// Return the total number of non-exited tasks tracked by the scheduler.
    pub fn task_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| t.state != TaskState::Exited)
            .count()
    }

    /// Block the currently running task and switch to the next ready task.
    #[allow(dead_code)]
    pub fn block_current(&mut self) {
        if let Some(cur_id) = self.current.take() {
            if let Some(task) = self.find_task_mut(cur_id) {
                if task.state == TaskState::Running {
                    task.state = TaskState::Blocked;
                }
            }
        }
        self.schedule_next();
    }

    /// Unblock a `Blocked` task and place it back on the ready queue.
    #[allow(dead_code)]
    pub fn unblock(&mut self, id: TaskId) -> bool {
        if let Some(task) = self.find_task_mut(id) {
            if task.state == TaskState::Blocked {
                task.state = TaskState::Ready;
                task.ticks_remaining = time_slice_for_priority(task.priority);
                self.ready_queue.push_back(id);
                return true;
            }
        }
        false
    }

    /// Mark the currently running task as `Exited` and schedule the next task.
    #[allow(dead_code)]
    pub fn exit_current(&mut self) {
        if let Some(cur_id) = self.current.take() {
            if let Some(task) = self.find_task_mut(cur_id) {
                task.state = TaskState::Exited;
            }
        }
        self.schedule_next();
    }

    /// Query the state of a task by `TaskId`.
    #[allow(dead_code)]
    pub fn task_state(&self, id: TaskId) -> Option<TaskState> {
        self.tasks.iter().find(|t| t.id == id).map(|t| t.state)
    }

    fn alloc_task(&mut self, name: &str, personality: Personality, priority: u8) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;
        self.tasks.push(Task::new(id, name, personality, priority));
        id
    }

    fn find_task_mut(&mut self, id: TaskId) -> Option<&mut Task> {
        self.tasks.iter_mut().find(|t| t.id == id)
    }

    fn preempt_current(&mut self) {
        if self.ready_queue.is_empty() {
            return;
        }
        if let Some(cur_id) = self.current.take() {
            if let Some(task) = self.find_task_mut(cur_id) {
                if task.state == TaskState::Running {
                    task.state = TaskState::Ready;
                    task.ticks_remaining = time_slice_for_priority(task.priority);
                    self.ready_queue.push_back(cur_id);
                }
            }
        }
        self.schedule_next();
    }

    fn schedule_next(&mut self) -> Option<TaskId> {
        // Pick the highest-priority Ready task in the queue (FIFO on ties).
        let mut best_idx: Option<usize> = None;
        let mut best_prio: u8 = 0;

        for (idx, &tid) in self.ready_queue.iter().enumerate() {
            if let Some(task) = self.tasks.iter().find(|t| t.id == tid) {
                if task.state == TaskState::Ready
                    && (best_idx.is_none() || task.priority > best_prio)
                {
                    best_idx = Some(idx);
                    best_prio = task.priority;
                }
            }
        }

        let next_id = self.ready_queue.remove(best_idx?)?;

        if let Some(task) = self.find_task_mut(next_id) {
            task.state = TaskState::Running;
            task.ticks_remaining = time_slice_for_priority(task.priority);
        }
        self.current = Some(next_id);
        Some(next_id)
    }
}

/// Initialize the global kernel scheduler.
pub fn init() {
    SCHEDULER.lock().init();
}

/// Spawn a new task on the global scheduler.
pub fn spawn(name: &str, personality: Personality, priority: u8) -> TaskId {
    SCHEDULER.lock().spawn(name, personality, priority)
}

/// Advance the global scheduler by one timer tick.
pub fn tick() -> Option<TaskId> {
    SCHEDULER.lock().tick()
}

/// Voluntarily yield the current task on the global scheduler.
pub fn yield_now() {
    SCHEDULER.lock().yield_now();
}

/// Return the currently running `TaskId` on the global scheduler.
pub fn current_task_id() -> Option<TaskId> {
    SCHEDULER.lock().current_task_id()
}

/// Return the number of active (non-exited) tasks on the global scheduler.
pub fn task_count() -> usize {
    SCHEDULER.lock().task_count()
}
