//! `no_std` Asynchronous `io_uring` SQ/CQ Ring Engine & Registered Buffer/FD Table.
//!
//! Provides:
//! - ABI-compatible Submission Queue Entry ([`IoUringSqe`]) and Completion Queue Entry
//!   ([`IoUringCqe`]) descriptors, plus [`SqRing`] and [`CqRing`] power-of-two ring buffers.
//! - Registered fixed buffer table ([`RegisteredBufferTable`]) and registered file descriptor
//!   table ([`RegisteredFdTable`]) with [`IoUringFile`] abstraction and [`VfsIoFile`] adapter.
//! - Core opcodes: `IORING_OP_NOP`, `IORING_OP_READ`, `IORING_OP_WRITE`, `IORING_OP_READ_FIXED`,
//!   `IORING_OP_WRITE_FIXED`, `IORING_OP_FSYNC`, `IORING_OP_POLL_ADD`, and `IORING_OP_TIMEOUT`.
//! - Linked SQE chains (`IOSQE_IO_LINK` / `IOSQE_IO_HARDLINK`) and `SQPOLL` kernel-poller
//!   idle/wakeup state machine ([`SqPollState`]).
//! - [`self_test`] exercising ring setup, NOP/READ/WRITE/FSYNC/POLL_ADD/TIMEOUT, registered
//!   buffers & FDs, linked SQE cancellation, and SQPOLL idle/wakeup transitions.

#![allow(dead_code)]

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use nexus_abi::AbiError;

use crate::mm::SpinLock;
use crate::serial_println;
use crate::time;
use crate::vfs::VfsNode;

/// Maximum supported SQ/CQ ring entries (power of two).
pub const MAX_RING_ENTRIES: u32 = 4096;

/// Default SQPOLL idle timeout in milliseconds.
pub const DEFAULT_SQPOLL_IDLE_MS: u32 = 1000;

/// Maximum number of registered buffers per ring.
pub const MAX_REGISTERED_BUFFERS: usize = 1024;

/// Maximum number of registered file descriptors per ring.
pub const MAX_REGISTERED_FILES: usize = 1024;

// ---------------------------------------------------------------------------
// Setup, Ring, Enter, SQE, and CQE Flags
// ---------------------------------------------------------------------------

pub const IORING_SETUP_IOPOLL: u32 = 1 << 0;
pub const IORING_SETUP_SQPOLL: u32 = 1 << 1;
pub const IORING_SETUP_SQ_AFF: u32 = 1 << 2;
pub const IORING_SETUP_CQSIZE: u32 = 1 << 3;
pub const IORING_SETUP_CLAMP: u32 = 1 << 4;
pub const IORING_SETUP_SINGLE_ISSUER: u32 = 1 << 12;

pub const IORING_SQ_NEED_WAKEUP: u32 = 1 << 0;
pub const IORING_SQ_CQ_OVERFLOW: u32 = 1 << 1;

pub const IORING_ENTER_GETEVENTS: u32 = 1 << 0;
pub const IORING_ENTER_SQ_WAKEUP: u32 = 1 << 1;
pub const IORING_ENTER_SQ_WAIT: u32 = 1 << 2;

pub const IOSQE_FIXED_FILE: u8 = 1 << 0;
pub const IOSQE_IO_DRAIN: u8 = 1 << 1;
pub const IOSQE_IO_LINK: u8 = 1 << 2;
pub const IOSQE_IO_HARDLINK: u8 = 1 << 3;
pub const IOSQE_ASYNC: u8 = 1 << 4;
pub const IOSQE_BUFFER_SELECT: u8 = 1 << 5;

pub const IORING_CQE_F_BUFFER: u32 = 1 << 0;
pub const IORING_CQE_F_MORE: u32 = 1 << 1;

pub const IORING_FSYNC_DATASYNC: u32 = 1 << 0;
pub const IORING_TIMEOUT_ABS: u32 = 1 << 0;

// ---------------------------------------------------------------------------
// Opcodes
// ---------------------------------------------------------------------------

pub const IORING_OP_NOP: u8 = 0;
pub const IORING_OP_READV: u8 = 1;
pub const IORING_OP_WRITEV: u8 = 2;
pub const IORING_OP_FSYNC: u8 = 3;
pub const IORING_OP_READ_FIXED: u8 = 4;
pub const IORING_OP_WRITE_FIXED: u8 = 5;
pub const IORING_OP_POLL_ADD: u8 = 6;
pub const IORING_OP_POLL_REMOVE: u8 = 7;
pub const IORING_OP_TIMEOUT: u8 = 11;
pub const IORING_OP_TIMEOUT_REMOVE: u8 = 12;
pub const IORING_OP_READ: u8 = 22;
pub const IORING_OP_WRITE: u8 = 23;

// ---------------------------------------------------------------------------
// Poll Masks & Negative Errno Constants
// ---------------------------------------------------------------------------

pub const POLLIN: u32 = 0x0001;
pub const POLLPRI: u32 = 0x0002;
pub const POLLOUT: u32 = 0x0004;
pub const POLLERR: u32 = 0x0008;
pub const POLLHUP: u32 = 0x0010;

pub const ENOENT: i32 = -2;
pub const EIO: i32 = -5;
pub const EBADF: i32 = -9;
pub const EAGAIN: i32 = -11;
pub const EFAULT: i32 = -14;
pub const EINVAL: i32 = -22;
pub const ENOSPC: i32 = -28;
pub const ETIME: i32 = -62;
pub const EOPNOTSUPP: i32 = -95;
pub const ECANCELED: i32 = -125;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by the `io_uring` engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoUringError {
    InvalidEntries,
    InvalidFlags,
    RingFull,
    RingEmpty,
    TooManyBuffers,
    TooManyFiles,
    InvalidBufferIndex,
    InvalidFd,
    OutOfBounds,
    NotFound,
}

impl IoUringError {
    pub const fn as_errno(self) -> i32 {
        match self {
            Self::InvalidEntries | Self::InvalidFlags | Self::InvalidBufferIndex => EINVAL,
            Self::RingFull | Self::TooManyBuffers | Self::TooManyFiles => ENOSPC,
            Self::RingEmpty => EAGAIN,
            Self::InvalidFd => EBADF,
            Self::OutOfBounds => EFAULT,
            Self::NotFound => ENOENT,
        }
    }
}

// ---------------------------------------------------------------------------
// Ring Descriptors
// ---------------------------------------------------------------------------

/// Submission Queue Entry (64 bytes, Linux `io_uring_sqe` layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoUringSqe {
    pub opcode: u8,
    pub flags: u8,
    pub ioprio: u16,
    pub fd: i32,
    pub off: u64,
    pub addr: u64,
    pub len: u32,
    pub op_flags: u32,
    pub user_data: u64,
    pub buf_index: u16,
    pub personality: u16,
    pub splice_fd_in: i32,
    pub addr3: u64,
    pub resv: u64,
}

impl IoUringSqe {
    /// Construct an `IORING_OP_NOP` SQE.
    pub const fn nop(user_data: u64) -> Self {
        Self {
            opcode: IORING_OP_NOP,
            flags: 0,
            ioprio: 0,
            fd: -1,
            off: 0,
            addr: 0,
            len: 0,
            op_flags: 0,
            user_data,
            buf_index: 0,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            resv: 0,
        }
    }

    /// Construct an `IORING_OP_READ` SQE.
    pub const fn read(fd: i32, addr: u64, len: u32, off: u64, user_data: u64) -> Self {
        Self {
            opcode: IORING_OP_READ,
            flags: 0,
            ioprio: 0,
            fd,
            off,
            addr,
            len,
            op_flags: 0,
            user_data,
            buf_index: 0,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            resv: 0,
        }
    }

    /// Construct an `IORING_OP_WRITE` SQE.
    pub const fn write(fd: i32, addr: u64, len: u32, off: u64, user_data: u64) -> Self {
        Self {
            opcode: IORING_OP_WRITE,
            flags: 0,
            ioprio: 0,
            fd,
            off,
            addr,
            len,
            op_flags: 0,
            user_data,
            buf_index: 0,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            resv: 0,
        }
    }

    /// Construct an `IORING_OP_READ_FIXED` SQE targeting a registered buffer.
    pub const fn read_fixed(fd: i32, buf_index: u16, len: u32, off: u64, user_data: u64) -> Self {
        Self {
            opcode: IORING_OP_READ_FIXED,
            flags: 0,
            ioprio: 0,
            fd,
            off,
            addr: 0,
            len,
            op_flags: 0,
            user_data,
            buf_index,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            resv: 0,
        }
    }

    /// Construct an `IORING_OP_WRITE_FIXED` SQE sourcing from a registered buffer.
    pub const fn write_fixed(fd: i32, buf_index: u16, len: u32, off: u64, user_data: u64) -> Self {
        Self {
            opcode: IORING_OP_WRITE_FIXED,
            flags: 0,
            ioprio: 0,
            fd,
            off,
            addr: 0,
            len,
            op_flags: 0,
            user_data,
            buf_index,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            resv: 0,
        }
    }

    /// Construct an `IORING_OP_FSYNC` SQE.
    pub const fn fsync(fd: i32, datasync: bool, user_data: u64) -> Self {
        Self {
            opcode: IORING_OP_FSYNC,
            flags: 0,
            ioprio: 0,
            fd,
            off: 0,
            addr: 0,
            len: 0,
            op_flags: if datasync { IORING_FSYNC_DATASYNC } else { 0 },
            user_data,
            buf_index: 0,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            resv: 0,
        }
    }

    /// Construct an `IORING_OP_POLL_ADD` SQE.
    pub const fn poll_add(fd: i32, poll_mask: u32, user_data: u64) -> Self {
        Self {
            opcode: IORING_OP_POLL_ADD,
            flags: 0,
            ioprio: 0,
            fd,
            off: 0,
            addr: 0,
            len: 0,
            op_flags: poll_mask,
            user_data,
            buf_index: 0,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            resv: 0,
        }
    }

    /// Construct an `IORING_OP_TIMEOUT` SQE.
    ///
    /// `timeout_ns` is relative unless `abs` is true. `count` specifies an optional
    /// CQE completion count threshold (0 for pure time-based expiry).
    pub const fn timeout(timeout_ns: u64, count: u64, abs: bool, user_data: u64) -> Self {
        Self {
            opcode: IORING_OP_TIMEOUT,
            flags: 0,
            ioprio: 0,
            fd: -1,
            off: count,
            addr: timeout_ns,
            len: 1,
            op_flags: if abs { IORING_TIMEOUT_ABS } else { 0 },
            user_data,
            buf_index: 0,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            resv: 0,
        }
    }

    /// Return a copy of this SQE with additional `IOSQE_*` flags set.
    pub const fn with_flags(mut self, flags: u8) -> Self {
        self.flags |= flags;
        self
    }
}

/// Completion Queue Entry (16 bytes, Linux `io_uring_cqe` layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoUringCqe {
    pub user_data: u64,
    pub res: i32,
    pub flags: u32,
}

/// Setup parameters for creating an `io_uring` instance.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoUringParams {
    pub sq_entries: u32,
    pub cq_entries: u32,
    pub flags: u32,
    pub sq_thread_cpu: u32,
    pub sq_thread_idle: u32,
    pub features: u32,
    pub resv: [u32; 4],
}

// ---------------------------------------------------------------------------
// Submission & Completion Rings
// ---------------------------------------------------------------------------

fn normalize_ring_entries(entries: u32, clamp: bool) -> Result<u32, IoUringError> {
    if entries == 0 {
        return Err(IoUringError::InvalidEntries);
    }
    let rounded = entries.next_power_of_two();
    if rounded > MAX_RING_ENTRIES {
        if clamp {
            Ok(MAX_RING_ENTRIES)
        } else {
            Err(IoUringError::InvalidEntries)
        }
    } else {
        Ok(rounded)
    }
}

/// Submission Queue ring buffer.
pub struct SqRing {
    pub head: AtomicU32,
    pub tail: AtomicU32,
    pub ring_mask: u32,
    pub ring_entries: u32,
    pub flags: AtomicU32,
    pub dropped: AtomicU32,
    pub array: Vec<u32>,
    pub sqes: Vec<IoUringSqe>,
}

impl SqRing {
    /// Allocate a new Submission Queue ring with `entries` slots (power of two).
    pub fn new(entries: u32, clamp: bool) -> Result<Self, IoUringError> {
        let count = normalize_ring_entries(entries, clamp)?;
        let mut array = Vec::with_capacity(count as usize);
        for idx in 0..count {
            array.push(idx);
        }
        let sqes = vec![IoUringSqe::default(); count as usize];
        Ok(Self {
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            ring_mask: count - 1,
            ring_entries: count,
            flags: AtomicU32::new(0),
            dropped: AtomicU32::new(0),
            array,
            sqes,
        })
    }

    /// Number of unconsumed SQEs currently in the ring.
    #[inline]
    pub fn pending(&self) -> u32 {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    /// Number of free slots available in the SQ ring.
    #[inline]
    pub fn free_slots(&self) -> u32 {
        self.ring_entries.saturating_sub(self.pending())
    }

    /// Push an SQE into the submission ring (producer side).
    pub fn push(&mut self, sqe: IoUringSqe) -> Result<(), IoUringError> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        if tail.wrapping_sub(head) >= self.ring_entries {
            return Err(IoUringError::RingFull);
        }
        let slot = (tail & self.ring_mask) as usize;
        self.sqes[slot] = sqe;
        self.array[slot] = slot as u32;
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Pop the next SQE from the submission ring (kernel consumer side).
    pub fn pop(&mut self) -> Option<IoUringSqe> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let ring_idx = (head & self.ring_mask) as usize;
        let sqe_idx = self.array[ring_idx] as usize;
        if sqe_idx >= self.sqes.len() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            self.head.store(head.wrapping_add(1), Ordering::Release);
            return None;
        }
        let sqe = self.sqes[sqe_idx];
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(sqe)
    }
}

/// Completion Queue ring buffer.
pub struct CqRing {
    pub head: AtomicU32,
    pub tail: AtomicU32,
    pub ring_mask: u32,
    pub ring_entries: u32,
    pub overflow: AtomicU32,
    pub cqes: Vec<IoUringCqe>,
}

impl CqRing {
    /// Allocate a new Completion Queue ring with `entries` slots (power of two).
    pub fn new(entries: u32, clamp: bool) -> Result<Self, IoUringError> {
        let count = normalize_ring_entries(entries, clamp)?;
        let cqes = vec![IoUringCqe::default(); count as usize];
        Ok(Self {
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            ring_mask: count - 1,
            ring_entries: count,
            overflow: AtomicU32::new(0),
            cqes,
        })
    }

    /// Number of unread CQEs currently in the ring.
    #[inline]
    pub fn ready(&self) -> u32 {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    /// Post a CQE into the completion ring (kernel producer side).
    /// Returns `false` and increments `overflow` if the ring is full.
    pub fn push(&mut self, cqe: IoUringCqe) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        if tail.wrapping_sub(head) >= self.ring_entries {
            self.overflow.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let slot = (tail & self.ring_mask) as usize;
        self.cqes[slot] = cqe;
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        true
    }

    /// Peek at the next unread CQE without advancing `head`.
    pub fn peek(&self) -> Option<IoUringCqe> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let slot = (head & self.ring_mask) as usize;
        Some(self.cqes[slot])
    }

    /// Pop the next unread CQE and advance `head` (consumer side).
    pub fn pop(&mut self) -> Option<IoUringCqe> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let slot = (head & self.ring_mask) as usize;
        let cqe = self.cqes[slot];
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(cqe)
    }
}

// ---------------------------------------------------------------------------
// Registered Buffer Table
// ---------------------------------------------------------------------------

/// Table of pre-registered fixed buffers for zero-lookup `READ_FIXED` / `WRITE_FIXED`.
#[derive(Default)]
pub struct RegisteredBufferTable {
    buffers: Vec<Vec<u8>>,
}

impl RegisteredBufferTable {
    pub const fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }

    /// Register a set of buffers, replacing any previously registered set.
    pub fn register(&mut self, bufs: Vec<Vec<u8>>) -> Result<(), IoUringError> {
        if bufs.len() > MAX_REGISTERED_BUFFERS {
            return Err(IoUringError::TooManyBuffers);
        }
        self.buffers = bufs;
        Ok(())
    }

    /// Register `sizes.len()` zero-initialized buffers of the given byte lengths.
    pub fn register_sizes(&mut self, sizes: &[usize]) -> Result<(), IoUringError> {
        if sizes.len() > MAX_REGISTERED_BUFFERS {
            return Err(IoUringError::TooManyBuffers);
        }
        let mut bufs = Vec::with_capacity(sizes.len());
        for &sz in sizes {
            bufs.push(vec![0u8; sz]);
        }
        self.buffers = bufs;
        Ok(())
    }

    /// Unregister all fixed buffers.
    pub fn unregister(&mut self) {
        self.buffers.clear();
    }

    /// Number of registered buffers.
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Returns `true` if no buffers are registered.
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Borrow a registered buffer immutably.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.buffers.get(index).map(Vec::as_slice)
    }

    /// Borrow a registered buffer mutably.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [u8]> {
        self.buffers.get_mut(index).map(Vec::as_mut_slice)
    }

    /// Copy bytes from a registered buffer into `dst`.
    pub fn read_from(
        &self,
        index: usize,
        offset: usize,
        dst: &mut [u8],
    ) -> Result<usize, IoUringError> {
        let buf = self
            .buffers
            .get(index)
            .ok_or(IoUringError::InvalidBufferIndex)?;
        if offset > buf.len() {
            return Err(IoUringError::OutOfBounds);
        }
        let avail = buf.len() - offset;
        let n = avail.min(dst.len());
        dst[..n].copy_from_slice(&buf[offset..offset + n]);
        Ok(n)
    }

    /// Copy bytes from `src` into a registered buffer.
    pub fn write_to(
        &mut self,
        index: usize,
        offset: usize,
        src: &[u8],
    ) -> Result<usize, IoUringError> {
        let buf = self
            .buffers
            .get_mut(index)
            .ok_or(IoUringError::InvalidBufferIndex)?;
        if offset > buf.len() {
            return Err(IoUringError::OutOfBounds);
        }
        let avail = buf.len() - offset;
        let n = avail.min(src.len());
        buf[offset..offset + n].copy_from_slice(&src[..n]);
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// File Abstraction & Registered FD Table
// ---------------------------------------------------------------------------

fn abi_err_to_errno(err: AbiError) -> i32 {
    match err {
        AbiError::InvalidArg => EINVAL,
        AbiError::NotFound => ENOENT,
        AbiError::AccessDenied => -13,
        AbiError::AlreadyExists => -17,
        AbiError::NoMemory => -12,
        AbiError::WouldBlock => EAGAIN,
        AbiError::NotSupported => EOPNOTSUPP,
        _ => EIO,
    }
}

/// Asynchronous-capable file/object backend for `io_uring` operations.
pub trait IoUringFile: Send + Sync {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize, i32>;
    fn write_at(&self, offset: u64, data: &[u8]) -> Result<usize, i32>;
    fn fsync(&self, datasync: bool) -> Result<(), i32>;
    fn poll_ready(&self, mask: u32) -> u32;
}

/// Adapter wrapping any [`VfsNode`] for use with `io_uring`.
pub struct VfsIoFile {
    node: Arc<dyn VfsNode>,
}

impl VfsIoFile {
    pub fn new(node: Arc<dyn VfsNode>) -> Self {
        Self { node }
    }
}

impl IoUringFile for VfsIoFile {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize, i32> {
        self.node
            .read(offset as usize, buf)
            .map_err(abi_err_to_errno)
    }

    fn write_at(&self, offset: u64, data: &[u8]) -> Result<usize, i32> {
        self.node
            .write(offset as usize, data)
            .map_err(abi_err_to_errno)
    }

    fn fsync(&self, _datasync: bool) -> Result<(), i32> {
        Ok(())
    }

    fn poll_ready(&self, mask: u32) -> u32 {
        mask & (POLLIN | POLLOUT)
    }
}

struct MemIoInner {
    data: Vec<u8>,
    poll_mask: u32,
    fsync_count: u64,
    datasync_count: u64,
}

/// In-memory [`IoUringFile`] implementation with controllable poll readiness and fsync tracking.
pub struct MemIoFile {
    inner: SpinLock<MemIoInner>,
}

impl MemIoFile {
    pub fn new(initial: &[u8], poll_mask: u32) -> Self {
        Self {
            inner: SpinLock::new(MemIoInner {
                data: initial.to_vec(),
                poll_mask,
                fsync_count: 0,
                datasync_count: 0,
            }),
        }
    }

    pub fn set_poll_mask(&self, mask: u32) {
        self.inner.lock().poll_mask = mask;
    }

    pub fn fsync_counts(&self) -> (u64, u64) {
        let guard = self.inner.lock();
        (guard.fsync_count, guard.datasync_count)
    }

    pub fn snapshot(&self) -> Vec<u8> {
        self.inner.lock().data.clone()
    }
}

impl IoUringFile for MemIoFile {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize, i32> {
        let guard = self.inner.lock();
        let off = offset as usize;
        if off >= guard.data.len() {
            return Ok(0);
        }
        let n = (guard.data.len() - off).min(buf.len());
        buf[..n].copy_from_slice(&guard.data[off..off + n]);
        Ok(n)
    }

    fn write_at(&self, offset: u64, data: &[u8]) -> Result<usize, i32> {
        let mut guard = self.inner.lock();
        let off = offset as usize;
        let end = off.saturating_add(data.len());
        if end > guard.data.len() {
            guard.data.resize(end, 0);
        }
        guard.data[off..off + data.len()].copy_from_slice(data);
        Ok(data.len())
    }

    fn fsync(&self, datasync: bool) -> Result<(), i32> {
        let mut guard = self.inner.lock();
        if datasync {
            guard.datasync_count += 1;
        } else {
            guard.fsync_count += 1;
        }
        Ok(())
    }

    fn poll_ready(&self, mask: u32) -> u32 {
        self.inner.lock().poll_mask & mask
    }
}

/// Table of pre-registered file descriptors (`IOSQE_FIXED_FILE`).
#[derive(Default)]
pub struct RegisteredFdTable {
    slots: Vec<Option<Arc<dyn IoUringFile>>>,
}

impl RegisteredFdTable {
    pub const fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Register a table of file slots, replacing any previous registration.
    pub fn register(
        &mut self,
        files: Vec<Option<Arc<dyn IoUringFile>>>,
    ) -> Result<(), IoUringError> {
        if files.len() > MAX_REGISTERED_FILES {
            return Err(IoUringError::TooManyFiles);
        }
        self.slots = files;
        Ok(())
    }

    /// Update a contiguous range of registered file slots starting at `offset`.
    pub fn update(
        &mut self,
        offset: usize,
        files: &[Option<Arc<dyn IoUringFile>>],
    ) -> Result<(), IoUringError> {
        let end = offset
            .checked_add(files.len())
            .ok_or(IoUringError::OutOfBounds)?;
        if end > self.slots.len() {
            return Err(IoUringError::InvalidFd);
        }
        for (dst, src) in self.slots[offset..end].iter_mut().zip(files.iter()) {
            *dst = src.clone();
        }
        Ok(())
    }

    /// Unregister all fixed file slots.
    pub fn unregister(&mut self) {
        self.slots.clear();
    }

    /// Number of registered file slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Returns `true` if no file slots are registered.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Look up a registered file by slot index.
    pub fn get(&self, index: usize) -> Option<&Arc<dyn IoUringFile>> {
        self.slots.get(index).and_then(|slot| slot.as_ref())
    }
}

// ---------------------------------------------------------------------------
// SQPOLL State Machine
// ---------------------------------------------------------------------------

/// Kernel-side `IORING_SETUP_SQPOLL` poller state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqPollState {
    pub enabled: bool,
    pub running: bool,
    pub need_wakeup: bool,
    pub sq_thread_cpu: u32,
    pub idle_timeout_ns: u64,
    pub last_activity_ns: u64,
    pub wakeups: u64,
    pub polls: u64,
}

impl SqPollState {
    pub fn new(enabled: bool, idle_ms: u32, sq_thread_cpu: u32, now_ns: u64) -> Self {
        let effective_ms = if idle_ms == 0 {
            DEFAULT_SQPOLL_IDLE_MS
        } else {
            idle_ms
        };
        Self {
            enabled,
            running: enabled,
            need_wakeup: false,
            sq_thread_cpu,
            idle_timeout_ns: (effective_ms as u64) * 1_000_000,
            last_activity_ns: now_ns,
            wakeups: 0,
            polls: 0,
        }
    }

    /// Record submission activity so the SQPOLL thread stays awake.
    pub fn record_activity(&mut self, now_ns: u64) {
        if self.enabled {
            self.last_activity_ns = now_ns;
            self.running = true;
            self.need_wakeup = false;
        }
    }

    /// Advance the SQPOLL state machine at `now_ns`.
    ///
    /// Returns `true` if the poller transitioned to parked (`IORING_SQ_NEED_WAKEUP`).
    pub fn tick(&mut self, now_ns: u64, has_pending_sqes: bool) -> bool {
        if !self.enabled {
            return false;
        }
        self.polls = self.polls.saturating_add(1);
        if has_pending_sqes {
            self.last_activity_ns = now_ns;
            self.running = true;
            self.need_wakeup = false;
            return false;
        }
        if self.running && now_ns.saturating_sub(self.last_activity_ns) >= self.idle_timeout_ns {
            self.running = false;
            self.need_wakeup = true;
            return true;
        }
        false
    }

    /// Wake up a parked SQPOLL thread (`IORING_ENTER_SQ_WAKEUP`).
    pub fn wakeup(&mut self, now_ns: u64) -> bool {
        if !self.enabled {
            return false;
        }
        let was_sleeping = !self.running || self.need_wakeup;
        self.running = true;
        self.need_wakeup = false;
        self.last_activity_ns = now_ns;
        if was_sleeping {
            self.wakeups = self.wakeups.saturating_add(1);
        }
        was_sleeping
    }
}

// ---------------------------------------------------------------------------
// Pending Async Operations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct PendingPoll {
    fd: i32,
    fixed_file: bool,
    mask: u32,
    user_data: u64,
}

#[derive(Debug, Clone, Copy)]
struct PendingTimeout {
    deadline_ns: u64,
    target_cqes: u64,
    user_data: u64,
}

// ---------------------------------------------------------------------------
// IoUringInstance Engine
// ---------------------------------------------------------------------------

/// Complete `io_uring` instance combining SQ/CQ rings, registered tables,
/// direct FD map, pending async operations, and SQPOLL state.
pub struct IoUringInstance {
    pub id: u64,
    pub params: IoUringParams,
    pub sq: SqRing,
    pub cq: CqRing,
    pub buffers: RegisteredBufferTable,
    pub files: RegisteredFdTable,
    pub sqpoll: SqPollState,
    direct_fds: BTreeMap<i32, Arc<dyn IoUringFile>>,
    pending_polls: Vec<PendingPoll>,
    pending_timeouts: Vec<PendingTimeout>,
    total_submitted: u64,
    total_completed: u64,
}

static NEXT_RING_ID: AtomicU64 = AtomicU64::new(1);

impl IoUringInstance {
    /// Create a new `io_uring` instance from `params`.
    pub fn new(mut params: IoUringParams) -> Result<Self, IoUringError> {
        let clamp = (params.flags & IORING_SETUP_CLAMP) != 0;
        let sq = SqRing::new(params.sq_entries, clamp)?;
        let cq_req = if (params.flags & IORING_SETUP_CQSIZE) != 0 && params.cq_entries > 0 {
            params.cq_entries
        } else {
            sq.ring_entries.saturating_mul(2)
        };
        let cq = CqRing::new(cq_req, clamp)?;

        params.sq_entries = sq.ring_entries;
        params.cq_entries = cq.ring_entries;

        let sqpoll_enabled = (params.flags & IORING_SETUP_SQPOLL) != 0;
        let now_ns = time::uptime_ns();
        let sqpoll = SqPollState::new(
            sqpoll_enabled,
            params.sq_thread_idle,
            params.sq_thread_cpu,
            now_ns,
        );

        Ok(Self {
            id: NEXT_RING_ID.fetch_add(1, Ordering::Relaxed),
            params,
            sq,
            cq,
            buffers: RegisteredBufferTable::new(),
            files: RegisteredFdTable::new(),
            sqpoll,
            direct_fds: BTreeMap::new(),
            pending_polls: Vec::new(),
            pending_timeouts: Vec::new(),
            total_submitted: 0,
            total_completed: 0,
        })
    }

    /// Attach a direct (non-fixed) file descriptor to this ring instance.
    pub fn insert_direct_fd(
        &mut self,
        fd: i32,
        file: Arc<dyn IoUringFile>,
    ) -> Result<(), IoUringError> {
        if fd < 0 {
            return Err(IoUringError::InvalidFd);
        }
        self.direct_fds.insert(fd, file);
        Ok(())
    }

    /// Remove a direct file descriptor from this ring instance.
    pub fn remove_direct_fd(&mut self, fd: i32) -> Option<Arc<dyn IoUringFile>> {
        self.direct_fds.remove(&fd)
    }

    /// Push an SQE onto the submission ring.
    pub fn push_sqe(&mut self, sqe: IoUringSqe) -> Result<(), IoUringError> {
        self.sq.push(sqe)
    }

    /// Pop a CQE from the completion ring.
    pub fn pop_cqe(&mut self) -> Option<IoUringCqe> {
        self.cq.pop()
    }

    /// Peek at the next CQE without consuming it.
    pub fn peek_cqe(&self) -> Option<IoUringCqe> {
        self.cq.peek()
    }

    /// Total SQEs submitted and CQEs posted over the lifetime of this ring.
    pub fn stats(&self) -> (u64, u64) {
        (self.total_submitted, self.total_completed)
    }

    fn resolve_file(&self, fd: i32, fixed: bool) -> Option<Arc<dyn IoUringFile>> {
        if fd < 0 {
            return None;
        }
        if fixed {
            self.files.get(fd as usize).cloned()
        } else {
            self.direct_fds.get(&fd).cloned()
        }
    }

    fn post_cqe(&mut self, user_data: u64, res: i32, flags: u32) {
        let cqe = IoUringCqe {
            user_data,
            res,
            flags,
        };
        if self.cq.push(cqe) {
            self.total_completed = self.total_completed.saturating_add(1);
        } else {
            self.sq
                .flags
                .fetch_or(IORING_SQ_CQ_OVERFLOW, Ordering::Release);
        }
    }

    /// Execute a single SQE. Returns `Some((res, flags))` if it completed synchronously,
    /// or `None` if it was queued as an asynchronous pending operation (`POLL_ADD` / `TIMEOUT`).
    fn execute_sqe(&mut self, sqe: &IoUringSqe, now_ns: u64) -> Option<(i32, u32)> {
        let fixed = (sqe.flags & IOSQE_FIXED_FILE) != 0;

        match sqe.opcode {
            IORING_OP_NOP => Some((0, 0)),

            IORING_OP_READ => {
                let Some(file) = self.resolve_file(sqe.fd, fixed) else {
                    return Some((EBADF, 0));
                };
                if sqe.len == 0 {
                    return Some((0, 0));
                }
                if sqe.addr == 0 {
                    return Some((EFAULT, 0));
                }
                // Safety: caller provides a valid writable buffer of `sqe.len` bytes at `sqe.addr`.
                let dst = unsafe {
                    core::slice::from_raw_parts_mut(sqe.addr as *mut u8, sqe.len as usize)
                };
                match file.read_at(sqe.off, dst) {
                    Ok(n) => Some((n as i32, 0)),
                    Err(e) => Some((e, 0)),
                }
            }

            IORING_OP_WRITE => {
                let Some(file) = self.resolve_file(sqe.fd, fixed) else {
                    return Some((EBADF, 0));
                };
                if sqe.len == 0 {
                    return Some((0, 0));
                }
                if sqe.addr == 0 {
                    return Some((EFAULT, 0));
                }
                // Safety: caller provides a valid readable buffer of `sqe.len` bytes at `sqe.addr`.
                let src =
                    unsafe { core::slice::from_raw_parts(sqe.addr as *const u8, sqe.len as usize) };
                match file.write_at(sqe.off, src) {
                    Ok(n) => Some((n as i32, 0)),
                    Err(e) => Some((e, 0)),
                }
            }

            IORING_OP_READ_FIXED => {
                let Some(file) = self.resolve_file(sqe.fd, fixed) else {
                    return Some((EBADF, 0));
                };
                let Some(buf) = self.buffers.get_mut(sqe.buf_index as usize) else {
                    return Some((EINVAL, 0));
                };
                let req_len = (sqe.len as usize).min(buf.len());
                match file.read_at(sqe.off, &mut buf[..req_len]) {
                    Ok(n) => Some((n as i32, 0)),
                    Err(e) => Some((e, 0)),
                }
            }

            IORING_OP_WRITE_FIXED => {
                let Some(file) = self.resolve_file(sqe.fd, fixed) else {
                    return Some((EBADF, 0));
                };
                let Some(buf) = self.buffers.get(sqe.buf_index as usize) else {
                    return Some((EINVAL, 0));
                };
                let req_len = (sqe.len as usize).min(buf.len());
                match file.write_at(sqe.off, &buf[..req_len]) {
                    Ok(n) => Some((n as i32, 0)),
                    Err(e) => Some((e, 0)),
                }
            }

            IORING_OP_FSYNC => {
                let Some(file) = self.resolve_file(sqe.fd, fixed) else {
                    return Some((EBADF, 0));
                };
                let datasync = (sqe.op_flags & IORING_FSYNC_DATASYNC) != 0;
                match file.fsync(datasync) {
                    Ok(()) => Some((0, 0)),
                    Err(e) => Some((e, 0)),
                }
            }

            IORING_OP_POLL_ADD => {
                let Some(file) = self.resolve_file(sqe.fd, fixed) else {
                    return Some((EBADF, 0));
                };
                let ready = file.poll_ready(sqe.op_flags);
                if ready != 0 {
                    Some((ready as i32, 0))
                } else {
                    self.pending_polls.push(PendingPoll {
                        fd: sqe.fd,
                        fixed_file: fixed,
                        mask: sqe.op_flags,
                        user_data: sqe.user_data,
                    });
                    None
                }
            }

            IORING_OP_TIMEOUT => {
                let is_abs = (sqe.op_flags & IORING_TIMEOUT_ABS) != 0;
                let deadline_ns = if is_abs {
                    sqe.addr
                } else {
                    now_ns.saturating_add(sqe.addr)
                };
                let target_cqes = if sqe.off > 0 {
                    self.total_completed.saturating_add(sqe.off)
                } else {
                    0
                };
                if target_cqes > 0 && self.total_completed >= target_cqes {
                    return Some((0, 0));
                }
                if now_ns >= deadline_ns {
                    return Some((ETIME, 0));
                }
                self.pending_timeouts.push(PendingTimeout {
                    deadline_ns,
                    target_cqes,
                    user_data: sqe.user_data,
                });
                None
            }

            _ => Some((EOPNOTSUPP, 0)),
        }
    }

    /// Drain up to `max_submit` SQEs (or all pending if `max_submit == 0`) from the SQ ring.
    pub fn submit(&mut self, max_submit: u32, now_ns: u64) -> usize {
        let limit = if max_submit == 0 {
            self.sq.pending()
        } else {
            max_submit.min(self.sq.pending())
        };

        let mut submitted = 0usize;
        let mut link_failed = false;

        for _ in 0..limit {
            let Some(sqe) = self.sq.pop() else {
                break;
            };
            submitted += 1;
            self.total_submitted = self.total_submitted.saturating_add(1);
            self.sqpoll.record_activity(now_ns);

            let is_link = (sqe.flags & IOSQE_IO_LINK) != 0;
            let is_hardlink = (sqe.flags & IOSQE_IO_HARDLINK) != 0;

            if link_failed {
                self.post_cqe(sqe.user_data, ECANCELED, 0);
                if !is_link && !is_hardlink {
                    link_failed = false;
                }
                continue;
            }

            if let Some((res, cqe_flags)) = self.execute_sqe(&sqe, now_ns) {
                self.post_cqe(sqe.user_data, res, cqe_flags);
                if res < 0 && is_link && !is_hardlink {
                    link_failed = true;
                } else if !is_link && !is_hardlink {
                    link_failed = false;
                }
            } else if !is_link && !is_hardlink {
                link_failed = false;
            }
        }

        submitted
    }

    /// Poll pending asynchronous operations (`POLL_ADD` and `TIMEOUT`) at `now_ns`.
    /// Returns the number of newly completed CQEs posted.
    pub fn poll_pending(&mut self, now_ns: u64) -> usize {
        let mut completed = 0usize;

        // Check pending POLL_ADD operations.
        let mut remaining_polls = Vec::with_capacity(self.pending_polls.len());
        let polls = core::mem::take(&mut self.pending_polls);
        for p in polls {
            if let Some(file) = self.resolve_file(p.fd, p.fixed_file) {
                let ready = file.poll_ready(p.mask);
                if ready != 0 {
                    self.post_cqe(p.user_data, ready as i32, 0);
                    completed += 1;
                } else {
                    remaining_polls.push(p);
                }
            } else {
                self.post_cqe(p.user_data, EBADF, 0);
                completed += 1;
            }
        }
        self.pending_polls = remaining_polls;

        // Check pending TIMEOUT operations.
        let mut remaining_timeouts = Vec::with_capacity(self.pending_timeouts.len());
        let timeouts = core::mem::take(&mut self.pending_timeouts);
        for t in timeouts {
            if t.target_cqes > 0 && self.total_completed >= t.target_cqes {
                self.post_cqe(t.user_data, 0, 0);
                completed += 1;
            } else if now_ns >= t.deadline_ns {
                self.post_cqe(t.user_data, ETIME, 0);
                completed += 1;
            } else {
                remaining_timeouts.push(t);
            }
        }
        self.pending_timeouts = remaining_timeouts;

        completed
    }

    /// Advance the `SQPOLL` kernel poller at `now_ns`, draining any pending SQEs
    /// and updating `IORING_SQ_NEED_WAKEUP` in `sq.flags`.
    pub fn sqpoll_tick(&mut self, now_ns: u64) -> usize {
        if !self.sqpoll.enabled {
            return 0;
        }
        let has_pending = self.sq.pending() > 0;
        let submitted = if has_pending && self.sqpoll.running {
            self.submit(0, now_ns)
        } else {
            0
        };
        let parked = self.sqpoll.tick(now_ns, self.sq.pending() > 0);
        if parked {
            self.sq
                .flags
                .fetch_or(IORING_SQ_NEED_WAKEUP, Ordering::Release);
        }
        submitted
    }

    /// `io_uring_enter` equivalent: optionally wake SQPOLL, submit `to_submit` SQEs,
    /// and poll pending events if `IORING_ENTER_GETEVENTS` or `min_complete > 0`.
    pub fn enter(
        &mut self,
        to_submit: u32,
        min_complete: u32,
        flags: u32,
        now_ns: u64,
    ) -> Result<u32, IoUringError> {
        if (flags & IORING_ENTER_SQ_WAKEUP) != 0 && self.sqpoll.wakeup(now_ns) {
            self.sq
                .flags
                .fetch_and(!IORING_SQ_NEED_WAKEUP, Ordering::Release);
        }

        let submitted = if to_submit > 0 {
            self.submit(to_submit, now_ns) as u32
        } else {
            0
        };

        if (flags & IORING_ENTER_GETEVENTS) != 0 || min_complete > 0 {
            self.poll_pending(now_ns);
        }

        Ok(submitted)
    }
}

// ---------------------------------------------------------------------------
// Self-Test
// ---------------------------------------------------------------------------

/// Run comprehensive `io_uring` self-tests covering ring descriptors, opcodes,
/// registered buffers/FDs, linked SQE chains, and SQPOLL state transitions.
pub fn self_test() -> Result<(), &'static str> {
    serial_println!("[io_uring] running self_test...");

    // 1. Ring creation & NOP
    let params = IoUringParams {
        sq_entries: 16,
        cq_entries: 32,
        flags: IORING_SETUP_CQSIZE | IORING_SETUP_SQPOLL,
        sq_thread_cpu: 0,
        sq_thread_idle: 50, // 50 ms
        features: 0,
        resv: [0; 4],
    };
    let mut ring = IoUringInstance::new(params).map_err(|_| "failed to create IoUringInstance")?;
    if ring.sq.ring_entries != 16 || ring.cq.ring_entries != 32 {
        return Err("unexpected ring entry counts");
    }

    ring.push_sqe(IoUringSqe::nop(0x1111))
        .map_err(|_| "push NOP failed")?;
    let submitted = ring
        .enter(1, 0, 0, 1_000_000)
        .map_err(|_| "enter NOP failed")?;
    if submitted != 1 {
        return Err("NOP submit count mismatch");
    }
    let cqe = ring.pop_cqe().ok_or("missing NOP CQE")?;
    if cqe.user_data != 0x1111 || cqe.res != 0 {
        return Err("NOP CQE mismatch");
    }

    // 2. Registered buffers & registered FDs (READ_FIXED / WRITE_FIXED / FSYNC)
    let file = Arc::new(MemIoFile::new(b"hello-nexus", 0));
    ring.files
        .register(vec![Some(file.clone())])
        .map_err(|_| "register files failed")?;
    ring.buffers
        .register_sizes(&[64, 64])
        .map_err(|_| "register buffers failed")?;

    ring.push_sqe(IoUringSqe::read_fixed(0, 0, 11, 0, 0x2222).with_flags(IOSQE_FIXED_FILE))
        .map_err(|_| "push READ_FIXED failed")?;
    ring.buffers
        .write_to(1, 0, b"-io_uring")
        .map_err(|_| "write_to registered buffer failed")?;
    ring.push_sqe(IoUringSqe::write_fixed(0, 1, 9, 11, 0x3333).with_flags(IOSQE_FIXED_FILE))
        .map_err(|_| "push WRITE_FIXED failed")?;
    ring.push_sqe(IoUringSqe::fsync(0, true, 0x4444).with_flags(IOSQE_FIXED_FILE))
        .map_err(|_| "push FSYNC failed")?;

    ring.enter(3, 0, 0, 2_000_000)
        .map_err(|_| "enter fixed ops failed")?;

    let c1 = ring.pop_cqe().ok_or("missing READ_FIXED CQE")?;
    let c2 = ring.pop_cqe().ok_or("missing WRITE_FIXED CQE")?;
    let c3 = ring.pop_cqe().ok_or("missing FSYNC CQE")?;
    if c1.res != 11 || c2.res != 9 || c3.res != 0 {
        return Err("fixed op CQE result mismatch");
    }
    if ring.buffers.get(0).map(|b| &b[..11]) != Some(b"hello-nexus") {
        return Err("READ_FIXED buffer content mismatch");
    }
    if file.snapshot().as_slice() != b"hello-nexus-io_uring" {
        return Err("WRITE_FIXED file content mismatch");
    }
    if file.fsync_counts() != (0, 1) {
        return Err("FSYNC datasync count mismatch");
    }

    // 3. Direct FD READ / WRITE
    let mut read_back = vec![0u8; 20];
    ring.insert_direct_fd(7, file.clone())
        .map_err(|_| "insert_direct_fd failed")?;
    ring.push_sqe(IoUringSqe::read(
        7,
        read_back.as_mut_ptr() as u64,
        20,
        0,
        0x5555,
    ))
    .map_err(|_| "push direct READ failed")?;
    ring.enter(1, 0, 0, 3_000_000)
        .map_err(|_| "enter direct READ failed")?;
    let c4 = ring.pop_cqe().ok_or("missing direct READ CQE")?;
    if c4.res != 20 || read_back.as_slice() != b"hello-nexus-io_uring" {
        return Err("direct READ mismatch");
    }

    // 4. POLL_ADD & TIMEOUT async completion
    ring.push_sqe(IoUringSqe::poll_add(7, POLLIN, 0x6666))
        .map_err(|_| "push POLL_ADD failed")?;
    ring.push_sqe(IoUringSqe::timeout(10_000_000, 0, false, 0x7777))
        .map_err(|_| "push TIMEOUT failed")?;
    ring.enter(2, 0, IORING_ENTER_GETEVENTS, 4_000_000)
        .map_err(|_| "enter async ops failed")?;
    if ring.pop_cqe().is_some() {
        return Err("POLL_ADD/TIMEOUT completed prematurely");
    }

    file.set_poll_mask(POLLIN);
    let done = ring.poll_pending(15_000_000);
    if done != 2 {
        return Err("expected both POLL_ADD and TIMEOUT to complete");
    }
    let cp = ring.pop_cqe().ok_or("missing POLL_ADD CQE")?;
    let ct = ring.pop_cqe().ok_or("missing TIMEOUT CQE")?;
    if cp.user_data != 0x6666 || cp.res != POLLIN as i32 {
        return Err("POLL_ADD CQE mismatch");
    }
    if ct.user_data != 0x7777 || ct.res != ETIME {
        return Err("TIMEOUT CQE mismatch");
    }

    // 5. Linked SQE chain cancellation on error
    ring.push_sqe(IoUringSqe::fsync(99, false, 0x8888).with_flags(IOSQE_IO_LINK))
        .map_err(|_| "push linked failing SQE failed")?;
    ring.push_sqe(IoUringSqe::nop(0x9999))
        .map_err(|_| "push linked trailing NOP failed")?;
    ring.enter(2, 0, 0, 16_000_000)
        .map_err(|_| "enter linked chain failed")?;
    let cl1 = ring.pop_cqe().ok_or("missing linked CQE 1")?;
    let cl2 = ring.pop_cqe().ok_or("missing linked CQE 2")?;
    if cl1.res != EBADF || cl2.res != ECANCELED {
        return Err("linked SQE cancellation mismatch");
    }

    // 6. SQPOLL idle park & wakeup
    ring.sqpoll_tick(100_000_000); // > 50 ms idle
    if (ring.sq.flags.load(Ordering::Acquire) & IORING_SQ_NEED_WAKEUP) == 0 {
        return Err("SQPOLL did not set IORING_SQ_NEED_WAKEUP after idle");
    }
    ring.enter(0, 0, IORING_ENTER_SQ_WAKEUP, 101_000_000)
        .map_err(|_| "SQPOLL wakeup failed")?;
    if (ring.sq.flags.load(Ordering::Acquire) & IORING_SQ_NEED_WAKEUP) != 0 {
        return Err("SQPOLL did not clear IORING_SQ_NEED_WAKEUP on wakeup");
    }
    if ring.sqpoll.wakeups != 1 {
        return Err("SQPOLL wakeup counter mismatch");
    }

    serial_println!("[io_uring] self_test passed");
    Ok(())
}
