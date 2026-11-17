#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Clean-room `no_std` + `alloc` POSIX runtime shim for Nexus OS.
//!
//! Translates POSIX-style file descriptor operations (`open`, `read`, `write`,
//! `close`, `getpid`, `exit`) into [`nexus_abi::SyscallNumber`] invocations over
//! an abstract [`SyscallTransport`].

extern crate alloc;

use core::fmt;
use nexus_abi::{AbiError, SyscallNumber};

/// Special `dirfd` value meaning "current working directory" for `openat`.
pub const AT_FDCWD: i32 = -100;

/// POSIX error number (`errno`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Errno(pub i32);

impl Errno {
    pub const EPERM: Self = Self(1);
    pub const ENOENT: Self = Self(2);
    pub const EIO: Self = Self(5);
    pub const EBADF: Self = Self(9);
    pub const EAGAIN: Self = Self(11);
    pub const ENOMEM: Self = Self(12);
    pub const EACCES: Self = Self(13);
    pub const EEXIST: Self = Self(17);
    pub const EINVAL: Self = Self(22);
    pub const ENOSYS: Self = Self(38);

    /// Map a Nexus OS [`AbiError`] to the corresponding POSIX [`Errno`].
    pub const fn from_abi_error(err: AbiError) -> Self {
        match err {
            AbiError::Ok => Self(0),
            AbiError::InvalidHandle => Self::EBADF,
            AbiError::AccessDenied => Self::EACCES,
            AbiError::InvalidArg => Self::EINVAL,
            AbiError::NotFound => Self::ENOENT,
            AbiError::NoMemory => Self::ENOMEM,
            AbiError::WouldBlock => Self::EAGAIN,
            AbiError::NotSupported => Self::ENOSYS,
            AbiError::AlreadyExists => Self::EEXIST,
            AbiError::BadState => Self::EIO,
        }
    }

    #[inline]
    pub const fn from_i32(code: i32) -> Self {
        Self(code)
    }

    #[inline]
    pub const fn as_i32(self) -> i32 {
        self.0
    }
}

impl fmt::Display for Errno {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match *self {
            Self::EPERM => "EPERM",
            Self::ENOENT => "ENOENT",
            Self::EIO => "EIO",
            Self::EBADF => "EBADF",
            Self::EAGAIN => "EAGAIN",
            Self::ENOMEM => "ENOMEM",
            Self::EACCES => "EACCES",
            Self::EEXIST => "EEXIST",
            Self::EINVAL => "EINVAL",
            Self::ENOSYS => "ENOSYS",
            _ => return write!(f, "Errno({})", self.0),
        };
        f.write_str(name)
    }
}

/// POSIX file descriptor wrapper.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fd(pub i32);

impl Fd {
    pub const STDIN: Self = Self(0);
    pub const STDOUT: Self = Self(1);
    pub const STDERR: Self = Self(2);
    pub const INVALID: Self = Self(-1);

    #[inline]
    pub const fn from_raw(fd: i32) -> Self {
        Self(fd)
    }

    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 >= 0
    }
}

/// POSIX `open` / `openat` flags bitmask.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpenFlags(u32);

impl OpenFlags {
    pub const O_RDONLY: Self = Self(0);
    pub const O_WRONLY: Self = Self(1 << 0);
    pub const O_RDWR: Self = Self(1 << 1);
    pub const O_CREAT: Self = Self(1 << 6);
    pub const O_EXCL: Self = Self(1 << 7);
    pub const O_TRUNC: Self = Self(1 << 9);
    pub const O_APPEND: Self = Self(1 << 10);
    pub const O_NONBLOCK: Self = Self(1 << 11);
    pub const O_CLOEXEC: Self = Self(1 << 19);

    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
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
}

/// POSIX `lseek` whence parameter.
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SeekWhence {
    Set = 0,
    Cur = 1,
    End = 2,
}

impl SeekWhence {
    pub const fn from_i32(val: i32) -> Option<Self> {
        match val {
            0 => Some(Self::Set),
            1 => Some(Self::Cur),
            2 => Some(Self::End),
            _ => None,
        }
    }
}

/// POSIX `stat` metadata structure.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PosixStat {
    pub st_dev: u64,
    pub st_ino: u64,
    pub st_mode: u32,
    pub st_nlink: u32,
    pub st_uid: u32,
    pub st_gid: u32,
    pub st_rdev: u64,
    pub st_size: i64,
    pub st_blksize: i64,
    pub st_blocks: i64,
    pub st_atime: i64,
    pub st_mtime: i64,
    pub st_ctime: i64,
}

/// Transport abstraction for invoking Nexus OS syscalls from user space.
pub trait SyscallTransport {
    fn invoke(&mut self, nr: SyscallNumber, args: [u64; 6]) -> Result<u64, AbiError>;
}

/// POSIX runtime shim backed by a [`SyscallTransport`].
pub struct PosixRuntime<T: SyscallTransport> {
    transport: T,
    pid: i32,
}

impl<T: SyscallTransport> PosixRuntime<T> {
    /// Create a new runtime with default PID `1`.
    pub const fn new(transport: T) -> Self {
        Self { transport, pid: 1 }
    }

    /// Create a new runtime with an explicit PID.
    pub const fn with_pid(transport: T, pid: i32) -> Self {
        Self { transport, pid }
    }

    /// Borrow the underlying syscall transport.
    pub const fn transport(&self) -> &T {
        &self.transport
    }

    /// Mutably borrow the underlying syscall transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Consume the runtime and return the underlying syscall transport.
    pub fn into_inner(self) -> T {
        self.transport
    }

    /// Open a file relative to the current working directory (`AT_FDCWD`).
    pub fn open(&mut self, path: &str, flags: OpenFlags) -> Result<Fd, Errno> {
        if path.is_empty() {
            return Err(Errno::EINVAL);
        }
        let args = [
            AT_FDCWD as i64 as u64,
            path.as_ptr() as u64,
            path.len() as u64,
            flags.bits() as u64,
            0,
            0,
        ];
        let raw = self
            .transport
            .invoke(SyscallNumber::PosixOpenAt, args)
            .map_err(Errno::from_abi_error)?;
        i32::try_from(raw).map(Fd).map_err(|_| Errno::EIO)
    }

    /// Read bytes from an open file descriptor into `buf`.
    pub fn read(&mut self, fd: Fd, buf: &mut [u8]) -> Result<usize, Errno> {
        if !fd.is_valid() {
            return Err(Errno::EBADF);
        }
        if buf.is_empty() {
            return Ok(0);
        }
        let args = [
            fd.as_raw() as u64,
            buf.as_mut_ptr() as u64,
            buf.len() as u64,
            0,
            0,
            0,
        ];
        let n = self
            .transport
            .invoke(SyscallNumber::PosixRead, args)
            .map_err(Errno::from_abi_error)?;
        usize::try_from(n).map_err(|_| Errno::EIO)
    }

    /// Write bytes from `buf` to an open file descriptor.
    pub fn write(&mut self, fd: Fd, buf: &[u8]) -> Result<usize, Errno> {
        if !fd.is_valid() {
            return Err(Errno::EBADF);
        }
        if buf.is_empty() {
            return Ok(0);
        }
        let args = [
            fd.as_raw() as u64,
            buf.as_ptr() as u64,
            buf.len() as u64,
            0,
            0,
            0,
        ];
        let n = self
            .transport
            .invoke(SyscallNumber::PosixWrite, args)
            .map_err(Errno::from_abi_error)?;
        usize::try_from(n).map_err(|_| Errno::EIO)
    }

    /// Close an open file descriptor.
    pub fn close(&mut self, fd: Fd) -> Result<(), Errno> {
        if !fd.is_valid() {
            return Err(Errno::EBADF);
        }
        let args = [fd.as_raw() as u64, 0, 0, 0, 0, 0];
        self.transport
            .invoke(SyscallNumber::PosixClose, args)
            .map(|_| ())
            .map_err(Errno::from_abi_error)
    }

    /// Return the current process ID.
    pub const fn getpid(&self) -> i32 {
        self.pid
    }

    /// Terminate the current process with `status`.
    pub fn exit(&mut self, status: i32) -> Result<(), Errno> {
        let args = [status as i64 as u64, 0, 0, 0, 0, 0];
        self.transport
            .invoke(SyscallNumber::ProcessExit, args)
            .map(|_| ())
            .map_err(Errno::from_abi_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::vec::Vec;

    struct MockTransport {
        calls: Vec<(SyscallNumber, [u64; 6])>,
        next_fd: i32,
        open_fds: BTreeMap<i32, Vec<u8>>,
        fail_next: Option<AbiError>,
    }

    impl MockTransport {
        fn new() -> Self {
            let mut open_fds = BTreeMap::new();
            open_fds.insert(1, Vec::new()); // stdout
            Self {
                calls: Vec::new(),
                next_fd: 3,
                open_fds,
                fail_next: None,
            }
        }
    }

    impl SyscallTransport for MockTransport {
        fn invoke(&mut self, nr: SyscallNumber, args: [u64; 6]) -> Result<u64, AbiError> {
            self.calls.push((nr, args));
            if let Some(err) = self.fail_next.take() {
                return Err(err);
            }
            match nr {
                SyscallNumber::PosixOpenAt => {
                    let path_len = args[2] as usize;
                    if path_len == 0 {
                        return Err(AbiError::InvalidArg);
                    }
                    let fd = self.next_fd;
                    self.next_fd += 1;
                    self.open_fds.insert(fd, b"hello nexus".to_vec());
                    Ok(fd as u64)
                }
                SyscallNumber::PosixRead => {
                    let fd = args[0] as i32;
                    let len = args[2] as usize;
                    let data = self.open_fds.get(&fd).ok_or(AbiError::InvalidHandle)?;
                    Ok(data.len().min(len) as u64)
                }
                SyscallNumber::PosixWrite => {
                    let fd = args[0] as i32;
                    let len = args[2] as usize;
                    if !self.open_fds.contains_key(&fd) {
                        return Err(AbiError::InvalidHandle);
                    }
                    Ok(len as u64)
                }
                SyscallNumber::PosixClose => {
                    let fd = args[0] as i32;
                    self.open_fds
                        .remove(&fd)
                        .map(|_| 0)
                        .ok_or(AbiError::InvalidHandle)
                }
                SyscallNumber::ProcessExit => Ok(0),
                _ => Err(AbiError::NotSupported),
            }
        }
    }

    #[test]
    fn errno_mapping_from_abi_error() {
        assert_eq!(Errno::from_abi_error(AbiError::NotFound), Errno::ENOENT);
        assert_eq!(Errno::from_abi_error(AbiError::AccessDenied), Errno::EACCES);
        assert_eq!(Errno::from_abi_error(AbiError::InvalidHandle), Errno::EBADF);
        assert_eq!(Errno::from_abi_error(AbiError::InvalidArg), Errno::EINVAL);
        assert_eq!(Errno::from_abi_error(AbiError::NoMemory), Errno::ENOMEM);
        assert_eq!(Errno::from_abi_error(AbiError::NotSupported), Errno::ENOSYS);
        assert_eq!(Errno::from_abi_error(AbiError::BadState), Errno::EIO);
        assert_eq!(Errno::from_abi_error(AbiError::WouldBlock), Errno::EAGAIN);
        assert_eq!(
            Errno::from_abi_error(AbiError::AlreadyExists),
            Errno::EEXIST
        );
        assert_eq!(Errno::from_abi_error(AbiError::Ok), Errno(0));
    }

    #[test]
    fn open_read_write_close_and_getpid() {
        let mut rt = PosixRuntime::with_pid(MockTransport::new(), 42);
        assert_eq!(rt.getpid(), 42);

        let fd = rt
            .open("/etc/motd", OpenFlags::O_RDONLY.union(OpenFlags::O_CLOEXEC))
            .expect("open should succeed");
        assert_eq!(fd, Fd(3));

        let mut buf = [0u8; 16];
        let n_read = rt.read(fd, &mut buf).expect("read should succeed");
        assert_eq!(n_read, 11);

        let n_written = rt.write(Fd::STDOUT, b"ok\n").expect("write should succeed");
        assert_eq!(n_written, 3);

        rt.close(fd).expect("close should succeed");
        assert_eq!(rt.close(fd), Err(Errno::EBADF));

        rt.exit(0).expect("exit should succeed");

        let calls = &rt.transport().calls;
        assert_eq!(calls[0].0, SyscallNumber::PosixOpenAt);
        assert_eq!(calls[0].1[0], AT_FDCWD as i64 as u64);
        assert_eq!(calls[1].0, SyscallNumber::PosixRead);
        assert_eq!(calls[2].0, SyscallNumber::PosixWrite);
        assert_eq!(calls[3].0, SyscallNumber::PosixClose);
        assert_eq!(calls[5].0, SyscallNumber::ProcessExit);
    }

    #[test]
    fn invalid_inputs_and_transport_errors() {
        let mut rt = PosixRuntime::new(MockTransport::new());
        assert_eq!(rt.getpid(), 1);
        assert_eq!(rt.open("", OpenFlags::O_RDONLY), Err(Errno::EINVAL));
        assert_eq!(rt.read(Fd::INVALID, &mut [0u8; 4]), Err(Errno::EBADF));
        assert_eq!(rt.write(Fd::INVALID, b"hi"), Err(Errno::EBADF));
        assert_eq!(rt.close(Fd::INVALID), Err(Errno::EBADF));

        rt.transport_mut().fail_next = Some(AbiError::NotFound);
        assert_eq!(rt.open("/missing", OpenFlags::O_RDONLY), Err(Errno::ENOENT));

        rt.transport_mut().fail_next = Some(AbiError::AccessDenied);
        assert_eq!(rt.open("/secret", OpenFlags::O_RDONLY), Err(Errno::EACCES));
    }

    #[test]
    fn types_and_flags_basics() {
        let flags = OpenFlags::O_RDWR.union(OpenFlags::O_CREAT);
        assert!(flags.contains(OpenFlags::O_RDWR));
        assert!(flags.contains(OpenFlags::O_CREAT));
        assert!(!flags.contains(OpenFlags::O_TRUNC));
        assert_eq!(SeekWhence::from_i32(0), Some(SeekWhence::Set));
        assert_eq!(SeekWhence::from_i32(1), Some(SeekWhence::Cur));
        assert_eq!(SeekWhence::from_i32(2), Some(SeekWhence::End));
        assert_eq!(SeekWhence::from_i32(99), None);

        let st = PosixStat::default();
        assert_eq!(st.st_size, 0);
    }
}
