#![no_std]
#![forbid(unsafe_code)]

//! Clean-room `no_std` + `alloc` Mach / Darwin (XNU) runtime shim for Nexus OS.
//!
//! Translates Mach port/IPC (`mach_task_self`, `mach_msg`) and BSD-layer
//! (`bsd_open`, `bsd_read`, `bsd_write`, `bsd_close`, `bsd_exit`) operations
//! into [`nexus_abi::SyscallNumber`] invocations over an abstract
//! [`DarwinTransport`].

extern crate alloc;

use core::fmt;
use nexus_abi::{AbiError, SyscallNumber};

/// Mach kernel return code (`kern_return_t`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernReturn(pub i32);

impl KernReturn {
    pub const KERN_SUCCESS: Self = Self(0);
    pub const KERN_INVALID_ADDRESS: Self = Self(1);
    pub const KERN_PROTECTION_FAILURE: Self = Self(2);
    pub const KERN_NO_SPACE: Self = Self(3);
    pub const KERN_INVALID_ARGUMENT: Self = Self(4);
    pub const KERN_FAILURE: Self = Self(5);
    pub const KERN_RESOURCE_SHORTAGE: Self = Self(6);
    pub const KERN_NOT_RECEIVER: Self = Self(7);
    pub const KERN_NO_ACCESS: Self = Self(8);
    pub const KERN_NOT_SUPPORTED: Self = Self(46);

    /// Map a Nexus OS [`AbiError`] to the corresponding [`KernReturn`].
    pub const fn from_abi_error(err: AbiError) -> Self {
        match err {
            AbiError::Ok => Self::KERN_SUCCESS,
            AbiError::InvalidHandle => Self::KERN_INVALID_ARGUMENT,
            AbiError::AccessDenied => Self::KERN_NO_ACCESS,
            AbiError::InvalidArg => Self::KERN_INVALID_ARGUMENT,
            AbiError::NotFound => Self::KERN_FAILURE,
            AbiError::NoMemory => Self::KERN_RESOURCE_SHORTAGE,
            AbiError::WouldBlock => Self::KERN_FAILURE,
            AbiError::NotSupported => Self::KERN_NOT_SUPPORTED,
            AbiError::AlreadyExists => Self::KERN_FAILURE,
            AbiError::BadState => Self::KERN_FAILURE,
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

    #[inline]
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for KernReturn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match *self {
            Self::KERN_SUCCESS => "KERN_SUCCESS",
            Self::KERN_INVALID_ADDRESS => "KERN_INVALID_ADDRESS",
            Self::KERN_PROTECTION_FAILURE => "KERN_PROTECTION_FAILURE",
            Self::KERN_NO_SPACE => "KERN_NO_SPACE",
            Self::KERN_INVALID_ARGUMENT => "KERN_INVALID_ARGUMENT",
            Self::KERN_FAILURE => "KERN_FAILURE",
            Self::KERN_RESOURCE_SHORTAGE => "KERN_RESOURCE_SHORTAGE",
            Self::KERN_NOT_RECEIVER => "KERN_NOT_RECEIVER",
            Self::KERN_NO_ACCESS => "KERN_NO_ACCESS",
            Self::KERN_NOT_SUPPORTED => "KERN_NOT_SUPPORTED",
            _ => return write!(f, "KernReturn({})", self.0),
        };
        f.write_str(name)
    }
}

/// Mach port right name (`mach_port_t`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MachPort(pub u32);

impl MachPort {
    pub const MACH_PORT_NULL: Self = Self(0);
    pub const MACH_PORT_DEAD: Self = Self(u32::MAX);

    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != 0 && self.0 != u32::MAX
    }
}

/// Common `mach_msg` option flags.
pub mod mach_msg_option {
    pub const MACH_MSG_OPTION_NONE: u32 = 0x0000_0000;
    pub const MACH_SEND_MSG: u32 = 0x0000_0001;
    pub const MACH_RCV_MSG: u32 = 0x0000_0002;
    pub const MACH_SEND_TIMEOUT: u32 = 0x0000_0010;
    pub const MACH_RCV_TIMEOUT: u32 = 0x0000_0100;
}

/// Transport abstraction for invoking Nexus OS Darwin/XNU-personality syscalls.
pub trait DarwinTransport {
    fn invoke(&mut self, nr: SyscallNumber, args: [u64; 6]) -> Result<u64, AbiError>;
}

/// Mach / Darwin runtime shim backed by a [`DarwinTransport`].
pub struct DarwinRuntime<T: DarwinTransport> {
    transport: T,
}

impl<T: DarwinTransport> DarwinRuntime<T> {
    /// Create a new Darwin runtime.
    pub const fn new(transport: T) -> Self {
        Self { transport }
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

    /// Return the Mach task port for the current process (`mach_task_self`).
    pub fn mach_task_self(&mut self) -> Result<MachPort, KernReturn> {
        let raw = self
            .transport
            .invoke(SyscallNumber::DarwinTaskSelf, [0; 6])
            .map_err(KernReturn::from_abi_error)?;
        let port = u32::try_from(raw)
            .map(MachPort::from_raw)
            .map_err(|_| KernReturn::KERN_FAILURE)?;
        if !port.is_valid() {
            return Err(KernReturn::KERN_INVALID_ARGUMENT);
        }
        Ok(port)
    }

    /// Send and/or receive a Mach IPC message (`mach_msg`).
    pub fn mach_msg(
        &mut self,
        port: MachPort,
        option: u32,
        send_buf: &[u8],
        recv_buf: &mut [u8],
    ) -> Result<usize, KernReturn> {
        if !port.is_valid() {
            return Err(KernReturn::KERN_INVALID_ARGUMENT);
        }
        let args = [
            port.as_raw() as u64,
            option as u64,
            send_buf.as_ptr() as u64,
            send_buf.len() as u64,
            recv_buf.as_mut_ptr() as u64,
            recv_buf.len() as u64,
        ];
        let n = self
            .transport
            .invoke(SyscallNumber::DarwinMachMsg, args)
            .map_err(KernReturn::from_abi_error)?;
        usize::try_from(n).map_err(|_| KernReturn::KERN_FAILURE)
    }

    /// Open a file via the Darwin BSD layer (`open`).
    pub fn bsd_open(&mut self, path: &str, flags: u32, mode: u32) -> Result<i32, KernReturn> {
        if path.is_empty() {
            return Err(KernReturn::KERN_INVALID_ARGUMENT);
        }
        let args = [
            path.as_ptr() as u64,
            path.len() as u64,
            flags as u64,
            mode as u64,
            0,
            0,
        ];
        let raw = self
            .transport
            .invoke(SyscallNumber::DarwinBsdOpen, args)
            .map_err(KernReturn::from_abi_error)?;
        i32::try_from(raw).map_err(|_| KernReturn::KERN_FAILURE)
    }

    /// Read bytes from a BSD file descriptor into `buf` (`read`).
    pub fn bsd_read(&mut self, fd: i32, buf: &mut [u8]) -> Result<usize, KernReturn> {
        if fd < 0 {
            return Err(KernReturn::KERN_INVALID_ARGUMENT);
        }
        if buf.is_empty() {
            return Ok(0);
        }
        let args = [
            fd as u64,
            buf.as_mut_ptr() as u64,
            buf.len() as u64,
            0,
            0,
            0,
        ];
        let n = self
            .transport
            .invoke(SyscallNumber::DarwinBsdRead, args)
            .map_err(KernReturn::from_abi_error)?;
        usize::try_from(n).map_err(|_| KernReturn::KERN_FAILURE)
    }

    /// Write bytes from `buf` to a BSD file descriptor (`write`).
    pub fn bsd_write(&mut self, fd: i32, buf: &[u8]) -> Result<usize, KernReturn> {
        if fd < 0 {
            return Err(KernReturn::KERN_INVALID_ARGUMENT);
        }
        if buf.is_empty() {
            return Ok(0);
        }
        let args = [fd as u64, buf.as_ptr() as u64, buf.len() as u64, 0, 0, 0];
        let n = self
            .transport
            .invoke(SyscallNumber::DarwinBsdWrite, args)
            .map_err(KernReturn::from_abi_error)?;
        usize::try_from(n).map_err(|_| KernReturn::KERN_FAILURE)
    }

    /// Close a BSD file descriptor (`close`).
    pub fn bsd_close(&mut self, fd: i32) -> Result<(), KernReturn> {
        if fd < 0 {
            return Err(KernReturn::KERN_INVALID_ARGUMENT);
        }
        let args = [fd as u64, 0, 0, 0, 0, 0];
        self.transport
            .invoke(SyscallNumber::DarwinBsdClose, args)
            .map(|_| ())
            .map_err(KernReturn::from_abi_error)
    }

    /// Terminate the current process with `status` (`exit`).
    pub fn bsd_exit(&mut self, status: i32) -> Result<(), KernReturn> {
        let args = [status as i64 as u64, 0, 0, 0, 0, 0];
        self.transport
            .invoke(SyscallNumber::ProcessExit, args)
            .map(|_| ())
            .map_err(KernReturn::from_abi_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::vec::Vec;

    struct MockDarwinTransport {
        calls: Vec<(SyscallNumber, [u64; 6])>,
        task_port: u32,
        next_fd: i32,
        fds: BTreeMap<i32, Vec<u8>>,
        fail_next: Option<AbiError>,
    }

    impl MockDarwinTransport {
        fn new() -> Self {
            let mut fds = BTreeMap::new();
            fds.insert(1, Vec::new());
            Self {
                calls: Vec::new(),
                task_port: 0x103,
                next_fd: 3,
                fds,
                fail_next: None,
            }
        }
    }

    impl DarwinTransport for MockDarwinTransport {
        fn invoke(&mut self, nr: SyscallNumber, args: [u64; 6]) -> Result<u64, AbiError> {
            self.calls.push((nr, args));
            if let Some(err) = self.fail_next.take() {
                return Err(err);
            }
            match nr {
                SyscallNumber::DarwinTaskSelf => Ok(self.task_port as u64),
                SyscallNumber::DarwinMachMsg => {
                    let port = args[0] as u32;
                    if port == 0 || port == u32::MAX {
                        return Err(AbiError::InvalidHandle);
                    }
                    let send_len = args[3] as usize;
                    let recv_len = args[5] as usize;
                    Ok(send_len.min(recv_len) as u64)
                }
                SyscallNumber::DarwinBsdOpen => {
                    let path_len = args[1] as usize;
                    if path_len == 0 {
                        return Err(AbiError::InvalidArg);
                    }
                    let fd = self.next_fd;
                    self.next_fd += 1;
                    self.fds.insert(fd, b"darwin-xnu".to_vec());
                    Ok(fd as u64)
                }
                SyscallNumber::DarwinBsdRead => {
                    let fd = args[0] as i32;
                    let len = args[2] as usize;
                    let data = self.fds.get(&fd).ok_or(AbiError::InvalidHandle)?;
                    Ok(data.len().min(len) as u64)
                }
                SyscallNumber::DarwinBsdWrite => {
                    let fd = args[0] as i32;
                    let len = args[2] as usize;
                    if !self.fds.contains_key(&fd) {
                        return Err(AbiError::InvalidHandle);
                    }
                    Ok(len as u64)
                }
                SyscallNumber::DarwinBsdClose => {
                    let fd = args[0] as i32;
                    self.fds
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
    fn kern_return_mapping_from_abi_error() {
        assert_eq!(
            KernReturn::from_abi_error(AbiError::Ok),
            KernReturn::KERN_SUCCESS
        );
        assert_eq!(
            KernReturn::from_abi_error(AbiError::InvalidArg),
            KernReturn::KERN_INVALID_ARGUMENT
        );
        assert_eq!(
            KernReturn::from_abi_error(AbiError::AccessDenied),
            KernReturn::KERN_NO_ACCESS
        );
        assert_eq!(
            KernReturn::from_abi_error(AbiError::NoMemory),
            KernReturn::KERN_RESOURCE_SHORTAGE
        );
        assert_eq!(
            KernReturn::from_abi_error(AbiError::BadState),
            KernReturn::KERN_FAILURE
        );
        assert!(KernReturn::KERN_SUCCESS.is_success());
        assert!(!KernReturn::KERN_FAILURE.is_success());
    }

    #[test]
    fn mach_and_bsd_lifecycle() {
        let mut rt = DarwinRuntime::new(MockDarwinTransport::new());

        let task = rt.mach_task_self().expect("mach_task_self should succeed");
        assert!(task.is_valid());
        assert_eq!(task, MachPort(0x103));

        let mut recv = [0u8; 64];
        let msg_len = rt
            .mach_msg(
                task,
                mach_msg_option::MACH_SEND_MSG | mach_msg_option::MACH_RCV_MSG,
                b"mach-ipc-ping",
                &mut recv,
            )
            .expect("mach_msg should succeed");
        assert_eq!(msg_len, 13);

        let fd = rt
            .bsd_open("/System/Library/CoreServices", 0, 0o644)
            .expect("bsd_open should succeed");
        assert_eq!(fd, 3);

        let mut buf = [0u8; 32];
        let n_read = rt.bsd_read(fd, &mut buf).expect("bsd_read should succeed");
        assert_eq!(n_read, 10);

        let n_written = rt
            .bsd_write(1, b"hello darwin\n")
            .expect("bsd_write should succeed");
        assert_eq!(n_written, 13);

        rt.bsd_close(fd).expect("bsd_close should succeed");
        assert_eq!(rt.bsd_close(fd), Err(KernReturn::KERN_INVALID_ARGUMENT));

        rt.bsd_exit(0).expect("bsd_exit should succeed");

        let calls = &rt.transport().calls;
        assert_eq!(calls[0].0, SyscallNumber::DarwinTaskSelf);
        assert_eq!(calls[1].0, SyscallNumber::DarwinMachMsg);
        assert_eq!(calls[2].0, SyscallNumber::DarwinBsdOpen);
        assert_eq!(calls[3].0, SyscallNumber::DarwinBsdRead);
        assert_eq!(calls[4].0, SyscallNumber::DarwinBsdWrite);
        assert_eq!(calls[5].0, SyscallNumber::DarwinBsdClose);
        assert_eq!(calls[7].0, SyscallNumber::ProcessExit);
    }

    #[test]
    fn invalid_inputs_and_transport_errors() {
        let mut rt = DarwinRuntime::new(MockDarwinTransport::new());

        assert_eq!(
            rt.mach_msg(
                MachPort::MACH_PORT_NULL,
                mach_msg_option::MACH_SEND_MSG,
                b"hi",
                &mut []
            ),
            Err(KernReturn::KERN_INVALID_ARGUMENT)
        );
        assert_eq!(
            rt.bsd_open("", 0, 0),
            Err(KernReturn::KERN_INVALID_ARGUMENT)
        );
        assert_eq!(
            rt.bsd_read(-1, &mut [0u8; 4]),
            Err(KernReturn::KERN_INVALID_ARGUMENT)
        );
        assert_eq!(
            rt.bsd_write(-1, b"hi"),
            Err(KernReturn::KERN_INVALID_ARGUMENT)
        );
        assert_eq!(rt.bsd_close(-1), Err(KernReturn::KERN_INVALID_ARGUMENT));

        rt.transport_mut().fail_next = Some(AbiError::AccessDenied);
        assert_eq!(
            rt.bsd_open("/private/etc/master.passwd", 0, 0),
            Err(KernReturn::KERN_NO_ACCESS)
        );

        rt.transport_mut().fail_next = Some(AbiError::NoMemory);
        assert_eq!(
            rt.mach_msg(
                MachPort(0x103),
                mach_msg_option::MACH_SEND_MSG,
                b"hi",
                &mut []
            ),
            Err(KernReturn::KERN_RESOURCE_SHORTAGE)
        );
    }
}
