#![no_std]
#![forbid(unsafe_code)]

//! Clean-room `no_std` + `alloc` Windows NT / Win32 runtime shim for Nexus OS.
//!
//! Translates Win32/NT-style handle, file, virtual memory, and process
//! operations into [`nexus_abi::SyscallNumber`] invocations over an abstract
//! [`Win32Transport`].

extern crate alloc;

use core::fmt;
use nexus_abi::{AbiError, SyscallNumber};

/// Windows NT status code (`NTSTATUS`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NtStatus(pub u32);

impl NtStatus {
    pub const STATUS_SUCCESS: Self = Self(0x0000_0000);
    pub const STATUS_UNSUCCESSFUL: Self = Self(0xC000_0001);
    pub const STATUS_NOT_IMPLEMENTED: Self = Self(0xC000_0002);
    pub const STATUS_INVALID_HANDLE: Self = Self(0xC000_0008);
    pub const STATUS_INVALID_PARAMETER: Self = Self(0xC000_000D);
    pub const STATUS_NO_MEMORY: Self = Self(0xC000_0017);
    pub const STATUS_ACCESS_DENIED: Self = Self(0xC000_0022);
    pub const STATUS_OBJECT_NAME_NOT_FOUND: Self = Self(0xC000_0034);
    pub const STATUS_OBJECT_NAME_COLLISION: Self = Self(0xC000_0035);
    pub const STATUS_PENDING: Self = Self(0x0000_0103);

    /// Map a Nexus OS [`AbiError`] to the corresponding [`NtStatus`].
    pub const fn from_abi_error(err: AbiError) -> Self {
        match err {
            AbiError::Ok => Self::STATUS_SUCCESS,
            AbiError::InvalidHandle => Self::STATUS_INVALID_HANDLE,
            AbiError::AccessDenied => Self::STATUS_ACCESS_DENIED,
            AbiError::InvalidArg => Self::STATUS_INVALID_PARAMETER,
            AbiError::NotFound => Self::STATUS_OBJECT_NAME_NOT_FOUND,
            AbiError::NoMemory => Self::STATUS_NO_MEMORY,
            AbiError::WouldBlock => Self::STATUS_PENDING,
            AbiError::NotSupported => Self::STATUS_NOT_IMPLEMENTED,
            AbiError::AlreadyExists => Self::STATUS_OBJECT_NAME_COLLISION,
            AbiError::BadState => Self::STATUS_UNSUCCESSFUL,
        }
    }

    #[inline]
    pub const fn from_u32(code: u32) -> Self {
        Self(code)
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Returns `true` if the status represents success (severity bit 31 is 0).
    #[inline]
    pub const fn is_success(self) -> bool {
        (self.0 & 0x8000_0000) == 0
    }
}

impl fmt::Display for NtStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match *self {
            Self::STATUS_SUCCESS => "STATUS_SUCCESS",
            Self::STATUS_UNSUCCESSFUL => "STATUS_UNSUCCESSFUL",
            Self::STATUS_NOT_IMPLEMENTED => "STATUS_NOT_IMPLEMENTED",
            Self::STATUS_INVALID_HANDLE => "STATUS_INVALID_HANDLE",
            Self::STATUS_INVALID_PARAMETER => "STATUS_INVALID_PARAMETER",
            Self::STATUS_NO_MEMORY => "STATUS_NO_MEMORY",
            Self::STATUS_ACCESS_DENIED => "STATUS_ACCESS_DENIED",
            Self::STATUS_OBJECT_NAME_NOT_FOUND => "STATUS_OBJECT_NAME_NOT_FOUND",
            Self::STATUS_OBJECT_NAME_COLLISION => "STATUS_OBJECT_NAME_COLLISION",
            Self::STATUS_PENDING => "STATUS_PENDING",
            _ => return write!(f, "NtStatus(0x{:08X})", self.0),
        };
        f.write_str(name)
    }
}

/// Windows NT / Win32 object handle (`HANDLE`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WinHandle(pub u64);

impl WinHandle {
    pub const NULL: Self = Self(0);
    pub const INVALID: Self = Self(u64::MAX);

    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != 0 && self.0 != u64::MAX
    }
}

/// Common Win32 `CreateFile` access rights.
pub mod access {
    pub const GENERIC_READ: u32 = 0x8000_0000;
    pub const GENERIC_WRITE: u32 = 0x4000_0000;
    pub const GENERIC_EXECUTE: u32 = 0x2000_0000;
    pub const GENERIC_ALL: u32 = 0x1000_0000;
}

/// Common Win32 `CreateFile` creation dispositions.
pub mod disposition {
    pub const CREATE_NEW: u32 = 1;
    pub const CREATE_ALWAYS: u32 = 2;
    pub const OPEN_EXISTING: u32 = 3;
    pub const OPEN_ALWAYS: u32 = 4;
    pub const TRUNCATE_EXISTING: u32 = 5;
}

/// Common Win32 `VirtualAlloc` allocation types and page protections.
pub mod memory {
    pub const MEM_COMMIT: u32 = 0x0000_1000;
    pub const MEM_RESERVE: u32 = 0x0000_2000;
    pub const MEM_RELEASE: u32 = 0x0000_8000;

    pub const PAGE_NOACCESS: u32 = 0x01;
    pub const PAGE_READONLY: u32 = 0x02;
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const PAGE_EXECUTE_READ: u32 = 0x20;
    pub const PAGE_EXECUTE_READWRITE: u32 = 0x40;
}

/// Transport abstraction for invoking Nexus OS NT-personality syscalls.
pub trait Win32Transport {
    fn invoke(&mut self, nr: SyscallNumber, args: [u64; 6]) -> Result<u64, AbiError>;
}

/// Windows NT / Win32 runtime shim backed by a [`Win32Transport`].
pub struct Win32Runtime<T: Win32Transport> {
    transport: T,
    last_error: NtStatus,
}

impl<T: Win32Transport> Win32Runtime<T> {
    /// Create a new Win32 runtime with `last_error` initialized to `STATUS_SUCCESS`.
    pub const fn new(transport: T) -> Self {
        Self {
            transport,
            last_error: NtStatus::STATUS_SUCCESS,
        }
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

    /// Return the last [`NtStatus`] error recorded by this runtime.
    pub const fn get_last_error(&self) -> NtStatus {
        self.last_error
    }

    fn record_err(&mut self, status: NtStatus) -> NtStatus {
        self.last_error = status;
        status
    }

    /// Open or create a file (`NtCreateFile` / `CreateFileW`).
    pub fn create_file(
        &mut self,
        path: &str,
        desired_access: u32,
        create_disposition: u32,
    ) -> Result<WinHandle, NtStatus> {
        if path.is_empty() {
            return Err(self.record_err(NtStatus::STATUS_INVALID_PARAMETER));
        }
        let args = [
            path.as_ptr() as u64,
            path.len() as u64,
            desired_access as u64,
            create_disposition as u64,
            0,
            0,
        ];
        match self.transport.invoke(SyscallNumber::NtCreateFile, args) {
            Ok(raw) => {
                let handle = WinHandle::from_raw(raw);
                if !handle.is_valid() {
                    return Err(self.record_err(NtStatus::STATUS_INVALID_HANDLE));
                }
                Ok(handle)
            }
            Err(err) => Err(self.record_err(NtStatus::from_abi_error(err))),
        }
    }

    /// Read bytes from an open handle into `buf` (`NtReadFile` / `ReadFile`).
    pub fn read_file(&mut self, handle: WinHandle, buf: &mut [u8]) -> Result<usize, NtStatus> {
        if !handle.is_valid() {
            return Err(self.record_err(NtStatus::STATUS_INVALID_HANDLE));
        }
        if buf.is_empty() {
            return Ok(0);
        }
        let args = [
            handle.as_raw(),
            buf.as_mut_ptr() as u64,
            buf.len() as u64,
            0,
            0,
            0,
        ];
        match self.transport.invoke(SyscallNumber::NtReadFile, args) {
            Ok(n) => usize::try_from(n).map_err(|_| self.record_err(NtStatus::STATUS_UNSUCCESSFUL)),
            Err(err) => Err(self.record_err(NtStatus::from_abi_error(err))),
        }
    }

    /// Write bytes from `buf` to an open handle (`NtWriteFile` / `WriteFile`).
    pub fn write_file(&mut self, handle: WinHandle, buf: &[u8]) -> Result<usize, NtStatus> {
        if !handle.is_valid() {
            return Err(self.record_err(NtStatus::STATUS_INVALID_HANDLE));
        }
        if buf.is_empty() {
            return Ok(0);
        }
        let args = [
            handle.as_raw(),
            buf.as_ptr() as u64,
            buf.len() as u64,
            0,
            0,
            0,
        ];
        match self.transport.invoke(SyscallNumber::NtWriteFile, args) {
            Ok(n) => usize::try_from(n).map_err(|_| self.record_err(NtStatus::STATUS_UNSUCCESSFUL)),
            Err(err) => Err(self.record_err(NtStatus::from_abi_error(err))),
        }
    }

    /// Close an open handle (`NtClose` / `CloseHandle`).
    pub fn close_handle(&mut self, handle: WinHandle) -> Result<(), NtStatus> {
        if !handle.is_valid() {
            return Err(self.record_err(NtStatus::STATUS_INVALID_HANDLE));
        }
        let args = [handle.as_raw(), 0, 0, 0, 0, 0];
        match self.transport.invoke(SyscallNumber::NtClose, args) {
            Ok(_) => Ok(()),
            Err(err) => Err(self.record_err(NtStatus::from_abi_error(err))),
        }
    }

    /// Reserve or commit virtual memory (`NtAllocateVirtualMemory` / `VirtualAlloc`).
    pub fn virtual_alloc(
        &mut self,
        base_addr: u64,
        size: usize,
        alloc_type: u32,
        protect: u32,
    ) -> Result<u64, NtStatus> {
        if size == 0 {
            return Err(self.record_err(NtStatus::STATUS_INVALID_PARAMETER));
        }
        let args = [
            base_addr,
            size as u64,
            alloc_type as u64,
            protect as u64,
            0,
            0,
        ];
        match self
            .transport
            .invoke(SyscallNumber::NtAllocateVirtualMemory, args)
        {
            Ok(addr) => Ok(addr),
            Err(err) => Err(self.record_err(NtStatus::from_abi_error(err))),
        }
    }

    /// Terminate the current process with `exit_code` (`ExitProcess`).
    pub fn exit_process(&mut self, exit_code: u32) -> Result<(), NtStatus> {
        let args = [exit_code as u64, 0, 0, 0, 0, 0];
        match self.transport.invoke(SyscallNumber::ProcessExit, args) {
            Ok(_) => Ok(()),
            Err(err) => Err(self.record_err(NtStatus::from_abi_error(err))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::vec::Vec;

    struct MockWin32Transport {
        calls: Vec<(SyscallNumber, [u64; 6])>,
        next_handle: u64,
        handles: BTreeMap<u64, Vec<u8>>,
        next_alloc: u64,
        fail_next: Option<AbiError>,
    }

    impl MockWin32Transport {
        fn new() -> Self {
            Self {
                calls: Vec::new(),
                next_handle: 0x100,
                handles: BTreeMap::new(),
                next_alloc: 0x0000_7FF0_0000_0000,
                fail_next: None,
            }
        }
    }

    impl Win32Transport for MockWin32Transport {
        fn invoke(&mut self, nr: SyscallNumber, args: [u64; 6]) -> Result<u64, AbiError> {
            self.calls.push((nr, args));
            if let Some(err) = self.fail_next.take() {
                return Err(err);
            }
            match nr {
                SyscallNumber::NtCreateFile => {
                    let path_len = args[1] as usize;
                    if path_len == 0 {
                        return Err(AbiError::InvalidArg);
                    }
                    let h = self.next_handle;
                    self.next_handle += 4;
                    self.handles.insert(h, b"win32-data".to_vec());
                    Ok(h)
                }
                SyscallNumber::NtReadFile => {
                    let h = args[0];
                    let len = args[2] as usize;
                    let data = self.handles.get(&h).ok_or(AbiError::InvalidHandle)?;
                    Ok(data.len().min(len) as u64)
                }
                SyscallNumber::NtWriteFile => {
                    let h = args[0];
                    let len = args[2] as usize;
                    if !self.handles.contains_key(&h) {
                        return Err(AbiError::InvalidHandle);
                    }
                    Ok(len as u64)
                }
                SyscallNumber::NtClose => {
                    let h = args[0];
                    self.handles
                        .remove(&h)
                        .map(|_| 0)
                        .ok_or(AbiError::InvalidHandle)
                }
                SyscallNumber::NtAllocateVirtualMemory => {
                    let base = args[0];
                    let size = args[1];
                    if size == 0 {
                        return Err(AbiError::InvalidArg);
                    }
                    if base != 0 {
                        Ok(base)
                    } else {
                        let addr = self.next_alloc;
                        self.next_alloc += (size + 0xFFF) & !0xFFF;
                        Ok(addr)
                    }
                }
                SyscallNumber::ProcessExit => Ok(0),
                _ => Err(AbiError::NotSupported),
            }
        }
    }

    #[test]
    fn nt_status_mapping_from_abi_error() {
        assert_eq!(
            NtStatus::from_abi_error(AbiError::Ok),
            NtStatus::STATUS_SUCCESS
        );
        assert_eq!(
            NtStatus::from_abi_error(AbiError::AccessDenied),
            NtStatus::STATUS_ACCESS_DENIED
        );
        assert_eq!(
            NtStatus::from_abi_error(AbiError::NotFound),
            NtStatus::STATUS_OBJECT_NAME_NOT_FOUND
        );
        assert_eq!(
            NtStatus::from_abi_error(AbiError::InvalidHandle),
            NtStatus::STATUS_INVALID_HANDLE
        );
        assert_eq!(
            NtStatus::from_abi_error(AbiError::NoMemory),
            NtStatus::STATUS_NO_MEMORY
        );
        assert_eq!(
            NtStatus::from_abi_error(AbiError::NotSupported),
            NtStatus::STATUS_NOT_IMPLEMENTED
        );
        assert!(NtStatus::STATUS_SUCCESS.is_success());
        assert!(!NtStatus::STATUS_ACCESS_DENIED.is_success());
    }

    #[test]
    fn file_virtual_alloc_and_exit_flow() {
        let mut rt = Win32Runtime::new(MockWin32Transport::new());
        assert_eq!(rt.get_last_error(), NtStatus::STATUS_SUCCESS);

        let handle = rt
            .create_file(
                "C:\\nexus\\test.txt",
                access::GENERIC_READ | access::GENERIC_WRITE,
                disposition::OPEN_EXISTING,
            )
            .expect("create_file should succeed");
        assert!(handle.is_valid());
        assert_eq!(handle, WinHandle(0x100));

        let mut buf = [0u8; 32];
        let n_read = rt
            .read_file(handle, &mut buf)
            .expect("read_file should succeed");
        assert_eq!(n_read, 10);

        let n_written = rt
            .write_file(handle, b"hello nt")
            .expect("write_file should succeed");
        assert_eq!(n_written, 8);

        let addr = rt
            .virtual_alloc(
                0,
                4096,
                memory::MEM_COMMIT | memory::MEM_RESERVE,
                memory::PAGE_READWRITE,
            )
            .expect("virtual_alloc should succeed");
        assert_eq!(addr, 0x0000_7FF0_0000_0000);

        rt.close_handle(handle)
            .expect("close_handle should succeed");
        assert_eq!(
            rt.close_handle(handle),
            Err(NtStatus::STATUS_INVALID_HANDLE)
        );
        assert_eq!(rt.get_last_error(), NtStatus::STATUS_INVALID_HANDLE);

        rt.exit_process(0).expect("exit_process should succeed");

        let calls = &rt.transport().calls;
        assert_eq!(calls[0].0, SyscallNumber::NtCreateFile);
        assert_eq!(calls[1].0, SyscallNumber::NtReadFile);
        assert_eq!(calls[2].0, SyscallNumber::NtWriteFile);
        assert_eq!(calls[3].0, SyscallNumber::NtAllocateVirtualMemory);
        assert_eq!(calls[4].0, SyscallNumber::NtClose);
        assert_eq!(calls[6].0, SyscallNumber::ProcessExit);
    }

    #[test]
    fn invalid_inputs_and_last_error_tracking() {
        let mut rt = Win32Runtime::new(MockWin32Transport::new());

        assert_eq!(
            rt.create_file("", access::GENERIC_READ, disposition::OPEN_EXISTING),
            Err(NtStatus::STATUS_INVALID_PARAMETER)
        );
        assert_eq!(rt.get_last_error(), NtStatus::STATUS_INVALID_PARAMETER);

        assert_eq!(
            rt.read_file(WinHandle::INVALID, &mut [0u8; 4]),
            Err(NtStatus::STATUS_INVALID_HANDLE)
        );
        assert_eq!(
            rt.write_file(WinHandle::NULL, b"abc"),
            Err(NtStatus::STATUS_INVALID_HANDLE)
        );
        assert_eq!(
            rt.virtual_alloc(0, 0, memory::MEM_COMMIT, memory::PAGE_READWRITE),
            Err(NtStatus::STATUS_INVALID_PARAMETER)
        );

        rt.transport_mut().fail_next = Some(AbiError::NoMemory);
        assert_eq!(
            rt.virtual_alloc(0, 8192, memory::MEM_COMMIT, memory::PAGE_READWRITE),
            Err(NtStatus::STATUS_NO_MEMORY)
        );
        assert_eq!(rt.get_last_error(), NtStatus::STATUS_NO_MEMORY);
    }
}
