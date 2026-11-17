//! Unified multi-personality syscall dispatcher.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

use nexus_abi::{AbiError, Personality, SyscallNumber};

use crate::mm::{self, SpinLock};
use crate::serial_println;
use crate::vfs::{self, OpenFlags, VfsNode};

struct OpenFile {
    node: Arc<dyn VfsNode>,
    offset: usize,
}

static NEXT_FD: AtomicU64 = AtomicU64::new(3); // 0, 1, 2 reserved for stdio
static FD_TABLE: SpinLock<BTreeMap<u64, OpenFile>> = SpinLock::new(BTreeMap::new());

/// Dispatch a syscall for the given execution personality.
pub fn dispatch(personality: Personality, raw_num: u64, args: [u64; 6]) -> Result<u64, AbiError> {
    let syscall = SyscallNumber::from_u64(raw_num).ok_or(AbiError::NotSupported)?;

    if syscall.personality() != personality {
        return Err(AbiError::AccessDenied);
    }

    match personality {
        Personality::Native => dispatch_native(syscall, args),
        Personality::Posix => dispatch_posix(syscall, args),
        Personality::Nt => dispatch_nt(syscall, args),
        Personality::Darwin => dispatch_darwin(syscall, args),
    }
}

/// Raw ABI wrapper returning `>= 0` on success or negative `AbiError` code on failure.
#[allow(dead_code)]
pub fn dispatch_raw(personality: Personality, raw_num: u64, args: [u64; 6]) -> i64 {
    match dispatch(personality, raw_num, args) {
        Ok(val) => val as i64,
        Err(err) => err as i64,
    }
}

fn dispatch_native(syscall: SyscallNumber, args: [u64; 6]) -> Result<u64, AbiError> {
    match syscall {
        SyscallNumber::DebugLog => {
            let ptr = args[0] as *const u8;
            let len = args[1] as usize;
            if !ptr.is_null() && len > 0 && len <= 4096 {
                let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
                if let Ok(msg) = core::str::from_utf8(bytes) {
                    serial_println!("[syscall::DebugLog] {}", msg);
                    return Ok(len as u64);
                }
            }
            serial_println!(
                "[syscall::DebugLog] arg0={:#x} arg1={:#x}",
                args[0],
                args[1]
            );
            Ok(0)
        }
        SyscallNumber::VmoCreate => {
            let size = args[0] as usize;
            if size == 0 || size > mm::PAGE_SIZE {
                return Err(AbiError::InvalidArg);
            }
            let frame = mm::alloc_frame().ok_or(AbiError::NoMemory)?;
            Ok(frame)
        }
        SyscallNumber::ThreadYield => {
            core::hint::spin_loop();
            Ok(0)
        }
        SyscallNumber::CapClose
        | SyscallNumber::CapDuplicate
        | SyscallNumber::CapTransfer
        | SyscallNumber::ChannelCreate
        | SyscallNumber::ChannelSend
        | SyscallNumber::ChannelRecv
        | SyscallNumber::VmoMap
        | SyscallNumber::VmoUnmap
        | SyscallNumber::ThreadSpawn
        | SyscallNumber::ThreadExit
        | SyscallNumber::ProcessSpawn
        | SyscallNumber::ProcessExit
        | SyscallNumber::WaitOne
        | SyscallNumber::Signal => Err(AbiError::NotSupported),
        _ => Err(AbiError::InvalidArg),
    }
}

fn dispatch_posix(syscall: SyscallNumber, args: [u64; 6]) -> Result<u64, AbiError> {
    match syscall {
        // args: [dirfd, path_ptr, path_len, flags, ...]
        SyscallNumber::PosixOpenAt => {
            vfs_open_fd(args[1], args[2], OpenFlags::from_bits(args[3] as u32))
        }
        // args: [fd, buf_ptr, buf_len, ...]
        SyscallNumber::PosixRead => vfs_read_fd(args[0], args[1], args[2]),
        // args: [fd, buf_ptr, buf_len, ...]
        SyscallNumber::PosixWrite => vfs_write_fd(args[0], args[1], args[2]),
        // args: [fd, ...]
        SyscallNumber::PosixClose => vfs_close_fd(args[0]),
        SyscallNumber::PosixMmap | SyscallNumber::PosixFork | SyscallNumber::PosixExecve => {
            Err(AbiError::NotSupported)
        }
        _ => Err(AbiError::InvalidArg),
    }
}

fn dispatch_nt(syscall: SyscallNumber, args: [u64; 6]) -> Result<u64, AbiError> {
    match syscall {
        // args: [path_ptr, path_len, desired_access, ...]
        SyscallNumber::NtCreateFile => {
            let flags = if args[2] != 0 {
                OpenFlags::from_bits(args[2] as u32)
            } else {
                OpenFlags::READ
            };
            vfs_open_fd(args[0], args[1], flags)
        }
        // args: [handle, buf_ptr, buf_len, ...]
        SyscallNumber::NtReadFile => vfs_read_fd(args[0], args[1], args[2]),
        // args: [handle, buf_ptr, buf_len, ...]
        SyscallNumber::NtWriteFile => vfs_write_fd(args[0], args[1], args[2]),
        // args: [handle, ...]
        SyscallNumber::NtClose => vfs_close_fd(args[0]),
        SyscallNumber::NtAllocateVirtualMemory | SyscallNumber::NtCreateProcessEx => {
            Err(AbiError::NotSupported)
        }
        _ => Err(AbiError::InvalidArg),
    }
}

fn dispatch_darwin(syscall: SyscallNumber, args: [u64; 6]) -> Result<u64, AbiError> {
    match syscall {
        // args: [path_ptr, path_len, flags, ...]
        SyscallNumber::DarwinBsdOpen => {
            vfs_open_fd(args[0], args[1], OpenFlags::from_bits(args[2] as u32))
        }
        // args: [fd, buf_ptr, buf_len, ...]
        SyscallNumber::DarwinBsdRead => vfs_read_fd(args[0], args[1], args[2]),
        // args: [fd, buf_ptr, buf_len, ...]
        SyscallNumber::DarwinBsdWrite => vfs_write_fd(args[0], args[1], args[2]),
        // args: [fd, ...]
        SyscallNumber::DarwinBsdClose => vfs_close_fd(args[0]),
        SyscallNumber::DarwinTaskSelf => Ok(1),
        SyscallNumber::DarwinMachMsg => Err(AbiError::NotSupported),
        _ => Err(AbiError::InvalidArg),
    }
}

fn read_path_arg(ptr: u64, len: u64) -> Result<&'static str, AbiError> {
    let len = len as usize;
    if ptr == 0 || len == 0 || len > 4096 {
        return Err(AbiError::InvalidArg);
    }
    let bytes = unsafe { core::slice::from_raw_parts(ptr as *const u8, len) };
    core::str::from_utf8(bytes).map_err(|_| AbiError::InvalidArg)
}

fn vfs_open_fd(path_ptr: u64, path_len: u64, flags: OpenFlags) -> Result<u64, AbiError> {
    let path = read_path_arg(path_ptr, path_len)?;
    let node = vfs::open(path, flags)?;
    let fd = NEXT_FD.fetch_add(1, Ordering::Relaxed);
    FD_TABLE.lock().insert(fd, OpenFile { node, offset: 0 });
    Ok(fd)
}

fn vfs_read_fd(fd: u64, buf_ptr: u64, buf_len: u64) -> Result<u64, AbiError> {
    let len = buf_len as usize;
    if buf_ptr == 0 || len > 65536 {
        return Err(AbiError::InvalidArg);
    }
    let buf = unsafe { core::slice::from_raw_parts_mut(buf_ptr as *mut u8, len) };
    let mut table = FD_TABLE.lock();
    let file = table.get_mut(&fd).ok_or(AbiError::InvalidHandle)?;
    let n = file.node.read(file.offset, buf)?;
    file.offset += n;
    Ok(n as u64)
}

fn vfs_write_fd(fd: u64, buf_ptr: u64, buf_len: u64) -> Result<u64, AbiError> {
    let len = buf_len as usize;
    if buf_ptr == 0 || len > 65536 {
        return Err(AbiError::InvalidArg);
    }
    let buf = unsafe { core::slice::from_raw_parts(buf_ptr as *const u8, len) };
    let mut table = FD_TABLE.lock();
    let file = table.get_mut(&fd).ok_or(AbiError::InvalidHandle)?;
    let n = file.node.write(file.offset, buf)?;
    file.offset += n;
    Ok(n as u64)
}

fn vfs_close_fd(fd: u64) -> Result<u64, AbiError> {
    FD_TABLE
        .lock()
        .remove(&fd)
        .map(|_| 0)
        .ok_or(AbiError::InvalidHandle)
}
