#![allow(dead_code)]

//! Built-in kernel recovery and diagnostic shell (`ksh`).
//!
//! Bridges [`nexus_shell`] with the kernel's [`crate::vfs`] so early-boot
//! diagnostics and recovery commands can run directly inside the kernel.

use alloc::string::String;
use alloc::vec::Vec;

use nexus_abi::AbiError;
use nexus_shell::{FsError, Shell, ShellFs, ShellOutput, SystemInfo};

use crate::vfs::{self, FileSystem, NodeType, OpenFlags};

/// Adapter exposing [`crate::vfs`] as a [`nexus_shell::ShellFs`].
#[derive(Copy, Clone, Debug, Default)]
pub struct KernelVfsAdapter;

impl KernelVfsAdapter {
    pub const fn new() -> Self {
        Self
    }
}

fn map_abi_err(err: AbiError) -> FsError {
    match err {
        AbiError::NotFound => FsError::NotFound,
        AbiError::AlreadyExists => FsError::AlreadyExists,
        AbiError::AccessDenied => FsError::PermissionDenied,
        AbiError::InvalidArg | AbiError::InvalidHandle => FsError::InvalidPath,
        _ => FsError::InvalidPath,
    }
}

impl ShellFs for KernelVfsAdapter {
    fn list_dir(&self, path: &str) -> Result<Vec<String>, FsError> {
        let st = vfs::stat(path).map_err(map_abi_err)?;
        if st.node_type != NodeType::Directory {
            return Err(FsError::NotADirectory);
        }
        vfs::list_dir(path).map_err(map_abi_err)
    }

    fn read_file(&self, path: &str) -> Result<Vec<u8>, FsError> {
        let st = vfs::stat(path).map_err(map_abi_err)?;
        if st.node_type == NodeType::Directory {
            return Err(FsError::IsADirectory);
        }
        vfs::read_to_vec(path).map_err(map_abi_err)
    }

    fn write_file(&mut self, path: &str, data: &[u8]) -> Result<(), FsError> {
        if let Ok(st) = vfs::stat(path) {
            if st.node_type == NodeType::Directory {
                return Err(FsError::IsADirectory);
            }
        }
        let flags = OpenFlags::WRITE
            .union(OpenFlags::CREATE)
            .union(OpenFlags::TRUNCATE);
        let node = vfs::open(path, flags).map_err(map_abi_err)?;
        node.write(0, data).map_err(map_abi_err)?;
        Ok(())
    }

    fn mkdir(&mut self, path: &str) -> Result<(), FsError> {
        let fs = vfs::root_fs().map_err(map_abi_err)?;
        fs.mkdir(path).map_err(map_abi_err)
    }

    fn remove(&mut self, path: &str) -> Result<(), FsError> {
        let fs = vfs::root_fs().map_err(map_abi_err)?;
        fs.unlink(path).map_err(map_abi_err)
    }

    fn exists(&self, path: &str) -> bool {
        vfs::stat(path).is_ok()
    }

    fn is_dir(&self, path: &str) -> bool {
        matches!(vfs::stat(path), Ok(st) if st.node_type == NodeType::Directory)
    }
}

/// Execute a single shell command line against the kernel VFS.
pub fn run_command(line: &str, sys: &SystemInfo) -> ShellOutput {
    let mut shell = Shell::new();
    let mut fs = KernelVfsAdapter::new();
    shell.eval(line, &mut fs, sys)
}

/// Run standard early-boot diagnostic commands and log their output to serial and framebuffer.
pub fn run_boot_diagnostics(sys: &SystemInfo) {
    let mut shell = Shell::new();
    let mut fs = KernelVfsAdapter::new();

    for cmd in ["uname -a", "lscpu", "free", "ls /"] {
        let out = shell.eval(cmd, &mut fs, sys);
        crate::serial_println!("ksh$ {}", cmd);
        crate::fb_println!("ksh$ {}", cmd);

        for line in out.stdout.lines() {
            crate::serial_println!("  {}", line);
            crate::fb_println!("  {}", line);
        }
        for line in out.stderr.lines() {
            crate::serial_println!("  [err] {}", line);
            crate::fb_println!("  [err] {}", line);
        }
    }
}
