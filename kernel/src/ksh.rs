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

/// Decode a PS/2 Set-1 scancode into an ASCII character, respecting `shift`.
///
/// Returns `None` for key-release (break) scancodes and non-printable/unmapped keys.
pub fn scancode_to_ascii(scancode: u8, shift: bool) -> Option<char> {
    if scancode & 0x80 != 0 {
        return None;
    }

    let ch = match scancode {
        0x02 => {
            if shift {
                '!'
            } else {
                '1'
            }
        }
        0x03 => {
            if shift {
                '@'
            } else {
                '2'
            }
        }
        0x04 => {
            if shift {
                '#'
            } else {
                '3'
            }
        }
        0x05 => {
            if shift {
                '$'
            } else {
                '4'
            }
        }
        0x06 => {
            if shift {
                '%'
            } else {
                '5'
            }
        }
        0x07 => {
            if shift {
                '^'
            } else {
                '6'
            }
        }
        0x08 => {
            if shift {
                '&'
            } else {
                '7'
            }
        }
        0x09 => {
            if shift {
                '*'
            } else {
                '8'
            }
        }
        0x0A => {
            if shift {
                '('
            } else {
                '9'
            }
        }
        0x0B => {
            if shift {
                ')'
            } else {
                '0'
            }
        }
        0x0C => {
            if shift {
                '_'
            } else {
                '-'
            }
        }
        0x0D => {
            if shift {
                '+'
            } else {
                '='
            }
        }
        0x0E => '\x08',
        0x0F => '\t',
        0x10 => {
            if shift {
                'Q'
            } else {
                'q'
            }
        }
        0x11 => {
            if shift {
                'W'
            } else {
                'w'
            }
        }
        0x12 => {
            if shift {
                'E'
            } else {
                'e'
            }
        }
        0x13 => {
            if shift {
                'R'
            } else {
                'r'
            }
        }
        0x14 => {
            if shift {
                'T'
            } else {
                't'
            }
        }
        0x15 => {
            if shift {
                'Y'
            } else {
                'y'
            }
        }
        0x16 => {
            if shift {
                'U'
            } else {
                'u'
            }
        }
        0x17 => {
            if shift {
                'I'
            } else {
                'i'
            }
        }
        0x18 => {
            if shift {
                'O'
            } else {
                'o'
            }
        }
        0x19 => {
            if shift {
                'P'
            } else {
                'p'
            }
        }
        0x1A => {
            if shift {
                '{'
            } else {
                '['
            }
        }
        0x1B => {
            if shift {
                '}'
            } else {
                ']'
            }
        }
        0x1C => '\n',
        0x1E => {
            if shift {
                'A'
            } else {
                'a'
            }
        }
        0x1F => {
            if shift {
                'S'
            } else {
                's'
            }
        }
        0x20 => {
            if shift {
                'D'
            } else {
                'd'
            }
        }
        0x21 => {
            if shift {
                'F'
            } else {
                'f'
            }
        }
        0x22 => {
            if shift {
                'G'
            } else {
                'g'
            }
        }
        0x23 => {
            if shift {
                'H'
            } else {
                'h'
            }
        }
        0x24 => {
            if shift {
                'J'
            } else {
                'j'
            }
        }
        0x25 => {
            if shift {
                'K'
            } else {
                'k'
            }
        }
        0x26 => {
            if shift {
                'L'
            } else {
                'l'
            }
        }
        0x27 => {
            if shift {
                ':'
            } else {
                ';'
            }
        }
        0x28 => {
            if shift {
                '"'
            } else {
                '\''
            }
        }
        0x29 => {
            if shift {
                '~'
            } else {
                '`'
            }
        }
        0x2B => {
            if shift {
                '|'
            } else {
                '\\'
            }
        }
        0x2C => {
            if shift {
                'Z'
            } else {
                'z'
            }
        }
        0x2D => {
            if shift {
                'X'
            } else {
                'x'
            }
        }
        0x2E => {
            if shift {
                'C'
            } else {
                'c'
            }
        }
        0x2F => {
            if shift {
                'V'
            } else {
                'v'
            }
        }
        0x30 => {
            if shift {
                'B'
            } else {
                'b'
            }
        }
        0x31 => {
            if shift {
                'N'
            } else {
                'n'
            }
        }
        0x32 => {
            if shift {
                'M'
            } else {
                'm'
            }
        }
        0x33 => {
            if shift {
                '<'
            } else {
                ','
            }
        }
        0x34 => {
            if shift {
                '>'
            } else {
                '.'
            }
        }
        0x35 => {
            if shift {
                '?'
            } else {
                '/'
            }
        }
        0x39 => ' ',
        _ => return None,
    };

    Some(ch)
}

/// Interactive line-buffered kernel shell driven by PS/2 scancodes.
#[derive(Clone, Debug, Default)]
pub struct InteractiveShell {
    line: String,
    shift_held: bool,
}

impl InteractiveShell {
    pub const fn new() -> Self {
        Self {
            line: String::new(),
            shift_held: false,
        }
    }

    pub fn line(&self) -> &str {
        &self.line
    }

    pub fn shift_held(&self) -> bool {
        self.shift_held
    }

    /// Feed a single PS/2 Set-1 scancode into the interactive shell.
    ///
    /// Echoes typed characters to serial and framebuffer consoles, and executes
    /// the buffered command via [`run_command`] when Enter is pressed.
    pub fn feed_scancode(&mut self, scancode: u8, sys: &SystemInfo) -> Option<ShellOutput> {
        match scancode {
            0x2A | 0x36 => {
                self.shift_held = true;
                return None;
            }
            0xAA | 0xB6 => {
                self.shift_held = false;
                return None;
            }
            _ => {}
        }

        let ch = scancode_to_ascii(scancode, self.shift_held)?;
        match ch {
            '\n' => {
                crate::serial_println!();
                crate::fb_println!();
                let cmd = core::mem::take(&mut self.line);
                let out = run_command(&cmd, sys);
                for line in out.stdout.lines() {
                    crate::serial_println!("{}", line);
                    crate::fb_println!("{}", line);
                }
                for line in out.stderr.lines() {
                    crate::serial_println!("[err] {}", line);
                    crate::fb_println!("[err] {}", line);
                }
                crate::serial_print!("ksh$ ");
                crate::fb_print!("ksh$ ");
                Some(out)
            }
            '\x08' => {
                if self.line.pop().is_some() {
                    crate::serial_print!("\x08 \x08");
                }
                None
            }
            c => {
                self.line.push(c);
                crate::serial_print!("{}", c);
                crate::fb_print!("{}", c);
                None
            }
        }
    }
}

/// Drain all pending PS/2 scancodes from the kernel ring buffer and feed them
/// into `shell`.
pub fn poll_keyboard_once(shell: &mut InteractiveShell, sys: &SystemInfo) {
    while let Some(scancode) = crate::apic::pop_scancode() {
        let _ = shell.feed_scancode(scancode, sys);
    }
}
