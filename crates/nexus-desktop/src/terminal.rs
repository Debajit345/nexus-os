use alloc::string::String;
use alloc::vec::Vec;
use nexus_shell::{MemShellFs, Shell, ShellOutput, SystemInfo};

/// Integrated desktop terminal application backed by `nexus_shell::Shell`.
#[derive(Clone, Debug)]
pub struct TerminalApp {
    pub shell: Shell,
    pub fs: MemShellFs,
    pub sys: SystemInfo,
    history: Vec<String>,
    transcript: Vec<String>,
}

impl TerminalApp {
    /// Create a new terminal session with a standard in-memory filesystem and default `SystemInfo`.
    pub fn new() -> Self {
        Self::with_system_info(SystemInfo {
            os_name: String::from("NexusOS"),
            version: String::from("0.1.0"),
            arch: String::from("x86_64"),
            cpu_vendor: String::from("NexusVirtualCPU"),
            total_ram_mib: 2048,
            free_ram_mib: 1536,
            online_cpus: 4,
            pci_count: 4,
        })
    }

    /// Create a new terminal session with custom `SystemInfo`.
    pub fn with_system_info(sys: SystemInfo) -> Self {
        Self {
            shell: Shell::new(),
            fs: MemShellFs::with_standard_layout(),
            sys,
            history: Vec::new(),
            transcript: Vec::new(),
        }
    }

    /// Execute a shell command line, recording it in history and transcript.
    pub fn exec(&mut self, command: &str) -> ShellOutput {
        self.history.push(String::from(command));
        let out = self.shell.eval(command, &mut self.fs, &self.sys);
        if !out.stdout.is_empty() {
            for line in out.stdout.lines() {
                self.transcript.push(String::from(line));
            }
        }
        if !out.stderr.is_empty() {
            for line in out.stderr.lines() {
                self.transcript.push(String::from(line));
            }
        }
        out
    }

    /// Current working directory of the shell.
    pub fn cwd(&self) -> &str {
        &self.shell.cwd
    }

    /// Executed command history.
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Captured output lines from executed commands.
    pub fn transcript(&self) -> &[String] {
        &self.transcript
    }

    /// Clear the captured output transcript.
    pub fn clear_transcript(&mut self) {
        self.transcript.clear();
    }
}

impl Default for TerminalApp {
    fn default() -> Self {
        Self::new()
    }
}
