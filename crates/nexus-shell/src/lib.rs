#![cfg_attr(not(any(test, feature = "std")), no_std)]

//! Clean-room POSIX shell and coreutils crate for Nexus OS.

extern crate alloc;

pub mod fs;
pub mod parser;
pub mod shell;
pub mod sys;

pub use fs::{normalize_path, resolve_path, FsError, MemShellFs, ShellFs};
pub use parser::ParseError;
pub use shell::{Shell, ShellOutput};
pub use sys::SystemInfo;

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Shell, MemShellFs, SystemInfo) {
        let shell = Shell::new();
        let fs = MemShellFs::with_standard_layout();
        let sys = SystemInfo {
            os_name: String::from("NexusOS"),
            version: String::from("0.1.0"),
            arch: String::from("x86_64"),
            cpu_vendor: String::from("AuthenticAMD"),
            total_ram_mib: 1024,
            free_ram_mib: 768,
            online_cpus: 4,
            pci_count: 6,
        };
        (shell, fs, sys)
    }

    #[test]
    fn echo_and_quoting() {
        let (mut sh, mut fs, sys) = setup();

        let out = sh.eval("echo hello world", &mut fs, &sys);
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, "hello world\n");

        let out = sh.eval("echo -n 'single $USER quote'", &mut fs, &sys);
        assert_eq!(out.stdout, "single $USER quote");

        let out = sh.eval("echo \"double $USER quote\"", &mut fs, &sys);
        assert_eq!(out.stdout, "double root quote\n");

        let out = sh.eval("echo \"unterminated", &mut fs, &sys);
        assert_eq!(out.exit_code, 2);
        assert!(out.stderr.contains("unterminated quote"));
    }

    #[test]
    fn mkdir_cd_pwd_ls_touch_cat_rm() {
        let (mut sh, mut fs, sys) = setup();

        assert_eq!(sh.eval("pwd", &mut fs, &sys).stdout, "/\n");

        let out = sh.eval("mkdir -p /tmp/a/b", &mut fs, &sys);
        assert_eq!(out.exit_code, 0);

        let out = sh.eval("cd /tmp/a/b", &mut fs, &sys);
        assert_eq!(out.exit_code, 0);
        assert_eq!(sh.cwd, "/tmp/a/b");
        assert_eq!(sh.eval("pwd", &mut fs, &sys).stdout, "/tmp/a/b\n");

        assert_eq!(sh.eval("touch note.txt", &mut fs, &sys).exit_code, 0);
        assert_eq!(
            sh.eval("echo 'hello nexus' > note.txt", &mut fs, &sys)
                .exit_code,
            0
        );
        assert_eq!(
            sh.eval("echo 'second line' >> note.txt", &mut fs, &sys)
                .exit_code,
            0
        );

        let cat_out = sh.eval("cat note.txt", &mut fs, &sys);
        assert_eq!(cat_out.exit_code, 0);
        assert_eq!(cat_out.stdout, "hello nexus\nsecond line\n");

        let ls_out = sh.eval("ls /tmp/a/b", &mut fs, &sys);
        assert_eq!(ls_out.exit_code, 0);
        assert_eq!(ls_out.stdout, "note.txt\n");

        assert_eq!(sh.eval("cd ..", &mut fs, &sys).exit_code, 0);
        assert_eq!(sh.cwd, "/tmp/a");

        // Non-recursive rm on non-empty dir fails
        assert_ne!(sh.eval("rm b", &mut fs, &sys).exit_code, 0);
        // Recursive rm succeeds
        assert_eq!(sh.eval("rm -rf b", &mut fs, &sys).exit_code, 0);
        assert!(!fs.exists("/tmp/a/b"));
    }

    #[test]
    fn export_and_env() {
        let (mut sh, mut fs, sys) = setup();

        let out = sh.eval("export FOO=bar BAZ=\"hello world\"", &mut fs, &sys);
        assert_eq!(out.exit_code, 0);

        let out = sh.eval("echo $FOO ${BAZ}", &mut fs, &sys);
        assert_eq!(out.stdout, "bar hello world\n");

        let env_out = sh.eval("env", &mut fs, &sys);
        assert_eq!(env_out.exit_code, 0);
        assert!(env_out.stdout.contains("FOO=bar\n"));
        assert!(env_out.stdout.contains("BAZ=hello world\n"));
    }

    #[test]
    fn pipelines_and_connectors() {
        let (mut sh, mut fs, sys) = setup();

        let out = sh.eval("cat /etc/os-release | grep NAME", &mut fs, &sys);
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, "NAME=\"Nexus OS\"\n");

        let out = sh.eval("cat /etc/os-release | wc -l", &mut fs, &sys);
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, "2\n");

        let out = sh.eval("false && echo nope || echo fallback", &mut fs, &sys);
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, "fallback\n");

        let out = sh.eval("true && echo ok; echo $?", &mut fs, &sys);
        assert_eq!(out.stdout, "ok\n0\n");
    }

    #[test]
    fn system_info_commands() {
        let (mut sh, mut fs, sys) = setup();

        let uname_a = sh.eval("uname -a", &mut fs, &sys);
        assert_eq!(uname_a.exit_code, 0);
        assert_eq!(uname_a.stdout, "NexusOS nexus 0.1.0 x86_64\n");

        assert_eq!(sh.eval("whoami", &mut fs, &sys).stdout, "root\n");
        assert_eq!(sh.eval("hostname", &mut fs, &sys).stdout, "nexus\n");

        let free_out = sh.eval("free -h", &mut fs, &sys);
        assert!(free_out.stdout.contains("1024Mi"));
        assert!(free_out.stdout.contains("768Mi"));

        let lscpu_out = sh.eval("lscpu", &mut fs, &sys);
        assert!(lscpu_out.stdout.contains("AuthenticAMD"));
        assert!(lscpu_out.stdout.contains("4"));

        let lspci_out = sh.eval("lspci", &mut fs, &sys);
        assert!(lspci_out.stdout.contains("6"));

        let ps_out = sh.eval("ps", &mut fs, &sys);
        assert!(ps_out.stdout.contains("init"));
        assert!(ps_out.stdout.contains("sh"));

        let clear_out = sh.eval("clear", &mut fs, &sys);
        assert!(clear_out.clear_screen);

        let help_out = sh.eval("help", &mut fs, &sys);
        assert!(help_out.stdout.contains("nexus-shell"));

        let exit_out = sh.eval("exit 42", &mut fs, &sys);
        assert!(exit_out.should_exit);
        assert_eq!(exit_out.exit_code, 42);
        assert!(!sh.history.is_empty());
    }
}
