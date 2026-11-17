use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::fs::{resolve_path, ShellFs};
use crate::parser::{parse_line, Connector, RedirectOut, SimpleCommand};
use crate::sys::SystemInfo;

/// Result of evaluating a shell command line.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShellOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub should_exit: bool,
    pub clear_screen: bool,
}

/// Clean-room POSIX-compatible shell interpreter.
#[derive(Clone, Debug)]
pub struct Shell {
    pub cwd: String,
    pub env: BTreeMap<String, String>,
    pub history: Vec<String>,
    pub last_exit_code: i32,
}

impl Default for Shell {
    fn default() -> Self {
        Self::new()
    }
}

impl Shell {
    /// Create a new shell initialized with default POSIX environment variables.
    pub fn new() -> Self {
        let mut env = BTreeMap::new();
        env.insert(String::from("HOME"), String::from("/root"));
        env.insert(String::from("USER"), String::from("root"));
        env.insert(String::from("HOSTNAME"), String::from("nexus"));
        env.insert(String::from("PATH"), String::from("/bin:/usr/bin"));
        env.insert(String::from("PWD"), String::from("/"));
        env.insert(String::from("SHELL"), String::from("/bin/sh"));
        env.insert(String::from("TERM"), String::from("xterm-256color"));

        Self {
            cwd: String::from("/"),
            env,
            history: Vec::new(),
            last_exit_code: 0,
        }
    }

    fn home(&self) -> &str {
        self.env.get("HOME").map(|s| s.as_str()).unwrap_or("/")
    }

    fn resolve(&self, path: &str) -> String {
        resolve_path(&self.cwd, self.home(), path)
    }

    /// Evaluate a command line (supporting quoting, variables, pipelines, `;`, `&&`, `||`, and redirections).
    pub fn eval(&mut self, line: &str, fs: &mut dyn ShellFs, sys: &SystemInfo) -> ShellOutput {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            self.history.push(String::from(trimmed));
        }

        let items = match parse_line(line, &self.env, self.last_exit_code) {
            Ok(items) => items,
            Err(err) => {
                self.last_exit_code = 2;
                return ShellOutput {
                    stdout: String::new(),
                    stderr: format!("sh: {}\n", err),
                    exit_code: 2,
                    should_exit: false,
                    clear_screen: false,
                };
            }
        };

        if items.is_empty() {
            return ShellOutput::default();
        }

        let mut total_out = ShellOutput::default();
        let mut skip_next = false;

        for item in items {
            if skip_next {
                skip_next = match item.connector {
                    Connector::And => total_out.exit_code != 0,
                    Connector::Or => total_out.exit_code == 0,
                    Connector::Seq | Connector::End => false,
                };
                continue;
            }

            let pipe_out = self.eval_pipeline(&item.commands, fs, sys);
            total_out.stdout.push_str(&pipe_out.stdout);
            total_out.stderr.push_str(&pipe_out.stderr);
            total_out.exit_code = pipe_out.exit_code;
            total_out.should_exit |= pipe_out.should_exit;
            total_out.clear_screen |= pipe_out.clear_screen;
            self.last_exit_code = pipe_out.exit_code;

            if total_out.should_exit {
                break;
            }

            skip_next = match item.connector {
                Connector::And => total_out.exit_code != 0,
                Connector::Or => total_out.exit_code == 0,
                Connector::Seq | Connector::End => false,
            };
        }

        total_out
    }

    fn eval_pipeline(
        &mut self,
        commands: &[SimpleCommand],
        fs: &mut dyn ShellFs,
        sys: &SystemInfo,
    ) -> ShellOutput {
        let mut stdin_buf = String::new();
        let mut combined_err = String::new();
        let mut last_out = ShellOutput::default();

        for (idx, cmd) in commands.iter().enumerate() {
            let mut cmd_stdin = if idx == 0 {
                String::new()
            } else {
                core::mem::take(&mut stdin_buf)
            };

            if let Some(ref in_path) = cmd.stdin_redirect {
                let resolved = self.resolve(in_path);
                match fs.read_file(&resolved) {
                    Ok(bytes) => {
                        cmd_stdin = String::from_utf8_lossy(&bytes).into_owned();
                    }
                    Err(e) => {
                        combined_err.push_str(&format!("sh: {}: {}\n", in_path, e));
                        last_out = ShellOutput {
                            stdout: String::new(),
                            stderr: combined_err,
                            exit_code: 1,
                            should_exit: false,
                            clear_screen: false,
                        };
                        return last_out;
                    }
                }
            }

            let mut out = self.eval_simple(&cmd.args, &cmd_stdin, fs, sys);
            combined_err.push_str(&out.stderr);

            if let Some(ref redir) = cmd.stdout_redirect {
                let (target, append) = match redir {
                    RedirectOut::Overwrite(p) => (p, false),
                    RedirectOut::Append(p) => (p, true),
                };
                let resolved = self.resolve(target);
                let mut data = if append && fs.exists(&resolved) {
                    fs.read_file(&resolved).unwrap_or_default()
                } else {
                    Vec::new()
                };
                data.extend_from_slice(out.stdout.as_bytes());
                if let Err(e) = fs.write_file(&resolved, &data) {
                    combined_err.push_str(&format!("sh: {}: {}\n", target, e));
                    out.exit_code = 1;
                }
                out.stdout.clear();
            }

            stdin_buf = core::mem::take(&mut out.stdout);
            last_out = out;
            if last_out.should_exit {
                break;
            }
        }

        last_out.stdout = stdin_buf;
        last_out.stderr = combined_err;
        last_out
    }

    fn eval_simple(
        &mut self,
        args: &[String],
        stdin: &str,
        fs: &mut dyn ShellFs,
        sys: &SystemInfo,
    ) -> ShellOutput {
        if args.is_empty() {
            return ShellOutput::default();
        }

        let cmd = args[0].as_str();
        let rest = &args[1..];

        match cmd {
            "echo" => cmd_echo(rest),
            "pwd" => ShellOutput {
                stdout: format!("{}\n", self.cwd),
                ..Default::default()
            },
            "cd" => self.cmd_cd(rest, fs),
            "ls" => self.cmd_ls(rest, fs),
            "cat" => self.cmd_cat(rest, stdin, fs),
            "mkdir" => self.cmd_mkdir(rest, fs),
            "touch" => self.cmd_touch(rest, fs),
            "rm" => self.cmd_rm(rest, fs),
            "uname" => cmd_uname(
                rest,
                sys,
                self.env
                    .get("HOSTNAME")
                    .map(|s| s.as_str())
                    .unwrap_or("nexus"),
            ),
            "env" => self.cmd_env(),
            "export" => self.cmd_export(rest),
            "whoami" => {
                let user = self.env.get("USER").map(|s| s.as_str()).unwrap_or("root");
                ShellOutput {
                    stdout: format!("{}\n", user),
                    ..Default::default()
                }
            }
            "hostname" => {
                let host = self
                    .env
                    .get("HOSTNAME")
                    .map(|s| s.as_str())
                    .unwrap_or("nexus");
                ShellOutput {
                    stdout: format!("{}\n", host),
                    ..Default::default()
                }
            }
            "free" => cmd_free(rest, sys),
            "lscpu" => cmd_lscpu(sys),
            "lspci" => cmd_lspci(sys),
            "ps" => cmd_ps(),
            "clear" => ShellOutput {
                stdout: String::from("\x1b[2J\x1b[H"),
                clear_screen: true,
                ..Default::default()
            },
            "help" => cmd_help(),
            "exit" => cmd_exit(rest),
            "true" => ShellOutput::default(),
            "false" => ShellOutput {
                exit_code: 1,
                ..Default::default()
            },
            "grep" => cmd_grep(rest, stdin, |p| self.resolve(p), fs),
            "head" => cmd_head(rest, stdin, |p| self.resolve(p), fs),
            "tail" => cmd_tail(rest, stdin, |p| self.resolve(p), fs),
            "wc" => cmd_wc(rest, stdin, |p| self.resolve(p), fs),
            other => ShellOutput {
                stderr: format!("sh: {}: command not found\n", other),
                exit_code: 127,
                ..Default::default()
            },
        }
    }

    fn cmd_cd(&mut self, args: &[String], fs: &dyn ShellFs) -> ShellOutput {
        let target = if args.is_empty() {
            String::from(self.home())
        } else if args[0] == "-" {
            match self.env.get("OLDPWD") {
                Some(old) => old.clone(),
                None => {
                    return ShellOutput {
                        stderr: String::from("cd: OLDPWD not set\n"),
                        exit_code: 1,
                        ..Default::default()
                    };
                }
            }
        } else {
            self.resolve(&args[0])
        };

        if !fs.exists(&target) {
            return ShellOutput {
                stderr: format!("cd: {}: No such file or directory\n", target),
                exit_code: 1,
                ..Default::default()
            };
        }
        if !fs.is_dir(&target) {
            return ShellOutput {
                stderr: format!("cd: {}: Not a directory\n", target),
                exit_code: 1,
                ..Default::default()
            };
        }

        let old = core::mem::replace(&mut self.cwd, target.clone());
        self.env.insert(String::from("OLDPWD"), old);
        self.env.insert(String::from("PWD"), target);
        ShellOutput::default()
    }

    fn cmd_ls(&self, args: &[String], fs: &dyn ShellFs) -> ShellOutput {
        let mut show_all = false;
        let mut long_fmt = false;
        let mut one_per_line = false;
        let mut paths: Vec<&str> = Vec::new();

        for arg in args {
            if arg.starts_with('-') && arg.len() > 1 {
                for ch in arg[1..].chars() {
                    match ch {
                        'a' => show_all = true,
                        'l' => long_fmt = true,
                        '1' => one_per_line = true,
                        _ => {
                            return ShellOutput {
                                stderr: format!("ls: invalid option -- '{}'\n", ch),
                                exit_code: 1,
                                ..Default::default()
                            };
                        }
                    }
                }
            } else {
                paths.push(arg.as_str());
            }
        }

        if paths.is_empty() {
            paths.push(".");
        }

        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut exit_code = 0;
        let multi = paths.len() > 1;

        for (idx, raw_path) in paths.iter().enumerate() {
            let resolved = self.resolve(raw_path);
            if !fs.exists(&resolved) {
                stderr.push_str(&format!(
                    "ls: cannot access '{}': No such file or directory\n",
                    raw_path
                ));
                exit_code = 1;
                continue;
            }

            if !fs.is_dir(&resolved) {
                if long_fmt {
                    let size = fs.read_file(&resolved).map(|b| b.len()).unwrap_or(0);
                    stdout.push_str(&format!(
                        "-rw-r--r-- 1 root root {:>6} {}\n",
                        size, raw_path
                    ));
                } else {
                    stdout.push_str(&format!("{}\n", raw_path));
                }
                continue;
            }

            if multi {
                if idx > 0 {
                    stdout.push('\n');
                }
                stdout.push_str(&format!("{}:\n", raw_path));
            }

            match fs.list_dir(&resolved) {
                Ok(entries) => {
                    let mut items: Vec<String> = Vec::new();
                    if show_all {
                        items.push(String::from("."));
                        items.push(String::from(".."));
                    }
                    for e in entries {
                        if !show_all && e.starts_with('.') {
                            continue;
                        }
                        items.push(e);
                    }

                    if long_fmt {
                        for name in &items {
                            let child = if name == "." {
                                resolved.clone()
                            } else if name == ".." {
                                resolve_path(&resolved, self.home(), "..")
                            } else if resolved == "/" {
                                format!("/{}", name)
                            } else {
                                format!("{}/{}", resolved, name)
                            };
                            if fs.is_dir(&child) {
                                stdout
                                    .push_str(&format!("drwxr-xr-x 1 root root      0 {}\n", name));
                            } else {
                                let size = fs.read_file(&child).map(|b| b.len()).unwrap_or(0);
                                stdout.push_str(&format!(
                                    "-rw-r--r-- 1 root root {:>6} {}\n",
                                    size, name
                                ));
                            }
                        }
                    } else if one_per_line {
                        for name in &items {
                            stdout.push_str(name);
                            stdout.push('\n');
                        }
                    } else if !items.is_empty() {
                        stdout.push_str(&items.join("  "));
                        stdout.push('\n');
                    }
                }
                Err(e) => {
                    stderr.push_str(&format!("ls: {}: {}\n", raw_path, e));
                    exit_code = 1;
                }
            }
        }

        ShellOutput {
            stdout,
            stderr,
            exit_code,
            ..Default::default()
        }
    }

    fn cmd_cat(&self, args: &[String], stdin: &str, fs: &dyn ShellFs) -> ShellOutput {
        if args.is_empty() {
            return ShellOutput {
                stdout: String::from(stdin),
                ..Default::default()
            };
        }

        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut exit_code = 0;

        for arg in args {
            if arg == "-" {
                stdout.push_str(stdin);
                continue;
            }
            let resolved = self.resolve(arg);
            match fs.read_file(&resolved) {
                Ok(bytes) => {
                    stdout.push_str(&String::from_utf8_lossy(&bytes));
                }
                Err(e) => {
                    stderr.push_str(&format!("cat: {}: {}\n", arg, e));
                    exit_code = 1;
                }
            }
        }

        ShellOutput {
            stdout,
            stderr,
            exit_code,
            ..Default::default()
        }
    }

    fn cmd_mkdir(&self, args: &[String], fs: &mut dyn ShellFs) -> ShellOutput {
        let mut parents = false;
        let mut paths: Vec<&str> = Vec::new();

        for arg in args {
            if arg == "-p" {
                parents = true;
            } else if arg.starts_with('-') {
                return ShellOutput {
                    stderr: format!("mkdir: invalid option '{}'\n", arg),
                    exit_code: 1,
                    ..Default::default()
                };
            } else {
                paths.push(arg.as_str());
            }
        }

        if paths.is_empty() {
            return ShellOutput {
                stderr: String::from("mkdir: missing operand\n"),
                exit_code: 1,
                ..Default::default()
            };
        }

        let mut stderr = String::new();
        let mut exit_code = 0;

        for raw in paths {
            let resolved = self.resolve(raw);
            if parents {
                let mut cur = String::new();
                for comp in resolved.trim_start_matches('/').split('/') {
                    if comp.is_empty() {
                        continue;
                    }
                    cur.push('/');
                    cur.push_str(comp);
                    if !fs.exists(&cur) {
                        if let Err(e) = fs.mkdir(&cur) {
                            stderr.push_str(&format!(
                                "mkdir: cannot create directory '{}': {}\n",
                                raw, e
                            ));
                            exit_code = 1;
                            break;
                        }
                    } else if !fs.is_dir(&cur) {
                        stderr.push_str(&format!(
                            "mkdir: cannot create directory '{}': File exists\n",
                            raw
                        ));
                        exit_code = 1;
                        break;
                    }
                }
            } else if let Err(e) = fs.mkdir(&resolved) {
                stderr.push_str(&format!(
                    "mkdir: cannot create directory '{}': {}\n",
                    raw, e
                ));
                exit_code = 1;
            }
        }

        ShellOutput {
            stderr,
            exit_code,
            ..Default::default()
        }
    }

    fn cmd_touch(&self, args: &[String], fs: &mut dyn ShellFs) -> ShellOutput {
        if args.is_empty() {
            return ShellOutput {
                stderr: String::from("touch: missing file operand\n"),
                exit_code: 1,
                ..Default::default()
            };
        }

        let mut stderr = String::new();
        let mut exit_code = 0;

        for arg in args {
            let resolved = self.resolve(arg);
            if !fs.exists(&resolved) {
                if let Err(e) = fs.write_file(&resolved, &[]) {
                    stderr.push_str(&format!("touch: cannot touch '{}': {}\n", arg, e));
                    exit_code = 1;
                }
            }
        }

        ShellOutput {
            stderr,
            exit_code,
            ..Default::default()
        }
    }

    fn cmd_rm(&self, args: &[String], fs: &mut dyn ShellFs) -> ShellOutput {
        let mut recursive = false;
        let mut force = false;
        let mut paths: Vec<&str> = Vec::new();

        for arg in args {
            if arg.starts_with('-') && arg.len() > 1 {
                for ch in arg[1..].chars() {
                    match ch {
                        'r' | 'R' => recursive = true,
                        'f' => force = true,
                        _ => {
                            return ShellOutput {
                                stderr: format!("rm: invalid option -- '{}'\n", ch),
                                exit_code: 1,
                                ..Default::default()
                            };
                        }
                    }
                }
            } else {
                paths.push(arg.as_str());
            }
        }

        if paths.is_empty() {
            if force {
                return ShellOutput::default();
            }
            return ShellOutput {
                stderr: String::from("rm: missing operand\n"),
                exit_code: 1,
                ..Default::default()
            };
        }

        let mut stderr = String::new();
        let mut exit_code = 0;

        for raw in paths {
            let resolved = self.resolve(raw);
            if !fs.exists(&resolved) {
                if !force {
                    stderr.push_str(&format!(
                        "rm: cannot remove '{}': No such file or directory\n",
                        raw
                    ));
                    exit_code = 1;
                }
                continue;
            }
            if let Err(e) = remove_path(fs, &resolved, recursive) {
                stderr.push_str(&format!("rm: cannot remove '{}': {}\n", raw, e));
                exit_code = 1;
            }
        }

        ShellOutput {
            stderr,
            exit_code,
            ..Default::default()
        }
    }

    fn cmd_env(&self) -> ShellOutput {
        let mut stdout = String::new();
        for (k, v) in &self.env {
            stdout.push_str(&format!("{}={}\n", k, v));
        }
        ShellOutput {
            stdout,
            ..Default::default()
        }
    }

    fn cmd_export(&mut self, args: &[String]) -> ShellOutput {
        if args.is_empty() {
            let mut stdout = String::new();
            for (k, v) in &self.env {
                stdout.push_str(&format!("export {}=\"{}\"\n", k, v));
            }
            return ShellOutput {
                stdout,
                ..Default::default()
            };
        }

        for arg in args {
            if let Some((k, v)) = arg.split_once('=') {
                if k.is_empty() {
                    return ShellOutput {
                        stderr: format!("export: `{}': not a valid identifier\n", arg),
                        exit_code: 1,
                        ..Default::default()
                    };
                }
                self.env.insert(String::from(k), String::from(v));
            } else {
                self.env.entry(arg.clone()).or_default();
            }
        }

        ShellOutput::default()
    }
}

fn remove_path(
    fs: &mut dyn ShellFs,
    path: &str,
    recursive: bool,
) -> Result<(), crate::fs::FsError> {
    if fs.is_dir(path) {
        if !recursive {
            return fs.remove(path);
        }
        let children = fs.list_dir(path)?;
        for child in children {
            let child_path = if path == "/" {
                format!("/{}", child)
            } else {
                format!("{}/{}", path, child)
            };
            remove_path(fs, &child_path, true)?;
        }
    }
    fs.remove(path)
}

fn cmd_echo(args: &[String]) -> ShellOutput {
    let mut no_newline = false;
    let mut start = 0;
    if args.first().map(|s| s.as_str()) == Some("-n") {
        no_newline = true;
        start = 1;
    }
    let mut stdout = args[start..].join(" ");
    if !no_newline {
        stdout.push('\n');
    }
    ShellOutput {
        stdout,
        ..Default::default()
    }
}

fn cmd_uname(args: &[String], sys: &SystemInfo, hostname: &str) -> ShellOutput {
    let mut all = false;
    let mut sysname = false;
    let mut nodename = false;
    let mut release = false;
    let mut machine = false;

    if args.is_empty() {
        sysname = true;
    } else {
        for arg in args {
            if arg.starts_with('-') && arg.len() > 1 {
                for ch in arg[1..].chars() {
                    match ch {
                        'a' => all = true,
                        's' => sysname = true,
                        'n' => nodename = true,
                        'r' => release = true,
                        'm' => machine = true,
                        'v' => {}
                        _ => {
                            return ShellOutput {
                                stderr: format!("uname: invalid option -- '{}'\n", ch),
                                exit_code: 1,
                                ..Default::default()
                            };
                        }
                    }
                }
            }
        }
    }

    let mut parts: Vec<&str> = Vec::new();
    if all {
        parts.push(&sys.os_name);
        parts.push(hostname);
        parts.push(&sys.version);
        parts.push(&sys.arch);
    } else {
        if sysname {
            parts.push(&sys.os_name);
        }
        if nodename {
            parts.push(hostname);
        }
        if release {
            parts.push(&sys.version);
        }
        if machine {
            parts.push(&sys.arch);
        }
    }

    ShellOutput {
        stdout: format!("{}\n", parts.join(" ")),
        ..Default::default()
    }
}

fn cmd_free(args: &[String], sys: &SystemInfo) -> ShellOutput {
    let human = args.iter().any(|a| a == "-h");
    let used = sys.total_ram_mib.saturating_sub(sys.free_ram_mib);
    let stdout = if human {
        format!(
            "               total        used        free\nMem:         {:>5}Mi     {:>5}Mi     {:>5}Mi\n",
            sys.total_ram_mib, used, sys.free_ram_mib
        )
    } else {
        format!(
            "               total        used        free\nMem:         {:>7}     {:>7}     {:>7}\n",
            sys.total_ram_mib, used, sys.free_ram_mib
        )
    };
    ShellOutput {
        stdout,
        ..Default::default()
    }
}

fn cmd_lscpu(sys: &SystemInfo) -> ShellOutput {
    let stdout = format!(
        "Architecture:        {}\nCPU(s):              {}\nVendor ID:           {}\n",
        sys.arch, sys.online_cpus, sys.cpu_vendor
    );
    ShellOutput {
        stdout,
        ..Default::default()
    }
}

fn cmd_lspci(sys: &SystemInfo) -> ShellOutput {
    let stdout = format!("PCI devices detected: {}\n", sys.pci_count);
    ShellOutput {
        stdout,
        ..Default::default()
    }
}

fn cmd_ps() -> ShellOutput {
    let stdout = String::from(
        "  PID TTY          TIME CMD\n    1 ttyS0    00:00:00 init\n    2 ttyS0    00:00:00 sh\n",
    );
    ShellOutput {
        stdout,
        ..Default::default()
    }
}

fn cmd_help() -> ShellOutput {
    let stdout = String::from(
        "Nexus OS POSIX Shell (nexus-shell)\n\
         Builtins: echo, pwd, cd, ls, cat, mkdir, touch, rm, uname, env, export,\n\
                   whoami, hostname, free, lscpu, lspci, ps, clear, help, exit,\n\
                   grep, head, tail, wc, true, false\n",
    );
    ShellOutput {
        stdout,
        ..Default::default()
    }
}

fn cmd_exit(args: &[String]) -> ShellOutput {
    let code = args
        .first()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);
    ShellOutput {
        exit_code: code,
        should_exit: true,
        ..Default::default()
    }
}

fn read_input_or_files<F: Fn(&str) -> String>(
    files: &[&str],
    stdin: &str,
    resolve: F,
    fs: &dyn ShellFs,
    cmd_name: &str,
) -> Result<String, ShellOutput> {
    if files.is_empty() {
        return Ok(String::from(stdin));
    }
    let mut buf = String::new();
    for f in files {
        let resolved = resolve(f);
        match fs.read_file(&resolved) {
            Ok(bytes) => buf.push_str(&String::from_utf8_lossy(&bytes)),
            Err(e) => {
                return Err(ShellOutput {
                    stderr: format!("{}: {}: {}\n", cmd_name, f, e),
                    exit_code: 1,
                    ..Default::default()
                });
            }
        }
    }
    Ok(buf)
}

fn cmd_grep<F: Fn(&str) -> String>(
    args: &[String],
    stdin: &str,
    resolve: F,
    fs: &dyn ShellFs,
) -> ShellOutput {
    let mut invert = false;
    let mut positional: Vec<&str> = Vec::new();
    for arg in args {
        if arg == "-v" {
            invert = true;
        } else {
            positional.push(arg.as_str());
        }
    }
    if positional.is_empty() {
        return ShellOutput {
            stderr: String::from("grep: missing pattern\n"),
            exit_code: 2,
            ..Default::default()
        };
    }
    let pattern = positional[0];
    let input = match read_input_or_files(&positional[1..], stdin, resolve, fs, "grep") {
        Ok(s) => s,
        Err(out) => return out,
    };

    let mut stdout = String::new();
    let mut matched = false;
    for line in input.lines() {
        let contains = line.contains(pattern);
        if contains ^ invert {
            stdout.push_str(line);
            stdout.push('\n');
            matched = true;
        }
    }
    ShellOutput {
        stdout,
        exit_code: if matched { 0 } else { 1 },
        ..Default::default()
    }
}

fn cmd_head<F: Fn(&str) -> String>(
    args: &[String],
    stdin: &str,
    resolve: F,
    fs: &dyn ShellFs,
) -> ShellOutput {
    let mut count = 10usize;
    let mut files: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-n" && i + 1 < args.len() {
            count = args[i + 1].parse().unwrap_or(10);
            i += 2;
        } else {
            files.push(args[i].as_str());
            i += 1;
        }
    }
    let input = match read_input_or_files(&files, stdin, resolve, fs, "head") {
        Ok(s) => s,
        Err(out) => return out,
    };
    let mut stdout = String::new();
    for line in input.lines().take(count) {
        stdout.push_str(line);
        stdout.push('\n');
    }
    ShellOutput {
        stdout,
        ..Default::default()
    }
}

fn cmd_tail<F: Fn(&str) -> String>(
    args: &[String],
    stdin: &str,
    resolve: F,
    fs: &dyn ShellFs,
) -> ShellOutput {
    let mut count = 10usize;
    let mut files: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-n" && i + 1 < args.len() {
            count = args[i + 1].parse().unwrap_or(10);
            i += 2;
        } else {
            files.push(args[i].as_str());
            i += 1;
        }
    }
    let input = match read_input_or_files(&files, stdin, resolve, fs, "tail") {
        Ok(s) => s,
        Err(out) => return out,
    };
    let lines: Vec<&str> = input.lines().collect();
    let start = lines.len().saturating_sub(count);
    let mut stdout = String::new();
    for line in &lines[start..] {
        stdout.push_str(line);
        stdout.push('\n');
    }
    ShellOutput {
        stdout,
        ..Default::default()
    }
}

fn cmd_wc<F: Fn(&str) -> String>(
    args: &[String],
    stdin: &str,
    resolve: F,
    fs: &dyn ShellFs,
) -> ShellOutput {
    let mut lines_only = false;
    let mut words_only = false;
    let mut bytes_only = false;
    let mut files: Vec<&str> = Vec::new();

    for arg in args {
        match arg.as_str() {
            "-l" => lines_only = true,
            "-w" => words_only = true,
            "-c" => bytes_only = true,
            _ => files.push(arg.as_str()),
        }
    }

    let input = match read_input_or_files(&files, stdin, resolve, fs, "wc") {
        Ok(s) => s,
        Err(out) => return out,
    };

    let line_count = input.bytes().filter(|&b| b == b'\n').count();
    let word_count = input.split_whitespace().count();
    let byte_count = input.len();

    let stdout = if lines_only {
        format!("{}\n", line_count)
    } else if words_only {
        format!("{}\n", word_count)
    } else if bytes_only {
        format!("{}\n", byte_count)
    } else {
        format!("{} {} {}\n", line_count, word_count, byte_count)
    };

    ShellOutput {
        stdout,
        ..Default::default()
    }
}
