use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Errors returned by [`ShellFs`] operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FsError {
    NotFound,
    AlreadyExists,
    NotADirectory,
    IsADirectory,
    DirectoryNotEmpty,
    InvalidPath,
    PermissionDenied,
}

impl fmt::Display for FsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "No such file or directory"),
            Self::AlreadyExists => write!(f, "File exists"),
            Self::NotADirectory => write!(f, "Not a directory"),
            Self::IsADirectory => write!(f, "Is a directory"),
            Self::DirectoryNotEmpty => write!(f, "Directory not empty"),
            Self::InvalidPath => write!(f, "Invalid path"),
            Self::PermissionDenied => write!(f, "Permission denied"),
        }
    }
}

/// Filesystem abstraction used by [`crate::Shell`].
pub trait ShellFs {
    fn list_dir(&self, path: &str) -> Result<Vec<String>, FsError>;
    fn read_file(&self, path: &str) -> Result<Vec<u8>, FsError>;
    fn write_file(&mut self, path: &str, data: &[u8]) -> Result<(), FsError>;
    fn mkdir(&mut self, path: &str) -> Result<(), FsError>;
    fn remove(&mut self, path: &str) -> Result<(), FsError>;
    fn exists(&self, path: &str) -> bool;

    /// Returns `true` if `path` exists and is a directory.
    fn is_dir(&self, path: &str) -> bool {
        self.list_dir(path).is_ok()
    }
}

/// Normalize a POSIX path, resolving `.` and `..` components.
pub fn normalize_path(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        match comp {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    if parts.is_empty() {
        String::from("/")
    } else {
        let mut out = String::new();
        for p in parts {
            out.push('/');
            out.push_str(p);
        }
        out
    }
}

/// Resolve `input` relative to `cwd` (and `home` for `~`).
pub fn resolve_path(cwd: &str, home: &str, input: &str) -> String {
    if input == "~" {
        return normalize_path(home);
    }
    if let Some(rest) = input.strip_prefix("~/") {
        let mut combined = String::from(home);
        if !combined.ends_with('/') {
            combined.push('/');
        }
        combined.push_str(rest);
        return normalize_path(&combined);
    }
    if input.starts_with('/') {
        normalize_path(input)
    } else {
        let mut combined = String::from(cwd);
        if !combined.ends_with('/') {
            combined.push('/');
        }
        combined.push_str(input);
        normalize_path(&combined)
    }
}

fn split_parent(norm_path: &str) -> Result<(&str, &str), FsError> {
    if norm_path == "/" {
        return Err(FsError::InvalidPath);
    }
    match norm_path.rfind('/') {
        Some(0) => Ok(("/", &norm_path[1..])),
        Some(idx) => Ok((&norm_path[..idx], &norm_path[idx + 1..])),
        None => Err(FsError::InvalidPath),
    }
}

#[derive(Clone, Debug)]
enum MemNode {
    File(Vec<u8>),
    Dir(BTreeMap<String, MemNode>),
}

/// In-memory [`ShellFs`] implementation for standalone use and testing.
#[derive(Clone, Debug)]
pub struct MemShellFs {
    root: MemNode,
}

impl Default for MemShellFs {
    fn default() -> Self {
        Self::new()
    }
}

impl MemShellFs {
    /// Create a new empty in-memory filesystem with `/` root directory.
    pub fn new() -> Self {
        Self {
            root: MemNode::Dir(BTreeMap::new()),
        }
    }

    /// Create a new in-memory filesystem pre-populated with standard POSIX directories.
    pub fn with_standard_layout() -> Self {
        let mut fs = Self::new();
        for dir in [
            "/bin", "/dev", "/etc", "/home", "/root", "/tmp", "/usr", "/var",
        ] {
            let _ = fs.mkdir(dir);
        }
        let _ = fs.write_file("/etc/hostname", b"nexus\n");
        let _ = fs.write_file("/etc/os-release", b"NAME=\"Nexus OS\"\nVERSION=\"0.1.0\"\n");
        fs
    }

    fn get_node(&self, norm_path: &str) -> Result<&MemNode, FsError> {
        if norm_path == "/" {
            return Ok(&self.root);
        }
        let mut cur = &self.root;
        for comp in norm_path.trim_start_matches('/').split('/') {
            if comp.is_empty() {
                continue;
            }
            match cur {
                MemNode::Dir(map) => {
                    cur = map.get(comp).ok_or(FsError::NotFound)?;
                }
                MemNode::File(_) => return Err(FsError::NotADirectory),
            }
        }
        Ok(cur)
    }

    fn get_dir_mut(&mut self, norm_path: &str) -> Result<&mut BTreeMap<String, MemNode>, FsError> {
        if norm_path == "/" {
            return match &mut self.root {
                MemNode::Dir(map) => Ok(map),
                MemNode::File(_) => Err(FsError::NotADirectory),
            };
        }
        let mut cur = &mut self.root;
        for comp in norm_path.trim_start_matches('/').split('/') {
            if comp.is_empty() {
                continue;
            }
            match cur {
                MemNode::Dir(map) => {
                    cur = map.get_mut(comp).ok_or(FsError::NotFound)?;
                }
                MemNode::File(_) => return Err(FsError::NotADirectory),
            }
        }
        match cur {
            MemNode::Dir(map) => Ok(map),
            MemNode::File(_) => Err(FsError::NotADirectory),
        }
    }
}

impl ShellFs for MemShellFs {
    fn list_dir(&self, path: &str) -> Result<Vec<String>, FsError> {
        let norm = normalize_path(path);
        match self.get_node(&norm)? {
            MemNode::Dir(map) => Ok(map.keys().cloned().collect()),
            MemNode::File(_) => Err(FsError::NotADirectory),
        }
    }

    fn read_file(&self, path: &str) -> Result<Vec<u8>, FsError> {
        let norm = normalize_path(path);
        match self.get_node(&norm)? {
            MemNode::File(data) => Ok(data.clone()),
            MemNode::Dir(_) => Err(FsError::IsADirectory),
        }
    }

    fn write_file(&mut self, path: &str, data: &[u8]) -> Result<(), FsError> {
        let norm = normalize_path(path);
        let (parent, name) = split_parent(&norm)?;
        let dir = self.get_dir_mut(parent)?;
        if let Some(MemNode::Dir(_)) = dir.get(name) {
            return Err(FsError::IsADirectory);
        }
        dir.insert(name.to_string(), MemNode::File(data.to_vec()));
        Ok(())
    }

    fn mkdir(&mut self, path: &str) -> Result<(), FsError> {
        let norm = normalize_path(path);
        if norm == "/" {
            return Err(FsError::AlreadyExists);
        }
        let (parent, name) = split_parent(&norm)?;
        let dir = self.get_dir_mut(parent)?;
        if dir.contains_key(name) {
            return Err(FsError::AlreadyExists);
        }
        dir.insert(name.to_string(), MemNode::Dir(BTreeMap::new()));
        Ok(())
    }

    fn remove(&mut self, path: &str) -> Result<(), FsError> {
        let norm = normalize_path(path);
        if norm == "/" {
            return Err(FsError::PermissionDenied);
        }
        let (parent, name) = split_parent(&norm)?;
        let dir = self.get_dir_mut(parent)?;
        match dir.get(name) {
            None => return Err(FsError::NotFound),
            Some(MemNode::Dir(children)) if !children.is_empty() => {
                return Err(FsError::DirectoryNotEmpty);
            }
            _ => {}
        }
        dir.remove(name);
        Ok(())
    }

    fn exists(&self, path: &str) -> bool {
        let norm = normalize_path(path);
        self.get_node(&norm).is_ok()
    }

    fn is_dir(&self, path: &str) -> bool {
        let norm = normalize_path(path);
        matches!(self.get_node(&norm), Ok(MemNode::Dir(_)))
    }
}
