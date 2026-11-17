//! Clean-room Virtual File System (VFS) and in-memory RamFs.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use nexus_abi::AbiError;

use crate::mm::SpinLock;

/// File open flags.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OpenFlags(u32);

#[allow(dead_code)]
impl OpenFlags {
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const CREATE: Self = Self(1 << 2);
    pub const TRUNCATE: Self = Self(1 << 3);
    pub const APPEND: Self = Self(1 << 4);
    pub const DIRECTORY: Self = Self(1 << 5);

    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NodeType {
    File,
    Directory,
    CharDevice,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct VfsStat {
    pub node_type: NodeType,
    pub size: usize,
    pub inode: u64,
}

/// Clean-room VFS node abstraction.
pub trait VfsNode: Send + Sync {
    fn stat(&self) -> Result<VfsStat, AbiError>;
    fn read(&self, offset: usize, buf: &mut [u8]) -> Result<usize, AbiError>;
    fn write(&self, offset: usize, data: &[u8]) -> Result<usize, AbiError>;
    fn readdir(&self) -> Result<Vec<String>, AbiError>;
}

/// Clean-room filesystem abstraction.
#[allow(dead_code)]
pub trait FileSystem: Send + Sync {
    fn name(&self) -> &str;
    fn open(&self, path: &str, flags: OpenFlags) -> Result<Arc<dyn VfsNode>, AbiError>;
    fn mkdir(&self, path: &str) -> Result<(), AbiError>;
    fn unlink(&self, path: &str) -> Result<(), AbiError>;
    fn stat(&self, path: &str) -> Result<VfsStat, AbiError>;
}

enum RamNodeKind {
    File(Vec<u8>),
    Directory(BTreeMap<String, Arc<RamNode>>),
}

pub struct RamNode {
    inode: u64,
    kind: SpinLock<RamNodeKind>,
}

impl RamNode {
    fn new_file(inode: u64, initial: &[u8]) -> Arc<Self> {
        Arc::new(Self {
            inode,
            kind: SpinLock::new(RamNodeKind::File(initial.to_vec())),
        })
    }

    fn new_dir(inode: u64) -> Arc<Self> {
        Arc::new(Self {
            inode,
            kind: SpinLock::new(RamNodeKind::Directory(BTreeMap::new())),
        })
    }

    fn truncate(&self) -> Result<(), AbiError> {
        let mut guard = self.kind.lock();
        match &mut *guard {
            RamNodeKind::File(data) => {
                data.clear();
                Ok(())
            }
            RamNodeKind::Directory(_) => Err(AbiError::InvalidArg),
        }
    }
}

impl VfsNode for RamNode {
    fn stat(&self) -> Result<VfsStat, AbiError> {
        let guard = self.kind.lock();
        match &*guard {
            RamNodeKind::File(data) => Ok(VfsStat {
                node_type: NodeType::File,
                size: data.len(),
                inode: self.inode,
            }),
            RamNodeKind::Directory(entries) => Ok(VfsStat {
                node_type: NodeType::Directory,
                size: entries.len(),
                inode: self.inode,
            }),
        }
    }

    fn read(&self, offset: usize, buf: &mut [u8]) -> Result<usize, AbiError> {
        let guard = self.kind.lock();
        match &*guard {
            RamNodeKind::File(data) => {
                if offset >= data.len() {
                    return Ok(0);
                }
                let avail = data.len() - offset;
                let to_copy = avail.min(buf.len());
                buf[..to_copy].copy_from_slice(&data[offset..offset + to_copy]);
                Ok(to_copy)
            }
            RamNodeKind::Directory(_) => Err(AbiError::InvalidArg),
        }
    }

    fn write(&self, offset: usize, data: &[u8]) -> Result<usize, AbiError> {
        let mut guard = self.kind.lock();
        match &mut *guard {
            RamNodeKind::File(buf) => {
                let end = offset.checked_add(data.len()).ok_or(AbiError::InvalidArg)?;
                if end > buf.len() {
                    buf.resize(end, 0);
                }
                buf[offset..end].copy_from_slice(data);
                Ok(data.len())
            }
            RamNodeKind::Directory(_) => Err(AbiError::InvalidArg),
        }
    }

    fn readdir(&self) -> Result<Vec<String>, AbiError> {
        let guard = self.kind.lock();
        match &*guard {
            RamNodeKind::Directory(entries) => Ok(entries.keys().cloned().collect()),
            RamNodeKind::File(_) => Err(AbiError::InvalidArg),
        }
    }
}

const DEFAULT_NEXUS_CONF: &[u8] = b"\
# Nexus OS declarative system configuration
[system]
name = \"nexus-os\"
version = \"0.1.0\"
arch = \"x86_64\"

[personalities]
native = true
posix = true
nt = true
darwin = true

[vfs]
root = \"ramfs\"
";

pub struct RamFs {
    root: Arc<RamNode>,
    next_inode: SpinLock<u64>,
}

impl RamFs {
    pub fn new() -> Self {
        let fs = Self {
            root: RamNode::new_dir(1),
            next_inode: SpinLock::new(2),
        };
        let _ = fs.mkdir("/bin");
        let _ = fs.mkdir("/etc");
        let _ = fs.mkdir("/dev");
        let _ = fs.mkdir("/tmp");
        if let Ok(node) = fs.open("/etc/nexus.conf", OpenFlags::CREATE.union(OpenFlags::WRITE)) {
            let _ = node.write(0, DEFAULT_NEXUS_CONF);
        }
        fs
    }

    fn alloc_inode(&self) -> u64 {
        let mut guard = self.next_inode.lock();
        let id = *guard;
        *guard += 1;
        id
    }

    fn split_parent_and_name(path: &str) -> Result<(Vec<&str>, &str), AbiError> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let (&name, parent) = parts.split_last().ok_or(AbiError::InvalidArg)?;
        Ok((parent.to_vec(), name))
    }

    fn resolve_dir(&self, components: &[&str]) -> Result<Arc<RamNode>, AbiError> {
        let mut curr = Arc::clone(&self.root);
        for &comp in components {
            let next = {
                let guard = curr.kind.lock();
                match &*guard {
                    RamNodeKind::Directory(entries) => {
                        entries.get(comp).cloned().ok_or(AbiError::NotFound)?
                    }
                    RamNodeKind::File(_) => return Err(AbiError::InvalidArg),
                }
            };
            curr = next;
        }
        Ok(curr)
    }

    fn resolve_node(&self, path: &str) -> Result<Arc<RamNode>, AbiError> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        self.resolve_dir(&parts)
    }
}

impl FileSystem for RamFs {
    fn name(&self) -> &str {
        "ramfs"
    }

    fn open(&self, path: &str, flags: OpenFlags) -> Result<Arc<dyn VfsNode>, AbiError> {
        if path == "/" {
            return Ok(Arc::clone(&self.root) as Arc<dyn VfsNode>);
        }

        let (parent_parts, name) = Self::split_parent_and_name(path)?;
        let parent = self.resolve_dir(&parent_parts)?;

        let mut guard = parent.kind.lock();
        let entries = match &mut *guard {
            RamNodeKind::Directory(entries) => entries,
            RamNodeKind::File(_) => return Err(AbiError::InvalidArg),
        };

        if let Some(existing) = entries.get(name) {
            let stat = existing.stat()?;
            if flags.contains(OpenFlags::DIRECTORY) && stat.node_type != NodeType::Directory {
                return Err(AbiError::InvalidArg);
            }
            if flags.contains(OpenFlags::TRUNCATE) && flags.contains(OpenFlags::WRITE) {
                existing.truncate()?;
            }
            return Ok(Arc::clone(existing) as Arc<dyn VfsNode>);
        }

        if !flags.contains(OpenFlags::CREATE) {
            return Err(AbiError::NotFound);
        }

        let inode = self.alloc_inode();
        let node = if flags.contains(OpenFlags::DIRECTORY) {
            RamNode::new_dir(inode)
        } else {
            RamNode::new_file(inode, &[])
        };
        entries.insert(name.to_string(), Arc::clone(&node));
        Ok(node as Arc<dyn VfsNode>)
    }

    fn mkdir(&self, path: &str) -> Result<(), AbiError> {
        let (parent_parts, name) = Self::split_parent_and_name(path)?;
        let parent = self.resolve_dir(&parent_parts)?;

        let mut guard = parent.kind.lock();
        let entries = match &mut *guard {
            RamNodeKind::Directory(entries) => entries,
            RamNodeKind::File(_) => return Err(AbiError::InvalidArg),
        };

        if entries.contains_key(name) {
            return Err(AbiError::AlreadyExists);
        }

        let inode = self.alloc_inode();
        entries.insert(name.to_string(), RamNode::new_dir(inode));
        Ok(())
    }

    fn unlink(&self, path: &str) -> Result<(), AbiError> {
        let (parent_parts, name) = Self::split_parent_and_name(path)?;
        let parent = self.resolve_dir(&parent_parts)?;

        let mut guard = parent.kind.lock();
        let entries = match &mut *guard {
            RamNodeKind::Directory(entries) => entries,
            RamNodeKind::File(_) => return Err(AbiError::InvalidArg),
        };

        entries.remove(name).ok_or(AbiError::NotFound)?;
        Ok(())
    }

    fn stat(&self, path: &str) -> Result<VfsStat, AbiError> {
        let node = self.resolve_node(path)?;
        node.stat()
    }
}

static ROOT_FS: SpinLock<Option<Arc<RamFs>>> = SpinLock::new(None);

pub fn init() {
    let fs = Arc::new(RamFs::new());
    *ROOT_FS.lock() = Some(fs);
}

pub fn root_fs() -> Result<Arc<RamFs>, AbiError> {
    ROOT_FS.lock().clone().ok_or(AbiError::BadState)
}

pub fn open(path: &str, flags: OpenFlags) -> Result<Arc<dyn VfsNode>, AbiError> {
    root_fs()?.open(path, flags)
}

#[allow(dead_code)]
pub fn stat(path: &str) -> Result<VfsStat, AbiError> {
    root_fs()?.stat(path)
}

pub fn list_dir(path: &str) -> Result<Vec<String>, AbiError> {
    let node = open(path, OpenFlags::READ.union(OpenFlags::DIRECTORY))?;
    node.readdir()
}

pub fn read_to_vec(path: &str) -> Result<Vec<u8>, AbiError> {
    let node = open(path, OpenFlags::READ)?;
    let st = node.stat()?;
    let mut buf = alloc::vec![0u8; st.size];
    let n = node.read(0, &mut buf)?;
    buf.truncate(n);
    Ok(buf)
}
