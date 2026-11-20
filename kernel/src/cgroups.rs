//! Namespaces (`pid`, `mnt`, `net`, `uts`, `ipc`, `user`) & Cgroups v2 Resource Controller.
//!
//! Provides:
//! - Linux-compatible `no_std` process isolation namespaces:
//!   - `PidNamespace`: hierarchical PID translation (virtual PID <-> host PID) and init reaper
//!   - `MntNamespace`: isolated mount table with bind/ro/nosuid/noexec/nodev flags
//!   - `NetNamespace`: isolated network interfaces, IPv4/IPv6 addresses, routes, and port bindings
//!   - `UtsNamespace`: isolated hostname and domainname
//!   - `IpcNamespace`: isolated message queues, semaphore sets, and shared memory segments
//!   - `UserNamespace`: UID/GID mapping extents and capability bounding sets
//!   - `NsProxy` & `NamespaceManager`: `clone_ns`, `unshare`, `setns`
//! - Unified Cgroups v2 hierarchy (`CgroupManager`):
//!   - `cpu`: `cpu.weight` (1..=10000), `cpu.max` (quota/period), burst/throttling & `cpu.stat`
//!   - `memory`: `memory.min`, `memory.low`, `memory.high`, `memory.max`, `memory.swap.max`,
//!     hierarchical accounting, pressure/OOM events
//!   - `io`: `io.weight`, `io.max` per-device `(major, minor)` `rbps`/`wbps`/`riops`/`wiops` throttling
//!   - `pids`: `pids.max` hierarchical fork limit & `pids.events`
//! - `self_test()` verification suite
#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::sync::TicketLock;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors returned by namespace and cgroup v2 operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CgroupError {
    NotFound,
    AlreadyExists,
    InvalidArgument,
    PermissionDenied,
    Busy,
    LimitExceeded,
    OutOfMemory,
    Throttled,
    ControllerDisabled,
    MaxDepthExceeded,
    InvalidState,
}

impl fmt::Display for CgroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "not found"),
            Self::AlreadyExists => write!(f, "already exists"),
            Self::InvalidArgument => write!(f, "invalid argument"),
            Self::PermissionDenied => write!(f, "permission denied"),
            Self::Busy => write!(f, "resource busy"),
            Self::LimitExceeded => write!(f, "resource limit exceeded"),
            Self::OutOfMemory => write!(f, "out of memory"),
            Self::Throttled => write!(f, "resource throttled"),
            Self::ControllerDisabled => write!(f, "controller not enabled"),
            Self::MaxDepthExceeded => write!(f, "maximum hierarchy depth exceeded"),
            Self::InvalidState => write!(f, "invalid state"),
        }
    }
}

// ---------------------------------------------------------------------------
// Namespace Identifiers & Clone Flags
// ---------------------------------------------------------------------------

/// Unique namespace identifier.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NsId(pub u64);

static NEXT_NS_ID: AtomicU64 = AtomicU64::new(1);

fn alloc_ns_id() -> NsId {
    NsId(NEXT_NS_ID.fetch_add(1, Ordering::Relaxed))
}

/// Namespace kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NsKind {
    Pid,
    Mnt,
    Net,
    Uts,
    Ipc,
    User,
}

/// Bitmask of `CLONE_NEW*` flags for `clone_ns` / `unshare`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct CloneFlags(pub u32);

impl CloneFlags {
    pub const NONE: Self = Self(0);
    pub const NEWNS: Self = Self(0x0002_0000);
    pub const NEWUTS: Self = Self(0x0400_0000);
    pub const NEWIPC: Self = Self(0x0800_0000);
    pub const NEWUSER: Self = Self(0x1000_0000);
    pub const NEWPID: Self = Self(0x2000_0000);
    pub const NEWNET: Self = Self(0x4000_0000);
    pub const ALL: Self = Self(
        Self::NEWNS.0
            | Self::NEWUTS.0
            | Self::NEWIPC.0
            | Self::NEWUSER.0
            | Self::NEWPID.0
            | Self::NEWNET.0,
    );

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

// ---------------------------------------------------------------------------
// 1. PID Namespace
// ---------------------------------------------------------------------------

/// Maximum nesting depth for PID and User namespaces (matches Linux 32).
pub const MAX_NS_DEPTH: usize = 32;

/// Hierarchical PID namespace mapping virtual PIDs to host (root) PIDs.
#[derive(Clone, Debug)]
pub struct PidNamespace {
    pub id: NsId,
    pub parent: Option<NsId>,
    pub level: u8,
    pub next_vpid: u64,
    pub reaper_host_pid: Option<u64>,
    /// Virtual PID -> Host PID
    pub vpid_to_host: BTreeMap<u64, u64>,
    /// Host PID -> Virtual PID
    pub host_to_vpid: BTreeMap<u64, u64>,
}

impl PidNamespace {
    pub fn new(id: NsId, parent: Option<NsId>, level: u8) -> Self {
        Self {
            id,
            parent,
            level,
            next_vpid: 1,
            reaper_host_pid: None,
            vpid_to_host: BTreeMap::new(),
            host_to_vpid: BTreeMap::new(),
        }
    }

    /// Allocate a virtual PID in this namespace for `host_pid`.
    /// The first process allocated becomes PID 1 (the namespace init/reaper).
    pub fn alloc_pid(&mut self, host_pid: u64) -> Result<u64, CgroupError> {
        if let Some(&existing) = self.host_to_vpid.get(&host_pid) {
            return Ok(existing);
        }
        let vpid = self.next_vpid;
        self.next_vpid = self
            .next_vpid
            .checked_add(1)
            .ok_or(CgroupError::LimitExceeded)?;
        if vpid == 1 {
            self.reaper_host_pid = Some(host_pid);
        }
        self.vpid_to_host.insert(vpid, host_pid);
        self.host_to_vpid.insert(host_pid, vpid);
        Ok(vpid)
    }

    /// Remove a process from this PID namespace.
    pub fn remove_host_pid(&mut self, host_pid: u64) -> Option<u64> {
        if let Some(vpid) = self.host_to_vpid.remove(&host_pid) {
            self.vpid_to_host.remove(&vpid);
            if self.reaper_host_pid == Some(host_pid) {
                self.reaper_host_pid = None;
            }
            Some(vpid)
        } else {
            None
        }
    }

    #[inline]
    pub fn translate_to_host(&self, vpid: u64) -> Option<u64> {
        self.vpid_to_host.get(&vpid).copied()
    }

    #[inline]
    pub fn translate_from_host(&self, host_pid: u64) -> Option<u64> {
        self.host_to_vpid.get(&host_pid).copied()
    }

    #[inline]
    pub fn process_count(&self) -> usize {
        self.vpid_to_host.len()
    }
}

// ---------------------------------------------------------------------------
// 2. Mount Namespace
// ---------------------------------------------------------------------------

/// Mount flags for `MntNamespace`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct MountFlags(pub u32);

impl MountFlags {
    pub const NONE: Self = Self(0);
    pub const RDONLY: Self = Self(1 << 0);
    pub const NOSUID: Self = Self(1 << 1);
    pub const NODEV: Self = Self(1 << 2);
    pub const NOEXEC: Self = Self(1 << 3);
    pub const BIND: Self = Self(1 << 4);
    pub const PRIVATE: Self = Self(1 << 5);
    pub const SHARED: Self = Self(1 << 6);

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

/// A single mount entry inside a mount namespace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MountEntry {
    pub mount_id: u64,
    pub source: String,
    pub target: String,
    pub fstype: String,
    pub flags: MountFlags,
}

/// Isolated mount table namespace.
#[derive(Clone, Debug)]
pub struct MntNamespace {
    pub id: NsId,
    pub next_mount_id: u64,
    pub mounts: Vec<MountEntry>,
}

impl MntNamespace {
    pub fn new(id: NsId) -> Self {
        let mut ns = Self {
            id,
            next_mount_id: 1,
            mounts: Vec::new(),
        };
        let _ = ns.mount("rootfs", "/", "ramfs", MountFlags::NONE);
        ns
    }

    pub fn clone_with_id(&self, new_id: NsId) -> Self {
        Self {
            id: new_id,
            next_mount_id: self.next_mount_id,
            mounts: self.mounts.clone(),
        }
    }

    pub fn mount(
        &mut self,
        source: &str,
        target: &str,
        fstype: &str,
        flags: MountFlags,
    ) -> Result<u64, CgroupError> {
        if target.is_empty() || !target.starts_with('/') {
            return Err(CgroupError::InvalidArgument);
        }
        let mount_id = self.next_mount_id;
        self.next_mount_id += 1;
        self.mounts.push(MountEntry {
            mount_id,
            source: source.to_string(),
            target: target.to_string(),
            fstype: fstype.to_string(),
            flags,
        });
        Ok(mount_id)
    }

    pub fn umount(&mut self, target: &str) -> Result<(), CgroupError> {
        if target == "/" {
            return Err(CgroupError::Busy);
        }
        if let Some(pos) = self.mounts.iter().rposition(|m| m.target == target) {
            self.mounts.remove(pos);
            Ok(())
        } else {
            Err(CgroupError::NotFound)
        }
    }

    /// Resolve the deepest mount point matching `path`.
    pub fn resolve_mount(&self, path: &str) -> Option<&MountEntry> {
        let mut best: Option<&MountEntry> = None;
        let mut best_len = 0usize;
        for entry in &self.mounts {
            let t = entry.target.as_str();
            let matches = if t == "/" {
                path.starts_with('/')
            } else {
                path == t || (path.starts_with(t) && path.as_bytes().get(t.len()) == Some(&b'/'))
            };
            if matches && t.len() >= best_len {
                best_len = t.len();
                best = Some(entry);
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// 3. Network Namespace
// ---------------------------------------------------------------------------

/// Virtual network interface inside a `NetNamespace`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetIf {
    pub name: String,
    pub mac: [u8; 6],
    pub ipv4: Option<[u8; 4]>,
    pub prefix_len: u8,
    pub mtu: u32,
    pub up: bool,
    pub peer_ns: Option<NsId>,
}

/// Routing table entry inside a `NetNamespace`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetRoute {
    pub dst: [u8; 4],
    pub prefix_len: u8,
    pub gateway: Option<[u8; 4]>,
    pub dev: String,
}

/// Isolated network stack namespace.
#[derive(Clone, Debug)]
pub struct NetNamespace {
    pub id: NsId,
    pub interfaces: Vec<NetIf>,
    pub routes: Vec<NetRoute>,
    pub bound_tcp_ports: Vec<u16>,
    pub bound_udp_ports: Vec<u16>,
}

impl NetNamespace {
    pub fn new(id: NsId) -> Self {
        let mut ns = Self {
            id,
            interfaces: Vec::new(),
            routes: Vec::new(),
            bound_tcp_ports: Vec::new(),
            bound_udp_ports: Vec::new(),
        };
        ns.interfaces.push(NetIf {
            name: String::from("lo"),
            mac: [0; 6],
            ipv4: Some([127, 0, 0, 1]),
            prefix_len: 8,
            mtu: 65536,
            up: true,
            peer_ns: None,
        });
        ns
    }

    pub fn add_interface(&mut self, iface: NetIf) -> Result<(), CgroupError> {
        if self.interfaces.iter().any(|i| i.name == iface.name) {
            return Err(CgroupError::AlreadyExists);
        }
        self.interfaces.push(iface);
        Ok(())
    }

    pub fn add_route(&mut self, route: NetRoute) -> Result<(), CgroupError> {
        if !self.interfaces.iter().any(|i| i.name == route.dev) {
            return Err(CgroupError::NotFound);
        }
        self.routes.push(route);
        Ok(())
    }

    pub fn bind_tcp_port(&mut self, port: u16) -> Result<(), CgroupError> {
        if port == 0 || self.bound_tcp_ports.contains(&port) {
            return Err(CgroupError::Busy);
        }
        self.bound_tcp_ports.push(port);
        Ok(())
    }

    pub fn unbind_tcp_port(&mut self, port: u16) {
        self.bound_tcp_ports.retain(|&p| p != port);
    }
}

// ---------------------------------------------------------------------------
// 4. UTS Namespace
// ---------------------------------------------------------------------------

/// Isolated hostname and domainname namespace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UtsNamespace {
    pub id: NsId,
    pub hostname: String,
    pub domainname: String,
    pub sysname: String,
    pub release: String,
    pub machine: String,
}

impl UtsNamespace {
    pub fn new(id: NsId) -> Self {
        Self {
            id,
            hostname: String::from("nexus"),
            domainname: String::from("(none)"),
            sysname: String::from("NexusOS"),
            release: String::from("0.1.0"),
            machine: String::from("x86_64"),
        }
    }

    pub fn clone_with_id(&self, new_id: NsId) -> Self {
        let mut cloned = self.clone();
        cloned.id = new_id;
        cloned
    }

    pub fn set_hostname(&mut self, name: &str) -> Result<(), CgroupError> {
        if name.is_empty() || name.len() > 64 {
            return Err(CgroupError::InvalidArgument);
        }
        self.hostname = name.to_string();
        Ok(())
    }

    pub fn set_domainname(&mut self, name: &str) -> Result<(), CgroupError> {
        if name.len() > 64 {
            return Err(CgroupError::InvalidArgument);
        }
        self.domainname = name.to_string();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 5. IPC Namespace
// ---------------------------------------------------------------------------

/// Isolated SysV/POSIX IPC namespace (message queues, semaphores, shared memory).
#[derive(Clone, Debug)]
pub struct IpcNamespace {
    pub id: NsId,
    pub next_ipc_id: u64,
    pub shm_segments: BTreeMap<u64, usize>,
    pub msg_queues: BTreeMap<u64, usize>,
    pub sem_sets: BTreeMap<u64, u32>,
    pub shm_max_bytes: usize,
    pub shm_used_bytes: usize,
}

impl IpcNamespace {
    pub fn new(id: NsId) -> Self {
        Self {
            id,
            next_ipc_id: 1,
            shm_segments: BTreeMap::new(),
            msg_queues: BTreeMap::new(),
            sem_sets: BTreeMap::new(),
            shm_max_bytes: 64 * 1024 * 1024,
            shm_used_bytes: 0,
        }
    }

    pub fn create_shm(&mut self, size: usize) -> Result<u64, CgroupError> {
        if size == 0 {
            return Err(CgroupError::InvalidArgument);
        }
        let new_used = self
            .shm_used_bytes
            .checked_add(size)
            .ok_or(CgroupError::LimitExceeded)?;
        if new_used > self.shm_max_bytes {
            return Err(CgroupError::LimitExceeded);
        }
        let id = self.next_ipc_id;
        self.next_ipc_id += 1;
        self.shm_used_bytes = new_used;
        self.shm_segments.insert(id, size);
        Ok(id)
    }

    pub fn remove_shm(&mut self, id: u64) -> Result<(), CgroupError> {
        let size = self.shm_segments.remove(&id).ok_or(CgroupError::NotFound)?;
        self.shm_used_bytes = self.shm_used_bytes.saturating_sub(size);
        Ok(())
    }

    pub fn create_msg_queue(&mut self, max_bytes: usize) -> u64 {
        let id = self.next_ipc_id;
        self.next_ipc_id += 1;
        self.msg_queues.insert(id, max_bytes);
        id
    }

    pub fn create_sem_set(&mut self, nsems: u32) -> Result<u64, CgroupError> {
        if nsems == 0 {
            return Err(CgroupError::InvalidArgument);
        }
        let id = self.next_ipc_id;
        self.next_ipc_id += 1;
        self.sem_sets.insert(id, nsems);
        Ok(id)
    }
}

// ---------------------------------------------------------------------------
// 6. User Namespace
// ---------------------------------------------------------------------------

/// Single UID/GID mapping extent (`inside_id .. inside_id + length` <-> `outside_id ..`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IdMapExtent {
    pub inside_id: u32,
    pub outside_id: u32,
    pub length: u32,
}

impl IdMapExtent {
    #[inline]
    pub fn map_to_outside(self, inside: u32) -> Option<u32> {
        let offset = inside.checked_sub(self.inside_id)?;
        if offset < self.length {
            self.outside_id.checked_add(offset)
        } else {
            None
        }
    }

    #[inline]
    pub fn map_to_inside(self, outside: u32) -> Option<u32> {
        let offset = outside.checked_sub(self.outside_id)?;
        if offset < self.length {
            self.inside_id.checked_add(offset)
        } else {
            None
        }
    }
}

/// Isolated user/group ID mapping and capability namespace.
#[derive(Clone, Debug)]
pub struct UserNamespace {
    pub id: NsId,
    pub parent: Option<NsId>,
    pub level: u8,
    pub owner_uid: u32,
    pub owner_gid: u32,
    pub uid_map: Vec<IdMapExtent>,
    pub gid_map: Vec<IdMapExtent>,
    pub cap_effective: u64,
}

impl UserNamespace {
    pub fn new_root(id: NsId) -> Self {
        Self {
            id,
            parent: None,
            level: 0,
            owner_uid: 0,
            owner_gid: 0,
            uid_map: alloc::vec![IdMapExtent {
                inside_id: 0,
                outside_id: 0,
                length: u32::MAX,
            }],
            gid_map: alloc::vec![IdMapExtent {
                inside_id: 0,
                outside_id: 0,
                length: u32::MAX,
            }],
            cap_effective: u64::MAX,
        }
    }

    pub fn new_child(id: NsId, parent: NsId, level: u8, owner_uid: u32, owner_gid: u32) -> Self {
        Self {
            id,
            parent: Some(parent),
            level,
            owner_uid,
            owner_gid,
            uid_map: Vec::new(),
            gid_map: Vec::new(),
            cap_effective: u64::MAX,
        }
    }

    pub fn set_uid_map(&mut self, extents: Vec<IdMapExtent>) -> Result<(), CgroupError> {
        if !self.uid_map.is_empty() && self.parent.is_some() {
            return Err(CgroupError::PermissionDenied);
        }
        Self::validate_extents(&extents)?;
        self.uid_map = extents;
        Ok(())
    }

    pub fn set_gid_map(&mut self, extents: Vec<IdMapExtent>) -> Result<(), CgroupError> {
        if !self.gid_map.is_empty() && self.parent.is_some() {
            return Err(CgroupError::PermissionDenied);
        }
        Self::validate_extents(&extents)?;
        self.gid_map = extents;
        Ok(())
    }

    fn validate_extents(extents: &[IdMapExtent]) -> Result<(), CgroupError> {
        if extents.is_empty() || extents.len() > 340 {
            return Err(CgroupError::InvalidArgument);
        }
        for ext in extents {
            if ext.length == 0 {
                return Err(CgroupError::InvalidArgument);
            }
        }
        Ok(())
    }

    pub fn map_uid_to_outside(&self, inside_uid: u32) -> Option<u32> {
        self.uid_map
            .iter()
            .find_map(|e| e.map_to_outside(inside_uid))
    }

    pub fn map_uid_to_inside(&self, outside_uid: u32) -> Option<u32> {
        self.uid_map
            .iter()
            .find_map(|e| e.map_to_inside(outside_uid))
    }

    pub fn map_gid_to_outside(&self, inside_gid: u32) -> Option<u32> {
        self.gid_map
            .iter()
            .find_map(|e| e.map_to_outside(inside_gid))
    }

    pub fn map_gid_to_inside(&self, outside_gid: u32) -> Option<u32> {
        self.gid_map
            .iter()
            .find_map(|e| e.map_to_inside(outside_gid))
    }
}

// ---------------------------------------------------------------------------
// NsProxy & NamespaceManager
// ---------------------------------------------------------------------------

/// Per-process bundle of active namespace references.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NsProxy {
    pub pid_ns: NsId,
    pub mnt_ns: NsId,
    pub net_ns: NsId,
    pub uts_ns: NsId,
    pub ipc_ns: NsId,
    pub user_ns: NsId,
}

/// Central registry and lifecycle manager for all 6 namespace types.
pub struct NamespaceManager {
    pub root_proxy: NsProxy,
    pub pid_namespaces: BTreeMap<NsId, PidNamespace>,
    pub mnt_namespaces: BTreeMap<NsId, MntNamespace>,
    pub net_namespaces: BTreeMap<NsId, NetNamespace>,
    pub uts_namespaces: BTreeMap<NsId, UtsNamespace>,
    pub ipc_namespaces: BTreeMap<NsId, IpcNamespace>,
    pub user_namespaces: BTreeMap<NsId, UserNamespace>,
    pub proc_proxies: BTreeMap<u64, NsProxy>,
}

impl NamespaceManager {
    pub fn new() -> Self {
        let pid_id = alloc_ns_id();
        let mnt_id = alloc_ns_id();
        let net_id = alloc_ns_id();
        let uts_id = alloc_ns_id();
        let ipc_id = alloc_ns_id();
        let user_id = alloc_ns_id();

        let mut pid_namespaces = BTreeMap::new();
        pid_namespaces.insert(pid_id, PidNamespace::new(pid_id, None, 0));

        let mut mnt_namespaces = BTreeMap::new();
        mnt_namespaces.insert(mnt_id, MntNamespace::new(mnt_id));

        let mut net_namespaces = BTreeMap::new();
        net_namespaces.insert(net_id, NetNamespace::new(net_id));

        let mut uts_namespaces = BTreeMap::new();
        uts_namespaces.insert(uts_id, UtsNamespace::new(uts_id));

        let mut ipc_namespaces = BTreeMap::new();
        ipc_namespaces.insert(ipc_id, IpcNamespace::new(ipc_id));

        let mut user_namespaces = BTreeMap::new();
        user_namespaces.insert(user_id, UserNamespace::new_root(user_id));

        let root_proxy = NsProxy {
            pid_ns: pid_id,
            mnt_ns: mnt_id,
            net_ns: net_id,
            uts_ns: uts_id,
            ipc_ns: ipc_id,
            user_ns: user_id,
        };

        Self {
            root_proxy,
            pid_namespaces,
            mnt_namespaces,
            net_namespaces,
            uts_namespaces,
            ipc_namespaces,
            user_namespaces,
            proc_proxies: BTreeMap::new(),
        }
    }

    /// Register a process with the root namespace proxy if not already registered.
    pub fn register_process(&mut self, host_pid: u64) -> Result<NsProxy, CgroupError> {
        if let Some(&proxy) = self.proc_proxies.get(&host_pid) {
            return Ok(proxy);
        }
        if let Some(pid_ns) = self.pid_namespaces.get_mut(&self.root_proxy.pid_ns) {
            pid_ns.alloc_pid(host_pid)?;
        }
        self.proc_proxies.insert(host_pid, self.root_proxy);
        Ok(self.root_proxy)
    }

    /// Clone namespaces for a child process according to `flags`.
    pub fn clone_ns(
        &mut self,
        parent_host_pid: u64,
        child_host_pid: u64,
        flags: CloneFlags,
    ) -> Result<NsProxy, CgroupError> {
        let parent_proxy = self
            .proc_proxies
            .get(&parent_host_pid)
            .copied()
            .unwrap_or(self.root_proxy);

        let mut child_proxy = parent_proxy;

        if flags.contains(CloneFlags::NEWUSER) {
            let parent_user = self
                .user_namespaces
                .get(&parent_proxy.user_ns)
                .ok_or(CgroupError::NotFound)?;
            if usize::from(parent_user.level) + 1 >= MAX_NS_DEPTH {
                return Err(CgroupError::MaxDepthExceeded);
            }
            let new_id = alloc_ns_id();
            let child_user =
                UserNamespace::new_child(new_id, parent_proxy.user_ns, parent_user.level + 1, 0, 0);
            self.user_namespaces.insert(new_id, child_user);
            child_proxy.user_ns = new_id;
        }

        if flags.contains(CloneFlags::NEWPID) {
            let parent_pid = self
                .pid_namespaces
                .get(&parent_proxy.pid_ns)
                .ok_or(CgroupError::NotFound)?;
            if usize::from(parent_pid.level) + 1 >= MAX_NS_DEPTH {
                return Err(CgroupError::MaxDepthExceeded);
            }
            let new_id = alloc_ns_id();
            let mut child_pid =
                PidNamespace::new(new_id, Some(parent_proxy.pid_ns), parent_pid.level + 1);
            child_pid.alloc_pid(child_host_pid)?;
            self.pid_namespaces.insert(new_id, child_pid);
            child_proxy.pid_ns = new_id;
        } else if let Some(pid_ns) = self.pid_namespaces.get_mut(&child_proxy.pid_ns) {
            pid_ns.alloc_pid(child_host_pid)?;
        }

        if flags.contains(CloneFlags::NEWNS) {
            let parent_mnt = self
                .mnt_namespaces
                .get(&parent_proxy.mnt_ns)
                .ok_or(CgroupError::NotFound)?;
            let new_id = alloc_ns_id();
            let cloned = parent_mnt.clone_with_id(new_id);
            self.mnt_namespaces.insert(new_id, cloned);
            child_proxy.mnt_ns = new_id;
        }

        if flags.contains(CloneFlags::NEWNET) {
            let new_id = alloc_ns_id();
            self.net_namespaces
                .insert(new_id, NetNamespace::new(new_id));
            child_proxy.net_ns = new_id;
        }

        if flags.contains(CloneFlags::NEWUTS) {
            let parent_uts = self
                .uts_namespaces
                .get(&parent_proxy.uts_ns)
                .ok_or(CgroupError::NotFound)?;
            let new_id = alloc_ns_id();
            let cloned = parent_uts.clone_with_id(new_id);
            self.uts_namespaces.insert(new_id, cloned);
            child_proxy.uts_ns = new_id;
        }

        if flags.contains(CloneFlags::NEWIPC) {
            let new_id = alloc_ns_id();
            self.ipc_namespaces
                .insert(new_id, IpcNamespace::new(new_id));
            child_proxy.ipc_ns = new_id;
        }

        self.proc_proxies.insert(child_host_pid, child_proxy);
        Ok(child_proxy)
    }

    /// Unshare selected namespaces for an existing process.
    pub fn unshare(&mut self, host_pid: u64, flags: CloneFlags) -> Result<NsProxy, CgroupError> {
        let current = self
            .proc_proxies
            .get(&host_pid)
            .copied()
            .unwrap_or(self.root_proxy);
        let mut updated = current;

        if flags.contains(CloneFlags::NEWUSER) {
            let parent_user = self
                .user_namespaces
                .get(&current.user_ns)
                .ok_or(CgroupError::NotFound)?;
            if usize::from(parent_user.level) + 1 >= MAX_NS_DEPTH {
                return Err(CgroupError::MaxDepthExceeded);
            }
            let new_id = alloc_ns_id();
            self.user_namespaces.insert(
                new_id,
                UserNamespace::new_child(new_id, current.user_ns, parent_user.level + 1, 0, 0),
            );
            updated.user_ns = new_id;
        }

        if flags.contains(CloneFlags::NEWNS) {
            let parent_mnt = self
                .mnt_namespaces
                .get(&current.mnt_ns)
                .ok_or(CgroupError::NotFound)?;
            let new_id = alloc_ns_id();
            self.mnt_namespaces
                .insert(new_id, parent_mnt.clone_with_id(new_id));
            updated.mnt_ns = new_id;
        }

        if flags.contains(CloneFlags::NEWNET) {
            let new_id = alloc_ns_id();
            self.net_namespaces
                .insert(new_id, NetNamespace::new(new_id));
            updated.net_ns = new_id;
        }

        if flags.contains(CloneFlags::NEWUTS) {
            let parent_uts = self
                .uts_namespaces
                .get(&current.uts_ns)
                .ok_or(CgroupError::NotFound)?;
            let new_id = alloc_ns_id();
            self.uts_namespaces
                .insert(new_id, parent_uts.clone_with_id(new_id));
            updated.uts_ns = new_id;
        }

        if flags.contains(CloneFlags::NEWIPC) {
            let new_id = alloc_ns_id();
            self.ipc_namespaces
                .insert(new_id, IpcNamespace::new(new_id));
            updated.ipc_ns = new_id;
        }

        if flags.contains(CloneFlags::NEWPID) {
            let parent_pid = self
                .pid_namespaces
                .get(&current.pid_ns)
                .ok_or(CgroupError::NotFound)?;
            if usize::from(parent_pid.level) + 1 >= MAX_NS_DEPTH {
                return Err(CgroupError::MaxDepthExceeded);
            }
            let new_id = alloc_ns_id();
            let mut child_pid =
                PidNamespace::new(new_id, Some(current.pid_ns), parent_pid.level + 1);
            child_pid.alloc_pid(host_pid)?;
            self.pid_namespaces.insert(new_id, child_pid);
            updated.pid_ns = new_id;
        }

        self.proc_proxies.insert(host_pid, updated);
        Ok(updated)
    }

    /// Reassociate `host_pid` with an existing namespace `target_ns` of `kind`.
    pub fn setns(
        &mut self,
        host_pid: u64,
        kind: NsKind,
        target_ns: NsId,
    ) -> Result<NsProxy, CgroupError> {
        let mut proxy = self
            .proc_proxies
            .get(&host_pid)
            .copied()
            .unwrap_or(self.root_proxy);

        match kind {
            NsKind::Pid => {
                let ns = self
                    .pid_namespaces
                    .get_mut(&target_ns)
                    .ok_or(CgroupError::NotFound)?;
                ns.alloc_pid(host_pid)?;
                proxy.pid_ns = target_ns;
            }
            NsKind::Mnt => {
                if !self.mnt_namespaces.contains_key(&target_ns) {
                    return Err(CgroupError::NotFound);
                }
                proxy.mnt_ns = target_ns;
            }
            NsKind::Net => {
                if !self.net_namespaces.contains_key(&target_ns) {
                    return Err(CgroupError::NotFound);
                }
                proxy.net_ns = target_ns;
            }
            NsKind::Uts => {
                if !self.uts_namespaces.contains_key(&target_ns) {
                    return Err(CgroupError::NotFound);
                }
                proxy.uts_ns = target_ns;
            }
            NsKind::Ipc => {
                if !self.ipc_namespaces.contains_key(&target_ns) {
                    return Err(CgroupError::NotFound);
                }
                proxy.ipc_ns = target_ns;
            }
            NsKind::User => {
                if !self.user_namespaces.contains_key(&target_ns) {
                    return Err(CgroupError::NotFound);
                }
                proxy.user_ns = target_ns;
            }
        }

        self.proc_proxies.insert(host_pid, proxy);
        Ok(proxy)
    }

    /// Clean up process namespace membership on exit.
    pub fn exit_process(&mut self, host_pid: u64) {
        if let Some(proxy) = self.proc_proxies.remove(&host_pid) {
            let mut cur = Some(proxy.pid_ns);
            while let Some(ns_id) = cur {
                if let Some(ns) = self.pid_namespaces.get_mut(&ns_id) {
                    ns.remove_host_pid(host_pid);
                    cur = ns.parent;
                } else {
                    break;
                }
            }
        }
    }
}

impl Default for NamespaceManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cgroups v2 Resource Controller
// ---------------------------------------------------------------------------

/// Unique cgroup v2 node identifier.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CgroupId(pub u64);

impl CgroupId {
    pub const ROOT: Self = Self(0);
}

/// Bitmask of cgroup v2 controllers (`cgroup.controllers` / `cgroup.subtree_control`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct ControllerMask(pub u32);

impl ControllerMask {
    pub const NONE: Self = Self(0);
    pub const CPU: Self = Self(1 << 0);
    pub const MEMORY: Self = Self(1 << 1);
    pub const IO: Self = Self(1 << 2);
    pub const PIDS: Self = Self(1 << 3);
    pub const ALL: Self = Self(Self::CPU.0 | Self::MEMORY.0 | Self::IO.0 | Self::PIDS.0);

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

// --- CPU Controller ---

/// `cpu.weight` bounds (Linux cgroup v2: 1..=10000, default 100).
pub const CPU_WEIGHT_MIN: u32 = 1;
pub const CPU_WEIGHT_DEFAULT: u32 = 100;
pub const CPU_WEIGHT_MAX: u32 = 10_000;
pub const CPU_DEFAULT_PERIOD_US: u64 = 100_000;

/// CPU controller configuration and runtime accounting (`cpu.weight`, `cpu.max`, `cpu.stat`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CpuController {
    pub weight: u32,
    /// `None` means `"max"` (unlimited).
    pub quota_us: Option<u64>,
    pub period_us: u64,
    pub burst_us: u64,
    pub period_used_us: u64,
    pub usage_usec: u64,
    pub user_usec: u64,
    pub system_usec: u64,
    pub nr_periods: u64,
    pub nr_throttled: u64,
    pub throttled_usec: u64,
}

impl Default for CpuController {
    fn default() -> Self {
        Self {
            weight: CPU_WEIGHT_DEFAULT,
            quota_us: None,
            period_us: CPU_DEFAULT_PERIOD_US,
            burst_us: 0,
            period_used_us: 0,
            usage_usec: 0,
            user_usec: 0,
            system_usec: 0,
            nr_periods: 0,
            nr_throttled: 0,
            throttled_usec: 0,
        }
    }
}

impl CpuController {
    pub fn set_weight(&mut self, weight: u32) -> Result<(), CgroupError> {
        if !(CPU_WEIGHT_MIN..=CPU_WEIGHT_MAX).contains(&weight) {
            return Err(CgroupError::InvalidArgument);
        }
        self.weight = weight;
        Ok(())
    }

    pub fn set_max(&mut self, quota_us: Option<u64>, period_us: u64) -> Result<(), CgroupError> {
        if !(1_000..=1_000_000).contains(&period_us) {
            return Err(CgroupError::InvalidArgument);
        }
        if let Some(q) = quota_us {
            if q < 1_000 {
                return Err(CgroupError::InvalidArgument);
            }
        }
        self.quota_us = quota_us;
        self.period_us = period_us;
        Ok(())
    }

    #[inline]
    pub fn is_throttled(&self) -> bool {
        if let Some(quota) = self.quota_us {
            self.period_used_us >= quota.saturating_add(self.burst_us)
        } else {
            false
        }
    }

    /// Charge CPU execution time (`user_us`, `sys_us`). Returns `Err(Throttled)` if quota exceeded.
    pub fn charge(&mut self, user_us: u64, sys_us: u64) -> Result<(), CgroupError> {
        let total = user_us.saturating_add(sys_us);
        self.usage_usec = self.usage_usec.saturating_add(total);
        self.user_usec = self.user_usec.saturating_add(user_us);
        self.system_usec = self.system_usec.saturating_add(sys_us);
        self.period_used_us = self.period_used_us.saturating_add(total);

        if let Some(quota) = self.quota_us {
            let limit = quota.saturating_add(self.burst_us);
            if self.period_used_us > limit {
                self.nr_throttled += 1;
                self.throttled_usec = self
                    .throttled_usec
                    .saturating_add(self.period_used_us - limit);
                return Err(CgroupError::Throttled);
            }
        }
        Ok(())
    }

    /// Advance to the next bandwidth period, resetting `period_used_us`.
    pub fn refill_period(&mut self) {
        self.nr_periods += 1;
        self.period_used_us = 0;
    }
}

// --- Memory Controller ---

/// Memory pressure / limit events (`memory.events`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryEvents {
    pub low: u64,
    pub high: u64,
    pub max: u64,
    pub oom: u64,
    pub oom_kill: u64,
}

/// Memory controller (`memory.min`, `memory.low`, `memory.high`, `memory.max`, `memory.current`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryController {
    pub min_bytes: u64,
    pub low_bytes: u64,
    pub high_bytes: u64,
    pub max_bytes: u64,
    pub swap_max_bytes: u64,
    pub current_bytes: u64,
    pub swap_current_bytes: u64,
    pub peak_bytes: u64,
    pub events: MemoryEvents,
}

impl Default for MemoryController {
    fn default() -> Self {
        Self {
            min_bytes: 0,
            low_bytes: 0,
            high_bytes: u64::MAX,
            max_bytes: u64::MAX,
            swap_max_bytes: u64::MAX,
            current_bytes: 0,
            swap_current_bytes: 0,
            peak_bytes: 0,
            events: MemoryEvents::default(),
        }
    }
}

/// Outcome of a memory charge check.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MemoryChargeStatus {
    Ok,
    HighPressure,
}

// --- I/O Controller ---

/// Block device identifier `(major, minor)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockDevId {
    pub major: u32,
    pub minor: u32,
}

/// Per-device I/O bandwidth and IOPS limits (`io.max`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IoLimit {
    pub rbps: u64,
    pub wbps: u64,
    pub riops: u64,
    pub wiops: u64,
}

impl Default for IoLimit {
    fn default() -> Self {
        Self {
            rbps: u64::MAX,
            wbps: u64::MAX,
            riops: u64::MAX,
            wiops: u64::MAX,
        }
    }
}

/// Per-device I/O accounting (`io.stat`) and current 1-second window counters.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct IoStat {
    pub rbytes: u64,
    pub wbytes: u64,
    pub rios: u64,
    pub wios: u64,
    pub window_rbytes: u64,
    pub window_wbytes: u64,
    pub window_rios: u64,
    pub window_wios: u64,
}

/// I/O direction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IoDir {
    Read,
    Write,
}

/// I/O controller (`io.weight`, `io.max`, `io.stat`).
#[derive(Clone, Debug)]
pub struct IoController {
    pub weight: u32,
    pub limits: BTreeMap<BlockDevId, IoLimit>,
    pub stats: BTreeMap<BlockDevId, IoStat>,
}

impl Default for IoController {
    fn default() -> Self {
        Self {
            weight: 100,
            limits: BTreeMap::new(),
            stats: BTreeMap::new(),
        }
    }
}

impl IoController {
    pub fn set_weight(&mut self, weight: u32) -> Result<(), CgroupError> {
        if !(1..=10_000).contains(&weight) {
            return Err(CgroupError::InvalidArgument);
        }
        self.weight = weight;
        Ok(())
    }

    pub fn set_limit(&mut self, dev: BlockDevId, limit: IoLimit) {
        self.limits.insert(dev, limit);
    }

    /// Charge an I/O operation against `dev` in the current 1-second window.
    pub fn charge_io(
        &mut self,
        dev: BlockDevId,
        dir: IoDir,
        bytes: u64,
    ) -> Result<(), CgroupError> {
        let limit = self.limits.get(&dev).copied().unwrap_or_default();
        let stat = self.stats.entry(dev).or_default();

        match dir {
            IoDir::Read => {
                let next_bytes = stat.window_rbytes.saturating_add(bytes);
                let next_ios = stat.window_rios.saturating_add(1);
                if next_bytes > limit.rbps || next_ios > limit.riops {
                    return Err(CgroupError::Throttled);
                }
                stat.window_rbytes = next_bytes;
                stat.window_rios = next_ios;
                stat.rbytes = stat.rbytes.saturating_add(bytes);
                stat.rios = stat.rios.saturating_add(1);
            }
            IoDir::Write => {
                let next_bytes = stat.window_wbytes.saturating_add(bytes);
                let next_ios = stat.window_wios.saturating_add(1);
                if next_bytes > limit.wbps || next_ios > limit.wiops {
                    return Err(CgroupError::Throttled);
                }
                stat.window_wbytes = next_bytes;
                stat.window_wios = next_ios;
                stat.wbytes = stat.wbytes.saturating_add(bytes);
                stat.wios = stat.wios.saturating_add(1);
            }
        }
        Ok(())
    }

    /// Reset per-second window counters for all devices.
    pub fn reset_window(&mut self) {
        for stat in self.stats.values_mut() {
            stat.window_rbytes = 0;
            stat.window_wbytes = 0;
            stat.window_rios = 0;
            stat.window_wios = 0;
        }
    }
}

// --- PIDs Controller ---

/// PIDs controller (`pids.max`, `pids.current`, `pids.peak`, `pids.events`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PidsController {
    pub max: u64,
    pub current: u64,
    pub peak: u64,
    pub events_max: u64,
}

impl Default for PidsController {
    fn default() -> Self {
        Self {
            max: u64::MAX,
            current: 0,
            peak: 0,
            events_max: 0,
        }
    }
}

// --- Cgroup Node & Manager ---

/// A single node in the unified cgroup v2 hierarchy.
#[derive(Clone, Debug)]
pub struct CgroupNode {
    pub id: CgroupId,
    pub name: String,
    pub path: String,
    pub parent: Option<CgroupId>,
    pub children: Vec<CgroupId>,
    pub procs: Vec<u64>,
    pub controllers: ControllerMask,
    pub subtree_control: ControllerMask,
    pub cpu: CpuController,
    pub memory: MemoryController,
    pub io: IoController,
    pub pids: PidsController,
}

impl CgroupNode {
    fn new(
        id: CgroupId,
        name: String,
        path: String,
        parent: Option<CgroupId>,
        controllers: ControllerMask,
    ) -> Self {
        Self {
            id,
            name,
            path,
            parent,
            children: Vec::new(),
            procs: Vec::new(),
            controllers,
            subtree_control: ControllerMask::NONE,
            cpu: CpuController::default(),
            memory: MemoryController::default(),
            io: IoController::default(),
            pids: PidsController::default(),
        }
    }
}

/// Unified Cgroups v2 hierarchy manager.
pub struct CgroupManager {
    next_id: u64,
    pub nodes: BTreeMap<CgroupId, CgroupNode>,
    pub path_index: BTreeMap<String, CgroupId>,
    pub proc_cgroup: BTreeMap<u64, CgroupId>,
}

impl CgroupManager {
    pub fn new() -> Self {
        let mut nodes = BTreeMap::new();
        let mut path_index = BTreeMap::new();
        let root = CgroupNode::new(
            CgroupId::ROOT,
            String::from("/"),
            String::from("/"),
            None,
            ControllerMask::ALL,
        );
        nodes.insert(CgroupId::ROOT, root);
        path_index.insert(String::from("/"), CgroupId::ROOT);

        Self {
            next_id: 1,
            nodes,
            path_index,
            proc_cgroup: BTreeMap::new(),
        }
    }

    /// Enable or disable controllers in `cgroup.subtree_control` for `id`.
    pub fn set_subtree_control(
        &mut self,
        id: CgroupId,
        mask: ControllerMask,
    ) -> Result<(), CgroupError> {
        let node = self.nodes.get_mut(&id).ok_or(CgroupError::NotFound)?;
        if !node.controllers.contains(mask) {
            return Err(CgroupError::ControllerDisabled);
        }
        // Cgroup v2 no-internal-process constraint: non-root cgroups with processes
        // cannot enable subtree_control.
        if id != CgroupId::ROOT && !node.procs.is_empty() && mask != ControllerMask::NONE {
            return Err(CgroupError::Busy);
        }
        node.subtree_control = mask;
        Ok(())
    }

    /// Create a child cgroup under `parent_id`.
    pub fn create_cgroup(
        &mut self,
        parent_id: CgroupId,
        name: &str,
    ) -> Result<CgroupId, CgroupError> {
        if name.is_empty() || name.contains('/') || name == "." || name == ".." {
            return Err(CgroupError::InvalidArgument);
        }
        let (parent_path, inherited_controllers) = {
            let parent = self.nodes.get(&parent_id).ok_or(CgroupError::NotFound)?;
            (parent.path.clone(), parent.subtree_control)
        };

        let full_path = if parent_path == "/" {
            alloc::format!("/{}", name)
        } else {
            alloc::format!("{}/{}", parent_path, name)
        };

        if self.path_index.contains_key(&full_path) {
            return Err(CgroupError::AlreadyExists);
        }

        let id = CgroupId(self.next_id);
        self.next_id += 1;

        let node = CgroupNode::new(
            id,
            name.to_string(),
            full_path.clone(),
            Some(parent_id),
            inherited_controllers,
        );
        self.nodes.insert(id, node);
        self.path_index.insert(full_path, id);
        if let Some(parent) = self.nodes.get_mut(&parent_id) {
            parent.children.push(id);
        }
        Ok(id)
    }

    /// Remove an empty leaf cgroup.
    pub fn remove_cgroup(&mut self, id: CgroupId) -> Result<(), CgroupError> {
        if id == CgroupId::ROOT {
            return Err(CgroupError::PermissionDenied);
        }
        let (parent_id, path) = {
            let node = self.nodes.get(&id).ok_or(CgroupError::NotFound)?;
            if !node.children.is_empty() || !node.procs.is_empty() {
                return Err(CgroupError::Busy);
            }
            (node.parent, node.path.clone())
        };
        self.nodes.remove(&id);
        self.path_index.remove(&path);
        if let Some(pid) = parent_id {
            if let Some(parent) = self.nodes.get_mut(&pid) {
                parent.children.retain(|&c| c != id);
            }
        }
        Ok(())
    }

    /// Collect ancestor chain from `id` up to `CgroupId::ROOT` (inclusive).
    fn ancestor_chain(&self, id: CgroupId) -> Result<Vec<CgroupId>, CgroupError> {
        let mut chain = Vec::new();
        let mut cur = Some(id);
        while let Some(cid) = cur {
            let node = self.nodes.get(&cid).ok_or(CgroupError::NotFound)?;
            chain.push(cid);
            cur = node.parent;
        }
        Ok(chain)
    }

    /// Attach or migrate `host_pid` into `target_id`, enforcing `pids.max` along the hierarchy.
    pub fn attach_process(
        &mut self,
        target_id: CgroupId,
        host_pid: u64,
    ) -> Result<(), CgroupError> {
        let target_chain = self.ancestor_chain(target_id)?;
        let old_id = self.proc_cgroup.get(&host_pid).copied();
        if old_id == Some(target_id) {
            return Ok(());
        }

        // Check pids.max on target ancestors that don't already contain the process.
        let old_chain = match old_id {
            Some(oid) => self.ancestor_chain(oid)?,
            None => Vec::new(),
        };

        for &cid in &target_chain {
            if old_chain.contains(&cid) {
                continue;
            }
            let node = self.nodes.get_mut(&cid).ok_or(CgroupError::NotFound)?;
            if node.controllers.contains(ControllerMask::PIDS) && node.pids.current >= node.pids.max
            {
                node.pids.events_max += 1;
                return Err(CgroupError::LimitExceeded);
            }
        }

        // Detach from old cgroup if present.
        if let Some(oid) = old_id {
            if let Some(old_node) = self.nodes.get_mut(&oid) {
                old_node.procs.retain(|&p| p != host_pid);
            }
            for &cid in &old_chain {
                if !target_chain.contains(&cid) {
                    if let Some(node) = self.nodes.get_mut(&cid) {
                        node.pids.current = node.pids.current.saturating_sub(1);
                    }
                }
            }
        }

        // Attach to target cgroup and increment pids.current on new ancestors.
        if let Some(target_node) = self.nodes.get_mut(&target_id) {
            target_node.procs.push(host_pid);
        }
        for &cid in &target_chain {
            if !old_chain.contains(&cid) {
                if let Some(node) = self.nodes.get_mut(&cid) {
                    node.pids.current += 1;
                    if node.pids.current > node.pids.peak {
                        node.pids.peak = node.pids.current;
                    }
                }
            }
        }
        self.proc_cgroup.insert(host_pid, target_id);
        Ok(())
    }

    /// Detach a process on exit.
    pub fn detach_process(&mut self, host_pid: u64) {
        if let Some(cid) = self.proc_cgroup.remove(&host_pid) {
            if let Some(node) = self.nodes.get_mut(&cid) {
                node.procs.retain(|&p| p != host_pid);
            }
            if let Ok(chain) = self.ancestor_chain(cid) {
                for anc in chain {
                    if let Some(node) = self.nodes.get_mut(&anc) {
                        node.pids.current = node.pids.current.saturating_sub(1);
                    }
                }
            }
        }
    }

    /// Charge memory bytes hierarchically from `id` up to root.
    pub fn charge_memory(
        &mut self,
        id: CgroupId,
        bytes: u64,
    ) -> Result<MemoryChargeStatus, CgroupError> {
        let chain = self.ancestor_chain(id)?;
        let mut high_pressure = false;

        // First pass: validate `memory.max` across all ancestors.
        for &cid in &chain {
            let node = self.nodes.get_mut(&cid).ok_or(CgroupError::NotFound)?;
            if !node.controllers.contains(ControllerMask::MEMORY) {
                continue;
            }
            let next = node
                .memory
                .current_bytes
                .checked_add(bytes)
                .ok_or(CgroupError::OutOfMemory)?;
            if next > node.memory.max_bytes {
                node.memory.events.max += 1;
                node.memory.events.oom += 1;
                return Err(CgroupError::OutOfMemory);
            }
        }

        // Second pass: commit charge and check `memory.high`.
        for &cid in &chain {
            let node = self.nodes.get_mut(&cid).ok_or(CgroupError::NotFound)?;
            node.memory.current_bytes = node.memory.current_bytes.saturating_add(bytes);
            if node.memory.current_bytes > node.memory.peak_bytes {
                node.memory.peak_bytes = node.memory.current_bytes;
            }
            if node.controllers.contains(ControllerMask::MEMORY)
                && node.memory.current_bytes > node.memory.high_bytes
            {
                node.memory.events.high += 1;
                high_pressure = true;
            }
        }

        if high_pressure {
            Ok(MemoryChargeStatus::HighPressure)
        } else {
            Ok(MemoryChargeStatus::Ok)
        }
    }

    /// Uncharge memory bytes hierarchically from `id` up to root.
    pub fn uncharge_memory(&mut self, id: CgroupId, bytes: u64) -> Result<(), CgroupError> {
        let chain = self.ancestor_chain(id)?;
        for &cid in &chain {
            if let Some(node) = self.nodes.get_mut(&cid) {
                node.memory.current_bytes = node.memory.current_bytes.saturating_sub(bytes);
            }
        }
        Ok(())
    }

    /// Charge CPU time hierarchically from `id` up to root.
    pub fn charge_cpu(
        &mut self,
        id: CgroupId,
        user_us: u64,
        sys_us: u64,
    ) -> Result<(), CgroupError> {
        let chain = self.ancestor_chain(id)?;
        let mut throttled = false;
        for &cid in &chain {
            let node = self.nodes.get_mut(&cid).ok_or(CgroupError::NotFound)?;
            if node.controllers.contains(ControllerMask::CPU)
                && node.cpu.charge(user_us, sys_us).is_err()
            {
                throttled = true;
            }
        }
        if throttled {
            Err(CgroupError::Throttled)
        } else {
            Ok(())
        }
    }

    /// Charge I/O hierarchically from `id` up to root.
    pub fn charge_io(
        &mut self,
        id: CgroupId,
        dev: BlockDevId,
        dir: IoDir,
        bytes: u64,
    ) -> Result<(), CgroupError> {
        let chain = self.ancestor_chain(id)?;
        for &cid in &chain {
            let node = self.nodes.get_mut(&cid).ok_or(CgroupError::NotFound)?;
            if node.controllers.contains(ControllerMask::IO) {
                node.io.charge_io(dev, dir, bytes)?;
            }
        }
        Ok(())
    }
}

impl Default for CgroupManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Global Kernel State
// ---------------------------------------------------------------------------

/// Combined global state for namespaces and cgroups v2.
pub struct IsolationSubsystem {
    pub ns: NamespaceManager,
    pub cgroups: CgroupManager,
}

impl IsolationSubsystem {
    pub fn new() -> Self {
        Self {
            ns: NamespaceManager::new(),
            cgroups: CgroupManager::new(),
        }
    }
}

impl Default for IsolationSubsystem {
    fn default() -> Self {
        Self::new()
    }
}

pub static ISOLATION: TicketLock<Option<Box<IsolationSubsystem>>> = TicketLock::new(None);

/// Initialize the global namespace & cgroup v2 subsystem.
pub fn init() {
    let mut guard = ISOLATION.lock();
    if guard.is_none() {
        *guard = Some(Box::new(IsolationSubsystem::new()));
    }
}

// ---------------------------------------------------------------------------
// Self-Test Suite
// ---------------------------------------------------------------------------

/// Verify all 6 namespace types and all 4 cgroup v2 controllers.
pub fn self_test() -> Result<(), &'static str> {
    // Heap-allocate managers to keep stack frames well under 2 KiB.
    let mut ns_mgr = Box::new(NamespaceManager::new());
    let mut cg_mgr = Box::new(CgroupManager::new());

    // 1. Register host PID 100 in root namespaces.
    let root_proxy = ns_mgr
        .register_process(100)
        .map_err(|_| "register_process failed")?;

    // 2. Clone all 6 namespaces for container init (host PID 200).
    let child_proxy = ns_mgr
        .clone_ns(100, 200, CloneFlags::ALL)
        .map_err(|_| "clone_ns(ALL) failed")?;

    if child_proxy.pid_ns == root_proxy.pid_ns
        || child_proxy.mnt_ns == root_proxy.mnt_ns
        || child_proxy.net_ns == root_proxy.net_ns
        || child_proxy.uts_ns == root_proxy.uts_ns
        || child_proxy.ipc_ns == root_proxy.ipc_ns
        || child_proxy.user_ns == root_proxy.user_ns
    {
        return Err("clone_ns(ALL) did not isolate all 6 namespaces");
    }

    // 3. PID namespace translation: host PID 200 -> vpid 1.
    {
        let pid_ns = ns_mgr
            .pid_namespaces
            .get(&child_proxy.pid_ns)
            .ok_or("missing child pid_ns")?;
        if pid_ns.translate_from_host(200) != Some(1) || pid_ns.translate_to_host(1) != Some(200) {
            return Err("pid_ns translation mismatch");
        }
    }

    // 4. UTS namespace isolation.
    {
        let uts = ns_mgr
            .uts_namespaces
            .get_mut(&child_proxy.uts_ns)
            .ok_or("missing child uts_ns")?;
        uts.set_hostname("container-1")
            .map_err(|_| "set_hostname failed")?;
        let root_uts = ns_mgr
            .uts_namespaces
            .get(&root_proxy.uts_ns)
            .ok_or("missing root uts_ns")?;
        if root_uts.hostname == "container-1" {
            return Err("uts_ns leaked hostname to root");
        }
    }

    // 5. Mount namespace isolation.
    {
        let mnt = ns_mgr
            .mnt_namespaces
            .get_mut(&child_proxy.mnt_ns)
            .ok_or("missing child mnt_ns")?;
        mnt.mount("tmpfs", "/tmp", "tmpfs", MountFlags::NOSUID)
            .map_err(|_| "mnt_ns mount failed")?;
        if mnt.resolve_mount("/tmp/file").map(|m| m.fstype.as_str()) != Some("tmpfs") {
            return Err("mnt_ns resolve_mount failed");
        }
        let root_mnt = ns_mgr
            .mnt_namespaces
            .get(&root_proxy.mnt_ns)
            .ok_or("missing root mnt_ns")?;
        if root_mnt
            .resolve_mount("/tmp/file")
            .map(|m| m.fstype.as_str())
            == Some("tmpfs")
        {
            return Err("mnt_ns leaked mount to root");
        }
    }

    // 6. Net namespace isolation.
    {
        let net = ns_mgr
            .net_namespaces
            .get_mut(&child_proxy.net_ns)
            .ok_or("missing child net_ns")?;
        net.bind_tcp_port(8080)
            .map_err(|_| "net_ns bind_tcp_port failed")?;
        let root_net = ns_mgr
            .net_namespaces
            .get_mut(&root_proxy.net_ns)
            .ok_or("missing root net_ns")?;
        root_net
            .bind_tcp_port(8080)
            .map_err(|_| "net_ns port isolation failed")?;
    }

    // 7. IPC namespace isolation.
    {
        let ipc = ns_mgr
            .ipc_namespaces
            .get_mut(&child_proxy.ipc_ns)
            .ok_or("missing child ipc_ns")?;
        let shm_id = ipc.create_shm(4096).map_err(|_| "ipc create_shm failed")?;
        ipc.remove_shm(shm_id)
            .map_err(|_| "ipc remove_shm failed")?;
    }

    // 8. User namespace UID/GID mapping.
    {
        let user = ns_mgr
            .user_namespaces
            .get_mut(&child_proxy.user_ns)
            .ok_or("missing child user_ns")?;
        user.set_uid_map(alloc::vec![IdMapExtent {
            inside_id: 0,
            outside_id: 100_000,
            length: 65_536,
        }])
        .map_err(|_| "set_uid_map failed")?;
        if user.map_uid_to_outside(0) != Some(100_000)
            || user.map_uid_to_inside(100_042) != Some(42)
        {
            return Err("user_ns uid mapping mismatch");
        }
    }

    // 9. unshare & setns.
    let unshared = ns_mgr
        .unshare(100, CloneFlags::NEWUTS)
        .map_err(|_| "unshare failed")?;
    if unshared.uts_ns == root_proxy.uts_ns {
        return Err("unshare(NEWUTS) did not create new uts_ns");
    }
    let rejoined = ns_mgr
        .setns(100, NsKind::Uts, root_proxy.uts_ns)
        .map_err(|_| "setns failed")?;
    if rejoined.uts_ns != root_proxy.uts_ns {
        return Err("setns did not restore root uts_ns");
    }
    ns_mgr.exit_process(200);

    // 10. Cgroups v2 hierarchy & controllers.
    cg_mgr
        .set_subtree_control(CgroupId::ROOT, ControllerMask::ALL)
        .map_err(|_| "root subtree_control failed")?;
    let slice_id = cg_mgr
        .create_cgroup(CgroupId::ROOT, "system.slice")
        .map_err(|_| "create system.slice failed")?;
    cg_mgr
        .set_subtree_control(slice_id, ControllerMask::ALL)
        .map_err(|_| "slice subtree_control failed")?;
    let svc_id = cg_mgr
        .create_cgroup(slice_id, "worker.service")
        .map_err(|_| "create worker.service failed")?;

    // PIDs controller: max = 2.
    {
        let svc = cg_mgr.nodes.get_mut(&svc_id).ok_or("missing svc")?;
        svc.pids.max = 2;
        svc.cpu
            .set_weight(250)
            .map_err(|_| "cpu set_weight failed")?;
        svc.cpu
            .set_max(Some(20_000), 100_000)
            .map_err(|_| "cpu set_max failed")?;
        svc.memory.high_bytes = 8192;
        svc.memory.max_bytes = 16384;
        svc.io.set_limit(
            BlockDevId { major: 8, minor: 0 },
            IoLimit {
                rbps: 4096,
                wbps: 4096,
                riops: 2,
                wiops: 2,
            },
        );
    }

    cg_mgr
        .attach_process(svc_id, 301)
        .map_err(|_| "attach 301 failed")?;
    cg_mgr
        .attach_process(svc_id, 302)
        .map_err(|_| "attach 302 failed")?;
    if cg_mgr.attach_process(svc_id, 303) != Err(CgroupError::LimitExceeded) {
        return Err("pids.max did not reject 3rd process");
    }

    // CPU quota throttling & refill.
    cg_mgr
        .charge_cpu(svc_id, 15_000, 5_000)
        .map_err(|_| "cpu charge within quota failed")?;
    if cg_mgr.charge_cpu(svc_id, 1_000, 0) != Err(CgroupError::Throttled) {
        return Err("cpu.max did not throttle over-quota usage");
    }
    if let Some(svc) = cg_mgr.nodes.get_mut(&svc_id) {
        svc.cpu.refill_period();
    }

    // Memory high/max & hierarchical accounting.
    if cg_mgr.charge_memory(svc_id, 12_288) != Ok(MemoryChargeStatus::HighPressure) {
        return Err("memory.high did not signal HighPressure");
    }
    if cg_mgr.charge_memory(svc_id, 8_192) != Err(CgroupError::OutOfMemory) {
        return Err("memory.max did not return OutOfMemory");
    }
    let slice_mem = cg_mgr
        .nodes
        .get(&slice_id)
        .map(|n| n.memory.current_bytes)
        .unwrap_or(0);
    if slice_mem != 12_288 {
        return Err("hierarchical memory accounting mismatch");
    }
    cg_mgr
        .uncharge_memory(svc_id, 12_288)
        .map_err(|_| "uncharge_memory failed")?;

    // I/O throttling.
    let dev = BlockDevId { major: 8, minor: 0 };
    cg_mgr
        .charge_io(svc_id, dev, IoDir::Read, 4096)
        .map_err(|_| "io charge within rbps failed")?;
    if cg_mgr.charge_io(svc_id, dev, IoDir::Read, 1024) != Err(CgroupError::Throttled) {
        return Err("io.max rbps did not throttle");
    }

    // Cleanup.
    cg_mgr.detach_process(301);
    cg_mgr.detach_process(302);
    cg_mgr
        .remove_cgroup(svc_id)
        .map_err(|_| "remove svc failed")?;
    cg_mgr
        .remove_cgroup(slice_id)
        .map_err(|_| "remove slice failed")?;

    Ok(())
}
