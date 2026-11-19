//! `no_std` Production CoW B-Tree Extent Filesystem & ACL/Security Engine.
//!
//! Provides:
//! - [`ExtentAllocator`]: Physical block extent allocator with coalescing free-extent
//!   B-tree and per-extent reference counting for Copy-on-Write sharing.
//! - [`CowBTree`]: Copy-on-Write B-Tree with `Arc`-based structural sharing, O(log N)
//!   path-copying mutations, node splitting/merging, and invariant verification.
//! - Case-preserving / case-insensitive (and optional case-sensitive) directory lookup.
//! - CoW snapshots (`create_snapshot`, `delete_snapshot`, `read_snapshot_file`,
//!   `rollback_snapshot`) and O(1) reflink file cloning (`clone_file`).
//! - POSIX.1e ACLs ([`PosixAcl`]) with mask clamping and `check_access`.
//! - Windows NT Security Descriptors ([`NtSecurityDescriptor`]) with SIDs, DACL/SACL,
//!   ACE inheritance, and canonical `SeAccessCheck`.
//! - [`CowFs::fsck`] full-filesystem consistency verifier (B-tree invariants, directory
//!   connectivity, extent bounds & CRC32C checksums, refcount & free-space accounting).
//! - [`self_test`] exercising all subsystems.

#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::serial_println;

/// Filesystem block size in bytes (4 KiB).
pub const BLOCK_SIZE: usize = 4096;

/// Superblock magic ("NXCF").
pub const COW_FS_MAGIC: u32 = 0x4E58_4346;

/// On-disk / logical format version.
pub const COW_FS_VERSION: u32 = 1;

/// Maximum filename length in bytes.
pub const MAX_NAME_LEN: usize = 255;

/// Root directory inode number.
pub const ROOT_INODE: u64 = 1;

/// Reflected CRC32C (Castagnoli) polynomial.
const CRC32C_POLY: u32 = 0x82F6_3B78;

/// B-Tree branching factor parameter `B` (max keys = 2*B - 1 = 15, min keys = B - 1 = 7).
const BTREE_B: usize = 8;
const BTREE_MAX_KEYS: usize = 2 * BTREE_B - 1;
const BTREE_MIN_KEYS: usize = BTREE_B - 1;

/// Errors returned by the CoW filesystem and security engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CowFsError {
    InvalidArgument,
    NotFound,
    AlreadyExists,
    NotADirectory,
    IsADirectory,
    DirectoryNotEmpty,
    NoSpace,
    PermissionDenied,
    InvalidSnapshot,
    ReadOnlySnapshot,
    CorruptMetadata(&'static str),
    ChecksumMismatch,
    NameTooLong,
    IoError(&'static str),
    AclInvalid,
    SecurityDescriptorInvalid,
}

/// Compute CRC32C (Castagnoli) over a byte slice.
pub fn crc32c(data: &[u8]) -> u32 {
    crc32c_update(0, data)
}

/// Incremental CRC32C update.
pub fn crc32c_update(initial: u32, data: &[u8]) -> u32 {
    let mut crc = !initial;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (CRC32C_POLY & mask);
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// Block Store Abstraction
// ---------------------------------------------------------------------------

/// Backing block storage trait for 4 KiB blocks.
pub trait BlockStore: Send + Sync {
    fn total_blocks(&self) -> u64;
    fn read_block(&self, block_idx: u64, buf: &mut [u8]) -> Result<(), CowFsError>;
    fn write_block(&mut self, block_idx: u64, data: &[u8]) -> Result<(), CowFsError>;
}

/// Heap-backed in-memory 4 KiB block store (zero-initialized on demand).
pub struct InMemoryBlockStore {
    total_blocks: u64,
    blocks: BTreeMap<u64, Box<[u8; BLOCK_SIZE]>>,
}

impl InMemoryBlockStore {
    pub fn new(total_blocks: u64) -> Self {
        Self {
            total_blocks,
            blocks: BTreeMap::new(),
        }
    }

    /// Corrupt a block's first byte for testing checksum verification.
    pub fn corrupt_block_for_test(&mut self, block_idx: u64) {
        if let Some(blk) = self.blocks.get_mut(&block_idx) {
            blk[0] ^= 0xFF;
        }
    }
}

impl BlockStore for InMemoryBlockStore {
    fn total_blocks(&self) -> u64 {
        self.total_blocks
    }

    fn read_block(&self, block_idx: u64, buf: &mut [u8]) -> Result<(), CowFsError> {
        if block_idx >= self.total_blocks || buf.len() < BLOCK_SIZE {
            return Err(CowFsError::InvalidArgument);
        }
        if let Some(blk) = self.blocks.get(&block_idx) {
            buf[..BLOCK_SIZE].copy_from_slice(&blk[..]);
        } else {
            buf[..BLOCK_SIZE].fill(0);
        }
        Ok(())
    }

    fn write_block(&mut self, block_idx: u64, data: &[u8]) -> Result<(), CowFsError> {
        if block_idx >= self.total_blocks || data.len() < BLOCK_SIZE {
            return Err(CowFsError::InvalidArgument);
        }
        let entry = self
            .blocks
            .entry(block_idx)
            .or_insert_with(|| Box::new([0u8; BLOCK_SIZE]));
        entry[..].copy_from_slice(&data[..BLOCK_SIZE]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Extent Allocator with Coalescing & Reference Counting
// ---------------------------------------------------------------------------

/// A contiguous range of physical blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalExtent {
    pub start_block: u64,
    pub block_count: u64,
}

/// Coalescing extent allocator with per-block reference counting for CoW.
#[derive(Debug, Clone)]
pub struct ExtentAllocator {
    total_blocks: u64,
    /// Free extents keyed by `start_block -> block_count`.
    free_by_start: BTreeMap<u64, u64>,
    /// Reference count per allocated physical block (`block_idx -> refcount`).
    refcounts: BTreeMap<u64, u32>,
    free_block_count: u64,
}

impl ExtentAllocator {
    pub fn new(total_blocks: u64) -> Result<Self, CowFsError> {
        if total_blocks == 0 {
            return Err(CowFsError::InvalidArgument);
        }
        let mut free_by_start = BTreeMap::new();
        free_by_start.insert(0, total_blocks);
        Ok(Self {
            total_blocks,
            free_by_start,
            refcounts: BTreeMap::new(),
            free_block_count: total_blocks,
        })
    }

    pub fn total_blocks(&self) -> u64 {
        self.total_blocks
    }

    pub fn free_blocks(&self) -> u64 {
        self.free_block_count
    }

    pub fn used_blocks(&self) -> u64 {
        self.total_blocks.saturating_sub(self.free_block_count)
    }

    pub fn free_extent_count(&self) -> usize {
        self.free_by_start.len()
    }

    /// Allocate a contiguous extent of `count` blocks (first-fit) with refcount = 1.
    pub fn alloc_extent(&mut self, count: u64) -> Result<PhysicalExtent, CowFsError> {
        if count == 0 {
            return Err(CowFsError::InvalidArgument);
        }
        if count > self.free_block_count {
            return Err(CowFsError::NoSpace);
        }

        let mut found: Option<(u64, u64)> = None;
        for (&start, &len) in &self.free_by_start {
            if len >= count {
                found = Some((start, len));
                break;
            }
        }

        let (start, len) = found.ok_or(CowFsError::NoSpace)?;
        self.free_by_start.remove(&start);
        if len > count {
            self.free_by_start.insert(start + count, len - count);
        }
        self.free_block_count -= count;

        for blk in start..(start + count) {
            self.refcounts.insert(blk, 1);
        }

        Ok(PhysicalExtent {
            start_block: start,
            block_count: count,
        })
    }

    /// Increment reference count for all blocks in an extent.
    pub fn inc_ref(&mut self, extent: PhysicalExtent) -> Result<(), CowFsError> {
        if extent.start_block.saturating_add(extent.block_count) > self.total_blocks {
            return Err(CowFsError::InvalidArgument);
        }
        for blk in extent.start_block..(extent.start_block + extent.block_count) {
            let rc = self
                .refcounts
                .get_mut(&blk)
                .ok_or(CowFsError::CorruptMetadata("inc_ref on unallocated block"))?;
            *rc = rc
                .checked_add(1)
                .ok_or(CowFsError::CorruptMetadata("refcount overflow"))?;
        }
        Ok(())
    }

    /// Decrement reference count for all blocks in an extent; frees blocks whose refcount hits 0.
    pub fn dec_ref(&mut self, extent: PhysicalExtent) -> Result<(), CowFsError> {
        if extent.start_block.saturating_add(extent.block_count) > self.total_blocks {
            return Err(CowFsError::InvalidArgument);
        }
        let mut run_start: Option<u64> = None;
        let mut run_len: u64 = 0;

        for blk in extent.start_block..(extent.start_block + extent.block_count) {
            let rc = self
                .refcounts
                .get_mut(&blk)
                .ok_or(CowFsError::CorruptMetadata("dec_ref on unallocated block"))?;
            if *rc <= 1 {
                self.refcounts.remove(&blk);
                match run_start {
                    Some(s) if s + run_len == blk => {
                        run_len += 1;
                    }
                    Some(s) => {
                        self.insert_and_coalesce(s, run_len);
                        run_start = Some(blk);
                        run_len = 1;
                    }
                    None => {
                        run_start = Some(blk);
                        run_len = 1;
                    }
                }
            } else {
                *rc -= 1;
                if let Some(s) = run_start.take() {
                    self.insert_and_coalesce(s, run_len);
                    run_len = 0;
                }
            }
        }

        if let Some(s) = run_start {
            self.insert_and_coalesce(s, run_len);
        }
        Ok(())
    }

    pub fn refcount(&self, block_idx: u64) -> u32 {
        self.refcounts.get(&block_idx).copied().unwrap_or(0)
    }

    fn insert_and_coalesce(&mut self, mut start: u64, mut len: u64) {
        if len == 0 {
            return;
        }
        self.free_block_count += len;

        // Coalesce with predecessor if adjacent.
        if let Some((&prev_start, &prev_len)) = self.free_by_start.range(..start).next_back() {
            if prev_start + prev_len == start {
                self.free_by_start.remove(&prev_start);
                start = prev_start;
                len += prev_len;
            }
        }

        // Coalesce with successor if adjacent.
        let next_key = start + len;
        if let Some(&next_len) = self.free_by_start.get(&next_key) {
            self.free_by_start.remove(&next_key);
            len += next_len;
        }

        self.free_by_start.insert(start, len);
    }
}

// ---------------------------------------------------------------------------
// Copy-on-Write B-Tree (Arc-based structural sharing)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CowNode<K: Ord + Clone, V: Clone> {
    keys: Vec<K>,
    values: Vec<V>,
    children: Vec<Arc<CowNode<K, V>>>,
}

impl<K: Ord + Clone, V: Clone> CowNode<K, V> {
    fn new_leaf() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn is_full(&self) -> bool {
        self.keys.len() >= BTREE_MAX_KEYS
    }
}

/// Copy-on-Write B-Tree with O(1) snapshot cloning and O(log N) path-copying writes.
#[derive(Debug, Clone)]
pub struct CowBTree<K: Ord + Clone, V: Clone> {
    root: Option<Arc<CowNode<K, V>>>,
    len: usize,
}

impl<K: Ord + Clone, V: Clone> Default for CowBTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Clone, V: Clone> CowBTree<K, V> {
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterative lookup (zero recursion, tiny stack frame).
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut cur = self.root.as_deref()?;
        loop {
            match cur.keys.binary_search(key) {
                Ok(idx) => return Some(&cur.values[idx]),
                Err(idx) => {
                    if cur.is_leaf() {
                        return None;
                    }
                    cur = cur.children[idx].as_ref();
                }
            }
        }
    }

    /// CoW insert: clones only nodes along the modified root-to-leaf path.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.root.is_none() {
            let mut leaf = CowNode::new_leaf();
            leaf.keys.push(key);
            leaf.values.push(value);
            self.root = Some(Arc::new(leaf));
            self.len = 1;
            return None;
        }

        let root_arc = self.root.as_mut().unwrap();
        if root_arc.is_full() {
            let old_root = self.root.take().unwrap();
            let mut new_root = CowNode {
                keys: Vec::new(),
                values: Vec::new(),
                children: vec![old_root],
            };
            Self::split_child(&mut new_root, 0);
            self.root = Some(Arc::new(new_root));
        }

        let root_mut = Arc::make_mut(self.root.as_mut().unwrap());
        let old = Self::insert_non_full(root_mut, key, value);
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    fn split_child(parent: &mut CowNode<K, V>, child_idx: usize) {
        let child = Arc::make_mut(&mut parent.children[child_idx]);
        let mid = BTREE_B - 1;

        let right_keys = child.keys.split_off(mid + 1);
        let right_values = child.values.split_off(mid + 1);
        let right_children = if child.is_leaf() {
            Vec::new()
        } else {
            child.children.split_off(mid + 1)
        };

        let med_key = child.keys.pop().unwrap();
        let med_val = child.values.pop().unwrap();

        let right = Arc::new(CowNode {
            keys: right_keys,
            values: right_values,
            children: right_children,
        });

        parent.keys.insert(child_idx, med_key);
        parent.values.insert(child_idx, med_val);
        parent.children.insert(child_idx + 1, right);
    }

    fn insert_non_full(node: &mut CowNode<K, V>, key: K, value: V) -> Option<V> {
        match node.keys.binary_search(&key) {
            Ok(idx) => {
                let old = core::mem::replace(&mut node.values[idx], value);
                Some(old)
            }
            Err(mut idx) => {
                if node.is_leaf() {
                    node.keys.insert(idx, key);
                    node.values.insert(idx, value);
                    None
                } else {
                    if node.children[idx].is_full() {
                        Self::split_child(node, idx);
                        match key.cmp(&node.keys[idx]) {
                            Ordering::Equal => {
                                let old = core::mem::replace(&mut node.values[idx], value);
                                return Some(old);
                            }
                            Ordering::Greater => idx += 1,
                            Ordering::Less => {}
                        }
                    }
                    let child = Arc::make_mut(&mut node.children[idx]);
                    Self::insert_non_full(child, key, value)
                }
            }
        }
    }

    /// CoW remove: clones only nodes along the modified path and rebalances.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let root_arc = self.root.as_mut()?;
        let root_mut = Arc::make_mut(root_arc);
        let removed = Self::remove_from_node(root_mut, key);
        if removed.is_some() {
            self.len -= 1;
            if let Some(r) = self.root.as_ref() {
                if r.keys.is_empty() {
                    if r.is_leaf() {
                        self.root = None;
                    } else {
                        let only_child = r.children[0].clone();
                        self.root = Some(only_child);
                    }
                }
            }
        }
        removed
    }

    fn remove_from_node(node: &mut CowNode<K, V>, key: &K) -> Option<V> {
        match node.keys.binary_search(key) {
            Ok(idx) => {
                if node.is_leaf() {
                    node.keys.remove(idx);
                    Some(node.values.remove(idx))
                } else if node.children[idx].keys.len() >= BTREE_B {
                    let (pred_k, pred_v) = Self::pop_max(Arc::make_mut(&mut node.children[idx]));
                    node.keys[idx] = pred_k;
                    let old = core::mem::replace(&mut node.values[idx], pred_v);
                    Some(old)
                } else if node.children[idx + 1].keys.len() >= BTREE_B {
                    let (succ_k, succ_v) =
                        Self::pop_min(Arc::make_mut(&mut node.children[idx + 1]));
                    node.keys[idx] = succ_k;
                    let old = core::mem::replace(&mut node.values[idx], succ_v);
                    Some(old)
                } else {
                    Self::merge_children(node, idx);
                    let child = Arc::make_mut(&mut node.children[idx]);
                    Self::remove_from_node(child, key)
                }
            }
            Err(mut idx) => {
                if node.is_leaf() {
                    return None;
                }
                if node.children[idx].keys.len() < BTREE_B {
                    idx = Self::ensure_child_min(node, idx);
                }
                let child = Arc::make_mut(&mut node.children[idx]);
                Self::remove_from_node(child, key)
            }
        }
    }

    fn pop_max(node: &mut CowNode<K, V>) -> (K, V) {
        if node.is_leaf() {
            (node.keys.pop().unwrap(), node.values.pop().unwrap())
        } else {
            let last = node.children.len() - 1;
            let idx = if node.children[last].keys.len() < BTREE_B {
                Self::ensure_child_min(node, last)
            } else {
                last
            };
            let child = Arc::make_mut(&mut node.children[idx]);
            Self::pop_max(child)
        }
    }

    fn pop_min(node: &mut CowNode<K, V>) -> (K, V) {
        if node.is_leaf() {
            (node.keys.remove(0), node.values.remove(0))
        } else {
            let idx = if node.children[0].keys.len() < BTREE_B {
                Self::ensure_child_min(node, 0)
            } else {
                0
            };
            let child = Arc::make_mut(&mut node.children[idx]);
            Self::pop_min(child)
        }
    }

    fn ensure_child_min(parent: &mut CowNode<K, V>, idx: usize) -> usize {
        if idx > 0 && parent.children[idx - 1].keys.len() >= BTREE_B {
            Self::borrow_from_prev(parent, idx);
            idx
        } else if idx + 1 < parent.children.len() && parent.children[idx + 1].keys.len() >= BTREE_B
        {
            Self::borrow_from_next(parent, idx);
            idx
        } else if idx + 1 < parent.children.len() {
            Self::merge_children(parent, idx);
            idx
        } else {
            Self::merge_children(parent, idx - 1);
            idx - 1
        }
    }

    fn borrow_from_prev(parent: &mut CowNode<K, V>, idx: usize) {
        let prev = Arc::make_mut(&mut parent.children[idx - 1]);
        let prev_k = prev.keys.pop().unwrap();
        let prev_v = prev.values.pop().unwrap();
        let prev_c = if prev.is_leaf() {
            None
        } else {
            Some(prev.children.pop().unwrap())
        };

        let p_k = core::mem::replace(&mut parent.keys[idx - 1], prev_k);
        let p_v = core::mem::replace(&mut parent.values[idx - 1], prev_v);

        let child = Arc::make_mut(&mut parent.children[idx]);
        child.keys.insert(0, p_k);
        child.values.insert(0, p_v);
        if let Some(c) = prev_c {
            child.children.insert(0, c);
        }
    }

    fn borrow_from_next(parent: &mut CowNode<K, V>, idx: usize) {
        let next = Arc::make_mut(&mut parent.children[idx + 1]);
        let next_k = next.keys.remove(0);
        let next_v = next.values.remove(0);
        let next_c = if next.is_leaf() {
            None
        } else {
            Some(next.children.remove(0))
        };

        let p_k = core::mem::replace(&mut parent.keys[idx], next_k);
        let p_v = core::mem::replace(&mut parent.values[idx], next_v);

        let child = Arc::make_mut(&mut parent.children[idx]);
        child.keys.push(p_k);
        child.values.push(p_v);
        if let Some(c) = next_c {
            child.children.push(c);
        }
    }

    fn merge_children(parent: &mut CowNode<K, V>, idx: usize) {
        let med_k = parent.keys.remove(idx);
        let med_v = parent.values.remove(idx);
        let right_arc = parent.children.remove(idx + 1);
        let right = (*right_arc).clone();

        let left = Arc::make_mut(&mut parent.children[idx]);
        left.keys.push(med_k);
        left.values.push(med_v);
        left.keys.extend(right.keys);
        left.values.extend(right.values);
        left.children.extend(right.children);
    }

    /// Visit all `(K, V)` pairs in sorted order using an explicit heap stack (no recursion).
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(&K, &V),
    {
        let Some(root) = self.root.as_deref() else {
            return;
        };
        let mut stack: Vec<(&CowNode<K, V>, usize)> = Vec::new();
        stack.push((root, 0));

        while let Some((node, idx)) = stack.pop() {
            if node.is_leaf() {
                for i in 0..node.keys.len() {
                    f(&node.keys[i], &node.values[i]);
                }
            } else if idx < node.keys.len() {
                stack.push((node, idx + 1));
                if idx > 0 {
                    f(&node.keys[idx - 1], &node.values[idx - 1]);
                }
                stack.push((node.children[idx].as_ref(), 0));
            } else {
                if !node.keys.is_empty() {
                    f(
                        &node.keys[node.keys.len() - 1],
                        &node.values[node.values.len() - 1],
                    );
                }
                stack.push((node.children[node.keys.len()].as_ref(), 0));
            }
        }
    }

    /// Collect all entries into a `Vec<(K, V)>`.
    pub fn entries(&self) -> Vec<(K, V)> {
        let mut out = Vec::with_capacity(self.len);
        self.for_each(|k, v| out.push((k.clone(), v.clone())));
        out
    }

    /// Verify B-tree structural invariants (sorted keys, occupancy, balanced depth, count).
    pub fn verify_invariants(&self) -> Result<usize, CowFsError> {
        let Some(root) = self.root.as_deref() else {
            if self.len != 0 {
                return Err(CowFsError::CorruptMetadata("empty root with non-zero len"));
            }
            return Ok(0);
        };

        let mut count = 0usize;
        let mut leaf_depth: Option<usize> = None;
        let mut stack: Vec<(&CowNode<K, V>, usize, bool)> = Vec::new();
        stack.push((root, 0, true));

        while let Some((node, depth, is_root)) = stack.pop() {
            if node.keys.len() != node.values.len() {
                return Err(CowFsError::CorruptMetadata("keys/values length mismatch"));
            }
            if !is_root && node.keys.len() < BTREE_MIN_KEYS {
                return Err(CowFsError::CorruptMetadata("btree node underflow"));
            }
            if node.keys.len() > BTREE_MAX_KEYS {
                return Err(CowFsError::CorruptMetadata("btree node overflow"));
            }
            for i in 1..node.keys.len() {
                if node.keys[i - 1] >= node.keys[i] {
                    return Err(CowFsError::CorruptMetadata(
                        "btree keys not strictly sorted",
                    ));
                }
            }
            count += node.keys.len();

            if node.is_leaf() {
                match leaf_depth {
                    Some(d) if d != depth => {
                        return Err(CowFsError::CorruptMetadata("unbalanced btree leaf depth"));
                    }
                    None => leaf_depth = Some(depth),
                    _ => {}
                }
            } else {
                if node.children.len() != node.keys.len() + 1 {
                    return Err(CowFsError::CorruptMetadata("btree children count mismatch"));
                }
                for child in &node.children {
                    stack.push((child.as_ref(), depth + 1, false));
                }
            }
        }

        if count != self.len {
            return Err(CowFsError::CorruptMetadata("btree len mismatch"));
        }
        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// POSIX.1e ACLs
// ---------------------------------------------------------------------------

/// POSIX ACL permission bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AclPerm(pub u8);

impl AclPerm {
    pub const NONE: Self = Self(0);
    pub const EXEC: Self = Self(1);
    pub const WRITE: Self = Self(2);
    pub const READ: Self = Self(4);
    pub const RW: Self = Self(6);
    pub const RX: Self = Self(5);
    pub const RWX: Self = Self(7);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// POSIX ACL entry tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AclTag {
    UserObj,
    User(u32),
    GroupObj,
    Group(u32),
    Mask,
    Other,
}

/// Single POSIX ACL entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AclEntry {
    pub tag: AclTag,
    pub perm: AclPerm,
}

/// POSIX.1e Access Control List.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PosixAcl {
    pub entries: Vec<AclEntry>,
    pub default_entries: Vec<AclEntry>,
}

impl PosixAcl {
    /// Construct a minimal POSIX ACL from standard Unix mode bits (`rwxrwxrwx`).
    pub fn from_mode(mode: u16) -> Self {
        let u = AclPerm(((mode >> 6) & 0x7) as u8);
        let g = AclPerm(((mode >> 3) & 0x7) as u8);
        let o = AclPerm((mode & 0x7) as u8);
        Self {
            entries: vec![
                AclEntry {
                    tag: AclTag::UserObj,
                    perm: u,
                },
                AclEntry {
                    tag: AclTag::GroupObj,
                    perm: g,
                },
                AclEntry {
                    tag: AclTag::Other,
                    perm: o,
                },
            ],
            default_entries: Vec::new(),
        }
    }

    /// Validate POSIX.1e ACL structural rules.
    pub fn validate(&self) -> Result<(), CowFsError> {
        Self::validate_list(&self.entries)?;
        if !self.default_entries.is_empty() {
            Self::validate_list(&self.default_entries)?;
        }
        Ok(())
    }

    fn validate_list(list: &[AclEntry]) -> Result<(), CowFsError> {
        let mut has_user_obj = false;
        let mut has_group_obj = false;
        let mut has_other = false;
        let mut has_mask = false;
        let mut has_named = false;

        for (i, e) in list.iter().enumerate() {
            if e.perm.0 > 7 {
                return Err(CowFsError::AclInvalid);
            }
            for other in &list[i + 1..] {
                if e.tag == other.tag {
                    return Err(CowFsError::AclInvalid);
                }
            }
            match e.tag {
                AclTag::UserObj => has_user_obj = true,
                AclTag::GroupObj => has_group_obj = true,
                AclTag::Other => has_other = true,
                AclTag::Mask => has_mask = true,
                AclTag::User(_) | AclTag::Group(_) => has_named = true,
            }
        }

        if !has_user_obj || !has_group_obj || !has_other {
            return Err(CowFsError::AclInvalid);
        }
        if has_named && !has_mask {
            return Err(CowFsError::AclInvalid);
        }
        Ok(())
    }

    /// Evaluate POSIX.1e access check for `(uid, gids)` against `owner_uid`/`owner_gid`.
    pub fn check_access(
        &self,
        uid: u32,
        gids: &[u32],
        requested: AclPerm,
        owner_uid: u32,
        owner_gid: u32,
    ) -> bool {
        if requested.0 == 0 {
            return true;
        }
        // Root bypasses read/write; exec requires at least one exec bit in ACL.
        if uid == 0 {
            if !requested.contains(AclPerm::EXEC) {
                return true;
            }
            let any_exec = self.entries.iter().any(|e| e.perm.contains(AclPerm::EXEC));
            return any_exec;
        }

        let mask = self
            .entries
            .iter()
            .find(|e| e.tag == AclTag::Mask)
            .map(|e| e.perm);

        // 1. Owner user
        if uid == owner_uid {
            if let Some(e) = self.entries.iter().find(|e| e.tag == AclTag::UserObj) {
                return e.perm.contains(requested);
            }
            return false;
        }

        // 2. Named user
        if let Some(e) = self.entries.iter().find(|e| e.tag == AclTag::User(uid)) {
            let eff = match mask {
                Some(m) => e.perm.intersect(m),
                None => e.perm,
            };
            return eff.contains(requested);
        }

        // 3. Owning group or named groups (union of matching entries, masked)
        let mut matched_group = false;
        let mut group_union = AclPerm::NONE;
        for e in &self.entries {
            match e.tag {
                AclTag::GroupObj if gids.contains(&owner_gid) => {
                    matched_group = true;
                    group_union = group_union.union(e.perm);
                }
                AclTag::Group(gid) if gids.contains(&gid) => {
                    matched_group = true;
                    group_union = group_union.union(e.perm);
                }
                _ => {}
            }
        }
        if matched_group {
            let eff = match mask {
                Some(m) => group_union.intersect(m),
                None => group_union,
            };
            return eff.contains(requested);
        }

        // 4. Other
        if let Some(e) = self.entries.iter().find(|e| e.tag == AclTag::Other) {
            return e.perm.contains(requested);
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Windows NT Security Descriptors (SID, ACE, ACL, SeAccessCheck)
// ---------------------------------------------------------------------------

/// Windows NT Security Identifier (SID).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sid {
    pub revision: u8,
    pub authority: [u8; 6],
    pub sub_authorities: Vec<u32>,
}

impl Sid {
    pub fn everyone() -> Self {
        Self {
            revision: 1,
            authority: [0, 0, 0, 0, 0, 1], // SECURITY_WORLD_SID_AUTHORITY
            sub_authorities: vec![0],
        }
    }

    pub fn system() -> Self {
        Self {
            revision: 1,
            authority: [0, 0, 0, 0, 0, 5], // SECURITY_NT_AUTHORITY
            sub_authorities: vec![18],
        }
    }

    pub fn administrators() -> Self {
        Self {
            revision: 1,
            authority: [0, 0, 0, 0, 0, 5],
            sub_authorities: vec![32, 544],
        }
    }

    pub fn from_uid(uid: u32) -> Self {
        Self {
            revision: 1,
            authority: [0, 0, 0, 0, 0, 5],
            sub_authorities: vec![21, 1000, uid],
        }
    }

    pub fn from_gid(gid: u32) -> Self {
        Self {
            revision: 1,
            authority: [0, 0, 0, 0, 0, 5],
            sub_authorities: vec![21, 2000, gid],
        }
    }
}

/// NT Access Mask constants.
pub mod nt_access {
    pub const FILE_READ_DATA: u32 = 0x0000_0001;
    pub const FILE_WRITE_DATA: u32 = 0x0000_0002;
    pub const FILE_APPEND_DATA: u32 = 0x0000_0004;
    pub const FILE_EXECUTE: u32 = 0x0000_0020;
    pub const DELETE: u32 = 0x0001_0000;
    pub const READ_CONTROL: u32 = 0x0002_0000;
    pub const WRITE_DAC: u32 = 0x0004_0000;
    pub const WRITE_OWNER: u32 = 0x0008_0000;
    pub const GENERIC_ALL: u32 = 0x1000_0000;
    pub const GENERIC_EXECUTE: u32 = 0x2000_0000;
    pub const GENERIC_WRITE: u32 = 0x4000_0000;
    pub const GENERIC_READ: u32 = 0x8000_0000;

    pub const FILE_GENERIC_READ: u32 = FILE_READ_DATA | READ_CONTROL;
    pub const FILE_GENERIC_WRITE: u32 = FILE_WRITE_DATA | FILE_APPEND_DATA | READ_CONTROL;
    pub const FILE_GENERIC_EXECUTE: u32 = FILE_EXECUTE | READ_CONTROL;
    pub const FILE_ALL_ACCESS: u32 = FILE_READ_DATA
        | FILE_WRITE_DATA
        | FILE_APPEND_DATA
        | FILE_EXECUTE
        | DELETE
        | READ_CONTROL
        | WRITE_DAC
        | WRITE_OWNER;

    /// Map generic access bits to file-specific access bits.
    pub fn map_generic(mask: u32) -> u32 {
        let mut out = mask & !(GENERIC_READ | GENERIC_WRITE | GENERIC_EXECUTE | GENERIC_ALL);
        if (mask & GENERIC_READ) != 0 {
            out |= FILE_GENERIC_READ;
        }
        if (mask & GENERIC_WRITE) != 0 {
            out |= FILE_GENERIC_WRITE;
        }
        if (mask & GENERIC_EXECUTE) != 0 {
            out |= FILE_GENERIC_EXECUTE;
        }
        if (mask & GENERIC_ALL) != 0 {
            out |= FILE_ALL_ACCESS;
        }
        out
    }
}

/// NT ACE type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AceType {
    AccessAllowed,
    AccessDenied,
    SystemAudit,
}

/// NT ACE inheritance flags.
pub mod ace_flags {
    pub const OBJECT_INHERIT: u8 = 0x01;
    pub const CONTAINER_INHERIT: u8 = 0x02;
    pub const NO_PROPAGATE_INHERIT: u8 = 0x04;
    pub const INHERIT_ONLY: u8 = 0x08;
    pub const INHERITED: u8 = 0x10;
}

/// NT Access Control Entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ace {
    pub ace_type: AceType,
    pub flags: u8,
    pub mask: u32,
    pub sid: Sid,
}

/// NT Access Control List (DACL or SACL).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NtAcl {
    pub revision: u8,
    pub aces: Vec<Ace>,
}

/// NT Security Token representing a caller's identity and groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityToken {
    pub user_sid: Sid,
    pub group_sids: Vec<Sid>,
    pub privileges: u32,
}

impl SecurityToken {
    pub fn from_unix(uid: u32, gids: &[u32]) -> Self {
        let user_sid = if uid == 0 {
            Sid::system()
        } else {
            Sid::from_uid(uid)
        };
        let mut group_sids = Vec::with_capacity(gids.len() + 1);
        for &g in gids {
            group_sids.push(Sid::from_gid(g));
        }
        if uid == 0 {
            group_sids.push(Sid::administrators());
        }
        group_sids.push(Sid::everyone());
        Self {
            user_sid,
            group_sids,
            privileges: 0,
        }
    }

    pub fn matches_sid(&self, sid: &Sid) -> bool {
        &self.user_sid == sid || self.group_sids.iter().any(|g| g == sid)
    }
}

/// Windows NT Security Descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NtSecurityDescriptor {
    pub owner: Sid,
    pub group: Sid,
    pub control: u16,
    pub dacl: Option<NtAcl>,
    pub sacl: Option<NtAcl>,
}

impl NtSecurityDescriptor {
    pub fn default_for_unix(uid: u32, gid: u32, mode: u16) -> Self {
        let owner = Sid::from_uid(uid);
        let group = Sid::from_gid(gid);
        let mut aces = Vec::new();

        let u_mask = Self::unix_bits_to_nt(((mode >> 6) & 0x7) as u8);
        if u_mask != 0 {
            aces.push(Ace {
                ace_type: AceType::AccessAllowed,
                flags: 0,
                mask: u_mask | nt_access::READ_CONTROL | nt_access::WRITE_DAC,
                sid: owner.clone(),
            });
        }
        let g_mask = Self::unix_bits_to_nt(((mode >> 3) & 0x7) as u8);
        if g_mask != 0 {
            aces.push(Ace {
                ace_type: AceType::AccessAllowed,
                flags: 0,
                mask: g_mask,
                sid: group.clone(),
            });
        }
        let o_mask = Self::unix_bits_to_nt((mode & 0x7) as u8);
        if o_mask != 0 {
            aces.push(Ace {
                ace_type: AceType::AccessAllowed,
                flags: 0,
                mask: o_mask,
                sid: Sid::everyone(),
            });
        }

        Self {
            owner,
            group,
            control: 0x0004, // SE_DACL_PRESENT
            dacl: Some(NtAcl { revision: 2, aces }),
            sacl: None,
        }
    }

    fn unix_bits_to_nt(bits: u8) -> u32 {
        let mut m = 0u32;
        if (bits & 4) != 0 {
            m |= nt_access::FILE_GENERIC_READ;
        }
        if (bits & 2) != 0 {
            m |= nt_access::FILE_GENERIC_WRITE | nt_access::DELETE;
        }
        if (bits & 1) != 0 {
            m |= nt_access::FILE_GENERIC_EXECUTE;
        }
        m
    }

    /// Canonical Windows NT `SeAccessCheck` evaluation.
    pub fn check_access(&self, token: &SecurityToken, desired_access: u32) -> bool {
        let desired = nt_access::map_generic(desired_access);
        if desired == 0 {
            return true;
        }
        // SYSTEM always has full access.
        if token.user_sid == Sid::system() {
            return true;
        }

        let Some(dacl) = &self.dacl else {
            // NULL DACL grants all access.
            return true;
        };
        if dacl.aces.is_empty() {
            // Empty DACL denies all access.
            return false;
        }

        let mut remaining = desired;
        let mut granted = 0u32;

        for ace in &dacl.aces {
            if (ace.flags & ace_flags::INHERIT_ONLY) != 0 {
                continue;
            }
            if !token.matches_sid(&ace.sid) {
                continue;
            }
            let ace_mask = nt_access::map_generic(ace.mask);
            match ace.ace_type {
                AceType::AccessDenied => {
                    if (remaining & ace_mask) != 0 {
                        return false;
                    }
                }
                AceType::AccessAllowed => {
                    granted |= ace_mask;
                    remaining &= !ace_mask;
                    if remaining == 0 {
                        return true;
                    }
                }
                AceType::SystemAudit => {}
            }
        }

        (granted & desired) == desired
    }

    /// Compute inherited Security Descriptor for a newly created child file or directory.
    pub fn inherit_for_child(&self, child_owner: Sid, child_group: Sid, is_dir: bool) -> Self {
        let mut child_aces = Vec::new();
        if let Some(parent_dacl) = &self.dacl {
            for ace in &parent_dacl.aces {
                let applies = if is_dir {
                    (ace.flags & ace_flags::CONTAINER_INHERIT) != 0
                } else {
                    (ace.flags & ace_flags::OBJECT_INHERIT) != 0
                };
                if !applies {
                    continue;
                }
                let mut new_flags = ace.flags | ace_flags::INHERITED;
                new_flags &= !ace_flags::INHERIT_ONLY;
                if !is_dir || (ace.flags & ace_flags::NO_PROPAGATE_INHERIT) != 0 {
                    new_flags &= !(ace_flags::OBJECT_INHERIT
                        | ace_flags::CONTAINER_INHERIT
                        | ace_flags::NO_PROPAGATE_INHERIT);
                }
                child_aces.push(Ace {
                    ace_type: ace.ace_type,
                    flags: new_flags,
                    mask: ace.mask,
                    sid: ace.sid.clone(),
                });
            }
        }

        Self {
            owner: child_owner,
            group: child_group,
            control: self.control,
            dacl: Some(NtAcl {
                revision: 2,
                aces: child_aces,
            }),
            sacl: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Filesystem Metadata Types & Case-Folding
// ---------------------------------------------------------------------------

/// Directory name matching policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseMode {
    /// Case-insensitive lookup, case-preserving storage (NTFS/APFS style).
    CaseInsensitivePreserving,
    /// Exact byte-for-byte case-sensitive lookup (POSIX style).
    CaseSensitive,
}

fn fold_name(name: &str, mode: CaseMode) -> String {
    match mode {
        CaseMode::CaseSensitive => name.to_string(),
        CaseMode::CaseInsensitivePreserving => {
            let mut out = String::with_capacity(name.len());
            for ch in name.chars() {
                for lc in ch.to_lowercase() {
                    out.push(lc);
                }
            }
            out
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InodeKind {
    File,
    Directory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inode {
    pub id: u64,
    pub kind: InodeKind,
    pub size: u64,
    pub uid: u32,
    pub gid: u32,
    pub mode: u16,
    pub nlink: u32,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InodeStat {
    pub id: u64,
    pub kind: InodeKind,
    pub size: u64,
    pub uid: u32,
    pub gid: u32,
    pub mode: u16,
    pub nlink: u32,
    pub generation: u64,
}

impl From<&Inode> for InodeStat {
    fn from(ino: &Inode) -> Self {
        Self {
            id: ino.id,
            kind: ino.kind,
            size: ino.size,
            uid: ino.uid,
            gid: ino.gid,
            mode: ino.mode,
            nlink: ino.nlink,
            generation: ino.generation,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DirKey {
    pub parent_inode: u64,
    pub folded_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    pub child_inode: u64,
    pub original_name: String,
    pub kind: InodeKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntryInfo {
    pub inode: u64,
    pub name: String,
    pub kind: InodeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExtentKey {
    pub inode_id: u64,
    pub logical_block: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileExtent {
    pub physical_block: u64,
    pub block_count: u64,
    pub checksum: u32,
}

/// Complete point-in-time CoW filesystem tree root.
#[derive(Debug, Clone)]
pub struct FsRoot {
    pub inodes: CowBTree<u64, Inode>,
    pub dirents: CowBTree<DirKey, DirEntry>,
    pub extents: CowBTree<ExtentKey, FileExtent>,
    pub acls: CowBTree<u64, PosixAcl>,
    pub sec_descs: CowBTree<u64, NtSecurityDescriptor>,
}

impl FsRoot {
    fn new_empty() -> Self {
        let mut inodes = CowBTree::new();
        let mut acls = CowBTree::new();
        let mut sec_descs = CowBTree::new();

        let root_inode = Inode {
            id: ROOT_INODE,
            kind: InodeKind::Directory,
            size: 0,
            uid: 0,
            gid: 0,
            mode: 0o755,
            nlink: 2,
            generation: 1,
        };
        inodes.insert(ROOT_INODE, root_inode);
        acls.insert(ROOT_INODE, PosixAcl::from_mode(0o755));
        sec_descs.insert(
            ROOT_INODE,
            NtSecurityDescriptor::default_for_unix(0, 0, 0o755),
        );

        Self {
            inodes,
            dirents: CowBTree::new(),
            extents: CowBTree::new(),
            acls,
            sec_descs,
        }
    }
}

/// Snapshot metadata + CoW root.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub id: u64,
    pub name: String,
    pub generation: u64,
    pub read_only: bool,
    pub root: FsRoot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotInfo {
    pub id: u64,
    pub name: String,
    pub generation: u64,
    pub read_only: bool,
}

/// Report produced by [`CowFs::fsck`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsckReport {
    pub inodes_checked: usize,
    pub dirents_checked: usize,
    pub extents_checked: usize,
    pub snapshots_checked: usize,
    pub blocks_referenced: u64,
    pub blocks_free: u64,
}

// ---------------------------------------------------------------------------
// CowFs Engine
// ---------------------------------------------------------------------------

/// Production `no_std` CoW B-Tree Extent Filesystem & ACL/Security Engine.
pub struct CowFs<S: BlockStore> {
    store: S,
    allocator: ExtentAllocator,
    case_mode: CaseMode,
    generation: u64,
    next_inode: u64,
    next_snapshot_id: u64,
    live: FsRoot,
    snapshots: BTreeMap<String, Snapshot>,
}

impl<S: BlockStore> CowFs<S> {
    pub fn new(store: S, case_mode: CaseMode) -> Result<Self, CowFsError> {
        let total = store.total_blocks();
        let allocator = ExtentAllocator::new(total)?;
        Ok(Self {
            store,
            allocator,
            case_mode,
            generation: 1,
            next_inode: ROOT_INODE + 1,
            next_snapshot_id: 1,
            live: FsRoot::new_empty(),
            snapshots: BTreeMap::new(),
        })
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    pub fn allocator(&self) -> &ExtentAllocator {
        &self.allocator
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn split_path(path: &str) -> Result<Vec<&str>, CowFsError> {
        if path.is_empty() || !path.starts_with('/') {
            return Err(CowFsError::InvalidArgument);
        }
        let mut parts = Vec::new();
        for seg in path.split('/') {
            if seg.is_empty() || seg == "." {
                continue;
            }
            if seg == ".." {
                return Err(CowFsError::InvalidArgument);
            }
            if seg.len() > MAX_NAME_LEN {
                return Err(CowFsError::NameTooLong);
            }
            parts.push(seg);
        }
        Ok(parts)
    }

    fn resolve_in_root(&self, root: &FsRoot, path: &str) -> Result<u64, CowFsError> {
        let parts = Self::split_path(path)?;
        let mut cur = ROOT_INODE;
        for seg in parts {
            let ino = root.inodes.get(&cur).ok_or(CowFsError::NotFound)?;
            if ino.kind != InodeKind::Directory {
                return Err(CowFsError::NotADirectory);
            }
            let key = DirKey {
                parent_inode: cur,
                folded_name: fold_name(seg, self.case_mode),
            };
            let de = root.dirents.get(&key).ok_or(CowFsError::NotFound)?;
            cur = de.child_inode;
        }
        Ok(cur)
    }

    fn resolve_parent(&self, path: &str) -> Result<(u64, String), CowFsError> {
        let parts = Self::split_path(path)?;
        if parts.is_empty() {
            return Err(CowFsError::InvalidArgument);
        }
        let leaf = parts[parts.len() - 1].to_string();
        let mut cur = ROOT_INODE;
        for &seg in &parts[..parts.len() - 1] {
            let ino = self.live.inodes.get(&cur).ok_or(CowFsError::NotFound)?;
            if ino.kind != InodeKind::Directory {
                return Err(CowFsError::NotADirectory);
            }
            let key = DirKey {
                parent_inode: cur,
                folded_name: fold_name(seg, self.case_mode),
            };
            let de = self.live.dirents.get(&key).ok_or(CowFsError::NotFound)?;
            cur = de.child_inode;
        }
        let parent_ino = self.live.inodes.get(&cur).ok_or(CowFsError::NotFound)?;
        if parent_ino.kind != InodeKind::Directory {
            return Err(CowFsError::NotADirectory);
        }
        Ok((cur, leaf))
    }

    fn check_inode_posix(
        root: &FsRoot,
        inode_id: u64,
        uid: u32,
        gids: &[u32],
        perm: AclPerm,
    ) -> Result<(), CowFsError> {
        let ino = root.inodes.get(&inode_id).ok_or(CowFsError::NotFound)?;
        let acl = root
            .acls
            .get(&inode_id)
            .ok_or(CowFsError::CorruptMetadata("missing inode acl"))?;
        if acl.check_access(uid, gids, perm, ino.uid, ino.gid) {
            Ok(())
        } else {
            Err(CowFsError::PermissionDenied)
        }
    }

    /// Create a regular file at `path`.
    pub fn create_file(
        &mut self,
        path: &str,
        uid: u32,
        gid: u32,
        mode: u16,
    ) -> Result<u64, CowFsError> {
        self.create_node(path, InodeKind::File, uid, gid, mode)
    }

    /// Create a directory at `path`.
    pub fn mkdir(&mut self, path: &str, uid: u32, gid: u32, mode: u16) -> Result<u64, CowFsError> {
        self.create_node(path, InodeKind::Directory, uid, gid, mode)
    }

    fn create_node(
        &mut self,
        path: &str,
        kind: InodeKind,
        uid: u32,
        gid: u32,
        mode: u16,
    ) -> Result<u64, CowFsError> {
        let (parent_id, name) = self.resolve_parent(path)?;
        Self::check_inode_posix(&self.live, parent_id, uid, &[gid], AclPerm::WRITE)?;

        let key = DirKey {
            parent_inode: parent_id,
            folded_name: fold_name(&name, self.case_mode),
        };
        if self.live.dirents.get(&key).is_some() {
            return Err(CowFsError::AlreadyExists);
        }

        self.generation += 1;
        let id = self.next_inode;
        self.next_inode += 1;

        let inode = Inode {
            id,
            kind,
            size: 0,
            uid,
            gid,
            mode: mode & 0o777,
            nlink: if kind == InodeKind::Directory { 2 } else { 1 },
            generation: self.generation,
        };

        let parent_sd = self.live.sec_descs.get(&parent_id).cloned();
        let child_sd = if let Some(psd) = parent_sd {
            let inherited = psd.inherit_for_child(
                Sid::from_uid(uid),
                Sid::from_gid(gid),
                kind == InodeKind::Directory,
            );
            if inherited
                .dacl
                .as_ref()
                .map(|d| d.aces.is_empty())
                .unwrap_or(true)
            {
                NtSecurityDescriptor::default_for_unix(uid, gid, mode)
            } else {
                inherited
            }
        } else {
            NtSecurityDescriptor::default_for_unix(uid, gid, mode)
        };

        self.live.inodes.insert(id, inode);
        self.live.acls.insert(id, PosixAcl::from_mode(mode));
        self.live.sec_descs.insert(id, child_sd);
        self.live.dirents.insert(
            key,
            DirEntry {
                child_inode: id,
                original_name: name,
                kind,
            },
        );

        if kind == InodeKind::Directory {
            if let Some(mut p) = self.live.inodes.get(&parent_id).cloned() {
                p.nlink += 1;
                self.live.inodes.insert(parent_id, p);
            }
        }

        Ok(id)
    }

    /// Stat a file or directory.
    pub fn stat(&self, path: &str) -> Result<InodeStat, CowFsError> {
        let id = self.resolve_in_root(&self.live, path)?;
        let ino = self.live.inodes.get(&id).ok_or(CowFsError::NotFound)?;
        Ok(InodeStat::from(ino))
    }

    /// List directory entries (preserving original case).
    pub fn readdir(
        &self,
        path: &str,
        uid: u32,
        gids: &[u32],
    ) -> Result<Vec<DirEntryInfo>, CowFsError> {
        let id = self.resolve_in_root(&self.live, path)?;
        let ino = self.live.inodes.get(&id).ok_or(CowFsError::NotFound)?;
        if ino.kind != InodeKind::Directory {
            return Err(CowFsError::NotADirectory);
        }
        Self::check_inode_posix(&self.live, id, uid, gids, AclPerm::READ)?;

        let mut out = Vec::new();
        self.live.dirents.for_each(|k, v| {
            if k.parent_inode == id {
                out.push(DirEntryInfo {
                    inode: v.child_inode,
                    name: v.original_name.clone(),
                    kind: v.kind,
                });
            }
        });
        Ok(out)
    }

    /// CoW write to a file at `offset`. Allocates new physical blocks, writes data,
    /// updates extent B-tree, and decrements refcounts on overwritten blocks.
    pub fn write_file(
        &mut self,
        path: &str,
        offset: u64,
        data: &[u8],
        uid: u32,
        gids: &[u32],
    ) -> Result<usize, CowFsError> {
        let id = self.resolve_in_root(&self.live, path)?;
        let mut ino = self
            .live
            .inodes
            .get(&id)
            .cloned()
            .ok_or(CowFsError::NotFound)?;
        if ino.kind != InodeKind::File {
            return Err(CowFsError::IsADirectory);
        }
        Self::check_inode_posix(&self.live, id, uid, gids, AclPerm::WRITE)?;

        if data.is_empty() {
            return Ok(0);
        }

        let mut block_buf = vec![0u8; BLOCK_SIZE];
        let mut written = 0usize;

        while written < data.len() {
            let cur_pos = offset + written as u64;
            let lba = cur_pos / BLOCK_SIZE as u64;
            let in_blk_off = (cur_pos % BLOCK_SIZE as u64) as usize;
            let chunk = core::cmp::min(BLOCK_SIZE - in_blk_off, data.len() - written);

            let ext_key = ExtentKey {
                inode_id: id,
                logical_block: lba,
            };

            // If partial block write and an existing extent covers this logical block, read it first.
            let old_ext = self.live.extents.get(&ext_key).copied();
            if in_blk_off != 0 || chunk < BLOCK_SIZE {
                if let Some(old) = old_ext {
                    self.store.read_block(old.physical_block, &mut block_buf)?;
                    if crc32c(&block_buf) != old.checksum {
                        return Err(CowFsError::ChecksumMismatch);
                    }
                } else {
                    block_buf.fill(0);
                }
            }

            block_buf[in_blk_off..in_blk_off + chunk]
                .copy_from_slice(&data[written..written + chunk]);
            let csum = crc32c(&block_buf);

            // Allocate new physical block (CoW: never overwrite in place).
            let new_phys = self.allocator.alloc_extent(1)?;
            if let Err(e) = self.store.write_block(new_phys.start_block, &block_buf) {
                let _ = self.allocator.dec_ref(new_phys);
                return Err(e);
            }

            self.live.extents.insert(
                ext_key,
                FileExtent {
                    physical_block: new_phys.start_block,
                    block_count: 1,
                    checksum: csum,
                },
            );

            if let Some(old) = old_ext {
                self.allocator.dec_ref(PhysicalExtent {
                    start_block: old.physical_block,
                    block_count: old.block_count,
                })?;
            }

            written += chunk;
        }

        self.generation += 1;
        let end_pos = offset + written as u64;
        if end_pos > ino.size {
            ino.size = end_pos;
        }
        ino.generation = self.generation;
        self.live.inodes.insert(id, ino);

        Ok(written)
    }

    /// Read from a file in the live tree, verifying CRC32C checksums on every block.
    pub fn read_file(
        &self,
        path: &str,
        offset: u64,
        buf: &mut [u8],
        uid: u32,
        gids: &[u32],
    ) -> Result<usize, CowFsError> {
        let id = self.resolve_in_root(&self.live, path)?;
        Self::check_inode_posix(&self.live, id, uid, gids, AclPerm::READ)?;
        self.read_inode_from_root(&self.live, id, offset, buf)
    }

    fn read_inode_from_root(
        &self,
        root: &FsRoot,
        id: u64,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<usize, CowFsError> {
        let ino = root.inodes.get(&id).ok_or(CowFsError::NotFound)?;
        if ino.kind != InodeKind::File {
            return Err(CowFsError::IsADirectory);
        }
        if offset >= ino.size || buf.is_empty() {
            return Ok(0);
        }

        let to_read = core::cmp::min(buf.len() as u64, ino.size - offset) as usize;
        let mut block_buf = vec![0u8; BLOCK_SIZE];
        let mut read_bytes = 0usize;

        while read_bytes < to_read {
            let cur_pos = offset + read_bytes as u64;
            let lba = cur_pos / BLOCK_SIZE as u64;
            let in_blk_off = (cur_pos % BLOCK_SIZE as u64) as usize;
            let chunk = core::cmp::min(BLOCK_SIZE - in_blk_off, to_read - read_bytes);

            let ext_key = ExtentKey {
                inode_id: id,
                logical_block: lba,
            };
            if let Some(ext) = root.extents.get(&ext_key) {
                self.store.read_block(ext.physical_block, &mut block_buf)?;
                if crc32c(&block_buf) != ext.checksum {
                    return Err(CowFsError::ChecksumMismatch);
                }
                buf[read_bytes..read_bytes + chunk]
                    .copy_from_slice(&block_buf[in_blk_off..in_blk_off + chunk]);
            } else {
                // Sparse hole reads as zeroes.
                buf[read_bytes..read_bytes + chunk].fill(0);
            }

            read_bytes += chunk;
        }

        Ok(read_bytes)
    }

    /// Reflink clone: creates a new file sharing all physical extents with `src_path` (CoW).
    pub fn clone_file(
        &mut self,
        src_path: &str,
        dst_path: &str,
        uid: u32,
        gids: &[u32],
    ) -> Result<u64, CowFsError> {
        let src_id = self.resolve_in_root(&self.live, src_path)?;
        let src_ino = self
            .live
            .inodes
            .get(&src_id)
            .cloned()
            .ok_or(CowFsError::NotFound)?;
        if src_ino.kind != InodeKind::File {
            return Err(CowFsError::IsADirectory);
        }
        Self::check_inode_posix(&self.live, src_id, uid, gids, AclPerm::READ)?;

        let gid = gids.first().copied().unwrap_or(0);
        let dst_id = self.create_file(dst_path, uid, gid, src_ino.mode)?;

        let mut src_extents = Vec::new();
        self.live.extents.for_each(|k, v| {
            if k.inode_id == src_id {
                src_extents.push((k.logical_block, *v));
            }
        });

        for (lba, ext) in src_extents {
            self.allocator.inc_ref(PhysicalExtent {
                start_block: ext.physical_block,
                block_count: ext.block_count,
            })?;
            self.live.extents.insert(
                ExtentKey {
                    inode_id: dst_id,
                    logical_block: lba,
                },
                ext,
            );
        }

        if let Some(mut dst_ino) = self.live.inodes.get(&dst_id).cloned() {
            dst_ino.size = src_ino.size;
            self.live.inodes.insert(dst_id, dst_ino);
        }

        Ok(dst_id)
    }

    /// Unlink a regular file and decrement refcounts on its extents.
    pub fn unlink(&mut self, path: &str, uid: u32, gids: &[u32]) -> Result<(), CowFsError> {
        let (parent_id, name) = self.resolve_parent(path)?;
        Self::check_inode_posix(&self.live, parent_id, uid, gids, AclPerm::WRITE)?;

        let key = DirKey {
            parent_inode: parent_id,
            folded_name: fold_name(&name, self.case_mode),
        };
        let de = self
            .live
            .dirents
            .get(&key)
            .cloned()
            .ok_or(CowFsError::NotFound)?;
        if de.kind == InodeKind::Directory {
            return Err(CowFsError::IsADirectory);
        }

        self.live.dirents.remove(&key);
        let mut ino = self
            .live
            .inodes
            .get(&de.child_inode)
            .cloned()
            .ok_or(CowFsError::NotFound)?;
        ino.nlink = ino.nlink.saturating_sub(1);

        if ino.nlink == 0 {
            let mut to_remove = Vec::new();
            self.live.extents.for_each(|k, v| {
                if k.inode_id == de.child_inode {
                    to_remove.push((*k, *v));
                }
            });
            for (ek, ev) in to_remove {
                self.live.extents.remove(&ek);
                self.allocator.dec_ref(PhysicalExtent {
                    start_block: ev.physical_block,
                    block_count: ev.block_count,
                })?;
            }
            self.live.inodes.remove(&de.child_inode);
            self.live.acls.remove(&de.child_inode);
            self.live.sec_descs.remove(&de.child_inode);
        } else {
            self.live.inodes.insert(de.child_inode, ino);
        }

        self.generation += 1;
        Ok(())
    }

    /// Remove an empty directory.
    pub fn rmdir(&mut self, path: &str, uid: u32, gids: &[u32]) -> Result<(), CowFsError> {
        let (parent_id, name) = self.resolve_parent(path)?;
        Self::check_inode_posix(&self.live, parent_id, uid, gids, AclPerm::WRITE)?;

        let key = DirKey {
            parent_inode: parent_id,
            folded_name: fold_name(&name, self.case_mode),
        };
        let de = self
            .live
            .dirents
            .get(&key)
            .cloned()
            .ok_or(CowFsError::NotFound)?;
        if de.kind != InodeKind::Directory {
            return Err(CowFsError::NotADirectory);
        }

        let mut has_children = false;
        self.live.dirents.for_each(|k, _| {
            if k.parent_inode == de.child_inode {
                has_children = true;
            }
        });
        if has_children {
            return Err(CowFsError::DirectoryNotEmpty);
        }

        self.live.dirents.remove(&key);
        self.live.inodes.remove(&de.child_inode);
        self.live.acls.remove(&de.child_inode);
        self.live.sec_descs.remove(&de.child_inode);

        if let Some(mut p) = self.live.inodes.get(&parent_id).cloned() {
            p.nlink = p.nlink.saturating_sub(1);
            self.live.inodes.insert(parent_id, p);
        }
        self.generation += 1;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // ACL & NT Security Descriptor APIs
    // -----------------------------------------------------------------------

    pub fn set_posix_acl(&mut self, path: &str, acl: PosixAcl, uid: u32) -> Result<(), CowFsError> {
        acl.validate()?;
        let id = self.resolve_in_root(&self.live, path)?;
        let ino = self.live.inodes.get(&id).ok_or(CowFsError::NotFound)?;
        if uid != 0 && uid != ino.uid {
            return Err(CowFsError::PermissionDenied);
        }
        self.live.acls.insert(id, acl);
        Ok(())
    }

    pub fn get_posix_acl(&self, path: &str) -> Result<PosixAcl, CowFsError> {
        let id = self.resolve_in_root(&self.live, path)?;
        self.live.acls.get(&id).cloned().ok_or(CowFsError::NotFound)
    }

    pub fn set_nt_security(
        &mut self,
        path: &str,
        sd: NtSecurityDescriptor,
        token: &SecurityToken,
    ) -> Result<(), CowFsError> {
        let id = self.resolve_in_root(&self.live, path)?;
        let cur_sd = self.live.sec_descs.get(&id).ok_or(CowFsError::NotFound)?;
        if !cur_sd.check_access(token, nt_access::WRITE_DAC) {
            return Err(CowFsError::PermissionDenied);
        }
        self.live.sec_descs.insert(id, sd);
        Ok(())
    }

    pub fn get_nt_security(&self, path: &str) -> Result<NtSecurityDescriptor, CowFsError> {
        let id = self.resolve_in_root(&self.live, path)?;
        self.live
            .sec_descs
            .get(&id)
            .cloned()
            .ok_or(CowFsError::NotFound)
    }

    pub fn check_nt_access(
        &self,
        path: &str,
        token: &SecurityToken,
        desired_access: u32,
    ) -> Result<bool, CowFsError> {
        let id = self.resolve_in_root(&self.live, path)?;
        let sd = self.live.sec_descs.get(&id).ok_or(CowFsError::NotFound)?;
        Ok(sd.check_access(token, desired_access))
    }

    // -----------------------------------------------------------------------
    // CoW Snapshots
    // -----------------------------------------------------------------------

    /// Create a read-only CoW snapshot in O(1) tree-root clone + extent refcount bump.
    pub fn create_snapshot(&mut self, name: &str) -> Result<u64, CowFsError> {
        if name.is_empty() || name.len() > MAX_NAME_LEN {
            return Err(CowFsError::InvalidArgument);
        }
        if self.snapshots.contains_key(name) {
            return Err(CowFsError::AlreadyExists);
        }

        let mut extents_to_ref = Vec::new();
        self.live.extents.for_each(|_, v| {
            extents_to_ref.push(PhysicalExtent {
                start_block: v.physical_block,
                block_count: v.block_count,
            });
        });
        for ext in extents_to_ref {
            self.allocator.inc_ref(ext)?;
        }

        let id = self.next_snapshot_id;
        self.next_snapshot_id += 1;
        let snap = Snapshot {
            id,
            name: name.to_string(),
            generation: self.generation,
            read_only: true,
            root: self.live.clone(),
        };
        self.snapshots.insert(name.to_string(), snap);
        Ok(id)
    }

    /// Delete a snapshot and release its extent references.
    pub fn delete_snapshot(&mut self, name: &str) -> Result<(), CowFsError> {
        let snap = self
            .snapshots
            .remove(name)
            .ok_or(CowFsError::InvalidSnapshot)?;

        let mut extents_to_unref = Vec::new();
        snap.root.extents.for_each(|_, v| {
            extents_to_unref.push(PhysicalExtent {
                start_block: v.physical_block,
                block_count: v.block_count,
            });
        });
        for ext in extents_to_unref {
            self.allocator.dec_ref(ext)?;
        }
        Ok(())
    }

    pub fn list_snapshots(&self) -> Vec<SnapshotInfo> {
        self.snapshots
            .values()
            .map(|s| SnapshotInfo {
                id: s.id,
                name: s.name.clone(),
                generation: s.generation,
                read_only: s.read_only,
            })
            .collect()
    }

    /// Read a file from a named snapshot.
    pub fn read_snapshot_file(
        &self,
        snap_name: &str,
        path: &str,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<usize, CowFsError> {
        let snap = self
            .snapshots
            .get(snap_name)
            .ok_or(CowFsError::InvalidSnapshot)?;
        let id = self.resolve_in_root(&snap.root, path)?;
        self.read_inode_from_root(&snap.root, id, offset, buf)
    }

    /// Roll back the live filesystem to a named snapshot (preserving the snapshot).
    pub fn rollback_snapshot(&mut self, name: &str) -> Result<(), CowFsError> {
        let snap_root = self
            .snapshots
            .get(name)
            .ok_or(CowFsError::InvalidSnapshot)?
            .root
            .clone();

        // Bump refcounts for the restored root first, then drop old live root's refcounts.
        let mut new_exts = Vec::new();
        snap_root.extents.for_each(|_, v| {
            new_exts.push(PhysicalExtent {
                start_block: v.physical_block,
                block_count: v.block_count,
            });
        });
        for ext in new_exts {
            self.allocator.inc_ref(ext)?;
        }

        let mut old_exts = Vec::new();
        self.live.extents.for_each(|_, v| {
            old_exts.push(PhysicalExtent {
                start_block: v.physical_block,
                block_count: v.block_count,
            });
        });
        for ext in old_exts {
            self.allocator.dec_ref(ext)?;
        }

        self.live = snap_root;
        self.generation += 1;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // fsck Consistency Verifier
    // -----------------------------------------------------------------------

    /// Verify full filesystem consistency across live tree, all snapshots,
    /// extent checksums, directory connectivity, ACLs, and refcount accounting.
    pub fn fsck(&self) -> Result<FsckReport, CowFsError> {
        let mut expected_refs: BTreeMap<u64, u32> = BTreeMap::new();
        let mut inodes_checked = 0usize;
        let mut dirents_checked = 0usize;
        let mut extents_checked = 0usize;

        let mut block_buf = vec![0u8; BLOCK_SIZE];

        self.verify_root(
            &self.live,
            &mut expected_refs,
            &mut block_buf,
            &mut inodes_checked,
            &mut dirents_checked,
            &mut extents_checked,
        )?;

        for snap in self.snapshots.values() {
            self.verify_root(
                &snap.root,
                &mut expected_refs,
                &mut block_buf,
                &mut inodes_checked,
                &mut dirents_checked,
                &mut extents_checked,
            )?;
        }

        // Verify exact refcount match against ExtentAllocator.
        let mut blocks_referenced = 0u64;
        for blk in 0..self.allocator.total_blocks() {
            let exp = expected_refs.get(&blk).copied().unwrap_or(0);
            let actual = self.allocator.refcount(blk);
            if exp != actual {
                return Err(CowFsError::CorruptMetadata("extent refcount mismatch"));
            }
            if exp > 0 {
                blocks_referenced += 1;
            }
        }

        if blocks_referenced + self.allocator.free_blocks() != self.allocator.total_blocks() {
            return Err(CowFsError::CorruptMetadata("block accounting mismatch"));
        }

        Ok(FsckReport {
            inodes_checked,
            dirents_checked,
            extents_checked,
            snapshots_checked: self.snapshots.len(),
            blocks_referenced,
            blocks_free: self.allocator.free_blocks(),
        })
    }

    fn verify_root(
        &self,
        root: &FsRoot,
        expected_refs: &mut BTreeMap<u64, u32>,
        block_buf: &mut [u8],
        inodes_checked: &mut usize,
        dirents_checked: &mut usize,
        extents_checked: &mut usize,
    ) -> Result<(), CowFsError> {
        *inodes_checked += root.inodes.verify_invariants()?;
        *dirents_checked += root.dirents.verify_invariants()?;
        *extents_checked += root.extents.verify_invariants()?;
        root.acls.verify_invariants()?;
        root.sec_descs.verify_invariants()?;

        if root.inodes.get(&ROOT_INODE).is_none() {
            return Err(CowFsError::CorruptMetadata("missing root inode"));
        }

        // Verify ACLs.
        let mut acl_err: Option<CowFsError> = None;
        root.acls.for_each(|_, acl| {
            if acl_err.is_none() {
                if let Err(e) = acl.validate() {
                    acl_err = Some(e);
                }
            }
        });
        if let Some(e) = acl_err {
            return Err(e);
        }

        // Verify directory entries point to valid inodes.
        let mut dir_err: Option<CowFsError> = None;
        root.dirents.for_each(|k, v| {
            if dir_err.is_none() {
                if root.inodes.get(&k.parent_inode).is_none() {
                    dir_err = Some(CowFsError::CorruptMetadata("dirent missing parent inode"));
                } else if let Some(child) = root.inodes.get(&v.child_inode) {
                    if child.kind != v.kind {
                        dir_err = Some(CowFsError::CorruptMetadata("dirent kind mismatch"));
                    }
                } else {
                    dir_err = Some(CowFsError::CorruptMetadata("dirent missing child inode"));
                }
            }
        });
        if let Some(e) = dir_err {
            return Err(e);
        }

        // Verify extents: bounds, checksums, and accumulate expected refcounts.
        let mut ext_err: Option<CowFsError> = None;
        root.extents.for_each(|k, v| {
            if ext_err.is_some() {
                return;
            }
            if root.inodes.get(&k.inode_id).is_none() {
                ext_err = Some(CowFsError::CorruptMetadata("orphan extent"));
                return;
            }
            if v.block_count == 0
                || v.physical_block.saturating_add(v.block_count) > self.allocator.total_blocks()
            {
                ext_err = Some(CowFsError::CorruptMetadata("extent out of bounds"));
                return;
            }
            if let Err(e) = self.store.read_block(v.physical_block, block_buf) {
                ext_err = Some(e);
                return;
            }
            if crc32c(block_buf) != v.checksum {
                ext_err = Some(CowFsError::ChecksumMismatch);
                return;
            }
            for blk in v.physical_block..(v.physical_block + v.block_count) {
                *expected_refs.entry(blk).or_insert(0) += 1;
            }
        });
        if let Some(e) = ext_err {
            return Err(e);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Self-Test Suite
// ---------------------------------------------------------------------------

/// Run comprehensive self-tests for the CoW B-Tree Extent Filesystem & ACL/Security Engine.
pub fn self_test() -> Result<(), &'static str> {
    serial_println!("[cow_fs] running self_test...");

    // 1. ExtentAllocator coalescing & refcounting
    {
        let mut alloc = ExtentAllocator::new(64).map_err(|_| "alloc init failed")?;
        let e1 = alloc.alloc_extent(8).map_err(|_| "alloc e1 failed")?;
        let e2 = alloc.alloc_extent(8).map_err(|_| "alloc e2 failed")?;
        if alloc.free_blocks() != 48 {
            return Err("unexpected free_blocks after alloc");
        }
        alloc.inc_ref(e1).map_err(|_| "inc_ref failed")?;
        alloc.dec_ref(e1).map_err(|_| "dec_ref 1 failed")?;
        if alloc.refcount(e1.start_block) != 1 {
            return Err("refcount should still be 1");
        }
        alloc.dec_ref(e1).map_err(|_| "dec_ref 2 failed")?;
        alloc.dec_ref(e2).map_err(|_| "dec_ref e2 failed")?;
        if alloc.free_blocks() != 64 || alloc.free_extent_count() != 1 {
            return Err("extent coalescing failed");
        }
    }

    // 2. CowBTree insert, split, snapshot sharing, remove, invariants
    {
        let mut tree: CowBTree<u64, u64> = CowBTree::new();
        for i in 0..100u64 {
            tree.insert(i, i * 10);
        }
        tree.verify_invariants()
            .map_err(|_| "btree invariants after insert")?;
        let snap = tree.clone();
        for i in 0..50u64 {
            tree.remove(&i);
        }
        tree.verify_invariants()
            .map_err(|_| "btree invariants after remove")?;
        if snap.len() != 100 || tree.len() != 50 {
            return Err("cow btree snapshot isolation failed");
        }
        if snap.get(&10) != Some(&100) || tree.get(&10).is_some() {
            return Err("cow btree lookup mismatch");
        }
    }

    // 3. CowFs: case-preserving/insensitive lookup, CoW writes, reflink, snapshots, rollback, fsck
    let store = InMemoryBlockStore::new(256);
    let mut fs =
        CowFs::new(store, CaseMode::CaseInsensitivePreserving).map_err(|_| "CowFs::new failed")?;

    fs.mkdir("/Docs", 1000, 1000, 0o755)
        .map_err(|_| "mkdir /Docs failed")?;
    fs.create_file("/docs/Report_Final.TXT", 1000, 1000, 0o644)
        .map_err(|_| "create_file failed")?;

    // Duplicate case-variant must fail with AlreadyExists
    if fs.create_file("/DOCS/report_final.txt", 1000, 1000, 0o644) != Err(CowFsError::AlreadyExists)
    {
        return Err("case-insensitive duplicate check failed");
    }

    // readdir preserves original casing
    let entries = fs
        .readdir("/DOCS", 1000, &[1000])
        .map_err(|_| "readdir failed")?;
    if entries.len() != 1 || entries[0].name != "Report_Final.TXT" {
        return Err("case-preserving readdir failed");
    }

    // Write data & take snapshot
    let payload_v1 = b"Nexus OS CoW B-Tree v1 payload!";
    fs.write_file("/docs/report_final.txt", 0, payload_v1, 1000, &[1000])
        .map_err(|_| "write v1 failed")?;
    fs.create_snapshot("snap1")
        .map_err(|_| "create_snapshot failed")?;

    // Overwrite in live tree (CoW)
    let payload_v2 = b"Nexus OS CoW B-Tree v2 OVERWRITTEN!";
    fs.write_file("/docs/report_final.txt", 0, payload_v2, 1000, &[1000])
        .map_err(|_| "write v2 failed")?;

    let mut buf = [0u8; 64];
    let n1 = fs
        .read_snapshot_file("snap1", "/docs/report_final.txt", 0, &mut buf)
        .map_err(|_| "read_snapshot_file failed")?;
    if &buf[..n1] != payload_v1 {
        return Err("snapshot data corrupted after CoW write");
    }

    let n2 = fs
        .read_file("/docs/report_final.txt", 0, &mut buf, 1000, &[1000])
        .map_err(|_| "read_file v2 failed")?;
    if &buf[..n2] != payload_v2 {
        return Err("live data mismatch after CoW write");
    }

    // Reflink clone
    fs.clone_file("/docs/report_final.txt", "/docs/clone.txt", 1000, &[1000])
        .map_err(|_| "clone_file failed")?;
    let n3 = fs
        .read_file("/docs/clone.txt", 0, &mut buf, 1000, &[1000])
        .map_err(|_| "read clone failed")?;
    if &buf[..n3] != payload_v2 {
        return Err("clone data mismatch");
    }

    // 4. POSIX ACLs
    let acl = PosixAcl {
        entries: vec![
            AclEntry {
                tag: AclTag::UserObj,
                perm: AclPerm::RW,
            },
            AclEntry {
                tag: AclTag::User(2000),
                perm: AclPerm::RW,
            },
            AclEntry {
                tag: AclTag::GroupObj,
                perm: AclPerm::READ,
            },
            AclEntry {
                tag: AclTag::Mask,
                perm: AclPerm::READ, // clamps User(2000) to READ only
            },
            AclEntry {
                tag: AclTag::Other,
                perm: AclPerm::NONE,
            },
        ],
        default_entries: Vec::new(),
    };
    fs.set_posix_acl("/docs/report_final.txt", acl, 1000)
        .map_err(|_| "set_posix_acl failed")?;
    if fs.write_file("/docs/report_final.txt", 0, b"x", 2000, &[2000])
        != Err(CowFsError::PermissionDenied)
    {
        return Err("POSIX ACL mask clamping failed");
    }
    if fs
        .read_file("/docs/report_final.txt", 0, &mut buf, 2000, &[2000])
        .is_err()
    {
        return Err("POSIX ACL named user read failed");
    }

    // 5. NT Security Descriptors
    let sd = NtSecurityDescriptor {
        owner: Sid::from_uid(1000),
        group: Sid::from_gid(1000),
        control: 0x0004,
        dacl: Some(NtAcl {
            revision: 2,
            aces: vec![
                Ace {
                    ace_type: AceType::AccessDenied,
                    flags: 0,
                    mask: nt_access::FILE_WRITE_DATA,
                    sid: Sid::from_uid(3000),
                },
                Ace {
                    ace_type: AceType::AccessAllowed,
                    flags: 0,
                    mask: nt_access::GENERIC_READ | nt_access::GENERIC_WRITE,
                    sid: Sid::everyone(),
                },
            ],
        }),
        sacl: None,
    };
    let owner_tok = SecurityToken::from_unix(1000, &[1000]);
    fs.set_nt_security("/docs/report_final.txt", sd, &owner_tok)
        .map_err(|_| "set_nt_security failed")?;

    let user3000 = SecurityToken::from_unix(3000, &[3000]);
    if fs
        .check_nt_access(
            "/docs/report_final.txt",
            &user3000,
            nt_access::FILE_WRITE_DATA,
        )
        .map_err(|_| "check_nt_access failed")?
    {
        return Err("NT AccessDenied ACE should block user 3000 write");
    }
    if !fs
        .check_nt_access(
            "/docs/report_final.txt",
            &user3000,
            nt_access::FILE_READ_DATA,
        )
        .map_err(|_| "check_nt_access read failed")?
    {
        return Err("NT AccessAllowed EVERYONE should allow user 3000 read");
    }

    // 6. fsck on healthy filesystem
    let report = fs.fsck().map_err(|_| "fsck failed on healthy fs")?;
    if report.snapshots_checked != 1 || report.extents_checked == 0 {
        return Err("unexpected fsck report counts");
    }

    // 7. Rollback & delete snapshot
    fs.rollback_snapshot("snap1")
        .map_err(|_| "rollback_snapshot failed")?;
    let n_rb = fs
        .read_file("/docs/report_final.txt", 0, &mut buf, 1000, &[1000])
        .map_err(|_| "read after rollback failed")?;
    if &buf[..n_rb] != payload_v1 {
        return Err("rollback did not restore v1 data");
    }
    fs.delete_snapshot("snap1")
        .map_err(|_| "delete_snapshot failed")?;
    fs.fsck()
        .map_err(|_| "fsck failed after rollback & delete_snapshot")?;

    serial_println!("[cow_fs] self_test PASSED");
    Ok(())
}
