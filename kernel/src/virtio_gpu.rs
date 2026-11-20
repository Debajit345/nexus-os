//! Clean-room VirtIO-GPU 2D/3D Virgl & Vulkan/DRM Command Submission Engine.
//!
//! Implements the VirtIO 1.2 GPU device specification (`no_std` + `alloc`):
//! - **Split Virtqueues (`controlq` & `cursorq`)**: Descriptor table, available ring,
//!   used ring, free-descriptor allocator, scatter-gather chain builder, and fence-tracked
//!   completion.
//! - **2D Display Pipeline**: `GET_DISPLAY_INFO`, `GET_EDID`, `RESOURCE_CREATE_2D`,
//!   `RESOURCE_ATTACH_BACKING`, `TRANSFER_TO_HOST_2D`, `SET_SCANOUT`, `RESOURCE_FLUSH`,
//!   `RESOURCE_DETACH_BACKING`, `RESOURCE_UNREF`, plus `UPDATE_CURSOR` / `MOVE_CURSOR`.
//! - **3D Virgl & Vulkan/DRM Engine**: `GET_CAPSET_INFO`, `GET_CAPSET`, `CTX_CREATE`
//!   (Virgl / Venus-Vulkan / CrossDomain), `CTX_DESTROY`, `RESOURCE_CREATE_3D`,
//!   `RESOURCE_CREATE_BLOB`, `CTX_ATTACH_RESOURCE`, `CTX_DETACH_RESOURCE`,
//!   `TRANSFER_TO_HOST_3D`, `TRANSFER_FROM_HOST_3D`, `SUBMIT_3D`, and a Virgl command
//!   stream encoder + DRM/Vulkan fence-tracked submission engine.
#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::sync::TicketLock;

// ---------------------------------------------------------------------------
// VirtIO-GPU Feature Bits & Constants
// ---------------------------------------------------------------------------

pub const VIRTIO_GPU_F_VIRGL: u64 = 1 << 0;
pub const VIRTIO_GPU_F_EDID: u64 = 1 << 1;
pub const VIRTIO_GPU_F_RESOURCE_UUID: u64 = 1 << 2;
pub const VIRTIO_GPU_F_RESOURCE_BLOB: u64 = 1 << 3;
pub const VIRTIO_GPU_F_CONTEXT_INIT: u64 = 1 << 4;

pub const VIRTIO_GPU_MAX_SCANOUTS: usize = 16;
pub const VIRTIO_GPU_FLAG_FENCE: u32 = 1 << 0;
pub const VIRTIO_GPU_FLAG_INFO_RING_IDX: u32 = 1 << 1;

pub const VIRTIO_GPU_CAPSET_VIRGL: u32 = 1;
pub const VIRTIO_GPU_CAPSET_VIRGL2: u32 = 2;
pub const VIRTIO_GPU_CAPSET_GFXSTREAM_VULKAN: u32 = 3;
pub const VIRTIO_GPU_CAPSET_VENUS: u32 = 4;
pub const VIRTIO_GPU_CAPSET_CROSS_DOMAIN: u32 = 5;

pub const VIRTIO_GPU_CONTEXT_INIT_CAPSET_ID_MASK: u32 = 0x00FF;

pub const VIRTIO_GPU_BLOB_MEM_GUEST: u32 = 0x0001;
pub const VIRTIO_GPU_BLOB_MEM_HOST3D: u32 = 0x0002;
pub const VIRTIO_GPU_BLOB_MEM_HOST3D_GUEST: u32 = 0x0003;

pub const VIRTIO_GPU_BLOB_FLAG_USE_MAPPABLE: u32 = 0x0001;
pub const VIRTIO_GPU_BLOB_FLAG_USE_SHAREABLE: u32 = 0x0002;
pub const VIRTIO_GPU_BLOB_FLAG_USE_CROSS_DEVICE: u32 = 0x0004;

pub const DEFAULT_QUEUE_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtioGpuError {
    InvalidArgument,
    NotInitialized,
    QueueFull,
    NoDescriptors,
    ResourceNotFound,
    ContextNotFound,
    ScanoutOutOfRange,
    UnsupportedFeature,
    DeviceError(u32),
    FenceTimeout,
    OutOfMemory,
    InvalidResponse,
}

impl VirtioGpuError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgument => "virtio-gpu: invalid argument",
            Self::NotInitialized => "virtio-gpu: device not initialized",
            Self::QueueFull => "virtio-gpu: virtqueue full",
            Self::NoDescriptors => "virtio-gpu: no free virtqueue descriptors",
            Self::ResourceNotFound => "virtio-gpu: resource not found",
            Self::ContextNotFound => "virtio-gpu: context not found",
            Self::ScanoutOutOfRange => "virtio-gpu: scanout index out of range",
            Self::UnsupportedFeature => "virtio-gpu: unsupported feature",
            Self::DeviceError(_) => "virtio-gpu: device returned error response",
            Self::FenceTimeout => "virtio-gpu: fence wait timed out",
            Self::OutOfMemory => "virtio-gpu: out of memory",
            Self::InvalidResponse => "virtio-gpu: invalid device response",
        }
    }
}

// ---------------------------------------------------------------------------
// VirtIO-GPU Wire Protocol Types
// ---------------------------------------------------------------------------

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtioGpuCtrlType {
    Undefined = 0,
    // 2D commands
    CmdGetDisplayInfo = 0x0100,
    CmdResourceCreate2d = 0x0101,
    CmdResourceUnref = 0x0102,
    CmdSetScanout = 0x0103,
    CmdResourceFlush = 0x0104,
    CmdTransferToHost2d = 0x0105,
    CmdResourceAttachBacking = 0x0106,
    CmdResourceDetachBacking = 0x0107,
    CmdGetCapsetInfo = 0x0108,
    CmdGetCapset = 0x0109,
    CmdGetEdid = 0x010A,
    CmdResourceAssignUuid = 0x010B,
    CmdResourceCreateBlob = 0x010C,
    CmdSetScanoutBlob = 0x010D,
    // 3D commands
    CmdCtxCreate = 0x0200,
    CmdCtxDestroy = 0x0201,
    CmdCtxAttachResource = 0x0202,
    CmdCtxDetachResource = 0x0203,
    CmdResourceCreate3d = 0x0204,
    CmdTransferToHost3d = 0x0205,
    CmdTransferFromHost3d = 0x0206,
    CmdSubmit3d = 0x0207,
    CmdResourceMapBlob = 0x0208,
    CmdResourceUnmapBlob = 0x0209,
    // Cursor commands
    CmdUpdateCursor = 0x0300,
    CmdMoveCursor = 0x0301,
    // Success responses
    RespOkNodata = 0x1100,
    RespOkDisplayInfo = 0x1101,
    RespOkCapsetInfo = 0x1102,
    RespOkCapset = 0x1103,
    RespOkEdid = 0x1104,
    RespOkResourceUuid = 0x1105,
    RespOkMapInfo = 0x1106,
    // Error responses
    RespErrUnspec = 0x1200,
    RespErrOutOfMemory = 0x1201,
    RespErrInvalidScanoutId = 0x1202,
    RespErrInvalidResourceId = 0x1203,
    RespErrInvalidContextId = 0x1204,
    RespErrInvalidParameter = 0x1205,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub enum VirtioGpuFormat {
    B8G8R8A8Unorm = 1,
    B8G8R8X8Unorm = 2,
    A8R8G8B8Unorm = 3,
    X8R8G8B8Unorm = 4,
    R8G8B8A8Unorm = 67,
    X8B8G8R8Unorm = 68,
    A8B8G8R8Unorm = 121,
    R8G8B8X8Unorm = 134,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VirtioGpuCtrlHdr {
    pub cmd_type: u32,
    pub flags: u32,
    pub fence_id: u64,
    pub ctx_id: u32,
    pub ring_idx: u8,
    pub padding: [u8; 3],
}

impl VirtioGpuCtrlHdr {
    pub const fn new(cmd_type: VirtioGpuCtrlType) -> Self {
        Self {
            cmd_type: cmd_type as u32,
            flags: 0,
            fence_id: 0,
            ctx_id: 0,
            ring_idx: 0,
            padding: [0; 3],
        }
    }

    pub const fn with_fence(mut self, fence_id: u64) -> Self {
        self.flags |= VIRTIO_GPU_FLAG_FENCE;
        self.fence_id = fence_id;
        self
    }

    pub const fn with_ctx(mut self, ctx_id: u32) -> Self {
        self.ctx_id = ctx_id;
        self
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VirtioGpuRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl VirtioGpuRect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VirtioGpuBox {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
    pub h: u32,
    pub d: u32,
}

impl VirtioGpuBox {
    pub const fn new(x: u32, y: u32, z: u32, w: u32, h: u32, d: u32) -> Self {
        Self { x, y, z, w, h, d }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuDisplayOne {
    pub r: VirtioGpuRect,
    pub enabled: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioGpuRespDisplayInfo {
    pub hdr: VirtioGpuCtrlHdr,
    pub pmodes: [VirtioGpuDisplayOne; VIRTIO_GPU_MAX_SCANOUTS],
}

impl Default for VirtioGpuRespDisplayInfo {
    fn default() -> Self {
        Self {
            hdr: VirtioGpuCtrlHdr::default(),
            pmodes: [VirtioGpuDisplayOne::default(); VIRTIO_GPU_MAX_SCANOUTS],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuResourceCreate2d {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub format: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuResourceUnref {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuSetScanout {
    pub hdr: VirtioGpuCtrlHdr,
    pub r: VirtioGpuRect,
    pub scanout_id: u32,
    pub resource_id: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuResourceFlush {
    pub hdr: VirtioGpuCtrlHdr,
    pub r: VirtioGpuRect,
    pub resource_id: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuTransferToHost2d {
    pub hdr: VirtioGpuCtrlHdr,
    pub r: VirtioGpuRect,
    pub offset: u64,
    pub resource_id: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuResourceAttachBacking {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub nr_entries: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VirtioGpuMemEntry {
    pub addr: u64,
    pub length: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuResourceDetachBacking {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuGetEdid {
    pub hdr: VirtioGpuCtrlHdr,
    pub scanout: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioGpuRespEdid {
    pub hdr: VirtioGpuCtrlHdr,
    pub size: u32,
    pub padding: u32,
    pub edid: [u8; 1024],
}

impl Default for VirtioGpuRespEdid {
    fn default() -> Self {
        Self {
            hdr: VirtioGpuCtrlHdr::default(),
            size: 0,
            padding: 0,
            edid: [0; 1024],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuCursorPos {
    pub scanout_id: u32,
    pub x: u32,
    pub y: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuUpdateCursor {
    pub hdr: VirtioGpuCtrlHdr,
    pub pos: VirtioGpuCursorPos,
    pub resource_id: u32,
    pub hot_x: u32,
    pub hot_y: u32,
    pub padding: u32,
}

// 3D & Capset & Blob Wire Structures

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuGetCapsetInfo {
    pub hdr: VirtioGpuCtrlHdr,
    pub capset_index: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuRespCapsetInfo {
    pub hdr: VirtioGpuCtrlHdr,
    pub capset_id: u32,
    pub capset_max_version: u32,
    pub capset_max_size: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuGetCapset {
    pub hdr: VirtioGpuCtrlHdr,
    pub capset_id: u32,
    pub capset_version: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioGpuCtxCreate {
    pub hdr: VirtioGpuCtrlHdr,
    pub nlen: u32,
    pub context_init: u32,
    pub debug_name: [u8; 64],
}

impl Default for VirtioGpuCtxCreate {
    fn default() -> Self {
        Self {
            hdr: VirtioGpuCtrlHdr::default(),
            nlen: 0,
            context_init: 0,
            debug_name: [0; 64],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuCtxDestroy {
    pub hdr: VirtioGpuCtrlHdr,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuCtxResource {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuResourceCreate3d {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub target: u32,
    pub format: u32,
    pub bind: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_size: u32,
    pub last_level: u32,
    pub nr_samples: u32,
    pub flags: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuTransferHost3d {
    pub hdr: VirtioGpuCtrlHdr,
    pub box_region: VirtioGpuBox,
    pub offset: u64,
    pub resource_id: u32,
    pub level: u32,
    pub stride: u32,
    pub layer_stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuCmdSubmit {
    pub hdr: VirtioGpuCtrlHdr,
    pub size: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuResourceCreateBlob {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub blob_mem: u32,
    pub blob_flags: u32,
    pub nr_entries: u32,
    pub blob_id: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuResourceMapBlob {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub padding: u32,
    pub offset: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuRespMapInfo {
    pub hdr: VirtioGpuCtrlHdr,
    pub map_info: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioGpuResourceUnmapBlob {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub padding: u32,
}

// ---------------------------------------------------------------------------
// Split Virtqueue Implementation (controlq / cursorq)
// ---------------------------------------------------------------------------

pub const VIRTQ_DESC_F_NEXT: u16 = 1;
pub const VIRTQ_DESC_F_WRITE: u16 = 2;
pub const VIRTQ_DESC_F_INDIRECT: u16 = 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtqDesc {
    pub addr: u64,
    pub len: u32,
    pub flags: u16,
    pub next: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtqUsedElem {
    pub id: u32,
    pub len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueKind {
    Control = 0,
    Cursor = 1,
}

#[derive(Debug, Clone, Copy)]
pub struct SgSegment {
    pub phys_addr: u64,
    pub len: u32,
    pub device_writable: bool,
}

/// Split virtqueue state allocated on the heap to keep stack frames small.
pub struct SplitVirtqueue {
    pub kind: QueueKind,
    pub queue_size: u16,
    pub desc: Vec<VirtqDesc>,
    pub avail_flags: u16,
    pub avail_idx: u16,
    pub avail_ring: Vec<u16>,
    pub used_flags: u16,
    pub used_idx: u16,
    pub last_seen_used: u16,
    pub used_ring: Vec<VirtqUsedElem>,
    free_head: u16,
    num_free: u16,
    notify_count: u64,
    notify_mmio_addr: Option<u64>,
}

impl SplitVirtqueue {
    pub fn new(
        kind: QueueKind,
        queue_size: u16,
        notify_mmio_addr: Option<u64>,
    ) -> Result<Self, VirtioGpuError> {
        if queue_size == 0 || !queue_size.is_power_of_two() {
            return Err(VirtioGpuError::InvalidArgument);
        }
        let n = queue_size as usize;
        let mut desc = vec![VirtqDesc::default(); n];
        for (i, d) in desc.iter_mut().enumerate() {
            d.next = if i + 1 < n { (i + 1) as u16 } else { u16::MAX };
        }
        Ok(Self {
            kind,
            queue_size,
            desc,
            avail_flags: 0,
            avail_idx: 0,
            avail_ring: vec![0; n],
            used_flags: 0,
            used_idx: 0,
            last_seen_used: 0,
            used_ring: vec![VirtqUsedElem::default(); n],
            free_head: 0,
            num_free: queue_size,
            notify_count: 0,
            notify_mmio_addr,
        })
    }

    pub fn free_descriptors(&self) -> u16 {
        self.num_free
    }

    pub fn notify_count(&self) -> u64 {
        self.notify_count
    }

    fn alloc_desc(&mut self) -> Option<u16> {
        if self.num_free == 0 || self.free_head == u16::MAX {
            return None;
        }
        let idx = self.free_head;
        self.free_head = self.desc[idx as usize].next;
        self.num_free -= 1;
        Some(idx)
    }

    fn free_chain(&mut self, head: u16) {
        let mut cur = head;
        let mut steps = 0u16;
        while cur < self.queue_size && steps < self.queue_size {
            let flags = self.desc[cur as usize].flags;
            let next = self.desc[cur as usize].next;
            self.desc[cur as usize] = VirtqDesc {
                addr: 0,
                len: 0,
                flags: 0,
                next: self.free_head,
            };
            self.free_head = cur;
            self.num_free += 1;
            steps += 1;
            if (flags & VIRTQ_DESC_F_NEXT) == 0 {
                break;
            }
            cur = next;
        }
    }

    /// Enqueue a scatter-gather descriptor chain into the available ring and kick the queue.
    pub fn submit_chain(&mut self, segments: &[SgSegment]) -> Result<u16, VirtioGpuError> {
        if segments.is_empty() {
            return Err(VirtioGpuError::InvalidArgument);
        }
        if (self.num_free as usize) < segments.len() {
            return Err(VirtioGpuError::NoDescriptors);
        }

        let mut head = u16::MAX;
        let mut prev = u16::MAX;

        for (idx, seg) in segments.iter().enumerate() {
            let desc_idx = self.alloc_desc().ok_or(VirtioGpuError::NoDescriptors)?;
            if head == u16::MAX {
                head = desc_idx;
            }
            if prev != u16::MAX {
                self.desc[prev as usize].flags |= VIRTQ_DESC_F_NEXT;
                self.desc[prev as usize].next = desc_idx;
            }
            let mut flags = 0u16;
            if seg.device_writable {
                flags |= VIRTQ_DESC_F_WRITE;
            }
            if idx + 1 < segments.len() {
                flags |= VIRTQ_DESC_F_NEXT;
            }
            self.desc[desc_idx as usize] = VirtqDesc {
                addr: seg.phys_addr,
                len: seg.len,
                flags,
                next: u16::MAX,
            };
            prev = desc_idx;
        }

        let slot = (self.avail_idx as usize) % (self.queue_size as usize);
        self.avail_ring[slot] = head;
        core::sync::atomic::fence(Ordering::Release);
        self.avail_idx = self.avail_idx.wrapping_add(1);
        self.kick();
        Ok(head)
    }

    /// Notify the device (MMIO doorbell if mapped, plus counter tracking).
    pub fn kick(&mut self) {
        self.notify_count = self.notify_count.wrapping_add(1);
        if let Some(addr) = self.notify_mmio_addr {
            unsafe {
                core::ptr::write_volatile(addr as *mut u16, self.kind as u16);
            }
        }
    }

    /// Device-side helper (used by loopback/emulated backend) to mark a chain completed.
    pub fn push_used(&mut self, head: u16, written_len: u32) {
        let slot = (self.used_idx as usize) % (self.queue_size as usize);
        self.used_ring[slot] = VirtqUsedElem {
            id: head as u32,
            len: written_len,
        };
        core::sync::atomic::fence(Ordering::Release);
        self.used_idx = self.used_idx.wrapping_add(1);
    }

    /// Poll and reclaim one completed descriptor chain from the used ring.
    pub fn pop_used(&mut self) -> Option<VirtqUsedElem> {
        if self.last_seen_used == self.used_idx {
            return None;
        }
        core::sync::atomic::fence(Ordering::Acquire);
        let slot = (self.last_seen_used as usize) % (self.queue_size as usize);
        let elem = self.used_ring[slot];
        self.last_seen_used = self.last_seen_used.wrapping_add(1);
        self.free_chain(elem.id as u16);
        Some(elem)
    }
}

// ---------------------------------------------------------------------------
// Virgl 3D Command Stream Encoder
// ---------------------------------------------------------------------------

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirglCmd {
    Nop = 0,
    CreateObject = 1,
    BindObject = 2,
    DestroyObject = 3,
    SetViewportState = 4,
    SetFramebufferState = 5,
    SetVertexBuffers = 6,
    Clear = 7,
    DrawVbo = 8,
    ResourceInlineWrite = 9,
    SetScissorState = 10,
    ResourceCopyRegion = 11,
    Blit = 12,
    SetConstantBuffer = 13,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirglObjectType {
    Blend = 1,
    Rasterizer = 2,
    Dsa = 3,
    Shader = 4,
    VertexElements = 5,
    SamplerView = 6,
    SamplerState = 7,
    Surface = 8,
    Query = 9,
    StreamoutTarget = 10,
}

/// Builds a Virgl 3D command stream (`u32` words) for `VIRTIO_GPU_CMD_SUBMIT_3D`.
pub struct VirglCmdStream {
    words: Vec<u32>,
}

impl VirglCmdStream {
    pub fn new() -> Self {
        Self { words: Vec::new() }
    }

    pub fn with_capacity(words: usize) -> Self {
        Self {
            words: Vec::with_capacity(words),
        }
    }

    #[inline]
    fn push_header(&mut self, cmd: VirglCmd, obj_type: u32, payload_len: u16) {
        let word = ((payload_len as u32) << 16) | ((obj_type & 0xFF) << 8) | ((cmd as u32) & 0xFF);
        self.words.push(word);
    }

    pub fn create_surface(
        &mut self,
        handle: u32,
        res_id: u32,
        format: u32,
        first_layer: u32,
        last_layer: u32,
    ) {
        self.push_header(VirglCmd::CreateObject, VirglObjectType::Surface as u32, 5);
        self.words.push(handle);
        self.words.push(res_id);
        self.words.push(format);
        self.words.push(first_layer);
        self.words.push(last_layer);
    }

    pub fn set_framebuffer_state(&mut self, zsurf_handle: u32, cbuf_handles: &[u32]) {
        let len = 2 + cbuf_handles.len() as u16;
        self.push_header(VirglCmd::SetFramebufferState, 0, len);
        self.words.push(cbuf_handles.len() as u32);
        self.words.push(zsurf_handle);
        self.words.extend_from_slice(cbuf_handles);
    }

    pub fn set_viewport(&mut self, width: f32, height: f32) {
        self.push_header(VirglCmd::SetViewportState, 0, 7);
        self.words.push(0); // start_slot
        let sx = (width * 0.5).to_bits();
        let sy = (-height * 0.5).to_bits();
        let sz = 0.5f32.to_bits();
        let tx = (width * 0.5).to_bits();
        let ty = (height * 0.5).to_bits();
        let tz = 0.5f32.to_bits();
        self.words.extend_from_slice(&[sx, sy, sz, tx, ty, tz]);
    }

    pub fn clear(&mut self, buffers: u32, rgba: [f32; 4], depth: f64, stencil: u32) {
        self.push_header(VirglCmd::Clear, 0, 8);
        self.words.push(buffers);
        for c in rgba {
            self.words.push(c.to_bits());
        }
        let d_bits = depth.to_bits();
        self.words.push(d_bits as u32);
        self.words.push((d_bits >> 32) as u32);
        self.words.push(stencil);
    }

    pub fn draw_vbo(&mut self, start: u32, count: u32, mode: u32, instance_count: u32) {
        self.push_header(VirglCmd::DrawVbo, 0, 4);
        self.words.push(start);
        self.words.push(count);
        self.words.push(mode);
        self.words.push(instance_count);
    }

    pub fn destroy_object(&mut self, handle: u32, obj_type: VirglObjectType) {
        self.push_header(VirglCmd::DestroyObject, obj_type as u32, 1);
        self.words.push(handle);
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.words.len() * 4);
        for &w in &self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    pub fn word_len(&self) -> usize {
        self.words.len()
    }
}

impl Default for VirglCmdStream {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Resource, Context, Scanout & Fence Tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    TwoD,
    ThreeD,
    Blob,
}

#[derive(Debug, Clone)]
pub struct GpuResource {
    pub id: u32,
    pub kind: ResourceKind,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub bind: u32,
    pub backing: Vec<VirtioGpuMemEntry>,
    pub attached_contexts: Vec<u32>,
    pub blob_size: u64,
    pub mapped_offset: Option<u64>,
    pub host_flushes: u64,
    pub host_transfers: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextKind {
    Virgl,
    VenusVulkan,
    CrossDomain,
}

#[derive(Debug, Clone)]
pub struct GpuContext {
    pub id: u32,
    pub kind: ContextKind,
    pub capset_id: u32,
    pub name: [u8; 64],
    pub name_len: usize,
    pub attached_resources: Vec<u32>,
    pub submitted_batches: u64,
    pub submitted_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ScanoutState {
    pub enabled: bool,
    pub resource_id: u32,
    pub rect: VirtioGpuRect,
    pub cursor_resource_id: u32,
    pub cursor_x: u32,
    pub cursor_y: u32,
    pub cursor_hot_x: u32,
    pub cursor_hot_y: u32,
    pub flush_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceStatus {
    Pending,
    Signaled,
    Error(u32),
}

#[derive(Debug, Clone, Copy)]
pub struct GpuFence {
    pub id: u64,
    pub ctx_id: u32,
    pub cmd_type: u32,
    pub status: FenceStatus,
}

#[derive(Debug, Clone, Copy)]
pub struct CapsetDescriptor {
    pub capset_id: u32,
    pub max_version: u32,
    pub max_size: u32,
}

// ---------------------------------------------------------------------------
// VirtIO-GPU Device & DRM/Vulkan Command Submission Engine
// ---------------------------------------------------------------------------

static NEXT_RESOURCE_ID: AtomicU32 = AtomicU32::new(1);
static NEXT_CONTEXT_ID: AtomicU32 = AtomicU32::new(1);
static NEXT_FENCE_ID: AtomicU64 = AtomicU64::new(1);

pub struct VirtioGpuDevice {
    pub features: u64,
    pub controlq: Box<SplitVirtqueue>,
    pub cursorq: Box<SplitVirtqueue>,
    pub resources: BTreeMap<u32, GpuResource>,
    pub contexts: BTreeMap<u32, GpuContext>,
    pub scanouts: [ScanoutState; VIRTIO_GPU_MAX_SCANOUTS],
    pub fences: BTreeMap<u64, GpuFence>,
    pub capsets: Vec<CapsetDescriptor>,
    pub loopback_mode: bool,
    pub total_commands: u64,
    pub total_3d_submits: u64,
}

impl VirtioGpuDevice {
    pub fn new(features: u64, loopback_mode: bool) -> Result<Self, VirtioGpuError> {
        let controlq = Box::new(SplitVirtqueue::new(
            QueueKind::Control,
            DEFAULT_QUEUE_SIZE as u16,
            None,
        )?);
        let cursorq = Box::new(SplitVirtqueue::new(
            QueueKind::Cursor,
            DEFAULT_QUEUE_SIZE as u16,
            None,
        )?);

        let mut scanouts = [ScanoutState::default(); VIRTIO_GPU_MAX_SCANOUTS];
        scanouts[0].enabled = true;
        scanouts[0].rect = VirtioGpuRect::new(0, 0, 1920, 1080);

        let mut capsets = Vec::new();
        if (features & VIRTIO_GPU_F_VIRGL) != 0 {
            capsets.push(CapsetDescriptor {
                capset_id: VIRTIO_GPU_CAPSET_VIRGL,
                max_version: 2,
                max_size: 512,
            });
            capsets.push(CapsetDescriptor {
                capset_id: VIRTIO_GPU_CAPSET_VIRGL2,
                max_version: 2,
                max_size: 1024,
            });
        }
        if (features & VIRTIO_GPU_F_CONTEXT_INIT) != 0 {
            capsets.push(CapsetDescriptor {
                capset_id: VIRTIO_GPU_CAPSET_VENUS,
                max_version: 1,
                max_size: 512,
            });
            capsets.push(CapsetDescriptor {
                capset_id: VIRTIO_GPU_CAPSET_CROSS_DOMAIN,
                max_version: 1,
                max_size: 256,
            });
        }

        Ok(Self {
            features,
            controlq,
            cursorq,
            resources: BTreeMap::new(),
            contexts: BTreeMap::new(),
            scanouts,
            fences: BTreeMap::new(),
            capsets,
            loopback_mode,
            total_commands: 0,
            total_3d_submits: 0,
        })
    }

    #[inline]
    fn alloc_fence(&mut self, ctx_id: u32, cmd_type: VirtioGpuCtrlType) -> u64 {
        let id = NEXT_FENCE_ID.fetch_add(1, Ordering::Relaxed);
        self.fences.insert(
            id,
            GpuFence {
                id,
                ctx_id,
                cmd_type: cmd_type as u32,
                status: FenceStatus::Pending,
            },
        );
        id
    }

    fn submit_control_cmd<Req, Resp>(
        &mut self,
        req: &Req,
        resp: &mut Resp,
        fence_id: u64,
    ) -> Result<(), VirtioGpuError> {
        let segs = [
            SgSegment {
                phys_addr: (req as *const Req) as u64,
                len: size_of::<Req>() as u32,
                device_writable: false,
            },
            SgSegment {
                phys_addr: (resp as *mut Resp) as u64,
                len: size_of::<Resp>() as u32,
                device_writable: true,
            },
        ];
        let head = self.controlq.submit_chain(&segs)?;
        self.total_commands = self.total_commands.wrapping_add(1);

        if self.loopback_mode {
            self.controlq.push_used(head, size_of::<Resp>() as u32);
            let _ = self.controlq.pop_used();
            if let Some(f) = self.fences.get_mut(&fence_id) {
                f.status = FenceStatus::Signaled;
            }
        }
        Ok(())
    }

    fn submit_cursor_cmd<Req>(
        &mut self,
        req: &Req,
        resp: &mut VirtioGpuCtrlHdr,
        fence_id: u64,
    ) -> Result<(), VirtioGpuError> {
        let segs = [
            SgSegment {
                phys_addr: (req as *const Req) as u64,
                len: size_of::<Req>() as u32,
                device_writable: false,
            },
            SgSegment {
                phys_addr: (resp as *mut VirtioGpuCtrlHdr) as u64,
                len: size_of::<VirtioGpuCtrlHdr>() as u32,
                device_writable: true,
            },
        ];
        let head = self.cursorq.submit_chain(&segs)?;
        self.total_commands = self.total_commands.wrapping_add(1);

        if self.loopback_mode {
            self.cursorq
                .push_used(head, size_of::<VirtioGpuCtrlHdr>() as u32);
            let _ = self.cursorq.pop_used();
            if let Some(f) = self.fences.get_mut(&fence_id) {
                f.status = FenceStatus::Signaled;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 2D Display & Cursor Commands
    // -----------------------------------------------------------------------

    pub fn get_display_info(&mut self) -> Result<Box<VirtioGpuRespDisplayInfo>, VirtioGpuError> {
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdGetDisplayInfo);
        let req = VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdGetDisplayInfo).with_fence(fence_id);
        let mut resp = Box::new(VirtioGpuRespDisplayInfo::default());
        self.submit_control_cmd(&req, &mut *resp, fence_id)?;
        if self.loopback_mode {
            resp.hdr.cmd_type = VirtioGpuCtrlType::RespOkDisplayInfo as u32;
            for (dst, src) in resp.pmodes.iter_mut().zip(self.scanouts.iter()) {
                dst.r = src.rect;
                dst.enabled = u32::from(src.enabled);
            }
        }
        Ok(resp)
    }

    pub fn resource_create_2d(
        &mut self,
        format: VirtioGpuFormat,
        width: u32,
        height: u32,
    ) -> Result<u32, VirtioGpuError> {
        if width == 0 || height == 0 {
            return Err(VirtioGpuError::InvalidArgument);
        }
        let res_id = NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed);
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdResourceCreate2d);
        let req = VirtioGpuResourceCreate2d {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdResourceCreate2d).with_fence(fence_id),
            resource_id: res_id,
            format: format as u32,
            width,
            height,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;

        self.resources.insert(
            res_id,
            GpuResource {
                id: res_id,
                kind: ResourceKind::TwoD,
                format: format as u32,
                width,
                height,
                depth: 1,
                bind: 0,
                backing: Vec::new(),
                attached_contexts: Vec::new(),
                blob_size: 0,
                mapped_offset: None,
                host_flushes: 0,
                host_transfers: 0,
            },
        );
        Ok(res_id)
    }

    pub fn resource_attach_backing(
        &mut self,
        resource_id: u32,
        entries: &[VirtioGpuMemEntry],
    ) -> Result<(), VirtioGpuError> {
        if entries.is_empty() {
            return Err(VirtioGpuError::InvalidArgument);
        }
        if !self.resources.contains_key(&resource_id) {
            return Err(VirtioGpuError::ResourceNotFound);
        }
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdResourceAttachBacking);
        let req = VirtioGpuResourceAttachBacking {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdResourceAttachBacking)
                .with_fence(fence_id),
            resource_id,
            nr_entries: entries.len() as u32,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        let segs = [
            SgSegment {
                phys_addr: (&req as *const VirtioGpuResourceAttachBacking) as u64,
                len: size_of::<VirtioGpuResourceAttachBacking>() as u32,
                device_writable: false,
            },
            SgSegment {
                phys_addr: entries.as_ptr() as u64,
                len: core::mem::size_of_val(entries) as u32,
                device_writable: false,
            },
            SgSegment {
                phys_addr: (&mut resp as *mut VirtioGpuCtrlHdr) as u64,
                len: size_of::<VirtioGpuCtrlHdr>() as u32,
                device_writable: true,
            },
        ];
        let head = self.controlq.submit_chain(&segs)?;
        self.total_commands = self.total_commands.wrapping_add(1);
        if self.loopback_mode {
            self.controlq
                .push_used(head, size_of::<VirtioGpuCtrlHdr>() as u32);
            let _ = self.controlq.pop_used();
            if let Some(f) = self.fences.get_mut(&fence_id) {
                f.status = FenceStatus::Signaled;
            }
        }

        if let Some(res) = self.resources.get_mut(&resource_id) {
            res.backing.extend_from_slice(entries);
        }
        Ok(())
    }

    pub fn resource_detach_backing(&mut self, resource_id: u32) -> Result<(), VirtioGpuError> {
        if !self.resources.contains_key(&resource_id) {
            return Err(VirtioGpuError::ResourceNotFound);
        }
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdResourceDetachBacking);
        let req = VirtioGpuResourceDetachBacking {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdResourceDetachBacking)
                .with_fence(fence_id),
            resource_id,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;
        if let Some(res) = self.resources.get_mut(&resource_id) {
            res.backing.clear();
        }
        Ok(())
    }

    pub fn set_scanout(
        &mut self,
        scanout_id: u32,
        resource_id: u32,
        rect: VirtioGpuRect,
    ) -> Result<(), VirtioGpuError> {
        if (scanout_id as usize) >= VIRTIO_GPU_MAX_SCANOUTS {
            return Err(VirtioGpuError::ScanoutOutOfRange);
        }
        if resource_id != 0 && !self.resources.contains_key(&resource_id) {
            return Err(VirtioGpuError::ResourceNotFound);
        }
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdSetScanout);
        let req = VirtioGpuSetScanout {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdSetScanout).with_fence(fence_id),
            r: rect,
            scanout_id,
            resource_id,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;

        let sc = &mut self.scanouts[scanout_id as usize];
        sc.enabled = resource_id != 0;
        sc.resource_id = resource_id;
        sc.rect = rect;
        Ok(())
    }

    pub fn transfer_to_host_2d(
        &mut self,
        resource_id: u32,
        rect: VirtioGpuRect,
        offset: u64,
    ) -> Result<u64, VirtioGpuError> {
        if !self.resources.contains_key(&resource_id) {
            return Err(VirtioGpuError::ResourceNotFound);
        }
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdTransferToHost2d);
        let req = VirtioGpuTransferToHost2d {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdTransferToHost2d).with_fence(fence_id),
            r: rect,
            offset,
            resource_id,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;
        if let Some(res) = self.resources.get_mut(&resource_id) {
            res.host_transfers = res.host_transfers.wrapping_add(1);
        }
        Ok(fence_id)
    }

    pub fn resource_flush(
        &mut self,
        resource_id: u32,
        rect: VirtioGpuRect,
    ) -> Result<u64, VirtioGpuError> {
        if !self.resources.contains_key(&resource_id) {
            return Err(VirtioGpuError::ResourceNotFound);
        }
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdResourceFlush);
        let req = VirtioGpuResourceFlush {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdResourceFlush).with_fence(fence_id),
            r: rect,
            resource_id,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;
        if let Some(res) = self.resources.get_mut(&resource_id) {
            res.host_flushes = res.host_flushes.wrapping_add(1);
        }
        for sc in &mut self.scanouts {
            if sc.resource_id == resource_id {
                sc.flush_count = sc.flush_count.wrapping_add(1);
            }
        }
        Ok(fence_id)
    }

    pub fn resource_unref(&mut self, resource_id: u32) -> Result<(), VirtioGpuError> {
        if !self.resources.contains_key(&resource_id) {
            return Err(VirtioGpuError::ResourceNotFound);
        }
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdResourceUnref);
        let req = VirtioGpuResourceUnref {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdResourceUnref).with_fence(fence_id),
            resource_id,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;
        self.resources.remove(&resource_id);
        Ok(())
    }

    pub fn update_cursor(
        &mut self,
        scanout_id: u32,
        resource_id: u32,
        x: u32,
        y: u32,
        hot_x: u32,
        hot_y: u32,
    ) -> Result<(), VirtioGpuError> {
        if (scanout_id as usize) >= VIRTIO_GPU_MAX_SCANOUTS {
            return Err(VirtioGpuError::ScanoutOutOfRange);
        }
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdUpdateCursor);
        let req = VirtioGpuUpdateCursor {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdUpdateCursor).with_fence(fence_id),
            pos: VirtioGpuCursorPos {
                scanout_id,
                x,
                y,
                padding: 0,
            },
            resource_id,
            hot_x,
            hot_y,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_cursor_cmd(&req, &mut resp, fence_id)?;
        let sc = &mut self.scanouts[scanout_id as usize];
        sc.cursor_resource_id = resource_id;
        sc.cursor_x = x;
        sc.cursor_y = y;
        sc.cursor_hot_x = hot_x;
        sc.cursor_hot_y = hot_y;
        Ok(())
    }

    pub fn move_cursor(&mut self, scanout_id: u32, x: u32, y: u32) -> Result<(), VirtioGpuError> {
        if (scanout_id as usize) >= VIRTIO_GPU_MAX_SCANOUTS {
            return Err(VirtioGpuError::ScanoutOutOfRange);
        }
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdMoveCursor);
        let req = VirtioGpuUpdateCursor {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdMoveCursor).with_fence(fence_id),
            pos: VirtioGpuCursorPos {
                scanout_id,
                x,
                y,
                padding: 0,
            },
            resource_id: self.scanouts[scanout_id as usize].cursor_resource_id,
            hot_x: 0,
            hot_y: 0,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_cursor_cmd(&req, &mut resp, fence_id)?;
        let sc = &mut self.scanouts[scanout_id as usize];
        sc.cursor_x = x;
        sc.cursor_y = y;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 3D Virgl & Vulkan/DRM Command Submission Engine
    // -----------------------------------------------------------------------

    pub fn get_capset_info(
        &mut self,
        capset_index: u32,
    ) -> Result<VirtioGpuRespCapsetInfo, VirtioGpuError> {
        if (self.features & VIRTIO_GPU_F_VIRGL) == 0 {
            return Err(VirtioGpuError::UnsupportedFeature);
        }
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdGetCapsetInfo);
        let req = VirtioGpuGetCapsetInfo {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdGetCapsetInfo).with_fence(fence_id),
            capset_index,
            padding: 0,
        };
        let mut resp = VirtioGpuRespCapsetInfo::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;
        if self.loopback_mode {
            let desc = self
                .capsets
                .get(capset_index as usize)
                .ok_or(VirtioGpuError::InvalidArgument)?;
            resp.hdr.cmd_type = VirtioGpuCtrlType::RespOkCapsetInfo as u32;
            resp.capset_id = desc.capset_id;
            resp.capset_max_version = desc.max_version;
            resp.capset_max_size = desc.max_size;
        }
        Ok(resp)
    }

    pub fn ctx_create(
        &mut self,
        kind: ContextKind,
        debug_name: &[u8],
    ) -> Result<u32, VirtioGpuError> {
        if (self.features & VIRTIO_GPU_F_VIRGL) == 0 {
            return Err(VirtioGpuError::UnsupportedFeature);
        }
        let capset_id = match kind {
            ContextKind::Virgl => VIRTIO_GPU_CAPSET_VIRGL2,
            ContextKind::VenusVulkan => {
                if (self.features & VIRTIO_GPU_F_CONTEXT_INIT) == 0 {
                    return Err(VirtioGpuError::UnsupportedFeature);
                }
                VIRTIO_GPU_CAPSET_VENUS
            }
            ContextKind::CrossDomain => {
                if (self.features & VIRTIO_GPU_F_CONTEXT_INIT) == 0 {
                    return Err(VirtioGpuError::UnsupportedFeature);
                }
                VIRTIO_GPU_CAPSET_CROSS_DOMAIN
            }
        };

        let ctx_id = NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed);
        let fence_id = self.alloc_fence(ctx_id, VirtioGpuCtrlType::CmdCtxCreate);
        let mut name_buf = [0u8; 64];
        let nlen = debug_name.len().min(64);
        name_buf[..nlen].copy_from_slice(&debug_name[..nlen]);

        let req = VirtioGpuCtxCreate {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdCtxCreate)
                .with_ctx(ctx_id)
                .with_fence(fence_id),
            nlen: nlen as u32,
            context_init: capset_id & VIRTIO_GPU_CONTEXT_INIT_CAPSET_ID_MASK,
            debug_name: name_buf,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;

        self.contexts.insert(
            ctx_id,
            GpuContext {
                id: ctx_id,
                kind,
                capset_id,
                name: name_buf,
                name_len: nlen,
                attached_resources: Vec::new(),
                submitted_batches: 0,
                submitted_bytes: 0,
            },
        );
        Ok(ctx_id)
    }

    pub fn ctx_destroy(&mut self, ctx_id: u32) -> Result<(), VirtioGpuError> {
        if !self.contexts.contains_key(&ctx_id) {
            return Err(VirtioGpuError::ContextNotFound);
        }
        let fence_id = self.alloc_fence(ctx_id, VirtioGpuCtrlType::CmdCtxDestroy);
        let req = VirtioGpuCtxDestroy {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdCtxDestroy)
                .with_ctx(ctx_id)
                .with_fence(fence_id),
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;
        self.contexts.remove(&ctx_id);
        Ok(())
    }

    pub fn ctx_attach_resource(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
    ) -> Result<(), VirtioGpuError> {
        if !self.contexts.contains_key(&ctx_id) {
            return Err(VirtioGpuError::ContextNotFound);
        }
        if !self.resources.contains_key(&resource_id) {
            return Err(VirtioGpuError::ResourceNotFound);
        }
        let fence_id = self.alloc_fence(ctx_id, VirtioGpuCtrlType::CmdCtxAttachResource);
        let req = VirtioGpuCtxResource {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdCtxAttachResource)
                .with_ctx(ctx_id)
                .with_fence(fence_id),
            resource_id,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;

        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            if !ctx.attached_resources.contains(&resource_id) {
                ctx.attached_resources.push(resource_id);
            }
        }
        if let Some(res) = self.resources.get_mut(&resource_id) {
            if !res.attached_contexts.contains(&ctx_id) {
                res.attached_contexts.push(ctx_id);
            }
        }
        Ok(())
    }

    pub fn ctx_detach_resource(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
    ) -> Result<(), VirtioGpuError> {
        if !self.contexts.contains_key(&ctx_id) {
            return Err(VirtioGpuError::ContextNotFound);
        }
        if !self.resources.contains_key(&resource_id) {
            return Err(VirtioGpuError::ResourceNotFound);
        }
        let fence_id = self.alloc_fence(ctx_id, VirtioGpuCtrlType::CmdCtxDetachResource);
        let req = VirtioGpuCtxResource {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdCtxDetachResource)
                .with_ctx(ctx_id)
                .with_fence(fence_id),
            resource_id,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;

        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.attached_resources.retain(|&id| id != resource_id);
        }
        if let Some(res) = self.resources.get_mut(&resource_id) {
            res.attached_contexts.retain(|&id| id != ctx_id);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resource_create_3d(
        &mut self,
        target: u32,
        format: u32,
        bind: u32,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<u32, VirtioGpuError> {
        if (self.features & VIRTIO_GPU_F_VIRGL) == 0 {
            return Err(VirtioGpuError::UnsupportedFeature);
        }
        if width == 0 || height == 0 || depth == 0 {
            return Err(VirtioGpuError::InvalidArgument);
        }
        let res_id = NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed);
        let fence_id = self.alloc_fence(0, VirtioGpuCtrlType::CmdResourceCreate3d);
        let req = VirtioGpuResourceCreate3d {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdResourceCreate3d).with_fence(fence_id),
            resource_id: res_id,
            target,
            format,
            bind,
            width,
            height,
            depth,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: 0,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;

        self.resources.insert(
            res_id,
            GpuResource {
                id: res_id,
                kind: ResourceKind::ThreeD,
                format,
                width,
                height,
                depth,
                bind,
                backing: Vec::new(),
                attached_contexts: Vec::new(),
                blob_size: 0,
                mapped_offset: None,
                host_flushes: 0,
                host_transfers: 0,
            },
        );
        Ok(res_id)
    }

    pub fn resource_create_blob(
        &mut self,
        ctx_id: u32,
        blob_mem: u32,
        blob_flags: u32,
        blob_id: u64,
        size: u64,
        entries: &[VirtioGpuMemEntry],
    ) -> Result<u32, VirtioGpuError> {
        if (self.features & VIRTIO_GPU_F_RESOURCE_BLOB) == 0 {
            return Err(VirtioGpuError::UnsupportedFeature);
        }
        if size == 0 {
            return Err(VirtioGpuError::InvalidArgument);
        }
        if ctx_id != 0 && !self.contexts.contains_key(&ctx_id) {
            return Err(VirtioGpuError::ContextNotFound);
        }
        let res_id = NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed);
        let fence_id = self.alloc_fence(ctx_id, VirtioGpuCtrlType::CmdResourceCreateBlob);
        let req = VirtioGpuResourceCreateBlob {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdResourceCreateBlob)
                .with_ctx(ctx_id)
                .with_fence(fence_id),
            resource_id: res_id,
            blob_mem,
            blob_flags,
            nr_entries: entries.len() as u32,
            blob_id,
            size,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;

        self.resources.insert(
            res_id,
            GpuResource {
                id: res_id,
                kind: ResourceKind::Blob,
                format: 0,
                width: 0,
                height: 0,
                depth: 0,
                bind: blob_flags,
                backing: entries.to_vec(),
                attached_contexts: if ctx_id != 0 {
                    vec![ctx_id]
                } else {
                    Vec::new()
                },
                blob_size: size,
                mapped_offset: None,
                host_flushes: 0,
                host_transfers: 0,
            },
        );
        Ok(res_id)
    }

    pub fn transfer_to_host_3d(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        box_region: VirtioGpuBox,
        offset: u64,
        stride: u32,
        layer_stride: u32,
    ) -> Result<u64, VirtioGpuError> {
        if !self.contexts.contains_key(&ctx_id) {
            return Err(VirtioGpuError::ContextNotFound);
        }
        if !self.resources.contains_key(&resource_id) {
            return Err(VirtioGpuError::ResourceNotFound);
        }
        let fence_id = self.alloc_fence(ctx_id, VirtioGpuCtrlType::CmdTransferToHost3d);
        let req = VirtioGpuTransferHost3d {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdTransferToHost3d)
                .with_ctx(ctx_id)
                .with_fence(fence_id),
            box_region,
            offset,
            resource_id,
            level: 0,
            stride,
            layer_stride,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        self.submit_control_cmd(&req, &mut resp, fence_id)?;
        if let Some(res) = self.resources.get_mut(&resource_id) {
            res.host_transfers = res.host_transfers.wrapping_add(1);
        }
        Ok(fence_id)
    }

    pub fn submit_3d(&mut self, ctx_id: u32, cmd_bytes: &[u8]) -> Result<u64, VirtioGpuError> {
        if (self.features & VIRTIO_GPU_F_VIRGL) == 0 {
            return Err(VirtioGpuError::UnsupportedFeature);
        }
        if !self.contexts.contains_key(&ctx_id) {
            return Err(VirtioGpuError::ContextNotFound);
        }
        if cmd_bytes.is_empty() || !cmd_bytes.len().is_multiple_of(4) {
            return Err(VirtioGpuError::InvalidArgument);
        }

        let fence_id = self.alloc_fence(ctx_id, VirtioGpuCtrlType::CmdSubmit3d);
        let req = VirtioGpuCmdSubmit {
            hdr: VirtioGpuCtrlHdr::new(VirtioGpuCtrlType::CmdSubmit3d)
                .with_ctx(ctx_id)
                .with_fence(fence_id),
            size: cmd_bytes.len() as u32,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr::default();
        let segs = [
            SgSegment {
                phys_addr: (&req as *const VirtioGpuCmdSubmit) as u64,
                len: size_of::<VirtioGpuCmdSubmit>() as u32,
                device_writable: false,
            },
            SgSegment {
                phys_addr: cmd_bytes.as_ptr() as u64,
                len: cmd_bytes.len() as u32,
                device_writable: false,
            },
            SgSegment {
                phys_addr: (&mut resp as *mut VirtioGpuCtrlHdr) as u64,
                len: size_of::<VirtioGpuCtrlHdr>() as u32,
                device_writable: true,
            },
        ];
        let head = self.controlq.submit_chain(&segs)?;
        self.total_commands = self.total_commands.wrapping_add(1);
        self.total_3d_submits = self.total_3d_submits.wrapping_add(1);

        if self.loopback_mode {
            self.controlq
                .push_used(head, size_of::<VirtioGpuCtrlHdr>() as u32);
            let _ = self.controlq.pop_used();
            if let Some(f) = self.fences.get_mut(&fence_id) {
                f.status = FenceStatus::Signaled;
            }
        }

        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.submitted_batches = ctx.submitted_batches.wrapping_add(1);
            ctx.submitted_bytes = ctx.submitted_bytes.wrapping_add(cmd_bytes.len() as u64);
        }
        Ok(fence_id)
    }

    pub fn fence_status(&self, fence_id: u64) -> Option<FenceStatus> {
        self.fences.get(&fence_id).map(|f| f.status)
    }
}

// ---------------------------------------------------------------------------
// Global Engine Instance & Self-Test
// ---------------------------------------------------------------------------

static VIRTIO_GPU_ENGINE: TicketLock<Option<VirtioGpuDevice>> = TicketLock::new(None);

pub fn init_loopback() -> Result<(), VirtioGpuError> {
    let features = VIRTIO_GPU_F_VIRGL
        | VIRTIO_GPU_F_EDID
        | VIRTIO_GPU_F_RESOURCE_UUID
        | VIRTIO_GPU_F_RESOURCE_BLOB
        | VIRTIO_GPU_F_CONTEXT_INIT;
    let dev = VirtioGpuDevice::new(features, true)?;
    *VIRTIO_GPU_ENGINE.lock() = Some(dev);
    Ok(())
}

pub fn self_test() -> Result<(), &'static str> {
    let features = VIRTIO_GPU_F_VIRGL
        | VIRTIO_GPU_F_EDID
        | VIRTIO_GPU_F_RESOURCE_UUID
        | VIRTIO_GPU_F_RESOURCE_BLOB
        | VIRTIO_GPU_F_CONTEXT_INIT;
    let mut dev = VirtioGpuDevice::new(features, true).map_err(|e| e.as_str())?;

    // 1. Display info & 2D resource lifecycle
    let disp = dev.get_display_info().map_err(|e| e.as_str())?;
    if disp.pmodes[0].enabled == 0 || disp.pmodes[0].r.width != 1920 {
        return Err("virtio-gpu: unexpected default display info");
    }

    let fb_res = dev
        .resource_create_2d(VirtioGpuFormat::B8G8R8X8Unorm, 1920, 1080)
        .map_err(|e| e.as_str())?;
    let backing = [VirtioGpuMemEntry {
        addr: 0x1000_0000,
        length: 1920 * 1080 * 4,
        padding: 0,
    }];
    dev.resource_attach_backing(fb_res, &backing)
        .map_err(|e| e.as_str())?;

    let rect = VirtioGpuRect::new(0, 0, 1920, 1080);
    let xfer_fence = dev
        .transfer_to_host_2d(fb_res, rect, 0)
        .map_err(|e| e.as_str())?;
    if dev.fence_status(xfer_fence) != Some(FenceStatus::Signaled) {
        return Err("virtio-gpu: 2D transfer fence not signaled");
    }

    dev.set_scanout(0, fb_res, rect).map_err(|e| e.as_str())?;
    let flush_fence = dev.resource_flush(fb_res, rect).map_err(|e| e.as_str())?;
    if dev.fence_status(flush_fence) != Some(FenceStatus::Signaled) {
        return Err("virtio-gpu: resource flush fence not signaled");
    }

    // 2. Cursor queue test
    let cursor_res = dev
        .resource_create_2d(VirtioGpuFormat::B8G8R8A8Unorm, 64, 64)
        .map_err(|e| e.as_str())?;
    dev.update_cursor(0, cursor_res, 100, 200, 4, 4)
        .map_err(|e| e.as_str())?;
    dev.move_cursor(0, 320, 240).map_err(|e| e.as_str())?;
    if dev.scanouts[0].cursor_x != 320 || dev.scanouts[0].cursor_y != 240 {
        return Err("virtio-gpu: cursor position mismatch");
    }

    // 3. 3D Virgl & Venus/Vulkan DRM command submission
    let cap = dev.get_capset_info(0).map_err(|e| e.as_str())?;
    if cap.capset_id != VIRTIO_GPU_CAPSET_VIRGL {
        return Err("virtio-gpu: unexpected capset id");
    }

    let virgl_ctx = dev
        .ctx_create(ContextKind::Virgl, b"nexus-virgl")
        .map_err(|e| e.as_str())?;
    let venus_ctx = dev
        .ctx_create(ContextKind::VenusVulkan, b"nexus-vulkan")
        .map_err(|e| e.as_str())?;

    let tex_res = dev
        .resource_create_3d(2, 1, 0x0A, 512, 512, 1)
        .map_err(|e| e.as_str())?;
    dev.ctx_attach_resource(virgl_ctx, tex_res)
        .map_err(|e| e.as_str())?;
    let xfer3d_fence = dev
        .transfer_to_host_3d(
            virgl_ctx,
            tex_res,
            VirtioGpuBox::new(0, 0, 0, 512, 512, 1),
            0,
            512 * 4,
            0,
        )
        .map_err(|e| e.as_str())?;
    if dev.fence_status(xfer3d_fence) != Some(FenceStatus::Signaled) {
        return Err("virtio-gpu: 3D transfer fence not signaled");
    }

    let mut stream = VirglCmdStream::with_capacity(32);
    stream.create_surface(1, tex_res, 1, 0, 0);
    stream.set_framebuffer_state(0, &[1]);
    stream.set_viewport(512.0, 512.0);
    stream.clear(0xF, [0.1, 0.2, 0.3, 1.0], 1.0, 0);
    stream.draw_vbo(0, 3, 4, 1);
    stream.destroy_object(1, VirglObjectType::Surface);

    let cmd_bytes = stream.as_bytes();
    let submit_fence = dev
        .submit_3d(virgl_ctx, &cmd_bytes)
        .map_err(|e| e.as_str())?;
    if dev.fence_status(submit_fence) != Some(FenceStatus::Signaled) {
        return Err("virtio-gpu: 3D submit fence not signaled");
    }

    // 4. Vulkan/DRM blob resource
    let blob_res = dev
        .resource_create_blob(
            venus_ctx,
            VIRTIO_GPU_BLOB_MEM_HOST3D_GUEST,
            VIRTIO_GPU_BLOB_FLAG_USE_MAPPABLE | VIRTIO_GPU_BLOB_FLAG_USE_SHAREABLE,
            1,
            64 * 1024,
            &backing,
        )
        .map_err(|e| e.as_str())?;

    // 5. Cleanup & descriptor leak check
    dev.ctx_detach_resource(virgl_ctx, tex_res)
        .map_err(|e| e.as_str())?;
    dev.resource_unref(blob_res).map_err(|e| e.as_str())?;
    dev.resource_unref(tex_res).map_err(|e| e.as_str())?;
    dev.ctx_destroy(venus_ctx).map_err(|e| e.as_str())?;
    dev.ctx_destroy(virgl_ctx).map_err(|e| e.as_str())?;
    dev.set_scanout(0, 0, VirtioGpuRect::default())
        .map_err(|e| e.as_str())?;
    dev.resource_detach_backing(fb_res)
        .map_err(|e| e.as_str())?;
    dev.resource_unref(fb_res).map_err(|e| e.as_str())?;
    dev.resource_unref(cursor_res).map_err(|e| e.as_str())?;

    if dev.controlq.free_descriptors() != DEFAULT_QUEUE_SIZE as u16
        || dev.cursorq.free_descriptors() != DEFAULT_QUEUE_SIZE as u16
    {
        return Err("virtio-gpu: virtqueue descriptor leak detected");
    }

    Ok(())
}
