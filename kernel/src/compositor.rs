//! Clean-room Kernel-Assisted Zero-Copy Wayland/Aqua Compositor & Damage-Tracking Engine.
//!
//! Provides:
//! - **DMA-BUF Surface Registry**: Zero-copy PRIME/DMA-BUF buffer descriptors, fence
//!   synchronization, Wayland-style pending/committed surface state, subsurface hierarchy,
//!   and direct-scanout eligibility detection.
//! - **Damage-Rect Clipping & Occlusion Culling**: Bounded dirty-rectangle accumulation,
//!   coalescing, output-bounds clamping, and opaque-region occlusion culling.
//! - **VSync Page-Flip Scheduler**: CRTC vblank tracking, frame-pacing deadlines, atomic
//!   page-flip queue with acquire-fence gating, and `wl_surface.frame` callback dispatch.
//! - **Aqua Glass & Shadow Pass**: Integer-only `no_std` Porter-Duff `SrcOver` blending,
//!   anti-aliased rounded-rectangle corner masking, separable box-blur frosted glass
//!   vibrancy, soft drop shadows, and 1px specular window rim highlights.
#![allow(dead_code)]

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::sync::TicketLock;

// ---------------------------------------------------------------------------
// Constants & IDs
// ---------------------------------------------------------------------------

pub const MAX_DAMAGE_RECTS: usize = 16;
pub const MAX_SURFACES: usize = 256;
pub const MAX_OUTPUTS: usize = 8;
pub const MAX_FRAME_CALLBACKS: usize = 64;
pub const DEFAULT_REFRESH_MHZ: u32 = 60_000;

static NEXT_DMABUF_ID: AtomicU32 = AtomicU32::new(1);
static NEXT_SURFACE_ID: AtomicU32 = AtomicU32::new(1);
static NEXT_FENCE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_CALLBACK_ID: AtomicU32 = AtomicU32::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositorError {
    InvalidArgument,
    NotFound,
    AlreadyExists,
    LimitReached,
    FenceNotSignaled,
    InvalidBuffer,
    OutputNotFound,
    FlipPending,
    OutOfMemory,
}

impl CompositorError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid argument",
            Self::NotFound => "object not found",
            Self::AlreadyExists => "object already exists",
            Self::LimitReached => "registry limit reached",
            Self::FenceNotSignaled => "acquire fence not signaled",
            Self::InvalidBuffer => "invalid DMA-BUF descriptor",
            Self::OutputNotFound => "output not found",
            Self::FlipPending => "page flip already pending",
            Self::OutOfMemory => "out of memory",
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry & Damage-Rect Clipping
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub const EMPTY: Self = Self {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    };

    #[inline]
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    #[inline]
    pub const fn right(&self) -> i64 {
        self.x as i64 + self.width as i64
    }

    #[inline]
    pub const fn bottom(&self) -> i64 {
        self.y as i64 + self.height as i64
    }

    #[inline]
    pub const fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    pub fn intersect(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::EMPTY;
        }
        let x0 = self.x.max(other.x) as i64;
        let y0 = self.y.max(other.y) as i64;
        let x1 = self.right().min(other.right());
        let y1 = self.bottom().min(other.bottom());
        if x1 <= x0 || y1 <= y0 {
            return Self::EMPTY;
        }
        Self {
            x: x0 as i32,
            y: y0 as i32,
            width: (x1 - x0) as u32,
            height: (y1 - y0) as u32,
        }
    }

    pub fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let x0 = (self.x as i64).min(other.x as i64);
        let y0 = (self.y as i64).min(other.y as i64);
        let x1 = self.right().max(other.right());
        let y1 = self.bottom().max(other.bottom());
        Self {
            x: x0 as i32,
            y: y0 as i32,
            width: (x1 - x0).max(0) as u32,
            height: (y1 - y0).max(0) as u32,
        }
    }

    #[inline]
    pub fn contains_rect(&self, other: &Self) -> bool {
        if other.is_empty() {
            return true;
        }
        if self.is_empty() {
            return false;
        }
        (other.x as i64) >= (self.x as i64)
            && (other.y as i64) >= (self.y as i64)
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    #[inline]
    pub fn translate(&self, dx: i32, dy: i32) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        Self {
            x: self.x.saturating_add(dx),
            y: self.y.saturating_add(dy),
            width: self.width,
            height: self.height,
        }
    }

    /// Expand rectangle uniformly on all sides (used for blur/shadow damage inflation).
    #[inline]
    pub fn inflate(&self, pad: u32) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        let p = pad as i32;
        Self {
            x: self.x.saturating_sub(p),
            y: self.y.saturating_sub(p),
            width: self.width.saturating_add(pad.saturating_mul(2)),
            height: self.height.saturating_add(pad.saturating_mul(2)),
        }
    }
}

/// Bounded damage-rectangle accumulator with coalescing and occlusion subtraction.
#[derive(Debug, Clone)]
pub struct DamageRegion {
    rects: [Rect; MAX_DAMAGE_RECTS],
    len: usize,
}

impl Default for DamageRegion {
    fn default() -> Self {
        Self::new()
    }
}

impl DamageRegion {
    pub const fn new() -> Self {
        Self {
            rects: [Rect::EMPTY; MAX_DAMAGE_RECTS],
            len: 0,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn as_slice(&self) -> &[Rect] {
        &self.rects[..self.len]
    }

    pub fn bounding_box(&self) -> Rect {
        let mut acc = Rect::EMPTY;
        for r in self.as_slice() {
            acc = acc.union(r);
        }
        acc
    }

    pub fn total_area(&self) -> u64 {
        let mut area = 0u64;
        for r in self.as_slice() {
            area = area.saturating_add(r.area());
        }
        area
    }

    /// Add a dirty rectangle, coalescing with an existing overlapping/contained rect
    /// or collapsing to the bounding box if capacity is exceeded.
    pub fn add(&mut self, rect: Rect) {
        if rect.is_empty() {
            return;
        }
        for existing in &mut self.rects[..self.len] {
            if existing.contains_rect(&rect) {
                return;
            }
            if rect.contains_rect(existing) {
                *existing = rect;
                return;
            }
            let merged = existing.union(&rect);
            // Coalesce if union adds <= 25% overhead over sum of areas.
            let sum_area = existing.area().saturating_add(rect.area());
            if !existing.intersect(&rect).is_empty() && merged.area() <= sum_area + (sum_area >> 2)
            {
                *existing = merged;
                return;
            }
        }
        if self.len < MAX_DAMAGE_RECTS {
            self.rects[self.len] = rect;
            self.len += 1;
        } else {
            let bbox = self.bounding_box().union(&rect);
            self.rects[0] = bbox;
            self.len = 1;
        }
    }

    /// Clip all damage rectangles to `bounds`, dropping empty results.
    pub fn clip_to(&mut self, bounds: &Rect) {
        let mut write_idx = 0usize;
        for read_idx in 0..self.len {
            let clipped = self.rects[read_idx].intersect(bounds);
            if !clipped.is_empty() {
                self.rects[write_idx] = clipped;
                write_idx += 1;
            }
        }
        self.len = write_idx;
    }

    /// Subtract an opaque occluder rectangle from all damage rectangles.
    /// Splits partially occluded rects into up to 4 non-overlapping bands.
    pub fn subtract_opaque(&mut self, occluder: &Rect) {
        if occluder.is_empty() || self.len == 0 {
            return;
        }
        let mut out = [Rect::EMPTY; MAX_DAMAGE_RECTS];
        let mut out_len = 0usize;

        for r in &self.rects[..self.len] {
            let inter = r.intersect(occluder);
            if inter.is_empty() {
                if out_len < MAX_DAMAGE_RECTS {
                    out[out_len] = *r;
                    out_len += 1;
                }
                continue;
            }
            if occluder.contains_rect(r) {
                continue;
            }

            // Top band
            if inter.y > r.y {
                let top = Rect::new(r.x, r.y, r.width, (inter.y - r.y) as u32);
                Self::push_or_merge(&mut out, &mut out_len, top);
            }
            // Bottom band
            if inter.bottom() < r.bottom() {
                let bot = Rect::new(
                    r.x,
                    inter.bottom() as i32,
                    r.width,
                    (r.bottom() - inter.bottom()) as u32,
                );
                Self::push_or_merge(&mut out, &mut out_len, bot);
            }
            // Left band (between inter.y and inter.bottom())
            if inter.x > r.x {
                let left = Rect::new(r.x, inter.y, (inter.x - r.x) as u32, inter.height);
                Self::push_or_merge(&mut out, &mut out_len, left);
            }
            // Right band
            if inter.right() < r.right() {
                let right = Rect::new(
                    inter.right() as i32,
                    inter.y,
                    (r.right() - inter.right()) as u32,
                    inter.height,
                );
                Self::push_or_merge(&mut out, &mut out_len, right);
            }
        }

        self.rects = out;
        self.len = out_len;
    }

    fn push_or_merge(out: &mut [Rect; MAX_DAMAGE_RECTS], out_len: &mut usize, rect: Rect) {
        if rect.is_empty() {
            return;
        }
        if *out_len < MAX_DAMAGE_RECTS {
            out[*out_len] = rect;
            *out_len += 1;
        } else {
            out[0] = out[0].union(&rect);
        }
    }
}

// ---------------------------------------------------------------------------
// DMA-BUF Surface Registry & Zero-Copy Buffer Descriptors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Argb8888,
    Xrgb8888,
    Abgr8888,
    Xbgr8888,
    Rgb565,
    Nv12,
}

impl PixelFormat {
    #[inline]
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Argb8888 | Self::Xrgb8888 | Self::Abgr8888 | Self::Xbgr8888 => 4,
            Self::Rgb565 => 2,
            Self::Nv12 => 1,
        }
    }

    #[inline]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Argb8888 | Self::Abgr8888)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferModifier {
    Linear,
    TiledX,
    TiledY,
    CompressedAfbc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncFence {
    pub id: u64,
    pub signaled: bool,
    pub timestamp_ns: u64,
}

impl SyncFence {
    pub fn new(signaled: bool) -> Self {
        Self {
            id: NEXT_FENCE_ID.fetch_add(1, Ordering::Relaxed),
            signaled,
            timestamp_ns: 0,
        }
    }

    #[inline]
    pub fn signal(&mut self, timestamp_ns: u64) {
        self.signaled = true;
        self.timestamp_ns = timestamp_ns;
    }
}

#[derive(Debug, Clone)]
pub struct DmaBufDescriptor {
    pub id: u32,
    pub owner_pid: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: PixelFormat,
    pub modifier: BufferModifier,
    pub phys_addr: u64,
    pub size_bytes: usize,
    pub ref_count: u32,
    pub acquire_fence: Option<SyncFence>,
}

impl DmaBufDescriptor {
    pub fn validate(&self) -> Result<(), CompositorError> {
        if self.width == 0 || self.height == 0 || self.phys_addr == 0 {
            return Err(CompositorError::InvalidBuffer);
        }
        let min_stride = self.width.saturating_mul(self.format.bytes_per_pixel());
        if self.stride < min_stride {
            return Err(CompositorError::InvalidBuffer);
        }
        let min_size = (self.stride as usize).saturating_mul(self.height as usize);
        if self.size_bytes < min_size {
            return Err(CompositorError::InvalidBuffer);
        }
        Ok(())
    }

    #[inline]
    pub fn is_fence_ready(&self) -> bool {
        match self.acquire_fence {
            Some(f) => f.signaled,
            None => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceRole {
    Toplevel,
    Popup,
    Subsurface,
    Cursor,
    MenuBar,
    Dock,
    Wallpaper,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlassMaterial {
    None,
    Light,
    Dark,
    UltraThin,
    Sidebar,
    MenuBar,
    Dock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AquaStyle {
    pub material: GlassMaterial,
    pub corner_radius: u16,
    pub opacity: u8,
    pub shadow_radius: u16,
    pub shadow_offset_y: i16,
    pub shadow_alpha: u8,
    pub specular_rim: bool,
}

impl Default for AquaStyle {
    fn default() -> Self {
        Self {
            material: GlassMaterial::None,
            corner_radius: 10,
            opacity: 255,
            shadow_radius: 12,
            shadow_offset_y: 4,
            shadow_alpha: 96,
            specular_rim: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SurfaceState {
    pub buffer_id: Option<u32>,
    pub position: (i32, i32),
    pub size: (u32, u32),
    pub scale: u32,
    pub z_order: i32,
    pub opaque_region: Option<Rect>,
    pub damage: DamageRegion,
    pub style: AquaStyle,
    pub visible: bool,
}

impl Default for SurfaceState {
    fn default() -> Self {
        Self {
            buffer_id: None,
            position: (0, 0),
            size: (0, 0),
            scale: 1,
            z_order: 0,
            opaque_region: None,
            damage: DamageRegion::new(),
            style: AquaStyle::default(),
            visible: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WaylandSurface {
    pub id: u32,
    pub owner_pid: u32,
    pub role: SurfaceRole,
    pub parent_id: Option<u32>,
    pub pending: SurfaceState,
    pub committed: SurfaceState,
    pub commit_serial: u64,
    pub pending_callbacks: Vec<u32>,
}

impl WaylandSurface {
    pub fn new(id: u32, owner_pid: u32, role: SurfaceRole, parent_id: Option<u32>) -> Self {
        Self {
            id,
            owner_pid,
            role,
            parent_id,
            pending: SurfaceState::default(),
            committed: SurfaceState::default(),
            commit_serial: 0,
            pending_callbacks: Vec::new(),
        }
    }

    #[inline]
    pub fn bounds(&self) -> Rect {
        Rect::new(
            self.committed.position.0,
            self.committed.position.1,
            self.committed.size.0,
            self.committed.size.1,
        )
    }

    #[inline]
    pub fn visual_bounds(&self) -> Rect {
        let base = self.bounds();
        if base.is_empty() {
            return Rect::EMPTY;
        }
        let pad = self.committed.style.shadow_radius as u32;
        let mut inflated = base.inflate(pad);
        if self.committed.style.shadow_offset_y > 0 {
            inflated.height = inflated
                .height
                .saturating_add(self.committed.style.shadow_offset_y as u32);
        }
        inflated
    }

    #[inline]
    pub fn is_fully_opaque(&self) -> bool {
        if !self.committed.visible || self.committed.style.opacity < 255 {
            return false;
        }
        if self.committed.style.material != GlassMaterial::None
            || self.committed.style.corner_radius > 0
        {
            return false;
        }
        match self.committed.opaque_region {
            Some(r) => r.contains_rect(&Rect::new(
                0,
                0,
                self.committed.size.0,
                self.committed.size.1,
            )),
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// VSync Page-Flip Scheduler
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipState {
    Idle,
    Scheduled,
    PendingVblank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameCallback {
    pub id: u32,
    pub surface_id: u32,
    pub armed_serial: u64,
}

#[derive(Debug, Clone)]
pub struct OutputState {
    pub id: u32,
    pub width: u32,
    pub height: u32,
    pub refresh_mhz: u32,
    pub vblank_period_ns: u64,
    pub vblank_seq: u64,
    pub last_vblank_ns: u64,
    pub front_dmabuf: Option<u32>,
    pub pending_dmabuf: Option<u32>,
    pub flip_state: FlipState,
    pub direct_scanout_surface: Option<u32>,
    pub vrr_enabled: bool,
    pub missed_vblanks: u64,
    pub total_flips: u64,
}

impl OutputState {
    pub fn new(id: u32, width: u32, height: u32, refresh_mhz: u32) -> Self {
        let hz = (refresh_mhz / 1000).max(1) as u64;
        let vblank_period_ns = 1_000_000_000 / hz;
        Self {
            id,
            width,
            height,
            refresh_mhz,
            vblank_period_ns,
            vblank_seq: 0,
            last_vblank_ns: 0,
            front_dmabuf: None,
            pending_dmabuf: None,
            flip_state: FlipState::Idle,
            direct_scanout_surface: None,
            vrr_enabled: false,
            missed_vblanks: 0,
            total_flips: 0,
        }
    }

    #[inline]
    pub fn bounds(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }

    #[inline]
    pub fn next_vblank_ns(&self) -> u64 {
        self.last_vblank_ns.saturating_add(self.vblank_period_ns)
    }

    /// Recommended compositor render deadline (2 ms before next vblank).
    #[inline]
    pub fn render_deadline_ns(&self) -> u64 {
        self.next_vblank_ns().saturating_sub(2_000_000)
    }
}

// ---------------------------------------------------------------------------
// Compositor Engine
// ---------------------------------------------------------------------------

pub struct CompositorEngine {
    dmabufs: BTreeMap<u32, DmaBufDescriptor>,
    surfaces: BTreeMap<u32, WaylandSurface>,
    outputs: BTreeMap<u32, OutputState>,
    frame_callbacks: Vec<FrameCallback>,
    output_damage: DamageRegion,
}

impl Default for CompositorEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DmaBufImport {
    pub owner_pid: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: PixelFormat,
    pub modifier: BufferModifier,
    pub phys_addr: u64,
    pub size_bytes: usize,
    pub acquire_fence: Option<SyncFence>,
}

impl CompositorEngine {
    pub const fn new() -> Self {
        Self {
            dmabufs: BTreeMap::new(),
            surfaces: BTreeMap::new(),
            outputs: BTreeMap::new(),
            frame_callbacks: Vec::new(),
            output_damage: DamageRegion::new(),
        }
    }

    pub fn register_output(
        &mut self,
        id: u32,
        width: u32,
        height: u32,
        refresh_mhz: u32,
    ) -> Result<(), CompositorError> {
        if width == 0 || height == 0 {
            return Err(CompositorError::InvalidArgument);
        }
        if self.outputs.len() >= MAX_OUTPUTS && !self.outputs.contains_key(&id) {
            return Err(CompositorError::LimitReached);
        }
        self.outputs
            .insert(id, OutputState::new(id, width, height, refresh_mhz));
        self.output_damage.add(Rect::new(0, 0, width, height));
        Ok(())
    }

    pub fn register_dmabuf(&mut self, import: DmaBufImport) -> Result<u32, CompositorError> {
        let id = NEXT_DMABUF_ID.fetch_add(1, Ordering::Relaxed);
        let desc = DmaBufDescriptor {
            id,
            owner_pid: import.owner_pid,
            width: import.width,
            height: import.height,
            stride: import.stride,
            format: import.format,
            modifier: import.modifier,
            phys_addr: import.phys_addr,
            size_bytes: import.size_bytes,
            ref_count: 1,
            acquire_fence: import.acquire_fence,
        };
        desc.validate()?;
        self.dmabufs.insert(id, desc);
        Ok(id)
    }

    pub fn signal_dmabuf_fence(
        &mut self,
        dmabuf_id: u32,
        timestamp_ns: u64,
    ) -> Result<(), CompositorError> {
        let buf = self
            .dmabufs
            .get_mut(&dmabuf_id)
            .ok_or(CompositorError::NotFound)?;
        if let Some(ref mut fence) = buf.acquire_fence {
            fence.signal(timestamp_ns);
        }
        Ok(())
    }

    pub fn release_dmabuf(&mut self, dmabuf_id: u32) -> Result<bool, CompositorError> {
        let buf = self
            .dmabufs
            .get_mut(&dmabuf_id)
            .ok_or(CompositorError::NotFound)?;
        if buf.ref_count > 1 {
            buf.ref_count -= 1;
            return Ok(false);
        }
        self.dmabufs.remove(&dmabuf_id);
        Ok(true)
    }

    pub fn create_surface(
        &mut self,
        owner_pid: u32,
        role: SurfaceRole,
        parent_id: Option<u32>,
    ) -> Result<u32, CompositorError> {
        if self.surfaces.len() >= MAX_SURFACES {
            return Err(CompositorError::LimitReached);
        }
        if let Some(pid) = parent_id {
            if !self.surfaces.contains_key(&pid) {
                return Err(CompositorError::NotFound);
            }
        }
        let id = NEXT_SURFACE_ID.fetch_add(1, Ordering::Relaxed);
        self.surfaces
            .insert(id, WaylandSurface::new(id, owner_pid, role, parent_id));
        Ok(id)
    }

    pub fn destroy_surface(&mut self, surface_id: u32) -> Result<(), CompositorError> {
        let surf = self
            .surfaces
            .remove(&surface_id)
            .ok_or(CompositorError::NotFound)?;
        self.output_damage.add(surf.visual_bounds());
        if let Some(buf_id) = surf.committed.buffer_id {
            let _ = self.release_dmabuf(buf_id);
        }
        self.frame_callbacks
            .retain(|cb| cb.surface_id != surface_id);
        Ok(())
    }

    pub fn attach_buffer(
        &mut self,
        surface_id: u32,
        dmabuf_id: u32,
    ) -> Result<(), CompositorError> {
        let (w, h) = {
            let buf = self
                .dmabufs
                .get_mut(&dmabuf_id)
                .ok_or(CompositorError::NotFound)?;
            buf.ref_count = buf.ref_count.saturating_add(1);
            (buf.width, buf.height)
        };
        let surf = self
            .surfaces
            .get_mut(&surface_id)
            .ok_or(CompositorError::NotFound)?;
        surf.pending.buffer_id = Some(dmabuf_id);
        surf.pending.size = (w, h);
        Ok(())
    }

    pub fn set_surface_geometry(
        &mut self,
        surface_id: u32,
        x: i32,
        y: i32,
        z_order: i32,
        opaque_region: Option<Rect>,
        style: AquaStyle,
    ) -> Result<(), CompositorError> {
        let surf = self
            .surfaces
            .get_mut(&surface_id)
            .ok_or(CompositorError::NotFound)?;
        surf.pending.position = (x, y);
        surf.pending.z_order = z_order;
        surf.pending.opaque_region = opaque_region;
        surf.pending.style = style;
        Ok(())
    }

    pub fn damage_surface(&mut self, surface_id: u32, rect: Rect) -> Result<(), CompositorError> {
        let surf = self
            .surfaces
            .get_mut(&surface_id)
            .ok_or(CompositorError::NotFound)?;
        surf.pending.damage.add(rect);
        Ok(())
    }

    pub fn request_frame_callback(&mut self, surface_id: u32) -> Result<u32, CompositorError> {
        if self.frame_callbacks.len() >= MAX_FRAME_CALLBACKS {
            return Err(CompositorError::LimitReached);
        }
        let surf = self
            .surfaces
            .get_mut(&surface_id)
            .ok_or(CompositorError::NotFound)?;
        let cb_id = NEXT_CALLBACK_ID.fetch_add(1, Ordering::Relaxed);
        surf.pending_callbacks.push(cb_id);
        Ok(cb_id)
    }

    /// Commit pending surface state, translate surface-local damage to output space,
    /// and arm any pending frame callbacks.
    pub fn commit_surface(&mut self, surface_id: u32) -> Result<u64, CompositorError> {
        let (old_buf, old_bounds, new_bounds, local_damage, callbacks, serial) = {
            let surf = self
                .surfaces
                .get_mut(&surface_id)
                .ok_or(CompositorError::NotFound)?;
            let old_buf = surf.committed.buffer_id;
            let old_bounds = surf.visual_bounds();

            surf.committed.buffer_id = surf.pending.buffer_id;
            surf.committed.position = surf.pending.position;
            surf.committed.size = surf.pending.size;
            surf.committed.scale = surf.pending.scale;
            surf.committed.z_order = surf.pending.z_order;
            surf.committed.opaque_region = surf.pending.opaque_region;
            surf.committed.style = surf.pending.style;
            surf.committed.visible = surf.pending.visible;

            let mut dmg = DamageRegion::new();
            for r in surf.pending.damage.as_slice() {
                dmg.add(*r);
            }
            surf.pending.damage.clear();
            surf.commit_serial = surf.commit_serial.saturating_add(1);

            let cbs = core::mem::take(&mut surf.pending_callbacks);
            (
                old_buf,
                old_bounds,
                surf.visual_bounds(),
                dmg,
                cbs,
                surf.commit_serial,
            )
        };

        if let Some(prev_id) = old_buf {
            if Some(prev_id)
                != self
                    .surfaces
                    .get(&surface_id)
                    .and_then(|s| s.committed.buffer_id)
            {
                let _ = self.release_dmabuf(prev_id);
            }
        }

        if old_bounds != new_bounds {
            self.output_damage.add(old_bounds);
            self.output_damage.add(new_bounds);
        } else if let Some(surf) = self.surfaces.get(&surface_id) {
            let (sx, sy) = surf.committed.position;
            let pad = surf.committed.style.shadow_radius as u32;
            for r in local_damage.as_slice() {
                self.output_damage.add(r.translate(sx, sy).inflate(pad));
            }
        }

        for cb_id in callbacks {
            if self.frame_callbacks.len() < MAX_FRAME_CALLBACKS {
                self.frame_callbacks.push(FrameCallback {
                    id: cb_id,
                    surface_id,
                    armed_serial: serial,
                });
            }
        }

        Ok(serial)
    }

    /// Compute clipped, occlusion-culled damage for a given output.
    pub fn compute_output_damage(&self, output_id: u32) -> Result<DamageRegion, CompositorError> {
        let out = self
            .outputs
            .get(&output_id)
            .ok_or(CompositorError::OutputNotFound)?;
        let mut region = self.output_damage.clone();
        region.clip_to(&out.bounds());

        // Collect visible surfaces sorted front-to-back (highest z_order first).
        let mut ordered: Vec<&WaylandSurface> = self
            .surfaces
            .values()
            .filter(|s| s.committed.visible && !s.bounds().is_empty())
            .collect();
        ordered.sort_by_key(|a| core::cmp::Reverse(a.committed.z_order));

        // If the topmost surface is fully opaque and covers the entire output,
        // any damage strictly behind it is occluded.
        if let Some(top) = ordered.first() {
            if top.is_fully_opaque() && top.bounds().contains_rect(&out.bounds()) {
                // Only damage within the top surface itself matters (already clipped to output).
                return Ok(region);
            }
        }

        Ok(region)
    }

    /// Compute per-surface visible damage after subtracting opaque regions of higher-z surfaces.
    pub fn compute_surface_visible_damage(
        &self,
        surface_id: u32,
        output_id: u32,
    ) -> Result<DamageRegion, CompositorError> {
        let out = self
            .outputs
            .get(&output_id)
            .ok_or(CompositorError::OutputNotFound)?;
        let target = self
            .surfaces
            .get(&surface_id)
            .ok_or(CompositorError::NotFound)?;
        if !target.committed.visible {
            return Ok(DamageRegion::new());
        }

        let mut dmg = self.output_damage.clone();
        dmg.clip_to(&target.visual_bounds());
        dmg.clip_to(&out.bounds());

        for other in self.surfaces.values() {
            if other.id == surface_id || !other.committed.visible {
                continue;
            }
            if other.committed.z_order > target.committed.z_order && other.is_fully_opaque() {
                dmg.subtract_opaque(&other.bounds());
            }
        }

        Ok(dmg)
    }

    /// Check if a single fullscreen opaque surface can bypass composition via zero-copy direct scanout.
    pub fn check_direct_scanout(
        &self,
        output_id: u32,
    ) -> Result<Option<(u32, u32)>, CompositorError> {
        let out = self
            .outputs
            .get(&output_id)
            .ok_or(CompositorError::OutputNotFound)?;

        let mut visible: Vec<&WaylandSurface> = self
            .surfaces
            .values()
            .filter(|s| s.committed.visible && !s.bounds().is_empty())
            .collect();
        visible.sort_by_key(|a| core::cmp::Reverse(a.committed.z_order));

        let Some(top) = visible.first() else {
            return Ok(None);
        };
        if !top.is_fully_opaque() {
            return Ok(None);
        }
        if top.bounds() != out.bounds() {
            return Ok(None);
        }
        let Some(buf_id) = top.committed.buffer_id else {
            return Ok(None);
        };
        let Some(buf) = self.dmabufs.get(&buf_id) else {
            return Ok(None);
        };
        if buf.width != out.width || buf.height != out.height || !buf.is_fence_ready() {
            return Ok(None);
        }
        Ok(Some((top.id, buf_id)))
    }

    /// Schedule an atomic page flip on `output_id` using `dmabuf_id`.
    pub fn schedule_page_flip(
        &mut self,
        output_id: u32,
        dmabuf_id: u32,
    ) -> Result<(), CompositorError> {
        let buf = self
            .dmabufs
            .get(&dmabuf_id)
            .ok_or(CompositorError::NotFound)?;
        if !buf.is_fence_ready() {
            return Err(CompositorError::FenceNotSignaled);
        }
        let out = self
            .outputs
            .get_mut(&output_id)
            .ok_or(CompositorError::OutputNotFound)?;
        if out.flip_state == FlipState::PendingVblank {
            return Err(CompositorError::FlipPending);
        }
        out.pending_dmabuf = Some(dmabuf_id);
        out.flip_state = FlipState::PendingVblank;
        Ok(())
    }

    /// Handle a hardware VSync / vblank interrupt for `output_id`.
    /// Completes any pending page flip, clears accumulated output damage, and returns
    /// the list of dispatched `FrameCallback` IDs.
    pub fn on_vblank(
        &mut self,
        output_id: u32,
        timestamp_ns: u64,
    ) -> Result<Vec<u32>, CompositorError> {
        let out = self
            .outputs
            .get_mut(&output_id)
            .ok_or(CompositorError::OutputNotFound)?;

        if out.last_vblank_ns > 0 && timestamp_ns > out.last_vblank_ns + out.vblank_period_ns * 2 {
            out.missed_vblanks = out.missed_vblanks.saturating_add(1);
        }
        out.vblank_seq = out.vblank_seq.saturating_add(1);
        out.last_vblank_ns = timestamp_ns;

        let mut flipped = false;
        if out.flip_state == FlipState::PendingVblank {
            if let Some(next_buf) = out.pending_dmabuf.take() {
                out.front_dmabuf = Some(next_buf);
                out.total_flips = out.total_flips.saturating_add(1);
                flipped = true;
            }
            out.flip_state = FlipState::Idle;
        }

        if flipped {
            self.output_damage.clear();
        }

        let dispatched: Vec<u32> = self.frame_callbacks.iter().map(|cb| cb.id).collect();
        self.frame_callbacks.clear();
        Ok(dispatched)
    }

    pub fn output_state(&self, output_id: u32) -> Option<&OutputState> {
        self.outputs.get(&output_id)
    }

    pub fn surface_count(&self) -> usize {
        self.surfaces.len()
    }

    pub fn dmabuf_count(&self) -> usize {
        self.dmabufs.len()
    }
}

// ---------------------------------------------------------------------------
// Aqua Glass, Shadow & Rounded-Rect Compositing Pass (no_std integer math)
// ---------------------------------------------------------------------------

/// Porter-Duff `SrcOver` blend of `src` (ARGB8888) over `dst` (ARGB8888) with extra `opacity` (0..=255).
#[inline]
pub fn blend_src_over(dst: u32, src: u32, opacity: u8) -> u32 {
    let sa = (((src >> 24) & 0xFF) * opacity as u32 + 127) / 255;
    if sa == 0 {
        return dst;
    }
    if sa == 255 {
        return src | 0xFF00_0000;
    }
    let inv_a = 255 - sa;
    let sr = (src >> 16) & 0xFF;
    let sg = (src >> 8) & 0xFF;
    let sb = src & 0xFF;

    let dr = (dst >> 16) & 0xFF;
    let dg = (dst >> 8) & 0xFF;
    let db = dst & 0xFF;

    let r = (sr * sa + dr * inv_a + 127) / 255;
    let g = (sg * sa + dg * inv_a + 127) / 255;
    let b = (sb * sa + db * inv_a + 127) / 255;

    0xFF00_0000 | (r << 16) | (g << 8) | b
}

/// Compute anti-aliased coverage (0..=255) for a pixel `(px, py)` inside a rounded rectangle
/// of size `(w, h)` with corner radius `radius`.
#[inline]
pub fn rounded_rect_coverage(px: u32, py: u32, w: u32, h: u32, radius: u16) -> u8 {
    if px >= w || py >= h {
        return 0;
    }
    let r = (radius as u32).min(w / 2).min(h / 2);
    if r == 0 {
        return 255;
    }

    let dx = if px < r {
        r - 1 - px
    } else if px >= w - r {
        px - (w - r)
    } else {
        return 255;
    };

    let dy = if py < r {
        r - 1 - py
    } else if py >= h - r {
        py - (h - r)
    } else {
        return 255;
    };

    let dist_sq = dx * dx + dy * dy;
    let r_sq = r * r;
    if dist_sq <= r_sq.saturating_sub(r) {
        255
    } else if dist_sq >= r_sq + r {
        0
    } else {
        // Linear anti-aliased falloff across the 1px boundary band.
        let span = (2 * r).max(1);
        let rem = (r_sq + r).saturating_sub(dist_sq);
        ((rem * 255) / span).min(255) as u8
    }
}

/// Separable box blur over `region` in an ARGB8888 framebuffer (`fb_w` x `fb_h`).
/// Uses a heap-allocated scratch buffer to keep kernel stack frames tiny.
pub fn aqua_box_blur_region(
    fb: &mut [u32],
    fb_w: u32,
    fb_h: u32,
    region: Rect,
    radius: u32,
) -> Result<(), CompositorError> {
    if radius == 0 || fb_w == 0 || fb_h == 0 {
        return Ok(());
    }
    let clipped = region.intersect(&Rect::new(0, 0, fb_w, fb_h));
    if clipped.is_empty() {
        return Ok(());
    }
    let expected_len = (fb_w as usize).saturating_mul(fb_h as usize);
    if fb.len() < expected_len {
        return Err(CompositorError::InvalidArgument);
    }

    let rw = clipped.width as usize;
    let rh = clipped.height as usize;
    let rx = clipped.x as usize;
    let ry = clipped.y as usize;
    let stride = fb_w as usize;
    let rad = radius.min(16) as isize;

    let mut scratch = vec![0u32; rw * rh];

    // Horizontal pass: fb -> scratch
    for row in 0..rh {
        let src_row = (ry + row) * stride;
        let dst_row = row * rw;
        for col in 0..rw {
            let mut sum_r = 0u32;
            let mut sum_g = 0u32;
            let mut sum_b = 0u32;
            let mut count = 0u32;
            let c_i = col as isize;
            let start = (c_i - rad).max(0) as usize;
            let end = (c_i + rad).min((rw - 1) as isize) as usize;
            for k in start..=end {
                let px = fb[src_row + rx + k];
                sum_r += (px >> 16) & 0xFF;
                sum_g += (px >> 8) & 0xFF;
                sum_b += px & 0xFF;
                count += 1;
            }
            scratch[dst_row + col] =
                0xFF00_0000 | ((sum_r / count) << 16) | ((sum_g / count) << 8) | (sum_b / count);
        }
    }

    // Vertical pass: scratch -> fb
    for row in 0..rh {
        let dst_row = (ry + row) * stride;
        let r_i = row as isize;
        let start = (r_i - rad).max(0) as usize;
        let end = (r_i + rad).min((rh - 1) as isize) as usize;
        for col in 0..rw {
            let mut sum_r = 0u32;
            let mut sum_g = 0u32;
            let mut sum_b = 0u32;
            let mut count = 0u32;
            for k in start..=end {
                let px = scratch[k * rw + col];
                sum_r += (px >> 16) & 0xFF;
                sum_g += (px >> 8) & 0xFF;
                sum_b += px & 0xFF;
                count += 1;
            }
            fb[dst_row + rx + col] =
                0xFF00_0000 | ((sum_r / count) << 16) | ((sum_g / count) << 8) | (sum_b / count);
        }
    }

    Ok(())
}

/// Apply Aqua frosted-glass vibrancy tint over `region` (after blur), clipped to `damage`.
pub fn aqua_glass_tint_pass(
    fb: &mut [u32],
    fb_w: u32,
    fb_h: u32,
    surface_rect: Rect,
    damage: Rect,
    style: &AquaStyle,
) {
    let (tint_rgb, tint_alpha) = match style.material {
        GlassMaterial::None => return,
        GlassMaterial::Light => (0x00F5_F7FAu32, 140u8),
        GlassMaterial::Dark => (0x001E_222Au32, 165u8),
        GlassMaterial::UltraThin => (0x00E8_EEF5u32, 90u8),
        GlassMaterial::Sidebar => (0x0026_2B36u32, 150u8),
        GlassMaterial::MenuBar => (0x0018_1C24u32, 175u8),
        GlassMaterial::Dock => (0x002A_303Cu32, 135u8),
    };

    let clip = surface_rect
        .intersect(&damage)
        .intersect(&Rect::new(0, 0, fb_w, fb_h));
    if clip.is_empty() {
        return;
    }

    let stride = fb_w as usize;
    let tint_src = ((tint_alpha as u32) << 24) | tint_rgb;

    for y in clip.y..(clip.y + clip.height as i32) {
        let ly = (y - surface_rect.y) as u32;
        let row_off = (y as usize) * stride;
        for x in clip.x..(clip.x + clip.width as i32) {
            let lx = (x - surface_rect.x) as u32;
            let cov = rounded_rect_coverage(
                lx,
                ly,
                surface_rect.width,
                surface_rect.height,
                style.corner_radius,
            );
            if cov == 0 {
                continue;
            }
            let idx = row_off + x as usize;
            let mut px = blend_src_over(fb[idx], tint_src, cov);
            if style.specular_rim && ly == 0 && cov > 128 {
                px = blend_src_over(px, 0x40FF_FFFF, 255);
            }
            fb[idx] = px;
        }
    }
}

/// Render a soft Aqua drop shadow behind `surface_rect`, clipped to `damage`.
pub fn aqua_drop_shadow_pass(
    fb: &mut [u32],
    fb_w: u32,
    fb_h: u32,
    surface_rect: Rect,
    damage: Rect,
    style: &AquaStyle,
) {
    if style.shadow_alpha == 0 || style.shadow_radius == 0 || surface_rect.is_empty() {
        return;
    }
    let rad = style.shadow_radius as i32;
    let shadow_rect = Rect::new(
        surface_rect.x - rad,
        surface_rect.y + style.shadow_offset_y as i32 - rad,
        surface_rect.width + (rad as u32) * 2,
        surface_rect.height + (rad as u32) * 2,
    );
    let clip = shadow_rect
        .intersect(&damage)
        .intersect(&Rect::new(0, 0, fb_w, fb_h));
    if clip.is_empty() {
        return;
    }

    let stride = fb_w as usize;
    let sx0 = surface_rect.x;
    let sy0 = surface_rect.y + style.shadow_offset_y as i32;
    let sx1 = sx0 + surface_rect.width as i32;
    let sy1 = sy0 + surface_rect.height as i32;

    for y in clip.y..(clip.y + clip.height as i32) {
        let dy = if y < sy0 {
            sy0 - y
        } else if y >= sy1 {
            y - sy1 + 1
        } else {
            0
        };
        let row_off = (y as usize) * stride;
        for x in clip.x..(clip.x + clip.width as i32) {
            // Skip interior of the surface itself when opaque/covered.
            if x >= surface_rect.x
                && x < surface_rect.x + surface_rect.width as i32
                && y >= surface_rect.y
                && y < surface_rect.y + surface_rect.height as i32
            {
                continue;
            }
            let dx = if x < sx0 {
                sx0 - x
            } else if x >= sx1 {
                x - sx1 + 1
            } else {
                0
            };
            let dist = dx.max(dy);
            if dist >= rad {
                continue;
            }
            let falloff = ((rad - dist) as u32 * style.shadow_alpha as u32) / (rad as u32);
            let idx = row_off + x as usize;
            fb[idx] = blend_src_over(fb[idx], 0xFF00_0000, falloff.min(255) as u8);
        }
    }
}

// ---------------------------------------------------------------------------
// Global Kernel Compositor Instance
// ---------------------------------------------------------------------------

pub static COMPOSITOR: TicketLock<CompositorEngine> = TicketLock::new(CompositorEngine::new());

// ---------------------------------------------------------------------------
// Self-Test
// ---------------------------------------------------------------------------

pub fn self_test() -> Result<(), &'static str> {
    let mut engine = CompositorEngine::new();
    engine
        .register_output(0, 1920, 1080, DEFAULT_REFRESH_MHZ)
        .map_err(|_| "register_output failed")?;

    // 1. DMA-BUF registration & fence gating
    let fence = SyncFence::new(false);
    let buf1 = engine
        .register_dmabuf(DmaBufImport {
            owner_pid: 100,
            width: 1920,
            height: 1080,
            stride: 1920 * 4,
            format: PixelFormat::Xrgb8888,
            modifier: BufferModifier::Linear,
            phys_addr: 0x1000_0000,
            size_bytes: 1920 * 1080 * 4,
            acquire_fence: Some(fence),
        })
        .map_err(|_| "register_dmabuf failed")?;

    if engine.schedule_page_flip(0, buf1).is_ok() {
        return Err("page flip should fail while acquire fence is unsignaled");
    }
    engine
        .signal_dmabuf_fence(buf1, 1_000_000)
        .map_err(|_| "signal_dmabuf_fence failed")?;

    // 2. Surface creation, direct scanout check, and damage clipping
    let surf1 = engine
        .create_surface(100, SurfaceRole::Toplevel, None)
        .map_err(|_| "create_surface failed")?;
    engine
        .attach_buffer(surf1, buf1)
        .map_err(|_| "attach_buffer failed")?;
    let opaque_style = AquaStyle {
        material: GlassMaterial::None,
        corner_radius: 0,
        opacity: 255,
        shadow_radius: 0,
        shadow_offset_y: 0,
        shadow_alpha: 0,
        specular_rim: false,
    };
    engine
        .set_surface_geometry(
            surf1,
            0,
            0,
            10,
            Some(Rect::new(0, 0, 1920, 1080)),
            opaque_style,
        )
        .map_err(|_| "set_surface_geometry failed")?;
    engine
        .damage_surface(surf1, Rect::new(100, 100, 200, 200))
        .map_err(|_| "damage_surface failed")?;
    let cb1 = engine
        .request_frame_callback(surf1)
        .map_err(|_| "request_frame_callback failed")?;
    engine
        .commit_surface(surf1)
        .map_err(|_| "commit_surface failed")?;

    let scanout = engine
        .check_direct_scanout(0)
        .map_err(|_| "check_direct_scanout failed")?;
    if scanout != Some((surf1, buf1)) {
        return Err("expected direct scanout eligibility for fullscreen opaque surface");
    }

    // 3. Occlusion culling: lower-z surface behind opaque fullscreen surf1 has 0 visible damage
    let surf2 = engine
        .create_surface(101, SurfaceRole::Toplevel, None)
        .map_err(|_| "create_surface 2 failed")?;
    engine
        .set_surface_geometry(surf2, 50, 50, 5, None, AquaStyle::default())
        .map_err(|_| "set_surface_geometry 2 failed")?;
    engine
        .commit_surface(surf2)
        .map_err(|_| "commit_surface 2 failed")?;
    let vis_dmg = engine
        .compute_surface_visible_damage(surf2, 0)
        .map_err(|_| "compute_surface_visible_damage failed")?;
    if !vis_dmg.is_empty() {
        return Err("occluded surface should have empty visible damage");
    }

    // 4. VSync page-flip & frame callback dispatch
    engine
        .schedule_page_flip(0, buf1)
        .map_err(|_| "schedule_page_flip failed")?;
    let dispatched = engine
        .on_vblank(0, 16_666_667)
        .map_err(|_| "on_vblank failed")?;
    if !dispatched.contains(&cb1) {
        return Err("frame callback not dispatched on vblank");
    }
    let out = engine.output_state(0).ok_or("missing output 0")?;
    if out.front_dmabuf != Some(buf1) || out.total_flips != 1 {
        return Err("page flip did not promote front buffer");
    }

    // 5. Aqua glass, blur, and drop-shadow pass on a compact 64x64 heap framebuffer
    let fb_w = 64u32;
    let fb_h = 64u32;
    let mut fb = vec![0xFF30_6090u32; (fb_w * fb_h) as usize];
    let win_rect = Rect::new(12, 12, 40, 40);
    let dmg = Rect::new(0, 0, fb_w, fb_h);
    let glass_style = AquaStyle {
        material: GlassMaterial::Dark,
        corner_radius: 8,
        opacity: 255,
        shadow_radius: 6,
        shadow_offset_y: 2,
        shadow_alpha: 120,
        specular_rim: true,
    };

    aqua_drop_shadow_pass(&mut fb, fb_w, fb_h, win_rect, dmg, &glass_style);
    aqua_box_blur_region(&mut fb, fb_w, fb_h, win_rect, 2)
        .map_err(|_| "aqua_box_blur_region failed")?;
    aqua_glass_tint_pass(&mut fb, fb_w, fb_h, win_rect, dmg, &glass_style);

    // Verify shadow darkened pixel just below window and glass tinted center pixel
    let shadow_px = fb[(53 * fb_w + 32) as usize];
    let center_px = fb[(32 * fb_w + 32) as usize];
    if shadow_px == 0xFF30_6090 || center_px == 0xFF30_6090 {
        return Err("Aqua glass/shadow pass did not modify framebuffer");
    }

    Ok(())
}
