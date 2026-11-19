//! Clean-room GPU Kernel Mode-Setting (KMS) & GEM/TTM VRAM Manager.
//!
//! Provides:
//! - **EDID 1.3/1.4 Parser**: Validates 128-byte base blocks, extracts manufacturer/product IDs,
//!   physical dimensions, established/standard timings, and 18-byte Detailed Timing Descriptors.
//! - **KMS Mode-Setting Pipeline**: CRTC, Encoder, Connector, Plane, and Framebuffer objects with
//!   routing validation, DPMS power states, vblank tracking, and atomic commit / rollback.
//! - **GEM/TTM Buffer Object & VRAM Manager**: Extent-based VRAM and GTT aperture allocators with
//!   coalescing, BO pinning, domain migration (VRAM <-> GTT <-> System), PRIME sharing, and
//!   reservation fence synchronization.
//! - **GPU Ring Submission Descriptors**: Multi-engine command rings (Graphics, Compute, CopyDma,
//!   VideoDecode) with fence-tracked submission and completion.
#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::sync::TicketLock;

// ---------------------------------------------------------------------------
// Constants & Error Types
// ---------------------------------------------------------------------------

pub const KMS_PAGE_SIZE: usize = 4096;
pub const EDID_BLOCK_SIZE: usize = 128;
pub const MAX_RING_SLOTS: usize = 64;

static NEXT_OBJECT_ID: AtomicU32 = AtomicU32::new(1);
static NEXT_FENCE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_PRIME_ID: AtomicU32 = AtomicU32::new(1);

#[inline]
fn alloc_obj_id() -> u32 {
    NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KmsError {
    InvalidArgument,
    NotFound,
    AlreadyExists,
    OutOfMemory,
    OutOfVram,
    OutOfGtt,
    Busy,
    Pinned,
    InvalidEdidHeader,
    InvalidEdidChecksum,
    InvalidEdidLength,
    InvalidMode,
    InvalidRouting,
    FramebufferTooSmall,
    RingFull,
    FenceTimeout,
    FenceError,
    Unsupported,
}

impl KmsError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid argument",
            Self::NotFound => "object not found",
            Self::AlreadyExists => "object already exists",
            Self::OutOfMemory => "out of system memory",
            Self::OutOfVram => "out of VRAM",
            Self::OutOfGtt => "out of GTT aperture",
            Self::Busy => "resource busy",
            Self::Pinned => "buffer object is pinned",
            Self::InvalidEdidHeader => "invalid EDID header",
            Self::InvalidEdidChecksum => "invalid EDID checksum",
            Self::InvalidEdidLength => "invalid EDID length",
            Self::InvalidMode => "invalid display mode",
            Self::InvalidRouting => "invalid CRTC/encoder/connector routing",
            Self::FramebufferTooSmall => "framebuffer smaller than mode",
            Self::RingFull => "GPU command ring full",
            Self::FenceTimeout => "fence wait timed out",
            Self::FenceError => "fence signaled with error",
            Self::Unsupported => "unsupported operation",
        }
    }
}

// ---------------------------------------------------------------------------
// Display Modes, Pixel Formats & EDID Parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeFlags {
    pub hsync_pos: bool,
    pub vsync_pos: bool,
    pub interlaced: bool,
    pub preferred: bool,
}

impl ModeFlags {
    pub const fn new(hsync_pos: bool, vsync_pos: bool, interlaced: bool, preferred: bool) -> Self {
        Self {
            hsync_pos,
            vsync_pos,
            interlaced,
            preferred,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayMode {
    pub clock_khz: u32,
    pub hdisplay: u16,
    pub hsync_start: u16,
    pub hsync_end: u16,
    pub htotal: u16,
    pub vdisplay: u16,
    pub vsync_start: u16,
    pub vsync_end: u16,
    pub vtotal: u16,
    pub refresh_hz: u16,
    pub flags: ModeFlags,
}

impl DisplayMode {
    pub const fn simple(width: u16, height: u16, refresh_hz: u16) -> Self {
        let htotal = width.saturating_add(160);
        let vtotal = height.saturating_add(45);
        let clock_khz = ((htotal as u32) * (vtotal as u32) * (refresh_hz as u32)) / 1000;
        Self {
            clock_khz,
            hdisplay: width,
            hsync_start: width.saturating_add(48),
            hsync_end: width.saturating_add(80),
            htotal,
            vdisplay: height,
            vsync_start: height.saturating_add(3),
            vsync_end: height.saturating_add(8),
            vtotal,
            refresh_hz,
            flags: ModeFlags::new(true, true, false, false),
        }
    }

    pub fn validate(&self) -> Result<(), KmsError> {
        if self.hdisplay == 0
            || self.vdisplay == 0
            || self.clock_khz == 0
            || self.hsync_start < self.hdisplay
            || self.hsync_end < self.hsync_start
            || self.htotal < self.hsync_end
            || self.vsync_start < self.vdisplay
            || self.vsync_end < self.vsync_start
            || self.vtotal < self.vsync_end
        {
            return Err(KmsError::InvalidMode);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Xrgb8888,
    Argb8888,
    Xbgr8888,
    Abgr8888,
    Rgb565,
}

impl PixelFormat {
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Xrgb8888 | Self::Argb8888 | Self::Xbgr8888 | Self::Abgr8888 => 4,
            Self::Rgb565 => 2,
        }
    }

    pub const fn bits_per_pixel(self) -> u32 {
        self.bytes_per_pixel() * 8
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdidInfo {
    pub manufacturer: [u8; 3],
    pub product_code: u16,
    pub serial_number: u32,
    pub manufacture_week: u8,
    pub manufacture_year: u16,
    pub version: u8,
    pub revision: u8,
    pub digital: bool,
    pub bits_per_color: u8,
    pub width_mm: u16,
    pub height_mm: u16,
    pub gamma_x100: u16,
    pub monitor_name: String,
    pub modes: Vec<DisplayMode>,
    pub extension_count: u8,
}

const EDID_HEADER: [u8; 8] = [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00];

/// Parse a 128-byte EDID 1.3/1.4 base block.
pub fn parse_edid(data: &[u8]) -> Result<EdidInfo, KmsError> {
    if data.len() < EDID_BLOCK_SIZE {
        return Err(KmsError::InvalidEdidLength);
    }
    let block = &data[..EDID_BLOCK_SIZE];
    if block[..8] != EDID_HEADER {
        return Err(KmsError::InvalidEdidHeader);
    }

    let mut sum: u8 = 0;
    for &b in block {
        sum = sum.wrapping_add(b);
    }
    if sum != 0 {
        return Err(KmsError::InvalidEdidChecksum);
    }

    let mfg_raw = u16::from_be_bytes([block[8], block[9]]);
    let c1 = (((mfg_raw >> 10) & 0x1F) as u8).saturating_add(b'A' - 1);
    let c2 = (((mfg_raw >> 5) & 0x1F) as u8).saturating_add(b'A' - 1);
    let c3 = ((mfg_raw & 0x1F) as u8).saturating_add(b'A' - 1);

    let product_code = u16::from_le_bytes([block[10], block[11]]);
    let serial_number = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);
    let manufacture_week = block[16];
    let manufacture_year = 1990u16.saturating_add(block[17] as u16);
    let version = block[18];
    let revision = block[19];

    let video_input = block[20];
    let digital = (video_input & 0x80) != 0;
    let bits_per_color = if digital {
        match (video_input >> 4) & 0x07 {
            1 => 6,
            2 => 8,
            3 => 10,
            4 => 12,
            5 => 14,
            6 => 16,
            _ => 8,
        }
    } else {
        8
    };

    let width_mm = (block[21] as u16) * 10;
    let height_mm = (block[22] as u16) * 10;
    let gamma_x100 = if block[23] == 0xFF {
        220
    } else {
        100u16.saturating_add(block[23] as u16)
    };

    let mut modes = Vec::new();
    let mut monitor_name = String::new();

    // Parse 4 x 18-byte Detailed Timing Descriptors (bytes 54..126).
    for i in 0..4 {
        let off = 54 + i * 18;
        let dtd = &block[off..off + 18];
        let pixel_clock_10khz = u16::from_le_bytes([dtd[0], dtd[1]]);
        if pixel_clock_10khz != 0 {
            if let Some(mode) = parse_dtd(dtd, i == 0) {
                modes.push(mode);
            }
        } else if dtd[3] == 0xFC {
            // Monitor Name descriptor (bytes 5..18, terminated by 0x0A or padded with 0x20).
            for &ch in &dtd[5..18] {
                if ch == 0x0A || ch == 0x00 {
                    break;
                }
                if ch.is_ascii_graphic() || ch == b' ' {
                    monitor_name.push(ch as char);
                }
            }
            while monitor_name.ends_with(' ') {
                monitor_name.pop();
            }
        }
    }

    // Established timings (bytes 35..37).
    let est1 = block[35];
    let est2 = block[36];
    if (est1 & 0x20) != 0 {
        push_unique_mode(&mut modes, DisplayMode::simple(640, 480, 60));
    }
    if (est1 & 0x01) != 0 {
        push_unique_mode(&mut modes, DisplayMode::simple(800, 600, 60));
    }
    if (est2 & 0x08) != 0 {
        push_unique_mode(&mut modes, DisplayMode::simple(1024, 768, 60));
    }

    // Standard timings (bytes 38..54, 8 x 2-byte entries).
    for i in 0..8 {
        let off = 38 + i * 2;
        let b0 = block[off];
        let b1 = block[off + 1];
        if b0 == 0x01 && b1 == 0x01 {
            continue;
        }
        if b0 == 0 {
            continue;
        }
        let width = ((b0 as u16) + 31) * 8;
        let height = match (b1 >> 6) & 0x03 {
            0 => (width * 10) / 16,
            1 => (width * 3) / 4,
            2 => (width * 4) / 5,
            _ => (width * 9) / 16,
        };
        let refresh = ((b1 & 0x3F) as u16) + 60;
        push_unique_mode(&mut modes, DisplayMode::simple(width, height, refresh));
    }

    Ok(EdidInfo {
        manufacturer: [c1, c2, c3],
        product_code,
        serial_number,
        manufacture_week,
        manufacture_year,
        version,
        revision,
        digital,
        bits_per_color,
        width_mm,
        height_mm,
        gamma_x100,
        monitor_name,
        modes,
        extension_count: block[126],
    })
}

fn push_unique_mode(modes: &mut Vec<DisplayMode>, mode: DisplayMode) {
    if !modes.iter().any(|m| {
        m.hdisplay == mode.hdisplay
            && m.vdisplay == mode.vdisplay
            && m.refresh_hz == mode.refresh_hz
    }) {
        modes.push(mode);
    }
}

fn parse_dtd(dtd: &[u8], preferred: bool) -> Option<DisplayMode> {
    let clock_khz = (u16::from_le_bytes([dtd[0], dtd[1]]) as u32) * 10;
    if clock_khz == 0 {
        return None;
    }

    let hactive = (dtd[2] as u16) | (((dtd[4] as u16) & 0xF0) << 4);
    let hblank = (dtd[3] as u16) | (((dtd[4] as u16) & 0x0F) << 8);
    let vactive = (dtd[5] as u16) | (((dtd[7] as u16) & 0xF0) << 4);
    let vblank = (dtd[6] as u16) | (((dtd[7] as u16) & 0x0F) << 8);

    let hsync_off = (dtd[8] as u16) | (((dtd[11] as u16) & 0xC0) << 2);
    let hsync_width = (dtd[9] as u16) | (((dtd[11] as u16) & 0x30) << 4);
    let vsync_off = ((dtd[10] as u16) >> 4) | (((dtd[11] as u16) & 0x0C) << 2);
    let vsync_width = ((dtd[10] as u16) & 0x0F) | (((dtd[11] as u16) & 0x03) << 4);

    let htotal = hactive.checked_add(hblank)?;
    let vtotal = vactive.checked_add(vblank)?;
    if htotal == 0 || vtotal == 0 {
        return None;
    }

    let hsync_start = hactive.checked_add(hsync_off)?;
    let hsync_end = hsync_start.checked_add(hsync_width)?;
    let vsync_start = vactive.checked_add(vsync_off)?;
    let vsync_end = vsync_start.checked_add(vsync_width)?;

    let total_pixels = (htotal as u64) * (vtotal as u64);
    let refresh_hz = ((clock_khz as u64 * 1000) / total_pixels) as u16;

    let flags_byte = dtd[17];
    let interlaced = (flags_byte & 0x80) != 0;
    let hsync_pos = (flags_byte & 0x02) != 0;
    let vsync_pos = (flags_byte & 0x04) != 0;

    let mode = DisplayMode {
        clock_khz,
        hdisplay: hactive,
        hsync_start,
        hsync_end,
        htotal,
        vdisplay: vactive,
        vsync_start,
        vsync_end,
        vtotal,
        refresh_hz: refresh_hz.max(1),
        flags: ModeFlags::new(hsync_pos, vsync_pos, interlaced, preferred),
    };
    mode.validate().ok()?;
    Some(mode)
}

// ---------------------------------------------------------------------------
// Fence Synchronization & Timeline
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FenceId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fence {
    pub id: FenceId,
    pub context: u64,
    pub seqno: u64,
    pub signaled: bool,
    pub error: Option<KmsError>,
}

pub struct FenceManager {
    fences: BTreeMap<u64, Fence>,
    timelines: BTreeMap<u64, u64>,
}

impl FenceManager {
    pub const fn new() -> Self {
        Self {
            fences: BTreeMap::new(),
            timelines: BTreeMap::new(),
        }
    }

    pub fn create_fence(&mut self, context: u64, seqno: u64) -> FenceId {
        let id = FenceId(NEXT_FENCE_ID.fetch_add(1, Ordering::Relaxed));
        let completed = self.timelines.get(&context).copied().unwrap_or(0);
        let signaled = seqno <= completed;
        self.fences.insert(
            id.0,
            Fence {
                id,
                context,
                seqno,
                signaled,
                error: None,
            },
        );
        id
    }

    pub fn signal_fence(&mut self, id: FenceId, error: Option<KmsError>) -> Result<(), KmsError> {
        let fence = self.fences.get_mut(&id.0).ok_or(KmsError::NotFound)?;
        fence.signaled = true;
        fence.error = error;
        let entry = self.timelines.entry(fence.context).or_insert(0);
        if fence.seqno > *entry {
            *entry = fence.seqno;
        }
        Ok(())
    }

    pub fn signal_timeline(&mut self, context: u64, seqno: u64) {
        let entry = self.timelines.entry(context).or_insert(0);
        if seqno > *entry {
            *entry = seqno;
        }
        for fence in self.fences.values_mut() {
            if fence.context == context && fence.seqno <= seqno {
                fence.signaled = true;
            }
        }
    }

    pub fn is_signaled(&self, id: FenceId) -> Result<bool, KmsError> {
        let fence = self.fences.get(&id.0).ok_or(KmsError::NotFound)?;
        if let Some(err) = fence.error {
            return Err(err);
        }
        Ok(fence.signaled)
    }

    pub fn wait_poll(&self, id: FenceId, max_spins: usize) -> Result<(), KmsError> {
        for _ in 0..max_spins {
            if self.is_signaled(id)? {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        if self.is_signaled(id)? {
            Ok(())
        } else {
            Err(KmsError::FenceTimeout)
        }
    }

    pub fn gc_signaled(&mut self, keep_recent: usize) {
        if self.fences.len() <= keep_recent {
            return;
        }
        let to_remove = self.fences.len() - keep_recent;
        let mut keys = Vec::new();
        for (&k, f) in &self.fences {
            if f.signaled {
                keys.push(k);
                if keys.len() >= to_remove {
                    break;
                }
            }
        }
        for k in keys {
            self.fences.remove(&k);
        }
    }
}

// ---------------------------------------------------------------------------
// VRAM & GTT Extent Allocator (GEM / TTM)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Extent {
    offset: u64,
    size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApertureStats {
    pub base: u64,
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub free_bytes: u64,
    pub free_extents: usize,
}

/// Coalescing extent allocator for VRAM and GTT apertures.
pub struct ExtentAllocator {
    base: u64,
    total_size: u64,
    used_bytes: u64,
    free_list: Vec<Extent>,
}

impl ExtentAllocator {
    pub fn new(base: u64, total_size: u64) -> Self {
        let mut free_list = Vec::new();
        if total_size > 0 {
            free_list.push(Extent {
                offset: 0,
                size: total_size,
            });
        }
        Self {
            base,
            total_size,
            used_bytes: 0,
            free_list,
        }
    }

    pub fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let align = align.max(KMS_PAGE_SIZE as u64);
        if !align.is_power_of_two() {
            return None;
        }

        for idx in 0..self.free_list.len() {
            let ext = self.free_list[idx];
            let aligned_off = (ext.offset + align - 1) & !(align - 1);
            let pad = aligned_off - ext.offset;
            if ext.size < pad.saturating_add(size) {
                continue;
            }

            let tail_off = aligned_off + size;
            let tail_size = (ext.offset + ext.size) - tail_off;

            self.free_list.remove(idx);
            if tail_size > 0 {
                self.free_list.insert(
                    idx,
                    Extent {
                        offset: tail_off,
                        size: tail_size,
                    },
                );
            }
            if pad > 0 {
                self.free_list.insert(
                    idx,
                    Extent {
                        offset: ext.offset,
                        size: pad,
                    },
                );
            }

            self.used_bytes += size;
            return Some(self.base + aligned_off);
        }
        None
    }

    pub fn free(&mut self, addr: u64, size: u64) -> Result<(), KmsError> {
        if size == 0 || addr < self.base {
            return Err(KmsError::InvalidArgument);
        }
        let offset = addr - self.base;
        if offset.saturating_add(size) > self.total_size {
            return Err(KmsError::InvalidArgument);
        }

        let mut pos = 0;
        while pos < self.free_list.len() && self.free_list[pos].offset < offset {
            pos += 1;
        }
        self.free_list.insert(pos, Extent { offset, size });
        self.used_bytes = self.used_bytes.saturating_sub(size);

        // Coalesce with next extent.
        if pos + 1 < self.free_list.len() {
            let cur = self.free_list[pos];
            let next = self.free_list[pos + 1];
            if cur.offset + cur.size == next.offset {
                self.free_list[pos].size += next.size;
                self.free_list.remove(pos + 1);
            }
        }
        // Coalesce with previous extent.
        if pos > 0 {
            let prev = self.free_list[pos - 1];
            let cur = self.free_list[pos];
            if prev.offset + prev.size == cur.offset {
                self.free_list[pos - 1].size += cur.size;
                self.free_list.remove(pos);
            }
        }
        Ok(())
    }

    pub fn stats(&self) -> ApertureStats {
        ApertureStats {
            base: self.base,
            total_bytes: self.total_size,
            used_bytes: self.used_bytes,
            free_bytes: self.total_size.saturating_sub(self.used_bytes),
            free_extents: self.free_list.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// GEM / TTM Buffer Objects
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryDomain {
    System,
    Gtt,
    Vram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    WriteBack,
    WriteCombine,
    Uncached,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoFlags {
    pub scanout: bool,
    pub cursor: bool,
    pub linear: bool,
    pub cpu_access: bool,
}

impl BoFlags {
    pub const fn default_render() -> Self {
        Self {
            scanout: false,
            cursor: false,
            linear: true,
            cpu_access: true,
        }
    }

    pub const fn scanout() -> Self {
        Self {
            scanout: true,
            cursor: false,
            linear: true,
            cpu_access: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BufferObject {
    pub handle: u32,
    pub size: usize,
    pub aligned_size: usize,
    pub domain: MemoryDomain,
    pub cache_mode: CacheMode,
    pub flags: BoFlags,
    pub vram_addr: Option<u64>,
    pub gtt_addr: Option<u64>,
    pub ref_count: u32,
    pub pin_count: u32,
    pub prime_id: Option<u32>,
    pub exclusive_fence: Option<FenceId>,
    pub shared_fences: Vec<FenceId>,
}

pub struct BoManager {
    vram: ExtentAllocator,
    gtt: ExtentAllocator,
    objects: BTreeMap<u32, BufferObject>,
    prime_map: BTreeMap<u32, u32>,
}

impl BoManager {
    pub fn new(vram_base: u64, vram_size: u64, gtt_base: u64, gtt_size: u64) -> Self {
        Self {
            vram: ExtentAllocator::new(vram_base, vram_size),
            gtt: ExtentAllocator::new(gtt_base, gtt_size),
            objects: BTreeMap::new(),
            prime_map: BTreeMap::new(),
        }
    }

    pub fn create_bo(
        &mut self,
        size: usize,
        domain: MemoryDomain,
        cache_mode: CacheMode,
        flags: BoFlags,
    ) -> Result<u32, KmsError> {
        if size == 0 {
            return Err(KmsError::InvalidArgument);
        }
        let aligned_size = size.div_ceil(KMS_PAGE_SIZE) * KMS_PAGE_SIZE;
        let align = if flags.scanout {
            64 * 1024
        } else {
            KMS_PAGE_SIZE as u64
        };

        let (vram_addr, gtt_addr) = match domain {
            MemoryDomain::Vram => {
                let addr = self
                    .vram
                    .alloc(aligned_size as u64, align)
                    .ok_or(KmsError::OutOfVram)?;
                (Some(addr), None)
            }
            MemoryDomain::Gtt => {
                let addr = self
                    .gtt
                    .alloc(aligned_size as u64, align)
                    .ok_or(KmsError::OutOfGtt)?;
                (None, Some(addr))
            }
            MemoryDomain::System => (None, None),
        };

        let handle = alloc_obj_id();
        self.objects.insert(
            handle,
            BufferObject {
                handle,
                size,
                aligned_size,
                domain,
                cache_mode,
                flags,
                vram_addr,
                gtt_addr,
                ref_count: 1,
                pin_count: 0,
                prime_id: None,
                exclusive_fence: None,
                shared_fences: Vec::new(),
            },
        );
        Ok(handle)
    }

    pub fn get(&self, handle: u32) -> Option<&BufferObject> {
        self.objects.get(&handle)
    }

    pub fn ref_bo(&mut self, handle: u32) -> Result<u32, KmsError> {
        let bo = self.objects.get_mut(&handle).ok_or(KmsError::NotFound)?;
        bo.ref_count = bo.ref_count.saturating_add(1);
        Ok(bo.ref_count)
    }

    pub fn unref_bo(&mut self, handle: u32) -> Result<bool, KmsError> {
        let bo = self.objects.get_mut(&handle).ok_or(KmsError::NotFound)?;
        if bo.ref_count > 1 {
            bo.ref_count -= 1;
            return Ok(false);
        }
        if bo.pin_count > 0 {
            return Err(KmsError::Pinned);
        }
        let removed = self.objects.remove(&handle).ok_or(KmsError::NotFound)?;
        if let Some(addr) = removed.vram_addr {
            let _ = self.vram.free(addr, removed.aligned_size as u64);
        }
        if let Some(addr) = removed.gtt_addr {
            let _ = self.gtt.free(addr, removed.aligned_size as u64);
        }
        if let Some(pid) = removed.prime_id {
            self.prime_map.remove(&pid);
        }
        Ok(true)
    }

    pub fn pin_bo(&mut self, handle: u32) -> Result<u32, KmsError> {
        let bo = self.objects.get_mut(&handle).ok_or(KmsError::NotFound)?;
        bo.pin_count = bo.pin_count.saturating_add(1);
        Ok(bo.pin_count)
    }

    pub fn unpin_bo(&mut self, handle: u32) -> Result<u32, KmsError> {
        let bo = self.objects.get_mut(&handle).ok_or(KmsError::NotFound)?;
        if bo.pin_count == 0 {
            return Err(KmsError::InvalidArgument);
        }
        bo.pin_count -= 1;
        Ok(bo.pin_count)
    }

    /// TTM-style domain migration (e.g. VRAM <-> GTT <-> System).
    pub fn migrate_bo(
        &mut self,
        handle: u32,
        target: MemoryDomain,
        fences: &FenceManager,
    ) -> Result<(), KmsError> {
        let bo = self.objects.get(&handle).ok_or(KmsError::NotFound)?;
        if bo.domain == target {
            return Ok(());
        }
        if bo.pin_count > 0 {
            return Err(KmsError::Pinned);
        }
        if let Some(fid) = bo.exclusive_fence {
            if !fences.is_signaled(fid)? {
                return Err(KmsError::Busy);
            }
        }
        for &fid in &bo.shared_fences {
            if !fences.is_signaled(fid)? {
                return Err(KmsError::Busy);
            }
        }

        let aligned_size = bo.aligned_size as u64;
        let align = if bo.flags.scanout {
            64 * 1024
        } else {
            KMS_PAGE_SIZE as u64
        };
        let old_vram = bo.vram_addr;
        let old_gtt = bo.gtt_addr;

        let (new_vram, new_gtt) = match target {
            MemoryDomain::Vram => {
                let addr = self
                    .vram
                    .alloc(aligned_size, align)
                    .ok_or(KmsError::OutOfVram)?;
                (Some(addr), None)
            }
            MemoryDomain::Gtt => {
                let addr = self
                    .gtt
                    .alloc(aligned_size, align)
                    .ok_or(KmsError::OutOfGtt)?;
                (None, Some(addr))
            }
            MemoryDomain::System => (None, None),
        };

        if let Some(addr) = old_vram {
            let _ = self.vram.free(addr, aligned_size);
        }
        if let Some(addr) = old_gtt {
            let _ = self.gtt.free(addr, aligned_size);
        }

        let bo_mut = self.objects.get_mut(&handle).ok_or(KmsError::NotFound)?;
        bo_mut.domain = target;
        bo_mut.vram_addr = new_vram;
        bo_mut.gtt_addr = new_gtt;
        Ok(())
    }

    pub fn attach_fence(
        &mut self,
        handle: u32,
        fence: FenceId,
        exclusive: bool,
    ) -> Result<(), KmsError> {
        let bo = self.objects.get_mut(&handle).ok_or(KmsError::NotFound)?;
        if exclusive {
            bo.exclusive_fence = Some(fence);
            bo.shared_fences.clear();
        } else {
            bo.shared_fences.push(fence);
        }
        Ok(())
    }

    pub fn prime_export(&mut self, handle: u32) -> Result<u32, KmsError> {
        let bo = self.objects.get_mut(&handle).ok_or(KmsError::NotFound)?;
        if let Some(pid) = bo.prime_id {
            return Ok(pid);
        }
        let pid = NEXT_PRIME_ID.fetch_add(1, Ordering::Relaxed);
        bo.prime_id = Some(pid);
        self.prime_map.insert(pid, handle);
        Ok(pid)
    }

    pub fn prime_import(&mut self, prime_id: u32) -> Result<u32, KmsError> {
        let handle = *self.prime_map.get(&prime_id).ok_or(KmsError::NotFound)?;
        self.ref_bo(handle)?;
        Ok(handle)
    }

    pub fn vram_stats(&self) -> ApertureStats {
        self.vram.stats()
    }

    pub fn gtt_stats(&self) -> ApertureStats {
        self.gtt.stats()
    }
}

// ---------------------------------------------------------------------------
// GPU Ring Submission Descriptors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingType {
    Graphics,
    Compute,
    CopyDma,
    VideoDecode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmdOpcode {
    Nop,
    ClearColor,
    BlitCopy,
    DrawIndexed,
    DispatchCompute,
    PageFlip,
    FenceSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingDescriptor {
    pub opcode: CmdOpcode,
    pub flags: u32,
    pub cmd_gpu_addr: u64,
    pub cmd_dwords: u32,
    pub seqno: u64,
    pub fence_id: FenceId,
}

pub struct CmdSubmission {
    pub opcode: CmdOpcode,
    pub flags: u32,
    pub cmd_gpu_addr: u64,
    pub cmd_dwords: u32,
    pub write_bos: Vec<u32>,
    pub read_bos: Vec<u32>,
}

pub struct GpuRing {
    pub ring_type: RingType,
    pub context_id: u64,
    pub ring_phys_addr: u64,
    pub doorbell_offset: u32,
    slots: Vec<Option<RingDescriptor>>,
    head: usize,
    tail: usize,
    count: usize,
    next_seqno: u64,
    completed_seqno: u64,
}

impl GpuRing {
    pub fn new(
        ring_type: RingType,
        context_id: u64,
        ring_phys_addr: u64,
        doorbell_offset: u32,
    ) -> Self {
        Self {
            ring_type,
            context_id,
            ring_phys_addr,
            doorbell_offset,
            slots: vec![None; MAX_RING_SLOTS],
            head: 0,
            tail: 0,
            count: 0,
            next_seqno: 1,
            completed_seqno: 0,
        }
    }

    pub fn free_slots(&self) -> usize {
        self.slots.len().saturating_sub(self.count)
    }

    pub fn pending_count(&self) -> usize {
        self.count
    }

    pub fn submit(
        &mut self,
        sub: &CmdSubmission,
        bos: &mut BoManager,
        fences: &mut FenceManager,
    ) -> Result<FenceId, KmsError> {
        if self.free_slots() == 0 {
            return Err(KmsError::RingFull);
        }
        for &h in &sub.write_bos {
            if bos.get(h).is_none() {
                return Err(KmsError::NotFound);
            }
        }
        for &h in &sub.read_bos {
            if bos.get(h).is_none() {
                return Err(KmsError::NotFound);
            }
        }

        let seqno = self.next_seqno;
        self.next_seqno += 1;
        let fence_id = fences.create_fence(self.context_id, seqno);

        for &h in &sub.write_bos {
            bos.attach_fence(h, fence_id, true)?;
        }
        for &h in &sub.read_bos {
            bos.attach_fence(h, fence_id, false)?;
        }

        let desc = RingDescriptor {
            opcode: sub.opcode,
            flags: sub.flags,
            cmd_gpu_addr: sub.cmd_gpu_addr,
            cmd_dwords: sub.cmd_dwords,
            seqno,
            fence_id,
        };
        self.slots[self.tail] = Some(desc);
        self.tail = (self.tail + 1) % self.slots.len();
        self.count += 1;

        Ok(fence_id)
    }

    pub fn complete_up_to(&mut self, seqno: u64, fences: &mut FenceManager) -> usize {
        let mut retired = 0;
        while self.count > 0 {
            let Some(desc) = self.slots[self.head] else {
                break;
            };
            if desc.seqno > seqno {
                break;
            }
            self.slots[self.head] = None;
            self.head = (self.head + 1) % self.slots.len();
            self.count -= 1;
            if desc.seqno > self.completed_seqno {
                self.completed_seqno = desc.seqno;
            }
            let _ = fences.signal_fence(desc.fence_id, None);
            retired += 1;
        }
        fences.signal_timeline(self.context_id, self.completed_seqno);
        retired
    }
}

// ---------------------------------------------------------------------------
// KMS Objects: Framebuffer, Plane, CRTC, Encoder, Connector
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Framebuffer {
    pub id: u32,
    pub bo_handle: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub format: PixelFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneType {
    Primary,
    Cursor,
    Overlay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    pub id: u32,
    pub plane_type: PlaneType,
    pub possible_crtcs_mask: u32,
    pub crtc_id: Option<u32>,
    pub fb_id: Option<u32>,
    pub crtc_x: i32,
    pub crtc_y: i32,
    pub crtc_w: u32,
    pub crtc_h: u32,
    pub zpos: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crtc {
    pub id: u32,
    pub index: u8,
    pub active: bool,
    pub mode: Option<DisplayMode>,
    pub fb_id: Option<u32>,
    pub x: u32,
    pub y: u32,
    pub vblank_count: u64,
    pub page_flip_pending: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderType {
    Dac,
    Tmds,
    Lvds,
    DpMst,
    Dsi,
    Virtual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Encoder {
    pub id: u32,
    pub encoder_type: EncoderType,
    pub possible_crtcs_mask: u32,
    pub crtc_id: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectorType {
    Vga,
    DviD,
    HdmiA,
    DisplayPort,
    Edp,
    Virtual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectorStatus {
    Connected,
    Disconnected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpmsState {
    On,
    Standby,
    Suspend,
    Off,
}

#[derive(Debug, Clone)]
pub struct Connector {
    pub id: u32,
    pub connector_type: ConnectorType,
    pub status: ConnectorStatus,
    pub dpms: DpmsState,
    pub possible_encoders: Vec<u32>,
    pub encoder_id: Option<u32>,
    pub modes: Vec<DisplayMode>,
    pub edid: Option<EdidInfo>,
}

#[derive(Debug, Clone)]
pub struct ModeSetRequest {
    pub crtc_id: u32,
    pub connector_id: u32,
    pub encoder_id: u32,
    pub fb_id: u32,
    pub mode: DisplayMode,
    pub x: u32,
    pub y: u32,
}

pub struct KmsDevice {
    pub crtcs: BTreeMap<u32, Crtc>,
    pub encoders: BTreeMap<u32, Encoder>,
    pub connectors: BTreeMap<u32, Connector>,
    pub planes: BTreeMap<u32, Plane>,
    pub framebuffers: BTreeMap<u32, Framebuffer>,
    pub bos: BoManager,
    pub fences: FenceManager,
    pub rings: Vec<GpuRing>,
}

impl KmsDevice {
    pub fn new(vram_base: u64, vram_size: u64, gtt_base: u64, gtt_size: u64) -> Self {
        let rings = vec![
            GpuRing::new(RingType::Graphics, 1, vram_base, 0x1000),
            GpuRing::new(RingType::Compute, 2, vram_base + 0x1000, 0x1004),
            GpuRing::new(RingType::CopyDma, 3, vram_base + 0x2000, 0x1008),
        ];
        Self {
            crtcs: BTreeMap::new(),
            encoders: BTreeMap::new(),
            connectors: BTreeMap::new(),
            planes: BTreeMap::new(),
            framebuffers: BTreeMap::new(),
            bos: BoManager::new(vram_base, vram_size, gtt_base, gtt_size),
            fences: FenceManager::new(),
            rings,
        }
    }

    pub fn add_crtc(&mut self, index: u8) -> u32 {
        let id = alloc_obj_id();
        self.crtcs.insert(
            id,
            Crtc {
                id,
                index,
                active: false,
                mode: None,
                fb_id: None,
                x: 0,
                y: 0,
                vblank_count: 0,
                page_flip_pending: false,
            },
        );
        id
    }

    pub fn add_encoder(&mut self, encoder_type: EncoderType, possible_crtcs_mask: u32) -> u32 {
        let id = alloc_obj_id();
        self.encoders.insert(
            id,
            Encoder {
                id,
                encoder_type,
                possible_crtcs_mask,
                crtc_id: None,
            },
        );
        id
    }

    pub fn add_connector(
        &mut self,
        connector_type: ConnectorType,
        possible_encoders: Vec<u32>,
    ) -> u32 {
        let id = alloc_obj_id();
        self.connectors.insert(
            id,
            Connector {
                id,
                connector_type,
                status: ConnectorStatus::Disconnected,
                dpms: DpmsState::Off,
                possible_encoders,
                encoder_id: None,
                modes: Vec::new(),
                edid: None,
            },
        );
        id
    }

    pub fn add_plane(&mut self, plane_type: PlaneType, possible_crtcs_mask: u32, zpos: u8) -> u32 {
        let id = alloc_obj_id();
        self.planes.insert(
            id,
            Plane {
                id,
                plane_type,
                possible_crtcs_mask,
                crtc_id: None,
                fb_id: None,
                crtc_x: 0,
                crtc_y: 0,
                crtc_w: 0,
                crtc_h: 0,
                zpos,
            },
        );
        id
    }

    pub fn attach_edid(&mut self, connector_id: u32, raw_edid: &[u8]) -> Result<(), KmsError> {
        let info = parse_edid(raw_edid)?;
        let conn = self
            .connectors
            .get_mut(&connector_id)
            .ok_or(KmsError::NotFound)?;
        conn.modes = info.modes.clone();
        conn.edid = Some(info);
        conn.status = ConnectorStatus::Connected;
        Ok(())
    }

    pub fn create_framebuffer(
        &mut self,
        bo_handle: u32,
        width: u32,
        height: u32,
        pitch: u32,
        format: PixelFormat,
    ) -> Result<u32, KmsError> {
        let min_pitch = width.saturating_mul(format.bytes_per_pixel());
        if width == 0 || height == 0 || pitch < min_pitch {
            return Err(KmsError::InvalidArgument);
        }
        let required_bytes = (pitch as usize).saturating_mul(height as usize);
        let bo = self.bos.get(bo_handle).ok_or(KmsError::NotFound)?;
        if bo.size < required_bytes {
            return Err(KmsError::FramebufferTooSmall);
        }
        self.bos.ref_bo(bo_handle)?;
        let id = alloc_obj_id();
        self.framebuffers.insert(
            id,
            Framebuffer {
                id,
                bo_handle,
                width,
                height,
                pitch,
                format,
            },
        );
        Ok(id)
    }

    pub fn destroy_framebuffer(&mut self, fb_id: u32) -> Result<(), KmsError> {
        for crtc in self.crtcs.values() {
            if crtc.fb_id == Some(fb_id) {
                return Err(KmsError::Busy);
            }
        }
        let fb = self.framebuffers.remove(&fb_id).ok_or(KmsError::NotFound)?;
        let _ = self.bos.unref_bo(fb.bo_handle);
        Ok(())
    }

    /// Validate and atomically apply a mode-set configuration.
    pub fn atomic_modeset(&mut self, req: &ModeSetRequest) -> Result<(), KmsError> {
        req.mode.validate()?;

        let crtc = self.crtcs.get(&req.crtc_id).ok_or(KmsError::NotFound)?;
        let crtc_bit = 1u32.checked_shl(crtc.index as u32).unwrap_or(0);

        let enc = self
            .encoders
            .get(&req.encoder_id)
            .ok_or(KmsError::NotFound)?;
        if (enc.possible_crtcs_mask & crtc_bit) == 0 {
            return Err(KmsError::InvalidRouting);
        }

        let conn = self
            .connectors
            .get(&req.connector_id)
            .ok_or(KmsError::NotFound)?;
        if !conn.possible_encoders.contains(&req.encoder_id) {
            return Err(KmsError::InvalidRouting);
        }

        let fb = self
            .framebuffers
            .get(&req.fb_id)
            .ok_or(KmsError::NotFound)?;
        if fb.width < req.x.saturating_add(req.mode.hdisplay as u32)
            || fb.height < req.y.saturating_add(req.mode.vdisplay as u32)
        {
            return Err(KmsError::FramebufferTooSmall);
        }

        let old_fb_bo = crtc
            .fb_id
            .and_then(|id| self.framebuffers.get(&id))
            .map(|f| f.bo_handle);
        let new_fb_bo = fb.bo_handle;

        // Pin new scanout BO before committing.
        self.bos.pin_bo(new_fb_bo)?;

        // Commit state.
        if let Some(c) = self.crtcs.get_mut(&req.crtc_id) {
            c.active = true;
            c.mode = Some(req.mode);
            c.fb_id = Some(req.fb_id);
            c.x = req.x;
            c.y = req.y;
        }
        if let Some(e) = self.encoders.get_mut(&req.encoder_id) {
            e.crtc_id = Some(req.crtc_id);
        }
        if let Some(cn) = self.connectors.get_mut(&req.connector_id) {
            cn.encoder_id = Some(req.encoder_id);
            cn.dpms = DpmsState::On;
        }
        for plane in self.planes.values_mut() {
            if plane.plane_type == PlaneType::Primary && (plane.possible_crtcs_mask & crtc_bit) != 0
            {
                plane.crtc_id = Some(req.crtc_id);
                plane.fb_id = Some(req.fb_id);
                plane.crtc_w = req.mode.hdisplay as u32;
                plane.crtc_h = req.mode.vdisplay as u32;
                break;
            }
        }

        if let Some(old_bo) = old_fb_bo {
            if old_bo != new_fb_bo {
                let _ = self.bos.unpin_bo(old_bo);
            }
        }
        Ok(())
    }

    pub fn page_flip(&mut self, crtc_id: u32, new_fb_id: u32) -> Result<FenceId, KmsError> {
        let crtc = self.crtcs.get(&crtc_id).ok_or(KmsError::NotFound)?;
        if !crtc.active {
            return Err(KmsError::InvalidArgument);
        }
        let mode = crtc.mode.ok_or(KmsError::InvalidMode)?;
        let new_fb = self
            .framebuffers
            .get(&new_fb_id)
            .ok_or(KmsError::NotFound)?;
        if new_fb.width < mode.hdisplay as u32 || new_fb.height < mode.vdisplay as u32 {
            return Err(KmsError::FramebufferTooSmall);
        }

        let new_bo = new_fb.bo_handle;
        let old_bo = crtc
            .fb_id
            .and_then(|id| self.framebuffers.get(&id))
            .map(|f| f.bo_handle);
        self.bos.pin_bo(new_bo)?;

        let sub = CmdSubmission {
            opcode: CmdOpcode::PageFlip,
            flags: 0,
            cmd_gpu_addr: 0,
            cmd_dwords: 4,
            write_bos: vec![new_bo],
            read_bos: Vec::new(),
        };
        let fence = self.rings[0].submit(&sub, &mut self.bos, &mut self.fences)?;

        if let Some(c) = self.crtcs.get_mut(&crtc_id) {
            c.fb_id = Some(new_fb_id);
            c.page_flip_pending = true;
        }
        if let Some(old) = old_bo {
            if old != new_bo {
                let _ = self.bos.unpin_bo(old);
            }
        }
        Ok(fence)
    }

    pub fn handle_vblank(&mut self, crtc_id: u32) -> Result<(), KmsError> {
        let crtc = self.crtcs.get_mut(&crtc_id).ok_or(KmsError::NotFound)?;
        crtc.vblank_count = crtc.vblank_count.saturating_add(1);
        crtc.page_flip_pending = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Global KMS Subsystem Instance
// ---------------------------------------------------------------------------

pub static KMS_DEVICE: TicketLock<Option<KmsDevice>> = TicketLock::new(None);

pub fn init(vram_base: u64, vram_size: u64, gtt_base: u64, gtt_size: u64) {
    let mut guard = KMS_DEVICE.lock();
    *guard = Some(KmsDevice::new(vram_base, vram_size, gtt_base, gtt_size));
}

// ---------------------------------------------------------------------------
// Self-Test
// ---------------------------------------------------------------------------

fn build_test_edid() -> Box<[u8; EDID_BLOCK_SIZE]> {
    let mut edid = Box::new([0u8; EDID_BLOCK_SIZE]);
    edid[..8].copy_from_slice(&EDID_HEADER);
    // Manufacturer "NXS" -> N=14, X=24, S=19 -> (14<<10)|(24<<5)|19 = 0x3B13
    edid[8] = 0x3B;
    edid[9] = 0x13;
    edid[10] = 0x01;
    edid[11] = 0x00;
    edid[16] = 12;
    edid[17] = 36; // 2026
    edid[18] = 1;
    edid[19] = 4;
    edid[20] = 0xA5; // Digital, 8-bit
    edid[21] = 60; // 600 mm
    edid[22] = 34; // 340 mm
    edid[23] = 120; // gamma 2.20

    // DTD 0: 1920x1080 @ 60Hz (148.5 MHz = 14850 * 10 kHz = 0x3A02)
    let dtd = &mut edid[54..72];
    dtd[0] = 0x02;
    dtd[1] = 0x3A;
    dtd[2] = 0x80; // hactive low (1920 = 0x780)
    dtd[3] = 0x18; // hblank low (280 = 0x118)
    dtd[4] = 0x71; // hactive high=7, hblank high=1
    dtd[5] = 0x38; // vactive low (1080 = 0x438)
    dtd[6] = 0x2D; // vblank low (45 = 0x02D)
    dtd[7] = 0x40; // vactive high=4, vblank high=0
    dtd[8] = 0x58; // hsync offset = 88
    dtd[9] = 0x2C; // hsync width = 44
    dtd[10] = 0x45; // vsync off=4, width=5
    dtd[11] = 0x00;
    dtd[17] = 0x1E; // non-interlaced, +hsync, +vsync

    // DTD 1: Monitor Name "NEXUS-DISP"
    let name_dtd = &mut edid[72..90];
    name_dtd[3] = 0xFC;
    name_dtd[5..15].copy_from_slice(b"NEXUS-DISP");
    name_dtd[15] = 0x0A;

    let mut sum: u8 = 0;
    for &b in &edid[..127] {
        sum = sum.wrapping_add(b);
    }
    edid[127] = 0u8.wrapping_sub(sum);
    edid
}

pub fn self_test() -> Result<(), &'static str> {
    // 1. EDID parsing & validation.
    let mut edid = build_test_edid();
    let parsed = parse_edid(&*edid).map_err(|e| e.as_str())?;
    if &parsed.manufacturer != b"NXS" || parsed.monitor_name != "NEXUS-DISP" {
        return Err("EDID manufacturer or monitor name mismatch");
    }
    if parsed.modes.is_empty()
        || parsed.modes[0].hdisplay != 1920
        || parsed.modes[0].vdisplay != 1080
    {
        return Err("EDID DTD 1920x1080 mode missing");
    }
    edid[127] ^= 0xFF;
    if parse_edid(&*edid) != Err(KmsError::InvalidEdidChecksum) {
        return Err("EDID bad checksum not rejected");
    }
    edid[127] ^= 0xFF;

    // 2. KMS device, CRTC/Encoder/Connector/Plane setup & atomic modeset.
    let mut dev = Box::new(KmsDevice::new(
        0xE000_0000,
        64 * 1024 * 1024,
        0x1000_0000,
        32 * 1024 * 1024,
    ));
    let crtc_id = dev.add_crtc(0);
    let enc_id = dev.add_encoder(EncoderType::DisplayPort, 0x1);
    let conn_id = dev.add_connector(ConnectorType::DisplayPort, vec![enc_id]);
    let _plane_id = dev.add_plane(PlaneType::Primary, 0x1, 0);
    dev.attach_edid(conn_id, &*edid).map_err(|e| e.as_str())?;

    let fb_size = 1920 * 1080 * 4;
    let bo1 = dev
        .bos
        .create_bo(
            fb_size,
            MemoryDomain::Vram,
            CacheMode::WriteCombine,
            BoFlags::scanout(),
        )
        .map_err(|e| e.as_str())?;
    let bo2 = dev
        .bos
        .create_bo(
            fb_size,
            MemoryDomain::Vram,
            CacheMode::WriteCombine,
            BoFlags::scanout(),
        )
        .map_err(|e| e.as_str())?;

    let fb1 = dev
        .create_framebuffer(bo1, 1920, 1080, 1920 * 4, PixelFormat::Xrgb8888)
        .map_err(|e| e.as_str())?;
    let fb2 = dev
        .create_framebuffer(bo2, 1920, 1080, 1920 * 4, PixelFormat::Xrgb8888)
        .map_err(|e| e.as_str())?;

    let mode = parsed.modes[0];
    dev.atomic_modeset(&ModeSetRequest {
        crtc_id,
        connector_id: conn_id,
        encoder_id: enc_id,
        fb_id: fb1,
        mode,
        x: 0,
        y: 0,
    })
    .map_err(|e| e.as_str())?;

    // 3. Page flip + GPU ring submission + fence completion.
    let flip_fence = dev.page_flip(crtc_id, fb2).map_err(|e| e.as_str())?;
    if dev.fences.is_signaled(flip_fence).map_err(|e| e.as_str())? {
        return Err("flip fence signaled prematurely");
    }
    let retired = dev.rings[0].complete_up_to(1, &mut dev.fences);
    if retired != 1 || !dev.fences.is_signaled(flip_fence).map_err(|e| e.as_str())? {
        return Err("ring completion did not signal flip fence");
    }
    dev.handle_vblank(crtc_id).map_err(|e| e.as_str())?;

    // 4. TTM migration & PRIME export/import.
    let scratch = dev
        .bos
        .create_bo(
            64 * 1024,
            MemoryDomain::Vram,
            CacheMode::WriteBack,
            BoFlags::default_render(),
        )
        .map_err(|e| e.as_str())?;
    dev.bos
        .migrate_bo(scratch, MemoryDomain::Gtt, &dev.fences)
        .map_err(|e| e.as_str())?;
    let prime_id = dev.bos.prime_export(scratch).map_err(|e| e.as_str())?;
    let imported = dev.bos.prime_import(prime_id).map_err(|e| e.as_str())?;
    if imported != scratch {
        return Err("PRIME import handle mismatch");
    }
    let _ = dev.bos.unref_bo(imported);
    let freed = dev.bos.unref_bo(scratch).map_err(|e| e.as_str())?;
    if !freed {
        return Err("scratch BO not freed on final unref");
    }

    Ok(())
}
