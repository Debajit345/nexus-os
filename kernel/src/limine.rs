//! Clean Limine boot protocol definitions (Base Revision 2).

use core::cell::UnsafeCell;
use core::ptr::NonNull;

#[repr(C)]
pub struct RequestsStartMarker([u64; 4]);

#[repr(C)]
pub struct RequestsEndMarker([u64; 2]);

#[used]
#[link_section = ".requests_start_marker"]
pub static REQUESTS_START_MARKER: RequestsStartMarker = RequestsStartMarker([
    0xf6b8f4b39de7d1ae,
    0xfab91a6940fcb9cf,
    0x785c6ed015d3e316,
    0x181e920a7852b9d9,
]);

#[used]
#[link_section = ".requests_end_marker"]
pub static REQUESTS_END_MARKER: RequestsEndMarker =
    RequestsEndMarker([0xadc0e0531bb10d03, 0x9572709f31764c62]);

#[repr(C)]
pub struct BaseRevision {
    id: [u64; 2],
    revision: UnsafeCell<u64>,
}

unsafe impl Sync for BaseRevision {}

impl BaseRevision {
    pub const fn new(revision: u64) -> Self {
        Self {
            id: [0xf9562b2d5c95a6c8, 0x6a7b384944536bdc],
            revision: UnsafeCell::new(revision),
        }
    }

    pub fn is_supported(&self) -> bool {
        unsafe { core::ptr::read_volatile(self.revision.get()) == 0 }
    }
}

#[repr(C)]
pub struct Request<Resp> {
    id: [u64; 4],
    revision: u64,
    response: UnsafeCell<*const Resp>,
}

unsafe impl<Resp> Sync for Request<Resp> {}

impl<Resp> Request<Resp> {
    pub const fn new(id: [u64; 4], revision: u64) -> Self {
        Self {
            id,
            revision,
            response: UnsafeCell::new(core::ptr::null()),
        }
    }

    pub fn response(&self) -> Option<&Resp> {
        let ptr = unsafe { core::ptr::read_volatile(self.response.get()) };
        NonNull::new(ptr as *mut Resp).map(|nn| unsafe { nn.as_ref() })
    }
}

pub const FRAMEBUFFER_REQUEST_ID: [u64; 4] = [
    0xc7b1dd30df4c8b88,
    0x0a82e883a194f07b,
    0x9d5827dcd881dd75,
    0xa3148604f6fab11b,
];

pub const HHDM_REQUEST_ID: [u64; 4] = [
    0xc7b1dd30df4c8b88,
    0x0a82e883a194f07b,
    0x48dcf1cb8ad2b852,
    0x63984e959a98244b,
];

pub const MEMMAP_REQUEST_ID: [u64; 4] = [
    0xc7b1dd30df4c8b88,
    0x0a82e883a194f07b,
    0x67cf3d9d378a806f,
    0xe304acdfc50c3c62,
];

pub const KERNEL_ADDRESS_REQUEST_ID: [u64; 4] = [
    0xc7b1dd30df4c8b88,
    0x0a82e883a194f07b,
    0x71ba76863cc55f63,
    0xb2644a48c516a487,
];

#[repr(C)]
pub struct FramebufferResponse {
    pub revision: u64,
    pub framebuffer_count: u64,
    pub framebuffers: *const *const Framebuffer,
}

impl FramebufferResponse {
    pub fn framebuffers(&self) -> &[&Framebuffer] {
        if self.framebuffers.is_null() || self.framebuffer_count == 0 {
            return &[];
        }
        unsafe {
            core::slice::from_raw_parts(
                self.framebuffers as *const &Framebuffer,
                self.framebuffer_count as usize,
            )
        }
    }
}

#[repr(C)]
pub struct Framebuffer {
    pub address: *mut u8,
    pub width: u64,
    pub height: u64,
    pub pitch: u64,
    pub bpp: u16,
    pub memory_model: u8,
    pub red_mask_size: u8,
    pub red_mask_shift: u8,
    pub green_mask_size: u8,
    pub green_mask_shift: u8,
    pub blue_mask_size: u8,
    pub blue_mask_shift: u8,
    pub unused: [u8; 7],
    pub edid_size: u64,
    pub edid: *const u8,
}

#[repr(C)]
pub struct HhdmResponse {
    pub revision: u64,
    pub offset: u64,
}

#[repr(C)]
pub struct MemmapResponse {
    pub revision: u64,
    pub entry_count: u64,
    pub entries: *const *const MemmapEntry,
}

impl MemmapResponse {
    pub fn entries(&self) -> &[&MemmapEntry] {
        if self.entries.is_null() || self.entry_count == 0 {
            return &[];
        }
        unsafe {
            core::slice::from_raw_parts(
                self.entries as *const &MemmapEntry,
                self.entry_count as usize,
            )
        }
    }
}

#[repr(C)]
pub struct MemmapEntry {
    pub base: u64,
    pub length: u64,
    pub entry_type: u64,
}

pub const MEMMAP_USABLE: u64 = 0;

#[repr(C)]
pub struct KernelAddressResponse {
    pub revision: u64,
    pub physical_base: u64,
    pub virtual_base: u64,
}
