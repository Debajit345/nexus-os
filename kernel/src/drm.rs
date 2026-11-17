//! Clean-room Kernel DRM/GPU subsystem (GEM buffers, KMS modesetting, PCI GPU probing).
#![allow(dead_code)]

use alloc::vec::Vec;
use core::ptr::write_bytes;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::mm::{alloc_frame, dealloc_frame, stats, PAGE_SIZE};
use crate::pci::{PciClass, PciDevice};

static NEXT_GEM_HANDLE: AtomicU32 = AtomicU32::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Intel,
    Amd,
    Nvidia,
    Virtio,
    Other,
}

impl GpuVendor {
    pub const fn from_vendor_id(vendor_id: u16) -> Self {
        match vendor_id {
            0x8086 => Self::Intel,
            0x1002 => Self::Amd,
            0x10DE => Self::Nvidia,
            0x1AF4 => Self::Virtio,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayMode {
    pub width: u32,
    pub height: u32,
    pub refresh_hz: u32,
}

impl DisplayMode {
    pub const fn new(width: u32, height: u32, refresh_hz: u32) -> Self {
        Self {
            width,
            height,
            refresh_hz,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemBuffer {
    pub handle: u32,
    pub phys_addr: u64,
    pub size: usize,
}

impl GemBuffer {
    /// Virtual address of the buffer via HHDM.
    #[inline]
    pub fn virt_addr(&self) -> u64 {
        self.phys_addr.wrapping_add(stats().hhdm_offset)
    }
}

#[derive(Debug, Clone)]
pub struct DrmDevice {
    pub id: u32,
    pub vendor: GpuVendor,
    pub vendor_id: u16,
    pub device_id: u16,
    pub pci_bus: u8,
    pub pci_dev: u8,
    pub pci_func: u8,
    pub mmio_bar: u64,
    pub vram_bar: u64,
    pub bars: [u64; 6],
    pub msi_cap: Option<u8>,
    pub msix_cap: Option<u8>,
    pub modes: Vec<DisplayMode>,
    pub active_mode: Option<DisplayMode>,
    pub active_fb: Option<GemBuffer>,
}

impl DrmDevice {
    pub fn from_pci(id: u32, dev: &PciDevice) -> Self {
        let vendor = GpuVendor::from_vendor_id(dev.vendor_id);
        // Heuristic BAR assignment:
        // - Intel/AMD/NVIDIA typically expose MMIO in BAR0 and VRAM/aperture in BAR2 (or BAR1/BAR4).
        // - VirtIO GPU exposes MMIO/capabilities across BAR0/BAR1/BAR4.
        let mmio_bar = dev.bars[0];
        let vram_bar = if dev.bars[2] != 0 {
            dev.bars[2]
        } else if dev.bars[1] != 0 {
            dev.bars[1]
        } else {
            dev.bars[4]
        };

        Self {
            id,
            vendor,
            vendor_id: dev.vendor_id,
            device_id: dev.device_id,
            pci_bus: dev.bus,
            pci_dev: dev.dev,
            pci_func: dev.func,
            mmio_bar,
            vram_bar,
            bars: dev.bars,
            msi_cap: dev.msi_cap,
            msix_cap: dev.msix_cap,
            modes: Vec::new(),
            active_mode: None,
            active_fb: None,
        }
    }

    pub fn has_msi(&self) -> bool {
        self.msi_cap.is_some() || self.msix_cap.is_some()
    }

    pub fn add_mode(&mut self, mode: DisplayMode) {
        if !self.modes.contains(&mode) {
            self.modes.push(mode);
        }
    }

    pub fn set_mode(&mut self, mode: DisplayMode, fb: Option<GemBuffer>) {
        self.add_mode(mode);
        self.active_mode = Some(mode);
        if let Some(buf) = fb {
            self.active_fb = Some(buf);
        }
    }
}

/// Probe PCI devices for GPUs (`PciClass::Gpu`, plus VirtIO display devices with class `0x03`).
pub fn probe_gpus(devs: &[PciDevice]) -> Vec<DrmDevice> {
    let mut gpus = Vec::new();
    let mut next_id = 0u32;

    for dev in devs {
        let is_gpu = dev.class() == PciClass::Gpu
            || (dev.class() == PciClass::Virtio && dev.class_code == 0x03);
        if !is_gpu {
            continue;
        }

        gpus.push(DrmDevice::from_pci(next_id, dev));
        next_id += 1;
    }

    gpus
}

/// Allocate a GEM buffer backed by physical frames from `crate::mm::alloc_frame()`.
pub fn alloc_gem_buffer(size: usize) -> Option<GemBuffer> {
    if size == 0 {
        return None;
    }

    let pages = size.div_ceil(PAGE_SIZE);
    let aligned_size = pages * PAGE_SIZE;
    let hhdm_offset = stats().hhdm_offset;

    let first_phys = alloc_frame()?;
    unsafe {
        write_bytes((first_phys + hhdm_offset) as *mut u8, 0, PAGE_SIZE);
    }

    let mut allocated = Vec::with_capacity(pages);
    allocated.push(first_phys);

    for _ in 1..pages {
        match alloc_frame() {
            Some(phys) => {
                unsafe {
                    write_bytes((phys + hhdm_offset) as *mut u8, 0, PAGE_SIZE);
                }
                allocated.push(phys);
            }
            None => {
                for p in allocated {
                    dealloc_frame(p);
                }
                return None;
            }
        }
    }

    let handle = NEXT_GEM_HANDLE.fetch_add(1, Ordering::Relaxed);
    Some(GemBuffer {
        handle,
        phys_addr: first_phys,
        size: aligned_size,
    })
}

/// Free a GEM buffer's backing physical frames (for contiguous allocations).
pub fn free_gem_buffer(buf: GemBuffer) {
    let pages = buf.size.div_ceil(PAGE_SIZE);
    for i in 0..pages {
        dealloc_frame(buf.phys_addr + (i * PAGE_SIZE) as u64);
    }
}
