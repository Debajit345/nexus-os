//! Clean-room Intel High Definition Audio (HDA) PCIe controller driver.
#![allow(dead_code)]

use alloc::vec::Vec;
use core::ptr::{read_volatile, write_bytes, write_volatile};

use crate::mm::{alloc_frame, dealloc_frame, stats, PAGE_SIZE};
use crate::pci::PciDevice;

/// PCI class/subclass for Multimedia Audio / High Definition Audio.
pub const PCI_CLASS_MULTIMEDIA: u8 = 0x04;
pub const PCI_SUBCLASS_HDA: u8 = 0x03;

/// Intel HDA MMIO register offsets (BAR0).
pub mod regs {
    pub const GCAP: usize = 0x00;
    pub const VMIN: usize = 0x02;
    pub const VMAJ: usize = 0x03;
    pub const OUTPAY: usize = 0x04;
    pub const INPAY: usize = 0x06;
    pub const GCTL: usize = 0x08;
    pub const WAKEEN: usize = 0x0C;
    pub const STATESTS: usize = 0x0E;
    pub const GSTS: usize = 0x10;
    pub const INTCTL: usize = 0x20;
    pub const INTSTS: usize = 0x24;
    pub const WALCLK: usize = 0x30;
    pub const SSYNC: usize = 0x38;
    pub const CORBLBASE: usize = 0x40;
    pub const CORBUBASE: usize = 0x44;
    pub const CORBWP: usize = 0x48;
    pub const CORBRP: usize = 0x4A;
    pub const CORBCTL: usize = 0x4C;
    pub const CORBSTS: usize = 0x4D;
    pub const CORBSIZE: usize = 0x4E;
    pub const RIRBLBASE: usize = 0x50;
    pub const RIRBUBASE: usize = 0x54;
    pub const RIRBWP: usize = 0x58;
    pub const RINTCNT: usize = 0x5A;
    pub const RIRBCTL: usize = 0x5C;
    pub const RIRBSTS: usize = 0x5D;
    pub const RIRBSIZE: usize = 0x5E;
    pub const DPLBASE: usize = 0x70;
    pub const DPUBASE: usize = 0x74;
    pub const SD_BASE: usize = 0x80;
    pub const SD_STRIDE: usize = 0x20;
}

const GCTL_CRST: u32 = 1 << 0;
const CORBRP_RST: u16 = 1 << 15;
const CORBCTL_RUN: u8 = 1 << 1;
const RIRBWP_RST: u16 = 1 << 15;
const RIRBCTL_DMA_EN: u8 = 1 << 1;

const POLL_TIMEOUT_ITERS: usize = 100_000;

/// Supported PCM sample rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleRate {
    Hz44100,
    Hz48000,
    Hz88200,
    Hz96000,
    Hz176400,
    Hz192000,
}

/// Supported channel configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    Mono,
    Stereo,
    Surround51,
    Surround71,
}

impl Channels {
    #[inline]
    pub const fn count(self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
        }
    }
}

/// Supported PCM sample bit depths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    Bits8,
    Bits16,
    Bits20,
    Bits24,
    Bits32,
}

/// Audio stream format descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    pub sample_rate: SampleRate,
    pub channels: Channels,
    pub bit_depth: BitDepth,
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self {
            sample_rate: SampleRate::Hz48000,
            channels: Channels::Stereo,
            bit_depth: BitDepth::Bits16,
        }
    }
}

impl AudioFormat {
    /// Encode into the 16-bit Intel HDA Stream Format register value.
    pub const fn to_hda_format(self) -> u16 {
        let (base, mult, div) = match self.sample_rate {
            SampleRate::Hz44100 => (1u16, 0u16, 0u16),
            SampleRate::Hz48000 => (0u16, 0u16, 0u16),
            SampleRate::Hz88200 => (1u16, 1u16, 0u16),
            SampleRate::Hz96000 => (0u16, 1u16, 0u16),
            SampleRate::Hz176400 => (1u16, 3u16, 0u16),
            SampleRate::Hz192000 => (0u16, 3u16, 0u16),
        };
        let bits = match self.bit_depth {
            BitDepth::Bits8 => 0u16,
            BitDepth::Bits16 => 1u16,
            BitDepth::Bits20 => 2u16,
            BitDepth::Bits24 => 3u16,
            BitDepth::Bits32 => 4u16,
        };
        let chan = (self.channels.count().saturating_sub(1) as u16) & 0x0F;
        (base << 14) | (mult << 11) | (div << 8) | (bits << 4) | chan
    }
}

/// Intel High Definition Audio controller instance.
#[derive(Debug)]
pub struct HdaController {
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub mmio_base: u64,
    pub gcap: u16,
    pub vmaj: u8,
    pub vmin: u8,
    pub oss_count: u8,
    pub iss_count: u8,
    pub bss_count: u8,
    pub codec_mask: u16,
    pub corb_phys: u64,
    pub corb_virt: u64,
    pub corb_entries: u16,
    pub rirb_phys: u64,
    pub rirb_virt: u64,
    pub rirb_entries: u16,
    corb_wp: u16,
    rirb_rp: u16,
}

impl HdaController {
    #[inline]
    fn read_u8(&self, offset: usize) -> u8 {
        unsafe { read_volatile((self.mmio_base + offset as u64) as *const u8) }
    }

    #[inline]
    fn read_u16(&self, offset: usize) -> u16 {
        unsafe { read_volatile((self.mmio_base + offset as u64) as *const u16) }
    }

    #[inline]
    fn read_u32(&self, offset: usize) -> u32 {
        unsafe { read_volatile((self.mmio_base + offset as u64) as *const u32) }
    }

    #[inline]
    fn write_u8(&self, offset: usize, val: u8) {
        unsafe { write_volatile((self.mmio_base + offset as u64) as *mut u8, val) }
    }

    #[inline]
    fn write_u16(&self, offset: usize, val: u16) {
        unsafe { write_volatile((self.mmio_base + offset as u64) as *mut u16, val) }
    }

    #[inline]
    fn write_u32(&self, offset: usize, val: u32) {
        unsafe { write_volatile((self.mmio_base + offset as u64) as *mut u32, val) }
    }

    /// Probe and initialize an Intel HDA controller from a PCI device descriptor.
    pub fn probe(dev: &PciDevice) -> Option<Self> {
        if dev.class_code != PCI_CLASS_MULTIMEDIA || dev.subclass != PCI_SUBCLASS_HDA {
            return None;
        }

        let bar0_phys = dev.bars[0] & !0xF;
        if bar0_phys == 0 {
            return None;
        }

        let hhdm = stats().hhdm_offset;
        let mmio_base = bar0_phys.wrapping_add(hhdm);

        let corb_phys = alloc_frame()?;
        let rirb_phys = match alloc_frame() {
            Some(p) => p,
            None => {
                dealloc_frame(corb_phys);
                return None;
            }
        };

        let corb_virt = corb_phys.wrapping_add(hhdm);
        let rirb_virt = rirb_phys.wrapping_add(hhdm);
        unsafe {
            write_bytes(corb_virt as *mut u8, 0, PAGE_SIZE);
            write_bytes(rirb_virt as *mut u8, 0, PAGE_SIZE);
        }

        let mut ctrl = Self {
            bus: dev.bus,
            dev: dev.dev,
            func: dev.func,
            vendor_id: dev.vendor_id,
            device_id: dev.device_id,
            mmio_base,
            gcap: 0,
            vmaj: 0,
            vmin: 0,
            oss_count: 0,
            iss_count: 0,
            bss_count: 0,
            codec_mask: 0,
            corb_phys,
            corb_virt,
            corb_entries: 256,
            rirb_phys,
            rirb_virt,
            rirb_entries: 256,
            corb_wp: 0,
            rirb_rp: 0,
        };

        ctrl.gcap = ctrl.read_u16(regs::GCAP);
        ctrl.vmin = ctrl.read_u8(regs::VMIN);
        ctrl.vmaj = ctrl.read_u8(regs::VMAJ);
        ctrl.oss_count = ((ctrl.gcap >> 12) & 0x0F) as u8;
        ctrl.iss_count = ((ctrl.gcap >> 8) & 0x0F) as u8;
        ctrl.bss_count = ((ctrl.gcap >> 3) & 0x1F) as u8;

        if !ctrl.reset_controller() {
            dealloc_frame(corb_phys);
            dealloc_frame(rirb_phys);
            return None;
        }

        ctrl.codec_mask = ctrl.read_u16(regs::STATESTS) & 0x7FFF;
        ctrl.init_corb_rirb();

        Some(ctrl)
    }

    fn reset_controller(&self) -> bool {
        // Enter reset (CRST = 0)
        let gctl = self.read_u32(regs::GCTL);
        self.write_u32(regs::GCTL, gctl & !GCTL_CRST);
        let mut ok = false;
        for _ in 0..POLL_TIMEOUT_ITERS {
            if (self.read_u32(regs::GCTL) & GCTL_CRST) == 0 {
                ok = true;
                break;
            }
            core::hint::spin_loop();
        }
        if !ok {
            return false;
        }

        // Exit reset (CRST = 1)
        let gctl = self.read_u32(regs::GCTL);
        self.write_u32(regs::GCTL, gctl | GCTL_CRST);
        for _ in 0..POLL_TIMEOUT_ITERS {
            if (self.read_u32(regs::GCTL) & GCTL_CRST) != 0 {
                return true;
            }
            core::hint::spin_loop();
        }
        false
    }

    fn init_corb_rirb(&mut self) {
        // Stop CORB and RIRB DMA before programming base addresses
        self.write_u8(regs::CORBCTL, 0);
        self.write_u8(regs::RIRBCTL, 0);

        // Program CORB (256 entries = 1024 bytes, size code 0x02)
        self.corb_entries = Self::select_ring_entries(self.read_u8(regs::CORBSIZE), 256);
        let corb_size_code = Self::size_code_for_entries(self.corb_entries);
        self.write_u8(regs::CORBSIZE, corb_size_code);
        self.write_u32(regs::CORBLBASE, self.corb_phys as u32);
        self.write_u32(regs::CORBUBASE, (self.corb_phys >> 32) as u32);

        // Reset CORB read pointer
        self.write_u16(regs::CORBRP, CORBRP_RST);
        for _ in 0..POLL_TIMEOUT_ITERS {
            if (self.read_u16(regs::CORBRP) & CORBRP_RST) != 0 {
                break;
            }
            core::hint::spin_loop();
        }
        self.write_u16(regs::CORBRP, 0);
        for _ in 0..POLL_TIMEOUT_ITERS {
            if (self.read_u16(regs::CORBRP) & CORBRP_RST) == 0 {
                break;
            }
            core::hint::spin_loop();
        }
        self.corb_wp = 0;
        self.write_u16(regs::CORBWP, 0);

        // Program RIRB (256 entries = 2048 bytes, size code 0x02)
        self.rirb_entries = Self::select_ring_entries(self.read_u8(regs::RIRBSIZE), 256);
        let rirb_size_code = Self::size_code_for_entries(self.rirb_entries);
        self.write_u8(regs::RIRBSIZE, rirb_size_code);
        self.write_u32(regs::RIRBLBASE, self.rirb_phys as u32);
        self.write_u32(regs::RIRBUBASE, (self.rirb_phys >> 32) as u32);
        self.write_u16(regs::RIRBWP, RIRBWP_RST);
        self.write_u16(regs::RINTCNT, 1);
        self.rirb_rp = 0;

        // Enable CORB and RIRB DMA engines
        self.write_u8(regs::CORBCTL, CORBCTL_RUN);
        self.write_u8(regs::RIRBCTL, RIRBCTL_DMA_EN);
    }

    const fn select_ring_entries(size_cap_reg: u8, preferred: u16) -> u16 {
        let cap = size_cap_reg >> 4;
        if (cap & 0x04) != 0 && preferred >= 256 {
            256
        } else if (cap & 0x02) != 0 && preferred >= 16 {
            16
        } else if (cap & 0x01) != 0 {
            2
        } else {
            256
        }
    }

    const fn size_code_for_entries(entries: u16) -> u8 {
        match entries {
            2 => 0x00,
            16 => 0x01,
            _ => 0x02,
        }
    }

    /// Send a 20-bit HDA codec verb to `(codec, nid)` and poll RIRB for the 64-bit response.
    pub fn send_verb(&mut self, codec: u8, nid: u16, verb: u32) -> Option<u64> {
        if self.corb_entries == 0 || self.rirb_entries == 0 {
            return None;
        }

        let cmd =
            ((codec as u32 & 0x0F) << 28) | ((nid as u32 & 0xFF) << 20) | (verb & 0x000F_FFFF);

        let next_wp = (self.corb_wp + 1) % self.corb_entries;
        let corb_ptr = self.corb_virt as *mut u32;
        unsafe {
            write_volatile(corb_ptr.add(next_wp as usize), cmd);
        }
        self.corb_wp = next_wp;
        self.write_u16(regs::CORBWP, next_wp);

        for _ in 0..POLL_TIMEOUT_ITERS {
            let rirb_wp = self.read_u16(regs::RIRBWP) & 0x00FF;
            if rirb_wp != self.rirb_rp {
                self.rirb_rp = (self.rirb_rp + 1) % self.rirb_entries;
                let rirb_ptr = self.rirb_virt as *const u64;
                let resp = unsafe { read_volatile(rirb_ptr.add(self.rirb_rp as usize)) };
                return Some(resp);
            }
            core::hint::spin_loop();
        }

        None
    }
}

/// Probe all PCI devices for Intel HDA audio controllers.
pub fn probe_audio(devs: &[PciDevice]) -> Vec<HdaController> {
    let mut controllers = Vec::new();
    for dev in devs {
        if let Some(ctrl) = HdaController::probe(dev) {
            controllers.push(ctrl);
        }
    }
    controllers
}
