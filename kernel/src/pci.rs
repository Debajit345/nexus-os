//! PCIe ECAM and legacy 0xCF8/0xCFC configuration space access and device enumeration.
#![allow(dead_code)]

use alloc::vec::Vec;
use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};

use crate::acpi::McfgAllocation;

const PCI_CONFIG_ADDR: u16 = 0x0CF8;
const PCI_CONFIG_DATA: u16 = 0x0CFC;

#[inline]
unsafe fn outl(port: u16, val: u32) {
    asm!("out dx, eax", in("dx") port, in("eax") val, options(nomem, nostack, preserves_flags));
}

#[inline]
unsafe fn inl(port: u16) -> u32 {
    let val: u32;
    asm!("in eax, dx", out("eax") val, in("dx") port, options(nomem, nostack, preserves_flags));
    val
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PciClass {
    Nvme,
    XhciUsb,
    Ahci,
    Virtio,
    Gpu,
    Network,
    Other,
}

impl PciClass {
    pub fn from_device(dev: &PciDevice) -> Self {
        if dev.vendor_id == 0x1AF4 {
            return Self::Virtio;
        }
        match (dev.class_code, dev.subclass, dev.prog_if) {
            (0x01, 0x08, 0x02) => Self::Nvme,
            (0x01, 0x06, 0x01) => Self::Ahci,
            (0x0C, 0x03, 0x30) => Self::XhciUsb,
            (0x03, _, _) => Self::Gpu,
            (0x02, _, _) => Self::Network,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PciDevice {
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: u8,
    pub subclass: u8,
    pub prog_if: u8,
    pub header_type: u8,
    pub irq_line: u8,
    pub bars: [u64; 6],
    pub msi_cap: Option<u8>,
    pub msix_cap: Option<u8>,
}

impl PciDevice {
    pub fn class(&self) -> PciClass {
        PciClass::from_device(self)
    }
}

#[inline]
fn ecam_addr(mcfg: &[McfgAllocation], bus: u8, dev: u8, func: u8, offset: u16) -> Option<u64> {
    if dev >= 32 || func >= 8 || offset >= 4096 {
        return None;
    }
    let alloc = mcfg
        .iter()
        .find(|a| a.segment == 0 && bus >= a.start_bus && bus <= a.end_bus)?;
    let phys = alloc.base_addr
        + (((bus - alloc.start_bus) as u64) << 20)
        + ((dev as u64) << 15)
        + ((func as u64) << 12)
        + ((offset as u64) & 0xFFC);
    let hhdm = crate::mm::stats().hhdm_offset;
    Some(phys.wrapping_add(hhdm))
}

#[inline]
fn legacy_addr(bus: u8, dev: u8, func: u8, offset: u16) -> u32 {
    0x8000_0000u32
        | ((bus as u32) << 16)
        | (((dev as u32) & 0x1F) << 11)
        | (((func as u32) & 0x07) << 8)
        | ((offset as u32) & 0xFC)
}

pub fn read_u32(mcfg: &[McfgAllocation], bus: u8, dev: u8, func: u8, offset: u16) -> u32 {
    if let Some(virt) = ecam_addr(mcfg, bus, dev, func, offset) {
        unsafe { read_volatile(virt as *const u32) }
    } else if offset < 256 {
        unsafe {
            outl(PCI_CONFIG_ADDR, legacy_addr(bus, dev, func, offset));
            inl(PCI_CONFIG_DATA)
        }
    } else {
        0xFFFF_FFFF
    }
}

pub fn write_u32(mcfg: &[McfgAllocation], bus: u8, dev: u8, func: u8, offset: u16, val: u32) {
    if let Some(virt) = ecam_addr(mcfg, bus, dev, func, offset) {
        unsafe { write_volatile(virt as *mut u32, val) };
    } else if offset < 256 {
        unsafe {
            outl(PCI_CONFIG_ADDR, legacy_addr(bus, dev, func, offset));
            outl(PCI_CONFIG_DATA, val);
        }
    }
}

pub fn read_u16(mcfg: &[McfgAllocation], bus: u8, dev: u8, func: u8, offset: u16) -> u16 {
    let dword = read_u32(mcfg, bus, dev, func, offset & !0x3);
    let shift = ((offset & 0x2) * 8) as u32;
    ((dword >> shift) & 0xFFFF) as u16
}

pub fn read_u8(mcfg: &[McfgAllocation], bus: u8, dev: u8, func: u8, offset: u16) -> u8 {
    let dword = read_u32(mcfg, bus, dev, func, offset & !0x3);
    let shift = ((offset & 0x3) * 8) as u32;
    ((dword >> shift) & 0xFF) as u8
}

fn read_bars(mcfg: &[McfgAllocation], bus: u8, dev: u8, func: u8, header_type: u8) -> [u64; 6] {
    let mut bars = [0u64; 6];
    let max_bars = match header_type & 0x7F {
        0x00 => 6,
        0x01 => 2,
        _ => 0,
    };

    let mut i = 0;
    while i < max_bars {
        let offset = 0x10 + (i as u16) * 4;
        let raw = read_u32(mcfg, bus, dev, func, offset);
        if raw == 0 {
            i += 1;
            continue;
        }

        if (raw & 0x1) != 0 {
            // I/O space BAR
            bars[i] = (raw & 0xFFFF_FFFC) as u64;
            i += 1;
        } else {
            // Memory space BAR
            let mem_type = (raw >> 1) & 0x3;
            let low = (raw & 0xFFFF_FFF0) as u64;
            if mem_type == 0x2 && i + 1 < max_bars {
                let high = read_u32(mcfg, bus, dev, func, offset + 4) as u64;
                bars[i] = (high << 32) | low;
                i += 2;
            } else {
                bars[i] = low;
                i += 1;
            }
        }
    }

    bars
}

fn find_msi_caps(mcfg: &[McfgAllocation], bus: u8, dev: u8, func: u8) -> (Option<u8>, Option<u8>) {
    let status = read_u16(mcfg, bus, dev, func, 0x06);
    if (status & 0x0010) == 0 {
        return (None, None);
    }

    let mut msi_cap = None;
    let mut msix_cap = None;
    let mut ptr = read_u8(mcfg, bus, dev, func, 0x34) & 0xFC;
    let mut steps = 0u8;

    while ptr >= 0x40 && steps < 48 {
        let cap_id = read_u8(mcfg, bus, dev, func, ptr as u16);
        match cap_id {
            0x05 if msi_cap.is_none() => msi_cap = Some(ptr),
            0x11 if msix_cap.is_none() => msix_cap = Some(ptr),
            _ => {}
        }
        let next = read_u8(mcfg, bus, dev, func, (ptr as u16) + 1) & 0xFC;
        if next == ptr {
            break;
        }
        ptr = next;
        steps += 1;
    }

    (msi_cap, msix_cap)
}

fn probe_function(mcfg: &[McfgAllocation], bus: u8, dev: u8, func: u8) -> Option<PciDevice> {
    let id_reg = read_u32(mcfg, bus, dev, func, 0x00);
    let vendor_id = (id_reg & 0xFFFF) as u16;
    if vendor_id == 0xFFFF {
        return None;
    }
    let device_id = ((id_reg >> 16) & 0xFFFF) as u16;

    let class_reg = read_u32(mcfg, bus, dev, func, 0x08);
    let prog_if = ((class_reg >> 8) & 0xFF) as u8;
    let subclass = ((class_reg >> 16) & 0xFF) as u8;
    let class_code = ((class_reg >> 24) & 0xFF) as u8;

    let header_type = read_u8(mcfg, bus, dev, func, 0x0E);
    let irq_line = read_u8(mcfg, bus, dev, func, 0x3C);
    let bars = read_bars(mcfg, bus, dev, func, header_type);
    let (msi_cap, msix_cap) = find_msi_caps(mcfg, bus, dev, func);

    Some(PciDevice {
        bus,
        dev,
        func,
        vendor_id,
        device_id,
        class_code,
        subclass,
        prog_if,
        header_type,
        irq_line,
        bars,
        msi_cap,
        msix_cap,
    })
}

fn scan_bus(mcfg: &[McfgAllocation], bus: u8, devices: &mut Vec<PciDevice>) {
    for dev in 0..32u8 {
        let Some(func0) = probe_function(mcfg, bus, dev, 0) else {
            continue;
        };
        let multi_func = (func0.header_type & 0x80) != 0;
        devices.push(func0);

        if multi_func {
            for func in 1..8u8 {
                if let Some(f) = probe_function(mcfg, bus, dev, func) {
                    devices.push(f);
                }
            }
        }
    }
}

/// Enumerate all PCI/PCIe devices using MCFG ECAM (or legacy 0xCF8/0xCFC if `mcfg` is empty).
pub fn enumerate(mcfg: &[McfgAllocation]) -> Vec<PciDevice> {
    let mut devices = Vec::new();

    if mcfg.is_empty() {
        for bus in 0..=255u8 {
            scan_bus(mcfg, bus, &mut devices);
        }
    } else {
        for alloc in mcfg {
            if alloc.segment != 0 {
                continue;
            }
            for bus in alloc.start_bus..=alloc.end_bus {
                scan_bus(mcfg, bus, &mut devices);
            }
        }
    }

    devices
}
