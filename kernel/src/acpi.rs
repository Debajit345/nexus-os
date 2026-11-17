//! ACPI table parser (RSDP, RSDT/XSDT, MADT, MCFG).
#![allow(dead_code)]

use alloc::vec::Vec;
use core::ptr::read_unaligned;

const RSDP_SIGNATURE: &[u8; 8] = b"RSD PTR ";

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct RsdpV1 {
    signature: [u8; 8],
    checksum: u8,
    oem_id: [u8; 6],
    revision: u8,
    rsdt_address: u32,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct RsdpV2 {
    v1: RsdpV1,
    length: u32,
    xsdt_address: u64,
    extended_checksum: u8,
    reserved: [u8; 3],
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct SdtHeader {
    pub signature: [u8; 4],
    pub length: u32,
    pub revision: u8,
    pub checksum: u8,
    pub oem_id: [u8; 6],
    pub oem_table_id: [u8; 8],
    pub oem_revision: u32,
    pub creator_id: u32,
    pub creator_revision: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalApicEntry {
    pub acpi_processor_id: u8,
    pub apic_id: u8,
    pub flags: u32,
}

impl LocalApicEntry {
    pub fn is_enabled(&self) -> bool {
        (self.flags & 0x1) != 0
    }

    pub fn is_online_capable(&self) -> bool {
        (self.flags & 0x2) != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoApicEntry {
    pub io_apic_id: u8,
    pub io_apic_addr: u32,
    pub gsi_base: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterruptOverride {
    pub bus: u8,
    pub source: u8,
    pub gsi: u32,
    pub flags: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McfgAllocation {
    pub base_addr: u64,
    pub segment: u16,
    pub start_bus: u8,
    pub end_bus: u8,
}

#[derive(Debug, Clone)]
pub struct AcpiInfo {
    pub revision: u8,
    pub lapic_addr: u64,
    pub madt_flags: u32,
    pub local_apics: Vec<LocalApicEntry>,
    pub io_apics: Vec<IoApicEntry>,
    pub interrupt_overrides: Vec<InterruptOverride>,
    pub mcfg_allocations: Vec<McfgAllocation>,
}

#[inline]
fn phys_to_virt(phys: u64) -> *const u8 {
    let hhdm = crate::mm::stats().hhdm_offset;
    (phys.wrapping_add(hhdm)) as *const u8
}

#[inline]
fn resolve_rsdp_ptr(rsdp_addr: *const u8) -> *const u8 {
    let addr = rsdp_addr as u64;
    let hhdm = crate::mm::stats().hhdm_offset;
    if hhdm != 0 && addr < hhdm {
        (addr.wrapping_add(hhdm)) as *const u8
    } else {
        rsdp_addr
    }
}

fn validate_checksum(ptr: *const u8, len: usize) -> bool {
    if ptr.is_null() || len == 0 {
        return false;
    }
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    let sum = bytes.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    sum == 0
}

unsafe fn read_sdt_header(phys: u64) -> Option<(*const u8, SdtHeader)> {
    if phys == 0 {
        return None;
    }
    let ptr = phys_to_virt(phys);
    let header = read_unaligned(ptr as *const SdtHeader);
    let len = header.length as usize;
    if len < core::mem::size_of::<SdtHeader>() {
        return None;
    }
    if !validate_checksum(ptr, len) {
        return None;
    }
    Some((ptr, header))
}

unsafe fn parse_madt(ptr: *const u8, header: &SdtHeader, info: &mut AcpiInfo) {
    let total_len = header.length as usize;
    let hdr_size = core::mem::size_of::<SdtHeader>();
    if total_len < hdr_size + 8 {
        return;
    }

    let lapic_addr32 = read_unaligned(ptr.add(hdr_size) as *const u32);
    let madt_flags = read_unaligned(ptr.add(hdr_size + 4) as *const u32);
    info.lapic_addr = lapic_addr32 as u64;
    info.madt_flags = madt_flags;

    let mut offset = hdr_size + 8;
    while offset + 2 <= total_len {
        let entry_type = *ptr.add(offset);
        let record_len = *ptr.add(offset + 1) as usize;
        if record_len < 2 || offset + record_len > total_len {
            break;
        }

        match entry_type {
            // Processor Local APIC
            0 if record_len >= 8 => {
                let acpi_processor_id = *ptr.add(offset + 2);
                let apic_id = *ptr.add(offset + 3);
                let flags = read_unaligned(ptr.add(offset + 4) as *const u32);
                info.local_apics.push(LocalApicEntry {
                    acpi_processor_id,
                    apic_id,
                    flags,
                });
            }
            // I/O APIC
            1 if record_len >= 12 => {
                let io_apic_id = *ptr.add(offset + 2);
                let io_apic_addr = read_unaligned(ptr.add(offset + 4) as *const u32);
                let gsi_base = read_unaligned(ptr.add(offset + 8) as *const u32);
                info.io_apics.push(IoApicEntry {
                    io_apic_id,
                    io_apic_addr,
                    gsi_base,
                });
            }
            // Interrupt Source Override
            2 if record_len >= 10 => {
                let bus = *ptr.add(offset + 2);
                let source = *ptr.add(offset + 3);
                let gsi = read_unaligned(ptr.add(offset + 4) as *const u32);
                let flags = read_unaligned(ptr.add(offset + 8) as *const u16);
                info.interrupt_overrides.push(InterruptOverride {
                    bus,
                    source,
                    gsi,
                    flags,
                });
            }
            // Local APIC Address Override (64-bit)
            5 if record_len >= 12 => {
                let addr64 = read_unaligned(ptr.add(offset + 4) as *const u64);
                info.lapic_addr = addr64;
            }
            _ => {}
        }

        offset += record_len;
    }
}

unsafe fn parse_mcfg(ptr: *const u8, header: &SdtHeader, info: &mut AcpiInfo) {
    let total_len = header.length as usize;
    let hdr_size = core::mem::size_of::<SdtHeader>();
    // 8 bytes reserved after SDT header
    let mut offset = hdr_size + 8;
    while offset + 16 <= total_len {
        let base_addr = read_unaligned(ptr.add(offset) as *const u64);
        let segment = read_unaligned(ptr.add(offset + 8) as *const u16);
        let start_bus = *ptr.add(offset + 10);
        let end_bus = *ptr.add(offset + 11);
        info.mcfg_allocations.push(McfgAllocation {
            base_addr,
            segment,
            start_bus,
            end_bus,
        });
        offset += 16;
    }
}

/// Initialize ACPI by parsing RSDP, RSDT/XSDT, MADT (`APIC`), and MCFG.
pub fn init(rsdp_addr: *const u8) -> Option<AcpiInfo> {
    if rsdp_addr.is_null() {
        return None;
    }

    let rsdp_ptr = resolve_rsdp_ptr(rsdp_addr);
    let v1 = unsafe { read_unaligned(rsdp_ptr as *const RsdpV1) };
    if &v1.signature != RSDP_SIGNATURE {
        return None;
    }
    if !validate_checksum(rsdp_ptr, core::mem::size_of::<RsdpV1>()) {
        return None;
    }

    let mut use_xsdt = false;
    let mut xsdt_phys = 0u64;
    if v1.revision >= 2 {
        let v2 = unsafe { read_unaligned(rsdp_ptr as *const RsdpV2) };
        let len = v2.length as usize;
        if len >= core::mem::size_of::<RsdpV2>()
            && validate_checksum(rsdp_ptr, len)
            && v2.xsdt_address != 0
        {
            use_xsdt = true;
            xsdt_phys = v2.xsdt_address;
        }
    }

    let mut info = AcpiInfo {
        revision: v1.revision,
        lapic_addr: 0,
        madt_flags: 0,
        local_apics: Vec::new(),
        io_apics: Vec::new(),
        interrupt_overrides: Vec::new(),
        mcfg_allocations: Vec::new(),
    };

    let hdr_size = core::mem::size_of::<SdtHeader>();

    unsafe {
        if use_xsdt {
            if let Some((xsdt_ptr, xsdt_hdr)) = read_sdt_header(xsdt_phys) {
                if &xsdt_hdr.signature == b"XSDT" {
                    let entries_bytes = (xsdt_hdr.length as usize).saturating_sub(hdr_size);
                    let count = entries_bytes / 8;
                    for i in 0..count {
                        let table_phys =
                            read_unaligned(xsdt_ptr.add(hdr_size + i * 8) as *const u64);
                        if let Some((tbl_ptr, tbl_hdr)) = read_sdt_header(table_phys) {
                            match &tbl_hdr.signature {
                                b"APIC" => parse_madt(tbl_ptr, &tbl_hdr, &mut info),
                                b"MCFG" => parse_mcfg(tbl_ptr, &tbl_hdr, &mut info),
                                _ => {}
                            }
                        }
                    }
                    return Some(info);
                }
            }
        }

        if v1.rsdt_address != 0 {
            if let Some((rsdt_ptr, rsdt_hdr)) = read_sdt_header(v1.rsdt_address as u64) {
                if &rsdt_hdr.signature == b"RSDT" {
                    let entries_bytes = (rsdt_hdr.length as usize).saturating_sub(hdr_size);
                    let count = entries_bytes / 4;
                    for i in 0..count {
                        let table_phys =
                            read_unaligned(rsdt_ptr.add(hdr_size + i * 4) as *const u32) as u64;
                        if let Some((tbl_ptr, tbl_hdr)) = read_sdt_header(table_phys) {
                            match &tbl_hdr.signature {
                                b"APIC" => parse_madt(tbl_ptr, &tbl_hdr, &mut info),
                                b"MCFG" => parse_mcfg(tbl_ptr, &tbl_hdr, &mut info),
                                _ => {}
                            }
                        }
                    }
                    return Some(info);
                }
            }
        }
    }

    None
}
