//! Clean-room xHCI (USB 3.x) host controller driver.
#![allow(dead_code)]

use core::ptr::{read_volatile, write_bytes, write_volatile};

use crate::mm::{alloc_frame, dealloc_frame, stats, PAGE_SIZE};
use crate::pci::{PciClass, PciDevice};

// ---------------------------------------------------------------------------
// Capability Register Offsets (relative to MMIO base)
// ---------------------------------------------------------------------------
pub const CAP_CAPLENGTH: usize = 0x00; // u8
pub const CAP_HCIVERSION: usize = 0x02; // u16
pub const CAP_HCSPARAMS1: usize = 0x04; // u32
pub const CAP_HCSPARAMS2: usize = 0x08; // u32
pub const CAP_HCSPARAMS3: usize = 0x0C; // u32
pub const CAP_HCCPARAMS1: usize = 0x10; // u32
pub const CAP_DBOFF: usize = 0x14; // u32
pub const CAP_RTSOFF: usize = 0x18; // u32
pub const CAP_HCCPARAMS2: usize = 0x1C; // u32

// ---------------------------------------------------------------------------
// Operational Register Offsets (relative to op_base = mmio_base + cap_length)
// ---------------------------------------------------------------------------
pub const OP_USBCMD: usize = 0x00; // u32
pub const OP_USBSTS: usize = 0x04; // u32
pub const OP_PAGESIZE: usize = 0x08; // u32
pub const OP_DNCTRL: usize = 0x14; // u32
pub const OP_CRCR: usize = 0x18; // u64
pub const OP_DCBAAP: usize = 0x30; // u64
pub const OP_CONFIG: usize = 0x38; // u32
pub const OP_PORTSC_BASE: usize = 0x400;
pub const OP_PORT_STRIDE: usize = 0x10;

// USBCMD bits
pub const USBCMD_RS: u32 = 1 << 0; // Run/Stop
pub const USBCMD_HCRST: u32 = 1 << 1; // Host Controller Reset
pub const USBCMD_INTE: u32 = 1 << 2; // Interrupter Enable
pub const USBCMD_HSEE: u32 = 1 << 3; // Host System Error Enable

// USBSTS bits
pub const USBSTS_HCH: u32 = 1 << 0; // HC Halted
pub const USBSTS_HSE: u32 = 1 << 2; // Host System Error
pub const USBSTS_EINT: u32 = 1 << 3; // Event Interrupt
pub const USBSTS_PCD: u32 = 1 << 4; // Port Change Detect
pub const USBSTS_CNR: u32 = 1 << 11; // Controller Not Ready

// CRCR bits
pub const CRCR_RCS: u64 = 1 << 0; // Ring Cycle State
pub const CRCR_CS: u64 = 1 << 1; // Command Stop
pub const CRCR_CA: u64 = 1 << 2; // Command Abort
pub const CRCR_CRR: u64 = 1 << 3; // Command Ring Running

// PORTSC bits
pub const PORTSC_CCS: u32 = 1 << 0; // Current Connect Status
pub const PORTSC_PED: u32 = 1 << 1; // Port Enabled/Disabled
pub const PORTSC_OCA: u32 = 1 << 3; // Over-current Active
pub const PORTSC_PR: u32 = 1 << 4; // Port Reset
pub const PORTSC_PP: u32 = 1 << 9; // Port Power
pub const PORTSC_CSC: u32 = 1 << 17; // Connect Status Change
pub const PORTSC_PRC: u32 = 1 << 21; // Port Reset Change

// ---------------------------------------------------------------------------
// TRB (Transfer Request Block) - 16 bytes
// ---------------------------------------------------------------------------
pub const TRB_TYPE_NORMAL: u32 = 1;
pub const TRB_TYPE_SETUP_STAGE: u32 = 2;
pub const TRB_TYPE_DATA_STAGE: u32 = 3;
pub const TRB_TYPE_STATUS_STAGE: u32 = 4;
pub const TRB_TYPE_LINK: u32 = 6;
pub const TRB_TYPE_NOOP: u32 = 8;
pub const TRB_TYPE_ENABLE_SLOT: u32 = 9;
pub const TRB_TYPE_DISABLE_SLOT: u32 = 10;
pub const TRB_TYPE_ADDRESS_DEVICE: u32 = 11;
pub const TRB_TYPE_CONFIGURE_ENDPOINT: u32 = 12;
pub const TRB_TYPE_EVALUATE_CONTEXT: u32 = 13;
pub const TRB_TYPE_RESET_ENDPOINT: u32 = 14;
pub const TRB_TYPE_STOP_ENDPOINT: u32 = 15;
pub const TRB_TYPE_SET_TR_DEQUEUE: u32 = 16;
pub const TRB_TYPE_RESET_DEVICE: u32 = 17;
pub const TRB_TYPE_NOOP_CMD: u32 = 23;
pub const TRB_TYPE_TRANSFER_EVENT: u32 = 32;
pub const TRB_TYPE_CMD_COMPLETION_EVENT: u32 = 33;
pub const TRB_TYPE_PORT_STATUS_CHANGE_EVENT: u32 = 34;

pub const TRB_CYCLE_BIT: u32 = 1 << 0;
pub const TRB_TC_BIT: u32 = 1 << 1; // Toggle Cycle (Link TRB)
pub const TRB_IOC_BIT: u32 = 1 << 5; // Interrupt On Completion

pub const CMD_RING_TRB_COUNT: usize = PAGE_SIZE / core::mem::size_of::<Trb>();

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Trb {
    pub parameter: u64,
    pub status: u32,
    pub control: u32,
}

impl Trb {
    pub const fn zeroed() -> Self {
        Self {
            parameter: 0,
            status: 0,
            control: 0,
        }
    }

    pub const fn new(parameter: u64, status: u32, trb_type: u32, flags: u32) -> Self {
        Self {
            parameter,
            status,
            control: ((trb_type & 0x3F) << 10) | (flags & !0xFC00),
        }
    }

    #[inline]
    pub const fn trb_type(&self) -> u32 {
        (self.control >> 10) & 0x3F
    }

    #[inline]
    pub const fn cycle_bit(&self) -> bool {
        (self.control & TRB_CYCLE_BIT) != 0
    }

    #[inline]
    pub fn set_cycle_bit(&mut self, cycle: bool) {
        if cycle {
            self.control |= TRB_CYCLE_BIT;
        } else {
            self.control &= !TRB_CYCLE_BIT;
        }
    }

    pub const fn noop_cmd(cycle: bool) -> Self {
        let flags = if cycle { TRB_CYCLE_BIT } else { 0 };
        Self::new(0, 0, TRB_TYPE_NOOP_CMD, flags)
    }

    pub const fn enable_slot_cmd(slot_type: u8, cycle: bool) -> Self {
        let flags = ((slot_type as u32 & 0x1F) << 16) | if cycle { TRB_CYCLE_BIT } else { 0 };
        Self::new(0, 0, TRB_TYPE_ENABLE_SLOT, flags)
    }

    pub const fn address_device_cmd(
        input_ctx_phys: u64,
        slot_id: u8,
        bsr: bool,
        cycle: bool,
    ) -> Self {
        let mut flags = (slot_id as u32) << 24;
        if bsr {
            flags |= 1 << 9;
        }
        if cycle {
            flags |= TRB_CYCLE_BIT;
        }
        Self::new(input_ctx_phys, 0, TRB_TYPE_ADDRESS_DEVICE, flags)
    }

    pub const fn link(next_segment_phys: u64, toggle_cycle: bool, cycle: bool) -> Self {
        let mut flags = 0;
        if toggle_cycle {
            flags |= TRB_TC_BIT;
        }
        if cycle {
            flags |= TRB_CYCLE_BIT;
        }
        Self::new(next_segment_phys, 0, TRB_TYPE_LINK, flags)
    }
}

// ---------------------------------------------------------------------------
// xHCI Host Controller
// ---------------------------------------------------------------------------
#[derive(Debug)]
pub struct XhciController {
    pub pci_bus: u8,
    pub pci_dev: u8,
    pub pci_func: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub mmio_phys: u64,
    pub mmio_base: u64,
    pub cap_length: u8,
    pub hci_version: u16,
    pub op_base: u64,
    pub rt_base: u64,
    pub db_base: u64,
    pub max_slots: u8,
    pub max_intrs: u16,
    pub max_ports: u8,
    pub context_size_64: bool,
    pub dcbaa_phys: u64,
    pub cmd_ring_phys: u64,
    pub cmd_ring_enqueue: usize,
    pub cmd_ring_cycle: bool,
    pub msi_cap: Option<u8>,
    pub msix_cap: Option<u8>,
}

impl XhciController {
    /// Probe a PCI device for an xHCI USB 3.x host controller.
    ///
    /// Maps BAR0 via `crate::mm::stats().hhdm_offset`, reads capability parameters
    /// (`max_slots`, `max_ports`, `DBOFF`, `RTSOFF`), and allocates the DCBAA and
    /// Command Ring frames via `crate::mm::alloc_frame()`.
    pub fn probe(dev: &PciDevice) -> Option<Self> {
        if dev.class() != PciClass::XhciUsb {
            return None;
        }

        let bar0 = dev.bars[0];
        if bar0 == 0 {
            return None;
        }

        let mmio_phys = bar0 & !0xF;
        let hhdm_offset = stats().hhdm_offset;
        let mmio_base = mmio_phys.wrapping_add(hhdm_offset);

        let cap_length = unsafe { read_volatile((mmio_base + CAP_CAPLENGTH as u64) as *const u8) };
        if cap_length < 0x20 {
            return None;
        }

        let hci_version =
            unsafe { read_volatile((mmio_base + CAP_HCIVERSION as u64) as *const u16) };
        let hcsparams1 =
            unsafe { read_volatile((mmio_base + CAP_HCSPARAMS1 as u64) as *const u32) };
        let hccparams1 =
            unsafe { read_volatile((mmio_base + CAP_HCCPARAMS1 as u64) as *const u32) };
        let dboff = unsafe { read_volatile((mmio_base + CAP_DBOFF as u64) as *const u32) };
        let rtsoff = unsafe { read_volatile((mmio_base + CAP_RTSOFF as u64) as *const u32) };

        let max_slots = (hcsparams1 & 0xFF) as u8;
        let max_intrs = ((hcsparams1 >> 8) & 0x7FF) as u16;
        let max_ports = ((hcsparams1 >> 24) & 0xFF) as u8;
        let context_size_64 = (hccparams1 & (1 << 2)) != 0;

        let op_base = mmio_base + cap_length as u64;
        let db_base = mmio_base + (dboff & !0x3) as u64;
        let rt_base = mmio_base + (rtsoff & !0x1F) as u64;

        let dcbaa_phys = alloc_frame()?;
        let cmd_ring_phys = match alloc_frame() {
            Some(p) => p,
            None => {
                dealloc_frame(dcbaa_phys);
                return None;
            }
        };

        // Zero-initialize DCBAA and Command Ring frames via HHDM.
        unsafe {
            write_bytes((dcbaa_phys + hhdm_offset) as *mut u8, 0, PAGE_SIZE);
            write_bytes((cmd_ring_phys + hhdm_offset) as *mut u8, 0, PAGE_SIZE);
        }

        // Place a Link TRB at the last slot of the Command Ring pointing back to start.
        let link_idx = CMD_RING_TRB_COUNT - 1;
        let ring_virt = (cmd_ring_phys + hhdm_offset) as *mut Trb;
        unsafe {
            write_volatile(
                ring_virt.add(link_idx),
                Trb::link(cmd_ring_phys, true, true),
            );
        }

        Some(Self {
            pci_bus: dev.bus,
            pci_dev: dev.dev,
            pci_func: dev.func,
            vendor_id: dev.vendor_id,
            device_id: dev.device_id,
            mmio_phys,
            mmio_base,
            cap_length,
            hci_version,
            op_base,
            rt_base,
            db_base,
            max_slots,
            max_intrs,
            max_ports,
            context_size_64,
            dcbaa_phys,
            cmd_ring_phys,
            cmd_ring_enqueue: 0,
            cmd_ring_cycle: true,
            msi_cap: dev.msi_cap,
            msix_cap: dev.msix_cap,
        })
    }

    #[inline]
    pub fn read_cap32(&self, offset: usize) -> u32 {
        unsafe { read_volatile((self.mmio_base + offset as u64) as *const u32) }
    }

    #[inline]
    pub fn read_op32(&self, offset: usize) -> u32 {
        unsafe { read_volatile((self.op_base + offset as u64) as *const u32) }
    }

    #[inline]
    pub fn write_op32(&self, offset: usize, val: u32) {
        unsafe { write_volatile((self.op_base + offset as u64) as *mut u32, val) }
    }

    #[inline]
    pub fn read_op64(&self, offset: usize) -> u64 {
        let lo = self.read_op32(offset) as u64;
        let hi = self.read_op32(offset + 4) as u64;
        (hi << 32) | lo
    }

    #[inline]
    pub fn write_op64(&self, offset: usize, val: u64) {
        self.write_op32(offset, val as u32);
        self.write_op32(offset + 4, (val >> 32) as u32);
    }

    #[inline]
    pub fn read_portsc(&self, port_index: u8) -> Option<u32> {
        if port_index >= self.max_ports {
            return None;
        }
        let off = OP_PORTSC_BASE + (port_index as usize) * OP_PORT_STRIDE;
        Some(self.read_op32(off))
    }

    #[inline]
    pub fn write_portsc(&self, port_index: u8, val: u32) -> bool {
        if port_index >= self.max_ports {
            return false;
        }
        let off = OP_PORTSC_BASE + (port_index as usize) * OP_PORT_STRIDE;
        self.write_op32(off, val);
        true
    }

    /// Halt and reset the xHCI controller, waiting for CNR to clear.
    pub fn reset(&self, max_spins: usize) -> bool {
        let cmd = self.read_op32(OP_USBCMD);
        self.write_op32(OP_USBCMD, cmd & !USBCMD_RS);

        let mut spins = 0;
        while (self.read_op32(OP_USBSTS) & USBSTS_HCH) == 0 {
            if spins >= max_spins {
                return false;
            }
            spins += 1;
            core::hint::spin_loop();
        }

        self.write_op32(OP_USBCMD, USBCMD_HCRST);
        spins = 0;
        while (self.read_op32(OP_USBCMD) & USBCMD_HCRST) != 0
            || (self.read_op32(OP_USBSTS) & USBSTS_CNR) != 0
        {
            if spins >= max_spins {
                return false;
            }
            spins += 1;
            core::hint::spin_loop();
        }

        true
    }

    /// Program `CONFIG`, `DCBAAP`, and `CRCR` with the allocated rings.
    pub fn init_rings(&self) {
        self.write_op32(OP_CONFIG, self.max_slots as u32);
        self.write_op64(OP_DCBAAP, self.dcbaa_phys);
        let crcr = (self.cmd_ring_phys & !0x3F) | if self.cmd_ring_cycle { CRCR_RCS } else { 0 };
        self.write_op64(OP_CRCR, crcr);
    }

    /// Start the host controller (`USBCMD.RS = 1`).
    pub fn start(&self) {
        let cmd = self.read_op32(OP_USBCMD);
        self.write_op32(OP_USBCMD, cmd | USBCMD_RS | USBCMD_INTE);
    }

    /// Stop the host controller (`USBCMD.RS = 0`).
    pub fn stop(&self) {
        let cmd = self.read_op32(OP_USBCMD);
        self.write_op32(OP_USBCMD, cmd & !USBCMD_RS);
    }

    /// Ring a doorbell register (`slot_id = 0` for Host Controller Command Ring).
    #[inline]
    pub fn ring_doorbell(&self, slot_id: u8, target: u32) {
        let addr = self.db_base + (slot_id as u64) * 4;
        unsafe { write_volatile(addr as *mut u32, target) };
    }

    /// Enqueue a command TRB onto the Command Ring and ring doorbell 0.
    pub fn enqueue_command(&mut self, mut trb: Trb) {
        let hhdm_offset = stats().hhdm_offset;
        let ring_virt = (self.cmd_ring_phys + hhdm_offset) as *mut Trb;

        trb.set_cycle_bit(self.cmd_ring_cycle);
        unsafe {
            write_volatile(ring_virt.add(self.cmd_ring_enqueue), trb);
        }

        self.cmd_ring_enqueue += 1;
        if self.cmd_ring_enqueue >= CMD_RING_TRB_COUNT - 1 {
            // Update Link TRB cycle bit and wrap around.
            let link = Trb::link(self.cmd_ring_phys, true, self.cmd_ring_cycle);
            unsafe {
                write_volatile(ring_virt.add(CMD_RING_TRB_COUNT - 1), link);
            }
            self.cmd_ring_enqueue = 0;
            self.cmd_ring_cycle = !self.cmd_ring_cycle;
        }

        self.ring_doorbell(0, 0);
    }

    /// Count how many root hub ports currently report a connected device (`PORTSC.CCS`).
    pub fn connected_ports(&self) -> u8 {
        let mut count = 0u8;
        for port in 0..self.max_ports {
            if let Some(portsc) = self.read_portsc(port) {
                if (portsc & PORTSC_CCS) != 0 {
                    count += 1;
                }
            }
        }
        count
    }
}
