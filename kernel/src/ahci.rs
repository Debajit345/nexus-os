//! Clean-room AHCI 1.3.1 / SATA 3.0 HBA Driver & NCQ Command Engine.
//!
//! Implements:
//! - HBA Generic Host Control & per-port MMIO register definitions
//! - SATA FIS structures (Reg H2D, Reg D2H, PIO Setup, DMA Setup, Set Device Bits, Received FIS)
//! - Command List, Command Header, Command Table, and PRDT scatter-gather builder
//! - 32-slot Native Command Queuing (NCQ) tag queue & PxSACT completion tracker
//! - ATA IDENTIFY DEVICE (0xEC) parser (LBA48, 512e/4Kn, NCQ depth, SATA Gen1-3, TRIM, strings)
//! - `BlockDevice` integration and `self_test()`
#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::mem::size_of;
use core::ptr::{read_volatile, write_bytes, write_volatile};

use crate::blk::{BlockDevice, SECTOR_SIZE};
use crate::mm::{alloc_frame, dealloc_frame, stats, SpinLock, PAGE_SIZE};
use crate::pci::{PciClass, PciDevice};

/// Maximum number of AHCI ports per HBA.
pub const MAX_PORTS: usize = 32;
/// Maximum number of command slots / NCQ tags per port.
pub const MAX_CMD_SLOTS: usize = 32;
/// Maximum bytes per single PRDT entry (4 MiB).
pub const MAX_PRDT_BYTES: usize = 4 * 1024 * 1024;
/// Default PRDT entries per command table in DMA frame layout.
pub const DEFAULT_PRDT_ENTRIES: usize = 8;
/// Polling timeout iterations for MMIO waits.
const POLL_TIMEOUT_ITERS: usize = 1_000_000;

/// AHCI driver error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AhciError {
    NoController,
    InvalidBar,
    HbaResetTimeout,
    BiosHandoffTimeout,
    PortNotImplemented,
    NoDevicePresent,
    UnsupportedDevice,
    PortBusy,
    CommandTimeout,
    TaskFileError(u8),
    QueueFull,
    InvalidTag,
    TagNotActive,
    InvalidBuffer,
    UnalignedAddress,
    PrdtOverflow,
    OutOfMemory,
    OutOfRange,
}

impl AhciError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoController => "ahci: no AHCI controller found",
            Self::InvalidBar => "ahci: invalid ABAR (BAR5)",
            Self::HbaResetTimeout => "ahci: HBA reset timeout",
            Self::BiosHandoffTimeout => "ahci: BIOS/OS handoff timeout",
            Self::PortNotImplemented => "ahci: port not implemented",
            Self::NoDevicePresent => "ahci: no SATA device present on port",
            Self::UnsupportedDevice => "ahci: unsupported device signature",
            Self::PortBusy => "ahci: port task file busy",
            Self::CommandTimeout => "ahci: command completion timeout",
            Self::TaskFileError(_) => "ahci: task file error",
            Self::QueueFull => "ahci: NCQ tag queue full",
            Self::InvalidTag => "ahci: invalid NCQ tag",
            Self::TagNotActive => "ahci: NCQ tag not active",
            Self::InvalidBuffer => "ahci: invalid buffer length",
            Self::UnalignedAddress => "ahci: unaligned DMA address",
            Self::PrdtOverflow => "ahci: PRDT entry capacity exceeded",
            Self::OutOfMemory => "ahci: out of DMA memory",
            Self::OutOfRange => "ahci: LBA out of range",
        }
    }
}

impl fmt::Display for AhciError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TaskFileError(err) => write!(f, "ahci: task file error (ERR={:#04x})", err),
            _ => f.write_str(self.as_str()),
        }
    }
}

impl From<AhciError> for &'static str {
    fn from(err: AhciError) -> Self {
        err.as_str()
    }
}

/// HBA Generic Host Control MMIO register offsets.
pub mod hba_regs {
    pub const CAP: usize = 0x00;
    pub const GHC: usize = 0x04;
    pub const IS: usize = 0x08;
    pub const PI: usize = 0x0C;
    pub const VS: usize = 0x10;
    pub const CCC_CTL: usize = 0x14;
    pub const CCC_PORTS: usize = 0x18;
    pub const EM_LOC: usize = 0x1C;
    pub const EM_CTL: usize = 0x20;
    pub const CAP2: usize = 0x24;
    pub const BOHC: usize = 0x28;
    pub const PORT_BASE: usize = 0x100;
    pub const PORT_STRIDE: usize = 0x80;
}

/// HBA Port MMIO register offsets (relative to `PORT_BASE + port * PORT_STRIDE`).
pub mod port_regs {
    pub const CLB: usize = 0x00;
    pub const CLBU: usize = 0x04;
    pub const FB: usize = 0x08;
    pub const FBU: usize = 0x0C;
    pub const IS: usize = 0x10;
    pub const IE: usize = 0x14;
    pub const CMD: usize = 0x18;
    pub const TFD: usize = 0x20;
    pub const SIG: usize = 0x24;
    pub const SSTS: usize = 0x28;
    pub const SCTL: usize = 0x2C;
    pub const SERR: usize = 0x30;
    pub const SACT: usize = 0x34;
    pub const CI: usize = 0x38;
    pub const SNTF: usize = 0x3C;
    pub const FBS: usize = 0x40;
}

// GHC bits
pub const GHC_HR: u32 = 1 << 0;
pub const GHC_IE: u32 = 1 << 1;
pub const GHC_AE: u32 = 1 << 31;

// CAP2 bits
pub const CAP2_BOH: u32 = 1 << 0;

// BOHC bits
pub const BOHC_BOS: u32 = 1 << 0;
pub const BOHC_OOS: u32 = 1 << 1;
pub const BOHC_BB: u32 = 1 << 4;

// PxCMD bits
pub const PXCMD_ST: u32 = 1 << 0;
pub const PXCMD_SUD: u32 = 1 << 1;
pub const PXCMD_POD: u32 = 1 << 2;
pub const PXCMD_CLO: u32 = 1 << 3;
pub const PXCMD_FRE: u32 = 1 << 4;
pub const PXCMD_FR: u32 = 1 << 14;
pub const PXCMD_CR: u32 = 1 << 15;
pub const PXCMD_ICC_MASK: u32 = 0xF << 28;
pub const PXCMD_ICC_ACTIVE: u32 = 0x1 << 28;

// PxTFD bits
pub const PXTFD_ERR: u32 = 1 << 0;
pub const PXTFD_DRQ: u32 = 1 << 3;
pub const PXTFD_BSY: u32 = 1 << 7;

// PxIS bits
pub const PXIS_DHRS: u32 = 1 << 0;
pub const PXIS_PSS: u32 = 1 << 1;
pub const PXIS_DSS: u32 = 1 << 2;
pub const PXIS_SDBS: u32 = 1 << 3;
pub const PXIS_TFES: u32 = 1 << 30;

// Device signatures (PxSIG)
pub const SATA_SIG_ATA: u32 = 0x0000_0101;
pub const SATA_SIG_ATAPI: u32 = 0xEB14_0101;
pub const SATA_SIG_SEMB: u32 = 0xC33C_0101;
pub const SATA_SIG_PM: u32 = 0x9669_0101;

// ATA commands
pub const ATA_CMD_READ_DMA_EXT: u8 = 0x25;
pub const ATA_CMD_WRITE_DMA_EXT: u8 = 0x35;
pub const ATA_CMD_READ_FPDMA_QUEUED: u8 = 0x60;
pub const ATA_CMD_WRITE_FPDMA_QUEUED: u8 = 0x61;
pub const ATA_CMD_FLUSH_CACHE_EXT: u8 = 0xEA;
pub const ATA_CMD_IDENTIFY: u8 = 0xEC;

// FIS types
pub const FIS_TYPE_REG_H2D: u8 = 0x27;
pub const FIS_TYPE_REG_D2H: u8 = 0x34;
pub const FIS_TYPE_DMA_ACT: u8 = 0x39;
pub const FIS_TYPE_DMA_SETUP: u8 = 0x41;
pub const FIS_TYPE_DATA: u8 = 0x46;
pub const FIS_TYPE_BIST: u8 = 0x58;
pub const FIS_TYPE_PIO_SETUP: u8 = 0x5F;
pub const FIS_TYPE_DEV_BITS: u8 = 0xA1;

/// Decoded HBA capabilities from `CAP`, `CAP2`, and `VS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HbaCapabilities {
    pub version_major: u16,
    pub version_minor: u16,
    pub num_ports: u8,
    pub num_cmd_slots: u8,
    pub interface_speed: u8,
    pub supports_64bit: bool,
    pub supports_ncq: bool,
    pub supports_staggered_spinup: bool,
    pub supports_ahci_only: bool,
    pub bios_handoff: bool,
}

impl HbaCapabilities {
    pub const fn decode(cap: u32, cap2: u32, vs: u32) -> Self {
        Self {
            version_major: (vs >> 16) as u16,
            version_minor: (vs & 0xFFFF) as u16,
            num_ports: ((cap & 0x1F) as u8) + 1,
            num_cmd_slots: (((cap >> 8) & 0x1F) as u8) + 1,
            interface_speed: ((cap >> 20) & 0x0F) as u8,
            supports_64bit: (cap & (1 << 31)) != 0,
            supports_ncq: (cap & (1 << 30)) != 0,
            supports_staggered_spinup: (cap & (1 << 27)) != 0,
            supports_ahci_only: (cap & (1 << 18)) != 0,
            bios_handoff: (cap2 & CAP2_BOH) != 0,
        }
    }
}

/// SATA link speed reported in `PxSSTS.SPD`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SataSpeed {
    None,
    Gen1_1_5Gbps,
    Gen2_3_0Gbps,
    Gen3_6_0Gbps,
    Unknown(u8),
}

impl SataSpeed {
    pub const fn from_ssts(ssts: u32) -> Self {
        match ((ssts >> 4) & 0x0F) as u8 {
            0 => Self::None,
            1 => Self::Gen1_1_5Gbps,
            2 => Self::Gen2_3_0Gbps,
            3 => Self::Gen3_6_0Gbps,
            other => Self::Unknown(other),
        }
    }
}

/// Attached device type on an AHCI port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortDeviceType {
    None,
    Sata,
    Satapi,
    Semb,
    PortMultiplier,
    Unknown(u32),
}

impl PortDeviceType {
    pub const fn from_sig(sig: u32) -> Self {
        match sig {
            SATA_SIG_ATA => Self::Sata,
            SATA_SIG_ATAPI => Self::Satapi,
            SATA_SIG_SEMB => Self::Semb,
            SATA_SIG_PM => Self::PortMultiplier,
            other => Self::Unknown(other),
        }
    }
}

/// Register - Host to Device FIS (20 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FisRegH2D {
    pub fis_type: u8,
    pub pmport_c: u8,
    pub command: u8,
    pub featurel: u8,
    pub lba0: u8,
    pub lba1: u8,
    pub lba2: u8,
    pub device: u8,
    pub lba3: u8,
    pub lba4: u8,
    pub lba5: u8,
    pub featureh: u8,
    pub countl: u8,
    pub counth: u8,
    pub icc: u8,
    pub control: u8,
    pub rsv1: [u8; 4],
}

const _: () = assert!(size_of::<FisRegH2D>() == 20);

impl FisRegH2D {
    /// Build a standard LBA48 command FIS (e.g. IDENTIFY, READ/WRITE DMA EXT, FLUSH).
    pub const fn new_command(command: u8, lba: u64, count: u16, lba_mode: bool) -> Self {
        Self {
            fis_type: FIS_TYPE_REG_H2D,
            pmport_c: 0x80, // C = 1 (command register update)
            command,
            featurel: 0,
            lba0: (lba & 0xFF) as u8,
            lba1: ((lba >> 8) & 0xFF) as u8,
            lba2: ((lba >> 16) & 0xFF) as u8,
            device: if lba_mode { 1 << 6 } else { 0 },
            lba3: ((lba >> 24) & 0xFF) as u8,
            lba4: ((lba >> 32) & 0xFF) as u8,
            lba5: ((lba >> 40) & 0xFF) as u8,
            featureh: 0,
            countl: (count & 0xFF) as u8,
            counth: (count >> 8) as u8,
            icc: 0,
            control: 0,
            rsv1: [0; 4],
        }
    }

    /// Build an NCQ FPDMA QUEUED command FIS (`READ FPDMA QUEUED` 0x60 or `WRITE FPDMA QUEUED` 0x61).
    ///
    /// Per SATA 3.0:
    /// - Sector count is placed in `featurel` / `featureh` (0 means 65536 sectors).
    /// - NCQ tag (0..31) is placed in `countl[7:3]` (`tag << 3`).
    /// - `device` has bit 6 set (LBA mode) and optionally bit 7 (FUA).
    pub const fn new_ncq(command: u8, tag: u8, lba: u64, sector_count: u16, fua: bool) -> Self {
        let dev = (1 << 6) | if fua { 1 << 7 } else { 0 };
        Self {
            fis_type: FIS_TYPE_REG_H2D,
            pmport_c: 0x80,
            command,
            featurel: (sector_count & 0xFF) as u8,
            lba0: (lba & 0xFF) as u8,
            lba1: ((lba >> 8) & 0xFF) as u8,
            lba2: ((lba >> 16) & 0xFF) as u8,
            device: dev,
            lba3: ((lba >> 24) & 0xFF) as u8,
            lba4: ((lba >> 32) & 0xFF) as u8,
            lba5: ((lba >> 40) & 0xFF) as u8,
            featureh: (sector_count >> 8) as u8,
            countl: (tag & 0x1F) << 3,
            counth: 0,
            icc: 0,
            control: 0,
            rsv1: [0; 4],
        }
    }
}

/// Register - Device to Host FIS (20 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FisRegD2H {
    pub fis_type: u8,
    pub pmport_i: u8,
    pub status: u8,
    pub error: u8,
    pub lba0: u8,
    pub lba1: u8,
    pub lba2: u8,
    pub device: u8,
    pub lba3: u8,
    pub lba4: u8,
    pub lba5: u8,
    pub rsv2: u8,
    pub countl: u8,
    pub counth: u8,
    pub rsv3: [u8; 6],
}

const _: () = assert!(size_of::<FisRegD2H>() == 20);

/// Set Device Bits - Device to Host FIS (8 bytes, used for NCQ completions).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FisSetDevBits {
    pub fis_type: u8,
    pub pmport_i_n: u8,
    pub status: u8,
    pub error: u8,
    pub sactive: u32,
}

const _: () = assert!(size_of::<FisSetDevBits>() == 8);

/// Received FIS structure (256 bytes, 256-byte aligned).
#[repr(C, align(256))]
#[derive(Debug, Clone, Copy)]
pub struct ReceivedFis {
    pub dsfis: [u8; 28],
    pub pad0: [u8; 4],
    pub psfis: [u8; 20],
    pub pad1: [u8; 12],
    pub rfis: FisRegD2H,
    pub pad2: [u8; 4],
    pub sdbfis: FisSetDevBits,
    pub ufis: [u8; 64],
    pub rsv: [u8; 96],
}

const _: () = assert!(size_of::<ReceivedFis>() == 256);

impl Default for ReceivedFis {
    fn default() -> Self {
        Self {
            dsfis: [0; 28],
            pad0: [0; 4],
            psfis: [0; 20],
            pad1: [0; 12],
            rfis: FisRegD2H::default(),
            pad2: [0; 4],
            sdbfis: FisSetDevBits::default(),
            ufis: [0; 64],
            rsv: [0; 96],
        }
    }
}

/// AHCI Command Header (32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommandHeader {
    /// Bits 4:0 = CFL (Command FIS length in DWORDs), bit 5 = A (ATAPI),
    /// bit 6 = W (Write), bit 7 = P (Prefetchable), bit 10 = C (Clear Busy).
    pub flags: u16,
    /// Physical Region Descriptor Table Length (number of entries).
    pub prdtl: u16,
    /// Physical Region Descriptor Byte Count transferred.
    pub prdbc: u32,
    /// Command Table Descriptor Base Address (lower 32 bits, 128-byte aligned).
    pub ctba: u32,
    /// Command Table Descriptor Base Address (upper 32 bits).
    pub ctbau: u32,
    pub rsv: [u32; 4],
}

const _: () = assert!(size_of::<CommandHeader>() == 32);

impl CommandHeader {
    pub fn configure(
        &mut self,
        cfl_dwords: u8,
        write: bool,
        prefetchable: bool,
        clear_busy: bool,
        prdtl: u16,
        cmd_table_phys: u64,
    ) {
        let mut flags = (cfl_dwords as u16) & 0x1F;
        if write {
            flags |= 1 << 6;
        }
        if prefetchable {
            flags |= 1 << 7;
        }
        if clear_busy {
            flags |= 1 << 10;
        }
        self.flags = flags;
        self.prdtl = prdtl;
        self.prdbc = 0;
        self.ctba = cmd_table_phys as u32;
        self.ctbau = (cmd_table_phys >> 32) as u32;
        self.rsv = [0; 4];
    }
}

/// AHCI Command List (32 Command Headers = 1024 bytes, 1024-byte aligned).
#[repr(C, align(1024))]
#[derive(Debug, Clone, Copy)]
pub struct CommandList {
    pub headers: [CommandHeader; MAX_CMD_SLOTS],
}

const _: () = assert!(size_of::<CommandList>() == 1024);

impl Default for CommandList {
    fn default() -> Self {
        Self {
            headers: [CommandHeader::default(); MAX_CMD_SLOTS],
        }
    }
}

/// Physical Region Descriptor Table (PRDT) entry (16 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrdtEntry {
    pub dba: u32,
    pub dbau: u32,
    pub rsv: u32,
    /// Bits 21:0 = byte count minus 1 (max 4 MiB - 1); bit 31 = Interrupt on Completion.
    pub dbc_i: u32,
}

const _: () = assert!(size_of::<PrdtEntry>() == 16);

impl PrdtEntry {
    pub fn new(
        phys_addr: u64,
        byte_count: usize,
        interrupt_on_complete: bool,
    ) -> Result<Self, AhciError> {
        if (phys_addr & 1) != 0 {
            return Err(AhciError::UnalignedAddress);
        }
        if byte_count == 0 || byte_count > MAX_PRDT_BYTES {
            return Err(AhciError::InvalidBuffer);
        }
        let mut dbc_i = ((byte_count - 1) as u32) & 0x003F_FFFF;
        if interrupt_on_complete {
            dbc_i |= 1 << 31;
        }
        Ok(Self {
            dba: phys_addr as u32,
            dbau: (phys_addr >> 32) as u32,
            rsv: 0,
            dbc_i,
        })
    }

    #[inline]
    pub const fn byte_count(&self) -> usize {
        ((self.dbc_i & 0x003F_FFFF) as usize) + 1
    }

    #[inline]
    pub const fn has_ioc(&self) -> bool {
        (self.dbc_i & (1 << 31)) != 0
    }
}

/// Command Table fixed header (128 bytes: 64B CFIS + 16B ACMD + 48B reserved).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
pub struct CommandTableHeader {
    pub cfis: [u8; 64],
    pub acmd: [u8; 16],
    pub rsv: [u8; 48],
}

const _: () = assert!(size_of::<CommandTableHeader>() == 128);

impl Default for CommandTableHeader {
    fn default() -> Self {
        Self {
            cfis: [0; 64],
            acmd: [0; 16],
            rsv: [0; 48],
        }
    }
}

impl CommandTableHeader {
    pub fn set_h2d_fis(&mut self, fis: &FisRegH2D) {
        self.cfis.fill(0);
        let bytes = unsafe {
            core::slice::from_raw_parts(
                (fis as *const FisRegH2D) as *const u8,
                size_of::<FisRegH2D>(),
            )
        };
        self.cfis[..size_of::<FisRegH2D>()].copy_from_slice(bytes);
    }
}

/// Scatter-gather PRDT builder that splits large segments (> 4 MiB) automatically.
pub struct PrdtBuilder<'a> {
    entries: &'a mut [PrdtEntry],
    count: usize,
}

impl<'a> PrdtBuilder<'a> {
    pub fn new(entries: &'a mut [PrdtEntry]) -> Self {
        Self { entries, count: 0 }
    }

    /// Append a physical memory segment, splitting into <= 4 MiB PRDT entries.
    pub fn push_segment(&mut self, mut phys_addr: u64, mut len: usize) -> Result<(), AhciError> {
        if (phys_addr & 1) != 0 {
            return Err(AhciError::UnalignedAddress);
        }
        if len == 0 {
            return Err(AhciError::InvalidBuffer);
        }
        while len > 0 {
            if self.count >= self.entries.len() {
                return Err(AhciError::PrdtOverflow);
            }
            let chunk = len.min(MAX_PRDT_BYTES);
            self.entries[self.count] = PrdtEntry::new(phys_addr, chunk, false)?;
            self.count += 1;
            phys_addr = phys_addr.wrapping_add(chunk as u64);
            len -= chunk;
        }
        Ok(())
    }

    /// Finalize the PRDT, optionally setting Interrupt-on-Completion on the last entry.
    pub fn finish(self, ioc_on_last: bool) -> usize {
        if ioc_on_last && self.count > 0 {
            self.entries[self.count - 1].dbc_i |= 1 << 31;
        }
        self.count
    }
}

/// Metadata tracked per NCQ command slot (0..31).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NcqSlotMeta {
    pub in_use: bool,
    pub is_write: bool,
    pub fua: bool,
    pub lba: u64,
    pub sector_count: u16,
    pub prdt_count: u16,
    pub bytes_expected: u32,
    pub request_id: u64,
}

/// 32-slot Native Command Queuing (NCQ) tag queue.
#[derive(Debug, Clone)]
pub struct NcqQueue {
    max_depth: u8,
    allocated_mask: u32,
    active_mask: u32,
    slots: [NcqSlotMeta; MAX_CMD_SLOTS],
    pub total_issued: u64,
    pub total_completed: u64,
    pub total_errors: u64,
}

impl NcqQueue {
    pub fn new(max_depth: u8) -> Self {
        let clamped = max_depth.clamp(1, MAX_CMD_SLOTS as u8);
        Self {
            max_depth: clamped,
            allocated_mask: 0,
            active_mask: 0,
            slots: [NcqSlotMeta::default(); MAX_CMD_SLOTS],
            total_issued: 0,
            total_completed: 0,
            total_errors: 0,
        }
    }

    #[inline]
    pub const fn max_depth(&self) -> u8 {
        self.max_depth
    }

    #[inline]
    pub const fn active_mask(&self) -> u32 {
        self.active_mask
    }

    #[inline]
    pub const fn allocated_mask(&self) -> u32 {
        self.allocated_mask
    }

    #[inline]
    pub const fn in_flight_count(&self) -> u32 {
        self.active_mask.count_ones()
    }

    #[inline]
    pub fn available_slots(&self) -> u32 {
        let valid_mask = if self.max_depth >= 32 {
            u32::MAX
        } else {
            (1u32 << self.max_depth) - 1
        };
        (valid_mask & !self.allocated_mask).count_ones()
    }

    /// Allocate a free NCQ tag in `0..max_depth`.
    pub fn allocate_tag(&mut self) -> Option<u8> {
        let tag = self.allocated_mask.trailing_ones() as u8;
        if tag >= self.max_depth {
            return None;
        }
        self.allocated_mask |= 1u32 << tag;
        Some(tag)
    }

    /// Mark an allocated tag as issued to hardware (`PxSACT` / `PxCI`).
    pub fn submit(&mut self, tag: u8, mut meta: NcqSlotMeta) -> Result<(), AhciError> {
        if tag >= self.max_depth {
            return Err(AhciError::InvalidTag);
        }
        let bit = 1u32 << tag;
        if (self.allocated_mask & bit) == 0 {
            return Err(AhciError::InvalidTag);
        }
        if (self.active_mask & bit) != 0 {
            return Err(AhciError::PortBusy);
        }
        meta.in_use = true;
        self.slots[tag as usize] = meta;
        self.active_mask |= bit;
        self.total_issued += 1;
        Ok(())
    }

    /// Reconcile hardware `PxSACT` register against `active_mask`.
    /// Returns the bitmask of tags that completed since the last check.
    pub fn complete_from_sact(&mut self, current_sact: u32) -> u32 {
        let completed = self.active_mask & !current_sact;
        if completed == 0 {
            return 0;
        }
        let mut remaining = completed;
        while remaining != 0 {
            let tag = remaining.trailing_zeros() as usize;
            self.slots[tag] = NcqSlotMeta::default();
            self.total_completed += 1;
            remaining &= remaining - 1;
        }
        self.active_mask &= !completed;
        self.allocated_mask &= !completed;
        completed
    }

    /// Complete a single tag explicitly.
    pub fn complete_tag(&mut self, tag: u8) -> Result<NcqSlotMeta, AhciError> {
        if tag >= self.max_depth {
            return Err(AhciError::InvalidTag);
        }
        let bit = 1u32 << tag;
        if (self.active_mask & bit) == 0 {
            return Err(AhciError::TagNotActive);
        }
        let meta = self.slots[tag as usize];
        self.slots[tag as usize] = NcqSlotMeta::default();
        self.active_mask &= !bit;
        self.allocated_mask &= !bit;
        self.total_completed += 1;
        Ok(meta)
    }

    /// Release an allocated tag that was never submitted.
    pub fn release_unsubmitted(&mut self, tag: u8) {
        if tag < self.max_depth {
            let bit = 1u32 << tag;
            if (self.active_mask & bit) == 0 {
                self.allocated_mask &= !bit;
            }
        }
    }

    /// Abort all active tags (e.g. after a task file error / COMRESET).
    pub fn abort_all(&mut self) -> u32 {
        let aborted = self.active_mask;
        let count = aborted.count_ones() as u64;
        self.total_errors += count;
        self.slots.fill(NcqSlotMeta::default());
        self.active_mask = 0;
        self.allocated_mask = 0;
        aborted
    }

    pub fn slot_meta(&self, tag: u8) -> Option<&NcqSlotMeta> {
        if tag < self.max_depth && (self.active_mask & (1u32 << tag)) != 0 {
            Some(&self.slots[tag as usize])
        } else {
            None
        }
    }
}

/// Parsed ATA `IDENTIFY DEVICE` (0xEC) information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentifyDevice {
    pub model: [u8; 40],
    pub model_len: usize,
    pub serial: [u8; 20],
    pub serial_len: usize,
    pub firmware: [u8; 8],
    pub firmware_len: usize,
    pub lba48_supported: bool,
    pub total_sectors: u64,
    pub logical_sector_size: u32,
    pub physical_sector_size: u32,
    pub ncq_supported: bool,
    pub ncq_queue_depth: u8,
    pub sata_gen1: bool,
    pub sata_gen2: bool,
    pub sata_gen3: bool,
    pub trim_supported: bool,
    pub write_cache_supported: bool,
    pub write_cache_enabled: bool,
    pub smart_supported: bool,
    pub smart_enabled: bool,
    pub rpm: u16,
}

impl IdentifyDevice {
    /// Parse a 256-word (512-byte) `IDENTIFY DEVICE` response.
    pub fn parse(words: &[u16; 256]) -> Self {
        let mut model = [b' '; 40];
        let model_len = decode_ata_string(&words[27..47], &mut model);

        let mut serial = [b' '; 20];
        let serial_len = decode_ata_string(&words[10..20], &mut serial);

        let mut firmware = [b' '; 8];
        let firmware_len = decode_ata_string(&words[23..27], &mut firmware);

        let lba48_supported = (words[83] & (1 << 10)) != 0;
        let total_sectors = if lba48_supported {
            (words[100] as u64)
                | ((words[101] as u64) << 16)
                | ((words[102] as u64) << 32)
                | ((words[103] as u64) << 48)
        } else {
            (words[60] as u64) | ((words[61] as u64) << 16)
        };

        let w106 = words[106];
        let logical_sector_size =
            if (w106 & (1 << 14)) != 0 && (w106 & (1 << 15)) == 0 && (w106 & (1 << 12)) != 0 {
                let words_per_sector = (words[117] as u32) | ((words[118] as u32) << 16);
                if words_per_sector >= 256 {
                    words_per_sector * 2
                } else {
                    SECTOR_SIZE as u32
                }
            } else {
                SECTOR_SIZE as u32
            };

        let physical_sector_size =
            if (w106 & (1 << 14)) != 0 && (w106 & (1 << 15)) == 0 && (w106 & (1 << 13)) != 0 {
                let shift = (w106 & 0x0F) as u32;
                logical_sector_size
                    .checked_shl(shift.min(16))
                    .unwrap_or(logical_sector_size)
            } else {
                logical_sector_size
            };

        let w76 = words[76];
        let ncq_supported = w76 != 0xFFFF && (w76 & (1 << 8)) != 0;
        let ncq_queue_depth = if ncq_supported {
            ((words[75] & 0x1F) as u8) + 1
        } else {
            1
        };

        Self {
            model,
            model_len,
            serial,
            serial_len,
            firmware,
            firmware_len,
            lba48_supported,
            total_sectors,
            logical_sector_size,
            physical_sector_size,
            ncq_supported,
            ncq_queue_depth,
            sata_gen1: w76 != 0xFFFF && (w76 & (1 << 1)) != 0,
            sata_gen2: w76 != 0xFFFF && (w76 & (1 << 2)) != 0,
            sata_gen3: w76 != 0xFFFF && (w76 & (1 << 3)) != 0,
            trim_supported: (words[169] & 1) != 0,
            write_cache_supported: (words[82] & (1 << 5)) != 0,
            write_cache_enabled: (words[85] & (1 << 5)) != 0,
            smart_supported: (words[82] & 1) != 0,
            smart_enabled: (words[85] & 1) != 0,
            rpm: words[217],
        }
    }

    /// Parse from a 512-byte little-endian buffer.
    pub fn parse_bytes(bytes: &[u8]) -> Result<Self, AhciError> {
        if bytes.len() < 512 {
            return Err(AhciError::InvalidBuffer);
        }
        let mut words = [0u16; 256];
        for (dst, chunk) in words.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
            *dst = u16::from_le_bytes(*chunk);
        }
        Ok(Self::parse(&words))
    }

    #[inline]
    pub const fn capacity_bytes(&self) -> u64 {
        self.total_sectors
            .saturating_mul(self.logical_sector_size as u64)
    }

    #[inline]
    pub const fn is_ssd(&self) -> bool {
        self.rpm == 1
    }

    pub fn model_str(&self) -> &str {
        core::str::from_utf8(&self.model[..self.model_len]).unwrap_or("")
    }

    pub fn serial_str(&self) -> &str {
        core::str::from_utf8(&self.serial[..self.serial_len]).unwrap_or("")
    }

    pub fn firmware_str(&self) -> &str {
        core::str::from_utf8(&self.firmware[..self.firmware_len]).unwrap_or("")
    }
}

/// Decode an ATA byte-swapped ASCII string from 16-bit words and trim trailing spaces/NULs.
fn decode_ata_string(words: &[u16], out: &mut [u8]) -> usize {
    for (dst_pair, &w) in out.as_chunks_mut::<2>().0.iter_mut().zip(words.iter()) {
        let hi = (w >> 8) as u8;
        let lo = (w & 0xFF) as u8;
        dst_pair[0] = if hi.is_ascii_graphic() || hi == b' ' {
            hi
        } else {
            b' '
        };
        dst_pair[1] = if lo.is_ascii_graphic() || lo == b' ' {
            lo
        } else {
            b' '
        };
    }
    let mut len = out.len();
    while len > 0 && (out[len - 1] == b' ' || out[len - 1] == 0) {
        len -= 1;
    }
    len
}

/// Encode an ASCII string into ATA byte-swapped 16-bit words (useful for tests/emulation).
pub fn encode_ata_string(src: &[u8], words: &mut [u16]) {
    for (idx, w) in words.iter_mut().enumerate() {
        let b0 = src.get(idx * 2).copied().unwrap_or(b' ');
        let b1 = src.get(idx * 2 + 1).copied().unwrap_or(b' ');
        *w = ((b0 as u16) << 8) | (b1 as u16);
    }
}

/// DMA memory layout for a single AHCI port (fits in one 4 KiB page for slot 0..7,
/// or two pages for full 32-slot command tables).
///
/// Page 0 (4 KiB):
/// - `0x000..0x400`: `CommandList` (1024 bytes, 1024-byte aligned)
/// - `0x400..0x500`: `ReceivedFis` (256 bytes, 256-byte aligned)
/// - `0x500..0x1000`: Command Tables 0..10 (256 bytes each: 128B header + 8 * 16B PRDT)
struct PortDma {
    frame_phys: u64,
    frame_virt: u64,
}

impl PortDma {
    const CL_OFFSET: usize = 0x000;
    const FB_OFFSET: usize = 0x400;
    const CT_BASE_OFFSET: usize = 0x500;
    const CT_STRIDE: usize = 256; // 128B CommandTableHeader + 8 * 16B PrdtEntry

    fn new(hhdm: u64) -> Result<Self, AhciError> {
        let frame_phys = alloc_frame().ok_or(AhciError::OutOfMemory)?;
        let frame_virt = frame_phys.wrapping_add(hhdm);
        unsafe {
            write_bytes(frame_virt as *mut u8, 0, PAGE_SIZE);
        }
        Ok(Self {
            frame_phys,
            frame_virt,
        })
    }

    #[inline]
    fn cl_phys(&self) -> u64 {
        self.frame_phys + Self::CL_OFFSET as u64
    }

    #[inline]
    fn fb_phys(&self) -> u64 {
        self.frame_phys + Self::FB_OFFSET as u64
    }

    #[inline]
    fn ct_phys(&self, slot: usize) -> u64 {
        self.frame_phys + (Self::CT_BASE_OFFSET + slot * Self::CT_STRIDE) as u64
    }

    #[inline]
    fn cmd_header_mut(&mut self, slot: usize) -> &mut CommandHeader {
        let ptr = (self.frame_virt + Self::CL_OFFSET as u64) as *mut CommandHeader;
        unsafe { &mut *ptr.add(slot) }
    }

    #[inline]
    fn cmd_table_header_mut(&mut self, slot: usize) -> &mut CommandTableHeader {
        let ptr = (self.frame_virt + (Self::CT_BASE_OFFSET + slot * Self::CT_STRIDE) as u64)
            as *mut CommandTableHeader;
        unsafe { &mut *ptr }
    }

    #[inline]
    fn prdt_slice_mut(&mut self, slot: usize) -> &mut [PrdtEntry] {
        let ptr = (self.frame_virt
            + (Self::CT_BASE_OFFSET + slot * Self::CT_STRIDE + size_of::<CommandTableHeader>())
                as u64) as *mut PrdtEntry;
        unsafe { core::slice::from_raw_parts_mut(ptr, DEFAULT_PRDT_ENTRIES) }
    }
}

impl Drop for PortDma {
    fn drop(&mut self) {
        dealloc_frame(self.frame_phys);
    }
}

/// Active AHCI port driver.
pub struct AhciPort {
    pub port_num: u8,
    port_mmio: u64,
    dma: PortDma,
    pub device_type: PortDeviceType,
    pub speed: SataSpeed,
    pub identify: Option<IdentifyDevice>,
    pub ncq: NcqQueue,
}

impl AhciPort {
    /// Probe and initialize an AHCI port if a SATA device is attached.
    pub fn init(abar_virt: u64, port_num: u8, hba_slots: u8, hhdm: u64) -> Result<Self, AhciError> {
        let port_mmio =
            abar_virt + (hba_regs::PORT_BASE + (port_num as usize) * hba_regs::PORT_STRIDE) as u64;
        let ssts = unsafe { read_mmio32(port_mmio, port_regs::SSTS) };
        let det = ssts & 0x0F;
        let ipm = (ssts >> 8) & 0x0F;
        if det != 3 || ipm != 1 {
            return Err(AhciError::NoDevicePresent);
        }

        let sig = unsafe { read_mmio32(port_mmio, port_regs::SIG) };
        let device_type = PortDeviceType::from_sig(sig);
        if device_type != PortDeviceType::Sata {
            return Err(AhciError::UnsupportedDevice);
        }

        let speed = SataSpeed::from_ssts(ssts);
        let dma = PortDma::new(hhdm)?;

        let mut port = Self {
            port_num,
            port_mmio,
            dma,
            device_type,
            speed,
            identify: None,
            ncq: NcqQueue::new(hba_slots.min(10)),
        };

        port.stop_cmd_engine()?;
        unsafe {
            write_mmio64(port.port_mmio, port_regs::CLB, port.dma.cl_phys());
            write_mmio64(port.port_mmio, port_regs::FB, port.dma.fb_phys());
            write_mmio32(port.port_mmio, port_regs::SERR, u32::MAX);
            write_mmio32(port.port_mmio, port_regs::IS, u32::MAX);
        }
        port.start_cmd_engine()?;

        let id = port.identify_device(hhdm)?;
        let effective_depth = if id.ncq_supported {
            id.ncq_queue_depth.min(hba_slots).min(10)
        } else {
            1
        };
        port.ncq = NcqQueue::new(effective_depth);
        port.identify = Some(id);

        Ok(port)
    }

    fn stop_cmd_engine(&self) -> Result<(), AhciError> {
        let mut cmd = unsafe { read_mmio32(self.port_mmio, port_regs::CMD) };
        cmd &= !(PXCMD_ST | PXCMD_FRE);
        unsafe { write_mmio32(self.port_mmio, port_regs::CMD, cmd) };

        for _ in 0..POLL_TIMEOUT_ITERS {
            let cur = unsafe { read_mmio32(self.port_mmio, port_regs::CMD) };
            if (cur & (PXCMD_FR | PXCMD_CR)) == 0 {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(AhciError::CommandTimeout)
    }

    fn start_cmd_engine(&self) -> Result<(), AhciError> {
        for _ in 0..POLL_TIMEOUT_ITERS {
            let cur = unsafe { read_mmio32(self.port_mmio, port_regs::CMD) };
            if (cur & PXCMD_CR) == 0 {
                break;
            }
            core::hint::spin_loop();
        }
        let mut cmd = unsafe { read_mmio32(self.port_mmio, port_regs::CMD) };
        cmd |= PXCMD_FRE | PXCMD_ST;
        unsafe { write_mmio32(self.port_mmio, port_regs::CMD, cmd) };
        Ok(())
    }

    fn wait_not_busy(&self) -> Result<(), AhciError> {
        for _ in 0..POLL_TIMEOUT_ITERS {
            let tfd = unsafe { read_mmio32(self.port_mmio, port_regs::TFD) };
            if (tfd & (PXTFD_BSY | PXTFD_DRQ)) == 0 {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(AhciError::PortBusy)
    }

    fn issue_polled_slot0(
        &mut self,
        fis: &FisRegH2D,
        buf_phys: u64,
        byte_len: usize,
        write: bool,
    ) -> Result<(), AhciError> {
        self.wait_not_busy()?;
        unsafe { write_mmio32(self.port_mmio, port_regs::IS, u32::MAX) };

        let ct_phys = self.dma.ct_phys(0);
        self.dma.cmd_table_header_mut(0).set_h2d_fis(fis);

        let prdtl = if byte_len > 0 {
            let mut builder = PrdtBuilder::new(self.dma.prdt_slice_mut(0));
            builder.push_segment(buf_phys, byte_len)?;
            builder.finish(true) as u16
        } else {
            0
        };

        let cfl_dwords = (size_of::<FisRegH2D>() / 4) as u8;
        self.dma
            .cmd_header_mut(0)
            .configure(cfl_dwords, write, false, false, prdtl, ct_phys);

        unsafe { write_mmio32(self.port_mmio, port_regs::CI, 1) };

        for _ in 0..POLL_TIMEOUT_ITERS {
            let ci = unsafe { read_mmio32(self.port_mmio, port_regs::CI) };
            let is = unsafe { read_mmio32(self.port_mmio, port_regs::IS) };
            if (is & PXIS_TFES) != 0 {
                let tfd = unsafe { read_mmio32(self.port_mmio, port_regs::TFD) };
                return Err(AhciError::TaskFileError(((tfd >> 8) & 0xFF) as u8));
            }
            if (ci & 1) == 0 {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(AhciError::CommandTimeout)
    }

    /// Issue `IDENTIFY DEVICE` (0xEC) and parse the 512-byte response.
    pub fn identify_device(&mut self, hhdm: u64) -> Result<IdentifyDevice, AhciError> {
        let dma_phys = alloc_frame().ok_or(AhciError::OutOfMemory)?;
        let dma_virt = dma_phys.wrapping_add(hhdm);
        unsafe { write_bytes(dma_virt as *mut u8, 0, 512) };

        let fis = FisRegH2D::new_command(ATA_CMD_IDENTIFY, 0, 0, false);
        let res = self.issue_polled_slot0(&fis, dma_phys, 512, false);
        if let Err(e) = res {
            dealloc_frame(dma_phys);
            return Err(e);
        }

        let buf = unsafe { core::slice::from_raw_parts(dma_virt as *const u8, 512) };
        let parsed = IdentifyDevice::parse_bytes(buf);
        dealloc_frame(dma_phys);
        parsed
    }

    /// Read sectors using `READ DMA EXT` (0x25).
    pub fn read_sectors(&mut self, lba: u64, buf: &mut [u8], hhdm: u64) -> Result<(), AhciError> {
        if buf.is_empty() || !buf.len().is_multiple_of(SECTOR_SIZE) {
            return Err(AhciError::InvalidBuffer);
        }
        let dma_phys = alloc_frame().ok_or(AhciError::OutOfMemory)?;
        let dma_virt = dma_phys.wrapping_add(hhdm);

        let max_sectors_per_page = PAGE_SIZE / SECTOR_SIZE;
        let mut remaining = buf.len() / SECTOR_SIZE;
        let mut cur_lba = lba;
        let mut offset = 0usize;

        while remaining > 0 {
            let chunk = remaining.min(max_sectors_per_page);
            let byte_len = chunk * SECTOR_SIZE;
            let fis = FisRegH2D::new_command(ATA_CMD_READ_DMA_EXT, cur_lba, chunk as u16, true);
            if let Err(e) = self.issue_polled_slot0(&fis, dma_phys, byte_len, false) {
                dealloc_frame(dma_phys);
                return Err(e);
            }
            unsafe {
                core::ptr::copy_nonoverlapping(
                    dma_virt as *const u8,
                    buf[offset..offset + byte_len].as_mut_ptr(),
                    byte_len,
                );
            }
            offset += byte_len;
            cur_lba += chunk as u64;
            remaining -= chunk;
        }

        dealloc_frame(dma_phys);
        Ok(())
    }

    /// Write sectors using `WRITE DMA EXT` (0x35).
    pub fn write_sectors(&mut self, lba: u64, buf: &[u8], hhdm: u64) -> Result<(), AhciError> {
        if buf.is_empty() || !buf.len().is_multiple_of(SECTOR_SIZE) {
            return Err(AhciError::InvalidBuffer);
        }
        let dma_phys = alloc_frame().ok_or(AhciError::OutOfMemory)?;
        let dma_virt = dma_phys.wrapping_add(hhdm);

        let max_sectors_per_page = PAGE_SIZE / SECTOR_SIZE;
        let mut remaining = buf.len() / SECTOR_SIZE;
        let mut cur_lba = lba;
        let mut offset = 0usize;

        while remaining > 0 {
            let chunk = remaining.min(max_sectors_per_page);
            let byte_len = chunk * SECTOR_SIZE;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    buf[offset..offset + byte_len].as_ptr(),
                    dma_virt as *mut u8,
                    byte_len,
                );
            }
            let fis = FisRegH2D::new_command(ATA_CMD_WRITE_DMA_EXT, cur_lba, chunk as u16, true);
            if let Err(e) = self.issue_polled_slot0(&fis, dma_phys, byte_len, true) {
                dealloc_frame(dma_phys);
                return Err(e);
            }
            offset += byte_len;
            cur_lba += chunk as u64;
            remaining -= chunk;
        }

        dealloc_frame(dma_phys);
        Ok(())
    }

    /// Submit an asynchronous NCQ read/write command (`READ/WRITE FPDMA QUEUED`).
    /// Returns the allocated NCQ tag (0..31).
    pub fn submit_ncq(
        &mut self,
        lba: u64,
        sector_count: u16,
        buf_phys: u64,
        write: bool,
        fua: bool,
        request_id: u64,
    ) -> Result<u8, AhciError> {
        let tag = self.ncq.allocate_tag().ok_or(AhciError::QueueFull)?;
        let byte_len = (sector_count as usize) * SECTOR_SIZE;
        let cmd = if write {
            ATA_CMD_WRITE_FPDMA_QUEUED
        } else {
            ATA_CMD_READ_FPDMA_QUEUED
        };
        let fis = FisRegH2D::new_ncq(cmd, tag, lba, sector_count, fua);

        let slot = tag as usize;
        let ct_phys = self.dma.ct_phys(slot);
        self.dma.cmd_table_header_mut(slot).set_h2d_fis(&fis);

        let mut builder = PrdtBuilder::new(self.dma.prdt_slice_mut(slot));
        if let Err(e) = builder.push_segment(buf_phys, byte_len) {
            self.ncq.release_unsubmitted(tag);
            return Err(e);
        }
        let prdtl = builder.finish(true) as u16;

        let cfl_dwords = (size_of::<FisRegH2D>() / 4) as u8;
        self.dma
            .cmd_header_mut(slot)
            .configure(cfl_dwords, write, true, false, prdtl, ct_phys);

        let meta = NcqSlotMeta {
            in_use: true,
            is_write: write,
            fua,
            lba,
            sector_count,
            prdt_count: prdtl,
            bytes_expected: byte_len as u32,
            request_id,
        };
        if let Err(e) = self.ncq.submit(tag, meta) {
            self.ncq.release_unsubmitted(tag);
            return Err(e);
        }

        // Per AHCI 1.3.1: PxSACT bit MUST be set before PxCI bit for NCQ commands.
        let bit = 1u32 << tag;
        unsafe {
            write_mmio32(self.port_mmio, port_regs::SACT, bit);
            write_mmio32(self.port_mmio, port_regs::CI, bit);
        }
        Ok(tag)
    }

    /// Poll `PxSACT` and return the bitmask of newly completed NCQ tags.
    pub fn poll_ncq(&mut self) -> Result<u32, AhciError> {
        let is = unsafe { read_mmio32(self.port_mmio, port_regs::IS) };
        if (is & PXIS_TFES) != 0 {
            let tfd = unsafe { read_mmio32(self.port_mmio, port_regs::TFD) };
            self.ncq.abort_all();
            return Err(AhciError::TaskFileError(((tfd >> 8) & 0xFF) as u8));
        }
        let sact = unsafe { read_mmio32(self.port_mmio, port_regs::SACT) };
        Ok(self.ncq.complete_from_sact(sact))
    }
}

/// AHCI Host Bus Adapter controller.
pub struct AhciController {
    pub abar_phys: u64,
    pub abar_virt: u64,
    pub caps: HbaCapabilities,
    pub ports_implemented: u32,
    pub ports: Vec<AhciPort>,
}

impl AhciController {
    /// Probe PCI devices for an AHCI controller (class 0x01, subclass 0x06, prog_if 0x01).
    pub fn probe_pci(devices: &[PciDevice]) -> Result<Self, AhciError> {
        let dev = devices
            .iter()
            .find(|d| d.class() == PciClass::Ahci)
            .ok_or(AhciError::NoController)?;

        let abar_phys = dev.bars[5] & !0xF;
        if abar_phys == 0 {
            return Err(AhciError::InvalidBar);
        }

        let hhdm = stats().hhdm_offset;
        let abar_virt = abar_phys.wrapping_add(hhdm);

        // Enable AHCI mode (GHC.AE)
        let ghc = unsafe { read_mmio32(abar_virt, hba_regs::GHC) };
        unsafe { write_mmio32(abar_virt, hba_regs::GHC, ghc | GHC_AE) };

        let cap = unsafe { read_mmio32(abar_virt, hba_regs::CAP) };
        let cap2 = unsafe { read_mmio32(abar_virt, hba_regs::CAP2) };
        let vs = unsafe { read_mmio32(abar_virt, hba_regs::VS) };
        let caps = HbaCapabilities::decode(cap, cap2, vs);

        if caps.bios_handoff {
            Self::perform_bios_handoff(abar_virt)?;
        }

        let pi = unsafe { read_mmio32(abar_virt, hba_regs::PI) };
        let mut ports = Vec::new();
        for port_num in 0..MAX_PORTS {
            if (pi & (1u32 << port_num)) != 0 {
                if let Ok(port) =
                    AhciPort::init(abar_virt, port_num as u8, caps.num_cmd_slots, hhdm)
                {
                    ports.push(port);
                }
            }
        }

        Ok(Self {
            abar_phys,
            abar_virt,
            caps,
            ports_implemented: pi,
            ports,
        })
    }

    fn perform_bios_handoff(abar_virt: u64) -> Result<(), AhciError> {
        let bohc = unsafe { read_mmio32(abar_virt, hba_regs::BOHC) };
        if (bohc & BOHC_BOS) == 0 {
            return Ok(());
        }
        unsafe { write_mmio32(abar_virt, hba_regs::BOHC, bohc | BOHC_OOS) };
        for _ in 0..POLL_TIMEOUT_ITERS {
            let cur = unsafe { read_mmio32(abar_virt, hba_regs::BOHC) };
            if (cur & BOHC_BOS) == 0 && (cur & BOHC_BB) == 0 {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(AhciError::BiosHandoffTimeout)
    }
}

/// `BlockDevice` wrapper around an initialized `AhciPort`.
pub struct AhciBlockDevice {
    port: SpinLock<AhciPort>,
    total_sectors: u64,
    sector_size: usize,
    hhdm: u64,
}

impl AhciBlockDevice {
    pub fn new(port: AhciPort, hhdm: u64) -> Self {
        let (total_sectors, sector_size) = match &port.identify {
            Some(id) => (id.total_sectors, id.logical_sector_size as usize),
            None => (0, SECTOR_SIZE),
        };
        Self {
            port: SpinLock::new(port),
            total_sectors,
            sector_size: sector_size.max(SECTOR_SIZE),
            hhdm,
        }
    }
}

impl BlockDevice for AhciBlockDevice {
    fn block_size(&self) -> usize {
        self.sector_size
    }

    fn block_count(&self) -> u64 {
        self.total_sectors
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        let sectors = (buf.len() / self.sector_size) as u64;
        if lba.saturating_add(sectors) > self.total_sectors {
            return Err(AhciError::OutOfRange.as_str());
        }
        let mut port = self.port.lock();
        port.read_sectors(lba, buf, self.hhdm).map_err(Into::into)
    }

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        let sectors = (buf.len() / self.sector_size) as u64;
        if lba.saturating_add(sectors) > self.total_sectors {
            return Err(AhciError::OutOfRange.as_str());
        }
        let mut port = self.port.lock();
        port.write_sectors(lba, buf, self.hhdm).map_err(Into::into)
    }
}

#[inline]
unsafe fn read_mmio32(base: u64, offset: usize) -> u32 {
    read_volatile((base + offset as u64) as *const u32)
}

#[inline]
unsafe fn write_mmio32(base: u64, offset: usize, val: u32) {
    write_volatile((base + offset as u64) as *mut u32, val);
}

#[inline]
unsafe fn write_mmio64(base: u64, offset: usize, val: u64) {
    write_mmio32(base, offset, val as u32);
    write_mmio32(base, offset + 4, (val >> 32) as u32);
}

/// Self-test for AHCI / SATA 3.0 HBA structures, PRDT builder, NCQ tag queue,
/// IDENTIFY DEVICE parser, and simulated command-slot wiring.
pub fn self_test() -> Result<(), &'static str> {
    // 1. Verify HBA capability decoding (AHCI 1.3.1, 6 ports, 32 slots, 64-bit, NCQ, Gen3, BOH)
    let cap = (5u32) | (31u32 << 8) | (3u32 << 20) | (1u32 << 30) | (1u32 << 31);
    let cap2 = CAP2_BOH;
    let vs = 0x0001_0301;
    let caps = HbaCapabilities::decode(cap, cap2, vs);
    if caps.version_major != 1
        || caps.version_minor != 0x0301
        || caps.num_ports != 6
        || caps.num_cmd_slots != 32
        || caps.interface_speed != 3
        || !caps.supports_64bit
        || !caps.supports_ncq
        || !caps.bios_handoff
    {
        return Err("ahci self_test: HbaCapabilities decode mismatch");
    }

    // 2. Verify FIS construction (LBA48 + NCQ FPDMA QUEUED)
    let lba = 0x0000_1234_5678_9ABCu64;
    let ncq_fis = FisRegH2D::new_ncq(ATA_CMD_READ_FPDMA_QUEUED, 17, lba, 64, true);
    if ncq_fis.fis_type != FIS_TYPE_REG_H2D
        || ncq_fis.pmport_c != 0x80
        || ncq_fis.command != ATA_CMD_READ_FPDMA_QUEUED
        || ncq_fis.featurel != 64
        || ncq_fis.featureh != 0
        || ncq_fis.countl != (17 << 3)
        || ncq_fis.device != ((1 << 6) | (1 << 7))
        || ncq_fis.lba0 != 0xBC
        || ncq_fis.lba3 != 0x56
        || ncq_fis.lba5 != 0x12
    {
        return Err("ahci self_test: FisRegH2D NCQ encoding mismatch");
    }

    // 3. Verify PRDT scatter-gather builder (including >4 MiB auto-split & IOC)
    let mut prdt = [PrdtEntry::default(); 8];
    let mut builder = PrdtBuilder::new(&mut prdt);
    builder.push_segment(0x1000_0000, 6 * 1024 * 1024)?; // splits into 4 MiB + 2 MiB
    builder.push_segment(0x2000_0000, 4096)?;
    let prdt_count = builder.finish(true);
    if prdt_count != 3
        || prdt[0].byte_count() != MAX_PRDT_BYTES
        || prdt[0].has_ioc()
        || prdt[1].byte_count() != 2 * 1024 * 1024
        || prdt[2].byte_count() != 4096
        || !prdt[2].has_ioc()
    {
        return Err("ahci self_test: PrdtBuilder scatter-gather mismatch");
    }

    // Unaligned address must be rejected
    let mut bad_prdt = [PrdtEntry::default(); 2];
    let mut bad_builder = PrdtBuilder::new(&mut bad_prdt);
    if bad_builder.push_segment(0x1001, 512).is_ok() {
        return Err("ahci self_test: unaligned PRDT address accepted");
    }

    // 4. Verify 32-slot NCQ tag queue allocation, out-of-order completion, and full-queue handling
    let mut ncq = NcqQueue::new(32);
    let mut tags = [0u8; 32];
    for (idx, slot_tag) in tags.iter_mut().enumerate() {
        let tag = ncq
            .allocate_tag()
            .ok_or("ahci self_test: NCQ tag alloc failed")?;
        *slot_tag = tag;
        ncq.submit(
            tag,
            NcqSlotMeta {
                in_use: true,
                is_write: (idx & 1) != 0,
                fua: false,
                lba: (idx as u64) * 8,
                sector_count: 8,
                prdt_count: 1,
                bytes_expected: 4096,
                request_id: idx as u64,
            },
        )?;
    }
    if ncq.allocate_tag().is_some() || ncq.in_flight_count() != 32 || ncq.available_slots() != 0 {
        return Err("ahci self_test: NCQ full-queue state mismatch");
    }

    // Simulate hardware completing tags 3, 15, and 31 out-of-order via PxSACT
    let sact_after = !((1u32 << 3) | (1u32 << 15) | (1u32 << 31));
    let completed = ncq.complete_from_sact(sact_after);
    if completed != ((1u32 << 3) | (1u32 << 15) | (1u32 << 31))
        || ncq.in_flight_count() != 29
        || ncq.available_slots() != 3
        || ncq.total_completed != 3
    {
        return Err("ahci self_test: NCQ PxSACT completion mismatch");
    }
    let reused = ncq
        .allocate_tag()
        .ok_or("ahci self_test: NCQ tag reuse failed")?;
    if reused != 3 {
        return Err("ahci self_test: expected lowest freed tag 3");
    }
    ncq.release_unsubmitted(reused);
    ncq.abort_all();
    if ncq.in_flight_count() != 0 || ncq.total_errors != 29 {
        return Err("ahci self_test: NCQ abort_all mismatch");
    }

    // 5. Verify IDENTIFY DEVICE parser (heap-allocated to keep stack frame < 2 KiB)
    let mut id_words = vec![0u16; 256];
    encode_ata_string(b"NEXUS-NVME-SATA-SSD", &mut id_words[27..47]);
    encode_ata_string(b"NXS00001234", &mut id_words[10..20]);
    encode_ata_string(b"1.0A", &mut id_words[23..27]);
    id_words[75] = 31; // queue depth 32
    id_words[76] = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 8); // Gen1/2/3 + NCQ
    id_words[82] = (1 << 0) | (1 << 5); // SMART + write cache supported
    id_words[83] = 1 << 10; // LBA48 supported
    id_words[85] = (1 << 0) | (1 << 5); // SMART + write cache enabled
    id_words[100] = 0x0000;
    id_words[101] = 0x0400; // 0x0400_0000 = 67,108,864 sectors (32 GiB @ 512B)
    id_words[106] = (1 << 14) | (1 << 13) | 3; // 4096B physical (512 << 3)
    id_words[169] = 1; // TRIM supported
    id_words[217] = 1; // SSD (non-rotating)

    let mut id_bytes = vec![0u8; 512];
    for (dst, &w) in id_bytes
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(id_words.iter())
    {
        *dst = w.to_le_bytes();
    }
    let parsed = IdentifyDevice::parse_bytes(&id_bytes)?;
    if parsed.model_str() != "NEXUS-NVME-SATA-SSD"
        || parsed.serial_str() != "NXS00001234"
        || parsed.firmware_str() != "1.0A"
        || !parsed.lba48_supported
        || parsed.total_sectors != 0x0400_0000
        || parsed.logical_sector_size != 512
        || parsed.physical_sector_size != 4096
        || !parsed.ncq_supported
        || parsed.ncq_queue_depth != 32
        || !parsed.sata_gen3
        || !parsed.trim_supported
        || !parsed.is_ssd()
    {
        return Err("ahci self_test: IdentifyDevice parse mismatch");
    }

    // 6. Verify CommandList + CommandTableHeader + PrdtEntry wiring on heap
    let mut cl = Box::<CommandList>::default();
    let mut ct = Box::<CommandTableHeader>::default();
    let ct_phys = (&*ct as *const CommandTableHeader) as u64;
    ct.set_h2d_fis(&ncq_fis);
    cl.headers[17].configure(
        (size_of::<FisRegH2D>() / 4) as u8,
        false,
        true,
        false,
        prdt_count as u16,
        ct_phys,
    );
    if cl.headers[17].prdtl != 3
        || cl.headers[17].ctba != ct_phys as u32
        || ct.cfis[0] != FIS_TYPE_REG_H2D
        || ct.cfis[2] != ATA_CMD_READ_FPDMA_QUEUED
    {
        return Err("ahci self_test: CommandList/CommandTable wiring mismatch");
    }

    Ok(())
}
