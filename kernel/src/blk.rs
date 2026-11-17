//! Block device abstraction, GPT partition table parser, and VirtIO-Blk driver skeleton.
#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;
use core::ptr::{read_volatile, write_volatile};

use crate::mm::{alloc_frame, stats, SpinLock};
use crate::pci::{PciClass, PciDevice};

/// Standard 512-byte sector size.
pub const SECTOR_SIZE: usize = 512;

/// Clean-room block device abstraction.
pub trait BlockDevice: Send + Sync {
    fn block_size(&self) -> usize;
    fn block_count(&self) -> u64;
    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str>;
    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str>;
}

/// GPT header signature ("EFI PART").
pub const GPT_SIGNATURE: [u8; 8] = *b"EFI PART";

/// Parsed GUID Partition Table (GPT) header (LBA 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GptHeader {
    pub signature: [u8; 8],
    pub revision: u32,
    pub header_size: u32,
    pub header_crc32: u32,
    pub my_lba: u64,
    pub alternate_lba: u64,
    pub first_usable_lba: u64,
    pub last_usable_lba: u64,
    pub disk_guid: [u8; 16],
    pub partition_entry_lba: u64,
    pub num_partition_entries: u32,
    pub size_of_partition_entry: u32,
    pub partition_entry_array_crc32: u32,
}

impl GptHeader {
    /// Parse a GPT header from LBA 1 bytes (at least 92 bytes).
    pub fn parse(lba1: &[u8]) -> Option<Self> {
        if lba1.len() < 92 {
            return None;
        }
        let mut signature = [0u8; 8];
        signature.copy_from_slice(&lba1[0..8]);
        if signature != GPT_SIGNATURE {
            return None;
        }

        let revision = read_le_u32(lba1, 8);
        let header_size = read_le_u32(lba1, 12);
        let header_crc32 = read_le_u32(lba1, 16);
        // bytes 20..24 reserved
        let my_lba = read_le_u64(lba1, 24);
        let alternate_lba = read_le_u64(lba1, 32);
        let first_usable_lba = read_le_u64(lba1, 40);
        let last_usable_lba = read_le_u64(lba1, 48);

        let mut disk_guid = [0u8; 16];
        disk_guid.copy_from_slice(&lba1[56..72]);

        let partition_entry_lba = read_le_u64(lba1, 72);
        let num_partition_entries = read_le_u32(lba1, 80);
        let size_of_partition_entry = read_le_u32(lba1, 84);
        let partition_entry_array_crc32 = read_le_u32(lba1, 88);

        if header_size < 92 || size_of_partition_entry < 128 {
            return None;
        }

        Some(Self {
            signature,
            revision,
            header_size,
            header_crc32,
            my_lba,
            alternate_lba,
            first_usable_lba,
            last_usable_lba,
            disk_guid,
            partition_entry_lba,
            num_partition_entries,
            size_of_partition_entry,
            partition_entry_array_crc32,
        })
    }
}

/// Parsed GPT partition entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GptPartition {
    pub type_guid: [u8; 16],
    pub unique_guid: [u8; 16],
    pub starting_lba: u64,
    pub ending_lba: u64,
    pub attributes: u64,
    pub name: String,
}

impl GptPartition {
    /// Number of sectors spanned by this partition (inclusive range).
    pub fn sector_count(&self) -> u64 {
        if self.ending_lba >= self.starting_lba {
            self.ending_lba - self.starting_lba + 1
        } else {
            0
        }
    }
}

/// Parse GPT header (LBA 1) and partition entry array into active `GptPartition`s.
pub fn parse_gpt(lba1: &[u8], entries: &[u8]) -> Vec<GptPartition> {
    let mut partitions = Vec::new();
    let Some(header) = GptHeader::parse(lba1) else {
        return partitions;
    };

    let entry_size = header.size_of_partition_entry as usize;
    let count = header.num_partition_entries as usize;

    for i in 0..count {
        let offset = match i.checked_mul(entry_size) {
            Some(off) => off,
            None => break,
        };
        if offset + 128 > entries.len() {
            break;
        }
        let entry = &entries[offset..offset + 128];

        let mut type_guid = [0u8; 16];
        type_guid.copy_from_slice(&entry[0..16]);
        if type_guid == [0u8; 16] {
            continue;
        }

        let mut unique_guid = [0u8; 16];
        unique_guid.copy_from_slice(&entry[16..32]);

        let starting_lba = read_le_u64(entry, 32);
        let ending_lba = read_le_u64(entry, 40);
        let attributes = read_le_u64(entry, 48);

        let mut name = String::new();
        for j in 0..36 {
            let ch = read_le_u16(entry, 56 + j * 2);
            if ch == 0 {
                break;
            }
            if let Some(c) = char::from_u32(ch as u32) {
                name.push(c);
            }
        }

        partitions.push(GptPartition {
            type_guid,
            unique_guid,
            starting_lba,
            ending_lba,
            attributes,
            name,
        });
    }

    partitions
}

#[inline]
fn read_le_u16(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([buf[offset], buf[offset + 1]])
}

#[inline]
fn read_le_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}

#[inline]
fn read_le_u64(buf: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
        buf[offset + 4],
        buf[offset + 5],
        buf[offset + 6],
        buf[offset + 7],
    ])
}

// VirtIO PCI constants
const VIRTIO_VENDOR_ID: u16 = 0x1AF4;
const VIRTIO_BLK_LEGACY_DEVICE_ID: u16 = 0x1001;
const VIRTIO_BLK_MODERN_DEVICE_ID: u16 = 0x1042;

// VirtIO legacy/MMIO status bits
const VIRTIO_STATUS_ACKNOWLEDGE: u8 = 1;
const VIRTIO_STATUS_DRIVER: u8 = 2;
const VIRTIO_STATUS_DRIVER_OK: u8 = 4;
const VIRTIO_STATUS_FEATURES_OK: u8 = 8;

// Legacy VirtIO PCI BAR0 register offsets
const VIRTIO_PCI_HOST_FEATURES: usize = 0x00;
const VIRTIO_PCI_GUEST_FEATURES: usize = 0x04;
const VIRTIO_PCI_QUEUE_PFN: usize = 0x08;
const VIRTIO_PCI_QUEUE_NUM: usize = 0x0C;
const VIRTIO_PCI_QUEUE_SEL: usize = 0x0E;
const VIRTIO_PCI_QUEUE_NOTIFY: usize = 0x10;
const VIRTIO_PCI_STATUS: usize = 0x12;
const VIRTIO_PCI_BLK_CAPACITY_LO: usize = 0x14;
const VIRTIO_PCI_BLK_CAPACITY_HI: usize = 0x18;

struct VirtioBlkState {
    queue_phys: u64,
    initialized: bool,
}

/// VirtIO-Blk PCI device skeleton.
pub struct VirtioBlkDevice {
    bus: u8,
    dev: u8,
    func: u8,
    device_id: u16,
    bar0_phys: u64,
    mmio_base: u64,
    capacity_sectors: u64,
    state: SpinLock<VirtioBlkState>,
}

impl VirtioBlkDevice {
    /// Probe a `PciDevice` for a VirtIO block controller (`PciClass::Virtio`).
    pub fn probe(dev: &PciDevice) -> Option<Self> {
        if dev.class() != PciClass::Virtio || dev.vendor_id != VIRTIO_VENDOR_ID {
            return None;
        }
        if dev.device_id != VIRTIO_BLK_LEGACY_DEVICE_ID
            && dev.device_id != VIRTIO_BLK_MODERN_DEVICE_ID
        {
            return None;
        }

        let bar0_phys = dev.bars[0];
        if bar0_phys == 0 {
            return None;
        }

        let hhdm = stats().hhdm_offset;
        let mmio_base = bar0_phys.wrapping_add(hhdm);

        // Allocate a physical frame for the virtqueue ring skeleton.
        let queue_phys = alloc_frame().unwrap_or(0);

        let mut capacity_sectors = 0u64;
        if dev.device_id == VIRTIO_BLK_LEGACY_DEVICE_ID && bar0_phys >= 0x10000 {
            // If BAR0 is MMIO-mapped, perform basic VirtIO status handshake & read capacity.
            unsafe {
                let status_ptr = (mmio_base + VIRTIO_PCI_STATUS as u64) as *mut u8;
                write_volatile(status_ptr, 0); // Reset
                write_volatile(status_ptr, VIRTIO_STATUS_ACKNOWLEDGE);
                write_volatile(status_ptr, VIRTIO_STATUS_ACKNOWLEDGE | VIRTIO_STATUS_DRIVER);

                let cap_lo =
                    read_volatile((mmio_base + VIRTIO_PCI_BLK_CAPACITY_LO as u64) as *const u32)
                        as u64;
                let cap_hi =
                    read_volatile((mmio_base + VIRTIO_PCI_BLK_CAPACITY_HI as u64) as *const u32)
                        as u64;
                capacity_sectors = (cap_hi << 32) | cap_lo;

                if queue_phys != 0 {
                    write_volatile((mmio_base + VIRTIO_PCI_QUEUE_SEL as u64) as *mut u16, 0);
                    write_volatile(
                        (mmio_base + VIRTIO_PCI_QUEUE_PFN as u64) as *mut u32,
                        (queue_phys >> 12) as u32,
                    );
                }

                write_volatile(
                    status_ptr,
                    VIRTIO_STATUS_ACKNOWLEDGE
                        | VIRTIO_STATUS_DRIVER
                        | VIRTIO_STATUS_FEATURES_OK
                        | VIRTIO_STATUS_DRIVER_OK,
                );
            }
        }

        Some(Self {
            bus: dev.bus,
            dev: dev.dev,
            func: dev.func,
            device_id: dev.device_id,
            bar0_phys,
            mmio_base,
            capacity_sectors,
            state: SpinLock::new(VirtioBlkState {
                queue_phys,
                initialized: queue_phys != 0,
            }),
        })
    }

    pub fn pci_location(&self) -> (u8, u8, u8) {
        (self.bus, self.dev, self.func)
    }

    pub fn bar0_phys(&self) -> u64 {
        self.bar0_phys
    }

    pub fn mmio_base(&self) -> u64 {
        self.mmio_base
    }
}

impl BlockDevice for VirtioBlkDevice {
    fn block_size(&self) -> usize {
        SECTOR_SIZE
    }

    fn block_count(&self) -> u64 {
        self.capacity_sectors
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        if buf.is_empty() || (buf.len() % SECTOR_SIZE) != 0 {
            return Err("virtio-blk: buffer must be a non-empty multiple of 512 bytes");
        }
        let sectors = (buf.len() / SECTOR_SIZE) as u64;
        if self.capacity_sectors != 0 && lba.saturating_add(sectors) > self.capacity_sectors {
            return Err("virtio-blk: LBA out of range");
        }
        let state = self.state.lock();
        if !state.initialized {
            return Err("virtio-blk: virtqueue not initialized");
        }
        Err("virtio-blk: I/O submission not yet implemented in skeleton")
    }

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        if buf.is_empty() || (buf.len() % SECTOR_SIZE) != 0 {
            return Err("virtio-blk: buffer must be a non-empty multiple of 512 bytes");
        }
        let sectors = (buf.len() / SECTOR_SIZE) as u64;
        if self.capacity_sectors != 0 && lba.saturating_add(sectors) > self.capacity_sectors {
            return Err("virtio-blk: LBA out of range");
        }
        let state = self.state.lock();
        if !state.initialized {
            return Err("virtio-blk: virtqueue not initialized");
        }
        Err("virtio-blk: I/O submission not yet implemented in skeleton")
    }
}
