//! Clean-room NVMe PCIe controller driver.
#![allow(dead_code)]

use core::mem::size_of;
use core::ptr::{read_volatile, write_bytes, write_volatile};

use crate::blk::{BlockDevice, SECTOR_SIZE};
use crate::mm::{alloc_frame, dealloc_frame, stats, SpinLock, PAGE_SIZE};
use crate::pci::{PciClass, PciDevice};

/// NVMe BAR0 MMIO register offsets.
pub mod regs {
    pub const CAP: usize = 0x0000;
    pub const VS: usize = 0x0008;
    pub const INTMS: usize = 0x000C;
    pub const INTMC: usize = 0x0010;
    pub const CC: usize = 0x0014;
    pub const CSTS: usize = 0x001C;
    pub const AQA: usize = 0x0024;
    pub const ASQ: usize = 0x0028;
    pub const ACQ: usize = 0x0030;
    pub const DOORBELL_BASE: usize = 0x1000;
}

// Controller Configuration (CC) bits
const CC_EN: u32 = 1 << 0;
const CC_IOSQES_64B: u32 = 6 << 16;
const CC_IOCQES_16B: u32 = 4 << 20;

// Controller Status (CSTS) bits
const CSTS_RDY: u32 = 1 << 0;
const CSTS_CFS: u32 = 1 << 1;

// NVMe Admin & NVM opcodes
const ADMIN_CREATE_IO_CQ: u8 = 0x05;
const ADMIN_CREATE_IO_SQ: u8 = 0x01;
const ADMIN_IDENTIFY: u8 = 0x06;
const NVM_CMD_WRITE: u8 = 0x01;
const NVM_CMD_READ: u8 = 0x02;

const ADMIN_QUEUE_DEPTH: u16 = 64;
const IO_QUEUE_DEPTH: u16 = 64;
const POLL_TIMEOUT_ITERS: usize = 1_000_000;

/// 64-byte NVMe Submission Queue Entry.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NvmeSqEntry {
    pub cdw0: u32,
    pub nsid: u32,
    pub rsvd2: u64,
    pub mptr: u64,
    pub prp1: u64,
    pub prp2: u64,
    pub cdw10: u32,
    pub cdw11: u32,
    pub cdw12: u32,
    pub cdw13: u32,
    pub cdw14: u32,
    pub cdw15: u32,
}

const _: () = assert!(size_of::<NvmeSqEntry>() == 64);

/// 16-byte NVMe Completion Queue Entry.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NvmeCqEntry {
    pub dw0: u32,
    pub dw1: u32,
    pub sq_head: u16,
    pub sq_id: u16,
    pub cid: u16,
    pub status: u16,
}

const _: () = assert!(size_of::<NvmeCqEntry>() == 16);

impl NvmeCqEntry {
    #[inline]
    pub fn phase(&self) -> bool {
        (self.status & 0x1) != 0
    }

    #[inline]
    pub fn status_code(&self) -> u16 {
        (self.status >> 1) & 0x7FFF
    }
}

struct QueuePair {
    sq_phys: u64,
    cq_phys: u64,
    sq_virt: u64,
    cq_virt: u64,
    depth: u16,
    sq_tail: u16,
    cq_head: u16,
    cq_phase: bool,
    next_cid: u16,
}

impl QueuePair {
    fn new(depth: u16, hhdm: u64) -> Option<Self> {
        let sq_phys = alloc_frame()?;
        let cq_phys = match alloc_frame() {
            Some(p) => p,
            None => {
                dealloc_frame(sq_phys);
                return None;
            }
        };
        let sq_virt = sq_phys.wrapping_add(hhdm);
        let cq_virt = cq_phys.wrapping_add(hhdm);
        unsafe {
            write_bytes(sq_virt as *mut u8, 0, PAGE_SIZE);
            write_bytes(cq_virt as *mut u8, 0, PAGE_SIZE);
        }
        Some(Self {
            sq_phys,
            cq_phys,
            sq_virt,
            cq_virt,
            depth,
            sq_tail: 0,
            cq_head: 0,
            cq_phase: true,
            next_cid: 1,
        })
    }
}

struct NvmeState {
    admin: QueuePair,
    io: Option<QueuePair>,
    ready: bool,
}

/// Clean-room NVMe PCIe controller driver.
pub struct NvmeController {
    bus: u8,
    dev: u8,
    func: u8,
    bar0_phys: u64,
    mmio_base: u64,
    cap: u64,
    vs: u32,
    dstrd: u32,
    block_size: usize,
    block_count: u64,
    nsid: u32,
    state: SpinLock<NvmeState>,
}

impl NvmeController {
    /// Probe a `PciDevice` (`PciClass::Nvme`), map BAR0, read `CAP`/`VS`, and prepare Admin SQ/CQ.
    pub fn probe(dev: &PciDevice) -> Option<Self> {
        if dev.class() != PciClass::Nvme {
            return None;
        }
        let bar0_phys = dev.bars[0];
        if bar0_phys == 0 {
            return None;
        }

        let hhdm = stats().hhdm_offset;
        let mmio_base = bar0_phys.wrapping_add(hhdm);

        let cap = unsafe { read_mmio64(mmio_base, regs::CAP) };
        let vs = unsafe { read_mmio32(mmio_base, regs::VS) };
        let dstrd = ((cap >> 32) & 0xF) as u32;

        let admin = QueuePair::new(ADMIN_QUEUE_DEPTH, hhdm)?;

        // Disable controller before configuring Admin Queue registers.
        let mut cc = unsafe { read_mmio32(mmio_base, regs::CC) };
        if (cc & CC_EN) != 0 {
            cc &= !CC_EN;
            unsafe { write_mmio32(mmio_base, regs::CC, cc) };
            let _ = wait_csts_ready(mmio_base, false);
        }

        // Program AQA, ASQ, ACQ.
        let aqa = (((ADMIN_QUEUE_DEPTH as u32) - 1) << 16) | ((ADMIN_QUEUE_DEPTH as u32) - 1);
        unsafe {
            write_mmio32(mmio_base, regs::AQA, aqa);
            write_mmio64(mmio_base, regs::ASQ, admin.sq_phys);
            write_mmio64(mmio_base, regs::ACQ, admin.cq_phys);
        }

        // Enable controller with 4 KiB pages, NVM command set, 64B SQE, 16B CQE.
        let new_cc = CC_EN | CC_IOSQES_64B | CC_IOCQES_16B;
        unsafe { write_mmio32(mmio_base, regs::CC, new_cc) };
        let ready = wait_csts_ready(mmio_base, true);

        let mut ctrl = Self {
            bus: dev.bus,
            dev: dev.dev,
            func: dev.func,
            bar0_phys,
            mmio_base,
            cap,
            vs,
            dstrd,
            block_size: SECTOR_SIZE,
            block_count: 0,
            nsid: 1,
            state: SpinLock::new(NvmeState {
                admin,
                io: None,
                ready,
            }),
        };

        if ready {
            ctrl.init_io_queues(hhdm);
        }

        Some(ctrl)
    }

    fn init_io_queues(&mut self, hhdm: u64) {
        let Some(io_qp) = QueuePair::new(IO_QUEUE_DEPTH, hhdm) else {
            return;
        };

        let mut state = self.state.lock();
        // Create I/O Completion Queue (qid = 1)
        let mut cq_cmd = NvmeSqEntry::default();
        cq_cmd.cdw0 = ADMIN_CREATE_IO_CQ as u32;
        cq_cmd.prp1 = io_qp.cq_phys;
        cq_cmd.cdw10 = (((IO_QUEUE_DEPTH as u32) - 1) << 16) | 1;
        cq_cmd.cdw11 = 1; // Physically contiguous (PC = 1)

        if submit_and_poll(self.mmio_base, self.dstrd, 0, &mut state.admin, cq_cmd).is_err() {
            dealloc_frame(io_qp.sq_phys);
            dealloc_frame(io_qp.cq_phys);
            return;
        }

        // Create I/O Submission Queue (qid = 1, cqid = 1)
        let mut sq_cmd = NvmeSqEntry::default();
        sq_cmd.cdw0 = ADMIN_CREATE_IO_SQ as u32;
        sq_cmd.prp1 = io_qp.sq_phys;
        sq_cmd.cdw10 = (((IO_QUEUE_DEPTH as u32) - 1) << 16) | 1;
        sq_cmd.cdw11 = (1 << 16) | 1; // CQID = 1, PC = 1

        if submit_and_poll(self.mmio_base, self.dstrd, 0, &mut state.admin, sq_cmd).is_err() {
            dealloc_frame(io_qp.sq_phys);
            dealloc_frame(io_qp.cq_phys);
            return;
        }

        // Identify Namespace 1 to discover block count if available.
        if let Some(id_phys) = alloc_frame() {
            let id_virt = id_phys.wrapping_add(hhdm);
            unsafe { write_bytes(id_virt as *mut u8, 0, PAGE_SIZE) };
            let mut id_cmd = NvmeSqEntry::default();
            id_cmd.cdw0 = ADMIN_IDENTIFY as u32;
            id_cmd.nsid = self.nsid;
            id_cmd.prp1 = id_phys;
            id_cmd.cdw10 = 0; // CNS = 0 (Identify Namespace)

            if submit_and_poll(self.mmio_base, self.dstrd, 0, &mut state.admin, id_cmd).is_ok() {
                let nsze = unsafe { read_volatile(id_virt as *const u64) };
                self.block_count = nsze;
            }
            dealloc_frame(id_phys);
        }

        state.io = Some(io_qp);
    }

    pub fn cap(&self) -> u64 {
        self.cap
    }

    pub fn version(&self) -> u32 {
        self.vs
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

impl BlockDevice for NvmeController {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn block_count(&self) -> u64 {
        self.block_count
    }

    fn read_blocks(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        if buf.is_empty() || (buf.len() % self.block_size) != 0 {
            return Err("nvme: buffer must be a non-empty multiple of block_size");
        }
        let blocks = buf.len() / self.block_size;
        if self.block_count != 0 && lba.saturating_add(blocks as u64) > self.block_count {
            return Err("nvme: LBA out of range");
        }

        let hhdm = stats().hhdm_offset;
        let mut state = self.state.lock();
        if !state.ready {
            return Err("nvme: controller not ready");
        }
        let io = state.io.as_mut().ok_or("nvme: I/O queue not initialized")?;

        let max_per_page = PAGE_SIZE / self.block_size;
        let mut offset = 0usize;
        let mut cur_lba = lba;
        let mut remaining = blocks;

        let dma_phys = alloc_frame().ok_or("nvme: out of DMA frames")?;
        let dma_virt = dma_phys.wrapping_add(hhdm);

        while remaining > 0 {
            let chunk = remaining.min(max_per_page);
            let byte_len = chunk * self.block_size;

            let mut cmd = NvmeSqEntry::default();
            cmd.cdw0 = NVM_CMD_READ as u32;
            cmd.nsid = self.nsid;
            cmd.prp1 = dma_phys;
            cmd.cdw10 = cur_lba as u32;
            cmd.cdw11 = (cur_lba >> 32) as u32;
            cmd.cdw12 = (chunk as u32) - 1;

            if let Err(e) = submit_and_poll(self.mmio_base, self.dstrd, 1, io, cmd) {
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

    fn write_blocks(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        if buf.is_empty() || (buf.len() % self.block_size) != 0 {
            return Err("nvme: buffer must be a non-empty multiple of block_size");
        }
        let blocks = buf.len() / self.block_size;
        if self.block_count != 0 && lba.saturating_add(blocks as u64) > self.block_count {
            return Err("nvme: LBA out of range");
        }

        let hhdm = stats().hhdm_offset;
        let mut state = self.state.lock();
        if !state.ready {
            return Err("nvme: controller not ready");
        }
        let io = state.io.as_mut().ok_or("nvme: I/O queue not initialized")?;

        let max_per_page = PAGE_SIZE / self.block_size;
        let mut offset = 0usize;
        let mut cur_lba = lba;
        let mut remaining = blocks;

        let dma_phys = alloc_frame().ok_or("nvme: out of DMA frames")?;
        let dma_virt = dma_phys.wrapping_add(hhdm);

        while remaining > 0 {
            let chunk = remaining.min(max_per_page);
            let byte_len = chunk * self.block_size;

            unsafe {
                core::ptr::copy_nonoverlapping(
                    buf[offset..offset + byte_len].as_ptr(),
                    dma_virt as *mut u8,
                    byte_len,
                );
            }

            let mut cmd = NvmeSqEntry::default();
            cmd.cdw0 = NVM_CMD_WRITE as u32;
            cmd.nsid = self.nsid;
            cmd.prp1 = dma_phys;
            cmd.cdw10 = cur_lba as u32;
            cmd.cdw11 = (cur_lba >> 32) as u32;
            cmd.cdw12 = (chunk as u32) - 1;

            if let Err(e) = submit_and_poll(self.mmio_base, self.dstrd, 1, io, cmd) {
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
}

fn submit_and_poll(
    mmio_base: u64,
    dstrd: u32,
    qid: u16,
    qp: &mut QueuePair,
    mut cmd: NvmeSqEntry,
) -> Result<NvmeCqEntry, &'static str> {
    let cid = qp.next_cid;
    qp.next_cid = qp.next_cid.wrapping_add(1);
    if qp.next_cid == 0 {
        qp.next_cid = 1;
    }
    cmd.cdw0 = (cmd.cdw0 & 0x0000_FFFF) | ((cid as u32) << 16);

    let sq_slot = (qp.sq_virt as *mut NvmeSqEntry).wrapping_add(qp.sq_tail as usize);
    unsafe { write_volatile(sq_slot, cmd) };

    qp.sq_tail = (qp.sq_tail + 1) % qp.depth;
    let sq_db_offset = regs::DOORBELL_BASE + ((2 * qid as usize) * (4 << dstrd));
    unsafe { write_mmio32(mmio_base, sq_db_offset, qp.sq_tail as u32) };

    let cq_slot = (qp.cq_virt as *const NvmeCqEntry).wrapping_add(qp.cq_head as usize);
    for _ in 0..POLL_TIMEOUT_ITERS {
        let cqe = unsafe { read_volatile(cq_slot) };
        if cqe.phase() == qp.cq_phase {
            qp.cq_head += 1;
            if qp.cq_head == qp.depth {
                qp.cq_head = 0;
                qp.cq_phase = !qp.cq_phase;
            }
            let cq_db_offset = regs::DOORBELL_BASE + ((2 * qid as usize + 1) * (4 << dstrd));
            unsafe { write_mmio32(mmio_base, cq_db_offset, qp.cq_head as u32) };

            if cqe.status_code() != 0 {
                return Err("nvme: command completed with error status");
            }
            return Ok(cqe);
        }
        core::hint::spin_loop();
    }

    Err("nvme: command completion timeout")
}

fn wait_csts_ready(mmio_base: u64, expected: bool) -> bool {
    for _ in 0..POLL_TIMEOUT_ITERS {
        let csts = unsafe { read_mmio32(mmio_base, regs::CSTS) };
        if (csts & CSTS_CFS) != 0 {
            return false;
        }
        let rdy = (csts & CSTS_RDY) != 0;
        if rdy == expected {
            return true;
        }
        core::hint::spin_loop();
    }
    false
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
unsafe fn read_mmio64(base: u64, offset: usize) -> u64 {
    let lo = read_mmio32(base, offset) as u64;
    let hi = read_mmio32(base, offset + 4) as u64;
    (hi << 32) | lo
}

#[inline]
unsafe fn write_mmio64(base: u64, offset: usize, val: u64) {
    write_mmio32(base, offset, val as u32);
    write_mmio32(base, offset + 4, (val >> 32) as u32);
}
