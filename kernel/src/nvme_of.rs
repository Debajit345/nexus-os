//! `no_std` NVMe-oF (NVMe over Fabrics / TCP) & RDMA Queue-Pair Engine.
//!
//! Provides:
//! - Castagnoli CRC32C digest calculation for NVMe/TCP header and data digests.
//! - NVMe/TCP PDU framing (`ICReq`, `ICResp`, `H2CTermReq`, `C2HTermReq`,
//!   `CapsuleCmd`, `CapsuleResp`, `H2CData`, `C2HData`, `R2T`) with optional
//!   header/data digests and PDU data offset (PDO) alignment.
//! - NVMe-oF 64-byte Submission Queue Entries (`NvmeOfSqe`), 16-byte Completion
//!   Queue Entries (`NvmeOfCqe`), SGL descriptors (Data Block, Keyed Data Block,
//!   Transport Data Block), and Fabrics Connect/Property/Disconnect commands.
//! - [`NvmeTcpQueue`] connection & command state machine.
//! - RDMA Queue-Pair Engine ([`RdmaEngine`]) with Protection Domains, Memory
//!   Regions (`lkey`/`rkey` + access flags), Scatter-Gather Elements, Send/Recv
//!   Work Requests, Completion Queues, IBTA QP state machine, and loopback
//!   execution for `Send`, `RdmaWrite`, `RdmaRead`, `AtomicCmpSwp`, and
//!   `AtomicFetchAdd`.
//! - [`NvmeRdmaQueue`] binding NVMe-oF capsules onto RDMA Queue Pairs.
//! - [`self_test`] exercising all protocol layers end-to-end.

#![allow(dead_code)]

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::mem::size_of;

use crate::mm::SpinLock;
use crate::serial_println;

// ---------------------------------------------------------------------------
// Constants & Error Types
// ---------------------------------------------------------------------------

/// Default NVMe/TCP port (IANA assigned).
pub const NVME_TCP_PORT: u16 = 4420;

/// Default NVMe/TCP discovery port.
pub const NVME_TCP_DISC_PORT: u16 = 8009;

/// NVMe/TCP Protocol Format Version 1.0.
pub const NVME_TCP_PFV_1_0: u16 = 0x0000;

/// NVMe Fabrics command opcode.
pub const NVME_FABRICS_OPCODE: u8 = 0x7F;

/// NVM Command Set opcodes.
pub const NVM_CMD_FLUSH: u8 = 0x00;
pub const NVM_CMD_WRITE: u8 = 0x01;
pub const NVM_CMD_READ: u8 = 0x02;

/// CRC32C digest size in bytes.
pub const NVME_TCP_DIGEST_LEN: usize = 4;

/// NVMe-oF Connect Data payload size in bytes.
pub const NVME_CONNECT_DATA_LEN: usize = 1024;

/// Errors returned by the NVMe-oF and RDMA engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvmeOfError {
    InvalidBuffer,
    BufferTooSmall,
    InvalidPduType,
    InvalidPduLength,
    InvalidHeaderLength,
    DigestMismatch,
    InvalidState,
    InvalidTransition,
    QueueFull,
    QueueEmpty,
    CommandNotFound,
    InvalidProtectionDomain,
    InvalidMemoryRegion,
    AccessViolation,
    BoundsExceeded,
    InvalidQueuePair,
    InvalidCompletionQueue,
    UnsupportedOpcode,
    ProtocolError,
}

impl NvmeOfError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidBuffer => "invalid buffer",
            Self::BufferTooSmall => "buffer too small",
            Self::InvalidPduType => "invalid NVMe/TCP PDU type",
            Self::InvalidPduLength => "invalid NVMe/TCP PDU length",
            Self::InvalidHeaderLength => "invalid NVMe/TCP header length",
            Self::DigestMismatch => "CRC32C digest mismatch",
            Self::InvalidState => "invalid connection or queue state",
            Self::InvalidTransition => "invalid QP state transition",
            Self::QueueFull => "queue is full",
            Self::QueueEmpty => "queue is empty",
            Self::CommandNotFound => "command ID not found",
            Self::InvalidProtectionDomain => "invalid protection domain",
            Self::InvalidMemoryRegion => "invalid memory region key",
            Self::AccessViolation => "memory region access violation",
            Self::BoundsExceeded => "memory region bounds exceeded",
            Self::InvalidQueuePair => "invalid queue pair",
            Self::InvalidCompletionQueue => "invalid completion queue",
            Self::UnsupportedOpcode => "unsupported opcode",
            Self::ProtocolError => "NVMe-oF protocol error",
        }
    }
}

impl fmt::Display for NvmeOfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// CRC32C (Castagnoli)
// ---------------------------------------------------------------------------

const CRC32C_POLY_REFLECTED: u32 = 0x82F6_3B78;

/// Update a running CRC32C (Castagnoli) checksum without large lookup tables.
#[inline]
pub fn crc32c_update(mut crc: u32, data: &[u8]) -> u32 {
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (CRC32C_POLY_REFLECTED & mask);
        }
    }
    crc
}

/// Compute the Castagnoli CRC32C checksum of `data`.
#[inline]
pub fn crc32c(data: &[u8]) -> u32 {
    !crc32c_update(0xFFFF_FFFF, data)
}

// ---------------------------------------------------------------------------
// Little-Endian Helpers
// ---------------------------------------------------------------------------

#[inline]
fn read_u16_le(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

#[inline]
fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
fn read_u64_le(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

#[inline]
fn write_u16_le(buf: &mut [u8], off: usize, val: u16) {
    buf[off..off + 2].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn write_u32_le(buf: &mut [u8], off: usize, val: u32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn write_u64_le(buf: &mut [u8], off: usize, val: u64) {
    buf[off..off + 8].copy_from_slice(&val.to_le_bytes());
}

// ---------------------------------------------------------------------------
// NVMe-oF SGL Descriptors & Capsule Command / Response
// ---------------------------------------------------------------------------

/// NVMe SGL Descriptor Type (upper 4 bits of `type_id`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SglType {
    DataBlock = 0x00,
    BitBucket = 0x01,
    Segment = 0x02,
    LastSegment = 0x03,
    KeyedDataBlock = 0x04,
    TransportDataBlock = 0x05,
}

/// NVMe SGL Descriptor Sub-Type (lower 4 bits of `type_id`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SglSubType {
    Address = 0x00,
    Offset = 0x01,
    Transport = 0x0A,
}

/// 16-byte NVMe Scatter-Gather List (SGL) Descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SglDescriptor {
    pub addr: u64,
    pub length: u32,
    pub key_or_rsvd: [u8; 3],
    pub type_id: u8,
}

const _: () = assert!(size_of::<SglDescriptor>() == 16);

impl SglDescriptor {
    /// Construct a standard Data Block SGL descriptor.
    pub const fn data_block(addr: u64, length: u32) -> Self {
        Self {
            addr,
            length,
            key_or_rsvd: [0; 3],
            type_id: (SglType::DataBlock as u8) << 4 | (SglSubType::Address as u8),
        }
    }

    /// Construct an Offset Data Block SGL descriptor (used for in-capsule data).
    pub const fn offset_data_block(offset: u64, length: u32) -> Self {
        Self {
            addr: offset,
            length,
            key_or_rsvd: [0; 3],
            type_id: (SglType::DataBlock as u8) << 4 | (SglSubType::Offset as u8),
        }
    }

    /// Construct a Transport Data Block SGL descriptor (used for NVMe/TCP H2C/C2H).
    pub const fn transport_data_block(length: u32) -> Self {
        Self {
            addr: 0,
            length,
            key_or_rsvd: [0; 3],
            type_id: (SglType::TransportDataBlock as u8) << 4 | (SglSubType::Transport as u8),
        }
    }

    /// Construct a Keyed Data Block SGL descriptor (used for NVMe-oF RDMA).
    /// Note: In NVMe-oF Keyed SGL, bytes 8..11 hold length (24-bit) and bytes 11..15 hold key (32-bit).
    /// Here we store `length` in `length` (24-bit effective) and `rkey` across `key_or_rsvd` + `addr`
    /// or encode cleanly into wire bytes via `to_bytes`/`from_bytes`.
    pub const fn keyed_data_block(addr: u64, length: u32, rkey: u32) -> Self {
        let rkey_bytes = rkey.to_le_bytes();
        Self {
            addr,
            length: (length & 0x00FF_FFFF) | ((rkey_bytes[0] as u32) << 24),
            key_or_rsvd: [rkey_bytes[1], rkey_bytes[2], rkey_bytes[3]],
            type_id: (SglType::KeyedDataBlock as u8) << 4 | (SglSubType::Address as u8),
        }
    }

    /// Extract the 24-bit length and 32-bit rkey from a Keyed Data Block SGL.
    pub const fn keyed_parts(&self) -> (u32, u32) {
        let len = self.length & 0x00FF_FFFF;
        let b0 = ((self.length >> 24) & 0xFF) as u8;
        let rkey = u32::from_le_bytes([
            b0,
            self.key_or_rsvd[0],
            self.key_or_rsvd[1],
            self.key_or_rsvd[2],
        ]);
        (len, rkey)
    }

    #[inline]
    pub const fn sgl_type(&self) -> u8 {
        self.type_id >> 4
    }

    #[inline]
    pub const fn sgl_subtype(&self) -> u8 {
        self.type_id & 0x0F
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_bytes(&self, out: &mut [u8]) -> Result<(), NvmeOfError> {
        if out.len() < 16 {
            return Err(NvmeOfError::BufferTooSmall);
        }
        write_u64_le(out, 0, self.addr);
        write_u32_le(out, 8, self.length);
        out[12..15].copy_from_slice(&self.key_or_rsvd);
        out[15] = self.type_id;
        Ok(())
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, NvmeOfError> {
        if buf.len() < 16 {
            return Err(NvmeOfError::BufferTooSmall);
        }
        Ok(Self {
            addr: read_u64_le(buf, 0),
            length: read_u32_le(buf, 8),
            key_or_rsvd: [buf[12], buf[13], buf[14]],
            type_id: buf[15],
        })
    }
}

/// NVMe Fabrics Command Type (`fctype` in SQE byte 4).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FabricsCmdType {
    PropertySet = 0x00,
    Connect = 0x01,
    PropertyGet = 0x04,
    AuthSend = 0x05,
    AuthRecv = 0x06,
    Disconnect = 0x08,
}

/// 64-byte NVMe-oF Submission Queue Entry (Capsule Command SQE).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NvmeOfSqe {
    pub cdw0: u32,
    pub nsid: u32,
    pub rsvd2: u64,
    pub mptr: u64,
    pub sgl1: SglDescriptor,
    pub cdw10: u32,
    pub cdw11: u32,
    pub cdw12: u32,
    pub cdw13: u32,
    pub cdw14: u32,
    pub cdw15: u32,
}

const _: () = assert!(size_of::<NvmeOfSqe>() == 64);

impl NvmeOfSqe {
    /// Build CDW0 with opcode, PSDT=01b (SGL for data transfer), and command ID.
    #[inline]
    pub const fn make_cdw0(opcode: u8, psdt_sgl: bool, cid: u16) -> u32 {
        let psdt = if psdt_sgl { 1u32 << 14 } else { 0 };
        (opcode as u32) | psdt | ((cid as u32) << 16)
    }

    #[inline]
    pub const fn opcode(&self) -> u8 {
        (self.cdw0 & 0xFF) as u8
    }

    #[inline]
    pub const fn cid(&self) -> u16 {
        ((self.cdw0 >> 16) & 0xFFFF) as u16
    }

    #[inline]
    pub const fn fctype(&self) -> u8 {
        (self.nsid & 0xFF) as u8
    }

    /// Construct a Fabrics Connect SQE.
    pub fn fabrics_connect(
        cid: u16,
        qid: u16,
        sqsize: u16,
        recfmt: u16,
        sgl1: SglDescriptor,
    ) -> Self {
        Self {
            cdw0: Self::make_cdw0(NVME_FABRICS_OPCODE, true, cid),
            nsid: FabricsCmdType::Connect as u32,
            rsvd2: 0,
            mptr: 0,
            sgl1,
            cdw10: (recfmt as u32) | ((qid as u32) << 16),
            cdw11: sqsize as u32,
            cdw12: 0,
            cdw13: 0,
            cdw14: 0,
            cdw15: 0,
        }
    }

    /// Construct a Fabrics Property Get SQE.
    pub fn fabrics_property_get(cid: u16, attrib: u8, offset: u32) -> Self {
        Self {
            cdw0: Self::make_cdw0(NVME_FABRICS_OPCODE, true, cid),
            nsid: FabricsCmdType::PropertyGet as u32,
            cdw10: attrib as u32,
            cdw11: offset,
            ..Default::default()
        }
    }

    /// Construct a Fabrics Property Set SQE.
    pub fn fabrics_property_set(cid: u16, attrib: u8, offset: u32, value: u64) -> Self {
        Self {
            cdw0: Self::make_cdw0(NVME_FABRICS_OPCODE, true, cid),
            nsid: FabricsCmdType::PropertySet as u32,
            cdw10: attrib as u32,
            cdw11: offset,
            cdw12: value as u32,
            cdw13: (value >> 32) as u32,
            ..Default::default()
        }
    }

    /// Construct a Fabrics Disconnect SQE.
    pub fn fabrics_disconnect(cid: u16, recfmt: u16) -> Self {
        Self {
            cdw0: Self::make_cdw0(NVME_FABRICS_OPCODE, true, cid),
            nsid: FabricsCmdType::Disconnect as u32,
            cdw10: recfmt as u32,
            ..Default::default()
        }
    }

    /// Construct an NVM Read SQE using an SGL descriptor.
    pub fn nvm_read(
        cid: u16,
        nsid: u32,
        slba: u64,
        nlb_zero_based: u16,
        sgl1: SglDescriptor,
    ) -> Self {
        Self {
            cdw0: Self::make_cdw0(NVM_CMD_READ, true, cid),
            nsid,
            rsvd2: 0,
            mptr: 0,
            sgl1,
            cdw10: slba as u32,
            cdw11: (slba >> 32) as u32,
            cdw12: nlb_zero_based as u32,
            cdw13: 0,
            cdw14: 0,
            cdw15: 0,
        }
    }

    /// Construct an NVM Write SQE using an SGL descriptor.
    pub fn nvm_write(
        cid: u16,
        nsid: u32,
        slba: u64,
        nlb_zero_based: u16,
        sgl1: SglDescriptor,
    ) -> Self {
        Self {
            cdw0: Self::make_cdw0(NVM_CMD_WRITE, true, cid),
            nsid,
            rsvd2: 0,
            mptr: 0,
            sgl1,
            cdw10: slba as u32,
            cdw11: (slba >> 32) as u32,
            cdw12: nlb_zero_based as u32,
            cdw13: 0,
            cdw14: 0,
            cdw15: 0,
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_bytes(&self, out: &mut [u8]) -> Result<(), NvmeOfError> {
        if out.len() < 64 {
            return Err(NvmeOfError::BufferTooSmall);
        }
        write_u32_le(out, 0, self.cdw0);
        write_u32_le(out, 4, self.nsid);
        write_u64_le(out, 8, self.rsvd2);
        write_u64_le(out, 16, self.mptr);
        self.sgl1.to_bytes(&mut out[24..40])?;
        write_u32_le(out, 40, self.cdw10);
        write_u32_le(out, 44, self.cdw11);
        write_u32_le(out, 48, self.cdw12);
        write_u32_le(out, 52, self.cdw13);
        write_u32_le(out, 56, self.cdw14);
        write_u32_le(out, 60, self.cdw15);
        Ok(())
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, NvmeOfError> {
        if buf.len() < 64 {
            return Err(NvmeOfError::BufferTooSmall);
        }
        Ok(Self {
            cdw0: read_u32_le(buf, 0),
            nsid: read_u32_le(buf, 4),
            rsvd2: read_u64_le(buf, 8),
            mptr: read_u64_le(buf, 16),
            sgl1: SglDescriptor::from_bytes(&buf[24..40])?,
            cdw10: read_u32_le(buf, 40),
            cdw11: read_u32_le(buf, 44),
            cdw12: read_u32_le(buf, 48),
            cdw13: read_u32_le(buf, 52),
            cdw14: read_u32_le(buf, 56),
            cdw15: read_u32_le(buf, 60),
        })
    }
}

/// 16-byte NVMe-oF Completion Queue Entry (Capsule Response CQE).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NvmeOfCqe {
    pub dw0: u32,
    pub dw1: u32,
    pub sq_head: u16,
    pub sq_id: u16,
    pub cid: u16,
    pub status: u16,
}

const _: () = assert!(size_of::<NvmeOfCqe>() == 16);

impl NvmeOfCqe {
    pub const fn success(cid: u16, sq_id: u16, sq_head: u16, dw0: u32) -> Self {
        Self {
            dw0,
            dw1: 0,
            sq_head,
            sq_id,
            cid,
            status: 0,
        }
    }

    pub const fn with_status(cid: u16, sq_id: u16, sct: u8, sc: u8) -> Self {
        let status = (((sct as u16) & 0x07) << 9) | ((sc as u16) << 1);
        Self {
            dw0: 0,
            dw1: 0,
            sq_head: 0,
            sq_id,
            cid,
            status,
        }
    }

    #[inline]
    pub const fn phase(&self) -> bool {
        (self.status & 0x1) != 0
    }

    #[inline]
    pub const fn status_code(&self) -> u8 {
        ((self.status >> 1) & 0xFF) as u8
    }

    #[inline]
    pub const fn status_type(&self) -> u8 {
        ((self.status >> 9) & 0x07) as u8
    }

    #[inline]
    pub const fn is_success(&self) -> bool {
        (self.status >> 1) == 0
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_bytes(&self, out: &mut [u8]) -> Result<(), NvmeOfError> {
        if out.len() < 16 {
            return Err(NvmeOfError::BufferTooSmall);
        }
        write_u32_le(out, 0, self.dw0);
        write_u32_le(out, 4, self.dw1);
        write_u16_le(out, 8, self.sq_head);
        write_u16_le(out, 10, self.sq_id);
        write_u16_le(out, 12, self.cid);
        write_u16_le(out, 14, self.status);
        Ok(())
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, NvmeOfError> {
        if buf.len() < 16 {
            return Err(NvmeOfError::BufferTooSmall);
        }
        Ok(Self {
            dw0: read_u32_le(buf, 0),
            dw1: read_u32_le(buf, 4),
            sq_head: read_u16_le(buf, 8),
            sq_id: read_u16_le(buf, 10),
            cid: read_u16_le(buf, 12),
            status: read_u16_le(buf, 14),
        })
    }
}

/// Encode a 1024-byte NVMe-oF Fabrics Connect Data payload into a heap `Vec<u8>`.
pub fn encode_connect_data(
    hostid: &[u8; 16],
    cntlid: u16,
    subnqn: &[u8],
    hostnqn: &[u8],
) -> Vec<u8> {
    let mut buf = vec![0u8; NVME_CONNECT_DATA_LEN];
    buf[0..16].copy_from_slice(hostid);
    write_u16_le(&mut buf, 16, cntlid);
    let sub_len = subnqn.len().min(255);
    buf[256..256 + sub_len].copy_from_slice(&subnqn[..sub_len]);
    let host_len = hostnqn.len().min(255);
    buf[512..512 + host_len].copy_from_slice(&hostnqn[..host_len]);
    buf
}

// ---------------------------------------------------------------------------
// NVMe/TCP PDU Framing
// ---------------------------------------------------------------------------

/// NVMe/TCP PDU Types.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PduType {
    ICReq = 0x00,
    ICResp = 0x01,
    H2CTermReq = 0x02,
    C2HTermReq = 0x03,
    CapsuleCmd = 0x04,
    CapsuleResp = 0x05,
    H2CData = 0x06,
    C2HData = 0x07,
    R2T = 0x09,
}

impl PduType {
    pub const fn from_u8(v: u8) -> Result<Self, NvmeOfError> {
        match v {
            0x00 => Ok(Self::ICReq),
            0x01 => Ok(Self::ICResp),
            0x02 => Ok(Self::H2CTermReq),
            0x03 => Ok(Self::C2HTermReq),
            0x04 => Ok(Self::CapsuleCmd),
            0x05 => Ok(Self::CapsuleResp),
            0x06 => Ok(Self::H2CData),
            0x07 => Ok(Self::C2HData),
            0x09 => Ok(Self::R2T),
            _ => Err(NvmeOfError::InvalidPduType),
        }
    }
}

/// NVMe/TCP PDU Common Header Flags.
pub mod pdu_flags {
    pub const HDGSTF: u8 = 1 << 0;
    pub const DDGSTF: u8 = 1 << 1;
    pub const DATA_LAST: u8 = 1 << 2;
    pub const DATA_SUCCESS: u8 = 1 << 3;
}

/// 8-byte NVMe/TCP Common PDU Header.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PduCommonHeader {
    pub pdu_type: u8,
    pub flags: u8,
    pub hlen: u8,
    pub pdo: u8,
    pub plen: u32,
}

const _: () = assert!(size_of::<PduCommonHeader>() == 8);

impl PduCommonHeader {
    #[allow(clippy::wrong_self_convention)]
    pub fn to_bytes(&self, out: &mut [u8]) -> Result<(), NvmeOfError> {
        if out.len() < 8 {
            return Err(NvmeOfError::BufferTooSmall);
        }
        out[0] = self.pdu_type;
        out[1] = self.flags;
        out[2] = self.hlen;
        out[3] = self.pdo;
        write_u32_le(out, 4, self.plen);
        Ok(())
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, NvmeOfError> {
        if buf.len() < 8 {
            return Err(NvmeOfError::BufferTooSmall);
        }
        let pdu_type = buf[0];
        let _ = PduType::from_u8(pdu_type)?;
        let hlen = buf[2];
        let plen = read_u32_le(buf, 4);
        if (hlen as u32) < 8 || (hlen as u32) > plen {
            return Err(NvmeOfError::InvalidHeaderLength);
        }
        Ok(Self {
            pdu_type,
            flags: buf[1],
            hlen,
            pdo: buf[3],
            plen,
        })
    }
}

/// Initialize Connection Request (`ICReq`, 128 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IcReqPdu {
    pub pfv: u16,
    pub hpda: u8,
    pub digest: u8,
    pub maxr2t: u32,
}

impl Default for IcReqPdu {
    fn default() -> Self {
        Self {
            pfv: NVME_TCP_PFV_1_0,
            hpda: 0,
            digest: 0,
            maxr2t: 16,
        }
    }
}

/// Initialize Connection Response (`ICResp`, 128 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IcRespPdu {
    pub pfv: u16,
    pub cpda: u8,
    pub digest: u8,
    pub maxh2cdata: u32,
}

impl Default for IcRespPdu {
    fn default() -> Self {
        Self {
            pfv: NVME_TCP_PFV_1_0,
            cpda: 0,
            digest: 0,
            maxh2cdata: 131_072,
        }
    }
}

/// Terminate Connection Request (`H2CTermReq` / `C2HTermReq`, 24 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermReqPdu {
    pub is_c2h: bool,
    pub fes: u16,
    pub fei: u32,
}

/// Ready to Transfer (`R2T`, 24 bytes + optional HDGST).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R2TPdu {
    pub cccid: u16,
    pub ttag: u16,
    pub r2to: u32,
    pub r2tl: u32,
}

/// Host-to-Controller Data (`H2CData`, 24 bytes header + optional HDGST + data + optional DDGST).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H2CDataPdu {
    pub flags: u8,
    pub cccid: u16,
    pub ttag: u16,
    pub datao: u32,
    pub datal: u32,
    pub data: Vec<u8>,
}

/// Controller-to-Host Data (`C2HData`, 24 bytes header + optional HDGST + data + optional DDGST).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C2HDataPdu {
    pub flags: u8,
    pub cccid: u16,
    pub datao: u32,
    pub datal: u32,
    pub data: Vec<u8>,
}

/// Decoded NVMe/TCP PDU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NvmeTcpPdu {
    ICReq(IcReqPdu),
    ICResp(IcRespPdu),
    TermReq(TermReqPdu),
    CapsuleCmd { sqe: NvmeOfSqe, data: Vec<u8> },
    CapsuleResp { cqe: NvmeOfCqe },
    H2CData(H2CDataPdu),
    C2HData(C2HDataPdu),
    R2T(R2TPdu),
}

/// Align `val` up to `align` (power of two).
#[inline]
const fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// Encode an [`NvmeTcpPdu`] into wire bytes, optionally appending CRC32C header/data digests.
pub fn encode_pdu(
    pdu: &NvmeTcpPdu,
    hdgst: bool,
    ddgst: bool,
    pda_units: u8,
) -> Result<Vec<u8>, NvmeOfError> {
    let pda_align = ((pda_units as usize) + 1) * 4;
    match pdu {
        NvmeTcpPdu::ICReq(req) => {
            let mut buf = vec![0u8; 128];
            let hdr = PduCommonHeader {
                pdu_type: PduType::ICReq as u8,
                flags: 0,
                hlen: 128,
                pdo: 0,
                plen: 128,
            };
            hdr.to_bytes(&mut buf[0..8])?;
            write_u16_le(&mut buf, 8, req.pfv);
            buf[10] = req.hpda;
            buf[11] = req.digest;
            write_u32_le(&mut buf, 12, req.maxr2t);
            Ok(buf)
        }
        NvmeTcpPdu::ICResp(resp) => {
            let mut buf = vec![0u8; 128];
            let hdr = PduCommonHeader {
                pdu_type: PduType::ICResp as u8,
                flags: 0,
                hlen: 128,
                pdo: 0,
                plen: 128,
            };
            hdr.to_bytes(&mut buf[0..8])?;
            write_u16_le(&mut buf, 8, resp.pfv);
            buf[10] = resp.cpda;
            buf[11] = resp.digest;
            write_u32_le(&mut buf, 12, resp.maxh2cdata);
            Ok(buf)
        }
        NvmeTcpPdu::TermReq(term) => {
            let mut buf = vec![0u8; 24];
            let ptype = if term.is_c2h {
                PduType::C2HTermReq
            } else {
                PduType::H2CTermReq
            };
            let hdr = PduCommonHeader {
                pdu_type: ptype as u8,
                flags: 0,
                hlen: 24,
                pdo: 0,
                plen: 24,
            };
            hdr.to_bytes(&mut buf[0..8])?;
            write_u16_le(&mut buf, 8, term.fes);
            write_u32_le(&mut buf, 10, term.fei);
            Ok(buf)
        }
        NvmeTcpPdu::CapsuleCmd { sqe, data } => {
            let hlen = 72usize;
            let hdgst_len = if hdgst { NVME_TCP_DIGEST_LEN } else { 0 };
            let has_data = !data.is_empty();
            let ddgst_len = if ddgst && has_data {
                NVME_TCP_DIGEST_LEN
            } else {
                0
            };
            let pdo = if has_data {
                align_up(hlen + hdgst_len, pda_align)
            } else {
                0
            };
            let plen = if has_data {
                pdo + data.len() + ddgst_len
            } else {
                hlen + hdgst_len
            };
            let mut flags = 0u8;
            if hdgst {
                flags |= pdu_flags::HDGSTF;
            }
            if ddgst && has_data {
                flags |= pdu_flags::DDGSTF;
            }
            let mut buf = vec![0u8; plen];
            let hdr = PduCommonHeader {
                pdu_type: PduType::CapsuleCmd as u8,
                flags,
                hlen: hlen as u8,
                pdo: pdo as u8,
                plen: plen as u32,
            };
            hdr.to_bytes(&mut buf[0..8])?;
            sqe.to_bytes(&mut buf[8..72])?;
            if hdgst {
                let c = crc32c(&buf[0..hlen]);
                write_u32_le(&mut buf, hlen, c);
            }
            if has_data {
                buf[pdo..pdo + data.len()].copy_from_slice(data);
                if ddgst {
                    let c = crc32c(&buf[pdo..pdo + data.len()]);
                    write_u32_le(&mut buf, pdo + data.len(), c);
                }
            }
            Ok(buf)
        }
        NvmeTcpPdu::CapsuleResp { cqe } => {
            let hlen = 24usize;
            let hdgst_len = if hdgst { NVME_TCP_DIGEST_LEN } else { 0 };
            let plen = hlen + hdgst_len;
            let flags = if hdgst { pdu_flags::HDGSTF } else { 0 };
            let mut buf = vec![0u8; plen];
            let hdr = PduCommonHeader {
                pdu_type: PduType::CapsuleResp as u8,
                flags,
                hlen: hlen as u8,
                pdo: 0,
                plen: plen as u32,
            };
            hdr.to_bytes(&mut buf[0..8])?;
            cqe.to_bytes(&mut buf[8..24])?;
            if hdgst {
                let c = crc32c(&buf[0..hlen]);
                write_u32_le(&mut buf, hlen, c);
            }
            Ok(buf)
        }
        NvmeTcpPdu::H2CData(h2c) => {
            let hlen = 24usize;
            let hdgst_len = if hdgst { NVME_TCP_DIGEST_LEN } else { 0 };
            let has_data = !h2c.data.is_empty();
            let ddgst_len = if ddgst && has_data {
                NVME_TCP_DIGEST_LEN
            } else {
                0
            };
            let pdo = if has_data {
                align_up(hlen + hdgst_len, pda_align)
            } else {
                0
            };
            let plen = if has_data {
                pdo + h2c.data.len() + ddgst_len
            } else {
                hlen + hdgst_len
            };
            let mut flags = h2c.flags & (pdu_flags::DATA_LAST);
            if hdgst {
                flags |= pdu_flags::HDGSTF;
            }
            if ddgst && has_data {
                flags |= pdu_flags::DDGSTF;
            }
            let mut buf = vec![0u8; plen];
            let hdr = PduCommonHeader {
                pdu_type: PduType::H2CData as u8,
                flags,
                hlen: hlen as u8,
                pdo: pdo as u8,
                plen: plen as u32,
            };
            hdr.to_bytes(&mut buf[0..8])?;
            write_u16_le(&mut buf, 8, h2c.cccid);
            write_u16_le(&mut buf, 10, h2c.ttag);
            write_u32_le(&mut buf, 12, h2c.datao);
            write_u32_le(&mut buf, 16, h2c.datal);
            if hdgst {
                let c = crc32c(&buf[0..hlen]);
                write_u32_le(&mut buf, hlen, c);
            }
            if has_data {
                buf[pdo..pdo + h2c.data.len()].copy_from_slice(&h2c.data);
                if ddgst {
                    let c = crc32c(&buf[pdo..pdo + h2c.data.len()]);
                    write_u32_le(&mut buf, pdo + h2c.data.len(), c);
                }
            }
            Ok(buf)
        }
        NvmeTcpPdu::C2HData(c2h) => {
            let hlen = 24usize;
            let hdgst_len = if hdgst { NVME_TCP_DIGEST_LEN } else { 0 };
            let has_data = !c2h.data.is_empty();
            let ddgst_len = if ddgst && has_data {
                NVME_TCP_DIGEST_LEN
            } else {
                0
            };
            let pdo = if has_data {
                align_up(hlen + hdgst_len, pda_align)
            } else {
                0
            };
            let plen = if has_data {
                pdo + c2h.data.len() + ddgst_len
            } else {
                hlen + hdgst_len
            };
            let mut flags = c2h.flags & (pdu_flags::DATA_LAST | pdu_flags::DATA_SUCCESS);
            if hdgst {
                flags |= pdu_flags::HDGSTF;
            }
            if ddgst && has_data {
                flags |= pdu_flags::DDGSTF;
            }
            let mut buf = vec![0u8; plen];
            let hdr = PduCommonHeader {
                pdu_type: PduType::C2HData as u8,
                flags,
                hlen: hlen as u8,
                pdo: pdo as u8,
                plen: plen as u32,
            };
            hdr.to_bytes(&mut buf[0..8])?;
            write_u16_le(&mut buf, 8, c2h.cccid);
            write_u32_le(&mut buf, 12, c2h.datao);
            write_u32_le(&mut buf, 16, c2h.datal);
            if hdgst {
                let c = crc32c(&buf[0..hlen]);
                write_u32_le(&mut buf, hlen, c);
            }
            if has_data {
                buf[pdo..pdo + c2h.data.len()].copy_from_slice(&c2h.data);
                if ddgst {
                    let c = crc32c(&buf[pdo..pdo + c2h.data.len()]);
                    write_u32_le(&mut buf, pdo + c2h.data.len(), c);
                }
            }
            Ok(buf)
        }
        NvmeTcpPdu::R2T(r2t) => {
            let hlen = 24usize;
            let hdgst_len = if hdgst { NVME_TCP_DIGEST_LEN } else { 0 };
            let plen = hlen + hdgst_len;
            let flags = if hdgst { pdu_flags::HDGSTF } else { 0 };
            let mut buf = vec![0u8; plen];
            let hdr = PduCommonHeader {
                pdu_type: PduType::R2T as u8,
                flags,
                hlen: hlen as u8,
                pdo: 0,
                plen: plen as u32,
            };
            hdr.to_bytes(&mut buf[0..8])?;
            write_u16_le(&mut buf, 8, r2t.cccid);
            write_u16_le(&mut buf, 10, r2t.ttag);
            write_u32_le(&mut buf, 12, r2t.r2to);
            write_u32_le(&mut buf, 16, r2t.r2tl);
            if hdgst {
                let c = crc32c(&buf[0..hlen]);
                write_u32_le(&mut buf, hlen, c);
            }
            Ok(buf)
        }
    }
}

/// Decode an [`NvmeTcpPdu`] from wire bytes, verifying CRC32C digests if flagged.
pub fn decode_pdu(buf: &[u8]) -> Result<(NvmeTcpPdu, usize), NvmeOfError> {
    let hdr = PduCommonHeader::from_bytes(buf)?;
    let plen = hdr.plen as usize;
    let hlen = hdr.hlen as usize;
    if buf.len() < plen {
        return Err(NvmeOfError::BufferTooSmall);
    }
    let has_hdgst = (hdr.flags & pdu_flags::HDGSTF) != 0;
    let has_ddgst = (hdr.flags & pdu_flags::DDGSTF) != 0;

    if has_hdgst {
        if plen < hlen + NVME_TCP_DIGEST_LEN {
            return Err(NvmeOfError::InvalidPduLength);
        }
        let expected = read_u32_le(buf, hlen);
        let actual = crc32c(&buf[0..hlen]);
        if expected != actual {
            return Err(NvmeOfError::DigestMismatch);
        }
    }

    let ptype = PduType::from_u8(hdr.pdu_type)?;
    let pdu = match ptype {
        PduType::ICReq => {
            if hlen != 128 || plen != 128 {
                return Err(NvmeOfError::InvalidPduLength);
            }
            NvmeTcpPdu::ICReq(IcReqPdu {
                pfv: read_u16_le(buf, 8),
                hpda: buf[10],
                digest: buf[11],
                maxr2t: read_u32_le(buf, 12),
            })
        }
        PduType::ICResp => {
            if hlen != 128 || plen != 128 {
                return Err(NvmeOfError::InvalidPduLength);
            }
            NvmeTcpPdu::ICResp(IcRespPdu {
                pfv: read_u16_le(buf, 8),
                cpda: buf[10],
                digest: buf[11],
                maxh2cdata: read_u32_le(buf, 12),
            })
        }
        PduType::H2CTermReq | PduType::C2HTermReq => {
            if hlen != 24 || plen != 24 {
                return Err(NvmeOfError::InvalidPduLength);
            }
            NvmeTcpPdu::TermReq(TermReqPdu {
                is_c2h: ptype == PduType::C2HTermReq,
                fes: read_u16_le(buf, 8),
                fei: read_u32_le(buf, 10),
            })
        }
        PduType::CapsuleCmd => {
            if hlen < 72 {
                return Err(NvmeOfError::InvalidHeaderLength);
            }
            let sqe = NvmeOfSqe::from_bytes(&buf[8..72])?;
            let data = extract_pdu_data(buf, &hdr, has_ddgst)?;
            NvmeTcpPdu::CapsuleCmd { sqe, data }
        }
        PduType::CapsuleResp => {
            if hlen != 24 {
                return Err(NvmeOfError::InvalidHeaderLength);
            }
            let cqe = NvmeOfCqe::from_bytes(&buf[8..24])?;
            NvmeTcpPdu::CapsuleResp { cqe }
        }
        PduType::H2CData => {
            if hlen != 24 {
                return Err(NvmeOfError::InvalidHeaderLength);
            }
            let data = extract_pdu_data(buf, &hdr, has_ddgst)?;
            NvmeTcpPdu::H2CData(H2CDataPdu {
                flags: hdr.flags,
                cccid: read_u16_le(buf, 8),
                ttag: read_u16_le(buf, 10),
                datao: read_u32_le(buf, 12),
                datal: read_u32_le(buf, 16),
                data,
            })
        }
        PduType::C2HData => {
            if hlen != 24 {
                return Err(NvmeOfError::InvalidHeaderLength);
            }
            let data = extract_pdu_data(buf, &hdr, has_ddgst)?;
            NvmeTcpPdu::C2HData(C2HDataPdu {
                flags: hdr.flags,
                cccid: read_u16_le(buf, 8),
                datao: read_u32_le(buf, 12),
                datal: read_u32_le(buf, 16),
                data,
            })
        }
        PduType::R2T => {
            if hlen != 24 {
                return Err(NvmeOfError::InvalidHeaderLength);
            }
            NvmeTcpPdu::R2T(R2TPdu {
                cccid: read_u16_le(buf, 8),
                ttag: read_u16_le(buf, 10),
                r2to: read_u32_le(buf, 12),
                r2tl: read_u32_le(buf, 16),
            })
        }
    };

    Ok((pdu, plen))
}

fn extract_pdu_data(
    buf: &[u8],
    hdr: &PduCommonHeader,
    has_ddgst: bool,
) -> Result<Vec<u8>, NvmeOfError> {
    let plen = hdr.plen as usize;
    let hlen = hdr.hlen as usize;
    let hdgst_len = if (hdr.flags & pdu_flags::HDGSTF) != 0 {
        NVME_TCP_DIGEST_LEN
    } else {
        0
    };
    let ddgst_len = if has_ddgst { NVME_TCP_DIGEST_LEN } else { 0 };
    let min_no_data = hlen + hdgst_len;
    if plen <= min_no_data {
        return Ok(Vec::new());
    }
    let pdo = if hdr.pdo == 0 {
        min_no_data
    } else {
        hdr.pdo as usize
    };
    if pdo < min_no_data || plen < pdo + ddgst_len {
        return Err(NvmeOfError::InvalidPduLength);
    }
    let data_end = plen - ddgst_len;
    let data_slice = &buf[pdo..data_end];
    if has_ddgst {
        let expected = read_u32_le(buf, data_end);
        let actual = crc32c(data_slice);
        if expected != actual {
            return Err(NvmeOfError::DigestMismatch);
        }
    }
    Ok(data_slice.to_vec())
}

// ---------------------------------------------------------------------------
// NVMe/TCP Connection & Queue State Machine
// ---------------------------------------------------------------------------

/// NVMe/TCP connection lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpConnState {
    Disconnected,
    IcReqSent,
    Connected,
    FabricsReady,
    Terminating,
    Closed,
}

/// In-flight NVMe/TCP command tracking entry.
#[derive(Debug, Clone)]
pub struct InFlightCmd {
    pub cid: u16,
    pub opcode: u8,
    pub write_payload: Vec<u8>,
    pub read_buffer: Vec<u8>,
    pub completed: bool,
    pub cqe: Option<NvmeOfCqe>,
}

/// NVMe/TCP Queue & Connection Engine.
pub struct NvmeTcpQueue {
    pub qid: u16,
    pub sq_size: u16,
    pub state: TcpConnState,
    pub hdgst_enabled: bool,
    pub ddgst_enabled: bool,
    pub hpda: u8,
    pub cpda: u8,
    pub maxh2cdata: u32,
    pub maxr2t: u32,
    next_cid: u16,
    in_flight: Vec<InFlightCmd>,
    tx_queue: VecDeque<Vec<u8>>,
}

impl NvmeTcpQueue {
    pub fn new(qid: u16, sq_size: u16, hdgst: bool, ddgst: bool) -> Self {
        Self {
            qid,
            sq_size,
            state: TcpConnState::Disconnected,
            hdgst_enabled: hdgst,
            ddgst_enabled: ddgst,
            hpda: 0,
            cpda: 0,
            maxh2cdata: 131_072,
            maxr2t: 16,
            next_cid: 1,
            in_flight: Vec::new(),
            tx_queue: VecDeque::new(),
        }
    }

    fn alloc_cid(&mut self) -> u16 {
        let cid = self.next_cid;
        self.next_cid = self.next_cid.wrapping_add(1);
        if self.next_cid == 0 {
            self.next_cid = 1;
        }
        cid
    }

    /// Initiate NVMe/TCP connection by queuing an `ICReq` PDU.
    pub fn initiate_connection(&mut self) -> Result<Vec<u8>, NvmeOfError> {
        if self.state != TcpConnState::Disconnected {
            return Err(NvmeOfError::InvalidState);
        }
        let mut digest = 0u8;
        if self.hdgst_enabled {
            digest |= 0x01;
        }
        if self.ddgst_enabled {
            digest |= 0x02;
        }
        let pdu = NvmeTcpPdu::ICReq(IcReqPdu {
            pfv: NVME_TCP_PFV_1_0,
            hpda: self.hpda,
            digest,
            maxr2t: self.maxr2t,
        });
        let bytes = encode_pdu(&pdu, false, false, 0)?;
        self.state = TcpConnState::IcReqSent;
        self.tx_queue.push_back(bytes.clone());
        Ok(bytes)
    }

    /// Submit a Fabrics Connect command capsule.
    pub fn submit_fabrics_connect(
        &mut self,
        hostid: &[u8; 16],
        cntlid: u16,
        subnqn: &[u8],
        hostnqn: &[u8],
    ) -> Result<u16, NvmeOfError> {
        if self.state != TcpConnState::Connected && self.state != TcpConnState::FabricsReady {
            return Err(NvmeOfError::InvalidState);
        }
        let cid = self.alloc_cid();
        let data = encode_connect_data(hostid, cntlid, subnqn, hostnqn);
        let sgl = SglDescriptor::offset_data_block(0, data.len() as u32);
        let sqe = NvmeOfSqe::fabrics_connect(cid, self.qid, self.sq_size, 0, sgl);
        let pdu = NvmeTcpPdu::CapsuleCmd { sqe, data };
        let bytes = encode_pdu(&pdu, self.hdgst_enabled, self.ddgst_enabled, self.cpda)?;
        self.in_flight.push(InFlightCmd {
            cid,
            opcode: NVME_FABRICS_OPCODE,
            write_payload: Vec::new(),
            read_buffer: Vec::new(),
            completed: false,
            cqe: None,
        });
        self.tx_queue.push_back(bytes);
        Ok(cid)
    }

    /// Submit an NVM Write command capsule (data sent via R2T -> H2CData).
    pub fn submit_write(
        &mut self,
        nsid: u32,
        slba: u64,
        nlb_zero_based: u16,
        payload: Vec<u8>,
    ) -> Result<u16, NvmeOfError> {
        if self.state != TcpConnState::FabricsReady {
            return Err(NvmeOfError::InvalidState);
        }
        let cid = self.alloc_cid();
        let sgl = SglDescriptor::transport_data_block(payload.len() as u32);
        let sqe = NvmeOfSqe::nvm_write(cid, nsid, slba, nlb_zero_based, sgl);
        let pdu = NvmeTcpPdu::CapsuleCmd {
            sqe,
            data: Vec::new(),
        };
        let bytes = encode_pdu(&pdu, self.hdgst_enabled, self.ddgst_enabled, self.cpda)?;
        self.in_flight.push(InFlightCmd {
            cid,
            opcode: NVM_CMD_WRITE,
            write_payload: payload,
            read_buffer: Vec::new(),
            completed: false,
            cqe: None,
        });
        self.tx_queue.push_back(bytes);
        Ok(cid)
    }

    /// Submit an NVM Read command capsule.
    pub fn submit_read(
        &mut self,
        nsid: u32,
        slba: u64,
        nlb_zero_based: u16,
        expected_bytes: u32,
    ) -> Result<u16, NvmeOfError> {
        if self.state != TcpConnState::FabricsReady {
            return Err(NvmeOfError::InvalidState);
        }
        let cid = self.alloc_cid();
        let sgl = SglDescriptor::transport_data_block(expected_bytes);
        let sqe = NvmeOfSqe::nvm_read(cid, nsid, slba, nlb_zero_based, sgl);
        let pdu = NvmeTcpPdu::CapsuleCmd {
            sqe,
            data: Vec::new(),
        };
        let bytes = encode_pdu(&pdu, self.hdgst_enabled, self.ddgst_enabled, self.cpda)?;
        self.in_flight.push(InFlightCmd {
            cid,
            opcode: NVM_CMD_READ,
            write_payload: Vec::new(),
            read_buffer: vec![0u8; expected_bytes as usize],
            completed: false,
            cqe: None,
        });
        self.tx_queue.push_back(bytes);
        Ok(cid)
    }

    /// Pop the next encoded PDU ready for TCP transmission.
    pub fn pop_tx_pdu(&mut self) -> Option<Vec<u8>> {
        self.tx_queue.pop_front()
    }

    /// Process an incoming wire PDU from the controller.
    pub fn process_rx_pdu(&mut self, raw: &[u8]) -> Result<(), NvmeOfError> {
        let (pdu, _) = decode_pdu(raw)?;
        match pdu {
            NvmeTcpPdu::ICResp(resp) => {
                if self.state != TcpConnState::IcReqSent {
                    return Err(NvmeOfError::InvalidState);
                }
                if resp.pfv != NVME_TCP_PFV_1_0 {
                    return Err(NvmeOfError::ProtocolError);
                }
                self.cpda = resp.cpda;
                self.maxh2cdata = resp.maxh2cdata.max(4096);
                self.hdgst_enabled = (resp.digest & 0x01) != 0;
                self.ddgst_enabled = (resp.digest & 0x02) != 0;
                self.state = TcpConnState::Connected;
                Ok(())
            }
            NvmeTcpPdu::R2T(r2t) => {
                let cmd = self
                    .in_flight
                    .iter()
                    .find(|c| c.cid == r2t.cccid)
                    .ok_or(NvmeOfError::CommandNotFound)?;
                let start = r2t.r2to as usize;
                let end = start.saturating_add(r2t.r2tl as usize);
                if end > cmd.write_payload.len() {
                    return Err(NvmeOfError::BoundsExceeded);
                }
                let chunk_max = self.maxh2cdata as usize;
                let mut offset = start;
                while offset < end {
                    let chunk_end = (offset + chunk_max).min(end);
                    let is_last = chunk_end == end;
                    let h2c = NvmeTcpPdu::H2CData(H2CDataPdu {
                        flags: if is_last { pdu_flags::DATA_LAST } else { 0 },
                        cccid: r2t.cccid,
                        ttag: r2t.ttag,
                        datao: offset as u32,
                        datal: (chunk_end - offset) as u32,
                        data: cmd.write_payload[offset..chunk_end].to_vec(),
                    });
                    let encoded =
                        encode_pdu(&h2c, self.hdgst_enabled, self.ddgst_enabled, self.cpda)?;
                    self.tx_queue.push_back(encoded);
                    offset = chunk_end;
                }
                Ok(())
            }
            NvmeTcpPdu::C2HData(c2h) => {
                let cmd = self
                    .in_flight
                    .iter_mut()
                    .find(|c| c.cid == c2h.cccid)
                    .ok_or(NvmeOfError::CommandNotFound)?;
                let start = c2h.datao as usize;
                let end = start.saturating_add(c2h.data.len());
                if end > cmd.read_buffer.len() {
                    return Err(NvmeOfError::BoundsExceeded);
                }
                cmd.read_buffer[start..end].copy_from_slice(&c2h.data);
                if (c2h.flags & pdu_flags::DATA_SUCCESS) != 0 {
                    cmd.completed = true;
                    cmd.cqe = Some(NvmeOfCqe::success(c2h.cccid, self.qid, 0, 0));
                }
                Ok(())
            }
            NvmeTcpPdu::CapsuleResp { cqe } => {
                let cmd = self
                    .in_flight
                    .iter_mut()
                    .find(|c| c.cid == cqe.cid)
                    .ok_or(NvmeOfError::CommandNotFound)?;
                cmd.completed = true;
                cmd.cqe = Some(cqe);
                if cmd.opcode == NVME_FABRICS_OPCODE && cqe.is_success() {
                    self.state = TcpConnState::FabricsReady;
                }
                Ok(())
            }
            NvmeTcpPdu::TermReq(_) => {
                self.state = TcpConnState::Closed;
                Ok(())
            }
            _ => Err(NvmeOfError::InvalidPduType),
        }
    }

    /// Take a completed command by `cid`.
    pub fn take_completed(&mut self, cid: u16) -> Option<InFlightCmd> {
        let pos = self
            .in_flight
            .iter()
            .position(|c| c.cid == cid && c.completed)?;
        Some(self.in_flight.remove(pos))
    }
}

// ---------------------------------------------------------------------------
// RDMA Queue-Pair Engine (PD, MR, SGE, WR, CQ, QP)
// ---------------------------------------------------------------------------

/// Memory Region access permission flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MrAccessFlags(pub u32);

impl MrAccessFlags {
    pub const LOCAL_WRITE: Self = Self(1 << 0);
    pub const REMOTE_WRITE: Self = Self(1 << 1);
    pub const REMOTE_READ: Self = Self(1 << 2);
    pub const REMOTE_ATOMIC: Self = Self(1 << 3);
    pub const BIND_MW: Self = Self(1 << 4);

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Protection Domain handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtectionDomain {
    pub pd_id: u32,
}

/// Registered RDMA Memory Region with backing buffer for kernel/loopback execution.
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub lkey: u32,
    pub rkey: u32,
    pub pd_id: u32,
    pub base_addr: u64,
    pub access: MrAccessFlags,
    pub buffer: Vec<u8>,
}

impl MemoryRegion {
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    fn slice_range(&self, addr: u64, len: usize) -> Result<core::ops::Range<usize>, NvmeOfError> {
        if addr < self.base_addr {
            return Err(NvmeOfError::BoundsExceeded);
        }
        let off = (addr - self.base_addr) as usize;
        let end = off.checked_add(len).ok_or(NvmeOfError::BoundsExceeded)?;
        if end > self.buffer.len() {
            return Err(NvmeOfError::BoundsExceeded);
        }
        Ok(off..end)
    }
}

/// 16-byte RDMA Scatter-Gather Element (`ibv_sge`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RdmaSge {
    pub addr: u64,
    pub length: u32,
    pub lkey: u32,
}

const _: () = assert!(size_of::<RdmaSge>() == 16);

/// RDMA Send Work Request Opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdmaWrOpcode {
    Send,
    SendWithImm,
    RdmaWrite,
    RdmaWriteWithImm,
    RdmaRead,
    AtomicCmpSwp,
    AtomicFetchAdd,
}

/// RDMA Send Work Request Flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendFlags(pub u32);

impl SendFlags {
    pub const NONE: Self = Self(0);
    pub const FENCE: Self = Self(1 << 0);
    pub const SIGNALED: Self = Self(1 << 1);
    pub const SOLICITED: Self = Self(1 << 2);
    pub const INLINE: Self = Self(1 << 3);

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

/// RDMA Send Work Request (`ibv_send_wr`).
#[derive(Debug, Clone)]
pub struct SendWr {
    pub wr_id: u64,
    pub opcode: RdmaWrOpcode,
    pub send_flags: SendFlags,
    pub sge_list: Vec<RdmaSge>,
    pub remote_addr: u64,
    pub rkey: u32,
    pub imm_data: u32,
    pub compare_add: u64,
    pub swap: u64,
}

/// RDMA Receive Work Request (`ibv_recv_wr`).
#[derive(Debug, Clone)]
pub struct RecvWr {
    pub wr_id: u64,
    pub sge_list: Vec<RdmaSge>,
}

/// RDMA Work Completion Status (`ibv_wc_status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WcStatus {
    Success,
    LocLenErr,
    LocQpOpErr,
    LocProtErr,
    WrFlushErr,
    RemAccessErr,
    RemOpErr,
    RnrRetryExcErr,
    TransportRetryExcErr,
}

/// RDMA Work Completion Opcode (`ibv_wc_opcode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WcOpcode {
    Send,
    RdmaWrite,
    RdmaRead,
    CompSwap,
    FetchAdd,
    Recv,
    RecvRdmaWithImm,
}

/// RDMA Work Completion (`ibv_wc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkCompletion {
    pub wr_id: u64,
    pub status: WcStatus,
    pub opcode: WcOpcode,
    pub vendor_err: u32,
    pub byte_len: u32,
    pub imm_data: u32,
    pub qp_num: u32,
    pub src_qp: u32,
    pub has_imm: bool,
}

/// RDMA Completion Queue.
#[derive(Debug, Clone)]
pub struct CompletionQueue {
    pub cq_id: u32,
    pub capacity: usize,
    pub armed_solicited: bool,
    pub armed_all: bool,
    entries: VecDeque<WorkCompletion>,
}

impl CompletionQueue {
    pub fn new(cq_id: u32, capacity: usize) -> Self {
        Self {
            cq_id,
            capacity: capacity.max(1),
            armed_solicited: false,
            armed_all: false,
            entries: VecDeque::new(),
        }
    }

    pub fn push_wc(&mut self, wc: WorkCompletion) -> Result<(), NvmeOfError> {
        if self.entries.len() >= self.capacity {
            return Err(NvmeOfError::QueueFull);
        }
        self.entries.push_back(wc);
        Ok(())
    }

    pub fn poll_cq(&mut self) -> Option<WorkCompletion> {
        self.entries.pop_front()
    }

    pub fn poll_batch(&mut self, max: usize) -> Vec<WorkCompletion> {
        let count = max.min(self.entries.len());
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(wc) = self.entries.pop_front() {
                out.push(wc);
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// RDMA Queue Pair Transport Type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpType {
    Rc,
    Uc,
    Ud,
}

/// IBTA Queue Pair State.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpState {
    Reset,
    Init,
    Rtr,
    Rts,
    Sqd,
    Sqe,
    Err,
}

/// Attributes for `modify_qp`.
#[derive(Debug, Clone, Copy, Default)]
pub struct QpModifyAttr {
    pub target_state: Option<QpState>,
    pub dest_qp_num: Option<u32>,
    pub sq_psn: Option<u32>,
    pub rq_psn: Option<u32>,
    pub max_rd_atomic: Option<u8>,
}

/// RDMA Queue Pair.
#[derive(Debug, Clone)]
pub struct QueuePair {
    pub qp_num: u32,
    pub qp_type: QpType,
    pub state: QpState,
    pub pd_id: u32,
    pub send_cq_id: u32,
    pub recv_cq_id: u32,
    pub dest_qp_num: Option<u32>,
    pub sq_psn: u32,
    pub rq_psn: u32,
    pub max_send_wr: usize,
    pub max_recv_wr: usize,
    pub sq: VecDeque<SendWr>,
    pub rq: VecDeque<RecvWr>,
}

impl QueuePair {
    pub fn new(
        qp_num: u32,
        qp_type: QpType,
        pd_id: u32,
        send_cq_id: u32,
        recv_cq_id: u32,
        max_send_wr: usize,
        max_recv_wr: usize,
    ) -> Self {
        Self {
            qp_num,
            qp_type,
            state: QpState::Reset,
            pd_id,
            send_cq_id,
            recv_cq_id,
            dest_qp_num: None,
            sq_psn: 0,
            rq_psn: 0,
            max_send_wr: max_send_wr.max(1),
            max_recv_wr: max_recv_wr.max(1),
            sq: VecDeque::new(),
            rq: VecDeque::new(),
        }
    }

    /// Validate and apply an IBTA QP state transition.
    pub fn modify(&mut self, attr: QpModifyAttr) -> Result<(), NvmeOfError> {
        if let Some(next) = attr.target_state {
            let valid = matches!(
                (self.state, next),
                (QpState::Reset, QpState::Init)
                    | (QpState::Init, QpState::Rtr)
                    | (QpState::Rtr, QpState::Rts)
                    | (QpState::Rts, QpState::Sqd)
                    | (QpState::Sqd, QpState::Rts)
                    | (_, QpState::Err)
                    | (_, QpState::Reset)
            );
            if !valid && self.state != next {
                return Err(NvmeOfError::InvalidTransition);
            }
            self.state = next;
            if next == QpState::Reset {
                self.sq.clear();
                self.rq.clear();
                self.dest_qp_num = None;
            }
        }
        if let Some(dqpn) = attr.dest_qp_num {
            self.dest_qp_num = Some(dqpn);
        }
        if let Some(psn) = attr.sq_psn {
            self.sq_psn = psn;
        }
        if let Some(psn) = attr.rq_psn {
            self.rq_psn = psn;
        }
        Ok(())
    }

    pub fn post_send(&mut self, wr: SendWr) -> Result<(), NvmeOfError> {
        if self.state != QpState::Rts {
            return Err(NvmeOfError::InvalidState);
        }
        if self.sq.len() >= self.max_send_wr {
            return Err(NvmeOfError::QueueFull);
        }
        self.sq.push_back(wr);
        Ok(())
    }

    pub fn post_recv(&mut self, wr: RecvWr) -> Result<(), NvmeOfError> {
        if !matches!(self.state, QpState::Init | QpState::Rtr | QpState::Rts) {
            return Err(NvmeOfError::InvalidState);
        }
        if self.rq.len() >= self.max_recv_wr {
            return Err(NvmeOfError::QueueFull);
        }
        self.rq.push_back(wr);
        Ok(())
    }
}

/// RDMA Engine managing PDs, MRs, CQs, QPs, and loopback work-request execution.
pub struct RdmaEngine {
    next_id: u32,
    pds: Vec<ProtectionDomain>,
    mrs: Vec<MemoryRegion>,
    cqs: Vec<CompletionQueue>,
    qps: Vec<QueuePair>,
}

impl RdmaEngine {
    pub const fn new() -> Self {
        Self {
            next_id: 1,
            pds: Vec::new(),
            mrs: Vec::new(),
            cqs: Vec::new(),
            qps: Vec::new(),
        }
    }

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        if self.next_id == 0 {
            self.next_id = 1;
        }
        id
    }

    pub fn alloc_pd(&mut self) -> u32 {
        let pd_id = self.alloc_id();
        self.pds.push(ProtectionDomain { pd_id });
        pd_id
    }

    pub fn reg_mr(
        &mut self,
        pd_id: u32,
        base_addr: u64,
        size: usize,
        access: MrAccessFlags,
    ) -> Result<(u32, u32), NvmeOfError> {
        if !self.pds.iter().any(|p| p.pd_id == pd_id) {
            return Err(NvmeOfError::InvalidProtectionDomain);
        }
        let lkey = 0x1000_0000 | self.alloc_id();
        let rkey = 0x2000_0000 | self.alloc_id();
        self.mrs.push(MemoryRegion {
            lkey,
            rkey,
            pd_id,
            base_addr,
            access,
            buffer: vec![0u8; size],
        });
        Ok((lkey, rkey))
    }

    pub fn mr_write_local(&mut self, lkey: u32, addr: u64, data: &[u8]) -> Result<(), NvmeOfError> {
        let mr = self
            .mrs
            .iter_mut()
            .find(|m| m.lkey == lkey)
            .ok_or(NvmeOfError::InvalidMemoryRegion)?;
        let range = mr.slice_range(addr, data.len())?;
        mr.buffer[range].copy_from_slice(data);
        Ok(())
    }

    pub fn mr_read_local(&self, lkey: u32, addr: u64, len: usize) -> Result<Vec<u8>, NvmeOfError> {
        let mr = self
            .mrs
            .iter()
            .find(|m| m.lkey == lkey)
            .ok_or(NvmeOfError::InvalidMemoryRegion)?;
        let range = mr.slice_range(addr, len)?;
        Ok(mr.buffer[range].to_vec())
    }

    pub fn create_cq(&mut self, capacity: usize) -> u32 {
        let cq_id = self.alloc_id();
        self.cqs.push(CompletionQueue::new(cq_id, capacity));
        cq_id
    }

    pub fn create_qp(
        &mut self,
        qp_type: QpType,
        pd_id: u32,
        send_cq_id: u32,
        recv_cq_id: u32,
        max_send_wr: usize,
        max_recv_wr: usize,
    ) -> Result<u32, NvmeOfError> {
        if !self.pds.iter().any(|p| p.pd_id == pd_id) {
            return Err(NvmeOfError::InvalidProtectionDomain);
        }
        if !self.cqs.iter().any(|c| c.cq_id == send_cq_id)
            || !self.cqs.iter().any(|c| c.cq_id == recv_cq_id)
        {
            return Err(NvmeOfError::InvalidCompletionQueue);
        }
        let qp_num = self.alloc_id();
        self.qps.push(QueuePair::new(
            qp_num,
            qp_type,
            pd_id,
            send_cq_id,
            recv_cq_id,
            max_send_wr,
            max_recv_wr,
        ));
        Ok(qp_num)
    }

    pub fn modify_qp(&mut self, qp_num: u32, attr: QpModifyAttr) -> Result<(), NvmeOfError> {
        let qp = self
            .qps
            .iter_mut()
            .find(|q| q.qp_num == qp_num)
            .ok_or(NvmeOfError::InvalidQueuePair)?;
        qp.modify(attr)
    }

    /// Connect two RC Queue Pairs in loopback (`Reset -> Init -> RTR -> RTS`).
    pub fn connect_loopback(&mut self, qp_a: u32, qp_b: u32) -> Result<(), NvmeOfError> {
        for &(local, remote) in &[(qp_a, qp_b), (qp_b, qp_a)] {
            self.modify_qp(
                local,
                QpModifyAttr {
                    target_state: Some(QpState::Init),
                    ..Default::default()
                },
            )?;
            self.modify_qp(
                local,
                QpModifyAttr {
                    target_state: Some(QpState::Rtr),
                    dest_qp_num: Some(remote),
                    rq_psn: Some(1),
                    ..Default::default()
                },
            )?;
            self.modify_qp(
                local,
                QpModifyAttr {
                    target_state: Some(QpState::Rts),
                    sq_psn: Some(1),
                    ..Default::default()
                },
            )?;
        }
        Ok(())
    }

    pub fn post_send(&mut self, qp_num: u32, wr: SendWr) -> Result<(), NvmeOfError> {
        let qp = self
            .qps
            .iter_mut()
            .find(|q| q.qp_num == qp_num)
            .ok_or(NvmeOfError::InvalidQueuePair)?;
        qp.post_send(wr)
    }

    pub fn post_recv(&mut self, qp_num: u32, wr: RecvWr) -> Result<(), NvmeOfError> {
        let qp = self
            .qps
            .iter_mut()
            .find(|q| q.qp_num == qp_num)
            .ok_or(NvmeOfError::InvalidQueuePair)?;
        qp.post_recv(wr)
    }

    pub fn poll_cq(&mut self, cq_id: u32) -> Result<Option<WorkCompletion>, NvmeOfError> {
        let cq = self
            .cqs
            .iter_mut()
            .find(|c| c.cq_id == cq_id)
            .ok_or(NvmeOfError::InvalidCompletionQueue)?;
        Ok(cq.poll_cq())
    }

    fn gather_local(&self, pd_id: u32, sges: &[RdmaSge]) -> Result<Vec<u8>, WcStatus> {
        let mut out = Vec::new();
        for sge in sges {
            let mr = self
                .mrs
                .iter()
                .find(|m| m.lkey == sge.lkey && m.pd_id == pd_id)
                .ok_or(WcStatus::LocProtErr)?;
            let range = mr
                .slice_range(sge.addr, sge.length as usize)
                .map_err(|_| WcStatus::LocLenErr)?;
            out.extend_from_slice(&mr.buffer[range]);
        }
        Ok(out)
    }

    fn scatter_local(
        &mut self,
        pd_id: u32,
        sges: &[RdmaSge],
        data: &[u8],
    ) -> Result<u32, WcStatus> {
        let mut offset = 0usize;
        for sge in sges {
            if offset >= data.len() {
                break;
            }
            let mr = self
                .mrs
                .iter_mut()
                .find(|m| m.lkey == sge.lkey && m.pd_id == pd_id)
                .ok_or(WcStatus::LocProtErr)?;
            if !mr.access.contains(MrAccessFlags::LOCAL_WRITE) {
                return Err(WcStatus::LocProtErr);
            }
            let take = (sge.length as usize).min(data.len() - offset);
            let range = mr
                .slice_range(sge.addr, take)
                .map_err(|_| WcStatus::LocLenErr)?;
            mr.buffer[range].copy_from_slice(&data[offset..offset + take]);
            offset += take;
        }
        if offset < data.len() {
            return Err(WcStatus::LocLenErr);
        }
        Ok(offset as u32)
    }

    fn push_cq(&mut self, cq_id: u32, wc: WorkCompletion) -> Result<(), NvmeOfError> {
        let cq = self
            .cqs
            .iter_mut()
            .find(|c| c.cq_id == cq_id)
            .ok_or(NvmeOfError::InvalidCompletionQueue)?;
        cq.push_wc(wc)
    }

    /// Execute all pending Send Work Requests on `qp_num` against its connected peer QP.
    pub fn process_sq(&mut self, qp_num: u32) -> Result<usize, NvmeOfError> {
        let mut processed = 0usize;
        loop {
            let (pd_id, send_cq_id, dest_qp_num, wr) = {
                let qp = self
                    .qps
                    .iter_mut()
                    .find(|q| q.qp_num == qp_num)
                    .ok_or(NvmeOfError::InvalidQueuePair)?;
                let Some(wr) = qp.sq.pop_front() else {
                    break;
                };
                (qp.pd_id, qp.send_cq_id, qp.dest_qp_num, wr)
            };

            let Some(remote_qpn) = dest_qp_num else {
                self.push_cq(
                    send_cq_id,
                    WorkCompletion {
                        wr_id: wr.wr_id,
                        status: WcStatus::LocQpOpErr,
                        opcode: WcOpcode::Send,
                        vendor_err: 0,
                        byte_len: 0,
                        imm_data: 0,
                        qp_num,
                        src_qp: qp_num,
                        has_imm: false,
                    },
                )?;
                processed += 1;
                continue;
            };

            let (remote_pd_id, remote_recv_cq_id) = {
                let rqp = self
                    .qps
                    .iter()
                    .find(|q| q.qp_num == remote_qpn)
                    .ok_or(NvmeOfError::InvalidQueuePair)?;
                (rqp.pd_id, rqp.recv_cq_id)
            };

            match wr.opcode {
                RdmaWrOpcode::Send | RdmaWrOpcode::SendWithImm => {
                    let has_imm = wr.opcode == RdmaWrOpcode::SendWithImm;
                    let payload = match self.gather_local(pd_id, &wr.sge_list) {
                        Ok(p) => p,
                        Err(status) => {
                            self.push_cq(
                                send_cq_id,
                                WorkCompletion {
                                    wr_id: wr.wr_id,
                                    status,
                                    opcode: WcOpcode::Send,
                                    vendor_err: 0,
                                    byte_len: 0,
                                    imm_data: 0,
                                    qp_num,
                                    src_qp: qp_num,
                                    has_imm: false,
                                },
                            )?;
                            processed += 1;
                            continue;
                        }
                    };
                    let recv_wr = {
                        let rqp = self
                            .qps
                            .iter_mut()
                            .find(|q| q.qp_num == remote_qpn)
                            .ok_or(NvmeOfError::InvalidQueuePair)?;
                        rqp.rq.pop_front()
                    };
                    let Some(rwr) = recv_wr else {
                        self.push_cq(
                            send_cq_id,
                            WorkCompletion {
                                wr_id: wr.wr_id,
                                status: WcStatus::RnrRetryExcErr,
                                opcode: WcOpcode::Send,
                                vendor_err: 0,
                                byte_len: 0,
                                imm_data: 0,
                                qp_num,
                                src_qp: qp_num,
                                has_imm: false,
                            },
                        )?;
                        processed += 1;
                        continue;
                    };
                    let recv_res = self.scatter_local(remote_pd_id, &rwr.sge_list, &payload);
                    match recv_res {
                        Ok(byte_len) => {
                            self.push_cq(
                                remote_recv_cq_id,
                                WorkCompletion {
                                    wr_id: rwr.wr_id,
                                    status: WcStatus::Success,
                                    opcode: WcOpcode::Recv,
                                    vendor_err: 0,
                                    byte_len,
                                    imm_data: wr.imm_data,
                                    qp_num: remote_qpn,
                                    src_qp: qp_num,
                                    has_imm,
                                },
                            )?;
                            if wr.send_flags.contains(SendFlags::SIGNALED) {
                                self.push_cq(
                                    send_cq_id,
                                    WorkCompletion {
                                        wr_id: wr.wr_id,
                                        status: WcStatus::Success,
                                        opcode: WcOpcode::Send,
                                        vendor_err: 0,
                                        byte_len,
                                        imm_data: 0,
                                        qp_num,
                                        src_qp: qp_num,
                                        has_imm: false,
                                    },
                                )?;
                            }
                        }
                        Err(status) => {
                            self.push_cq(
                                remote_recv_cq_id,
                                WorkCompletion {
                                    wr_id: rwr.wr_id,
                                    status,
                                    opcode: WcOpcode::Recv,
                                    vendor_err: 0,
                                    byte_len: 0,
                                    imm_data: 0,
                                    qp_num: remote_qpn,
                                    src_qp: qp_num,
                                    has_imm: false,
                                },
                            )?;
                        }
                    }
                }
                RdmaWrOpcode::RdmaWrite | RdmaWrOpcode::RdmaWriteWithImm => {
                    let payload = match self.gather_local(pd_id, &wr.sge_list) {
                        Ok(p) => p,
                        Err(status) => {
                            self.push_cq(
                                send_cq_id,
                                WorkCompletion {
                                    wr_id: wr.wr_id,
                                    status,
                                    opcode: WcOpcode::RdmaWrite,
                                    vendor_err: 0,
                                    byte_len: 0,
                                    imm_data: 0,
                                    qp_num,
                                    src_qp: qp_num,
                                    has_imm: false,
                                },
                            )?;
                            processed += 1;
                            continue;
                        }
                    };
                    let write_status = {
                        match self
                            .mrs
                            .iter_mut()
                            .find(|m| m.rkey == wr.rkey && m.pd_id == remote_pd_id)
                        {
                            None => Err(WcStatus::RemAccessErr),
                            Some(rmr) => {
                                if !rmr.access.contains(MrAccessFlags::REMOTE_WRITE) {
                                    Err(WcStatus::RemAccessErr)
                                } else {
                                    match rmr.slice_range(wr.remote_addr, payload.len()) {
                                        Ok(range) => {
                                            rmr.buffer[range].copy_from_slice(&payload);
                                            Ok(payload.len() as u32)
                                        }
                                        Err(_) => Err(WcStatus::RemAccessErr),
                                    }
                                }
                            }
                        }
                    };
                    match write_status {
                        Ok(byte_len) => {
                            if wr.opcode == RdmaWrOpcode::RdmaWriteWithImm {
                                let recv_wr = {
                                    let rqp = self
                                        .qps
                                        .iter_mut()
                                        .find(|q| q.qp_num == remote_qpn)
                                        .ok_or(NvmeOfError::InvalidQueuePair)?;
                                    rqp.rq.pop_front()
                                };
                                if let Some(rwr) = recv_wr {
                                    self.push_cq(
                                        remote_recv_cq_id,
                                        WorkCompletion {
                                            wr_id: rwr.wr_id,
                                            status: WcStatus::Success,
                                            opcode: WcOpcode::RecvRdmaWithImm,
                                            vendor_err: 0,
                                            byte_len,
                                            imm_data: wr.imm_data,
                                            qp_num: remote_qpn,
                                            src_qp: qp_num,
                                            has_imm: true,
                                        },
                                    )?;
                                }
                            }
                            if wr.send_flags.contains(SendFlags::SIGNALED) {
                                self.push_cq(
                                    send_cq_id,
                                    WorkCompletion {
                                        wr_id: wr.wr_id,
                                        status: WcStatus::Success,
                                        opcode: WcOpcode::RdmaWrite,
                                        vendor_err: 0,
                                        byte_len,
                                        imm_data: 0,
                                        qp_num,
                                        src_qp: qp_num,
                                        has_imm: false,
                                    },
                                )?;
                            }
                        }
                        Err(status) => {
                            self.push_cq(
                                send_cq_id,
                                WorkCompletion {
                                    wr_id: wr.wr_id,
                                    status,
                                    opcode: WcOpcode::RdmaWrite,
                                    vendor_err: 0,
                                    byte_len: 0,
                                    imm_data: 0,
                                    qp_num,
                                    src_qp: qp_num,
                                    has_imm: false,
                                },
                            )?;
                        }
                    }
                }
                RdmaWrOpcode::RdmaRead => {
                    let total_len: usize = wr.sge_list.iter().map(|s| s.length as usize).sum();
                    let remote_data = {
                        match self
                            .mrs
                            .iter()
                            .find(|m| m.rkey == wr.rkey && m.pd_id == remote_pd_id)
                        {
                            None => Err(WcStatus::RemAccessErr),
                            Some(rmr) => {
                                if !rmr.access.contains(MrAccessFlags::REMOTE_READ) {
                                    Err(WcStatus::RemAccessErr)
                                } else {
                                    match rmr.slice_range(wr.remote_addr, total_len) {
                                        Ok(range) => Ok(rmr.buffer[range].to_vec()),
                                        Err(_) => Err(WcStatus::RemAccessErr),
                                    }
                                }
                            }
                        }
                    };
                    match remote_data {
                        Ok(data) => {
                            let res = self.scatter_local(pd_id, &wr.sge_list, &data);
                            let (status, byte_len) = match res {
                                Ok(n) => (WcStatus::Success, n),
                                Err(st) => (st, 0),
                            };
                            if wr.send_flags.contains(SendFlags::SIGNALED)
                                || status != WcStatus::Success
                            {
                                self.push_cq(
                                    send_cq_id,
                                    WorkCompletion {
                                        wr_id: wr.wr_id,
                                        status,
                                        opcode: WcOpcode::RdmaRead,
                                        vendor_err: 0,
                                        byte_len,
                                        imm_data: 0,
                                        qp_num,
                                        src_qp: qp_num,
                                        has_imm: false,
                                    },
                                )?;
                            }
                        }
                        Err(status) => {
                            self.push_cq(
                                send_cq_id,
                                WorkCompletion {
                                    wr_id: wr.wr_id,
                                    status,
                                    opcode: WcOpcode::RdmaRead,
                                    vendor_err: 0,
                                    byte_len: 0,
                                    imm_data: 0,
                                    qp_num,
                                    src_qp: qp_num,
                                    has_imm: false,
                                },
                            )?;
                        }
                    }
                }
                RdmaWrOpcode::AtomicCmpSwp | RdmaWrOpcode::AtomicFetchAdd => {
                    let is_cas = wr.opcode == RdmaWrOpcode::AtomicCmpSwp;
                    let wc_op = if is_cas {
                        WcOpcode::CompSwap
                    } else {
                        WcOpcode::FetchAdd
                    };
                    let atomic_res = {
                        match self
                            .mrs
                            .iter_mut()
                            .find(|m| m.rkey == wr.rkey && m.pd_id == remote_pd_id)
                        {
                            None => Err(WcStatus::RemAccessErr),
                            Some(rmr) => {
                                if !rmr.access.contains(MrAccessFlags::REMOTE_ATOMIC) {
                                    Err(WcStatus::RemAccessErr)
                                } else {
                                    match rmr.slice_range(wr.remote_addr, 8) {
                                        Ok(range) => {
                                            let old = read_u64_le(&rmr.buffer, range.start);
                                            let new_val = if is_cas {
                                                if old == wr.compare_add {
                                                    wr.swap
                                                } else {
                                                    old
                                                }
                                            } else {
                                                old.wrapping_add(wr.compare_add)
                                            };
                                            write_u64_le(&mut rmr.buffer, range.start, new_val);
                                            Ok(old.to_le_bytes())
                                        }
                                        Err(_) => Err(WcStatus::RemAccessErr),
                                    }
                                }
                            }
                        }
                    };
                    match atomic_res {
                        Ok(old_bytes) => {
                            let res = self.scatter_local(pd_id, &wr.sge_list, &old_bytes);
                            let (status, byte_len) = match res {
                                Ok(n) => (WcStatus::Success, n),
                                Err(st) => (st, 0),
                            };
                            if wr.send_flags.contains(SendFlags::SIGNALED)
                                || status != WcStatus::Success
                            {
                                self.push_cq(
                                    send_cq_id,
                                    WorkCompletion {
                                        wr_id: wr.wr_id,
                                        status,
                                        opcode: wc_op,
                                        vendor_err: 0,
                                        byte_len,
                                        imm_data: 0,
                                        qp_num,
                                        src_qp: qp_num,
                                        has_imm: false,
                                    },
                                )?;
                            }
                        }
                        Err(status) => {
                            self.push_cq(
                                send_cq_id,
                                WorkCompletion {
                                    wr_id: wr.wr_id,
                                    status,
                                    opcode: wc_op,
                                    vendor_err: 0,
                                    byte_len: 0,
                                    imm_data: 0,
                                    qp_num,
                                    src_qp: qp_num,
                                    has_imm: false,
                                },
                            )?;
                        }
                    }
                }
            }
            processed += 1;
        }
        Ok(processed)
    }
}

// ---------------------------------------------------------------------------
// NVMe-oF over RDMA Queue Binding
// ---------------------------------------------------------------------------

/// Binds NVMe-oF capsule submission/completion onto an RDMA Queue Pair.
pub struct NvmeRdmaQueue {
    pub qid: u16,
    pub qp_num: u32,
    pub sq_mr_lkey: u32,
    pub sq_mr_addr: u64,
    pub cq_mr_lkey: u32,
    pub cq_mr_addr: u64,
    next_cid: u16,
}

impl NvmeRdmaQueue {
    pub const fn new(
        qid: u16,
        qp_num: u32,
        sq_mr_lkey: u32,
        sq_mr_addr: u64,
        cq_mr_lkey: u32,
        cq_mr_addr: u64,
    ) -> Self {
        Self {
            qid,
            qp_num,
            sq_mr_lkey,
            sq_mr_addr,
            cq_mr_lkey,
            cq_mr_addr,
            next_cid: 1,
        }
    }

    /// Post a 64-byte NVMe-oF SQE capsule over the RDMA Queue Pair.
    pub fn post_capsule_cmd(
        &mut self,
        engine: &mut RdmaEngine,
        wr_id: u64,
        sqe: &NvmeOfSqe,
    ) -> Result<(), NvmeOfError> {
        let mut raw = [0u8; 64];
        sqe.to_bytes(&mut raw)?;
        engine.mr_write_local(self.sq_mr_lkey, self.sq_mr_addr, &raw)?;
        engine.post_send(
            self.qp_num,
            SendWr {
                wr_id,
                opcode: RdmaWrOpcode::Send,
                send_flags: SendFlags::SIGNALED,
                sge_list: vec![RdmaSge {
                    addr: self.sq_mr_addr,
                    length: 64,
                    lkey: self.sq_mr_lkey,
                }],
                remote_addr: 0,
                rkey: 0,
                imm_data: 0,
                compare_add: 0,
                swap: 0,
            },
        )
    }

    /// Post a 16-byte receive buffer for an incoming NVMe-oF CQE capsule.
    pub fn post_recv_cqe(&self, engine: &mut RdmaEngine, wr_id: u64) -> Result<(), NvmeOfError> {
        engine.post_recv(
            self.qp_num,
            RecvWr {
                wr_id,
                sge_list: vec![RdmaSge {
                    addr: self.cq_mr_addr,
                    length: 16,
                    lkey: self.cq_mr_lkey,
                }],
            },
        )
    }

    /// Read the received 16-byte CQE from the local CQ Memory Region.
    pub fn read_cqe(&self, engine: &RdmaEngine) -> Result<NvmeOfCqe, NvmeOfError> {
        let bytes = engine.mr_read_local(self.cq_mr_lkey, self.cq_mr_addr, 16)?;
        NvmeOfCqe::from_bytes(&bytes)
    }
}

/// Global kernel NVMe-oF RDMA engine instance.
pub static RDMA_ENGINE: SpinLock<RdmaEngine> = SpinLock::new(RdmaEngine::new());

// ---------------------------------------------------------------------------
// Self-Test
// ---------------------------------------------------------------------------

/// Run comprehensive self-tests for NVMe/TCP PDU framing, NVMe-oF capsules,
/// RDMA Queue-Pair Engine, and NVMe-oF over RDMA binding.
pub fn self_test() -> Result<(), &'static str> {
    // 1. CRC32C Castagnoli standard test vector: "123456789" -> 0xE3069283
    if crc32c(b"123456789") != 0xE306_9283 {
        return Err("nvme_of: crc32c test vector mismatch");
    }

    // 2. NVMe/TCP PDU encode/decode round-trip with HDGST + DDGST
    let test_data = vec![0xA5u8; 64];
    let sgl = SglDescriptor::offset_data_block(0, test_data.len() as u32);
    let sqe = NvmeOfSqe::nvm_write(7, 1, 0x100, 0, sgl);
    let cap_pdu = NvmeTcpPdu::CapsuleCmd {
        sqe,
        data: test_data.clone(),
    };
    let mut wire = encode_pdu(&cap_pdu, true, true, 1).map_err(|_| "nvme_of: encode failed")?;
    let (decoded, consumed) = decode_pdu(&wire).map_err(|_| "nvme_of: decode failed")?;
    if consumed != wire.len() {
        return Err("nvme_of: consumed length mismatch");
    }
    match decoded {
        NvmeTcpPdu::CapsuleCmd { sqe: d_sqe, data } => {
            if d_sqe.cid() != 7 || data != test_data {
                return Err("nvme_of: capsule cmd round-trip mismatch");
            }
        }
        _ => return Err("nvme_of: unexpected decoded PDU type"),
    }

    // Corrupt a data byte and verify DigestMismatch
    let last_data_idx = wire.len() - NVME_TCP_DIGEST_LEN - 1;
    wire[last_data_idx] ^= 0xFF;
    if decode_pdu(&wire) != Err(NvmeOfError::DigestMismatch) {
        return Err("nvme_of: failed to detect corrupted data digest");
    }

    // 3. NVMe/TCP Connection & Command State Machine (ICReq -> ICResp -> Connect -> Write/R2T -> Read/C2H)
    let mut tcp_q = NvmeTcpQueue::new(0, 64, true, true);
    let icreq_bytes = tcp_q
        .initiate_connection()
        .map_err(|_| "nvme_of: initiate_connection failed")?;
    let (icreq_dec, _) = decode_pdu(&icreq_bytes).map_err(|_| "nvme_of: decode icreq failed")?;
    if !matches!(icreq_dec, NvmeTcpPdu::ICReq(_)) {
        return Err("nvme_of: expected ICReq");
    }
    let _ = tcp_q.pop_tx_pdu();

    let icresp_bytes = encode_pdu(
        &NvmeTcpPdu::ICResp(IcRespPdu {
            pfv: NVME_TCP_PFV_1_0,
            cpda: 0,
            digest: 0x03,
            maxh2cdata: 4096,
        }),
        false,
        false,
        0,
    )
    .map_err(|_| "nvme_of: encode icresp failed")?;
    tcp_q
        .process_rx_pdu(&icresp_bytes)
        .map_err(|_| "nvme_of: process icresp failed")?;
    if tcp_q.state != TcpConnState::Connected {
        return Err("nvme_of: expected Connected state");
    }

    let hostid = [0x11u8; 16];
    let conn_cid = tcp_q
        .submit_fabrics_connect(
            &hostid,
            0xFFFF,
            b"nqn.2026-11.io.nexus:nvme0",
            b"nqn.2026-11.io.nexus:host0",
        )
        .map_err(|_| "nvme_of: submit_fabrics_connect failed")?;
    let _ = tcp_q.pop_tx_pdu();

    let conn_resp = encode_pdu(
        &NvmeTcpPdu::CapsuleResp {
            cqe: NvmeOfCqe::success(conn_cid, 0, 1, 0),
        },
        true,
        true,
        0,
    )
    .map_err(|_| "nvme_of: encode connect resp failed")?;
    tcp_q
        .process_rx_pdu(&conn_resp)
        .map_err(|_| "nvme_of: process connect resp failed")?;
    if tcp_q.state != TcpConnState::FabricsReady {
        return Err("nvme_of: expected FabricsReady state");
    }
    let _ = tcp_q
        .take_completed(conn_cid)
        .ok_or("nvme_of: connect completion missing")?;

    // Write with R2T -> H2CData
    let write_cid = tcp_q
        .submit_write(1, 0, 0, vec![0x5Au8; 512])
        .map_err(|_| "nvme_of: submit_write failed")?;
    let _ = tcp_q.pop_tx_pdu();
    let r2t_bytes = encode_pdu(
        &NvmeTcpPdu::R2T(R2TPdu {
            cccid: write_cid,
            ttag: 42,
            r2to: 0,
            r2tl: 512,
        }),
        true,
        true,
        0,
    )
    .map_err(|_| "nvme_of: encode r2t failed")?;
    tcp_q
        .process_rx_pdu(&r2t_bytes)
        .map_err(|_| "nvme_of: process r2t failed")?;
    let h2c_wire = tcp_q.pop_tx_pdu().ok_or("nvme_of: missing H2CData TX")?;
    let (h2c_dec, _) = decode_pdu(&h2c_wire).map_err(|_| "nvme_of: decode H2CData failed")?;
    match h2c_dec {
        NvmeTcpPdu::H2CData(h) => {
            if h.cccid != write_cid || h.ttag != 42 || h.data.len() != 512 {
                return Err("nvme_of: H2CData fields mismatch");
            }
        }
        _ => return Err("nvme_of: expected H2CData"),
    }

    // 4. RDMA Queue-Pair Engine: PD, MR, CQ, QP, Send/Recv, RDMA Write/Read, Atomics
    let mut rdma = RdmaEngine::new();
    let pd_a = rdma.alloc_pd();
    let pd_b = rdma.alloc_pd();
    let rw_flags = MrAccessFlags::LOCAL_WRITE
        .union(MrAccessFlags::REMOTE_WRITE)
        .union(MrAccessFlags::REMOTE_READ)
        .union(MrAccessFlags::REMOTE_ATOMIC);
    let (lkey_a, _rkey_a) = rdma
        .reg_mr(pd_a, 0x1000, 4096, rw_flags)
        .map_err(|_| "nvme_of: reg_mr A failed")?;
    let (lkey_b, rkey_b) = rdma
        .reg_mr(pd_b, 0x2000, 4096, rw_flags)
        .map_err(|_| "nvme_of: reg_mr B failed")?;

    let cq_a = rdma.create_cq(16);
    let cq_b = rdma.create_cq(16);
    let qp_a = rdma
        .create_qp(QpType::Rc, pd_a, cq_a, cq_a, 16, 16)
        .map_err(|_| "nvme_of: create_qp A failed")?;
    let qp_b = rdma
        .create_qp(QpType::Rc, pd_b, cq_b, cq_b, 16, 16)
        .map_err(|_| "nvme_of: create_qp B failed")?;
    rdma.connect_loopback(qp_a, qp_b)
        .map_err(|_| "nvme_of: connect_loopback failed")?;

    // RDMA Write + RDMA Read
    rdma.mr_write_local(lkey_a, 0x1000, b"NEXUS-RDMA-OK")
        .map_err(|_| "nvme_of: mr_write_local failed")?;
    rdma.post_send(
        qp_a,
        SendWr {
            wr_id: 101,
            opcode: RdmaWrOpcode::RdmaWrite,
            send_flags: SendFlags::SIGNALED,
            sge_list: vec![RdmaSge {
                addr: 0x1000,
                length: 13,
                lkey: lkey_a,
            }],
            remote_addr: 0x2000,
            rkey: rkey_b,
            imm_data: 0,
            compare_add: 0,
            swap: 0,
        },
    )
    .map_err(|_| "nvme_of: post_send RdmaWrite failed")?;
    rdma.process_sq(qp_a)
        .map_err(|_| "nvme_of: process_sq RdmaWrite failed")?;
    let wc = rdma
        .poll_cq(cq_a)
        .map_err(|_| "nvme_of: poll_cq failed")?
        .ok_or("nvme_of: missing RdmaWrite WC")?;
    if wc.status != WcStatus::Success || wc.wr_id != 101 {
        return Err("nvme_of: RdmaWrite WC failed");
    }
    let remote_bytes = rdma
        .mr_read_local(lkey_b, 0x2000, 13)
        .map_err(|_| "nvme_of: mr_read_local failed")?;
    if remote_bytes.as_slice() != b"NEXUS-RDMA-OK" {
        return Err("nvme_of: RdmaWrite data mismatch");
    }

    // Atomic Fetch-and-Add
    rdma.mr_write_local(lkey_b, 0x2100, &10u64.to_le_bytes())
        .map_err(|_| "nvme_of: atomic init failed")?;
    rdma.post_send(
        qp_a,
        SendWr {
            wr_id: 102,
            opcode: RdmaWrOpcode::AtomicFetchAdd,
            send_flags: SendFlags::SIGNALED,
            sge_list: vec![RdmaSge {
                addr: 0x1100,
                length: 8,
                lkey: lkey_a,
            }],
            remote_addr: 0x2100,
            rkey: rkey_b,
            imm_data: 0,
            compare_add: 5,
            swap: 0,
        },
    )
    .map_err(|_| "nvme_of: post_send AtomicFetchAdd failed")?;
    rdma.process_sq(qp_a)
        .map_err(|_| "nvme_of: process_sq AtomicFetchAdd failed")?;
    let _ = rdma
        .poll_cq(cq_a)
        .map_err(|_| "nvme_of: poll_cq atomic failed")?;
    let old_val = read_u64_le(
        &rdma
            .mr_read_local(lkey_a, 0x1100, 8)
            .map_err(|_| "nvme_of: read atomic old failed")?,
        0,
    );
    let new_val = read_u64_le(
        &rdma
            .mr_read_local(lkey_b, 0x2100, 8)
            .map_err(|_| "nvme_of: read atomic new failed")?,
        0,
    );
    if old_val != 10 || new_val != 15 {
        return Err("nvme_of: AtomicFetchAdd result mismatch");
    }

    // 5. NVMe-oF over RDMA Capsule Send/Recv
    let mut nvme_rdma_host = NvmeRdmaQueue::new(1, qp_a, lkey_a, 0x1200, lkey_a, 0x1300);
    let nvme_rdma_ctrl = NvmeRdmaQueue::new(1, qp_b, lkey_b, 0x2200, lkey_b, 0x2300);
    nvme_rdma_ctrl
        .post_recv_cqe(&mut rdma, 201)
        .map_err(|_| "nvme_of: post_recv_cqe failed")?;
    let keyed_sgl = SglDescriptor::keyed_data_block(0x2000, 512, rkey_b);
    let (klen, krkey) = keyed_sgl.keyed_parts();
    if klen != 512 || krkey != rkey_b {
        return Err("nvme_of: keyed SGL encode/decode mismatch");
    }
    let rdma_sqe = NvmeOfSqe::nvm_read(9, 1, 0, 0, keyed_sgl);
    nvme_rdma_host
        .post_capsule_cmd(&mut rdma, 202, &rdma_sqe)
        .map_err(|_| "nvme_of: post_capsule_cmd failed")?;
    rdma.process_sq(qp_a)
        .map_err(|_| "nvme_of: process_sq capsule failed")?;
    let wc_send = rdma
        .poll_cq(cq_a)
        .map_err(|_| "nvme_of: poll_cq send failed")?
        .ok_or("nvme_of: missing capsule send WC")?;
    let wc_recv = rdma
        .poll_cq(cq_b)
        .map_err(|_| "nvme_of: poll_cq recv failed")?
        .ok_or("nvme_of: missing capsule recv WC")?;
    if wc_send.status != WcStatus::Success
        || wc_recv.status != WcStatus::Success
        || wc_recv.byte_len != 64
    {
        return Err("nvme_of: NVMe-oF RDMA capsule transfer failed");
    }

    serial_println!("nvme_of: self_test passed (CRC32C, NVMe/TCP PDUs, RDMA QP/CQ/MR, NVMe-RDMA)");
    Ok(())
}
