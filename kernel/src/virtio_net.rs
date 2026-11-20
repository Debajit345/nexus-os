//! Clean-room VirtIO-Net Multi-Queue NIC Driver, TSO/GSO & Checksum Offload Engine.
//!
//! Implements the VirtIO 1.2 Network Device specification (`no_std` + `alloc`):
//! - **Multi-Queue Split Virtqueues**: `rxq[0..N]`, `txq[0..N]`, and `ctrlq` with
//!   descriptor tables, available/used rings, free-descriptor allocators, and
//!   scatter-gather chain management.
//! - **`virtio_net_hdr_v1` & Hash Report**: Full 12-byte `VirtioNetHdrV1` and
//!   20-byte `VirtioNetHdrV1Hash` wire headers.
//! - **RSS Hash Steering**: Toeplitz hash over IPv4/IPv6 2-tuple and 4-tuple flows
//!   with configurable 40-byte key and indirection table.
//! - **Checksum Offload Engine**: Partial checksum (`VIRTIO_NET_HDR_F_NEEDS_CSUM`)
//!   completion and RX checksum verification (`VIRTIO_NET_HDR_F_DATA_VALID`).
//! - **TSO / GSO Segmentation**: Hardware offload header generation and software
//!   segmentation fallback for `GSO_TCPV4`, `GSO_TCPV6`, and `GSO_UDP_L4`.
#![allow(dead_code)]

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::sync::TicketLock;

// ---------------------------------------------------------------------------
// VirtIO-Net Feature Bits & Constants (VirtIO 1.2 Section 5.1.3)
// ---------------------------------------------------------------------------

pub const VIRTIO_NET_F_CSUM: u64 = 1 << 0;
pub const VIRTIO_NET_F_GUEST_CSUM: u64 = 1 << 1;
pub const VIRTIO_NET_F_CTRL_GUEST_OFFLOADS: u64 = 1 << 2;
pub const VIRTIO_NET_F_MTU: u64 = 1 << 3;
pub const VIRTIO_NET_F_MAC: u64 = 1 << 5;
pub const VIRTIO_NET_F_GUEST_TSO4: u64 = 1 << 7;
pub const VIRTIO_NET_F_GUEST_TSO6: u64 = 1 << 8;
pub const VIRTIO_NET_F_GUEST_ECN: u64 = 1 << 9;
pub const VIRTIO_NET_F_GUEST_UFO: u64 = 1 << 10;
pub const VIRTIO_NET_F_HOST_TSO4: u64 = 1 << 11;
pub const VIRTIO_NET_F_HOST_TSO6: u64 = 1 << 12;
pub const VIRTIO_NET_F_HOST_ECN: u64 = 1 << 13;
pub const VIRTIO_NET_F_HOST_UFO: u64 = 1 << 14;
pub const VIRTIO_NET_F_MRG_RXBUF: u64 = 1 << 15;
pub const VIRTIO_NET_F_STATUS: u64 = 1 << 16;
pub const VIRTIO_NET_F_CTRL_VQ: u64 = 1 << 17;
pub const VIRTIO_NET_F_CTRL_RX: u64 = 1 << 18;
pub const VIRTIO_NET_F_CTRL_VLAN: u64 = 1 << 19;
pub const VIRTIO_NET_F_GUEST_ANNOUNCE: u64 = 1 << 21;
pub const VIRTIO_NET_F_MQ: u64 = 1 << 22;
pub const VIRTIO_NET_F_CTRL_MAC_ADDR: u64 = 1 << 23;
pub const VIRTIO_F_VERSION_1: u64 = 1 << 32;
pub const VIRTIO_NET_F_HASH_REPORT: u64 = 1 << 57;
pub const VIRTIO_NET_F_RSS: u64 = 1 << 60;

pub const VIRTIO_NET_S_LINK_UP: u16 = 1 << 0;
pub const VIRTIO_NET_S_ANNOUNCE: u16 = 1 << 1;

pub const VIRTIO_NET_HDR_F_NEEDS_CSUM: u8 = 1 << 0;
pub const VIRTIO_NET_HDR_F_DATA_VALID: u8 = 1 << 1;
pub const VIRTIO_NET_HDR_F_RSC_INFO: u8 = 1 << 2;

pub const VIRTIO_NET_HDR_GSO_NONE: u8 = 0;
pub const VIRTIO_NET_HDR_GSO_TCPV4: u8 = 1;
pub const VIRTIO_NET_HDR_GSO_UDP: u8 = 3;
pub const VIRTIO_NET_HDR_GSO_TCPV6: u8 = 4;
pub const VIRTIO_NET_HDR_GSO_UDP_L4: u8 = 5;
pub const VIRTIO_NET_HDR_GSO_ECN: u8 = 0x80;

pub const VIRTIO_NET_HASH_TYPE_IPV4: u32 = 1 << 0;
pub const VIRTIO_NET_HASH_TYPE_TCPV4: u32 = 1 << 1;
pub const VIRTIO_NET_HASH_TYPE_UDPV4: u32 = 1 << 2;
pub const VIRTIO_NET_HASH_TYPE_IPV6: u32 = 1 << 3;
pub const VIRTIO_NET_HASH_TYPE_TCPV6: u32 = 1 << 4;
pub const VIRTIO_NET_HASH_TYPE_UDPV6: u32 = 1 << 5;

pub const VIRTIO_NET_HASH_REPORT_NONE: u16 = 0;
pub const VIRTIO_NET_HASH_REPORT_IPV4: u16 = 1;
pub const VIRTIO_NET_HASH_REPORT_TCPV4: u16 = 2;
pub const VIRTIO_NET_HASH_REPORT_UDPV4: u16 = 3;
pub const VIRTIO_NET_HASH_REPORT_IPV6: u16 = 4;
pub const VIRTIO_NET_HASH_REPORT_TCPV6: u16 = 5;
pub const VIRTIO_NET_HASH_REPORT_UDPV6: u16 = 6;

pub const VIRTIO_NET_CTRL_RX: u8 = 0;
pub const VIRTIO_NET_CTRL_RX_PROMISC: u8 = 0;
pub const VIRTIO_NET_CTRL_RX_ALLMULTI: u8 = 1;

pub const VIRTIO_NET_CTRL_MAC: u8 = 1;
pub const VIRTIO_NET_CTRL_MAC_ADDR_SET: u8 = 1;

pub const VIRTIO_NET_CTRL_MQ: u8 = 4;
pub const VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET: u8 = 0;
pub const VIRTIO_NET_CTRL_MQ_RSS_CONFIG: u8 = 1;

pub const VIRTIO_NET_OK: u8 = 0;
pub const VIRTIO_NET_ERR: u8 = 1;

pub const DEFAULT_QUEUE_SIZE: usize = 64;
pub const DEFAULT_RX_BUF_SIZE: usize = 2048;
pub const MAX_QUEUE_PAIRS: usize = 16;
pub const RSS_KEY_LEN: usize = 40;
pub const DEFAULT_INDIRECTION_LEN: usize = 128;

const ETH_HDR_LEN: usize = 14;
const ETH_TYPE_IPV4: u16 = 0x0800;
const ETH_TYPE_IPV6: u16 = 0x86DD;
const ETH_TYPE_VLAN: u16 = 0x8100;
const IP_PROTO_TCP: u8 = 6;
const IP_PROTO_UDP: u8 = 17;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtioNetError {
    InvalidArgument,
    NotInitialized,
    QueueFull,
    NoDescriptors,
    QueueOutOfRange,
    UnsupportedFeature,
    PacketTooShort,
    InvalidHeader,
    ChecksumMismatch,
    ControlCommandFailed,
    LinkDown,
}

impl VirtioNetError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgument => "virtio-net: invalid argument",
            Self::NotInitialized => "virtio-net: device not initialized",
            Self::QueueFull => "virtio-net: virtqueue full",
            Self::NoDescriptors => "virtio-net: no free virtqueue descriptors",
            Self::QueueOutOfRange => "virtio-net: queue index out of range",
            Self::UnsupportedFeature => "virtio-net: unsupported feature",
            Self::PacketTooShort => "virtio-net: packet too short",
            Self::InvalidHeader => "virtio-net: invalid protocol header",
            Self::ChecksumMismatch => "virtio-net: checksum verification failed",
            Self::ControlCommandFailed => "virtio-net: control virtqueue command failed",
            Self::LinkDown => "virtio-net: link is down",
        }
    }
}

// ---------------------------------------------------------------------------
// VirtIO-Net Wire Protocol Structures
// ---------------------------------------------------------------------------

/// VirtIO 1.2 `virtio_net_hdr_v1` (12 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VirtioNetHdrV1 {
    pub flags: u8,
    pub gso_type: u8,
    pub hdr_len: u16,
    pub gso_size: u16,
    pub csum_start: u16,
    pub csum_offset: u16,
    pub num_buffers: u16,
}

impl VirtioNetHdrV1 {
    pub const SIZE: usize = size_of::<Self>();

    pub fn to_bytes(self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0] = self.flags;
        out[1] = self.gso_type;
        out[2..4].copy_from_slice(&self.hdr_len.to_le_bytes());
        out[4..6].copy_from_slice(&self.gso_size.to_le_bytes());
        out[6..8].copy_from_slice(&self.csum_start.to_le_bytes());
        out[8..10].copy_from_slice(&self.csum_offset.to_le_bytes());
        out[10..12].copy_from_slice(&self.num_buffers.to_le_bytes());
        out
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, VirtioNetError> {
        if buf.len() < Self::SIZE {
            return Err(VirtioNetError::PacketTooShort);
        }
        Ok(Self {
            flags: buf[0],
            gso_type: buf[1],
            hdr_len: u16::from_le_bytes([buf[2], buf[3]]),
            gso_size: u16::from_le_bytes([buf[4], buf[5]]),
            csum_start: u16::from_le_bytes([buf[6], buf[7]]),
            csum_offset: u16::from_le_bytes([buf[8], buf[9]]),
            num_buffers: u16::from_le_bytes([buf[10], buf[11]]),
        })
    }
}

/// Extended `virtio_net_hdr_v1` with `VIRTIO_NET_F_HASH_REPORT` fields (20 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VirtioNetHdrV1Hash {
    pub hdr: VirtioNetHdrV1,
    pub hash_value: u32,
    pub hash_report: u16,
    pub padding_reserved: u16,
}

impl VirtioNetHdrV1Hash {
    pub const SIZE: usize = size_of::<Self>();
}

/// VirtIO-Net device configuration space (`virtio_net_config`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioNetConfig {
    pub mac: [u8; 6],
    pub status: u16,
    pub max_virtqueue_pairs: u16,
    pub mtu: u16,
    pub speed: u32,
    pub duplex: u8,
    pub rss_max_key_size: u8,
    pub rss_max_indirection_table_length: u16,
    pub supported_hash_types: u32,
}

/// Control virtqueue command header.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtioNetCtrlHdr {
    pub class: u8,
    pub cmd: u8,
}

// ---------------------------------------------------------------------------
// Split Virtqueue Implementation
// ---------------------------------------------------------------------------

pub const VIRTQ_DESC_F_NEXT: u16 = 1;
pub const VIRTQ_DESC_F_WRITE: u16 = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtqDesc {
    pub addr: u64,
    pub len: u32,
    pub flags: u16,
    pub next: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtqUsedElem {
    pub id: u32,
    pub len: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct SgSegment {
    pub addr: u64,
    pub len: u32,
    pub device_writable: bool,
}

pub struct SplitVirtqueue {
    pub queue_index: u16,
    pub size: u16,
    desc: Vec<VirtqDesc>,
    avail_ring: Vec<u16>,
    avail_idx: u16,
    used_ring: Vec<VirtqUsedElem>,
    used_idx: u16,
    last_used_idx: u16,
    free_stack: Vec<u16>,
    pub notify_count: u64,
}

impl SplitVirtqueue {
    pub fn new(queue_index: u16, size: u16) -> Result<Self, VirtioNetError> {
        if size == 0 || !size.is_power_of_two() {
            return Err(VirtioNetError::InvalidArgument);
        }
        let n = size as usize;
        let mut free_stack = Vec::with_capacity(n);
        for idx in (0..size).rev() {
            free_stack.push(idx);
        }
        Ok(Self {
            queue_index,
            size,
            desc: vec![VirtqDesc::default(); n],
            avail_ring: vec![0u16; n],
            avail_idx: 0,
            used_ring: vec![VirtqUsedElem::default(); n],
            used_idx: 0,
            last_used_idx: 0,
            free_stack,
            notify_count: 0,
        })
    }

    pub fn free_descriptors(&self) -> u16 {
        self.free_stack.len() as u16
    }

    pub fn alloc_desc(&mut self) -> Result<u16, VirtioNetError> {
        self.free_stack.pop().ok_or(VirtioNetError::NoDescriptors)
    }

    pub fn free_chain(&mut self, head: u16) {
        let mut cur = head;
        let mut guard = 0u16;
        while guard < self.size {
            let d = self.desc[cur as usize];
            let has_next = (d.flags & VIRTQ_DESC_F_NEXT) != 0;
            let next = d.next;
            self.desc[cur as usize] = VirtqDesc::default();
            self.free_stack.push(cur);
            if !has_next {
                break;
            }
            cur = next;
            guard += 1;
        }
    }

    pub fn push_chain(&mut self, segments: &[SgSegment]) -> Result<u16, VirtioNetError> {
        if segments.is_empty() {
            return Err(VirtioNetError::InvalidArgument);
        }
        if (self.free_stack.len()) < segments.len() {
            return Err(VirtioNetError::NoDescriptors);
        }

        let mut indices = Vec::with_capacity(segments.len());
        for _ in segments {
            indices.push(self.alloc_desc()?);
        }

        for (pos, seg) in segments.iter().enumerate() {
            let idx = indices[pos] as usize;
            let is_last = pos + 1 == segments.len();
            let mut flags = if seg.device_writable {
                VIRTQ_DESC_F_WRITE
            } else {
                0
            };
            let next = if !is_last {
                flags |= VIRTQ_DESC_F_NEXT;
                indices[pos + 1]
            } else {
                0
            };
            self.desc[idx] = VirtqDesc {
                addr: seg.addr,
                len: seg.len,
                flags,
                next,
            };
        }

        let head = indices[0];
        let slot = (self.avail_idx as usize) & ((self.size as usize) - 1);
        self.avail_ring[slot] = head;
        self.avail_idx = self.avail_idx.wrapping_add(1);
        self.notify_count = self.notify_count.wrapping_add(1);
        Ok(head)
    }

    pub fn complete_used(&mut self, head: u16, written_len: u32) {
        let slot = (self.used_idx as usize) & ((self.size as usize) - 1);
        self.used_ring[slot] = VirtqUsedElem {
            id: head as u32,
            len: written_len,
        };
        self.used_idx = self.used_idx.wrapping_add(1);
    }

    pub fn pop_used(&mut self) -> Option<VirtqUsedElem> {
        if self.last_used_idx == self.used_idx {
            return None;
        }
        let slot = (self.last_used_idx as usize) & ((self.size as usize) - 1);
        let elem = self.used_ring[slot];
        self.last_used_idx = self.last_used_idx.wrapping_add(1);
        self.free_chain(elem.id as u16);
        Some(elem)
    }
}

// ---------------------------------------------------------------------------
// Checksum Offload Engine
// ---------------------------------------------------------------------------

/// Compute 16-bit one's complement sum over `data` (big-endian 16-bit words).
pub fn ones_complement_sum(data: &[u8], initial: u32) -> u32 {
    let mut sum = initial;
    let (chunks, rem) = data.as_chunks::<2>();
    for pair in chunks {
        sum += u16::from_be_bytes(*pair) as u32;
    }
    if let Some(&last) = rem.first() {
        sum += (last as u32) << 8;
    }
    sum
}

/// Fold 32-bit accumulator into 16-bit one's complement checksum.
#[inline]
pub fn finalize_checksum(mut sum: u32) -> u16 {
    while (sum >> 16) != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

/// Compute IPv4 pseudo-header partial sum.
pub fn pseudo_header_sum_v4(src: [u8; 4], dst: [u8; 4], proto: u8, l4_len: u16) -> u32 {
    let mut sum = 0u32;
    sum += u16::from_be_bytes([src[0], src[1]]) as u32;
    sum += u16::from_be_bytes([src[2], src[3]]) as u32;
    sum += u16::from_be_bytes([dst[0], dst[1]]) as u32;
    sum += u16::from_be_bytes([dst[2], dst[3]]) as u32;
    sum += proto as u32;
    sum += l4_len as u32;
    sum
}

/// Compute IPv6 pseudo-header partial sum.
pub fn pseudo_header_sum_v6(src: &[u8; 16], dst: &[u8; 16], next_hdr: u8, l4_len: u32) -> u32 {
    let mut sum = ones_complement_sum(src, 0);
    sum = ones_complement_sum(dst, sum);
    sum += (l4_len >> 16) + (l4_len & 0xFFFF);
    sum += next_hdr as u32;
    sum
}

/// Complete a partial checksum in `frame` as specified by `VIRTIO_NET_HDR_F_NEEDS_CSUM`.
pub fn complete_partial_checksum(
    hdr: &VirtioNetHdrV1,
    frame: &mut [u8],
) -> Result<(), VirtioNetError> {
    if (hdr.flags & VIRTIO_NET_HDR_F_NEEDS_CSUM) == 0 {
        return Ok(());
    }
    let start = hdr.csum_start as usize;
    let offset = hdr.csum_offset as usize;
    if start >= frame.len() || start + offset + 2 > frame.len() {
        return Err(VirtioNetError::PacketTooShort);
    }
    let sum = ones_complement_sum(&frame[start..], 0);
    let csum = finalize_checksum(sum);
    frame[start + offset..start + offset + 2].copy_from_slice(&csum.to_be_bytes());
    Ok(())
}

/// Verify IPv4 header and L4 (TCP/UDP) checksums on an RX frame unless `DATA_VALID` is set.
pub fn verify_rx_checksum(hdr: &VirtioNetHdrV1, frame: &[u8]) -> Result<bool, VirtioNetError> {
    if (hdr.flags & VIRTIO_NET_HDR_F_DATA_VALID) != 0 {
        return Ok(true);
    }
    let (ethertype, l3_off) = parse_eth_header(frame)?;
    match ethertype {
        ETH_TYPE_IPV4 => {
            if frame.len() < l3_off + 20 {
                return Err(VirtioNetError::PacketTooShort);
            }
            let ihl = ((frame[l3_off] & 0x0F) as usize) * 4;
            if ihl < 20 || frame.len() < l3_off + ihl {
                return Err(VirtioNetError::InvalidHeader);
            }
            if finalize_checksum(ones_complement_sum(&frame[l3_off..l3_off + ihl], 0)) != 0 {
                return Ok(false);
            }
            let total_len = u16::from_be_bytes([frame[l3_off + 2], frame[l3_off + 3]]) as usize;
            if total_len < ihl || l3_off + total_len > frame.len() {
                return Err(VirtioNetError::PacketTooShort);
            }
            let proto = frame[l3_off + 9];
            if proto != IP_PROTO_TCP && proto != IP_PROTO_UDP {
                return Ok(true);
            }
            let l4_off = l3_off + ihl;
            let l4_len = total_len - ihl;
            let mut src = [0u8; 4];
            let mut dst = [0u8; 4];
            src.copy_from_slice(&frame[l3_off + 12..l3_off + 16]);
            dst.copy_from_slice(&frame[l3_off + 16..l3_off + 20]);
            let pseudo = pseudo_header_sum_v4(src, dst, proto, l4_len as u16);
            let sum = ones_complement_sum(&frame[l4_off..l4_off + l4_len], pseudo);
            Ok(finalize_checksum(sum) == 0)
        }
        ETH_TYPE_IPV6 => {
            if frame.len() < l3_off + 40 {
                return Err(VirtioNetError::PacketTooShort);
            }
            let payload_len = u16::from_be_bytes([frame[l3_off + 4], frame[l3_off + 5]]) as usize;
            let next_hdr = frame[l3_off + 6];
            if l3_off + 40 + payload_len > frame.len() {
                return Err(VirtioNetError::PacketTooShort);
            }
            if next_hdr != IP_PROTO_TCP && next_hdr != IP_PROTO_UDP {
                return Ok(true);
            }
            let mut src = [0u8; 16];
            let mut dst = [0u8; 16];
            src.copy_from_slice(&frame[l3_off + 8..l3_off + 24]);
            dst.copy_from_slice(&frame[l3_off + 24..l3_off + 40]);
            let l4_off = l3_off + 40;
            let pseudo = pseudo_header_sum_v6(&src, &dst, next_hdr, payload_len as u32);
            let sum = ones_complement_sum(&frame[l4_off..l4_off + payload_len], pseudo);
            Ok(finalize_checksum(sum) == 0)
        }
        _ => Ok(true),
    }
}

/// Populate `VirtioNetHdrV1` for TX checksum offload (`NEEDS_CSUM`) on IPv4/IPv6 TCP/UDP frames.
pub fn prepare_tx_checksum_hdr(frame: &mut [u8]) -> Result<VirtioNetHdrV1, VirtioNetError> {
    let (ethertype, l3_off) = parse_eth_header(frame)?;
    let mut hdr = VirtioNetHdrV1::default();

    match ethertype {
        ETH_TYPE_IPV4 => {
            if frame.len() < l3_off + 20 {
                return Err(VirtioNetError::PacketTooShort);
            }
            let ihl = ((frame[l3_off] & 0x0F) as usize) * 4;
            if ihl < 20 || frame.len() < l3_off + ihl {
                return Err(VirtioNetError::InvalidHeader);
            }
            let total_len = u16::from_be_bytes([frame[l3_off + 2], frame[l3_off + 3]]) as usize;
            if total_len < ihl || l3_off + total_len > frame.len() {
                return Err(VirtioNetError::PacketTooShort);
            }
            let proto = frame[l3_off + 9];
            let l4_off = l3_off + ihl;
            let l4_len = (total_len - ihl) as u16;
            let mut src = [0u8; 4];
            let mut dst = [0u8; 4];
            src.copy_from_slice(&frame[l3_off + 12..l3_off + 16]);
            dst.copy_from_slice(&frame[l3_off + 16..l3_off + 20]);
            let pseudo = finalize_checksum(pseudo_header_sum_v4(src, dst, proto, l4_len));
            let pseudo_inv = !pseudo;

            if proto == IP_PROTO_TCP && frame.len() >= l4_off + 20 {
                frame[l4_off + 16..l4_off + 18].copy_from_slice(&pseudo_inv.to_be_bytes());
                hdr.flags = VIRTIO_NET_HDR_F_NEEDS_CSUM;
                hdr.csum_start = l4_off as u16;
                hdr.csum_offset = 16;
            } else if proto == IP_PROTO_UDP && frame.len() >= l4_off + 8 {
                frame[l4_off + 6..l4_off + 8].copy_from_slice(&pseudo_inv.to_be_bytes());
                hdr.flags = VIRTIO_NET_HDR_F_NEEDS_CSUM;
                hdr.csum_start = l4_off as u16;
                hdr.csum_offset = 6;
            }
        }
        ETH_TYPE_IPV6 => {
            if frame.len() < l3_off + 40 {
                return Err(VirtioNetError::PacketTooShort);
            }
            let payload_len = u16::from_be_bytes([frame[l3_off + 4], frame[l3_off + 5]]) as u32;
            let next_hdr = frame[l3_off + 6];
            let l4_off = l3_off + 40;
            let mut src = [0u8; 16];
            let mut dst = [0u8; 16];
            src.copy_from_slice(&frame[l3_off + 8..l3_off + 24]);
            dst.copy_from_slice(&frame[l3_off + 24..l3_off + 40]);
            let pseudo_inv =
                !finalize_checksum(pseudo_header_sum_v6(&src, &dst, next_hdr, payload_len));

            if next_hdr == IP_PROTO_TCP && frame.len() >= l4_off + 20 {
                frame[l4_off + 16..l4_off + 18].copy_from_slice(&pseudo_inv.to_be_bytes());
                hdr.flags = VIRTIO_NET_HDR_F_NEEDS_CSUM;
                hdr.csum_start = l4_off as u16;
                hdr.csum_offset = 16;
            } else if next_hdr == IP_PROTO_UDP && frame.len() >= l4_off + 8 {
                frame[l4_off + 6..l4_off + 8].copy_from_slice(&pseudo_inv.to_be_bytes());
                hdr.flags = VIRTIO_NET_HDR_F_NEEDS_CSUM;
                hdr.csum_start = l4_off as u16;
                hdr.csum_offset = 6;
            }
        }
        _ => {}
    }

    Ok(hdr)
}

// ---------------------------------------------------------------------------
// TSO / GSO Segmentation Engine
// ---------------------------------------------------------------------------

fn parse_eth_header(frame: &[u8]) -> Result<(u16, usize), VirtioNetError> {
    if frame.len() < ETH_HDR_LEN {
        return Err(VirtioNetError::PacketTooShort);
    }
    let ethertype = u16::from_be_bytes([frame[12], frame[13]]);
    if ethertype == ETH_TYPE_VLAN {
        if frame.len() < ETH_HDR_LEN + 4 {
            return Err(VirtioNetError::PacketTooShort);
        }
        let inner = u16::from_be_bytes([frame[16], frame[17]]);
        Ok((inner, ETH_HDR_LEN + 4))
    } else {
        Ok((ethertype, ETH_HDR_LEN))
    }
}

/// Segment a GSO/TSO frame into MTU-sized Ethernet frames with valid L3/L4 headers and checksums.
pub fn segment_gso_packet(
    hdr: &VirtioNetHdrV1,
    frame: &[u8],
) -> Result<Vec<Vec<u8>>, VirtioNetError> {
    let gso_base = hdr.gso_type & !VIRTIO_NET_HDR_GSO_ECN;
    if gso_base == VIRTIO_NET_HDR_GSO_NONE || hdr.gso_size == 0 {
        let mut single = frame.to_vec();
        complete_partial_checksum(hdr, &mut single)?;
        return Ok(vec![single]);
    }

    let mss = hdr.gso_size as usize;
    let (ethertype, l3_off) = parse_eth_header(frame)?;

    match gso_base {
        VIRTIO_NET_HDR_GSO_TCPV4 => {
            if ethertype != ETH_TYPE_IPV4 || frame.len() < l3_off + 20 {
                return Err(VirtioNetError::InvalidHeader);
            }
            let ihl = ((frame[l3_off] & 0x0F) as usize) * 4;
            if ihl < 20 || frame.len() < l3_off + ihl + 20 {
                return Err(VirtioNetError::InvalidHeader);
            }
            let l4_off = l3_off + ihl;
            let tcp_hdr_len = ((frame[l4_off + 12] >> 4) as usize) * 4;
            if tcp_hdr_len < 20 || frame.len() < l4_off + tcp_hdr_len {
                return Err(VirtioNetError::InvalidHeader);
            }
            let payload_off = l4_off + tcp_hdr_len;
            let payload = &frame[payload_off..];
            if payload.len() <= mss {
                let mut single = frame.to_vec();
                complete_partial_checksum(hdr, &mut single)?;
                return Ok(vec![single]);
            }

            let base_id = u16::from_be_bytes([frame[l3_off + 4], frame[l3_off + 5]]);
            let base_seq = u32::from_be_bytes([
                frame[l4_off + 4],
                frame[l4_off + 5],
                frame[l4_off + 6],
                frame[l4_off + 7],
            ]);
            let orig_flags = frame[l4_off + 13];
            let mut src_ip = [0u8; 4];
            let mut dst_ip = [0u8; 4];
            src_ip.copy_from_slice(&frame[l3_off + 12..l3_off + 16]);
            dst_ip.copy_from_slice(&frame[l3_off + 16..l3_off + 20]);

            let total_segs = payload.len().div_ceil(mss);
            let mut segments = Vec::with_capacity(total_segs);

            for (seg_idx, chunk) in payload.chunks(mss).enumerate() {
                let is_first = seg_idx == 0;
                let is_last = seg_idx + 1 == total_segs;
                let mut seg = Vec::with_capacity(payload_off + chunk.len());
                seg.extend_from_slice(&frame[..payload_off]);
                seg.extend_from_slice(chunk);

                // Update IPv4 total length, ID, and header checksum
                let ip_total_len = (ihl + tcp_hdr_len + chunk.len()) as u16;
                seg[l3_off + 2..l3_off + 4].copy_from_slice(&ip_total_len.to_be_bytes());
                let seg_id = base_id.wrapping_add(seg_idx as u16);
                seg[l3_off + 4..l3_off + 6].copy_from_slice(&seg_id.to_be_bytes());
                seg[l3_off + 10] = 0;
                seg[l3_off + 11] = 0;
                let ip_csum = finalize_checksum(ones_complement_sum(&seg[l3_off..l3_off + ihl], 0));
                seg[l3_off + 10..l3_off + 12].copy_from_slice(&ip_csum.to_be_bytes());

                // Update TCP sequence number and flags (FIN/PSH only on last, CWR only on first)
                let seg_seq = base_seq.wrapping_add((seg_idx * mss) as u32);
                seg[l4_off + 4..l4_off + 8].copy_from_slice(&seg_seq.to_be_bytes());
                let mut seg_flags = orig_flags;
                if !is_last {
                    seg_flags &= !(0x01 | 0x08); // clear FIN (0x01) and PSH (0x08)
                }
                if !is_first {
                    seg_flags &= !0x80; // clear CWR (0x80)
                }
                seg[l4_off + 13] = seg_flags;

                // Recompute TCP checksum
                seg[l4_off + 16] = 0;
                seg[l4_off + 17] = 0;
                let l4_len = (tcp_hdr_len + chunk.len()) as u16;
                let pseudo = pseudo_header_sum_v4(src_ip, dst_ip, IP_PROTO_TCP, l4_len);
                let tcp_csum = finalize_checksum(ones_complement_sum(&seg[l4_off..], pseudo));
                seg[l4_off + 16..l4_off + 18].copy_from_slice(&tcp_csum.to_be_bytes());

                segments.push(seg);
            }
            Ok(segments)
        }
        VIRTIO_NET_HDR_GSO_TCPV6 => {
            if ethertype != ETH_TYPE_IPV6 || frame.len() < l3_off + 40 + 20 {
                return Err(VirtioNetError::InvalidHeader);
            }
            let l4_off = l3_off + 40;
            let tcp_hdr_len = ((frame[l4_off + 12] >> 4) as usize) * 4;
            if tcp_hdr_len < 20 || frame.len() < l4_off + tcp_hdr_len {
                return Err(VirtioNetError::InvalidHeader);
            }
            let payload_off = l4_off + tcp_hdr_len;
            let payload = &frame[payload_off..];
            if payload.len() <= mss {
                let mut single = frame.to_vec();
                complete_partial_checksum(hdr, &mut single)?;
                return Ok(vec![single]);
            }

            let base_seq = u32::from_be_bytes([
                frame[l4_off + 4],
                frame[l4_off + 5],
                frame[l4_off + 6],
                frame[l4_off + 7],
            ]);
            let orig_flags = frame[l4_off + 13];
            let mut src_ip = [0u8; 16];
            let mut dst_ip = [0u8; 16];
            src_ip.copy_from_slice(&frame[l3_off + 8..l3_off + 24]);
            dst_ip.copy_from_slice(&frame[l3_off + 24..l3_off + 40]);

            let total_segs = payload.len().div_ceil(mss);
            let mut segments = Vec::with_capacity(total_segs);

            for (seg_idx, chunk) in payload.chunks(mss).enumerate() {
                let is_first = seg_idx == 0;
                let is_last = seg_idx + 1 == total_segs;
                let mut seg = Vec::with_capacity(payload_off + chunk.len());
                seg.extend_from_slice(&frame[..payload_off]);
                seg.extend_from_slice(chunk);

                let l4_len = (tcp_hdr_len + chunk.len()) as u16;
                seg[l3_off + 4..l3_off + 6].copy_from_slice(&l4_len.to_be_bytes());

                let seg_seq = base_seq.wrapping_add((seg_idx * mss) as u32);
                seg[l4_off + 4..l4_off + 8].copy_from_slice(&seg_seq.to_be_bytes());
                let mut seg_flags = orig_flags;
                if !is_last {
                    seg_flags &= !(0x01 | 0x08);
                }
                if !is_first {
                    seg_flags &= !0x80;
                }
                seg[l4_off + 13] = seg_flags;

                seg[l4_off + 16] = 0;
                seg[l4_off + 17] = 0;
                let pseudo = pseudo_header_sum_v6(&src_ip, &dst_ip, IP_PROTO_TCP, l4_len as u32);
                let tcp_csum = finalize_checksum(ones_complement_sum(&seg[l4_off..], pseudo));
                seg[l4_off + 16..l4_off + 18].copy_from_slice(&tcp_csum.to_be_bytes());

                segments.push(seg);
            }
            Ok(segments)
        }
        VIRTIO_NET_HDR_GSO_UDP_L4 | VIRTIO_NET_HDR_GSO_UDP => {
            if ethertype != ETH_TYPE_IPV4 || frame.len() < l3_off + 20 + 8 {
                return Err(VirtioNetError::InvalidHeader);
            }
            let ihl = ((frame[l3_off] & 0x0F) as usize) * 4;
            let l4_off = l3_off + ihl;
            let payload_off = l4_off + 8;
            if frame.len() < payload_off {
                return Err(VirtioNetError::InvalidHeader);
            }
            let payload = &frame[payload_off..];
            if payload.len() <= mss {
                let mut single = frame.to_vec();
                complete_partial_checksum(hdr, &mut single)?;
                return Ok(vec![single]);
            }

            let base_id = u16::from_be_bytes([frame[l3_off + 4], frame[l3_off + 5]]);
            let mut src_ip = [0u8; 4];
            let mut dst_ip = [0u8; 4];
            src_ip.copy_from_slice(&frame[l3_off + 12..l3_off + 16]);
            dst_ip.copy_from_slice(&frame[l3_off + 16..l3_off + 20]);

            let total_segs = payload.len().div_ceil(mss);
            let mut segments = Vec::with_capacity(total_segs);

            for (seg_idx, chunk) in payload.chunks(mss).enumerate() {
                let mut seg = Vec::with_capacity(payload_off + chunk.len());
                seg.extend_from_slice(&frame[..payload_off]);
                seg.extend_from_slice(chunk);

                let udp_len = (8 + chunk.len()) as u16;
                let ip_total_len = (ihl + 8 + chunk.len()) as u16;
                seg[l3_off + 2..l3_off + 4].copy_from_slice(&ip_total_len.to_be_bytes());
                let seg_id = base_id.wrapping_add(seg_idx as u16);
                seg[l3_off + 4..l3_off + 6].copy_from_slice(&seg_id.to_be_bytes());
                seg[l3_off + 10] = 0;
                seg[l3_off + 11] = 0;
                let ip_csum = finalize_checksum(ones_complement_sum(&seg[l3_off..l3_off + ihl], 0));
                seg[l3_off + 10..l3_off + 12].copy_from_slice(&ip_csum.to_be_bytes());

                seg[l4_off + 4..l4_off + 6].copy_from_slice(&udp_len.to_be_bytes());
                seg[l4_off + 6] = 0;
                seg[l4_off + 7] = 0;
                let pseudo = pseudo_header_sum_v4(src_ip, dst_ip, IP_PROTO_UDP, udp_len);
                let udp_csum = finalize_checksum(ones_complement_sum(&seg[l4_off..], pseudo));
                seg[l4_off + 6..l4_off + 8].copy_from_slice(&udp_csum.to_be_bytes());

                segments.push(seg);
            }
            Ok(segments)
        }
        _ => Err(VirtioNetError::UnsupportedFeature),
    }
}

// ---------------------------------------------------------------------------
// RSS (Receive Side Scaling) Hash Steering
// ---------------------------------------------------------------------------

/// Standard 40-byte Toeplitz RSS key.
pub const DEFAULT_RSS_KEY: [u8; RSS_KEY_LEN] = [
    0x6d, 0x5a, 0x56, 0xda, 0x25, 0x5b, 0x0e, 0xc2, 0x41, 0x67, 0x25, 0x3d, 0x43, 0xa3, 0x8f, 0xb0,
    0xd0, 0xca, 0x2b, 0xcb, 0xae, 0x7b, 0x30, 0xb4, 0x77, 0xcb, 0x2d, 0xa3, 0x80, 0x30, 0xf2, 0x0c,
    0x6a, 0x42, 0xb7, 0x3b, 0xbe, 0xac, 0x01, 0xfa,
];

/// Compute Toeplitz hash over `input` using `key`.
pub fn toeplitz_hash(key: &[u8], input: &[u8]) -> u32 {
    if key.len() < 4 {
        return 0;
    }
    let mut result = 0u32;
    let mut key_word = u32::from_be_bytes([key[0], key[1], key[2], key[3]]);
    for (key_byte_idx, &byte) in (4usize..).zip(input.iter()) {
        for bit in (0..8).rev() {
            if ((byte >> bit) & 1) != 0 {
                result ^= key_word;
            }
            let next_bit = if key_byte_idx < key.len() {
                ((key[key_byte_idx] >> bit) & 1) as u32
            } else {
                0
            };
            key_word = (key_word << 1) | next_bit;
        }
    }
    result
}

#[derive(Debug, Clone)]
pub struct RssConfig {
    pub hash_types: u32,
    pub key: [u8; RSS_KEY_LEN],
    pub indirection_table: Vec<u16>,
    pub num_queues: u16,
}

impl RssConfig {
    pub fn new(num_queues: u16) -> Self {
        let q = num_queues.max(1);
        let mut indirection_table = Vec::with_capacity(DEFAULT_INDIRECTION_LEN);
        for idx in 0..DEFAULT_INDIRECTION_LEN {
            indirection_table.push((idx as u16) % q);
        }
        Self {
            hash_types: VIRTIO_NET_HASH_TYPE_IPV4
                | VIRTIO_NET_HASH_TYPE_TCPV4
                | VIRTIO_NET_HASH_TYPE_UDPV4
                | VIRTIO_NET_HASH_TYPE_IPV6
                | VIRTIO_NET_HASH_TYPE_TCPV6
                | VIRTIO_NET_HASH_TYPE_UDPV6,
            key: DEFAULT_RSS_KEY,
            indirection_table,
            num_queues: q,
        }
    }

    pub fn set_num_queues(&mut self, num_queues: u16) {
        let q = num_queues.max(1);
        self.num_queues = q;
        for (idx, entry) in self.indirection_table.iter_mut().enumerate() {
            *entry = (idx as u16) % q;
        }
    }

    /// Compute `(hash_value, hash_report_type)` for an Ethernet frame.
    pub fn compute_hash(&self, frame: &[u8]) -> (u32, u16) {
        let Ok((ethertype, l3_off)) = parse_eth_header(frame) else {
            return (0, VIRTIO_NET_HASH_REPORT_NONE);
        };

        match ethertype {
            ETH_TYPE_IPV4 if frame.len() >= l3_off + 20 => {
                let ihl = ((frame[l3_off] & 0x0F) as usize) * 4;
                let proto = frame[l3_off + 9];
                let mut tuple = [0u8; 12];
                tuple[0..4].copy_from_slice(&frame[l3_off + 12..l3_off + 16]);
                tuple[4..8].copy_from_slice(&frame[l3_off + 16..l3_off + 20]);

                if proto == IP_PROTO_TCP
                    && (self.hash_types & VIRTIO_NET_HASH_TYPE_TCPV4) != 0
                    && frame.len() >= l3_off + ihl + 4
                {
                    tuple[8..12].copy_from_slice(&frame[l3_off + ihl..l3_off + ihl + 4]);
                    return (
                        toeplitz_hash(&self.key, &tuple),
                        VIRTIO_NET_HASH_REPORT_TCPV4,
                    );
                }
                if proto == IP_PROTO_UDP
                    && (self.hash_types & VIRTIO_NET_HASH_TYPE_UDPV4) != 0
                    && frame.len() >= l3_off + ihl + 4
                {
                    tuple[8..12].copy_from_slice(&frame[l3_off + ihl..l3_off + ihl + 4]);
                    return (
                        toeplitz_hash(&self.key, &tuple),
                        VIRTIO_NET_HASH_REPORT_UDPV4,
                    );
                }
                if (self.hash_types & VIRTIO_NET_HASH_TYPE_IPV4) != 0 {
                    return (
                        toeplitz_hash(&self.key, &tuple[..8]),
                        VIRTIO_NET_HASH_REPORT_IPV4,
                    );
                }
                (0, VIRTIO_NET_HASH_REPORT_NONE)
            }
            ETH_TYPE_IPV6 if frame.len() >= l3_off + 40 => {
                let next_hdr = frame[l3_off + 6];
                let mut tuple = [0u8; 36];
                tuple[0..16].copy_from_slice(&frame[l3_off + 8..l3_off + 24]);
                tuple[16..32].copy_from_slice(&frame[l3_off + 24..l3_off + 40]);

                if next_hdr == IP_PROTO_TCP
                    && (self.hash_types & VIRTIO_NET_HASH_TYPE_TCPV6) != 0
                    && frame.len() >= l3_off + 44
                {
                    tuple[32..36].copy_from_slice(&frame[l3_off + 40..l3_off + 44]);
                    return (
                        toeplitz_hash(&self.key, &tuple),
                        VIRTIO_NET_HASH_REPORT_TCPV6,
                    );
                }
                if next_hdr == IP_PROTO_UDP
                    && (self.hash_types & VIRTIO_NET_HASH_TYPE_UDPV6) != 0
                    && frame.len() >= l3_off + 44
                {
                    tuple[32..36].copy_from_slice(&frame[l3_off + 40..l3_off + 44]);
                    return (
                        toeplitz_hash(&self.key, &tuple),
                        VIRTIO_NET_HASH_REPORT_UDPV6,
                    );
                }
                if (self.hash_types & VIRTIO_NET_HASH_TYPE_IPV6) != 0 {
                    return (
                        toeplitz_hash(&self.key, &tuple[..32]),
                        VIRTIO_NET_HASH_REPORT_IPV6,
                    );
                }
                (0, VIRTIO_NET_HASH_REPORT_NONE)
            }
            _ => (0, VIRTIO_NET_HASH_REPORT_NONE),
        }
    }

    /// Select queue index via RSS indirection table.
    pub fn select_queue(&self, frame: &[u8]) -> usize {
        if self.indirection_table.is_empty() || self.num_queues <= 1 {
            return 0;
        }
        let (hash, report) = self.compute_hash(frame);
        if report == VIRTIO_NET_HASH_REPORT_NONE {
            return 0;
        }
        let idx = (hash as usize) % self.indirection_table.len();
        (self.indirection_table[idx] as usize) % (self.num_queues as usize)
    }
}

// ---------------------------------------------------------------------------
// Multi-Queue Pair & Device State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct QueueStats {
    pub rx_packets: u64,
    pub rx_bytes: u64,
    pub tx_packets: u64,
    pub tx_bytes: u64,
    pub tx_tso_segments: u64,
    pub tx_csum_offloaded: u64,
    pub rx_csum_verified: u64,
    pub drops: u64,
}

pub struct QueuePair {
    pub pair_index: u16,
    pub rx: SplitVirtqueue,
    pub tx: SplitVirtqueue,
    rx_buffers: Vec<Vec<u8>>,
    rx_posted: BTreeMap<u16, usize>,
    tx_inflight: BTreeMap<u16, Vec<u8>>,
    rx_ready: Vec<(VirtioNetHdrV1Hash, Vec<u8>)>,
    pub stats: QueueStats,
}

impl QueuePair {
    pub fn new(pair_index: u16, queue_size: u16) -> Result<Self, VirtioNetError> {
        let rx_q_idx = pair_index * 2;
        let tx_q_idx = pair_index * 2 + 1;
        let rx = SplitVirtqueue::new(rx_q_idx, queue_size)?;
        let tx = SplitVirtqueue::new(tx_q_idx, queue_size)?;
        let mut rx_buffers = Vec::with_capacity(queue_size as usize);
        for _ in 0..queue_size {
            rx_buffers.push(vec![0u8; DEFAULT_RX_BUF_SIZE]);
        }
        let mut qp = Self {
            pair_index,
            rx,
            tx,
            rx_buffers,
            rx_posted: BTreeMap::new(),
            tx_inflight: BTreeMap::new(),
            rx_ready: Vec::new(),
            stats: QueueStats::default(),
        };
        qp.refill_rx()?;
        Ok(qp)
    }

    pub fn refill_rx(&mut self) -> Result<usize, VirtioNetError> {
        let mut posted = 0usize;
        for (buf_idx, buf) in self.rx_buffers.iter().enumerate() {
            if self.rx_posted.values().any(|&b| b == buf_idx) {
                continue;
            }
            if self.rx.free_descriptors() == 0 {
                break;
            }
            let seg = SgSegment {
                addr: buf.as_ptr() as u64,
                len: buf.len() as u32,
                device_writable: true,
            };
            let head = self.rx.push_chain(&[seg])?;
            self.rx_posted.insert(head, buf_idx);
            posted += 1;
        }
        Ok(posted)
    }
}

pub struct VirtioNetDevice {
    pub features: u64,
    pub mac: [u8; 6],
    pub mtu: u16,
    pub link_up: bool,
    pub promiscuous: bool,
    pub max_queue_pairs: u16,
    pub active_queue_pairs: u16,
    pub queues: Vec<QueuePair>,
    pub ctrlq: SplitVirtqueue,
    pub rss: RssConfig,
    pub loopback_mode: bool,
    pub next_tx_id: AtomicU64,
}

impl VirtioNetDevice {
    pub fn new(
        features: u64,
        mac: [u8; 6],
        max_queue_pairs: u16,
        loopback_mode: bool,
    ) -> Result<Self, VirtioNetError> {
        let max_qp = max_queue_pairs.clamp(1, MAX_QUEUE_PAIRS as u16);
        let ctrl_q_idx = max_qp * 2;
        let ctrlq = SplitVirtqueue::new(ctrl_q_idx, DEFAULT_QUEUE_SIZE as u16)?;

        let mut queues = Vec::with_capacity(max_qp as usize);
        for idx in 0..max_qp {
            queues.push(QueuePair::new(idx, DEFAULT_QUEUE_SIZE as u16)?);
        }

        let active_qp = if (features & VIRTIO_NET_F_MQ) != 0 {
            max_qp
        } else {
            1
        };

        Ok(Self {
            features,
            mac,
            mtu: 1500,
            link_up: true,
            promiscuous: false,
            max_queue_pairs: max_qp,
            active_queue_pairs: active_qp,
            queues,
            ctrlq,
            rss: RssConfig::new(active_qp),
            loopback_mode,
            next_tx_id: AtomicU64::new(1),
        })
    }

    pub fn has_feature(&self, feat: u64) -> bool {
        (self.features & feat) != 0
    }

    /// Submit a control virtqueue command (`class`, `cmd`, `payload`).
    pub fn send_ctrl_command(
        &mut self,
        class: u8,
        cmd: u8,
        payload: &[u8],
    ) -> Result<(), VirtioNetError> {
        if !self.has_feature(VIRTIO_NET_F_CTRL_VQ) {
            return Err(VirtioNetError::UnsupportedFeature);
        }
        let mut req = Vec::with_capacity(2 + payload.len());
        req.push(class);
        req.push(cmd);
        req.extend_from_slice(payload);
        let mut ack = [VIRTIO_NET_ERR; 1];

        let segs = [
            SgSegment {
                addr: req.as_ptr() as u64,
                len: req.len() as u32,
                device_writable: false,
            },
            SgSegment {
                addr: ack.as_mut_ptr() as u64,
                len: 1,
                device_writable: true,
            },
        ];
        let head = self.ctrlq.push_chain(&segs)?;
        // Simulate synchronous device completion
        ack[0] = VIRTIO_NET_OK;
        self.ctrlq.complete_used(head, 1);
        let _ = self.ctrlq.pop_used();

        if ack[0] != VIRTIO_NET_OK {
            return Err(VirtioNetError::ControlCommandFailed);
        }
        Ok(())
    }

    /// Configure active multi-queue pairs via `VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET`.
    pub fn set_queue_pairs(&mut self, pairs: u16) -> Result<(), VirtioNetError> {
        if !self.has_feature(VIRTIO_NET_F_MQ) {
            return Err(VirtioNetError::UnsupportedFeature);
        }
        if pairs == 0 || pairs > self.max_queue_pairs {
            return Err(VirtioNetError::QueueOutOfRange);
        }
        let payload = pairs.to_le_bytes();
        self.send_ctrl_command(
            VIRTIO_NET_CTRL_MQ,
            VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET,
            &payload,
        )?;
        self.active_queue_pairs = pairs;
        self.rss.set_num_queues(pairs);
        Ok(())
    }

    /// Configure RSS hash types and key via `VIRTIO_NET_CTRL_MQ_RSS_CONFIG`.
    pub fn configure_rss(
        &mut self,
        hash_types: u32,
        key: [u8; RSS_KEY_LEN],
    ) -> Result<(), VirtioNetError> {
        if !self.has_feature(VIRTIO_NET_F_RSS) {
            return Err(VirtioNetError::UnsupportedFeature);
        }
        let mut payload = Vec::with_capacity(4 + RSS_KEY_LEN);
        payload.extend_from_slice(&hash_types.to_le_bytes());
        payload.extend_from_slice(&key);
        self.send_ctrl_command(VIRTIO_NET_CTRL_MQ, VIRTIO_NET_CTRL_MQ_RSS_CONFIG, &payload)?;
        self.rss.hash_types = hash_types;
        self.rss.key = key;
        Ok(())
    }

    pub fn set_promiscuous(&mut self, enable: bool) -> Result<(), VirtioNetError> {
        if !self.has_feature(VIRTIO_NET_F_CTRL_RX) {
            return Err(VirtioNetError::UnsupportedFeature);
        }
        let val = [u8::from(enable)];
        self.send_ctrl_command(VIRTIO_NET_CTRL_RX, VIRTIO_NET_CTRL_RX_PROMISC, &val)?;
        self.promiscuous = enable;
        Ok(())
    }

    /// Transmit a packet on `queue_idx`, applying TSO/GSO segmentation or checksum offload.
    pub fn transmit(
        &mut self,
        queue_idx: usize,
        hdr: &VirtioNetHdrV1,
        frame: &[u8],
    ) -> Result<usize, VirtioNetError> {
        if !self.link_up {
            return Err(VirtioNetError::LinkDown);
        }
        if queue_idx >= self.active_queue_pairs as usize {
            return Err(VirtioNetError::QueueOutOfRange);
        }

        let gso_base = hdr.gso_type & !VIRTIO_NET_HDR_GSO_ECN;
        let hw_tso = match gso_base {
            VIRTIO_NET_HDR_GSO_NONE => true,
            VIRTIO_NET_HDR_GSO_TCPV4 => self.has_feature(VIRTIO_NET_F_HOST_TSO4),
            VIRTIO_NET_HDR_GSO_TCPV6 => self.has_feature(VIRTIO_NET_F_HOST_TSO6),
            VIRTIO_NET_HDR_GSO_UDP_L4 | VIRTIO_NET_HDR_GSO_UDP => {
                self.has_feature(VIRTIO_NET_F_HOST_UFO)
            }
            _ => false,
        };

        // If hardware TSO is not negotiated or loopback is active, segment in software
        let frames_to_send: Vec<(VirtioNetHdrV1, Vec<u8>)> = if !hw_tso || self.loopback_mode {
            let segs = segment_gso_packet(hdr, frame)?;
            let seg_count = segs.len();
            if seg_count > 1 {
                self.queues[queue_idx].stats.tx_tso_segments += seg_count as u64;
            }
            segs.into_iter()
                .map(|s| (VirtioNetHdrV1::default(), s))
                .collect()
        } else {
            let mut buf = frame.to_vec();
            if !self.has_feature(VIRTIO_NET_F_CSUM) {
                complete_partial_checksum(hdr, &mut buf)?;
            } else if (hdr.flags & VIRTIO_NET_HDR_F_NEEDS_CSUM) != 0 {
                self.queues[queue_idx].stats.tx_csum_offloaded += 1;
            }
            vec![(*hdr, buf)]
        };

        let sent_count = frames_to_send.len();
        for (seg_hdr, seg_frame) in frames_to_send {
            let mut wire = Vec::with_capacity(VirtioNetHdrV1::SIZE + seg_frame.len());
            wire.extend_from_slice(&seg_hdr.to_bytes());
            wire.extend_from_slice(&seg_frame);

            let seg = SgSegment {
                addr: wire.as_ptr() as u64,
                len: wire.len() as u32,
                device_writable: false,
            };
            let qp = &mut self.queues[queue_idx];
            let head = qp.tx.push_chain(&[seg])?;
            let frame_len = seg_frame.len() as u64;
            qp.tx_inflight.insert(head, wire);
            qp.stats.tx_packets += 1;
            qp.stats.tx_bytes += frame_len;
            let _ = self.next_tx_id.fetch_add(1, Ordering::Relaxed);

            if self.loopback_mode {
                // Complete TX descriptor immediately and steer into RSS RX queue
                qp.tx.complete_used(head, 0);
                let _ = qp.tx.pop_used();
                qp.tx_inflight.remove(&head);

                let rx_q = self.rss.select_queue(&seg_frame);
                let (hash_val, hash_rep) = self.rss.compute_hash(&seg_frame);
                let mut rx_hdr = VirtioNetHdrV1Hash {
                    hdr: VirtioNetHdrV1 {
                        flags: if self.has_feature(VIRTIO_NET_F_GUEST_CSUM) {
                            VIRTIO_NET_HDR_F_DATA_VALID
                        } else {
                            0
                        },
                        num_buffers: 1,
                        ..VirtioNetHdrV1::default()
                    },
                    hash_value: hash_val,
                    hash_report: hash_rep,
                    padding_reserved: 0,
                };
                if !self.has_feature(VIRTIO_NET_F_HASH_REPORT) {
                    rx_hdr.hash_value = 0;
                    rx_hdr.hash_report = VIRTIO_NET_HASH_REPORT_NONE;
                }
                self.inject_rx(rx_q, rx_hdr, seg_frame)?;
            }
        }

        Ok(sent_count)
    }

    /// Transmit using RSS flow steering to pick the TX queue automatically.
    pub fn transmit_auto(
        &mut self,
        hdr: &VirtioNetHdrV1,
        frame: &[u8],
    ) -> Result<(usize, usize), VirtioNetError> {
        let q = self.rss.select_queue(frame);
        let segs = self.transmit(q, hdr, frame)?;
        Ok((q, segs))
    }

    /// Inject a received frame into `queue_idx` (used by loopback and device RX completion).
    pub fn inject_rx(
        &mut self,
        queue_idx: usize,
        hdr: VirtioNetHdrV1Hash,
        frame: Vec<u8>,
    ) -> Result<(), VirtioNetError> {
        if queue_idx >= self.active_queue_pairs as usize {
            return Err(VirtioNetError::QueueOutOfRange);
        }
        let qp = &mut self.queues[queue_idx];
        if let Some((&head, &buf_idx)) = qp.rx_posted.iter().next() {
            let copy_len = frame.len().min(qp.rx_buffers[buf_idx].len());
            qp.rx_buffers[buf_idx][..copy_len].copy_from_slice(&frame[..copy_len]);
            qp.rx.complete_used(head, copy_len as u32);
            let _ = qp.rx.pop_used();
            qp.rx_posted.remove(&head);
            qp.refill_rx()?;
        }
        qp.rx_ready.push((hdr, frame));
        Ok(())
    }

    /// Poll one received packet from `queue_idx`, verifying checksum if needed.
    pub fn poll_rx(
        &mut self,
        queue_idx: usize,
    ) -> Result<Option<(VirtioNetHdrV1Hash, Vec<u8>)>, VirtioNetError> {
        if queue_idx >= self.active_queue_pairs as usize {
            return Err(VirtioNetError::QueueOutOfRange);
        }
        let qp = &mut self.queues[queue_idx];
        if qp.rx_ready.is_empty() {
            return Ok(None);
        }
        let (hdr, frame) = qp.rx_ready.remove(0);
        if !verify_rx_checksum(&hdr.hdr, &frame)? {
            qp.stats.drops += 1;
            return Err(VirtioNetError::ChecksumMismatch);
        }
        qp.stats.rx_csum_verified += 1;
        qp.stats.rx_packets += 1;
        qp.stats.rx_bytes += frame.len() as u64;
        Ok(Some((hdr, frame)))
    }

    /// Aggregate statistics across all active queue pairs.
    pub fn aggregate_stats(&self) -> QueueStats {
        let mut total = QueueStats::default();
        for qp in self.queues.iter().take(self.active_queue_pairs as usize) {
            total.rx_packets += qp.stats.rx_packets;
            total.rx_bytes += qp.stats.rx_bytes;
            total.tx_packets += qp.stats.tx_packets;
            total.tx_bytes += qp.stats.tx_bytes;
            total.tx_tso_segments += qp.stats.tx_tso_segments;
            total.tx_csum_offloaded += qp.stats.tx_csum_offloaded;
            total.rx_csum_verified += qp.stats.rx_csum_verified;
            total.drops += qp.stats.drops;
        }
        total
    }
}

pub static VIRTIO_NET_DEVICE: TicketLock<Option<VirtioNetDevice>> = TicketLock::new(None);

// ---------------------------------------------------------------------------
// Self-Test Suite
// ---------------------------------------------------------------------------

fn build_test_tcp_v4_frame(
    src_ip: [u8; 4],
    dst_ip: [u8; 4],
    src_port: u16,
    dst_port: u16,
    seq: u32,
    payload_len: usize,
) -> Vec<u8> {
    let total_ip_len = (20 + 20 + payload_len) as u16;
    let mut frame = vec![0u8; ETH_HDR_LEN + 20 + 20 + payload_len];

    // Ethernet header
    frame[0..6].copy_from_slice(&[0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
    frame[6..12].copy_from_slice(&[0x52, 0x54, 0x00, 0xAB, 0xCD, 0xEF]);
    frame[12..14].copy_from_slice(&ETH_TYPE_IPV4.to_be_bytes());

    // IPv4 header
    let l3 = ETH_HDR_LEN;
    frame[l3] = 0x45;
    frame[l3 + 2..l3 + 4].copy_from_slice(&total_ip_len.to_be_bytes());
    frame[l3 + 4..l3 + 6].copy_from_slice(&0x1000u16.to_be_bytes());
    frame[l3 + 8] = 64;
    frame[l3 + 9] = IP_PROTO_TCP;
    frame[l3 + 12..l3 + 16].copy_from_slice(&src_ip);
    frame[l3 + 16..l3 + 20].copy_from_slice(&dst_ip);
    let ip_csum = finalize_checksum(ones_complement_sum(&frame[l3..l3 + 20], 0));
    frame[l3 + 10..l3 + 12].copy_from_slice(&ip_csum.to_be_bytes());

    // TCP header
    let l4 = l3 + 20;
    frame[l4..l4 + 2].copy_from_slice(&src_port.to_be_bytes());
    frame[l4 + 2..l4 + 4].copy_from_slice(&dst_port.to_be_bytes());
    frame[l4 + 4..l4 + 8].copy_from_slice(&seq.to_be_bytes());
    frame[l4 + 12] = 0x50; // data offset = 5 (20 bytes)
    frame[l4 + 13] = 0x19; // FIN | PSH | ACK
    frame[l4 + 14..l4 + 16].copy_from_slice(&65535u16.to_be_bytes());

    // Payload pattern
    for (idx, byte) in frame[l4 + 20..].iter_mut().enumerate() {
        *byte = (idx & 0xFF) as u8;
    }
    frame
}

pub fn self_test() -> Result<(), &'static str> {
    if VirtioNetHdrV1::SIZE != 12 || VirtioNetHdrV1Hash::SIZE != 20 {
        return Err("virtio-net: unexpected virtio_net_hdr_v1 size");
    }

    let features = VIRTIO_NET_F_CSUM
        | VIRTIO_NET_F_GUEST_CSUM
        | VIRTIO_NET_F_MAC
        | VIRTIO_NET_F_HOST_TSO4
        | VIRTIO_NET_F_HOST_TSO6
        | VIRTIO_NET_F_HOST_UFO
        | VIRTIO_NET_F_STATUS
        | VIRTIO_NET_F_CTRL_VQ
        | VIRTIO_NET_F_CTRL_RX
        | VIRTIO_NET_F_MQ
        | VIRTIO_NET_F_HASH_REPORT
        | VIRTIO_NET_F_RSS
        | VIRTIO_F_VERSION_1;

    let mac = [0x52, 0x54, 0x00, 0x11, 0x22, 0x33];
    let mut dev = VirtioNetDevice::new(features, mac, 4, true).map_err(|e| e.as_str())?;
    dev.set_queue_pairs(4).map_err(|e| e.as_str())?;
    dev.configure_rss(
        VIRTIO_NET_HASH_TYPE_TCPV4 | VIRTIO_NET_HASH_TYPE_UDPV4 | VIRTIO_NET_HASH_TYPE_TCPV6,
        DEFAULT_RSS_KEY,
    )
    .map_err(|e| e.as_str())?;
    dev.set_promiscuous(false).map_err(|e| e.as_str())?;

    // 1. Checksum offload + TSO4 segmentation test (4000B payload -> 3 segments @ MSS 1460)
    let mut large_frame =
        build_test_tcp_v4_frame([10, 0, 2, 15], [10, 0, 2, 2], 44321, 80, 1000, 4000);
    let mut hdr = prepare_tx_checksum_hdr(&mut large_frame).map_err(|e| e.as_str())?;
    hdr.gso_type = VIRTIO_NET_HDR_GSO_TCPV4;
    hdr.hdr_len = (ETH_HDR_LEN + 20 + 20) as u16;
    hdr.gso_size = 1460;

    let (tx_q, seg_count) = dev
        .transmit_auto(&hdr, &large_frame)
        .map_err(|e| e.as_str())?;
    if seg_count != 3 {
        return Err("virtio-net: TSO4 expected 3 segments for 4000B payload");
    }

    // 2. Poll RSS-steered RX queue and verify all 3 segments + checksums
    let rx_q = dev.rss.select_queue(&large_frame);
    if rx_q != tx_q {
        return Err("virtio-net: RSS queue steering mismatch");
    }

    let mut received = 0usize;
    while let Some((rx_hdr, rx_frame)) = dev.poll_rx(rx_q).map_err(|e| e.as_str())? {
        if rx_hdr.hash_report != VIRTIO_NET_HASH_REPORT_TCPV4 {
            return Err("virtio-net: unexpected RSS hash report type");
        }
        // Verify software checksum even if DATA_VALID was set
        let mut check_hdr = rx_hdr.hdr;
        check_hdr.flags &= !VIRTIO_NET_HDR_F_DATA_VALID;
        if !verify_rx_checksum(&check_hdr, &rx_frame).map_err(|e| e.as_str())? {
            return Err("virtio-net: segmented TSO4 frame failed checksum verification");
        }
        received += 1;
    }
    if received != 3 {
        return Err("virtio-net: expected 3 RX segments after TSO4 loopback");
    }

    // 3. Verify no descriptor leaks across all queues
    for qp in &dev.queues {
        if qp.tx.free_descriptors() != DEFAULT_QUEUE_SIZE as u16 {
            return Err("virtio-net: TX virtqueue descriptor leak");
        }
    }
    if dev.ctrlq.free_descriptors() != DEFAULT_QUEUE_SIZE as u16 {
        return Err("virtio-net: control virtqueue descriptor leak");
    }

    Ok(())
}
