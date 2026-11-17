//! Network stack primitives (Ethernet, ARP, IPv4, UDP, TCP) and VirtIO-Net driver.
#![allow(dead_code)]

use alloc::vec::Vec;
use core::fmt;
use core::ptr::{read_volatile, write_volatile};

use crate::mm::{alloc_frame, stats, SpinLock};
use crate::pci::{PciClass, PciDevice};

/// 6-byte IEEE 802 MAC address.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct MacAddr(pub [u8; 6]);

impl MacAddr {
    pub const ZERO: Self = Self([0, 0, 0, 0, 0, 0]);
    pub const BROADCAST: Self = Self([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

    pub const fn new(bytes: [u8; 6]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 6] {
        &self.0
    }

    pub const fn is_broadcast(&self) -> bool {
        self.0[0] == 0xFF
            && self.0[1] == 0xFF
            && self.0[2] == 0xFF
            && self.0[3] == 0xFF
            && self.0[4] == 0xFF
            && self.0[5] == 0xFF
    }

    pub const fn is_multicast(&self) -> bool {
        (self.0[0] & 0x01) != 0
    }
}

impl fmt::Debug for MacAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5]
        )
    }
}

/// 4-byte IPv4 address.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Ipv4Addr(pub [u8; 4]);

impl Ipv4Addr {
    pub const UNSPECIFIED: Self = Self([0, 0, 0, 0]);
    pub const BROADCAST: Self = Self([255, 255, 255, 255]);
    pub const LOCALHOST: Self = Self([127, 0, 0, 1]);

    pub const fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        Self([a, b, c, d])
    }

    pub const fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }

    pub const fn to_be_u32(&self) -> u32 {
        u32::from_be_bytes(self.0)
    }

    pub const fn from_be_u32(val: u32) -> Self {
        Self(val.to_be_bytes())
    }
}

impl fmt::Debug for Ipv4Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}.{}", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

/// Ethernet II frame header (14 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EthernetHeader {
    pub dst: MacAddr,
    pub src: MacAddr,
    pub ethertype: u16,
}

impl EthernetHeader {
    pub const SIZE: usize = 14;
    pub const ETHERTYPE_IPV4: u16 = 0x0800;
    pub const ETHERTYPE_ARP: u16 = 0x0806;
    pub const ETHERTYPE_IPV6: u16 = 0x86DD;

    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        let mut dst = [0u8; 6];
        let mut src = [0u8; 6];
        dst.copy_from_slice(&buf[0..6]);
        src.copy_from_slice(&buf[6..12]);
        let ethertype = u16::from_be_bytes([buf[12], buf[13]]);
        Some(Self {
            dst: MacAddr(dst),
            src: MacAddr(src),
            ethertype,
        })
    }

    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < Self::SIZE {
            return None;
        }
        buf[0..6].copy_from_slice(&self.dst.0);
        buf[6..12].copy_from_slice(&self.src.0);
        buf[12..14].copy_from_slice(&self.ethertype.to_be_bytes());
        Some(Self::SIZE)
    }
}

/// ARP packet for Ethernet + IPv4 (28 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArpPacket {
    pub htype: u16,
    pub ptype: u16,
    pub hlen: u8,
    pub plen: u8,
    pub oper: u16,
    pub sha: MacAddr,
    pub spa: Ipv4Addr,
    pub tha: MacAddr,
    pub tpa: Ipv4Addr,
}

impl ArpPacket {
    pub const SIZE: usize = 28;
    pub const HTYPE_ETHERNET: u16 = 1;
    pub const PTYPE_IPV4: u16 = 0x0800;
    pub const OP_REQUEST: u16 = 1;
    pub const OP_REPLY: u16 = 2;

    pub const fn new_request(sha: MacAddr, spa: Ipv4Addr, tpa: Ipv4Addr) -> Self {
        Self {
            htype: Self::HTYPE_ETHERNET,
            ptype: Self::PTYPE_IPV4,
            hlen: 6,
            plen: 4,
            oper: Self::OP_REQUEST,
            sha,
            spa,
            tha: MacAddr::ZERO,
            tpa,
        }
    }

    pub const fn new_reply(sha: MacAddr, spa: Ipv4Addr, tha: MacAddr, tpa: Ipv4Addr) -> Self {
        Self {
            htype: Self::HTYPE_ETHERNET,
            ptype: Self::PTYPE_IPV4,
            hlen: 6,
            plen: 4,
            oper: Self::OP_REPLY,
            sha,
            spa,
            tha,
            tpa,
        }
    }

    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        let htype = u16::from_be_bytes([buf[0], buf[1]]);
        let ptype = u16::from_be_bytes([buf[2], buf[3]]);
        let hlen = buf[4];
        let plen = buf[5];
        let oper = u16::from_be_bytes([buf[6], buf[7]]);
        if hlen != 6 || plen != 4 {
            return None;
        }
        let mut sha = [0u8; 6];
        let mut spa = [0u8; 4];
        let mut tha = [0u8; 6];
        let mut tpa = [0u8; 4];
        sha.copy_from_slice(&buf[8..14]);
        spa.copy_from_slice(&buf[14..18]);
        tha.copy_from_slice(&buf[18..24]);
        tpa.copy_from_slice(&buf[24..28]);
        Some(Self {
            htype,
            ptype,
            hlen,
            plen,
            oper,
            sha: MacAddr(sha),
            spa: Ipv4Addr(spa),
            tha: MacAddr(tha),
            tpa: Ipv4Addr(tpa),
        })
    }

    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < Self::SIZE {
            return None;
        }
        buf[0..2].copy_from_slice(&self.htype.to_be_bytes());
        buf[2..4].copy_from_slice(&self.ptype.to_be_bytes());
        buf[4] = self.hlen;
        buf[5] = self.plen;
        buf[6..8].copy_from_slice(&self.oper.to_be_bytes());
        buf[8..14].copy_from_slice(&self.sha.0);
        buf[14..18].copy_from_slice(&self.spa.0);
        buf[18..24].copy_from_slice(&self.tha.0);
        buf[24..28].copy_from_slice(&self.tpa.0);
        Some(Self::SIZE)
    }
}

/// IPv4 header (minimum 20 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ipv4Header {
    pub version_ihl: u8,
    pub dscp_ecn: u8,
    pub total_length: u16,
    pub identification: u16,
    pub flags_fragment: u16,
    pub ttl: u8,
    pub protocol: u8,
    pub checksum: u16,
    pub src: Ipv4Addr,
    pub dst: Ipv4Addr,
}

impl Ipv4Header {
    pub const MIN_SIZE: usize = 20;
    pub const PROTO_ICMP: u8 = 1;
    pub const PROTO_TCP: u8 = 6;
    pub const PROTO_UDP: u8 = 17;

    pub fn new(src: Ipv4Addr, dst: Ipv4Addr, protocol: u8, payload_len: u16) -> Self {
        Self {
            version_ihl: 0x45,
            dscp_ecn: 0,
            total_length: (Self::MIN_SIZE as u16).saturating_add(payload_len),
            identification: 0,
            flags_fragment: 0x4000, // Don't Fragment
            ttl: 64,
            protocol,
            checksum: 0,
            src,
            dst,
        }
    }

    pub const fn version(&self) -> u8 {
        self.version_ihl >> 4
    }

    pub const fn ihl_bytes(&self) -> usize {
        ((self.version_ihl & 0x0F) as usize) * 4
    }

    /// Compute RFC 1071 16-bit one's complement checksum over `data`.
    pub fn compute_checksum(data: &[u8]) -> u16 {
        let mut sum: u32 = 0;
        let mut i = 0;
        while i + 1 < data.len() {
            let word = u16::from_be_bytes([data[i], data[i + 1]]) as u32;
            sum = sum.wrapping_add(word);
            i += 2;
        }
        if i < data.len() {
            sum = sum.wrapping_add((data[i] as u32) << 8);
        }
        while (sum >> 16) != 0 {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        !(sum as u16)
    }

    pub fn verify_checksum(header_bytes: &[u8]) -> bool {
        if header_bytes.len() < Self::MIN_SIZE {
            return false;
        }
        Self::compute_checksum(header_bytes) == 0
    }

    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::MIN_SIZE {
            return None;
        }
        let version_ihl = buf[0];
        if (version_ihl >> 4) != 4 {
            return None;
        }
        let ihl = ((version_ihl & 0x0F) as usize) * 4;
        if ihl < Self::MIN_SIZE || buf.len() < ihl {
            return None;
        }
        let mut src = [0u8; 4];
        let mut dst = [0u8; 4];
        src.copy_from_slice(&buf[12..16]);
        dst.copy_from_slice(&buf[16..20]);
        Some(Self {
            version_ihl,
            dscp_ecn: buf[1],
            total_length: u16::from_be_bytes([buf[2], buf[3]]),
            identification: u16::from_be_bytes([buf[4], buf[5]]),
            flags_fragment: u16::from_be_bytes([buf[6], buf[7]]),
            ttl: buf[8],
            protocol: buf[9],
            checksum: u16::from_be_bytes([buf[10], buf[11]]),
            src: Ipv4Addr(src),
            dst: Ipv4Addr(dst),
        })
    }

    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < Self::MIN_SIZE {
            return None;
        }
        buf[0] = self.version_ihl;
        buf[1] = self.dscp_ecn;
        buf[2..4].copy_from_slice(&self.total_length.to_be_bytes());
        buf[4..6].copy_from_slice(&self.identification.to_be_bytes());
        buf[6..8].copy_from_slice(&self.flags_fragment.to_be_bytes());
        buf[8] = self.ttl;
        buf[9] = self.protocol;
        buf[10] = 0;
        buf[11] = 0;
        buf[12..16].copy_from_slice(&self.src.0);
        buf[16..20].copy_from_slice(&self.dst.0);
        let csum = Self::compute_checksum(&buf[..Self::MIN_SIZE]);
        buf[10..12].copy_from_slice(&csum.to_be_bytes());
        Some(Self::MIN_SIZE)
    }
}

/// UDP header (8 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdpHeader {
    pub src_port: u16,
    pub dst_port: u16,
    pub length: u16,
    pub checksum: u16,
}

impl UdpHeader {
    pub const SIZE: usize = 8;

    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            src_port: u16::from_be_bytes([buf[0], buf[1]]),
            dst_port: u16::from_be_bytes([buf[2], buf[3]]),
            length: u16::from_be_bytes([buf[4], buf[5]]),
            checksum: u16::from_be_bytes([buf[6], buf[7]]),
        })
    }

    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < Self::SIZE {
            return None;
        }
        buf[0..2].copy_from_slice(&self.src_port.to_be_bytes());
        buf[2..4].copy_from_slice(&self.dst_port.to_be_bytes());
        buf[4..6].copy_from_slice(&self.length.to_be_bytes());
        buf[6..8].copy_from_slice(&self.checksum.to_be_bytes());
        Some(Self::SIZE)
    }
}

/// TCP header (minimum 20 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcpHeader {
    pub src_port: u16,
    pub dst_port: u16,
    pub seq_num: u32,
    pub ack_num: u32,
    pub data_offset_flags: u16,
    pub window_size: u16,
    pub checksum: u16,
    pub urgent_ptr: u16,
}

impl TcpHeader {
    pub const MIN_SIZE: usize = 20;
    pub const FLAG_FIN: u16 = 0x0001;
    pub const FLAG_SYN: u16 = 0x0002;
    pub const FLAG_RST: u16 = 0x0004;
    pub const FLAG_PSH: u16 = 0x0008;
    pub const FLAG_ACK: u16 = 0x0010;
    pub const FLAG_URG: u16 = 0x0020;

    pub const fn data_offset_bytes(&self) -> usize {
        (((self.data_offset_flags >> 12) & 0x0F) as usize) * 4
    }

    pub const fn flags(&self) -> u16 {
        self.data_offset_flags & 0x01FF
    }

    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::MIN_SIZE {
            return None;
        }
        let data_offset_flags = u16::from_be_bytes([buf[12], buf[13]]);
        let hdr_len = (((data_offset_flags >> 12) & 0x0F) as usize) * 4;
        if hdr_len < Self::MIN_SIZE || buf.len() < hdr_len {
            return None;
        }
        Some(Self {
            src_port: u16::from_be_bytes([buf[0], buf[1]]),
            dst_port: u16::from_be_bytes([buf[2], buf[3]]),
            seq_num: u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]),
            ack_num: u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]),
            data_offset_flags,
            window_size: u16::from_be_bytes([buf[14], buf[15]]),
            checksum: u16::from_be_bytes([buf[16], buf[17]]),
            urgent_ptr: u16::from_be_bytes([buf[18], buf[19]]),
        })
    }

    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < Self::MIN_SIZE {
            return None;
        }
        buf[0..2].copy_from_slice(&self.src_port.to_be_bytes());
        buf[2..4].copy_from_slice(&self.dst_port.to_be_bytes());
        buf[4..8].copy_from_slice(&self.seq_num.to_be_bytes());
        buf[8..12].copy_from_slice(&self.ack_num.to_be_bytes());
        buf[12..14].copy_from_slice(&self.data_offset_flags.to_be_bytes());
        buf[14..16].copy_from_slice(&self.window_size.to_be_bytes());
        buf[16..18].copy_from_slice(&self.checksum.to_be_bytes());
        buf[18..20].copy_from_slice(&self.urgent_ptr.to_be_bytes());
        Some(Self::MIN_SIZE)
    }
}

/// Network interface device abstraction.
pub trait NetDevice: Send + Sync {
    fn mac_addr(&self) -> MacAddr;
    fn send_frame(&mut self, frame: &[u8]) -> Result<(), &'static str>;
    fn recv_frame(&mut self, buf: &mut [u8]) -> Option<usize>;
}

// VirtIO-Net PCI constants
const VIRTIO_VENDOR_ID: u16 = 0x1AF4;
const VIRTIO_NET_LEGACY_DEVICE_ID: u16 = 0x1000;
const VIRTIO_NET_MODERN_DEVICE_ID: u16 = 0x1041;

const VIRTIO_STATUS_ACKNOWLEDGE: u8 = 1;
const VIRTIO_STATUS_DRIVER: u8 = 2;
const VIRTIO_STATUS_DRIVER_OK: u8 = 4;
const VIRTIO_STATUS_FEATURES_OK: u8 = 8;

const VIRTIO_PCI_QUEUE_PFN: usize = 0x08;
const VIRTIO_PCI_QUEUE_SEL: usize = 0x0E;
const VIRTIO_PCI_STATUS: usize = 0x12;
const VIRTIO_PCI_NET_MAC_OFFSET: usize = 0x14;

/// VirtIO-Net PCI device driver skeleton.
pub struct VirtioNetDevice {
    bus: u8,
    dev: u8,
    func: u8,
    device_id: u16,
    bar0_phys: u64,
    mmio_base: u64,
    mac: MacAddr,
    rx_queue_phys: u64,
    tx_queue_phys: u64,
    initialized: bool,
}

impl VirtioNetDevice {
    /// Probe a `PciDevice` for a VirtIO network controller (`0x1000` legacy or `0x1041` modern).
    pub fn probe(dev: &PciDevice) -> Option<Self> {
        let class = dev.class();
        if class != PciClass::Network && class != PciClass::Virtio {
            return None;
        }
        if dev.vendor_id != VIRTIO_VENDOR_ID {
            return None;
        }
        if dev.device_id != VIRTIO_NET_LEGACY_DEVICE_ID
            && dev.device_id != VIRTIO_NET_MODERN_DEVICE_ID
        {
            return None;
        }

        let bar0_phys = dev.bars[0];
        let hhdm = stats().hhdm_offset;
        let mmio_base = if bar0_phys != 0 {
            bar0_phys.wrapping_add(hhdm)
        } else {
            0
        };

        let rx_queue_phys = alloc_frame().unwrap_or(0);
        let tx_queue_phys = alloc_frame().unwrap_or(0);

        let mut mac_bytes = [0x52, 0x54, 0x00, dev.bus, dev.dev, dev.func];

        if dev.device_id == VIRTIO_NET_LEGACY_DEVICE_ID && bar0_phys >= 0x10000 {
            unsafe {
                let status_ptr = (mmio_base + VIRTIO_PCI_STATUS as u64) as *mut u8;
                write_volatile(status_ptr, 0);
                write_volatile(status_ptr, VIRTIO_STATUS_ACKNOWLEDGE);
                write_volatile(status_ptr, VIRTIO_STATUS_ACKNOWLEDGE | VIRTIO_STATUS_DRIVER);

                for (i, byte) in mac_bytes.iter_mut().enumerate() {
                    *byte = read_volatile(
                        (mmio_base + (VIRTIO_PCI_NET_MAC_OFFSET + i) as u64) as *const u8,
                    );
                }

                if rx_queue_phys != 0 {
                    write_volatile((mmio_base + VIRTIO_PCI_QUEUE_SEL as u64) as *mut u16, 0);
                    write_volatile(
                        (mmio_base + VIRTIO_PCI_QUEUE_PFN as u64) as *mut u32,
                        (rx_queue_phys >> 12) as u32,
                    );
                }
                if tx_queue_phys != 0 {
                    write_volatile((mmio_base + VIRTIO_PCI_QUEUE_SEL as u64) as *mut u16, 1);
                    write_volatile(
                        (mmio_base + VIRTIO_PCI_QUEUE_PFN as u64) as *mut u32,
                        (tx_queue_phys >> 12) as u32,
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
            mac: MacAddr(mac_bytes),
            rx_queue_phys,
            tx_queue_phys,
            initialized: rx_queue_phys != 0 && tx_queue_phys != 0,
        })
    }

    pub fn pci_location(&self) -> (u8, u8, u8) {
        (self.bus, self.dev, self.func)
    }

    pub fn device_id(&self) -> u16 {
        self.device_id
    }

    pub fn bar0_phys(&self) -> u64 {
        self.bar0_phys
    }
}

impl NetDevice for VirtioNetDevice {
    fn mac_addr(&self) -> MacAddr {
        self.mac
    }

    fn send_frame(&mut self, frame: &[u8]) -> Result<(), &'static str> {
        if frame.is_empty() || frame.len() > 1514 {
            return Err("virtio-net: invalid Ethernet frame length");
        }
        if !self.initialized {
            return Err("virtio-net: virtqueues not initialized");
        }
        Err("virtio-net: TX ring submission not yet implemented in skeleton")
    }

    fn recv_frame(&mut self, _buf: &mut [u8]) -> Option<usize> {
        None
    }
}

/// Interface IPv4 configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterfaceConfig {
    pub mac: MacAddr,
    pub ip: Ipv4Addr,
    pub netmask: Ipv4Addr,
    pub gateway: Ipv4Addr,
}

impl Default for InterfaceConfig {
    fn default() -> Self {
        Self {
            mac: MacAddr::ZERO,
            ip: Ipv4Addr::new(10, 0, 2, 15),
            netmask: Ipv4Addr::new(255, 255, 255, 0),
            gateway: Ipv4Addr::new(10, 0, 2, 2),
        }
    }
}

/// ARP cache entry mapping an IPv4 address to a MAC address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArpEntry {
    pub ip: Ipv4Addr,
    pub mac: MacAddr,
}

/// Minimal kernel network stack holding interface config, ARP cache, and probed NICs.
pub struct NetStack {
    pub config: InterfaceConfig,
    pub arp_cache: Vec<ArpEntry>,
    pub nics: Vec<VirtioNetDevice>,
}

impl NetStack {
    pub const fn new() -> Self {
        Self {
            config: InterfaceConfig {
                mac: MacAddr::ZERO,
                ip: Ipv4Addr::new(10, 0, 2, 15),
                netmask: Ipv4Addr::new(255, 255, 255, 0),
                gateway: Ipv4Addr::new(10, 0, 2, 2),
            },
            arp_cache: Vec::new(),
            nics: Vec::new(),
        }
    }

    /// Probe PCI devices for VirtIO-Net NICs and register them. Returns count found.
    pub fn probe_nics(&mut self, devs: &[PciDevice]) -> usize {
        let mut found = 0usize;
        for dev in devs {
            if let Some(nic) = VirtioNetDevice::probe(dev) {
                if self.nics.is_empty() {
                    self.config.mac = nic.mac_addr();
                }
                self.nics.push(nic);
                found += 1;
            }
        }
        found
    }

    pub fn nic_count(&self) -> usize {
        self.nics.len()
    }

    pub fn lookup_arp(&self, ip: Ipv4Addr) -> Option<MacAddr> {
        self.arp_cache
            .iter()
            .find(|entry| entry.ip == ip)
            .map(|entry| entry.mac)
    }

    pub fn insert_arp(&mut self, ip: Ipv4Addr, mac: MacAddr) {
        if let Some(entry) = self.arp_cache.iter_mut().find(|e| e.ip == ip) {
            entry.mac = mac;
        } else {
            self.arp_cache.push(ArpEntry { ip, mac });
        }
    }

    /// Process an incoming Ethernet frame (updates ARP cache on ARP packets).
    pub fn handle_frame(&mut self, frame: &[u8]) {
        let Some(eth) = EthernetHeader::parse(frame) else {
            return;
        };
        let payload = &frame[EthernetHeader::SIZE..];
        if eth.ethertype == EthernetHeader::ETHERTYPE_ARP {
            if let Some(arp) = ArpPacket::parse(payload) {
                self.insert_arp(arp.spa, arp.sha);
            }
        }
    }
}

static NET_STACK: SpinLock<NetStack> = SpinLock::new(NetStack::new());

/// Probe PCI devices into the global `NetStack` and return the number of NICs found.
pub fn probe_nics(devs: &[PciDevice]) -> usize {
    NET_STACK.lock().probe_nics(devs)
}

/// Access the global `NetStack` under its spinlock.
pub fn with_net_stack<R>(f: impl FnOnce(&mut NetStack) -> R) -> R {
    let mut guard = NET_STACK.lock();
    f(&mut guard)
}
