//! Production IPv6 (NDP/SLAAC), TCP Congestion Control (CUBIC/BBR),
//! and WireGuard ChaCha20-Poly1305 AEAD Offload Engine.
//!
//! Designed for `no_std` x86_64 kernel execution with small stack frames (<2 KiB)
//! and pure integer / fixed-point arithmetic (no floating-point instructions).
#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::net::MacAddr;

// ============================================================================
// 1. IPv6 Addressing, Header, ICMPv6, NDP & SLAAC
// ============================================================================

/// 16-byte IPv6 address.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Ipv6Addr(pub [u8; 16]);

impl Ipv6Addr {
    pub const UNSPECIFIED: Self = Self([0; 16]);
    pub const LOOPBACK: Self = Self([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    pub const ALL_NODES_MULTICAST: Self =
        Self([0xff, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    pub const ALL_ROUTERS_MULTICAST: Self =
        Self([0xff, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]);

    pub const fn new(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn from_segments(segs: [u16; 8]) -> Self {
        let mut b = [0u8; 16];
        let mut i = 0;
        while i < 8 {
            let be = segs[i].to_be_bytes();
            b[i * 2] = be[0];
            b[i * 2 + 1] = be[1];
            i += 1;
        }
        Self(b)
    }

    pub const fn segments(&self) -> [u16; 8] {
        let mut segs = [0u16; 8];
        let mut i = 0;
        while i < 8 {
            segs[i] = u16::from_be_bytes([self.0[i * 2], self.0[i * 2 + 1]]);
            i += 1;
        }
        segs
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    pub const fn is_unspecified(&self) -> bool {
        u128::from_be_bytes(self.0) == 0
    }

    pub const fn is_loopback(&self) -> bool {
        u128::from_be_bytes(self.0) == 1
    }

    pub const fn is_multicast(&self) -> bool {
        self.0[0] == 0xff
    }

    pub const fn is_link_local(&self) -> bool {
        self.0[0] == 0xfe && (self.0[1] & 0xc0) == 0x80
    }

    /// Generate an IPv6 address from a /64 prefix and a 48-bit MAC using Modified EUI-64 (RFC 4291).
    pub const fn from_eui64(prefix: [u8; 8], mac: MacAddr) -> Self {
        let m = mac.0;
        Self([
            prefix[0],
            prefix[1],
            prefix[2],
            prefix[3],
            prefix[4],
            prefix[5],
            prefix[6],
            prefix[7],
            m[0] ^ 0x02,
            m[1],
            m[2],
            0xff,
            0xfe,
            m[3],
            m[4],
            m[5],
        ])
    }

    /// Generate a link-local IPv6 address (`fe80::/64` + EUI-64).
    pub const fn link_local_from_mac(mac: MacAddr) -> Self {
        Self::from_eui64([0xfe, 0x80, 0, 0, 0, 0, 0, 0], mac)
    }

    /// Compute the Solicited-Node Multicast address (`ff02::1:ffXX:XXXX`) for NDP.
    pub const fn solicited_node_multicast(&self) -> Self {
        Self([
            0xff, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0xff, self.0[13], self.0[14], self.0[15],
        ])
    }
}

impl fmt::Debug for Ipv6Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.segments();
        write!(
            f,
            "{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}",
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]
        )
    }
}

/// Fixed 40-byte IPv6 header (RFC 8200).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ipv6Header {
    pub traffic_class: u8,
    pub flow_label: u32,
    pub payload_len: u16,
    pub next_header: u8,
    pub hop_limit: u8,
    pub src: Ipv6Addr,
    pub dst: Ipv6Addr,
}

impl Ipv6Header {
    pub const SIZE: usize = 40;
    pub const NEXT_TCP: u8 = 6;
    pub const NEXT_UDP: u8 = 17;
    pub const NEXT_ICMPV6: u8 = 58;

    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        let vtf = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let version = (vtf >> 28) as u8;
        if version != 6 {
            return None;
        }
        let traffic_class = ((vtf >> 20) & 0xff) as u8;
        let flow_label = vtf & 0x000f_ffff;
        let payload_len = u16::from_be_bytes([buf[4], buf[5]]);
        let next_header = buf[6];
        let hop_limit = buf[7];
        let mut src = [0u8; 16];
        let mut dst = [0u8; 16];
        src.copy_from_slice(&buf[8..24]);
        dst.copy_from_slice(&buf[24..40]);
        Some(Self {
            traffic_class,
            flow_label,
            payload_len,
            next_header,
            hop_limit,
            src: Ipv6Addr(src),
            dst: Ipv6Addr(dst),
        })
    }

    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < Self::SIZE {
            return None;
        }
        let vtf: u32 =
            (6u32 << 28) | ((self.traffic_class as u32) << 20) | (self.flow_label & 0x000f_ffff);
        buf[0..4].copy_from_slice(&vtf.to_be_bytes());
        buf[4..6].copy_from_slice(&self.payload_len.to_be_bytes());
        buf[6] = self.next_header;
        buf[7] = self.hop_limit;
        buf[8..24].copy_from_slice(&self.src.0);
        buf[24..40].copy_from_slice(&self.dst.0);
        Some(Self::SIZE)
    }

    /// Compute the 32-bit ones-complement sum of the IPv6 pseudo-header (RFC 8200 Section 8.1).
    pub fn pseudo_header_sum(src: &Ipv6Addr, dst: &Ipv6Addr, next_header: u8, len: u32) -> u32 {
        let mut sum: u32 = 0;
        let mut i = 0;
        while i < 16 {
            sum += u16::from_be_bytes([src.0[i], src.0[i + 1]]) as u32;
            sum += u16::from_be_bytes([dst.0[i], dst.0[i + 1]]) as u32;
            i += 2;
        }
        sum += (len >> 16) & 0xffff;
        sum += len & 0xffff;
        sum += next_header as u32;
        sum
    }
}

/// Compute RFC 1071 / RFC 8200 Internet checksum over an IPv6 pseudo-header + payload.
pub fn ipv6_upper_checksum(src: &Ipv6Addr, dst: &Ipv6Addr, next_header: u8, payload: &[u8]) -> u16 {
    let mut sum = Ipv6Header::pseudo_header_sum(src, dst, next_header, payload.len() as u32);
    let mut i = 0;
    while i + 1 < payload.len() {
        sum += u16::from_be_bytes([payload[i], payload[i + 1]]) as u32;
        i += 2;
    }
    if i < payload.len() {
        sum += (payload[i] as u32) << 8;
    }
    while (sum >> 16) != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

/// ICMPv6 message types (RFC 4443 & RFC 4861).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Icmpv6Type {
    EchoRequest = 128,
    EchoReply = 129,
    RouterSolicitation = 133,
    RouterAdvertisement = 134,
    NeighborSolicitation = 135,
    NeighborAdvertisement = 136,
}

impl Icmpv6Type {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            128 => Some(Self::EchoRequest),
            129 => Some(Self::EchoReply),
            133 => Some(Self::RouterSolicitation),
            134 => Some(Self::RouterAdvertisement),
            135 => Some(Self::NeighborSolicitation),
            136 => Some(Self::NeighborAdvertisement),
            _ => None,
        }
    }
}

/// ICMPv6 common 4-byte header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Icmpv6Header {
    pub icmp_type: u8,
    pub code: u8,
    pub checksum: u16,
}

impl Icmpv6Header {
    pub const SIZE: usize = 4;

    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            icmp_type: buf[0],
            code: buf[1],
            checksum: u16::from_be_bytes([buf[2], buf[3]]),
        })
    }

    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < Self::SIZE {
            return None;
        }
        buf[0] = self.icmp_type;
        buf[1] = self.code;
        buf[2..4].copy_from_slice(&self.checksum.to_be_bytes());
        Some(Self::SIZE)
    }

    pub fn verify_checksum(src: &Ipv6Addr, dst: &Ipv6Addr, icmp_packet: &[u8]) -> bool {
        if icmp_packet.len() < Self::SIZE {
            return false;
        }
        ipv6_upper_checksum(src, dst, Ipv6Header::NEXT_ICMPV6, icmp_packet) == 0
    }
}

/// NDP Option types (RFC 4861 Section 4.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdpOption {
    SourceLinkLayer(MacAddr),
    TargetLinkLayer(MacAddr),
    PrefixInfo {
        prefix_len: u8,
        on_link: bool,
        autonomous: bool,
        valid_lifetime: u32,
        preferred_lifetime: u32,
        prefix: [u8; 8],
    },
    Mtu(u32),
}

impl NdpOption {
    pub const OPT_SRC_LL: u8 = 1;
    pub const OPT_TGT_LL: u8 = 2;
    pub const OPT_PREFIX_INFO: u8 = 3;
    pub const OPT_MTU: u8 = 5;

    /// Parse the first NDP option from `buf`, returning `(option, bytes_consumed)`.
    pub fn parse(buf: &[u8]) -> Option<(Option<Self>, usize)> {
        if buf.len() < 2 {
            return None;
        }
        let opt_type = buf[0];
        let len_units = buf[1] as usize;
        if len_units == 0 {
            return None;
        }
        let byte_len = len_units * 8;
        if buf.len() < byte_len {
            return None;
        }
        let parsed = match opt_type {
            Self::OPT_SRC_LL if byte_len >= 8 => {
                let mut mac = [0u8; 6];
                mac.copy_from_slice(&buf[2..8]);
                Some(Self::SourceLinkLayer(MacAddr(mac)))
            }
            Self::OPT_TGT_LL if byte_len >= 8 => {
                let mut mac = [0u8; 6];
                mac.copy_from_slice(&buf[2..8]);
                Some(Self::TargetLinkLayer(MacAddr(mac)))
            }
            Self::OPT_PREFIX_INFO if byte_len >= 32 => {
                let prefix_len = buf[2];
                let flags = buf[3];
                let valid_lifetime = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
                let preferred_lifetime = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);
                let mut prefix = [0u8; 8];
                prefix.copy_from_slice(&buf[16..24]);
                Some(Self::PrefixInfo {
                    prefix_len,
                    on_link: (flags & 0x80) != 0,
                    autonomous: (flags & 0x40) != 0,
                    valid_lifetime,
                    preferred_lifetime,
                    prefix,
                })
            }
            Self::OPT_MTU if byte_len >= 8 => {
                let mtu = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
                Some(Self::Mtu(mtu))
            }
            _ => None,
        };
        Some((parsed, byte_len))
    }
}

/// NDP Neighbor Cache entry state (RFC 4861 Section 7.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborState {
    Incomplete,
    Reachable,
    Stale,
    Delay,
    Probe,
}

#[derive(Debug, Clone, Copy)]
pub struct NeighborEntry {
    pub ip: Ipv6Addr,
    pub mac: MacAddr,
    pub state: NeighborState,
    pub is_router: bool,
    pub last_updated_ms: u64,
}

/// NDP Neighbor Cache.
pub struct NeighborCache {
    entries: Vec<NeighborEntry>,
    max_entries: usize,
    reachable_time_ms: u64,
}

impl NeighborCache {
    pub fn new(max_entries: usize, reachable_time_ms: u64) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
            reachable_time_ms,
        }
    }

    pub fn lookup(&self, ip: &Ipv6Addr) -> Option<&NeighborEntry> {
        self.entries.iter().find(|e| &e.ip == ip)
    }

    pub fn update(
        &mut self,
        ip: Ipv6Addr,
        mac: MacAddr,
        state: NeighborState,
        is_router: bool,
        now_ms: u64,
    ) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.ip == ip) {
            entry.mac = mac;
            entry.state = state;
            entry.is_router = is_router;
            entry.last_updated_ms = now_ms;
            return;
        }
        if self.entries.len() >= self.max_entries && !self.entries.is_empty() {
            self.entries.remove(0);
        }
        self.entries.push(NeighborEntry {
            ip,
            mac,
            state,
            is_router,
            last_updated_ms: now_ms,
        });
    }

    /// Transition `Reachable` entries older than `reachable_time_ms` to `Stale`.
    pub fn tick(&mut self, now_ms: u64) {
        for entry in &mut self.entries {
            if entry.state == NeighborState::Reachable
                && now_ms.saturating_sub(entry.last_updated_ms) >= self.reachable_time_ms
            {
                entry.state = NeighborState::Stale;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// SLAAC address lifecycle state (RFC 4862).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlaacAddrState {
    Tentative,
    Preferred,
    Deprecated,
}

#[derive(Debug, Clone, Copy)]
pub struct SlaacAddress {
    pub addr: Ipv6Addr,
    pub prefix_len: u8,
    pub state: SlaacAddrState,
    pub valid_until_s: u64,
    pub preferred_until_s: u64,
}

/// Stateless Address Autoconfiguration (SLAAC) & NDP packet builder/processor.
pub struct SlaacEngine {
    pub mac: MacAddr,
    pub link_local: SlaacAddress,
    pub global_addrs: Vec<SlaacAddress>,
    pub default_router: Option<Ipv6Addr>,
    pub link_mtu: u32,
    pub neighbors: NeighborCache,
}

impl SlaacEngine {
    pub fn new(mac: MacAddr) -> Self {
        let ll = Ipv6Addr::link_local_from_mac(mac);
        Self {
            mac,
            link_local: SlaacAddress {
                addr: ll,
                prefix_len: 64,
                state: SlaacAddrState::Tentative,
                valid_until_s: u64::MAX,
                preferred_until_s: u64::MAX,
            },
            global_addrs: Vec::new(),
            default_router: None,
            link_mtu: 1500,
            neighbors: NeighborCache::new(64, 30_000),
        }
    }

    /// Mark link-local DAD complete (no duplicate detected).
    pub fn complete_dad(&mut self) {
        if self.link_local.state == SlaacAddrState::Tentative {
            self.link_local.state = SlaacAddrState::Preferred;
        }
        for addr in &mut self.global_addrs {
            if addr.state == SlaacAddrState::Tentative {
                addr.state = SlaacAddrState::Preferred;
            }
        }
    }

    /// Build an ICMPv6 Router Solicitation (type 133) packet (16 bytes with Source LL option).
    pub fn build_router_solicitation(&self, out: &mut [u8]) -> Option<usize> {
        const RS_LEN: usize = 16;
        if out.len() < RS_LEN {
            return None;
        }
        out[0] = Icmpv6Type::RouterSolicitation as u8;
        out[1] = 0;
        out[2..4].copy_from_slice(&[0, 0]);
        out[4..8].copy_from_slice(&[0, 0, 0, 0]); // Reserved
        out[8] = NdpOption::OPT_SRC_LL;
        out[9] = 1; // 8 bytes
        out[10..16].copy_from_slice(&self.mac.0);
        let csum = ipv6_upper_checksum(
            &self.link_local.addr,
            &Ipv6Addr::ALL_ROUTERS_MULTICAST,
            Ipv6Header::NEXT_ICMPV6,
            &out[..RS_LEN],
        );
        out[2..4].copy_from_slice(&csum.to_be_bytes());
        Some(RS_LEN)
    }

    /// Build an ICMPv6 Neighbor Solicitation (type 135) packet (32 bytes with Source LL option).
    pub fn build_neighbor_solicitation(
        &self,
        target: &Ipv6Addr,
        dst: &Ipv6Addr,
        out: &mut [u8],
    ) -> Option<usize> {
        const NS_LEN: usize = 32;
        if out.len() < NS_LEN {
            return None;
        }
        out[0] = Icmpv6Type::NeighborSolicitation as u8;
        out[1] = 0;
        out[2..4].copy_from_slice(&[0, 0]);
        out[4..8].copy_from_slice(&[0, 0, 0, 0]); // Reserved
        out[8..24].copy_from_slice(&target.0);
        out[24] = NdpOption::OPT_SRC_LL;
        out[25] = 1;
        out[26..32].copy_from_slice(&self.mac.0);
        let csum = ipv6_upper_checksum(
            &self.link_local.addr,
            dst,
            Ipv6Header::NEXT_ICMPV6,
            &out[..NS_LEN],
        );
        out[2..4].copy_from_slice(&csum.to_be_bytes());
        Some(NS_LEN)
    }

    /// Build an ICMPv6 Neighbor Advertisement (type 136) packet (32 bytes with Target LL option).
    pub fn build_neighbor_advertisement(
        &self,
        target: &Ipv6Addr,
        dst: &Ipv6Addr,
        solicited: bool,
        override_flag: bool,
        out: &mut [u8],
    ) -> Option<usize> {
        const NA_LEN: usize = 32;
        if out.len() < NA_LEN {
            return None;
        }
        out[0] = Icmpv6Type::NeighborAdvertisement as u8;
        out[1] = 0;
        out[2..4].copy_from_slice(&[0, 0]);
        let mut flags: u8 = 0;
        if solicited {
            flags |= 0x40;
        }
        if override_flag {
            flags |= 0x20;
        }
        out[4] = flags;
        out[5..8].copy_from_slice(&[0, 0, 0]);
        out[8..24].copy_from_slice(&target.0);
        out[24] = NdpOption::OPT_TGT_LL;
        out[25] = 1;
        out[26..32].copy_from_slice(&self.mac.0);
        let csum = ipv6_upper_checksum(
            &self.link_local.addr,
            dst,
            Ipv6Header::NEXT_ICMPV6,
            &out[..NA_LEN],
        );
        out[2..4].copy_from_slice(&csum.to_be_bytes());
        Some(NA_LEN)
    }

    /// Process an incoming Router Advertisement (ICMPv6 type 134) and configure SLAAC prefixes.
    pub fn process_router_advertisement(
        &mut self,
        src: Ipv6Addr,
        icmp_payload: &[u8],
        now_s: u64,
        now_ms: u64,
    ) -> bool {
        // RA fixed body is 16 bytes (4-byte ICMPv6 header + 12-byte RA fields)
        if icmp_payload.len() < 16 || icmp_payload[0] != Icmpv6Type::RouterAdvertisement as u8 {
            return false;
        }
        let router_lifetime = u16::from_be_bytes([icmp_payload[6], icmp_payload[7]]);
        if router_lifetime > 0 {
            self.default_router = Some(src);
        }

        let mut offset = 16;
        while offset < icmp_payload.len() {
            let Some((opt, consumed)) = NdpOption::parse(&icmp_payload[offset..]) else {
                break;
            };
            if let Some(parsed) = opt {
                match parsed {
                    NdpOption::SourceLinkLayer(mac) => {
                        self.neighbors
                            .update(src, mac, NeighborState::Reachable, true, now_ms);
                    }
                    NdpOption::Mtu(mtu) if mtu >= 1280 => {
                        self.link_mtu = mtu;
                    }
                    NdpOption::PrefixInfo {
                        prefix_len: 64,
                        autonomous: true,
                        valid_lifetime,
                        preferred_lifetime,
                        prefix,
                        ..
                    } if valid_lifetime > 0 => {
                        let new_addr = Ipv6Addr::from_eui64(prefix, self.mac);
                        let valid_until = now_s.saturating_add(valid_lifetime as u64);
                        let pref_until = now_s.saturating_add(preferred_lifetime as u64);
                        if let Some(existing) =
                            self.global_addrs.iter_mut().find(|a| a.addr == new_addr)
                        {
                            existing.valid_until_s = valid_until;
                            existing.preferred_until_s = pref_until;
                            existing.state = if preferred_lifetime > 0 {
                                SlaacAddrState::Preferred
                            } else {
                                SlaacAddrState::Deprecated
                            };
                        } else {
                            self.global_addrs.push(SlaacAddress {
                                addr: new_addr,
                                prefix_len: 64,
                                state: SlaacAddrState::Preferred,
                                valid_until_s: valid_until,
                                preferred_until_s: pref_until,
                            });
                        }
                    }
                    _ => {}
                }
            }
            offset += consumed;
        }
        true
    }
}

// ============================================================================
// 2. TCP Congestion Control (CUBIC & BBR) - Pure Integer / Fixed-Point
// ============================================================================

/// Integer cube root: floor(cbrt(n)).
const fn icbrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    // Binary search over [0, 2_642_245] (since 2_642_246^3 > u64::MAX)
    let mut lo = 0u64;
    let mut hi = 2_642_245u64;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let cube = mid * mid * mid;
        if cube <= n {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// TCP CUBIC congestion control state (RFC 8312 / RFC 9438).
#[derive(Debug, Clone)]
pub struct CubicState {
    pub mss: u32,
    pub cwnd: u32,
    pub ssthresh: u32,
    pub w_max: u32,
    pub w_last_max: u32,
    pub epoch_start_us: u64,
    pub origin_point: u32,
    pub d_min_us: u64,
    pub k_ms: u64,
    pub tcp_cwnd: u32,
    pub ack_cnt: u32,
}

impl CubicState {
    /// Beta = 0.7 (717 / 1024)
    const BETA_NUM: u64 = 717;
    const BETA_DEN: u64 = 1024;
    /// C = 0.4 (410 / 1024)
    const C_NUM: u64 = 410;
    const C_DEN: u64 = 1024;

    pub fn new(mss: u32) -> Self {
        let init_cwnd = mss.saturating_mul(10);
        Self {
            mss,
            cwnd: init_cwnd,
            ssthresh: u32::MAX,
            w_max: 0,
            w_last_max: 0,
            epoch_start_us: 0,
            origin_point: 0,
            d_min_us: u64::MAX,
            k_ms: 0,
            tcp_cwnd: init_cwnd,
            ack_cnt: 0,
        }
    }

    pub fn on_ack(&mut self, acked_bytes: u32, rtt_us: u64, now_us: u64) {
        if rtt_us > 0 && rtt_us < self.d_min_us {
            self.d_min_us = rtt_us;
        }

        // Slow start
        if self.cwnd < self.ssthresh {
            self.cwnd = self.cwnd.saturating_add(acked_bytes.min(self.mss));
            return;
        }

        // Congestion avoidance (CUBIC)
        if self.epoch_start_us == 0 {
            self.epoch_start_us = now_us;
            if self.cwnd < self.w_max {
                // K = cbrt((W_max - cwnd) / (C * MSS)) in seconds -> * 1000 for ms
                let diff_segs = ((self.w_max - self.cwnd) as u64) / (self.mss as u64).max(1);
                // Scale by 10^9 so cbrt gives milliseconds directly:
                // K_ms = cbrt(diff_segs * 1024 * 1_000_000_000 / 410)
                let scaled = diff_segs
                    .saturating_mul(Self::C_DEN)
                    .saturating_mul(1_000_000_000)
                    / Self::C_NUM;
                self.k_ms = icbrt(scaled);
                self.origin_point = self.w_max;
            } else {
                self.k_ms = 0;
                self.origin_point = self.cwnd;
            }
            self.ack_cnt = 0;
            self.tcp_cwnd = self.cwnd;
        }

        let d_min_ms = if self.d_min_us == u64::MAX {
            0
        } else {
            self.d_min_us / 1000
        };
        let t_ms = now_us.saturating_sub(self.epoch_start_us) / 1000 + d_min_ms;
        let dt_ms = t_ms.abs_diff(self.k_ms);

        // delta_segs = C * (dt_ms / 1000)^3 = (410 * dt_ms^3) / (1024 * 10^9)
        let dt_cubed = dt_ms.saturating_mul(dt_ms).saturating_mul(dt_ms);
        let delta_bytes = (Self::C_NUM.saturating_mul(dt_cubed) / (Self::C_DEN * 1_000_000_000))
            .saturating_mul(self.mss as u64);

        let target = if t_ms < self.k_ms {
            (self.origin_point as u64).saturating_sub(delta_bytes)
        } else {
            (self.origin_point as u64).saturating_add(delta_bytes)
        };

        // Reno-friendly region
        self.ack_cnt = self.ack_cnt.saturating_add(acked_bytes);
        if self.ack_cnt >= self.cwnd {
            self.ack_cnt -= self.cwnd;
            self.tcp_cwnd = self.tcp_cwnd.saturating_add(self.mss);
        }

        let target_u32 = target.min(u32::MAX as u64) as u32;
        let desired = target_u32.max(self.tcp_cwnd);
        if desired > self.cwnd {
            let inc = ((desired - self.cwnd) as u64 * self.mss as u64) / (self.cwnd as u64).max(1);
            self.cwnd = self.cwnd.saturating_add((inc as u32).max(1));
        }
    }

    pub fn on_loss(&mut self) {
        self.epoch_start_us = 0;
        // Fast convergence
        if self.cwnd < self.w_last_max {
            self.w_last_max = self.cwnd;
            self.w_max = ((self.cwnd as u64 * (Self::BETA_DEN + Self::BETA_NUM))
                / (2 * Self::BETA_DEN)) as u32;
        } else {
            self.w_last_max = self.cwnd;
            self.w_max = self.cwnd;
        }
        let reduced = ((self.cwnd as u64 * Self::BETA_NUM) / Self::BETA_DEN) as u32;
        self.ssthresh = reduced.max(self.mss.saturating_mul(2));
        self.cwnd = self.ssthresh;
    }

    pub fn on_timeout(&mut self) {
        self.epoch_start_us = 0;
        self.w_max = self.cwnd;
        self.ssthresh = (self.cwnd / 2).max(self.mss.saturating_mul(2));
        self.cwnd = self.mss;
    }
}

/// BBR congestion control phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BbrMode {
    Startup,
    Drain,
    ProbeBw,
    ProbeRtt,
}

/// TCP BBR congestion control state (fixed-point 8-bit gain scaling: 256 = 1.0x).
#[derive(Debug, Clone)]
pub struct BbrState {
    pub mss: u32,
    pub mode: BbrMode,
    pub cwnd: u32,
    pub btl_bw_bps: u64,
    pub rt_prop_us: u64,
    pub rt_prop_stamp_us: u64,
    pub pacing_gain_fp: u32,
    pub cwnd_gain_fp: u32,
    pub cycle_idx: usize,
    pub cycle_stamp_us: u64,
    pub full_bw_bps: u64,
    pub full_bw_count: u8,
    pub probe_rtt_done_us: u64,
}

impl BbrState {
    const GAIN_UNIT: u32 = 256;
    const STARTUP_PACING_GAIN: u32 = 739; // ~2.885x
    const STARTUP_CWND_GAIN: u32 = 512; // 2.0x
    const DRAIN_PACING_GAIN: u32 = 192; // 0.75x
    const PROBE_BW_GAINS: [u32; 8] = [320, 192, 256, 256, 256, 256, 256, 256];
    const RT_PROP_FILTER_LEN_US: u64 = 10_000_000; // 10s
    const PROBE_RTT_DURATION_US: u64 = 200_000; // 200ms

    pub fn new(mss: u32) -> Self {
        Self {
            mss,
            mode: BbrMode::Startup,
            cwnd: mss.saturating_mul(10),
            btl_bw_bps: 0,
            rt_prop_us: u64::MAX,
            rt_prop_stamp_us: 0,
            pacing_gain_fp: Self::STARTUP_PACING_GAIN,
            cwnd_gain_fp: Self::STARTUP_CWND_GAIN,
            cycle_idx: 0,
            cycle_stamp_us: 0,
            full_bw_bps: 0,
            full_bw_count: 0,
            probe_rtt_done_us: 0,
        }
    }

    pub fn pacing_rate_bps(&self) -> u64 {
        (self.btl_bw_bps.saturating_mul(self.pacing_gain_fp as u64)) / (Self::GAIN_UNIT as u64)
    }

    pub fn bdp_bytes(&self) -> u32 {
        if self.rt_prop_us == u64::MAX || self.btl_bw_bps == 0 {
            return self.mss.saturating_mul(10);
        }
        // bdp = (btl_bw_bps / 8) * (rt_prop_us / 1_000_000)
        let bytes = (self.btl_bw_bps / 8).saturating_mul(self.rt_prop_us) / 1_000_000;
        (bytes as u32).max(self.mss.saturating_mul(4))
    }

    pub fn on_ack(
        &mut self,
        delivered_bytes: u32,
        rtt_us: u64,
        interval_us: u64,
        inflight_bytes: u32,
        now_us: u64,
    ) {
        // Update bottleneck bandwidth estimate
        if interval_us > 0 && delivered_bytes > 0 {
            let bw_bps = (delivered_bytes as u64)
                .saturating_mul(8)
                .saturating_mul(1_000_000)
                / interval_us;
            if bw_bps > self.btl_bw_bps {
                self.btl_bw_bps = bw_bps;
            }
        }

        // Update min RTT filter
        let rt_expired = now_us.saturating_sub(self.rt_prop_stamp_us) > Self::RT_PROP_FILTER_LEN_US;
        if rtt_us > 0 && (rtt_us <= self.rt_prop_us || rt_expired) {
            self.rt_prop_us = rtt_us;
            self.rt_prop_stamp_us = now_us;
        }

        match self.mode {
            BbrMode::Startup => {
                // Check if bandwidth grew by >= 25%
                if self.btl_bw_bps >= self.full_bw_bps.saturating_mul(5) / 4 {
                    self.full_bw_bps = self.btl_bw_bps;
                    self.full_bw_count = 0;
                } else {
                    self.full_bw_count = self.full_bw_count.saturating_add(1);
                    if self.full_bw_count >= 3 {
                        self.mode = BbrMode::Drain;
                        self.pacing_gain_fp = Self::DRAIN_PACING_GAIN;
                        self.cwnd_gain_fp = Self::STARTUP_CWND_GAIN;
                    }
                }
            }
            BbrMode::Drain => {
                if inflight_bytes <= self.bdp_bytes() {
                    self.mode = BbrMode::ProbeBw;
                    self.cycle_idx = 0;
                    self.cycle_stamp_us = now_us;
                    self.pacing_gain_fp = Self::PROBE_BW_GAINS[0];
                    self.cwnd_gain_fp = Self::STARTUP_CWND_GAIN;
                }
            }
            BbrMode::ProbeBw => {
                let cycle_len = if self.rt_prop_us == u64::MAX {
                    10_000
                } else {
                    self.rt_prop_us
                };
                if now_us.saturating_sub(self.cycle_stamp_us) >= cycle_len {
                    self.cycle_idx = (self.cycle_idx + 1) % Self::PROBE_BW_GAINS.len();
                    self.cycle_stamp_us = now_us;
                    self.pacing_gain_fp = Self::PROBE_BW_GAINS[self.cycle_idx];
                }
                if rt_expired {
                    self.mode = BbrMode::ProbeRtt;
                    self.pacing_gain_fp = Self::GAIN_UNIT;
                    self.cwnd_gain_fp = Self::GAIN_UNIT;
                    self.probe_rtt_done_us = now_us.saturating_add(Self::PROBE_RTT_DURATION_US);
                }
            }
            BbrMode::ProbeRtt => {
                if now_us >= self.probe_rtt_done_us {
                    self.mode = BbrMode::ProbeBw;
                    self.rt_prop_stamp_us = now_us;
                    self.cycle_idx = 0;
                    self.cycle_stamp_us = now_us;
                    self.pacing_gain_fp = Self::PROBE_BW_GAINS[0];
                    self.cwnd_gain_fp = Self::STARTUP_CWND_GAIN;
                }
            }
        }

        // Update cwnd
        if self.mode == BbrMode::ProbeRtt {
            self.cwnd = self.mss.saturating_mul(4);
        } else {
            let target = ((self.bdp_bytes() as u64).saturating_mul(self.cwnd_gain_fp as u64)
                / (Self::GAIN_UNIT as u64)) as u32;
            self.cwnd = target.max(self.mss.saturating_mul(4));
        }
    }
}

/// Unified TCP Congestion Controller supporting CUBIC and BBR.
#[derive(Debug, Clone)]
pub enum TcpCongestionController {
    Cubic(CubicState),
    Bbr(BbrState),
}

impl TcpCongestionController {
    pub fn new_cubic(mss: u32) -> Self {
        Self::Cubic(CubicState::new(mss))
    }

    pub fn new_bbr(mss: u32) -> Self {
        Self::Bbr(BbrState::new(mss))
    }

    pub fn cwnd(&self) -> u32 {
        match self {
            Self::Cubic(c) => c.cwnd,
            Self::Bbr(b) => b.cwnd,
        }
    }

    pub fn on_ack(
        &mut self,
        acked_bytes: u32,
        rtt_us: u64,
        interval_us: u64,
        inflight_bytes: u32,
        now_us: u64,
    ) {
        match self {
            Self::Cubic(c) => c.on_ack(acked_bytes, rtt_us, now_us),
            Self::Bbr(b) => b.on_ack(acked_bytes, rtt_us, interval_us, inflight_bytes, now_us),
        }
    }

    pub fn on_loss(&mut self) {
        match self {
            Self::Cubic(c) => c.on_loss(),
            Self::Bbr(b) => {
                b.cwnd = (b.cwnd / 2).max(b.mss.saturating_mul(4));
            }
        }
    }
}

// ============================================================================
// 3. WireGuard / ChaCha20-Poly1305 AEAD Offload Engine (RFC 8439)
// ============================================================================

#[inline]
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(16);

    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(12);

    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(8);

    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(7);
}

/// Compute a single 64-byte ChaCha20 keystream block (RFC 8439 Section 2.3).
pub fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut state = [0u32; 16];
    state[0] = 0x6170_7865;
    state[1] = 0x3320_646e;
    state[2] = 0x7962_2d32;
    state[3] = 0x6b20_6574;

    let mut i = 0;
    while i < 8 {
        let off = i * 4;
        state[4 + i] = u32::from_le_bytes([key[off], key[off + 1], key[off + 2], key[off + 3]]);
        i += 1;
    }
    state[12] = counter;
    state[13] = u32::from_le_bytes([nonce[0], nonce[1], nonce[2], nonce[3]]);
    state[14] = u32::from_le_bytes([nonce[4], nonce[5], nonce[6], nonce[7]]);
    state[15] = u32::from_le_bytes([nonce[8], nonce[9], nonce[10], nonce[11]]);

    let mut working = state;
    let mut round = 0;
    while round < 10 {
        // Column rounds
        quarter_round(&mut working, 0, 4, 8, 12);
        quarter_round(&mut working, 1, 5, 9, 13);
        quarter_round(&mut working, 2, 6, 10, 14);
        quarter_round(&mut working, 3, 7, 11, 15);
        // Diagonal rounds
        quarter_round(&mut working, 0, 5, 10, 15);
        quarter_round(&mut working, 1, 6, 11, 12);
        quarter_round(&mut working, 2, 7, 8, 13);
        quarter_round(&mut working, 3, 4, 9, 14);
        round += 1;
    }

    let mut out = [0u8; 64];
    let mut idx = 0;
    while idx < 16 {
        let word = working[idx].wrapping_add(state[idx]).to_le_bytes();
        let off = idx * 4;
        out[off..off + 4].copy_from_slice(&word);
        idx += 1;
    }
    out
}

/// XOR `data` in-place with ChaCha20 keystream starting at `counter`.
pub fn chacha20_xor(key: &[u8; 32], mut counter: u32, nonce: &[u8; 12], data: &mut [u8]) {
    let mut offset = 0;
    while offset < data.len() {
        let block = chacha20_block(key, counter, nonce);
        let chunk = (data.len() - offset).min(64);
        let mut i = 0;
        while i < chunk {
            data[offset + i] ^= block[i];
            i += 1;
        }
        offset += chunk;
        counter = counter.wrapping_add(1);
    }
}

/// Poly1305 one-time authenticator (RFC 8439 Section 2.5) using 3x 44-bit limbs in `u128`.
pub struct Poly1305 {
    r0: u64,
    r1: u64,
    r2: u64,
    h0: u64,
    h1: u64,
    h2: u64,
    pad0: u64,
    pad1: u64,
    buf: [u8; 16],
    buf_len: usize,
}

impl Poly1305 {
    pub fn new(key: &[u8; 32]) -> Self {
        let t0 = u64::from_le_bytes([
            key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
        ]);
        let t1 = u64::from_le_bytes([
            key[8], key[9], key[10], key[11], key[12], key[13], key[14], key[15],
        ]);

        // Clamp r per RFC 8439: r &= 0x0ffffffc0ffffffc0ffffffc0fffffff
        let r_lo = t0 & 0x0fff_fffc_0fff_ffff;
        let r_hi = t1 & 0x0fff_fffc_0fff_fffc;

        // Split 128-bit r into 44 + 44 + 40 bits
        let r0 = r_lo & 0x0fff_ffff_ffff;
        let r1 = ((r_lo >> 44) | (r_hi << 20)) & 0x0fff_ffff_ffff;
        let r2 = (r_hi >> 24) & 0x03ff_ffff_ffff;

        let pad0 = u64::from_le_bytes([
            key[16], key[17], key[18], key[19], key[20], key[21], key[22], key[23],
        ]);
        let pad1 = u64::from_le_bytes([
            key[24], key[25], key[26], key[27], key[28], key[29], key[30], key[31],
        ]);

        Self {
            r0,
            r1,
            r2,
            h0: 0,
            h1: 0,
            h2: 0,
            pad0,
            pad1,
            buf: [0u8; 16],
            buf_len: 0,
        }
    }

    fn block(&mut self, chunk: &[u8], hibit: u64) {
        let mut buf = [0u8; 16];
        buf[..chunk.len()].copy_from_slice(chunk);
        if chunk.len() < 16 {
            buf[chunk.len()] = 1;
        }
        let t0 = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);
        let t1 = u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]);

        let m0 = t0 & 0x0fff_ffff_ffff;
        let m1 = ((t0 >> 44) | (t1 << 20)) & 0x0fff_ffff_ffff;
        let m2 = (t1 >> 24) | (hibit << 40);

        let h0 = self.h0 + m0;
        let h1 = self.h1 + m1;
        let h2 = self.h2 + m2;

        let r0 = self.r0 as u128;
        let r1 = self.r1 as u128;
        let r2 = self.r2 as u128;
        let s1 = r1 * 20;
        let s2 = r2 * 20;

        let d0 = (h0 as u128) * r0 + (h1 as u128) * s2 + (h2 as u128) * s1;
        let mut d1 = (h0 as u128) * r1 + (h1 as u128) * r0 + (h2 as u128) * s2;
        let mut d2 = (h0 as u128) * r2 + (h1 as u128) * r1 + (h2 as u128) * r0;

        let mut c = d0 >> 44;
        let mut nh0 = (d0 as u64) & 0x0fff_ffff_ffff;
        d1 += c;
        c = d1 >> 44;
        let mut nh1 = (d1 as u64) & 0x0fff_ffff_ffff;
        d2 += c;
        c = d2 >> 42;
        let nh2 = (d2 as u64) & 0x03ff_ffff_ffff;

        nh0 += (c as u64) * 5;
        let c2 = nh0 >> 44;
        nh0 &= 0x0fff_ffff_ffff;
        nh1 += c2;

        self.h0 = nh0;
        self.h1 = nh1;
        self.h2 = nh2;
    }

    pub fn update(&mut self, data: &[u8]) {
        let mut off = 0;
        if self.buf_len > 0 {
            let take = (16 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            off += take;
            if self.buf_len == 16 {
                let blk = self.buf;
                self.block(&blk, 1);
                self.buf_len = 0;
            }
        }
        while off + 16 <= data.len() {
            self.block(&data[off..off + 16], 1);
            off += 16;
        }
        if off < data.len() {
            let rem = data.len() - off;
            self.buf[..rem].copy_from_slice(&data[off..]);
            self.buf_len = rem;
        }
    }

    pub fn update_pad16(&mut self, data: &[u8]) {
        self.update(data);
        let rem = data.len() % 16;
        if rem != 0 {
            let zeros = [0u8; 16];
            self.update(&zeros[..16 - rem]);
        }
    }

    pub fn finalize(mut self) -> [u8; 16] {
        if self.buf_len > 0 {
            let blk = self.buf;
            let len = self.buf_len;
            self.block(&blk[..len], 0);
        }
        let mut c = self.h1 >> 44;
        self.h1 &= 0x0fff_ffff_ffff;
        self.h2 += c;
        c = self.h2 >> 42;
        self.h2 &= 0x03ff_ffff_ffff;
        self.h0 += c * 5;
        c = self.h0 >> 44;
        self.h0 &= 0x0fff_ffff_ffff;
        self.h1 += c;
        c = self.h1 >> 44;
        self.h1 &= 0x0fff_ffff_ffff;
        self.h2 += c;
        c = self.h2 >> 42;
        self.h2 &= 0x03ff_ffff_ffff;
        self.h0 += c * 5;
        c = self.h0 >> 44;
        self.h0 &= 0x0fff_ffff_ffff;
        self.h1 += c;

        // Compute h + 5 - 2^130
        let mut g0 = self.h0 + 5;
        c = g0 >> 44;
        g0 &= 0x0fff_ffff_ffff;
        let mut g1 = self.h1 + c;
        c = g1 >> 44;
        g1 &= 0x0fff_ffff_ffff;
        let g2 = (self.h2 + c).wrapping_sub(1u64 << 42);

        // Select h if g2 bit 63 is set (underflow), else g
        let mask = (g2 >> 63).wrapping_sub(1);
        self.h0 = (self.h0 & !mask) | (g0 & mask);
        self.h1 = (self.h1 & !mask) | (g1 & mask);
        self.h2 = (self.h2 & !mask) | (g2 & mask);

        // Reassemble 128-bit h and add pad
        let h_lo = self.h0 | (self.h1 << 44);
        let h_hi = (self.h1 >> 20) | (self.h2 << 24);

        let f0 = (h_lo as u128) + (self.pad0 as u128);
        let f1 = (h_hi as u128) + (self.pad1 as u128) + (f0 >> 64);

        let mut tag = [0u8; 16];
        tag[0..8].copy_from_slice(&(f0 as u64).to_le_bytes());
        tag[8..16].copy_from_slice(&(f1 as u64).to_le_bytes());
        tag
    }
}

/// Constant-time 16-byte tag comparison.
#[inline]
pub fn ct_eq_16(a: &[u8; 16], b: &[u8; 16]) -> bool {
    let mut diff = 0u8;
    let mut i = 0;
    while i < 16 {
        diff |= a[i] ^ b[i];
        i += 1;
    }
    diff == 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AeadError {
    InvalidTag,
    BufferTooSmall,
    ReplayDetected,
    UnknownPeer,
}

/// RFC 8439 ChaCha20-Poly1305 AEAD.
pub struct ChaCha20Poly1305;

impl ChaCha20Poly1305 {
    fn poly1305_key_gen(key: &[u8; 32], nonce: &[u8; 12]) -> [u8; 32] {
        let block = chacha20_block(key, 0, nonce);
        let mut otk = [0u8; 32];
        otk.copy_from_slice(&block[..32]);
        otk
    }

    fn compute_tag(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
        let otk = Self::poly1305_key_gen(key, nonce);
        let mut poly = Poly1305::new(&otk);
        poly.update_pad16(aad);
        poly.update_pad16(ciphertext);
        let mut lens = [0u8; 16];
        lens[0..8].copy_from_slice(&(aad.len() as u64).to_le_bytes());
        lens[8..16].copy_from_slice(&(ciphertext.len() as u64).to_le_bytes());
        poly.update(&lens);
        poly.finalize()
    }

    pub fn seal_in_place(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], buf: &mut [u8]) -> [u8; 16] {
        chacha20_xor(key, 1, nonce, buf);
        Self::compute_tag(key, nonce, aad, buf)
    }

    pub fn open_in_place(
        key: &[u8; 32],
        nonce: &[u8; 12],
        aad: &[u8],
        buf: &mut [u8],
        tag: &[u8; 16],
    ) -> Result<(), AeadError> {
        let expected = Self::compute_tag(key, nonce, aad, buf);
        if !ct_eq_16(&expected, tag) {
            return Err(AeadError::InvalidTag);
        }
        chacha20_xor(key, 1, nonce, buf);
        Ok(())
    }
}

/// WireGuard Transport Data Header (16 bytes, Type = 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgTransportHeader {
    pub msg_type: u8,
    pub receiver_index: u32,
    pub counter: u64,
}

impl WgTransportHeader {
    pub const SIZE: usize = 16;
    pub const TAG_SIZE: usize = 16;
    pub const TYPE_TRANSPORT: u8 = 4;

    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE || buf[0] != Self::TYPE_TRANSPORT {
            return None;
        }
        let receiver_index = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let counter = u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]);
        Some(Self {
            msg_type: buf[0],
            receiver_index,
            counter,
        })
    }

    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < Self::SIZE {
            return None;
        }
        buf[0] = Self::TYPE_TRANSPORT;
        buf[1] = 0;
        buf[2] = 0;
        buf[3] = 0;
        buf[4..8].copy_from_slice(&self.receiver_index.to_le_bytes());
        buf[8..16].copy_from_slice(&self.counter.to_le_bytes());
        Some(Self::SIZE)
    }
}

/// 64-packet sliding anti-replay window for WireGuard transport counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReplayWindow {
    pub last_counter: u64,
    pub bitmap: u64,
}

impl ReplayWindow {
    pub fn check_and_update(&mut self, counter: u64) -> bool {
        if counter > self.last_counter {
            let shift = counter - self.last_counter;
            self.bitmap = if shift >= 64 {
                1
            } else {
                (self.bitmap << shift) | 1
            };
            self.last_counter = counter;
            true
        } else {
            let diff = self.last_counter - counter;
            if diff >= 64 {
                return false;
            }
            let mask = 1u64 << diff;
            if (self.bitmap & mask) != 0 {
                return false;
            }
            self.bitmap |= mask;
            true
        }
    }
}

#[derive(Debug, Clone)]
pub struct WgPeerSession {
    pub local_index: u32,
    pub remote_index: u32,
    pub tx_key: [u8; 32],
    pub rx_key: [u8; 32],
    pub tx_counter: u64,
    pub replay: ReplayWindow,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OffloadStats {
    pub packets_encrypted: u64,
    pub packets_decrypted: u64,
    pub auth_failures: u64,
    pub replay_drops: u64,
}

/// WireGuard ChaCha20-Poly1305 Offload Engine.
pub struct WireGuardOffloadEngine {
    peers: Vec<WgPeerSession>,
    pub stats: OffloadStats,
}

impl WireGuardOffloadEngine {
    pub fn new() -> Self {
        Self {
            peers: Vec::new(),
            stats: OffloadStats::default(),
        }
    }

    pub fn add_peer(&mut self, session: WgPeerSession) {
        self.peers.push(session);
    }

    #[inline]
    fn wg_nonce(counter: u64) -> [u8; 12] {
        let mut nonce = [0u8; 12];
        nonce[4..12].copy_from_slice(&counter.to_le_bytes());
        nonce
    }

    /// Encrypt `plaintext` into `out` as a WireGuard Type-4 Transport packet.
    /// Returns total packet length (`16 + plaintext.len() + 16`).
    pub fn encrypt_transport(
        &mut self,
        local_index: u32,
        plaintext: &[u8],
        out: &mut [u8],
    ) -> Result<usize, AeadError> {
        let total_len = WgTransportHeader::SIZE + plaintext.len() + WgTransportHeader::TAG_SIZE;
        if out.len() < total_len {
            return Err(AeadError::BufferTooSmall);
        }
        let peer = self
            .peers
            .iter_mut()
            .find(|p| p.local_index == local_index)
            .ok_or(AeadError::UnknownPeer)?;

        let counter = peer.tx_counter;
        peer.tx_counter = peer.tx_counter.wrapping_add(1);

        let hdr = WgTransportHeader {
            msg_type: WgTransportHeader::TYPE_TRANSPORT,
            receiver_index: peer.remote_index,
            counter,
        };
        hdr.encode(&mut out[..WgTransportHeader::SIZE]);

        let ct_start = WgTransportHeader::SIZE;
        let ct_end = ct_start + plaintext.len();
        out[ct_start..ct_end].copy_from_slice(plaintext);

        let nonce = Self::wg_nonce(counter);
        let tag =
            ChaCha20Poly1305::seal_in_place(&peer.tx_key, &nonce, &[], &mut out[ct_start..ct_end]);
        out[ct_end..total_len].copy_from_slice(&tag);

        self.stats.packets_encrypted += 1;
        Ok(total_len)
    }

    /// Decrypt a WireGuard Type-4 Transport packet in-place, returning the plaintext slice length.
    pub fn decrypt_transport(&mut self, packet: &mut [u8]) -> Result<usize, AeadError> {
        if packet.len() < WgTransportHeader::SIZE + WgTransportHeader::TAG_SIZE {
            return Err(AeadError::BufferTooSmall);
        }
        let hdr = WgTransportHeader::parse(packet).ok_or(AeadError::InvalidTag)?;
        let peer = self
            .peers
            .iter_mut()
            .find(|p| p.local_index == hdr.receiver_index)
            .ok_or(AeadError::UnknownPeer)?;

        let ct_start = WgTransportHeader::SIZE;
        let ct_end = packet.len() - WgTransportHeader::TAG_SIZE;
        let mut tag = [0u8; 16];
        tag.copy_from_slice(&packet[ct_end..]);

        let nonce = Self::wg_nonce(hdr.counter);
        if ChaCha20Poly1305::open_in_place(
            &peer.rx_key,
            &nonce,
            &[],
            &mut packet[ct_start..ct_end],
            &tag,
        )
        .is_err()
        {
            self.stats.auth_failures += 1;
            return Err(AeadError::InvalidTag);
        }

        if !peer.replay.check_and_update(hdr.counter) {
            self.stats.replay_drops += 1;
            return Err(AeadError::ReplayDetected);
        }

        self.stats.packets_decrypted += 1;
        Ok(ct_end - ct_start)
    }
}

impl Default for WireGuardOffloadEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 4. Self-Test Suite
// ============================================================================

pub fn self_test() -> Result<(), &'static str> {
    // 1. IPv6 EUI-64, Solicited-Node Multicast, NDP & SLAAC
    let mac = MacAddr::new([0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
    let mut slaac = SlaacEngine::new(mac);
    if !slaac.link_local.addr.is_link_local() {
        return Err("net_v6: link-local address invalid");
    }
    let snm = slaac.link_local.addr.solicited_node_multicast();
    if !snm.is_multicast() || snm.0[12] != 0xff || snm.0[15] != 0x56 {
        return Err("net_v6: solicited-node multicast invalid");
    }
    slaac.complete_dad();
    if slaac.link_local.state != SlaacAddrState::Preferred {
        return Err("net_v6: DAD completion failed");
    }

    let mut rs_buf = vec![0u8; 32];
    let rs_len = slaac
        .build_router_solicitation(&mut rs_buf)
        .ok_or("net_v6: RS build failed")?;
    if !Icmpv6Header::verify_checksum(
        &slaac.link_local.addr,
        &Ipv6Addr::ALL_ROUTERS_MULTICAST,
        &rs_buf[..rs_len],
    ) {
        return Err("net_v6: RS ICMPv6 checksum invalid");
    }

    // Simulate Router Advertisement with 2001:db8:1::/64 prefix
    let mut ra = vec![0u8; 48];
    ra[0] = Icmpv6Type::RouterAdvertisement as u8;
    ra[6..8].copy_from_slice(&1800u16.to_be_bytes()); // router lifetime
    ra[16] = NdpOption::OPT_PREFIX_INFO;
    ra[17] = 4; // 32 bytes
    ra[18] = 64; // /64
    ra[19] = 0xc0; // L + A flags
    ra[20..24].copy_from_slice(&86400u32.to_be_bytes());
    ra[24..28].copy_from_slice(&14400u32.to_be_bytes());
    ra[32..40].copy_from_slice(&[0x20, 0x01, 0x0d, 0xb8, 0x00, 0x01, 0, 0]);
    let router_ip = Ipv6Addr::from_segments([0xfe80, 0, 0, 0, 0, 0, 0, 1]);
    if !slaac.process_router_advertisement(router_ip, &ra, 100, 100_000) {
        return Err("net_v6: RA processing failed");
    }
    if slaac.global_addrs.len() != 1 || slaac.default_router != Some(router_ip) {
        return Err("net_v6: SLAAC global address not configured");
    }

    // 2. TCP CUBIC & BBR Congestion Control
    let mut cubic = TcpCongestionController::new_cubic(1460);
    let init_cwnd = cubic.cwnd();
    cubic.on_ack(1460, 20_000, 20_000, init_cwnd, 100_000);
    if cubic.cwnd() <= init_cwnd {
        return Err("net_v6: CUBIC slow start did not grow cwnd");
    }
    let pre_loss = cubic.cwnd();
    cubic.on_loss();
    if cubic.cwnd() >= pre_loss {
        return Err("net_v6: CUBIC loss did not reduce cwnd");
    }

    let mut bbr = TcpCongestionController::new_bbr(1460);
    for i in 1..=6u64 {
        bbr.on_ack(14_600, 10_000, 10_000, 14_600, i * 10_000);
    }
    if let TcpCongestionController::Bbr(ref state) = bbr {
        if state.btl_bw_bps == 0 || state.rt_prop_us != 10_000 {
            return Err("net_v6: BBR bandwidth/RTT estimation failed");
        }
    }

    // 3. RFC 8439 ChaCha20-Poly1305 AEAD & WireGuard Offload Engine
    // RFC 8439 Section 2.8.2 test vector
    let key: [u8; 32] = [
        0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e,
        0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d,
        0x9e, 0x9f,
    ];
    let nonce: [u8; 12] = [
        0x07, 0x00, 0x00, 0x00, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    ];
    let aad: [u8; 12] = [
        0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    ];
    let mut pt = *b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
    let expected_tag: [u8; 16] = [
        0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60, 0x06,
        0x91,
    ];
    let tag = ChaCha20Poly1305::seal_in_place(&key, &nonce, &aad, &mut pt);
    if !ct_eq_16(&tag, &expected_tag) {
        return Err("net_v6: RFC 8439 ChaCha20-Poly1305 tag mismatch");
    }
    ChaCha20Poly1305::open_in_place(&key, &nonce, &aad, &mut pt, &tag)
        .map_err(|_| "net_v6: RFC 8439 AEAD open failed")?;
    if &pt[..6] != b"Ladies" {
        return Err("net_v6: RFC 8439 AEAD plaintext mismatch");
    }

    // WireGuard transport encrypt/decrypt + anti-replay
    let mut wg = WireGuardOffloadEngine::new();
    wg.add_peer(WgPeerSession {
        local_index: 1,
        remote_index: 1,
        tx_key: key,
        rx_key: key,
        tx_counter: 1,
        replay: ReplayWindow::default(),
    });
    let mut pkt = vec![0u8; 64];
    let pkt_len = wg
        .encrypt_transport(1, b"nexus-wg-payload", &mut pkt)
        .map_err(|_| "net_v6: WG encrypt failed")?;
    let mut pkt_copy = pkt[..pkt_len].to_vec();
    let dec_len = wg
        .decrypt_transport(&mut pkt[..pkt_len])
        .map_err(|_| "net_v6: WG decrypt failed")?;
    if &pkt[WgTransportHeader::SIZE..WgTransportHeader::SIZE + dec_len] != b"nexus-wg-payload" {
        return Err("net_v6: WG decrypted payload mismatch");
    }
    // Replay must be rejected
    if wg.decrypt_transport(&mut pkt_copy) != Err(AeadError::ReplayDetected) {
        return Err("net_v6: WG anti-replay failed");
    }

    Ok(())
}
