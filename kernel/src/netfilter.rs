//! Stateful Packet Filter, NAT/Conntrack & eBPF-Style XDP Hook Engine.
//!
//! Provides:
//! - 5-tuple connection tracking (`ConntrackTable`) with TCP state machine & timeout GC
//! - SNAT / DNAT / Masquerade with in-place IPv4 & TCP/UDP checksum updates
//! - Stateful firewall chains across Netfilter hooks (`PreRouting`, `Input`, `Forward`, `Output`, `PostRouting`)
//! - eBPF-style 64-bit XDP bytecode verifier (DAG/loop-free, register/type/bounds safety) & VM
//!
//! Designed for `no_std` x86_64 kernel execution with small stack frames (<2 KiB).
#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::mm::SpinLock;
use crate::net::{EthernetHeader, Ipv4Addr, Ipv4Header, TcpHeader, UdpHeader};

// ============================================================================
// 1. Errors & Common Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetfilterError {
    PacketTooShort,
    NotIpv4,
    InvalidIpHeader,
    InvalidL4Header,
    TableFull,
    NoFreeNatPort,
    ChainLoop,
    InvalidChain,
    XdpVerify(XdpVerifyError),
    XdpRuntime(XdpRuntimeError),
}

impl fmt::Display for NetfilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PacketTooShort => write!(f, "packet too short"),
            Self::NotIpv4 => write!(f, "not an IPv4 packet"),
            Self::InvalidIpHeader => write!(f, "invalid IPv4 header"),
            Self::InvalidL4Header => write!(f, "invalid L4 header"),
            Self::TableFull => write!(f, "conntrack table full"),
            Self::NoFreeNatPort => write!(f, "no free NAT port"),
            Self::ChainLoop => write!(f, "firewall chain jump loop"),
            Self::InvalidChain => write!(f, "invalid firewall chain"),
            Self::XdpVerify(e) => write!(f, "XDP verify error: {:?}", e),
            Self::XdpRuntime(e) => write!(f, "XDP runtime error: {:?}", e),
        }
    }
}

pub const PROTO_ICMP: u8 = 1;
pub const PROTO_TCP: u8 = 6;
pub const PROTO_UDP: u8 = 17;

/// 5-tuple identifying a network flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FiveTuple {
    pub src_ip: Ipv4Addr,
    pub dst_ip: Ipv4Addr,
    pub src_port: u16,
    pub dst_port: u16,
    pub proto: u8,
}

impl FiveTuple {
    pub const fn new(
        src_ip: Ipv4Addr,
        dst_ip: Ipv4Addr,
        src_port: u16,
        dst_port: u16,
        proto: u8,
    ) -> Self {
        Self {
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            proto,
        }
    }

    /// Return the reverse-direction 5-tuple.
    pub const fn reverse(&self) -> Self {
        Self {
            src_ip: self.dst_ip,
            dst_ip: self.src_ip,
            src_port: self.dst_port,
            dst_port: self.src_port,
            proto: self.proto,
        }
    }
}

/// Parsed offsets and metadata for an IPv4 packet.
#[derive(Debug, Clone, Copy)]
pub struct ParsedPacket {
    pub l3_offset: usize,
    pub l3_len: usize,
    pub l4_offset: usize,
    pub total_len: usize,
    pub tuple: FiveTuple,
    pub tcp_flags: u16,
}

/// Parse an Ethernet+IPv4 or raw IPv4 packet into `ParsedPacket`.
pub fn parse_ipv4_packet(buf: &[u8], has_eth: bool) -> Result<ParsedPacket, NetfilterError> {
    let l3_offset = if has_eth {
        let eth = EthernetHeader::parse(buf).ok_or(NetfilterError::PacketTooShort)?;
        if eth.ethertype != EthernetHeader::ETHERTYPE_IPV4 {
            return Err(NetfilterError::NotIpv4);
        }
        EthernetHeader::SIZE
    } else {
        0
    };

    if buf.len() < l3_offset + Ipv4Header::MIN_SIZE {
        return Err(NetfilterError::PacketTooShort);
    }

    let ip_slice = &buf[l3_offset..];
    let ip = Ipv4Header::parse(ip_slice).ok_or(NetfilterError::InvalidIpHeader)?;
    let ihl = ((ip.version_ihl & 0x0F) as usize) * 4;
    if ihl < Ipv4Header::MIN_SIZE || ip_slice.len() < ihl {
        return Err(NetfilterError::InvalidIpHeader);
    }

    let l4_offset = l3_offset + ihl;
    let l4_slice = &buf[l4_offset..];

    let (src_port, dst_port, tcp_flags) = match ip.protocol {
        PROTO_TCP => {
            let tcp = TcpHeader::parse(l4_slice).ok_or(NetfilterError::InvalidL4Header)?;
            (tcp.src_port, tcp.dst_port, tcp.flags())
        }
        PROTO_UDP => {
            let udp = UdpHeader::parse(l4_slice).ok_or(NetfilterError::InvalidL4Header)?;
            (udp.src_port, udp.dst_port, 0)
        }
        PROTO_ICMP => {
            if l4_slice.len() < 8 {
                return Err(NetfilterError::InvalidL4Header);
            }
            let icmp_id = u16::from_be_bytes([l4_slice[4], l4_slice[5]]);
            (icmp_id, icmp_id, 0)
        }
        _ => (0, 0, 0),
    };

    Ok(ParsedPacket {
        l3_offset,
        l3_len: ihl,
        l4_offset,
        total_len: buf.len(),
        tuple: FiveTuple::new(ip.src, ip.dst, src_port, dst_port, ip.protocol),
        tcp_flags,
    })
}

// ============================================================================
// 2. Checksum Helpers (RFC 1071 / RFC 1624)
// ============================================================================

/// Compute RFC 1071 16-bit one's complement checksum.
pub fn rfc1071_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let (chunks, rem) = data.as_chunks::<2>();
    for chunk in chunks {
        sum += u16::from_be_bytes(*chunk) as u32;
    }
    if let Some(&b) = rem.first() {
        sum += (b as u32) << 8;
    }
    while (sum >> 16) != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

/// Compute TCP/UDP pseudo-header + L4 payload checksum.
pub fn l4_checksum(src: Ipv4Addr, dst: Ipv4Addr, proto: u8, l4_data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let s = src.as_bytes();
    let d = dst.as_bytes();
    sum += u16::from_be_bytes([s[0], s[1]]) as u32;
    sum += u16::from_be_bytes([s[2], s[3]]) as u32;
    sum += u16::from_be_bytes([d[0], d[1]]) as u32;
    sum += u16::from_be_bytes([d[2], d[3]]) as u32;
    sum += proto as u32;
    sum += l4_data.len() as u32;

    let (chunks, rem) = l4_data.as_chunks::<2>();
    for chunk in chunks {
        sum += u16::from_be_bytes(*chunk) as u32;
    }
    if let Some(&b) = rem.first() {
        sum += (b as u32) << 8;
    }
    while (sum >> 16) != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

/// Incremental RFC 1624 checksum update when replacing a 16-bit field.
#[inline]
pub fn csum_update_u16(old_csum: u16, old_val: u16, new_val: u16) -> u16 {
    let mut sum = (!old_csum) as u32;
    sum += (!old_val) as u32;
    sum += new_val as u32;
    while (sum >> 16) != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

/// Incremental RFC 1624 checksum update when replacing a 32-bit IPv4 address.
#[inline]
pub fn csum_update_u32(old_csum: u16, old_ip: Ipv4Addr, new_ip: Ipv4Addr) -> u16 {
    let o = old_ip.as_bytes();
    let n = new_ip.as_bytes();
    let c1 = csum_update_u16(
        old_csum,
        u16::from_be_bytes([o[0], o[1]]),
        u16::from_be_bytes([n[0], n[1]]),
    );
    csum_update_u16(
        c1,
        u16::from_be_bytes([o[2], o[3]]),
        u16::from_be_bytes([n[2], n[3]]),
    )
}

// ============================================================================
// 3. 5-Tuple Conntrack Table
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtState {
    New,
    Established,
    Related,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpCtState {
    None,
    SynSent,
    SynRecv,
    Established,
    FinWait,
    CloseWait,
    TimeWait,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatKind {
    None,
    Snat,
    Dnat,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtDirection {
    Original,
    Reply,
}

#[derive(Debug, Clone, Copy)]
pub struct ConntrackEntry {
    pub id: u64,
    pub orig_tuple: FiveTuple,
    pub reply_tuple: FiveTuple,
    pub state: CtState,
    pub tcp_state: TcpCtState,
    pub nat_kind: NatKind,
    pub last_seen_tick: u64,
    pub timeout_ticks: u64,
    pub orig_packets: u64,
    pub orig_bytes: u64,
    pub reply_packets: u64,
    pub reply_bytes: u64,
    pub mark: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConntrackStats {
    pub active_entries: usize,
    pub total_created: u64,
    pub total_expired: u64,
    pub total_hits: u64,
}

pub struct ConntrackTable {
    entries: Vec<ConntrackEntry>,
    max_entries: usize,
    next_id: u64,
    total_created: u64,
    total_expired: u64,
    total_hits: u64,
}

impl ConntrackTable {
    pub const DEFAULT_MAX_ENTRIES: usize = 4096;
    pub const TIMEOUT_NEW: u64 = 30;
    pub const TIMEOUT_ESTABLISHED: u64 = 300;
    pub const TIMEOUT_TIME_WAIT: u64 = 15;
    pub const TIMEOUT_UDP: u64 = 60;

    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
            next_id: 1,
            total_created: 0,
            total_expired: 0,
            total_hits: 0,
        }
    }

    /// Look up an existing entry matching either `orig_tuple` or `reply_tuple`.
    pub fn lookup(&self, tuple: &FiveTuple) -> Option<(usize, CtDirection)> {
        for (idx, entry) in self.entries.iter().enumerate() {
            if &entry.orig_tuple == tuple {
                return Some((idx, CtDirection::Original));
            }
            if &entry.reply_tuple == tuple {
                return Some((idx, CtDirection::Reply));
            }
        }
        None
    }

    pub fn get(&self, idx: usize) -> Option<&ConntrackEntry> {
        self.entries.get(idx)
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut ConntrackEntry> {
        self.entries.get_mut(idx)
    }

    /// Check if a reply tuple (e.g. NAT-allocated port) is already in use.
    pub fn is_reply_tuple_in_use(&self, reply: &FiveTuple) -> bool {
        self.entries
            .iter()
            .any(|e| &e.reply_tuple == reply || &e.orig_tuple == reply)
    }

    /// Insert a new conntrack entry, evicting expired entries first if full.
    pub fn insert(
        &mut self,
        orig_tuple: FiveTuple,
        reply_tuple: FiveTuple,
        nat_kind: NatKind,
        tcp_flags: u16,
        now_tick: u64,
    ) -> Result<usize, NetfilterError> {
        if self.entries.len() >= self.max_entries {
            self.gc_expired(now_tick);
        }
        if self.entries.len() >= self.max_entries {
            return Err(NetfilterError::TableFull);
        }

        let (tcp_state, timeout_ticks) = if orig_tuple.proto == PROTO_TCP {
            if (tcp_flags & TcpHeader::FLAG_SYN) != 0 && (tcp_flags & TcpHeader::FLAG_ACK) == 0 {
                (TcpCtState::SynSent, Self::TIMEOUT_NEW)
            } else {
                (TcpCtState::None, Self::TIMEOUT_NEW)
            }
        } else {
            (TcpCtState::None, Self::TIMEOUT_UDP)
        };

        let id = self.next_id;
        self.next_id += 1;
        self.total_created += 1;

        self.entries.push(ConntrackEntry {
            id,
            orig_tuple,
            reply_tuple,
            state: CtState::New,
            tcp_state,
            nat_kind,
            last_seen_tick: now_tick,
            timeout_ticks,
            orig_packets: 0,
            orig_bytes: 0,
            reply_packets: 0,
            reply_bytes: 0,
            mark: 0,
        });

        Ok(self.entries.len() - 1)
    }

    /// Update connection state and counters for a matched packet.
    pub fn update_on_packet(
        &mut self,
        idx: usize,
        dir: CtDirection,
        pkt_len: usize,
        tcp_flags: u16,
        now_tick: u64,
    ) {
        let Some(entry) = self.entries.get_mut(idx) else {
            return;
        };
        self.total_hits += 1;
        entry.last_seen_tick = now_tick;

        match dir {
            CtDirection::Original => {
                entry.orig_packets += 1;
                entry.orig_bytes += pkt_len as u64;
            }
            CtDirection::Reply => {
                entry.reply_packets += 1;
                entry.reply_bytes += pkt_len as u64;
                if entry.state == CtState::New {
                    entry.state = CtState::Established;
                }
            }
        }

        if entry.orig_tuple.proto == PROTO_TCP {
            let syn = (tcp_flags & TcpHeader::FLAG_SYN) != 0;
            let ack = (tcp_flags & TcpHeader::FLAG_ACK) != 0;
            let fin = (tcp_flags & TcpHeader::FLAG_FIN) != 0;
            let rst = (tcp_flags & TcpHeader::FLAG_RST) != 0;

            if rst {
                entry.tcp_state = TcpCtState::Closed;
                entry.timeout_ticks = Self::TIMEOUT_TIME_WAIT;
                return;
            }

            entry.tcp_state = match (entry.tcp_state, dir, syn, ack, fin) {
                (TcpCtState::SynSent, CtDirection::Reply, true, true, false) => {
                    entry.timeout_ticks = Self::TIMEOUT_NEW;
                    TcpCtState::SynRecv
                }
                (TcpCtState::SynRecv, CtDirection::Original, false, true, false) => {
                    entry.state = CtState::Established;
                    entry.timeout_ticks = Self::TIMEOUT_ESTABLISHED;
                    TcpCtState::Established
                }
                (TcpCtState::Established, _, _, _, true) => {
                    entry.timeout_ticks = Self::TIMEOUT_TIME_WAIT;
                    TcpCtState::FinWait
                }
                (TcpCtState::FinWait, _, _, true, true) => {
                    entry.timeout_ticks = Self::TIMEOUT_TIME_WAIT;
                    TcpCtState::TimeWait
                }
                (TcpCtState::FinWait, _, _, true, false) => {
                    entry.timeout_ticks = Self::TIMEOUT_TIME_WAIT;
                    TcpCtState::CloseWait
                }
                (TcpCtState::CloseWait, _, _, _, true) => {
                    entry.timeout_ticks = Self::TIMEOUT_TIME_WAIT;
                    TcpCtState::TimeWait
                }
                (state, _, _, _, _) => state,
            };
        } else if dir == CtDirection::Reply {
            entry.timeout_ticks = Self::TIMEOUT_ESTABLISHED;
        }
    }

    /// Remove all expired entries as of `now_tick`. Returns number of evicted entries.
    pub fn gc_expired(&mut self, now_tick: u64) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|e| now_tick.saturating_sub(e.last_seen_tick) < e.timeout_ticks);
        let removed = before - self.entries.len();
        self.total_expired += removed as u64;
        removed
    }

    pub fn stats(&self) -> ConntrackStats {
        ConntrackStats {
            active_entries: self.entries.len(),
            total_created: self.total_created,
            total_expired: self.total_expired,
            total_hits: self.total_hits,
        }
    }
}

// ============================================================================
// 4. SNAT / DNAT / Masquerade & In-Place Packet Rewriting
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IpCidr {
    pub addr: Ipv4Addr,
    pub prefix_len: u8,
}

impl IpCidr {
    pub const ANY: Self = Self {
        addr: Ipv4Addr::UNSPECIFIED,
        prefix_len: 0,
    };

    pub const fn new(addr: Ipv4Addr, prefix_len: u8) -> Self {
        Self { addr, prefix_len }
    }

    pub const fn host(addr: Ipv4Addr) -> Self {
        Self {
            addr,
            prefix_len: 32,
        }
    }

    pub fn matches(&self, ip: Ipv4Addr) -> bool {
        if self.prefix_len == 0 {
            return true;
        }
        let shift = 32u32.saturating_sub(self.prefix_len as u32);
        let mask = if shift >= 32 { 0 } else { !0u32 << shift };
        (self.addr.to_be_u32() & mask) == (ip.to_be_u32() & mask)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortRange {
    pub min: u16,
    pub max: u16,
}

impl PortRange {
    pub const ANY: Self = Self { min: 0, max: 65535 };

    pub const fn single(port: u16) -> Self {
        Self {
            min: port,
            max: port,
        }
    }

    pub const fn new(min: u16, max: u16) -> Self {
        Self { min, max }
    }

    pub const fn contains(&self, port: u16) -> bool {
        port >= self.min && port <= self.max
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatAction {
    Snat {
        to_ip: Ipv4Addr,
        port_min: u16,
        port_max: u16,
    },
    Dnat {
        to_ip: Ipv4Addr,
        to_port: Option<u16>,
    },
    Masquerade {
        out_ip: Ipv4Addr,
        port_min: u16,
        port_max: u16,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct NatRule {
    pub id: u32,
    pub src_cidr: IpCidr,
    pub dst_cidr: IpCidr,
    pub proto: Option<u8>,
    pub dst_port: PortRange,
    pub action: NatAction,
    pub hits: u64,
}

pub struct NatTable {
    rules: Vec<NatRule>,
    next_ephemeral: u16,
}

impl NatTable {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            next_ephemeral: 40000,
        }
    }

    pub fn add_rule(&mut self, rule: NatRule) {
        self.rules.push(rule);
    }

    pub fn rules(&self) -> &[NatRule] {
        &self.rules
    }

    /// Find a matching DNAT rule for `PreRouting`.
    pub fn match_dnat(&mut self, tuple: &FiveTuple) -> Option<(Ipv4Addr, u16)> {
        for rule in &mut self.rules {
            if let NatAction::Dnat { to_ip, to_port } = rule.action {
                if rule.proto.is_some_and(|p| p != tuple.proto) {
                    continue;
                }
                if !rule.src_cidr.matches(tuple.src_ip) || !rule.dst_cidr.matches(tuple.dst_ip) {
                    continue;
                }
                if !rule.dst_port.contains(tuple.dst_port) {
                    continue;
                }
                rule.hits += 1;
                return Some((to_ip, to_port.unwrap_or(tuple.dst_port)));
            }
        }
        None
    }

    /// Find a matching SNAT/Masquerade rule for `PostRouting` and allocate a non-colliding port.
    pub fn match_snat(
        &mut self,
        tuple: &FiveTuple,
        ct: &ConntrackTable,
    ) -> Result<Option<(Ipv4Addr, u16)>, NetfilterError> {
        for rule in &mut self.rules {
            let (to_ip, pmin, pmax) = match rule.action {
                NatAction::Snat {
                    to_ip,
                    port_min,
                    port_max,
                } => (to_ip, port_min, port_max),
                NatAction::Masquerade {
                    out_ip,
                    port_min,
                    port_max,
                } => (out_ip, port_min, port_max),
                NatAction::Dnat { .. } => continue,
            };

            if rule.proto.is_some_and(|p| p != tuple.proto) {
                continue;
            }
            if !rule.src_cidr.matches(tuple.src_ip) || !rule.dst_cidr.matches(tuple.dst_ip) {
                continue;
            }
            if !rule.dst_port.contains(tuple.dst_port) {
                continue;
            }

            rule.hits += 1;
            let port = if tuple.proto == PROTO_TCP || tuple.proto == PROTO_UDP {
                Self::alloc_port(&mut self.next_ephemeral, tuple, to_ip, pmin, pmax, ct)?
            } else {
                tuple.src_port
            };
            return Ok(Some((to_ip, port)));
        }
        Ok(None)
    }

    fn alloc_port(
        cursor: &mut u16,
        orig: &FiveTuple,
        nat_ip: Ipv4Addr,
        pmin: u16,
        pmax: u16,
        ct: &ConntrackTable,
    ) -> Result<u16, NetfilterError> {
        let lo = pmin.max(1024);
        let hi = pmax.max(lo);
        let span = (hi - lo) as usize + 1;

        // Try original port first if within range and free.
        if orig.src_port >= lo && orig.src_port <= hi {
            let candidate_reply = FiveTuple::new(
                orig.dst_ip,
                nat_ip,
                orig.dst_port,
                orig.src_port,
                orig.proto,
            );
            if !ct.is_reply_tuple_in_use(&candidate_reply) {
                return Ok(orig.src_port);
            }
        }

        for _ in 0..span {
            if *cursor < lo || *cursor > hi {
                *cursor = lo;
            }
            let port = *cursor;
            *cursor = if port == hi { lo } else { port + 1 };
            let candidate_reply =
                FiveTuple::new(orig.dst_ip, nat_ip, orig.dst_port, port, orig.proto);
            if !ct.is_reply_tuple_in_use(&candidate_reply) {
                return Ok(port);
            }
        }
        Err(NetfilterError::NoFreeNatPort)
    }
}

/// Rewrite IPv4 src/dst IP and L4 src/dst port in-place, updating IPv4 and TCP/UDP checksums.
pub fn apply_nat_rewrite(
    buf: &mut [u8],
    parsed: &ParsedPacket,
    new_src_ip: Option<Ipv4Addr>,
    new_dst_ip: Option<Ipv4Addr>,
    new_src_port: Option<u16>,
    new_dst_port: Option<u16>,
) -> Result<(), NetfilterError> {
    let l3 = parsed.l3_offset;
    let l4 = parsed.l4_offset;
    if buf.len() < l3 + Ipv4Header::MIN_SIZE {
        return Err(NetfilterError::PacketTooShort);
    }

    let old_src_ip = parsed.tuple.src_ip;
    let old_dst_ip = parsed.tuple.dst_ip;
    let old_src_port = parsed.tuple.src_port;
    let old_dst_port = parsed.tuple.dst_port;

    let mut ip_csum = u16::from_be_bytes([buf[l3 + 10], buf[l3 + 11]]);

    if let Some(sip) = new_src_ip {
        if sip != old_src_ip {
            buf[l3 + 12..l3 + 16].copy_from_slice(sip.as_bytes());
            ip_csum = csum_update_u32(ip_csum, old_src_ip, sip);
        }
    }
    if let Some(dip) = new_dst_ip {
        if dip != old_dst_ip {
            buf[l3 + 16..l3 + 20].copy_from_slice(dip.as_bytes());
            ip_csum = csum_update_u32(ip_csum, old_dst_ip, dip);
        }
    }
    buf[l3 + 10..l3 + 12].copy_from_slice(&ip_csum.to_be_bytes());

    match parsed.tuple.proto {
        PROTO_TCP => {
            if buf.len() < l4 + TcpHeader::MIN_SIZE {
                return Err(NetfilterError::InvalidL4Header);
            }
            let mut l4_csum = u16::from_be_bytes([buf[l4 + 16], buf[l4 + 17]]);
            if let Some(sip) = new_src_ip {
                if sip != old_src_ip {
                    l4_csum = csum_update_u32(l4_csum, old_src_ip, sip);
                }
            }
            if let Some(dip) = new_dst_ip {
                if dip != old_dst_ip {
                    l4_csum = csum_update_u32(l4_csum, old_dst_ip, dip);
                }
            }
            if let Some(sp) = new_src_port {
                if sp != old_src_port {
                    buf[l4..l4 + 2].copy_from_slice(&sp.to_be_bytes());
                    l4_csum = csum_update_u16(l4_csum, old_src_port, sp);
                }
            }
            if let Some(dp) = new_dst_port {
                if dp != old_dst_port {
                    buf[l4 + 2..l4 + 4].copy_from_slice(&dp.to_be_bytes());
                    l4_csum = csum_update_u16(l4_csum, old_dst_port, dp);
                }
            }
            buf[l4 + 16..l4 + 18].copy_from_slice(&l4_csum.to_be_bytes());
        }
        PROTO_UDP => {
            if buf.len() < l4 + UdpHeader::SIZE {
                return Err(NetfilterError::InvalidL4Header);
            }
            let mut l4_csum = u16::from_be_bytes([buf[l4 + 6], buf[l4 + 7]]);
            if let Some(sp) = new_src_port {
                if sp != old_src_port {
                    buf[l4..l4 + 2].copy_from_slice(&sp.to_be_bytes());
                }
            }
            if let Some(dp) = new_dst_port {
                if dp != old_dst_port {
                    buf[l4 + 2..l4 + 4].copy_from_slice(&dp.to_be_bytes());
                }
            }
            // UDP checksum 0 means "no checksum" in IPv4; only update if non-zero.
            if l4_csum != 0 {
                if let Some(sip) = new_src_ip {
                    if sip != old_src_ip {
                        l4_csum = csum_update_u32(l4_csum, old_src_ip, sip);
                    }
                }
                if let Some(dip) = new_dst_ip {
                    if dip != old_dst_ip {
                        l4_csum = csum_update_u32(l4_csum, old_dst_ip, dip);
                    }
                }
                if let Some(sp) = new_src_port {
                    if sp != old_src_port {
                        l4_csum = csum_update_u16(l4_csum, old_src_port, sp);
                    }
                }
                if let Some(dp) = new_dst_port {
                    if dp != old_dst_port {
                        l4_csum = csum_update_u16(l4_csum, old_dst_port, dp);
                    }
                }
                if l4_csum == 0 {
                    l4_csum = 0xFFFF;
                }
                buf[l4 + 6..l4 + 8].copy_from_slice(&l4_csum.to_be_bytes());
            }
        }
        _ => {}
    }

    Ok(())
}

// ============================================================================
// 5. Stateful Firewall Rules & Hook Chains
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookPoint {
    PreRouting = 0,
    Input = 1,
    Forward = 2,
    Output = 3,
    PostRouting = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Accept,
    Drop,
    Reject,
    Jump(usize),
    Return,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtStateMask(pub u8);

impl CtStateMask {
    pub const NEW: u8 = 1 << 0;
    pub const ESTABLISHED: u8 = 1 << 1;
    pub const RELATED: u8 = 1 << 2;
    pub const INVALID: u8 = 1 << 3;
    pub const ANY: Self = Self(0x0F);

    pub const fn from_state(state: CtState) -> u8 {
        match state {
            CtState::New => Self::NEW,
            CtState::Established => Self::ESTABLISHED,
            CtState::Related => Self::RELATED,
            CtState::Invalid => Self::INVALID,
        }
    }

    pub const fn matches(&self, state: CtState) -> bool {
        (self.0 & Self::from_state(state)) != 0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FirewallRule {
    pub id: u32,
    pub ifindex: Option<u32>,
    pub src_cidr: IpCidr,
    pub dst_cidr: IpCidr,
    pub proto: Option<u8>,
    pub src_port: PortRange,
    pub dst_port: PortRange,
    pub ct_states: CtStateMask,
    pub tcp_flags_mask: u16,
    pub tcp_flags_match: u16,
    pub verdict: Verdict,
    pub packets: u64,
    pub bytes: u64,
}

impl FirewallRule {
    pub const fn new(id: u32, verdict: Verdict) -> Self {
        Self {
            id,
            ifindex: None,
            src_cidr: IpCidr::ANY,
            dst_cidr: IpCidr::ANY,
            proto: None,
            src_port: PortRange::ANY,
            dst_port: PortRange::ANY,
            ct_states: CtStateMask::ANY,
            tcp_flags_mask: 0,
            tcp_flags_match: 0,
            verdict,
            packets: 0,
            bytes: 0,
        }
    }

    pub fn matches(
        &self,
        ifindex: u32,
        tuple: &FiveTuple,
        ct_state: CtState,
        tcp_flags: u16,
    ) -> bool {
        if self.ifindex.is_some_and(|i| i != ifindex) {
            return false;
        }
        if self.proto.is_some_and(|p| p != tuple.proto) {
            return false;
        }
        if !self.src_cidr.matches(tuple.src_ip) || !self.dst_cidr.matches(tuple.dst_ip) {
            return false;
        }
        if !self.src_port.contains(tuple.src_port) || !self.dst_port.contains(tuple.dst_port) {
            return false;
        }
        if !self.ct_states.matches(ct_state) {
            return false;
        }
        if (tcp_flags & self.tcp_flags_mask) != self.tcp_flags_match {
            return false;
        }
        true
    }
}

pub struct FirewallChain {
    pub policy: Verdict,
    pub rules: Vec<FirewallRule>,
}

impl FirewallChain {
    pub fn new(policy: Verdict) -> Self {
        Self {
            policy,
            rules: Vec::new(),
        }
    }
}

pub struct FirewallTable {
    hooks: [FirewallChain; 5],
    custom_chains: Vec<FirewallChain>,
}

impl FirewallTable {
    pub fn new(default_policy: Verdict) -> Self {
        Self {
            hooks: [
                FirewallChain::new(Verdict::Accept),
                FirewallChain::new(default_policy),
                FirewallChain::new(default_policy),
                FirewallChain::new(Verdict::Accept),
                FirewallChain::new(Verdict::Accept),
            ],
            custom_chains: Vec::new(),
        }
    }

    pub fn set_hook_policy(&mut self, hook: HookPoint, policy: Verdict) {
        self.hooks[hook as usize].policy = policy;
    }

    pub fn add_hook_rule(&mut self, hook: HookPoint, rule: FirewallRule) {
        self.hooks[hook as usize].rules.push(rule);
    }

    pub fn create_custom_chain(&mut self, policy: Verdict) -> usize {
        let idx = self.custom_chains.len();
        self.custom_chains.push(FirewallChain::new(policy));
        idx
    }

    pub fn add_custom_rule(
        &mut self,
        chain_idx: usize,
        rule: FirewallRule,
    ) -> Result<(), NetfilterError> {
        let chain = self
            .custom_chains
            .get_mut(chain_idx)
            .ok_or(NetfilterError::InvalidChain)?;
        chain.rules.push(rule);
        Ok(())
    }

    pub fn evaluate(
        &mut self,
        hook: HookPoint,
        ifindex: u32,
        tuple: &FiveTuple,
        ct_state: CtState,
        tcp_flags: u16,
        pkt_len: usize,
    ) -> Result<Verdict, NetfilterError> {
        self.eval_chain(
            None,
            Some(hook),
            ifindex,
            tuple,
            ct_state,
            tcp_flags,
            pkt_len,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_chain(
        &mut self,
        custom_idx: Option<usize>,
        hook: Option<HookPoint>,
        ifindex: u32,
        tuple: &FiveTuple,
        ct_state: CtState,
        tcp_flags: u16,
        pkt_len: usize,
        depth: usize,
    ) -> Result<Verdict, NetfilterError> {
        if depth > 8 {
            return Err(NetfilterError::ChainLoop);
        }

        let chain = match (custom_idx, hook) {
            (Some(idx), _) => self
                .custom_chains
                .get_mut(idx)
                .ok_or(NetfilterError::InvalidChain)?,
            (None, Some(h)) => &mut self.hooks[h as usize],
            (None, None) => return Err(NetfilterError::InvalidChain),
        };

        let mut jump_target: Option<usize> = None;
        let mut matched_verdict: Option<Verdict> = None;

        for rule in &mut chain.rules {
            if rule.matches(ifindex, tuple, ct_state, tcp_flags) {
                rule.packets += 1;
                rule.bytes += pkt_len as u64;
                match rule.verdict {
                    Verdict::Jump(target) => {
                        jump_target = Some(target);
                        break;
                    }
                    Verdict::Return => {
                        return Ok(Verdict::Return);
                    }
                    v => {
                        matched_verdict = Some(v);
                        break;
                    }
                }
            }
        }

        if let Some(v) = matched_verdict {
            return Ok(v);
        }

        if let Some(target) = jump_target {
            let sub = self.eval_chain(
                Some(target),
                None,
                ifindex,
                tuple,
                ct_state,
                tcp_flags,
                pkt_len,
                depth + 1,
            )?;
            if sub != Verdict::Return {
                return Ok(sub);
            }
            // If sub returned Return, fall through to current chain policy.
        }

        let policy = match (custom_idx, hook) {
            (Some(idx), _) => self.custom_chains[idx].policy,
            (None, Some(h)) => self.hooks[h as usize].policy,
            (None, None) => Verdict::Drop,
        };
        Ok(policy)
    }
}

// ============================================================================
// 6. eBPF-Style XDP Hook Engine (ISA, Verifier, Maps, VM)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdpAction {
    Aborted,
    Drop,
    Pass,
    Tx,
    Redirect(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdpVerifyError {
    EmptyProgram,
    ProgramTooLong,
    InvalidRegister,
    ReadOnlyR10,
    InvalidOpcode,
    JumpOutOfBounds,
    LoopDetected,
    UninitializedRegister,
    DivisionByZero,
    StackOutOfBounds,
    InvalidMemoryAccess,
    InvalidHelper,
    ExitWithoutR0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdpRuntimeError {
    StepLimitExceeded,
    OutOfBoundsPacketAccess,
    OutOfBoundsStackAccess,
    InvalidPointer,
    DivisionByZero,
    InvalidMapId,
}

/// eBPF-style 64-bit instruction (8 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BpfInsn {
    pub opcode: u8,
    pub dst_reg: u8,
    pub src_reg: u8,
    pub off: i16,
    pub imm: i32,
}

// Instruction classes (low 3 bits of opcode)
pub const BPF_LD: u8 = 0x00;
pub const BPF_LDX: u8 = 0x01;
pub const BPF_ST: u8 = 0x02;
pub const BPF_STX: u8 = 0x03;
pub const BPF_ALU: u8 = 0x04;
pub const BPF_JMP: u8 = 0x05;
pub const BPF_JMP32: u8 = 0x06;
pub const BPF_ALU64: u8 = 0x07;

// Source operand bit
pub const BPF_K: u8 = 0x00;
pub const BPF_X: u8 = 0x08;

// Size modifiers for LDX/ST/STX
pub const BPF_W: u8 = 0x00; // 32-bit
pub const BPF_H: u8 = 0x08; // 16-bit
pub const BPF_B: u8 = 0x10; // 8-bit
pub const BPF_DW: u8 = 0x18; // 64-bit
pub const BPF_MEM: u8 = 0x60;

// ALU operations (high 4 bits)
pub const BPF_ADD: u8 = 0x00;
pub const BPF_SUB: u8 = 0x10;
pub const BPF_MUL: u8 = 0x20;
pub const BPF_DIV: u8 = 0x30;
pub const BPF_OR: u8 = 0x40;
pub const BPF_AND: u8 = 0x50;
pub const BPF_LSH: u8 = 0x60;
pub const BPF_RSH: u8 = 0x70;
pub const BPF_NEG: u8 = 0x80;
pub const BPF_MOD: u8 = 0x90;
pub const BPF_XOR: u8 = 0xa0;
pub const BPF_MOV: u8 = 0xb0;
pub const BPF_ARSH: u8 = 0xc0;
pub const BPF_END: u8 = 0xd0;

// JMP operations (high 4 bits)
pub const BPF_JA: u8 = 0x00;
pub const BPF_JEQ: u8 = 0x10;
pub const BPF_JGT: u8 = 0x20;
pub const BPF_JGE: u8 = 0x30;
pub const BPF_JSET: u8 = 0x40;
pub const BPF_JNE: u8 = 0x50;
pub const BPF_JSGT: u8 = 0x60;
pub const BPF_JSGE: u8 = 0x70;
pub const BPF_CALL: u8 = 0x80;
pub const BPF_EXIT: u8 = 0x90;
pub const BPF_JLT: u8 = 0xa0;
pub const BPF_JLE: u8 = 0xb0;
pub const BPF_JSLT: u8 = 0xc0;
pub const BPF_JSLE: u8 = 0xd0;

// Helper IDs
pub const HELPER_MAP_LOOKUP_ELEM: i32 = 1;
pub const HELPER_MAP_UPDATE_ELEM: i32 = 2;
pub const HELPER_MAP_DELETE_ELEM: i32 = 3;
pub const HELPER_KTIME_GET_NS: i32 = 4;
pub const HELPER_XDP_ADJUST_HEAD: i32 = 5;

// Virtual address regions for XDP VM
const CTX_BASE: u64 = 0x1000_0000;
const PKT_BASE: u64 = 0x2000_0000;
const STACK_TOP: u64 = 0x3000_0000;
const STACK_SIZE: usize = 512;
const MAP_VAL_BASE: u64 = 0x4000_0000;

impl BpfInsn {
    pub const fn new(opcode: u8, dst_reg: u8, src_reg: u8, off: i16, imm: i32) -> Self {
        Self {
            opcode,
            dst_reg,
            src_reg,
            off,
            imm,
        }
    }

    pub const fn mov64_imm(dst: u8, imm: i32) -> Self {
        Self::new(BPF_ALU64 | BPF_MOV | BPF_K, dst, 0, 0, imm)
    }

    pub const fn mov64_reg(dst: u8, src: u8) -> Self {
        Self::new(BPF_ALU64 | BPF_MOV | BPF_X, dst, src, 0, 0)
    }

    pub const fn alu64_imm(op: u8, dst: u8, imm: i32) -> Self {
        Self::new(BPF_ALU64 | op | BPF_K, dst, 0, 0, imm)
    }

    pub const fn alu64_reg(op: u8, dst: u8, src: u8) -> Self {
        Self::new(BPF_ALU64 | op | BPF_X, dst, src, 0, 0)
    }

    pub const fn ldx_mem(size: u8, dst: u8, src: u8, off: i16) -> Self {
        Self::new(BPF_LDX | size | BPF_MEM, dst, src, off, 0)
    }

    pub const fn stx_mem(size: u8, dst: u8, src: u8, off: i16) -> Self {
        Self::new(BPF_STX | size | BPF_MEM, dst, src, off, 0)
    }

    pub const fn st_mem(size: u8, dst: u8, off: i16, imm: i32) -> Self {
        Self::new(BPF_ST | size | BPF_MEM, dst, 0, off, imm)
    }

    pub const fn jmp_imm(op: u8, dst: u8, imm: i32, off: i16) -> Self {
        Self::new(BPF_JMP | op | BPF_K, dst, 0, off, imm)
    }

    pub const fn jmp_reg(op: u8, dst: u8, src: u8, off: i16) -> Self {
        Self::new(BPF_JMP | op | BPF_X, dst, src, off, 0)
    }

    pub const fn ja(off: i16) -> Self {
        Self::new(BPF_JMP | BPF_JA, 0, 0, off, 0)
    }

    pub const fn call(helper_id: i32) -> Self {
        Self::new(BPF_JMP | BPF_CALL, 0, 0, 0, helper_id)
    }

    pub const fn exit() -> Self {
        Self::new(BPF_JMP | BPF_EXIT, 0, 0, 0, 0)
    }

    pub const fn be16(dst: u8) -> Self {
        Self::new(BPF_ALU | BPF_END | BPF_X, dst, 0, 0, 16)
    }

    pub const fn be32(dst: u8) -> Self {
        Self::new(BPF_ALU | BPF_END | BPF_X, dst, 0, 0, 32)
    }
}

/// Simple fixed-key/value eBPF map (array or hash-like linear table).
pub struct XdpMap {
    pub key_size: usize,
    pub value_size: usize,
    pub max_entries: usize,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
}

impl XdpMap {
    pub fn new(key_size: usize, value_size: usize, max_entries: usize) -> Self {
        Self {
            key_size,
            value_size,
            max_entries,
            entries: Vec::new(),
        }
    }

    pub fn lookup(&self, key: &[u8]) -> Option<&[u8]> {
        if key.len() != self.key_size {
            return None;
        }
        self.entries
            .iter()
            .find(|(k, _)| k.as_slice() == key)
            .map(|(_, v)| v.as_slice())
    }

    pub fn update(&mut self, key: &[u8], val: &[u8]) -> bool {
        if key.len() != self.key_size || val.len() != self.value_size {
            return false;
        }
        if let Some((_, existing)) = self.entries.iter_mut().find(|(k, _)| k.as_slice() == key) {
            existing.copy_from_slice(val);
            return true;
        }
        if self.entries.len() >= self.max_entries {
            return false;
        }
        self.entries.push((key.to_vec(), val.to_vec()));
        true
    }

    pub fn delete(&mut self, key: &[u8]) -> bool {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k.as_slice() == key) {
            self.entries.swap_remove(pos);
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegType {
    Uninitialized,
    Scalar,
    PtrToCtx,
    PtrToPacket,
    PtrToPacketEnd,
    PtrToStack,
    PtrToMapValue,
}

/// Verified XDP program wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedXdpProgram {
    insns: Vec<BpfInsn>,
}

impl VerifiedXdpProgram {
    pub fn insns(&self) -> &[BpfInsn] {
        &self.insns
    }
}

/// Static eBPF-style bytecode verifier.
pub struct XdpVerifier;

impl XdpVerifier {
    pub const MAX_INSNS: usize = 4096;

    pub fn verify(insns: &[BpfInsn]) -> Result<VerifiedXdpProgram, XdpVerifyError> {
        if insns.is_empty() {
            return Err(XdpVerifyError::EmptyProgram);
        }
        if insns.len() > Self::MAX_INSNS {
            return Err(XdpVerifyError::ProgramTooLong);
        }

        // Pass 1: Structural, opcode, register, and jump-target validation.
        for (pc, insn) in insns.iter().enumerate() {
            if insn.dst_reg > 10 || insn.src_reg > 10 {
                return Err(XdpVerifyError::InvalidRegister);
            }
            let class = insn.opcode & 0x07;
            match class {
                BPF_ALU | BPF_ALU64 => {
                    if insn.dst_reg == 10 {
                        return Err(XdpVerifyError::ReadOnlyR10);
                    }
                    let op = insn.opcode & 0xF0;
                    let src_bit = insn.opcode & 0x08;
                    match op {
                        BPF_ADD | BPF_SUB | BPF_MUL | BPF_DIV | BPF_OR | BPF_AND | BPF_LSH
                        | BPF_RSH | BPF_NEG | BPF_MOD | BPF_XOR | BPF_MOV | BPF_ARSH => {
                            if (op == BPF_DIV || op == BPF_MOD) && src_bit == BPF_K && insn.imm == 0
                            {
                                return Err(XdpVerifyError::DivisionByZero);
                            }
                        }
                        BPF_END if class == BPF_ALU => {
                            if !matches!(insn.imm, 16 | 32 | 64) {
                                return Err(XdpVerifyError::InvalidOpcode);
                            }
                        }
                        _ => return Err(XdpVerifyError::InvalidOpcode),
                    }
                }
                BPF_LDX => {
                    if insn.dst_reg == 10 {
                        return Err(XdpVerifyError::ReadOnlyR10);
                    }
                    if (insn.opcode & 0xE0) != BPF_MEM {
                        return Err(XdpVerifyError::InvalidOpcode);
                    }
                }
                BPF_ST | BPF_STX => {
                    if (insn.opcode & 0xE0) != BPF_MEM {
                        return Err(XdpVerifyError::InvalidOpcode);
                    }
                }
                BPF_JMP | BPF_JMP32 => {
                    let op = insn.opcode & 0xF0;
                    match op {
                        BPF_EXIT => {}
                        BPF_CALL => {
                            if !matches!(
                                insn.imm,
                                HELPER_MAP_LOOKUP_ELEM
                                    | HELPER_MAP_UPDATE_ELEM
                                    | HELPER_MAP_DELETE_ELEM
                                    | HELPER_KTIME_GET_NS
                                    | HELPER_XDP_ADJUST_HEAD
                            ) {
                                return Err(XdpVerifyError::InvalidHelper);
                            }
                        }
                        BPF_JA | BPF_JEQ | BPF_JGT | BPF_JGE | BPF_JSET | BPF_JNE | BPF_JSGT
                        | BPF_JSGE | BPF_JLT | BPF_JLE | BPF_JSLT | BPF_JSLE => {
                            let target = (pc as isize) + 1 + (insn.off as isize);
                            if target < 0 || (target as usize) >= insns.len() {
                                return Err(XdpVerifyError::JumpOutOfBounds);
                            }
                        }
                        _ => return Err(XdpVerifyError::InvalidOpcode),
                    }
                }
                _ => return Err(XdpVerifyError::InvalidOpcode),
            }
        }

        // Pass 2: Cycle detection (DAG check) via iterative DFS with 3-color marking.
        // 0 = unvisited, 1 = visiting (on stack), 2 = visited.
        let mut color = vec![0u8; insns.len()];
        let mut dfs_stack: Vec<(usize, u8)> = Vec::new();
        dfs_stack.push((0, 0));

        while let Some((pc, phase)) = dfs_stack.pop() {
            if phase == 1 {
                color[pc] = 2;
                continue;
            }
            if color[pc] == 2 {
                continue;
            }
            if color[pc] == 1 {
                return Err(XdpVerifyError::LoopDetected);
            }
            color[pc] = 1;
            dfs_stack.push((pc, 1));

            let insn = insns[pc];
            let class = insn.opcode & 0x07;
            if class == BPF_JMP || class == BPF_JMP32 {
                let op = insn.opcode & 0xF0;
                match op {
                    BPF_EXIT => {}
                    BPF_JA => {
                        let target = ((pc as isize) + 1 + (insn.off as isize)) as usize;
                        if color[target] == 1 {
                            return Err(XdpVerifyError::LoopDetected);
                        }
                        if color[target] == 0 {
                            dfs_stack.push((target, 0));
                        }
                    }
                    BPF_CALL => {
                        if pc + 1 >= insns.len() {
                            return Err(XdpVerifyError::JumpOutOfBounds);
                        }
                        if color[pc + 1] == 1 {
                            return Err(XdpVerifyError::LoopDetected);
                        }
                        if color[pc + 1] == 0 {
                            dfs_stack.push((pc + 1, 0));
                        }
                    }
                    _ => {
                        let target = ((pc as isize) + 1 + (insn.off as isize)) as usize;
                        if color[target] == 1 {
                            return Err(XdpVerifyError::LoopDetected);
                        }
                        if color[target] == 0 {
                            dfs_stack.push((target, 0));
                        }
                        if pc + 1 >= insns.len() {
                            return Err(XdpVerifyError::JumpOutOfBounds);
                        }
                        if color[pc + 1] == 1 {
                            return Err(XdpVerifyError::LoopDetected);
                        }
                        if color[pc + 1] == 0 {
                            dfs_stack.push((pc + 1, 0));
                        }
                    }
                }
            } else {
                if pc + 1 >= insns.len() {
                    return Err(XdpVerifyError::JumpOutOfBounds);
                }
                if color[pc + 1] == 1 {
                    return Err(XdpVerifyError::LoopDetected);
                }
                if color[pc + 1] == 0 {
                    dfs_stack.push((pc + 1, 0));
                }
            }
        }

        // Pass 3: Abstract interpretation of register types & stack bounds along reachable paths.
        let mut init_regs = [RegType::Uninitialized; 11];
        init_regs[1] = RegType::PtrToCtx;
        init_regs[10] = RegType::PtrToStack;

        let mut worklist: Vec<(usize, [RegType; 11])> = Vec::new();
        worklist.push((0, init_regs));

        while let Some((pc, mut regs)) = worklist.pop() {
            let insn = insns[pc];
            let class = insn.opcode & 0x07;
            let op = insn.opcode & 0xF0;
            let src_bit = insn.opcode & 0x08;

            match class {
                BPF_ALU | BPF_ALU64 => {
                    if op == BPF_MOV {
                        if src_bit == BPF_X {
                            if regs[insn.src_reg as usize] == RegType::Uninitialized {
                                return Err(XdpVerifyError::UninitializedRegister);
                            }
                            regs[insn.dst_reg as usize] = regs[insn.src_reg as usize];
                        } else {
                            regs[insn.dst_reg as usize] = RegType::Scalar;
                        }
                    } else if op == BPF_NEG || op == BPF_END {
                        if regs[insn.dst_reg as usize] == RegType::Uninitialized {
                            return Err(XdpVerifyError::UninitializedRegister);
                        }
                        regs[insn.dst_reg as usize] = RegType::Scalar;
                    } else {
                        if regs[insn.dst_reg as usize] == RegType::Uninitialized {
                            return Err(XdpVerifyError::UninitializedRegister);
                        }
                        if src_bit == BPF_X && regs[insn.src_reg as usize] == RegType::Uninitialized
                        {
                            return Err(XdpVerifyError::UninitializedRegister);
                        }
                        // Pointer arithmetic on packet/stack preserves pointer type for ADD/SUB.
                        let dst_t = regs[insn.dst_reg as usize];
                        if (op == BPF_ADD || op == BPF_SUB)
                            && matches!(dst_t, RegType::PtrToPacket | RegType::PtrToStack)
                        {
                            regs[insn.dst_reg as usize] = dst_t;
                        } else {
                            regs[insn.dst_reg as usize] = RegType::Scalar;
                        }
                    }
                    worklist.push((pc + 1, regs));
                }
                BPF_LDX => {
                    let src_t = regs[insn.src_reg as usize];
                    match src_t {
                        RegType::Uninitialized => {
                            return Err(XdpVerifyError::UninitializedRegister)
                        }
                        RegType::Scalar => return Err(XdpVerifyError::InvalidMemoryAccess),
                        RegType::PtrToStack => {
                            if insn.off >= 0 || (insn.off as i32) < -(STACK_SIZE as i32) {
                                return Err(XdpVerifyError::StackOutOfBounds);
                            }
                            regs[insn.dst_reg as usize] = RegType::Scalar;
                        }
                        RegType::PtrToCtx => {
                            regs[insn.dst_reg as usize] = match insn.off {
                                0 => RegType::PtrToPacket,
                                4 => RegType::PtrToPacketEnd,
                                8 | 12 => RegType::Scalar,
                                _ => return Err(XdpVerifyError::InvalidMemoryAccess),
                            };
                        }
                        RegType::PtrToPacket | RegType::PtrToMapValue => {
                            regs[insn.dst_reg as usize] = RegType::Scalar;
                        }
                        RegType::PtrToPacketEnd => return Err(XdpVerifyError::InvalidMemoryAccess),
                    }
                    worklist.push((pc + 1, regs));
                }
                BPF_ST | BPF_STX => {
                    if class == BPF_STX && regs[insn.src_reg as usize] == RegType::Uninitialized {
                        return Err(XdpVerifyError::UninitializedRegister);
                    }
                    match regs[insn.dst_reg as usize] {
                        RegType::Uninitialized => {
                            return Err(XdpVerifyError::UninitializedRegister)
                        }
                        RegType::PtrToStack => {
                            if insn.off >= 0 || (insn.off as i32) < -(STACK_SIZE as i32) {
                                return Err(XdpVerifyError::StackOutOfBounds);
                            }
                        }
                        RegType::PtrToPacket | RegType::PtrToMapValue => {}
                        _ => return Err(XdpVerifyError::InvalidMemoryAccess),
                    }
                    worklist.push((pc + 1, regs));
                }
                BPF_JMP | BPF_JMP32 => match op {
                    BPF_EXIT => {
                        if regs[0] == RegType::Uninitialized {
                            return Err(XdpVerifyError::ExitWithoutR0);
                        }
                    }
                    BPF_JA => {
                        let target = ((pc as isize) + 1 + (insn.off as isize)) as usize;
                        worklist.push((target, regs));
                    }
                    BPF_CALL => {
                        // Caller-saved R1..R5 clobbered; R0 set by helper.
                        regs[0] = if insn.imm == HELPER_MAP_LOOKUP_ELEM {
                            RegType::PtrToMapValue
                        } else {
                            RegType::Scalar
                        };
                        for r in &mut regs[1..=5] {
                            *r = RegType::Uninitialized;
                        }
                        worklist.push((pc + 1, regs));
                    }
                    _ => {
                        if regs[insn.dst_reg as usize] == RegType::Uninitialized {
                            return Err(XdpVerifyError::UninitializedRegister);
                        }
                        if src_bit == BPF_X && regs[insn.src_reg as usize] == RegType::Uninitialized
                        {
                            return Err(XdpVerifyError::UninitializedRegister);
                        }
                        let target = ((pc as isize) + 1 + (insn.off as isize)) as usize;
                        worklist.push((target, regs));
                        worklist.push((pc + 1, regs));
                    }
                },
                _ => return Err(XdpVerifyError::InvalidOpcode),
            }
        }

        Ok(VerifiedXdpProgram {
            insns: insns.to_vec(),
        })
    }
}

pub struct XdpContext<'a> {
    pub data: &'a mut [u8],
    pub data_offset: usize,
    pub ingress_ifindex: u32,
    pub rx_queue_index: u32,
    pub now_ns: u64,
}

pub struct XdpVm;

impl XdpVm {
    pub const MAX_STEPS: usize = 65536;

    pub fn execute(
        prog: &VerifiedXdpProgram,
        ctx: &mut XdpContext<'_>,
        maps: &mut [XdpMap],
    ) -> Result<XdpAction, XdpRuntimeError> {
        let mut regs = [0u64; 11];
        regs[1] = CTX_BASE;
        regs[10] = STACK_TOP;

        // Heap-allocate 512-byte stack & map scratch buffer to keep kernel stack frame tiny.
        let mut stack: Box<[u8; STACK_SIZE]> = Box::new([0u8; STACK_SIZE]);
        let mut map_val_buf: Box<[u8; 256]> = Box::new([0u8; 256]);
        let mut map_val_len: usize = 0;

        let insns = prog.insns();
        let mut pc: usize = 0;
        let mut steps: usize = 0;

        while pc < insns.len() {
            steps += 1;
            if steps > Self::MAX_STEPS {
                return Err(XdpRuntimeError::StepLimitExceeded);
            }

            let insn = insns[pc];
            let class = insn.opcode & 0x07;
            let op = insn.opcode & 0xF0;
            let src_bit = insn.opcode & 0x08;

            match class {
                BPF_ALU64 => {
                    let dst = regs[insn.dst_reg as usize];
                    let src = if src_bit == BPF_X {
                        regs[insn.src_reg as usize]
                    } else {
                        insn.imm as i64 as u64
                    };
                    regs[insn.dst_reg as usize] = match op {
                        BPF_ADD => dst.wrapping_add(src),
                        BPF_SUB => dst.wrapping_sub(src),
                        BPF_MUL => dst.wrapping_mul(src),
                        BPF_DIV => {
                            if src == 0 {
                                return Err(XdpRuntimeError::DivisionByZero);
                            }
                            dst / src
                        }
                        BPF_MOD => {
                            if src == 0 {
                                return Err(XdpRuntimeError::DivisionByZero);
                            }
                            dst % src
                        }
                        BPF_OR => dst | src,
                        BPF_AND => dst & src,
                        BPF_LSH => dst.wrapping_shl((src & 63) as u32),
                        BPF_RSH => dst.wrapping_shr((src & 63) as u32),
                        BPF_ARSH => ((dst as i64).wrapping_shr((src & 63) as u32)) as u64,
                        BPF_XOR => dst ^ src,
                        BPF_NEG => (-(dst as i64)) as u64,
                        BPF_MOV => src,
                        _ => dst,
                    };
                    pc += 1;
                }
                BPF_ALU => {
                    if op == BPF_END {
                        let v = regs[insn.dst_reg as usize];
                        regs[insn.dst_reg as usize] = match insn.imm {
                            16 => (v as u16).swap_bytes() as u64,
                            32 => (v as u32).swap_bytes() as u64,
                            64 => v.swap_bytes(),
                            _ => v,
                        };
                    } else {
                        let dst = regs[insn.dst_reg as usize] as u32;
                        let src = if src_bit == BPF_X {
                            regs[insn.src_reg as usize] as u32
                        } else {
                            insn.imm as u32
                        };
                        let res = match op {
                            BPF_ADD => dst.wrapping_add(src),
                            BPF_SUB => dst.wrapping_sub(src),
                            BPF_MUL => dst.wrapping_mul(src),
                            BPF_DIV => {
                                if src == 0 {
                                    return Err(XdpRuntimeError::DivisionByZero);
                                }
                                dst / src
                            }
                            BPF_MOD => {
                                if src == 0 {
                                    return Err(XdpRuntimeError::DivisionByZero);
                                }
                                dst % src
                            }
                            BPF_OR => dst | src,
                            BPF_AND => dst & src,
                            BPF_LSH => dst.wrapping_shl(src & 31),
                            BPF_RSH => dst.wrapping_shr(src & 31),
                            BPF_ARSH => ((dst as i32).wrapping_shr(src & 31)) as u32,
                            BPF_XOR => dst ^ src,
                            BPF_NEG => (-(dst as i32)) as u32,
                            BPF_MOV => src,
                            _ => dst,
                        };
                        regs[insn.dst_reg as usize] = res as u64;
                    }
                    pc += 1;
                }
                BPF_LDX => {
                    let addr = (regs[insn.src_reg as usize] as i64 + insn.off as i64) as u64;
                    let size = match insn.opcode & 0x18 {
                        BPF_B => 1,
                        BPF_H => 2,
                        BPF_W => 4,
                        BPF_DW => 8,
                        _ => 4,
                    };
                    regs[insn.dst_reg as usize] =
                        Self::mem_read(addr, size, ctx, &stack, &map_val_buf[..map_val_len])?;
                    pc += 1;
                }
                BPF_ST | BPF_STX => {
                    let addr = (regs[insn.dst_reg as usize] as i64 + insn.off as i64) as u64;
                    let size = match insn.opcode & 0x18 {
                        BPF_B => 1,
                        BPF_H => 2,
                        BPF_W => 4,
                        BPF_DW => 8,
                        _ => 4,
                    };
                    let val = if class == BPF_STX {
                        regs[insn.src_reg as usize]
                    } else {
                        insn.imm as i64 as u64
                    };
                    Self::mem_write(
                        addr,
                        size,
                        val,
                        ctx,
                        &mut stack,
                        &mut map_val_buf[..map_val_len],
                    )?;
                    pc += 1;
                }
                BPF_JMP | BPF_JMP32 => {
                    let is_32 = class == BPF_JMP32;
                    match op {
                        BPF_EXIT => {
                            let code = regs[0] as u32;
                            return Ok(match code {
                                0 => XdpAction::Aborted,
                                1 => XdpAction::Drop,
                                2 => XdpAction::Pass,
                                3 => XdpAction::Tx,
                                r => XdpAction::Redirect(r),
                            });
                        }
                        BPF_JA => {
                            pc = ((pc as isize) + 1 + (insn.off as isize)) as usize;
                        }
                        BPF_CALL => {
                            regs[0] = Self::call_helper(
                                insn.imm,
                                &regs,
                                ctx,
                                &stack,
                                maps,
                                &mut map_val_buf,
                                &mut map_val_len,
                            )?;
                            pc += 1;
                        }
                        _ => {
                            let dst = regs[insn.dst_reg as usize];
                            let src = if src_bit == BPF_X {
                                regs[insn.src_reg as usize]
                            } else {
                                insn.imm as i64 as u64
                            };
                            let taken = if is_32 {
                                let d = dst as u32;
                                let s = src as u32;
                                match op {
                                    BPF_JEQ => d == s,
                                    BPF_JNE => d != s,
                                    BPF_JGT => d > s,
                                    BPF_JGE => d >= s,
                                    BPF_JLT => d < s,
                                    BPF_JLE => d <= s,
                                    BPF_JSET => (d & s) != 0,
                                    BPF_JSGT => (d as i32) > (s as i32),
                                    BPF_JSGE => (d as i32) >= (s as i32),
                                    BPF_JSLT => (d as i32) < (s as i32),
                                    BPF_JSLE => (d as i32) <= (s as i32),
                                    _ => false,
                                }
                            } else {
                                match op {
                                    BPF_JEQ => dst == src,
                                    BPF_JNE => dst != src,
                                    BPF_JGT => dst > src,
                                    BPF_JGE => dst >= src,
                                    BPF_JLT => dst < src,
                                    BPF_JLE => dst <= src,
                                    BPF_JSET => (dst & src) != 0,
                                    BPF_JSGT => (dst as i64) > (src as i64),
                                    BPF_JSGE => (dst as i64) >= (src as i64),
                                    BPF_JSLT => (dst as i64) < (src as i64),
                                    BPF_JSLE => (dst as i64) <= (src as i64),
                                    _ => false,
                                }
                            };
                            if taken {
                                pc = ((pc as isize) + 1 + (insn.off as isize)) as usize;
                            } else {
                                pc += 1;
                            }
                        }
                    }
                }
                _ => pc += 1,
            }
        }

        Ok(XdpAction::Pass)
    }

    fn mem_read(
        addr: u64,
        size: usize,
        ctx: &XdpContext<'_>,
        stack: &[u8; STACK_SIZE],
        map_val: &[u8],
    ) -> Result<u64, XdpRuntimeError> {
        if (CTX_BASE..CTX_BASE + 16).contains(&addr) {
            let off = (addr - CTX_BASE) as usize;
            return Ok(match off {
                0 => PKT_BASE + ctx.data_offset as u64,
                4 => PKT_BASE + ctx.data.len() as u64,
                8 => ctx.ingress_ifindex as u64,
                12 => ctx.rx_queue_index as u64,
                _ => return Err(XdpRuntimeError::InvalidPointer),
            });
        }

        let slice = if addr >= PKT_BASE && addr < PKT_BASE + (ctx.data.len() as u64) {
            let off = (addr - PKT_BASE) as usize;
            if off + size > ctx.data.len() {
                return Err(XdpRuntimeError::OutOfBoundsPacketAccess);
            }
            &ctx.data[off..off + size]
        } else if addr >= STACK_TOP - (STACK_SIZE as u64) && addr < STACK_TOP {
            let off = (addr - (STACK_TOP - STACK_SIZE as u64)) as usize;
            if off + size > STACK_SIZE {
                return Err(XdpRuntimeError::OutOfBoundsStackAccess);
            }
            &stack[off..off + size]
        } else if addr >= MAP_VAL_BASE && addr < MAP_VAL_BASE + (map_val.len() as u64) {
            let off = (addr - MAP_VAL_BASE) as usize;
            if off + size > map_val.len() {
                return Err(XdpRuntimeError::InvalidPointer);
            }
            &map_val[off..off + size]
        } else {
            return Err(XdpRuntimeError::OutOfBoundsPacketAccess);
        };

        Ok(match size {
            1 => slice[0] as u64,
            2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
            4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
            8 => u64::from_le_bytes([
                slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
            ]),
            _ => return Err(XdpRuntimeError::InvalidPointer),
        })
    }

    fn mem_write(
        addr: u64,
        size: usize,
        val: u64,
        ctx: &mut XdpContext<'_>,
        stack: &mut [u8; STACK_SIZE],
        map_val: &mut [u8],
    ) -> Result<(), XdpRuntimeError> {
        let slice = if addr >= PKT_BASE && addr < PKT_BASE + (ctx.data.len() as u64) {
            let off = (addr - PKT_BASE) as usize;
            if off + size > ctx.data.len() {
                return Err(XdpRuntimeError::OutOfBoundsPacketAccess);
            }
            &mut ctx.data[off..off + size]
        } else if addr >= STACK_TOP - (STACK_SIZE as u64) && addr < STACK_TOP {
            let off = (addr - (STACK_TOP - STACK_SIZE as u64)) as usize;
            if off + size > STACK_SIZE {
                return Err(XdpRuntimeError::OutOfBoundsStackAccess);
            }
            &mut stack[off..off + size]
        } else if addr >= MAP_VAL_BASE && addr < MAP_VAL_BASE + (map_val.len() as u64) {
            let off = (addr - MAP_VAL_BASE) as usize;
            if off + size > map_val.len() {
                return Err(XdpRuntimeError::InvalidPointer);
            }
            &mut map_val[off..off + size]
        } else {
            return Err(XdpRuntimeError::InvalidPointer);
        };

        match size {
            1 => slice[0] = val as u8,
            2 => slice.copy_from_slice(&(val as u16).to_le_bytes()),
            4 => slice.copy_from_slice(&(val as u32).to_le_bytes()),
            8 => slice.copy_from_slice(&val.to_le_bytes()),
            _ => return Err(XdpRuntimeError::InvalidPointer),
        }
        Ok(())
    }

    fn read_stack_slice(
        addr: u64,
        len: usize,
        stack: &[u8; STACK_SIZE],
    ) -> Result<&[u8], XdpRuntimeError> {
        let base = STACK_TOP - STACK_SIZE as u64;
        if addr < base || addr >= STACK_TOP {
            return Err(XdpRuntimeError::OutOfBoundsStackAccess);
        }
        let off = (addr - base) as usize;
        if off + len > STACK_SIZE {
            return Err(XdpRuntimeError::OutOfBoundsStackAccess);
        }
        Ok(&stack[off..off + len])
    }

    fn call_helper(
        helper_id: i32,
        regs: &[u64; 11],
        ctx: &mut XdpContext<'_>,
        stack: &[u8; STACK_SIZE],
        maps: &mut [XdpMap],
        map_val_buf: &mut [u8; 256],
        map_val_len: &mut usize,
    ) -> Result<u64, XdpRuntimeError> {
        match helper_id {
            HELPER_MAP_LOOKUP_ELEM => {
                let map_id = regs[1] as usize;
                let map = maps.get(map_id).ok_or(XdpRuntimeError::InvalidMapId)?;
                let key = Self::read_stack_slice(regs[2], map.key_size, stack)?;
                if let Some(val) = map.lookup(key) {
                    let copy_len = val.len().min(map_val_buf.len());
                    map_val_buf[..copy_len].copy_from_slice(&val[..copy_len]);
                    *map_val_len = copy_len;
                    Ok(MAP_VAL_BASE)
                } else {
                    Ok(0)
                }
            }
            HELPER_MAP_UPDATE_ELEM => {
                let map_id = regs[1] as usize;
                let map = maps.get_mut(map_id).ok_or(XdpRuntimeError::InvalidMapId)?;
                let key = Self::read_stack_slice(regs[2], map.key_size, stack)?;
                let val = Self::read_stack_slice(regs[3], map.value_size, stack)?;
                Ok(if map.update(key, val) { 0 } else { u64::MAX })
            }
            HELPER_MAP_DELETE_ELEM => {
                let map_id = regs[1] as usize;
                let map = maps.get_mut(map_id).ok_or(XdpRuntimeError::InvalidMapId)?;
                let key = Self::read_stack_slice(regs[2], map.key_size, stack)?;
                Ok(if map.delete(key) { 0 } else { u64::MAX })
            }
            HELPER_KTIME_GET_NS => Ok(ctx.now_ns),
            HELPER_XDP_ADJUST_HEAD => {
                let delta = regs[2] as i32 as isize;
                let new_off = (ctx.data_offset as isize) + delta;
                if new_off < 0 || (new_off as usize) > ctx.data.len() {
                    Ok(u64::MAX)
                } else {
                    ctx.data_offset = new_off as usize;
                    Ok(0)
                }
            }
            _ => Ok(u64::MAX),
        }
    }
}

// ============================================================================
// 7. Unified Netfilter Engine & Global State
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetfilterDecision {
    Accept,
    Drop,
    Reject,
    XdpTx,
    XdpRedirect(u32),
}

pub struct NetfilterEngine {
    pub conntrack: ConntrackTable,
    pub nat: NatTable,
    pub firewall: FirewallTable,
    pub xdp_hooks: Vec<(u32, VerifiedXdpProgram)>,
    pub xdp_maps: Vec<XdpMap>,
}

impl NetfilterEngine {
    pub fn new() -> Self {
        Self {
            conntrack: ConntrackTable::new(ConntrackTable::DEFAULT_MAX_ENTRIES),
            nat: NatTable::new(),
            firewall: FirewallTable::new(Verdict::Accept),
            xdp_hooks: Vec::new(),
            xdp_maps: Vec::new(),
        }
    }

    pub fn attach_xdp(&mut self, ifindex: u32, prog: VerifiedXdpProgram) {
        if let Some((_, existing)) = self.xdp_hooks.iter_mut().find(|(i, _)| *i == ifindex) {
            *existing = prog;
        } else {
            self.xdp_hooks.push((ifindex, prog));
        }
    }

    pub fn add_xdp_map(&mut self, map: XdpMap) -> usize {
        let id = self.xdp_maps.len();
        self.xdp_maps.push(map);
        id
    }

    /// Process a packet through XDP (on `PreRouting`), Conntrack, NAT, and Firewall.
    pub fn process_packet(
        &mut self,
        ifindex: u32,
        hook: HookPoint,
        frame: &mut [u8],
        has_eth: bool,
        now_tick: u64,
    ) -> Result<NetfilterDecision, NetfilterError> {
        // 1. XDP hook on PreRouting
        if hook == HookPoint::PreRouting {
            if let Some((_, prog)) = self.xdp_hooks.iter().find(|(i, _)| *i == ifindex) {
                let mut ctx = XdpContext {
                    data: frame,
                    data_offset: 0,
                    ingress_ifindex: ifindex,
                    rx_queue_index: 0,
                    now_ns: now_tick.saturating_mul(1_000_000),
                };
                let action = XdpVm::execute(prog, &mut ctx, &mut self.xdp_maps)
                    .map_err(NetfilterError::XdpRuntime)?;
                match action {
                    XdpAction::Aborted | XdpAction::Drop => return Ok(NetfilterDecision::Drop),
                    XdpAction::Tx => return Ok(NetfilterDecision::XdpTx),
                    XdpAction::Redirect(target) => {
                        return Ok(NetfilterDecision::XdpRedirect(target));
                    }
                    XdpAction::Pass => {}
                }
            }
        }

        // 2. Parse L3/L4
        let mut parsed = parse_ipv4_packet(frame, has_eth)?;

        // 3. Conntrack lookup or creation
        let (ct_idx, dir) = if let Some((idx, d)) = self.conntrack.lookup(&parsed.tuple) {
            (idx, d)
        } else {
            let mut reply = parsed.tuple.reverse();
            let mut nat_kind = NatKind::None;

            if hook == HookPoint::PreRouting {
                if let Some((new_dst_ip, new_dst_port)) = self.nat.match_dnat(&parsed.tuple) {
                    reply.src_ip = new_dst_ip;
                    reply.src_port = new_dst_port;
                    nat_kind = NatKind::Dnat;
                }
            } else if hook == HookPoint::PostRouting {
                if let Some((new_src_ip, new_src_port)) =
                    self.nat.match_snat(&parsed.tuple, &self.conntrack)?
                {
                    reply.dst_ip = new_src_ip;
                    reply.dst_port = new_src_port;
                    nat_kind = NatKind::Snat;
                }
            }

            let idx =
                self.conntrack
                    .insert(parsed.tuple, reply, nat_kind, parsed.tcp_flags, now_tick)?;
            (idx, CtDirection::Original)
        };

        // 4. Apply NAT translation (forward or reverse)
        let entry = *self
            .conntrack
            .get(ct_idx)
            .ok_or(NetfilterError::TableFull)?;

        if entry.nat_kind != NatKind::None {
            match dir {
                CtDirection::Original => {
                    let expected_dst_ip = entry.reply_tuple.src_ip;
                    let expected_dst_port = entry.reply_tuple.src_port;
                    let expected_src_ip = entry.reply_tuple.dst_ip;
                    let expected_src_port = entry.reply_tuple.dst_port;

                    let new_sip =
                        (expected_src_ip != parsed.tuple.src_ip).then_some(expected_src_ip);
                    let new_dip =
                        (expected_dst_ip != parsed.tuple.dst_ip).then_some(expected_dst_ip);
                    let new_sp =
                        (expected_src_port != parsed.tuple.src_port).then_some(expected_src_port);
                    let new_dp =
                        (expected_dst_port != parsed.tuple.dst_port).then_some(expected_dst_port);

                    if new_sip.is_some()
                        || new_dip.is_some()
                        || new_sp.is_some()
                        || new_dp.is_some()
                    {
                        apply_nat_rewrite(frame, &parsed, new_sip, new_dip, new_sp, new_dp)?;
                        parsed = parse_ipv4_packet(frame, has_eth)?;
                    }
                }
                CtDirection::Reply => {
                    let orig_src_ip = entry.orig_tuple.src_ip;
                    let orig_src_port = entry.orig_tuple.src_port;
                    let orig_dst_ip = entry.orig_tuple.dst_ip;
                    let orig_dst_port = entry.orig_tuple.dst_port;

                    let new_sip = (orig_dst_ip != parsed.tuple.src_ip).then_some(orig_dst_ip);
                    let new_dip = (orig_src_ip != parsed.tuple.dst_ip).then_some(orig_src_ip);
                    let new_sp = (orig_dst_port != parsed.tuple.src_port).then_some(orig_dst_port);
                    let new_dp = (orig_src_port != parsed.tuple.dst_port).then_some(orig_src_port);

                    if new_sip.is_some()
                        || new_dip.is_some()
                        || new_sp.is_some()
                        || new_dp.is_some()
                    {
                        apply_nat_rewrite(frame, &parsed, new_sip, new_dip, new_sp, new_dp)?;
                        parsed = parse_ipv4_packet(frame, has_eth)?;
                    }
                }
            }
        }

        // 5. Update conntrack state & counters
        let ct_state = entry.state;
        self.conntrack
            .update_on_packet(ct_idx, dir, frame.len(), parsed.tcp_flags, now_tick);

        // 6. Evaluate firewall rules
        let verdict = self.firewall.evaluate(
            hook,
            ifindex,
            &parsed.tuple,
            ct_state,
            parsed.tcp_flags,
            frame.len(),
        )?;

        Ok(match verdict {
            Verdict::Accept | Verdict::Return => NetfilterDecision::Accept,
            Verdict::Drop => NetfilterDecision::Drop,
            Verdict::Reject => NetfilterDecision::Reject,
            Verdict::Jump(_) => NetfilterDecision::Drop,
        })
    }
}

pub static NETFILTER: SpinLock<Option<NetfilterEngine>> = SpinLock::new(None);

pub fn init() {
    let mut guard = NETFILTER.lock();
    if guard.is_none() {
        *guard = Some(NetfilterEngine::new());
    }
}

// ============================================================================
// 8. Self-Test Suite
// ============================================================================

fn build_test_ipv4_tcp(
    src_ip: Ipv4Addr,
    dst_ip: Ipv4Addr,
    src_port: u16,
    dst_port: u16,
    flags: u16,
) -> Vec<u8> {
    let mut pkt = vec![0u8; 40];
    // IPv4 header (20 bytes)
    pkt[0] = 0x45;
    pkt[2..4].copy_from_slice(&40u16.to_be_bytes());
    pkt[8] = 64;
    pkt[9] = PROTO_TCP;
    pkt[12..16].copy_from_slice(src_ip.as_bytes());
    pkt[16..20].copy_from_slice(dst_ip.as_bytes());
    let ip_csum = rfc1071_checksum(&pkt[0..20]);
    pkt[10..12].copy_from_slice(&ip_csum.to_be_bytes());

    // TCP header (20 bytes)
    pkt[20..22].copy_from_slice(&src_port.to_be_bytes());
    pkt[22..24].copy_from_slice(&dst_port.to_be_bytes());
    pkt[32] = 0x50; // data offset = 5 (20 bytes)
    pkt[33] = (flags & 0xFF) as u8;
    pkt[34..36].copy_from_slice(&65535u16.to_be_bytes());
    let tcp_csum = l4_checksum(src_ip, dst_ip, PROTO_TCP, &pkt[20..40]);
    pkt[36..38].copy_from_slice(&tcp_csum.to_be_bytes());
    pkt
}

fn build_test_ipv4_udp(
    src_ip: Ipv4Addr,
    dst_ip: Ipv4Addr,
    src_port: u16,
    dst_port: u16,
) -> Vec<u8> {
    let mut pkt = vec![0u8; 28];
    pkt[0] = 0x45;
    pkt[2..4].copy_from_slice(&28u16.to_be_bytes());
    pkt[8] = 64;
    pkt[9] = PROTO_UDP;
    pkt[12..16].copy_from_slice(src_ip.as_bytes());
    pkt[16..20].copy_from_slice(dst_ip.as_bytes());
    let ip_csum = rfc1071_checksum(&pkt[0..20]);
    pkt[10..12].copy_from_slice(&ip_csum.to_be_bytes());

    pkt[20..22].copy_from_slice(&src_port.to_be_bytes());
    pkt[22..24].copy_from_slice(&dst_port.to_be_bytes());
    pkt[24..26].copy_from_slice(&8u16.to_be_bytes());
    let udp_csum = l4_checksum(src_ip, dst_ip, PROTO_UDP, &pkt[20..28]);
    pkt[26..28].copy_from_slice(&udp_csum.to_be_bytes());
    pkt
}

pub fn self_test() -> Result<(), &'static str> {
    // 1. XDP Verifier negative tests
    if XdpVerifier::verify(&[]).is_ok() {
        return Err("verifier accepted empty program");
    }
    // Uninitialized R2 read
    let bad_uninit = [BpfInsn::mov64_reg(0, 2), BpfInsn::exit()];
    if XdpVerifier::verify(&bad_uninit) != Err(XdpVerifyError::UninitializedRegister) {
        return Err("verifier failed to catch uninitialized register");
    }
    // Backward jump / loop
    let bad_loop = [BpfInsn::mov64_imm(0, 2), BpfInsn::ja(-1), BpfInsn::exit()];
    if XdpVerifier::verify(&bad_loop) != Err(XdpVerifyError::LoopDetected) {
        return Err("verifier failed to catch loop");
    }
    // Write to R10
    let bad_r10 = [
        BpfInsn::mov64_imm(10, 0),
        BpfInsn::mov64_imm(0, 2),
        BpfInsn::exit(),
    ];
    if XdpVerifier::verify(&bad_r10) != Err(XdpVerifyError::ReadOnlyR10) {
        return Err("verifier failed to catch R10 write");
    }
    // Division by zero
    let bad_div0 = [
        BpfInsn::mov64_imm(0, 10),
        BpfInsn::alu64_imm(BPF_DIV, 0, 0),
        BpfInsn::exit(),
    ];
    if XdpVerifier::verify(&bad_div0) != Err(XdpVerifyError::DivisionByZero) {
        return Err("verifier failed to catch div by zero");
    }

    // 2. XDP Verifier + VM positive test: drop if IPv4 proto == UDP (17), else pass, and count in map
    let xdp_prog_insns = [
        // R2 = ctx->data, R3 = ctx->data_end
        BpfInsn::ldx_mem(BPF_W, 2, 1, 0),
        BpfInsn::ldx_mem(BPF_W, 3, 1, 4),
        // R4 = R2 + 20; if R4 > R3 goto pass
        BpfInsn::mov64_reg(4, 2),
        BpfInsn::alu64_imm(BPF_ADD, 4, 20),
        BpfInsn::jmp_reg(BPF_JGT, 4, 3, 4),
        // R5 = *(u8 *)(R2 + 9) (IPv4 protocol)
        BpfInsn::ldx_mem(BPF_B, 5, 2, 9),
        // if R5 == 17 (UDP) goto drop
        BpfInsn::jmp_imm(BPF_JEQ, 5, PROTO_UDP as i32, 2),
        // pass: R0 = 2; exit
        BpfInsn::mov64_imm(0, 2),
        BpfInsn::exit(),
        // drop: R0 = 1; exit
        BpfInsn::mov64_imm(0, 1),
        BpfInsn::exit(),
    ];
    let verified_xdp =
        XdpVerifier::verify(&xdp_prog_insns).map_err(|_| "valid XDP program failed verify")?;

    let mut engine = NetfilterEngine::new();
    engine.attach_xdp(1, verified_xdp);

    let lan_ip = Ipv4Addr::new(10, 0, 0, 2);
    let wan_ip = Ipv4Addr::new(203, 0, 113, 5);
    let srv_ip = Ipv4Addr::new(93, 184, 216, 34);

    // UDP packet on ifindex 1 should be dropped by XDP
    let mut udp_pkt = build_test_ipv4_udp(lan_ip, srv_ip, 5000, 53);
    let dec_udp = engine
        .process_packet(1, HookPoint::PreRouting, &mut udp_pkt, false, 10)
        .map_err(|_| "XDP UDP process failed")?;
    if dec_udp != NetfilterDecision::Drop {
        return Err("XDP did not drop UDP packet");
    }

    // 3. Stateful Firewall + Conntrack + SNAT test
    // Configure Input policy = Drop, allow ESTABLISHED/RELATED, allow outbound LAN->WAN
    engine
        .firewall
        .set_hook_policy(HookPoint::Input, Verdict::Drop);
    let mut allow_est = FirewallRule::new(1, Verdict::Accept);
    allow_est.ct_states = CtStateMask(CtStateMask::ESTABLISHED | CtStateMask::RELATED);
    engine.firewall.add_hook_rule(HookPoint::Input, allow_est);

    // SNAT rule: 10.0.0.0/24 -> 203.0.113.5
    engine.nat.add_rule(NatRule {
        id: 1,
        src_cidr: IpCidr::new(Ipv4Addr::new(10, 0, 0, 0), 24),
        dst_cidr: IpCidr::ANY,
        proto: Some(PROTO_TCP),
        dst_port: PortRange::ANY,
        action: NatAction::Snat {
            to_ip: wan_ip,
            port_min: 30000,
            port_max: 30100,
        },
        hits: 0,
    });

    // Outbound SYN on PostRouting (SNAT)
    let mut syn_pkt = build_test_ipv4_tcp(lan_ip, srv_ip, 12345, 443, TcpHeader::FLAG_SYN);
    let dec_syn = engine
        .process_packet(2, HookPoint::PostRouting, &mut syn_pkt, false, 100)
        .map_err(|_| "SNAT SYN failed")?;
    if dec_syn != NetfilterDecision::Accept {
        return Err("SNAT SYN not accepted");
    }
    let parsed_snat = parse_ipv4_packet(&syn_pkt, false).map_err(|_| "parse SNAT failed")?;
    if parsed_snat.tuple.src_ip != wan_ip {
        return Err("SNAT did not rewrite src_ip");
    }
    if rfc1071_checksum(&syn_pkt[0..20]) != 0 {
        return Err("SNAT IPv4 checksum invalid");
    }
    if l4_checksum(wan_ip, srv_ip, PROTO_TCP, &syn_pkt[20..40]) != 0 {
        return Err("SNAT TCP checksum invalid");
    }

    // Reply SYN-ACK on Input (reverse SNAT + ESTABLISHED)
    let mut synack_pkt = build_test_ipv4_tcp(
        srv_ip,
        wan_ip,
        443,
        parsed_snat.tuple.src_port,
        TcpHeader::FLAG_SYN | TcpHeader::FLAG_ACK,
    );
    let dec_reply = engine
        .process_packet(2, HookPoint::Input, &mut synack_pkt, false, 101)
        .map_err(|_| "reply SYN-ACK failed")?;
    if dec_reply != NetfilterDecision::Accept {
        return Err("reply SYN-ACK not accepted");
    }
    let parsed_rev = parse_ipv4_packet(&synack_pkt, false).map_err(|_| "parse rev failed")?;
    if parsed_rev.tuple.dst_ip != lan_ip || parsed_rev.tuple.dst_port != 12345 {
        return Err("reverse SNAT did not restore original dst");
    }
    if rfc1071_checksum(&synack_pkt[0..20]) != 0 {
        return Err("reverse SNAT IPv4 checksum invalid");
    }
    if l4_checksum(srv_ip, lan_ip, PROTO_TCP, &synack_pkt[20..40]) != 0 {
        return Err("reverse SNAT TCP checksum invalid");
    }

    // Unsolicited inbound SYN on Input should be dropped by default policy
    let mut unsol_pkt = build_test_ipv4_tcp(srv_ip, wan_ip, 9999, 22, TcpHeader::FLAG_SYN);
    let dec_unsol = engine
        .process_packet(2, HookPoint::Input, &mut unsol_pkt, false, 102)
        .map_err(|_| "unsolicited SYN failed")?;
    if dec_unsol != NetfilterDecision::Drop {
        return Err("unsolicited inbound SYN was not dropped");
    }

    // 4. Conntrack GC test
    let evicted = engine.conntrack.gc_expired(1000);
    if evicted == 0 {
        return Err("conntrack GC did not evict expired entries");
    }

    Ok(())
}
