#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SocketAddrV4 {
    pub ip: [u8; 4],
    pub port: u16,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SocketKind {
    Tcp,
    Udp,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SocketState {
    Created,
    Bound,
    Connected,
    Closed,
}

struct SocketEntry {
    kind: SocketKind,
    state: SocketState,
    local: Option<SocketAddrV4>,
    remote: Option<SocketAddrV4>,
    rx_queue: VecDeque<u8>,
}

#[derive(Default)]
pub struct SocketTable {
    next_fd: u32,
    sockets: BTreeMap<u32, SocketEntry>,
}

impl SocketTable {
    pub const fn new() -> Self {
        Self {
            next_fd: 1,
            sockets: BTreeMap::new(),
        }
    }

    pub fn socket(&mut self, kind: SocketKind) -> u32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        self.sockets.insert(
            fd,
            SocketEntry {
                kind,
                state: SocketState::Created,
                local: None,
                remote: None,
                rx_queue: VecDeque::new(),
            },
        );
        fd
    }

    pub fn bind(&mut self, fd: u32, addr: SocketAddrV4) -> Result<(), &'static str> {
        let s = self.sockets.get_mut(&fd).ok_or("bad fd")?;
        s.local = Some(addr);
        s.state = SocketState::Bound;
        Ok(())
    }

    pub fn connect(&mut self, fd: u32, addr: SocketAddrV4) -> Result<(), &'static str> {
        let s = self.sockets.get_mut(&fd).ok_or("bad fd")?;
        s.remote = Some(addr);
        s.state = SocketState::Connected;
        Ok(())
    }

    pub fn send(&mut self, fd: u32, data: &[u8]) -> Result<usize, &'static str> {
        let s = self.sockets.get_mut(&fd).ok_or("bad fd")?;
        if s.state != SocketState::Connected && s.kind == SocketKind::Tcp {
            return Err("not connected");
        }
        s.rx_queue.extend(data.iter().copied());
        Ok(data.len())
    }

    pub fn recv(&mut self, fd: u32, out: &mut [u8]) -> Result<usize, &'static str> {
        let s = self.sockets.get_mut(&fd).ok_or("bad fd")?;
        let mut n = 0;
        while n < out.len() {
            if let Some(b) = s.rx_queue.pop_front() {
                out[n] = b;
                n += 1;
            } else {
                break;
            }
        }
        Ok(n)
    }

    pub fn close(&mut self, fd: u32) -> Result<(), &'static str> {
        let s = self.sockets.get_mut(&fd).ok_or("bad fd")?;
        s.state = SocketState::Closed;
        Ok(())
    }
}

pub struct DnsQuery;

impl DnsQuery {
    pub fn encode_a_record(domain: &str, tx_id: u16) -> Vec<u8> {
        let mut pkt = Vec::with_capacity(32 + domain.len());
        pkt.extend_from_slice(&tx_id.to_be_bytes());
        pkt.extend_from_slice(&0x0100u16.to_be_bytes()); // standard query, RD=1
        pkt.extend_from_slice(&1u16.to_be_bytes()); // QDCOUNT=1
        pkt.extend_from_slice(&[0u8; 6]); // AN/NS/AR = 0
        for label in domain.split('.') {
            if !label.is_empty() && label.len() <= 63 {
                pkt.push(label.len() as u8);
                pkt.extend_from_slice(label.as_bytes());
            }
        }
        pkt.push(0);
        pkt.extend_from_slice(&1u16.to_be_bytes()); // QTYPE=A
        pkt.extend_from_slice(&1u16.to_be_bytes()); // QCLASS=IN
        pkt
    }
}

pub struct DhcpDiscover;

impl DhcpDiscover {
    pub fn build(mac: [u8; 6]) -> Vec<u8> {
        let mut pkt = alloc::vec![0u8; 244];
        pkt[0] = 1; // BOOTREQUEST
        pkt[1] = 1; // Ethernet
        pkt[2] = 6; // HLEN
        pkt[28..34].copy_from_slice(&mac);
        pkt[236..240].copy_from_slice(&[99, 130, 83, 99]); // DHCP magic cookie
        pkt[240..243].copy_from_slice(&[53, 1, 1]); // Option 53: DHCPDISCOVER
        pkt[243] = 255; // End
        pkt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn socket_dns_and_dhcp() {
        let mut tbl = SocketTable::new();
        let fd = tbl.socket(SocketKind::Tcp);
        let addr = SocketAddrV4 {
            ip: [127, 0, 0, 1],
            port: 8080,
        };
        tbl.bind(fd, addr).unwrap();
        tbl.connect(fd, addr).unwrap();
        assert_eq!(tbl.send(fd, b"GET /").unwrap(), 5);
        let mut buf = [0u8; 8];
        assert_eq!(tbl.recv(fd, &mut buf).unwrap(), 5);
        assert_eq!(&buf[..5], b"GET /");
        tbl.close(fd).unwrap();

        let dns = DnsQuery::encode_a_record("nexus.os", 0x1234);
        assert_eq!(&dns[0..2], &[0x12, 0x34]);
        let dhcp = DhcpDiscover::build([0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
        assert_eq!(&dhcp[236..240], &[99, 130, 83, 99]);
    }
}
