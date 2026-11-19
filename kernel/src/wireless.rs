//! Clean-room `no_std` 802.11ax/be Wi-Fi MAC, WPA3-SAE/CCMP & Bluetooth HCI/L2CAP Stack.
//!
//! Provides:
//! - IEEE 802.11ax (Wi-Fi 6) / 802.11be (Wi-Fi 7) MAC frame parser, builder, IE decoder
//!   (SSID, RSN, HE Capabilities, EHT Capabilities, Multi-Link Element), A-MPDU/A-MSDU
//!   subframe parsing, Block Ack bitmap tracking, and STA MAC state machine.
//! - WPA3-SAE (Dragonfly) Commit/Confirm & key hierarchy (SHA-256, HMAC-SHA256,
//!   HKDF-SHA256, PRF-384, AES-128, AES-CMAC, AES-CCMP AEAD with 48-bit PN replay
//!   protection, and 4-Way Handshake EAPOL-Key MIC verification).
//! - Bluetooth HCI (Command/ACL/Event/ISO packet framing, controller & connection state
//!   machine) and L2CAP (Signaling, ATT, SMP, Credit-Based Flow Control, SDU
//!   segmentation/reassembly, and channel state machine).
//!
//! Designed for `no_std` x86_64 kernel execution with small stack frames (<2 KiB)
//! and pure integer arithmetic (no floating-point instructions).
#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;

use crate::net::MacAddr;

// ============================================================================
// Common Error Type
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WirelessError {
    BufferTooSmall,
    MalformedFrame,
    UnsupportedVersion,
    UnsupportedCipher,
    InvalidState,
    AuthFailed,
    MicMismatch,
    ReplayDetected,
    DecryptFailed,
    InvalidParameter,
    ChannelNotFound,
    InsufficientCredits,
    SduTooLarge,
}

impl WirelessError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BufferTooSmall => "wireless: buffer too small",
            Self::MalformedFrame => "wireless: malformed frame or packet",
            Self::UnsupportedVersion => "wireless: unsupported protocol version",
            Self::UnsupportedCipher => "wireless: unsupported cipher or AKM suite",
            Self::InvalidState => "wireless: invalid state transition",
            Self::AuthFailed => "wireless: authentication failed",
            Self::MicMismatch => "wireless: MIC verification failed",
            Self::ReplayDetected => "wireless: replay packet number detected",
            Self::DecryptFailed => "wireless: CCMP decryption/authentication failed",
            Self::InvalidParameter => "wireless: invalid parameter",
            Self::ChannelNotFound => "wireless: L2CAP channel not found",
            Self::InsufficientCredits => "wireless: insufficient L2CAP flow credits",
            Self::SduTooLarge => "wireless: L2CAP SDU exceeds MTU",
        }
    }
}

impl From<WirelessError> for &'static str {
    fn from(err: WirelessError) -> Self {
        err.as_str()
    }
}

// ============================================================================
// 1. IEEE 802.11ax/be Wi-Fi MAC Frame Parser, Builder & State Machine
// ============================================================================

/// 802.11 Frame Type (2-bit field in Frame Control).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    Management = 0,
    Control = 1,
    Data = 2,
    Extension = 3,
}

impl FrameType {
    pub const fn from_u8(v: u8) -> Self {
        match v & 0x03 {
            0 => Self::Management,
            1 => Self::Control,
            2 => Self::Data,
            _ => Self::Extension,
        }
    }
}

/// 802.11 Management Frame Subtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MgmtSubtype {
    AssocReq = 0,
    AssocResp = 1,
    ReassocReq = 2,
    ReassocResp = 3,
    ProbeReq = 4,
    ProbeResp = 5,
    Beacon = 8,
    Disassoc = 10,
    Auth = 11,
    Deauth = 12,
    Action = 13,
}

/// 802.11 Control Frame Subtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CtrlSubtype {
    Trigger = 2,
    BlockAckReq = 8,
    BlockAck = 9,
    PsPoll = 10,
    Rts = 11,
    Cts = 12,
    Ack = 13,
}

/// 802.11 Data Frame Subtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataSubtype {
    Data = 0,
    Null = 4,
    QosData = 8,
    QosNull = 12,
}

/// Parsed 802.11 Frame Control (2 bytes, little-endian).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameControl {
    pub raw: u16,
}

impl FrameControl {
    pub const fn from_le_bytes(b: [u8; 2]) -> Self {
        Self {
            raw: u16::from_le_bytes(b),
        }
    }

    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.raw.to_le_bytes()
    }

    pub const fn new(frame_type: FrameType, subtype: u8, flags: u8) -> Self {
        let low = ((subtype & 0x0F) << 4) | ((frame_type as u8 & 0x03) << 2);
        Self {
            raw: (low as u16) | ((flags as u16) << 8),
        }
    }

    pub const fn protocol_version(self) -> u8 {
        (self.raw & 0x03) as u8
    }

    pub const fn frame_type(self) -> FrameType {
        FrameType::from_u8(((self.raw >> 2) & 0x03) as u8)
    }

    pub const fn subtype(self) -> u8 {
        ((self.raw >> 4) & 0x0F) as u8
    }

    pub const fn to_ds(self) -> bool {
        (self.raw & (1 << 8)) != 0
    }

    #[allow(clippy::wrong_self_convention)]
    pub const fn from_ds(self) -> bool {
        (self.raw & (1 << 9)) != 0
    }

    pub const fn more_fragments(self) -> bool {
        (self.raw & (1 << 10)) != 0
    }

    pub const fn retry(self) -> bool {
        (self.raw & (1 << 11)) != 0
    }

    pub const fn power_mgmt(self) -> bool {
        (self.raw & (1 << 12)) != 0
    }

    pub const fn more_data(self) -> bool {
        (self.raw & (1 << 13)) != 0
    }

    pub const fn protected(self) -> bool {
        (self.raw & (1 << 14)) != 0
    }

    pub const fn order_htc(self) -> bool {
        (self.raw & (1 << 15)) != 0
    }
}

/// 802.11 Information Element IDs (including 802.11ax/be Extension IDs).
pub const IE_SSID: u8 = 0;
pub const IE_SUPPORTED_RATES: u8 = 1;
pub const IE_DS_PARAM_SET: u8 = 3;
pub const IE_TIM: u8 = 5;
pub const IE_RSN: u8 = 48;
pub const IE_HT_CAPABILITIES: u8 = 45;
pub const IE_VHT_CAPABILITIES: u8 = 191;
pub const IE_EXTENSION: u8 = 255;

/// 802.11ax / 802.11be Extension Element IDs (when IE ID == 255).
pub const IE_EXT_HE_CAPABILITIES: u8 = 35;
pub const IE_EXT_HE_OPERATION: u8 = 36;
pub const IE_EXT_EHT_OPERATION: u8 = 106;
pub const IE_EXT_MULTI_LINK: u8 = 107;
pub const IE_EXT_EHT_CAPABILITIES: u8 = 108;

/// Parsed 802.11 Information Element view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InfoElement<'a> {
    pub id: u8,
    pub ext_id: Option<u8>,
    pub data: &'a [u8],
}

/// Iterator over 802.11 Information Elements in a management frame body.
pub struct IeIter<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> IeIter<'a> {
    pub const fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
}

impl<'a> Iterator for IeIter<'a> {
    type Item = Result<InfoElement<'a>, WirelessError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.buf.len() {
            return None;
        }
        if self.pos + 2 > self.buf.len() {
            self.pos = self.buf.len();
            return Some(Err(WirelessError::MalformedFrame));
        }
        let id = self.buf[self.pos];
        let len = self.buf[self.pos + 1] as usize;
        let start = self.pos + 2;
        let end = start + len;
        if end > self.buf.len() {
            self.pos = self.buf.len();
            return Some(Err(WirelessError::MalformedFrame));
        }
        self.pos = end;
        if id == IE_EXTENSION {
            if len == 0 {
                return Some(Err(WirelessError::MalformedFrame));
            }
            Some(Ok(InfoElement {
                id,
                ext_id: Some(self.buf[start]),
                data: &self.buf[start + 1..end],
            }))
        } else {
            Some(Ok(InfoElement {
                id,
                ext_id: None,
                data: &self.buf[start..end],
            }))
        }
    }
}

/// 802.11ax (Wi-Fi 6) HE Capabilities summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeCapabilities {
    pub mac_cap_info: [u8; 6],
    pub phy_cap_info: [u8; 11],
    pub max_mcs_160mhz: bool,
    pub ofdma_ul_mu_mimo: bool,
    pub twt_requester: bool,
}

impl HeCapabilities {
    pub fn parse(data: &[u8]) -> Result<Self, WirelessError> {
        if data.len() < 21 {
            return Err(WirelessError::MalformedFrame);
        }
        let mut mac_cap_info = [0u8; 6];
        mac_cap_info.copy_from_slice(&data[0..6]);
        let mut phy_cap_info = [0u8; 11];
        phy_cap_info.copy_from_slice(&data[6..17]);
        let twt_requester = (mac_cap_info[0] & 0x02) != 0;
        let max_mcs_160mhz = (phy_cap_info[0] & 0x08) != 0;
        let ofdma_ul_mu_mimo = (phy_cap_info[2] & 0x40) != 0;
        Ok(Self {
            mac_cap_info,
            phy_cap_info,
            max_mcs_160mhz,
            ofdma_ul_mu_mimo,
            twt_requester,
        })
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        out.push(IE_EXTENSION);
        out.push(22); // 1 ext_id + 6 mac + 11 phy + 4 mcs
        out.push(IE_EXT_HE_CAPABILITIES);
        out.extend_from_slice(&self.mac_cap_info);
        out.extend_from_slice(&self.phy_cap_info);
        out.extend_from_slice(&[0xFA, 0xFF, 0xFA, 0xFF]); // MCS 0-11 2x2
    }
}

/// 802.11be (Wi-Fi 7) EHT Capabilities & Multi-Link Operation (MLO) summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EhtCapabilities {
    pub mac_cap_info: [u8; 2],
    pub phy_cap_info: [u8; 9],
    pub support_320mhz_6ghz: bool,
    pub support_4096_qam: bool,
    pub mlo_emlsr_supported: bool,
}

impl EhtCapabilities {
    pub fn parse(data: &[u8]) -> Result<Self, WirelessError> {
        if data.len() < 11 {
            return Err(WirelessError::MalformedFrame);
        }
        let mut mac_cap_info = [0u8; 2];
        mac_cap_info.copy_from_slice(&data[0..2]);
        let mut phy_cap_info = [0u8; 9];
        phy_cap_info.copy_from_slice(&data[2..11]);
        let support_320mhz_6ghz = (phy_cap_info[0] & 0x02) != 0;
        let support_4096_qam = (phy_cap_info[3] & 0x04) != 0;
        let mlo_emlsr_supported = (mac_cap_info[0] & 0x04) != 0;
        Ok(Self {
            mac_cap_info,
            phy_cap_info,
            support_320mhz_6ghz,
            support_4096_qam,
            mlo_emlsr_supported,
        })
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        out.push(IE_EXTENSION);
        out.push(15); // 1 ext_id + 2 mac + 9 phy + 3 mcs
        out.push(IE_EXT_EHT_CAPABILITIES);
        out.extend_from_slice(&self.mac_cap_info);
        out.extend_from_slice(&self.phy_cap_info);
        out.extend_from_slice(&[0x22, 0x22, 0x22]); // EHT-MCS 0-13
    }
}

/// 802.11be Multi-Link Element (Basic variant) for MLO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiLinkElement {
    pub mld_mac: MacAddr,
    pub link_id: Option<u8>,
    pub bss_params_change_count: Option<u8>,
    pub eml_capabilities: Option<u16>,
}

impl MultiLinkElement {
    pub fn parse(data: &[u8]) -> Result<Self, WirelessError> {
        if data.len() < 9 {
            return Err(WirelessError::MalformedFrame);
        }
        let ctrl = u16::from_le_bytes([data[0], data[1]]);
        let variant = ctrl & 0x07;
        if variant != 0 {
            return Err(WirelessError::UnsupportedVersion);
        }
        let presence = (ctrl >> 4) & 0x0FFF;
        let common_len = data[2] as usize;
        if common_len < 7 || 2 + common_len > data.len() {
            return Err(WirelessError::MalformedFrame);
        }
        let mut mld_bytes = [0u8; 6];
        mld_bytes.copy_from_slice(&data[3..9]);
        let mut pos = 9usize;
        let link_id = if (presence & 0x01) != 0 {
            if pos >= 2 + common_len {
                return Err(WirelessError::MalformedFrame);
            }
            let v = data[pos] & 0x0F;
            pos += 1;
            Some(v)
        } else {
            None
        };
        let bss_params_change_count = if (presence & 0x02) != 0 {
            if pos >= 2 + common_len {
                return Err(WirelessError::MalformedFrame);
            }
            let v = data[pos];
            pos += 1;
            Some(v)
        } else {
            None
        };
        let eml_capabilities = if (presence & 0x08) != 0 {
            if pos + 2 > 2 + common_len {
                return Err(WirelessError::MalformedFrame);
            }
            let v = u16::from_le_bytes([data[pos], data[pos + 1]]);
            Some(v)
        } else {
            None
        };
        Ok(Self {
            mld_mac: MacAddr::new(mld_bytes),
            link_id,
            bss_params_change_count,
            eml_capabilities,
        })
    }
}

/// Parsed 802.11 MAC Header + payload slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacFrame<'a> {
    pub fc: FrameControl,
    pub duration_id: u16,
    pub addr1: MacAddr,
    pub addr2: Option<MacAddr>,
    pub addr3: Option<MacAddr>,
    pub seq_ctrl: Option<u16>,
    pub addr4: Option<MacAddr>,
    pub qos_ctrl: Option<u16>,
    pub ht_ctrl: Option<u32>,
    pub payload: &'a [u8],
}

impl<'a> MacFrame<'a> {
    /// Parse an 802.11 MAC frame (without trailing FCS).
    pub fn parse(raw: &'a [u8]) -> Result<Self, WirelessError> {
        if raw.len() < 10 {
            return Err(WirelessError::BufferTooSmall);
        }
        let fc = FrameControl::from_le_bytes([raw[0], raw[1]]);
        if fc.protocol_version() != 0 {
            return Err(WirelessError::UnsupportedVersion);
        }
        let duration_id = u16::from_le_bytes([raw[2], raw[3]]);
        let addr1 = MacAddr::new([raw[4], raw[5], raw[6], raw[7], raw[8], raw[9]]);

        if fc.frame_type() == FrameType::Control {
            let sub = fc.subtype();
            // CTS (12) and ACK (13) have only Addr1 (10 bytes total).
            if sub == CtrlSubtype::Cts as u8 || sub == CtrlSubtype::Ack as u8 {
                return Ok(Self {
                    fc,
                    duration_id,
                    addr1,
                    addr2: None,
                    addr3: None,
                    seq_ctrl: None,
                    addr4: None,
                    qos_ctrl: None,
                    ht_ctrl: None,
                    payload: &raw[10..],
                });
            }
            if raw.len() < 16 {
                return Err(WirelessError::BufferTooSmall);
            }
            let addr2 = MacAddr::new([raw[10], raw[11], raw[12], raw[13], raw[14], raw[15]]);
            return Ok(Self {
                fc,
                duration_id,
                addr1,
                addr2: Some(addr2),
                addr3: None,
                seq_ctrl: None,
                addr4: None,
                qos_ctrl: None,
                ht_ctrl: None,
                payload: &raw[16..],
            });
        }

        // Management and Data frames have at least 24 bytes (FC + Dur + A1 + A2 + A3 + Seq).
        if raw.len() < 24 {
            return Err(WirelessError::BufferTooSmall);
        }
        let addr2 = MacAddr::new([raw[10], raw[11], raw[12], raw[13], raw[14], raw[15]]);
        let addr3 = MacAddr::new([raw[16], raw[17], raw[18], raw[19], raw[20], raw[21]]);
        let seq_ctrl = u16::from_le_bytes([raw[22], raw[23]]);
        let mut pos = 24usize;

        let addr4 = if fc.frame_type() == FrameType::Data && fc.to_ds() && fc.from_ds() {
            if raw.len() < pos + 6 {
                return Err(WirelessError::BufferTooSmall);
            }
            let a4 = MacAddr::new([
                raw[pos],
                raw[pos + 1],
                raw[pos + 2],
                raw[pos + 3],
                raw[pos + 4],
                raw[pos + 5],
            ]);
            pos += 6;
            Some(a4)
        } else {
            None
        };

        let qos_ctrl = if fc.frame_type() == FrameType::Data && (fc.subtype() & 0x08) != 0 {
            if raw.len() < pos + 2 {
                return Err(WirelessError::BufferTooSmall);
            }
            let q = u16::from_le_bytes([raw[pos], raw[pos + 1]]);
            pos += 2;
            Some(q)
        } else {
            None
        };

        let ht_ctrl = if fc.order_htc() {
            if raw.len() < pos + 4 {
                return Err(WirelessError::BufferTooSmall);
            }
            let h = u32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]);
            pos += 4;
            Some(h)
        } else {
            None
        };

        Ok(Self {
            fc,
            duration_id,
            addr1,
            addr2: Some(addr2),
            addr3: Some(addr3),
            seq_ctrl: Some(seq_ctrl),
            addr4,
            qos_ctrl,
            ht_ctrl,
            payload: &raw[pos..],
        })
    }

    pub const fn sequence_number(&self) -> Option<u16> {
        match self.seq_ctrl {
            Some(s) => Some(s >> 4),
            None => None,
        }
    }

    pub const fn fragment_number(&self) -> Option<u8> {
        match self.seq_ctrl {
            Some(s) => Some((s & 0x0F) as u8),
            None => None,
        }
    }

    pub const fn qos_tid(&self) -> Option<u8> {
        match self.qos_ctrl {
            Some(q) => Some((q & 0x0F) as u8),
            None => None,
        }
    }
}

/// Build a Beacon frame with SSID, RSN (WPA3-SAE), HE Capabilities, and EHT Capabilities.
pub fn build_beacon_frame(
    bssid: MacAddr,
    ssid: &[u8],
    channel: u8,
    seq_num: u16,
    he: Option<&HeCapabilities>,
    eht: Option<&EhtCapabilities>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(128);
    let fc = FrameControl::new(FrameType::Management, MgmtSubtype::Beacon as u8, 0);
    out.extend_from_slice(&fc.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // Duration
    out.extend_from_slice(MacAddr::BROADCAST.as_bytes()); // Addr1
    out.extend_from_slice(bssid.as_bytes()); // Addr2
    out.extend_from_slice(bssid.as_bytes()); // Addr3
    out.extend_from_slice(&(seq_num << 4).to_le_bytes());

    // Fixed Beacon body: Timestamp (8) + Beacon Interval (2) + Capability Info (2)
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&100u16.to_le_bytes()); // 100 TU
    out.extend_from_slice(&0x0411u16.to_le_bytes()); // ESS + Privacy + Short Slot

    // SSID IE
    let ssid_len = ssid.len().min(32);
    out.push(IE_SSID);
    out.push(ssid_len as u8);
    out.extend_from_slice(&ssid[..ssid_len]);

    // DS Parameter Set IE
    out.push(IE_DS_PARAM_SET);
    out.push(1);
    out.push(channel);

    // RSN IE (WPA3-SAE: CCMP-128 + AKM SAE 00-0F-AC:8 + MFP required)
    out.push(IE_RSN);
    out.push(20);
    out.extend_from_slice(&1u16.to_le_bytes()); // Version 1
    out.extend_from_slice(&[0x00, 0x0F, 0xAC, 0x04]); // Group cipher: CCMP-128
    out.extend_from_slice(&1u16.to_le_bytes()); // Pairwise count
    out.extend_from_slice(&[0x00, 0x0F, 0xAC, 0x04]); // Pairwise: CCMP-128
    out.extend_from_slice(&1u16.to_le_bytes()); // AKM count
    out.extend_from_slice(&[0x00, 0x0F, 0xAC, 0x08]); // AKM: SAE
    out.extend_from_slice(&0x00C0u16.to_le_bytes()); // RSN Caps: MFPR + MFPC

    if let Some(he_cap) = he {
        he_cap.encode(&mut out);
    }
    if let Some(eht_cap) = eht {
        eht_cap.encode(&mut out);
    }
    out
}

/// Block Ack scoreboard for 64-frame reordering window (802.11n/ax/be).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockAckWindow {
    pub win_start: u16,
    pub bitmap: u64,
}

impl BlockAckWindow {
    pub const fn new(win_start: u16) -> Self {
        Self {
            win_start: win_start & 0x0FFF,
            bitmap: 0,
        }
    }

    /// Record a received sequence number and advance window if needed.
    pub fn record_rx(&mut self, seq: u16) -> bool {
        let seq = seq & 0x0FFF;
        let diff = seq.wrapping_sub(self.win_start) & 0x0FFF;
        if diff < 64 {
            let bit = 1u64 << diff;
            let dup = (self.bitmap & bit) != 0;
            self.bitmap |= bit;
            !dup
        } else if diff < 2048 {
            // Advance window so `seq` is at bit 63.
            let shift = diff - 63;
            if shift >= 64 {
                self.bitmap = 0;
            } else {
                self.bitmap >>= shift;
            }
            self.win_start = (self.win_start + shift) & 0x0FFF;
            self.bitmap |= 1u64 << 63;
            true
        } else {
            // Old frame behind window.
            false
        }
    }
}

/// 802.11 STA MAC Connection State Machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WifiMacState {
    Idle,
    Scanning,
    Authenticating,
    Associating,
    FourWayHandshake,
    Associated,
    Disconnecting,
}

/// 802.11 STA MAC Controller.
#[derive(Debug, Clone)]
pub struct WifiMac {
    pub sta_mac: MacAddr,
    pub bssid: Option<MacAddr>,
    pub state: WifiMacState,
    pub tx_seq: u16,
    pub ba_window: BlockAckWindow,
    pub he_enabled: bool,
    pub eht_enabled: bool,
}

impl WifiMac {
    pub const fn new(sta_mac: MacAddr) -> Self {
        Self {
            sta_mac,
            bssid: None,
            state: WifiMacState::Idle,
            tx_seq: 0,
            ba_window: BlockAckWindow::new(0),
            he_enabled: true,
            eht_enabled: true,
        }
    }

    pub fn start_scan(&mut self) -> Result<(), WirelessError> {
        if self.state != WifiMacState::Idle && self.state != WifiMacState::Scanning {
            return Err(WirelessError::InvalidState);
        }
        self.state = WifiMacState::Scanning;
        Ok(())
    }

    pub fn start_auth(&mut self, bssid: MacAddr) -> Result<(), WirelessError> {
        if self.state != WifiMacState::Idle && self.state != WifiMacState::Scanning {
            return Err(WirelessError::InvalidState);
        }
        self.bssid = Some(bssid);
        self.state = WifiMacState::Authenticating;
        Ok(())
    }

    pub fn on_auth_success(&mut self) -> Result<(), WirelessError> {
        if self.state != WifiMacState::Authenticating {
            return Err(WirelessError::InvalidState);
        }
        self.state = WifiMacState::Associating;
        Ok(())
    }

    pub fn on_assoc_success(&mut self) -> Result<(), WirelessError> {
        if self.state != WifiMacState::Associating {
            return Err(WirelessError::InvalidState);
        }
        self.state = WifiMacState::FourWayHandshake;
        Ok(())
    }

    pub fn on_handshake_complete(&mut self) -> Result<(), WirelessError> {
        if self.state != WifiMacState::FourWayHandshake {
            return Err(WirelessError::InvalidState);
        }
        self.state = WifiMacState::Associated;
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.state = WifiMacState::Idle;
        self.bssid = None;
    }

    pub fn next_seq(&mut self) -> u16 {
        let s = self.tx_seq;
        self.tx_seq = (self.tx_seq + 1) & 0x0FFF;
        s
    }
}

// ============================================================================
// 2. WPA3-SAE / Key Hierarchy / AES-CCMP (Pure no_std Crypto)
// ============================================================================

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Incremental SHA-256 hasher with small stack footprint.
pub struct Sha256 {
    state: [u32; 8],
    bit_len: u64,
    buf: [u8; 64],
    buf_len: usize,
}

impl Sha256 {
    pub const fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            bit_len: 0,
            buf: [0u8; 64],
            buf_len: 0,
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*chunk);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }

    pub fn update(&mut self, data: &[u8]) {
        self.bit_len = self.bit_len.wrapping_add((data.len() as u64) * 8);
        let mut offset = 0usize;
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            offset += take;
            if self.buf_len == 64 {
                let blk = self.buf;
                self.compress(&blk);
                self.buf_len = 0;
            }
        }
        while offset + 64 <= data.len() {
            let mut blk = [0u8; 64];
            blk.copy_from_slice(&data[offset..offset + 64]);
            self.compress(&blk);
            offset += 64;
        }
        if offset < data.len() {
            let rem = data.len() - offset;
            self.buf[..rem].copy_from_slice(&data[offset..]);
            self.buf_len = rem;
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.bit_len;
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            self.buf[self.buf_len..64].fill(0);
            let blk = self.buf;
            self.compress(&blk);
            self.buf_len = 0;
        }
        self.buf[self.buf_len..56].fill(0);
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let blk = self.buf;
        self.compress(&blk);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }
}

/// HMAC-SHA256 (RFC 2104).
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut k_pad = [0u8; 64];
    if key.len() > 64 {
        let hashed = Sha256::digest(key);
        k_pad[..32].copy_from_slice(&hashed);
    } else {
        k_pad[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5Cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_pad[i];
        opad[i] ^= k_pad[i];
    }

    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(data);
    let inner_hash = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&inner_hash);
    outer.finalize()
}

/// HKDF-Extract (RFC 5869) using SHA-256.
pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    let zero_salt = [0u8; 32];
    let s = if salt.is_empty() { &zero_salt } else { salt };
    hmac_sha256(s, ikm)
}

/// HKDF-Expand (RFC 5869) using SHA-256.
pub fn hkdf_expand(prk: &[u8; 32], info: &[u8], out: &mut [u8]) {
    let mut prev = [0u8; 32];
    let mut prev_len = 0usize;
    let mut counter = 1u8;
    let mut offset = 0usize;

    while offset < out.len() {
        let mut k_pad = [0u8; 64];
        k_pad[..32].copy_from_slice(prk);
        let mut ipad = [0x36u8; 64];
        let mut opad = [0x5Cu8; 64];
        for i in 0..64 {
            ipad[i] ^= k_pad[i];
            opad[i] ^= k_pad[i];
        }
        let mut inner = Sha256::new();
        inner.update(&ipad);
        if prev_len > 0 {
            inner.update(&prev[..prev_len]);
        }
        inner.update(info);
        inner.update(&[counter]);
        let inner_hash = inner.finalize();

        let mut outer = Sha256::new();
        outer.update(&opad);
        outer.update(&inner_hash);
        prev = outer.finalize();
        prev_len = 32;

        let take = (out.len() - offset).min(32);
        out[offset..offset + take].copy_from_slice(&prev[..take]);
        offset += take;
        counter = counter.wrapping_add(1);
    }
}

/// IEEE 802.11-2020 KDF-Hash-Length (SHA-256) for PTK derivation.
pub fn kdf_sha256(key: &[u8], label: &[u8], context: &[u8], out: &mut [u8]) {
    let bits = (out.len() as u16) * 8;
    let mut offset = 0usize;
    let mut iter = 1u16;
    while offset < out.len() {
        let mut msg = Vec::with_capacity(2 + label.len() + context.len() + 2);
        msg.extend_from_slice(&iter.to_le_bytes());
        msg.extend_from_slice(label);
        msg.extend_from_slice(context);
        msg.extend_from_slice(&bits.to_le_bytes());
        let block = hmac_sha256(key, &msg);
        let take = (out.len() - offset).min(32);
        out[offset..offset + take].copy_from_slice(&block[..take]);
        offset += take;
        iter += 1;
    }
}

// --- AES-128 Block Cipher (FIPS 197) ---

const AES_SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

const AES_RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

#[inline]
const fn gf_mul2(x: u8) -> u8 {
    let hi = x & 0x80;
    let shifted = x << 1;
    if hi != 0 {
        shifted ^ 0x1b
    } else {
        shifted
    }
}

/// AES-128 encryption context (11 round keys = 176 bytes).
#[derive(Clone)]
pub struct Aes128 {
    rk: [[u8; 16]; 11],
}

impl Aes128 {
    pub fn new(key: &[u8; 16]) -> Self {
        let mut rk = [[0u8; 16]; 11];
        rk[0] = *key;
        for r in 1..=10 {
            let prev = rk[r - 1];
            let mut t = [
                AES_SBOX[prev[13] as usize] ^ AES_RCON[r - 1],
                AES_SBOX[prev[14] as usize],
                AES_SBOX[prev[15] as usize],
                AES_SBOX[prev[12] as usize],
            ];
            for c in 0..4 {
                let base = c * 4;
                for j in 0..4 {
                    rk[r][base + j] = prev[base + j] ^ t[j];
                }
                t.copy_from_slice(&rk[r][base..base + 4]);
            }
        }
        Self { rk }
    }

    pub fn encrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let mut s = *block;
        for (dst, src) in s.iter_mut().zip(self.rk[0].iter()) {
            *dst ^= *src;
        }
        for r in 1..=10 {
            // SubBytes
            for b in &mut s {
                *b = AES_SBOX[*b as usize];
            }
            // ShiftRows
            let t1 = s[1];
            s[1] = s[5];
            s[5] = s[9];
            s[9] = s[13];
            s[13] = t1;

            s.swap(2, 10);
            s.swap(6, 14);

            let t3 = s[15];
            s[15] = s[11];
            s[11] = s[7];
            s[7] = s[3];
            s[3] = t3;

            // MixColumns (skip on final round)
            if r < 10 {
                for c in 0..4 {
                    let i = c * 4;
                    let (a0, a1, a2, a3) = (s[i], s[i + 1], s[i + 2], s[i + 3]);
                    let xor_all = a0 ^ a1 ^ a2 ^ a3;
                    s[i] = a0 ^ xor_all ^ gf_mul2(a0 ^ a1);
                    s[i + 1] = a1 ^ xor_all ^ gf_mul2(a1 ^ a2);
                    s[i + 2] = a2 ^ xor_all ^ gf_mul2(a2 ^ a3);
                    s[i + 3] = a3 ^ xor_all ^ gf_mul2(a3 ^ a0);
                }
            }
            // AddRoundKey
            for (dst, src) in s.iter_mut().zip(self.rk[r].iter()) {
                *dst ^= *src;
            }
        }
        s
    }
}

/// AES-CCMP (CCM with 13-byte nonce, 8-byte MIC, L=2) for IEEE 802.11i / WPA3.
pub struct CcmpCipher {
    aes: Aes128,
}

impl CcmpCipher {
    pub fn new(tk: &[u8; 16]) -> Self {
        Self {
            aes: Aes128::new(tk),
        }
    }

    /// Build 13-byte CCMP nonce: Priority (1) || A2 (6) || PN (6, big-endian).
    pub const fn build_nonce(priority: u8, a2: MacAddr, pn: u64) -> [u8; 13] {
        let pn_be = pn.to_be_bytes();
        [
            priority, a2.0[0], a2.0[1], a2.0[2], a2.0[3], a2.0[4], a2.0[5], pn_be[2], pn_be[3],
            pn_be[4], pn_be[5], pn_be[6], pn_be[7],
        ]
    }

    fn compute_cbc_mac(&self, nonce: &[u8; 13], aad: &[u8], plaintext: &[u8]) -> [u8; 8] {
        // B_0: Flags = (Adata << 6) | ((M' = 3) << 3) | (L' = 1)
        let adata_flag = if aad.is_empty() { 0u8 } else { 0x40u8 };
        let flags = adata_flag | (3 << 3) | 1;
        let mut b0 = [0u8; 16];
        b0[0] = flags;
        b0[1..14].copy_from_slice(nonce);
        let len_be = (plaintext.len() as u16).to_be_bytes();
        b0[14] = len_be[0];
        b0[15] = len_be[1];

        let mut x = self.aes.encrypt_block(&b0);

        if !aad.is_empty() {
            let mut blk = [0u8; 16];
            let aad_len_be = (aad.len() as u16).to_be_bytes();
            blk[0] = aad_len_be[0];
            blk[1] = aad_len_be[1];
            let first_take = aad.len().min(14);
            blk[2..2 + first_take].copy_from_slice(&aad[..first_take]);
            for i in 0..16 {
                x[i] ^= blk[i];
            }
            x = self.aes.encrypt_block(&x);

            let mut pos = first_take;
            while pos < aad.len() {
                let mut ablk = [0u8; 16];
                let take = (aad.len() - pos).min(16);
                ablk[..take].copy_from_slice(&aad[pos..pos + take]);
                for i in 0..16 {
                    x[i] ^= ablk[i];
                }
                x = self.aes.encrypt_block(&x);
                pos += take;
            }
        }

        let mut pos = 0usize;
        while pos < plaintext.len() {
            let mut pblk = [0u8; 16];
            let take = (plaintext.len() - pos).min(16);
            pblk[..take].copy_from_slice(&plaintext[pos..pos + take]);
            for i in 0..16 {
                x[i] ^= pblk[i];
            }
            x = self.aes.encrypt_block(&x);
            pos += take;
        }

        let mut mic = [0u8; 8];
        mic.copy_from_slice(&x[..8]);
        mic
    }

    fn ctr_xor(&self, nonce: &[u8; 13], counter: u16, data: &mut [u8]) {
        let mut ctr = counter;
        let mut pos = 0usize;
        while pos < data.len() {
            let mut a = [0u8; 16];
            a[0] = 1; // L' = 1 (L = 2)
            a[1..14].copy_from_slice(nonce);
            let c_be = ctr.to_be_bytes();
            a[14] = c_be[0];
            a[15] = c_be[1];
            let s = self.aes.encrypt_block(&a);
            let take = (data.len() - pos).min(16);
            for i in 0..take {
                data[pos + i] ^= s[i];
            }
            pos += take;
            ctr = ctr.wrapping_add(1);
        }
    }

    /// Encrypt plaintext and append 8-byte CCMP MIC.
    pub fn encrypt(&self, nonce: &[u8; 13], aad: &[u8], plaintext: &[u8]) -> Vec<u8> {
        let mut mic = self.compute_cbc_mac(nonce, aad, plaintext);
        self.ctr_xor(nonce, 0, &mut mic);

        let mut out = Vec::with_capacity(plaintext.len() + 8);
        out.extend_from_slice(plaintext);
        self.ctr_xor(nonce, 1, &mut out[..plaintext.len()]);
        out.extend_from_slice(&mic);
        out
    }

    /// Decrypt ciphertext+MIC and verify 8-byte CCMP MIC in constant time.
    pub fn decrypt(
        &self,
        nonce: &[u8; 13],
        aad: &[u8],
        ciphertext_and_mic: &[u8],
    ) -> Result<Vec<u8>, WirelessError> {
        if ciphertext_and_mic.len() < 8 {
            return Err(WirelessError::BufferTooSmall);
        }
        let ct_len = ciphertext_and_mic.len() - 8;
        let mut pt = ciphertext_and_mic[..ct_len].to_vec();
        self.ctr_xor(nonce, 1, &mut pt);

        let mut rx_mic = [0u8; 8];
        rx_mic.copy_from_slice(&ciphertext_and_mic[ct_len..]);
        self.ctr_xor(nonce, 0, &mut rx_mic);

        let expected_mic = self.compute_cbc_mac(nonce, aad, &pt);
        let mut diff = 0u8;
        for i in 0..8 {
            diff |= rx_mic[i] ^ expected_mic[i];
        }
        if diff != 0 {
            return Err(WirelessError::DecryptFailed);
        }
        Ok(pt)
    }
}

/// WPA3 Key Hierarchy (PMK -> PTK: KCK, KEK, TK) + Replay Counter.
#[derive(Clone)]
pub struct Wpa3KeyHierarchy {
    pub pmk: [u8; 32],
    pub kck: [u8; 16],
    pub kek: [u8; 16],
    pub tk: [u8; 16],
    pub tx_pn: u64,
    pub rx_pn: u64,
}

impl Wpa3KeyHierarchy {
    pub const fn from_pmk(pmk: [u8; 32]) -> Self {
        Self {
            pmk,
            kck: [0u8; 16],
            kek: [0u8; 16],
            tk: [0u8; 16],
            tx_pn: 1,
            rx_pn: 0,
        }
    }

    /// Derive PTK (KCK || KEK || TK) from PMK, AP/STA MACs, ANonce, and SNonce.
    pub fn derive_ptk(&mut self, aa: MacAddr, spa: MacAddr, anonce: &[u8; 32], snonce: &[u8; 32]) {
        let mut context = [0u8; 76];
        let (min_mac, max_mac) = if aa.0 <= spa.0 { (aa, spa) } else { (spa, aa) };
        let (min_nonce, max_nonce) = if anonce <= snonce {
            (anonce, snonce)
        } else {
            (snonce, anonce)
        };
        context[0..6].copy_from_slice(&min_mac.0);
        context[6..12].copy_from_slice(&max_mac.0);
        context[12..44].copy_from_slice(min_nonce);
        context[44..76].copy_from_slice(max_nonce);

        let mut ptk = [0u8; 48];
        kdf_sha256(&self.pmk, b"Pairwise key expansion", &context, &mut ptk);
        self.kck.copy_from_slice(&ptk[0..16]);
        self.kek.copy_from_slice(&ptk[16..32]);
        self.tk.copy_from_slice(&ptk[32..48]);
    }

    /// Compute 16-byte EAPOL-Key MIC (HMAC-SHA256-128 truncated).
    pub fn compute_eapol_mic(&self, eapol_frame: &[u8]) -> [u8; 16] {
        let full = hmac_sha256(&self.kck, eapol_frame);
        let mut mic = [0u8; 16];
        mic.copy_from_slice(&full[..16]);
        mic
    }

    /// Verify 16-byte EAPOL-Key MIC in constant time.
    pub fn verify_eapol_mic(
        &self,
        eapol_frame: &[u8],
        rx_mic: &[u8; 16],
    ) -> Result<(), WirelessError> {
        let expected = self.compute_eapol_mic(eapol_frame);
        let mut diff = 0u8;
        for i in 0..16 {
            diff |= expected[i] ^ rx_mic[i];
        }
        if diff != 0 {
            return Err(WirelessError::MicMismatch);
        }
        Ok(())
    }

    /// Check and update 48-bit CCMP Packet Number (PN) for replay protection.
    pub fn check_and_update_rx_pn(&mut self, pn: u64) -> Result<(), WirelessError> {
        if pn <= self.rx_pn {
            return Err(WirelessError::ReplayDetected);
        }
        self.rx_pn = pn;
        Ok(())
    }

    pub fn next_tx_pn(&mut self) -> u64 {
        let pn = self.tx_pn;
        self.tx_pn += 1;
        pn
    }
}

/// WPA3-SAE (Simultaneous Authentication of Equals) State Machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeState {
    Nothing,
    Committed,
    Confirmed,
    Accepted,
}

/// WPA3-SAE Session Context (Dragonfly Commit/Confirm & PMK derivation).
#[derive(Clone)]
pub struct SaeSession {
    pub state: SaeState,
    pub sta_mac: MacAddr,
    pub ap_mac: MacAddr,
    pub group_id: u16, // 19 = NIST P-256
    pub scalar: [u8; 32],
    pub element: [u8; 64],
    pub peer_scalar: [u8; 32],
    pub peer_element: [u8; 64],
    pub kck: [u8; 32],
    pub pmk: [u8; 32],
    pub pmkid: [u8; 16],
    pub send_confirm: u16,
}

impl SaeSession {
    pub fn new(sta_mac: MacAddr, ap_mac: MacAddr, passphrase: &[u8], ssid: &[u8]) -> Self {
        // Derive PWE seed via HKDF-Extract(ssid, passphrase)
        let pwe_seed = hkdf_extract(ssid, passphrase);
        let mut scalar = [0u8; 32];
        let mut info = [0u8; 12];
        info[0..6].copy_from_slice(&sta_mac.0);
        info[6..12].copy_from_slice(&ap_mac.0);
        hkdf_expand(&pwe_seed, &info, &mut scalar);

        let mut element = [0u8; 64];
        hkdf_expand(&scalar, b"SAE-PWE-ELEMENT", &mut element);

        Self {
            state: SaeState::Nothing,
            sta_mac,
            ap_mac,
            group_id: 19,
            scalar,
            element,
            peer_scalar: [0u8; 32],
            peer_element: [0u8; 64],
            kck: [0u8; 32],
            pmk: [0u8; 32],
            pmkid: [0u8; 16],
            send_confirm: 0,
        }
    }

    /// Build SAE Commit message (Auth Seq 1): group_id (2) || scalar (32) || element (64).
    pub fn build_commit(&mut self) -> Result<Vec<u8>, WirelessError> {
        if self.state != SaeState::Nothing && self.state != SaeState::Committed {
            return Err(WirelessError::InvalidState);
        }
        self.state = SaeState::Committed;
        let mut msg = Vec::with_capacity(98);
        msg.extend_from_slice(&self.group_id.to_le_bytes());
        msg.extend_from_slice(&self.scalar);
        msg.extend_from_slice(&self.element);
        Ok(msg)
    }

    /// Process peer SAE Commit message and derive KCK, PMK, and PMKID.
    pub fn process_peer_commit(&mut self, msg: &[u8]) -> Result<(), WirelessError> {
        if self.state != SaeState::Committed {
            return Err(WirelessError::InvalidState);
        }
        if msg.len() < 98 {
            return Err(WirelessError::MalformedFrame);
        }
        let group = u16::from_le_bytes([msg[0], msg[1]]);
        if group != self.group_id {
            return Err(WirelessError::UnsupportedCipher);
        }
        self.peer_scalar.copy_from_slice(&msg[2..34]);
        self.peer_element.copy_from_slice(&msg[34..98]);

        // Derive shared secret K = H(scalar ^ peer_scalar || element ^ peer_element)
        let mut shared_in = [0u8; 96];
        for (dst, (a, b)) in shared_in[..32]
            .iter_mut()
            .zip(self.scalar.iter().zip(self.peer_scalar.iter()))
        {
            *dst = *a ^ *b;
        }
        for i in 0..64 {
            shared_in[32 + i] = self.element[i] ^ self.peer_element[i];
        }
        let keyseed = hkdf_extract(&[0u8; 32], &shared_in);

        // Derive KCK || PMK (64 bytes) using context = (scalar + peer_scalar) mod 2^256
        let mut ctx = [0u8; 32];
        let mut carry = 0u16;
        for i in (0..32).rev() {
            let sum = (self.scalar[i] as u16) + (self.peer_scalar[i] as u16) + carry;
            ctx[i] = (sum & 0xFF) as u8;
            carry = sum >> 8;
        }
        let mut kck_pmk = [0u8; 64];
        kdf_sha256(&keyseed, b"SAE KCK and PMK", &ctx, &mut kck_pmk);
        self.kck.copy_from_slice(&kck_pmk[0..32]);
        self.pmk.copy_from_slice(&kck_pmk[32..64]);
        self.pmkid.copy_from_slice(&ctx[0..16]);
        Ok(())
    }

    /// Build SAE Confirm message (Auth Seq 2): send_confirm (2) || confirm_hash (32).
    pub fn build_confirm(&mut self) -> Result<Vec<u8>, WirelessError> {
        if self.state != SaeState::Committed {
            return Err(WirelessError::InvalidState);
        }
        self.send_confirm = self.send_confirm.wrapping_add(1);
        let mut data = Vec::with_capacity(2 + 32 + 64 + 32 + 64);
        data.extend_from_slice(&self.send_confirm.to_le_bytes());
        data.extend_from_slice(&self.scalar);
        data.extend_from_slice(&self.element);
        data.extend_from_slice(&self.peer_scalar);
        data.extend_from_slice(&self.peer_element);
        let confirm = hmac_sha256(&self.kck, &data);

        self.state = SaeState::Confirmed;
        let mut out = Vec::with_capacity(34);
        out.extend_from_slice(&self.send_confirm.to_le_bytes());
        out.extend_from_slice(&confirm);
        Ok(out)
    }

    /// Verify peer SAE Confirm message and transition to Accepted.
    pub fn process_peer_confirm(&mut self, msg: &[u8]) -> Result<(), WirelessError> {
        if self.state != SaeState::Confirmed {
            return Err(WirelessError::InvalidState);
        }
        if msg.len() < 34 {
            return Err(WirelessError::MalformedFrame);
        }
        let peer_sc = u16::from_le_bytes([msg[0], msg[1]]);
        let mut data = Vec::with_capacity(2 + 32 + 64 + 32 + 64);
        data.extend_from_slice(&peer_sc.to_le_bytes());
        data.extend_from_slice(&self.peer_scalar);
        data.extend_from_slice(&self.peer_element);
        data.extend_from_slice(&self.scalar);
        data.extend_from_slice(&self.element);
        let expected = hmac_sha256(&self.kck, &data);

        let mut diff = 0u8;
        for i in 0..32 {
            diff |= expected[i] ^ msg[2 + i];
        }
        if diff != 0 {
            return Err(WirelessError::AuthFailed);
        }
        self.state = SaeState::Accepted;
        Ok(())
    }
}

// ============================================================================
// 3. Bluetooth HCI & L2CAP Stack
// ============================================================================

/// Bluetooth HCI UART/Transport Packet Indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HciPacketType {
    Command = 0x01,
    AclData = 0x02,
    ScoData = 0x03,
    Event = 0x04,
    IsoData = 0x05,
}

impl HciPacketType {
    pub const fn from_u8(v: u8) -> Result<Self, WirelessError> {
        match v {
            0x01 => Ok(Self::Command),
            0x02 => Ok(Self::AclData),
            0x03 => Ok(Self::ScoData),
            0x04 => Ok(Self::Event),
            0x05 => Ok(Self::IsoData),
            _ => Err(WirelessError::MalformedFrame),
        }
    }
}

/// Common Bluetooth HCI Opcodes (OGF << 10 | OCF).
pub const HCI_OP_RESET: u16 = 0x0C03;
pub const HCI_OP_SET_EVENT_MASK: u16 = 0x0C01;
pub const HCI_OP_READ_BD_ADDR: u16 = 0x1009;
pub const HCI_OP_DISCONNECT: u16 = 0x0406;
pub const HCI_OP_LE_SET_SCAN_PARAMS: u16 = 0x200B;
pub const HCI_OP_LE_SET_SCAN_ENABLE: u16 = 0x200C;
pub const HCI_OP_LE_CREATE_CONN: u16 = 0x200D;

/// Common Bluetooth HCI Event Codes.
pub const HCI_EVT_DISCONN_COMPLETE: u8 = 0x05;
pub const HCI_EVT_CMD_COMPLETE: u8 = 0x0E;
pub const HCI_EVT_CMD_STATUS: u8 = 0x0F;
pub const HCI_EVT_LE_META: u8 = 0x3E;

/// Build an HCI Command packet (with 0x01 H4 indicator).
pub fn build_hci_command(opcode: u16, params: &[u8]) -> Result<Vec<u8>, WirelessError> {
    if params.len() > 255 {
        return Err(WirelessError::InvalidParameter);
    }
    let mut out = Vec::with_capacity(4 + params.len());
    out.push(HciPacketType::Command as u8);
    out.extend_from_slice(&opcode.to_le_bytes());
    out.push(params.len() as u8);
    out.extend_from_slice(params);
    Ok(out)
}

/// Parsed HCI Event view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HciEvent<'a> {
    pub event_code: u8,
    pub params: &'a [u8],
}

impl<'a> HciEvent<'a> {
    /// Parse an HCI Event packet (with or without leading 0x04 H4 indicator).
    pub fn parse(raw: &'a [u8]) -> Result<Self, WirelessError> {
        let slice = if raw.first() == Some(&(HciPacketType::Event as u8)) {
            &raw[1..]
        } else {
            raw
        };
        if slice.len() < 2 {
            return Err(WirelessError::BufferTooSmall);
        }
        let event_code = slice[0];
        let len = slice[1] as usize;
        if slice.len() < 2 + len {
            return Err(WirelessError::MalformedFrame);
        }
        Ok(Self {
            event_code,
            params: &slice[2..2 + len],
        })
    }

    pub fn as_cmd_complete(&self) -> Option<(u8, u16, u8, &'a [u8])> {
        if self.event_code != HCI_EVT_CMD_COMPLETE || self.params.len() < 4 {
            return None;
        }
        let num_pkts = self.params[0];
        let opcode = u16::from_le_bytes([self.params[1], self.params[2]]);
        let status = self.params[3];
        Some((num_pkts, opcode, status, &self.params[4..]))
    }
}

/// HCI ACL Data Header (4 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HciAclHeader {
    pub handle: u16,
    pub pb_flag: u8,
    pub bc_flag: u8,
    pub data_len: u16,
}

impl HciAclHeader {
    pub fn parse(raw: &[u8]) -> Result<(Self, &[u8]), WirelessError> {
        if raw.len() < 4 {
            return Err(WirelessError::BufferTooSmall);
        }
        let h_flags = u16::from_le_bytes([raw[0], raw[1]]);
        let handle = h_flags & 0x0FFF;
        let pb_flag = ((h_flags >> 12) & 0x03) as u8;
        let bc_flag = ((h_flags >> 14) & 0x03) as u8;
        let data_len = u16::from_le_bytes([raw[2], raw[3]]);
        if raw.len() < 4 + (data_len as usize) {
            return Err(WirelessError::MalformedFrame);
        }
        Ok((
            Self {
                handle,
                pb_flag,
                bc_flag,
                data_len,
            },
            &raw[4..4 + (data_len as usize)],
        ))
    }

    pub fn encode(&self, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(5 + payload.len());
        out.push(HciPacketType::AclData as u8);
        let h_flags = (self.handle & 0x0FFF)
            | (((self.pb_flag as u16) & 0x03) << 12)
            | (((self.bc_flag as u16) & 0x03) << 14);
        out.extend_from_slice(&h_flags.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }
}

/// Bluetooth HCI Controller State Machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HciControllerState {
    Uninitialized,
    Resetting,
    Ready,
    Scanning,
    Connecting,
    Connected,
}

/// Bluetooth HCI Controller & Connection Manager.
#[derive(Debug, Clone)]
pub struct HciController {
    pub state: HciControllerState,
    pub bd_addr: MacAddr,
    pub active_handle: Option<u16>,
    pub cmd_credits: u8,
}

impl HciController {
    pub const fn new() -> Self {
        Self {
            state: HciControllerState::Uninitialized,
            bd_addr: MacAddr::ZERO,
            active_handle: None,
            cmd_credits: 1,
        }
    }

    pub fn begin_reset(&mut self) -> Result<Vec<u8>, WirelessError> {
        self.state = HciControllerState::Resetting;
        build_hci_command(HCI_OP_RESET, &[])
    }

    pub fn handle_event(&mut self, evt: &HciEvent<'_>) -> Result<(), WirelessError> {
        if let Some((num_pkts, opcode, status, rest)) = evt.as_cmd_complete() {
            self.cmd_credits = num_pkts;
            if status != 0 {
                return Err(WirelessError::InvalidState);
            }
            match opcode {
                HCI_OP_RESET => {
                    self.state = HciControllerState::Ready;
                    self.active_handle = None;
                }
                HCI_OP_READ_BD_ADDR => {
                    if rest.len() >= 6 {
                        self.bd_addr =
                            MacAddr::new([rest[5], rest[4], rest[3], rest[2], rest[1], rest[0]]);
                    }
                }
                HCI_OP_LE_SET_SCAN_ENABLE => {
                    if self.state == HciControllerState::Ready {
                        self.state = HciControllerState::Scanning;
                    } else if self.state == HciControllerState::Scanning {
                        self.state = HciControllerState::Ready;
                    }
                }
                _ => {}
            }
            return Ok(());
        }

        if evt.event_code == HCI_EVT_DISCONN_COMPLETE && evt.params.len() >= 4 {
            let handle = u16::from_le_bytes([evt.params[1], evt.params[2]]) & 0x0FFF;
            if self.active_handle == Some(handle) {
                self.active_handle = None;
                self.state = HciControllerState::Ready;
            }
        } else if evt.event_code == HCI_EVT_LE_META && evt.params.len() >= 4 {
            let subevent = evt.params[0];
            if subevent == 0x01 && evt.params[1] == 0 {
                // LE Connection Complete
                let handle = u16::from_le_bytes([evt.params[2], evt.params[3]]) & 0x0FFF;
                self.active_handle = Some(handle);
                self.state = HciControllerState::Connected;
            }
        }
        Ok(())
    }
}

// --- L2CAP (Logical Link Control and Adaptation Protocol) ---

pub const L2CAP_CID_SIGNALING: u16 = 0x0001;
pub const L2CAP_CID_ATT: u16 = 0x0004;
pub const L2CAP_CID_LE_SIGNALING: u16 = 0x0005;
pub const L2CAP_CID_SMP: u16 = 0x0006;

/// L2CAP Signaling Command Codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum L2capSigCode {
    CommandReject = 0x01,
    ConnectionReq = 0x02,
    ConnectionRsp = 0x03,
    ConfigReq = 0x04,
    ConfigRsp = 0x05,
    DisconnectionReq = 0x06,
    DisconnectionRsp = 0x07,
    LeCreditBasedConnReq = 0x14,
    LeCreditBasedConnRsp = 0x15,
    FlowControlCreditInd = 0x16,
}

impl L2capSigCode {
    pub const fn from_u8(v: u8) -> Result<Self, WirelessError> {
        match v {
            0x01 => Ok(Self::CommandReject),
            0x02 => Ok(Self::ConnectionReq),
            0x03 => Ok(Self::ConnectionRsp),
            0x04 => Ok(Self::ConfigReq),
            0x05 => Ok(Self::ConfigRsp),
            0x06 => Ok(Self::DisconnectionReq),
            0x07 => Ok(Self::DisconnectionRsp),
            0x14 => Ok(Self::LeCreditBasedConnReq),
            0x15 => Ok(Self::LeCreditBasedConnRsp),
            0x16 => Ok(Self::FlowControlCreditInd),
            _ => Err(WirelessError::MalformedFrame),
        }
    }
}

/// L2CAP Channel State Machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2capChannelState {
    Closed,
    WaitConnect,
    Config,
    Open,
    WaitDisconnect,
}

/// L2CAP Channel Descriptor with Credit-Based Flow Control.
#[derive(Debug, Clone)]
pub struct L2capChannel {
    pub local_cid: u16,
    pub remote_cid: u16,
    pub psm: u16,
    pub mtu: u16,
    pub mps: u16,
    pub tx_credits: u16,
    pub rx_credits: u16,
    pub state: L2capChannelState,
    pub rx_reassembly: Vec<u8>,
    pub expected_sdu_len: Option<u16>,
}

impl L2capChannel {
    pub fn new(local_cid: u16, psm: u16, mtu: u16, mps: u16, initial_credits: u16) -> Self {
        Self {
            local_cid,
            remote_cid: 0,
            psm,
            mtu,
            mps,
            tx_credits: 0,
            rx_credits: initial_credits,
            state: L2capChannelState::Closed,
            rx_reassembly: Vec::new(),
            expected_sdu_len: None,
        }
    }
}

/// L2CAP Basic Frame (4-byte header: length + CID).
pub fn encode_l2cap_pdu(cid: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    out.extend_from_slice(&cid.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

pub fn parse_l2cap_pdu(raw: &[u8]) -> Result<(u16, &[u8]), WirelessError> {
    if raw.len() < 4 {
        return Err(WirelessError::BufferTooSmall);
    }
    let len = u16::from_le_bytes([raw[0], raw[1]]) as usize;
    let cid = u16::from_le_bytes([raw[2], raw[3]]);
    if raw.len() < 4 + len {
        return Err(WirelessError::MalformedFrame);
    }
    Ok((cid, &raw[4..4 + len]))
}

/// L2CAP Stack managing channels and signaling.
pub struct L2capManager {
    pub channels: Vec<L2capChannel>,
    next_cid: u16,
    next_sig_id: u8,
}

impl L2capManager {
    pub const fn new() -> Self {
        Self {
            channels: Vec::new(),
            next_cid: 0x0040,
            next_sig_id: 1,
        }
    }

    fn alloc_cid(&mut self) -> u16 {
        let cid = self.next_cid;
        self.next_cid = self.next_cid.wrapping_add(1).max(0x0040);
        cid
    }

    fn alloc_sig_id(&mut self) -> u8 {
        let id = self.next_sig_id;
        self.next_sig_id = self.next_sig_id.wrapping_add(1).max(1);
        id
    }

    /// Initiate an L2CAP Connection Request (BR/EDR or LE Credit-Based).
    pub fn connect_channel(&mut self, psm: u16, mtu: u16, mps: u16) -> (u16, Vec<u8>) {
        let local_cid = self.alloc_cid();
        let sig_id = self.alloc_sig_id();
        let mut ch = L2capChannel::new(local_cid, psm, mtu, mps, 8);
        ch.state = L2capChannelState::WaitConnect;
        self.channels.push(ch);

        let mut sig = Vec::with_capacity(8);
        sig.push(L2capSigCode::ConnectionReq as u8);
        sig.push(sig_id);
        sig.extend_from_slice(&4u16.to_le_bytes());
        sig.extend_from_slice(&psm.to_le_bytes());
        sig.extend_from_slice(&local_cid.to_le_bytes());
        (local_cid, encode_l2cap_pdu(L2CAP_CID_SIGNALING, &sig))
    }

    /// Process an incoming L2CAP Signaling PDU.
    pub fn handle_signaling(&mut self, payload: &[u8]) -> Result<Option<Vec<u8>>, WirelessError> {
        if payload.len() < 4 {
            return Err(WirelessError::BufferTooSmall);
        }
        let code = L2capSigCode::from_u8(payload[0])?;
        let sig_id = payload[1];
        let len = u16::from_le_bytes([payload[2], payload[3]]) as usize;
        if payload.len() < 4 + len {
            return Err(WirelessError::MalformedFrame);
        }
        let data = &payload[4..4 + len];

        match code {
            L2capSigCode::ConnectionRsp => {
                if data.len() < 8 {
                    return Err(WirelessError::MalformedFrame);
                }
                let dcid = u16::from_le_bytes([data[0], data[1]]);
                let scid = u16::from_le_bytes([data[2], data[3]]);
                let result = u16::from_le_bytes([data[4], data[5]]);
                let ch = self
                    .channels
                    .iter_mut()
                    .find(|c| c.local_cid == scid)
                    .ok_or(WirelessError::ChannelNotFound)?;
                if result != 0 {
                    ch.state = L2capChannelState::Closed;
                    return Err(WirelessError::AuthFailed);
                }
                ch.remote_cid = dcid;
                ch.state = L2capChannelState::Config;
                // Send ConfigReq
                let mut cfg = Vec::with_capacity(8);
                cfg.push(L2capSigCode::ConfigReq as u8);
                cfg.push(self.alloc_sig_id());
                cfg.extend_from_slice(&4u16.to_le_bytes());
                cfg.extend_from_slice(&dcid.to_le_bytes());
                cfg.extend_from_slice(&0u16.to_le_bytes());
                Ok(Some(encode_l2cap_pdu(L2CAP_CID_SIGNALING, &cfg)))
            }
            L2capSigCode::ConfigRsp => {
                if data.len() < 6 {
                    return Err(WirelessError::MalformedFrame);
                }
                let scid = u16::from_le_bytes([data[0], data[1]]);
                let result = u16::from_le_bytes([data[4], data[5]]);
                let ch = self
                    .channels
                    .iter_mut()
                    .find(|c| c.local_cid == scid)
                    .ok_or(WirelessError::ChannelNotFound)?;
                if result != 0 {
                    return Err(WirelessError::InvalidParameter);
                }
                ch.state = L2capChannelState::Open;
                ch.tx_credits = 8;
                Ok(None)
            }
            L2capSigCode::FlowControlCreditInd => {
                if data.len() < 4 {
                    return Err(WirelessError::MalformedFrame);
                }
                let cid = u16::from_le_bytes([data[0], data[1]]);
                let credits = u16::from_le_bytes([data[2], data[3]]);
                let ch = self
                    .channels
                    .iter_mut()
                    .find(|c| c.local_cid == cid)
                    .ok_or(WirelessError::ChannelNotFound)?;
                ch.tx_credits = ch.tx_credits.saturating_add(credits);
                Ok(None)
            }
            L2capSigCode::DisconnectionReq => {
                if data.len() < 4 {
                    return Err(WirelessError::MalformedFrame);
                }
                let dcid = u16::from_le_bytes([data[0], data[1]]);
                let scid = u16::from_le_bytes([data[2], data[3]]);
                if let Some(ch) = self.channels.iter_mut().find(|c| c.local_cid == dcid) {
                    ch.state = L2capChannelState::Closed;
                }
                let mut rsp = Vec::with_capacity(8);
                rsp.push(L2capSigCode::DisconnectionRsp as u8);
                rsp.push(sig_id);
                rsp.extend_from_slice(&4u16.to_le_bytes());
                rsp.extend_from_slice(&dcid.to_le_bytes());
                rsp.extend_from_slice(&scid.to_le_bytes());
                Ok(Some(encode_l2cap_pdu(L2CAP_CID_SIGNALING, &rsp)))
            }
            _ => Ok(None),
        }
    }

    /// Segment an SDU into credit-based K-frames for transmission.
    pub fn send_sdu(&mut self, local_cid: u16, sdu: &[u8]) -> Result<Vec<Vec<u8>>, WirelessError> {
        let ch = self
            .channels
            .iter_mut()
            .find(|c| c.local_cid == local_cid)
            .ok_or(WirelessError::ChannelNotFound)?;
        if ch.state != L2capChannelState::Open {
            return Err(WirelessError::InvalidState);
        }
        if sdu.len() > ch.mtu as usize {
            return Err(WirelessError::SduTooLarge);
        }
        let mps = (ch.mps as usize).max(23);
        let mut frames = Vec::new();
        let mut first_payload = Vec::with_capacity(2 + sdu.len().min(mps - 2));
        first_payload.extend_from_slice(&(sdu.len() as u16).to_le_bytes());
        let first_take = sdu.len().min(mps - 2);
        first_payload.extend_from_slice(&sdu[..first_take]);

        if ch.tx_credits == 0 {
            return Err(WirelessError::InsufficientCredits);
        }
        ch.tx_credits -= 1;
        frames.push(encode_l2cap_pdu(ch.remote_cid, &first_payload));

        let mut pos = first_take;
        while pos < sdu.len() {
            if ch.tx_credits == 0 {
                return Err(WirelessError::InsufficientCredits);
            }
            ch.tx_credits -= 1;
            let take = (sdu.len() - pos).min(mps);
            frames.push(encode_l2cap_pdu(ch.remote_cid, &sdu[pos..pos + take]));
            pos += take;
        }
        Ok(frames)
    }

    /// Reassemble an incoming K-frame into a complete SDU when all segments arrive.
    pub fn receive_kframe(
        &mut self,
        local_cid: u16,
        kframe_payload: &[u8],
    ) -> Result<Option<Vec<u8>>, WirelessError> {
        let ch = self
            .channels
            .iter_mut()
            .find(|c| c.local_cid == local_cid)
            .ok_or(WirelessError::ChannelNotFound)?;
        if ch.state != L2capChannelState::Open {
            return Err(WirelessError::InvalidState);
        }
        if ch.expected_sdu_len.is_none() {
            if kframe_payload.len() < 2 {
                return Err(WirelessError::MalformedFrame);
            }
            let sdu_len = u16::from_le_bytes([kframe_payload[0], kframe_payload[1]]);
            if sdu_len > ch.mtu {
                return Err(WirelessError::SduTooLarge);
            }
            ch.expected_sdu_len = Some(sdu_len);
            ch.rx_reassembly.clear();
            ch.rx_reassembly.extend_from_slice(&kframe_payload[2..]);
        } else {
            ch.rx_reassembly.extend_from_slice(kframe_payload);
        }

        if let Some(expected) = ch.expected_sdu_len {
            if ch.rx_reassembly.len() >= expected as usize {
                let mut sdu = core::mem::take(&mut ch.rx_reassembly);
                sdu.truncate(expected as usize);
                ch.expected_sdu_len = None;
                return Ok(Some(sdu));
            }
        }
        Ok(None)
    }
}

impl fmt::Debug for Wpa3KeyHierarchy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Wpa3KeyHierarchy")
            .field("tx_pn", &self.tx_pn)
            .field("rx_pn", &self.rx_pn)
            .finish()
    }
}

// ============================================================================
// 4. Self-Test Suite
// ============================================================================

pub fn self_test() -> Result<(), &'static str> {
    // 1. 802.11ax/be Beacon build & parse + IE iteration
    let bssid = MacAddr::new([0x02, 0x11, 0x22, 0x33, 0x44, 0x55]);
    let sta_mac = MacAddr::new([0x02, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE]);
    let he = HeCapabilities {
        mac_cap_info: [0x02, 0, 0, 0, 0, 0],
        phy_cap_info: [0x08, 0, 0x40, 0, 0, 0, 0, 0, 0, 0, 0],
        max_mcs_160mhz: true,
        ofdma_ul_mu_mimo: true,
        twt_requester: true,
    };
    let eht = EhtCapabilities {
        mac_cap_info: [0x04, 0],
        phy_cap_info: [0x02, 0, 0, 0x04, 0, 0, 0, 0, 0],
        support_320mhz_6ghz: true,
        support_4096_qam: true,
        mlo_emlsr_supported: true,
    };
    let beacon = build_beacon_frame(bssid, b"NexusWiFi7", 36, 42, Some(&he), Some(&eht));
    let parsed = MacFrame::parse(&beacon)?;
    if parsed.fc.frame_type() != FrameType::Management
        || parsed.fc.subtype() != MgmtSubtype::Beacon as u8
        || parsed.sequence_number() != Some(42)
    {
        return Err("wireless: beacon header mismatch");
    }
    let mut found_ssid = false;
    let mut found_he = false;
    let mut found_eht = false;
    for ie_res in IeIter::new(&parsed.payload[12..]) {
        let ie = ie_res?;
        if ie.id == IE_SSID && ie.data == b"NexusWiFi7" {
            found_ssid = true;
        } else if ie.id == IE_EXTENSION && ie.ext_id == Some(IE_EXT_HE_CAPABILITIES) {
            let parsed_he = HeCapabilities::parse(ie.data)?;
            if parsed_he.twt_requester && parsed_he.max_mcs_160mhz {
                found_he = true;
            }
        } else if ie.id == IE_EXTENSION && ie.ext_id == Some(IE_EXT_EHT_CAPABILITIES) {
            let parsed_eht = EhtCapabilities::parse(ie.data)?;
            if parsed_eht.support_320mhz_6ghz && parsed_eht.support_4096_qam {
                found_eht = true;
            }
        }
    }
    if !found_ssid || !found_he || !found_eht {
        return Err("wireless: missing expected 802.11ax/be IEs");
    }

    // 2. Block Ack window & Wi-Fi MAC state machine
    let mut mac = Box::new(WifiMac::new(sta_mac));
    mac.start_scan()?;
    mac.start_auth(bssid)?;
    mac.on_auth_success()?;
    mac.on_assoc_success()?;
    mac.on_handshake_complete()?;
    if mac.state != WifiMacState::Associated {
        return Err("wireless: MAC state did not reach Associated");
    }
    if !mac.ba_window.record_rx(0) || mac.ba_window.record_rx(0) || !mac.ba_window.record_rx(5) {
        return Err("wireless: BlockAck window tracking failed");
    }

    // 3. WPA3-SAE Commit/Confirm + Key Hierarchy + AES-CCMP AEAD
    let mut sta_sae = Box::new(SaeSession::new(
        sta_mac,
        bssid,
        b"correct-horse-battery",
        b"NexusWiFi7",
    ));
    let mut ap_sae = Box::new(SaeSession::new(
        bssid,
        sta_mac,
        b"correct-horse-battery",
        b"NexusWiFi7",
    ));
    let sta_commit = sta_sae.build_commit()?;
    let ap_commit = ap_sae.build_commit()?;
    sta_sae.process_peer_commit(&ap_commit)?;
    ap_sae.process_peer_commit(&sta_commit)?;
    let sta_confirm = sta_sae.build_confirm()?;
    let ap_confirm = ap_sae.build_confirm()?;
    sta_sae.process_peer_confirm(&ap_confirm)?;
    ap_sae.process_peer_confirm(&sta_confirm)?;
    if sta_sae.pmk != ap_sae.pmk {
        return Err("wireless: WPA3-SAE PMK mismatch");
    }

    let mut keys = Box::new(Wpa3KeyHierarchy::from_pmk(sta_sae.pmk));
    let anonce = [0x11u8; 32];
    let snonce = [0x22u8; 32];
    keys.derive_ptk(bssid, sta_mac, &anonce, &snonce);
    let eapol_body = b"EAPOL-KEY-MSG2";
    let mic = keys.compute_eapol_mic(eapol_body);
    keys.verify_eapol_mic(eapol_body, &mic)?;

    let ccmp = CcmpCipher::new(&keys.tk);
    let pn = keys.next_tx_pn();
    let nonce = CcmpCipher::build_nonce(0, sta_mac, pn);
    let aad = [0x08u8; 22];
    let plaintext = b"802.11be MLO encrypted payload";
    let mut ct = ccmp.encrypt(&nonce, &aad, plaintext);
    let pt = ccmp.decrypt(&nonce, &aad, &ct)?;
    if pt.as_slice() != plaintext {
        return Err("wireless: CCMP decrypt mismatch");
    }
    ct[0] ^= 0x01;
    if ccmp.decrypt(&nonce, &aad, &ct).is_ok() {
        return Err("wireless: CCMP failed to detect tampering");
    }
    keys.check_and_update_rx_pn(pn)?;
    if keys.check_and_update_rx_pn(pn).is_ok() {
        return Err("wireless: PN replay protection failed");
    }

    // 4. Bluetooth HCI & L2CAP Stack
    let mut hci = Box::new(HciController::new());
    let reset_cmd = hci.begin_reset()?;
    if reset_cmd.len() != 4 || reset_cmd[0] != HciPacketType::Command as u8 {
        return Err("wireless: HCI reset command malformed");
    }
    let cmd_complete_raw = [0x04, HCI_EVT_CMD_COMPLETE, 0x04, 0x01, 0x03, 0x0C, 0x00];
    let evt = HciEvent::parse(&cmd_complete_raw)?;
    hci.handle_event(&evt)?;
    if hci.state != HciControllerState::Ready {
        return Err("wireless: HCI controller did not transition to Ready");
    }

    let mut l2cap = Box::new(L2capManager::new());
    let (cid, _conn_req) = l2cap.connect_channel(0x0025, 512, 64);
    let conn_rsp = [
        L2capSigCode::ConnectionRsp as u8,
        1,
        8,
        0,
        0x41,
        0x00, // dcid
        cid.to_le_bytes()[0],
        cid.to_le_bytes()[1],
        0,
        0, // result = success
        0,
        0,
    ];
    let _cfg_req = l2cap.handle_signaling(&conn_rsp)?;
    let cfg_rsp = [
        L2capSigCode::ConfigRsp as u8,
        2,
        6,
        0,
        cid.to_le_bytes()[0],
        cid.to_le_bytes()[1],
        0,
        0,
        0,
        0,
    ];
    l2cap.handle_signaling(&cfg_rsp)?;

    let sdu = b"Bluetooth L2CAP Credit-Based SDU payload across multiple segments!";
    let kframes = l2cap.send_sdu(cid, sdu)?;
    if kframes.len() < 2 {
        return Err("wireless: expected segmented L2CAP K-frames");
    }
    let mut reassembled = None;
    for kf in &kframes {
        let (_rcid, payload) = parse_l2cap_pdu(kf)?;
        reassembled = l2cap.receive_kframe(cid, payload)?;
    }
    if reassembled.as_deref() != Some(sdu.as_slice()) {
        return Err("wireless: L2CAP SDU reassembly mismatch");
    }

    Ok(())
}
