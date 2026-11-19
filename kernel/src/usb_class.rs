//! Clean-room USB4/xHCI Isochronous Transfers & USB HID/Audio/Mass-Storage Class Drivers.
//!
//! Provides `no_std` building blocks for:
//! - xHCI / USB4-tunneled Isochronous TRB ring management and ESIT payload calculation
//! - USB HID Report Descriptor parser and boot-protocol keyboard/mouse decoders
//! - USB Audio Class 2.0 (UAC2) descriptor parser and control request builders
//! - USB Mass Storage Bulk-Only Transport (BOT) & UASP SCSI command builders
#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;

use crate::xhci::{Trb, TRB_CYCLE_BIT, TRB_IOC_BIT, TRB_TYPE_LINK};

// ---------------------------------------------------------------------------
// Common Error Type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsbClassError {
    InvalidParameter,
    RingFull,
    RingEmpty,
    BufferTooSmall,
    MalformedDescriptor,
    UnsupportedFormat,
    InvalidSignature,
    StackOverflow,
}

impl UsbClassError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidParameter => "usb_class: invalid parameter",
            Self::RingFull => "usb_class: isochronous TRB ring full",
            Self::RingEmpty => "usb_class: isochronous TRB ring empty",
            Self::BufferTooSmall => "usb_class: buffer too small",
            Self::MalformedDescriptor => "usb_class: malformed USB descriptor",
            Self::UnsupportedFormat => "usb_class: unsupported descriptor or format",
            Self::InvalidSignature => "usb_class: invalid protocol signature",
            Self::StackOverflow => "usb_class: HID global state stack overflow",
        }
    }
}

impl From<UsbClassError> for &'static str {
    fn from(err: UsbClassError) -> Self {
        err.as_str()
    }
}

// ---------------------------------------------------------------------------
// 1. USB4 / xHCI Isochronous Transfers & TRB Ring
// ---------------------------------------------------------------------------

pub const TRB_TYPE_ISOCH: u32 = 5;
pub const TRB_TYPE_EVENT_DATA: u32 = 7;

pub const TRB_ISP_BIT: u32 = 1 << 2;
pub const TRB_CHAIN_BIT: u32 = 1 << 4;
pub const TRB_IDT_BIT: u32 = 1 << 6;
pub const TRB_BEI_BIT: u32 = 1 << 9;
pub const TRB_SIA_BIT: u32 = 1 << 31;

/// xHCI Transfer Event completion codes relevant to isochronous streams.
pub const XHCI_CC_SUCCESS: u8 = 1;
pub const XHCI_CC_SHORT_PACKET: u8 = 13;
pub const XHCI_CC_RING_UNDERRUN: u8 = 14;
pub const XHCI_CC_RING_OVERRUN: u8 = 15;
pub const XHCI_CC_MISSED_SERVICE: u8 = 23;

/// Parameters for constructing an xHCI Isochronous Transfer Request Block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsochTrbParams {
    pub buf_phys: u64,
    pub len: u32,
    pub frame_id: u16,
    pub sia: bool,
    pub ioc: bool,
    pub isp: bool,
    pub chain: bool,
    pub tbc: u8,
    pub tlbpc: u8,
    pub td_size: u8,
    pub intr_target: u16,
}

impl Default for IsochTrbParams {
    fn default() -> Self {
        Self {
            buf_phys: 0,
            len: 0,
            frame_id: 0,
            sia: true,
            ioc: true,
            isp: false,
            chain: false,
            tbc: 0,
            tlbpc: 0,
            td_size: 0,
            intr_target: 0,
        }
    }
}

/// Build an xHCI Isochronous TRB (Type 5) per xHCI 1.2 Section 6.4.1.3.
pub const fn build_isoch_trb(params: &IsochTrbParams, cycle: bool) -> Trb {
    let status = (params.len & 0x1_FFFF)
        | (((params.td_size as u32) & 0x1F) << 17)
        | (((params.intr_target as u32) & 0x3FF) << 22);

    let mut flags = ((params.tbc as u32) & 0x3) << 7;
    flags |= ((params.tlbpc as u32) & 0xF) << 16;
    if !params.sia {
        flags |= ((params.frame_id as u32) & 0x7FF) << 20;
    } else {
        flags |= TRB_SIA_BIT;
    }
    if cycle {
        flags |= TRB_CYCLE_BIT;
    }
    if params.isp {
        flags |= TRB_ISP_BIT;
    }
    if params.chain {
        flags |= TRB_CHAIN_BIT;
    }
    if params.ioc {
        flags |= TRB_IOC_BIT;
    }

    Trb::new(params.buf_phys, status, TRB_TYPE_ISOCH, flags)
}

/// USB4 / SuperSpeedPlus Isochronous Endpoint configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsochEndpointConfig {
    pub endpoint_addr: u8,
    pub max_packet_size: u16,
    pub max_burst: u8,
    pub mult: u8,
    pub interval: u8,
    pub usb4_tunneled: bool,
}

impl IsochEndpointConfig {
    /// Compute Extended Service Interval Time (ESIT) payload in bytes.
    pub const fn max_esit_payload(&self) -> u32 {
        let packets_per_burst = (self.max_burst as u32) + 1;
        let bursts_per_interval = (self.mult as u32) + 1;
        (self.max_packet_size as u32) * packets_per_burst * bursts_per_interval
    }

    /// Compute service interval in microseconds (`125 * 2^(interval - 1)` for HS/SS/USB4).
    pub const fn interval_us(&self) -> u32 {
        if self.interval == 0 {
            return 125;
        }
        let shift = if self.interval > 16 {
            15
        } else {
            (self.interval - 1) as u32
        };
        125u32 << shift
    }
}

/// Outcome of a completed Isochronous TRB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsochCompletion {
    pub trb_index: usize,
    pub completion_code: u8,
    pub residual_bytes: u32,
    pub underrun_or_missed: bool,
}

/// Heap-backed xHCI Isochronous Transfer Ring with Link TRB wrap-around.
#[derive(Debug)]
pub struct IsochTrbRing {
    trbs: Vec<Trb>,
    ring_phys: u64,
    enqueue_idx: usize,
    dequeue_idx: usize,
    pending: usize,
    cycle: bool,
    pub completed_count: u64,
    pub underrun_count: u64,
}

impl IsochTrbRing {
    /// Allocate a new Isochronous TRB ring with `capacity` slots (last slot is Link TRB).
    pub fn new(capacity: usize, ring_phys: u64) -> Result<Self, UsbClassError> {
        if capacity < 4 {
            return Err(UsbClassError::InvalidParameter);
        }
        let mut trbs = vec![Trb::zeroed(); capacity];
        trbs[capacity - 1] = Trb::link(ring_phys, true, true);
        Ok(Self {
            trbs,
            ring_phys,
            enqueue_idx: 0,
            dequeue_idx: 0,
            pending: 0,
            cycle: true,
            completed_count: 0,
            underrun_count: 0,
        })
    }

    #[inline]
    pub fn usable_slots(&self) -> usize {
        self.trbs.len() - 1
    }

    #[inline]
    pub fn available_slots(&self) -> usize {
        self.usable_slots().saturating_sub(self.pending)
    }

    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending
    }

    #[inline]
    pub fn cycle_state(&self) -> bool {
        self.cycle
    }

    pub fn trb_at(&self, idx: usize) -> Option<&Trb> {
        self.trbs.get(idx)
    }

    /// Enqueue an Isochronous TRB, updating the Link TRB and toggling cycle on wrap.
    pub fn enqueue_isoch(&mut self, params: &IsochTrbParams) -> Result<usize, UsbClassError> {
        if self.available_slots() == 0 {
            return Err(UsbClassError::RingFull);
        }
        let idx = self.enqueue_idx;
        self.trbs[idx] = build_isoch_trb(params, self.cycle);
        self.pending += 1;
        self.enqueue_idx += 1;

        let link_idx = self.trbs.len() - 1;
        if self.enqueue_idx >= link_idx {
            self.trbs[link_idx] = Trb::link(self.ring_phys, true, self.cycle);
            self.enqueue_idx = 0;
            self.cycle = !self.cycle;
        }
        Ok(idx)
    }

    /// Retire the oldest pending Isochronous TRB upon receiving a Transfer Event.
    pub fn complete_one(
        &mut self,
        completion_code: u8,
        residual_bytes: u32,
    ) -> Result<IsochCompletion, UsbClassError> {
        if self.pending == 0 {
            return Err(UsbClassError::RingEmpty);
        }
        let idx = self.dequeue_idx;
        self.dequeue_idx += 1;
        if self.dequeue_idx >= self.trbs.len() - 1 {
            self.dequeue_idx = 0;
        }
        self.pending -= 1;
        self.completed_count += 1;

        let underrun_or_missed = matches!(
            completion_code,
            XHCI_CC_RING_UNDERRUN | XHCI_CC_RING_OVERRUN | XHCI_CC_MISSED_SERVICE
        );
        if underrun_or_missed {
            self.underrun_count += 1;
        }

        Ok(IsochCompletion {
            trb_index: idx,
            completion_code,
            residual_bytes,
            underrun_or_missed,
        })
    }
}

// ---------------------------------------------------------------------------
// 2. USB HID Report Descriptor Parser & Boot Protocol Decoders
// ---------------------------------------------------------------------------

pub const USB_CLASS_HID: u8 = 0x03;
pub const HID_SUBCLASS_BOOT: u8 = 0x01;
pub const HID_PROTO_KEYBOARD: u8 = 0x01;
pub const HID_PROTO_MOUSE: u8 = 0x02;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HidReportKind {
    Input,
    Output,
    Feature,
}

#[derive(Debug, Clone, Copy, Default)]
struct HidGlobalState {
    usage_page: u16,
    logical_min: i32,
    logical_max: i32,
    physical_min: i32,
    physical_max: i32,
    report_size: u8,
    report_count: u16,
    report_id: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HidField {
    pub report_id: u8,
    pub kind: HidReportKind,
    pub usage_page: u16,
    pub usage: u16,
    pub bit_offset: u32,
    pub bit_size: u8,
    pub logical_min: i32,
    pub logical_max: i32,
    pub flags: u16,
}

impl HidField {
    #[inline]
    pub const fn is_constant(&self) -> bool {
        (self.flags & 0x01) != 0
    }

    #[inline]
    pub const fn is_variable(&self) -> bool {
        (self.flags & 0x02) != 0
    }

    #[inline]
    pub const fn is_relative(&self) -> bool {
        (self.flags & 0x04) != 0
    }
}

#[derive(Debug, Default)]
pub struct HidReportDescriptor {
    pub fields: Vec<HidField>,
    pub input_bit_len: u32,
    pub output_bit_len: u32,
    pub feature_bit_len: u32,
}

fn decode_hid_unsigned(data: &[u8]) -> u32 {
    match data.len() {
        0 => 0,
        1 => data[0] as u32,
        2 => u16::from_le_bytes([data[0], data[1]]) as u32,
        4 => u32::from_le_bytes([data[0], data[1], data[2], data[3]]),
        _ => 0,
    }
}

fn decode_hid_signed(data: &[u8]) -> i32 {
    match data.len() {
        0 => 0,
        1 => (data[0] as i8) as i32,
        2 => i16::from_le_bytes([data[0], data[1]]) as i32,
        4 => i32::from_le_bytes([data[0], data[1], data[2], data[3]]),
        _ => 0,
    }
}

impl HidReportDescriptor {
    /// Parse a raw USB HID Report Descriptor byte stream.
    pub fn parse(bytes: &[u8]) -> Result<Self, UsbClassError> {
        let mut desc = Self::default();
        let mut global = HidGlobalState::default();
        let mut stack = [HidGlobalState::default(); 4];
        let mut stack_depth: usize = 0;

        let mut local_usages: Vec<u16> = Vec::new();
        let mut usage_min: Option<u16> = None;
        let mut usage_max: Option<u16> = None;

        let mut i = 0usize;
        while i < bytes.len() {
            let prefix = bytes[i];
            i += 1;

            // Long item (0xFE)
            if prefix == 0xFE {
                if i + 2 > bytes.len() {
                    return Err(UsbClassError::MalformedDescriptor);
                }
                let data_len = bytes[i] as usize;
                i += 2; // skip bDataSize and bLongItemTag
                if i + data_len > bytes.len() {
                    return Err(UsbClassError::MalformedDescriptor);
                }
                i += data_len;
                continue;
            }

            let size_code = prefix & 0x03;
            let data_len = match size_code {
                0 => 0,
                1 => 1,
                2 => 2,
                3 => 4,
                _ => 0,
            };
            if i + data_len > bytes.len() {
                return Err(UsbClassError::MalformedDescriptor);
            }
            let data = &bytes[i..i + data_len];
            i += data_len;

            let item_type = (prefix >> 2) & 0x03;
            let tag = (prefix >> 4) & 0x0F;

            match item_type {
                // Main items
                0 => {
                    let kind = match tag {
                        0x8 => Some(HidReportKind::Input),
                        0x9 => Some(HidReportKind::Output),
                        0xB => Some(HidReportKind::Feature),
                        _ => None,
                    };
                    if let Some(report_kind) = kind {
                        let flags = decode_hid_unsigned(data) as u16;
                        let count = global.report_count as usize;
                        let bit_size = global.report_size;
                        let bit_cursor = match report_kind {
                            HidReportKind::Input => &mut desc.input_bit_len,
                            HidReportKind::Output => &mut desc.output_bit_len,
                            HidReportKind::Feature => &mut desc.feature_bit_len,
                        };

                        for idx in 0..count {
                            let usage = if let Some(&u) = local_usages.get(idx) {
                                u
                            } else if let (Some(umin), Some(umax)) = (usage_min, usage_max) {
                                let candidate = umin.saturating_add(idx as u16);
                                if candidate <= umax {
                                    candidate
                                } else {
                                    umax
                                }
                            } else {
                                local_usages.last().copied().unwrap_or(0)
                            };

                            desc.fields.push(HidField {
                                report_id: global.report_id,
                                kind: report_kind,
                                usage_page: global.usage_page,
                                usage,
                                bit_offset: *bit_cursor,
                                bit_size,
                                logical_min: global.logical_min,
                                logical_max: global.logical_max,
                                flags,
                            });
                            *bit_cursor = bit_cursor.saturating_add(bit_size as u32);
                        }
                    }
                    // Main items clear local state
                    local_usages.clear();
                    usage_min = None;
                    usage_max = None;
                }
                // Global items
                1 => match tag {
                    0x0 => global.usage_page = decode_hid_unsigned(data) as u16,
                    0x1 => global.logical_min = decode_hid_signed(data),
                    0x2 => global.logical_max = decode_hid_signed(data),
                    0x3 => global.physical_min = decode_hid_signed(data),
                    0x4 => global.physical_max = decode_hid_signed(data),
                    0x7 => global.report_size = decode_hid_unsigned(data) as u8,
                    0x8 => global.report_id = decode_hid_unsigned(data) as u8,
                    0x9 => global.report_count = decode_hid_unsigned(data) as u16,
                    0xA => {
                        if stack_depth >= stack.len() {
                            return Err(UsbClassError::StackOverflow);
                        }
                        stack[stack_depth] = global;
                        stack_depth += 1;
                    }
                    0xB => {
                        if stack_depth == 0 {
                            return Err(UsbClassError::MalformedDescriptor);
                        }
                        stack_depth -= 1;
                        global = stack[stack_depth];
                    }
                    _ => {}
                },
                // Local items
                2 => match tag {
                    0x0 => local_usages.push(decode_hid_unsigned(data) as u16),
                    0x1 => usage_min = Some(decode_hid_unsigned(data) as u16),
                    0x2 => usage_max = Some(decode_hid_unsigned(data) as u16),
                    _ => {}
                },
                _ => {}
            }
        }

        Ok(desc)
    }

    /// Extract an unsigned bitfield value from a raw HID report buffer.
    pub fn extract_unsigned(report: &[u8], field: &HidField) -> Option<u32> {
        if field.bit_size == 0 || field.bit_size > 32 {
            return None;
        }
        let end_bit = field.bit_offset.checked_add(field.bit_size as u32)?;
        if (end_bit as usize) > report.len() * 8 {
            return None;
        }
        let mut val = 0u32;
        for b in 0..field.bit_size {
            let bit_pos = (field.bit_offset + b as u32) as usize;
            let byte = report[bit_pos / 8];
            let bit = (byte >> (bit_pos % 8)) & 1;
            val |= (bit as u32) << b;
        }
        Some(val)
    }

    /// Extract a signed bitfield value (sign-extended if `logical_min < 0`).
    pub fn extract_signed(report: &[u8], field: &HidField) -> Option<i32> {
        let raw = Self::extract_unsigned(report, field)?;
        if field.logical_min < 0 && field.bit_size < 32 {
            let sign_bit = 1u32 << (field.bit_size - 1);
            if (raw & sign_bit) != 0 {
                let mask = !((1u32 << field.bit_size) - 1);
                return Some((raw | mask) as i32);
            }
        }
        Some(raw as i32)
    }
}

/// Standard 8-byte USB HID Boot Keyboard report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootKeyboardReport {
    pub modifiers: u8,
    pub keycodes: [u8; 6],
}

pub fn parse_boot_keyboard(data: &[u8]) -> Option<BootKeyboardReport> {
    if data.len() < 8 {
        return None;
    }
    Some(BootKeyboardReport {
        modifiers: data[0],
        keycodes: [data[2], data[3], data[4], data[5], data[6], data[7]],
    })
}

/// Standard 3- or 4-byte USB HID Boot Mouse report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootMouseReport {
    pub buttons: u8,
    pub dx: i8,
    pub dy: i8,
    pub wheel: i8,
}

pub fn parse_boot_mouse(data: &[u8]) -> Option<BootMouseReport> {
    if data.len() < 3 {
        return None;
    }
    Some(BootMouseReport {
        buttons: data[0],
        dx: data[1] as i8,
        dy: data[2] as i8,
        wheel: if data.len() >= 4 { data[3] as i8 } else { 0 },
    })
}

// ---------------------------------------------------------------------------
// 3. USB Audio Class 2.0 (UAC2) Descriptors & Control Requests
// ---------------------------------------------------------------------------

pub const USB_CLASS_AUDIO: u8 = 0x01;
pub const AUDIO_SUBCLASS_CONTROL: u8 = 0x01;
pub const AUDIO_SUBCLASS_STREAMING: u8 = 0x02;
pub const AUDIO_PROTO_UAC2: u8 = 0x20;

pub const USB_DESC_CS_INTERFACE: u8 = 0x24;
pub const USB_DESC_CS_ENDPOINT: u8 = 0x25;

// UAC2 AudioControl Interface Descriptor Subtypes
pub const UAC2_AC_HEADER: u8 = 0x01;
pub const UAC2_AC_INPUT_TERMINAL: u8 = 0x02;
pub const UAC2_AC_OUTPUT_TERMINAL: u8 = 0x03;
pub const UAC2_AC_FEATURE_UNIT: u8 = 0x06;
pub const UAC2_AC_CLOCK_SOURCE: u8 = 0x0A;
pub const UAC2_AC_CLOCK_SELECTOR: u8 = 0x0B;

// UAC2 AudioStreaming Interface Descriptor Subtypes
pub const UAC2_AS_GENERAL: u8 = 0x01;
pub const UAC2_AS_FORMAT_TYPE: u8 = 0x02;

// UAC2 Control Selectors
pub const UAC2_CS_SAM_FREQ_CONTROL: u8 = 0x01;
pub const UAC2_FU_MUTE_CONTROL: u8 = 0x01;
pub const UAC2_FU_VOLUME_CONTROL: u8 = 0x02;
pub const UAC2_REQ_CUR: u8 = 0x01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Uac2ClockSource {
    pub clock_id: u8,
    pub attributes: u8,
    pub controls: u8,
    pub assoc_terminal: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Uac2Terminal {
    pub terminal_id: u8,
    pub terminal_type: u16,
    pub assoc_terminal: u8,
    pub clock_source_id: u8,
    pub nr_channels: u8,
    pub is_input: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Uac2FeatureUnit {
    pub unit_id: u8,
    pub source_id: u8,
    pub master_controls: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Uac2AsGeneral {
    pub terminal_link: u8,
    pub controls: u8,
    pub format_type: u8,
    pub formats: u32,
    pub nr_channels: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Uac2FormatTypeI {
    pub format_type: u8,
    pub subslot_size: u8,
    pub bit_resolution: u8,
}

#[derive(Debug, Default)]
pub struct Uac2DeviceConfig {
    pub clock_sources: Vec<Uac2ClockSource>,
    pub terminals: Vec<Uac2Terminal>,
    pub feature_units: Vec<Uac2FeatureUnit>,
    pub as_general: Option<Uac2AsGeneral>,
    pub format_type_i: Option<Uac2FormatTypeI>,
}

impl Uac2DeviceConfig {
    /// Parse a concatenated stream of UAC2 class-specific interface descriptors.
    pub fn parse(stream: &[u8]) -> Result<Self, UsbClassError> {
        let mut cfg = Self::default();
        let mut offset = 0usize;

        while offset < stream.len() {
            let len = stream[offset] as usize;
            if len < 3 || offset + len > stream.len() {
                return Err(UsbClassError::MalformedDescriptor);
            }
            let desc = &stream[offset..offset + len];
            offset += len;

            if desc[1] != USB_DESC_CS_INTERFACE {
                continue;
            }

            let subtype = desc[2];
            match subtype {
                UAC2_AC_CLOCK_SOURCE if len >= 8 => {
                    cfg.clock_sources.push(Uac2ClockSource {
                        clock_id: desc[3],
                        attributes: desc[4],
                        controls: desc[5],
                        assoc_terminal: desc[6],
                    });
                }
                UAC2_AC_INPUT_TERMINAL if len >= 17 => {
                    cfg.terminals.push(Uac2Terminal {
                        terminal_id: desc[3],
                        terminal_type: u16::from_le_bytes([desc[4], desc[5]]),
                        assoc_terminal: desc[6],
                        clock_source_id: desc[7],
                        nr_channels: desc[8],
                        is_input: true,
                    });
                }
                UAC2_AC_OUTPUT_TERMINAL if len >= 12 => {
                    cfg.terminals.push(Uac2Terminal {
                        terminal_id: desc[3],
                        terminal_type: u16::from_le_bytes([desc[4], desc[5]]),
                        assoc_terminal: desc[6],
                        clock_source_id: desc[8],
                        nr_channels: 0,
                        is_input: false,
                    });
                }
                UAC2_AC_FEATURE_UNIT if len >= 10 => {
                    let master_controls = u32::from_le_bytes([desc[5], desc[6], desc[7], desc[8]]);
                    cfg.feature_units.push(Uac2FeatureUnit {
                        unit_id: desc[3],
                        source_id: desc[4],
                        master_controls,
                    });
                }
                UAC2_AS_GENERAL if len >= 16 => {
                    cfg.as_general = Some(Uac2AsGeneral {
                        terminal_link: desc[3],
                        controls: desc[4],
                        format_type: desc[5],
                        formats: u32::from_le_bytes([desc[6], desc[7], desc[8], desc[9]]),
                        nr_channels: desc[10],
                    });
                }
                UAC2_AS_FORMAT_TYPE if len >= 6 => {
                    cfg.format_type_i = Some(Uac2FormatTypeI {
                        format_type: desc[3],
                        subslot_size: desc[4],
                        bit_resolution: desc[5],
                    });
                }
                _ => {}
            }
        }

        Ok(cfg)
    }
}

/// Build a UAC2 `SET_CUR` control transfer setup packet + 4-byte payload for sample rate.
pub const fn build_uac2_set_sample_rate(
    clock_id: u8,
    ac_interface: u8,
    sample_rate_hz: u32,
) -> ([u8; 8], [u8; 4]) {
    let w_value = (UAC2_CS_SAM_FREQ_CONTROL as u16) << 8;
    let w_index = ((clock_id as u16) << 8) | (ac_interface as u16);
    let setup = [
        0x21, // Host-to-device, Class, Interface
        UAC2_REQ_CUR,
        (w_value & 0xFF) as u8,
        (w_value >> 8) as u8,
        (w_index & 0xFF) as u8,
        (w_index >> 8) as u8,
        4,
        0,
    ];
    let payload = sample_rate_hz.to_le_bytes();
    (setup, payload)
}

/// Build a UAC2 `SET_CUR` control transfer setup packet + 2-byte payload for volume (1/256 dB).
pub const fn build_uac2_set_volume(
    unit_id: u8,
    ac_interface: u8,
    channel: u8,
    volume_q8_db: i16,
) -> ([u8; 8], [u8; 2]) {
    let w_value = ((UAC2_FU_VOLUME_CONTROL as u16) << 8) | (channel as u16);
    let w_index = ((unit_id as u16) << 8) | (ac_interface as u16);
    let setup = [
        0x21,
        UAC2_REQ_CUR,
        (w_value & 0xFF) as u8,
        (w_value >> 8) as u8,
        (w_index & 0xFF) as u8,
        (w_index >> 8) as u8,
        2,
        0,
    ];
    let payload = volume_q8_db.to_le_bytes();
    (setup, payload)
}

// ---------------------------------------------------------------------------
// 4. USB Mass Storage (BOT & UASP) SCSI Command Builder
// ---------------------------------------------------------------------------

pub const USB_CLASS_MASS_STORAGE: u8 = 0x08;
pub const MSC_SUBCLASS_SCSI: u8 = 0x06;
pub const MSC_PROTO_BOT: u8 = 0x50;
pub const MSC_PROTO_UASP: u8 = 0x62;

pub const CBW_SIGNATURE: u32 = 0x4342_5355; // "USBC"
pub const CSW_SIGNATURE: u32 = 0x5342_5355; // "USBS"

pub const CBW_FLAG_DATA_OUT: u8 = 0x00;
pub const CBW_FLAG_DATA_IN: u8 = 0x80;

pub const UASP_IU_COMMAND: u8 = 0x01;
pub const UASP_IU_SENSE: u8 = 0x03;
pub const UASP_IU_RESPONSE: u8 = 0x04;
pub const UASP_IU_TASK_MGMT: u8 = 0x05;

/// SCSI Command Descriptor Block (6, 10, 12, or 16 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScsiCdb {
    pub bytes: [u8; 16],
    pub len: u8,
}

impl ScsiCdb {
    pub const fn test_unit_ready() -> Self {
        let mut bytes = [0u8; 16];
        bytes[0] = 0x00;
        Self { bytes, len: 6 }
    }

    pub const fn request_sense(alloc_len: u8) -> Self {
        let mut bytes = [0u8; 16];
        bytes[0] = 0x03;
        bytes[4] = alloc_len;
        Self { bytes, len: 6 }
    }

    pub const fn inquiry(alloc_len: u8) -> Self {
        let mut bytes = [0u8; 16];
        bytes[0] = 0x12;
        bytes[4] = alloc_len;
        Self { bytes, len: 6 }
    }

    pub const fn read_capacity_10() -> Self {
        let mut bytes = [0u8; 16];
        bytes[0] = 0x25;
        Self { bytes, len: 10 }
    }

    pub const fn read_10(lba: u32, blocks: u16) -> Self {
        let mut bytes = [0u8; 16];
        let lba_be = lba.to_be_bytes();
        let blk_be = blocks.to_be_bytes();
        bytes[0] = 0x28;
        bytes[2] = lba_be[0];
        bytes[3] = lba_be[1];
        bytes[4] = lba_be[2];
        bytes[5] = lba_be[3];
        bytes[7] = blk_be[0];
        bytes[8] = blk_be[1];
        Self { bytes, len: 10 }
    }

    pub const fn write_10(lba: u32, blocks: u16) -> Self {
        let mut bytes = [0u8; 16];
        let lba_be = lba.to_be_bytes();
        let blk_be = blocks.to_be_bytes();
        bytes[0] = 0x2A;
        bytes[2] = lba_be[0];
        bytes[3] = lba_be[1];
        bytes[4] = lba_be[2];
        bytes[5] = lba_be[3];
        bytes[7] = blk_be[0];
        bytes[8] = blk_be[1];
        Self { bytes, len: 10 }
    }

    pub const fn read_16(lba: u64, blocks: u32) -> Self {
        let mut bytes = [0u8; 16];
        let lba_be = lba.to_be_bytes();
        let blk_be = blocks.to_be_bytes();
        bytes[0] = 0x88;
        bytes[2] = lba_be[0];
        bytes[3] = lba_be[1];
        bytes[4] = lba_be[2];
        bytes[5] = lba_be[3];
        bytes[6] = lba_be[4];
        bytes[7] = lba_be[5];
        bytes[8] = lba_be[6];
        bytes[9] = lba_be[7];
        bytes[10] = blk_be[0];
        bytes[11] = blk_be[1];
        bytes[12] = blk_be[2];
        bytes[13] = blk_be[3];
        Self { bytes, len: 16 }
    }

    pub const fn write_16(lba: u64, blocks: u32) -> Self {
        let mut bytes = [0u8; 16];
        let lba_be = lba.to_be_bytes();
        let blk_be = blocks.to_be_bytes();
        bytes[0] = 0x8A;
        bytes[2] = lba_be[0];
        bytes[3] = lba_be[1];
        bytes[4] = lba_be[2];
        bytes[5] = lba_be[3];
        bytes[6] = lba_be[4];
        bytes[7] = lba_be[5];
        bytes[8] = lba_be[6];
        bytes[9] = lba_be[7];
        bytes[10] = blk_be[0];
        bytes[11] = blk_be[1];
        bytes[12] = blk_be[2];
        bytes[13] = blk_be[3];
        Self { bytes, len: 16 }
    }
}

/// Bulk-Only Transport Command Block Wrapper (31 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandBlockWrapper {
    pub tag: u32,
    pub data_transfer_length: u32,
    pub flags: u8,
    pub lun: u8,
    pub cdb: ScsiCdb,
}

impl CommandBlockWrapper {
    pub fn to_bytes(self) -> Result<[u8; 31], UsbClassError> {
        if self.cdb.len == 0 || self.cdb.len > 16 {
            return Err(UsbClassError::InvalidParameter);
        }
        let mut out = [0u8; 31];
        out[0..4].copy_from_slice(&CBW_SIGNATURE.to_le_bytes());
        out[4..8].copy_from_slice(&self.tag.to_le_bytes());
        out[8..12].copy_from_slice(&self.data_transfer_length.to_le_bytes());
        out[12] = self.flags;
        out[13] = self.lun & 0x0F;
        out[14] = self.cdb.len;
        out[15..31].copy_from_slice(&self.cdb.bytes);
        Ok(out)
    }
}

/// Bulk-Only Transport Command Status Wrapper (13 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandStatusWrapper {
    pub tag: u32,
    pub data_residue: u32,
    pub status: u8,
}

impl CommandStatusWrapper {
    pub fn from_bytes(data: &[u8]) -> Result<Self, UsbClassError> {
        if data.len() < 13 {
            return Err(UsbClassError::BufferTooSmall);
        }
        let sig = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if sig != CSW_SIGNATURE {
            return Err(UsbClassError::InvalidSignature);
        }
        Ok(Self {
            tag: u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
            data_residue: u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
            status: data[12],
        })
    }
}

/// UASP Command Information Unit (32 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UaspCommandIu {
    pub tag: u16,
    pub task_attr: u8,
    pub priority: u8,
    pub lun: [u8; 8],
    pub cdb: ScsiCdb,
}

impl UaspCommandIu {
    pub fn to_bytes(self) -> Result<[u8; 32], UsbClassError> {
        if self.cdb.len == 0 || self.cdb.len > 16 {
            return Err(UsbClassError::InvalidParameter);
        }
        let mut out = [0u8; 32];
        out[0] = UASP_IU_COMMAND;
        out[1] = 0;
        out[2..4].copy_from_slice(&self.tag.to_be_bytes());
        out[4] = ((self.priority & 0x0F) << 3) | (self.task_attr & 0x07);
        out[5] = 0;
        out[6] = 0; // Additional CDB length (0 for <= 16-byte CDB)
        out[7] = 0;
        out[8..16].copy_from_slice(&self.lun);
        out[16..32].copy_from_slice(&self.cdb.bytes);
        Ok(out)
    }
}

/// Parsed SCSI READ CAPACITY (10) response (8 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScsiReadCapacity10Data {
    pub last_lba: u32,
    pub block_size: u32,
}

impl ScsiReadCapacity10Data {
    pub fn from_bytes(data: &[u8]) -> Result<Self, UsbClassError> {
        if data.len() < 8 {
            return Err(UsbClassError::BufferTooSmall);
        }
        Ok(Self {
            last_lba: u32::from_be_bytes([data[0], data[1], data[2], data[3]]),
            block_size: u32::from_be_bytes([data[4], data[5], data[6], data[7]]),
        })
    }
}

// ---------------------------------------------------------------------------
// 5. Self-Test Suite
// ---------------------------------------------------------------------------

fn test_isoch_ring() -> Result<(), &'static str> {
    let ep = IsochEndpointConfig {
        endpoint_addr: 0x81,
        max_packet_size: 1024,
        max_burst: 2,
        mult: 1,
        interval: 1,
        usb4_tunneled: true,
    };
    if ep.max_esit_payload() != 6144 || ep.interval_us() != 125 {
        return Err("usb_class: ESIT payload or interval mismatch");
    }

    let mut ring = IsochTrbRing::new(8, 0x1000_0000)?;
    if ring.usable_slots() != 7 || ring.available_slots() != 7 {
        return Err("usb_class: unexpected ring capacity");
    }

    let params = IsochTrbParams {
        buf_phys: 0x2000_0000,
        len: 1024,
        frame_id: 42,
        sia: false,
        ioc: true,
        isp: true,
        chain: false,
        tbc: 2,
        tlbpc: 1,
        td_size: 0,
        intr_target: 0,
    };

    // Enqueue 7 TRBs to trigger Link TRB wrap-around and cycle toggle.
    for _ in 0..7 {
        ring.enqueue_isoch(&params)?;
    }
    if ring.cycle_state() {
        return Err("usb_class: cycle bit did not toggle on wrap");
    }
    let link = ring.trb_at(7).ok_or("usb_class: missing link TRB")?;
    if link.trb_type() != TRB_TYPE_LINK {
        return Err("usb_class: last slot is not Link TRB");
    }

    let c1 = ring.complete_one(XHCI_CC_SUCCESS, 0)?;
    let c2 = ring.complete_one(XHCI_CC_RING_UNDERRUN, 1024)?;
    if c1.underrun_or_missed || !c2.underrun_or_missed || ring.underrun_count != 1 {
        return Err("usb_class: completion tracking mismatch");
    }
    Ok(())
}

fn test_hid_parser() -> Result<(), &'static str> {
    // Minimal 3-button + X/Y relative mouse HID report descriptor
    let mouse_desc: [u8; 45] = [
        0x05, 0x01, // Usage Page (Generic Desktop)
        0x09, 0x02, // Usage (Mouse)
        0xA1, 0x01, // Collection (Application)
        0x05, 0x09, //   Usage Page (Button)
        0x19, 0x01, //   Usage Minimum (1)
        0x29, 0x03, //   Usage Maximum (3)
        0x15, 0x00, //   Logical Minimum (0)
        0x25, 0x01, //   Logical Maximum (1)
        0x95, 0x03, //   Report Count (3)
        0x75, 0x01, //   Report Size (1)
        0x81, 0x02, //   Input (Data, Var, Abs)
        0x95, 0x01, //   Report Count (1)
        0x75, 0x05, //   Report Size (5)
        0x81, 0x03, //   Input (Const, Var, Abs)
        0x05, 0x01, //   Usage Page (Generic Desktop)
        0x09, 0x30, //   Usage (X)
        0x09, 0x31, //   Usage (Y)
        0x15, 0x81, //   Logical Minimum (-127)
        0x25, 0x7F, //   Logical Maximum (127)
        0x75, 0x08, //   Report Size (8)
        0x95, 0x02, //   Report Count (2)
        0x81, 0x06, //   Input (Data, Var, Rel)
        0xC0, // End Collection
    ];

    let parsed = HidReportDescriptor::parse(&mouse_desc)?;
    if parsed.input_bit_len != 24 || parsed.fields.len() != 6 {
        return Err("usb_class: HID mouse descriptor bit length or field count mismatch");
    }

    let report = [0x05u8, 0xFEu8, 0x10u8]; // buttons 1 & 3, dx = -2, dy = +16
    let btn1 = HidReportDescriptor::extract_unsigned(&report, &parsed.fields[0])
        .ok_or("usb_class: failed to extract btn1")?;
    let dx = HidReportDescriptor::extract_signed(&report, &parsed.fields[4])
        .ok_or("usb_class: failed to extract dx")?;
    let dy = HidReportDescriptor::extract_signed(&report, &parsed.fields[5])
        .ok_or("usb_class: failed to extract dy")?;
    if btn1 != 1 || dx != -2 || dy != 16 {
        return Err("usb_class: HID field extraction mismatch");
    }

    let boot_kbd = parse_boot_keyboard(&[0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00])
        .ok_or("usb_class: boot keyboard parse failed")?;
    if boot_kbd.modifiers != 0x02 || boot_kbd.keycodes[0] != 0x04 {
        return Err("usb_class: boot keyboard values mismatch");
    }
    Ok(())
}

fn test_uac2_descriptors() -> Result<(), &'static str> {
    let stream: [u8; 47] = [
        // Clock Source (8 bytes)
        8,
        USB_DESC_CS_INTERFACE,
        UAC2_AC_CLOCK_SOURCE,
        0x10,
        0x01,
        0x07,
        0x00,
        0x00,
        // Input Terminal (17 bytes)
        17,
        USB_DESC_CS_INTERFACE,
        UAC2_AC_INPUT_TERMINAL,
        0x01,
        0x01,
        0x01,
        0x00,
        0x10,
        0x02,
        0x03,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        // AS General (16 bytes)
        16,
        USB_DESC_CS_INTERFACE,
        UAC2_AS_GENERAL,
        0x01,
        0x00,
        0x01,
        0x01,
        0x00,
        0x00,
        0x00,
        0x02,
        0x03,
        0x00,
        0x00,
        0x00,
        0x00,
        // Format Type I (6 bytes)
        6,
        USB_DESC_CS_INTERFACE,
        UAC2_AS_FORMAT_TYPE,
        0x01,
        0x04,
        0x18,
    ];

    let cfg = Uac2DeviceConfig::parse(&stream)?;
    if cfg.clock_sources.len() != 1 || cfg.terminals.len() != 1 {
        return Err("usb_class: UAC2 clock/terminal count mismatch");
    }
    let fmt = cfg
        .format_type_i
        .ok_or("usb_class: missing UAC2 format type I")?;
    if fmt.subslot_size != 4 || fmt.bit_resolution != 24 {
        return Err("usb_class: UAC2 format type I mismatch");
    }

    let (setup, payload) = build_uac2_set_sample_rate(0x10, 0, 96_000);
    if setup[0] != 0x21 || u32::from_le_bytes(payload) != 96_000 {
        return Err("usb_class: UAC2 sample rate setup mismatch");
    }
    Ok(())
}

fn test_mass_storage() -> Result<(), &'static str> {
    let cdb = ScsiCdb::read_10(0x1234_5678, 8);
    let cbw = CommandBlockWrapper {
        tag: 0xDEAD_BEEF,
        data_transfer_length: 4096,
        flags: CBW_FLAG_DATA_IN,
        lun: 0,
        cdb,
    };
    let cbw_bytes = cbw.to_bytes()?;
    if &cbw_bytes[0..4] != b"USBC" || cbw_bytes[15] != 0x28 {
        return Err("usb_class: BOT CBW encoding mismatch");
    }

    let mut csw_raw = [0u8; 13];
    csw_raw[0..4].copy_from_slice(&CSW_SIGNATURE.to_le_bytes());
    csw_raw[4..8].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    let csw = CommandStatusWrapper::from_bytes(&csw_raw)?;
    if csw.tag != 0xDEAD_BEEF || csw.status != 0 {
        return Err("usb_class: BOT CSW decode mismatch");
    }

    let uasp = UaspCommandIu {
        tag: 0x0042,
        task_attr: 0,
        priority: 0,
        lun: [0; 8],
        cdb: ScsiCdb::read_16(0x0000_0001_0000_0000, 16),
    };
    let iu_bytes = uasp.to_bytes()?;
    if iu_bytes[0] != UASP_IU_COMMAND || iu_bytes[16] != 0x88 {
        return Err("usb_class: UASP Command IU encoding mismatch");
    }

    let cap =
        ScsiReadCapacity10Data::from_bytes(&[0x00, 0x0F, 0xFF, 0xFF, 0x00, 0x00, 0x02, 0x00])?;
    if cap.last_lba != 0x000F_FFFF || cap.block_size != 512 {
        return Err("usb_class: SCSI ReadCapacity10 decode mismatch");
    }
    Ok(())
}

/// Run self-tests for xHCI Isochronous TRB ring, HID parser, UAC2 descriptors, and BOT/UASP SCSI.
pub fn self_test() -> Result<(), &'static str> {
    test_isoch_ring()?;
    test_hid_parser()?;
    test_uac2_descriptors()?;
    test_mass_storage()?;
    Ok(())
}
