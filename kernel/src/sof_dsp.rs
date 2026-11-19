//! Clean-room Intel HDA / SOF (Sound Open Firmware) DSP Pipeline & Audio DMA Engine.
//!
//! Provides `no_std` building blocks for:
//! - Intel HDA CORB/RIRB command/response ring buffers and verb encoding
//! - Intel HDA Buffer Descriptor List (BDL) & cyclic stream DMA engine
//! - SOF IPC4 protocol headers, message builders, and AudioDataFormat ABI encoding
//! - SOF IPC4 topology/pipeline graph builder with cycle detection and topological ordering
//! - Fixed-point Q15 DSP gain/mixer frame processing
#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;

use crate::audio::{AudioFormat, BitDepth, Channels, SampleRate};

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SofError {
    InvalidParameter,
    BufferTooSmall,
    RingFull,
    RingEmpty,
    InvalidAlignment,
    InvalidState,
    PipelineNotFound,
    ModuleNotFound,
    RouteExists,
    RouteNotFound,
    CycleDetected,
    IpcTimeout,
    IpcError(u32),
    UnsupportedFormat,
    DmaError,
}

impl SofError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidParameter => "sof_dsp: invalid parameter",
            Self::BufferTooSmall => "sof_dsp: buffer too small",
            Self::RingFull => "sof_dsp: ring buffer full",
            Self::RingEmpty => "sof_dsp: ring buffer empty",
            Self::InvalidAlignment => "sof_dsp: invalid DMA alignment",
            Self::InvalidState => "sof_dsp: invalid pipeline or stream state",
            Self::PipelineNotFound => "sof_dsp: pipeline not found",
            Self::ModuleNotFound => "sof_dsp: module instance not found",
            Self::RouteExists => "sof_dsp: module binding already exists",
            Self::RouteNotFound => "sof_dsp: module binding not found",
            Self::CycleDetected => "sof_dsp: cycle detected in DSP topology",
            Self::IpcTimeout => "sof_dsp: IPC4 mailbox timeout",
            Self::IpcError(_) => "sof_dsp: IPC4 firmware error",
            Self::UnsupportedFormat => "sof_dsp: unsupported audio format",
            Self::DmaError => "sof_dsp: HDA DMA engine error",
        }
    }
}

impl From<SofError> for &'static str {
    fn from(err: SofError) -> Self {
        err.as_str()
    }
}

// ---------------------------------------------------------------------------
// 1. Intel HDA / cAVS / ACE DSP Register Constants
// ---------------------------------------------------------------------------

pub mod dsp_regs {
    /// Audio DSP Control & Status (cAVS / ACE).
    pub const ADSPCS: usize = 0x04;
    /// Host-to-DSP IPC Target Doorbell / Initiator.
    pub const HIPCTDR: usize = 0x40;
    /// Host-to-DSP IPC Target Acknowledge.
    pub const HIPCTDA: usize = 0x44;
    /// Host-to-DSP IPC Target Data (extension).
    pub const HIPCTDD: usize = 0x48;
    /// DSP-to-Host IPC Initiator.
    pub const HIPCT: usize = 0x50;
    /// DSP-to-Host IPC Extension.
    pub const HIPCTE: usize = 0x54;
    /// DSP-to-Host IPC Control.
    pub const HIPCTL: usize = 0x58;

    /// SRAM Mailbox windows (relative to DSP BAR).
    pub const SRAM_OUTBOX_BASE: usize = 0x80000;
    pub const SRAM_INBOX_BASE: usize = 0x81000;
    pub const SRAM_DEBUG_BASE: usize = 0x82000;
    pub const SRAM_MAILBOX_SIZE: usize = 0x1000;

    /// Doorbell busy bit (bit 31).
    pub const IPC_BUSY: u32 = 1 << 31;
    /// Doorbell done/ack bit (bit 31).
    pub const IPC_DONE: u32 = 1 << 31;
}

pub mod sd_regs {
    /// Stream Descriptor Control (24-bit).
    pub const SD_CTL: usize = 0x00;
    /// Stream Descriptor Status (8-bit).
    pub const SD_STS: usize = 0x03;
    /// Link Position in Current Buffer (32-bit).
    pub const SD_LPIB: usize = 0x04;
    /// Cyclic Buffer Length (32-bit).
    pub const SD_CBL: usize = 0x08;
    /// Last Valid Index (16-bit).
    pub const SD_LVI: usize = 0x0C;
    /// FIFO Watermark (16-bit).
    pub const SD_FIFOW: usize = 0x0E;
    /// FIFO Size (16-bit).
    pub const SD_FIFOS: usize = 0x10;
    /// Stream Format (16-bit).
    pub const SD_FMT: usize = 0x12;
    /// Buffer Descriptor List Pointer Lower (32-bit).
    pub const SD_BDPL: usize = 0x18;
    /// Buffer Descriptor List Pointer Upper (32-bit).
    pub const SD_BDPU: usize = 0x1C;

    pub const SD_CTL_SRST: u32 = 1 << 0;
    pub const SD_CTL_RUN: u32 = 1 << 1;
    pub const SD_CTL_IOCE: u32 = 1 << 2;
    pub const SD_CTL_FEIE: u32 = 1 << 3;
    pub const SD_CTL_DEIE: u32 = 1 << 4;
    pub const SD_STS_BCIS: u8 = 1 << 2;
    pub const SD_STS_FIFOE: u8 = 1 << 3;
    pub const SD_STS_DESE: u8 = 1 << 4;
}

// ---------------------------------------------------------------------------
// 2. Intel HDA CORB / RIRB Ring Buffers
// ---------------------------------------------------------------------------

/// Encoded 32-bit Intel HDA CORB verb command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorbVerb {
    pub codec_addr: u8,
    pub nid: u16,
    pub verb: u32,
}

impl CorbVerb {
    #[inline]
    pub const fn new(codec_addr: u8, nid: u16, verb: u32) -> Self {
        Self {
            codec_addr: codec_addr & 0x0F,
            nid: nid & 0xFF,
            verb: verb & 0x000F_FFFF,
        }
    }

    #[inline]
    pub const fn encode(self) -> u32 {
        ((self.codec_addr as u32 & 0x0F) << 28)
            | ((self.nid as u32 & 0xFF) << 20)
            | (self.verb & 0x000F_FFFF)
    }

    #[inline]
    pub const fn decode(raw: u32) -> Self {
        Self {
            codec_addr: ((raw >> 28) & 0x0F) as u8,
            nid: ((raw >> 20) & 0xFF) as u16,
            verb: raw & 0x000F_FFFF,
        }
    }
}

/// 64-bit Intel HDA RIRB response entry.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RirbEntry {
    pub response: u32,
    pub resp_ex: u32,
}

impl RirbEntry {
    #[inline]
    pub const fn new(response: u32, codec_addr: u8, unsolicited: bool) -> Self {
        let ex = (codec_addr as u32 & 0x0F) | if unsolicited { 1 << 4 } else { 0 };
        Self {
            response,
            resp_ex: ex,
        }
    }

    #[inline]
    pub const fn codec_addr(self) -> u8 {
        (self.resp_ex & 0x0F) as u8
    }

    #[inline]
    pub const fn is_unsolicited(self) -> bool {
        (self.resp_ex & (1 << 4)) != 0
    }
}

/// Valid HDA CORB/RIRB ring sizes per Intel HDA specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingSize {
    Entries2 = 2,
    Entries16 = 16,
    Entries256 = 256,
}

impl RingSize {
    #[inline]
    pub const fn count(self) -> u16 {
        self as u16
    }

    #[inline]
    pub const fn size_code(self) -> u8 {
        match self {
            Self::Entries2 => 0x00,
            Self::Entries16 => 0x01,
            Self::Entries256 => 0x02,
        }
    }
}

/// Heap-backed Intel HDA Command Output Ring Buffer (CORB).
#[derive(Debug, Clone)]
pub struct CorbRing {
    entries: Vec<u32>,
    capacity: u16,
    wp: u16,
    rp: u16,
}

impl CorbRing {
    pub fn new(size: RingSize) -> Self {
        let cap = size.count();
        Self {
            entries: vec![0u32; cap as usize],
            capacity: cap,
            wp: 0,
            rp: 0,
        }
    }

    #[inline]
    pub const fn capacity(&self) -> u16 {
        self.capacity
    }

    #[inline]
    pub const fn write_ptr(&self) -> u16 {
        self.wp
    }

    #[inline]
    pub const fn read_ptr(&self) -> u16 {
        self.rp
    }

    #[inline]
    pub fn pending_count(&self) -> u16 {
        if self.wp >= self.rp {
            self.wp - self.rp
        } else {
            self.capacity - self.rp + self.wp
        }
    }

    pub fn reset(&mut self) {
        self.entries.fill(0);
        self.wp = 0;
        self.rp = 0;
    }

    /// Enqueue a verb into the CORB ring (advances WP).
    pub fn push_verb(&mut self, verb: CorbVerb) -> Result<u16, SofError> {
        let next_wp = (self.wp + 1) % self.capacity;
        if next_wp == self.rp {
            return Err(SofError::RingFull);
        }
        self.entries[next_wp as usize] = verb.encode();
        self.wp = next_wp;
        Ok(self.wp)
    }

    /// Consume the next verb as the HDA controller hardware would (advances RP).
    pub fn pop_for_hw(&mut self) -> Result<CorbVerb, SofError> {
        if self.rp == self.wp {
            return Err(SofError::RingEmpty);
        }
        let next_rp = (self.rp + 1) % self.capacity;
        let raw = self.entries[next_rp as usize];
        self.rp = next_rp;
        Ok(CorbVerb::decode(raw))
    }
}

/// Heap-backed Intel HDA Response Input Ring Buffer (RIRB).
#[derive(Debug, Clone)]
pub struct RirbRing {
    entries: Vec<RirbEntry>,
    capacity: u16,
    wp: u16,
    rp: u16,
}

impl RirbRing {
    pub fn new(size: RingSize) -> Self {
        let cap = size.count();
        Self {
            entries: vec![RirbEntry::default(); cap as usize],
            capacity: cap,
            wp: 0,
            rp: 0,
        }
    }

    #[inline]
    pub const fn capacity(&self) -> u16 {
        self.capacity
    }

    #[inline]
    pub const fn write_ptr(&self) -> u16 {
        self.wp
    }

    #[inline]
    pub const fn read_ptr(&self) -> u16 {
        self.rp
    }

    #[inline]
    pub fn pending_count(&self) -> u16 {
        if self.wp >= self.rp {
            self.wp - self.rp
        } else {
            self.capacity - self.rp + self.wp
        }
    }

    pub fn reset(&mut self) {
        self.entries.fill(RirbEntry::default());
        self.wp = 0;
        self.rp = 0;
    }

    /// Push a response entry as the HDA controller hardware would (advances WP).
    pub fn push_from_hw(&mut self, entry: RirbEntry) -> Result<u16, SofError> {
        let next_wp = (self.wp + 1) % self.capacity;
        if next_wp == self.rp {
            return Err(SofError::RingFull);
        }
        self.entries[next_wp as usize] = entry;
        self.wp = next_wp;
        Ok(self.wp)
    }

    /// Pop the next response entry for the driver (advances RP).
    pub fn pop_response(&mut self) -> Result<RirbEntry, SofError> {
        if self.rp == self.wp {
            return Err(SofError::RingEmpty);
        }
        let next_rp = (self.rp + 1) % self.capacity;
        let entry = self.entries[next_rp as usize];
        self.rp = next_rp;
        Ok(entry)
    }
}

// ---------------------------------------------------------------------------
// 3. Intel HDA Buffer Descriptor List (BDL) & Stream DMA Engine
// ---------------------------------------------------------------------------

/// Required physical address alignment for HDA BDL buffers (128 bytes).
pub const HDA_BDL_ALIGN: u64 = 128;
/// Maximum number of entries in an HDA BDL table.
pub const HDA_BDL_MAX_ENTRIES: usize = 256;

/// Single 16-byte Intel HDA Buffer Descriptor List entry.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BdlEntry {
    pub addr: u64,
    pub length: u32,
    pub ioc: u32,
}

impl BdlEntry {
    pub fn new(
        phys_addr: u64,
        length: u32,
        interrupt_on_completion: bool,
    ) -> Result<Self, SofError> {
        if phys_addr == 0 || (phys_addr & (HDA_BDL_ALIGN - 1)) != 0 {
            return Err(SofError::InvalidAlignment);
        }
        if length == 0 {
            return Err(SofError::InvalidParameter);
        }
        Ok(Self {
            addr: phys_addr,
            length,
            ioc: if interrupt_on_completion { 1 } else { 0 },
        })
    }

    #[inline]
    pub const fn has_ioc(self) -> bool {
        (self.ioc & 1) != 0
    }
}

/// Intel HDA Buffer Descriptor List (2..=256 entries).
#[derive(Debug, Clone)]
pub struct BdlTable {
    entries: Vec<BdlEntry>,
    total_bytes: u32,
}

impl BdlTable {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_bytes: 0,
        }
    }

    pub fn add_entry(&mut self, phys_addr: u64, length: u32, ioc: bool) -> Result<usize, SofError> {
        if self.entries.len() >= HDA_BDL_MAX_ENTRIES {
            return Err(SofError::RingFull);
        }
        let entry = BdlEntry::new(phys_addr, length, ioc)?;
        let new_total = self
            .total_bytes
            .checked_add(length)
            .ok_or(SofError::InvalidParameter)?;
        self.entries.push(entry);
        self.total_bytes = new_total;
        Ok(self.entries.len() - 1)
    }

    /// Build a uniform cyclic BDL from a contiguous physical buffer split into `periods`.
    pub fn from_contiguous(
        base_phys: u64,
        period_bytes: u32,
        periods: usize,
        ioc_every_period: bool,
    ) -> Result<Self, SofError> {
        if !(2..=HDA_BDL_MAX_ENTRIES).contains(&periods) {
            return Err(SofError::InvalidParameter);
        }
        if period_bytes == 0 || (period_bytes as u64 & (HDA_BDL_ALIGN - 1)) != 0 {
            return Err(SofError::InvalidAlignment);
        }
        let mut table = Self::new();
        for i in 0..periods {
            let addr = base_phys
                .checked_add((i as u64) * (period_bytes as u64))
                .ok_or(SofError::InvalidParameter)?;
            table.add_entry(addr, period_bytes, ioc_every_period)?;
        }
        Ok(table)
    }

    pub fn validate(&self) -> Result<(), SofError> {
        if self.entries.len() < 2 || self.entries.len() > HDA_BDL_MAX_ENTRIES {
            return Err(SofError::DmaError);
        }
        if self.total_bytes == 0 {
            return Err(SofError::DmaError);
        }
        Ok(())
    }

    #[inline]
    pub fn entries(&self) -> &[BdlEntry] {
        &self.entries
    }

    #[inline]
    pub const fn total_bytes(&self) -> u32 {
        self.total_bytes
    }

    #[inline]
    pub fn lvi(&self) -> u16 {
        self.entries.len().saturating_sub(1) as u16
    }

    pub fn ioc_count(&self) -> usize {
        self.entries.iter().filter(|e| e.has_ioc()).count()
    }
}

impl Default for BdlTable {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamDirection {
    Playback,
    Capture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    Stopped,
    Prepared,
    Running,
    Paused,
    Xrun,
}

/// Intel HDA Stream DMA Engine state machine.
#[derive(Debug, Clone)]
pub struct HdaStreamDma {
    pub stream_id: u8,
    pub direction: StreamDirection,
    pub format: AudioFormat,
    pub bdl: BdlTable,
    pub state: StreamState,
    pub lpib: u32,
    pub appl_ptr: u32,
    pub completed_periods: u64,
    pub ioc_events: u64,
}

impl HdaStreamDma {
    pub fn new(
        stream_id: u8,
        direction: StreamDirection,
        format: AudioFormat,
        bdl: BdlTable,
    ) -> Result<Self, SofError> {
        if stream_id == 0 || stream_id > 15 {
            return Err(SofError::InvalidParameter);
        }
        bdl.validate()?;
        Ok(Self {
            stream_id,
            direction,
            format,
            bdl,
            state: StreamState::Stopped,
            lpib: 0,
            appl_ptr: 0,
            completed_periods: 0,
            ioc_events: 0,
        })
    }

    pub fn prepare(&mut self) -> Result<(), SofError> {
        self.bdl.validate()?;
        self.lpib = 0;
        self.appl_ptr = 0;
        self.completed_periods = 0;
        self.ioc_events = 0;
        self.state = StreamState::Prepared;
        Ok(())
    }

    pub fn start(&mut self) -> Result<(), SofError> {
        match self.state {
            StreamState::Prepared | StreamState::Paused => {
                self.state = StreamState::Running;
                Ok(())
            }
            _ => Err(SofError::InvalidState),
        }
    }

    pub fn pause(&mut self) -> Result<(), SofError> {
        if self.state != StreamState::Running {
            return Err(SofError::InvalidState);
        }
        self.state = StreamState::Paused;
        Ok(())
    }

    pub fn stop(&mut self) {
        self.state = StreamState::Stopped;
        self.lpib = 0;
    }

    /// Advance the hardware Link Position in Buffer (LPIB) by `bytes` and count
    /// crossed BDL period boundaries and IOC interrupts.
    pub fn advance_lpib(&mut self, bytes: u32) -> Result<u64, SofError> {
        if self.state != StreamState::Running {
            return Err(SofError::InvalidState);
        }
        let cbl = self.bdl.total_bytes();
        if cbl == 0 {
            return Err(SofError::DmaError);
        }

        let mut remaining = bytes;
        let mut new_iocs = 0u64;

        while remaining > 0 {
            let (entry_idx, offset_in_entry) = self.locate_offset(self.lpib);
            let entry = self.bdl.entries()[entry_idx];
            let to_end = entry.length - offset_in_entry;
            if remaining >= to_end {
                remaining -= to_end;
                self.lpib = (self.lpib + to_end) % cbl;
                self.completed_periods += 1;
                if entry.has_ioc() {
                    self.ioc_events += 1;
                    new_iocs += 1;
                }
            } else {
                self.lpib = (self.lpib + remaining) % cbl;
                remaining = 0;
            }
        }

        Ok(new_iocs)
    }

    fn locate_offset(&self, pos: u32) -> (usize, u32) {
        let mut acc = 0u32;
        for (idx, entry) in self.bdl.entries().iter().enumerate() {
            if pos < acc + entry.length {
                return (idx, pos - acc);
            }
            acc += entry.length;
        }
        (0, 0)
    }

    #[inline]
    pub fn hda_format_reg(&self) -> u16 {
        self.format.to_hda_format()
    }

    #[inline]
    pub fn cbl(&self) -> u32 {
        self.bdl.total_bytes()
    }

    #[inline]
    pub fn lvi(&self) -> u16 {
        self.bdl.lvi()
    }
}

// ---------------------------------------------------------------------------
// 4. SOF IPC4 Protocol & AudioDataFormat ABI
// ---------------------------------------------------------------------------

/// IPC4 message target (Global vs Module).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Ipc4MsgTarget {
    FwGenMsg = 0,
    ModuleMsg = 1,
}

/// IPC4 message direction (Request vs Reply/Notification).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Ipc4MsgDir {
    Request = 0,
    Reply = 1,
}

/// IPC4 Global message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Ipc4GlbType {
    BootConfig = 0,
    RomControl = 1,
    IpcGtwCmd = 2,
    PerfMeasurementsCmd = 13,
    ChainDma = 14,
    LoadLibrary = 16,
    CreatePipeline = 17,
    DeletePipeline = 18,
    SetPipelineState = 19,
    GetPipelineState = 20,
    Notification = 27,
}

/// IPC4 Module message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Ipc4ModType {
    InitInstance = 0,
    ConfigGet = 1,
    ConfigSet = 2,
    LargeConfigGet = 3,
    LargeConfigSet = 4,
    Bind = 5,
    Unbind = 6,
    SetDx = 7,
    SetD0ix = 8,
    DeleteInstance = 11,
}

/// IPC4 Pipeline states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PipelineState {
    Invalid = 0,
    Uninitialized = 1,
    Reset = 2,
    Paused = 3,
    Running = 4,
    Eos = 5,
}

/// IPC4 Reply status codes.
pub const IPC4_SUCCESS: u32 = 0;
pub const IPC4_INVALID_REQUEST: u32 = 4;
pub const IPC4_OUT_OF_MEMORY: u32 = 5;
pub const IPC4_BUSY: u32 = 6;
pub const IPC4_FAILURE: u32 = 11;

/// 64-bit SOF IPC4 header (`primary` + `extension`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ipc4Header {
    pub primary: u32,
    pub extension: u32,
}

impl Ipc4Header {
    /// Build a Global `CREATE_PIPELINE` header.
    pub const fn create_pipeline(
        instance_id: u8,
        ppl_mem_size_pages: u16,
        ppl_priority: u8,
        lp: bool,
        core_id: u8,
    ) -> Self {
        let primary = (ppl_mem_size_pages as u32 & 0x07FF)
            | ((ppl_priority as u32 & 0x1F) << 11)
            | ((instance_id as u32) << 16)
            | ((Ipc4GlbType::CreatePipeline as u32) << 24)
            | ((Ipc4MsgDir::Request as u32) << 29)
            | ((Ipc4MsgTarget::FwGenMsg as u32) << 30);
        let extension = (if lp { 1u32 } else { 0u32 }) | ((core_id as u32 & 0x0F) << 1);
        Self { primary, extension }
    }

    /// Build a Global `DELETE_PIPELINE` header.
    pub const fn delete_pipeline(instance_id: u8) -> Self {
        let primary = ((instance_id as u32) << 16)
            | ((Ipc4GlbType::DeletePipeline as u32) << 24)
            | ((Ipc4MsgDir::Request as u32) << 29)
            | ((Ipc4MsgTarget::FwGenMsg as u32) << 30);
        Self {
            primary,
            extension: 0,
        }
    }

    /// Build a Global `SET_PIPELINE_STATE` header.
    pub const fn set_pipeline_state(instance_id: u8, state: PipelineState) -> Self {
        let primary = (state as u32 & 0xFFFF)
            | ((instance_id as u32) << 16)
            | ((Ipc4GlbType::SetPipelineState as u32) << 24)
            | ((Ipc4MsgDir::Request as u32) << 29)
            | ((Ipc4MsgTarget::FwGenMsg as u32) << 30);
        Self {
            primary,
            extension: 0,
        }
    }

    /// Build a Module `INIT_INSTANCE` header.
    pub const fn init_module(
        module_id: u16,
        instance_id: u8,
        param_block_size_dwords: u16,
        ppl_instance_id: u8,
        core_id: u8,
    ) -> Self {
        let primary = (module_id as u32)
            | ((instance_id as u32) << 16)
            | ((Ipc4ModType::InitInstance as u32) << 24)
            | ((Ipc4MsgDir::Request as u32) << 29)
            | ((Ipc4MsgTarget::ModuleMsg as u32) << 30);
        let extension = (param_block_size_dwords as u32)
            | ((ppl_instance_id as u32) << 16)
            | ((core_id as u32 & 0x0F) << 24);
        Self { primary, extension }
    }

    /// Build a Module `DELETE_INSTANCE` header.
    pub const fn delete_module(module_id: u16, instance_id: u8) -> Self {
        let primary = (module_id as u32)
            | ((instance_id as u32) << 16)
            | ((Ipc4ModType::DeleteInstance as u32) << 24)
            | ((Ipc4MsgDir::Request as u32) << 29)
            | ((Ipc4MsgTarget::ModuleMsg as u32) << 30);
        Self {
            primary,
            extension: 0,
        }
    }

    /// Build a Module `BIND` or `UNBIND` header.
    pub const fn bind_modules(
        bind: bool,
        src_module_id: u16,
        src_instance_id: u8,
        dst_module_id: u16,
        dst_instance_id: u8,
        src_queue: u8,
        dst_queue: u8,
    ) -> Self {
        let msg_type = if bind {
            Ipc4ModType::Bind as u32
        } else {
            Ipc4ModType::Unbind as u32
        };
        let primary = (src_module_id as u32)
            | ((src_instance_id as u32) << 16)
            | (msg_type << 24)
            | ((Ipc4MsgDir::Request as u32) << 29)
            | ((Ipc4MsgTarget::ModuleMsg as u32) << 30);
        let extension = (dst_module_id as u32)
            | ((dst_instance_id as u32) << 16)
            | ((dst_queue as u32 & 0x07) << 24)
            | ((src_queue as u32 & 0x07) << 27);
        Self { primary, extension }
    }

    /// Build a Module `LARGE_CONFIG_SET` header.
    pub const fn large_config_set(
        module_id: u16,
        instance_id: u8,
        large_param_id: u8,
        data_off_size: u32,
        init_block: bool,
        final_block: bool,
    ) -> Self {
        let primary = (module_id as u32)
            | ((instance_id as u32) << 16)
            | ((Ipc4ModType::LargeConfigSet as u32) << 24)
            | ((Ipc4MsgDir::Request as u32) << 29)
            | ((Ipc4MsgTarget::ModuleMsg as u32) << 30);
        let extension = (data_off_size & 0x000F_FFFF)
            | ((large_param_id as u32) << 20)
            | (if final_block { 1 << 28 } else { 0 })
            | (if init_block { 1 << 29 } else { 0 });
        Self { primary, extension }
    }

    #[inline]
    pub const fn target(self) -> Ipc4MsgTarget {
        if ((self.primary >> 30) & 1) == 0 {
            Ipc4MsgTarget::FwGenMsg
        } else {
            Ipc4MsgTarget::ModuleMsg
        }
    }

    #[inline]
    pub const fn direction(self) -> Ipc4MsgDir {
        if ((self.primary >> 29) & 1) == 0 {
            Ipc4MsgDir::Request
        } else {
            Ipc4MsgDir::Reply
        }
    }

    #[inline]
    pub const fn msg_type(self) -> u8 {
        ((self.primary >> 24) & 0x1F) as u8
    }

    #[inline]
    pub const fn reply_status(self) -> u32 {
        self.primary & 0x00FF_FFFF
    }

    #[inline]
    pub const fn make_reply(self, status: u32) -> Self {
        let primary = (self.primary & 0xFF00_0000)
            | ((Ipc4MsgDir::Reply as u32) << 29)
            | (status & 0x00FF_FFFF);
        Self {
            primary,
            extension: self.extension,
        }
    }
}

/// Complete IPC4 message (header + optional payload bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ipc4Message {
    pub header: Ipc4Header,
    pub payload: Vec<u8>,
}

impl Ipc4Message {
    pub fn new(header: Ipc4Header, payload: Vec<u8>) -> Self {
        Self { header, payload }
    }

    pub fn empty(header: Ipc4Header) -> Self {
        Self {
            header,
            payload: Vec::new(),
        }
    }

    /// Serialize header (8 bytes LE) + payload into a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.payload.len());
        out.extend_from_slice(&self.header.primary.to_le_bytes());
        out.extend_from_slice(&self.header.extension.to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    /// Deserialize from a byte slice (minimum 8 bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SofError> {
        if bytes.len() < 8 {
            return Err(SofError::BufferTooSmall);
        }
        let primary = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let extension = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        Ok(Self {
            header: Ipc4Header { primary, extension },
            payload: bytes[8..].to_vec(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InterleavingStyle {
    PerChannel = 0,
    PerSample = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SampleType {
    IntMsb = 0,
    IntLsb = 1,
    Float = 2,
}

/// SOF IPC4 `AudioDataFormat` descriptor (24 bytes / 6 dwords in ABI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioDataFormat {
    pub sampling_frequency: u32,
    pub bit_depth: u8,
    pub valid_bit_depth: u8,
    pub channels: u8,
    pub channel_map: u32,
    pub interleaving_style: InterleavingStyle,
    pub sample_type: SampleType,
}

impl AudioDataFormat {
    pub const fn pcm_48k_stereo_16() -> Self {
        Self {
            sampling_frequency: 48_000,
            bit_depth: 16,
            valid_bit_depth: 16,
            channels: 2,
            channel_map: 0xFFFF_FF10,
            interleaving_style: InterleavingStyle::PerSample,
            sample_type: SampleType::IntLsb,
        }
    }

    pub const fn pcm_48k_stereo_32() -> Self {
        Self {
            sampling_frequency: 48_000,
            bit_depth: 32,
            valid_bit_depth: 24,
            channels: 2,
            channel_map: 0xFFFF_FF10,
            interleaving_style: InterleavingStyle::PerSample,
            sample_type: SampleType::IntLsb,
        }
    }

    pub fn from_audio_format(fmt: AudioFormat) -> Self {
        let sampling_frequency = match fmt.sample_rate {
            SampleRate::Hz44100 => 44_100,
            SampleRate::Hz48000 => 48_000,
            SampleRate::Hz88200 => 88_200,
            SampleRate::Hz96000 => 96_000,
            SampleRate::Hz176400 => 176_400,
            SampleRate::Hz192000 => 192_000,
        };
        let (bit_depth, valid_bit_depth) = match fmt.bit_depth {
            BitDepth::Bits8 => (8, 8),
            BitDepth::Bits16 => (16, 16),
            BitDepth::Bits20 => (32, 20),
            BitDepth::Bits24 => (32, 24),
            BitDepth::Bits32 => (32, 32),
        };
        let channels = fmt.channels.count();
        let channel_map = match fmt.channels {
            Channels::Mono => 0xFFFF_FFF0,
            Channels::Stereo => 0xFFFF_FF10,
            Channels::Surround51 => 0xFF54_3210,
            Channels::Surround71 => 0x7654_3210,
        };
        Self {
            sampling_frequency,
            bit_depth,
            valid_bit_depth,
            channels,
            channel_map,
            interleaving_style: InterleavingStyle::PerSample,
            sample_type: SampleType::IntLsb,
        }
    }

    pub fn validate(&self) -> Result<(), SofError> {
        if self.sampling_frequency == 0 || self.channels == 0 || self.channels > 8 {
            return Err(SofError::UnsupportedFormat);
        }
        if !matches!(self.bit_depth, 8 | 16 | 24 | 32) {
            return Err(SofError::UnsupportedFormat);
        }
        if self.valid_bit_depth == 0 || self.valid_bit_depth > self.bit_depth {
            return Err(SofError::UnsupportedFormat);
        }
        Ok(())
    }

    #[inline]
    pub const fn frame_size_bytes(&self) -> u32 {
        (self.channels as u32) * ((self.bit_depth as u32) / 8)
    }

    #[inline]
    pub const fn bytes_per_second(&self) -> u32 {
        self.sampling_frequency * self.frame_size_bytes()
    }

    /// Encode into the 6-dword (24-byte) SOF IPC4 `AudioDataFormat` ABI structure.
    pub const fn to_ipc4_dwords(self) -> [u32; 6] {
        let w1 = (self.bit_depth as u32)
            | ((self.valid_bit_depth as u32) << 8)
            | ((self.channels as u32) << 16)
            | ((self.sample_type as u32) << 24);
        let w2 = self.channel_map;
        let w3 = self.interleaving_style as u32;
        [self.sampling_frequency, w1, w2, w3, 0, 0]
    }
}

// ---------------------------------------------------------------------------
// 5. SOF IPC4 Topology & Pipeline Builder
// ---------------------------------------------------------------------------

/// Standard SOF IPC4 DSP module types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleType {
    HostCopier,
    DaiCopier,
    Gain,
    Mixer,
    Src,
    EqIir,
    EqFir,
    Drc,
    Kpb,
    SmartAmp,
}

impl ModuleType {
    /// Canonical SOF IPC4 module IDs.
    pub const fn default_module_id(self) -> u16 {
        match self {
            Self::HostCopier => 0x0001,
            Self::DaiCopier => 0x0002,
            Self::Gain => 0x0006,
            Self::Mixer => 0x0007,
            Self::Src => 0x0008,
            Self::EqIir => 0x0009,
            Self::EqFir => 0x000A,
            Self::Drc => 0x000B,
            Self::Kpb => 0x000C,
            Self::SmartAmp => 0x000D,
        }
    }
}

/// Unique key identifying a module instance in the DSP topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModuleKey {
    pub module_id: u16,
    pub instance_id: u8,
}

impl ModuleKey {
    #[inline]
    pub const fn new(module_id: u16, instance_id: u8) -> Self {
        Self {
            module_id,
            instance_id,
        }
    }
}

/// Directed audio route between two module pins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuleBinding {
    pub src: ModuleKey,
    pub src_pin: u8,
    pub dst: ModuleKey,
    pub dst_pin: u8,
}

/// DSP module instance configuration.
#[derive(Debug, Clone)]
pub struct ModuleInstance {
    pub key: ModuleKey,
    pub module_type: ModuleType,
    pub pipeline_id: u8,
    pub core_id: u8,
    pub input_format: AudioDataFormat,
    pub output_format: AudioDataFormat,
    pub ibs_bytes: u32,
    pub obs_bytes: u32,
    pub cpc: u32,
    pub is_pages: u32,
    /// Gain in Q16.16 linear multiplier (0x0001_0000 = 1.0 / 0 dB).
    pub gain_q16: i32,
}

impl ModuleInstance {
    pub fn new(
        module_type: ModuleType,
        instance_id: u8,
        pipeline_id: u8,
        format: AudioDataFormat,
    ) -> Result<Self, SofError> {
        format.validate()?;
        // 1 ms chunk size aligned to frame size
        let chunk_bytes = (format.bytes_per_second() / 1000).max(format.frame_size_bytes());
        Ok(Self {
            key: ModuleKey::new(module_type.default_module_id(), instance_id),
            module_type,
            pipeline_id,
            core_id: 0,
            input_format: format,
            output_format: format,
            ibs_bytes: chunk_bytes,
            obs_bytes: chunk_bytes,
            cpc: 100_000,
            is_pages: 1,
            gain_q16: 0x0001_0000,
        })
    }

    /// Build the IPC4 `INIT_INSTANCE` message with BaseModuleCfg + AudioDataFormat payload.
    pub fn build_init_ipc4(&self) -> Ipc4Message {
        let mut payload = Vec::with_capacity(40);
        // BaseModuleCfg: cpc, is_pages, ibs, obs (4 dwords)
        payload.extend_from_slice(&self.cpc.to_le_bytes());
        payload.extend_from_slice(&self.is_pages.to_le_bytes());
        payload.extend_from_slice(&self.ibs_bytes.to_le_bytes());
        payload.extend_from_slice(&self.obs_bytes.to_le_bytes());
        // Input AudioDataFormat (6 dwords)
        for dw in self.input_format.to_ipc4_dwords() {
            payload.extend_from_slice(&dw.to_le_bytes());
        }
        let dwords = (payload.len() / 4) as u16;
        let header = Ipc4Header::init_module(
            self.key.module_id,
            self.key.instance_id,
            dwords,
            self.pipeline_id,
            self.core_id,
        );
        Ipc4Message::new(header, payload)
    }
}

/// SOF DSP Pipeline descriptor.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub id: u8,
    pub priority: u8,
    pub core_id: u8,
    pub lp_mode: bool,
    pub mem_pages: u16,
    pub state: PipelineState,
}

impl Pipeline {
    pub const fn new(id: u8, priority: u8, mem_pages: u16) -> Self {
        Self {
            id,
            priority: priority & 0x1F,
            core_id: 0,
            lp_mode: false,
            mem_pages: mem_pages & 0x07FF,
            state: PipelineState::Uninitialized,
        }
    }

    pub fn build_create_ipc4(&self) -> Ipc4Message {
        Ipc4Message::empty(Ipc4Header::create_pipeline(
            self.id,
            self.mem_pages,
            self.priority,
            self.lp_mode,
            self.core_id,
        ))
    }

    pub fn build_set_state_ipc4(&self, target: PipelineState) -> Ipc4Message {
        Ipc4Message::empty(Ipc4Header::set_pipeline_state(self.id, target))
    }
}

/// Complete SOF IPC4 DSP Topology & Pipeline Builder.
#[derive(Debug, Clone)]
pub struct SofTopology {
    pipelines: Vec<Pipeline>,
    modules: Vec<ModuleInstance>,
    bindings: Vec<ModuleBinding>,
}

impl SofTopology {
    pub fn new() -> Self {
        Self {
            pipelines: Vec::new(),
            modules: Vec::new(),
            bindings: Vec::new(),
        }
    }

    pub fn add_pipeline(&mut self, pipeline: Pipeline) -> Result<(), SofError> {
        if self.pipelines.iter().any(|p| p.id == pipeline.id) {
            return Err(SofError::InvalidParameter);
        }
        self.pipelines.push(pipeline);
        Ok(())
    }

    pub fn add_module(&mut self, module: ModuleInstance) -> Result<(), SofError> {
        if !self.pipelines.iter().any(|p| p.id == module.pipeline_id) {
            return Err(SofError::PipelineNotFound);
        }
        if self.modules.iter().any(|m| m.key == module.key) {
            return Err(SofError::InvalidParameter);
        }
        self.modules.push(module);
        Ok(())
    }

    pub fn bind(
        &mut self,
        src: ModuleKey,
        src_pin: u8,
        dst: ModuleKey,
        dst_pin: u8,
    ) -> Result<(), SofError> {
        if src == dst {
            return Err(SofError::CycleDetected);
        }
        if !self.modules.iter().any(|m| m.key == src) || !self.modules.iter().any(|m| m.key == dst)
        {
            return Err(SofError::ModuleNotFound);
        }
        let binding = ModuleBinding {
            src,
            src_pin,
            dst,
            dst_pin,
        };
        if self.bindings.contains(&binding) {
            return Err(SofError::RouteExists);
        }
        self.bindings.push(binding);
        if self.topological_order().is_err() {
            self.bindings.pop();
            return Err(SofError::CycleDetected);
        }
        Ok(())
    }

    pub fn unbind(
        &mut self,
        src: ModuleKey,
        src_pin: u8,
        dst: ModuleKey,
        dst_pin: u8,
    ) -> Result<(), SofError> {
        let pos = self
            .bindings
            .iter()
            .position(|b| {
                b.src == src && b.src_pin == src_pin && b.dst == dst && b.dst_pin == dst_pin
            })
            .ok_or(SofError::RouteNotFound)?;
        self.bindings.remove(pos);
        Ok(())
    }

    /// Compute Kahn's topological order of all module instances; returns `CycleDetected` on cycles.
    pub fn topological_order(&self) -> Result<Vec<ModuleKey>, SofError> {
        let n = self.modules.len();
        let mut in_degree = vec![0usize; n];
        for b in &self.bindings {
            if let Some(dst_idx) = self.modules.iter().position(|m| m.key == b.dst) {
                in_degree[dst_idx] += 1;
            }
        }

        let mut queue = Vec::new();
        for (idx, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push(idx);
            }
        }

        let mut order = Vec::with_capacity(n);
        let mut head = 0usize;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            let u_key = self.modules[u].key;
            order.push(u_key);

            for b in &self.bindings {
                if b.src == u_key {
                    if let Some(v) = self.modules.iter().position(|m| m.key == b.dst) {
                        in_degree[v] -= 1;
                        if in_degree[v] == 0 {
                            queue.push(v);
                        }
                    }
                }
            }
        }

        if order.len() != n {
            return Err(SofError::CycleDetected);
        }
        Ok(order)
    }

    /// Compile the full IPC4 bring-up sequence:
    /// `CREATE_PIPELINE` -> `INIT_INSTANCE` (topological) -> `BIND` -> `SET_PIPELINE_STATE(Paused -> Running)`.
    pub fn compile_ipc4_sequence(&self) -> Result<Vec<Ipc4Message>, SofError> {
        let order = self.topological_order()?;
        let mut msgs =
            Vec::with_capacity(self.pipelines.len() * 3 + self.modules.len() + self.bindings.len());

        for ppl in &self.pipelines {
            msgs.push(ppl.build_create_ipc4());
        }

        for key in &order {
            if let Some(m) = self.modules.iter().find(|m| m.key == *key) {
                msgs.push(m.build_init_ipc4());
            }
        }

        for b in &self.bindings {
            let hdr = Ipc4Header::bind_modules(
                true,
                b.src.module_id,
                b.src.instance_id,
                b.dst.module_id,
                b.dst.instance_id,
                b.src_pin,
                b.dst_pin,
            );
            msgs.push(Ipc4Message::empty(hdr));
        }

        for ppl in &self.pipelines {
            msgs.push(ppl.build_set_state_ipc4(PipelineState::Paused));
            msgs.push(ppl.build_set_state_ipc4(PipelineState::Running));
        }

        Ok(msgs)
    }

    /// Compile the IPC4 teardown sequence in reverse order.
    pub fn teardown_ipc4_sequence(&self) -> Vec<Ipc4Message> {
        let mut msgs =
            Vec::with_capacity(self.pipelines.len() * 2 + self.modules.len() + self.bindings.len());

        for ppl in self.pipelines.iter().rev() {
            msgs.push(ppl.build_set_state_ipc4(PipelineState::Reset));
        }

        for b in self.bindings.iter().rev() {
            let hdr = Ipc4Header::bind_modules(
                false,
                b.src.module_id,
                b.src.instance_id,
                b.dst.module_id,
                b.dst.instance_id,
                b.src_pin,
                b.dst_pin,
            );
            msgs.push(Ipc4Message::empty(hdr));
        }

        for m in self.modules.iter().rev() {
            msgs.push(Ipc4Message::empty(Ipc4Header::delete_module(
                m.key.module_id,
                m.key.instance_id,
            )));
        }

        for ppl in self.pipelines.iter().rev() {
            msgs.push(Ipc4Message::empty(Ipc4Header::delete_pipeline(ppl.id)));
        }

        msgs
    }

    #[inline]
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    #[inline]
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    #[inline]
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }
}

impl Default for SofTopology {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 6. Fixed-Point Q15 DSP Frame Processor (Gain & Mixer)
// ---------------------------------------------------------------------------

/// Apply Q16.16 linear gain to a slice of signed 16-bit PCM samples with saturation.
pub fn dsp_apply_gain_i16(samples: &mut [i16], gain_q16: i32) {
    for s in samples.iter_mut() {
        let scaled = ((*s as i64) * (gain_q16 as i64)) >> 16;
        *s = scaled.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
    }
}

/// Mix `src` into `dst` (16-bit PCM) with saturating addition.
pub fn dsp_mix_i16(dst: &mut [i16], src: &[i16]) -> Result<(), SofError> {
    if dst.len() != src.len() {
        return Err(SofError::BufferTooSmall);
    }
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = d.saturating_add(s);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 7. Self-Test Suite
// ---------------------------------------------------------------------------

pub fn self_test() -> Result<(), &'static str> {
    // 1. HDA CORB / RIRB ring buffer test
    let mut corb = CorbRing::new(RingSize::Entries16);
    let mut rirb = RirbRing::new(RingSize::Entries16);

    let v1 = CorbVerb::new(0, 0x01, 0xF0000); // Get Parameter: Vendor ID
    let v2 = CorbVerb::new(2, 0x05, 0x70500); // Set Power State D0
    corb.push_verb(v1)?;
    corb.push_verb(v2)?;
    if corb.pending_count() != 2 {
        return Err("sof_dsp: CORB pending count mismatch");
    }
    let popped1 = corb.pop_for_hw()?;
    let popped2 = corb.pop_for_hw()?;
    if popped1 != v1 || popped2 != v2 {
        return Err("sof_dsp: CORB verb round-trip mismatch");
    }

    rirb.push_from_hw(RirbEntry::new(0x10EC_0295, 0, false))?;
    rirb.push_from_hw(RirbEntry::new(0x0000_0001, 2, true))?;
    let r1 = rirb.pop_response()?;
    let r2 = rirb.pop_response()?;
    if r1.response != 0x10EC_0295
        || r1.is_unsolicited()
        || !r2.is_unsolicited()
        || r2.codec_addr() != 2
    {
        return Err("sof_dsp: RIRB response decode mismatch");
    }

    // 2. HDA BDL & Stream DMA Engine test
    if BdlEntry::new(0x1004, 4096, true).is_ok() {
        return Err("sof_dsp: unaligned BDL entry should fail");
    }
    let bdl = BdlTable::from_contiguous(0x0020_0000, 4096, 4, true)?;
    if bdl.cbl_check_helper() != 16384 || bdl.ioc_count() != 4 || bdl.lvi() != 3 {
        return Err("sof_dsp: BDL table geometry mismatch");
    }
    let mut dma = HdaStreamDma::new(1, StreamDirection::Playback, AudioFormat::default(), bdl)?;
    dma.prepare()?;
    dma.start()?;
    let iocs = dma.advance_lpib(10_240)?; // crosses 2 periods (8192 bytes) + 2048 into 3rd
    if iocs != 2 || dma.lpib != 10_240 || dma.completed_periods != 2 {
        return Err("sof_dsp: HDA DMA LPIB/IOC tracking mismatch");
    }
    let wrap_iocs = dma.advance_lpib(6_144)?; // wraps back to 0 (crosses periods 3 and 4)
    if wrap_iocs != 2 || dma.lpib != 0 || dma.completed_periods != 4 {
        return Err("sof_dsp: HDA DMA cyclic wraparound mismatch");
    }
    dma.pause()?;
    dma.stop();

    // 3. SOF IPC4 Header & AudioDataFormat ABI test
    let fmt = AudioDataFormat::pcm_48k_stereo_16();
    fmt.validate()?;
    if fmt.frame_size_bytes() != 4 || fmt.bytes_per_second() != 192_000 {
        return Err("sof_dsp: AudioDataFormat size calculation mismatch");
    }
    let dwords = fmt.to_ipc4_dwords();
    if dwords[0] != 48_000 || (dwords[1] & 0xFF) != 16 {
        return Err("sof_dsp: AudioDataFormat IPC4 ABI encoding mismatch");
    }

    let hdr = Ipc4Header::create_pipeline(1, 4, 0, false, 0);
    let msg = Ipc4Message::new(hdr, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    let raw = msg.to_bytes();
    let decoded = Ipc4Message::from_bytes(&raw)?;
    if decoded != msg || decoded.header.target() != Ipc4MsgTarget::FwGenMsg {
        return Err("sof_dsp: IPC4 message serialization mismatch");
    }

    // 4. SOF IPC4 Topology & Pipeline Builder test
    let mut topo = SofTopology::new();
    topo.add_pipeline(Pipeline::new(1, 0, 4))?;
    topo.add_pipeline(Pipeline::new(2, 1, 4))?;

    let host_out = ModuleInstance::new(ModuleType::HostCopier, 0, 1, fmt)?;
    let gain = ModuleInstance::new(ModuleType::Gain, 0, 1, fmt)?;
    let mixer = ModuleInstance::new(ModuleType::Mixer, 0, 1, fmt)?;
    let dai_out = ModuleInstance::new(ModuleType::DaiCopier, 0, 1, fmt)?;
    let dai_in = ModuleInstance::new(ModuleType::DaiCopier, 1, 2, fmt)?;
    let host_in = ModuleInstance::new(ModuleType::HostCopier, 1, 2, fmt)?;

    let k_host_out = host_out.key;
    let k_gain = gain.key;
    let k_mixer = mixer.key;
    let k_dai_out = dai_out.key;
    let k_dai_in = dai_in.key;
    let k_host_in = host_in.key;

    topo.add_module(host_out)?;
    topo.add_module(gain)?;
    topo.add_module(mixer)?;
    topo.add_module(dai_out)?;
    topo.add_module(dai_in)?;
    topo.add_module(host_in)?;

    topo.bind(k_host_out, 0, k_gain, 0)?;
    topo.bind(k_gain, 0, k_mixer, 0)?;
    topo.bind(k_mixer, 0, k_dai_out, 0)?;
    topo.bind(k_dai_in, 0, k_host_in, 0)?;

    // Cycle detection check
    if topo.bind(k_dai_out, 0, k_host_out, 0) != Err(SofError::CycleDetected) {
        return Err("sof_dsp: cycle detection failed");
    }

    let order = topo.topological_order()?;
    if order.len() != 6 {
        return Err("sof_dsp: topological order length mismatch");
    }

    let bringup = topo.compile_ipc4_sequence()?;
    // 2 create_pipeline + 6 init_instance + 4 bind + 4 set_pipeline_state = 16
    if bringup.len() != 16 {
        return Err("sof_dsp: compiled IPC4 bring-up sequence length mismatch");
    }
    let teardown = topo.teardown_ipc4_sequence();
    // 2 reset + 4 unbind + 6 delete_instance + 2 delete_pipeline = 14
    if teardown.len() != 14 {
        return Err("sof_dsp: compiled IPC4 teardown sequence length mismatch");
    }

    // 5. Fixed-point DSP Gain & Mixer test
    let mut buf_a = [1000i16, -2000, 30000, -30000];
    let buf_b = [500i16, -500, 5000, -5000];
    dsp_apply_gain_i16(&mut buf_a, 0x0000_8000); // 0.5x (-6 dB)
    if buf_a != [500, -1000, 15000, -15000] {
        return Err("sof_dsp: Q16 gain scaling mismatch");
    }
    dsp_mix_i16(&mut buf_a, &buf_b)?;
    if buf_a != [1000, -1500, 20000, -20000] {
        return Err("sof_dsp: saturating mixer mismatch");
    }

    Ok(())
}

impl BdlTable {
    #[inline]
    const fn cbl_check_helper(&self) -> u32 {
        self.total_bytes
    }
}
