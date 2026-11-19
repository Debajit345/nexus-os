//! Clean-room ACPI AML bytecode parser and power/thermal subsystem.
//!
//! Provides `no_std` DSDT/SSDT table validation, iterative AML bytecode decoding
//! (Scope, Device, Method, Name, Package, Buffer, OperationRegion, Processor,
//! PowerResource, ThermalZone), `_PRT` PCIe IRQ routing extraction, and
//! thermal/battery/sleep-state descriptor extraction.
#![allow(dead_code)]

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ptr::read_unaligned;

use crate::acpi::SdtHeader;

const MAX_SCOPE_DEPTH: usize = 48;
const MAX_MULTI_NAME_SEGS: usize = 32;
const MAX_PACKAGE_ELEMENTS: usize = 256;
const MAX_PARSE_STEPS: usize = 65_536;

/// Errors returned by ACPI AML validation and decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmlError {
    TruncatedTable,
    InvalidSignature,
    InvalidLength,
    BadChecksum,
    UnexpectedEof,
    InvalidPkgLength,
    InvalidNameString,
    InvalidOpcode(u8),
    InvalidExtOpcode(u8),
    ScopeDepthExceeded,
    PackageTooLarge,
    StepLimitExceeded,
    MalformedObject,
}

impl AmlError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TruncatedTable => "truncated ACPI AML table",
            Self::InvalidSignature => "invalid AML table signature (expected DSDT or SSDT)",
            Self::InvalidLength => "invalid AML table length",
            Self::BadChecksum => "ACPI AML table checksum mismatch",
            Self::UnexpectedEof => "unexpected end of AML bytecode",
            Self::InvalidPkgLength => "invalid AML PkgLength encoding",
            Self::InvalidNameString => "invalid AML NameString encoding",
            Self::InvalidOpcode(_) => "invalid or unsupported AML opcode",
            Self::InvalidExtOpcode(_) => "invalid or unsupported AML extended opcode",
            Self::ScopeDepthExceeded => "AML scope nesting depth exceeded",
            Self::PackageTooLarge => "AML package element count exceeded limit",
            Self::StepLimitExceeded => "AML parser step limit exceeded",
            Self::MalformedObject => "malformed AML object",
        }
    }
}

/// Kind of AML definition block table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmlTableKind {
    Dsdt,
    Ssdt,
}

/// Validated DSDT or SSDT header metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmlTableHeader {
    pub kind: AmlTableKind,
    pub length: u32,
    pub revision: u8,
    pub checksum: u8,
    pub oem_id: [u8; 6],
    pub oem_table_id: [u8; 8],
    pub oem_revision: u32,
    pub creator_id: u32,
    pub creator_revision: u32,
}

/// Validate an in-memory DSDT or SSDT table and return its header and AML bytecode slice.
pub fn validate_aml_table(bytes: &[u8]) -> Result<(AmlTableHeader, &[u8]), AmlError> {
    let hdr_size = core::mem::size_of::<SdtHeader>();
    if bytes.len() < hdr_size {
        return Err(AmlError::TruncatedTable);
    }

    let hdr = unsafe { read_unaligned(bytes.as_ptr() as *const SdtHeader) };
    let kind = match &hdr.signature {
        b"DSDT" => AmlTableKind::Dsdt,
        b"SSDT" => AmlTableKind::Ssdt,
        _ => return Err(AmlError::InvalidSignature),
    };

    let total_len = hdr.length as usize;
    if total_len < hdr_size || total_len > bytes.len() {
        return Err(AmlError::InvalidLength);
    }

    let sum = bytes[..total_len]
        .iter()
        .fold(0u8, |acc, &b| acc.wrapping_add(b));
    if sum != 0 {
        return Err(AmlError::BadChecksum);
    }

    let parsed = AmlTableHeader {
        kind,
        length: hdr.length,
        revision: hdr.revision,
        checksum: hdr.checksum,
        oem_id: hdr.oem_id,
        oem_table_id: hdr.oem_table_id,
        oem_revision: hdr.oem_revision,
        creator_id: hdr.creator_id,
        creator_revision: hdr.creator_revision,
    };

    Ok((parsed, &bytes[hdr_size..total_len]))
}

/// Validate a DSDT or SSDT table at a physical address using the kernel HHDM offset.
///
/// # Safety
/// `phys` must point to a valid, mapped ACPI DSDT/SSDT table in physical memory.
pub unsafe fn validate_aml_table_phys(
    phys: u64,
) -> Result<(AmlTableHeader, &'static [u8]), AmlError> {
    if phys == 0 {
        return Err(AmlError::TruncatedTable);
    }
    let hhdm = crate::mm::stats().hhdm_offset;
    let ptr = phys.wrapping_add(hhdm) as *const u8;
    let hdr_size = core::mem::size_of::<SdtHeader>();
    let hdr = read_unaligned(ptr as *const SdtHeader);
    let total_len = hdr.length as usize;
    if total_len < hdr_size || total_len > 16 * 1024 * 1024 {
        return Err(AmlError::InvalidLength);
    }
    let slice = core::slice::from_raw_parts(ptr, total_len);
    validate_aml_table(slice)
}

/// Compute the ACPI 8-bit checksum byte for a table buffer (with checksum field at index 9).
pub fn compute_acpi_checksum(bytes: &[u8]) -> u8 {
    let sum = bytes.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    0u8.wrapping_sub(sum)
}

/// Decode an ACPI AML `PkgLength` starting at `pos`.
///
/// Returns `(pkg_length, next_pos)` where `pkg_length` includes the `PkgLength` encoding bytes.
pub fn decode_pkg_length(code: &[u8], pos: usize) -> Result<(usize, usize), AmlError> {
    let lead = *code.get(pos).ok_or(AmlError::UnexpectedEof)?;
    let follow_bytes = (lead >> 6) as usize;
    let header_len = 1 + follow_bytes;
    if pos + header_len > code.len() {
        return Err(AmlError::UnexpectedEof);
    }

    let pkg_len = if follow_bytes == 0 {
        (lead & 0x3F) as usize
    } else {
        let mut len = (lead & 0x0F) as usize;
        for i in 0..follow_bytes {
            let b = code[pos + 1 + i] as usize;
            len |= b << (4 + i * 8);
        }
        len
    };

    if pkg_len < header_len || pos + pkg_len > code.len() {
        return Err(AmlError::InvalidPkgLength);
    }

    Ok((pkg_len, pos + header_len))
}

#[inline]
fn is_lead_name_char(b: u8) -> bool {
    b.is_ascii_uppercase() || b == b'_'
}

#[inline]
fn is_name_char(b: u8) -> bool {
    b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'
}

fn decode_name_seg(code: &[u8], pos: usize) -> Result<([u8; 4], usize), AmlError> {
    if pos + 4 > code.len() {
        return Err(AmlError::UnexpectedEof);
    }
    let seg = [code[pos], code[pos + 1], code[pos + 2], code[pos + 3]];
    if !is_lead_name_char(seg[0])
        || !is_name_char(seg[1])
        || !is_name_char(seg[2])
        || !is_name_char(seg[3])
    {
        return Err(AmlError::InvalidNameString);
    }
    Ok((seg, pos + 4))
}

/// Canonical ACPI namespace path (e.g. `\_SB.PCI0._PRT`).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AmlPath {
    pub segments: Vec<[u8; 4]>,
}

impl AmlPath {
    pub fn root() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn from_segments(segments: Vec<[u8; 4]>) -> Self {
        Self { segments }
    }

    pub fn push(&mut self, seg: [u8; 4]) {
        self.segments.push(seg);
    }

    pub fn pop(&mut self) {
        self.segments.pop();
    }

    pub fn last_seg(&self) -> Option<[u8; 4]> {
        self.segments.last().copied()
    }

    pub fn ends_with_seg(&self, seg: &[u8; 4]) -> bool {
        self.segments.last() == Some(seg)
    }

    pub fn parent(&self) -> Self {
        if self.segments.is_empty() {
            Self::root()
        } else {
            Self {
                segments: self.segments[..self.segments.len() - 1].to_vec(),
            }
        }
    }

    pub fn starts_with(&self, prefix: &AmlPath) -> bool {
        self.segments.len() >= prefix.segments.len()
            && self.segments[..prefix.segments.len()] == prefix.segments[..]
    }

    pub fn to_string_path(&self) -> String {
        if self.segments.is_empty() {
            return String::from("\\");
        }
        let mut out = String::with_capacity(1 + self.segments.len() * 5);
        out.push('\\');
        for (idx, seg) in self.segments.iter().enumerate() {
            if idx > 0 {
                out.push('.');
            }
            for &b in seg {
                out.push(b as char);
            }
        }
        out
    }
}

impl fmt::Debug for AmlPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_path())
    }
}

/// Decoded relative or absolute AML NameString.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmlName {
    pub root: bool,
    pub parent_ups: u8,
    pub segments: Vec<[u8; 4]>,
}

impl AmlName {
    /// Resolve this NameString against `current_scope`.
    pub fn resolve(&self, current_scope: &AmlPath) -> AmlPath {
        let mut segs = if self.root {
            Vec::new()
        } else {
            let keep = current_scope
                .segments
                .len()
                .saturating_sub(self.parent_ups as usize);
            current_scope.segments[..keep].to_vec()
        };
        segs.extend_from_slice(&self.segments);
        AmlPath { segments: segs }
    }
}

/// Decode an AML `NameString` starting at `pos`.
pub fn decode_name_string(code: &[u8], mut pos: usize) -> Result<(AmlName, usize), AmlError> {
    let mut root = false;
    let mut parent_ups = 0u8;

    let first = *code.get(pos).ok_or(AmlError::UnexpectedEof)?;
    if first == 0x5C {
        // RootChar '\'
        root = true;
        pos += 1;
    } else {
        while pos < code.len() && code[pos] == 0x5E {
            // ParentPrefixChar '^'
            parent_ups = parent_ups.saturating_add(1);
            pos += 1;
        }
    }

    let tag = *code.get(pos).ok_or(AmlError::UnexpectedEof)?;
    let mut segments = Vec::new();

    match tag {
        0x00 => {
            // NullName
            pos += 1;
        }
        0x2E => {
            // DualNamePrefix
            pos += 1;
            let (s1, p1) = decode_name_seg(code, pos)?;
            let (s2, p2) = decode_name_seg(code, p1)?;
            segments.push(s1);
            segments.push(s2);
            pos = p2;
        }
        0x2F => {
            // MultiNamePrefix
            pos += 1;
            let count = *code.get(pos).ok_or(AmlError::UnexpectedEof)? as usize;
            pos += 1;
            if count == 0 || count > MAX_MULTI_NAME_SEGS {
                return Err(AmlError::InvalidNameString);
            }
            segments.reserve(count);
            for _ in 0..count {
                let (seg, next) = decode_name_seg(code, pos)?;
                segments.push(seg);
                pos = next;
            }
        }
        _ => {
            let (seg, next) = decode_name_seg(code, pos)?;
            segments.push(seg);
            pos = next;
        }
    }

    Ok((
        AmlName {
            root,
            parent_ups,
            segments,
        },
        pos,
    ))
}

/// Decode a 32-bit compressed EISAID (e.g. `0x0A0CD041` -> `"PNP0C0A"`).
pub fn decode_eisa_id(raw: u32) -> [u8; 7] {
    // EISAID is stored as a little-endian 32-bit integer in ACPI AML:
    // byte 0..1 = compressed 3-char vendor ID, byte 2..3 = 4 hex digits.
    let be = raw.to_le_bytes();
    let c1 = ((be[0] >> 2) & 0x1F) + b'@';
    let c2 = (((be[0] & 0x03) << 3) | ((be[1] >> 5) & 0x07)) + b'@';
    let c3 = (be[1] & 0x1F) + b'@';

    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let h1 = HEX[((be[2] >> 4) & 0x0F) as usize];
    let h2 = HEX[(be[2] & 0x0F) as usize];
    let h3 = HEX[((be[3] >> 4) & 0x0F) as usize];
    let h4 = HEX[(be[3] & 0x0F) as usize];

    [c1, c2, c3, h1, h2, h3, h4]
}

/// Decoded AML data value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AmlValue {
    Uninitialized,
    Integer(u64),
    String(String),
    Buffer(Vec<u8>),
    Package(Vec<AmlValue>),
    NameRef(AmlPath),
}

impl AmlValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Integer(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_package(&self) -> Option<&[AmlValue]> {
        match self {
            Self::Package(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    pub fn as_name_ref(&self) -> Option<&AmlPath> {
        match self {
            Self::NameRef(p) => Some(p),
            _ => None,
        }
    }

    /// Interpret `_HID` / `_CID` as a 7-char PNP/ACPI ID string when possible.
    pub fn as_pnp_id(&self) -> Option<String> {
        match self {
            Self::String(s) => Some(s.clone()),
            Self::Integer(v) => {
                let bytes = decode_eisa_id(*v as u32);
                core::str::from_utf8(&bytes).ok().map(String::from)
            }
            _ => None,
        }
    }
}

/// Decode an AML `DataRefObject` / simple `TermArg` at `pos` within `scope`.
///
/// Uses an explicit iterative work stack for nested `Package`s to keep call stack frames small.
pub fn decode_data_object(
    code: &[u8],
    pos: usize,
    scope: &AmlPath,
) -> Result<(AmlValue, usize), AmlError> {
    enum PendingPkg {
        Fixed {
            end: usize,
            expected: usize,
            items: Vec<AmlValue>,
        },
    }

    let mut stack: Vec<PendingPkg> = Vec::new();
    let mut cur = pos;

    loop {
        // If we are inside a package and reached its end or element count, close it.
        while let Some(top) = stack.last() {
            let PendingPkg::Fixed {
                end,
                expected,
                items,
            } = top;
            if cur >= *end || items.len() >= *expected {
                let finished = match stack.pop() {
                    Some(PendingPkg::Fixed { end, items, .. }) => {
                        cur = end;
                        AmlValue::Package(items)
                    }
                    None => unreachable!(),
                };
                if let Some(PendingPkg::Fixed { items, .. }) = stack.last_mut() {
                    items.push(finished);
                } else {
                    return Ok((finished, cur));
                }
            } else {
                break;
            }
        }

        let op = *code.get(cur).ok_or(AmlError::UnexpectedEof)?;
        let val = match op {
            0x00 => {
                // ZeroOp
                cur += 1;
                AmlValue::Integer(0)
            }
            0x01 => {
                // OneOp
                cur += 1;
                AmlValue::Integer(1)
            }
            0xFF => {
                // OnesOp
                cur += 1;
                AmlValue::Integer(u64::MAX)
            }
            0x0A => {
                // BytePrefix
                let b = *code.get(cur + 1).ok_or(AmlError::UnexpectedEof)?;
                cur += 2;
                AmlValue::Integer(b as u64)
            }
            0x0B => {
                // WordPrefix
                if cur + 3 > code.len() {
                    return Err(AmlError::UnexpectedEof);
                }
                let v = u16::from_le_bytes([code[cur + 1], code[cur + 2]]);
                cur += 3;
                AmlValue::Integer(v as u64)
            }
            0x0C => {
                // DWordPrefix
                if cur + 5 > code.len() {
                    return Err(AmlError::UnexpectedEof);
                }
                let v = u32::from_le_bytes([
                    code[cur + 1],
                    code[cur + 2],
                    code[cur + 3],
                    code[cur + 4],
                ]);
                cur += 5;
                AmlValue::Integer(v as u64)
            }
            0x0E => {
                // QWordPrefix
                if cur + 9 > code.len() {
                    return Err(AmlError::UnexpectedEof);
                }
                let v = u64::from_le_bytes([
                    code[cur + 1],
                    code[cur + 2],
                    code[cur + 3],
                    code[cur + 4],
                    code[cur + 5],
                    code[cur + 6],
                    code[cur + 7],
                    code[cur + 8],
                ]);
                cur += 9;
                AmlValue::Integer(v)
            }
            0x0D => {
                // StringPrefix (null-terminated ASCII)
                let mut end = cur + 1;
                while end < code.len() && code[end] != 0 {
                    end += 1;
                }
                if end >= code.len() {
                    return Err(AmlError::UnexpectedEof);
                }
                let s = core::str::from_utf8(&code[cur + 1..end])
                    .map(String::from)
                    .unwrap_or_else(|_| String::from("<non-utf8>"));
                cur = end + 1;
                AmlValue::String(s)
            }
            0x11 => {
                // BufferOp: PkgLength, BufferSize (TermArg), ByteList
                let (pkg_len, body_start) = decode_pkg_length(code, cur + 1)?;
                let pkg_end = cur + 1 + pkg_len;
                let (_buf_size, data_start) = decode_simple_int(code, body_start)?;
                let raw = if data_start <= pkg_end {
                    code[data_start..pkg_end].to_vec()
                } else {
                    Vec::new()
                };
                cur = pkg_end;
                AmlValue::Buffer(raw)
            }
            0x12 => {
                // PackageOp: PkgLength, NumElements (u8), PackageElementList
                if stack.len() >= MAX_SCOPE_DEPTH {
                    return Err(AmlError::ScopeDepthExceeded);
                }
                let (pkg_len, body_start) = decode_pkg_length(code, cur + 1)?;
                let pkg_end = cur + 1 + pkg_len;
                let num_elems = *code.get(body_start).ok_or(AmlError::UnexpectedEof)? as usize;
                if num_elems > MAX_PACKAGE_ELEMENTS {
                    return Err(AmlError::PackageTooLarge);
                }
                cur = body_start + 1;
                stack.push(PendingPkg::Fixed {
                    end: pkg_end,
                    expected: num_elems,
                    items: Vec::with_capacity(num_elems.min(32)),
                });
                continue;
            }
            0x13 => {
                // VarPackageOp: PkgLength, VarNumElements (TermArg), PackageElementList
                if stack.len() >= MAX_SCOPE_DEPTH {
                    return Err(AmlError::ScopeDepthExceeded);
                }
                let (pkg_len, body_start) = decode_pkg_length(code, cur + 1)?;
                let pkg_end = cur + 1 + pkg_len;
                let (num_elems_u64, elem_start) = decode_simple_int(code, body_start)?;
                let num_elems = num_elems_u64 as usize;
                if num_elems > MAX_PACKAGE_ELEMENTS {
                    return Err(AmlError::PackageTooLarge);
                }
                cur = elem_start;
                stack.push(PendingPkg::Fixed {
                    end: pkg_end,
                    expected: num_elems,
                    items: Vec::with_capacity(num_elems.min(32)),
                });
                continue;
            }
            0x5B => {
                // ExtOpPrefix inside data object (e.g. RevisionOp 0x5B 0x30)
                let ext = *code.get(cur + 1).ok_or(AmlError::UnexpectedEof)?;
                if ext == 0x30 {
                    cur += 2;
                    AmlValue::Integer(2)
                } else {
                    return Err(AmlError::InvalidExtOpcode(ext));
                }
            }
            0x5C | 0x5E | 0x2E | 0x2F | b'A'..=b'Z' | b'_' => {
                let (name, next) = decode_name_string(code, cur)?;
                cur = next;
                AmlValue::NameRef(name.resolve(scope))
            }
            other => return Err(AmlError::InvalidOpcode(other)),
        };

        if let Some(PendingPkg::Fixed { items, .. }) = stack.last_mut() {
            items.push(val);
        } else {
            return Ok((val, cur));
        }
    }
}

fn decode_simple_int(code: &[u8], pos: usize) -> Result<(u64, usize), AmlError> {
    let op = *code.get(pos).ok_or(AmlError::UnexpectedEof)?;
    match op {
        0x00 => Ok((0, pos + 1)),
        0x01 => Ok((1, pos + 1)),
        0xFF => Ok((u64::MAX, pos + 1)),
        0x0A => {
            let b = *code.get(pos + 1).ok_or(AmlError::UnexpectedEof)?;
            Ok((b as u64, pos + 2))
        }
        0x0B => {
            if pos + 3 > code.len() {
                return Err(AmlError::UnexpectedEof);
            }
            Ok((
                u16::from_le_bytes([code[pos + 1], code[pos + 2]]) as u64,
                pos + 3,
            ))
        }
        0x0C => {
            if pos + 5 > code.len() {
                return Err(AmlError::UnexpectedEof);
            }
            Ok((
                u32::from_le_bytes([code[pos + 1], code[pos + 2], code[pos + 3], code[pos + 4]])
                    as u64,
                pos + 5,
            ))
        }
        0x0E => {
            if pos + 9 > code.len() {
                return Err(AmlError::UnexpectedEof);
            }
            Ok((
                u64::from_le_bytes([
                    code[pos + 1],
                    code[pos + 2],
                    code[pos + 3],
                    code[pos + 4],
                    code[pos + 5],
                    code[pos + 6],
                    code[pos + 7],
                    code[pos + 8],
                ]),
                pos + 9,
            ))
        }
        other => Err(AmlError::InvalidOpcode(other)),
    }
}

/// OperationRegion address space IDs (ACPI Spec 19.6.96).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionSpace {
    SystemMemory,
    SystemIo,
    PciConfig,
    EmbeddedControl,
    SmBus,
    SystemCmos,
    PciBarTarget,
    Ipmi,
    GeneralPurposeIo,
    GenericSerialBus,
    Other(u8),
}

impl RegionSpace {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0x00 => Self::SystemMemory,
            0x01 => Self::SystemIo,
            0x02 => Self::PciConfig,
            0x03 => Self::EmbeddedControl,
            0x04 => Self::SmBus,
            0x05 => Self::SystemCmos,
            0x06 => Self::PciBarTarget,
            0x07 => Self::Ipmi,
            0x08 => Self::GeneralPurposeIo,
            0x09 => Self::GenericSerialBus,
            other => Self::Other(other),
        }
    }
}

/// Decoded AML `Method` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmlMethod {
    pub path: AmlPath,
    pub arg_count: u8,
    pub serialized: bool,
    pub sync_level: u8,
    pub body_offset: usize,
    pub body_len: usize,
}

/// Decoded AML `Device` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmlDevice {
    pub path: AmlPath,
}

/// Decoded AML `Processor` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmlProcessor {
    pub path: AmlPath,
    pub proc_id: u8,
    pub pblk_addr: u32,
    pub pblk_len: u8,
}

/// Decoded AML `PowerResource` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmlPowerResource {
    pub path: AmlPath,
    pub system_level: u8,
    pub resource_order: u16,
}

/// Decoded AML `OperationRegion` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmlOpRegion {
    pub path: AmlPath,
    pub space: RegionSpace,
    pub offset: u64,
    pub length: u64,
}

/// Decoded AML `Name` object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmlNamedObject {
    pub path: AmlPath,
    pub value: AmlValue,
}

/// Decoded AML `Alias` object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmlAlias {
    pub alias: AmlPath,
    pub target: AmlPath,
}

/// Parsed ACPI AML namespace populated from one or more DSDT/SSDT tables.
#[derive(Debug, Clone, Default)]
pub struct AmlNamespace {
    pub devices: Vec<AmlDevice>,
    pub methods: Vec<AmlMethod>,
    pub names: Vec<AmlNamedObject>,
    pub aliases: Vec<AmlAlias>,
    pub processors: Vec<AmlProcessor>,
    pub power_resources: Vec<AmlPowerResource>,
    pub thermal_zones: Vec<AmlPath>,
    pub op_regions: Vec<AmlOpRegion>,
}

impl AmlNamespace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse an AML bytecode stream (from DSDT or SSDT) into this namespace.
    pub fn parse_aml(&mut self, code: &[u8]) -> Result<(), AmlError> {
        struct ScopeFrame {
            path: AmlPath,
            end: usize,
        }

        let mut stack: Vec<ScopeFrame> = Vec::with_capacity(8);
        stack.push(ScopeFrame {
            path: AmlPath::root(),
            end: code.len(),
        });

        let mut pos = 0usize;
        let mut steps = 0usize;

        while let Some(top) = stack.last() {
            if pos >= top.end {
                let finished_end = top.end;
                stack.pop();
                pos = finished_end;
                continue;
            }

            steps += 1;
            if steps > MAX_PARSE_STEPS {
                return Err(AmlError::StepLimitExceeded);
            }

            let cur_scope = top.path.clone();
            let op = code[pos];

            match op {
                // ScopeOp: 0x10 PkgLength NameString TermList
                0x10 => {
                    let (pkg_len, body_start) = decode_pkg_length(code, pos + 1)?;
                    let pkg_end = pos + 1 + pkg_len;
                    let (name, next) = decode_name_string(code, body_start)?;
                    let resolved = name.resolve(&cur_scope);
                    if stack.len() >= MAX_SCOPE_DEPTH {
                        return Err(AmlError::ScopeDepthExceeded);
                    }
                    pos = next;
                    stack.push(ScopeFrame {
                        path: resolved,
                        end: pkg_end,
                    });
                }
                // NameOp: 0x08 NameString DataRefObject
                0x08 => {
                    let (name, val_pos) = decode_name_string(code, pos + 1)?;
                    let resolved = name.resolve(&cur_scope);
                    let (value, next) = decode_data_object(code, val_pos, &cur_scope)?;
                    self.names.push(AmlNamedObject {
                        path: resolved,
                        value,
                    });
                    pos = next;
                }
                // AliasOp: 0x06 NameString NameString
                0x06 => {
                    let (target_name, p1) = decode_name_string(code, pos + 1)?;
                    let (alias_name, p2) = decode_name_string(code, p1)?;
                    self.aliases.push(AmlAlias {
                        alias: alias_name.resolve(&cur_scope),
                        target: target_name.resolve(&cur_scope),
                    });
                    pos = p2;
                }
                // MethodOp: 0x14 PkgLength NameString MethodFlags TermList
                0x14 => {
                    let (pkg_len, body_start) = decode_pkg_length(code, pos + 1)?;
                    let pkg_end = pos + 1 + pkg_len;
                    let (name, flags_pos) = decode_name_string(code, body_start)?;
                    let flags = *code.get(flags_pos).ok_or(AmlError::UnexpectedEof)?;
                    let method_body = flags_pos + 1;
                    let resolved = name.resolve(&cur_scope);
                    self.methods.push(AmlMethod {
                        path: resolved,
                        arg_count: flags & 0x07,
                        serialized: (flags & 0x08) != 0,
                        sync_level: (flags >> 4) & 0x0F,
                        body_offset: method_body,
                        body_len: pkg_end.saturating_sub(method_body),
                    });
                    pos = pkg_end;
                }
                // IfOp (0xA0), ElseOp (0xA1), WhileOp (0xA2): skip via PkgLength
                0xA0..=0xA2 => {
                    let (pkg_len, _) = decode_pkg_length(code, pos + 1)?;
                    pos = pos + 1 + pkg_len;
                }
                // NoopOp (0xA3)
                0xA3 => {
                    pos += 1;
                }
                // ExtOpPrefix: 0x5B
                0x5B => {
                    let ext = *code.get(pos + 1).ok_or(AmlError::UnexpectedEof)?;
                    match ext {
                        // MutexOp: 0x5B 0x01 NameString SyncFlags
                        0x01 => {
                            let (_name, next) = decode_name_string(code, pos + 2)?;
                            if next >= code.len() {
                                return Err(AmlError::UnexpectedEof);
                            }
                            pos = next + 1;
                        }
                        // EventOp: 0x5B 0x02 NameString
                        0x02 => {
                            let (_name, next) = decode_name_string(code, pos + 2)?;
                            pos = next;
                        }
                        // OpRegionOp: 0x5B 0x80 NameString RegionSpace RegionOffset RegionLen
                        0x80 => {
                            let (name, p1) = decode_name_string(code, pos + 2)?;
                            let space_byte = *code.get(p1).ok_or(AmlError::UnexpectedEof)?;
                            let (offset, p2) = decode_simple_int(code, p1 + 1)?;
                            let (length, p3) = decode_simple_int(code, p2)?;
                            self.op_regions.push(AmlOpRegion {
                                path: name.resolve(&cur_scope),
                                space: RegionSpace::from_u8(space_byte),
                                offset,
                                length,
                            });
                            pos = p3;
                        }
                        // FieldOp (0x81), IndexFieldOp (0x86), BankFieldOp (0x87): skip via PkgLength
                        0x81 | 0x86 | 0x87 => {
                            let (pkg_len, _) = decode_pkg_length(code, pos + 2)?;
                            pos = pos + 2 + pkg_len;
                        }
                        // DeviceOp: 0x5B 0x82 PkgLength NameString TermList
                        0x82 => {
                            let (pkg_len, body_start) = decode_pkg_length(code, pos + 2)?;
                            let pkg_end = pos + 2 + pkg_len;
                            let (name, next) = decode_name_string(code, body_start)?;
                            let resolved = name.resolve(&cur_scope);
                            self.devices.push(AmlDevice {
                                path: resolved.clone(),
                            });
                            if stack.len() >= MAX_SCOPE_DEPTH {
                                return Err(AmlError::ScopeDepthExceeded);
                            }
                            pos = next;
                            stack.push(ScopeFrame {
                                path: resolved,
                                end: pkg_end,
                            });
                        }
                        // ProcessorOp: 0x5B 0x83 PkgLength NameString ProcID PblkAddr PblkLen TermList
                        0x83 => {
                            let (pkg_len, body_start) = decode_pkg_length(code, pos + 2)?;
                            let pkg_end = pos + 2 + pkg_len;
                            let (name, p1) = decode_name_string(code, body_start)?;
                            if p1 + 6 > pkg_end {
                                return Err(AmlError::UnexpectedEof);
                            }
                            let proc_id = code[p1];
                            let pblk_addr = u32::from_le_bytes([
                                code[p1 + 1],
                                code[p1 + 2],
                                code[p1 + 3],
                                code[p1 + 4],
                            ]);
                            let pblk_len = code[p1 + 5];
                            let resolved = name.resolve(&cur_scope);
                            self.processors.push(AmlProcessor {
                                path: resolved.clone(),
                                proc_id,
                                pblk_addr,
                                pblk_len,
                            });
                            if stack.len() >= MAX_SCOPE_DEPTH {
                                return Err(AmlError::ScopeDepthExceeded);
                            }
                            pos = p1 + 6;
                            stack.push(ScopeFrame {
                                path: resolved,
                                end: pkg_end,
                            });
                        }
                        // PowerResOp: 0x5B 0x84 PkgLength NameString SystemLevel ResourceOrder TermList
                        0x84 => {
                            let (pkg_len, body_start) = decode_pkg_length(code, pos + 2)?;
                            let pkg_end = pos + 2 + pkg_len;
                            let (name, p1) = decode_name_string(code, body_start)?;
                            if p1 + 3 > pkg_end {
                                return Err(AmlError::UnexpectedEof);
                            }
                            let system_level = code[p1];
                            let resource_order = u16::from_le_bytes([code[p1 + 1], code[p1 + 2]]);
                            let resolved = name.resolve(&cur_scope);
                            self.power_resources.push(AmlPowerResource {
                                path: resolved.clone(),
                                system_level,
                                resource_order,
                            });
                            if stack.len() >= MAX_SCOPE_DEPTH {
                                return Err(AmlError::ScopeDepthExceeded);
                            }
                            pos = p1 + 3;
                            stack.push(ScopeFrame {
                                path: resolved,
                                end: pkg_end,
                            });
                        }
                        // ThermalZoneOp: 0x5B 0x85 PkgLength NameString TermList
                        0x85 => {
                            let (pkg_len, body_start) = decode_pkg_length(code, pos + 2)?;
                            let pkg_end = pos + 2 + pkg_len;
                            let (name, next) = decode_name_string(code, body_start)?;
                            let resolved = name.resolve(&cur_scope);
                            self.thermal_zones.push(resolved.clone());
                            if stack.len() >= MAX_SCOPE_DEPTH {
                                return Err(AmlError::ScopeDepthExceeded);
                            }
                            pos = next;
                            stack.push(ScopeFrame {
                                path: resolved,
                                end: pkg_end,
                            });
                        }
                        other => return Err(AmlError::InvalidExtOpcode(other)),
                    }
                }
                other => return Err(AmlError::InvalidOpcode(other)),
            }
        }

        Ok(())
    }

    /// Look up a named object by exact path.
    pub fn find_name(&self, path: &AmlPath) -> Option<&AmlValue> {
        self.names
            .iter()
            .find(|n| &n.path == path)
            .map(|n| &n.value)
    }

    /// Look up a child named object under `parent` with 4-byte segment `seg`.
    pub fn find_child_name(&self, parent: &AmlPath, seg: &[u8; 4]) -> Option<&AmlValue> {
        self.names
            .iter()
            .find(|n| {
                n.path.segments.len() == parent.segments.len() + 1
                    && n.path.starts_with(parent)
                    && n.path.ends_with_seg(seg)
            })
            .map(|n| &n.value)
    }
}

/// PCI legacy interrupt pin (`INTA#`..`INTD#`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PciIntPin {
    IntA = 0,
    IntB = 1,
    IntC = 2,
    IntD = 3,
}

impl PciIntPin {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::IntA),
            1 => Some(Self::IntB),
            2 => Some(Self::IntC),
            3 => Some(Self::IntD),
            _ => None,
        }
    }
}

/// Target of a `_PRT` routing entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrtSource {
    /// Hardwired Global System Interrupt (Source == 0).
    Gsi(u32),
    /// Routed through a PCI Interrupt Link device (e.g. `\_SB.LNKA`) at `index`.
    LinkDevice { path: AmlPath, index: u32 },
}

/// Decoded `_PRT` PCIe IRQ routing entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrtEntry {
    pub scope: AmlPath,
    pub pci_device: u8,
    pub pci_function: u16,
    pub pin: PciIntPin,
    pub source: PrtSource,
}

/// Extract all `_PRT` PCIe IRQ routing entries from the parsed AML namespace.
pub fn extract_prt_entries(ns: &AmlNamespace) -> Vec<PrtEntry> {
    let mut out = Vec::new();
    for obj in &ns.names {
        if !obj.path.ends_with_seg(b"_PRT") {
            continue;
        }
        let scope = obj.path.parent();
        let Some(pkg) = obj.value.as_package() else {
            continue;
        };
        for elem in pkg {
            let Some(fields) = elem.as_package() else {
                continue;
            };
            if fields.len() < 4 {
                continue;
            }
            let Some(addr) = fields[0].as_u64() else {
                continue;
            };
            let Some(pin_raw) = fields[1].as_u64() else {
                continue;
            };
            let Some(pin) = PciIntPin::from_u8(pin_raw as u8) else {
                continue;
            };
            let Some(source_index) = fields[3].as_u64() else {
                continue;
            };

            let source = match &fields[2] {
                AmlValue::Integer(0) => PrtSource::Gsi(source_index as u32),
                AmlValue::NameRef(link_path) => PrtSource::LinkDevice {
                    path: link_path.clone(),
                    index: source_index as u32,
                },
                _ => continue,
            };

            let pci_device = ((addr >> 16) & 0x1F) as u8;
            let pci_function = (addr & 0xFFFF) as u16;

            out.push(PrtEntry {
                scope: scope.clone(),
                pci_device,
                pci_function,
                pin,
                source,
            });
        }
    }
    out
}

/// Resolve a PCI device + interrupt pin against extracted `_PRT` entries.
pub fn resolve_pci_irq(entries: &[PrtEntry], device: u8, pin: PciIntPin) -> Option<&PrtEntry> {
    entries
        .iter()
        .find(|e| e.pci_device == device && e.pin == pin)
}

/// Convert ACPI tenths-of-Kelvin temperature to tenths of degrees Celsius.
#[inline]
pub const fn decikelvin_to_celsius_tenths(dk: u64) -> i32 {
    (dk as i32) - 2732
}

/// Extracted ACPI ThermalZone descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThermalZoneDescriptor {
    pub path: AmlPath,
    pub tmp_decikelvin: Option<u64>,
    pub crt_decikelvin: Option<u64>,
    pub hot_decikelvin: Option<u64>,
    pub psv_decikelvin: Option<u64>,
    pub ac0_decikelvin: Option<u64>,
    pub tc1: Option<u64>,
    pub tc2: Option<u64>,
    pub tsp: Option<u64>,
    pub tzp: Option<u64>,
}

/// Extracted ACPI Control Method Battery (`PNP0C0A`) descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatteryDescriptor {
    pub path: AmlPath,
    pub uid: Option<u64>,
    pub sta: Option<u64>,
    pub design_capacity: Option<u64>,
    pub last_full_capacity: Option<u64>,
    pub design_voltage_mv: Option<u64>,
    pub state_flags: Option<u64>,
    pub present_rate: Option<u64>,
    pub remaining_capacity: Option<u64>,
    pub present_voltage_mv: Option<u64>,
    pub model_number: Option<String>,
}

/// Extracted ACPI AC Adapter (`ACPI0003`) descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcAdapterDescriptor {
    pub path: AmlPath,
    pub sta: Option<u64>,
    pub psr_online: Option<bool>,
}

/// Extracted ACPI sleep state (`_S0`..`_S5`) `SLP_TYP` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SleepStateDescriptor {
    pub state: u8,
    pub slp_typa: u8,
    pub slp_typb: u8,
}

/// Aggregated power, thermal, battery, and sleep-state descriptors from AML.
#[derive(Debug, Clone, Default)]
pub struct PowerThermalInfo {
    pub thermal_zones: Vec<ThermalZoneDescriptor>,
    pub batteries: Vec<BatteryDescriptor>,
    pub ac_adapters: Vec<AcAdapterDescriptor>,
    pub sleep_states: Vec<SleepStateDescriptor>,
}

/// Extract thermal zones, batteries, AC adapters, and `_S0`..`_S5` sleep states from `ns`.
pub fn extract_power_thermal_info(ns: &AmlNamespace) -> PowerThermalInfo {
    let mut info = PowerThermalInfo::default();

    for tz_path in &ns.thermal_zones {
        info.thermal_zones.push(ThermalZoneDescriptor {
            path: tz_path.clone(),
            tmp_decikelvin: ns
                .find_child_name(tz_path, b"_TMP")
                .and_then(AmlValue::as_u64),
            crt_decikelvin: ns
                .find_child_name(tz_path, b"_CRT")
                .and_then(AmlValue::as_u64),
            hot_decikelvin: ns
                .find_child_name(tz_path, b"_HOT")
                .and_then(AmlValue::as_u64),
            psv_decikelvin: ns
                .find_child_name(tz_path, b"_PSV")
                .and_then(AmlValue::as_u64),
            ac0_decikelvin: ns
                .find_child_name(tz_path, b"_AC0")
                .and_then(AmlValue::as_u64),
            tc1: ns
                .find_child_name(tz_path, b"_TC1")
                .and_then(AmlValue::as_u64),
            tc2: ns
                .find_child_name(tz_path, b"_TC2")
                .and_then(AmlValue::as_u64),
            tsp: ns
                .find_child_name(tz_path, b"_TSP")
                .and_then(AmlValue::as_u64),
            tzp: ns
                .find_child_name(tz_path, b"_TZP")
                .and_then(AmlValue::as_u64),
        });
    }

    for dev in &ns.devices {
        let hid = ns
            .find_child_name(&dev.path, b"_HID")
            .and_then(AmlValue::as_pnp_id);
        match hid.as_deref() {
            Some("PNP0C0A") => {
                let bif = ns
                    .find_child_name(&dev.path, b"_BIF")
                    .and_then(AmlValue::as_package);
                let bst = ns
                    .find_child_name(&dev.path, b"_BST")
                    .and_then(AmlValue::as_package);

                info.batteries.push(BatteryDescriptor {
                    path: dev.path.clone(),
                    uid: ns
                        .find_child_name(&dev.path, b"_UID")
                        .and_then(AmlValue::as_u64),
                    sta: ns
                        .find_child_name(&dev.path, b"_STA")
                        .and_then(AmlValue::as_u64),
                    design_capacity: bif.and_then(|p| p.get(1)).and_then(AmlValue::as_u64),
                    last_full_capacity: bif.and_then(|p| p.get(2)).and_then(AmlValue::as_u64),
                    design_voltage_mv: bif.and_then(|p| p.get(4)).and_then(AmlValue::as_u64),
                    model_number: bif
                        .and_then(|p| p.get(9))
                        .and_then(AmlValue::as_str)
                        .map(String::from),
                    state_flags: bst.and_then(|p| p.first()).and_then(AmlValue::as_u64),
                    present_rate: bst.and_then(|p| p.get(1)).and_then(AmlValue::as_u64),
                    remaining_capacity: bst.and_then(|p| p.get(2)).and_then(AmlValue::as_u64),
                    present_voltage_mv: bst.and_then(|p| p.get(3)).and_then(AmlValue::as_u64),
                });
            }
            Some("ACPI0003") => {
                let psr = ns
                    .find_child_name(&dev.path, b"_PSR")
                    .and_then(AmlValue::as_u64)
                    .map(|v| v != 0);
                info.ac_adapters.push(AcAdapterDescriptor {
                    path: dev.path.clone(),
                    sta: ns
                        .find_child_name(&dev.path, b"_STA")
                        .and_then(AmlValue::as_u64),
                    psr_online: psr,
                });
            }
            _ => {}
        }
    }

    let sleep_segs: [(u8, &[u8; 4]); 6] = [
        (0, b"_S0_"),
        (1, b"_S1_"),
        (2, b"_S2_"),
        (3, b"_S3_"),
        (4, b"_S4_"),
        (5, b"_S5_"),
    ];
    let root = AmlPath::root();
    for (state, seg) in sleep_segs {
        if let Some(pkg) = ns
            .find_child_name(&root, seg)
            .and_then(AmlValue::as_package)
        {
            if let (Some(a), Some(b)) = (
                pkg.first().and_then(AmlValue::as_u64),
                pkg.get(1).and_then(AmlValue::as_u64),
            ) {
                info.sleep_states.push(SleepStateDescriptor {
                    state,
                    slp_typa: a as u8,
                    slp_typb: b as u8,
                });
            }
        }
    }

    info
}

// ---------------------------------------------------------------------------
// Clean-room self-test helpers & suite
// ---------------------------------------------------------------------------

fn encode_pkg_length(body_len: usize) -> Vec<u8> {
    if body_len < 63 {
        vec![(body_len + 1) as u8]
    } else if body_len + 2 <= 4095 {
        let total = body_len + 2;
        vec![0x40 | ((total & 0x0F) as u8), ((total >> 4) & 0xFF) as u8]
    } else {
        let total = body_len + 3;
        vec![
            0x80 | ((total & 0x0F) as u8),
            ((total >> 4) & 0xFF) as u8,
            ((total >> 12) & 0xFF) as u8,
        ]
    }
}

fn wrap_pkg(opcode: &[u8], body: &[u8]) -> Vec<u8> {
    let pkg_len = encode_pkg_length(body.len());
    let mut out = Vec::with_capacity(opcode.len() + pkg_len.len() + body.len());
    out.extend_from_slice(opcode);
    out.extend_from_slice(&pkg_len);
    out.extend_from_slice(body);
    out
}

fn build_synthetic_dsdt(aml_body: &[u8]) -> Vec<u8> {
    let hdr_size = core::mem::size_of::<SdtHeader>();
    let total_len = hdr_size + aml_body.len();
    let mut table = vec![0u8; total_len];
    table[0..4].copy_from_slice(b"DSDT");
    table[4..8].copy_from_slice(&(total_len as u32).to_le_bytes());
    table[8] = 2; // revision
    table[10..16].copy_from_slice(b"NEXUS ");
    table[16..24].copy_from_slice(b"NEXUSAML");
    table[hdr_size..].copy_from_slice(aml_body);
    table[9] = compute_acpi_checksum(&table);
    table
}

/// Run clean-room self-tests for DSDT/SSDT validation, AML decoding, `_PRT`, and power/thermal extraction.
pub fn self_test() -> Result<(), &'static str> {
    // 1. EISAID decoding check: PNP0C0A -> 0x0A0CD041
    if &decode_eisa_id(0x0A0CD041) != b"PNP0C0A" {
        return Err("aml self_test: EISAID decode mismatch");
    }

    // 2. Build synthetic AML payload:
    //    - Name(_S5_, Package() { 0x05, 0x00, 0x00, 0x00 })
    //    - Scope(\_SB_) {
    //        Device(PCI0) {
    //          Name(_HID, "PNP0A08")
    //          Name(_PRT, Package() {
    //            Package() { 0x0001FFFF, 0, 0, 16 },
    //            Package() { 0x0002FFFF, 1, \_SB_.LNKA, 0 }
    //          })
    //          Method(_OSC, 4, NotSerialized) { Noop }
    //        }
    //        Device(BAT0) {
    //          Name(_HID, 0x0A0CD041)
    //          Name(_STA, 0x1F)
    //          Name(_BIF, Package() { 1, 50000, 48000, 1, 11400 })
    //          Name(_BST, Package() { 0, 1200, 45000, 11500 })
    //        }
    //        Device(AC__) {
    //          Name(_HID, "ACPI0003")
    //          Name(_PSR, 1)
    //        }
    //        ThermalZone(TZ00) {
    //          Name(_TMP, 3132)
    //          Name(_CRT, 3732)
    //          Name(_PSV, 3582)
    //          Name(_TZP, 100)
    //        }
    //      }
    let mut s5_pkg_body = vec![4u8]; // 4 elements
    s5_pkg_body.extend_from_slice(&[0x0A, 0x05, 0x00, 0x00, 0x00]);
    let s5_pkg = wrap_pkg(&[0x12], &s5_pkg_body);
    let mut aml = Vec::new();
    aml.push(0x08);
    aml.extend_from_slice(b"_S5_");
    aml.extend_from_slice(&s5_pkg);

    // _PRT entry 1: { 0x0001FFFF, 0, 0, 16 }
    let mut prt_e1_body = vec![4u8];
    prt_e1_body.push(0x0C);
    prt_e1_body.extend_from_slice(&0x0001_FFFFu32.to_le_bytes());
    prt_e1_body.push(0x00); // pin 0 (INTA)
    prt_e1_body.push(0x00); // source 0 (hardwired GSI)
    prt_e1_body.extend_from_slice(&[0x0A, 16]); // GSI 16
    let prt_e1 = wrap_pkg(&[0x12], &prt_e1_body);

    // _PRT entry 2: { 0x0002FFFF, 1, \_SB_.LNKA, 0 }
    let mut prt_e2_body = vec![4u8];
    prt_e2_body.push(0x0C);
    prt_e2_body.extend_from_slice(&0x0002_FFFFu32.to_le_bytes());
    prt_e2_body.push(0x01); // pin 1 (INTB)
    prt_e2_body.extend_from_slice(b"\x5C\x2E_SB_LNKA"); // \_SB_.LNKA
    prt_e2_body.push(0x00); // index 0
    let prt_e2 = wrap_pkg(&[0x12], &prt_e2_body);

    let mut prt_pkg_body = vec![2u8];
    prt_pkg_body.extend_from_slice(&prt_e1);
    prt_pkg_body.extend_from_slice(&prt_e2);
    let prt_pkg = wrap_pkg(&[0x12], &prt_pkg_body);

    // Method(_OSC, 4, NotSerialized) { Noop }
    let osc_method = wrap_pkg(&[0x14], b"_OSC\x04\xA3");

    // Device(PCI0)
    let mut pci0_body = Vec::new();
    pci0_body.extend_from_slice(b"PCI0");
    pci0_body.extend_from_slice(b"\x08_HID\x0DPNP0A08\x00");
    pci0_body.extend_from_slice(b"\x08_PRT");
    pci0_body.extend_from_slice(&prt_pkg);
    pci0_body.extend_from_slice(&osc_method);
    let pci0_dev = wrap_pkg(&[0x5B, 0x82], &pci0_body);

    // Device(BAT0)
    let mut bif_body = vec![5u8, 0x01];
    bif_body.push(0x0C);
    bif_body.extend_from_slice(&50_000u32.to_le_bytes());
    bif_body.push(0x0C);
    bif_body.extend_from_slice(&48_000u32.to_le_bytes());
    bif_body.push(0x01);
    bif_body.push(0x0C);
    bif_body.extend_from_slice(&11_400u32.to_le_bytes());
    let bif_pkg = wrap_pkg(&[0x12], &bif_body);

    let mut bst_body = vec![4u8, 0x00];
    bst_body.push(0x0B);
    bst_body.extend_from_slice(&1_200u16.to_le_bytes());
    bst_body.push(0x0C);
    bst_body.extend_from_slice(&45_000u32.to_le_bytes());
    bst_body.push(0x0B);
    bst_body.extend_from_slice(&11_500u16.to_le_bytes());
    let bst_pkg = wrap_pkg(&[0x12], &bst_body);

    let mut bat0_body = Vec::new();
    bat0_body.extend_from_slice(b"BAT0");
    bat0_body.extend_from_slice(b"\x08_HID\x0C");
    bat0_body.extend_from_slice(&0x0A0C_D041u32.to_le_bytes());
    bat0_body.extend_from_slice(b"\x08_STA\x0A\x1F");
    bat0_body.extend_from_slice(b"\x08_BIF");
    bat0_body.extend_from_slice(&bif_pkg);
    bat0_body.extend_from_slice(b"\x08_BST");
    bat0_body.extend_from_slice(&bst_pkg);
    let bat0_dev = wrap_pkg(&[0x5B, 0x82], &bat0_body);

    // Device(AC__)
    let mut ac_body = Vec::new();
    ac_body.extend_from_slice(b"AC__");
    ac_body.extend_from_slice(b"\x08_HID\x0DACPI0003\x00");
    ac_body.extend_from_slice(b"\x08_PSR\x01");
    let ac_dev = wrap_pkg(&[0x5B, 0x82], &ac_body);

    // ThermalZone(TZ00)
    let mut tz_body = Vec::new();
    tz_body.extend_from_slice(b"TZ00");
    tz_body.extend_from_slice(b"\x08_TMP\x0B");
    tz_body.extend_from_slice(&3132u16.to_le_bytes());
    tz_body.extend_from_slice(b"\x08_CRT\x0B");
    tz_body.extend_from_slice(&3732u16.to_le_bytes());
    tz_body.extend_from_slice(b"\x08_PSV\x0B");
    tz_body.extend_from_slice(&3582u16.to_le_bytes());
    tz_body.extend_from_slice(b"\x08_TZP\x0A\x64");
    let tz_zone = wrap_pkg(&[0x5B, 0x85], &tz_body);

    // Scope(\_SB_)
    let mut sb_body = Vec::new();
    sb_body.extend_from_slice(b"\x5C_SB_");
    sb_body.extend_from_slice(&pci0_dev);
    sb_body.extend_from_slice(&bat0_dev);
    sb_body.extend_from_slice(&ac_dev);
    sb_body.extend_from_slice(&tz_zone);
    let sb_scope = wrap_pkg(&[0x10], &sb_body);
    aml.extend_from_slice(&sb_scope);

    // 3. Validate DSDT table & parse AML namespace
    let mut dsdt = build_synthetic_dsdt(&aml);
    let (hdr, bytecode) = validate_aml_table(&dsdt).map_err(|e| e.as_str())?;
    if hdr.kind != AmlTableKind::Dsdt {
        return Err("aml self_test: expected DSDT kind");
    }

    let mut ns = AmlNamespace::new();
    ns.parse_aml(bytecode).map_err(|e| e.as_str())?;

    // Negative checksum test
    dsdt[9] ^= 0xFF;
    if validate_aml_table(&dsdt) != Err(AmlError::BadChecksum) {
        return Err("aml self_test: bad checksum not detected");
    }

    if ns.devices.len() != 3 || ns.methods.len() != 1 || ns.thermal_zones.len() != 1 {
        return Err("aml self_test: unexpected namespace object counts");
    }

    // 5. Verify _PRT extraction
    let prt = extract_prt_entries(&ns);
    if prt.len() != 2 {
        return Err("aml self_test: expected 2 _PRT entries");
    }
    let irq1 =
        resolve_pci_irq(&prt, 1, PciIntPin::IntA).ok_or("aml self_test: missing PCI dev 1 INTA")?;
    if irq1.source != PrtSource::Gsi(16) {
        return Err("aml self_test: unexpected GSI for dev 1 INTA");
    }
    let irq2 =
        resolve_pci_irq(&prt, 2, PciIntPin::IntB).ok_or("aml self_test: missing PCI dev 2 INTB")?;
    match &irq2.source {
        PrtSource::LinkDevice { path, index }
            if path.to_string_path() == "\\_SB_.LNKA" && *index == 0 => {}
        _ => return Err("aml self_test: unexpected LinkDevice for dev 2 INTB"),
    }

    // 6. Verify power/thermal extraction
    let pt = extract_power_thermal_info(&ns);
    if pt.thermal_zones.len() != 1 {
        return Err("aml self_test: expected 1 thermal zone");
    }
    let tz = &pt.thermal_zones[0];
    if tz.tmp_decikelvin != Some(3132)
        || decikelvin_to_celsius_tenths(3132) != 400
        || tz.crt_decikelvin != Some(3732)
    {
        return Err("aml self_test: thermal zone values mismatch");
    }
    if pt.batteries.len() != 1
        || pt.batteries[0].design_capacity != Some(50_000)
        || pt.batteries[0].remaining_capacity != Some(45_000)
    {
        return Err("aml self_test: battery descriptor mismatch");
    }
    if pt.ac_adapters.len() != 1 || pt.ac_adapters[0].psr_online != Some(true) {
        return Err("aml self_test: AC adapter descriptor mismatch");
    }
    if pt.sleep_states.len() != 1
        || pt.sleep_states[0].state != 5
        || pt.sleep_states[0].slp_typa != 5
    {
        return Err("aml self_test: _S5 sleep state mismatch");
    }

    Ok(())
}
