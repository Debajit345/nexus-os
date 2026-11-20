//! TPM 2.0 Measured Boot, PCR Extend/Quote & Kernel Keyring / Signature Verifier.
//!
//! Provides:
//! - TPM 2.0 CRB (Command Response Buffer) and TIS (TPM Interface Specification)
//!   MMIO transport + ACPI `TPM2` table parser + software TPM 2.0 emulator fallback
//! - TPM 2.0 command/response framing (`Startup`, `SelfTest`, `GetRandom`,
//!   `PCR_Read`, `PCR_Extend`, `Quote`)
//! - Pure `no_std` SHA-256, SHA-512, HMAC-SHA256, and RFC 8032 Ed25519 verifier
//! - 24-register SHA-256 PCR bank, TCG-style measured boot event log & replay verifier
//! - Kernel Keyring with key revocation, usage enforcement, and `NXSG` signed image verifier
//! - `self_test()` verification suite
#![allow(dead_code)]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::ptr::{read_unaligned, read_volatile, write_volatile};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::mm::SpinLock;

// ---------------------------------------------------------------------------
// Constants: TPM 2.0 MMIO, TIS, CRB, Commands & Algorithms
// ---------------------------------------------------------------------------

pub const TPM_MMIO_BASE: u64 = 0xFED4_0000;
pub const TPM_LOCALITY_SIZE: u64 = 0x1000;
pub const NUM_PCRS: usize = 24;

// Standard Measured Boot PCR assignments
pub const PCR_SRTM_BIOS: u8 = 0;
pub const PCR_PLATFORM_CONFIG: u8 = 1;
pub const PCR_OPTION_ROM: u8 = 2;
pub const PCR_BOOT_LOADER: u8 = 4;
pub const PCR_BOOT_CONFIG: u8 = 5;
pub const PCR_KERNEL_IMAGE: u8 = 8;
pub const PCR_KERNEL_CMDLINE: u8 = 9;
pub const PCR_INITRD: u8 = 10;
pub const PCR_KERNEL_MODULES: u8 = 11;
pub const PCR_POLICY: u8 = 12;

// TIS Register Offsets (Locality 0)
const TIS_ACCESS: usize = 0x0000;
const TIS_INT_ENABLE: usize = 0x0008;
const TIS_INT_STATUS: usize = 0x0010;
const TIS_INTF_CAPABILITY: usize = 0x0014;
const TIS_STS: usize = 0x0018;
const TIS_DATA_FIFO: usize = 0x0024;
const TIS_INTERFACE_ID: usize = 0x0030;
const TIS_DID_VID: usize = 0x0F00;
const TIS_RID: usize = 0x0F04;

// TIS_ACCESS bits
const TIS_ACCESS_VALID: u8 = 1 << 7;
const TIS_ACCESS_ACTIVE_LOCALITY: u8 = 1 << 5;
const TIS_ACCESS_REQUEST_USE: u8 = 1 << 1;

// TIS_STS bits
const TIS_STS_VALID: u32 = 1 << 7;
const TIS_STS_COMMAND_READY: u32 = 1 << 6;
const TIS_STS_GO: u32 = 1 << 5;
const TIS_STS_DATA_AVAIL: u32 = 1 << 4;
const TIS_STS_EXPECT: u32 = 1 << 3;

// CRB Register Offsets (Locality 0)
const CRB_LOC_STATE: usize = 0x0000;
const CRB_LOC_CTRL: usize = 0x0008;
const CRB_LOC_STS: usize = 0x000C;
const CRB_INTF_ID: usize = 0x0030;
const CRB_CTRL_REQ: usize = 0x0040;
const CRB_CTRL_STS: usize = 0x0044;
const CRB_CTRL_CANCEL: usize = 0x0048;
const CRB_CTRL_START: usize = 0x004C;
const CRB_CTRL_CMD_SIZE: usize = 0x0058;
const CRB_CTRL_CMD_LADDR: usize = 0x005C;
const CRB_CTRL_CMD_HADDR: usize = 0x0060;
const CRB_CTRL_RSP_SIZE: usize = 0x0064;
const CRB_CTRL_RSP_LADDR: usize = 0x0068;
const CRB_CTRL_RSP_HADDR: usize = 0x006C;
const CRB_DATA_BUFFER: usize = 0x0080;

const CRB_LOC_STATE_ASSIGNED: u32 = 1 << 1;
const CRB_LOC_STATE_VALID: u32 = 1 << 7;
const CRB_LOC_CTRL_REQUEST: u32 = 1 << 0;
const CRB_LOC_CTRL_RELINQUISH: u32 = 1 << 1;
const CRB_CTRL_REQ_CMD_READY: u32 = 1 << 0;
const CRB_CTRL_REQ_GO_IDLE: u32 = 1 << 1;
const CRB_CTRL_STS_ERROR: u32 = 1 << 0;
const CRB_CTRL_STS_IDLE: u32 = 1 << 1;
const CRB_CTRL_START_BIT: u32 = 1 << 0;

// TPM 2.0 Tags, Commands, Algorithms, Return Codes
pub const TPM_ST_NO_SESSIONS: u16 = 0x8001;
pub const TPM_ST_SESSIONS: u16 = 0x8002;
pub const TPM_ST_ATTEST_QUOTE: u16 = 0x8018;
pub const TPM_GENERATED_VALUE: u32 = 0xFF54_4347; // "\xffTCG"

pub const TPM2_CC_SELF_TEST: u32 = 0x0000_0143;
pub const TPM2_CC_STARTUP: u32 = 0x0000_0144;
pub const TPM2_CC_QUOTE: u32 = 0x0000_0158;
pub const TPM2_CC_GET_CAPABILITY: u32 = 0x0000_017A;
pub const TPM2_CC_GET_RANDOM: u32 = 0x0000_017B;
pub const TPM2_CC_PCR_READ: u32 = 0x0000_017E;
pub const TPM2_CC_PCR_EXTEND: u32 = 0x0000_0182;

pub const TPM2_SU_CLEAR: u16 = 0x0000;
pub const TPM2_SU_STATE: u16 = 0x0001;

pub const TPM_ALG_SHA256: u16 = 0x000B;
pub const TPM_ALG_NULL: u16 = 0x0010;
pub const TPM_ALG_EDDSA: u16 = 0x0027;

pub const TPM_RC_SUCCESS: u32 = 0x0000_0000;
pub const TPM_RC_INITIALIZE: u32 = 0x0000_0100;
pub const TPM_RC_FAILURE: u32 = 0x0000_0101;
pub const TPM_RC_VALUE: u32 = 0x0000_0084;
pub const TPM_RC_COMMAND_CODE: u32 = 0x0000_0143;

const TPM_RH_NULL: u32 = 0x4000_0007;
const TPM_POLL_LIMIT: usize = 100_000;

const NXSG_MAGIC: [u8; 4] = *b"NXSG";
const NXSG_VERSION: u16 = 1;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpmError {
    NotInitialized,
    DeviceNotFound,
    LocalityBusy,
    Timeout,
    InvalidResponse,
    CommandFailed(u32),
    InvalidPcr(u8),
    BufferTooSmall,
    KeyNotFound(u32),
    KeyRevoked(u32),
    InvalidKeyFormat,
    InvalidSignature,
    UnsupportedAlgorithm,
    EventLogCorrupted,
    QuoteVerificationFailed,
}

impl TpmError {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotInitialized => "TPM subsystem not initialized",
            Self::DeviceNotFound => "TPM hardware device not found",
            Self::LocalityBusy => "TPM locality busy",
            Self::Timeout => "TPM MMIO operation timed out",
            Self::InvalidResponse => "invalid TPM 2.0 response frame",
            Self::CommandFailed(_) => "TPM 2.0 command returned error code",
            Self::InvalidPcr(_) => "invalid PCR index (must be 0..23)",
            Self::BufferTooSmall => "TPM command/response buffer too small",
            Self::KeyNotFound(_) => "keyring key not found",
            Self::KeyRevoked(_) => "keyring key has been revoked",
            Self::InvalidKeyFormat => "invalid public key or key length",
            Self::InvalidSignature => "cryptographic signature verification failed",
            Self::UnsupportedAlgorithm => "unsupported cryptographic algorithm",
            Self::EventLogCorrupted => "measured boot event log replay mismatch",
            Self::QuoteVerificationFailed => "TPM quote attestation verification failed",
        }
    }
}

impl fmt::Display for TpmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandFailed(rc) => write!(f, "TPM 2.0 command failed (rc={:#010x})", rc),
            Self::InvalidPcr(pcr) => write!(f, "invalid PCR index {}", pcr),
            Self::KeyNotFound(id) => write!(f, "keyring key {} not found", id),
            Self::KeyRevoked(id) => write!(f, "keyring key {} is revoked", id),
            _ => f.write_str(self.as_str()),
        }
    }
}

impl From<TpmError> for &'static str {
    fn from(err: TpmError) -> Self {
        err.as_str()
    }
}

// ---------------------------------------------------------------------------
// Constant-Time Comparison
// ---------------------------------------------------------------------------

#[inline]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// Pure no_std SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------

const SHA256_IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    bit_len: u64,
    buf: [u8; 64],
    buf_len: usize,
}

impl Sha256 {
    pub const fn new() -> Self {
        Self {
            state: SHA256_IV,
            bit_len: 0,
            buf: [0u8; 64],
            buf_len: 0,
        }
    }

    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
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
                let block = self.buf;
                Self::compress(&mut self.state, &block);
                self.buf_len = 0;
            }
        }

        while offset + 64 <= data.len() {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[offset..offset + 64]);
            Self::compress(&mut self.state, &block);
            offset += 64;
        }

        if offset < data.len() {
            let rem = data.len() - offset;
            self.buf[..rem].copy_from_slice(&data[offset..]);
            self.buf_len = rem;
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let total_bits = self.bit_len;
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if self.buf_len > 56 {
            self.buf[self.buf_len..64].fill(0);
            let block = self.buf;
            Self::compress(&mut self.state, &block);
            self.buf_len = 0;
        }

        self.buf[self.buf_len..56].fill(0);
        self.buf[56..64].copy_from_slice(&total_bits.to_be_bytes());
        let block = self.buf;
        Self::compress(&mut self.state, &block);

        let mut out = [0u8; 32];
        for (chunk, &word) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.state.iter()) {
            *chunk = word.to_be_bytes();
        }
        out
    }

    fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (dst, &chunk) in w[..16].iter_mut().zip(block.as_chunks::<4>().0.iter()) {
            *dst = u32::from_be_bytes(chunk);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for (&k, &wi) in SHA256_K.iter().zip(w.iter()) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k)
                .wrapping_add(wi);
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

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
}

pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut k_pad = [0u8; 64];
    if key.len() > 64 {
        let hashed = Sha256::digest(key);
        k_pad[..32].copy_from_slice(&hashed);
    } else {
        k_pad[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for (dst, &k) in ipad.iter_mut().zip(k_pad.iter()) {
        *dst ^= k;
    }
    for (dst, &k) in opad.iter_mut().zip(k_pad.iter()) {
        *dst ^= k;
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

// ---------------------------------------------------------------------------
// Pure no_std SHA-512 (FIPS 180-4, used by Ed25519)
// ---------------------------------------------------------------------------

const SHA512_IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

const SHA512_K: [u64; 80] = [
    0x428a_2f98_d728_ae22,
    0x7137_4491_23ef_65cd,
    0xb5c0_fbcf_ec4d_3b2f,
    0xe9b5_dba5_8189_dbbc,
    0x3956_c25b_f348_b538,
    0x59f1_11f1_b605_d019,
    0x923f_82a4_af19_4f9b,
    0xab1c_5ed5_da6d_8118,
    0xd807_aa98_a303_0242,
    0x1283_5b01_4570_6fbe,
    0x2431_85be_4ee4_b28c,
    0x550c_7dc3_d5ff_b4e2,
    0x72be_5d74_f27b_896f,
    0x80de_b1fe_3b16_96b1,
    0x9bdc_06a7_25c7_1235,
    0xc19b_f174_cf69_2694,
    0xe49b_69c1_9ef1_4ad2,
    0xefbe_4786_384f_25e3,
    0x0fc1_9dc6_8b8c_d5b5,
    0x240c_a1cc_77ac_9c65,
    0x2de9_2c6f_592b_0275,
    0x4a74_84aa_6ea6_e483,
    0x5cb0_a9dc_bd41_fbd4,
    0x76f9_88da_8311_53b5,
    0x983e_5152_ee66_dfab,
    0xa831_c66d_2db4_3210,
    0xb003_27c8_98fb_213f,
    0xbf59_7fc7_beef_0ee4,
    0xc6e0_0bf3_3da8_8fc2,
    0xd5a7_9147_930a_a725,
    0x06ca_6351_e003_826f,
    0x1429_2967_0a0e_6e70,
    0x27b7_0a85_46d2_2ffc,
    0x2e1b_2138_5c26_c926,
    0x4d2c_6dfc_5ac4_2aed,
    0x5338_0d13_9d95_b3df,
    0x650a_7354_8baf_63de,
    0x766a_0abb_3c77_b2a8,
    0x81c2_c92e_47ed_aee6,
    0x9272_2c85_1482_353b,
    0xa2bf_e8a1_4cf1_0364,
    0xa81a_664b_bc42_3001,
    0xc24b_8b70_d0f8_9791,
    0xc76c_51a3_0654_be30,
    0xd192_e819_d6ef_5218,
    0xd699_0624_5565_a910,
    0xf40e_3585_5771_202a,
    0x106a_a070_32bb_d1b8,
    0x19a4_c116_b8d2_d0c8,
    0x1e37_6c08_5141_ab53,
    0x2748_774c_df8e_eb99,
    0x34b0_bcb5_e19b_48a8,
    0x391c_0cb3_c5c9_5a63,
    0x4ed8_aa4a_e341_8acb,
    0x5b9c_ca4f_7763_e373,
    0x682e_6ff3_d6b2_b8a3,
    0x748f_82ee_5def_b2fc,
    0x78a5_636f_4317_2f60,
    0x84c8_7814_a1f0_ab72,
    0x8cc7_0208_1a64_39ec,
    0x90be_fffa_2363_1e28,
    0xa450_6ceb_de82_bde9,
    0xbef9_a3f7_b2c6_7915,
    0xc671_78f2_e372_532b,
    0xca27_3ece_ea26_619c,
    0xd186_b8c7_21c0_c207,
    0xeada_7dd6_cde0_eb1e,
    0xf57d_4f7f_ee6e_d178,
    0x06f0_67aa_7217_6fba,
    0x0a63_7dc5_a2c8_98a6,
    0x113f_9804_bef9_0dae,
    0x1b71_0b35_131c_471b,
    0x28db_77f5_2304_7d84,
    0x32ca_ab7b_40c7_2493,
    0x3c9e_be0a_15c9_bebc,
    0x431d_67c4_9c10_0d4c,
    0x4cc5_d4be_cb3e_42b6,
    0x597f_299c_fc65_7e2a,
    0x5fcb_6fab_3ad6_faec,
    0x6c44_198c_4a47_5817,
];

struct Sha512 {
    state: [u64; 8],
    bit_len: u128,
    buf: [u8; 128],
    buf_len: usize,
}

impl Sha512 {
    const fn new() -> Self {
        Self {
            state: SHA512_IV,
            bit_len: 0,
            buf: [0u8; 128],
            buf_len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.bit_len = self.bit_len.wrapping_add((data.len() as u128) * 8);
        let mut offset = 0usize;

        if self.buf_len > 0 {
            let take = (128 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            offset += take;
            if self.buf_len == 128 {
                let block = self.buf;
                Self::compress(&mut self.state, &block);
                self.buf_len = 0;
            }
        }

        while offset + 128 <= data.len() {
            let mut block = [0u8; 128];
            block.copy_from_slice(&data[offset..offset + 128]);
            Self::compress(&mut self.state, &block);
            offset += 128;
        }

        if offset < data.len() {
            let rem = data.len() - offset;
            self.buf[..rem].copy_from_slice(&data[offset..]);
            self.buf_len = rem;
        }
    }

    fn finalize(mut self) -> [u8; 64] {
        let total_bits = self.bit_len;
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if self.buf_len > 112 {
            self.buf[self.buf_len..128].fill(0);
            let block = self.buf;
            Self::compress(&mut self.state, &block);
            self.buf_len = 0;
        }

        self.buf[self.buf_len..112].fill(0);
        self.buf[112..128].copy_from_slice(&total_bits.to_be_bytes());
        let block = self.buf;
        Self::compress(&mut self.state, &block);

        let mut out = [0u8; 64];
        for (chunk, &word) in out.as_chunks_mut::<8>().0.iter_mut().zip(self.state.iter()) {
            *chunk = word.to_be_bytes();
        }
        out
    }

    fn compress(state: &mut [u64; 8], block: &[u8; 128]) {
        let mut w = [0u64; 80];
        for (dst, &chunk) in w[..16].iter_mut().zip(block.as_chunks::<8>().0.iter()) {
            *dst = u64::from_be_bytes(chunk);
        }
        for i in 16..80 {
            let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
            let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for (&k, &wi) in SHA512_K.iter().zip(w.iter()) {
            let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k)
                .wrapping_add(wi);
            let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
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

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
}

// ---------------------------------------------------------------------------
// Pure no_std Ed25519 Signature Verifier (RFC 8032)
// ---------------------------------------------------------------------------

const MASK51: u64 = (1u64 << 51) - 1;

/// Field element in GF(2^255 - 19) in radix 2^51.
#[derive(Clone, Copy)]
struct Fe([u64; 5]);

impl Fe {
    const ZERO: Self = Self([0, 0, 0, 0, 0]);
    const ONE: Self = Self([1, 0, 0, 0, 0]);

    /// d = -121665 / 121666 mod p
    const EDWARDS_D: Self = Self([
        0x0003_4dca_1359_78a3,
        0x0001_a828_3b15_6ebd,
        0x0005_e7a2_6001_c029,
        0x0007_39c6_63a0_3cbb,
        0x0005_2036_cee2_b6ff,
    ]);

    /// sqrt(-1) mod p = 2^((p-1)/4) mod p
    const SQRT_M1: Self = Self([
        0x0006_1b27_4a0e_a0b0,
        0x0000_d5a5_fc8f_189d,
        0x0007_ef5e_9cbd_0c60,
        0x0007_8595_a680_4c9e,
        0x0002_b832_4804_fc1d,
    ]);

    fn from_bytes(s: &[u8; 32]) -> Self {
        let load64 = |off: usize| -> u64 {
            u64::from_le_bytes([
                s[off],
                s[off + 1],
                s[off + 2],
                s[off + 3],
                s[off + 4],
                s[off + 5],
                s[off + 6],
                s[off + 7],
            ])
        };
        let t0 = load64(0);
        let t1 = load64(6) >> 3;
        let t2 = load64(12) >> 6;
        let t3 = load64(19) >> 1;
        let t4 = (load64(24) >> 12) & ((1u64 << 51) - 1);
        Self([
            t0 & MASK51,
            t1 & MASK51,
            t2 & MASK51,
            t3 & MASK51,
            t4 & ((1u64 << 50) - 1),
        ])
    }

    fn reduce(mut h: [u64; 5]) -> Self {
        let mut c = h[0] >> 51;
        h[0] &= MASK51;
        h[1] += c;
        c = h[1] >> 51;
        h[1] &= MASK51;
        h[2] += c;
        c = h[2] >> 51;
        h[2] &= MASK51;
        h[3] += c;
        c = h[3] >> 51;
        h[3] &= MASK51;
        h[4] += c;
        c = h[4] >> 51;
        h[4] &= MASK51;
        h[0] += c * 19;
        c = h[0] >> 51;
        h[0] &= MASK51;
        h[1] += c;
        Self(h)
    }

    fn to_bytes(self) -> [u8; 32] {
        let mut h = Self::reduce(self.0).0;
        // Canonical reduction mod p = 2^255 - 19
        let mut q = (h[0] + 19) >> 51;
        q = (h[1] + q) >> 51;
        q = (h[2] + q) >> 51;
        q = (h[3] + q) >> 51;
        q = (h[4] + q) >> 51;

        h[0] += 19 * q;
        let mut c = h[0] >> 51;
        h[0] &= MASK51;
        h[1] += c;
        c = h[1] >> 51;
        h[1] &= MASK51;
        h[2] += c;
        c = h[2] >> 51;
        h[2] &= MASK51;
        h[3] += c;
        c = h[3] >> 51;
        h[3] &= MASK51;
        h[4] += c;
        h[4] &= MASK51;

        let mut out = [0u8; 32];
        let w0 = h[0] | (h[1] << 51);
        let w1 = (h[1] >> 13) | (h[2] << 38);
        let w2 = (h[2] >> 26) | (h[3] << 25);
        let w3 = (h[3] >> 39) | (h[4] << 12);
        out[0..8].copy_from_slice(&w0.to_le_bytes());
        out[8..16].copy_from_slice(&w1.to_le_bytes());
        out[16..24].copy_from_slice(&w2.to_le_bytes());
        out[24..32].copy_from_slice(&w3.to_le_bytes());
        out
    }

    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
        ])
    }

    fn sub(self, rhs: Self) -> Self {
        // Add 2*p to avoid underflow (each limb of 2*p is 2*MASK51, except limb 0 is 2*(MASK51-18))
        let r = Self::reduce(rhs.0).0;
        Self::reduce([
            self.0[0] + 0x000f_ffff_ffff_ffda - r[0],
            self.0[1] + 0x000f_ffff_ffff_fffe - r[1],
            self.0[2] + 0x000f_ffff_ffff_fffe - r[2],
            self.0[3] + 0x000f_ffff_ffff_fffe - r[3],
            self.0[4] + 0x000f_ffff_ffff_fffe - r[4],
        ])
    }

    fn neg(self) -> Self {
        Self::ZERO.sub(self)
    }

    fn mul(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;
        let a0 = a[0] as u128;
        let a1 = a[1] as u128;
        let a2 = a[2] as u128;
        let a3 = a[3] as u128;
        let a4 = a[4] as u128;

        let b0 = b[0] as u128;
        let b1 = b[1] as u128;
        let b2 = b[2] as u128;
        let b3 = b[3] as u128;
        let b4 = b[4] as u128;

        let b1_19 = b1 * 19;
        let b2_19 = b2 * 19;
        let b3_19 = b3 * 19;
        let b4_19 = b4 * 19;

        let mut t0 = a0 * b0 + a1 * b4_19 + a2 * b3_19 + a3 * b2_19 + a4 * b1_19;
        let mut t1 = a0 * b1 + a1 * b0 + a2 * b4_19 + a3 * b3_19 + a4 * b2_19;
        let mut t2 = a0 * b2 + a1 * b1 + a2 * b0 + a3 * b4_19 + a4 * b3_19;
        let mut t3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + a4 * b4_19;
        let mut t4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

        let mask = MASK51 as u128;
        t1 += t0 >> 51;
        let r0 = (t0 & mask) as u64;
        t2 += t1 >> 51;
        let r1 = (t1 & mask) as u64;
        t3 += t2 >> 51;
        let r2 = (t2 & mask) as u64;
        t4 += t3 >> 51;
        let r3 = (t3 & mask) as u64;
        t0 = (r0 as u128) + (t4 >> 51) * 19;
        let r4 = (t4 & mask) as u64;
        let r0_final = (t0 & mask) as u64;
        let r1_final = r1 + ((t0 >> 51) as u64);

        Self([r0_final, r1_final, r2, r3, r4])
    }

    fn sq(self) -> Self {
        self.mul(self)
    }

    fn pow22523(self) -> Self {
        let mut t0 = self.sq();
        let mut t1 = t0.sq().sq();
        t1 = self.mul(t1);
        t0 = t0.mul(t1);
        t0 = t0.sq();
        t0 = t1.mul(t0);
        t1 = t0.sq();
        for _ in 1..5 {
            t1 = t1.sq();
        }
        t0 = t1.mul(t0);
        t1 = t0.sq();
        for _ in 1..10 {
            t1 = t1.sq();
        }
        t1 = t1.mul(t0);
        let mut t2 = t1.sq();
        for _ in 1..20 {
            t2 = t2.sq();
        }
        t1 = t2.mul(t1);
        t1 = t1.sq();
        for _ in 1..10 {
            t1 = t1.sq();
        }
        t0 = t1.mul(t0);
        t1 = t0.sq();
        for _ in 1..50 {
            t1 = t1.sq();
        }
        t1 = t1.mul(t0);
        t2 = t1.sq();
        for _ in 1..100 {
            t2 = t2.sq();
        }
        t1 = t2.mul(t1);
        t1 = t1.sq();
        for _ in 1..50 {
            t1 = t1.sq();
        }
        t0 = t1.mul(t0);
        t0 = t0.sq().sq();
        t0.mul(self)
    }

    fn invert(self) -> Self {
        // a^(p-2) = a^(2^255 - 21)
        let t0 = self.sq();
        let mut t1 = t0.sq().sq();
        t1 = self.mul(t1);
        let t0 = t0.mul(t1);
        let t2 = t0.sq();
        let t1 = t1.mul(t2);
        let mut t2 = t1.sq();
        for _ in 1..5 {
            t2 = t2.sq();
        }
        let t1 = t2.mul(t1);
        let mut t2 = t1.sq();
        for _ in 1..10 {
            t2 = t2.sq();
        }
        let t2 = t2.mul(t1);
        let mut t3 = t2.sq();
        for _ in 1..20 {
            t3 = t3.sq();
        }
        let t2 = t3.mul(t2);
        let mut t2 = t2.sq();
        for _ in 1..10 {
            t2 = t2.sq();
        }
        let t1 = t2.mul(t1);
        let mut t2 = t1.sq();
        for _ in 1..50 {
            t2 = t2.sq();
        }
        let t2 = t2.mul(t1);
        let mut t3 = t2.sq();
        for _ in 1..100 {
            t3 = t3.sq();
        }
        let t2 = t3.mul(t2);
        let mut t2 = t2.sq();
        for _ in 1..50 {
            t2 = t2.sq();
        }
        let mut t1 = t2.mul(t1);
        for _ in 0..5 {
            t1 = t1.sq();
        }
        t1.mul(t0)
    }

    fn is_zero(self) -> bool {
        self.to_bytes().iter().all(|&b| b == 0)
    }

    fn is_negative(self) -> bool {
        (self.to_bytes()[0] & 1) != 0
    }
}

#[derive(Clone, Copy)]
struct GeExtended {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

impl GeExtended {
    const IDENTITY: Self = Self {
        x: Fe::ZERO,
        y: Fe::ONE,
        z: Fe::ONE,
        t: Fe::ZERO,
    };

    /// Ed25519 basepoint B
    fn basepoint() -> Self {
        // y = 4/5 mod p
        let y_bytes: [u8; 32] = [
            0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
            0x66, 0x66, 0x66, 0x66,
        ];
        Self::from_bytes(&y_bytes).unwrap_or(Self::IDENTITY)
    }

    fn from_bytes(s: &[u8; 32]) -> Option<Self> {
        let y = Fe::from_bytes(s);
        let y2 = y.sq();
        let u = y2.sub(Fe::ONE);
        let v = y2.mul(Fe::EDWARDS_D).add(Fe::ONE);
        let v3 = v.sq().mul(v);
        let v7 = v3.sq().mul(v);
        let mut x = u.mul(v3).mul(u.mul(v7).pow22523());

        let vx2 = v.mul(x.sq());
        if !ct_eq(&vx2.to_bytes(), &u.to_bytes()) {
            if ct_eq(&vx2.to_bytes(), &u.neg().to_bytes()) {
                x = x.mul(Fe::SQRT_M1);
            } else {
                return None;
            }
        }

        let sign = (s[31] >> 7) != 0;
        if x.is_zero() && sign {
            return None;
        }
        if x.is_negative() != sign {
            x = x.neg();
        }

        Some(Self {
            x,
            y,
            z: Fe::ONE,
            t: x.mul(y),
        })
    }

    fn to_bytes(self) -> [u8; 32] {
        let recip = self.z.invert();
        let x = self.x.mul(recip);
        let y = self.y.mul(recip);
        let mut out = y.to_bytes();
        if x.is_negative() {
            out[31] |= 0x80;
        }
        out
    }

    fn double(self) -> Self {
        let a = self.x.sq();
        let b = self.y.sq();
        let c = self.z.sq().add(self.z.sq());
        let d = a.neg();
        let e = self.x.add(self.y).sq().sub(a).sub(b);
        let g = d.add(b);
        let f = g.sub(c);
        let h = d.sub(b);
        Self {
            x: e.mul(f),
            y: g.mul(h),
            z: f.mul(g),
            t: e.mul(h),
        }
    }

    fn add(self, rhs: Self) -> Self {
        let a = self.y.sub(self.x).mul(rhs.y.sub(rhs.x));
        let b = self.y.add(self.x).mul(rhs.y.add(rhs.x));
        let c = self
            .t
            .mul(rhs.t)
            .mul(Fe::EDWARDS_D)
            .add(self.t.mul(rhs.t).mul(Fe::EDWARDS_D));
        let d = self.z.mul(rhs.z).add(self.z.mul(rhs.z));
        let e = b.sub(a);
        let f = d.sub(c);
        let g = d.add(c);
        let h = b.add(a);
        Self {
            x: e.mul(f),
            y: g.mul(h),
            z: f.mul(g),
            t: e.mul(h),
        }
    }

    fn scalar_mul(self, scalar: &[u8; 32]) -> Self {
        let mut acc = Self::IDENTITY;
        for &byte in scalar.iter().rev() {
            for bit in (0..8).rev() {
                acc = acc.double();
                if ((byte >> bit) & 1) != 0 {
                    acc = acc.add(self);
                }
            }
        }
        acc
    }
}

/// Reduce a 64-byte little-endian integer modulo L = 2^252 + 27742317777372353535851937790883648493.
fn sc_reduce_64(input: &[u8; 64]) -> [u8; 32] {
    // L in little-endian 32-bit limbs (8 limbs)
    const L: [u64; 8] = [
        0x5cf5_d3ed,
        0x5812_631a,
        0xa2f7_9cd6,
        0x14de_f9de,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x1000_0000,
    ];

    // Bit-by-bit shift-subtract from bit 511 down to 252.
    let mut rem = [0u32; 16];
    for (dst, &chunk) in rem.iter_mut().zip(input.as_chunks::<4>().0.iter()) {
        *dst = u32::from_le_bytes(chunk);
    }

    for shift in (0..=259).rev() {
        let word_off = shift / 32;
        let bit_off = shift % 32;

        // Build shifted L (9 words max) and compare with rem[word_off..]
        let mut shifted = [0u32; 9];
        let mut carry = 0u64;
        for (dst, &lw) in shifted[..8].iter_mut().zip(L.iter()) {
            let v = (lw << bit_off) | carry;
            *dst = v as u32;
            carry = v >> 32;
        }
        shifted[8] = carry as u32;

        // Check if rem >= (L << shift)
        let mut ge = true;
        let top_rem = if word_off + 8 < 16 {
            rem[word_off + 8]
        } else {
            0
        };
        // Any non-zero word above word_off + 8 means rem > shifted
        let has_higher = rem.iter().skip(word_off + 9).any(|&w| w != 0);
        if !has_higher {
            if top_rem != shifted[8] {
                ge = top_rem > shifted[8];
            } else {
                for idx in (0..8).rev() {
                    let rw = if word_off + idx < 16 {
                        rem[word_off + idx]
                    } else {
                        0
                    };
                    if rw != shifted[idx] {
                        ge = rw > shifted[idx];
                        break;
                    }
                }
            }
        }

        if ge {
            let mut borrow = 0i64;
            for (idx, &sw) in shifted.iter().enumerate() {
                if word_off + idx >= 16 {
                    break;
                }
                let diff = (rem[word_off + idx] as i64) - (sw as i64) - borrow;
                if diff < 0 {
                    rem[word_off + idx] = (diff + (1i64 << 32)) as u32;
                    borrow = 1;
                } else {
                    rem[word_off + idx] = diff as u32;
                    borrow = 0;
                }
            }
        }
    }

    let mut out = [0u8; 32];
    for (chunk, &w) in out.as_chunks_mut::<4>().0.iter_mut().zip(rem[..8].iter()) {
        *chunk = w.to_le_bytes();
    }
    out
}

fn sc_is_canonical(s: &[u8; 32]) -> bool {
    // L in little-endian bytes
    const L_BYTES: [u8; 32] = [
        0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde,
        0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x10,
    ];
    for (&sb, &lb) in s.iter().rev().zip(L_BYTES.iter().rev()) {
        if sb < lb {
            return true;
        }
        if sb > lb {
            return false;
        }
    }
    false
}

/// Verify an RFC 8032 Ed25519 signature (32-byte pubkey, 64-byte signature).
pub fn ed25519_verify(pubkey: &[u8], message: &[u8], signature: &[u8]) -> Result<(), TpmError> {
    if pubkey.len() != 32 {
        return Err(TpmError::InvalidKeyFormat);
    }
    if signature.len() != 64 {
        return Err(TpmError::InvalidSignature);
    }

    let mut pk_bytes = [0u8; 32];
    pk_bytes.copy_from_slice(pubkey);
    let a_point = GeExtended::from_bytes(&pk_bytes).ok_or(TpmError::InvalidKeyFormat)?;

    let mut r_bytes = [0u8; 32];
    r_bytes.copy_from_slice(&signature[..32]);
    let mut s_bytes = [0u8; 32];
    s_bytes.copy_from_slice(&signature[32..64]);

    if !sc_is_canonical(&s_bytes) {
        return Err(TpmError::InvalidSignature);
    }
    let r_point = GeExtended::from_bytes(&r_bytes).ok_or(TpmError::InvalidSignature)?;

    let mut hasher = Sha512::new();
    hasher.update(&r_bytes);
    hasher.update(&pk_bytes);
    hasher.update(message);
    let h_digest = hasher.finalize();
    let h_scalar = sc_reduce_64(&h_digest);

    // Check [S]B == R + [h]A
    let lhs = GeExtended::basepoint().scalar_mul(&s_bytes);
    let rhs = r_point.add(a_point.scalar_mul(&h_scalar));

    if ct_eq(&lhs.to_bytes(), &rhs.to_bytes()) {
        Ok(())
    } else {
        Err(TpmError::InvalidSignature)
    }
}

// ---------------------------------------------------------------------------
// ACPI TPM2 Table Parser
// ---------------------------------------------------------------------------

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct AcpiSdtHeader {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem_id: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    creator_id: u32,
    creator_revision: u32,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct AcpiTpm2Raw {
    header: AcpiSdtHeader,
    platform_class: u16,
    reserved: u16,
    control_area_pa: u64,
    start_method: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpmInterfaceKind {
    Tis,
    Crb,
    Simulated,
}

#[derive(Debug, Clone, Copy)]
pub struct Tpm2AcpiInfo {
    pub platform_class: u16,
    pub control_area_pa: u64,
    pub start_method: u32,
    pub interface_kind: TpmInterfaceKind,
}

pub fn parse_acpi_tpm2_table(table_bytes: &[u8]) -> Option<Tpm2AcpiInfo> {
    if table_bytes.len() < core::mem::size_of::<AcpiTpm2Raw>() {
        return None;
    }
    let raw = unsafe { read_unaligned(table_bytes.as_ptr() as *const AcpiTpm2Raw) };
    if &raw.header.signature != b"TPM2" {
        return None;
    }
    let len = raw.header.length as usize;
    if len < core::mem::size_of::<AcpiTpm2Raw>() || len > table_bytes.len() {
        return None;
    }
    let sum = table_bytes[..len]
        .iter()
        .fold(0u8, |acc, &b| acc.wrapping_add(b));
    if sum != 0 {
        return None;
    }
    let interface_kind = match raw.start_method {
        6 => TpmInterfaceKind::Tis,
        7 | 8 | 11 => TpmInterfaceKind::Crb,
        _ => TpmInterfaceKind::Crb,
    };
    Some(Tpm2AcpiInfo {
        platform_class: raw.platform_class,
        control_area_pa: raw.control_area_pa,
        start_method: raw.start_method,
        interface_kind,
    })
}

// ---------------------------------------------------------------------------
// TPM 2.0 Command / Response Framing
// ---------------------------------------------------------------------------

pub struct TpmCommandBuilder {
    buf: Vec<u8>,
}

impl TpmCommandBuilder {
    pub fn new(tag: u16, command_code: u32) -> Self {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&tag.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes()); // placeholder for size
        buf.extend_from_slice(&command_code.to_be_bytes());
        Self { buf }
    }

    pub fn push_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn push_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn push_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn push_bytes(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    pub fn finish(mut self) -> Vec<u8> {
        let len = self.buf.len() as u32;
        self.buf[2..6].copy_from_slice(&len.to_be_bytes());
        self.buf
    }
}

pub struct TpmResponseReader<'a> {
    buf: &'a [u8],
    pos: usize,
    pub tag: u16,
    pub size: u32,
    pub rc: u32,
}

impl<'a> TpmResponseReader<'a> {
    pub fn parse(buf: &'a [u8]) -> Result<Self, TpmError> {
        if buf.len() < 10 {
            return Err(TpmError::InvalidResponse);
        }
        let tag = u16::from_be_bytes([buf[0], buf[1]]);
        let size = u32::from_be_bytes([buf[2], buf[3], buf[4], buf[5]]);
        let rc = u32::from_be_bytes([buf[6], buf[7], buf[8], buf[9]]);
        if (size as usize) > buf.len() || size < 10 {
            return Err(TpmError::InvalidResponse);
        }
        Ok(Self {
            buf: &buf[..size as usize],
            pos: 10,
            tag,
            size,
            rc,
        })
    }

    pub fn check_rc(&self) -> Result<(), TpmError> {
        if self.rc == TPM_RC_SUCCESS {
            Ok(())
        } else {
            Err(TpmError::CommandFailed(self.rc))
        }
    }

    pub fn read_u16(&mut self) -> Result<u16, TpmError> {
        if self.pos + 2 > self.buf.len() {
            return Err(TpmError::InvalidResponse);
        }
        let v = u16::from_be_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    pub fn read_u32(&mut self) -> Result<u32, TpmError> {
        if self.pos + 4 > self.buf.len() {
            return Err(TpmError::InvalidResponse);
        }
        let v = u32::from_be_bytes([
            self.buf[self.pos],
            self.buf[self.pos + 1],
            self.buf[self.pos + 2],
            self.buf[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    pub fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], TpmError> {
        if self.pos + len > self.buf.len() {
            return Err(TpmError::InvalidResponse);
        }
        let slice = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }
}

pub fn build_cmd_startup(startup_type: u16) -> Vec<u8> {
    let mut b = TpmCommandBuilder::new(TPM_ST_NO_SESSIONS, TPM2_CC_STARTUP);
    b.push_u16(startup_type);
    b.finish()
}

pub fn build_cmd_self_test(full_test: bool) -> Vec<u8> {
    let mut b = TpmCommandBuilder::new(TPM_ST_NO_SESSIONS, TPM2_CC_SELF_TEST);
    b.push_u8(if full_test { 1 } else { 0 });
    b.finish()
}

pub fn build_cmd_get_random(bytes_requested: u16) -> Vec<u8> {
    let mut b = TpmCommandBuilder::new(TPM_ST_NO_SESSIONS, TPM2_CC_GET_RANDOM);
    b.push_u16(bytes_requested);
    b.finish()
}

pub fn build_cmd_pcr_read(pcr_index: u8) -> Result<Vec<u8>, TpmError> {
    if (pcr_index as usize) >= NUM_PCRS {
        return Err(TpmError::InvalidPcr(pcr_index));
    }
    let mut b = TpmCommandBuilder::new(TPM_ST_NO_SESSIONS, TPM2_CC_PCR_READ);
    b.push_u32(1); // TPML_PCR_SELECTION count = 1
    b.push_u16(TPM_ALG_SHA256);
    b.push_u8(3); // sizeofSelect = 3 bytes (24 PCRs)
    let mut sel = [0u8; 3];
    sel[(pcr_index / 8) as usize] |= 1 << (pcr_index % 8);
    b.push_bytes(&sel);
    Ok(b.finish())
}

pub fn build_cmd_pcr_extend(pcr_index: u8, digest: &[u8; 32]) -> Result<Vec<u8>, TpmError> {
    if (pcr_index as usize) >= NUM_PCRS {
        return Err(TpmError::InvalidPcr(pcr_index));
    }
    let mut b = TpmCommandBuilder::new(TPM_ST_SESSIONS, TPM2_CC_PCR_EXTEND);
    b.push_u32(pcr_index as u32); // TPMI_DH_PCR handle
                                  // Empty password authorization session (TPM_RS_PW = 0x40000009)
    b.push_u32(9); // authorizationSize
    b.push_u32(0x4000_0009); // TPM_RS_PW
    b.push_u16(0); // nonceCaller size
    b.push_u8(0); // sessionAttributes
    b.push_u16(0); // hmac size
                   // TPML_DIGEST_VALUES
    b.push_u32(1); // count = 1
    b.push_u16(TPM_ALG_SHA256);
    b.push_bytes(digest);
    Ok(b.finish())
}

pub fn build_cmd_quote(pcr_mask: u32, nonce: &[u8]) -> Vec<u8> {
    let mut b = TpmCommandBuilder::new(TPM_ST_SESSIONS, TPM2_CC_QUOTE);
    b.push_u32(0x8101_0001); // signHandle (persistent AK handle)
    b.push_u32(9); // authorizationSize
    b.push_u32(0x4000_0009); // TPM_RS_PW
    b.push_u16(0);
    b.push_u8(0);
    b.push_u16(0);
    // qualifyingData (TPM2B_DATA)
    let nonce_len = nonce.len().min(64) as u16;
    b.push_u16(nonce_len);
    b.push_bytes(&nonce[..nonce_len as usize]);
    // inScheme (TPMT_SIG_SCHEME)
    b.push_u16(TPM_ALG_NULL);
    // PCRselect (TPML_PCR_SELECTION)
    b.push_u32(1);
    b.push_u16(TPM_ALG_SHA256);
    b.push_u8(3);
    b.push_u8((pcr_mask & 0xFF) as u8);
    b.push_u8(((pcr_mask >> 8) & 0xFF) as u8);
    b.push_u8(((pcr_mask >> 16) & 0xFF) as u8);
    b.finish()
}

// ---------------------------------------------------------------------------
// MMIO TIS & CRB Transports
// ---------------------------------------------------------------------------

#[inline]
unsafe fn mmio_read8(base: u64, offset: usize) -> u8 {
    read_volatile((base + offset as u64) as *const u8)
}

#[inline]
unsafe fn mmio_write8(base: u64, offset: usize, val: u8) {
    write_volatile((base + offset as u64) as *mut u8, val);
}

#[inline]
unsafe fn mmio_read32(base: u64, offset: usize) -> u32 {
    read_volatile((base + offset as u64) as *const u32)
}

#[inline]
unsafe fn mmio_write32(base: u64, offset: usize, val: u32) {
    write_volatile((base + offset as u64) as *mut u32, val);
}

pub fn probe_mmio_interface(hhdm_offset: u64, phys_base: u64) -> TpmInterfaceKind {
    let virt = phys_base.wrapping_add(hhdm_offset);
    let did_vid = unsafe { mmio_read32(virt, TIS_DID_VID) };
    if did_vid == 0xFFFF_FFFF || did_vid == 0x0000_0000 {
        return TpmInterfaceKind::Simulated;
    }
    let intf_id = unsafe { mmio_read32(virt, TIS_INTERFACE_ID) };
    let intf_type = intf_id & 0x0F;
    match intf_type {
        0x01 => TpmInterfaceKind::Crb,
        0x00 | 0x0F => TpmInterfaceKind::Tis,
        _ => TpmInterfaceKind::Tis,
    }
}

pub struct MmioTisTransport {
    virt_base: u64,
}

impl MmioTisTransport {
    pub fn new(hhdm_offset: u64, phys_base: u64) -> Self {
        Self {
            virt_base: phys_base.wrapping_add(hhdm_offset),
        }
    }

    fn request_locality(&self) -> Result<(), TpmError> {
        unsafe {
            mmio_write8(self.virt_base, TIS_ACCESS, TIS_ACCESS_REQUEST_USE);
            for _ in 0..TPM_POLL_LIMIT {
                let acc = mmio_read8(self.virt_base, TIS_ACCESS);
                if (acc & TIS_ACCESS_VALID) != 0 && (acc & TIS_ACCESS_ACTIVE_LOCALITY) != 0 {
                    return Ok(());
                }
                core::hint::spin_loop();
            }
        }
        Err(TpmError::LocalityBusy)
    }

    fn release_locality(&self) {
        unsafe {
            mmio_write8(self.virt_base, TIS_ACCESS, TIS_ACCESS_ACTIVE_LOCALITY);
        }
    }

    fn wait_sts(&self, mask: u32, expected: u32) -> Result<u32, TpmError> {
        unsafe {
            for _ in 0..TPM_POLL_LIMIT {
                let sts = mmio_read32(self.virt_base, TIS_STS);
                if (sts & TIS_STS_VALID) != 0 && (sts & mask) == expected {
                    return Ok(sts);
                }
                core::hint::spin_loop();
            }
        }
        Err(TpmError::Timeout)
    }

    pub fn submit(&self, cmd: &[u8]) -> Result<Vec<u8>, TpmError> {
        if cmd.len() < 10 {
            return Err(TpmError::BufferTooSmall);
        }
        self.request_locality()?;

        let res = (|| -> Result<Vec<u8>, TpmError> {
            unsafe {
                mmio_write32(self.virt_base, TIS_STS, TIS_STS_COMMAND_READY);
            }
            self.wait_sts(TIS_STS_COMMAND_READY, TIS_STS_COMMAND_READY)?;

            for &byte in cmd {
                unsafe {
                    mmio_write8(self.virt_base, TIS_DATA_FIFO, byte);
                }
            }

            unsafe {
                mmio_write32(self.virt_base, TIS_STS, TIS_STS_GO);
            }
            self.wait_sts(TIS_STS_DATA_AVAIL, TIS_STS_DATA_AVAIL)?;

            let mut rsp = Vec::with_capacity(64);
            for _ in 0..10 {
                let b = unsafe { mmio_read8(self.virt_base, TIS_DATA_FIFO) };
                rsp.push(b);
            }
            let total_len = u32::from_be_bytes([rsp[2], rsp[3], rsp[4], rsp[5]]) as usize;
            if !(10..=4096).contains(&total_len) {
                return Err(TpmError::InvalidResponse);
            }
            while rsp.len() < total_len {
                let sts = unsafe { mmio_read32(self.virt_base, TIS_STS) };
                if (sts & TIS_STS_DATA_AVAIL) == 0 {
                    break;
                }
                let b = unsafe { mmio_read8(self.virt_base, TIS_DATA_FIFO) };
                rsp.push(b);
            }
            unsafe {
                mmio_write32(self.virt_base, TIS_STS, TIS_STS_COMMAND_READY);
            }
            Ok(rsp)
        })();

        self.release_locality();
        res
    }
}

pub struct MmioCrbTransport {
    virt_base: u64,
}

impl MmioCrbTransport {
    pub fn new(hhdm_offset: u64, phys_base: u64) -> Self {
        Self {
            virt_base: phys_base.wrapping_add(hhdm_offset),
        }
    }

    fn request_locality(&self) -> Result<(), TpmError> {
        unsafe {
            mmio_write32(self.virt_base, CRB_LOC_CTRL, CRB_LOC_CTRL_REQUEST);
            for _ in 0..TPM_POLL_LIMIT {
                let state = mmio_read32(self.virt_base, CRB_LOC_STATE);
                if (state & CRB_LOC_STATE_VALID) != 0 && (state & CRB_LOC_STATE_ASSIGNED) != 0 {
                    return Ok(());
                }
                core::hint::spin_loop();
            }
        }
        Err(TpmError::LocalityBusy)
    }

    fn release_locality(&self) {
        unsafe {
            mmio_write32(self.virt_base, CRB_LOC_CTRL, CRB_LOC_CTRL_RELINQUISH);
        }
    }

    pub fn submit(&self, cmd: &[u8]) -> Result<Vec<u8>, TpmError> {
        if cmd.len() < 10 || cmd.len() > 3968 {
            return Err(TpmError::BufferTooSmall);
        }
        self.request_locality()?;

        let res = (|| -> Result<Vec<u8>, TpmError> {
            unsafe {
                mmio_write32(self.virt_base, CRB_CTRL_REQ, CRB_CTRL_REQ_CMD_READY);
                for _ in 0..TPM_POLL_LIMIT {
                    let sts = mmio_read32(self.virt_base, CRB_CTRL_STS);
                    if (sts & CRB_CTRL_STS_ERROR) != 0 {
                        return Err(TpmError::CommandFailed(TPM_RC_FAILURE));
                    }
                    if (sts & CRB_CTRL_STS_IDLE) == 0 {
                        break;
                    }
                    core::hint::spin_loop();
                }

                for (idx, &byte) in cmd.iter().enumerate() {
                    mmio_write8(self.virt_base, CRB_DATA_BUFFER + idx, byte);
                }

                mmio_write32(self.virt_base, CRB_CTRL_START, CRB_CTRL_START_BIT);
                for _ in 0..TPM_POLL_LIMIT {
                    let start = mmio_read32(self.virt_base, CRB_CTRL_START);
                    if (start & CRB_CTRL_START_BIT) == 0 {
                        break;
                    }
                    core::hint::spin_loop();
                }

                let mut hdr = [0u8; 10];
                for (idx, dst) in hdr.iter_mut().enumerate() {
                    *dst = mmio_read8(self.virt_base, CRB_DATA_BUFFER + idx);
                }
                let total_len = u32::from_be_bytes([hdr[2], hdr[3], hdr[4], hdr[5]]) as usize;
                if !(10..=3968).contains(&total_len) {
                    return Err(TpmError::InvalidResponse);
                }
                let mut rsp = Vec::with_capacity(total_len);
                rsp.extend_from_slice(&hdr);
                for idx in 10..total_len {
                    rsp.push(mmio_read8(self.virt_base, CRB_DATA_BUFFER + idx));
                }
                mmio_write32(self.virt_base, CRB_CTRL_REQ, CRB_CTRL_REQ_GO_IDLE);
                Ok(rsp)
            }
        })();

        self.release_locality();
        res
    }
}

// ---------------------------------------------------------------------------
// PCR Bank & Measured Boot Event Log
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    SCrtmVersion,
    EfiBootServicesApp,
    Ipl,
    KernelImage,
    Initrd,
    Cmdline,
    KernelModule,
    SecurityPolicy,
    Separator,
}

impl EventType {
    pub fn code(self) -> u32 {
        match self {
            Self::SCrtmVersion => 0x0000_0008,
            Self::EfiBootServicesApp => 0x8000_0003,
            Self::Ipl => 0x0000_000D,
            Self::KernelImage => 0x8000_0100,
            Self::Initrd => 0x8000_0101,
            Self::Cmdline => 0x8000_0102,
            Self::KernelModule => 0x8000_0103,
            Self::SecurityPolicy => 0x8000_0104,
            Self::Separator => 0x0000_0004,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MeasuredEvent {
    pub sequence: u64,
    pub pcr_index: u8,
    pub event_type: EventType,
    pub digest: [u8; 32],
    pub description: String,
}

pub struct PcrBank {
    pcrs: [[u8; 32]; NUM_PCRS],
}

impl PcrBank {
    pub const fn new() -> Self {
        Self {
            pcrs: [[0u8; 32]; NUM_PCRS],
        }
    }

    pub fn reset(&mut self) {
        for pcr in self.pcrs.iter_mut() {
            *pcr = [0u8; 32];
        }
    }

    pub fn read(&self, pcr_index: u8) -> Result<[u8; 32], TpmError> {
        let idx = pcr_index as usize;
        if idx >= NUM_PCRS {
            return Err(TpmError::InvalidPcr(pcr_index));
        }
        Ok(self.pcrs[idx])
    }

    pub fn extend(&mut self, pcr_index: u8, digest: &[u8; 32]) -> Result<[u8; 32], TpmError> {
        let idx = pcr_index as usize;
        if idx >= NUM_PCRS {
            return Err(TpmError::InvalidPcr(pcr_index));
        }
        let mut hasher = Sha256::new();
        hasher.update(&self.pcrs[idx]);
        hasher.update(digest);
        let next = hasher.finalize();
        self.pcrs[idx] = next;
        Ok(next)
    }

    pub fn composite_digest(&self, pcr_mask: u32) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for (idx, pcr) in self.pcrs.iter().enumerate() {
            if (pcr_mask & (1u32 << idx)) != 0 {
                hasher.update(pcr);
            }
        }
        hasher.finalize()
    }
}

// ---------------------------------------------------------------------------
// Software TPM 2.0 Emulator (Wire-Compatible Framing)
// ---------------------------------------------------------------------------

pub struct SoftwareTpm {
    started: bool,
    pcr_bank: PcrBank,
    prng_state: u64,
}

impl SoftwareTpm {
    pub const fn new() -> Self {
        Self {
            started: false,
            pcr_bank: PcrBank::new(),
            prng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_random_bytes(&mut self, count: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            self.prng_state ^= self.prng_state << 13;
            self.prng_state ^= self.prng_state >> 7;
            self.prng_state ^= self.prng_state << 17;
            let digest = Sha256::digest(&self.prng_state.to_le_bytes());
            let take = (count - out.len()).min(32);
            out.extend_from_slice(&digest[..take]);
        }
        out
    }

    pub fn submit(&mut self, cmd: &[u8]) -> Result<Vec<u8>, TpmError> {
        if cmd.len() < 10 {
            return Err(TpmError::BufferTooSmall);
        }
        let size = u32::from_be_bytes([cmd[2], cmd[3], cmd[4], cmd[5]]) as usize;
        let cc = u32::from_be_bytes([cmd[6], cmd[7], cmd[8], cmd[9]]);
        if size > cmd.len() || size < 10 {
            return Err(TpmError::InvalidResponse);
        }

        match cc {
            TPM2_CC_STARTUP => {
                self.started = true;
                Ok(Self::ok_response(&[]))
            }
            TPM2_CC_SELF_TEST => {
                if !self.started {
                    return Ok(Self::err_response(TPM_RC_INITIALIZE));
                }
                Ok(Self::ok_response(&[]))
            }
            TPM2_CC_GET_RANDOM => {
                if !self.started || size < 12 {
                    return Ok(Self::err_response(TPM_RC_INITIALIZE));
                }
                let req = u16::from_be_bytes([cmd[10], cmd[11]]) as usize;
                let count = req.min(64);
                let bytes = self.next_random_bytes(count);
                let mut body = Vec::with_capacity(2 + count);
                body.extend_from_slice(&(count as u16).to_be_bytes());
                body.extend_from_slice(&bytes);
                Ok(Self::ok_response(&body))
            }
            TPM2_CC_PCR_READ => {
                if !self.started || size < 20 {
                    return Ok(Self::err_response(TPM_RC_VALUE));
                }
                let sel0 = cmd[17];
                let sel1 = cmd[18];
                let sel2 = cmd[19];
                let mask = (sel0 as u32) | ((sel1 as u32) << 8) | ((sel2 as u32) << 16);
                let mut digests = Vec::new();
                let mut count = 0u32;
                for idx in 0..NUM_PCRS {
                    if (mask & (1u32 << idx)) != 0 {
                        let val = self.pcr_bank.read(idx as u8)?;
                        digests.extend_from_slice(&32u16.to_be_bytes());
                        digests.extend_from_slice(&val);
                        count += 1;
                    }
                }
                let mut body = Vec::with_capacity(16 + digests.len());
                body.extend_from_slice(&1u32.to_be_bytes()); // pcrUpdateCounter
                body.extend_from_slice(&1u32.to_be_bytes()); // TPML_PCR_SELECTION count
                body.extend_from_slice(&TPM_ALG_SHA256.to_be_bytes());
                body.push(3);
                body.push(sel0);
                body.push(sel1);
                body.push(sel2);
                body.extend_from_slice(&count.to_be_bytes());
                body.extend_from_slice(&digests);
                Ok(Self::ok_response(&body))
            }
            TPM2_CC_PCR_EXTEND => {
                if !self.started || size < 10 + 4 + 4 + 9 + 4 + 2 + 32 {
                    return Ok(Self::err_response(TPM_RC_VALUE));
                }
                let pcr_handle = u32::from_be_bytes([cmd[10], cmd[11], cmd[12], cmd[13]]);
                let auth_size = u32::from_be_bytes([cmd[14], cmd[15], cmd[16], cmd[17]]) as usize;
                let digest_off = 18 + auth_size + 4 + 2;
                if digest_off + 32 > size {
                    return Ok(Self::err_response(TPM_RC_VALUE));
                }
                let mut digest = [0u8; 32];
                digest.copy_from_slice(&cmd[digest_off..digest_off + 32]);
                self.pcr_bank.extend(pcr_handle as u8, &digest)?;
                Ok(Self::ok_response(&[]))
            }
            _ => Ok(Self::err_response(TPM_RC_COMMAND_CODE)),
        }
    }

    fn ok_response(body: &[u8]) -> Vec<u8> {
        let total = (10 + body.len()) as u32;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&TPM_ST_NO_SESSIONS.to_be_bytes());
        out.extend_from_slice(&total.to_be_bytes());
        out.extend_from_slice(&TPM_RC_SUCCESS.to_be_bytes());
        out.extend_from_slice(body);
        out
    }

    fn err_response(rc: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(10);
        out.extend_from_slice(&TPM_ST_NO_SESSIONS.to_be_bytes());
        out.extend_from_slice(&10u32.to_be_bytes());
        out.extend_from_slice(&rc.to_be_bytes());
        out
    }
}

// ---------------------------------------------------------------------------
// TPM Quote Attestation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TpmQuote {
    pub attest_bytes: Vec<u8>,
    pub pcr_mask: u32,
    pub nonce: Vec<u8>,
    pub pcr_digest: [u8; 32],
    pub signature: Vec<u8>,
    pub sig_alg: KeyAlgorithm,
}

pub fn build_tpms_attest_quote(
    pcr_mask: u32,
    nonce: &[u8],
    pcr_digest: &[u8; 32],
    clock_ticks: u64,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(128);
    out.extend_from_slice(&TPM_GENERATED_VALUE.to_be_bytes());
    out.extend_from_slice(&TPM_ST_ATTEST_QUOTE.to_be_bytes());
    // qualifiedSigner (TPM2B_NAME: 2-byte len + 34-byte SHA256 name)
    out.extend_from_slice(&34u16.to_be_bytes());
    out.extend_from_slice(&TPM_ALG_SHA256.to_be_bytes());
    out.extend_from_slice(&[0xAAu8; 32]);
    // extraData (nonce)
    let nlen = nonce.len().min(64) as u16;
    out.extend_from_slice(&nlen.to_be_bytes());
    out.extend_from_slice(&nonce[..nlen as usize]);
    // clockInfo (clock u64, resetCount u32, restartCount u32, safe u8)
    out.extend_from_slice(&clock_ticks.to_be_bytes());
    out.extend_from_slice(&1u32.to_be_bytes());
    out.extend_from_slice(&1u32.to_be_bytes());
    out.push(1);
    // firmwareVersion
    out.extend_from_slice(&0x0002_0000_0001_0000u64.to_be_bytes());
    // TPMS_QUOTE_INFO: pcrSelect (TPML_PCR_SELECTION) + pcrDigest (TPM2B_DIGEST)
    out.extend_from_slice(&1u32.to_be_bytes());
    out.extend_from_slice(&TPM_ALG_SHA256.to_be_bytes());
    out.push(3);
    out.push((pcr_mask & 0xFF) as u8);
    out.push(((pcr_mask >> 8) & 0xFF) as u8);
    out.push(((pcr_mask >> 16) & 0xFF) as u8);
    out.extend_from_slice(&32u16.to_be_bytes());
    out.extend_from_slice(pcr_digest);
    out
}

pub fn verify_quote_hmac(
    quote: &TpmQuote,
    expected_nonce: &[u8],
    expected_pcr_mask: u32,
    expected_pcr_digest: &[u8; 32],
    attestation_key: &[u8],
) -> Result<(), TpmError> {
    if quote.sig_alg != KeyAlgorithm::HmacSha256 {
        return Err(TpmError::UnsupportedAlgorithm);
    }
    if quote.pcr_mask != expected_pcr_mask {
        return Err(TpmError::QuoteVerificationFailed);
    }
    if !ct_eq(&quote.nonce, expected_nonce) {
        return Err(TpmError::QuoteVerificationFailed);
    }
    if !ct_eq(&quote.pcr_digest, expected_pcr_digest) {
        return Err(TpmError::QuoteVerificationFailed);
    }
    if quote.attest_bytes.len() < 40 {
        return Err(TpmError::QuoteVerificationFailed);
    }
    let magic = u32::from_be_bytes([
        quote.attest_bytes[0],
        quote.attest_bytes[1],
        quote.attest_bytes[2],
        quote.attest_bytes[3],
    ]);
    let attest_type = u16::from_be_bytes([quote.attest_bytes[4], quote.attest_bytes[5]]);
    if magic != TPM_GENERATED_VALUE || attest_type != TPM_ST_ATTEST_QUOTE {
        return Err(TpmError::QuoteVerificationFailed);
    }
    let expected_mac = hmac_sha256(attestation_key, &quote.attest_bytes);
    if !ct_eq(&expected_mac, &quote.signature) {
        return Err(TpmError::InvalidSignature);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Kernel Keyring & Signature Verifier
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAlgorithm {
    Ed25519,
    HmacSha256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyUsage {
    KernelModule,
    Firmware,
    BootImage,
    Attestation,
    Policy,
}

#[derive(Debug, Clone)]
pub struct KeyEntry {
    pub id: u32,
    pub label: String,
    pub algorithm: KeyAlgorithm,
    pub usage: KeyUsage,
    pub key_bytes: Vec<u8>,
    pub fingerprint: [u8; 32],
    pub revoked: bool,
}

pub struct KernelKeyring {
    next_id: u32,
    keys: BTreeMap<u32, KeyEntry>,
}

impl KernelKeyring {
    pub const fn new() -> Self {
        Self {
            next_id: 1,
            keys: BTreeMap::new(),
        }
    }

    pub fn add_key(
        &mut self,
        label: &str,
        algorithm: KeyAlgorithm,
        usage: KeyUsage,
        key_bytes: &[u8],
    ) -> Result<u32, TpmError> {
        match algorithm {
            KeyAlgorithm::Ed25519 => {
                if key_bytes.len() != 32 {
                    return Err(TpmError::InvalidKeyFormat);
                }
                let mut pk = [0u8; 32];
                pk.copy_from_slice(key_bytes);
                if GeExtended::from_bytes(&pk).is_none() {
                    return Err(TpmError::InvalidKeyFormat);
                }
            }
            KeyAlgorithm::HmacSha256 => {
                if key_bytes.len() < 16 {
                    return Err(TpmError::InvalidKeyFormat);
                }
            }
        }

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let fingerprint = Sha256::digest(key_bytes);
        self.keys.insert(
            id,
            KeyEntry {
                id,
                label: String::from(label),
                algorithm,
                usage,
                key_bytes: key_bytes.to_vec(),
                fingerprint,
                revoked: false,
            },
        );
        Ok(id)
    }

    pub fn revoke_key(&mut self, key_id: u32) -> Result<(), TpmError> {
        let entry = self
            .keys
            .get_mut(&key_id)
            .ok_or(TpmError::KeyNotFound(key_id))?;
        entry.revoked = true;
        Ok(())
    }

    pub fn get_key(&self, key_id: u32) -> Option<&KeyEntry> {
        self.keys.get(&key_id)
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    pub fn verify(
        &self,
        key_id: u32,
        required_usage: KeyUsage,
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), TpmError> {
        let entry = self
            .keys
            .get(&key_id)
            .ok_or(TpmError::KeyNotFound(key_id))?;
        if entry.revoked {
            return Err(TpmError::KeyRevoked(key_id));
        }
        if entry.usage != required_usage {
            return Err(TpmError::InvalidSignature);
        }
        match entry.algorithm {
            KeyAlgorithm::Ed25519 => ed25519_verify(&entry.key_bytes, message, signature),
            KeyAlgorithm::HmacSha256 => {
                if signature.len() != 32 {
                    return Err(TpmError::InvalidSignature);
                }
                let expected = hmac_sha256(&entry.key_bytes, message);
                if ct_eq(&expected, signature) {
                    Ok(())
                } else {
                    Err(TpmError::InvalidSignature)
                }
            }
        }
    }

    /// Verify an `NXSG` signed image envelope:
    /// `[0..4] magic ("NXSG"), [4..6] version (u16 LE), [6..8] alg (u16 LE: 1=Ed25519, 2=HmacSha256),`
    /// `[8..12] key_id (u32 LE), [12..16] sig_len (u32 LE), [16..20] payload_len (u32 LE),`
    /// `[20..20+sig_len] signature, [20+sig_len..] payload`.
    pub fn verify_signed_image<'a>(
        &self,
        image: &'a [u8],
        required_usage: KeyUsage,
    ) -> Result<(u32, &'a [u8]), TpmError> {
        if image.len() < 20 || image[0..4] != NXSG_MAGIC {
            return Err(TpmError::InvalidSignature);
        }
        let version = u16::from_le_bytes([image[4], image[5]]);
        let alg_id = u16::from_le_bytes([image[6], image[7]]);
        let key_id = u32::from_le_bytes([image[8], image[9], image[10], image[11]]);
        let sig_len = u32::from_le_bytes([image[12], image[13], image[14], image[15]]) as usize;
        let payload_len = u32::from_le_bytes([image[16], image[17], image[18], image[19]]) as usize;

        if version != NXSG_VERSION {
            return Err(TpmError::InvalidSignature);
        }
        let expected_total = 20usize
            .checked_add(sig_len)
            .and_then(|v| v.checked_add(payload_len))
            .ok_or(TpmError::InvalidSignature)?;
        if image.len() != expected_total {
            return Err(TpmError::InvalidSignature);
        }

        let entry = self
            .keys
            .get(&key_id)
            .ok_or(TpmError::KeyNotFound(key_id))?;
        let expected_alg = match alg_id {
            1 => KeyAlgorithm::Ed25519,
            2 => KeyAlgorithm::HmacSha256,
            _ => return Err(TpmError::UnsupportedAlgorithm),
        };
        if entry.algorithm != expected_alg {
            return Err(TpmError::InvalidSignature);
        }

        let sig = &image[20..20 + sig_len];
        let payload = &image[20 + sig_len..];
        self.verify(key_id, required_usage, payload, sig)?;
        Ok((key_id, payload))
    }
}

pub fn build_signed_image(
    alg: KeyAlgorithm,
    key_id: u32,
    signature: &[u8],
    payload: &[u8],
) -> Vec<u8> {
    let alg_id: u16 = match alg {
        KeyAlgorithm::Ed25519 => 1,
        KeyAlgorithm::HmacSha256 => 2,
    };
    let mut out = Vec::with_capacity(20 + signature.len() + payload.len());
    out.extend_from_slice(&NXSG_MAGIC);
    out.extend_from_slice(&NXSG_VERSION.to_le_bytes());
    out.extend_from_slice(&alg_id.to_le_bytes());
    out.extend_from_slice(&key_id.to_le_bytes());
    out.extend_from_slice(&(signature.len() as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(signature);
    out.extend_from_slice(payload);
    out
}

// ---------------------------------------------------------------------------
// Global TPM Subsystem
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct TpmInfo {
    pub interface: TpmInterfaceKind,
    pub mmio_base: u64,
    pub hardware_present: bool,
    pub event_count: usize,
    pub key_count: usize,
}

pub struct TpmSubsystem {
    interface: TpmInterfaceKind,
    hhdm_offset: u64,
    mmio_base: u64,
    emu: SoftwareTpm,
    event_log: Vec<MeasuredEvent>,
    next_seq: u64,
    keyring: KernelKeyring,
    attestation_secret: [u8; 32],
}

impl TpmSubsystem {
    pub fn new(hhdm_offset: u64, mmio_base: u64, interface: TpmInterfaceKind) -> Self {
        let mut emu = SoftwareTpm::new();
        let _ = emu.submit(&build_cmd_startup(TPM2_SU_CLEAR));
        let _ = emu.submit(&build_cmd_self_test(false));
        Self {
            interface,
            hhdm_offset,
            mmio_base,
            emu,
            event_log: Vec::new(),
            next_seq: 1,
            keyring: KernelKeyring::new(),
            attestation_secret: Sha256::digest(b"nexus-os-tpm-attestation-key-v1"),
        }
    }

    pub fn submit_raw(&mut self, cmd: &[u8]) -> Result<Vec<u8>, TpmError> {
        match self.interface {
            TpmInterfaceKind::Tis => {
                let tis = MmioTisTransport::new(self.hhdm_offset, self.mmio_base);
                tis.submit(cmd)
            }
            TpmInterfaceKind::Crb => {
                let crb = MmioCrbTransport::new(self.hhdm_offset, self.mmio_base);
                crb.submit(cmd)
            }
            TpmInterfaceKind::Simulated => self.emu.submit(cmd),
        }
    }

    pub fn measure_and_extend(
        &mut self,
        pcr_index: u8,
        event_type: EventType,
        data: &[u8],
        description: &str,
    ) -> Result<[u8; 32], TpmError> {
        let digest = Sha256::digest(data);
        let cmd = build_cmd_pcr_extend(pcr_index, &digest)?;
        let rsp = self.submit_raw(&cmd)?;
        let reader = TpmResponseReader::parse(&rsp)?;
        reader.check_rc()?;

        // Keep software shadow bank in sync when hardware is active
        if self.interface != TpmInterfaceKind::Simulated {
            self.emu.pcr_bank.extend(pcr_index, &digest)?;
        }

        let seq = self.next_seq;
        self.next_seq += 1;
        self.event_log.push(MeasuredEvent {
            sequence: seq,
            pcr_index,
            event_type,
            digest,
            description: String::from(description),
        });
        self.emu.pcr_bank.read(pcr_index)
    }

    pub fn read_pcr(&self, pcr_index: u8) -> Result<[u8; 32], TpmError> {
        self.emu.pcr_bank.read(pcr_index)
    }

    pub fn verify_event_log(&self) -> Result<(), TpmError> {
        let mut replay = PcrBank::new();
        for ev in &self.event_log {
            replay.extend(ev.pcr_index, &ev.digest)?;
        }
        for idx in 0..NUM_PCRS {
            let expected = self.emu.pcr_bank.read(idx as u8)?;
            let replayed = replay.read(idx as u8)?;
            if !ct_eq(&expected, &replayed) {
                return Err(TpmError::EventLogCorrupted);
            }
        }
        Ok(())
    }

    pub fn quote(&self, pcr_mask: u32, nonce: &[u8]) -> Result<TpmQuote, TpmError> {
        let pcr_digest = self.emu.pcr_bank.composite_digest(pcr_mask);
        let attest_bytes = build_tpms_attest_quote(pcr_mask, nonce, &pcr_digest, self.next_seq);
        let signature = hmac_sha256(&self.attestation_secret, &attest_bytes).to_vec();
        Ok(TpmQuote {
            attest_bytes,
            pcr_mask,
            nonce: nonce.to_vec(),
            pcr_digest,
            signature,
            sig_alg: KeyAlgorithm::HmacSha256,
        })
    }
}

static TPM_SUBSYSTEM: SpinLock<Option<TpmSubsystem>> = SpinLock::new(None);
static TPM_INITIALIZED: AtomicBool = AtomicBool::new(false);
static TPM_EXTEND_COUNT: AtomicU64 = AtomicU64::new(0);

pub fn init(hhdm_offset: u64, acpi_tpm2_phys: Option<u64>) -> TpmInfo {
    let mmio_base = acpi_tpm2_phys.unwrap_or(TPM_MMIO_BASE);
    let interface = probe_mmio_interface(hhdm_offset, mmio_base);
    let hardware_present = interface != TpmInterfaceKind::Simulated;

    let mut sub = TpmSubsystem::new(hhdm_offset, mmio_base, interface);
    let _ = sub.measure_and_extend(
        PCR_KERNEL_IMAGE,
        EventType::KernelImage,
        b"nexus-kernel-v0.1.0",
        "Nexus OS microkernel image",
    );

    let info = TpmInfo {
        interface,
        mmio_base,
        hardware_present,
        event_count: sub.event_log.len(),
        key_count: sub.keyring.key_count(),
    };

    *TPM_SUBSYSTEM.lock() = Some(sub);
    TPM_INITIALIZED.store(true, Ordering::Release);
    info
}

pub fn is_initialized() -> bool {
    TPM_INITIALIZED.load(Ordering::Acquire)
}

pub fn measure_and_extend(
    pcr_index: u8,
    event_type: EventType,
    data: &[u8],
    description: &str,
) -> Result<[u8; 32], TpmError> {
    let mut guard = TPM_SUBSYSTEM.lock();
    let sub = guard.as_mut().ok_or(TpmError::NotInitialized)?;
    let res = sub.measure_and_extend(pcr_index, event_type, data, description)?;
    TPM_EXTEND_COUNT.fetch_add(1, Ordering::Relaxed);
    Ok(res)
}

pub fn read_pcr(pcr_index: u8) -> Result<[u8; 32], TpmError> {
    let guard = TPM_SUBSYSTEM.lock();
    let sub = guard.as_ref().ok_or(TpmError::NotInitialized)?;
    sub.read_pcr(pcr_index)
}

pub fn quote(pcr_mask: u32, nonce: &[u8]) -> Result<TpmQuote, TpmError> {
    let guard = TPM_SUBSYSTEM.lock();
    let sub = guard.as_ref().ok_or(TpmError::NotInitialized)?;
    sub.quote(pcr_mask, nonce)
}

pub fn add_trusted_key(
    label: &str,
    algorithm: KeyAlgorithm,
    usage: KeyUsage,
    key_bytes: &[u8],
) -> Result<u32, TpmError> {
    let mut guard = TPM_SUBSYSTEM.lock();
    let sub = guard.as_mut().ok_or(TpmError::NotInitialized)?;
    sub.keyring.add_key(label, algorithm, usage, key_bytes)
}

pub fn revoke_key(key_id: u32) -> Result<(), TpmError> {
    let mut guard = TPM_SUBSYSTEM.lock();
    let sub = guard.as_mut().ok_or(TpmError::NotInitialized)?;
    sub.keyring.revoke_key(key_id)
}

pub fn verify_signature(
    key_id: u32,
    required_usage: KeyUsage,
    message: &[u8],
    signature: &[u8],
) -> Result<(), TpmError> {
    let guard = TPM_SUBSYSTEM.lock();
    let sub = guard.as_ref().ok_or(TpmError::NotInitialized)?;
    sub.keyring
        .verify(key_id, required_usage, message, signature)
}

// ---------------------------------------------------------------------------
// Self-Test Suite
// ---------------------------------------------------------------------------

pub fn self_test() -> Result<(), &'static str> {
    // 1. SHA-256 NIST FIPS 180-4 test vector ("abc")
    let abc_hash = Sha256::digest(b"abc");
    let expected_abc: [u8; 32] = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];
    if !ct_eq(&abc_hash, &expected_abc) {
        return Err("tpm: SHA-256 test vector mismatch");
    }

    // 2. HMAC-SHA256 RFC 4231 Test Case 1
    let hmac_key = [0x0bu8; 20];
    let hmac_out = hmac_sha256(&hmac_key, b"Hi There");
    let expected_hmac: [u8; 32] = [
        0xb0, 0x34, 0x4c, 0x61, 0xd8, 0xdb, 0x38, 0x53, 0x5c, 0xa8, 0xaf, 0xce, 0xaf, 0x0b, 0xf1,
        0x2b, 0x88, 0x1d, 0xc2, 0x00, 0xc9, 0x83, 0x3d, 0xa7, 0x26, 0xe9, 0x37, 0x6c, 0x2e, 0x32,
        0xcf, 0xf7,
    ];
    if !ct_eq(&hmac_out, &expected_hmac) {
        return Err("tpm: HMAC-SHA256 test vector mismatch");
    }

    // 3. Ed25519 RFC 8032 Section 7.1 Test Vector 1 (empty message)
    let ed_pk: [u8; 32] = [
        0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07,
        0x3a, 0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07,
        0x51, 0x1a,
    ];
    let ed_sig: [u8; 64] = [
        0xe5, 0x56, 0x43, 0x00, 0xc3, 0x60, 0xac, 0x72, 0x90, 0x86, 0xe2, 0xcc, 0x80, 0x6e, 0x82,
        0x8a, 0x84, 0x87, 0x7f, 0x1e, 0xb8, 0xe5, 0xd9, 0x74, 0xd8, 0x73, 0xe0, 0x65, 0x22, 0x49,
        0x01, 0x55, 0x5f, 0xb8, 0x82, 0x15, 0x90, 0xa3, 0x3b, 0xac, 0xc6, 0x1e, 0x39, 0x70, 0x1c,
        0xf9, 0xb4, 0x6b, 0xd2, 0x5b, 0xf5, 0xf0, 0x59, 0x5b, 0xbe, 0x24, 0x65, 0x51, 0x41, 0x43,
        0x8e, 0x7a, 0x10, 0x0b,
    ];
    ed25519_verify(&ed_pk, b"", &ed_sig).map_err(|e| e.as_str())?;
    if ed25519_verify(&ed_pk, b"tampered", &ed_sig).is_ok() {
        return Err("tpm: Ed25519 accepted tampered message");
    }

    // 4. TPM 2.0 Command Framing + Software TPM PCR Extend / Read / Quote / Event Log Replay
    let mut sub = TpmSubsystem::new(0, TPM_MMIO_BASE, TpmInterfaceKind::Simulated);
    let rnd_rsp = sub
        .submit_raw(&build_cmd_get_random(16))
        .map_err(|e| e.as_str())?;
    let mut rnd_reader = TpmResponseReader::parse(&rnd_rsp).map_err(|e| e.as_str())?;
    rnd_reader.check_rc().map_err(|e| e.as_str())?;
    let rnd_len = rnd_reader.read_u16().map_err(|e| e.as_str())?;
    if rnd_len != 16 {
        return Err("tpm: GetRandom returned unexpected length");
    }

    sub.measure_and_extend(
        PCR_KERNEL_IMAGE,
        EventType::KernelImage,
        b"kernel-elf-bytes",
        "kernel",
    )
    .map_err(|e| e.as_str())?;
    sub.measure_and_extend(
        PCR_KERNEL_CMDLINE,
        EventType::Cmdline,
        b"root=/dev/nvme0n1p2 quiet",
        "cmdline",
    )
    .map_err(|e| e.as_str())?;
    sub.verify_event_log().map_err(|e| e.as_str())?;

    let pcr_mask = (1u32 << PCR_KERNEL_IMAGE) | (1u32 << PCR_KERNEL_CMDLINE);
    let nonce = b"challenge-nonce-2026";
    let q = sub.quote(pcr_mask, nonce).map_err(|e| e.as_str())?;
    let expected_composite = sub.emu.pcr_bank.composite_digest(pcr_mask);
    verify_quote_hmac(
        &q,
        nonce,
        pcr_mask,
        &expected_composite,
        &sub.attestation_secret,
    )
    .map_err(|e| e.as_str())?;

    // 5. Kernel Keyring + NXSG Signed Image + Revocation
    let ed_id = sub
        .keyring
        .add_key(
            "root-ed25519",
            KeyAlgorithm::Ed25519,
            KeyUsage::BootImage,
            &ed_pk,
        )
        .map_err(|e| e.as_str())?;
    let signed_img = build_signed_image(KeyAlgorithm::Ed25519, ed_id, &ed_sig, b"");
    let (verified_id, payload) = sub
        .keyring
        .verify_signed_image(&signed_img, KeyUsage::BootImage)
        .map_err(|e| e.as_str())?;
    if verified_id != ed_id || !payload.is_empty() {
        return Err("tpm: NXSG signed image verification mismatch");
    }

    sub.keyring.revoke_key(ed_id).map_err(|e| e.as_str())?;
    if sub
        .keyring
        .verify_signed_image(&signed_img, KeyUsage::BootImage)
        .is_ok()
    {
        return Err("tpm: revoked key was accepted");
    }

    Ok(())
}
