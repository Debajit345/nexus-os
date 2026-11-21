//! Kernel TLS 1.3 (kTLS) & QUIC Transport Offload Engine.
//!
//! Provides a pure `no_std` implementation of:
//! - SHA-256, HMAC-SHA256, HKDF-SHA256, and TLS 1.3 `HKDF-Expand-Label` (RFC 8446)
//! - TLS 1.3 Key Schedule & QUIC v1 Key Derivation (RFC 9001)
//! - AES-128-GCM / AES-256-GCM (FIPS 197 + NIST SP 800-38D)
//! - ChaCha20-Poly1305 AEAD (RFC 8439)
//! - kTLS 1.3 Record Framing, Encryption, Decryption, Padding & KeyUpdate
//! - QUIC VarInt, Long/Short Headers, Header Protection, Frame Codec & Stream Mux
//! - `self_test()` verification suite with NIST / RFC test vectors
//!
//! Designed for `no_std` x86_64 kernel execution with small stack frames (<2 KiB)
//! and constant-time tag verification.
#![allow(dead_code)]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::fmt;
use core::hint::spin_loop;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// ============================================================================
// 1. Errors, Constants & Cipher Suites
// ============================================================================

/// Errors returned by the kTLS 1.3 and QUIC offload engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsQuicError {
    InvalidRecord,
    RecordTooLarge,
    UnsupportedVersion,
    UnsupportedCipher,
    AuthFailed,
    SequenceOverflow,
    KeyUpdateRequired,
    InvalidPacket,
    InvalidVarInt,
    StreamNotFound,
    StreamClosed,
    FlowControlExceeded,
    BufferTooSmall,
    InvalidState,
    CryptoError,
}

impl TlsQuicError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRecord => "invalid TLS record",
            Self::RecordTooLarge => "TLS record exceeds maximum length",
            Self::UnsupportedVersion => "unsupported TLS/QUIC version",
            Self::UnsupportedCipher => "unsupported cipher suite",
            Self::AuthFailed => "AEAD authentication tag verification failed",
            Self::SequenceOverflow => "record/packet sequence number overflow",
            Self::KeyUpdateRequired => "key update required",
            Self::InvalidPacket => "invalid QUIC packet",
            Self::InvalidVarInt => "invalid QUIC variable-length integer",
            Self::StreamNotFound => "QUIC stream not found",
            Self::StreamClosed => "QUIC stream closed",
            Self::FlowControlExceeded => "QUIC flow control limit exceeded",
            Self::BufferTooSmall => "output buffer too small",
            Self::InvalidState => "invalid connection or session state",
            Self::CryptoError => "cryptographic operation failed",
        }
    }
}

impl fmt::Display for TlsQuicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Supported TLS 1.3 / QUIC AEAD cipher suites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CipherSuite {
    TlsAes128GcmSha256 = 0x1301,
    TlsAes256GcmSha256 = 0x1302,
    TlsChacha20Poly1305Sha256 = 0x1303,
}

impl CipherSuite {
    pub const fn from_u16(val: u16) -> Result<Self, TlsQuicError> {
        match val {
            0x1301 => Ok(Self::TlsAes128GcmSha256),
            0x1302 => Ok(Self::TlsAes256GcmSha256),
            0x1303 => Ok(Self::TlsChacha20Poly1305Sha256),
            _ => Err(TlsQuicError::UnsupportedCipher),
        }
    }

    pub const fn key_len(self) -> usize {
        match self {
            Self::TlsAes128GcmSha256 => 16,
            Self::TlsAes256GcmSha256 | Self::TlsChacha20Poly1305Sha256 => 32,
        }
    }

    pub const fn iv_len(self) -> usize {
        12
    }

    pub const fn tag_len(self) -> usize {
        16
    }
}

/// TLS 1.3 legacy record version (0x0303 = TLS 1.2).
pub const TLS_LEGACY_VERSION: u16 = 0x0303;
/// TLS 1.3 protocol version.
pub const TLS_VERSION_1_3: u16 = 0x0304;
/// Maximum TLS 1.3 plaintext fragment length (16 KiB).
pub const TLS_MAX_PLAINTEXT_LEN: usize = 16384;
/// Maximum TLS 1.3 ciphertext length (16 KiB + 256 bytes expansion).
pub const TLS_MAX_CIPHERTEXT_LEN: usize = 16384 + 256;
/// QUIC v1 version number (RFC 9000).
pub const QUIC_VERSION_1: u32 = 0x0000_0001;

/// Constant-time byte slice equality check.
#[inline]
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ============================================================================
// 2. Minimal SpinLock for no_std Global Offload Registry
// ============================================================================

struct LocalSpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for LocalSpinLock<T> {}
unsafe impl<T: Send> Send for LocalSpinLock<T> {}

impl<T> LocalSpinLock<T> {
    const fn new(val: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(val),
        }
    }

    fn lock(&self) -> LocalSpinGuard<'_, T> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            while self.locked.load(Ordering::Relaxed) {
                spin_loop();
            }
        }
        LocalSpinGuard { lock: self }
    }
}

struct LocalSpinGuard<'a, T> {
    lock: &'a LocalSpinLock<T>,
}

impl<T> core::ops::Deref for LocalSpinGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> core::ops::DerefMut for LocalSpinGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for LocalSpinGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

// ============================================================================
// 3. SHA-256, HMAC-SHA256 & HKDF-SHA256 (FIPS 180-4, RFC 2104, RFC 5869)
// ============================================================================

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

const SHA256_INIT: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// Streaming SHA-256 hasher with small stack footprint.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Sha256 {
    pub const fn new() -> Self {
        Self {
            state: SHA256_INIT,
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        let (chunks, _) = block.as_chunks::<4>();
        for (dst, src) in w[..16].iter_mut().zip(chunks.iter()) {
            *dst = u32::from_be_bytes(*src);
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
        for (&wi, &ki) in w.iter().zip(SHA256_K.iter()) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(ki)
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

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                Self::compress(&mut self.state, &block);
                self.buf_len = 0;
            }
        }
        let (blocks, rem) = data.as_chunks::<64>();
        for block in blocks {
            Self::compress(&mut self.state, block);
        }
        if !rem.is_empty() {
            self.buf[..rem.len()].copy_from_slice(rem);
            self.buf_len = rem.len();
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            self.buf[self.buf_len..64].fill(0);
            let block = self.buf;
            Self::compress(&mut self.state, &block);
            self.buf_len = 0;
        }
        self.buf[self.buf_len..56].fill(0);
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        Self::compress(&mut self.state, &block);

        let mut out = [0u8; 32];
        for (dst, &word) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.state.iter()) {
            *dst = word.to_be_bytes();
        }
        out
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute SHA-256 digest of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

/// Compute HMAC-SHA256 over multiple slices.
pub fn hmac_sha256_parts(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut k_block = [0u8; 64];
    if key.len() > 64 {
        let hashed = sha256(key);
        k_block[..32].copy_from_slice(&hashed);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for (i, &kb) in k_block.iter().enumerate() {
        ipad[i] ^= kb;
        opad[i] ^= kb;
    }

    let mut inner = Sha256::new();
    inner.update(&ipad);
    for part in parts {
        inner.update(part);
    }
    let inner_hash = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&inner_hash);
    outer.finalize()
}

/// Compute HMAC-SHA256(key, data).
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    hmac_sha256_parts(key, &[data])
}

/// HKDF-Extract(salt, ikm) -> PRK (RFC 5869).
pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    let zero_salt = [0u8; 32];
    let actual_salt = if salt.is_empty() { &zero_salt } else { salt };
    hmac_sha256(actual_salt, ikm)
}

/// HKDF-Expand(prk, info, out) (RFC 5869).
pub fn hkdf_expand(prk: &[u8; 32], info: &[u8], out: &mut [u8]) -> Result<(), TlsQuicError> {
    if out.len() > 255 * 32 {
        return Err(TlsQuicError::CryptoError);
    }
    let mut t = [0u8; 32];
    let mut t_len = 0usize;
    let mut offset = 0usize;
    let mut counter = 1u8;

    while offset < out.len() {
        let c_byte = [counter];
        t = hmac_sha256_parts(prk, &[&t[..t_len], info, &c_byte]);
        t_len = 32;
        let take = (out.len() - offset).min(32);
        out[offset..offset + take].copy_from_slice(&t[..take]);
        offset += take;
        counter = counter.wrapping_add(1);
    }
    Ok(())
}

/// TLS 1.3 `HKDF-Expand-Label(Secret, Label, Context, Length)` (RFC 8446 Section 7.1).
pub fn hkdf_expand_label(
    secret: &[u8; 32],
    label: &[u8],
    context: &[u8],
    out: &mut [u8],
) -> Result<(), TlsQuicError> {
    let full_label_len = 6 + label.len();
    if full_label_len > 255 || context.len() > 255 || out.len() > u16::MAX as usize {
        return Err(TlsQuicError::CryptoError);
    }
    let mut info = Vec::with_capacity(2 + 1 + full_label_len + 1 + context.len());
    info.extend_from_slice(&(out.len() as u16).to_be_bytes());
    info.push(full_label_len as u8);
    info.extend_from_slice(b"tls13 ");
    info.extend_from_slice(label);
    info.push(context.len() as u8);
    info.extend_from_slice(context);
    hkdf_expand(secret, &info, out)
}

/// TLS 1.3 `Derive-Secret(Secret, Label, TranscriptHash)` (RFC 8446 Section 7.1).
pub fn derive_secret(secret: &[u8; 32], label: &[u8], transcript_hash: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let _ = hkdf_expand_label(secret, label, transcript_hash, &mut out);
    out
}

// ============================================================================
// 4. TLS 1.3 Key Schedule & QUIC v1 Key Derivation
// ============================================================================

/// Derived traffic key and IV for kTLS 1.3 record protection.
#[derive(Clone, Copy)]
pub struct TrafficKeys {
    pub key: [u8; 32],
    pub key_len: usize,
    pub iv: [u8; 12],
}

/// Derived QUIC packet protection keys (AEAD key, IV, and Header Protection key).
#[derive(Clone, Copy)]
pub struct QuicKeys {
    pub cipher: CipherSuite,
    pub key: [u8; 32],
    pub key_len: usize,
    pub iv: [u8; 12],
    pub hp_key: [u8; 32],
}

/// TLS 1.3 Key Schedule helper (RFC 8446 Section 7.1).
pub struct Tls13KeySchedule;

impl Tls13KeySchedule {
    /// SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    pub const EMPTY_HASH: [u8; 32] = [
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9,
        0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52,
        0xb8, 0x55,
    ];

    /// Compute Early Secret from optional PSK (defaults to 32 zero bytes).
    pub fn early_secret(psk: Option<&[u8]>) -> [u8; 32] {
        let zeros = [0u8; 32];
        hkdf_extract(&zeros, psk.unwrap_or(&zeros))
    }

    /// Compute Handshake Secret from Early Secret and ECDHE shared secret.
    pub fn handshake_secret(early_secret: &[u8; 32], shared_secret: &[u8]) -> [u8; 32] {
        let derived = derive_secret(early_secret, b"derived", &Self::EMPTY_HASH);
        hkdf_extract(&derived, shared_secret)
    }

    /// Compute Master Secret from Handshake Secret.
    pub fn master_secret(handshake_secret: &[u8; 32]) -> [u8; 32] {
        let derived = derive_secret(handshake_secret, b"derived", &Self::EMPTY_HASH);
        let zeros = [0u8; 32];
        hkdf_extract(&derived, &zeros)
    }

    /// Derive TLS 1.3 traffic key and 12-byte IV from a traffic secret.
    pub fn derive_traffic_keys(traffic_secret: &[u8; 32], cipher: CipherSuite) -> TrafficKeys {
        let key_len = cipher.key_len();
        let mut key = [0u8; 32];
        let mut iv = [0u8; 12];
        let _ = hkdf_expand_label(traffic_secret, b"key", &[], &mut key[..key_len]);
        let _ = hkdf_expand_label(traffic_secret, b"iv", &[], &mut iv);
        TrafficKeys { key, key_len, iv }
    }

    /// Advance a TLS 1.3 application traffic secret (`KeyUpdate`, RFC 8446 Section 7.2).
    pub fn update_traffic_secret(traffic_secret: &[u8; 32]) -> [u8; 32] {
        let mut next = [0u8; 32];
        let _ = hkdf_expand_label(traffic_secret, b"traffic upd", &[], &mut next);
        next
    }

    /// Derive QUIC packet protection keys (`quic key`, `quic iv`, `quic hp`, RFC 9001 Section 5.1).
    pub fn quic_derive_keys(traffic_secret: &[u8; 32], cipher: CipherSuite) -> QuicKeys {
        let key_len = cipher.key_len();
        let mut key = [0u8; 32];
        let mut iv = [0u8; 12];
        let mut hp_key = [0u8; 32];
        let _ = hkdf_expand_label(traffic_secret, b"quic key", &[], &mut key[..key_len]);
        let _ = hkdf_expand_label(traffic_secret, b"quic iv", &[], &mut iv);
        let _ = hkdf_expand_label(traffic_secret, b"quic hp", &[], &mut hp_key[..key_len]);
        QuicKeys {
            cipher,
            key,
            key_len,
            iv,
            hp_key,
        }
    }

    /// Derive QUIC v1 Initial client and server keys from Destination Connection ID (RFC 9001 Section 5.2).
    pub fn quic_initial_keys(dcid: &[u8]) -> (QuicKeys, QuicKeys) {
        const QUIC_V1_INITIAL_SALT: [u8; 20] = [
            0x38, 0x76, 0x2c, 0xf7, 0xf5, 0x59, 0x34, 0xb3, 0x4d, 0x17, 0x9a, 0xe6, 0xa4, 0xc8,
            0x0c, 0xad, 0xcc, 0xbb, 0x7f, 0x0a,
        ];
        let initial_secret = hkdf_extract(&QUIC_V1_INITIAL_SALT, dcid);
        let mut client_secret = [0u8; 32];
        let mut server_secret = [0u8; 32];
        let _ = hkdf_expand_label(&initial_secret, b"client in", &[], &mut client_secret);
        let _ = hkdf_expand_label(&initial_secret, b"server in", &[], &mut server_secret);
        (
            Self::quic_derive_keys(&client_secret, CipherSuite::TlsAes128GcmSha256),
            Self::quic_derive_keys(&server_secret, CipherSuite::TlsAes128GcmSha256),
        )
    }
}

// ============================================================================
// 5. AES-128 / AES-256 & AES-GCM (FIPS 197 + NIST SP 800-38D)
// ============================================================================

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

const AES_RCON: [u32; 11] = [
    0x0000_0000,
    0x0100_0000,
    0x0200_0000,
    0x0400_0000,
    0x0800_0000,
    0x1000_0000,
    0x2000_0000,
    0x4000_0000,
    0x8000_0000,
    0x1b00_0000,
    0x3600_0000,
];

#[inline]
fn sub_word(w: u32) -> u32 {
    let b = w.to_be_bytes();
    u32::from_be_bytes([
        AES_SBOX[b[0] as usize],
        AES_SBOX[b[1] as usize],
        AES_SBOX[b[2] as usize],
        AES_SBOX[b[3] as usize],
    ])
}

#[inline]
fn xtime(x: u8) -> u8 {
    (x << 1) ^ (((x >> 7) & 1) * 0x1b)
}

struct AesCipher {
    rk: [u32; 60],
    rounds: usize,
}

impl AesCipher {
    fn new(key: &[u8]) -> Result<Self, TlsQuicError> {
        let (nk, rounds) = match key.len() {
            16 => (4usize, 10usize),
            32 => (8usize, 14usize),
            _ => return Err(TlsQuicError::CryptoError),
        };
        let mut rk = [0u32; 60];
        let (chunks, _) = key.as_chunks::<4>();
        for (dst, src) in rk[..nk].iter_mut().zip(chunks.iter()) {
            *dst = u32::from_be_bytes(*src);
        }
        let total_words = 4 * (rounds + 1);
        for i in nk..total_words {
            let mut temp = rk[i - 1];
            if i.is_multiple_of(nk) {
                temp = sub_word(temp.rotate_left(8)) ^ AES_RCON[i / nk];
            } else if nk > 6 && i % nk == 4 {
                temp = sub_word(temp);
            }
            rk[i] = rk[i - nk] ^ temp;
        }
        Ok(Self { rk, rounds })
    }

    fn encrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let mut s = *block;
        self.add_round_key(&mut s, 0);

        for round in 1..self.rounds {
            for b in &mut s {
                *b = AES_SBOX[*b as usize];
            }
            Self::shift_rows(&mut s);
            Self::mix_columns(&mut s);
            self.add_round_key(&mut s, round);
        }

        for b in &mut s {
            *b = AES_SBOX[*b as usize];
        }
        Self::shift_rows(&mut s);
        self.add_round_key(&mut s, self.rounds);
        s
    }

    #[inline]
    fn add_round_key(&self, s: &mut [u8; 16], round: usize) {
        let base = round * 4;
        for (col, &rk_word) in s
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(self.rk[base..base + 4].iter())
        {
            let k = rk_word.to_be_bytes();
            col[0] ^= k[0];
            col[1] ^= k[1];
            col[2] ^= k[2];
            col[3] ^= k[3];
        }
    }

    #[inline]
    fn shift_rows(s: &mut [u8; 16]) {
        // Row 1: shift left 1
        let t1 = s[1];
        s[1] = s[5];
        s[5] = s[9];
        s[9] = s[13];
        s[13] = t1;
        // Row 2: shift left 2
        s.swap(2, 10);
        s.swap(6, 14);
        // Row 3: shift left 3 (right 1)
        let t3 = s[15];
        s[15] = s[11];
        s[11] = s[7];
        s[7] = s[3];
        s[3] = t3;
    }

    #[inline]
    fn mix_columns(s: &mut [u8; 16]) {
        for col in s.as_chunks_mut::<4>().0 {
            let [a0, a1, a2, a3] = *col;
            let t = a0 ^ a1 ^ a2 ^ a3;
            col[0] ^= t ^ xtime(a0 ^ a1);
            col[1] ^= t ^ xtime(a1 ^ a2);
            col[2] ^= t ^ xtime(a2 ^ a3);
            col[3] ^= t ^ xtime(a3 ^ a0);
        }
    }
}

/// GF(2^128) multiplication for GHASH (NIST SP 800-38D Section 6.3).
fn gf128_mul(x: u128, y: u128) -> u128 {
    const R: u128 = 0xe100_0000_0000_0000_0000_0000_0000_0000;
    let mut z = 0u128;
    let mut v = y;
    for i in 0..128 {
        let bit = (x >> (127 - i)) & 1;
        let mask = 0u128.wrapping_sub(bit);
        z ^= v & mask;
        let lsb = v & 1;
        v >>= 1;
        v ^= R & 0u128.wrapping_sub(lsb);
    }
    z
}

fn ghash(h: u128, aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    let mut y = 0u128;

    let (aad_blocks, aad_rem) = aad.as_chunks::<16>();
    for blk in aad_blocks {
        y ^= u128::from_be_bytes(*blk);
        y = gf128_mul(y, h);
    }
    if !aad_rem.is_empty() {
        let mut padded = [0u8; 16];
        padded[..aad_rem.len()].copy_from_slice(aad_rem);
        y ^= u128::from_be_bytes(padded);
        y = gf128_mul(y, h);
    }

    let (ct_blocks, ct_rem) = ciphertext.as_chunks::<16>();
    for blk in ct_blocks {
        y ^= u128::from_be_bytes(*blk);
        y = gf128_mul(y, h);
    }
    if !ct_rem.is_empty() {
        let mut padded = [0u8; 16];
        padded[..ct_rem.len()].copy_from_slice(ct_rem);
        y ^= u128::from_be_bytes(padded);
        y = gf128_mul(y, h);
    }

    let mut len_block = [0u8; 16];
    len_block[..8].copy_from_slice(&((aad.len() as u64) * 8).to_be_bytes());
    len_block[8..16].copy_from_slice(&((ciphertext.len() as u64) * 8).to_be_bytes());
    y ^= u128::from_be_bytes(len_block);
    y = gf128_mul(y, h);
    y.to_be_bytes()
}

#[inline]
fn inc32(counter: &mut [u8; 16]) {
    let c =
        u32::from_be_bytes([counter[12], counter[13], counter[14], counter[15]]).wrapping_add(1);
    counter[12..16].copy_from_slice(&c.to_be_bytes());
}

fn aes_ctr_xor(aes: &AesCipher, j0: &[u8; 16], data: &mut [u8]) {
    let mut counter = *j0;
    let (blocks, rem) = data.as_chunks_mut::<16>();
    for blk in blocks {
        inc32(&mut counter);
        let ks = aes.encrypt_block(&counter);
        for (d, &k) in blk.iter_mut().zip(ks.iter()) {
            *d ^= k;
        }
    }
    if !rem.is_empty() {
        inc32(&mut counter);
        let ks = aes.encrypt_block(&counter);
        for (d, &k) in rem.iter_mut().zip(ks.iter()) {
            *d ^= k;
        }
    }
}

/// AES-GCM authenticated encryption (12-byte nonce, 16-byte tag).
pub fn aes_gcm_seal(
    key: &[u8],
    nonce: &[u8; 12],
    aad: &[u8],
    plaintext: &[u8],
) -> Result<(Vec<u8>, [u8; 16]), TlsQuicError> {
    let aes = AesCipher::new(key)?;
    let h = u128::from_be_bytes(aes.encrypt_block(&[0u8; 16]));

    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(nonce);
    j0[15] = 1;

    let mut ciphertext = plaintext.to_vec();
    aes_ctr_xor(&aes, &j0, &mut ciphertext);

    let s = ghash(h, aad, &ciphertext);
    let e_j0 = aes.encrypt_block(&j0);
    let mut tag = [0u8; 16];
    for (t, (&sb, &eb)) in tag.iter_mut().zip(s.iter().zip(e_j0.iter())) {
        *t = sb ^ eb;
    }
    Ok((ciphertext, tag))
}

/// AES-GCM authenticated decryption (12-byte nonce, 16-byte tag).
pub fn aes_gcm_open(
    key: &[u8],
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
    tag: &[u8; 16],
) -> Result<Vec<u8>, TlsQuicError> {
    let aes = AesCipher::new(key)?;
    let h = u128::from_be_bytes(aes.encrypt_block(&[0u8; 16]));

    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(nonce);
    j0[15] = 1;

    let s = ghash(h, aad, ciphertext);
    let e_j0 = aes.encrypt_block(&j0);
    let mut expected_tag = [0u8; 16];
    for (t, (&sb, &eb)) in expected_tag.iter_mut().zip(s.iter().zip(e_j0.iter())) {
        *t = sb ^ eb;
    }
    if !ct_eq(&expected_tag, tag) {
        return Err(TlsQuicError::AuthFailed);
    }

    let mut plaintext = ciphertext.to_vec();
    aes_ctr_xor(&aes, &j0, &mut plaintext);
    Ok(plaintext)
}

// ============================================================================
// 6. ChaCha20-Poly1305 AEAD (RFC 8439)
// ============================================================================

#[inline]
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut state = [0u32; 16];
    state[0] = 0x6170_7865;
    state[1] = 0x3320_646e;
    state[2] = 0x7962_2d32;
    state[3] = 0x6b20_6574;
    for (dst, src) in state[4..12].iter_mut().zip(key.as_chunks::<4>().0.iter()) {
        *dst = u32::from_le_bytes(*src);
    }
    state[12] = counter;
    for (dst, src) in state[13..16]
        .iter_mut()
        .zip(nonce.as_chunks::<4>().0.iter())
    {
        *dst = u32::from_le_bytes(*src);
    }

    let mut working = state;
    for _ in 0..10 {
        quarter_round(&mut working, 0, 4, 8, 12);
        quarter_round(&mut working, 1, 5, 9, 13);
        quarter_round(&mut working, 2, 6, 10, 14);
        quarter_round(&mut working, 3, 7, 11, 15);
        quarter_round(&mut working, 0, 5, 10, 15);
        quarter_round(&mut working, 1, 6, 11, 12);
        quarter_round(&mut working, 2, 7, 8, 13);
        quarter_round(&mut working, 3, 4, 9, 14);
    }

    let mut out = [0u8; 64];
    for (dst, (&w, &s)) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(working.iter().zip(state.iter()))
    {
        *dst = w.wrapping_add(s).to_le_bytes();
    }
    out
}

fn chacha20_xor(key: &[u8; 32], mut counter: u32, nonce: &[u8; 12], data: &mut [u8]) {
    let (blocks, rem) = data.as_chunks_mut::<64>();
    for blk in blocks {
        let ks = chacha20_block(key, counter, nonce);
        counter = counter.wrapping_add(1);
        for (d, &k) in blk.iter_mut().zip(ks.iter()) {
            *d ^= k;
        }
    }
    if !rem.is_empty() {
        let ks = chacha20_block(key, counter, nonce);
        for (d, &k) in rem.iter_mut().zip(ks.iter()) {
            *d ^= k;
        }
    }
}

struct Poly1305 {
    r: [u32; 5],
    h: [u32; 5],
    pad: [u32; 4],
}

impl Poly1305 {
    fn new(key: &[u8; 32]) -> Self {
        let t0 = u32::from_le_bytes([key[0], key[1], key[2], key[3]]);
        let t1 = u32::from_le_bytes([key[4], key[5], key[6], key[7]]);
        let t2 = u32::from_le_bytes([key[8], key[9], key[10], key[11]]);
        let t3 = u32::from_le_bytes([key[12], key[13], key[14], key[15]]);

        let r = [
            t0 & 0x03ff_ffff,
            ((t0 >> 26) | (t1 << 6)) & 0x03ff_ff03,
            ((t1 >> 20) | (t2 << 12)) & 0x03ff_c0ff,
            ((t2 >> 14) | (t3 << 18)) & 0x03f0_3fff,
            (t3 >> 8) & 0x000f_ffff,
        ];

        let pad = [
            u32::from_le_bytes([key[16], key[17], key[18], key[19]]),
            u32::from_le_bytes([key[20], key[21], key[22], key[23]]),
            u32::from_le_bytes([key[24], key[25], key[26], key[27]]),
            u32::from_le_bytes([key[28], key[29], key[30], key[31]]),
        ];

        Self {
            r,
            h: [0u32; 5],
            pad,
        }
    }

    fn block(&mut self, msg: &[u8], hibit: u32) {
        let mut buf = [0u8; 16];
        buf[..msg.len()].copy_from_slice(msg);
        if msg.len() < 16 {
            buf[msg.len()] = 1;
        }

        let t0 = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let t1 = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let t2 = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let t3 = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);

        self.h[0] = self.h[0].wrapping_add(t0 & 0x03ff_ffff);
        self.h[1] = self.h[1].wrapping_add(((t0 >> 26) | (t1 << 6)) & 0x03ff_ffff);
        self.h[2] = self.h[2].wrapping_add(((t1 >> 20) | (t2 << 12)) & 0x03ff_ffff);
        self.h[3] = self.h[3].wrapping_add(((t2 >> 14) | (t3 << 18)) & 0x03ff_ffff);
        self.h[4] = self.h[4].wrapping_add((t3 >> 8) | hibit);

        let [r0, r1, r2, r3, r4] = self.r;
        let s1 = r1 * 5;
        let s2 = r2 * 5;
        let s3 = r3 * 5;
        let s4 = r4 * 5;

        let [h0, h1, h2, h3, h4] = self.h.map(u64::from);
        let d0 = h0 * u64::from(r0)
            + h1 * u64::from(s4)
            + h2 * u64::from(s3)
            + h3 * u64::from(s2)
            + h4 * u64::from(s1);
        let mut d1 = h0 * u64::from(r1)
            + h1 * u64::from(r0)
            + h2 * u64::from(s4)
            + h3 * u64::from(s3)
            + h4 * u64::from(s2);
        let mut d2 = h0 * u64::from(r2)
            + h1 * u64::from(r1)
            + h2 * u64::from(r0)
            + h3 * u64::from(s4)
            + h4 * u64::from(s3);
        let mut d3 = h0 * u64::from(r3)
            + h1 * u64::from(r2)
            + h2 * u64::from(r1)
            + h3 * u64::from(r0)
            + h4 * u64::from(s4);
        let mut d4 = h0 * u64::from(r4)
            + h1 * u64::from(r3)
            + h2 * u64::from(r2)
            + h3 * u64::from(r1)
            + h4 * u64::from(r0);

        let mut c = d0 >> 26;
        self.h[0] = (d0 as u32) & 0x03ff_ffff;
        d1 += c;
        c = d1 >> 26;
        self.h[1] = (d1 as u32) & 0x03ff_ffff;
        d2 += c;
        c = d2 >> 26;
        self.h[2] = (d2 as u32) & 0x03ff_ffff;
        d3 += c;
        c = d3 >> 26;
        self.h[3] = (d3 as u32) & 0x03ff_ffff;
        d4 += c;
        c = d4 >> 26;
        self.h[4] = (d4 as u32) & 0x03ff_ffff;
        self.h[0] = self.h[0].wrapping_add((c as u32) * 5);
        c = u64::from(self.h[0] >> 26);
        self.h[0] &= 0x03ff_ffff;
        self.h[1] = self.h[1].wrapping_add(c as u32);
    }

    fn update_padded(&mut self, data: &[u8]) {
        let (blocks, rem) = data.as_chunks::<16>();
        for blk in blocks {
            self.block(blk, 1 << 24);
        }
        if !rem.is_empty() {
            let mut padded = [0u8; 16];
            padded[..rem.len()].copy_from_slice(rem);
            self.block(&padded, 1 << 24);
        }
    }

    fn finalize(mut self) -> [u8; 16] {
        let mut c = self.h[1] >> 26;
        self.h[1] &= 0x03ff_ffff;
        self.h[2] = self.h[2].wrapping_add(c);
        c = self.h[2] >> 26;
        self.h[2] &= 0x03ff_ffff;
        self.h[3] = self.h[3].wrapping_add(c);
        c = self.h[3] >> 26;
        self.h[3] &= 0x03ff_ffff;
        self.h[4] = self.h[4].wrapping_add(c);
        c = self.h[4] >> 26;
        self.h[4] &= 0x03ff_ffff;
        self.h[0] = self.h[0].wrapping_add(c * 5);
        c = self.h[0] >> 26;
        self.h[0] &= 0x03ff_ffff;
        self.h[1] = self.h[1].wrapping_add(c);

        let mut g0 = self.h[0].wrapping_add(5);
        c = g0 >> 26;
        g0 &= 0x03ff_ffff;
        let mut g1 = self.h[1].wrapping_add(c);
        c = g1 >> 26;
        g1 &= 0x03ff_ffff;
        let mut g2 = self.h[2].wrapping_add(c);
        c = g2 >> 26;
        g2 &= 0x03ff_ffff;
        let mut g3 = self.h[3].wrapping_add(c);
        c = g3 >> 26;
        g3 &= 0x03ff_ffff;
        let g4 = self.h[4].wrapping_add(c).wrapping_sub(1 << 26);

        let mask = (g4 >> 31).wrapping_sub(1);
        let nmask = !mask;
        self.h[0] = (self.h[0] & nmask) | (g0 & mask);
        self.h[1] = (self.h[1] & nmask) | (g1 & mask);
        self.h[2] = (self.h[2] & nmask) | (g2 & mask);
        self.h[3] = (self.h[3] & nmask) | (g3 & mask);
        self.h[4] = (self.h[4] & nmask) | (g4 & mask);

        let f0 = u64::from(self.h[0] | (self.h[1] << 26)) + u64::from(self.pad[0]);
        let f1 =
            u64::from((self.h[1] >> 6) | (self.h[2] << 20)) + u64::from(self.pad[1]) + (f0 >> 32);
        let f2 =
            u64::from((self.h[2] >> 12) | (self.h[3] << 14)) + u64::from(self.pad[2]) + (f1 >> 32);
        let f3 =
            u64::from((self.h[3] >> 18) | (self.h[4] << 8)) + u64::from(self.pad[3]) + (f2 >> 32);

        let mut tag = [0u8; 16];
        tag[0..4].copy_from_slice(&(f0 as u32).to_le_bytes());
        tag[4..8].copy_from_slice(&(f1 as u32).to_le_bytes());
        tag[8..12].copy_from_slice(&(f2 as u32).to_le_bytes());
        tag[12..16].copy_from_slice(&(f3 as u32).to_le_bytes());
        tag
    }
}

fn poly1305_aead_tag(otk: &[u8; 32], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    let mut poly = Poly1305::new(otk);
    poly.update_padded(aad);
    poly.update_padded(ciphertext);
    let mut lens = [0u8; 16];
    lens[..8].copy_from_slice(&(aad.len() as u64).to_le_bytes());
    lens[8..16].copy_from_slice(&(ciphertext.len() as u64).to_le_bytes());
    poly.block(&lens, 1 << 24);
    poly.finalize()
}

/// ChaCha20-Poly1305 AEAD encryption (RFC 8439).
pub fn chacha20_poly1305_seal(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    plaintext: &[u8],
) -> (Vec<u8>, [u8; 16]) {
    let block0 = chacha20_block(key, 0, nonce);
    let mut otk = [0u8; 32];
    otk.copy_from_slice(&block0[..32]);

    let mut ciphertext = plaintext.to_vec();
    chacha20_xor(key, 1, nonce, &mut ciphertext);
    let tag = poly1305_aead_tag(&otk, aad, &ciphertext);
    (ciphertext, tag)
}

/// ChaCha20-Poly1305 AEAD decryption (RFC 8439).
pub fn chacha20_poly1305_open(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
    tag: &[u8; 16],
) -> Result<Vec<u8>, TlsQuicError> {
    let block0 = chacha20_block(key, 0, nonce);
    let mut otk = [0u8; 32];
    otk.copy_from_slice(&block0[..32]);

    let expected = poly1305_aead_tag(&otk, aad, ciphertext);
    if !ct_eq(&expected, tag) {
        return Err(TlsQuicError::AuthFailed);
    }

    let mut plaintext = ciphertext.to_vec();
    chacha20_xor(key, 1, nonce, &mut plaintext);
    Ok(plaintext)
}

// ============================================================================
// 7. Kernel TLS 1.3 (kTLS) Record Framing & Offload Engine
// ============================================================================

/// TLS 1.3 record content types (RFC 8446 Section 5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContentType {
    Invalid = 0,
    ChangeCipherSpec = 20,
    Alert = 21,
    Handshake = 22,
    ApplicationData = 23,
}

impl ContentType {
    pub const fn from_u8(val: u8) -> Result<Self, TlsQuicError> {
        match val {
            20 => Ok(Self::ChangeCipherSpec),
            21 => Ok(Self::Alert),
            22 => Ok(Self::Handshake),
            23 => Ok(Self::ApplicationData),
            _ => Err(TlsQuicError::InvalidRecord),
        }
    }
}

/// 5-byte TLS record header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlsRecordHeader {
    pub content_type: ContentType,
    pub legacy_version: u16,
    pub length: u16,
}

impl TlsRecordHeader {
    pub fn encode(&self) -> [u8; 5] {
        let len_be = self.length.to_be_bytes();
        let ver_be = self.legacy_version.to_be_bytes();
        [
            self.content_type as u8,
            ver_be[0],
            ver_be[1],
            len_be[0],
            len_be[1],
        ]
    }

    pub fn decode(buf: &[u8]) -> Result<Self, TlsQuicError> {
        if buf.len() < 5 {
            return Err(TlsQuicError::InvalidRecord);
        }
        Ok(Self {
            content_type: ContentType::from_u8(buf[0])?,
            legacy_version: u16::from_be_bytes([buf[1], buf[2]]),
            length: u16::from_be_bytes([buf[3], buf[4]]),
        })
    }
}

/// Per-direction kTLS 1.3 cryptographic state.
#[derive(Clone)]
pub struct KtlsDirectionState {
    pub cipher: CipherSuite,
    pub traffic_secret: [u8; 32],
    pub keys: TrafficKeys,
    pub seq: u64,
    pub records_processed: u64,
    pub bytes_processed: u64,
}

impl KtlsDirectionState {
    pub fn new(cipher: CipherSuite, traffic_secret: [u8; 32]) -> Self {
        let keys = Tls13KeySchedule::derive_traffic_keys(&traffic_secret, cipher);
        Self {
            cipher,
            traffic_secret,
            keys,
            seq: 0,
            records_processed: 0,
            bytes_processed: 0,
        }
    }

    /// Construct per-record nonce: `iv XOR padded_seq` (RFC 8446 Section 5.3).
    pub fn nonce_for_seq(&self, seq: u64) -> [u8; 12] {
        let mut nonce = self.keys.iv;
        let seq_be = seq.to_be_bytes();
        for (n, &s) in nonce[4..12].iter_mut().zip(seq_be.iter()) {
            *n ^= s;
        }
        nonce
    }

    /// Perform TLS 1.3 `KeyUpdate` for this direction.
    pub fn key_update(&mut self) {
        self.traffic_secret = Tls13KeySchedule::update_traffic_secret(&self.traffic_secret);
        self.keys = Tls13KeySchedule::derive_traffic_keys(&self.traffic_secret, self.cipher);
        self.seq = 0;
    }
}

/// Kernel TLS 1.3 offload session (bidirectional TX/RX).
pub struct KtlsSession {
    pub id: u64,
    pub tx: KtlsDirectionState,
    pub rx: KtlsDirectionState,
}

impl KtlsSession {
    pub fn new(id: u64, cipher: CipherSuite, tx_secret: [u8; 32], rx_secret: [u8; 32]) -> Self {
        Self {
            id,
            tx: KtlsDirectionState::new(cipher, tx_secret),
            rx: KtlsDirectionState::new(cipher, rx_secret),
        }
    }

    /// Encrypt a TLS 1.3 record (`TLSInnerPlaintext` -> `TLSCiphertext`).
    pub fn encrypt_record(
        &mut self,
        content_type: ContentType,
        plaintext: &[u8],
        padding: usize,
    ) -> Result<Vec<u8>, TlsQuicError> {
        if plaintext.len() > TLS_MAX_PLAINTEXT_LEN {
            return Err(TlsQuicError::RecordTooLarge);
        }
        let inner_len = plaintext.len() + 1 + padding;
        let ct_len = inner_len + self.tx.cipher.tag_len();
        if ct_len > TLS_MAX_CIPHERTEXT_LEN {
            return Err(TlsQuicError::RecordTooLarge);
        }
        if self.tx.seq == u64::MAX {
            return Err(TlsQuicError::SequenceOverflow);
        }

        let mut inner = Vec::with_capacity(inner_len);
        inner.extend_from_slice(plaintext);
        inner.push(content_type as u8);
        inner.resize(inner_len, 0);

        let hdr = TlsRecordHeader {
            content_type: ContentType::ApplicationData,
            legacy_version: TLS_LEGACY_VERSION,
            length: ct_len as u16,
        };
        let aad = hdr.encode();
        let nonce = self.tx.nonce_for_seq(self.tx.seq);

        let (ct, tag) = match self.tx.cipher {
            CipherSuite::TlsAes128GcmSha256 | CipherSuite::TlsAes256GcmSha256 => aes_gcm_seal(
                &self.tx.keys.key[..self.tx.keys.key_len],
                &nonce,
                &aad,
                &inner,
            )?,
            CipherSuite::TlsChacha20Poly1305Sha256 => {
                chacha20_poly1305_seal(&self.tx.keys.key, &nonce, &aad, &inner)
            }
        };

        self.tx.seq += 1;
        self.tx.records_processed += 1;
        self.tx.bytes_processed += plaintext.len() as u64;

        let mut record = Vec::with_capacity(5 + ct_len);
        record.extend_from_slice(&aad);
        record.extend_from_slice(&ct);
        record.extend_from_slice(&tag);
        Ok(record)
    }

    /// Decrypt a TLS 1.3 `TLSCiphertext` record and strip trailing zero padding.
    pub fn decrypt_record(
        &mut self,
        record: &[u8],
    ) -> Result<(ContentType, Vec<u8>), TlsQuicError> {
        let hdr = TlsRecordHeader::decode(record)?;
        if hdr.content_type != ContentType::ApplicationData
            || hdr.legacy_version != TLS_LEGACY_VERSION
        {
            return Err(TlsQuicError::InvalidRecord);
        }
        let payload_len = hdr.length as usize;
        if !(17..=TLS_MAX_CIPHERTEXT_LEN).contains(&payload_len) || record.len() < 5 + payload_len {
            return Err(TlsQuicError::InvalidRecord);
        }
        if self.rx.seq == u64::MAX {
            return Err(TlsQuicError::SequenceOverflow);
        }

        let aad = &record[..5];
        let ct_end = 5 + payload_len - 16;
        let ciphertext = &record[5..ct_end];
        let mut tag = [0u8; 16];
        tag.copy_from_slice(&record[ct_end..5 + payload_len]);
        let nonce = self.rx.nonce_for_seq(self.rx.seq);

        let mut inner = match self.rx.cipher {
            CipherSuite::TlsAes128GcmSha256 | CipherSuite::TlsAes256GcmSha256 => aes_gcm_open(
                &self.rx.keys.key[..self.rx.keys.key_len],
                &nonce,
                aad,
                ciphertext,
                &tag,
            )?,
            CipherSuite::TlsChacha20Poly1305Sha256 => {
                chacha20_poly1305_open(&self.rx.keys.key, &nonce, aad, ciphertext, &tag)?
            }
        };

        // Strip trailing zero padding to find the real ContentType byte.
        while inner.last() == Some(&0) {
            inner.pop();
        }
        let ct_byte = inner.pop().ok_or(TlsQuicError::InvalidRecord)?;
        let real_ct = ContentType::from_u8(ct_byte)?;

        self.rx.seq += 1;
        self.rx.records_processed += 1;
        self.rx.bytes_processed += inner.len() as u64;

        Ok((real_ct, inner))
    }
}

// ============================================================================
// 8. QUIC Transport Offload Engine (RFC 9000 / RFC 9001)
// ============================================================================

/// Encode a QUIC variable-length integer (RFC 9000 Section 16).
pub fn encode_varint(val: u64, out: &mut Vec<u8>) -> Result<(), TlsQuicError> {
    if val < (1 << 6) {
        out.push(val as u8);
    } else if val < (1 << 14) {
        let v = (val as u16) | 0x4000;
        out.extend_from_slice(&v.to_be_bytes());
    } else if val < (1 << 30) {
        let v = (val as u32) | 0x8000_0000;
        out.extend_from_slice(&v.to_be_bytes());
    } else if val < (1 << 62) {
        let v = val | 0xc000_0000_0000_0000;
        out.extend_from_slice(&v.to_be_bytes());
    } else {
        return Err(TlsQuicError::InvalidVarInt);
    }
    Ok(())
}

/// Decode a QUIC variable-length integer, returning `(value, bytes_consumed)`.
pub fn decode_varint(buf: &[u8]) -> Result<(u64, usize), TlsQuicError> {
    if buf.is_empty() {
        return Err(TlsQuicError::InvalidVarInt);
    }
    let prefix = buf[0] >> 6;
    let len = 1usize << prefix;
    if buf.len() < len {
        return Err(TlsQuicError::InvalidVarInt);
    }
    let val = match len {
        1 => u64::from(buf[0] & 0x3f),
        2 => u64::from(u16::from_be_bytes([buf[0] & 0x3f, buf[1]])),
        4 => u64::from(u32::from_be_bytes([buf[0] & 0x3f, buf[1], buf[2], buf[3]])),
        8 => u64::from_be_bytes([
            buf[0] & 0x3f,
            buf[1],
            buf[2],
            buf[3],
            buf[4],
            buf[5],
            buf[6],
            buf[7],
        ]),
        _ => return Err(TlsQuicError::InvalidVarInt),
    };
    Ok((val, len))
}

/// QUIC packet types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuicPacketType {
    Initial,
    ZeroRtt,
    Handshake,
    Retry,
    OneRtt,
}

/// QUIC Long Header (Initial, 0-RTT, Handshake).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuicLongHeader {
    pub packet_type: QuicPacketType,
    pub version: u32,
    pub dcid: Vec<u8>,
    pub scid: Vec<u8>,
    pub token: Vec<u8>,
    pub pn_len: u8,
}

/// QUIC Short Header (1-RTT).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuicShortHeader {
    pub spin_bit: bool,
    pub key_phase: bool,
    pub dcid: Vec<u8>,
    pub pn_len: u8,
}

/// QUIC Frame types (RFC 9000 Section 19).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuicFrame {
    Padding(usize),
    Ping,
    Ack {
        largest_acked: u64,
        ack_delay: u64,
        first_ack_range: u64,
    },
    Crypto {
        offset: u64,
        data: Vec<u8>,
    },
    Stream {
        stream_id: u64,
        offset: u64,
        fin: bool,
        data: Vec<u8>,
    },
    MaxData(u64),
    MaxStreamData {
        stream_id: u64,
        max_data: u64,
    },
    ConnectionClose {
        error_code: u64,
        frame_type: u64,
        reason: Vec<u8>,
    },
}

impl QuicFrame {
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), TlsQuicError> {
        match self {
            Self::Padding(count) => {
                out.resize(out.len() + *count, 0);
            }
            Self::Ping => out.push(0x01),
            Self::Ack {
                largest_acked,
                ack_delay,
                first_ack_range,
            } => {
                out.push(0x02);
                encode_varint(*largest_acked, out)?;
                encode_varint(*ack_delay, out)?;
                encode_varint(0, out)?; // ack_range_count = 0
                encode_varint(*first_ack_range, out)?;
            }
            Self::Crypto { offset, data } => {
                out.push(0x06);
                encode_varint(*offset, out)?;
                encode_varint(data.len() as u64, out)?;
                out.extend_from_slice(data);
            }
            Self::Stream {
                stream_id,
                offset,
                fin,
                data,
            } => {
                let mut ftype = 0x08u8 | 0x02; // LEN bit always set
                if *offset > 0 {
                    ftype |= 0x04; // OFF bit
                }
                if *fin {
                    ftype |= 0x01; // FIN bit
                }
                out.push(ftype);
                encode_varint(*stream_id, out)?;
                if *offset > 0 {
                    encode_varint(*offset, out)?;
                }
                encode_varint(data.len() as u64, out)?;
                out.extend_from_slice(data);
            }
            Self::MaxData(max) => {
                out.push(0x10);
                encode_varint(*max, out)?;
            }
            Self::MaxStreamData {
                stream_id,
                max_data,
            } => {
                out.push(0x11);
                encode_varint(*stream_id, out)?;
                encode_varint(*max_data, out)?;
            }
            Self::ConnectionClose {
                error_code,
                frame_type,
                reason,
            } => {
                out.push(0x1c);
                encode_varint(*error_code, out)?;
                encode_varint(*frame_type, out)?;
                encode_varint(reason.len() as u64, out)?;
                out.extend_from_slice(reason);
            }
        }
        Ok(())
    }

    pub fn decode(buf: &[u8]) -> Result<(Self, usize), TlsQuicError> {
        if buf.is_empty() {
            return Err(TlsQuicError::InvalidPacket);
        }
        let ftype = buf[0];
        let mut pos = 1usize;
        match ftype {
            0x00 => {
                while pos < buf.len() && buf[pos] == 0x00 {
                    pos += 1;
                }
                Ok((Self::Padding(pos), pos))
            }
            0x01 => Ok((Self::Ping, 1)),
            0x02 => {
                let (largest_acked, n) = decode_varint(&buf[pos..])?;
                pos += n;
                let (ack_delay, n) = decode_varint(&buf[pos..])?;
                pos += n;
                let (range_count, n) = decode_varint(&buf[pos..])?;
                pos += n;
                let (first_ack_range, n) = decode_varint(&buf[pos..])?;
                pos += n;
                for _ in 0..range_count {
                    let (_, g) = decode_varint(&buf[pos..])?;
                    pos += g;
                    let (_, r) = decode_varint(&buf[pos..])?;
                    pos += r;
                }
                Ok((
                    Self::Ack {
                        largest_acked,
                        ack_delay,
                        first_ack_range,
                    },
                    pos,
                ))
            }
            0x06 => {
                let (offset, n) = decode_varint(&buf[pos..])?;
                pos += n;
                let (len, n) = decode_varint(&buf[pos..])?;
                pos += n;
                let len = len as usize;
                if buf.len() < pos + len {
                    return Err(TlsQuicError::InvalidPacket);
                }
                let data = buf[pos..pos + len].to_vec();
                pos += len;
                Ok((Self::Crypto { offset, data }, pos))
            }
            0x08..=0x0f => {
                let has_off = (ftype & 0x04) != 0;
                let has_len = (ftype & 0x02) != 0;
                let fin = (ftype & 0x01) != 0;
                let (stream_id, n) = decode_varint(&buf[pos..])?;
                pos += n;
                let offset = if has_off {
                    let (off, n) = decode_varint(&buf[pos..])?;
                    pos += n;
                    off
                } else {
                    0
                };
                let data = if has_len {
                    let (len, n) = decode_varint(&buf[pos..])?;
                    pos += n;
                    let len = len as usize;
                    if buf.len() < pos + len {
                        return Err(TlsQuicError::InvalidPacket);
                    }
                    let d = buf[pos..pos + len].to_vec();
                    pos += len;
                    d
                } else {
                    let d = buf[pos..].to_vec();
                    pos = buf.len();
                    d
                };
                Ok((
                    Self::Stream {
                        stream_id,
                        offset,
                        fin,
                        data,
                    },
                    pos,
                ))
            }
            0x10 => {
                let (max, n) = decode_varint(&buf[pos..])?;
                Ok((Self::MaxData(max), pos + n))
            }
            0x11 => {
                let (stream_id, n) = decode_varint(&buf[pos..])?;
                pos += n;
                let (max_data, n) = decode_varint(&buf[pos..])?;
                Ok((
                    Self::MaxStreamData {
                        stream_id,
                        max_data,
                    },
                    pos + n,
                ))
            }
            0x1c => {
                let (error_code, n) = decode_varint(&buf[pos..])?;
                pos += n;
                let (frame_type, n) = decode_varint(&buf[pos..])?;
                pos += n;
                let (rlen, n) = decode_varint(&buf[pos..])?;
                pos += n;
                let rlen = rlen as usize;
                if buf.len() < pos + rlen {
                    return Err(TlsQuicError::InvalidPacket);
                }
                let reason = buf[pos..pos + rlen].to_vec();
                Ok((
                    Self::ConnectionClose {
                        error_code,
                        frame_type,
                        reason,
                    },
                    pos + rlen,
                ))
            }
            _ => Err(TlsQuicError::InvalidPacket),
        }
    }
}

/// Compute 5-byte QUIC Header Protection mask (RFC 9001 Section 5.4).
fn quic_hp_mask(keys: &QuicKeys, sample: &[u8; 16]) -> Result<[u8; 5], TlsQuicError> {
    let mut mask = [0u8; 5];
    match keys.cipher {
        CipherSuite::TlsAes128GcmSha256 | CipherSuite::TlsAes256GcmSha256 => {
            let aes = AesCipher::new(&keys.hp_key[..keys.key_len])?;
            let out = aes.encrypt_block(sample);
            mask.copy_from_slice(&out[..5]);
        }
        CipherSuite::TlsChacha20Poly1305Sha256 => {
            let counter = u32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
            let mut nonce = [0u8; 12];
            nonce.copy_from_slice(&sample[4..16]);
            let out = chacha20_block(&keys.hp_key, counter, &nonce);
            mask.copy_from_slice(&out[..5]);
        }
    }
    Ok(mask)
}

/// Construct QUIC AEAD nonce: `iv XOR padded_pn` (RFC 9001 Section 5.3).
fn quic_nonce(iv: &[u8; 12], pn: u64) -> [u8; 12] {
    let mut nonce = *iv;
    let pn_be = pn.to_be_bytes();
    for (n, &p) in nonce[4..12].iter_mut().zip(pn_be.iter()) {
        *n ^= p;
    }
    nonce
}

/// Per-stream state with out-of-order reassembly and flow control.
pub struct QuicStream {
    pub id: u64,
    pub send_offset: u64,
    pub recv_offset: u64,
    pub max_send_data: u64,
    pub max_recv_data: u64,
    pub fin_sent: bool,
    pub fin_recv: bool,
    pub recv_buf: Vec<u8>,
    pub reorder_chunks: BTreeMap<u64, Vec<u8>>,
}

impl QuicStream {
    pub fn new(id: u64, initial_max_data: u64) -> Self {
        Self {
            id,
            send_offset: 0,
            recv_offset: 0,
            max_send_data: initial_max_data,
            max_recv_data: initial_max_data,
            fin_sent: false,
            fin_recv: false,
            recv_buf: Vec::new(),
            reorder_chunks: BTreeMap::new(),
        }
    }

    fn push_recv_data(&mut self, offset: u64, data: &[u8], fin: bool) -> Result<(), TlsQuicError> {
        let end = offset.saturating_add(data.len() as u64);
        if end > self.max_recv_data {
            return Err(TlsQuicError::FlowControlExceeded);
        }
        if fin {
            self.fin_recv = true;
        }
        if end <= self.recv_offset {
            return Ok(());
        }
        if offset <= self.recv_offset {
            let skip = (self.recv_offset - offset) as usize;
            self.recv_buf.extend_from_slice(&data[skip..]);
            self.recv_offset = end;
            // Drain contiguous buffered chunks.
            while let Some((&next_off, _)) = self.reorder_chunks.first_key_value() {
                if next_off > self.recv_offset {
                    break;
                }
                let chunk = self.reorder_chunks.remove(&next_off).unwrap_or_default();
                let chunk_end = next_off.saturating_add(chunk.len() as u64);
                if chunk_end > self.recv_offset {
                    let cskip = (self.recv_offset - next_off) as usize;
                    self.recv_buf.extend_from_slice(&chunk[cskip..]);
                    self.recv_offset = chunk_end;
                }
            }
        } else {
            self.reorder_chunks.insert(offset, data.to_vec());
        }
        Ok(())
    }
}

/// QUIC 1-RTT Transport Offload Connection.
pub struct QuicConnection {
    pub id: u64,
    pub dcid: Vec<u8>,
    pub scid: Vec<u8>,
    pub tx_keys: QuicKeys,
    pub rx_keys: QuicKeys,
    pub next_pn: u64,
    pub largest_recv_pn: u64,
    pub next_client_bidi_stream: u64,
    pub next_server_bidi_stream: u64,
    pub streams: BTreeMap<u64, QuicStream>,
    pub packets_tx: u64,
    pub packets_rx: u64,
}

impl QuicConnection {
    pub fn new(
        id: u64,
        dcid: Vec<u8>,
        scid: Vec<u8>,
        tx_keys: QuicKeys,
        rx_keys: QuicKeys,
    ) -> Self {
        Self {
            id,
            dcid,
            scid,
            tx_keys,
            rx_keys,
            next_pn: 0,
            largest_recv_pn: 0,
            next_client_bidi_stream: 0,
            next_server_bidi_stream: 1,
            streams: BTreeMap::new(),
            packets_tx: 0,
            packets_rx: 0,
        }
    }

    /// Open a new bidirectional QUIC stream.
    pub fn open_stream(&mut self, is_client: bool, initial_max_data: u64) -> u64 {
        let sid = if is_client {
            let s = self.next_client_bidi_stream;
            self.next_client_bidi_stream += 4;
            s
        } else {
            let s = self.next_server_bidi_stream;
            self.next_server_bidi_stream += 4;
            s
        };
        self.streams
            .insert(sid, QuicStream::new(sid, initial_max_data));
        sid
    }

    /// Frame stream data into a 1-RTT QUIC packet with AEAD + Header Protection.
    pub fn send_stream_data(
        &mut self,
        stream_id: u64,
        data: &[u8],
        fin: bool,
    ) -> Result<Vec<u8>, TlsQuicError> {
        let stream = self
            .streams
            .get_mut(&stream_id)
            .ok_or(TlsQuicError::StreamNotFound)?;
        if stream.fin_sent {
            return Err(TlsQuicError::StreamClosed);
        }
        let new_end = stream.send_offset.saturating_add(data.len() as u64);
        if new_end > stream.max_send_data {
            return Err(TlsQuicError::FlowControlExceeded);
        }

        let frame = QuicFrame::Stream {
            stream_id,
            offset: stream.send_offset,
            fin,
            data: data.to_vec(),
        };
        stream.send_offset = new_end;
        if fin {
            stream.fin_sent = true;
        }

        let mut payload = Vec::with_capacity(data.len() + 16);
        frame.encode(&mut payload)?;
        // Ensure at least 4 bytes of payload so sample at pn_offset + 4 is always valid.
        if payload.len() < 4 {
            payload.resize(4, 0);
        }

        let pn = self.next_pn;
        self.next_pn += 1;
        let pn_len = 4usize;
        let first_byte = 0x40u8 | ((pn_len as u8) - 1); // Short header, fixed bit=1, 4-byte PN

        let mut header = Vec::with_capacity(1 + self.dcid.len() + pn_len);
        header.push(first_byte);
        header.extend_from_slice(&self.dcid);
        let pn_offset = header.len();
        header.extend_from_slice(&(pn as u32).to_be_bytes());

        let nonce = quic_nonce(&self.tx_keys.iv, pn);
        let (ct, tag) = match self.tx_keys.cipher {
            CipherSuite::TlsAes128GcmSha256 | CipherSuite::TlsAes256GcmSha256 => aes_gcm_seal(
                &self.tx_keys.key[..self.tx_keys.key_len],
                &nonce,
                &header,
                &payload,
            )?,
            CipherSuite::TlsChacha20Poly1305Sha256 => {
                chacha20_poly1305_seal(&self.tx_keys.key, &nonce, &header, &payload)
            }
        };

        let mut packet = Vec::with_capacity(header.len() + ct.len() + tag.len());
        packet.extend_from_slice(&header);
        packet.extend_from_slice(&ct);
        packet.extend_from_slice(&tag);

        // Apply Header Protection (RFC 9001 Section 5.4.1).
        let sample_offset = pn_offset + 4;
        if packet.len() < sample_offset + 16 {
            return Err(TlsQuicError::InvalidPacket);
        }
        let mut sample = [0u8; 16];
        sample.copy_from_slice(&packet[sample_offset..sample_offset + 16]);
        let mask = quic_hp_mask(&self.tx_keys, &sample)?;
        packet[0] ^= mask[0] & 0x1f;
        for (p, &m) in packet[pn_offset..pn_offset + pn_len]
            .iter_mut()
            .zip(mask[1..1 + pn_len].iter())
        {
            *p ^= m;
        }

        self.packets_tx += 1;
        Ok(packet)
    }

    /// Receive, unprotect, decrypt, and demultiplex a 1-RTT QUIC packet.
    pub fn recv_packet(&mut self, raw_packet: &[u8]) -> Result<Vec<QuicFrame>, TlsQuicError> {
        let dcid_len = self.dcid.len();
        if raw_packet.len() < 1 + dcid_len + 4 + 16 {
            return Err(TlsQuicError::InvalidPacket);
        }
        if (raw_packet[0] & 0x80) != 0 {
            return Err(TlsQuicError::InvalidPacket);
        }

        let mut packet = raw_packet.to_vec();
        let pn_offset = 1 + dcid_len;
        let sample_offset = pn_offset + 4;
        if packet.len() < sample_offset + 16 {
            return Err(TlsQuicError::InvalidPacket);
        }
        let mut sample = [0u8; 16];
        sample.copy_from_slice(&packet[sample_offset..sample_offset + 16]);
        let mask = quic_hp_mask(&self.rx_keys, &sample)?;

        packet[0] ^= mask[0] & 0x1f;
        let pn_len = ((packet[0] & 0x03) as usize) + 1;
        for (p, &m) in packet[pn_offset..pn_offset + pn_len]
            .iter_mut()
            .zip(mask[1..1 + pn_len].iter())
        {
            *p ^= m;
        }

        let mut pn_bytes = [0u8; 8];
        pn_bytes[8 - pn_len..8].copy_from_slice(&packet[pn_offset..pn_offset + pn_len]);
        let pn = u64::from_be_bytes(pn_bytes);

        let aad_end = pn_offset + pn_len;
        let ct_end = packet.len() - 16;
        let mut tag = [0u8; 16];
        tag.copy_from_slice(&packet[ct_end..]);
        let nonce = quic_nonce(&self.rx_keys.iv, pn);

        let plaintext = match self.rx_keys.cipher {
            CipherSuite::TlsAes128GcmSha256 | CipherSuite::TlsAes256GcmSha256 => aes_gcm_open(
                &self.rx_keys.key[..self.rx_keys.key_len],
                &nonce,
                &packet[..aad_end],
                &packet[aad_end..ct_end],
                &tag,
            )?,
            CipherSuite::TlsChacha20Poly1305Sha256 => chacha20_poly1305_open(
                &self.rx_keys.key,
                &nonce,
                &packet[..aad_end],
                &packet[aad_end..ct_end],
                &tag,
            )?,
        };

        if pn > self.largest_recv_pn {
            self.largest_recv_pn = pn;
        }
        self.packets_rx += 1;

        let mut frames = Vec::new();
        let mut pos = 0usize;
        while pos < plaintext.len() {
            let (frame, consumed) = QuicFrame::decode(&plaintext[pos..])?;
            pos += consumed;
            if let QuicFrame::Stream {
                stream_id,
                offset,
                fin,
                ref data,
            } = frame
            {
                let stream = self
                    .streams
                    .entry(stream_id)
                    .or_insert_with(|| QuicStream::new(stream_id, 1024 * 1024));
                stream.push_recv_data(offset, data, fin)?;
            }
            frames.push(frame);
        }
        Ok(frames)
    }

    /// Read reassembled in-order bytes from a QUIC stream.
    pub fn read_stream(&mut self, stream_id: u64, max_len: usize) -> Result<Vec<u8>, TlsQuicError> {
        let stream = self
            .streams
            .get_mut(&stream_id)
            .ok_or(TlsQuicError::StreamNotFound)?;
        let take = max_len.min(stream.recv_buf.len());
        let out: Vec<u8> = stream.recv_buf.drain(..take).collect();
        Ok(out)
    }
}

// ============================================================================
// 9. Global Offload Engine Registry
// ============================================================================

pub struct TlsQuicEngine {
    pub ktls_sessions: BTreeMap<u64, KtlsSession>,
    pub quic_conns: BTreeMap<u64, QuicConnection>,
}

impl TlsQuicEngine {
    const fn new() -> Self {
        Self {
            ktls_sessions: BTreeMap::new(),
            quic_conns: BTreeMap::new(),
        }
    }
}

static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);
static ENGINE: LocalSpinLock<TlsQuicEngine> = LocalSpinLock::new(TlsQuicEngine::new());

pub fn register_ktls_session(cipher: CipherSuite, tx_secret: [u8; 32], rx_secret: [u8; 32]) -> u64 {
    let id = NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed);
    let session = KtlsSession::new(id, cipher, tx_secret, rx_secret);
    ENGINE.lock().ktls_sessions.insert(id, session);
    id
}

pub fn register_quic_connection(
    dcid: Vec<u8>,
    scid: Vec<u8>,
    tx_keys: QuicKeys,
    rx_keys: QuicKeys,
) -> u64 {
    let id = NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed);
    let conn = QuicConnection::new(id, dcid, scid, tx_keys, rx_keys);
    ENGINE.lock().quic_conns.insert(id, conn);
    id
}

// ============================================================================
// 10. Self-Test Verification Suite
// ============================================================================

pub fn self_test() -> Result<(), &'static str> {
    // 1. SHA-256("abc") NIST vector
    let digest = sha256(b"abc");
    let expected_sha256: [u8; 32] = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];
    if digest != expected_sha256 {
        return Err("SHA-256 NIST vector mismatch");
    }

    // 2. HMAC-SHA256 RFC 4231 Test Case 2 ("Jefe" / "what do ya want for nothing?")
    let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
    let expected_hmac: [u8; 32] = [
        0x5b, 0xdc, 0xc1, 0x46, 0xbf, 0x60, 0x75, 0x4e, 0x6a, 0x04, 0x24, 0x26, 0x08, 0x95, 0x75,
        0xc7, 0x5a, 0x00, 0x3f, 0x08, 0x9d, 0x27, 0x39, 0x83, 0x9d, 0xec, 0x58, 0xb9, 0x64, 0xec,
        0x38, 0x43,
    ];
    if mac != expected_hmac {
        return Err("HMAC-SHA256 RFC 4231 vector mismatch");
    }

    // 3. HKDF-SHA256 RFC 5869 Test Case 1
    let ikm = [0x0bu8; 22];
    let salt: [u8; 13] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
    ];
    let info: [u8; 10] = [0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9];
    let prk = hkdf_extract(&salt, &ikm);
    let mut okm = [0u8; 42];
    hkdf_expand(&prk, &info, &mut okm).map_err(|_| "HKDF expand failed")?;
    let expected_okm_prefix: [u8; 8] = [0x3c, 0xb2, 0x5f, 0x25, 0xfa, 0xac, 0xd5, 0x7a];
    if okm[..8] != expected_okm_prefix {
        return Err("HKDF-SHA256 RFC 5869 vector mismatch");
    }

    // 4. AES-128-GCM NIST SP 800-38D Test Case 2 (zero key, zero nonce, 16-byte zero plaintext)
    let aes_key = [0u8; 16];
    let aes_nonce = [0u8; 12];
    let (ct, tag) = aes_gcm_seal(&aes_key, &aes_nonce, &[], &[0u8; 16])
        .map_err(|_| "AES-128-GCM seal failed")?;
    let expected_ct: [u8; 16] = [
        0x03, 0x88, 0xda, 0xce, 0x60, 0xb6, 0xa3, 0x92, 0xf3, 0x28, 0xc2, 0xb9, 0x71, 0xb2, 0xfe,
        0x78,
    ];
    let expected_tag: [u8; 16] = [
        0xab, 0x6e, 0x47, 0xd4, 0x2c, 0xec, 0x13, 0xbd, 0xf5, 0x3a, 0x67, 0xb2, 0x12, 0x57, 0xbd,
        0xdf,
    ];
    if ct.as_slice() != expected_ct || tag != expected_tag {
        return Err("AES-128-GCM NIST vector mismatch");
    }
    let pt = aes_gcm_open(&aes_key, &aes_nonce, &[], &ct, &tag)
        .map_err(|_| "AES-128-GCM open failed")?;
    if pt != [0u8; 16] {
        return Err("AES-128-GCM plaintext mismatch");
    }

    // 5. AES-256-GCM round-trip + tamper detection
    let aes256_key = [0x42u8; 32];
    let (ct256, mut tag256) = aes_gcm_seal(&aes256_key, &aes_nonce, b"aad", b"nexus-ktls-256")
        .map_err(|_| "AES-256-GCM seal failed")?;
    let pt256 = aes_gcm_open(&aes256_key, &aes_nonce, b"aad", &ct256, &tag256)
        .map_err(|_| "AES-256-GCM open failed")?;
    if pt256 != b"nexus-ktls-256" {
        return Err("AES-256-GCM round-trip mismatch");
    }
    tag256[0] ^= 0x01;
    if aes_gcm_open(&aes256_key, &aes_nonce, b"aad", &ct256, &tag256).is_ok() {
        return Err("AES-256-GCM failed to reject tampered tag");
    }

    // 6. ChaCha20-Poly1305 round-trip + tamper detection
    let cc_key = [0x80u8; 32];
    let cc_nonce = [0x07u8; 12];
    let (cc_ct, mut cc_tag) = chacha20_poly1305_seal(&cc_key, &cc_nonce, b"hdr", b"chacha-ktls");
    let cc_pt = chacha20_poly1305_open(&cc_key, &cc_nonce, b"hdr", &cc_ct, &cc_tag)
        .map_err(|_| "ChaCha20-Poly1305 open failed")?;
    if cc_pt != b"chacha-ktls" {
        return Err("ChaCha20-Poly1305 round-trip mismatch");
    }
    cc_tag[15] ^= 0x80;
    if chacha20_poly1305_open(&cc_key, &cc_nonce, b"hdr", &cc_ct, &cc_tag).is_ok() {
        return Err("ChaCha20-Poly1305 failed to reject tampered tag");
    }

    // 7. kTLS 1.3 record framing, padding, and KeyUpdate
    let early = Tls13KeySchedule::early_secret(None);
    let hs = Tls13KeySchedule::handshake_secret(&early, &[0x11u8; 32]);
    let master = Tls13KeySchedule::master_secret(&hs);
    let c_secret = derive_secret(&master, b"c ap traffic", &Tls13KeySchedule::EMPTY_HASH);
    let s_secret = derive_secret(&master, b"s ap traffic", &Tls13KeySchedule::EMPTY_HASH);

    let mut client_ktls = KtlsSession::new(1, CipherSuite::TlsAes128GcmSha256, c_secret, s_secret);
    let mut server_ktls = KtlsSession::new(2, CipherSuite::TlsAes128GcmSha256, s_secret, c_secret);

    let rec = client_ktls
        .encrypt_record(ContentType::ApplicationData, b"GET / HTTP/1.1\r\n", 12)
        .map_err(|_| "kTLS encrypt failed")?;
    let (ct_type, dec) = server_ktls
        .decrypt_record(&rec)
        .map_err(|_| "kTLS decrypt failed")?;
    if ct_type != ContentType::ApplicationData || dec != b"GET / HTTP/1.1\r\n" {
        return Err("kTLS record round-trip mismatch");
    }

    client_ktls.tx.key_update();
    server_ktls.rx.key_update();
    let rec2 = client_ktls
        .encrypt_record(ContentType::Alert, &[1, 0], 0)
        .map_err(|_| "kTLS post-KeyUpdate encrypt failed")?;
    let (ct_type2, dec2) = server_ktls
        .decrypt_record(&rec2)
        .map_err(|_| "kTLS post-KeyUpdate decrypt failed")?;
    if ct_type2 != ContentType::Alert || dec2 != [1, 0] {
        return Err("kTLS KeyUpdate round-trip mismatch");
    }

    // 8. QUIC VarInt, Initial keys, 1-RTT packet protection & out-of-order stream mux
    let mut vbuf = Vec::new();
    for &val in &[
        0u64,
        63,
        64,
        16383,
        16384,
        1_073_741_823,
        4_611_686_018_427_387_903,
    ] {
        vbuf.clear();
        encode_varint(val, &mut vbuf).map_err(|_| "QUIC varint encode failed")?;
        let (decoded, used) = decode_varint(&vbuf).map_err(|_| "QUIC varint decode failed")?;
        if decoded != val || used != vbuf.len() {
            return Err("QUIC varint round-trip mismatch");
        }
    }

    let dcid = vec![0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
    let (q_client_keys, q_server_keys) = Tls13KeySchedule::quic_initial_keys(&dcid);
    let mut q_client = QuicConnection::new(
        1,
        dcid.clone(),
        vec![1, 2, 3, 4],
        q_client_keys,
        q_server_keys,
    );
    let mut q_server = QuicConnection::new(2, dcid, vec![5, 6, 7, 8], q_server_keys, q_client_keys);

    let sid = q_client.open_stream(true, 65536);
    let pkt1 = q_client
        .send_stream_data(sid, b"hello ", false)
        .map_err(|_| "QUIC send pkt1 failed")?;
    let pkt2 = q_client
        .send_stream_data(sid, b"quic!", true)
        .map_err(|_| "QUIC send pkt2 failed")?;

    // Deliver pkt2 before pkt1 to verify out-of-order stream reassembly.
    q_server
        .recv_packet(&pkt2)
        .map_err(|_| "QUIC recv pkt2 failed")?;
    let early_read = q_server
        .read_stream(sid, 64)
        .map_err(|_| "QUIC early read failed")?;
    if !early_read.is_empty() {
        return Err("QUIC out-of-order data leaked before gap filled");
    }
    q_server
        .recv_packet(&pkt1)
        .map_err(|_| "QUIC recv pkt1 failed")?;
    let full_read = q_server
        .read_stream(sid, 64)
        .map_err(|_| "QUIC full read failed")?;
    if full_read != b"hello quic!" {
        return Err("QUIC stream reassembly mismatch");
    }

    Ok(())
}
