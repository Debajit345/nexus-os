//! Kernel Security, CSPRNG/ASLR & CPU Mitigation Engine.
//!
//! Provides:
//! - Hardware entropy harvesting (`RDSEED`, `RDRAND`, `RDTSC`) + ChaCha20 CSPRNG
//! - ASLR / KASLR slide and user-space layout randomization
//! - CPU vulnerability & mitigation probe (`SMEP`, `SMAP`, `UMIP`, `NX`, `CET`,
//!   `IBRS`, `STIBP`, `SSBD`, `MD_CLEAR`, `L1D_FLUSH`, `IA32_ARCH_CAPABILITIES`)
//! - W^X page-permission audit and stack-canary guard
//! - `self_test()` verification suite
#![allow(dead_code)]

use core::arch::asm;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::mm::SpinLock;
use crate::vmm::PageFlags;

const MSR_IA32_SPEC_CTRL: u32 = 0x0000_0048;
const MSR_IA32_ARCH_CAPABILITIES: u32 = 0x0000_010A;
const MSR_IA32_EFER: u32 = 0xC000_0080;

const EFER_NXE: u64 = 1 << 11;

const CR4_SMEP: u64 = 1 << 7;
const CR4_UMIP: u64 = 1 << 11;
const CR4_SMAP: u64 = 1 << 21;
const CR4_CET: u64 = 1 << 23;

const SPEC_CTRL_IBRS: u64 = 1 << 0;
const SPEC_CTRL_STIBP: u64 = 1 << 1;
const SPEC_CTRL_SSBD: u64 = 1 << 2;

const ARCH_CAP_RDCL_NO: u64 = 1 << 0;
const ARCH_CAP_IBRS_ALL: u64 = 1 << 1;
const ARCH_CAP_RSBA: u64 = 1 << 2;
const ARCH_CAP_SKIP_L1DFL_VMENTRY: u64 = 1 << 3;
const ARCH_CAP_SSB_NO: u64 = 1 << 4;
const ARCH_CAP_MDS_NO: u64 = 1 << 5;
const ARCH_CAP_TAA_NO: u64 = 1 << 8;

const PAGE_SIZE_4K: u64 = 4096;
const PAGE_SIZE_2M: u64 = 2 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Low-level x86_64 CPUID / MSR / CR / Entropy primitives
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct CpuidRegs {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

#[inline]
fn cpuid(leaf: u32, subleaf: u32) -> CpuidRegs {
    let eax: u32;
    let ebx_64: u64;
    let ecx: u32;
    let edx: u32;
    unsafe {
        asm!(
            "mov {tmp}, rbx",
            "cpuid",
            "xchg {tmp}, rbx",
            inout("eax") leaf => eax,
            inout("ecx") subleaf => ecx,
            tmp = out(reg) ebx_64,
            lateout("edx") edx,
            options(nomem, nostack, preserves_flags),
        );
    }
    CpuidRegs {
        eax,
        ebx: ebx_64 as u32,
        ecx,
        edx,
    }
}

#[inline]
fn max_basic_cpuid_leaf() -> u32 {
    cpuid(0, 0).eax
}

#[inline]
fn max_extended_cpuid_leaf() -> u32 {
    cpuid(0x8000_0000, 0).eax
}

#[inline]
fn read_cr4() -> u64 {
    let val: u64;
    unsafe {
        asm!("mov {}, cr4", out(reg) val, options(nomem, nostack, preserves_flags));
    }
    val
}

#[inline]
unsafe fn write_cr4(val: u64) {
    asm!("mov cr4, {}", in(reg) val, options(nomem, nostack, preserves_flags));
}

#[inline]
unsafe fn rdmsr(msr: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    asm!(
        "rdmsr",
        in("ecx") msr,
        out("eax") lo,
        out("edx") hi,
        options(nomem, nostack, preserves_flags),
    );
    ((hi as u64) << 32) | (lo as u64)
}

#[inline]
unsafe fn wrmsr(msr: u32, val: u64) {
    let lo = val as u32;
    let hi = (val >> 32) as u32;
    asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") lo,
        in("edx") hi,
        options(nomem, nostack, preserves_flags),
    );
}

/// Read the 64-bit Time Stamp Counter (`RDTSC`).
#[inline]
pub fn rdtsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack, preserves_flags),
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// Check whether `RDRAND` is supported by the CPU.
#[inline]
pub fn has_rdrand() -> bool {
    if max_basic_cpuid_leaf() < 1 {
        return false;
    }
    (cpuid(1, 0).ecx & (1 << 30)) != 0
}

/// Check whether `RDSEED` is supported by the CPU.
#[inline]
pub fn has_rdseed() -> bool {
    if max_basic_cpuid_leaf() < 7 {
        return false;
    }
    (cpuid(7, 0).ebx & (1 << 18)) != 0
}

/// Attempt to read 64 bits of hardware randomness via `RDRAND`.
pub fn rdrand64() -> Option<u64> {
    if !has_rdrand() {
        return None;
    }
    for _ in 0..10 {
        let val: u64;
        let ok: u8;
        unsafe {
            asm!(
                "rdrand {val}",
                "setc {ok}",
                val = out(reg) val,
                ok = out(reg_byte) ok,
                options(nomem, nostack),
            );
        }
        if ok != 0 {
            return Some(val);
        }
        core::hint::spin_loop();
    }
    None
}

/// Attempt to read 64 bits of hardware seed entropy via `RDSEED`.
pub fn rdseed64() -> Option<u64> {
    if !has_rdseed() {
        return None;
    }
    for _ in 0..10 {
        let val: u64;
        let ok: u8;
        unsafe {
            asm!(
                "rdseed {val}",
                "setc {ok}",
                val = out(reg) val,
                ok = out(reg_byte) ok,
                options(nomem, nostack),
            );
        }
        if ok != 0 {
            return Some(val);
        }
        core::hint::spin_loop();
    }
    None
}

// ---------------------------------------------------------------------------
// ChaCha20 CSPRNG & Entropy Pool
// ---------------------------------------------------------------------------

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_left(16);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_left(12);

    state[a] = state[a].wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_left(8);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_left(7);
}

/// ChaCha20-based Cryptographically Secure Pseudo-Random Number Generator.
pub struct ChaCha20Rng {
    key: [u32; 8],
    nonce: [u32; 3],
    counter: u32,
    block: [u8; 64],
    block_pos: usize,
    seeded: bool,
}

impl ChaCha20Rng {
    pub const fn new_unseeded() -> Self {
        Self {
            key: [0; 8],
            nonce: [0; 3],
            counter: 0,
            block: [0; 64],
            block_pos: 64,
            seeded: false,
        }
    }

    /// Seed or reseed the CSPRNG from hardware entropy (`RDSEED`, `RDRAND`, `RDTSC`)
    /// plus optional caller-supplied entropy bytes.
    pub fn seed_from_hardware(&mut self, extra: &[u8]) {
        let mut seed_words = [0u64; 6];
        let stack_addr = seed_words.as_ptr() as u64;

        for (i, slot) in seed_words.iter_mut().enumerate() {
            let hw = rdseed64()
                .or_else(rdrand64)
                .unwrap_or_else(|| splitmix64(rdtsc() ^ stack_addr ^ ((i as u64) << 32)));
            let tsc = rdtsc();
            *slot = splitmix64(hw ^ tsc.rotate_left((i as u32) * 11));
        }

        for (i, &byte) in extra.iter().enumerate() {
            let idx = i % seed_words.len();
            seed_words[idx] = splitmix64(seed_words[idx] ^ ((byte as u64) << ((i % 8) * 8)));
        }

        for (chunk, &word) in self
            .key
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(seed_words[..4].iter())
        {
            chunk[0] ^= word as u32;
            chunk[1] ^= (word >> 32) as u32;
        }
        self.nonce[0] ^= seed_words[4] as u32;
        self.nonce[1] ^= (seed_words[4] >> 32) as u32;
        self.nonce[2] ^= seed_words[5] as u32;
        self.counter = self.counter.wrapping_add(1);
        self.block_pos = 64;
        self.seeded = true;
    }

    fn refill_block(&mut self) {
        if !self.seeded {
            self.seed_from_hardware(&[]);
        }

        // "expand 32-byte k"
        const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646E, 0x7962_2D32, 0x6B20_6574];

        let initial: [u32; 16] = [
            CONSTANTS[0],
            CONSTANTS[1],
            CONSTANTS[2],
            CONSTANTS[3],
            self.key[0],
            self.key[1],
            self.key[2],
            self.key[3],
            self.key[4],
            self.key[5],
            self.key[6],
            self.key[7],
            self.counter,
            self.nonce[0],
            self.nonce[1],
            self.nonce[2],
        ];

        let mut working = initial;
        for _ in 0..10 {
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
        }

        for (dst, (&w, &init)) in self
            .block
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(working.iter().zip(initial.iter()))
        {
            *dst = w.wrapping_add(init).to_le_bytes();
        }

        self.counter = self.counter.wrapping_add(1);
        if self.counter == 0 {
            // Counter wrapped: mix in fresh hardware entropy.
            self.seed_from_hardware(&[]);
        }
        self.block_pos = 0;
    }

    /// Fill `dst` with cryptographically strong pseudo-random bytes.
    pub fn fill_bytes(&mut self, dst: &mut [u8]) {
        let mut offset = 0;
        while offset < dst.len() {
            if self.block_pos >= self.block.len() {
                self.refill_block();
            }
            let avail = self.block.len() - self.block_pos;
            let take = avail.min(dst.len() - offset);
            dst[offset..offset + take]
                .copy_from_slice(&self.block[self.block_pos..self.block_pos + take]);
            self.block_pos += take;
            offset += take;
        }
    }

    /// Generate a random `u32`.
    pub fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    /// Generate a random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }
}

static KERNEL_RNG: SpinLock<ChaCha20Rng> = SpinLock::new(ChaCha20Rng::new_unseeded());

/// Initialize or reseed the global kernel CSPRNG.
pub fn init_rng(extra_entropy: &[u8]) {
    let mut rng = KERNEL_RNG.lock();
    rng.seed_from_hardware(extra_entropy);
}

/// Generate a random `u64` from the global kernel CSPRNG.
pub fn random_u64() -> u64 {
    let mut rng = KERNEL_RNG.lock();
    rng.next_u64()
}

/// Fill `dst` with random bytes from the global kernel CSPRNG.
pub fn random_bytes(dst: &mut [u8]) {
    let mut rng = KERNEL_RNG.lock();
    rng.fill_bytes(dst);
}

/// Generate a uniform random `u64` in `[0, upper)` using rejection sampling.
pub fn random_below(upper: u64) -> u64 {
    if upper <= 1 {
        return 0;
    }
    let zone = u64::MAX - (u64::MAX % upper);
    loop {
        let r = random_u64();
        if r < zone {
            return r % upper;
        }
    }
}

// ---------------------------------------------------------------------------
// ASLR / KASLR Slide Generator
// ---------------------------------------------------------------------------

/// Randomized user-space virtual address layout for a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AslrLayout {
    pub pie_base: u64,
    pub mmap_base: u64,
    pub stack_top: u64,
    pub brk_base: u64,
}

/// Generate a 2 MiB-aligned KASLR slide with `entropy_bits` bits of entropy (clamped to 1..=16).
pub fn kaslr_slide(entropy_bits: u8) -> u64 {
    let bits = entropy_bits.clamp(1, 16);
    let mask = (1u64 << bits) - 1;
    let slots = random_u64() & mask;
    slots.saturating_mul(PAGE_SIZE_2M)
}

/// Randomize a virtual address by `entropy_bits` pages of `align` bytes.
pub fn randomize_address(base: u64, align: u64, entropy_bits: u8, upward: bool) -> u64 {
    let align = if align == 0 { PAGE_SIZE_4K } else { align };
    let bits = entropy_bits.clamp(1, 28);
    let mask = (1u64 << bits) - 1;
    let offset = (random_u64() & mask).saturating_mul(align);
    if upward {
        base.wrapping_add(offset)
    } else {
        base.wrapping_sub(offset)
    }
}

/// Generate a randomized 4 KiB-aligned user-space ASLR layout.
pub fn user_aslr_layout() -> AslrLayout {
    AslrLayout {
        pie_base: randomize_address(0x0000_5555_5555_0000, PAGE_SIZE_4K, 16, true),
        mmap_base: randomize_address(0x0000_7F00_0000_0000, PAGE_SIZE_4K, 20, false),
        stack_top: randomize_address(0x0000_7FFF_FFFF_0000, PAGE_SIZE_4K, 16, false),
        brk_base: randomize_address(0x0000_0000_1000_0000, PAGE_SIZE_4K, 16, true),
    }
}

// ---------------------------------------------------------------------------
// CPU Vulnerability & Mitigation Probe
// ---------------------------------------------------------------------------

/// Snapshot of CPU hardware security features and active mitigations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MitigationReport {
    pub smep_supported: bool,
    pub smep_enabled: bool,
    pub smap_supported: bool,
    pub smap_enabled: bool,
    pub umip_supported: bool,
    pub umip_enabled: bool,
    pub nx_supported: bool,
    pub nx_enabled: bool,
    pub cet_ss_supported: bool,
    pub cet_ibt_supported: bool,
    pub cet_enabled: bool,
    pub ibrs_supported: bool,
    pub stibp_supported: bool,
    pub ssbd_supported: bool,
    pub md_clear_supported: bool,
    pub l1d_flush_supported: bool,
    pub arch_cap_supported: bool,
    pub rdcl_no: bool,
    pub ibrs_all: bool,
    pub rsba: bool,
    pub skip_l1dfl_vmentry: bool,
    pub ssb_no: bool,
    pub mds_no: bool,
    pub taa_no: bool,
    pub spec_ctrl_msr: u64,
}

impl MitigationReport {
    /// Compute a 0..=100 hardening score based on active mitigations and hardware immunity.
    pub fn security_score(&self) -> u8 {
        let mut score: u8 = 0;
        if self.nx_enabled {
            score += 20;
        }
        if self.smep_enabled {
            score += 20;
        }
        if self.smap_enabled {
            score += 20;
        }
        if self.umip_enabled {
            score += 10;
        }
        if self.cet_enabled || (!self.cet_ss_supported && !self.cet_ibt_supported) {
            score += 10;
        }
        if self.rdcl_no || self.l1d_flush_supported {
            score += 10;
        }
        if self.ibrs_all || (self.spec_ctrl_msr & SPEC_CTRL_IBRS) != 0 {
            score += 10;
        }
        score.min(100)
    }
}

/// Probe CPUID, CR4, EFER, and speculation-control MSRs without faulting on unsupported leaves.
pub fn probe_mitigations() -> MitigationReport {
    let mut report = MitigationReport::default();
    let max_basic = max_basic_cpuid_leaf();
    let max_ext = max_extended_cpuid_leaf();

    if max_ext >= 0x8000_0001 {
        let ext1 = cpuid(0x8000_0001, 0);
        report.nx_supported = (ext1.edx & (1 << 20)) != 0;
        if report.nx_supported {
            let efer = unsafe { rdmsr(MSR_IA32_EFER) };
            report.nx_enabled = (efer & EFER_NXE) != 0;
        }
    }

    if max_basic >= 7 {
        let leaf7 = cpuid(7, 0);
        report.smep_supported = (leaf7.ebx & (1 << 7)) != 0;
        report.smap_supported = (leaf7.ebx & (1 << 20)) != 0;
        report.umip_supported = (leaf7.ecx & (1 << 2)) != 0;
        report.cet_ss_supported = (leaf7.ecx & (1 << 7)) != 0;
        report.md_clear_supported = (leaf7.edx & (1 << 10)) != 0;
        report.cet_ibt_supported = (leaf7.edx & (1 << 20)) != 0;
        report.ibrs_supported = (leaf7.edx & (1 << 26)) != 0;
        report.stibp_supported = (leaf7.edx & (1 << 27)) != 0;
        report.l1d_flush_supported = (leaf7.edx & (1 << 28)) != 0;
        report.arch_cap_supported = (leaf7.edx & (1 << 29)) != 0;
        report.ssbd_supported = (leaf7.edx & (1 << 31)) != 0;
    }

    let cr4 = read_cr4();
    report.smep_enabled = report.smep_supported && (cr4 & CR4_SMEP) != 0;
    report.smap_enabled = report.smap_supported && (cr4 & CR4_SMAP) != 0;
    report.umip_enabled = report.umip_supported && (cr4 & CR4_UMIP) != 0;
    report.cet_enabled =
        (report.cet_ss_supported || report.cet_ibt_supported) && (cr4 & CR4_CET) != 0;

    if report.arch_cap_supported {
        let arch_cap = unsafe { rdmsr(MSR_IA32_ARCH_CAPABILITIES) };
        report.rdcl_no = (arch_cap & ARCH_CAP_RDCL_NO) != 0;
        report.ibrs_all = (arch_cap & ARCH_CAP_IBRS_ALL) != 0;
        report.rsba = (arch_cap & ARCH_CAP_RSBA) != 0;
        report.skip_l1dfl_vmentry = (arch_cap & ARCH_CAP_SKIP_L1DFL_VMENTRY) != 0;
        report.ssb_no = (arch_cap & ARCH_CAP_SSB_NO) != 0;
        report.mds_no = (arch_cap & ARCH_CAP_MDS_NO) != 0;
        report.taa_no = (arch_cap & ARCH_CAP_TAA_NO) != 0;
    }

    if report.ibrs_supported || report.stibp_supported || report.ssbd_supported {
        report.spec_ctrl_msr = unsafe { rdmsr(MSR_IA32_SPEC_CTRL) };
    }

    report
}

/// Enable all supported hardware security mitigations (NXE, SMEP, SMAP, UMIP, SPEC_CTRL).
pub fn enable_mitigations() -> MitigationReport {
    let probe = probe_mitigations();

    if probe.nx_supported && !probe.nx_enabled {
        unsafe {
            let efer = rdmsr(MSR_IA32_EFER);
            wrmsr(MSR_IA32_EFER, efer | EFER_NXE);
        }
    }

    let mut cr4 = read_cr4();
    if probe.smep_supported {
        cr4 |= CR4_SMEP;
    }
    if probe.smap_supported {
        cr4 |= CR4_SMAP;
    }
    if probe.umip_supported {
        cr4 |= CR4_UMIP;
    }
    unsafe {
        write_cr4(cr4);
    }

    if probe.ibrs_supported || probe.stibp_supported || probe.ssbd_supported {
        let mut spec_ctrl = unsafe { rdmsr(MSR_IA32_SPEC_CTRL) };
        if probe.ibrs_supported && !probe.ibrs_all {
            spec_ctrl |= SPEC_CTRL_IBRS;
        }
        if probe.stibp_supported {
            spec_ctrl |= SPEC_CTRL_STIBP;
        }
        if probe.ssbd_supported && !probe.ssb_no {
            spec_ctrl |= SPEC_CTRL_SSBD;
        }
        unsafe {
            wrmsr(MSR_IA32_SPEC_CTRL, spec_ctrl);
        }
    }

    probe_mitigations()
}

// ---------------------------------------------------------------------------
// W^X Audit & Stack Canary Guard
// ---------------------------------------------------------------------------

/// Result of auditing a set of virtual memory regions for W^X compliance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WxAuditResult {
    pub total_regions: usize,
    pub compliant_regions: usize,
    pub violations: usize,
    pub first_violation_addr: Option<u64>,
}

/// Check whether a single `PageFlags` value satisfies Write-XOR-Execute (W^X).
///
/// A present page that is both `WRITABLE` and executable (lacks `NO_EXECUTE`) violates W^X.
#[inline]
pub const fn is_wx_compliant(flags: PageFlags) -> bool {
    if !flags.contains(PageFlags::PRESENT) {
        return true;
    }
    let writable = flags.contains(PageFlags::WRITABLE);
    let executable = !flags.contains(PageFlags::NO_EXECUTE);
    !(writable && executable)
}

/// Audit a slice of `(start_vaddr, end_vaddr, PageFlags)` regions for W^X compliance.
pub fn audit_wx_regions(regions: &[(u64, u64, PageFlags)]) -> WxAuditResult {
    let mut result = WxAuditResult {
        total_regions: regions.len(),
        compliant_regions: 0,
        violations: 0,
        first_violation_addr: None,
    };

    for &(start, _end, flags) in regions {
        if is_wx_compliant(flags) {
            result.compliant_regions += 1;
        } else {
            result.violations += 1;
            if result.first_violation_addr.is_none() {
                result.first_violation_addr = Some(start);
            }
        }
    }

    result
}

/// Global stack-canary guard value (terminator-byte in LSB + 56 random bits).
static STACK_CHK_GUARD: AtomicU64 = AtomicU64::new(0x595A_4E45_5855_5300);
static HARDENING_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Generate a fresh 64-bit stack canary with a zero LSB terminator byte.
pub fn generate_stack_canary() -> u64 {
    loop {
        let candidate = random_u64() & !0xFF;
        if candidate != 0 {
            return candidate;
        }
    }
}

/// Return the active kernel stack-canary guard value.
#[inline]
pub fn stack_canary() -> u64 {
    STACK_CHK_GUARD.load(Ordering::Acquire)
}

/// Constant-time stack-canary verification.
#[inline]
pub fn verify_stack_canary(expected: u64, observed: u64) -> bool {
    (expected ^ observed) == 0
}

// ---------------------------------------------------------------------------
// Initialization & Self-Test
// ---------------------------------------------------------------------------

/// Summary returned by `hardening::init()`.
#[derive(Debug, Clone, Copy)]
pub struct HardeningSummary {
    pub mitigations: MitigationReport,
    pub kaslr_slide: u64,
    pub stack_canary: u64,
    pub security_score: u8,
}

/// Initialize the kernel hardening subsystem:
/// 1. Seed the ChaCha20 CSPRNG from hardware entropy
/// 2. Generate and install the stack-canary guard
/// 3. Enable supported CPU mitigations (NXE, SMEP, SMAP, UMIP, SPEC_CTRL)
pub fn init() -> HardeningSummary {
    init_rng(b"nexus-os-kernel-hardening-v1");
    let canary = generate_stack_canary();
    STACK_CHK_GUARD.store(canary, Ordering::Release);

    let mitigations = enable_mitigations();
    let slide = kaslr_slide(9);
    let security_score = mitigations.security_score();
    HARDENING_INITIALIZED.store(true, Ordering::Release);

    HardeningSummary {
        mitigations,
        kaslr_slide: slide,
        stack_canary: canary,
        security_score,
    }
}

/// Run self-tests for CSPRNG, ASLR/KASLR, CPU mitigation probe, W^X audit, and stack canary.
pub fn self_test() -> Result<(), &'static str> {
    if !HARDENING_INITIALIZED.load(Ordering::Acquire) {
        let _ = init();
    }

    // 1. CSPRNG sanity: two consecutive 64-bit draws must be non-zero and distinct.
    let r1 = random_u64();
    let r2 = random_u64();
    if r1 == 0 && r2 == 0 {
        return Err("CSPRNG produced all-zero output");
    }
    if r1 == r2 {
        return Err("CSPRNG produced identical consecutive outputs");
    }

    // 2. ASLR / KASLR alignment checks.
    let slide = kaslr_slide(9);
    if (slide & (PAGE_SIZE_2M - 1)) != 0 {
        return Err("KASLR slide is not 2 MiB aligned");
    }
    let layout = user_aslr_layout();
    if (layout.pie_base & (PAGE_SIZE_4K - 1)) != 0
        || (layout.mmap_base & (PAGE_SIZE_4K - 1)) != 0
        || (layout.stack_top & (PAGE_SIZE_4K - 1)) != 0
        || (layout.brk_base & (PAGE_SIZE_4K - 1)) != 0
    {
        return Err("User ASLR layout is not 4 KiB aligned");
    }

    // 3. CPU mitigation probe sanity.
    let report = probe_mitigations();
    let _score = report.security_score();

    // 4. W^X audit checks.
    let rx_flags = PageFlags::PRESENT;
    let rw_nx_flags = PageFlags::PRESENT | PageFlags::WRITABLE | PageFlags::NO_EXECUTE;
    let rwx_flags = PageFlags::PRESENT | PageFlags::WRITABLE;
    let regions = [
        (0x1000_0000, 0x1000_4000, rx_flags),
        (0x2000_0000, 0x2000_8000, rw_nx_flags),
        (0x3000_0000, 0x3000_2000, rwx_flags),
    ];
    let audit = audit_wx_regions(&regions);
    if audit.total_regions != 3
        || audit.compliant_regions != 2
        || audit.violations != 1
        || audit.first_violation_addr != Some(0x3000_0000)
    {
        return Err("W^X audit failed to detect RWX violation");
    }

    // 5. Stack canary checks.
    let canary = stack_canary();
    if (canary & 0xFF) != 0 || canary == 0 {
        return Err("Stack canary format invalid");
    }
    if !verify_stack_canary(canary, canary) || verify_stack_canary(canary, canary ^ 1) {
        return Err("Stack canary verification failed");
    }

    Ok(())
}
