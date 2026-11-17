#![allow(dead_code)]
//! x86_64 CPUID detection, modern Intel/AMD feature enablement, and per-CPU GS state.

use core::arch::asm;

const MSR_IA32_EFER: u32 = 0xC000_0080;
const MSR_IA32_GS_BASE: u32 = 0xC000_0101;
const MSR_IA32_KERNEL_GS_BASE: u32 = 0xC000_0102;

const EFER_NXE: u64 = 1 << 11;

const CR0_MP: u64 = 1 << 1;
const CR0_EM: u64 = 1 << 2;
const CR0_TS: u64 = 1 << 3;
const CR0_NE: u64 = 1 << 5;

const CR4_SMEP: u64 = 1 << 7;
const CR4_OSFXSR: u64 = 1 << 9;
const CR4_OSXMMEXCPT: u64 = 1 << 10;
const CR4_UMIP: u64 = 1 << 11;
const CR4_FSGSBASE: u64 = 1 << 16;
const CR4_PCIDE: u64 = 1 << 17;
const CR4_OSXSAVE: u64 = 1 << 18;
const CR4_SMAP: u64 = 1 << 21;

const XCR0_X87: u64 = 1 << 0;
const XCR0_SSE: u64 = 1 << 1;
const XCR0_AVX: u64 = 1 << 2;
const XCR0_OPMASK: u64 = 1 << 5;
const XCR0_ZMM_HI256: u64 = 1 << 6;
const XCR0_HI16_ZMM: u64 = 1 << 7;
const XCR0_AVX512_MASK: u64 = XCR0_OPMASK | XCR0_ZMM_HI256 | XCR0_HI16_ZMM;

pub const MAX_CPUS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVendor {
    Intel,
    Amd,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreType {
    Performance,
    Efficiency,
    Unified,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub xsave: bool,
    pub fsgsbase: bool,
    pub smep: bool,
    pub smap: bool,
    pub umip: bool,
    pub pcid: bool,
    pub invpcid: bool,
    pub nxe: bool,
    pub x2apic: bool,
    pub tsc_deadline: bool,
    pub rdrand: bool,
    pub rdseed: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct CpuInfo {
    pub vendor: CpuVendor,
    pub brand: [u8; 48],
    pub family: u32,
    pub model: u32,
    pub stepping: u32,
    pub core_type: CoreType,
    pub features: CpuFeatures,
}

impl CpuInfo {
    pub fn brand_str(&self) -> &str {
        let len = self
            .brand
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.brand.len());
        let s = core::str::from_utf8(&self.brand[..len]).unwrap_or("Unknown CPU");
        s.trim()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PerCpu {
    pub self_ptr: u64,
    pub cpu_id: u32,
    pub online: bool,
    pub kernel_rsp: u64,
    pub user_rsp: u64,
    pub current_task_id: u64,
}

impl PerCpu {
    const fn new() -> Self {
        Self {
            self_ptr: 0,
            cpu_id: 0,
            online: false,
            kernel_rsp: 0,
            user_rsp: 0,
            current_task_id: 0,
        }
    }
}

static mut PER_CPU: [PerCpu; MAX_CPUS] = [PerCpu::new(); MAX_CPUS];

#[derive(Debug, Clone, Copy)]
struct CpuidResult {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

#[inline]
fn cpuid(leaf: u32, subleaf: u32) -> CpuidResult {
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
    CpuidResult {
        eax,
        ebx: ebx_64 as u32,
        ecx,
        edx,
    }
}

#[inline]
unsafe fn read_cr0() -> u64 {
    let val: u64;
    asm!("mov {}, cr0", out(reg) val, options(nomem, nostack, preserves_flags));
    val
}

#[inline]
unsafe fn write_cr0(val: u64) {
    asm!("mov cr0, {}", in(reg) val, options(nomem, nostack, preserves_flags));
}

#[inline]
unsafe fn read_cr4() -> u64 {
    let val: u64;
    asm!("mov {}, cr4", out(reg) val, options(nomem, nostack, preserves_flags));
    val
}

#[inline]
unsafe fn write_cr4(val: u64) {
    asm!("mov cr4, {}", in(reg) val, options(nomem, nostack, preserves_flags));
}

#[inline]
unsafe fn rdmsr(msr: u32) -> u64 {
    let low: u32;
    let high: u32;
    asm!(
        "rdmsr",
        in("ecx") msr,
        out("eax") low,
        out("edx") high,
        options(nomem, nostack, preserves_flags),
    );
    ((high as u64) << 32) | (low as u64)
}

#[inline]
unsafe fn wrmsr(msr: u32, val: u64) {
    let low = val as u32;
    let high = (val >> 32) as u32;
    asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") low,
        in("edx") high,
        options(nomem, nostack, preserves_flags),
    );
}

#[inline]
unsafe fn xgetbv(xcr: u32) -> u64 {
    let low: u32;
    let high: u32;
    asm!(
        "xgetbv",
        in("ecx") xcr,
        out("eax") low,
        out("edx") high,
        options(nomem, nostack, preserves_flags),
    );
    ((high as u64) << 32) | (low as u64)
}

#[inline]
unsafe fn xsetbv(xcr: u32, val: u64) {
    let low = val as u32;
    let high = (val >> 32) as u32;
    asm!(
        "xsetbv",
        in("ecx") xcr,
        in("eax") low,
        in("edx") high,
        options(nomem, nostack, preserves_flags),
    );
}

fn detect_vendor(leaf0: CpuidResult) -> CpuVendor {
    // "GenuineIntel" -> EBX="Genu", EDX="ineI", ECX="ntel"
    if leaf0.ebx == 0x756e_6547 && leaf0.edx == 0x4965_6e69 && leaf0.ecx == 0x6c65_746e {
        CpuVendor::Intel
    // "AuthenticAMD" -> EBX="Auth", EDX="enti", ECX="cAMD"
    } else if leaf0.ebx == 0x6874_7541 && leaf0.edx == 0x6974_6e65 && leaf0.ecx == 0x444d_4163 {
        CpuVendor::Amd
    } else {
        CpuVendor::Unknown
    }
}

fn read_brand_string(max_ext_leaf: u32) -> [u8; 48] {
    let mut brand = [0u8; 48];
    if max_ext_leaf < 0x8000_0004 {
        return brand;
    }
    for (i, leaf) in [0x8000_0002u32, 0x8000_0003, 0x8000_0004]
        .iter()
        .enumerate()
    {
        let res = cpuid(*leaf, 0);
        let base = i * 16;
        brand[base..base + 4].copy_from_slice(&res.eax.to_le_bytes());
        brand[base + 4..base + 8].copy_from_slice(&res.ebx.to_le_bytes());
        brand[base + 8..base + 12].copy_from_slice(&res.ecx.to_le_bytes());
        brand[base + 12..base + 16].copy_from_slice(&res.edx.to_le_bytes());
    }
    brand
}

/// Detect CPU vendor, model, hybrid core type, and supported features via CPUID,
/// then enable supported architectural features in CR0, CR4, IA32_EFER, and XCR0.
pub fn detect_and_enable() -> CpuInfo {
    let leaf0 = cpuid(0, 0);
    let max_basic_leaf = leaf0.eax;
    let vendor = detect_vendor(leaf0);

    let ext0 = cpuid(0x8000_0000, 0);
    let max_ext_leaf = ext0.eax;
    let brand = read_brand_string(max_ext_leaf);

    let mut family = 0u32;
    let mut model = 0u32;
    let mut stepping = 0u32;
    let mut features = CpuFeatures::default();
    let mut fxsr = false;

    if max_basic_leaf >= 1 {
        let leaf1 = cpuid(1, 0);
        stepping = leaf1.eax & 0xF;
        let base_model = (leaf1.eax >> 4) & 0xF;
        let base_family = (leaf1.eax >> 8) & 0xF;
        let ext_model = (leaf1.eax >> 16) & 0xF;
        let ext_family = (leaf1.eax >> 20) & 0xFF;

        family = if base_family == 0xF {
            base_family + ext_family
        } else {
            base_family
        };
        model = if base_family == 0x6 || base_family == 0xF {
            (ext_model << 4) | base_model
        } else {
            base_model
        };

        features.pcid = (leaf1.ecx & (1 << 17)) != 0;
        features.x2apic = (leaf1.ecx & (1 << 21)) != 0;
        features.tsc_deadline = (leaf1.ecx & (1 << 24)) != 0;
        features.xsave = (leaf1.ecx & (1 << 26)) != 0;
        features.avx = (leaf1.ecx & (1 << 28)) != 0;
        features.rdrand = (leaf1.ecx & (1 << 30)) != 0;

        fxsr = (leaf1.edx & (1 << 24)) != 0;
        features.sse = (leaf1.edx & (1 << 25)) != 0;
        features.sse2 = (leaf1.edx & (1 << 26)) != 0;
    }

    let mut is_hybrid = false;
    if max_basic_leaf >= 7 {
        let leaf7 = cpuid(7, 0);
        features.fsgsbase = (leaf7.ebx & (1 << 0)) != 0;
        features.avx2 = (leaf7.ebx & (1 << 5)) != 0;
        features.smep = (leaf7.ebx & (1 << 7)) != 0;
        features.invpcid = (leaf7.ebx & (1 << 10)) != 0;
        features.avx512f = (leaf7.ebx & (1 << 16)) != 0;
        features.rdseed = (leaf7.ebx & (1 << 18)) != 0;
        features.smap = (leaf7.ebx & (1 << 20)) != 0;

        features.umip = (leaf7.ecx & (1 << 2)) != 0;
        is_hybrid = (leaf7.edx & (1 << 15)) != 0;
    }

    if max_ext_leaf >= 0x8000_0001 {
        let ext1 = cpuid(0x8000_0001, 0);
        features.nxe = (ext1.edx & (1 << 20)) != 0;
    }

    let core_type = if vendor == CpuVendor::Intel && is_hybrid && max_basic_leaf >= 0x1A {
        let leaf1a = cpuid(0x1A, 0);
        match (leaf1a.eax >> 24) & 0xFF {
            0x20 => CoreType::Efficiency,
            0x40 => CoreType::Performance,
            _ => CoreType::Unified,
        }
    } else {
        CoreType::Unified
    };

    unsafe {
        // CR0: enable Monitor Coprocessor (MP) and Numeric Error (NE), clear EM and TS.
        let mut cr0 = read_cr0();
        cr0 &= !(CR0_EM | CR0_TS);
        cr0 |= CR0_MP | CR0_NE;
        write_cr0(cr0);

        // CR4: enable only features reported by CPUID.
        let mut cr4 = read_cr4();
        if fxsr || features.sse {
            cr4 |= CR4_OSFXSR | CR4_OSXMMEXCPT;
        }
        if features.xsave {
            cr4 |= CR4_OSXSAVE;
        }
        if features.fsgsbase {
            cr4 |= CR4_FSGSBASE;
        }
        if features.smep {
            cr4 |= CR4_SMEP;
        }
        if features.smap {
            cr4 |= CR4_SMAP;
        }
        if features.umip {
            cr4 |= CR4_UMIP;
        }
        if features.pcid {
            cr4 |= CR4_PCIDE;
        }
        write_cr4(cr4);

        // IA32_EFER.NXE
        if features.nxe {
            let efer = rdmsr(MSR_IA32_EFER);
            wrmsr(MSR_IA32_EFER, efer | EFER_NXE);
        }

        // XCR0: enable x87 + SSE + AVX + AVX-512 if supported by both CPUID and XSAVE leaf 0x0D.
        if features.xsave && max_basic_leaf >= 0x0D {
            let leaf_d = cpuid(0x0D, 0);
            let supported_xcr0 = ((leaf_d.edx as u64) << 32) | (leaf_d.eax as u64);
            let mut xcr0 = xgetbv(0);
            xcr0 |= XCR0_X87;
            if (supported_xcr0 & XCR0_SSE) != 0 {
                xcr0 |= XCR0_SSE;
            }
            if features.avx && (supported_xcr0 & XCR0_AVX) != 0 {
                xcr0 |= XCR0_AVX;
            }
            if features.avx512f && (supported_xcr0 & XCR0_AVX512_MASK) == XCR0_AVX512_MASK {
                xcr0 |= XCR0_AVX512_MASK;
            }
            xsetbv(0, xcr0);
        }
    }

    CpuInfo {
        vendor,
        brand,
        family,
        model,
        stepping,
        core_type,
        features,
    }
}

/// Initialize per-CPU state for `cpu_id` and point `IA32_GS_BASE` at it
/// (clearing `IA32_KERNEL_GS_BASE` for user/kernel `swapgs` transitions).
pub fn init_per_cpu_gs(cpu_id: u32) {
    let idx = cpu_id as usize;
    if idx >= MAX_CPUS {
        return;
    }
    unsafe {
        let ptr = core::ptr::addr_of_mut!(PER_CPU[idx]);
        let addr = ptr as u64;
        (*ptr).self_ptr = addr;
        (*ptr).cpu_id = cpu_id;
        (*ptr).online = true;
        wrmsr(MSR_IA32_GS_BASE, addr);
        wrmsr(MSR_IA32_KERNEL_GS_BASE, 0);
    }
}

/// Return a pointer to the current CPU's `PerCpu` struct via `gs:[0]`.
pub fn current_per_cpu() -> *mut PerCpu {
    let ptr: u64;
    unsafe {
        asm!("mov {}, gs:[0]", out(reg) ptr, options(nomem, nostack, preserves_flags));
    }
    ptr as *mut PerCpu
}
