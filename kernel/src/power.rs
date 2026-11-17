//! Power management: ACPI S-states, Intel HWP / AMD CPPC detection, performance profiles, reboot & shutdown.
#![allow(dead_code)]

use core::arch::asm;

use crate::cpu::CpuVendor;
use crate::mm::SpinLock;

const MSR_IA32_PM_ENABLE: u32 = 0x770;
const MSR_IA32_HWP_CAPABILITIES: u32 = 0x771;
const MSR_IA32_HWP_REQUEST: u32 = 0x774;

/// ACPI system power states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerState {
    S0Working,
    S3Suspend,
    S5SoftOff,
}

/// CPU energy/performance preference profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfProfile {
    PowerSaver,
    Balanced,
    Performance,
}

/// Detected power management capabilities and current state.
#[derive(Debug, Clone, Copy)]
pub struct PowerInfo {
    pub vendor: CpuVendor,
    pub state: PowerState,
    pub profile: PerfProfile,
    pub hwp_supported: bool,
    pub hwp_enabled: bool,
    pub hwp_epp: bool,
    pub amd_hw_pstate: bool,
    pub amd_cppc: bool,
    pub invariant_tsc: bool,
}

impl PowerInfo {
    const fn new() -> Self {
        Self {
            vendor: CpuVendor::Unknown,
            state: PowerState::S0Working,
            profile: PerfProfile::Balanced,
            hwp_supported: false,
            hwp_enabled: false,
            hwp_epp: false,
            amd_hw_pstate: false,
            amd_cppc: false,
            invariant_tsc: false,
        }
    }
}

static POWER_INFO: SpinLock<PowerInfo> = SpinLock::new(PowerInfo::new());

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

#[inline]
unsafe fn outb(port: u16, val: u8) {
    asm!("out dx, al", in("dx") port, in("al") val, options(nomem, nostack, preserves_flags));
}

#[inline]
unsafe fn inb(port: u16) -> u8 {
    let val: u8;
    asm!("in al, dx", out("al") val, in("dx") port, options(nomem, nostack, preserves_flags));
    val
}

#[inline]
unsafe fn outw(port: u16, val: u16) {
    asm!("out dx, ax", in("dx") port, in("ax") val, options(nomem, nostack, preserves_flags));
}

/// Detect Intel HWP / AMD CPPC & P-state capabilities and initialize power management.
pub fn init(vendor: CpuVendor) -> PowerInfo {
    let mut info = PowerInfo::new();
    info.vendor = vendor;
    info.state = PowerState::S0Working;
    info.profile = PerfProfile::Balanced;

    let max_basic = cpuid(0, 0).eax;
    let max_ext = cpuid(0x8000_0000, 0).eax;

    if max_basic >= 0x06 {
        let leaf6 = cpuid(0x06, 0);
        if vendor == CpuVendor::Intel {
            info.hwp_supported = (leaf6.eax & (1 << 7)) != 0;
            info.hwp_epp = (leaf6.eax & (1 << 11)) != 0;
            if info.hwp_supported {
                unsafe {
                    let pm_enable = rdmsr(MSR_IA32_PM_ENABLE);
                    if (pm_enable & 1) == 0 {
                        wrmsr(MSR_IA32_PM_ENABLE, pm_enable | 1);
                    }
                    info.hwp_enabled = (rdmsr(MSR_IA32_PM_ENABLE) & 1) != 0;
                }
            }
        }
    }

    if max_ext >= 0x8000_0007 {
        let ext7 = cpuid(0x8000_0007, 0);
        info.amd_hw_pstate = (ext7.edx & (1 << 7)) != 0;
        info.invariant_tsc = (ext7.edx & (1 << 8)) != 0;
    }

    if vendor == CpuVendor::Amd && max_ext >= 0x8000_0008 {
        let ext8 = cpuid(0x8000_0008, 0);
        info.amd_cppc = (ext8.ebx & (1 << 27)) != 0;
    }

    *POWER_INFO.lock() = info;
    if info.hwp_enabled {
        set_perf_profile(PerfProfile::Balanced);
    }
    info
}

/// Return a snapshot of the current `PowerInfo`.
pub fn info() -> PowerInfo {
    *POWER_INFO.lock()
}

/// Set the active performance profile (updates Intel HWP EPP when enabled).
pub fn set_perf_profile(profile: PerfProfile) {
    let mut guard = POWER_INFO.lock();
    guard.profile = profile;

    if guard.hwp_enabled {
        let epp: u64 = match profile {
            PerfProfile::PowerSaver => 0xC0,
            PerfProfile::Balanced => 0x80,
            PerfProfile::Performance => 0x00,
        };
        unsafe {
            let mut req = rdmsr(MSR_IA32_HWP_REQUEST);
            if guard.hwp_epp {
                req = (req & !(0xFFu64 << 24)) | (epp << 24);
            }
            wrmsr(MSR_IA32_HWP_REQUEST, req);
        }
    }
}

#[repr(C, packed)]
struct NullIdtr {
    limit: u16,
    base: u64,
}

/// Reboot the machine via 8042 keyboard controller reset (`0x64`), with triple-fault fallback.
pub fn reboot() -> ! {
    unsafe {
        asm!("cli", options(nomem, nostack));

        // Wait for 8042 input buffer to clear, then pulse CPU reset line (0xFE).
        for _ in 0..100_000 {
            if (inb(0x64) & 0x02) == 0 {
                break;
            }
        }
        outb(0x64, 0xFE);

        // Triple-fault fallback: load null IDT and trigger an interrupt.
        let null_idtr = NullIdtr { limit: 0, base: 0 };
        asm!(
            "lidt [{0}]",
            "int3",
            in(reg) &null_idtr,
            options(nostack),
        );

        loop {
            asm!("hlt", options(nomem, nostack));
        }
    }
}

/// Power off QEMU/Bochs/VirtualBox via ACPI PM1a / debug-exit I/O ports.
pub fn shutdown_qemu() {
    POWER_INFO.lock().state = PowerState::S5SoftOff;
    unsafe {
        outw(0x604, 0x2000); // QEMU PIIX4 / Q35 ACPI shutdown
        outw(0xB004, 0x2000); // Bochs / older QEMU
        outw(0x4004, 0x3400); // VirtualBox
        outb(0x501, 0x00); // QEMU isa-debug-exit
    }
}
