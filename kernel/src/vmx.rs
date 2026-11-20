#![allow(dead_code)]
//! Hardware Virtualization (Intel VT-x VMX / AMD-V SVM) Hypervisor & EPT/NPT Nested Paging Engine.
//!
//! Provides:
//! - CPUID & MSR capability detection for Intel VT-x (VMX + EPT + VPID) and AMD-V (SVM + NPT + ASID)
//! - VMCS field encoder/decoder and control-MSR bit adjustment
//! - 4-level EPT (Extended Page Tables) and AMD NPT guest-physical page table engine (4 KiB / 2 MiB / 1 GiB)
//! - AMD-V VMCB layout and SVM exit definitions
//! - VM-exit dispatcher, hypercall ABI, and vCPU state machine
//! - `self_test()` exercising encoding, EPT mapping/translation, and simulated VM-exit dispatch

extern crate alloc;

use alloc::vec::Vec;
use core::arch::asm;
use core::arch::x86_64::{__cpuid, __cpuid_count};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::cpu::{self, CpuVendor};
use crate::mm::{alloc_frame, dealloc_frame, stats, SpinLock, PAGE_SIZE};
use crate::serial_println;

// ---------------------------------------------------------------------------
// Constants & MSRs
// ---------------------------------------------------------------------------

const PHYS_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;
const TABLE_ENTRIES: usize = 512;

// Intel VMX MSRs
const MSR_IA32_FEATURE_CONTROL: u32 = 0x0000_003A;
const MSR_IA32_VMX_BASIC: u32 = 0x0000_0480;
const MSR_IA32_VMX_PINBASED_CTLS: u32 = 0x0000_0481;
const MSR_IA32_VMX_PROCBASED_CTLS: u32 = 0x0000_0482;
const MSR_IA32_VMX_EXIT_CTLS: u32 = 0x0000_0483;
const MSR_IA32_VMX_ENTRY_CTLS: u32 = 0x0000_0484;
const MSR_IA32_VMX_MISC: u32 = 0x0000_0485;
const MSR_IA32_VMX_CR0_FIXED0: u32 = 0x0000_0486;
const MSR_IA32_VMX_CR0_FIXED1: u32 = 0x0000_0487;
const MSR_IA32_VMX_CR4_FIXED0: u32 = 0x0000_0488;
const MSR_IA32_VMX_CR4_FIXED1: u32 = 0x0000_0489;
const MSR_IA32_VMX_PROCBASED_CTLS2: u32 = 0x0000_048B;
const MSR_IA32_VMX_EPT_VPID_CAP: u32 = 0x0000_048C;
const MSR_IA32_VMX_TRUE_PINBASED_CTLS: u32 = 0x0000_048D;
const MSR_IA32_VMX_TRUE_PROCBASED_CTLS: u32 = 0x0000_048E;
const MSR_IA32_VMX_TRUE_EXIT_CTLS: u32 = 0x0000_048F;
const MSR_IA32_VMX_TRUE_ENTRY_CTLS: u32 = 0x0000_0490;

// Common x86 MSRs
const MSR_IA32_APIC_BASE: u32 = 0x0000_001B;
const MSR_IA32_EFER: u32 = 0xC000_0080;
const MSR_IA32_TSC: u32 = 0x0000_0010;

// AMD SVM MSRs
const MSR_AMD_VM_CR: u32 = 0xC001_0114;
const MSR_AMD_VM_HSAVE_PA: u32 = 0xC001_0117;

// Feature Control bits
const FEATURE_CONTROL_LOCK: u64 = 1 << 0;
const FEATURE_CONTROL_VMXON_OUTSIDE_SMX: u64 = 1 << 2;

// CR0 / CR4 bits
const CR0_NE: u64 = 1 << 5;
const CR4_VMXE: u64 = 1 << 13;

// EFER bits
const EFER_SVME: u64 = 1 << 12;

// AMD VM_CR bits
const VM_CR_SVMDIS: u64 = 1 << 4;

// ---------------------------------------------------------------------------
// Error & Result Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmxError {
    Unsupported,
    DisabledInBios,
    AlreadyEnabled,
    VmxonFailed,
    VmclearFailed,
    VmptrldFailed,
    VmreadFailed,
    VmwriteFailed,
    VmlaunchFailed,
    VmresumeFailed,
    OutOfMemory,
    InvalidAddress,
    InvalidGuestState,
    InvalidVmcsField,
    SvmNotSupported,
    EptMappingConflict,
    NotInitialized,
}

pub type VmxResult<T> = Result<T, VmxError>;

// ---------------------------------------------------------------------------
// Capability Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtBackend {
    None,
    IntelVmx,
    AmdSvm,
}

#[derive(Debug, Clone, Copy)]
pub struct VirtCapabilities {
    pub backend: VirtBackend,
    pub vmx_supported: bool,
    pub svm_supported: bool,
    pub ept_supported: bool,
    pub vpid_supported: bool,
    pub unrestricted_guest: bool,
    pub vmfunc_supported: bool,
    pub pml_supported: bool,
    pub npt_supported: bool,
    pub nrip_save: bool,
    pub decode_assists: bool,
    pub flush_by_asid: bool,
    pub asid_count: u32,
    pub vmcs_revision_id: u32,
    pub vmcs_region_size: u16,
    pub ept_execute_only: bool,
    pub ept_page_walk_4: bool,
    pub ept_wb: bool,
    pub ept_2m: bool,
    pub ept_1g: bool,
    pub bios_locked: bool,
    pub bios_enabled: bool,
}

impl VirtCapabilities {
    pub const fn empty() -> Self {
        Self {
            backend: VirtBackend::None,
            vmx_supported: false,
            svm_supported: false,
            ept_supported: false,
            vpid_supported: false,
            unrestricted_guest: false,
            vmfunc_supported: false,
            pml_supported: false,
            npt_supported: false,
            nrip_save: false,
            decode_assists: false,
            flush_by_asid: false,
            asid_count: 0,
            vmcs_revision_id: 0,
            vmcs_region_size: 0,
            ept_execute_only: false,
            ept_page_walk_4: false,
            ept_wb: false,
            ept_2m: false,
            ept_1g: false,
            bios_locked: false,
            bios_enabled: false,
        }
    }
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
        options(nomem, nostack, preserves_flags)
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
        options(nomem, nostack, preserves_flags)
    );
}

#[inline]
fn read_cr0() -> u64 {
    let val: u64;
    unsafe {
        asm!("mov {}, cr0", out(reg) val, options(nomem, nostack, preserves_flags));
    }
    val
}

#[inline]
unsafe fn write_cr0(val: u64) {
    asm!("mov cr0, {}", in(reg) val, options(nomem, nostack, preserves_flags));
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

/// Detect Intel VT-x (VMX) or AMD-V (SVM) hardware virtualization capabilities.
pub fn detect_capabilities() -> VirtCapabilities {
    let mut caps = VirtCapabilities::empty();
    let cpu_info = cpu::detect_and_enable();

    match cpu_info.vendor {
        CpuVendor::Intel => {
            let leaf1 = __cpuid(1);
            let vmx = (leaf1.ecx & (1 << 5)) != 0;
            caps.vmx_supported = vmx;
            if vmx {
                caps.backend = VirtBackend::IntelVmx;
                let feat_ctrl = unsafe { rdmsr(MSR_IA32_FEATURE_CONTROL) };
                caps.bios_locked = (feat_ctrl & FEATURE_CONTROL_LOCK) != 0;
                caps.bios_enabled = (feat_ctrl & FEATURE_CONTROL_VMXON_OUTSIDE_SMX) != 0;

                let vmx_basic = unsafe { rdmsr(MSR_IA32_VMX_BASIC) };
                caps.vmcs_revision_id = (vmx_basic & 0x7FFF_FFFF) as u32;
                caps.vmcs_region_size = ((vmx_basic >> 32) & 0x1FFF) as u16;

                let proc_ctls = unsafe { rdmsr(MSR_IA32_VMX_PROCBASED_CTLS) };
                let secondary_allowed = ((proc_ctls >> 32) & (1 << 31)) != 0;
                if secondary_allowed {
                    let ctls2 = unsafe { rdmsr(MSR_IA32_VMX_PROCBASED_CTLS2) };
                    let allowed2 = (ctls2 >> 32) as u32;
                    caps.ept_supported = (allowed2 & (1 << 1)) != 0;
                    caps.vpid_supported = (allowed2 & (1 << 5)) != 0;
                    caps.unrestricted_guest = (allowed2 & (1 << 7)) != 0;
                    caps.vmfunc_supported = (allowed2 & (1 << 13)) != 0;
                    caps.pml_supported = (allowed2 & (1 << 17)) != 0;
                }

                if caps.ept_supported || caps.vpid_supported {
                    let ept_vpid = unsafe { rdmsr(MSR_IA32_VMX_EPT_VPID_CAP) };
                    caps.ept_execute_only = (ept_vpid & (1 << 0)) != 0;
                    caps.ept_page_walk_4 = (ept_vpid & (1 << 6)) != 0;
                    caps.ept_wb = (ept_vpid & (1 << 14)) != 0;
                    caps.ept_2m = (ept_vpid & (1 << 16)) != 0;
                    caps.ept_1g = (ept_vpid & (1 << 17)) != 0;
                }
            }
        }
        CpuVendor::Amd => {
            let max_ext = __cpuid(0x8000_0000).eax;
            if max_ext >= 0x8000_0001 {
                let ext1 = __cpuid(0x8000_0001);
                let svm = (ext1.ecx & (1 << 2)) != 0;
                caps.svm_supported = svm;
                if svm {
                    caps.backend = VirtBackend::AmdSvm;
                    let vm_cr = unsafe { rdmsr(MSR_AMD_VM_CR) };
                    caps.bios_locked = (vm_cr & VM_CR_SVMDIS) != 0;
                    caps.bios_enabled = !caps.bios_locked;

                    if max_ext >= 0x8000_000A {
                        let svm_leaf = __cpuid(0x8000_000A);
                        caps.vmcs_revision_id = svm_leaf.eax & 0xFF;
                        caps.asid_count = svm_leaf.ebx;
                        caps.npt_supported = (svm_leaf.edx & (1 << 0)) != 0;
                        caps.nrip_save = (svm_leaf.edx & (1 << 3)) != 0;
                        caps.flush_by_asid = (svm_leaf.edx & (1 << 6)) != 0;
                        caps.decode_assists = (svm_leaf.edx & (1 << 7)) != 0;
                    }
                }
            }
        }
        CpuVendor::Unknown => {}
    }

    caps
}

// ---------------------------------------------------------------------------
// VMCS Field Encoder & Constants
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VmcsFieldWidth {
    Bits16 = 0,
    Bits64 = 1,
    Bits32 = 2,
    Natural = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VmcsFieldType {
    Control = 0,
    VmExitInfo = 1,
    GuestState = 2,
    HostState = 3,
}

/// Intel SDM Vol. 3B Appendix B VMCS field encoding helper.
pub struct VmcsField;

impl VmcsField {
    /// Encode a VMCS field identifier:
    /// - bit 0: access type (0 = full, 1 = high)
    /// - bits 9:1: index
    /// - bits 11:10: field type
    /// - bits 14:13: width
    #[inline]
    pub const fn encode(
        access_high: bool,
        index: u32,
        field_type: VmcsFieldType,
        width: VmcsFieldWidth,
    ) -> u32 {
        let access = if access_high { 1 } else { 0 };
        access | ((index & 0x1FF) << 1) | ((field_type as u32) << 10) | ((width as u32) << 13)
    }

    /// Decode a VMCS field encoding into `(width, field_type, index, access_high)`.
    #[inline]
    pub const fn decode(encoding: u32) -> Option<(VmcsFieldWidth, VmcsFieldType, u32, bool)> {
        let access_high = (encoding & 1) != 0;
        let index = (encoding >> 1) & 0x1FF;
        let field_type = match (encoding >> 10) & 0x3 {
            0 => VmcsFieldType::Control,
            1 => VmcsFieldType::VmExitInfo,
            2 => VmcsFieldType::GuestState,
            3 => VmcsFieldType::HostState,
            _ => return None,
        };
        let width = match (encoding >> 13) & 0x3 {
            0 => VmcsFieldWidth::Bits16,
            1 => VmcsFieldWidth::Bits64,
            2 => VmcsFieldWidth::Bits32,
            3 => VmcsFieldWidth::Natural,
            _ => return None,
        };
        Some((width, field_type, index, access_high))
    }
}

pub mod vmcs_field {
    use super::{VmcsField, VmcsFieldType, VmcsFieldWidth};

    // 16-bit Control
    pub const VIRTUAL_PROCESSOR_ID: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::Control, VmcsFieldWidth::Bits16);

    // 16-bit Guest State
    pub const GUEST_ES_SELECTOR: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::GuestState, VmcsFieldWidth::Bits16);
    pub const GUEST_CS_SELECTOR: u32 =
        VmcsField::encode(false, 1, VmcsFieldType::GuestState, VmcsFieldWidth::Bits16);
    pub const GUEST_SS_SELECTOR: u32 =
        VmcsField::encode(false, 2, VmcsFieldType::GuestState, VmcsFieldWidth::Bits16);
    pub const GUEST_DS_SELECTOR: u32 =
        VmcsField::encode(false, 3, VmcsFieldType::GuestState, VmcsFieldWidth::Bits16);
    pub const GUEST_FS_SELECTOR: u32 =
        VmcsField::encode(false, 4, VmcsFieldType::GuestState, VmcsFieldWidth::Bits16);
    pub const GUEST_GS_SELECTOR: u32 =
        VmcsField::encode(false, 5, VmcsFieldType::GuestState, VmcsFieldWidth::Bits16);
    pub const GUEST_LDTR_SELECTOR: u32 =
        VmcsField::encode(false, 6, VmcsFieldType::GuestState, VmcsFieldWidth::Bits16);
    pub const GUEST_TR_SELECTOR: u32 =
        VmcsField::encode(false, 7, VmcsFieldType::GuestState, VmcsFieldWidth::Bits16);

    // 16-bit Host State
    pub const HOST_ES_SELECTOR: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::HostState, VmcsFieldWidth::Bits16);
    pub const HOST_CS_SELECTOR: u32 =
        VmcsField::encode(false, 1, VmcsFieldType::HostState, VmcsFieldWidth::Bits16);
    pub const HOST_SS_SELECTOR: u32 =
        VmcsField::encode(false, 2, VmcsFieldType::HostState, VmcsFieldWidth::Bits16);
    pub const HOST_DS_SELECTOR: u32 =
        VmcsField::encode(false, 3, VmcsFieldType::HostState, VmcsFieldWidth::Bits16);
    pub const HOST_FS_SELECTOR: u32 =
        VmcsField::encode(false, 4, VmcsFieldType::HostState, VmcsFieldWidth::Bits16);
    pub const HOST_GS_SELECTOR: u32 =
        VmcsField::encode(false, 5, VmcsFieldType::HostState, VmcsFieldWidth::Bits16);
    pub const HOST_TR_SELECTOR: u32 =
        VmcsField::encode(false, 6, VmcsFieldType::HostState, VmcsFieldWidth::Bits16);

    // 64-bit Control
    pub const IO_BITMAP_A: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::Control, VmcsFieldWidth::Bits64);
    pub const IO_BITMAP_B: u32 =
        VmcsField::encode(false, 1, VmcsFieldType::Control, VmcsFieldWidth::Bits64);
    pub const MSR_BITMAP: u32 =
        VmcsField::encode(false, 2, VmcsFieldType::Control, VmcsFieldWidth::Bits64);
    pub const EPT_POINTER: u32 =
        VmcsField::encode(false, 13, VmcsFieldType::Control, VmcsFieldWidth::Bits64);

    // 64-bit Read-Only Data
    pub const GUEST_PHYSICAL_ADDRESS: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::VmExitInfo, VmcsFieldWidth::Bits64);

    // 64-bit Guest State
    pub const VMCS_LINK_POINTER: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::GuestState, VmcsFieldWidth::Bits64);
    pub const GUEST_IA32_PAT: u32 =
        VmcsField::encode(false, 2, VmcsFieldType::GuestState, VmcsFieldWidth::Bits64);
    pub const GUEST_IA32_EFER: u32 =
        VmcsField::encode(false, 3, VmcsFieldType::GuestState, VmcsFieldWidth::Bits64);

    // 64-bit Host State
    pub const HOST_IA32_PAT: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::HostState, VmcsFieldWidth::Bits64);
    pub const HOST_IA32_EFER: u32 =
        VmcsField::encode(false, 1, VmcsFieldType::HostState, VmcsFieldWidth::Bits64);

    // 32-bit Control
    pub const PIN_BASED_VM_EXEC_CONTROL: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::Control, VmcsFieldWidth::Bits32);
    pub const CPU_BASED_VM_EXEC_CONTROL: u32 =
        VmcsField::encode(false, 1, VmcsFieldType::Control, VmcsFieldWidth::Bits32);
    pub const EXCEPTION_BITMAP: u32 =
        VmcsField::encode(false, 2, VmcsFieldType::Control, VmcsFieldWidth::Bits32);
    pub const VM_EXIT_CONTROLS: u32 =
        VmcsField::encode(false, 6, VmcsFieldType::Control, VmcsFieldWidth::Bits32);
    pub const VM_ENTRY_CONTROLS: u32 =
        VmcsField::encode(false, 9, VmcsFieldType::Control, VmcsFieldWidth::Bits32);
    pub const SECONDARY_VM_EXEC_CONTROL: u32 =
        VmcsField::encode(false, 15, VmcsFieldType::Control, VmcsFieldWidth::Bits32);

    // 32-bit Read-Only Data
    pub const VM_INSTRUCTION_ERROR: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::VmExitInfo, VmcsFieldWidth::Bits32);
    pub const EXIT_REASON: u32 =
        VmcsField::encode(false, 1, VmcsFieldType::VmExitInfo, VmcsFieldWidth::Bits32);
    pub const VM_EXIT_INTERRUPTION_INFO: u32 =
        VmcsField::encode(false, 2, VmcsFieldType::VmExitInfo, VmcsFieldWidth::Bits32);
    pub const VM_EXIT_INSTRUCTION_LEN: u32 =
        VmcsField::encode(false, 6, VmcsFieldType::VmExitInfo, VmcsFieldWidth::Bits32);

    // 32-bit Guest State
    pub const GUEST_ES_LIMIT: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_CS_LIMIT: u32 =
        VmcsField::encode(false, 1, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_SS_LIMIT: u32 =
        VmcsField::encode(false, 2, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_DS_LIMIT: u32 =
        VmcsField::encode(false, 3, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_FS_LIMIT: u32 =
        VmcsField::encode(false, 4, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_GS_LIMIT: u32 =
        VmcsField::encode(false, 5, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_LDTR_LIMIT: u32 =
        VmcsField::encode(false, 6, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_TR_LIMIT: u32 =
        VmcsField::encode(false, 7, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_GDTR_LIMIT: u32 =
        VmcsField::encode(false, 8, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_IDTR_LIMIT: u32 =
        VmcsField::encode(false, 9, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_ES_AR_BYTES: u32 =
        VmcsField::encode(false, 10, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_CS_AR_BYTES: u32 =
        VmcsField::encode(false, 11, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_SS_AR_BYTES: u32 =
        VmcsField::encode(false, 12, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_DS_AR_BYTES: u32 =
        VmcsField::encode(false, 13, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_FS_AR_BYTES: u32 =
        VmcsField::encode(false, 14, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_GS_AR_BYTES: u32 =
        VmcsField::encode(false, 15, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_LDTR_AR_BYTES: u32 =
        VmcsField::encode(false, 16, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_TR_AR_BYTES: u32 =
        VmcsField::encode(false, 17, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_INTERRUPTIBILITY_INFO: u32 =
        VmcsField::encode(false, 18, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);
    pub const GUEST_ACTIVITY_STATE: u32 =
        VmcsField::encode(false, 19, VmcsFieldType::GuestState, VmcsFieldWidth::Bits32);

    // Natural-Width Read-Only Data
    pub const EXIT_QUALIFICATION: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::VmExitInfo, VmcsFieldWidth::Natural);
    pub const GUEST_LINEAR_ADDRESS: u32 =
        VmcsField::encode(false, 5, VmcsFieldType::VmExitInfo, VmcsFieldWidth::Natural);

    // Natural-Width Guest State
    pub const GUEST_CR0: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::GuestState, VmcsFieldWidth::Natural);
    pub const GUEST_CR3: u32 =
        VmcsField::encode(false, 1, VmcsFieldType::GuestState, VmcsFieldWidth::Natural);
    pub const GUEST_CR4: u32 =
        VmcsField::encode(false, 2, VmcsFieldType::GuestState, VmcsFieldWidth::Natural);
    pub const GUEST_ES_BASE: u32 =
        VmcsField::encode(false, 3, VmcsFieldType::GuestState, VmcsFieldWidth::Natural);
    pub const GUEST_CS_BASE: u32 =
        VmcsField::encode(false, 4, VmcsFieldType::GuestState, VmcsFieldWidth::Natural);
    pub const GUEST_SS_BASE: u32 =
        VmcsField::encode(false, 5, VmcsFieldType::GuestState, VmcsFieldWidth::Natural);
    pub const GUEST_DS_BASE: u32 =
        VmcsField::encode(false, 6, VmcsFieldType::GuestState, VmcsFieldWidth::Natural);
    pub const GUEST_FS_BASE: u32 =
        VmcsField::encode(false, 7, VmcsFieldType::GuestState, VmcsFieldWidth::Natural);
    pub const GUEST_GS_BASE: u32 =
        VmcsField::encode(false, 8, VmcsFieldType::GuestState, VmcsFieldWidth::Natural);
    pub const GUEST_LDTR_BASE: u32 =
        VmcsField::encode(false, 9, VmcsFieldType::GuestState, VmcsFieldWidth::Natural);
    pub const GUEST_TR_BASE: u32 = VmcsField::encode(
        false,
        10,
        VmcsFieldType::GuestState,
        VmcsFieldWidth::Natural,
    );
    pub const GUEST_GDTR_BASE: u32 = VmcsField::encode(
        false,
        11,
        VmcsFieldType::GuestState,
        VmcsFieldWidth::Natural,
    );
    pub const GUEST_IDTR_BASE: u32 = VmcsField::encode(
        false,
        12,
        VmcsFieldType::GuestState,
        VmcsFieldWidth::Natural,
    );
    pub const GUEST_DR7: u32 = VmcsField::encode(
        false,
        13,
        VmcsFieldType::GuestState,
        VmcsFieldWidth::Natural,
    );
    pub const GUEST_RSP: u32 = VmcsField::encode(
        false,
        14,
        VmcsFieldType::GuestState,
        VmcsFieldWidth::Natural,
    );
    pub const GUEST_RIP: u32 = VmcsField::encode(
        false,
        15,
        VmcsFieldType::GuestState,
        VmcsFieldWidth::Natural,
    );
    pub const GUEST_RFLAGS: u32 = VmcsField::encode(
        false,
        16,
        VmcsFieldType::GuestState,
        VmcsFieldWidth::Natural,
    );

    // Natural-Width Host State
    pub const HOST_CR0: u32 =
        VmcsField::encode(false, 0, VmcsFieldType::HostState, VmcsFieldWidth::Natural);
    pub const HOST_CR3: u32 =
        VmcsField::encode(false, 1, VmcsFieldType::HostState, VmcsFieldWidth::Natural);
    pub const HOST_CR4: u32 =
        VmcsField::encode(false, 2, VmcsFieldType::HostState, VmcsFieldWidth::Natural);
    pub const HOST_FS_BASE: u32 =
        VmcsField::encode(false, 3, VmcsFieldType::HostState, VmcsFieldWidth::Natural);
    pub const HOST_GS_BASE: u32 =
        VmcsField::encode(false, 4, VmcsFieldType::HostState, VmcsFieldWidth::Natural);
    pub const HOST_TR_BASE: u32 =
        VmcsField::encode(false, 5, VmcsFieldType::HostState, VmcsFieldWidth::Natural);
    pub const HOST_GDTR_BASE: u32 =
        VmcsField::encode(false, 6, VmcsFieldType::HostState, VmcsFieldWidth::Natural);
    pub const HOST_IDTR_BASE: u32 =
        VmcsField::encode(false, 7, VmcsFieldType::HostState, VmcsFieldWidth::Natural);
    pub const HOST_RSP: u32 =
        VmcsField::encode(false, 10, VmcsFieldType::HostState, VmcsFieldWidth::Natural);
    pub const HOST_RIP: u32 =
        VmcsField::encode(false, 11, VmcsFieldType::HostState, VmcsFieldWidth::Natural);
}

/// Adjust VMX control bits against the capability MSR's allowed-0 (low 32) and allowed-1 (high 32) masks.
#[inline]
pub const fn adjust_vmx_controls(desired: u32, capability_msr: u64) -> u32 {
    let allowed_0 = capability_msr as u32;
    let allowed_1 = (capability_msr >> 32) as u32;
    (desired | allowed_0) & allowed_1
}

// ---------------------------------------------------------------------------
// Low-Level VMX Instruction Wrappers
// ---------------------------------------------------------------------------

#[inline]
unsafe fn vmxon(vmxon_phys: u64) -> VmxResult<()> {
    let ok: u8;
    asm!(
        "vmxon [{0}]",
        "setna {1}",
        in(reg) &vmxon_phys,
        out(reg_byte) ok,
        options(nostack)
    );
    if ok == 0 {
        Ok(())
    } else {
        Err(VmxError::VmxonFailed)
    }
}

#[inline]
unsafe fn vmxoff() {
    asm!("vmxoff", options(nomem, nostack));
}

#[inline]
unsafe fn vmclear(vmcs_phys: u64) -> VmxResult<()> {
    let ok: u8;
    asm!(
        "vmclear [{0}]",
        "setna {1}",
        in(reg) &vmcs_phys,
        out(reg_byte) ok,
        options(nostack)
    );
    if ok == 0 {
        Ok(())
    } else {
        Err(VmxError::VmclearFailed)
    }
}

#[inline]
unsafe fn vmptrld(vmcs_phys: u64) -> VmxResult<()> {
    let ok: u8;
    asm!(
        "vmptrld [{0}]",
        "setna {1}",
        in(reg) &vmcs_phys,
        out(reg_byte) ok,
        options(nostack)
    );
    if ok == 0 {
        Ok(())
    } else {
        Err(VmxError::VmptrldFailed)
    }
}

#[inline]
pub unsafe fn vmread(field: u32) -> VmxResult<u64> {
    let val: u64;
    let fail: u8;
    asm!(
        "vmread {0}, {1:r}",
        "setna {2}",
        out(reg) val,
        in(reg) field as u64,
        out(reg_byte) fail,
        options(nomem, nostack)
    );
    if fail == 0 {
        Ok(val)
    } else {
        Err(VmxError::VmreadFailed)
    }
}

#[inline]
pub unsafe fn vmwrite(field: u32, val: u64) -> VmxResult<()> {
    let fail: u8;
    asm!(
        "vmwrite {0:r}, {1}",
        "setna {2}",
        in(reg) field as u64,
        in(reg) val,
        out(reg_byte) fail,
        options(nomem, nostack)
    );
    if fail == 0 {
        Ok(())
    } else {
        Err(VmxError::VmwriteFailed)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InveptDescriptor {
    pub eptp: u64,
    pub reserved: u64,
}

#[inline]
pub unsafe fn invept(invalidation_type: u64, eptp: u64) -> VmxResult<()> {
    let desc = InveptDescriptor { eptp, reserved: 0 };
    let fail: u8;
    asm!(
        "invept {0}, [{1}]",
        "setna {2}",
        in(reg) invalidation_type,
        in(reg) &desc,
        out(reg_byte) fail,
        options(nostack)
    );
    if fail == 0 {
        Ok(())
    } else {
        Err(VmxError::VmwriteFailed)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InvvpidDescriptor {
    pub vpid: u16,
    pub reserved: [u16; 3],
    pub linear_addr: u64,
}

#[inline]
pub unsafe fn invvpid(invalidation_type: u64, vpid: u16, linear_addr: u64) -> VmxResult<()> {
    let desc = InvvpidDescriptor {
        vpid,
        reserved: [0; 3],
        linear_addr,
    };
    let fail: u8;
    asm!(
        "invvpid {0}, [{1}]",
        "setna {2}",
        in(reg) invalidation_type,
        in(reg) &desc,
        out(reg_byte) fail,
        options(nostack)
    );
    if fail == 0 {
        Ok(())
    } else {
        Err(VmxError::VmwriteFailed)
    }
}

// ---------------------------------------------------------------------------
// EPT (Extended Page Tables) 4-Level Guest-Physical Paging Engine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum EptMemoryType {
    Uncacheable = 0,
    WriteCombining = 1,
    WriteThrough = 4,
    WriteProtected = 5,
    WriteBack = 6,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EptFlags(pub u64);

impl EptFlags {
    pub const NONE: Self = Self(0);
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const EXECUTE: Self = Self(1 << 2);
    pub const RW: Self = Self((1 << 0) | (1 << 1));
    pub const RX: Self = Self((1 << 0) | (1 << 2));
    pub const RWX: Self = Self((1 << 0) | (1 << 1) | (1 << 2));
    pub const IGNORE_PAT: Self = Self(1 << 6);
    pub const LARGE_PAGE: Self = Self(1 << 7);
    pub const ACCESSED: Self = Self(1 << 8);
    pub const DIRTY: Self = Self(1 << 9);
    pub const EXECUTE_USER: Self = Self(1 << 10);

    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub const fn is_present(self) -> bool {
        (self.0 & 0x7) != 0
    }
}

#[repr(C, align(4096))]
struct EptPageTable {
    entries: [u64; TABLE_ENTRIES],
}

impl EptPageTable {
    fn zero(&mut self) {
        self.entries.fill(0);
    }
}

#[inline]
fn hhdm_offset() -> u64 {
    stats().hhdm_offset
}

#[inline]
unsafe fn ept_table_from_phys<'a>(phys: u64) -> &'a mut EptPageTable {
    let virt = hhdm_offset().wrapping_add(phys & PHYS_ADDR_MASK);
    &mut *(virt as *mut EptPageTable)
}

#[inline]
fn alloc_zeroed_table() -> VmxResult<u64> {
    let phys = alloc_frame().ok_or(VmxError::OutOfMemory)?;
    unsafe {
        ept_table_from_phys(phys).zero();
    }
    Ok(phys)
}

#[inline]
const fn ept_pml4_idx(gpa: u64) -> usize {
    ((gpa >> 39) & 0x1FF) as usize
}

#[inline]
const fn ept_pdpt_idx(gpa: u64) -> usize {
    ((gpa >> 30) & 0x1FF) as usize
}

#[inline]
const fn ept_pd_idx(gpa: u64) -> usize {
    ((gpa >> 21) & 0x1FF) as usize
}

#[inline]
const fn ept_pt_idx(gpa: u64) -> usize {
    ((gpa >> 12) & 0x1FF) as usize
}

/// 4-level Intel EPT (Extended Page Table) guest-physical address space.
pub struct EptTable {
    pml4_phys: u64,
    allocated_tables: Vec<u64>,
    mapped_count: usize,
    mem_type: EptMemoryType,
    enable_ad: bool,
}

impl EptTable {
    /// Allocate a new 4-level EPT hierarchy rooted at a fresh PML4 frame.
    pub fn new(mem_type: EptMemoryType, enable_ad: bool) -> VmxResult<Self> {
        let pml4_phys = alloc_zeroed_table()?;
        let allocated_tables = alloc::vec![pml4_phys];
        Ok(Self {
            pml4_phys,
            allocated_tables,
            mapped_count: 0,
            mem_type,
            enable_ad,
        })
    }

    /// Construct the 64-bit EPTP (Extended Page Table Pointer) value for VMCS `EPT_POINTER`.
    pub fn eptp(&self) -> u64 {
        let mt = (self.mem_type as u64) & 0x7;
        let walk_len = 3u64 << 3; // 4-level walk => value 3
        let ad_bit = if self.enable_ad { 1u64 << 6 } else { 0 };
        (self.pml4_phys & PHYS_ADDR_MASK) | mt | walk_len | ad_bit
    }

    pub fn pml4_phys(&self) -> u64 {
        self.pml4_phys
    }

    pub fn mapped_pages(&self) -> usize {
        self.mapped_count
    }

    fn ensure_next_table(&mut self, parent_phys: u64, idx: usize) -> VmxResult<u64> {
        let parent = unsafe { ept_table_from_phys(parent_phys) };
        let entry = parent.entries[idx];
        if (entry & 0x7) != 0 {
            if (entry & EptFlags::LARGE_PAGE.bits()) != 0 {
                return Err(VmxError::EptMappingConflict);
            }
            return Ok(entry & PHYS_ADDR_MASK);
        }
        let child_phys = alloc_zeroed_table()?;
        self.allocated_tables.push(child_phys);
        parent.entries[idx] = (child_phys & PHYS_ADDR_MASK) | EptFlags::RWX.bits();
        Ok(child_phys)
    }

    /// Map a 4 KiB guest-physical page `gpa` -> host-physical `hpa`.
    pub fn map_4k(
        &mut self,
        gpa: u64,
        hpa: u64,
        flags: EptFlags,
        mem_type: EptMemoryType,
    ) -> VmxResult<()> {
        if (gpa & 0xFFF) != 0 || (hpa & 0xFFF) != 0 {
            return Err(VmxError::InvalidAddress);
        }
        let pdpt_phys = self.ensure_next_table(self.pml4_phys, ept_pml4_idx(gpa))?;
        let pd_phys = self.ensure_next_table(pdpt_phys, ept_pdpt_idx(gpa))?;
        let pt_phys = self.ensure_next_table(pd_phys, ept_pd_idx(gpa))?;

        let pt = unsafe { ept_table_from_phys(pt_phys) };
        let idx = ept_pt_idx(gpa);
        let was_present = (pt.entries[idx] & 0x7) != 0;
        let mt_bits = ((mem_type as u64) & 0x7) << 3;
        pt.entries[idx] = (hpa & PHYS_ADDR_MASK) | (flags.bits() & 0x7FF) | mt_bits;
        if !was_present && flags.is_present() {
            self.mapped_count += 1;
        }
        Ok(())
    }

    /// Map a 2 MiB large guest-physical page `gpa` -> host-physical `hpa`.
    pub fn map_2m(
        &mut self,
        gpa: u64,
        hpa: u64,
        flags: EptFlags,
        mem_type: EptMemoryType,
    ) -> VmxResult<()> {
        const ALIGN_2M: u64 = 2 * 1024 * 1024;
        if !gpa.is_multiple_of(ALIGN_2M) || !hpa.is_multiple_of(ALIGN_2M) {
            return Err(VmxError::InvalidAddress);
        }
        let pdpt_phys = self.ensure_next_table(self.pml4_phys, ept_pml4_idx(gpa))?;
        let pd_phys = self.ensure_next_table(pdpt_phys, ept_pdpt_idx(gpa))?;

        let pd = unsafe { ept_table_from_phys(pd_phys) };
        let idx = ept_pd_idx(gpa);
        let existing = pd.entries[idx];
        if (existing & 0x7) != 0 && (existing & EptFlags::LARGE_PAGE.bits()) == 0 {
            return Err(VmxError::EptMappingConflict);
        }
        let was_present = (existing & 0x7) != 0;
        let mt_bits = ((mem_type as u64) & 0x7) << 3;
        pd.entries[idx] = (hpa & 0x000F_FFFF_FFE0_0000)
            | (flags.bits() & 0x7FF)
            | EptFlags::LARGE_PAGE.bits()
            | mt_bits;
        if !was_present && flags.is_present() {
            self.mapped_count += 512;
        }
        Ok(())
    }

    /// Map a 1 GiB huge guest-physical page `gpa` -> host-physical `hpa`.
    pub fn map_1g(
        &mut self,
        gpa: u64,
        hpa: u64,
        flags: EptFlags,
        mem_type: EptMemoryType,
    ) -> VmxResult<()> {
        const ALIGN_1G: u64 = 1024 * 1024 * 1024;
        if !gpa.is_multiple_of(ALIGN_1G) || !hpa.is_multiple_of(ALIGN_1G) {
            return Err(VmxError::InvalidAddress);
        }
        let pdpt_phys = self.ensure_next_table(self.pml4_phys, ept_pml4_idx(gpa))?;
        let pdpt = unsafe { ept_table_from_phys(pdpt_phys) };
        let idx = ept_pdpt_idx(gpa);
        let existing = pdpt.entries[idx];
        if (existing & 0x7) != 0 && (existing & EptFlags::LARGE_PAGE.bits()) == 0 {
            return Err(VmxError::EptMappingConflict);
        }
        let was_present = (existing & 0x7) != 0;
        let mt_bits = ((mem_type as u64) & 0x7) << 3;
        pdpt.entries[idx] = (hpa & 0x000F_FFFF_C000_0000)
            | (flags.bits() & 0x7FF)
            | EptFlags::LARGE_PAGE.bits()
            | mt_bits;
        if !was_present && flags.is_present() {
            self.mapped_count += 512 * 512;
        }
        Ok(())
    }

    /// Map a contiguous guest-physical range using 1 GiB / 2 MiB / 4 KiB pages where aligned.
    pub fn map_range(
        &mut self,
        gpa_start: u64,
        hpa_start: u64,
        size: u64,
        flags: EptFlags,
        mem_type: EptMemoryType,
    ) -> VmxResult<()> {
        const SIZE_1G: u64 = 1024 * 1024 * 1024;
        const SIZE_2M: u64 = 2 * 1024 * 1024;
        const SIZE_4K: u64 = PAGE_SIZE as u64;

        if !gpa_start.is_multiple_of(SIZE_4K)
            || !hpa_start.is_multiple_of(SIZE_4K)
            || !size.is_multiple_of(SIZE_4K)
        {
            return Err(VmxError::InvalidAddress);
        }

        let mut offset = 0u64;
        while offset < size {
            let gpa = gpa_start + offset;
            let hpa = hpa_start + offset;
            let rem = size - offset;
            if rem >= SIZE_1G && gpa.is_multiple_of(SIZE_1G) && hpa.is_multiple_of(SIZE_1G) {
                self.map_1g(gpa, hpa, flags, mem_type)?;
                offset += SIZE_1G;
            } else if rem >= SIZE_2M && gpa.is_multiple_of(SIZE_2M) && hpa.is_multiple_of(SIZE_2M) {
                self.map_2m(gpa, hpa, flags, mem_type)?;
                offset += SIZE_2M;
            } else {
                self.map_4k(gpa, hpa, flags, mem_type)?;
                offset += SIZE_4K;
            }
        }
        Ok(())
    }

    /// Translate a guest-physical address to `(host_phys, flags, page_size_bytes)`.
    pub fn translate(&self, gpa: u64) -> Option<(u64, EptFlags, usize)> {
        let pml4 = unsafe { ept_table_from_phys(self.pml4_phys) };
        let pml4e = pml4.entries[ept_pml4_idx(gpa)];
        if (pml4e & 0x7) == 0 {
            return None;
        }

        let pdpt = unsafe { ept_table_from_phys(pml4e & PHYS_ADDR_MASK) };
        let pdpte = pdpt.entries[ept_pdpt_idx(gpa)];
        if (pdpte & 0x7) == 0 {
            return None;
        }
        if (pdpte & EptFlags::LARGE_PAGE.bits()) != 0 {
            let base = pdpte & 0x000F_FFFF_C000_0000;
            let off = gpa & 0x3FFF_FFFF;
            return Some((base | off, EptFlags(pdpte & 0x7FF), 1024 * 1024 * 1024));
        }

        let pd = unsafe { ept_table_from_phys(pdpte & PHYS_ADDR_MASK) };
        let pde = pd.entries[ept_pd_idx(gpa)];
        if (pde & 0x7) == 0 {
            return None;
        }
        if (pde & EptFlags::LARGE_PAGE.bits()) != 0 {
            let base = pde & 0x000F_FFFF_FFE0_0000;
            let off = gpa & 0x1F_FFFF;
            return Some((base | off, EptFlags(pde & 0x7FF), 2 * 1024 * 1024));
        }

        let pt = unsafe { ept_table_from_phys(pde & PHYS_ADDR_MASK) };
        let pte = pt.entries[ept_pt_idx(gpa)];
        if (pte & 0x7) == 0 {
            return None;
        }
        let base = pte & PHYS_ADDR_MASK;
        let off = gpa & 0xFFF;
        Some((base | off, EptFlags(pte & 0x7FF), PAGE_SIZE))
    }

    /// Update R/W/X permissions on an existing 4 KiB EPT leaf entry.
    pub fn update_permissions(&mut self, gpa: u64, flags: EptFlags) -> VmxResult<()> {
        let pml4 = unsafe { ept_table_from_phys(self.pml4_phys) };
        let pml4e = pml4.entries[ept_pml4_idx(gpa)];
        if (pml4e & 0x7) == 0 {
            return Err(VmxError::InvalidAddress);
        }
        let pdpt = unsafe { ept_table_from_phys(pml4e & PHYS_ADDR_MASK) };
        let pdpte = pdpt.entries[ept_pdpt_idx(gpa)];
        if (pdpte & 0x7) == 0 || (pdpte & EptFlags::LARGE_PAGE.bits()) != 0 {
            return Err(VmxError::InvalidAddress);
        }
        let pd = unsafe { ept_table_from_phys(pdpte & PHYS_ADDR_MASK) };
        let pde = pd.entries[ept_pd_idx(gpa)];
        if (pde & 0x7) == 0 || (pde & EptFlags::LARGE_PAGE.bits()) != 0 {
            return Err(VmxError::InvalidAddress);
        }
        let pt = unsafe { ept_table_from_phys(pde & PHYS_ADDR_MASK) };
        let idx = ept_pt_idx(gpa);
        let pte = pt.entries[idx];
        if (pte & 0x7) == 0 {
            return Err(VmxError::InvalidAddress);
        }
        pt.entries[idx] = (pte & !0x7) | (flags.bits() & 0x7);
        Ok(())
    }

    /// Unmap a 4 KiB guest-physical page and return its previous host-physical address.
    pub fn unmap_4k(&mut self, gpa: u64) -> VmxResult<u64> {
        let pml4 = unsafe { ept_table_from_phys(self.pml4_phys) };
        let pml4e = pml4.entries[ept_pml4_idx(gpa)];
        if (pml4e & 0x7) == 0 {
            return Err(VmxError::InvalidAddress);
        }
        let pdpt = unsafe { ept_table_from_phys(pml4e & PHYS_ADDR_MASK) };
        let pdpte = pdpt.entries[ept_pdpt_idx(gpa)];
        if (pdpte & 0x7) == 0 || (pdpte & EptFlags::LARGE_PAGE.bits()) != 0 {
            return Err(VmxError::InvalidAddress);
        }
        let pd = unsafe { ept_table_from_phys(pdpte & PHYS_ADDR_MASK) };
        let pde = pd.entries[ept_pd_idx(gpa)];
        if (pde & 0x7) == 0 || (pde & EptFlags::LARGE_PAGE.bits()) != 0 {
            return Err(VmxError::InvalidAddress);
        }
        let pt = unsafe { ept_table_from_phys(pde & PHYS_ADDR_MASK) };
        let idx = ept_pt_idx(gpa);
        let pte = pt.entries[idx];
        if (pte & 0x7) == 0 {
            return Err(VmxError::InvalidAddress);
        }
        pt.entries[idx] = 0;
        self.mapped_count = self.mapped_count.saturating_sub(1);
        Ok(pte & PHYS_ADDR_MASK)
    }
}

impl Drop for EptTable {
    fn drop(&mut self) {
        for &phys in &self.allocated_tables {
            dealloc_frame(phys);
        }
        self.allocated_tables.clear();
    }
}

// ---------------------------------------------------------------------------
// AMD-V NPT (Nested Page Tables) & VMCB Layout
// ---------------------------------------------------------------------------

/// AMD-V Nested Page Table (uses standard x86_64 4-level page table format for `nCR3`).
pub struct NptTable {
    pml4_phys: u64,
    allocated_tables: Vec<u64>,
}

impl NptTable {
    pub fn new() -> VmxResult<Self> {
        let pml4_phys = alloc_zeroed_table()?;
        let allocated_tables = alloc::vec![pml4_phys];
        Ok(Self {
            pml4_phys,
            allocated_tables,
        })
    }

    pub fn ncr3(&self) -> u64 {
        self.pml4_phys & PHYS_ADDR_MASK
    }

    fn ensure_next(&mut self, parent_phys: u64, idx: usize) -> VmxResult<u64> {
        let parent = unsafe { ept_table_from_phys(parent_phys) };
        let entry = parent.entries[idx];
        if (entry & 1) != 0 {
            return Ok(entry & PHYS_ADDR_MASK);
        }
        let child = alloc_zeroed_table()?;
        self.allocated_tables.push(child);
        // Present | Writable | User (NPT requires User bit for guest accesses)
        parent.entries[idx] = (child & PHYS_ADDR_MASK) | 0x7;
        Ok(child)
    }

    pub fn map_4k(
        &mut self,
        gpa: u64,
        hpa: u64,
        writable: bool,
        executable: bool,
    ) -> VmxResult<()> {
        if (gpa & 0xFFF) != 0 || (hpa & 0xFFF) != 0 {
            return Err(VmxError::InvalidAddress);
        }
        let pdpt = self.ensure_next(self.pml4_phys, ept_pml4_idx(gpa))?;
        let pd = self.ensure_next(pdpt, ept_pdpt_idx(gpa))?;
        let pt = self.ensure_next(pd, ept_pd_idx(gpa))?;
        let pt_tbl = unsafe { ept_table_from_phys(pt) };
        let mut flags = 0x5u64; // Present | User
        if writable {
            flags |= 1 << 1;
        }
        if !executable {
            flags |= 1 << 63; // NX
        }
        pt_tbl.entries[ept_pt_idx(gpa)] = (hpa & PHYS_ADDR_MASK) | flags;
        Ok(())
    }

    pub fn translate(&self, gpa: u64) -> Option<u64> {
        let pml4 = unsafe { ept_table_from_phys(self.pml4_phys) };
        let e4 = pml4.entries[ept_pml4_idx(gpa)];
        if (e4 & 1) == 0 {
            return None;
        }
        let pdpt = unsafe { ept_table_from_phys(e4 & PHYS_ADDR_MASK) };
        let e3 = pdpt.entries[ept_pdpt_idx(gpa)];
        if (e3 & 1) == 0 {
            return None;
        }
        let pd = unsafe { ept_table_from_phys(e3 & PHYS_ADDR_MASK) };
        let e2 = pd.entries[ept_pd_idx(gpa)];
        if (e2 & 1) == 0 {
            return None;
        }
        let pt = unsafe { ept_table_from_phys(e2 & PHYS_ADDR_MASK) };
        let e1 = pt.entries[ept_pt_idx(gpa)];
        if (e1 & 1) == 0 {
            return None;
        }
        Some((e1 & PHYS_ADDR_MASK) | (gpa & 0xFFF))
    }
}

impl Drop for NptTable {
    fn drop(&mut self) {
        for &phys in &self.allocated_tables {
            dealloc_frame(phys);
        }
        self.allocated_tables.clear();
    }
}

/// AMD-V VMCB Control Area (1024 bytes, offset 0x000..0x400).
#[repr(C)]
pub struct VmcbControlArea {
    pub intercept_cr_read: u16,
    pub intercept_cr_write: u16,
    pub intercept_dr_read: u16,
    pub intercept_dr_write: u16,
    pub intercept_exceptions: u32,
    pub intercept_misc1: u32,
    pub intercept_misc2: u32,
    pub intercept_misc3: u32,
    pub reserved1: [u8; 36],
    pub pause_filter_threshold: u16,
    pub pause_filter_count: u16,
    pub iopm_base_pa: u64,
    pub msrpm_base_pa: u64,
    pub tsc_offset: u64,
    pub guest_asid: u32,
    pub tlb_control: u32,
    pub v_tpr: u64,
    pub interrupt_shadow: u64,
    pub exit_code: u64,
    pub exit_info1: u64,
    pub exit_info2: u64,
    pub exit_int_info: u64,
    pub np_enable: u64,
    pub avic_apic_bar: u64,
    pub ghcb_pa: u64,
    pub event_inj: u64,
    pub n_cr3: u64,
    pub lbr_virt_enable: u64,
    pub vmcb_clean: u32,
    pub reserved2: u32,
    pub next_rip: u64,
    pub insn_len: u8,
    pub insn_bytes: [u8; 15],
    pub avic_backing_page: u64,
    pub reserved3: u64,
    pub avic_logical_table: u64,
    pub avic_physical_table: u64,
    pub reserved4: u64,
    pub vmsa_pa: u64,
    pub reserved5: [u8; 752],
}

/// AMD-V VMCB Segment Register descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VmcbSegment {
    pub selector: u16,
    pub attrib: u16,
    pub limit: u32,
    pub base: u64,
}

/// AMD-V VMCB State Save Area (offset 0x400..0x1000).
#[repr(C)]
pub struct VmcbStateSaveArea {
    pub es: VmcbSegment,
    pub cs: VmcbSegment,
    pub ss: VmcbSegment,
    pub ds: VmcbSegment,
    pub fs: VmcbSegment,
    pub gs: VmcbSegment,
    pub gdtr: VmcbSegment,
    pub ldtr: VmcbSegment,
    pub idtr: VmcbSegment,
    pub tr: VmcbSegment,
    pub reserved1: [u8; 43],
    pub cpl: u8,
    pub reserved2: u32,
    pub efer: u64,
    pub reserved3: [u8; 112],
    pub cr4: u64,
    pub cr3: u64,
    pub cr0: u64,
    pub dr7: u64,
    pub dr6: u64,
    pub rflags: u64,
    pub rip: u64,
    pub reserved4: [u8; 88],
    pub rsp: u64,
    pub reserved5: [u8; 24],
    pub rax: u64,
    pub star: u64,
    pub lstar: u64,
    pub cstar: u64,
    pub sfmask: u64,
    pub kernel_gs_base: u64,
    pub sysenter_cs: u64,
    pub sysenter_esp: u64,
    pub sysenter_eip: u64,
    pub cr2: u64,
    pub reserved6: [u8; 32],
    pub g_pat: u64,
    pub dbgctl: u64,
    pub br_from: u64,
    pub br_to: u64,
    pub last_excp_from: u64,
    pub last_excp_to: u64,
    pub reserved7: [u8; 2408],
}

/// 4 KiB-aligned AMD-V Virtual Machine Control Block.
#[repr(C, align(4096))]
pub struct Vmcb {
    pub control: VmcbControlArea,
    pub save: VmcbStateSaveArea,
}

const _: () = assert!(core::mem::size_of::<VmcbControlArea>() == 1024);
const _: () = assert!(core::mem::size_of::<Vmcb>() == 4096);

// ---------------------------------------------------------------------------
// VM-Exit Reasons, Qualifiers & Dispatcher
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmxExitReason {
    ExceptionOrNmi,
    ExternalInterrupt,
    TripleFault,
    InitSignal,
    Sipi,
    InterruptWindow,
    NmiWindow,
    TaskSwitch,
    Cpuid,
    Getsec,
    Hlt,
    Invd,
    Invlpg,
    Rdpmc,
    Rdtsc,
    Rsm,
    Vmcall,
    Vmclear,
    Vmlaunch,
    Vmptrld,
    Vmptrst,
    Vmread,
    Vmresume,
    Vmwrite,
    Vmxoff,
    Vmxon,
    CrAccess,
    MovDr,
    IoInstruction,
    MsrRead,
    MsrWrite,
    InvalidGuestState,
    MsrLoading,
    Mwait,
    MonitorTrapFlag,
    Monitor,
    Pause,
    MachineCheck,
    TprBelowThreshold,
    ApicAccess,
    VirtualizedEoi,
    GdtrOrIdtrAccess,
    LdtrOrTrAccess,
    EptViolation,
    EptMisconfiguration,
    Invept,
    Rdtscp,
    VmxPreemptionTimer,
    Invvpid,
    Wbinvd,
    Xsetbv,
    ApicWrite,
    Rdrand,
    Invpcid,
    Vmfunc,
    Unknown(u32),
}

impl VmxExitReason {
    pub const fn from_raw(raw: u32) -> Self {
        match raw & 0xFFFF {
            0 => Self::ExceptionOrNmi,
            1 => Self::ExternalInterrupt,
            2 => Self::TripleFault,
            3 => Self::InitSignal,
            4 => Self::Sipi,
            7 => Self::InterruptWindow,
            8 => Self::NmiWindow,
            9 => Self::TaskSwitch,
            10 => Self::Cpuid,
            11 => Self::Getsec,
            12 => Self::Hlt,
            13 => Self::Invd,
            14 => Self::Invlpg,
            15 => Self::Rdpmc,
            16 => Self::Rdtsc,
            17 => Self::Rsm,
            18 => Self::Vmcall,
            19 => Self::Vmclear,
            20 => Self::Vmlaunch,
            21 => Self::Vmptrld,
            22 => Self::Vmptrst,
            23 => Self::Vmread,
            24 => Self::Vmresume,
            25 => Self::Vmwrite,
            26 => Self::Vmxoff,
            27 => Self::Vmxon,
            28 => Self::CrAccess,
            29 => Self::MovDr,
            30 => Self::IoInstruction,
            31 => Self::MsrRead,
            32 => Self::MsrWrite,
            33 => Self::InvalidGuestState,
            34 => Self::MsrLoading,
            36 => Self::Mwait,
            37 => Self::MonitorTrapFlag,
            39 => Self::Monitor,
            40 => Self::Pause,
            41 => Self::MachineCheck,
            43 => Self::TprBelowThreshold,
            44 => Self::ApicAccess,
            45 => Self::VirtualizedEoi,
            46 => Self::GdtrOrIdtrAccess,
            47 => Self::LdtrOrTrAccess,
            48 => Self::EptViolation,
            49 => Self::EptMisconfiguration,
            50 => Self::Invept,
            51 => Self::Rdtscp,
            52 => Self::VmxPreemptionTimer,
            53 => Self::Invvpid,
            54 => Self::Wbinvd,
            55 => Self::Xsetbv,
            56 => Self::ApicWrite,
            57 => Self::Rdrand,
            58 => Self::Invpcid,
            59 => Self::Vmfunc,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EptViolationInfo {
    pub read_access: bool,
    pub write_access: bool,
    pub exec_access: bool,
    pub ept_readable: bool,
    pub ept_writable: bool,
    pub ept_executable: bool,
    pub linear_addr_valid: bool,
}

impl EptViolationInfo {
    pub const fn from_qualification(qual: u64) -> Self {
        Self {
            read_access: (qual & (1 << 0)) != 0,
            write_access: (qual & (1 << 1)) != 0,
            exec_access: (qual & (1 << 2)) != 0,
            ept_readable: (qual & (1 << 3)) != 0,
            ept_writable: (qual & (1 << 4)) != 0,
            ept_executable: (qual & (1 << 5)) != 0,
            linear_addr_valid: (qual & (1 << 7)) != 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoExitInfo {
    pub size_bytes: u8,
    pub is_in: bool,
    pub is_string: bool,
    pub is_rep: bool,
    pub port: u16,
}

impl IoExitInfo {
    pub const fn from_qualification(qual: u64) -> Self {
        let size_bytes = ((qual & 0x7) as u8) + 1;
        Self {
            size_bytes,
            is_in: (qual & (1 << 3)) != 0,
            is_string: (qual & (1 << 4)) != 0,
            is_rep: (qual & (1 << 5)) != 0,
            port: ((qual >> 16) & 0xFFFF) as u16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum HypercallId {
    Ping = 0,
    GetVersion = 1,
    ConsoleWrite = 2,
    Yield = 3,
    Shutdown = 4,
    MapMemory = 5,
}

impl HypercallId {
    pub const fn from_u64(val: u64) -> Option<Self> {
        match val {
            0 => Some(Self::Ping),
            1 => Some(Self::GetVersion),
            2 => Some(Self::ConsoleWrite),
            3 => Some(Self::Yield),
            4 => Some(Self::Shutdown),
            5 => Some(Self::MapMemory),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmExitAction {
    Resume,
    AdvanceRipAndResume(u32),
    Halt,
    HypercallHandled(u64),
    MmioFault { gpa: u64, is_write: bool, size: u8 },
    Shutdown,
    Unhandled(VmxExitReason),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GuestRegs {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub rsp: u64,
    pub rflags: u64,
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
}

/// Virtual CPU state and VM-exit dispatcher.
pub struct Vcpu {
    pub id: u32,
    pub vpid: u16,
    pub asid: u32,
    pub backend: VirtBackend,
    pub vmcs_phys: u64,
    pub vmcb_phys: u64,
    pub msr_bitmap_phys: u64,
    pub io_bitmap_a_phys: u64,
    pub io_bitmap_b_phys: u64,
    pub regs: GuestRegs,
    pub launched: bool,
    pub halted: bool,
    pub exit_count: u64,
    pub tsc_offset: u64,
    pub apic_base: u64,
    pub efer: u64,
}

impl Vcpu {
    /// Create a lightweight vCPU descriptor without allocating hardware VMCS/VMCB frames yet.
    pub const fn new_unbacked(id: u32, backend: VirtBackend) -> Self {
        Self {
            id,
            vpid: (id as u16).saturating_add(1),
            asid: id.saturating_add(1),
            backend,
            vmcs_phys: 0,
            vmcb_phys: 0,
            msr_bitmap_phys: 0,
            io_bitmap_a_phys: 0,
            io_bitmap_b_phys: 0,
            regs: GuestRegs {
                rax: 0,
                rbx: 0,
                rcx: 0,
                rdx: 0,
                rsi: 0,
                rdi: 0,
                rbp: 0,
                r8: 0,
                r9: 0,
                r10: 0,
                r11: 0,
                r12: 0,
                r13: 0,
                r14: 0,
                r15: 0,
                rip: 0,
                rsp: 0,
                rflags: 0x2,
                cr0: 0x8000_0031,
                cr2: 0,
                cr3: 0,
                cr4: 0x20,
            },
            launched: false,
            halted: false,
            exit_count: 0,
            tsc_offset: 0,
            apic_base: 0xFEE0_0900,
            efer: 0x500,
        }
    }

    /// Allocate 4 KiB-aligned VMCS/VMCB and bitmap frames for this vCPU.
    pub fn allocate_regions(&mut self, revision_id: u32) -> VmxResult<()> {
        let region_phys = alloc_zeroed_table()?;
        let msr_phys = alloc_zeroed_table()?;
        let io_a_phys = alloc_zeroed_table()?;
        let io_b_phys = alloc_zeroed_table()?;

        unsafe {
            let virt = hhdm_offset().wrapping_add(region_phys & PHYS_ADDR_MASK) as *mut u32;
            *virt = revision_id & 0x7FFF_FFFF;
        }

        match self.backend {
            VirtBackend::IntelVmx => self.vmcs_phys = region_phys,
            VirtBackend::AmdSvm => self.vmcb_phys = region_phys,
            VirtBackend::None => self.vmcs_phys = region_phys,
        }
        self.msr_bitmap_phys = msr_phys;
        self.io_bitmap_a_phys = io_a_phys;
        self.io_bitmap_b_phys = io_b_phys;
        Ok(())
    }

    /// Dispatch a VM-exit and update guest register/control state accordingly.
    pub fn dispatch_vm_exit(
        &mut self,
        reason: VmxExitReason,
        qualification: u64,
        guest_phys: u64,
        instr_len: u32,
        ept: Option<&EptTable>,
    ) -> VmExitAction {
        self.exit_count += 1;

        match reason {
            VmxExitReason::Cpuid => {
                let leaf = self.regs.rax as u32;
                let subleaf = self.regs.rcx as u32;
                if leaf == 0x4000_0000 {
                    // Hypervisor CPUID leaf: "NexusHV\0\0\0\0\0"
                    self.regs.rax = 0x4000_0001;
                    self.regs.rbx = u32::from_le_bytes(*b"Nexu") as u64;
                    self.regs.rcx = u32::from_le_bytes(*b"sHV\0") as u64;
                    self.regs.rdx = 0;
                } else if leaf == 0x4000_0001 {
                    self.regs.rax = 0x0001_0000; // NexusHV v1.0
                    self.regs.rbx = 0;
                    self.regs.rcx = 0;
                    self.regs.rdx = 0;
                } else {
                    let res = __cpuid_count(leaf, subleaf);
                    let mut ecx = res.ecx;
                    if leaf == 1 {
                        // Hide VMX (bit 5), advertise hypervisor present (bit 31)
                        ecx &= !(1 << 5);
                        ecx |= 1 << 31;
                    }
                    self.regs.rax = res.eax as u64;
                    self.regs.rbx = res.ebx as u64;
                    self.regs.rcx = ecx as u64;
                    self.regs.rdx = res.edx as u64;
                }
                self.regs.rip = self.regs.rip.wrapping_add(instr_len as u64);
                VmExitAction::AdvanceRipAndResume(instr_len)
            }

            VmxExitReason::Vmcall => {
                let ret = match HypercallId::from_u64(self.regs.rax) {
                    Some(HypercallId::Ping) => 0x4E45_5855, // "NEXU"
                    Some(HypercallId::GetVersion) => 0x0001_0000,
                    Some(HypercallId::ConsoleWrite) => self.regs.rsi,
                    Some(HypercallId::Yield) => 0,
                    Some(HypercallId::Shutdown) => {
                        self.halted = true;
                        return VmExitAction::Shutdown;
                    }
                    Some(HypercallId::MapMemory) => 0,
                    None => u64::MAX,
                };
                self.regs.rax = ret;
                self.regs.rip = self.regs.rip.wrapping_add(instr_len as u64);
                VmExitAction::HypercallHandled(ret)
            }

            VmxExitReason::Hlt => {
                self.halted = true;
                self.regs.rip = self.regs.rip.wrapping_add(instr_len as u64);
                VmExitAction::Halt
            }

            VmxExitReason::IoInstruction => {
                let io = IoExitInfo::from_qualification(qualification);
                if io.is_in && (0x3F8..=0x3FF).contains(&io.port) {
                    // Emulate COM1 LSR ready (0x60) on port 0x3FD
                    self.regs.rax = if io.port == 0x3FD { 0x60 } else { 0xFF };
                }
                self.regs.rip = self.regs.rip.wrapping_add(instr_len as u64);
                VmExitAction::AdvanceRipAndResume(instr_len)
            }

            VmxExitReason::MsrRead => {
                let msr = self.regs.rcx as u32;
                let val = match msr {
                    MSR_IA32_APIC_BASE => self.apic_base,
                    MSR_IA32_EFER => self.efer,
                    MSR_IA32_TSC => self.tsc_offset,
                    _ => 0,
                };
                self.regs.rax = val & 0xFFFF_FFFF;
                self.regs.rdx = val >> 32;
                self.regs.rip = self.regs.rip.wrapping_add(instr_len as u64);
                VmExitAction::AdvanceRipAndResume(instr_len)
            }

            VmxExitReason::MsrWrite => {
                let msr = self.regs.rcx as u32;
                let val = (self.regs.rax & 0xFFFF_FFFF) | ((self.regs.rdx & 0xFFFF_FFFF) << 32);
                match msr {
                    MSR_IA32_APIC_BASE => self.apic_base = val,
                    MSR_IA32_EFER => self.efer = val,
                    MSR_IA32_TSC => self.tsc_offset = val,
                    _ => {}
                }
                self.regs.rip = self.regs.rip.wrapping_add(instr_len as u64);
                VmExitAction::AdvanceRipAndResume(instr_len)
            }

            VmxExitReason::CrAccess => {
                let cr_num = (qualification & 0xF) as u8;
                let access_type = ((qualification >> 4) & 0x3) as u8;
                let gpr = ((qualification >> 8) & 0xF) as u8;
                let gpr_val = self.read_gpr(gpr);
                if access_type == 0 {
                    // MOV to CR
                    match cr_num {
                        0 => self.regs.cr0 = gpr_val,
                        3 => self.regs.cr3 = gpr_val,
                        4 => self.regs.cr4 = gpr_val,
                        _ => {}
                    }
                } else if access_type == 1 {
                    // MOV from CR
                    let cr_val = match cr_num {
                        0 => self.regs.cr0,
                        3 => self.regs.cr3,
                        4 => self.regs.cr4,
                        _ => 0,
                    };
                    self.write_gpr(gpr, cr_val);
                }
                self.regs.rip = self.regs.rip.wrapping_add(instr_len as u64);
                VmExitAction::AdvanceRipAndResume(instr_len)
            }

            VmxExitReason::EptViolation => {
                let info = EptViolationInfo::from_qualification(qualification);
                let mapped = ept.and_then(|t| t.translate(guest_phys)).is_some();
                if !mapped {
                    VmExitAction::MmioFault {
                        gpa: guest_phys,
                        is_write: info.write_access,
                        size: 4,
                    }
                } else {
                    VmExitAction::Resume
                }
            }

            VmxExitReason::ExternalInterrupt | VmxExitReason::InterruptWindow => {
                VmExitAction::Resume
            }

            VmxExitReason::TripleFault => {
                self.halted = true;
                VmExitAction::Shutdown
            }

            other => VmExitAction::Unhandled(other),
        }
    }

    fn read_gpr(&self, idx: u8) -> u64 {
        match idx {
            0 => self.regs.rax,
            1 => self.regs.rcx,
            2 => self.regs.rdx,
            3 => self.regs.rbx,
            4 => self.regs.rsp,
            5 => self.regs.rbp,
            6 => self.regs.rsi,
            7 => self.regs.rdi,
            8 => self.regs.r8,
            9 => self.regs.r9,
            10 => self.regs.r10,
            11 => self.regs.r11,
            12 => self.regs.r12,
            13 => self.regs.r13,
            14 => self.regs.r14,
            15 => self.regs.r15,
            _ => 0,
        }
    }

    fn write_gpr(&mut self, idx: u8, val: u64) {
        match idx {
            0 => self.regs.rax = val,
            1 => self.regs.rcx = val,
            2 => self.regs.rdx = val,
            3 => self.regs.rbx = val,
            4 => self.regs.rsp = val,
            5 => self.regs.rbp = val,
            6 => self.regs.rsi = val,
            7 => self.regs.rdi = val,
            8 => self.regs.r8 = val,
            9 => self.regs.r9 = val,
            10 => self.regs.r10 = val,
            11 => self.regs.r11 = val,
            12 => self.regs.r12 = val,
            13 => self.regs.r13 = val,
            14 => self.regs.r14 = val,
            15 => self.regs.r15 = val,
            _ => {}
        }
    }
}

impl Drop for Vcpu {
    fn drop(&mut self) {
        if self.vmcs_phys != 0 {
            dealloc_frame(self.vmcs_phys);
        }
        if self.vmcb_phys != 0 {
            dealloc_frame(self.vmcb_phys);
        }
        if self.msr_bitmap_phys != 0 {
            dealloc_frame(self.msr_bitmap_phys);
        }
        if self.io_bitmap_a_phys != 0 {
            dealloc_frame(self.io_bitmap_a_phys);
        }
        if self.io_bitmap_b_phys != 0 {
            dealloc_frame(self.io_bitmap_b_phys);
        }
    }
}

// ---------------------------------------------------------------------------
// Global Hypervisor State & Initialization
// ---------------------------------------------------------------------------

struct HypervisorState {
    initialized: bool,
    vmxon_phys: u64,
    caps: VirtCapabilities,
}

impl HypervisorState {
    const fn new() -> Self {
        Self {
            initialized: false,
            vmxon_phys: 0,
            caps: VirtCapabilities::empty(),
        }
    }
}

static HYPERVISOR: SpinLock<HypervisorState> = SpinLock::new(HypervisorState::new());
static VMX_ACTIVE: AtomicBool = AtomicBool::new(false);
static TOTAL_VM_EXITS: AtomicU64 = AtomicU64::new(0);

/// Enable hardware virtualization (VMXON on Intel or EFER.SVME on AMD) if supported and unlocked.
pub fn init() -> VmxResult<VirtCapabilities> {
    let mut hv = HYPERVISOR.lock();
    if hv.initialized {
        return Ok(hv.caps);
    }

    let caps = detect_capabilities();
    match caps.backend {
        VirtBackend::None => return Err(VmxError::Unsupported),
        VirtBackend::IntelVmx => {
            if caps.bios_locked && !caps.bios_enabled {
                return Err(VmxError::DisabledInBios);
            }
            if !caps.bios_locked {
                unsafe {
                    let cur = rdmsr(MSR_IA32_FEATURE_CONTROL);
                    wrmsr(
                        MSR_IA32_FEATURE_CONTROL,
                        cur | FEATURE_CONTROL_LOCK | FEATURE_CONTROL_VMXON_OUTSIDE_SMX,
                    );
                }
            }

            let cr0_fixed0 = unsafe { rdmsr(MSR_IA32_VMX_CR0_FIXED0) };
            let cr0_fixed1 = unsafe { rdmsr(MSR_IA32_VMX_CR0_FIXED1) };
            let cr4_fixed0 = unsafe { rdmsr(MSR_IA32_VMX_CR4_FIXED0) };
            let cr4_fixed1 = unsafe { rdmsr(MSR_IA32_VMX_CR4_FIXED1) };

            let new_cr0 = ((read_cr0() | CR0_NE) | cr0_fixed0) & cr0_fixed1;
            let new_cr4 = ((read_cr4() | CR4_VMXE) | cr4_fixed0) & cr4_fixed1;
            unsafe {
                write_cr0(new_cr0);
                write_cr4(new_cr4);
            }

            let vmxon_phys = alloc_zeroed_table()?;
            unsafe {
                let virt = hhdm_offset().wrapping_add(vmxon_phys & PHYS_ADDR_MASK) as *mut u32;
                *virt = caps.vmcs_revision_id & 0x7FFF_FFFF;
                if let Err(e) = vmxon(vmxon_phys) {
                    dealloc_frame(vmxon_phys);
                    return Err(e);
                }
            }
            hv.vmxon_phys = vmxon_phys;
        }
        VirtBackend::AmdSvm => {
            if caps.bios_locked {
                return Err(VmxError::DisabledInBios);
            }
            unsafe {
                let efer = rdmsr(MSR_IA32_EFER);
                wrmsr(MSR_IA32_EFER, efer | EFER_SVME);
            }
        }
    }

    hv.caps = caps;
    hv.initialized = true;
    VMX_ACTIVE.store(true, Ordering::Release);
    Ok(caps)
}

pub fn is_active() -> bool {
    VMX_ACTIVE.load(Ordering::Acquire)
}

pub fn total_vm_exits() -> u64 {
    TOTAL_VM_EXITS.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Self-Test
// ---------------------------------------------------------------------------

/// Run non-destructive self-tests for VMCS encoding, EPT 4-level paging, and VM-exit dispatch.
pub fn self_test() -> bool {
    let caps = detect_capabilities();
    serial_println!(
        "VMX self_test: backend={:?} vmx={} svm={} ept={} npt={}",
        caps.backend,
        caps.vmx_supported,
        caps.svm_supported,
        caps.ept_supported,
        caps.npt_supported
    );

    // 1. Verify VMCS field encoding & decoding round-trip
    let encoded = vmcs_field::EPT_POINTER;
    let decoded = VmcsField::decode(encoded);
    if decoded != Some((VmcsFieldWidth::Bits64, VmcsFieldType::Control, 13, false)) {
        serial_println!("VMX self_test FAILED: VMCS field decode mismatch");
        return false;
    }
    if vmcs_field::GUEST_RIP != 0x681E || vmcs_field::EXIT_REASON != 0x4402 {
        serial_println!(
            "VMX self_test FAILED: VMCS constants mismatch (GUEST_RIP={:#x}, EXIT_REASON={:#x})",
            vmcs_field::GUEST_RIP,
            vmcs_field::EXIT_REASON
        );
        return false;
    }

    // 2. Verify VMX control bit adjustment
    let msr_sim: u64 = (0x0000_FFFFu64 << 32) | 0x0000_0005u64;
    let adjusted = adjust_vmx_controls(0x0001_0002, msr_sim);
    if adjusted != 0x0000_0007 {
        serial_println!("VMX self_test FAILED: adjust_vmx_controls={:#x}", adjusted);
        return false;
    }

    // 3. Verify EPT 4-level page table mapping, translation, permission update, and unmap
    if stats().total_frames > 0 {
        let mut ept = match EptTable::new(EptMemoryType::WriteBack, true) {
            Ok(t) => t,
            Err(e) => {
                serial_println!("VMX self_test FAILED: EptTable::new {:?}", e);
                return false;
            }
        };

        if ept
            .map_4k(0x1000, 0x20_0000, EptFlags::RWX, EptMemoryType::WriteBack)
            .is_err()
        {
            serial_println!("VMX self_test FAILED: EPT map_4k");
            return false;
        }
        if ept
            .map_2m(0x40_0000, 0x80_0000, EptFlags::RW, EptMemoryType::WriteBack)
            .is_err()
        {
            serial_println!("VMX self_test FAILED: EPT map_2m");
            return false;
        }
        if ept
            .map_1g(
                0x4000_0000,
                0x8000_0000,
                EptFlags::RX,
                EptMemoryType::WriteBack,
            )
            .is_err()
        {
            serial_println!("VMX self_test FAILED: EPT map_1g");
            return false;
        }

        match ept.translate(0x1234) {
            Some((hpa, flags, sz))
                if hpa == 0x20_0234 && flags.contains(EptFlags::RWX) && sz == 4096 => {}
            other => {
                serial_println!("VMX self_test FAILED: EPT 4K translate {:?}", other);
                return false;
            }
        }
        match ept.translate(0x40_1000) {
            Some((hpa, _, sz)) if hpa == 0x80_1000 && sz == 2 * 1024 * 1024 => {}
            other => {
                serial_println!("VMX self_test FAILED: EPT 2M translate {:?}", other);
                return false;
            }
        }
        match ept.translate(0x4000_5000) {
            Some((hpa, _, sz)) if hpa == 0x8000_5000 && sz == 1024 * 1024 * 1024 => {}
            other => {
                serial_println!("VMX self_test FAILED: EPT 1G translate {:?}", other);
                return false;
            }
        }

        if ept.update_permissions(0x1000, EptFlags::READ).is_err() {
            serial_println!("VMX self_test FAILED: EPT update_permissions");
            return false;
        }
        if ept.unmap_4k(0x1000) != Ok(0x20_0000) || ept.translate(0x1000).is_some() {
            serial_println!("VMX self_test FAILED: EPT unmap_4k");
            return false;
        }

        // 4. Verify VM-exit dispatcher on simulated exits
        let mut vcpu = Vcpu::new_unbacked(0, caps.backend);
        vcpu.regs.rax = 0x4000_0000;
        vcpu.regs.rip = 0x1000;
        let act_cpuid = vcpu.dispatch_vm_exit(VmxExitReason::Cpuid, 0, 0, 2, Some(&ept));
        if act_cpuid != VmExitAction::AdvanceRipAndResume(2)
            || vcpu.regs.rip != 0x1002
            || vcpu.regs.rbx != u32::from_le_bytes(*b"Nexu") as u64
        {
            serial_println!("VMX self_test FAILED: CPUID exit dispatch");
            return false;
        }

        vcpu.regs.rax = HypercallId::Ping as u64;
        let act_vmcall = vcpu.dispatch_vm_exit(VmxExitReason::Vmcall, 0, 0, 3, Some(&ept));
        if act_vmcall != VmExitAction::HypercallHandled(0x4E45_5855) {
            serial_println!("VMX self_test FAILED: VMCALL exit dispatch");
            return false;
        }

        let act_ept =
            vcpu.dispatch_vm_exit(VmxExitReason::EptViolation, 0x2, 0xDEAD_0000, 0, Some(&ept));
        if act_ept
            != (VmExitAction::MmioFault {
                gpa: 0xDEAD_0000,
                is_write: true,
                size: 4,
            })
        {
            serial_println!("VMX self_test FAILED: EPT violation dispatch");
            return false;
        }

        let act_hlt = vcpu.dispatch_vm_exit(VmxExitReason::Hlt, 0, 0, 1, Some(&ept));
        if act_hlt != VmExitAction::Halt || !vcpu.halted {
            serial_println!("VMX self_test FAILED: HLT exit dispatch");
            return false;
        }
    }

    serial_println!("VMX self_test PASSED");
    true
}
