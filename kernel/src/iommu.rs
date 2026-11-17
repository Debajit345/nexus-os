//! Intel VT-d / AMD-Vi IOMMU DMA remapping and kernel security sandbox policy.
#![allow(dead_code)]

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use nexus_abi::Personality;

use crate::cpu::CpuVendor;
use crate::mm::SpinLock;
use crate::vmm::PageFlags;

/// Hardware IOMMU implementation detected on the platform.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IommuKind {
    IntelVtd,
    AmdVi,
    None,
}

/// Isolated DMA remapping domain identifier.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DmaDomainId(pub u16);

/// A single IOVA-to-physical DMA mapping within a domain.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DmaMapping {
    pub iova: u64,
    pub phys: u64,
    pub size: usize,
    pub read: bool,
    pub write: bool,
}

/// Security sandbox profile assigned to a task or driver.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SandboxProfile {
    Strict,
    PosixCompat,
    NtCompat,
    DarwinCompat,
    KernelDriver,
}

struct IommuState {
    kind: IommuKind,
    next_domain: u16,
    domains: BTreeMap<DmaDomainId, Vec<DmaMapping>>,
}

impl IommuState {
    const fn new() -> Self {
        Self {
            kind: IommuKind::None,
            next_domain: 1,
            domains: BTreeMap::new(),
        }
    }
}

static IOMMU: SpinLock<IommuState> = SpinLock::new(IommuState::new());

/// Kernel IOMMU & DMA isolation manager.
pub struct IommuManager;

impl IommuManager {
    /// Initialize the IOMMU subsystem based on the detected CPU vendor.
    pub fn init(vendor: CpuVendor) -> IommuKind {
        let kind = match vendor {
            CpuVendor::Intel => IommuKind::IntelVtd,
            CpuVendor::Amd => IommuKind::AmdVi,
            CpuVendor::Unknown => IommuKind::None,
        };
        let mut state = IOMMU.lock();
        state.kind = kind;
        kind
    }

    /// Return the active IOMMU kind.
    pub fn kind() -> IommuKind {
        IOMMU.lock().kind
    }

    /// Allocate a new isolated DMA domain.
    pub fn create_domain() -> DmaDomainId {
        let mut state = IOMMU.lock();
        let id = DmaDomainId(state.next_domain);
        state.next_domain = state.next_domain.wrapping_add(1).max(1);
        state.domains.insert(id, Vec::new());
        id
    }

    /// Map an IOVA range to physical memory inside `domain`.
    pub fn map_dma(domain: DmaDomainId, mapping: DmaMapping) -> Result<(), &'static str> {
        if mapping.size == 0 || (mapping.iova & 0xFFF) != 0 || (mapping.phys & 0xFFF) != 0 {
            return Err("unaligned or empty DMA mapping");
        }
        let end = mapping
            .iova
            .checked_add(mapping.size as u64)
            .ok_or("IOVA overflow")?;
        let _ = mapping
            .phys
            .checked_add(mapping.size as u64)
            .ok_or("physical address overflow")?;

        let mut state = IOMMU.lock();
        let entries = state.domains.get_mut(&domain).ok_or("unknown DMA domain")?;
        for existing in entries.iter() {
            let ex_end = existing.iova.saturating_add(existing.size as u64);
            if mapping.iova < ex_end && end > existing.iova {
                return Err("overlapping IOVA range");
            }
        }
        entries.push(mapping);
        Ok(())
    }

    /// Unmap an IOVA range from `domain`.
    pub fn unmap_dma(domain: DmaDomainId, iova: u64) -> bool {
        let mut state = IOMMU.lock();
        if let Some(entries) = state.domains.get_mut(&domain) {
            let before = entries.len();
            entries.retain(|m| m.iova != iova);
            return entries.len() != before;
        }
        false
    }

    /// Translate an IOVA within `domain` to a physical address.
    pub fn translate(domain: DmaDomainId, iova: u64) -> Option<u64> {
        let state = IOMMU.lock();
        let entries = state.domains.get(&domain)?;
        for m in entries {
            let end = m.iova.saturating_add(m.size as u64);
            if iova >= m.iova && iova < end {
                return Some(m.phys + (iova - m.iova));
            }
        }
        None
    }
}

/// Enforce Write-XOR-Execute (W^X): reject pages that are both writable and executable.
#[inline]
pub fn enforce_wx(flags: PageFlags) -> bool {
    let writable = flags.contains(PageFlags::WRITABLE);
    let executable = !flags.contains(PageFlags::NO_EXECUTE);
    !(writable && executable)
}

/// Check whether a task with `profile` is permitted to invoke syscalls for `personality`.
#[inline]
pub fn check_syscall_allowed(profile: SandboxProfile, personality: Personality) -> bool {
    match profile {
        SandboxProfile::Strict => matches!(personality, Personality::Native),
        SandboxProfile::PosixCompat => {
            matches!(personality, Personality::Native | Personality::Posix)
        }
        SandboxProfile::NtCompat => matches!(personality, Personality::Native | Personality::Nt),
        SandboxProfile::DarwinCompat => {
            matches!(personality, Personality::Native | Personality::Darwin)
        }
        SandboxProfile::KernelDriver => true,
    }
}
