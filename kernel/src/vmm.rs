//! 4-level x86_64 Virtual Memory Manager (PML4 -> PDPT -> PD -> PT).

use core::arch::asm;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

use crate::mm::{alloc_frame, stats, PAGE_SIZE};

const PHYS_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;
const HUGE_1G_ADDR_MASK: u64 = 0x000F_FFFF_C000_0000;
const HUGE_2M_ADDR_MASK: u64 = 0x000F_FFFF_FFE0_0000;
const PAGE_OFFSET_MASK: u64 = 0xFFF;
const HUGE_1G_OFFSET_MASK: u64 = 0x3FFF_FFFF;
const HUGE_2M_OFFSET_MASK: u64 = 0x1F_FFFF;
const TABLE_ENTRIES: usize = 512;

/// x86_64 page table entry flags.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PageFlags(u64);

#[allow(dead_code)]
impl PageFlags {
    pub const PRESENT: Self = Self(1 << 0);
    pub const WRITABLE: Self = Self(1 << 1);
    pub const USER: Self = Self(1 << 2);
    pub const WRITE_THROUGH: Self = Self(1 << 3);
    pub const NO_CACHE: Self = Self(1 << 4);
    pub const ACCESSED: Self = Self(1 << 5);
    pub const DIRTY: Self = Self(1 << 6);
    pub const HUGE_PAGE: Self = Self(1 << 7);
    pub const GLOBAL: Self = Self(1 << 8);
    pub const NO_EXECUTE: Self = Self(1 << 63);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for PageFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PageFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PageFlags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PageFlags {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl Not for PageFlags {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

#[repr(C, align(4096))]
struct PageTable {
    entries: [u64; TABLE_ENTRIES],
}

impl PageTable {
    fn zero(&mut self) {
        self.entries.fill(0);
    }
}

#[inline]
fn hhdm_offset() -> u64 {
    stats().hhdm_offset
}

#[inline]
unsafe fn table_from_phys<'a>(phys: u64) -> &'a mut PageTable {
    let virt = hhdm_offset().wrapping_add(phys & PHYS_ADDR_MASK);
    &mut *(virt as *mut PageTable)
}

#[inline]
const fn pml4_index(virt: u64) -> usize {
    ((virt >> 39) & 0x1FF) as usize
}

#[inline]
const fn pdpt_index(virt: u64) -> usize {
    ((virt >> 30) & 0x1FF) as usize
}

#[inline]
const fn pd_index(virt: u64) -> usize {
    ((virt >> 21) & 0x1FF) as usize
}

#[inline]
const fn pt_index(virt: u64) -> usize {
    ((virt >> 12) & 0x1FF) as usize
}

#[inline]
fn read_cr3() -> u64 {
    let cr3: u64;
    unsafe {
        asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack, preserves_flags));
    }
    cr3 & PHYS_ADDR_MASK
}

/// Represents a 4-level x86_64 virtual address space rooted at a PML4 physical frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AddressSpace {
    pub pml4_phys: u64,
}

#[allow(dead_code)]
impl AddressSpace {
    /// Wrap the currently active address space by reading CR3.
    pub fn new_from_current() -> Self {
        Self {
            pml4_phys: read_cr3(),
        }
    }

    /// Allocate a fresh PML4 for a user address space and copy upper-half kernel mappings (entries 256..512).
    pub fn new_user() -> Option<Self> {
        let new_pml4_phys = alloc_frame()?;
        let cur_pml4_phys = read_cr3();

        unsafe {
            let new_pml4 = table_from_phys(new_pml4_phys);
            new_pml4.zero();

            let cur_pml4 = table_from_phys(cur_pml4_phys);
            for i in 256..TABLE_ENTRIES {
                new_pml4.entries[i] = cur_pml4.entries[i];
            }
        }

        Some(Self {
            pml4_phys: new_pml4_phys,
        })
    }

    /// Map a 4 KiB virtual page `virt` to physical frame `phys` with `flags`.
    pub fn map_page(&mut self, virt: u64, phys: u64, flags: PageFlags) -> Result<(), &'static str> {
        if (virt & (PAGE_SIZE as u64 - 1)) != 0 || (phys & (PAGE_SIZE as u64 - 1)) != 0 {
            return Err("unaligned virtual or physical address");
        }

        let user_bit = if flags.contains(PageFlags::USER) {
            PageFlags::USER.bits()
        } else {
            0
        };
        let intermediate_flags = PageFlags::PRESENT.bits() | PageFlags::WRITABLE.bits() | user_bit;

        unsafe {
            let pml4 = table_from_phys(self.pml4_phys);
            let pdpt_phys =
                Self::get_or_alloc_table(&mut pml4.entries[pml4_index(virt)], intermediate_flags)?;

            let pdpt = table_from_phys(pdpt_phys);
            let pdpt_entry = &mut pdpt.entries[pdpt_index(virt)];
            if (*pdpt_entry & PageFlags::PRESENT.bits()) != 0
                && (*pdpt_entry & PageFlags::HUGE_PAGE.bits()) != 0
            {
                return Err("cannot map 4KiB page inside 1GiB huge page");
            }
            let pd_phys = Self::get_or_alloc_table(pdpt_entry, intermediate_flags)?;

            let pd = table_from_phys(pd_phys);
            let pd_entry = &mut pd.entries[pd_index(virt)];
            if (*pd_entry & PageFlags::PRESENT.bits()) != 0
                && (*pd_entry & PageFlags::HUGE_PAGE.bits()) != 0
            {
                return Err("cannot map 4KiB page inside 2MiB huge page");
            }
            let pt_phys = Self::get_or_alloc_table(pd_entry, intermediate_flags)?;

            let pt = table_from_phys(pt_phys);
            pt.entries[pt_index(virt)] =
                (phys & PHYS_ADDR_MASK) | flags.bits() | PageFlags::PRESENT.bits();
        }

        Self::tlb_flush(virt);
        Ok(())
    }

    /// Unmap a 4 KiB virtual page `virt`, returning its physical frame address if it was mapped.
    pub fn unmap_page(&mut self, virt: u64) -> Option<u64> {
        unsafe {
            let pml4 = table_from_phys(self.pml4_phys);
            let pml4_entry = pml4.entries[pml4_index(virt)];
            if (pml4_entry & PageFlags::PRESENT.bits()) == 0 {
                return None;
            }

            let pdpt = table_from_phys(pml4_entry);
            let pdpt_entry = pdpt.entries[pdpt_index(virt)];
            if (pdpt_entry & PageFlags::PRESENT.bits()) == 0
                || (pdpt_entry & PageFlags::HUGE_PAGE.bits()) != 0
            {
                return None;
            }

            let pd = table_from_phys(pdpt_entry);
            let pd_entry = pd.entries[pd_index(virt)];
            if (pd_entry & PageFlags::PRESENT.bits()) == 0
                || (pd_entry & PageFlags::HUGE_PAGE.bits()) != 0
            {
                return None;
            }

            let pt = table_from_phys(pd_entry);
            let pt_entry = &mut pt.entries[pt_index(virt)];
            if (*pt_entry & PageFlags::PRESENT.bits()) == 0 {
                return None;
            }

            let phys = *pt_entry & PHYS_ADDR_MASK;
            *pt_entry = 0;
            Self::tlb_flush(virt);
            Some(phys)
        }
    }

    /// Translate a virtual address to its mapped physical address, if present.
    pub fn translate(&self, virt: u64) -> Option<u64> {
        unsafe {
            let pml4 = table_from_phys(self.pml4_phys);
            let pml4_entry = pml4.entries[pml4_index(virt)];
            if (pml4_entry & PageFlags::PRESENT.bits()) == 0 {
                return None;
            }

            let pdpt = table_from_phys(pml4_entry);
            let pdpt_entry = pdpt.entries[pdpt_index(virt)];
            if (pdpt_entry & PageFlags::PRESENT.bits()) == 0 {
                return None;
            }
            if (pdpt_entry & PageFlags::HUGE_PAGE.bits()) != 0 {
                return Some((pdpt_entry & HUGE_1G_ADDR_MASK) | (virt & HUGE_1G_OFFSET_MASK));
            }

            let pd = table_from_phys(pdpt_entry);
            let pd_entry = pd.entries[pd_index(virt)];
            if (pd_entry & PageFlags::PRESENT.bits()) == 0 {
                return None;
            }
            if (pd_entry & PageFlags::HUGE_PAGE.bits()) != 0 {
                return Some((pd_entry & HUGE_2M_ADDR_MASK) | (virt & HUGE_2M_OFFSET_MASK));
            }

            let pt = table_from_phys(pd_entry);
            let pt_entry = pt.entries[pt_index(virt)];
            if (pt_entry & PageFlags::PRESENT.bits()) == 0 {
                return None;
            }

            Some((pt_entry & PHYS_ADDR_MASK) | (virt & PAGE_OFFSET_MASK))
        }
    }

    /// Activate this address space by loading its PML4 physical address into CR3.
    pub fn activate(&self) {
        unsafe {
            asm!(
                "mov cr3, {}",
                in(reg) (self.pml4_phys & PHYS_ADDR_MASK),
                options(nostack, preserves_flags)
            );
        }
    }

    /// Invalidate the TLB entry for `virt`.
    #[inline]
    pub fn tlb_flush(virt: u64) {
        unsafe {
            asm!("invlpg [{}]", in(reg) virt, options(nostack, preserves_flags));
        }
    }

    unsafe fn get_or_alloc_table(entry: &mut u64, flags: u64) -> Result<u64, &'static str> {
        if (*entry & PageFlags::PRESENT.bits()) != 0 {
            // Ensure intermediate permissions (e.g. USER / WRITABLE) are propagated if needed.
            *entry |= flags;
            return Ok(*entry & PHYS_ADDR_MASK);
        }

        let frame = alloc_frame().ok_or("out of physical frames for page table")?;
        let table = table_from_phys(frame);
        table.zero();
        *entry = (frame & PHYS_ADDR_MASK) | flags;
        Ok(frame)
    }
}
