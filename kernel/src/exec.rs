#![allow(dead_code)]

//! Multi-OS process loader for ELF64, PE32+, and Mach-O 64-bit binaries.

use core::ptr;

use nexus_abi::Personality;
use nexus_compat::{inspect_binary, BinaryFormat, CompatError};

use crate::iommu::enforce_wx;
use crate::mm::{alloc_frame, stats, PAGE_SIZE};
use crate::sched::{spawn, TaskId};
use crate::usermode::UserContext;
use crate::vmm::{AddressSpace, PageFlags};

/// Base virtual address of the user-mode stack region.
pub const USER_STACK_BASE: u64 = 0x7FFF_FFFF_0000;

/// Number of 4 KiB pages allocated for the initial user-mode stack (16 KiB).
pub const USER_STACK_PAGES: usize = 4;

/// Top of the initial user-mode stack (16-byte aligned).
pub const USER_STACK_TOP: u64 = USER_STACK_BASE + (USER_STACK_PAGES as u64) * (PAGE_SIZE as u64);

/// A validated, mapped, and scheduled user-space process ready for Ring 3 entry.
#[derive(Debug, Clone, Copy)]
pub struct SpawnedProcess {
    pub task_id: TaskId,
    pub personality: Personality,
    pub format: BinaryFormat,
    pub entry_rip: u64,
    pub user_rsp: u64,
    pub address_space: AddressSpace,
}

impl SpawnedProcess {
    /// Build a [`UserContext`] configured for entering Ring 3 at this process's entry point.
    pub const fn user_context(&self) -> UserContext {
        UserContext::new(self.entry_rip, self.user_rsp).with_personality(self.personality)
    }
}

/// Inspect an ELF64, PE32+, or Mach-O 64-bit binary, map its segments into a fresh
/// user [`AddressSpace`] with W^X enforcement, allocate a user stack, and spawn a
/// scheduler task.
pub fn load_and_prepare(
    name: &str,
    binary: &[u8],
    priority: u8,
) -> Result<SpawnedProcess, &'static str> {
    let image = inspect_binary(binary).map_err(compat_err_str)?;
    let mut address_space =
        AddressSpace::new_user().ok_or("failed to allocate user address space")?;

    let hhdm = stats().hhdm_offset;
    let page_size = PAGE_SIZE as u64;

    for seg in &image.segments {
        if seg.virtual_size == 0 {
            continue;
        }

        let mut flags = PageFlags::PRESENT | PageFlags::USER;
        if seg.writable {
            flags |= PageFlags::WRITABLE;
        }
        if !seg.executable {
            flags |= PageFlags::NO_EXECUTE;
        }

        if !enforce_wx(flags) {
            return Err("segment violates W^X policy");
        }

        if seg.file_size > 0 {
            let file_end = seg
                .file_offset
                .checked_add(seg.file_size)
                .ok_or("segment file range overflow")?;
            if file_end as usize > binary.len() {
                return Err("segment file data out of bounds");
            }
        }

        let seg_start = seg.virtual_address & !(page_size - 1);
        let seg_end = seg
            .virtual_address
            .checked_add(seg.virtual_size)
            .ok_or("segment virtual address overflow")?;
        let aligned_end = (seg_end + page_size - 1) & !(page_size - 1);

        let file_vaddr_end = seg
            .virtual_address
            .checked_add(seg.file_size)
            .ok_or("segment file virtual range overflow")?;

        let mut vaddr = seg_start;
        while vaddr < aligned_end {
            let phys = if let Some(existing_phys) = address_space.translate(vaddr) {
                existing_phys & !(page_size - 1)
            } else {
                let frame = alloc_frame().ok_or("out of physical frames for segment")?;
                unsafe {
                    zero_frame(hhdm, frame);
                }
                address_space.map_page(vaddr, frame, flags)?;
                frame
            };

            if seg.file_size > 0 {
                let page_end = vaddr + page_size;
                let copy_start = vaddr.max(seg.virtual_address);
                let copy_end = page_end.min(file_vaddr_end);

                if copy_start < copy_end {
                    let dst_off = (copy_start - vaddr) as usize;
                    let src_off = (seg.file_offset + (copy_start - seg.virtual_address)) as usize;
                    let len = (copy_end - copy_start) as usize;

                    unsafe {
                        let dst = (hhdm.wrapping_add(phys) as *mut u8).add(dst_off);
                        ptr::copy_nonoverlapping(binary[src_off..].as_ptr(), dst, len);
                    }
                }
            }

            vaddr += page_size;
        }
    }

    let stack_flags =
        PageFlags::PRESENT | PageFlags::USER | PageFlags::WRITABLE | PageFlags::NO_EXECUTE;
    if !enforce_wx(stack_flags) {
        return Err("user stack violates W^X policy");
    }

    for i in 0..USER_STACK_PAGES {
        let vaddr = USER_STACK_BASE + (i as u64) * page_size;
        let frame = alloc_frame().ok_or("out of physical frames for user stack")?;
        unsafe {
            zero_frame(hhdm, frame);
        }
        address_space.map_page(vaddr, frame, stack_flags)?;
    }

    let task_id = spawn(name, image.personality, priority);

    Ok(SpawnedProcess {
        task_id,
        personality: image.personality,
        format: image.format,
        entry_rip: image.entry_point,
        user_rsp: USER_STACK_TOP,
        address_space,
    })
}

#[inline]
unsafe fn zero_frame(hhdm_offset: u64, phys: u64) {
    let ptr = hhdm_offset.wrapping_add(phys) as *mut u8;
    ptr::write_bytes(ptr, 0, PAGE_SIZE);
}

const fn compat_err_str(err: CompatError) -> &'static str {
    match err {
        CompatError::TooShort => "binary too short",
        CompatError::UnknownFormat => "unknown binary format",
        CompatError::Elf(_) => "invalid ELF64 binary",
        CompatError::Pe(_) => "invalid PE32+ binary",
        CompatError::MachO(_) => "invalid Mach-O 64-bit binary",
    }
}
