#![no_std]
#![no_main]

extern crate alloc;

mod cap;
mod console;
mod gdt;
mod idt;
mod limine;
mod mm;
mod serial;
mod syscall;
mod vfs;

use core::arch::asm;
use core::panic::PanicInfo;

use nexus_abi::{AbiError, ObjectType, Personality, Rights, SyscallNumber};

use crate::cap::CapTable;
use crate::limine::{
    BaseRevision, FramebufferResponse, HhdmResponse, KernelAddressResponse, MemmapResponse,
    Request, FRAMEBUFFER_REQUEST_ID, HHDM_REQUEST_ID, KERNEL_ADDRESS_REQUEST_ID, MEMMAP_REQUEST_ID,
    MEMMAP_USABLE,
};

#[used]
#[link_section = ".requests"]
static BASE_REVISION: BaseRevision = BaseRevision::new(2);

#[used]
#[link_section = ".requests"]
static FRAMEBUFFER_REQUEST: Request<FramebufferResponse> = Request::new(FRAMEBUFFER_REQUEST_ID, 0);

#[used]
#[link_section = ".requests"]
static HHDM_REQUEST: Request<HhdmResponse> = Request::new(HHDM_REQUEST_ID, 0);

#[used]
#[link_section = ".requests"]
static MEMMAP_REQUEST: Request<MemmapResponse> = Request::new(MEMMAP_REQUEST_ID, 0);

#[used]
#[link_section = ".requests"]
static KERNEL_ADDRESS_REQUEST: Request<KernelAddressResponse> =
    Request::new(KERNEL_ADDRESS_REQUEST_ID, 0);

#[no_mangle]
pub extern "C" fn _start() -> ! {
    serial::init();
    serial_println!("Nexus OS microkernel booting (x86_64, Limine)...");

    if !BASE_REVISION.is_supported() {
        serial_println!("ERROR: Limine base revision 2 not supported by bootloader");
        halt_loop();
    }

    if let Some(fb_resp) = FRAMEBUFFER_REQUEST.response() {
        if let Some(fb) = fb_resp.framebuffers().first() {
            console::init(fb);
            fb_println!("Nexus OS v0.1.0 (x86_64 capability microkernel)");
            fb_println!("Framebuffer: {}x{} @ {} bpp", fb.width, fb.height, fb.bpp);
            serial_println!(
                "Framebuffer initialized: {}x{} ({} bpp)",
                fb.width,
                fb.height,
                fb.bpp
            );
        }
    }

    gdt::init();
    idt::init();
    serial_println!("GDT, TSS, and IDT initialized.");
    fb_println!("GDT, TSS, and IDT initialized.");

    let hhdm_offset = if let Some(hhdm) = HHDM_REQUEST.response() {
        serial_println!("HHDM offset: {:#x}", hhdm.offset);
        hhdm.offset
    } else {
        0
    };

    if let Some(kaddr) = KERNEL_ADDRESS_REQUEST.response() {
        serial_println!(
            "Kernel phys={:#x} virt={:#x}",
            kaddr.physical_base,
            kaddr.virtual_base
        );
    }

    if let Some(memmap) = MEMMAP_REQUEST.response() {
        let mut usable_bytes: u64 = 0;
        for entry in memmap.entries() {
            if entry.entry_type == MEMMAP_USABLE {
                usable_bytes += entry.length;
            }
        }
        serial_println!(
            "Memory map: {} entries, {} MiB usable",
            memmap.entry_count,
            usable_bytes / (1024 * 1024)
        );
        fb_println!("Usable RAM: {} MiB", usable_bytes / (1024 * 1024));

        let mm_stats = mm::init(memmap, hhdm_offset);
        serial_println!(
            "MM initialized: {} total frames, {} free frames, heap={} KiB",
            mm_stats.total_frames,
            mm_stats.free_frames,
            mm_stats.heap_size / 1024
        );
        fb_println!(
            "MM ready: {} frames, heap {} KiB",
            mm_stats.total_frames,
            mm_stats.heap_size / 1024
        );
    }

    // Initialize VFS (RamFs) and run self-test
    vfs::init();
    if let Ok(entries) = vfs::list_dir("/") {
        serial_println!("VFS mounted (ramfs): / -> {:?}", entries);
        fb_println!("VFS mounted (ramfs): {} root entries", entries.len());
    }
    if let Ok(conf) = vfs::read_to_vec("/etc/nexus.conf") {
        serial_println!("VFS /etc/nexus.conf: {} bytes", conf.len());
        fb_println!("VFS /etc/nexus.conf: {} bytes", conf.len());
    }

    // Unified syscall router self-test across personalities
    let msg = b"hello from Native DebugLog";
    let native_res = syscall::dispatch(
        Personality::Native,
        SyscallNumber::DebugLog as u64,
        [msg.as_ptr() as u64, msg.len() as u64, 0, 0, 0, 0],
    );
    let conf_path = b"/etc/nexus.conf";
    let posix_fd = syscall::dispatch(
        Personality::Posix,
        SyscallNumber::PosixOpenAt as u64,
        [
            0,
            conf_path.as_ptr() as u64,
            conf_path.len() as u64,
            vfs::OpenFlags::READ.bits() as u64,
            0,
            0,
        ],
    );
    if let Ok(fd) = posix_fd {
        let mut buf = [0u8; 32];
        let _ = syscall::dispatch(
            Personality::Posix,
            SyscallNumber::PosixRead as u64,
            [fd, buf.as_mut_ptr() as u64, buf.len() as u64, 0, 0, 0],
        );
        let _ = syscall::dispatch(
            Personality::Posix,
            SyscallNumber::PosixClose as u64,
            [fd, 0, 0, 0, 0, 0],
        );
    }
    let nt_res = syscall::dispatch(
        Personality::Nt,
        SyscallNumber::NtAllocateVirtualMemory as u64,
        [0; 6],
    );
    let darwin_res = syscall::dispatch(
        Personality::Darwin,
        SyscallNumber::DarwinTaskSelf as u64,
        [0; 6],
    );
    let invalid_res = syscall::dispatch(Personality::Native, 0xDEAD_BEEF, [0; 6]);
    let mismatch_res = syscall::dispatch(
        Personality::Posix,
        SyscallNumber::NtCreateFile as u64,
        [0; 6],
    );
    serial_println!(
        "Syscall self-test: native={:?}, posix_fd={:?}, nt={:?}, darwin={:?}, invalid={:?}, mismatch={:?}",
        native_res,
        posix_fd,
        nt_res,
        darwin_res,
        invalid_res,
        mismatch_res
    );
    fb_println!(
        "Syscall router ready (invalid={:?}, darwin={:?}).",
        invalid_res == Err(AbiError::NotSupported),
        darwin_res
    );

    // Smoke-test the capability table and multi-personality ABI
    let mut cap_table = CapTable::new();
    let root_proc = cap_table
        .insert(ObjectType::Process, Rights::ALL, 0x1000)
        .expect("insert root process cap");
    let posix_ctx = cap_table
        .insert(
            ObjectType::PersonalityContext,
            Rights::READ.union(Rights::WRITE).union(Rights::DUPLICATE),
            Personality::Posix as usize,
        )
        .expect("insert posix personality cap");
    let posix_ro = cap_table
        .duplicate(posix_ctx, Rights::READ)
        .expect("duplicate posix cap with attenuated rights");
    let _ = cap_table
        .lookup(posix_ro, Rights::READ)
        .expect("lookup attenuated posix cap");
    cap_table.close(posix_ro).expect("close duplicated cap");

    serial_println!(
        "Capability core ready: {} active caps (root={:#x}, posix={:#x})",
        cap_table.count(),
        root_proc.as_raw(),
        posix_ctx.as_raw()
    );
    fb_println!("Capability core ready ({} active caps).", cap_table.count());

    // Trigger a breakpoint (#BP) to verify IDT dispatch works cleanly
    unsafe {
        asm!("int3", options(nomem, nostack));
    }

    serial_println!("Nexus OS early boot complete. Halting.");
    fb_println!("Early boot complete.");
    halt_loop();
}

fn halt_loop() -> ! {
    loop {
        unsafe {
            asm!("hlt", options(nomem, nostack));
        }
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("KERNEL PANIC: {}", info);
    fb_println!("KERNEL PANIC: {}", info);
    loop {
        unsafe {
            asm!("cli; hlt", options(nomem, nostack));
        }
    }
}
