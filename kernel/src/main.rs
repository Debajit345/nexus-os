#![no_std]
#![no_main]

mod cap;
mod console;
mod gdt;
mod idt;
mod limine;
mod serial;

use core::arch::asm;
use core::panic::PanicInfo;

use nexus_abi::{ObjectType, Personality, Rights};

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

    if let Some(hhdm) = HHDM_REQUEST.response() {
        serial_println!("HHDM offset: {:#x}", hhdm.offset);
    }

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
    }

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
