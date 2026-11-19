#![no_std]
#![no_main]

extern crate alloc;

mod acpi;
mod aml;
mod apic;
mod audio;
mod blk;
mod blk_mq;
mod cap;
mod console;
mod cpu;
mod desktop_boot;
mod drm;
mod exec;
mod fs_mount;
mod gdt;
mod hardening;
mod idt;
mod iommu;
mod ipc;
mod ksh;
mod limine;
mod mm;
mod net;
mod nvme;
mod page_cache;
mod pci;
mod power;
mod proc;
mod rcu;
mod ring3_test;
mod sched;
mod serial;
mod slab;
mod smp;
mod sync;
mod syscall;
mod time;
mod usermode;
mod vfs;
mod vmm;
mod xhci;

use core::arch::asm;
use core::panic::PanicInfo;

use nexus_abi::{AbiError, ObjectType, Personality, Rights, SyscallNumber};

use crate::cap::CapTable;
use crate::limine::{
    BaseRevision, FramebufferResponse, HhdmResponse, KernelAddressResponse, MemmapResponse,
    Request, RsdpResponse, SmpResponse, FRAMEBUFFER_REQUEST_ID, HHDM_REQUEST_ID,
    KERNEL_ADDRESS_REQUEST_ID, MEMMAP_REQUEST_ID, MEMMAP_USABLE, RSDP_REQUEST_ID, SMP_REQUEST_ID,
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

#[used]
#[link_section = ".requests"]
static RSDP_REQUEST: Request<RsdpResponse> = Request::new(RSDP_REQUEST_ID, 0);

#[used]
#[link_section = ".requests"]
static SMP_REQUEST: Request<SmpResponse> = Request::new(SMP_REQUEST_ID, 0);

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
    let cpu_info = cpu::detect_and_enable();
    cpu::init_per_cpu_gs(0);
    usermode::init_syscall_msrs();
    serial_println!(
        "CPU: {:?} ({}) core={:?} avx2={} avx512={} smep={} smap={} x2apic={}",
        cpu_info.vendor,
        cpu_info.brand_str(),
        cpu_info.core_type,
        cpu_info.features.avx2,
        cpu_info.features.avx512f,
        cpu_info.features.smep,
        cpu_info.features.smap,
        cpu_info.features.x2apic
    );
    fb_println!("CPU: {:?} ({:?})", cpu_info.vendor, cpu_info.core_type);
    serial_println!("GDT, TSS, IDT, and Ring 3 syscall/sysret MSRs initialized.");
    fb_println!("GDT, TSS, IDT, and syscall MSRs ready.");

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

    // Initialize VMM, Scheduler, IPC, and PIC/PIT/Keyboard
    let kernel_as = vmm::AddressSpace::new_from_current();
    let k_phys = kernel_as.translate(kernel_as.pml4_phys);
    serial_println!(
        "VMM active: PML4={:#x} (self-check={:?})",
        kernel_as.pml4_phys,
        k_phys.is_some()
    );
    fb_println!("VMM active: PML4={:#x}", kernel_as.pml4_phys);

    sched::init();
    let init_tid = sched::spawn("init", Personality::Native, 128);
    let posix_tid = sched::spawn("posix-sh", Personality::Posix, 100);
    serial_println!(
        "Scheduler ready: {} tasks (init={:?}, posix={:?})",
        sched::task_count(),
        init_tid,
        posix_tid
    );
    fb_println!("Scheduler ready: {} tasks", sched::task_count());

    let (ep_a, ep_b) = ipc::create_channel();
    let _ = ipc::send(
        ep_a,
        ipc::Message::new(init_tid.as_u64(), 1, alloc::vec![0x4E, 0x58], alloc::vec![]),
    );
    let ipc_ok = ipc::recv(ep_b).is_some();
    let qid = ipc::create_queue();
    let _ = ipc::post(
        qid,
        ipc::CompletionEvent {
            user_data: 42,
            status: 0,
            bytes: 2,
        },
    );
    let cq_ok = ipc::poll(qid).is_some();
    serial_println!(
        "IPC & Async CQ ready: channel_ok={}, cq_ok={}",
        ipc_ok,
        cq_ok
    );
    fb_println!("IPC & Async CQ ready.");

    apic::init_pic_pit_keyboard();
    apic::init_pit(100);
    serial_println!("PIC/PIT (100 Hz) and PS/2 keyboard IRQ initialized.");

    // ACPI (MADT/MCFG) & PCIe enumeration
    let mcfg_allocs = if let Some(rsdp) = RSDP_REQUEST.response() {
        if let Some(acpi_info) = acpi::init(rsdp.address) {
            serial_println!(
                "ACPI: lapic={:#x}, local_apics={}, io_apics={}, mcfg_regions={}",
                acpi_info.lapic_addr,
                acpi_info.local_apics.len(),
                acpi_info.io_apics.len(),
                acpi_info.mcfg_allocations.len()
            );
            fb_println!(
                "ACPI: {} LAPICs, {} MCFG regions",
                acpi_info.local_apics.len(),
                acpi_info.mcfg_allocations.len()
            );
            acpi_info.mcfg_allocations
        } else {
            alloc::vec::Vec::new()
        }
    } else {
        alloc::vec::Vec::new()
    };

    let pci_devs = pci::enumerate(&mcfg_allocs);
    serial_println!("PCIe: enumerated {} devices", pci_devs.len());
    fb_println!("PCIe: {} devices enumerated", pci_devs.len());

    // IOMMU & Security Policy
    let iommu_kind = iommu::IommuManager::init(cpu_info.vendor);
    let wx_ok = !iommu::enforce_wx(vmm::PageFlags::PRESENT.union(vmm::PageFlags::WRITABLE));
    serial_println!("IOMMU: {:?}, W^X enforced={}", iommu_kind, wx_ok);
    fb_println!("IOMMU: {:?}, W^X={}", iommu_kind, wx_ok);

    // Probe NVMe, VirtIO-Blk, xHCI USB 3.x, and DRM GPUs from PCIe bus
    let mut nvme_count = 0usize;
    let mut xhci_count = 0usize;
    for dev in &pci_devs {
        match dev.class() {
            pci::PciClass::Nvme => {
                if nvme::NvmeController::probe(dev).is_some() {
                    nvme_count += 1;
                }
            }
            pci::PciClass::XhciUsb if xhci::XhciController::probe(dev).is_some() => {
                xhci_count += 1;
            }
            _ => {}
        }
    }
    let gpus = drm::probe_gpus(&pci_devs);
    let audio_devs = audio::probe_audio(&pci_devs);
    let mut net_stack = net::NetStack::new();
    let nic_count = net_stack.probe_nics(&pci_devs);
    let power_info = power::init(cpu_info.vendor);
    time::init();
    let rtc = time::read_rtc();
    serial_println!(
        "Drivers probed: nvme={}, xhci={}, drm_gpus={}, audio={}, nics={}, hwp={}, cppc={}, rtc={:04}-{:02}-{:02}",
        nvme_count,
        xhci_count,
        gpus.len(),
        audio_devs.len(),
        nic_count,
        power_info.hwp_supported,
        power_info.amd_cppc,
        rtc.year,
        rtc.month,
        rtc.day
    );
    fb_println!(
        "Drivers: {} NVMe, {} xHCI, {} GPU, {} Audio, {} NIC",
        nvme_count,
        xhci_count,
        gpus.len(),
        audio_devs.len(),
        nic_count
    );

    // SMP multi-core bootstrap
    let online_cpus = if let Some(smp_resp) = SMP_REQUEST.response() {
        let online = smp::init(smp_resp);
        serial_println!(
            "SMP: {}/{} CPUs online (BSP LAPIC={})",
            online,
            smp_resp.cpu_count,
            smp::bsp_lapic_id()
        );
        fb_println!("SMP: {}/{} CPUs online", online, smp_resp.cpu_count);
        online
    } else {
        1
    };

    // Built-in kernel recovery/diagnostic shell self-test
    let mm_now = mm::stats();
    let sys_info = nexus_shell::SystemInfo {
        os_name: alloc::string::String::from("Nexus OS"),
        version: alloc::string::String::from("0.1.0"),
        arch: alloc::string::String::from("x86_64"),
        cpu_vendor: alloc::format!("{:?}", cpu_info.vendor),
        total_ram_mib: ((mm_now.total_frames * mm::PAGE_SIZE) / (1024 * 1024)) as u64,
        free_ram_mib: ((mm_now.free_frames * mm::PAGE_SIZE) / (1024 * 1024)) as u64,
        online_cpus: online_cpus as u32,
        pci_count: pci_devs.len() as u32,
    };
    ksh::run_boot_diagnostics(&sys_info);
    hardening::init();
    let hardening_ok = hardening::self_test();
    let proc_ok = proc::self_test();
    let sync_ok = sync::self_test();
    let page_cache_ok = page_cache::self_test();
    serial_println!(
        "Production kernel self-tests: hardening={:?} proc={:?} sync={} page_cache={:?}",
        hardening_ok,
        proc_ok,
        sync_ok,
        page_cache_ok
    );
    let slab_ok = slab::self_test();
    let aml_ok = aml::self_test();
    let blk_mq_ok = blk_mq::self_test();
    let rcu_ok = rcu::self_test();
    serial_println!(
        "Production kernel self-tests (Stages 25-28): slab={:?} aml={:?} blk_mq={:?} rcu={:?}",
        slab_ok,
        aml_ok,
        blk_mq_ok,
        rcu_ok
    );
    fs_mount::run_fs_selftest();
    ring3_test::run_ring3_exec_selftest();
    desktop_boot::run_init_selftest();
    if let Some(fb_resp) = FRAMEBUFFER_REQUEST.response() {
        if let Some(fb) = fb_resp.framebuffers().first() {
            desktop_boot::render_desktop_preview(fb);
        }
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
