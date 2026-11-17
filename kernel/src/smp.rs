#![allow(dead_code)]

//! Symmetric Multiprocessing (SMP) Application Processor (AP) bootstrapper.

use core::arch::asm;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::limine::{SmpInfo, SmpResponse};
use crate::mm::SpinLock;
use crate::serial_println;

static BSP_LAPIC_ID: AtomicU32 = AtomicU32::new(0);
static ONLINE_CPUS: AtomicUsize = AtomicUsize::new(1);
static AP_INIT_LOCK: SpinLock<()> = SpinLock::new(());

/// Bootstrap all Application Processors reported by Limine and return the online CPU count.
pub fn init(smp: &SmpResponse) -> usize {
    BSP_LAPIC_ID.store(smp.bsp_lapic_id, Ordering::Release);
    ONLINE_CPUS.store(1, Ordering::Release);

    let cpus = smp.cpus();
    let total_cpus = cpus.len().max(1);
    let entry_addr = ap_entry as *const () as u64;

    for cpu in cpus {
        if cpu.lapic_id == smp.bsp_lapic_id {
            continue;
        }
        cpu.goto_address.store(entry_addr, Ordering::Release);
    }

    // Wait briefly for APs to complete initialization and increment ONLINE_CPUS.
    for _ in 0..10_000_000u32 {
        if ONLINE_CPUS.load(Ordering::Acquire) >= total_cpus {
            break;
        }
        core::hint::spin_loop();
    }

    online_cpus()
}

/// Entry point jumped to by each Application Processor via Limine's `goto_address`.
pub extern "C" fn ap_entry(info: &SmpInfo) -> ! {
    {
        let _guard = AP_INIT_LOCK.lock();
        crate::gdt::init();
        crate::idt::init();
        serial_println!(
            "[SMP] AP online: processor_id={} lapic_id={}",
            info.processor_id,
            info.lapic_id
        );
    }

    ONLINE_CPUS.fetch_add(1, Ordering::AcqRel);

    loop {
        unsafe {
            asm!("cli; hlt", options(nomem, nostack));
        }
    }
}

/// Return the number of CPUs currently online (including the BSP).
pub fn online_cpus() -> usize {
    ONLINE_CPUS.load(Ordering::Acquire)
}

/// Return the recorded Bootstrap Processor (BSP) LAPIC ID.
pub fn bsp_lapic_id() -> u32 {
    BSP_LAPIC_ID.load(Ordering::Acquire)
}
