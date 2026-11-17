//! 8259 PIC, Local APIC (MMIO), 8254 PIT timer, and PS/2 keyboard scancode buffer.
#![allow(dead_code)]

use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// x86 Port I/O & MSR helpers
// ---------------------------------------------------------------------------

#[inline]
pub unsafe fn outb(port: u16, val: u8) {
    asm!("out dx, al", in("dx") port, in("al") val, options(nomem, nostack, preserves_flags));
}

#[inline]
pub unsafe fn inb(port: u16) -> u8 {
    let val: u8;
    asm!("in al, dx", out("al") val, in("dx") port, options(nomem, nostack, preserves_flags));
    val
}

#[inline]
unsafe fn io_wait() {
    outb(0x80, 0);
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

// ---------------------------------------------------------------------------
// 8259 Programmable Interrupt Controller (PIC)
// ---------------------------------------------------------------------------

const PIC1_CMD: u16 = 0x20;
const PIC1_DATA: u16 = 0x21;
const PIC2_CMD: u16 = 0xA0;
const PIC2_DATA: u16 = 0xA1;

const ICW1_ICW4: u8 = 0x01;
const ICW1_INIT: u8 = 0x10;
const ICW4_8086: u8 = 0x01;
const PIC_EOI: u8 = 0x20;

pub const PIC1_OFFSET: u8 = 32;
pub const PIC2_OFFSET: u8 = 40;

/// Remap the 8259 PICs to vectors `offset1..offset1+8` and `offset2..offset2+8`.
pub fn remap_pic(offset1: u8, offset2: u8) {
    unsafe {
        let mask1 = inb(PIC1_DATA);
        let mask2 = inb(PIC2_DATA);

        // Start initialization sequence (cascade mode)
        outb(PIC1_CMD, ICW1_INIT | ICW1_ICW4);
        io_wait();
        outb(PIC2_CMD, ICW1_INIT | ICW1_ICW4);
        io_wait();

        // ICW2: Vector offsets
        outb(PIC1_DATA, offset1);
        io_wait();
        outb(PIC2_DATA, offset2);
        io_wait();

        // ICW3: Master/Slave wiring (IRQ2 cascade)
        outb(PIC1_DATA, 4);
        io_wait();
        outb(PIC2_DATA, 2);
        io_wait();

        // ICW4: 8086/88 mode
        outb(PIC1_DATA, ICW4_8086);
        io_wait();
        outb(PIC2_DATA, ICW4_8086);
        io_wait();

        // Restore saved masks
        outb(PIC1_DATA, mask1);
        outb(PIC2_DATA, mask2);
    }
}

/// Remap the 8259 PICs to vectors 32..47 and mask all IRQ lines.
pub fn disable_pic() {
    remap_pic(PIC1_OFFSET, PIC2_OFFSET);
    unsafe {
        outb(PIC1_DATA, 0xFF);
        outb(PIC2_DATA, 0xFF);
    }
}

/// Remap the 8259 PICs to vectors 32..47 and unmask only IRQ0 (PIT) and IRQ1 (PS/2 keyboard).
pub fn init_pic_pit_keyboard() {
    remap_pic(PIC1_OFFSET, PIC2_OFFSET);
    unsafe {
        // Unmask IRQ0 (bit 0) and IRQ1 (bit 1) on master PIC; mask all on slave PIC.
        outb(PIC1_DATA, 0xFC);
        outb(PIC2_DATA, 0xFF);
    }
}

/// Unmask a specific legacy IRQ line (0..15).
pub fn unmask_pic_irq(irq: u8) {
    unsafe {
        if irq < 8 {
            let mask = inb(PIC1_DATA) & !(1 << irq);
            outb(PIC1_DATA, mask);
        } else if irq < 16 {
            let mask = inb(PIC2_DATA) & !(1 << (irq - 8));
            outb(PIC2_DATA, mask);
        }
    }
}

/// Mask a specific legacy IRQ line (0..15).
pub fn mask_pic_irq(irq: u8) {
    unsafe {
        if irq < 8 {
            let mask = inb(PIC1_DATA) | (1 << irq);
            outb(PIC1_DATA, mask);
        } else if irq < 16 {
            let mask = inb(PIC2_DATA) | (1 << (irq - 8));
            outb(PIC2_DATA, mask);
        }
    }
}

/// Send End-Of-Interrupt to the 8259 PIC for `irq` (0..15).
pub fn pic_eoi(irq: u8) {
    unsafe {
        if irq >= 8 {
            outb(PIC2_CMD, PIC_EOI);
        }
        outb(PIC1_CMD, PIC_EOI);
    }
}

// ---------------------------------------------------------------------------
// Local APIC (MMIO via IA32_APIC_BASE MSR 0x1B + HHDM)
// ---------------------------------------------------------------------------

const IA32_APIC_BASE_MSR: u32 = 0x1B;
const IA32_APIC_BASE_ENABLE: u64 = 1 << 11;
const IA32_APIC_BASE_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

pub const LAPIC_REG_ID: u32 = 0x020;
pub const LAPIC_REG_VERSION: u32 = 0x030;
pub const LAPIC_REG_TPR: u32 = 0x080;
pub const LAPIC_REG_EOI: u32 = 0x0B0;
pub const LAPIC_REG_SVR: u32 = 0x0F0;
pub const LAPIC_REG_ESR: u32 = 0x280;
pub const LAPIC_REG_LVT_TIMER: u32 = 0x320;
pub const LAPIC_REG_TIMER_INITCNT: u32 = 0x380;
pub const LAPIC_REG_TIMER_CURRCNT: u32 = 0x390;
pub const LAPIC_REG_TIMER_DIV: u32 = 0x3E0;

const LAPIC_SVR_ENABLE: u32 = 1 << 8;
const LAPIC_SPURIOUS_VECTOR: u32 = 0xFF;

static LAPIC_VIRT_BASE: AtomicU64 = AtomicU64::new(0);

#[inline]
fn lapic_virt_base() -> u64 {
    let cached = LAPIC_VIRT_BASE.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    let hhdm = crate::mm::stats().hhdm_offset;
    let msr = unsafe { rdmsr(IA32_APIC_BASE_MSR) };
    let phys = msr & IA32_APIC_BASE_ADDR_MASK;
    let virt = hhdm.wrapping_add(phys);
    LAPIC_VIRT_BASE.store(virt, Ordering::Release);
    virt
}

/// Read a 32-bit Local APIC MMIO register at `reg` offset.
#[inline]
pub fn lapic_read(reg: u32) -> u32 {
    let base = lapic_virt_base();
    if base == 0 {
        return 0;
    }
    unsafe { read_volatile((base + reg as u64) as *const u32) }
}

/// Write a 32-bit value to a Local APIC MMIO register at `reg` offset.
#[inline]
pub fn lapic_write(reg: u32, val: u32) {
    let base = lapic_virt_base();
    if base == 0 {
        return;
    }
    unsafe { write_volatile((base + reg as u64) as *mut u32, val) }
}

/// Initialize the Local APIC: enable via `IA32_APIC_BASE` MSR, set TPR=0, and enable SVR.
pub fn init_lapic() {
    unsafe {
        let mut msr = rdmsr(IA32_APIC_BASE_MSR);
        if (msr & IA32_APIC_BASE_ENABLE) == 0 {
            msr |= IA32_APIC_BASE_ENABLE;
            wrmsr(IA32_APIC_BASE_MSR, msr);
        }
        let hhdm = crate::mm::stats().hhdm_offset;
        let phys = msr & IA32_APIC_BASE_ADDR_MASK;
        LAPIC_VIRT_BASE.store(hhdm.wrapping_add(phys), Ordering::Release);
    }

    // Clear Error Status Register
    lapic_write(LAPIC_REG_ESR, 0);
    // Accept all interrupt priorities
    lapic_write(LAPIC_REG_TPR, 0);
    // Software-enable Local APIC with spurious vector 0xFF
    lapic_write(LAPIC_REG_SVR, LAPIC_SVR_ENABLE | LAPIC_SPURIOUS_VECTOR);
}

/// Signal End-Of-Interrupt to the Local APIC.
#[inline]
pub fn lapic_eoi() {
    lapic_write(LAPIC_REG_EOI, 0);
}

/// Return the current CPU's Local APIC ID.
#[inline]
pub fn lapic_id() -> u32 {
    lapic_read(LAPIC_REG_ID) >> 24
}

/// Configure the Local APIC timer in periodic mode on `vector`.
pub fn init_lapic_timer(vector: u8, initial_count: u32, divide_config: u32) {
    lapic_write(LAPIC_REG_TIMER_DIV, divide_config);
    // Bit 17 = periodic mode (0b01 << 17)
    lapic_write(LAPIC_REG_LVT_TIMER, (1 << 17) | (vector as u32));
    lapic_write(LAPIC_REG_TIMER_INITCNT, initial_count);
}

// ---------------------------------------------------------------------------
// 8254 Programmable Interval Timer (PIT) Channel 0
// ---------------------------------------------------------------------------

const PIT_CH0_DATA: u16 = 0x40;
const PIT_CMD: u16 = 0x43;
const PIT_BASE_HZ: u32 = 1_193_182;

static PIT_TICKS: AtomicU64 = AtomicU64::new(0);

/// Initialize PIT channel 0 in square-wave mode (mode 3) at `hz` Hz.
pub fn init_pit(hz: u32) {
    let target_hz = if hz == 0 { 100 } else { hz };
    let divisor = (PIT_BASE_HZ / target_hz).clamp(1, u16::MAX as u32) as u16;

    unsafe {
        // Channel 0, lobyte/hibyte, mode 3 (square wave), binary
        outb(PIT_CMD, 0x36);
        outb(PIT_CH0_DATA, (divisor & 0xFF) as u8);
        outb(PIT_CH0_DATA, ((divisor >> 8) & 0xFF) as u8);
    }
}

/// Increment and return the monotonic PIT tick counter.
#[inline]
pub fn tick() -> u64 {
    PIT_TICKS.fetch_add(1, Ordering::Relaxed) + 1
}

/// Read the current PIT tick counter.
#[inline]
pub fn ticks() -> u64 {
    PIT_TICKS.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// PS/2 Keyboard Scancode Ring Buffer (lock-free SPSC)
// ---------------------------------------------------------------------------

const PS2_DATA_PORT: u16 = 0x60;
const PS2_STATUS_PORT: u16 = 0x64;
const SCANCODE_BUF_CAP: usize = 256;

struct ScancodeRing {
    buf: [AtomicU8; SCANCODE_BUF_CAP],
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl ScancodeRing {
    const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: AtomicU8 = AtomicU8::new(0);
        Self {
            buf: [ZERO; SCANCODE_BUF_CAP],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    fn push(&self, scancode: u8) {
        let head = self.head.load(Ordering::Relaxed);
        let next = (head + 1) % SCANCODE_BUF_CAP;
        if next == self.tail.load(Ordering::Acquire) {
            // Buffer full; drop oldest or newest (drop newest here).
            return;
        }
        self.buf[head].store(scancode, Ordering::Relaxed);
        self.head.store(next, Ordering::Release);
    }

    fn pop(&self) -> Option<u8> {
        let tail = self.tail.load(Ordering::Relaxed);
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }
        let val = self.buf[tail].load(Ordering::Relaxed);
        let next = (tail + 1) % SCANCODE_BUF_CAP;
        self.tail.store(next, Ordering::Release);
        Some(val)
    }
}

static SCANCODE_RING: ScancodeRing = ScancodeRing::new();

/// Read a byte from the PS/2 controller data port (`0x60`).
#[inline]
pub fn read_ps2_data() -> u8 {
    unsafe { inb(PS2_DATA_PORT) }
}

/// Read the PS/2 controller status register (`0x64`).
#[inline]
pub fn read_ps2_status() -> u8 {
    unsafe { inb(PS2_STATUS_PORT) }
}

/// Push a raw PS/2 scancode into the kernel ring buffer.
#[inline]
pub fn push_scancode(scancode: u8) {
    SCANCODE_RING.push(scancode);
}

/// Pop the next raw PS/2 scancode from the kernel ring buffer, if any.
#[inline]
pub fn pop_scancode() -> Option<u8> {
    SCANCODE_RING.pop()
}
