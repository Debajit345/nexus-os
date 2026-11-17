//! x86_64 Global Descriptor Table (GDT) and Task State Segment (TSS).

use core::arch::asm;
use core::mem::size_of;

pub const KERNEL_CS: u16 = 0x08;
pub const KERNEL_DS: u16 = 0x10;
#[allow(dead_code)]
pub const USER_DS: u16 = 0x18 | 3;
#[allow(dead_code)]
pub const USER_CS: u16 = 0x20 | 3;
pub const TSS_SEL: u16 = 0x28;

pub const DOUBLE_FAULT_IST_INDEX: u8 = 1;

#[repr(C, packed)]
struct Tss {
    reserved1: u32,
    rsp: [u64; 3],
    reserved2: u64,
    ist: [u64; 7],
    reserved3: u64,
    reserved4: u16,
    iomap_base: u16,
}

impl Tss {
    const fn new() -> Self {
        Self {
            reserved1: 0,
            rsp: [0; 3],
            reserved2: 0,
            ist: [0; 7],
            reserved3: 0,
            reserved4: 0,
            iomap_base: size_of::<Tss>() as u16,
        }
    }
}

#[repr(C, packed)]
struct GdtDescriptor {
    limit: u16,
    base: u64,
}

#[repr(C, align(16))]
struct GdtTable {
    entries: [u64; 7],
}

const DF_STACK_SIZE: usize = 16384;
static mut DF_STACK: [u8; DF_STACK_SIZE] = [0; DF_STACK_SIZE];
static mut TSS: Tss = Tss::new();
static mut GDT: GdtTable = GdtTable {
    entries: [
        0x0000_0000_0000_0000, // 0x00: Null
        0x00AF_9A00_0000_FFFF, // 0x08: Kernel Code (64-bit)
        0x00CF_9200_0000_FFFF, // 0x10: Kernel Data
        0x00CF_F200_0000_FFFF, // 0x18: User Data
        0x00AF_FA00_0000_FFFF, // 0x20: User Code (64-bit)
        0,                     // 0x28: TSS low
        0,                     // 0x30: TSS high
    ],
};

pub fn init() {
    unsafe {
        let df_stack_top = core::ptr::addr_of!(DF_STACK) as u64 + DF_STACK_SIZE as u64;
        TSS.ist[0] = df_stack_top;

        let tss_base = core::ptr::addr_of!(TSS) as u64;
        let tss_limit = (size_of::<Tss>() - 1) as u64;

        let low = (tss_limit & 0xFFFF)
            | ((tss_base & 0xFF_FFFF) << 16)
            | (0x89u64 << 40)
            | (((tss_limit >> 16) & 0xF) << 48)
            | (((tss_base >> 24) & 0xFF) << 56);
        let high = tss_base >> 32;

        GDT.entries[5] = low;
        GDT.entries[6] = high;

        let gdtr = GdtDescriptor {
            limit: (size_of::<GdtTable>() - 1) as u16,
            base: core::ptr::addr_of!(GDT) as u64,
        };

        asm!(
            "lgdt [{gdtr}]",
            "push {kcs}",
            "lea rax, [rip + 2f]",
            "push rax",
            "retfq",
            "2:",
            "mov ds, {kds:x}",
            "mov es, {kds:x}",
            "mov ss, {kds:x}",
            "xor eax, eax",
            "mov fs, ax",
            "mov gs, ax",
            "ltr {tss:x}",
            gdtr = in(reg) &gdtr,
            kcs = const KERNEL_CS as u64,
            kds = in(reg) KERNEL_DS,
            tss = in(reg) TSS_SEL,
            out("rax") _,
        );
    }
}
