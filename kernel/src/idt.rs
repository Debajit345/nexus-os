//! x86_64 Interrupt Descriptor Table (IDT) and exception handlers.

use core::arch::{asm, global_asm};
use core::mem::size_of;

use crate::gdt::{DOUBLE_FAULT_IST_INDEX, KERNEL_CS};
use crate::serial_println;

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct IdtEntry {
    offset_low: u16,
    selector: u16,
    ist: u8,
    type_attr: u8,
    offset_mid: u16,
    offset_high: u32,
    zero: u32,
}

impl IdtEntry {
    const fn missing() -> Self {
        Self {
            offset_low: 0,
            selector: 0,
            ist: 0,
            type_attr: 0,
            offset_mid: 0,
            offset_high: 0,
            zero: 0,
        }
    }

    fn set_handler(&mut self, handler: u64, ist: u8) {
        self.offset_low = handler as u16;
        self.selector = KERNEL_CS;
        self.ist = ist & 0x7;
        self.type_attr = 0x8E; // Present, DPL=0, 64-bit Interrupt Gate
        self.offset_mid = (handler >> 16) as u16;
        self.offset_high = (handler >> 32) as u32;
        self.zero = 0;
    }
}

#[repr(C, packed)]
struct IdtDescriptor {
    limit: u16,
    base: u64,
}

#[repr(C)]
pub struct InterruptFrame {
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rbp: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rbx: u64,
    pub rax: u64,
    pub vector: u64,
    pub error_code: u64,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

static mut IDT: [IdtEntry; 256] = [IdtEntry::missing(); 256];

extern "C" {
    fn isr_stub_0();
    fn isr_stub_1();
    fn isr_stub_3();
    fn isr_stub_6();
    fn isr_stub_8();
    fn isr_stub_13();
    fn isr_stub_14();
    fn isr_stub_default();
}

global_asm!(
    r#"
.macro ISR_NOERR vec
.global isr_stub_\vec
isr_stub_\vec:
    push 0
    push \vec
    jmp isr_common
.endm

.macro ISR_ERR vec
.global isr_stub_\vec
isr_stub_\vec:
    push \vec
    jmp isr_common
.endm

ISR_NOERR 0
ISR_NOERR 1
ISR_NOERR 3
ISR_NOERR 6
ISR_ERR   8
ISR_ERR   13
ISR_ERR   14

.global isr_stub_default
isr_stub_default:
    push 0
    push 255
    jmp isr_common

isr_common:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15

    mov rdi, rsp
    call isr_dispatch

    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rbp
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    add rsp, 16
    iretq
"#
);

#[no_mangle]
extern "C" fn isr_dispatch(frame: &InterruptFrame) {
    match frame.vector {
        3 => {
            serial_println!("[IDT] Breakpoint (#BP) at RIP={:#x}", frame.rip);
        }
        8 => {
            serial_println!(
                "[IDT] DOUBLE FAULT (#DF) err={:#x} RIP={:#x} RSP={:#x}",
                frame.error_code,
                frame.rip,
                frame.rsp
            );
            halt_forever();
        }
        13 => {
            serial_println!(
                "[IDT] GENERAL PROTECTION FAULT (#GP) err={:#x} RIP={:#x}",
                frame.error_code,
                frame.rip
            );
            halt_forever();
        }
        14 => {
            let cr2: u64;
            unsafe {
                asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack, preserves_flags));
            }
            serial_println!(
                "[IDT] PAGE FAULT (#PF) addr={:#x} err={:#x} RIP={:#x}",
                cr2,
                frame.error_code,
                frame.rip
            );
            halt_forever();
        }
        vec => {
            serial_println!(
                "[IDT] Exception vec={} err={:#x} RIP={:#x}",
                vec,
                frame.error_code,
                frame.rip
            );
            halt_forever();
        }
    }
}

fn halt_forever() -> ! {
    loop {
        unsafe {
            asm!("cli; hlt", options(nomem, nostack));
        }
    }
}

pub fn init() {
    unsafe {
        let default_addr = isr_stub_default as *const () as u64;
        let idt_ptr = core::ptr::addr_of_mut!(IDT);
        for i in 0..256 {
            (*idt_ptr)[i].set_handler(default_addr, 0);
        }

        IDT[0].set_handler(isr_stub_0 as *const () as u64, 0);
        IDT[1].set_handler(isr_stub_1 as *const () as u64, 0);
        IDT[3].set_handler(isr_stub_3 as *const () as u64, 0);
        IDT[6].set_handler(isr_stub_6 as *const () as u64, 0);
        IDT[8].set_handler(isr_stub_8 as *const () as u64, DOUBLE_FAULT_IST_INDEX);
        IDT[13].set_handler(isr_stub_13 as *const () as u64, 0);
        IDT[14].set_handler(isr_stub_14 as *const () as u64, 0);

        let idtr = IdtDescriptor {
            limit: (size_of::<[IdtEntry; 256]>() - 1) as u16,
            base: core::ptr::addr_of!(IDT) as u64,
        };

        asm!("lidt [{idtr}]", idtr = in(reg) &idtr, options(readonly, nostack, preserves_flags));
    }
}
