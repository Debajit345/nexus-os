#![allow(dead_code)]

//! x86_64 Ring 3 user-mode transition (`iretq`) and `syscall`/`sysret` support.

use core::arch::{asm, global_asm};

use nexus_abi::Personality;

use crate::gdt::{KERNEL_CS, KERNEL_DS, USER_CS, USER_DS};

const IA32_EFER: u32 = 0xC000_0080;
const IA32_STAR: u32 = 0xC000_0081;
const IA32_LSTAR: u32 = 0xC000_0082;
const IA32_FMASK: u32 = 0xC000_0084;
const IA32_KERNEL_GS_BASE: u32 = 0xC000_0102;

const EFER_SCE: u64 = 1 << 0;
const RFLAGS_TF: u64 = 1 << 8;
const RFLAGS_IF: u64 = 1 << 9;
const RFLAGS_DF: u64 = 1 << 10;
const USER_RFLAGS_DEFAULT: u64 = RFLAGS_IF | (1 << 1);

const SYSCALL_STACK_SIZE: usize = 16384;

#[repr(C, align(16))]
struct SyscallStack([u8; SYSCALL_STACK_SIZE]);

static mut SYSCALL_STACK: SyscallStack = SyscallStack([0; SYSCALL_STACK_SIZE]);

#[repr(C)]
struct PerCpuSyscall {
    kernel_rsp: u64,
    user_rsp: u64,
}

static mut PER_CPU_SYSCALL: PerCpuSyscall = PerCpuSyscall {
    kernel_rsp: 0,
    user_rsp: 0,
};

/// Saved register state pushed by `syscall_entry`.
#[repr(C)]
pub struct SyscallRegs {
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub rbp: u64,
    pub rbx: u64,
    pub r9: u64,
    pub r8: u64,
    pub r10: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rax: u64,
    pub rcx: u64,
    pub r11: u64,
    pub rsp: u64,
}

/// User-mode execution context for entering Ring 3 via `iretq`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UserContext {
    pub rip: u64,
    pub rsp: u64,
    pub rflags: u64,
    pub cs: u16,
    pub ss: u16,
    pub personality: Personality,
}

impl UserContext {
    pub const fn new(entry_rip: u64, user_rsp: u64) -> Self {
        Self {
            rip: entry_rip,
            rsp: user_rsp,
            rflags: USER_RFLAGS_DEFAULT,
            cs: USER_CS,
            ss: USER_DS,
            personality: Personality::Native,
        }
    }

    pub const fn with_personality(mut self, personality: Personality) -> Self {
        self.personality = personality;
        self
    }

    pub unsafe fn enter(&self) -> ! {
        enter_usermode_iretq(self.rip, self.rsp)
    }
}

extern "C" {
    fn syscall_entry();
}

global_asm!(
    r#"
.global syscall_entry
syscall_entry:
    swapgs
    mov gs:[8], rsp
    mov rsp, gs:[0]

    push qword ptr gs:[8]
    push r11
    push rcx
    push rax
    push rdi
    push rsi
    push rdx
    push r10
    push r8
    push r9
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    mov rdi, rsp
    call syscall_rust_handler

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    pop r9
    pop r8
    pop r10
    pop rdx
    pop rsi
    pop rdi
    add rsp, 8
    pop rcx
    pop r11
    pop rsp
    swapgs
    sysretq
"#
);

#[no_mangle]
extern "C" fn syscall_rust_handler(regs: &SyscallRegs) -> i64 {
    let args = [regs.rdi, regs.rsi, regs.rdx, regs.r10, regs.r8, regs.r9];
    crate::syscall::dispatch_raw(Personality::Native, regs.rax, args)
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
unsafe fn wrmsr(msr: u32, value: u64) {
    let low = value as u32;
    let high = (value >> 32) as u32;
    asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") low,
        in("edx") high,
        options(nomem, nostack, preserves_flags)
    );
}

/// Enable `IA32_EFER.SCE` and program `IA32_STAR`, `IA32_LSTAR`, `IA32_FMASK`,
/// and `IA32_KERNEL_GS_BASE` for Ring 3 `syscall`/`sysretq`.
pub fn init_syscall_msrs() {
    // Verify GDT selector layout required by sysretq:
    // STAR[63:48] + 8  | 3 == USER_DS (0x1B)
    // STAR[63:48] + 16 | 3 == USER_CS (0x23)
    const _: () = assert!(KERNEL_DS == KERNEL_CS + 8);
    const _: () = assert!(USER_DS == (USER_CS - 8));

    unsafe {
        let stack_top = core::ptr::addr_of!(SYSCALL_STACK) as u64 + SYSCALL_STACK_SIZE as u64;
        let per_cpu_ptr = core::ptr::addr_of_mut!(PER_CPU_SYSCALL);
        (*per_cpu_ptr).kernel_rsp = stack_top;
        (*per_cpu_ptr).user_rsp = 0;

        wrmsr(IA32_KERNEL_GS_BASE, per_cpu_ptr as u64);

        let efer = rdmsr(IA32_EFER);
        wrmsr(IA32_EFER, efer | EFER_SCE);

        let star = (((USER_CS as u64) - 16) << 48) | ((KERNEL_CS as u64) << 32);
        wrmsr(IA32_STAR, star);

        wrmsr(IA32_LSTAR, syscall_entry as *const () as u64);
        wrmsr(IA32_FMASK, RFLAGS_TF | RFLAGS_IF | RFLAGS_DF);
    }
}

/// Transition to Ring 3 at `entry_rip` with stack `user_rsp` using `iretq`.
pub unsafe fn enter_usermode_iretq(entry_rip: u64, user_rsp: u64) -> ! {
    asm!(
        "cli",
        "mov ds, {uds:x}",
        "mov es, {uds:x}",
        "push {uss}",
        "push {ursp}",
        "push {urflags}",
        "push {ucs}",
        "push {urip}",
        "iretq",
        uds = in(reg) USER_DS,
        uss = in(reg) USER_DS as u64,
        ursp = in(reg) user_rsp,
        urflags = in(reg) USER_RFLAGS_DEFAULT,
        ucs = in(reg) USER_CS as u64,
        urip = in(reg) entry_rip,
        options(noreturn)
    );
}
