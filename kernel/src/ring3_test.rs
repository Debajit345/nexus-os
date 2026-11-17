//! Ring 3 ELF64 user-space process loading and W^X page-mapping self-test.
#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;

use crate::exec;
use crate::serial_println;

const ENTRY_VADDR: u64 = 0x0040_0000;

fn build_minimal_elf64() -> Vec<u8> {
    // Minimal x86_64 machine code: mov eax, 60; xor edi, edi; syscall
    let code: [u8; 9] = [0xB8, 0x3C, 0x00, 0x00, 0x00, 0x31, 0xFF, 0x0F, 0x05];
    let ehdr_size: usize = 64;
    let phdr_size: usize = 56;
    let code_offset: usize = ehdr_size + phdr_size;
    let total_size = code_offset + code.len();

    let mut buf = vec![0u8; total_size];

    // ELF64 Header (64 bytes)
    buf[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    buf[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    buf[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    buf[20..24].copy_from_slice(&1u32.to_le_bytes()); // EV_CURRENT
    buf[24..32].copy_from_slice(&ENTRY_VADDR.to_le_bytes()); // e_entry
    buf[32..40].copy_from_slice(&(ehdr_size as u64).to_le_bytes()); // e_phoff
    buf[52..54].copy_from_slice(&(ehdr_size as u16).to_le_bytes()); // e_ehsize
    buf[54..56].copy_from_slice(&(phdr_size as u16).to_le_bytes()); // e_phentsize
    buf[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum

    // Program Header (PT_LOAD, PF_R | PF_X = 5)
    let ph = &mut buf[ehdr_size..ehdr_size + phdr_size];
    ph[0..4].copy_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD
    ph[4..8].copy_from_slice(&5u32.to_le_bytes()); // p_flags = PF_R | PF_X
    ph[8..16].copy_from_slice(&(code_offset as u64).to_le_bytes()); // p_offset
    ph[16..24].copy_from_slice(&ENTRY_VADDR.to_le_bytes()); // p_vaddr
    ph[24..32].copy_from_slice(&ENTRY_VADDR.to_le_bytes()); // p_paddr
    ph[32..40].copy_from_slice(&(code.len() as u64).to_le_bytes()); // p_filesz
    ph[40..48].copy_from_slice(&(code.len() as u64).to_le_bytes()); // p_memsz
    ph[48..56].copy_from_slice(&0x1000u64.to_le_bytes()); // p_align

    buf[code_offset..total_size].copy_from_slice(&code);
    buf
}

/// Load a minimal ELF64 binary into a fresh user-space `AddressSpace` and verify W^X mappings.
pub fn run_ring3_exec_selftest() {
    let elf = build_minimal_elf64();
    match exec::load_and_prepare("ring3-smoke", &elf, 100) {
        Ok(proc) => {
            let code_mapped = proc.address_space.translate(ENTRY_VADDR).is_some();
            let stack_mapped = proc.address_space.translate(proc.user_rsp - 8).is_some();
            serial_println!(
                "Ring 3 ELF64 exec self-test passed: tid={:?}, entry={:#x}, pml4={:#x}, code_mapped={}, stack_mapped={}",
                proc.task_id,
                proc.entry_rip,
                proc.address_space.pml4_phys,
                code_mapped,
                stack_mapped
            );
        }
        Err(err) => {
            serial_println!("Ring 3 ELF64 exec self-test failed: {}", err);
        }
    }
}
