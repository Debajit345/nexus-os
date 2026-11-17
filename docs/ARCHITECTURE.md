# Nexus OS Architecture

Nexus OS is a clean-room, `x86_64` hybrid operating system written in Rust. Its
design synthesizes proven architectural primitives from three OS families into a
single, memory-safe kernel and userspace platform, paired with a GPU/Vulkan-first
graphics roadmap. For a detailed breakdown of what Nexus OS keeps and eliminates
from macOS, Linux, and Windows, see the [OS Feature Matrix](OS_FEATURE_MATRIX.md).

---

## 1. Design Pillars

| Pillar | Inspiration | Role in Nexus OS |
| :--- | :--- | :--- |
| **Unified Object & Async I/O** | Windows NT | Kernel object namespace, per-process handle tables with access rights, IRP-style async I/O packets, and I/O Completion Ports (IOCP). |
| **Capability IPC & UX** | macOS / Mach | Port-based message passing, send/receive rights, out-of-line memory descriptors, and a low-latency compositor-driven desktop UX. |
| **POSIX & VFS** | Linux | Hierarchical Virtual File System (VFS), mount namespaces, file descriptors mapped onto kernel handles, and POSIX process/signal semantics. |
| **GPU / Vulkan-First** | Modern Graphics | Direct-to-GPU memory management, explicit synchronization primitives, zero-copy surface sharing, and a Vulkan-first compositor. |

---

## 2. High-Level System Layers

```text
+-----------------------------------------------------------------------+
|                         Userspace Personalities                       |
|   [ POSIX / Linux ABI ]   [ Win32 / NT ABI ]   [ Mach / Native UX ]   |
+-----------------------------------------------------------------------+
|                     System Services & Compositor                      |
|   [ VFS Server ]   [ Vulkan Compositor ]   [ Device / Driver Hosts ]  |
+-----------------------------------------------------------------------+
|                        Nexus Hybrid Kernel (Rust)                     |
|  +------------------+  +-------------------+  +--------------------+  |
|  |  Object Manager  |  |   Mach-Style IPC  |  |   Async I/O (IOCP) |  |
|  +------------------+  +-------------------+  +--------------------+  |
|  +------------------+  +-------------------+  +--------------------+  |
|  |  VFS Core / Vnode|  |  Memory & Paging  |  |  Preemptive Sched  |  |
|  +------------------+  +-------------------+  +--------------------+  |
|  +-----------------------------------------------------------------+  |
|  |          x86_64 HAL (APIC, IDT/GDT, Syscall, PCIe, IOMMU)       |  |
|  +-----------------------------------------------------------------+  |
+-----------------------------------------------------------------------+
```

---

## 3. Core Kernel Subsystems

### 3.1 Hardware Abstraction & Boot (`x86_64`)
- **Boot**: UEFI / Limine-compatible `x86_64-unknown-none` entry point with
  higher-half kernel mapping.
- **Memory Management**: 4-level (and optional 5-level) paging, physical frame
  allocator, slab/buddy kernel heap, and copy-on-write virtual memory objects
  (VMOs).
- **Interrupts & Syscalls**: APIC/x2APIC timer and IPI routing, `SYSCALL`/`SYSRET`
  fast path, and per-CPU kernel stacks.

### 3.2 NT-Style Object Manager & Handle Tables
- Every kernel resource (files, ports, threads, processes, VMOs, completion
  ports, GPU fences) is a reference-counted **Kernel Object** with a uniform
  header and type descriptor.
- Processes interact with objects exclusively through **Handles** stored in a
  per-process handle table, each carrying an explicit **Access Mask** (read,
  write, execute, duplicate, transfer).
- An internal hierarchical object namespace (`/Device`, `/ObjectTypes`,
  `/Sessions`) coexists cleanly with the POSIX VFS mount tree.

### 3.3 Asynchronous I/O & Completion Ports
- I/O operations are modeled as **I/O Request Packets (IRPs)** that flow
  asynchronously through layered drivers without blocking kernel threads.
- **I/O Completion Ports (IOCP)** allow userspace event loops and thread pools
  to dequeue completed I/O, timers, and IPC notifications in a single scalable
  wait call.
- Synchronous POSIX `read`/`write` calls are implemented as thin wrappers over
  the async I/O pipeline.

### 3.4 Mach-Style Capability IPC
- **Ports** are unidirectional, kernel-protected message queues.
- **Port Rights** (`Send`, `SendOnce`, `Receive`) are capabilities that can be
  transferred across processes via IPC messages, enabling fine-grained
  sandboxing and service discovery.
- Large payloads use **Out-of-Line (OOL)** virtual memory descriptors for
  zero-copy or copy-on-write transfer between address spaces.

### 3.5 Linux-Style VFS & POSIX Layer
- A clean-room **VFS** provides `Inode`/`Vnode`, `Dentry` cache, `Superblock`,
  and mount-point resolution.
- POSIX file descriptors (`fd`) map directly to kernel handles, allowing
  seamless interoperability between POSIX `epoll`/`poll` and NT-style IOCP.

---

## 4. GPU / Vulkan-First Roadmap

Rather than treating graphics as a legacy 2D framebuffer with 3D bolted on,
Nexus OS is architected around modern GPU pipelines:

1. **Phase 1 (Bring-up)**: UEFI GOP / linear framebuffer console, serial
   logging, and basic software blitting for early kernel debugging.
2. **Phase 2 (DRM/KMS-style Core & VirtIO-GPU)**: Kernel mode-setting,
   DMA-BUF-style shared memory objects (VMOs), and explicit sync fences
   integrated with IOCP.
3. **Phase 3 (Vulkan-First Compositor)**: A userspace compositor speaking
   directly to GPU command queues, supporting zero-copy client surfaces,
   damage tracking, and hardware-accelerated window effects.
4. **Phase 4 (Native & Compatibility Graphics)**: Clean-room Vulkan ICD
   interface, Mesa/Gallium-compatible userspace driver hosting, and translation
   layers for POSIX/Wayland and Win32/DXVK-style workloads.

---

## 5. Implemented Kernel Modules & Workspace Crates

To maximize development velocity and reliability, core subsystems live in
platform-independent `crates/*` packages that compile and run unit tests on the
host (`cargo test`), while `kernel/` (`nexus-kernel`) binds them to bare-metal
`x86_64` hardware.

### `nexus-kernel` Modules (`kernel/src/`)
- **Boot & CPU HAL**: `limine` (Base Revision 2), `gdt` (GDT/TSS), `idt` (exceptions/interrupts), `apic` (8259 PIC, Local APIC, PIT, PS/2), `cpu` (CPUID, AVX2/AVX-512, SMEP/SMAP, per-CPU GS), `smp` (AP bootstrap), `time` (RTC/TSC timekeeping), `serial` (COM1 UART), `console` (linear framebuffer 8x8 text console).
- **Memory & Isolation**: `mm` (physical frame allocator & kernel heap), `vmm` (4-level PML4 paging), `iommu` (Intel VT-d / AMD-Vi DMA remapping & sandbox policy).
- **Capabilities, IPC, Execution & Scheduling**: `cap` (generation-checked capability table), `ipc` (Mach-style zero-copy ports + NT/io_uring async completion queues), `sched` (preemptive priority + round-robin scheduler), `syscall` (multi-personality dispatcher), `usermode` (Ring 3 `iretq` & `syscall`/`sysret`), `exec` (multi-OS ELF64/PE32+/Mach-O loader), `ksh` (kernel recovery shell), `desktop_boot` (early-boot `nexus-init` self-test & live `nexus-desktop` preview).
- **Storage, Bus, USB, Audio, Net & Power**: `acpi` (RSDP/RSDT/XSDT/MADT/MCFG), `pci` (PCIe ECAM & legacy config space), `vfs` (VFS & RamFs), `blk` (block abstraction, GPT, VirtIO-Blk), `nvme` (NVMe PCIe controller), `xhci` (USB 3.x host controller), `audio` (Intel HDA audio controller), `net` (Ethernet/ARP/IPv4/UDP/TCP & VirtIO-Net), `power` (ACPI S-states, Intel HWP / AMD CPPC, profiles, reboot/shutdown).
- **Graphics**: `drm` (clean-room GEM buffers, KMS modesetting, PCI GPU probing).

### Workspace Crates (`crates/`)
- `nexus-abi`: Shared `no_std` ABI types (`CapHandle`, `ObjectType`, `Rights`, `Personality`, `SyscallNumber`, `AbiError`).
- `nexus-elf`: Clean-room `no_std` (`#![forbid(unsafe_code)]`) ELF64 parser and validator for Linux binaries.
- `nexus-pe`: Clean-room `no_std` (`#![forbid(unsafe_code)]`) PE32+ parser and validator for Windows binaries.
- `nexus-macho`: Clean-room `no_std` (`#![forbid(unsafe_code)]`) Mach-O 64-bit parser and validator for macOS binaries.
- `nexus-compat`: Unified ELF64/PE32+/Mach-O inspection and multi-personality syscall translation.
- `nexus-gfx`: 2D compositor primitives (`Surface`, `Color`, `Rect`), Aqua-inspired desktop chrome (`MenuBar`, `Dock`, `WindowFrame`), and GPU/DRM command submission traits.
- `nexus-shell`: Clean-room `no_std` POSIX shell, command parser, in-memory VFS (`MemShellFs`), and system builtins (`uname`, `free`, `lscpu`, `lspci`).
- `nexus-desktop`: Clean-room `no_std` (`#![forbid(unsafe_code)]`) macOS-inspired desktop compositor, window manager, light/dark themes, and integrated terminal.
- `nexus-init`: Clean-room `no_std` (`#![forbid(unsafe_code)]`) init system, declarative `/etc/nexus.conf` parser, and dependency-aware service manager.
- `nexus-pkg`: Clean-room `no_std` (`#![forbid(unsafe_code)]`) multi-OS package manifest, dependency resolver, and registry (native, Linux ELF, Windows PE, macOS `.app`).
- `nexus-gpu-bridge`: Clean-room `no_std` (`#![forbid(unsafe_code)]`) GPU API translation bridge (Direct3D 11/12, Metal, OpenGL, and Vulkan).
- `nexus-fs`: Clean-room `no_std` filesystem implementations (FAT32 and NexusFS).
