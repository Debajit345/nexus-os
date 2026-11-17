# OS Feature Matrix: macOS, Linux, and Windows in Nexus OS

Nexus OS is a clean-room, `x86_64` hybrid microkernel operating system written in Rust. Rather than cloning any single legacy OS, Nexus OS systematically evaluates the core-to-desktop architecture of **macOS (XNU/Darwin/Aqua)**, **Linux (Kernel/POSIX/DRM/Wayland)**, and **Windows (NT/Win32/WDDM/DirectX)**, keeping their strongest primitives, eliminating their historical liabilities, and unifying them into a memory-safe, capability-based design.

---

## 1. macOS (XNU / Darwin / Aqua)

### What is Good (Keep & Unify)
- **Mach Capability IPC**: Port-based message passing with explicit `Send`, `SendOnce`, and `Receive` rights, plus Out-of-Line (OOL) virtual memory descriptors for zero-copy large transfers.
- **Unified Desktop UX (Aqua)**: Global top menu bar, dock, consistent traffic-light window controls, and a compositor-first display pipeline where every window is a GPU-backed surface.
- **Clean Graphics & Audio Abstractions**: Metal-style explicit command submission and low-latency audio/media pipelines without legacy 2D GDI/X11 baggage.
- **Mach-O 64-bit Binary Format**: Clean load-command structure (`LC_SEGMENT_64`, `LC_MAIN`, `LC_LOAD_DYLIB`) for structured executable and dynamic library loading.
- **Structured Sandboxing**: Capability-scoped access to hardware, files, and IPC endpoints rather than ambient root authority.

### What is Bad (Eliminate)
- **Apple Hardware Lock-In**: Tight coupling to proprietary Apple Silicon / T2 / SMC firmware and closed-source drivers.
- **Closed-Source Kernel & Compositor**: Opaque WindowServer and XNU internals that prevent community auditing and clean-room driver development.
- **Objective-C / Swift Runtime Coupling**: System frameworks and I/O Kit C++ ABI fragility baked into core OS interfaces.
- **Opaque Entitlements & TCC UX**: Confusing permission prompts and undocumented private entitlements.
- **Legacy HFS+ / Case-Insensitivity Quirks**: Filesystem edge cases that break POSIX tooling expectations.

### Integration into Nexus OS
| macOS Feature | Nexus OS Integration | Kernel Modules & Crates |
| :--- | :--- | :--- |
| Mach Ports & OOL IPC | Zero-copy capability channels/ports with transferable rights | `kernel/src/ipc.rs`, `kernel/src/cap.rs`, `crates/nexus-abi` |
| Aqua Desktop Chrome | Global menu bar, dock, traffic-light window frames, ARGB8888 surfaces | `crates/nexus-gfx` (`chrome`, `surface`, `color`, `geometry`) |
| Metal-Style GPU Pipeline | Clean-room DRM/GEM/KMS + explicit GPU command submission traits | `kernel/src/drm.rs`, `crates/nexus-gfx` (`gpu`) |
| Mach-O 64-bit Binaries | Bounds-checked, `#![forbid(unsafe_code)]` Mach-O parser & Darwin personality | `crates/nexus-macho`, `crates/nexus-compat`, `kernel/src/syscall.rs` |
| Sandboxing & Isolation | Per-process capability tables + VT-d/AMD-Vi IOMMU DMA remapping | `kernel/src/cap.rs`, `kernel/src/iommu.rs` |

---

## 2. Linux (Kernel / POSIX / DRM / Wayland)

### What is Good (Keep & Unify)
- **Virtual File System (VFS) & POSIX Semantics**: Hierarchical `/` namespace, `inode`/`dentry`/`superblock` abstractions, mount points, and standard POSIX file/process ergonomics.
- **`io_uring` Asynchronous I/O**: Shared-memory submission and completion rings that eliminate syscall overhead for high-throughput I/O.
- **DRM / KMS / GEM / Vulkan-First Graphics**: Kernel mode-setting, GEM buffer objects, explicit sync fences, and direct userspace Vulkan/Mesa rendering.
- **Wayland Compositor Model**: Every frame is perfect, zero-tearing buffer swaps, and strict client isolation (clients cannot snoop on other clients' input or framebuffers).
- **Modern Hardware & Storage Stack**: PCIe ECAM enumeration, ACPI table parsing (MADT/MCFG), NVMe multi-queue storage, xHCI USB 3.x, SMP, and IOMMU isolation.
- **ELF64 & POSIX Shell Tooling**: Clean ELF64 program headers and composable POSIX shell/coreutils workflows.

### What is Bad (Eliminate)
- **C Memory Unsafety**: Buffer overflows, use-after-free, and data races across kernel subsystems and third-party C drivers.
- **Unstable Internal Driver ABI**: Out-of-tree drivers break across kernel releases, forcing monolithic coupling.
- **Monolithic Blast Radius**: A bug in a single filesystem, network, or GPU driver panics the entire kernel.
- **Legacy X11 & `ioctl` Proliferation**: Decades of untyped `ioctl` multiplexers and X11 security/tearing baggage.
- **`fork()` Complexity**: Heavyweight copy-on-write process cloning semantics that complicate multithreaded runtimes and capability inheritance.

### Integration into Nexus OS
| Linux Feature | Nexus OS Integration | Kernel Modules & Crates |
| :--- | :--- | :--- |
| VFS & RamFs | Clean-room VFS with mountable filesystems and POSIX path resolution | `kernel/src/vfs.rs`, `crates/nexus-shell` (`fs`) |
| `io_uring` Completion Rings | Unified async completion queues paired with Mach-style IPC | `kernel/src/ipc.rs`, `crates/nexus-abi` |
| DRM / KMS / GEM | Kernel DRM subsystem with GEM buffers, KMS modesetting, and PCI GPU probing | `kernel/src/drm.rs`, `crates/nexus-gfx` (`gpu`) |
| ELF64 & Linux Syscalls | `#![forbid(unsafe_code)]` ELF64 validator + Linux personality syscall dispatch | `crates/nexus-elf`, `crates/nexus-compat`, `kernel/src/syscall.rs` |
| Hardware Bring-up & Drivers | Rust-safe `x86_64` HAL, ACPI, PCIe, NVMe, VirtIO-Blk, xHCI, VirtIO-Net, SMP | `kernel/src/{cpu,acpi,pci,mm,vmm,smp,blk,nvme,xhci,net,power}.rs` |
| POSIX Shell & Coreutils | `no_std` POSIX shell parser, builtins (`uname`, `free`, `lscpu`, `lspci`), and VFS | `crates/nexus-shell` |

---

## 3. Windows (NT / Win32 / WDDM / DirectX)

### What is Good (Keep & Unify)
- **NT Object Manager & Handle Tables**: Every kernel resource (files, ports, threads, VMOs, completion ports) is a typed, reference-counted object accessed via per-process handles with explicit access rights.
- **I/O Completion Ports (IOCP) & Async IRPs**: Scalable, non-blocking I/O completion queues designed from day one for thread pools and event loops.
- **WDDM & DirectX 12 Explicit GPU Model**: Hardware-scheduled GPU command queues, explicit memory residency, and fine-grained synchronization fences.
- **PE32+ Binary Format**: Structured COFF/PE32+ headers, section tables, and import/export directories for Windows application compatibility.
- **ACPI Power & Performance Profiles**: Structured S-states, processor performance states (Intel HWP / AMD CPPC), and thermal/power policies.

### What is Bad (Eliminate)
- **Windows Registry**: Global, opaque, mutable binary database that creates configuration rot and brittle application state.
- **Legacy 16-bit / 32-bit Win32 & GDI Cruft**: Decades of backward-compatibility shims, `win32k.sys` kernel-mode GUI attack surface, and inconsistent APIs.
- **C/C++ Kernel & Driver Unsafety**: Memory corruption vulnerabilities in kernel-mode drivers and legacy subsystems.
- **Drive-Letter Path Fragmentation**: `C:\`, `D:\`, and UNC path inconsistencies instead of a single unified VFS tree.
- **Closed-Source Kernel & Heavy COM Boilerplate**: Proprietary internals and verbose ABI ceremony.

### Integration into Nexus OS
| Windows Feature | Nexus OS Integration | Kernel Modules & Crates |
| :--- | :--- | :--- |
| Object Manager & Handles | Generation-counted `CapHandle` (slot + generation + flags) and `Rights` masks | `kernel/src/cap.rs`, `crates/nexus-abi` |
| IOCP Async Completion | Unified async completion queues integrated with IPC ports | `kernel/src/ipc.rs`, `crates/nexus-abi` |
| PE32+ Binaries & Win32 ABI | `#![forbid(unsafe_code)]` PE32+ validator + Windows NT personality dispatch | `crates/nexus-pe`, `crates/nexus-compat`, `kernel/src/syscall.rs` |
| Explicit GPU & Power | DRM/GEM command submission + ACPI S-states, HWP/CPPC, and power profiles | `kernel/src/drm.rs`, `kernel/src/power.rs` |

---

## 4. Unified Nexus OS Architecture Matrix

The table below maps every layer of Nexus OS to its exact implementation in `kernel/src/` and `crates/`:

| System Layer | Unified Design Choice | `nexus-kernel` Modules | Workspace Crates |
| :--- | :--- | :--- | :--- |
| **Boot & `x86_64` HAL** | Limine v2 boot, GDT/TSS, IDT, APIC/x2APIC, CPUID/HWP/CPPC, SMP | `limine`, `gdt`, `idt`, `apic`, `cpu`, `smp`, `serial`, `console` | `nexus-abi` |
| **Memory & Isolation** | Physical frame + heap allocator, 4-level paging, VT-d/AMD-Vi IOMMU | `mm`, `vmm`, `iommu` | `nexus-abi` |
| **Capabilities & Objects** | NT-style typed objects + generation-checked capability handles | `cap` | `nexus-abi` |
| **IPC & Async I/O** | Mach-style zero-copy ports + NT IOCP / Linux `io_uring` completion queues | `ipc` | `nexus-abi` |
| **Scheduling & Usermode** | Preemptive priority + round-robin scheduler, Ring 3 `syscall`/`sysret` | `sched`, `usermode`, `syscall` | `nexus-abi` |
| **Storage & VFS** | Linux-style VFS/RamFs, GPT partitions, VirtIO-Blk, NVMe PCIe | `vfs`, `blk`, `nvme` | `nexus-shell` |
| **Bus, USB, Net & Power** | ACPI (MADT/MCFG), PCIe ECAM, xHCI USB 3.x, VirtIO-Net, ACPI power | `acpi`, `pci`, `xhci`, `net`, `power` | `nexus-shell` |
| **Graphics & Desktop UX** | DRM/KMS/GEM + Aqua-inspired menu bar, dock, and window chrome | `drm`, `console` | `nexus-gfx` |
| **Multi-Personality ABI** | Native, POSIX/Linux (ELF64), Windows NT (PE32+), Darwin/XNU (Mach-O) | `syscall`, `usermode` | `nexus-elf`, `nexus-pe`, `nexus-macho`, `nexus-compat` |
| **Shell & Coreutils** | Clean-room `no_std` POSIX shell, pipelines, redirects, and sysinfo builtins | `vfs`, `console`, `serial` | `nexus-shell` |
