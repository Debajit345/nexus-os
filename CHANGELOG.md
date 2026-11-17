# Changelog

All notable changes to **Nexus OS** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-18

Full-stack `v0.2.0` milestone covering Stages 11 through 20 across 21 workspace crates and `nexus-kernel`.

### Added

- **Stage 11 — Kernel FS Mount, Ring 3 Exec Smoke Test & POSIX Shim**: `kernel/src/fs_mount.rs`, `kernel/src/ring3_test.rs`, and `crates/nexus-libc`.
- **Stage 12 — Windows NT/Win32 Runtime Shim**: `crates/nexus-win32` (`NtStatus`, `WinHandle`, `Win32Runtime`).
- **Stage 13 — macOS Darwin/Mach Runtime Shim**: `crates/nexus-darwin` (`KernReturn`, `MachPort`, `DarwinRuntime`).
- **Stage 14 — Audio DSP Mixer**: `crates/nexus-audio` (`AudioMixer`, `MixerChannel`, soft-clipping & stereo pan).
- **Stage 15 — HID Input & Shortcut Router**: `crates/nexus-input` (`InputQueue`, `match_shortcut` for macOS/Linux/Windows).
- **Stage 16 — User-Space Socket, DNS & DHCP Stack**: `crates/nexus-net-stack` (`SocketTable`, `DnsQuery`, `DhcpDiscover`).
- **Stage 17 — Capability Sandbox Policy Engine**: `crates/nexus-sandbox` (`AppPolicy`, `SandboxEnforcer`).
- **Stage 18 — Hypervisor & Container Isolation**: `crates/nexus-vm` (`IsolationManager`, `VmConfig`, `ContainerSpec`).
- **Stage 19 — Built-in Core Apps**: `crates/nexus-apps` (`FinderApp`, `ActivityMonitorApp`, `SettingsApp`).
- **Stage 20 — Full Workspace Integration & `v0.2.0` Release**: 21 workspace crates + `nexus-kernel`, 89/89 unit tests, 0-warning `clippy -D warnings`, and QEMU Boot CI.

## [0.1.0] - 2026-11-18

Initial `v0.1.0` milestone release covering Phases 1 through 10 of the clean-room
`x86_64` hybrid operating system.

### Added

- **Phase 1 — Clean-Room Kernel Foundation & ABI**:
  - Bare-metal `x86_64-unknown-none` kernel (`nexus-kernel`) with Limine Base Revision 2 boot, GDT/TSS, IDT exception/interrupt handlers, COM1 serial UART, linear framebuffer 8x8 text console, and generation-checked capability table (`cap`).
  - Shared `no_std` ABI crate (`nexus-abi`) defining `CapHandle`, `ObjectType`, `Rights`, `Personality`, `SyscallNumber`, and `AbiError`.
  - Clean-room legal policy (`docs/LEGAL_CLEAN_ROOM.md`), architecture specification (`docs/ARCHITECTURE.md`), and OS feature matrix (`docs/OS_FEATURE_MATRIX.md`).
- **Phase 2 — Memory, VFS, Multi-OS Binary Loaders & 2D Graphics**:
  - Kernel physical frame allocator and heap (`mm`), VFS and RamFs (`vfs`), and multi-personality syscall dispatcher (`syscall`).
  - Bounds-checked, `#![forbid(unsafe_code)]` binary parsers and validators for Linux ELF64 (`nexus-elf`), Windows PE32+ (`nexus-pe`), and macOS Mach-O 64-bit (`nexus-macho`).
  - 2D compositor primitives, Aqua-inspired desktop chrome, and GPU/DRM traits (`nexus-gfx`).
- **Phase 3 — Paging, Preemptive Scheduling, IPC & Interrupts**:
  - 4-level PML4 virtual memory manager (`vmm`), preemptive priority + round-robin scheduler (`sched`), Mach-style zero-copy ports and NT/io_uring async completion queues (`ipc`), and 8259 PIC / Local APIC / PIT / PS/2 support (`apic`).
- **Phase 4 — Intel/Ryzen Hardware Support, ACPI, PCIe & SMP**:
  - Intel and AMD Ryzen CPUID feature detection, AVX2/AVX-512, SMEP/SMAP, and per-CPU GS (`cpu`).
  - ACPI RSDP/RSDT/XSDT/MADT/MCFG parsing (`acpi`), PCIe ECAM and legacy config space enumeration (`pci`), AP bootstrap (`smp`), and Ring 3 `iretq` / `syscall`/`sysret` usermode transitions (`usermode`).
- **Phase 5 — Storage, USB 3.x, DRM GPU & IOMMU Isolation**:
  - Block device abstraction, GPT partitioning, and VirtIO-Blk (`blk`), NVMe PCIe controller (`nvme`), xHCI USB 3.x host controller (`xhci`), clean-room DRM GEM/KMS (`drm`), and Intel VT-d / AMD-Vi DMA remapping (`iommu`).
- **Phase 6 — Networking, Power Management, POSIX Shell & Multi-OS Compat**:
  - Ethernet/ARP/IPv4/UDP/TCP stack and VirtIO-Net (`net`), ACPI S-states and Intel HWP / AMD CPPC power profiles (`power`).
  - Clean-room `no_std` POSIX shell with parser, in-memory VFS, and system builtins (`nexus-shell`), plus unified ELF64/PE32+/Mach-O inspection and syscall translation (`nexus-compat`).
- **Phase 7 — Audio, Timekeeping, Multi-OS Exec & Recovery Shell**:
  - Intel HDA audio controller (`audio`), RTC/TSC timekeeping (`time`), multi-OS executable loader (`exec`), and interactive kernel recovery shell (`ksh`).
- **Phase 8 — macOS-Inspired Desktop Compositor, Init System & QEMU Boot CI**:
  - Clean-room desktop compositor, window manager, light/dark themes, and integrated terminal (`nexus-desktop`).
  - Declarative `/etc/nexus.conf` parser and dependency-aware service manager (`nexus-init`).
  - Automated QEMU boot verification and workspace test/clippy checks in GitHub Actions (`.github/workflows/ci.yml`).
- **Phase 9 — GPU Translation Bridge, Package Manager & Live Desktop Boot**:
  - Clean-room GPU API translation bridge for Direct3D 11/12, Metal, OpenGL, and Vulkan (`nexus-gpu-bridge`).
  - Multi-OS package manifest, dependency resolver, and registry supporting native, Linux ELF, Windows PE, and macOS `.app` bundles (`nexus-pkg`).
  - Early-boot `nexus-init` self-test and live `nexus-desktop` framebuffer preview (`desktop_boot`).
- **Phase 10 — Filesystems & Release Documentation**:
  - Clean-room `no_std` FAT32 and NexusFS filesystem implementations (`nexus-fs`).
  - Updated `README.md`, `docs/ARCHITECTURE.md`, and `CHANGELOG.md` for `v0.1.0`.
