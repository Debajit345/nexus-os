# Nexus OS

**Nexus OS** is a clean-room, `x86_64` hybrid operating system written in Rust.
It unifies the strongest architectural ideas from three major OS families into a
single, memory-safe kernel and userspace platform:

- **Linux**: POSIX semantics, Virtual File System (VFS), and familiar
  process/descriptor abstractions.
- **Windows NT**: Unified Object Manager, handle tables, and true asynchronous
  I/O (I/O Completion Ports / IRP-style pipelines).
- **macOS / Mach**: Capability-based Mach IPC (ports & messages) and a
  responsive, compositor-driven user experience.
- **GPU / Vulkan-First Roadmap**: Designed from the ground up for modern
  hardware-accelerated rendering, zero-copy buffer sharing, and a Vulkan-first
  graphics stack.

> **Status**: Early active development. Contributions, architectural feedback,
> and clean-room implementations are warmly welcomed!

---

## Why Nexus OS?

Modern operating systems often carry decades of legacy C/C++ technical debt and
fragmented I/O or IPC models. Nexus OS explores what happens when you combine:

1. **Rust's memory safety** at the kernel and driver boundary.
2. **NT-style Object & Async I/O** as the core kernel substrate.
3. **Mach-style capability IPC** for modular system services and sandboxing.
4. **POSIX/VFS compatibility** for broad developer and tooling ergonomics.
5. **A GPU/Vulkan-first display pipeline** rather than bolting modern graphics
   onto a legacy framebuffer.

---

## Repository Layout

- `kernel/`: Bare-metal `x86_64-unknown-none` microkernel (`nexus-kernel`) with
  modules for `cpu`, `acpi`, `pci`, `mm`, `vmm`, `cap`, `sched`, `ipc`, `vfs`,
  `syscall`, `usermode`, `smp`, `blk`, `nvme`, `xhci`, `drm`, `iommu`, `net`,
  `power`, `apic`, `gdt`, `idt`, `limine`, `serial`, and `console`.
- `crates/`: Host-testable `no_std` core subsystems:
  - `nexus-abi`: Shared capability handles, object types, personalities, and syscall ABI.
  - `nexus-elf`: Bounds-checked ELF64 parser and validator (`#![forbid(unsafe_code)]`).
  - `nexus-pe`: Bounds-checked PE32+ parser and validator (`#![forbid(unsafe_code)]`).
  - `nexus-macho`: Bounds-checked Mach-O 64-bit parser and validator (`#![forbid(unsafe_code)]`).
  - `nexus-compat`: Unified binary inspection and multi-personality syscall translation.
  - `nexus-gfx`: 2D compositor primitives, Aqua-inspired desktop chrome, and GPU/DRM traits.
  - `nexus-shell`: Clean-room POSIX shell, parser, in-memory VFS, and system builtins.
- `docs/`: Architecture specifications, OS feature matrix, and clean-room legal guidelines.
- `.github/`: CI workflows and issue templates.

---

## Quick Start

### Prerequisites

- Rust toolchain (nightly or stable with `x86_64-unknown-none` target)
- `qemu-system-x86_64` (for running kernel images)

```bash
rustup target add x86_64-unknown-none
```

### Build & Test

```bash
# Run unit tests for host-compatible crates
cargo test --workspace --exclude nexus-kernel

# Type-check the bare-metal x86_64 kernel
cargo check -p nexus-kernel --target x86_64-unknown-none
```

---

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md): Hybrid kernel design,
  implemented modules, workspace crates, and GPU/Vulkan-first roadmap.
- [OS Feature Matrix](docs/OS_FEATURE_MATRIX.md): Core-to-desktop comparison of
  macOS, Linux, and Windows (what is kept vs eliminated) and how each maps to
  Nexus OS crates and kernel modules.
- [Clean-Room Legal Policy](docs/LEGAL_CLEAN_ROOM.md): Mandatory rules for
  clean-room development.
- [Contributing Guide](CONTRIBUTING.md): How to submit PRs, run tests, and
  follow project standards.
- [Code of Conduct](CODE_OF_CONDUCT.md): Community expectations.

---

## Contributing

We welcome contributors! Whether you are interested in `x86_64` boot/paging,
async I/O, VFS, Mach-style IPC, or Vulkan/GPU drivers, please read
[CONTRIBUTING.md](CONTRIBUTING.md) and [docs/LEGAL_CLEAN_ROOM.md](docs/LEGAL_CLEAN_ROOM.md)
before opening a pull request.

---

## License

Nexus OS is dual-licensed under either of:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
