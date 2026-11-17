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

- `kernel/`: Bare-metal `x86_64-unknown-none` kernel crate.
- `crates/`: Host-testable core subsystems (object manager, VFS, IPC, async
  I/O, scheduler, etc.).
- `docs/`: Architecture specifications and clean-room legal guidelines.
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
  subsystems, and GPU/Vulkan-first roadmap.
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
