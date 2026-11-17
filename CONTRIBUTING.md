# Contributing to Nexus OS

Thank you for your interest in contributing to **Nexus OS**! We welcome
contributors of all backgrounds who want to help build a clean-room,
memory-safe, hybrid `x86_64` operating system in Rust.

Before submitting any code, documentation, or design proposals, please read this
guide and our [Code of Conduct](CODE_OF_CONDUCT.md) carefully.

---

## 1. Strict Clean-Room Policy (Mandatory)

Nexus OS implements compatibility layers inspired by Linux (POSIX/VFS),
Windows NT (Object Manager & Async I/O), and macOS (Mach IPC & UX). To protect
the project and all contributors legally, **every contribution must be 100%
clean-room**.

**You MUST NOT:**
- Copy, adapt, or reference proprietary Windows, macOS, or other closed-source
  operating system source code (including leaked source trees).
- Submit code derived from decompiled, disassembled, or reverse-engineered
  proprietary binaries (e.g., `ntoskrnl.exe`, `kernel32.dll`, `XNU` proprietary
  extensions, or closed-source GPU drivers).
- Use NDA-protected documentation, internal symbols, or private SDKs.
- Copy GPL-only code into MIT/Apache-2.0 licensed crates.

**You MAY:**
- Implement behavior from public specifications (POSIX / Single UNIX
  Specification, public Microsoft Learn / WDK API documentation, public Mach
  research papers, Vulkan / Khronos specifications, UEFI / ACPI / PCIe specs).
- Write original Rust implementations from scratch based on documented
  interfaces and observable black-box behavior.

See [docs/LEGAL_CLEAN_ROOM.md](docs/LEGAL_CLEAN_ROOM.md) for the full policy.
Pull requests that violate this policy will be rejected immediately.

---

## 2. Licensing

Nexus OS is dual-licensed under:
- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

By submitting a pull request, you agree that your contribution will be
dual-licensed under `MIT OR Apache-2.0` without any additional terms or
conditions.

---

## 3. Getting Started

1. Fork the repository at <https://github.com/Debajit345/nexus-os>.
2. Clone your fork and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
3. Ensure you have the Rust toolchain and `x86_64-unknown-none` target:
   ```bash
   rustup target add x86_64-unknown-none
   ```
4. Run host tests and kernel checks before pushing:
   ```bash
   cargo test --workspace --exclude nexus-kernel
   cargo check -p nexus-kernel --target x86_64-unknown-none
   cargo fmt --all -- --check
   ```

---

## 4. Pull Request Guidelines

- **Keep PRs focused**: One logical change per PR makes review faster and
  safer.
- **Write tests**: Host-side crates (`crates/*`) should include unit tests
  runnable via `cargo test`.
- **Document unsafe code**: Any `unsafe` block in Rust must include a
  `// SAFETY: ...` comment explaining why the invariants hold.
- **No generated artifacts**: Do not commit `target/`, `*.iso`, `*.img`, or
  editor swap files.
- **Describe your change**: Explain *what* changed, *why*, and *how* you tested
  it. If implementing a compatibility API, cite the public specification used.

---

## 5. Issue Guidelines

- **Bug reports**: Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md)
  template and include reproduction steps, target/QEMU details, and logs.
- **Feature requests**: Use the
  [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template and
  describe the use case and architectural fit.
- **Security issues**: Please do not open public issues for exploitable kernel
  vulnerabilities; contact the maintainers privately first.
