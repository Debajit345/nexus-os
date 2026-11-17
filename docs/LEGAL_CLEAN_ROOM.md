# Clean-Room Legal Policy

Nexus OS is an independent, clean-room operating system project dual-licensed
under **MIT OR Apache-2.0**. Because Nexus OS implements compatibility layers
inspired by Linux (POSIX/VFS), Windows NT (Object Manager & Async I/O), and
macOS (Mach IPC & UX), strict clean-room engineering practices are mandatory for
all contributors.

---

## 1. Core Principle

**Clean-room implementation** means writing original code solely from public
specifications, published academic literature, and observable black-box
behavior, without any exposure to proprietary source code or reverse-engineered
internals.

---

## 2. Prohibited Sources (Strictly Forbidden)

Contributors **MUST NOT** use, reference, copy, or derive code from:

1. **Proprietary or Leaked Source Code**:
   - Any leaked Windows, macOS, or proprietary driver source code.
   - Any closed-source OS components obtained under NDA or employment.
2. **Decompiled or Disassembled Binaries**:
   - Output from disassemblers or decompilers (e.g., IDA Pro, Ghidra, Binary
     Ninja, `objdump`) run against proprietary binaries such as `ntoskrnl.exe`,
     `ntdll.dll`, `kernel32.dll`, macOS `kernel`, or proprietary GPU drivers.
3. **Incompatible Copyleft Code**:
   - Do not copy GPL-only (e.g., GPLv2/GPLv3) source code into Nexus OS crates,
     as Nexus OS is dual-licensed under `MIT OR Apache-2.0`.
4. **Undocumented Internal Symbols**:
   - Private symbol files (`.pdb` internals not published in official SDKs) or
     proprietary header dumps not released under a permissive license.

---

## 3. Permitted Sources

Contributors **MAY** consult and cite:

1. **Public Standards & Specifications**:
   - POSIX / The Open Group Base Specifications (Single UNIX Specification).
   - Khronos Group specifications (Vulkan, SPIR-V, OpenGL).
   - Hardware specifications (Intel/AMD `x86_64` SDM/APM, UEFI, ACPI, PCIe,
     NVMe, VirtIO).
2. **Official Public Documentation**:
   - Public Microsoft Learn / Windows Driver Kit (WDK) API documentation
     describing documented structures, flags, and behavioral contracts.
   - Public Apple Developer documentation and original CMU Mach 3.0 / 4.0
     academic papers.
3. **Permissively Licensed Open-Source References**:
   - Specifications or code licensed under MIT, BSD, Apache-2.0, or Zlib
     (with proper attribution and license compliance).
4. **Black-Box Behavioral Testing**:
   - Writing small, original test programs against public APIs to observe
     documented return codes, error values, and edge-case behavior.

---

## 4. Two-Team / Chinese-Wall Practice for Complex APIs

When implementing complex undocumented or subtly specified compatibility
behaviors:

1. **Specification Author**: Studies public documentation or black-box test
   results and writes a plain-English behavioral specification (inputs, outputs,
   error codes, invariants) without including any proprietary code.
2. **Implementer**: Writes the Rust implementation in Nexus OS solely from that
   written specification.

---

## 5. Contributor Attestation

By submitting a pull request to Nexus OS, you attest that:

- Your contribution is your own original work (or properly attributed
  permissively licensed work).
- You have not violated any of the prohibitions in Section 2.
- You have the legal right to license your contribution under `MIT OR Apache-2.0`.

If you have any doubt about whether a reference or technique is permissible,
open a discussion or ask a maintainer **before** writing or submitting code.
