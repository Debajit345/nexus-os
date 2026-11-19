# Nexus OS — Production Kernel Roadmap (NT / Linux / XNU Parity)

This document catalogs the complete engineering roadmap required to bring the **Nexus OS** `x86_64` kernel (`nexus-kernel`) from its `v0.2.0` clean-room architectural foundation to production parity with mature kernels such as **Windows NT**, **Linux**, and **Apple XNU**.

---

## 1. Process, Thread & Virtual Memory Management
- [x] **Stage 21 (`kernel/src/proc.rs`)**: Per-process PID/TID table, per-process PML4 root, `VmArea` / VAD region tracking with W^X & Copy-on-Write (CoW) flags, demand-paging fault handler, `fork`/`exec`/`wait`/`exit` state machine, TLS (`FS_BASE`/`GS_BASE`), and unified signal/SEH/Mach-exception queue.
- [ ] **Real Ring 3 Dynamic Linker Support**: `ld-linux-x86-64.so.2` (ELF), `ntdll.dll` (PE32+), and `dyld` (Mach-O) user-space bootstrap with `AT_*` auxiliary vectors, PEB/TEB, and `commpage`.
- [ ] **Demand Paging & Swap**: Page-fault-driven anonymous zero-fill, file-backed `mmap` / `MapViewOfFile`, CoW page splitting with physical frame refcounting, and compressed/disk swap.
- [x] **Stage 25 (`kernel/src/slab.rs`)**: Slab/magazine kernel allocator (16B..4KiB size classes, per-CPU magazines), 2 MiB / 1 GiB huge-page descriptors, and NUMA node frame pools.
- [ ] **SMP Preemptive Scheduler**: Per-CPU runqueues, work-stealing load balancer, CFS/EEVDF + real-time deadline classes, and Intel Hybrid P-core/E-core Thread Director scheduling.

## 2. Synchronization, IPC & Event Multiplexing
- [x] **Stage 22 (`kernel/src/sync.rs`)**: Ticket spinlocks, `RwLock`, `Futex` wait-queue hash table (backing Linux `futex`, Win32 `WaitOnAddress`, and Mach `ulock`), and unified `EventMux` (backing `epoll`, Windows `IOCP`, and BSD/macOS `kqueue`).
- [x] **Stage 28 (`kernel/src/rcu.rs`)**: Epoch-based quiescent-state RCU grace-period reclamation and Lockdep deadlock cycle detector.
- [ ] **Zero-Copy Mach/NT IPC**: Out-of-line (OOL) descriptor page remapping and shared-memory ring buffers for compositor/audio/driver servers.

## 3. Storage, Filesystems & DMA
- [x] **Stage 23 (`kernel/src/page_cache.rs`)**: 4 KiB LRU page cache with dirty-page writeback, crash-safe Write-Ahead Log (WAL) with CRC32C checksums, and scatter-gather `SgList` DMA descriptor builder.
- [ ] **Production CoW / Journaling Filesystem**: Full B-tree extent allocator, snapshots, POSIX ACLs + NT Security Descriptors, case-insensitive/case-preserving APFS/NTFS semantics, and `fsck` recovery.
- [x] **Stage 27 (`kernel/src/blk_mq.rs`)**: Multi-queue `blk-mq` per-CPU staging queues, hardware dispatch queues, request merging/elevator sorting, and tag allocator.

## 4. Security, Isolation & Hardware Mitigations
- [x] **Stage 24 (`kernel/src/hardening.rs`)**: `RDRAND`/`RDSEED`/TSC entropy pool + CSPRNG, ASLR/KASLR slide generator, CPU vulnerability & mitigation probe (`IBRS`, `STIBP`, `SSBD`, `MD_CLEAR`, `SMEP`, `SMAP`, `UMIP`, `CET`), W^X & stack-canary audit.
- [ ] **Full IOMMU & Confidential Computing**: Per-PCIe-BDF VT-d / AMD-Vi page tables, interrupt remapping, and AMD SEV-SNP / Intel TDX guest support.
- [ ] **Kernel Fuzzing & Formal Verification**: Syscall fuzzer (Syzkaller-style), KASAN/KMSAN/UBSAN kernel sanitizers, and Miri/Kani verification of `unsafe` boundaries.

## 5. Hardware Drivers, ACPI & GPU Stack
- [x] **Stage 26 (`kernel/src/aml.rs`)**: Clean-room ACPI AML bytecode parser (`Scope`/`Device`/`Method`/`Name`/`Package`), `_PRT` PCIe IRQ routing, and thermal/battery descriptors.
- [ ] **GPU Kernel Mode-Setting (KMS) & Vulkan**: Native Intel Xe/i915, AMD AMDGPU, and NVIDIA OpenRM command submission, GEM/TTM VRAM manager, and Wayland/Aqua-style zero-copy compositor.
- [ ] **Networking & Wireless**: Full TCP congestion control (CUBIC/BBR), IPv6, WireGuard/TLS offload, Intel/Realtek/Broadcom Ethernet, and 802.11ax/be Wi-Fi + Bluetooth stack.
- [ ] **USB4 / Thunderbolt & Audio**: Full xHCI isochronous transfers, USB HID/Audio/Video classes, and Intel HDA / SOF DSP firmware loading.
