# Nexus OS — Production Kernel Roadmap (NT / Linux / XNU Parity)

This document catalogs the complete engineering roadmap required to bring the **Nexus OS** `x86_64` kernel (`nexus-kernel`) from its `v0.2.0` clean-room architectural foundation to production parity with mature kernels such as **Windows NT**, **Linux**, and **Apple XNU**.

---

## 1. Process, Thread & Virtual Memory Management
- [x] **Stage 21 (`kernel/src/proc.rs`)**: Per-process PID/TID table, per-process PML4 root, `VmArea` / VAD region tracking with W^X & Copy-on-Write (CoW) flags, demand-paging fault handler, `fork`/`exec`/`wait`/`exit` state machine, TLS (`FS_BASE`/`GS_BASE`), and unified signal/SEH/Mach-exception queue.
- [x] **Stage 29 (`kernel/src/dynlink.rs`)**: Ring 3 dynamic linker (`PT_DYNAMIC`, SysV/GNU hash, `R_X86_64_*`), ELF `AT_*` auxv, Windows PEB/TEB + `.reloc`, and macOS `dyld` `commpage` + `LC_MAIN` bootstrap.
- [x] **Stage 34 (`kernel/src/zswap.rs`)**: Demand paging, compressed ZRAM/Zswap pool (LZ-style compression + CRC32C), active/inactive LRU reclaim, and OOM score evaluator.
- [x] **Stage 25 (`kernel/src/slab.rs`)**: Slab/magazine kernel allocator (16B..4KiB size classes, per-CPU magazines), 2 MiB / 1 GiB huge-page descriptors, and NUMA node frame pools.
- [x] **Stage 33 (`kernel/src/eevdf.rs`)**: SMP preemptive EEVDF + `SCHED_DEADLINE` scheduler, work-stealing load balancer, and Intel Hybrid P/E-core Thread Director.

## 2. Synchronization, IPC & Event Multiplexing
- [x] **Stage 22 (`kernel/src/sync.rs`)**: Ticket spinlocks, `RwLock`, `Futex` wait-queue hash table (backing Linux `futex`, Win32 `WaitOnAddress`, and Mach `ulock`), and unified `EventMux` (backing `epoll`, Windows `IOCP`, and BSD/macOS `kqueue`).
- [x] **Stage 28 (`kernel/src/rcu.rs`)**: Epoch-based quiescent-state RCU grace-period reclamation and Lockdep deadlock cycle detector.
- [x] **Stage 37 (`kernel/src/ool_ipc.rs`)**: Zero-copy Mach/NT OOL page-descriptor remapping, port rights transfer, ALPC section views, and lock-free SPSC/MPSC ring buffers.

## 3. Storage, Filesystems & DMA
- [x] **Stage 23 (`kernel/src/page_cache.rs`)**: 4 KiB LRU page cache with dirty-page writeback, crash-safe Write-Ahead Log (WAL) with CRC32C checksums, and scatter-gather `SgList` DMA descriptor builder.
- [x] **Stage 30 (`kernel/src/cow_fs.rs`)**: CoW B-tree extent filesystem, O(1) snapshots/reflinks, POSIX ACLs + NT Security Descriptors, case-preserving/insensitive lookup, and `fsck` verifier.
- [x] **Stage 27 (`kernel/src/blk_mq.rs`)**: Multi-queue `blk-mq` per-CPU staging queues, hardware dispatch queues, request merging/elevator sorting, and tag allocator.

## 4. Security, Isolation & Hardware Mitigations
- [x] **Stage 24 (`kernel/src/hardening.rs`)**: `RDRAND`/`RDSEED`/TSC entropy pool + CSPRNG, ASLR/KASLR slide generator, CPU vulnerability & mitigation probe (`IBRS`, `STIBP`, `SSBD`, `MD_CLEAR`, `SMEP`, `SMAP`, `UMIP`, `CET`), W^X & stack-canary audit.
- [x] **Stage 35 (`kernel/src/iommu_vtd.rs`)**: Per-PCIe-BDF VT-d/AMD-Vi 4-level IOVA page tables, IRTE interrupt remapping, and AMD SEV-SNP / Intel TDX confidential computing probe.
- [x] **Stage 38 (`kernel/src/kasan.rs`)**: KASAN 8:1 shadow memory redzone tracker (UAF/OOB/double-free) and deterministic syscall fuzzer harness across all 4 personalities.

## 5. Hardware Drivers, ACPI & GPU Stack
- [x] **Stage 26 (`kernel/src/aml.rs`)**: Clean-room ACPI AML bytecode parser (`Scope`/`Device`/`Method`/`Name`/`Package`), `_PRT` PCIe IRQ routing, and thermal/battery descriptors.
- [x] **Stage 31 (`kernel/src/kms.rs`)**: GPU Kernel Mode-Setting (CRTC/Connector/Encoder, EDID parser), GEM/TTM VRAM allocator with fence sync, and GPU ring submission descriptors.
- [x] **Stage 32 (`kernel/src/net_v6.rs`)**: IPv6 NDP/SLAAC, TCP CUBIC/BBR congestion control, and WireGuard ChaCha20-Poly1305 AEAD offload engine.
- [x] **Stage 36 (`kernel/src/usb_class.rs`)**: USB4/xHCI isochronous TRB ring, USB HID report parser, UAC2 audio descriptors, and BOT/UASP SCSI command builder.
- [x] **Stage 39 (`kernel/src/wireless.rs`)**: 802.11ax/be Wi-Fi MAC, WPA3-SAE/CCMP key hierarchy, and Bluetooth HCI/L2CAP stack.
- [x] **Stage 40 (`kernel/src/sof_dsp.rs`)**: Intel HDA / SOF IPC4 DSP topology/pipeline builder and CORB/RIRB + BDL audio DMA engine.
- [x] **Stage 41 (`kernel/src/virtio_gpu.rs`)**: VirtIO-GPU 2D/3D Virgl & Venus/Vulkan command submission engine.
- [x] **Stage 42 (`kernel/src/ahci.rs`)**: AHCI / SATA 3.0 HBA driver & 32-slot NCQ command engine.
- [x] **Stage 43 (`kernel/src/netfilter.rs`)**: Stateful packet filter, 5-tuple NAT/conntrack & eBPF-style XDP hook engine.
- [x] **Stage 44 (`kernel/src/cgroups.rs`)**: Linux-compatible namespaces (`pid`/`mnt`/`net`/`uts`/`ipc`/`user`) & Cgroups v2 resource controller.
- [x] **Stage 45 (`kernel/src/perf.rs`)**: Kernel tracing, PMU performance counters & verified eBPF probe engine.
- [x] **Stage 46 (`kernel/src/tpm.rs`)**: TPM 2.0 measured boot, PCR extend/quote & kernel keyring / Ed25519 signature verifier.
