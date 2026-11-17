#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/target/iso"
KERNEL_BIN="${ROOT_DIR}/target/x86_64-unknown-none/debug/nexus-kernel"
ISO_PATH="${ROOT_DIR}/target/nexus-os.iso"
LIMINE_DIR="${LIMINE_DIR:-${ROOT_DIR}/target/limine}"

echo "[1/4] Building nexus-kernel (x86_64-unknown-none)..."
cargo build --manifest-path "${ROOT_DIR}/Cargo.toml" --target x86_64-unknown-none -p nexus-kernel

if [[ ! -d "${LIMINE_DIR}" ]]; then
    echo "[2/4] Fetching Limine binary release (v8.x-binary)..."
    git clone --depth=1 --branch=v8.x-binary https://github.com/limine-bootloader/limine.git "${LIMINE_DIR}"
    make -C "${LIMINE_DIR}"
fi

echo "[3/4] Staging ISO image..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/boot/limine" "${BUILD_DIR}/EFI/BOOT"

cp "${KERNEL_BIN}" "${BUILD_DIR}/boot/nexus-kernel"
cat > "${BUILD_DIR}/boot/limine/limine.conf" <<'EOF'
timeout: 0

/Nexus OS
    protocol: limine
    path: boot():/boot/nexus-kernel
EOF

cp "${LIMINE_DIR}/limine-bios.sys" \
   "${LIMINE_DIR}/limine-bios-cd.bin" \
   "${LIMINE_DIR}/limine-uefi-cd.bin" \
   "${BUILD_DIR}/boot/limine/"
cp "${LIMINE_DIR}/BOOTX64.EFI" "${BUILD_DIR}/EFI/BOOT/"

xorriso -as mkisofs -R -r -J -b boot/limine/limine-bios-cd.bin \
    -no-emul-boot -boot-load-size 4 -boot-info-table -hfsplus \
    -apm-block-size 2048 --efi-boot boot/limine/limine-uefi-cd.bin \
    -efi-boot-part --efi-boot-image --protective-msdos-label \
    "${BUILD_DIR}" -o "${ISO_PATH}"

"${LIMINE_DIR}/limine" bios-install "${ISO_PATH}"

echo "[4/4] Booting Nexus OS in QEMU..."
exec qemu-system-x86_64 \
    -M q35 \
    -m 256M \
    -cdrom "${ISO_PATH}" \
    -serial stdio \
    -no-reboot \
    -no-shutdown
