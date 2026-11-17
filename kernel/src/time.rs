//! CMOS RTC wall-clock and TSC/PIT monotonic nanosecond timekeeping.
#![allow(dead_code)]

use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::apic::{inb, outb};

const CMOS_ADDR: u16 = 0x70;
const CMOS_DATA: u16 = 0x71;

/// Wall-clock date and time read from the CMOS RTC.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DateTime {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
}

static BOOT_TSC: AtomicU64 = AtomicU64::new(0);
static TSC_HZ: AtomicU64 = AtomicU64::new(2_000_000_000); // 2 GHz default fallback

/// Read the x86_64 Time Stamp Counter (`rdtsc`).
#[inline]
pub fn rdtsc() -> u64 {
    let low: u32;
    let high: u32;
    unsafe {
        asm!(
            "rdtsc",
            out("eax") low,
            out("edx") high,
            options(nomem, nostack, preserves_flags)
        );
    }
    ((high as u64) << 32) | (low as u64)
}

#[inline]
unsafe fn cmos_read(reg: u8) -> u8 {
    outb(CMOS_ADDR, reg);
    inb(CMOS_DATA)
}

#[inline]
fn bcd_to_bin(val: u8) -> u8 {
    (val & 0x0F) + ((val >> 4) * 10)
}

/// Read the current UTC date and time from the CMOS RTC.
pub fn read_rtc() -> DateTime {
    unsafe {
        // Wait until RTC update-in-progress (bit 7 of status register A) clears.
        while (cmos_read(0x0A) & 0x80) != 0 {}

        let mut sec = cmos_read(0x00);
        let mut min = cmos_read(0x02);
        let mut hour = cmos_read(0x04);
        let mut day = cmos_read(0x07);
        let mut month = cmos_read(0x08);
        let mut year = cmos_read(0x09);
        let reg_b = cmos_read(0x0B);

        if (reg_b & 0x04) == 0 {
            sec = bcd_to_bin(sec);
            min = bcd_to_bin(min);
            hour = bcd_to_bin(hour & 0x7F) | (hour & 0x80);
            day = bcd_to_bin(day);
            month = bcd_to_bin(month);
            year = bcd_to_bin(year);
        }

        if (reg_b & 0x02) == 0 && (hour & 0x80) != 0 {
            hour = ((hour & 0x7F) + 12) % 24;
        }

        DateTime {
            year: 2000 + year as u16,
            month,
            day,
            hour,
            minute: min,
            second: sec,
        }
    }
}

/// Convert a `DateTime` (year >= 1970) to an approximate Unix timestamp in seconds.
pub fn unix_timestamp() -> u64 {
    let dt = read_rtc();
    let y = dt.year as u64;
    let m = dt.month.clamp(1, 12) as u64;
    let d = dt.day.clamp(1, 31) as u64;

    let mut days: u64 = 0;
    for yr in 1970..y {
        let leap = (yr.is_multiple_of(4) && !yr.is_multiple_of(100)) || yr.is_multiple_of(400);
        days += if leap { 366 } else { 365 };
    }

    const MONTH_DAYS: [u64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let leap_cur = (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400);
    for (idx, &mdays) in MONTH_DAYS.iter().enumerate().take(m as usize - 1) {
        days += mdays;
        if idx == 1 && leap_cur {
            days += 1;
        }
    }
    days += d - 1;

    days * 86_400 + (dt.hour as u64) * 3_600 + (dt.minute as u64) * 60 + (dt.second as u64)
}

/// Initialize monotonic TSC timekeeping using CPUID leaf `0x15` when available.
pub fn init() {
    BOOT_TSC.store(rdtsc(), Ordering::Relaxed);

    let max_leaf: u32;
    unsafe {
        asm!(
            "push rbx",
            "cpuid",
            "pop rbx",
            inout("eax") 0u32 => max_leaf,
            out("ecx") _,
            out("edx") _,
            options(nomem, preserves_flags)
        );
    }

    if max_leaf >= 0x15 {
        let denom: u32;
        let numer: u32;
        let crystal_hz: u32;
        unsafe {
            asm!(
                "push rbx",
                "cpuid",
                "mov {0:e}, ebx",
                "pop rbx",
                out(reg) numer,
                inout("eax") 0x15u32 => denom,
                out("ecx") crystal_hz,
                out("edx") _,
                options(nomem, preserves_flags)
            );
        }
        if denom != 0 && numer != 0 && crystal_hz != 0 {
            let hz = (crystal_hz as u64) * (numer as u64) / (denom as u64);
            if hz > 0 {
                TSC_HZ.store(hz, Ordering::Relaxed);
            }
        }
    }
}

/// Return the calibrated TSC frequency in Hz.
#[inline]
pub fn tsc_hz() -> u64 {
    TSC_HZ.load(Ordering::Relaxed)
}

/// Return monotonic nanoseconds elapsed since `time::init()`.
#[inline]
pub fn uptime_ns() -> u64 {
    let delta = rdtsc().saturating_sub(BOOT_TSC.load(Ordering::Relaxed));
    let hz = tsc_hz().max(1);
    ((delta as u128 * 1_000_000_000u128) / hz as u128) as u64
}
