#![allow(dead_code)]

//! Early-boot integration for `nexus-init` and `nexus-desktop`.
//!
//! Provides a self-test for declarative `/etc/nexus.conf` parsing and
//! dependency-ordered service startup, plus a live macOS-inspired desktop
//! preview rendered via `nexus-desktop` and `nexus-gfx`.

use alloc::string::String;
use alloc::vec;

use nexus_desktop::{DesktopSession, DesktopTheme, TerminalApp};
use nexus_gfx::{Color, Rect, Size, Surface};
use nexus_init::{parse_config, ServiceManager, ServiceSpec};

use crate::limine::Framebuffer;
use crate::{serial_println, vfs};

/// Parses `/etc/nexus.conf`, registers core system services (`vfsd`, `gpud`,
/// `desktopd`), starts them in dependency order, and logs the running count.
pub fn run_init_selftest() {
    let mut manager = match vfs::read_to_vec("/etc/nexus.conf") {
        Ok(bytes) => match core::str::from_utf8(&bytes) {
            Ok(text) => match parse_config(text) {
                Ok(cfg) => {
                    serial_println!(
                        "Init: parsed /etc/nexus.conf (hostname={:?}, services={})",
                        cfg.hostname,
                        cfg.services.len()
                    );
                    ServiceManager::from_config(&cfg).unwrap_or_else(|_| ServiceManager::new())
                }
                Err(err) => {
                    serial_println!("Init: /etc/nexus.conf parse warning: {}", err);
                    ServiceManager::new()
                }
            },
            Err(_) => {
                serial_println!("Init: /etc/nexus.conf is not valid UTF-8");
                ServiceManager::new()
            }
        },
        Err(err) => {
            serial_println!("Init: failed to read /etc/nexus.conf: {:?}", err);
            ServiceManager::new()
        }
    };

    let mut vfsd = ServiceSpec::new("vfsd", "/sbin/vfsd");
    vfsd.restart_on_exit = true;
    vfsd.sandbox_strict = true;

    let mut gpud = ServiceSpec::new("gpud", "/sbin/gpud");
    gpud.dependencies.push(String::from("vfsd"));
    gpud.restart_on_exit = true;

    let mut desktopd = ServiceSpec::new("desktopd", "/sbin/desktopd");
    desktopd.dependencies.push(String::from("vfsd"));
    desktopd.dependencies.push(String::from("gpud"));
    desktopd.restart_on_exit = true;

    for spec in [vfsd, gpud, desktopd] {
        if manager.status(&spec.name).is_none() {
            if let Err(err) = manager.register(spec) {
                serial_println!("Init: service registration error: {}", err);
            }
        }
    }

    match manager.start_all() {
        Ok(order) => {
            serial_println!(
                "Init: started {} services in order {:?} (running_count={})",
                order.len(),
                order,
                manager.running_count()
            );
        }
        Err(err) => {
            serial_println!(
                "Init: service startup error: {} (running_count={})",
                err,
                manager.running_count()
            );
        }
    }
}

/// Renders a live macOS-inspired desktop preview (MenuBar, Dock, and Terminal
/// window) into a compact offscreen `Surface` and blits it onto the Limine
/// framebuffer when 32bpp is active.
pub fn render_desktop_preview(fb: &Framebuffer) {
    if fb.address.is_null() || fb.width == 0 || fb.height == 0 || fb.bpp != 32 {
        serial_println!(
            "Desktop preview skipped (addr_null={}, {}x{} @ {} bpp)",
            fb.address.is_null(),
            fb.width,
            fb.height,
            fb.bpp
        );
        return;
    }

    let preview_w = (fb.width as u32).min(320);
    let preview_h = (fb.height as u32).min(200);
    let screen = Size::new(preview_w, preview_h);

    let mut session = DesktopSession::with_theme(screen, DesktopTheme::dark());
    let win_w = preview_w.saturating_sub(64).max(120);
    let win_h = preview_h.saturating_sub(84).max(80);
    let _term_win = session.open_window("Terminal", Rect::new(32, 34, win_w, win_h));

    let mut term = TerminalApp::new();
    let _ = term.exec("echo 'Nexus Desktop Ready'");

    let mut pixels = vec![0u32; (preview_w as usize) * (preview_h as usize)];
    let Ok(mut surface) = Surface::new(&mut pixels, preview_w, preview_h, preview_w) else {
        serial_println!("Desktop preview: failed to create offscreen surface");
        return;
    };
    session.render(&mut surface);

    let dst_x = (fb.width as u32).saturating_sub(preview_w + 16) as usize;
    let dst_y = 16usize.min((fb.height as usize).saturating_sub(preview_h as usize));
    let pitch = fb.pitch as usize;

    for y in 0..(preview_h as usize) {
        let row_offset = (dst_y + y) * pitch;
        for x in 0..(preview_w as usize) {
            let raw = pixels[y * (preview_w as usize) + x];
            let color = Color::from_raw(raw);
            let packed = ((color.r() as u32) << fb.red_mask_shift)
                | ((color.g() as u32) << fb.green_mask_shift)
                | ((color.b() as u32) << fb.blue_mask_shift);
            let offset = row_offset + (dst_x + x) * 4;
            unsafe {
                let ptr = fb.address.add(offset) as *mut u32;
                core::ptr::write_volatile(ptr, packed);
            }
        }
    }

    serial_println!(
        "Desktop preview rendered: {}x{} at ({}, {}), windows={}, dock_items={}, term_lines={}",
        preview_w,
        preview_h,
        dst_x,
        dst_y,
        session.windows.len(),
        session.dock_item_count(),
        term.transcript().len()
    );
}
