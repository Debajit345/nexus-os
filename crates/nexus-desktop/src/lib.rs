#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Clean-room macOS-inspired desktop compositor, window manager, and
//! integrated terminal for Nexus OS.

extern crate alloc;

pub mod session;
pub mod terminal;
pub mod theme;
pub mod window;

pub use session::{ClickAction, DesktopSession};
pub use terminal::TerminalApp;
pub use theme::{DesktopTheme, ThemeMode};
pub use window::DesktopWindow;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use nexus_gfx::{Point, Rect, Size, Surface, TrafficLightButton};

    #[test]
    fn window_creation_and_z_order_focus() {
        let mut session = DesktopSession::new(Size::new(1280, 800));
        let w1 = session.open_window("Finder", Rect::new(80, 80, 400, 300));
        let w2 = session.open_window("Terminal", Rect::new(160, 120, 480, 320));

        assert_eq!(session.windows.len(), 2);
        assert_eq!(session.focused_window().map(|w| w.id), Some(w2));
        assert!(!session.window(w1).unwrap().focused);
        assert!(session.window(w2).unwrap().focused);

        // Clicking inside w1's non-overlapping region brings w1 to top and focuses it.
        let action = session.handle_click(Point::new(100, 150));
        assert_eq!(action, ClickAction::Focused(w1));
        assert_eq!(session.focused_window().map(|w| w.id), Some(w1));
        assert_eq!(session.windows.last().map(|w| w.id), Some(w1));
        assert!(!session.window(w2).unwrap().focused);
    }

    #[test]
    fn traffic_light_close_minimize_and_zoom() {
        let mut session = DesktopSession::new(Size::new(1280, 800));
        let w1 = session.open_window("Notes", Rect::new(100, 100, 400, 300));
        let w2 = session.open_window("Terminal", Rect::new(200, 140, 420, 300));

        // Minimize w2 via its yellow traffic-light button.
        let min_pt = session
            .window(w2)
            .unwrap()
            .frame
            .traffic_light_center(TrafficLightButton::Minimize);
        assert_eq!(session.handle_click(min_pt), ClickAction::Minimized(w2));
        assert!(session.window(w2).unwrap().minimized);
        assert_eq!(session.focused_window().map(|w| w.id), Some(w1));

        // Zoom w1 via its green traffic-light button.
        let zoom_pt = session
            .window(w1)
            .unwrap()
            .frame
            .traffic_light_center(TrafficLightButton::Zoom);
        assert_eq!(session.handle_click(zoom_pt), ClickAction::Zoomed(w1));
        assert!(session.window(w1).unwrap().maximized);
        assert_eq!(
            session.window(w1).unwrap().bounds,
            session.layout.work_area(session.dock_item_count())
        );

        // Close w1 via its red traffic-light button.
        let close_pt = session
            .window(w1)
            .unwrap()
            .frame
            .traffic_light_center(TrafficLightButton::Close);
        assert_eq!(session.handle_click(close_pt), ClickAction::Closed(w1));
        assert!(session.window(w1).is_none());
        assert_eq!(session.windows.len(), 1);
    }

    #[test]
    fn render_desktop_into_surface() {
        let screen = Size::new(640, 400);
        let mut session = DesktopSession::with_theme(screen, DesktopTheme::light());
        let w1 = session.open_window("Terminal", Rect::new(60, 60, 300, 200));
        assert!(session.window(w1).is_some());

        let mut pixels = vec![0u32; (screen.width * screen.height) as usize];
        let mut surface = Surface::new(&mut pixels, screen.width, screen.height, screen.width)
            .expect("valid surface");
        session.render(&mut surface);

        // Top-left pixel is inside the MenuBar, not zeroed.
        let bar_px = surface.get_pixel(10, 10).unwrap();
        assert_ne!(bar_px.as_raw(), 0);

        // Window body pixel matches light theme window_body.
        let body_px = surface.get_pixel(150, 150).unwrap();
        assert_eq!(body_px, session.theme.window_body);
    }

    #[test]
    fn terminal_app_executes_shell_commands() {
        let mut term = TerminalApp::new();
        assert_eq!(term.cwd(), "/");

        let out = term.exec("mkdir -p /home/nexus && cd /home/nexus && pwd");
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, "/home/nexus\n");
        assert_eq!(term.cwd(), "/home/nexus");

        let echo_out = term.exec("echo 'hello desktop'");
        assert_eq!(echo_out.exit_code, 0);
        assert_eq!(echo_out.stdout, "hello desktop\n");
        assert_eq!(term.history().len(), 2);
        assert!(term.transcript().iter().any(|l| l == "hello desktop"));
    }
}
