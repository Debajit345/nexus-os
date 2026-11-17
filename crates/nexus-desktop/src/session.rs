use alloc::string::String;
use alloc::vec::Vec;
use nexus_gfx::{Color, DesktopLayout, MenuBarHit, Point, Rect, Size, Surface, WindowHit};

use crate::theme::DesktopTheme;
use crate::window::DesktopWindow;

/// Outcome of a pointer click handled by `DesktopSession::handle_click`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ClickAction {
    None,
    MenuBar(MenuBarHit),
    DockItem(usize),
    Focused(u64),
    Closed(u64),
    Minimized(u64),
    Zoomed(u64),
}

/// macOS-inspired desktop compositor and window manager session.
#[derive(Clone, Debug)]
pub struct DesktopSession {
    pub screen: Size,
    pub theme: DesktopTheme,
    pub layout: DesktopLayout,
    /// Windows ordered from bottom-most (index 0) to top-most (last).
    pub windows: Vec<DesktopWindow>,
    /// Minimum number of dock slots shown even when fewer windows are open.
    pub pinned_dock_items: usize,
    pub cursor: Option<Point>,
    next_window_id: u64,
}

impl DesktopSession {
    /// Create a new desktop session with the default dark theme.
    pub fn new(screen: Size) -> Self {
        Self::with_theme(screen, DesktopTheme::default())
    }

    /// Create a new desktop session with a specific `DesktopTheme`.
    pub fn with_theme(screen: Size, theme: DesktopTheme) -> Self {
        let layout = DesktopLayout::new(screen, theme.menu_bar(), theme.dock());
        Self {
            screen,
            theme,
            layout,
            windows: Vec::new(),
            pinned_dock_items: 4,
            cursor: None,
            next_window_id: 1,
        }
    }

    /// Number of items currently displayed in the dock.
    pub fn dock_item_count(&self) -> usize {
        self.pinned_dock_items.max(self.windows.len())
    }

    /// Open a new window, place it at the top of the z-order, and focus it.
    pub fn open_window(&mut self, title: impl Into<String>, bounds: Rect) -> u64 {
        let id = self.next_window_id;
        self.next_window_id = self.next_window_id.saturating_add(1);

        for win in &mut self.windows {
            win.set_focused(false);
            self.theme.apply_to_frame(&mut win.frame, false);
        }

        let mut win = DesktopWindow::new(id, title, bounds);
        self.theme.apply_to_frame(&mut win.frame, true);
        self.windows.push(win);
        id
    }

    /// Close a window by `id`. Returns `true` if the window existed.
    pub fn close_window(&mut self, id: u64) -> bool {
        let Some(pos) = self.windows.iter().position(|w| w.id == id) else {
            return false;
        };
        let was_focused = self.windows[pos].focused;
        self.windows.remove(pos);
        if was_focused {
            self.focus_topmost_visible();
        }
        true
    }

    /// Bring window `id` to the top of the z-order, unminimize it, and focus it.
    pub fn focus_window(&mut self, id: u64) -> bool {
        let Some(pos) = self.windows.iter().position(|w| w.id == id) else {
            return false;
        };
        let mut target = self.windows.remove(pos);
        target.minimized = false;
        for win in &mut self.windows {
            win.set_focused(false);
            self.theme.apply_to_frame(&mut win.frame, false);
        }
        target.set_focused(true);
        self.theme.apply_to_frame(&mut target.frame, true);
        self.windows.push(target);
        true
    }

    /// Minimize window `id` and transfer focus to the next visible window.
    pub fn minimize_window(&mut self, id: u64) -> bool {
        let Some(win) = self.windows.iter_mut().find(|w| w.id == id) else {
            return false;
        };
        let was_focused = win.focused;
        win.minimized = true;
        win.set_focused(false);
        self.theme.apply_to_frame(&mut win.frame, false);
        if was_focused {
            self.focus_topmost_visible();
        }
        true
    }

    /// Toggle maximized (zoomed) state for window `id` within the desktop work area.
    pub fn toggle_maximize_window(&mut self, id: u64) -> bool {
        let work_area = self.layout.work_area(self.dock_item_count());
        let Some(win) = self.windows.iter_mut().find(|w| w.id == id) else {
            return false;
        };
        if win.maximized {
            if let Some(prev) = win.restore_bounds.take() {
                win.set_bounds(prev);
            }
            win.maximized = false;
        } else {
            win.restore_bounds = Some(win.bounds);
            win.set_bounds(work_area);
            win.maximized = true;
        }
        true
    }

    /// Return the currently focused window, if any.
    pub fn focused_window(&self) -> Option<&DesktopWindow> {
        self.windows
            .iter()
            .rev()
            .find(|w| w.focused && !w.minimized)
    }

    /// Look up a window by `id`.
    pub fn window(&self, id: u64) -> Option<&DesktopWindow> {
        self.windows.iter().find(|w| w.id == id)
    }

    /// Look up a mutable window by `id`.
    pub fn window_mut(&mut self, id: u64) -> Option<&mut DesktopWindow> {
        self.windows.iter_mut().find(|w| w.id == id)
    }

    /// Handle a pointer click at `point` across menu bar, dock, and windows.
    pub fn handle_click(&mut self, point: Point) -> ClickAction {
        let menu_hit = self.layout.menu_bar.hit_test(self.screen, point);
        if menu_hit != MenuBarHit::None {
            return ClickAction::MenuBar(menu_hit);
        }

        let dock_count = self.dock_item_count();
        if let Some(idx) = self
            .layout
            .dock
            .hit_test(self.screen, dock_count, point, self.cursor)
        {
            if let Some(win_id) = self.windows.get(idx).map(|w| w.id) {
                self.focus_window(win_id);
            }
            return ClickAction::DockItem(idx);
        }

        let mut hit_result: Option<(u64, WindowHit)> = None;
        for win in self.windows.iter().rev() {
            let hit = win.hit_test(point);
            if hit != WindowHit::Outside {
                hit_result = Some((win.id, hit));
                break;
            }
        }

        let Some((id, hit)) = hit_result else {
            return ClickAction::None;
        };

        match hit {
            WindowHit::CloseButton => {
                self.close_window(id);
                ClickAction::Closed(id)
            }
            WindowHit::MinimizeButton => {
                self.minimize_window(id);
                ClickAction::Minimized(id)
            }
            WindowHit::ZoomButton => {
                self.toggle_maximize_window(id);
                self.focus_window(id);
                ClickAction::Zoomed(id)
            }
            WindowHit::Titlebar | WindowHit::Content => {
                self.focus_window(id);
                ClickAction::Focused(id)
            }
            WindowHit::Outside => ClickAction::None,
        }
    }

    /// Render wallpaper, visible windows in z-order, global `MenuBar`, and `Dock`.
    pub fn render(&self, surface: &mut Surface<'_>) {
        self.render_wallpaper(surface);
        for win in &self.windows {
            win.render(surface);
        }
        self.layout.menu_bar.render(surface);
        self.layout
            .dock
            .render(surface, self.dock_item_count(), self.cursor);
    }

    fn render_wallpaper(&self, surface: &mut Surface<'_>) {
        let w = surface.width();
        let h = surface.height();
        if w == 0 || h == 0 {
            return;
        }
        let top = self.theme.wallpaper_top;
        let bot = self.theme.wallpaper_bottom;
        if h == 1 || top == bot {
            surface.fill_rect(surface.bounds(), top);
            return;
        }
        let denom = (h - 1) as i32;
        for y in 0..h {
            let t = y as i32;
            let r = lerp_u8(top.r(), bot.r(), t, denom);
            let g = lerp_u8(top.g(), bot.g(), t, denom);
            let b = lerp_u8(top.b(), bot.b(), t, denom);
            surface.fill_rect(Rect::new(0, y as i32, w, 1), Color::from_rgb(r, g, b));
        }
    }

    fn focus_topmost_visible(&mut self) {
        let top_id = self
            .windows
            .iter()
            .rev()
            .find(|w| !w.minimized)
            .map(|w| w.id);
        for win in &mut self.windows {
            let is_focus = Some(win.id) == top_id;
            win.set_focused(is_focus);
            self.theme.apply_to_frame(&mut win.frame, is_focus);
        }
    }
}

#[inline]
fn lerp_u8(a: u8, b: u8, num: i32, denom: i32) -> u8 {
    let a = a as i32;
    let b = b as i32;
    (a + ((b - a) * num) / denom).clamp(0, 255) as u8
}
