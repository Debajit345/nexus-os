use alloc::string::String;
use nexus_gfx::{Point, Rect, Surface, WindowFrame, WindowHit};

/// A managed desktop window with macOS-inspired chrome and state flags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DesktopWindow {
    pub id: u64,
    pub title: String,
    pub bounds: Rect,
    pub restore_bounds: Option<Rect>,
    pub minimized: bool,
    pub maximized: bool,
    pub focused: bool,
    pub frame: WindowFrame,
}

impl DesktopWindow {
    /// Create a new focused, non-minimized window with the given `id`, `title`, and `bounds`.
    pub fn new(id: u64, title: impl Into<String>, bounds: Rect) -> Self {
        Self {
            id,
            title: title.into(),
            bounds,
            restore_bounds: None,
            minimized: false,
            maximized: false,
            focused: true,
            frame: WindowFrame::new(bounds),
        }
    }

    /// Update the window bounds and synchronize its `WindowFrame`.
    pub fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
        self.frame.bounds = bounds;
    }

    /// Update the window focus state and synchronize its `WindowFrame`.
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        self.frame.focused = focused;
    }

    /// Hit-test a point against this window's frame. Minimized windows never hit.
    pub fn hit_test(&self, point: Point) -> WindowHit {
        if self.minimized {
            return WindowHit::Outside;
        }
        self.frame.hit_test(point)
    }

    /// Render the window frame onto `surface` if not minimized.
    pub fn render(&self, surface: &mut Surface<'_>) {
        if self.minimized {
            return;
        }
        self.frame.render(surface);
    }
}
