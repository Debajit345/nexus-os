use crate::color::Color;
use crate::geometry::{Point, Rect, Size};
use crate::surface::Surface;

/// Hit-test result for the global top menu bar.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MenuBarHit {
    None,
    AppleMenu,
    AppMenuArea,
    StatusArea,
}

/// macOS-inspired global top menu bar.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MenuBar {
    pub height: u32,
    pub apple_menu_width: u32,
    pub status_area_width: u32,
    pub background: Color,
    pub border_color: Color,
    pub accent_color: Color,
}

impl Default for MenuBar {
    fn default() -> Self {
        Self {
            height: 28,
            apple_menu_width: 40,
            status_area_width: 180,
            background: Color::from_argb(0xD8, 0x1E, 0x1E, 0x24),
            border_color: Color::from_argb(0x40, 0xFF, 0xFF, 0xFF),
            accent_color: Color::from_rgb(0xE5, 0xE7, 0xEB),
        }
    }
}

impl MenuBar {
    #[inline]
    pub const fn bounds(&self, screen: Size) -> Rect {
        Rect::new(0, 0, screen.width, self.height)
    }

    #[inline]
    pub const fn apple_menu_rect(&self, screen: Size) -> Rect {
        let w = if self.apple_menu_width < screen.width {
            self.apple_menu_width
        } else {
            screen.width
        };
        Rect::new(0, 0, w, self.height)
    }

    #[inline]
    pub const fn status_area_rect(&self, screen: Size) -> Rect {
        let w = if self.status_area_width < screen.width {
            self.status_area_width
        } else {
            screen.width
        };
        let x = (screen.width - w) as i32;
        Rect::new(x, 0, w, self.height)
    }

    pub const fn hit_test(&self, screen: Size, point: Point) -> MenuBarHit {
        if !self.bounds(screen).contains(point) {
            return MenuBarHit::None;
        }
        if self.apple_menu_rect(screen).contains(point) {
            return MenuBarHit::AppleMenu;
        }
        if self.status_area_rect(screen).contains(point) {
            return MenuBarHit::StatusArea;
        }
        MenuBarHit::AppMenuArea
    }

    pub fn render(&self, surface: &mut Surface<'_>) {
        let screen = surface.size();
        let bar = self.bounds(screen);
        surface.fill_rect(bar, self.background);
        if self.height > 0 {
            surface.fill_rect(
                Rect::new(0, (self.height - 1) as i32, screen.width, 1),
                self.border_color,
            );
        }
        // Subtle system emblem dot in the Apple/Nexus menu slot.
        let apple = self.apple_menu_rect(screen);
        if apple.width >= 12 && apple.height >= 12 {
            surface.draw_circle(
                apple.x + (apple.width as i32) / 2,
                apple.y + (apple.height as i32) / 2,
                5,
                self.accent_color,
            );
        }
    }
}

/// Computed layout for a single dock icon.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DockItemLayout {
    pub index: usize,
    pub rect: Rect,
    /// Scale factor in permille (`1000` = 1.0x, `1500` = 1.5x).
    pub scale_permille: u32,
}

/// macOS-inspired bottom centered icon dock with magnification and spacing math.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dock {
    pub icon_size: u32,
    pub spacing: u32,
    pub padding: u32,
    pub bottom_margin: u32,
    pub corner_radius: u32,
    /// Maximum magnification in permille (`1000` = no magnification, `1500` = 1.5x).
    pub max_magnification_permille: u32,
    /// Horizontal influence radius in pixels for cursor magnification.
    pub magnification_radius: u32,
    pub background: Color,
    pub border_color: Color,
    pub icon_placeholder_color: Color,
}

impl Default for Dock {
    fn default() -> Self {
        Self {
            icon_size: 48,
            spacing: 10,
            padding: 10,
            bottom_margin: 8,
            corner_radius: 16,
            max_magnification_permille: 1500,
            magnification_radius: 96,
            background: Color::from_argb(0xB8, 0x24, 0x26, 0x2E),
            border_color: Color::from_argb(0x48, 0xFF, 0xFF, 0xFF),
            icon_placeholder_color: Color::from_argb(0xE0, 0x5E, 0x81, 0xAC),
        }
    }
}

impl Dock {
    /// Unmagnified dock container rectangle centered at the bottom of `screen`.
    pub fn dock_rect(&self, screen: Size, item_count: usize) -> Rect {
        if item_count == 0 {
            return Rect::ZERO;
        }
        let count = item_count as u32;
        let icons_w = count
            .saturating_mul(self.icon_size)
            .saturating_add(count.saturating_sub(1).saturating_mul(self.spacing));
        let total_w = icons_w.saturating_add(self.padding.saturating_mul(2));
        let total_h = self
            .icon_size
            .saturating_add(self.padding.saturating_mul(2));
        let x = ((screen.width as i64 - total_w as i64) / 2) as i32;
        let y = (screen.height as i64 - self.bottom_margin as i64 - total_h as i64) as i32;
        Rect::new(x, y, total_w, total_h)
    }

    /// Compute the magnification scale (in permille) for icon `index` given `cursor`.
    pub fn item_scale_permille(
        &self,
        screen: Size,
        item_count: usize,
        index: usize,
        cursor: Option<Point>,
    ) -> u32 {
        if index >= item_count
            || self.max_magnification_permille <= 1000
            || self.magnification_radius == 0
        {
            return 1000;
        }
        let Some(cursor) = cursor else {
            return 1000;
        };

        let base_dock = self.dock_rect(screen, item_count);
        let active_top = base_dock.y.saturating_sub(self.icon_size as i32);
        let active_bottom = base_dock.bottom() + self.bottom_margin as i64;
        if (cursor.y as i64) < active_top as i64 || (cursor.y as i64) > active_bottom {
            return 1000;
        }

        let step = (self.icon_size + self.spacing) as i64;
        let first_cx = base_dock.x as i64 + self.padding as i64 + (self.icon_size as i64) / 2;
        let cx = first_cx + (index as i64) * step;
        let dist = (cursor.x as i64 - cx).unsigned_abs() as u64;
        let radius = self.magnification_radius as u64;
        if dist >= radius {
            return 1000;
        }

        let t = radius - dist;
        let extra_max = (self.max_magnification_permille - 1000) as u64;
        let extra = (extra_max * t * t) / (radius * radius);
        1000 + extra as u32
    }

    /// Compute the layout of icon `index` (centered as a group, bottom-aligned to the dock baseline).
    pub fn item_layout(
        &self,
        screen: Size,
        item_count: usize,
        index: usize,
        cursor: Option<Point>,
    ) -> Option<DockItemLayout> {
        if index >= item_count {
            return None;
        }

        let mut total_icons_w: u32 = 0;
        let mut target_scale = 1000u32;
        let mut target_size = self.icon_size;
        let mut prefix_w: u32 = 0;

        for i in 0..item_count {
            let scale = self.item_scale_permille(screen, item_count, i, cursor);
            let size = (self.icon_size.saturating_mul(scale) + 500) / 1000;
            if i < index {
                prefix_w = prefix_w.saturating_add(size).saturating_add(self.spacing);
            } else if i == index {
                target_scale = scale;
                target_size = size;
            }
            total_icons_w = total_icons_w.saturating_add(size);
            if i + 1 < item_count {
                total_icons_w = total_icons_w.saturating_add(self.spacing);
            }
        }

        let base_dock = self.dock_rect(screen, item_count);
        let baseline_y = base_dock.bottom() - self.padding as i64;
        let start_x = ((screen.width as i64 - total_icons_w as i64) / 2) as i32;
        let x = start_x.saturating_add(prefix_w as i32);
        let y = (baseline_y - target_size as i64) as i32;

        Some(DockItemLayout {
            index,
            rect: Rect::new(x, y, target_size, target_size),
            scale_permille: target_scale,
        })
    }

    /// Hit-test which dock icon (if any) is at `point`.
    pub fn hit_test(
        &self,
        screen: Size,
        item_count: usize,
        point: Point,
        cursor: Option<Point>,
    ) -> Option<usize> {
        for i in 0..item_count {
            if let Some(layout) = self.item_layout(screen, item_count, i, cursor) {
                if layout.rect.contains(point) {
                    return Some(i);
                }
            }
        }
        None
    }

    pub fn render(&self, surface: &mut Surface<'_>, item_count: usize, cursor: Option<Point>) {
        if item_count == 0 {
            return;
        }
        let screen = surface.size();
        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        for i in 0..item_count {
            if let Some(layout) = self.item_layout(screen, item_count, i, cursor) {
                if layout.rect.x < min_x {
                    min_x = layout.rect.x;
                }
                let r = layout.rect.right() as i32;
                if r > max_x {
                    max_x = r;
                }
            }
        }

        let base_dock = self.dock_rect(screen, item_count);
        let bg_rect = Rect::new(
            min_x.saturating_sub(self.padding as i32),
            base_dock.y,
            (max_x - min_x).max(0) as u32 + self.padding.saturating_mul(2),
            base_dock.height,
        );
        surface.fill_rounded_rect(bg_rect, self.corner_radius, self.background);

        for i in 0..item_count {
            if let Some(layout) = self.item_layout(screen, item_count, i, cursor) {
                surface.fill_rounded_rect(
                    layout.rect,
                    (layout.rect.width / 5).max(4),
                    self.icon_placeholder_color,
                );
            }
        }
    }
}

/// Traffic-light window control buttons.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TrafficLightButton {
    Close,
    Minimize,
    Zoom,
}

/// Hit-test result for a `WindowFrame`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WindowHit {
    Outside,
    CloseButton,
    MinimizeButton,
    ZoomButton,
    Titlebar,
    Content,
}

/// macOS-inspired rounded-corner window frame with traffic-light buttons.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WindowFrame {
    pub bounds: Rect,
    pub titlebar_height: u32,
    pub corner_radius: u32,
    pub focused: bool,
    pub titlebar_color: Color,
    pub body_color: Color,
    pub border_color: Color,
}

impl WindowFrame {
    pub const TRAFFIC_LIGHT_RADIUS: u32 = 6;
    pub const TRAFFIC_LIGHT_SPACING: i32 = 20;
    pub const TRAFFIC_LIGHT_LEFT_MARGIN: i32 = 16;

    pub const CLOSE_COLOR: Color = Color::from_rgb(0xFF, 0x5F, 0x56);
    pub const MINIMIZE_COLOR: Color = Color::from_rgb(0xFF, 0xBD, 0x2E);
    pub const ZOOM_COLOR: Color = Color::from_rgb(0x27, 0xC9, 0x3F);
    pub const UNFOCUSED_BUTTON_COLOR: Color = Color::from_rgb(0x55, 0x58, 0x62);

    pub const fn new(bounds: Rect) -> Self {
        Self {
            bounds,
            titlebar_height: 28,
            corner_radius: 10,
            focused: true,
            titlebar_color: Color::from_rgb(0x2B, 0x2D, 0x36),
            body_color: Color::from_rgb(0x1E, 0x20, 0x28),
            border_color: Color::from_argb(0x60, 0xFF, 0xFF, 0xFF),
        }
    }

    #[inline]
    pub const fn titlebar_rect(&self) -> Rect {
        let h = if self.titlebar_height < self.bounds.height {
            self.titlebar_height
        } else {
            self.bounds.height
        };
        Rect::new(self.bounds.x, self.bounds.y, self.bounds.width, h)
    }

    #[inline]
    pub const fn content_rect(&self) -> Rect {
        let tb_h = self.titlebar_rect().height;
        Rect::new(
            self.bounds.x,
            self.bounds.y.saturating_add(tb_h as i32),
            self.bounds.width,
            self.bounds.height.saturating_sub(tb_h),
        )
    }

    #[inline]
    pub const fn traffic_light_center(&self, button: TrafficLightButton) -> Point {
        let idx = match button {
            TrafficLightButton::Close => 0,
            TrafficLightButton::Minimize => 1,
            TrafficLightButton::Zoom => 2,
        };
        let cx =
            self.bounds.x + Self::TRAFFIC_LIGHT_LEFT_MARGIN + idx * Self::TRAFFIC_LIGHT_SPACING;
        let cy = self.bounds.y + (self.titlebar_rect().height as i32) / 2;
        Point::new(cx, cy)
    }

    pub const fn hit_test(&self, point: Point) -> WindowHit {
        if !self.bounds.contains(point) {
            return WindowHit::Outside;
        }
        if point_in_circle(
            point,
            self.traffic_light_center(TrafficLightButton::Close),
            Self::TRAFFIC_LIGHT_RADIUS,
        ) {
            return WindowHit::CloseButton;
        }
        if point_in_circle(
            point,
            self.traffic_light_center(TrafficLightButton::Minimize),
            Self::TRAFFIC_LIGHT_RADIUS,
        ) {
            return WindowHit::MinimizeButton;
        }
        if point_in_circle(
            point,
            self.traffic_light_center(TrafficLightButton::Zoom),
            Self::TRAFFIC_LIGHT_RADIUS,
        ) {
            return WindowHit::ZoomButton;
        }
        if self.titlebar_rect().contains(point) {
            return WindowHit::Titlebar;
        }
        WindowHit::Content
    }

    pub fn render(&self, surface: &mut Surface<'_>) {
        if self.bounds.is_empty() {
            return;
        }
        surface.fill_rounded_rect(self.bounds, self.corner_radius, self.body_color);
        let tb = self.titlebar_rect();
        surface.fill_rounded_rect(tb, self.corner_radius, self.titlebar_color);
        // Square off the bottom edge of the titlebar so it meets the content area cleanly.
        if tb.height > self.corner_radius {
            let strip_h = self.corner_radius.min(tb.height);
            surface.fill_rect(
                Rect::new(
                    tb.x,
                    (tb.bottom() - strip_h as i64) as i32,
                    tb.width,
                    strip_h,
                ),
                self.titlebar_color,
            );
        }

        let (c_close, c_min, c_zoom) = if self.focused {
            (Self::CLOSE_COLOR, Self::MINIMIZE_COLOR, Self::ZOOM_COLOR)
        } else {
            (
                Self::UNFOCUSED_BUTTON_COLOR,
                Self::UNFOCUSED_BUTTON_COLOR,
                Self::UNFOCUSED_BUTTON_COLOR,
            )
        };

        let p_close = self.traffic_light_center(TrafficLightButton::Close);
        let p_min = self.traffic_light_center(TrafficLightButton::Minimize);
        let p_zoom = self.traffic_light_center(TrafficLightButton::Zoom);

        surface.draw_circle(p_close.x, p_close.y, Self::TRAFFIC_LIGHT_RADIUS, c_close);
        surface.draw_circle(p_min.x, p_min.y, Self::TRAFFIC_LIGHT_RADIUS, c_min);
        surface.draw_circle(p_zoom.x, p_zoom.y, Self::TRAFFIC_LIGHT_RADIUS, c_zoom);
    }
}

/// Computes the usable desktop work area between `MenuBar` and `Dock`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DesktopLayout {
    pub screen: Size,
    pub menu_bar: MenuBar,
    pub dock: Dock,
}

impl DesktopLayout {
    pub const fn new(screen: Size, menu_bar: MenuBar, dock: Dock) -> Self {
        Self {
            screen,
            menu_bar,
            dock,
        }
    }

    pub fn work_area(&self, dock_item_count: usize) -> Rect {
        let top = self.menu_bar.height.min(self.screen.height);
        let dock_reserve = if dock_item_count == 0 {
            0
        } else {
            self.dock
                .dock_rect(self.screen, dock_item_count)
                .height
                .saturating_add(self.dock.bottom_margin)
        };
        let usable_h = self
            .screen
            .height
            .saturating_sub(top)
            .saturating_sub(dock_reserve);
        Rect::new(0, top as i32, self.screen.width, usable_h)
    }
}

#[inline]
const fn point_in_circle(p: Point, center: Point, radius: u32) -> bool {
    let dx = p.x as i64 - center.x as i64;
    let dy = p.y as i64 - center.y as i64;
    let r = radius as i64;
    dx * dx + dy * dy <= r * r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menu_bar_hit_testing() {
        let screen = Size::new(1280, 800);
        let bar = MenuBar::default();
        assert_eq!(
            bar.hit_test(screen, Point::new(10, 10)),
            MenuBarHit::AppleMenu
        );
        assert_eq!(
            bar.hit_test(screen, Point::new(200, 10)),
            MenuBarHit::AppMenuArea
        );
        assert_eq!(
            bar.hit_test(screen, Point::new(1200, 10)),
            MenuBarHit::StatusArea
        );
        assert_eq!(bar.hit_test(screen, Point::new(200, 40)), MenuBarHit::None);
    }

    #[test]
    fn dock_centering_and_magnification() {
        let screen = Size::new(1280, 800);
        let dock = Dock::default();
        let rect = dock.dock_rect(screen, 5);
        // Centered horizontally
        assert_eq!(rect.x + (rect.width as i32) / 2, 640);

        // Hovering over center icon (index 2) magnifies it more than outer icons
        let center_item = dock.item_layout(screen, 5, 2, None).unwrap();
        let cursor = Point::new(
            center_item.rect.x + (center_item.rect.width as i32) / 2,
            center_item.rect.y + (center_item.rect.height as i32) / 2,
        );
        let mag_center = dock.item_layout(screen, 5, 2, Some(cursor)).unwrap();
        let mag_edge = dock.item_layout(screen, 5, 0, Some(cursor)).unwrap();
        assert_eq!(mag_center.scale_permille, dock.max_magnification_permille);
        assert!(mag_center.rect.width > mag_edge.rect.width);
        assert_eq!(dock.hit_test(screen, 5, cursor, Some(cursor)), Some(2));
    }

    #[test]
    fn window_frame_traffic_light_hit_testing() {
        let frame = WindowFrame::new(Rect::new(100, 80, 400, 300));
        let close = frame.traffic_light_center(TrafficLightButton::Close);
        let min = frame.traffic_light_center(TrafficLightButton::Minimize);
        let zoom = frame.traffic_light_center(TrafficLightButton::Zoom);

        assert_eq!(frame.hit_test(close), WindowHit::CloseButton);
        assert_eq!(frame.hit_test(min), WindowHit::MinimizeButton);
        assert_eq!(frame.hit_test(zoom), WindowHit::ZoomButton);
        assert_eq!(frame.hit_test(Point::new(300, 90)), WindowHit::Titlebar);
        assert_eq!(frame.hit_test(Point::new(300, 200)), WindowHit::Content);
        assert_eq!(frame.hit_test(Point::new(10, 10)), WindowHit::Outside);
    }

    #[test]
    fn desktop_work_area_excludes_bar_and_dock() {
        let layout = DesktopLayout::new(Size::new(1280, 800), MenuBar::default(), Dock::default());
        let work = layout.work_area(5);
        assert_eq!(work.y, 28);
        assert!(work.height < 800 - 28);
    }
}
