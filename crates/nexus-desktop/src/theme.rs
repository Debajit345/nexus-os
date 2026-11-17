use nexus_gfx::{Color, Dock, MenuBar, WindowFrame};

/// Light or dark appearance mode for the desktop environment.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ThemeMode {
    Dark,
    Light,
}

/// macOS-inspired glassmorphic desktop theme palette.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DesktopTheme {
    pub mode: ThemeMode,
    pub wallpaper_top: Color,
    pub wallpaper_bottom: Color,
    pub accent_color: Color,
    pub glass_tint: Color,
    pub glass_border: Color,
    pub window_titlebar: Color,
    pub window_titlebar_unfocused: Color,
    pub window_body: Color,
    pub menu_bar_bg: Color,
    pub dock_bg: Color,
    pub corner_radius: u32,
}

impl DesktopTheme {
    /// Default dark glass theme.
    pub const fn dark() -> Self {
        Self {
            mode: ThemeMode::Dark,
            wallpaper_top: Color::from_rgb(0x14, 0x1B, 0x2D),
            wallpaper_bottom: Color::from_rgb(0x2A, 0x1F, 0x3D),
            accent_color: Color::from_rgb(0x0A, 0x84, 0xFF),
            glass_tint: Color::from_argb(0xC0, 0x1E, 0x20, 0x28),
            glass_border: Color::from_argb(0x48, 0xFF, 0xFF, 0xFF),
            window_titlebar: Color::from_rgb(0x2B, 0x2D, 0x36),
            window_titlebar_unfocused: Color::from_rgb(0x23, 0x25, 0x2C),
            window_body: Color::from_rgb(0x1E, 0x20, 0x28),
            menu_bar_bg: Color::from_argb(0xD8, 0x1E, 0x1E, 0x24),
            dock_bg: Color::from_argb(0xB8, 0x24, 0x26, 0x2E),
            corner_radius: 12,
        }
    }

    /// Light frosted glass theme.
    pub const fn light() -> Self {
        Self {
            mode: ThemeMode::Light,
            wallpaper_top: Color::from_rgb(0xD8, 0xE6, 0xF8),
            wallpaper_bottom: Color::from_rgb(0xF2, 0xE4, 0xF5),
            accent_color: Color::from_rgb(0x00, 0x7A, 0xFF),
            glass_tint: Color::from_argb(0xCC, 0xF4, 0xF5, 0xF8),
            glass_border: Color::from_argb(0x60, 0xFF, 0xFF, 0xFF),
            window_titlebar: Color::from_rgb(0xEC, 0xEE, 0xF2),
            window_titlebar_unfocused: Color::from_rgb(0xE2, 0xE4, 0xE8),
            window_body: Color::from_rgb(0xFA, 0xFB, 0xFD),
            menu_bar_bg: Color::from_argb(0xD8, 0xF2, 0xF4, 0xF8),
            dock_bg: Color::from_argb(0xB8, 0xEC, 0xEE, 0xF4),
            corner_radius: 12,
        }
    }

    /// Build a `MenuBar` styled with this theme.
    pub fn menu_bar(&self) -> MenuBar {
        MenuBar {
            background: self.menu_bar_bg,
            border_color: self.glass_border,
            accent_color: self.accent_color,
            ..MenuBar::default()
        }
    }

    /// Build a `Dock` styled with this theme.
    pub fn dock(&self) -> Dock {
        Dock {
            corner_radius: self.corner_radius.saturating_add(4),
            background: self.dock_bg,
            border_color: self.glass_border,
            icon_placeholder_color: self.accent_color,
            ..Dock::default()
        }
    }

    /// Apply theme colors and corner radius to a `WindowFrame`.
    pub fn apply_to_frame(&self, frame: &mut WindowFrame, focused: bool) {
        frame.focused = focused;
        frame.corner_radius = self.corner_radius;
        frame.titlebar_color = if focused {
            self.window_titlebar
        } else {
            self.window_titlebar_unfocused
        };
        frame.body_color = self.window_body;
        frame.border_color = self.glass_border;
    }
}

impl Default for DesktopTheme {
    fn default() -> Self {
        Self::dark()
    }
}
