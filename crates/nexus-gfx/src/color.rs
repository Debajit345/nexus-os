/// ARGB8888 color packed as `0xAARRGGBB`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Color(pub u32);

impl Color {
    pub const TRANSPARENT: Self = Self(0x0000_0000);
    pub const BLACK: Self = Self(0xFF00_0000);
    pub const WHITE: Self = Self(0xFFFF_FFFF);
    pub const RED: Self = Self(0xFFFF_0000);
    pub const GREEN: Self = Self(0xFF00_FF00);
    pub const BLUE: Self = Self(0xFF00_00FF);

    #[inline]
    pub const fn from_argb(a: u8, r: u8, g: u8, b: u8) -> Self {
        Self(((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32))
    }

    #[inline]
    pub const fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self::from_argb(0xFF, r, g, b)
    }

    #[inline]
    pub const fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::from_argb(a, r, g, b)
    }

    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn a(self) -> u8 {
        ((self.0 >> 24) & 0xFF) as u8
    }

    #[inline]
    pub const fn r(self) -> u8 {
        ((self.0 >> 16) & 0xFF) as u8
    }

    #[inline]
    pub const fn g(self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }

    #[inline]
    pub const fn b(self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    #[inline]
    pub const fn with_alpha(self, a: u8) -> Self {
        Self((self.0 & 0x00FF_FFFF) | ((a as u32) << 24))
    }

    #[inline]
    pub const fn is_opaque(self) -> bool {
        self.a() == 0xFF
    }

    #[inline]
    pub const fn is_transparent(self) -> bool {
        self.a() == 0
    }

    /// Porter-Duff source-over alpha blending (`self` over `dst`).
    #[inline]
    pub const fn blend_over(self, dst: Self) -> Self {
        let sa = self.a() as u32;
        if sa == 255 {
            return self;
        }
        if sa == 0 {
            return dst;
        }

        let inv_sa = 255 - sa;
        let da = dst.a() as u32;

        if da == 255 {
            let r = div255_round(self.r() as u32 * sa + dst.r() as u32 * inv_sa);
            let g = div255_round(self.g() as u32 * sa + dst.g() as u32 * inv_sa);
            let b = div255_round(self.b() as u32 * sa + dst.b() as u32 * inv_sa);
            return Self::from_argb(255, r as u8, g as u8, b as u8);
        }

        let out_a = sa + div255_round(da * inv_sa);
        if out_a == 0 {
            return Self::TRANSPARENT;
        }

        let den = out_a * 255;
        let half = den / 2;
        let r = (self.r() as u32 * sa * 255 + dst.r() as u32 * da * inv_sa + half) / den;
        let g = (self.g() as u32 * sa * 255 + dst.g() as u32 * da * inv_sa + half) / den;
        let b = (self.b() as u32 * sa * 255 + dst.b() as u32 * da * inv_sa + half) / den;

        Self::from_argb(out_a as u8, r as u8, g as u8, b as u8)
    }

    /// Linearly interpolate between `self` (`t = 0`) and `other` (`t = 255`).
    #[inline]
    pub const fn lerp(self, other: Self, t: u8) -> Self {
        let t = t as u32;
        let inv = 255 - t;
        let a = div255_round(self.a() as u32 * inv + other.a() as u32 * t);
        let r = div255_round(self.r() as u32 * inv + other.r() as u32 * t);
        let g = div255_round(self.g() as u32 * inv + other.g() as u32 * t);
        let b = div255_round(self.b() as u32 * inv + other.b() as u32 * t);
        Self::from_argb(a as u8, r as u8, g as u8, b as u8)
    }
}

#[inline]
const fn div255_round(x: u32) -> u32 {
    (x + 127) / 255
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_roundtrip() {
        let c = Color::from_argb(0x80, 0x12, 0x34, 0x56);
        assert_eq!(c.a(), 0x80);
        assert_eq!(c.r(), 0x12);
        assert_eq!(c.g(), 0x34);
        assert_eq!(c.b(), 0x56);
        assert_eq!(c.as_raw(), 0x8012_3456);
    }

    #[test]
    fn blend_over_opaque_and_transparent() {
        let src = Color::from_argb(0xFF, 10, 20, 30);
        let dst = Color::from_argb(0xFF, 200, 210, 220);
        assert_eq!(src.blend_over(dst), src);
        assert_eq!(Color::TRANSPARENT.blend_over(dst), dst);
    }

    #[test]
    fn blend_over_half_alpha_on_opaque() {
        let src = Color::from_argb(128, 255, 0, 0);
        let dst = Color::from_rgb(0, 0, 255);
        let out = src.blend_over(dst);
        assert_eq!(out.a(), 255);
        assert!((out.r() as i32 - 128).abs() <= 1);
        assert_eq!(out.g(), 0);
        assert!((out.b() as i32 - 127).abs() <= 1);
    }

    #[test]
    fn blend_over_translucent_on_translucent() {
        let src = Color::from_argb(128, 200, 100, 50);
        let dst = Color::from_argb(128, 20, 40, 60);
        let out = src.blend_over(dst);
        // out_a = 128 + 128 * 127 / 255 = 192
        assert!((out.a() as i32 - 192).abs() <= 1);
        assert!(out.r() > dst.r() && out.r() < src.r());
    }
}
