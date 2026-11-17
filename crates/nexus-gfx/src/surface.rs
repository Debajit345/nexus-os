use crate::color::Color;
use crate::geometry::{Rect, Size};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SurfaceError {
    InvalidDimensions,
    StrideTooSmall,
    BufferTooSmall,
}

/// Immutable view over an ARGB8888 pixel buffer.
#[derive(Copy, Clone, Debug)]
pub struct SurfaceRef<'a> {
    pixels: &'a [u32],
    width: u32,
    height: u32,
    stride: u32,
}

impl<'a> SurfaceRef<'a> {
    pub fn new(
        pixels: &'a [u32],
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<Self, SurfaceError> {
        validate_layout(pixels.len(), width, height, stride)?;
        Ok(Self {
            pixels,
            width,
            height,
            stride,
        })
    }

    #[inline]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    pub const fn stride(&self) -> u32 {
        self.stride
    }

    #[inline]
    pub const fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }

    #[inline]
    pub const fn bounds(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }

    #[inline]
    pub fn pixels(&self) -> &[u32] {
        self.pixels
    }

    #[inline]
    pub fn get_pixel(&self, x: i32, y: i32) -> Option<Color> {
        if x < 0 || y < 0 || (x as u32) >= self.width || (y as u32) >= self.height {
            return None;
        }
        let idx = (y as usize) * (self.stride as usize) + (x as usize);
        Some(Color::from_raw(self.pixels[idx]))
    }
}

/// Mutable slice-backed ARGB8888 pixel surface.
#[derive(Debug)]
pub struct Surface<'a> {
    pixels: &'a mut [u32],
    width: u32,
    height: u32,
    stride: u32,
}

impl<'a> Surface<'a> {
    pub fn new(
        pixels: &'a mut [u32],
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<Self, SurfaceError> {
        validate_layout(pixels.len(), width, height, stride)?;
        Ok(Self {
            pixels,
            width,
            height,
            stride,
        })
    }

    #[inline]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    pub const fn stride(&self) -> u32 {
        self.stride
    }

    #[inline]
    pub const fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }

    #[inline]
    pub const fn bounds(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }

    #[inline]
    pub fn as_ref(&self) -> SurfaceRef<'_> {
        SurfaceRef {
            pixels: self.pixels,
            width: self.width,
            height: self.height,
            stride: self.stride,
        }
    }

    #[inline]
    pub fn pixels(&self) -> &[u32] {
        self.pixels
    }

    #[inline]
    pub fn pixels_mut(&mut self) -> &mut [u32] {
        self.pixels
    }

    #[inline]
    pub fn get_pixel(&self, x: i32, y: i32) -> Option<Color> {
        self.as_ref().get_pixel(x, y)
    }

    #[inline]
    pub fn put_pixel(&mut self, x: i32, y: i32, color: Color) {
        if x < 0 || y < 0 || (x as u32) >= self.width || (y as u32) >= self.height {
            return;
        }
        let idx = (y as usize) * (self.stride as usize) + (x as usize);
        self.pixels[idx] = color.as_raw();
    }

    #[inline]
    pub fn blend_pixel(&mut self, x: i32, y: i32, color: Color) {
        if color.is_transparent() {
            return;
        }
        if x < 0 || y < 0 || (x as u32) >= self.width || (y as u32) >= self.height {
            return;
        }
        let idx = (y as usize) * (self.stride as usize) + (x as usize);
        if color.is_opaque() {
            self.pixels[idx] = color.as_raw();
        } else {
            let dst = Color::from_raw(self.pixels[idx]);
            self.pixels[idx] = color.blend_over(dst).as_raw();
        }
    }

    pub fn clear(&mut self, color: Color) {
        let raw = color.as_raw();
        let w = self.width as usize;
        let stride = self.stride as usize;
        for y in 0..(self.height as usize) {
            let row_start = y * stride;
            self.pixels[row_start..row_start + w].fill(raw);
        }
    }

    pub fn fill_rect(&mut self, rect: Rect, color: Color) {
        if color.is_transparent() {
            return;
        }
        let Some(clipped) = rect.intersection(self.bounds()) else {
            return;
        };
        let x0 = clipped.x as usize;
        let y0 = clipped.y as usize;
        let w = clipped.width as usize;
        let h = clipped.height as usize;
        let stride = self.stride as usize;

        if color.is_opaque() {
            let raw = color.as_raw();
            for row in 0..h {
                let start = (y0 + row) * stride + x0;
                self.pixels[start..start + w].fill(raw);
            }
        } else {
            for row in 0..h {
                let start = (y0 + row) * stride + x0;
                for px in &mut self.pixels[start..start + w] {
                    let dst = Color::from_raw(*px);
                    *px = color.blend_over(dst).as_raw();
                }
            }
        }
    }

    pub fn stroke_rect(&mut self, rect: Rect, thickness: u32, color: Color) {
        if rect.is_empty() || thickness == 0 || color.is_transparent() {
            return;
        }
        let t = thickness.min(rect.width).min(rect.height);
        // Top
        self.fill_rect(Rect::new(rect.x, rect.y, rect.width, t), color);
        // Bottom
        let bottom_y = (rect.bottom() - t as i64) as i32;
        self.fill_rect(Rect::new(rect.x, bottom_y, rect.width, t), color);
        // Left & Right (excluding top/bottom overlap)
        let mid_h = rect.height.saturating_sub(2 * t);
        if mid_h > 0 {
            let mid_y = rect.y.saturating_add(t as i32);
            self.fill_rect(Rect::new(rect.x, mid_y, t, mid_h), color);
            let right_x = (rect.right() - t as i64) as i32;
            self.fill_rect(Rect::new(right_x, mid_y, t, mid_h), color);
        }
    }

    pub fn fill_rounded_rect(&mut self, rect: Rect, radius: u32, color: Color) {
        if rect.is_empty() || color.is_transparent() {
            return;
        }
        let max_r = (rect.width.min(rect.height)) / 2;
        let r = radius.min(max_r);
        if r == 0 {
            self.fill_rect(rect, color);
            return;
        }

        let Some(clipped) = rect.intersection(self.bounds()) else {
            return;
        };

        let r_i64 = r as i64;
        let r_sq = r_i64 * r_i64;
        let left_cx = rect.x as i64 + r_i64;
        let right_cx = rect.right() - r_i64 - 1;
        let top_cy = rect.y as i64 + r_i64;
        let bottom_cy = rect.bottom() - r_i64 - 1;

        for py in clipped.y..(clipped.bottom() as i32) {
            let py64 = py as i64;
            for px in clipped.x..(clipped.right() as i32) {
                let px64 = px as i64;
                let dx = if px64 < left_cx {
                    left_cx - px64
                } else if px64 > right_cx {
                    px64 - right_cx
                } else {
                    0
                };
                let dy = if py64 < top_cy {
                    top_cy - py64
                } else if py64 > bottom_cy {
                    py64 - bottom_cy
                } else {
                    0
                };

                if dx == 0 || dy == 0 || (dx * dx + dy * dy) <= r_sq {
                    self.blend_pixel(px, py, color);
                }
            }
        }
    }

    pub fn draw_circle(&mut self, cx: i32, cy: i32, radius: u32, color: Color) {
        if radius == 0 || color.is_transparent() {
            return;
        }
        let r = radius as i64;
        let r_sq = r * r;
        let bbox = Rect::new(
            cx.saturating_sub(radius as i32),
            cy.saturating_sub(radius as i32),
            radius.saturating_mul(2).saturating_add(1),
            radius.saturating_mul(2).saturating_add(1),
        );
        let Some(clipped) = bbox.intersection(self.bounds()) else {
            return;
        };

        for py in clipped.y..(clipped.bottom() as i32) {
            let dy = py as i64 - cy as i64;
            for px in clipped.x..(clipped.right() as i32) {
                let dx = px as i64 - cx as i64;
                if dx * dx + dy * dy <= r_sq {
                    self.blend_pixel(px, py, color);
                }
            }
        }
    }

    pub fn blit(&mut self, src: &SurfaceRef<'_>, src_rect: Rect, dst_x: i32, dst_y: i32) {
        self.blit_internal(src, src_rect, dst_x, dst_y, false);
    }

    pub fn blit_blend(&mut self, src: &SurfaceRef<'_>, src_rect: Rect, dst_x: i32, dst_y: i32) {
        self.blit_internal(src, src_rect, dst_x, dst_y, true);
    }

    fn blit_internal(
        &mut self,
        src: &SurfaceRef<'_>,
        src_rect: Rect,
        dst_x: i32,
        dst_y: i32,
        blend: bool,
    ) {
        let Some(src_clipped) = src_rect.intersection(src.bounds()) else {
            return;
        };
        let dx_shift = src_clipped.x - src_rect.x;
        let dy_shift = src_clipped.y - src_rect.y;
        let dst_rect = Rect::new(
            dst_x.saturating_add(dx_shift),
            dst_y.saturating_add(dy_shift),
            src_clipped.width,
            src_clipped.height,
        );
        let Some(dst_clipped) = dst_rect.intersection(self.bounds()) else {
            return;
        };

        let src_x0 = (src_clipped.x + (dst_clipped.x - dst_rect.x)) as usize;
        let src_y0 = (src_clipped.y + (dst_clipped.y - dst_rect.y)) as usize;
        let dst_x0 = dst_clipped.x as usize;
        let dst_y0 = dst_clipped.y as usize;
        let w = dst_clipped.width as usize;
        let h = dst_clipped.height as usize;

        let src_stride = src.stride() as usize;
        let dst_stride = self.stride as usize;
        let src_pixels = src.pixels();

        for row in 0..h {
            let s_start = (src_y0 + row) * src_stride + src_x0;
            let d_start = (dst_y0 + row) * dst_stride + dst_x0;
            if !blend {
                self.pixels[d_start..d_start + w]
                    .copy_from_slice(&src_pixels[s_start..s_start + w]);
            } else {
                for col in 0..w {
                    let sc = Color::from_raw(src_pixels[s_start + col]);
                    if sc.is_transparent() {
                        continue;
                    }
                    if sc.is_opaque() {
                        self.pixels[d_start + col] = sc.as_raw();
                    } else {
                        let dc = Color::from_raw(self.pixels[d_start + col]);
                        self.pixels[d_start + col] = sc.blend_over(dc).as_raw();
                    }
                }
            }
        }
    }
}

fn validate_layout(len: usize, width: u32, height: u32, stride: u32) -> Result<(), SurfaceError> {
    if width == 0 || height == 0 {
        return Err(SurfaceError::InvalidDimensions);
    }
    if stride < width {
        return Err(SurfaceError::StrideTooSmall);
    }
    let required = (height as usize)
        .checked_sub(1)
        .and_then(|h| h.checked_mul(stride as usize))
        .and_then(|off| off.checked_add(width as usize))
        .ok_or(SurfaceError::BufferTooSmall)?;
    if len < required {
        return Err(SurfaceError::BufferTooSmall);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_fill_and_blend() {
        let mut buf = [0u32; 16 * 16];
        let mut surf = Surface::new(&mut buf, 16, 16, 16).unwrap();
        surf.clear(Color::BLACK);
        surf.fill_rect(Rect::new(2, 2, 4, 4), Color::from_argb(128, 255, 0, 0));
        let px = surf.get_pixel(3, 3).unwrap();
        assert_eq!(px.a(), 255);
        assert!((px.r() as i32 - 128).abs() <= 1);
        assert_eq!(surf.get_pixel(0, 0), Some(Color::BLACK));
    }

    #[test]
    fn surface_blit_blend_clips_cleanly() {
        let mut src_buf = [Color::from_argb(128, 0, 255, 0).as_raw(); 4 * 4];
        let src = SurfaceRef::new(&src_buf, 4, 4, 4).unwrap();
        let mut dst_buf = [Color::BLUE.as_raw(); 8 * 8];
        let mut dst = Surface::new(&mut dst_buf, 8, 8, 8).unwrap();

        dst.blit_blend(&src, Rect::new(0, 0, 4, 4), 6, 6);
        let blended = dst.get_pixel(7, 7).unwrap();
        assert!(blended.g() > 100 && blended.b() > 100);
        assert_eq!(dst.get_pixel(0, 0), Some(Color::BLUE));
        let _ = &mut src_buf;
    }

    #[test]
    fn surface_rounded_rect_cuts_corners() {
        let mut buf = [Color::BLACK.as_raw(); 20 * 20];
        let mut surf = Surface::new(&mut buf, 20, 20, 20).unwrap();
        surf.fill_rounded_rect(Rect::new(0, 0, 20, 20), 6, Color::WHITE);
        // Corner (0, 0) should remain black, center (10, 10) should be white.
        assert_eq!(surf.get_pixel(0, 0), Some(Color::BLACK));
        assert_eq!(surf.get_pixel(10, 10), Some(Color::WHITE));
    }
}
