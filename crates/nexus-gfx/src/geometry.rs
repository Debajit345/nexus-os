/// 2D integer point.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    #[inline]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub const fn offset(self, dx: i32, dy: i32) -> Self {
        Self {
            x: self.x.saturating_add(dx),
            y: self.y.saturating_add(dy),
        }
    }
}

/// 2D unsigned size.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    #[inline]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    #[inline]
    pub const fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// 2D axis-aligned rectangle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub const ZERO: Self = Self {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    };

    #[inline]
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    #[inline]
    pub const fn from_origin_size(origin: Point, size: Size) -> Self {
        Self {
            x: origin.x,
            y: origin.y,
            width: size.width,
            height: size.height,
        }
    }

    #[inline]
    pub const fn origin(self) -> Point {
        Point::new(self.x, self.y)
    }

    #[inline]
    pub const fn size(self) -> Size {
        Size::new(self.width, self.height)
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    #[inline]
    pub const fn right(self) -> i64 {
        self.x as i64 + self.width as i64
    }

    #[inline]
    pub const fn bottom(self) -> i64 {
        self.y as i64 + self.height as i64
    }

    #[inline]
    pub const fn contains(self, p: Point) -> bool {
        if self.is_empty() {
            return false;
        }
        let px = p.x as i64;
        let py = p.y as i64;
        px >= self.x as i64 && px < self.right() && py >= self.y as i64 && py < self.bottom()
    }

    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        (self.x as i64) < other.right()
            && self.right() > (other.x as i64)
            && (self.y as i64) < other.bottom()
            && self.bottom() > (other.y as i64)
    }

    #[inline]
    pub const fn intersection(self, other: Self) -> Option<Self> {
        if !self.intersects(other) {
            return None;
        }
        let x0 = if self.x > other.x { self.x } else { other.x };
        let y0 = if self.y > other.y { self.y } else { other.y };
        let r0 = self.right();
        let r1 = other.right();
        let x1 = if r0 < r1 { r0 } else { r1 };
        let b0 = self.bottom();
        let b1 = other.bottom();
        let y1 = if b0 < b1 { b0 } else { b1 };

        let w = (x1 - x0 as i64) as u32;
        let h = (y1 - y0 as i64) as u32;
        Some(Self::new(x0, y0, w, h))
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        let x0 = if self.x < other.x { self.x } else { other.x };
        let y0 = if self.y < other.y { self.y } else { other.y };
        let r0 = self.right();
        let r1 = other.right();
        let x1 = if r0 > r1 { r0 } else { r1 };
        let b0 = self.bottom();
        let b1 = other.bottom();
        let y1 = if b0 > b1 { b0 } else { b1 };

        Self::new(x0, y0, (x1 - x0 as i64) as u32, (y1 - y0 as i64) as u32)
    }

    #[inline]
    pub const fn inset(self, dx: i32, dy: i32) -> Self {
        let nx = self.x.saturating_add(dx);
        let ny = self.y.saturating_add(dy);
        let raw_w = self.width as i64 - 2 * dx as i64;
        let raw_h = self.height as i64 - 2 * dy as i64;
        let nw = if raw_w > 0 { raw_w as u32 } else { 0 };
        let nh = if raw_h > 0 { raw_h as u32 } else { 0 };
        Self::new(nx, ny, nw, nh)
    }

    #[inline]
    pub const fn offset(self, dx: i32, dy: i32) -> Self {
        Self::new(
            self.x.saturating_add(dx),
            self.y.saturating_add(dy),
            self.width,
            self.height,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_contains_and_intersection() {
        let a = Rect::new(10, 20, 100, 50);
        let b = Rect::new(50, 40, 100, 50);
        assert!(a.contains(Point::new(10, 20)));
        assert!(!a.contains(Point::new(110, 20)));
        assert!(a.intersects(b));
        assert_eq!(a.intersection(b), Some(Rect::new(50, 40, 60, 30)));
        assert_eq!(a.union(b), Rect::new(10, 20, 140, 70));
    }

    #[test]
    fn rect_inset_clamps_to_zero() {
        let r = Rect::new(0, 0, 10, 10);
        assert_eq!(r.inset(6, 6), Rect::new(6, 6, 0, 0));
    }
}
