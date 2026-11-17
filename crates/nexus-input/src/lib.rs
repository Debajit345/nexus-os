#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Clean-room `no_std` + `alloc` HID event router and cross-platform shortcut matcher for Nexus OS.

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// Logical keyboard key codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    Space,
    Enter,
    Escape,
    Tab,
    Backspace,
    Delete,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Unknown(u32),
}

/// Keyboard modifier state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub cmd: bool,
}

impl Modifiers {
    /// No modifiers pressed.
    pub const fn none() -> Self {
        Self {
            shift: false,
            ctrl: false,
            alt: false,
            cmd: false,
        }
    }

    /// Only `ctrl` pressed.
    pub const fn ctrl() -> Self {
        Self {
            shift: false,
            ctrl: true,
            alt: false,
            cmd: false,
        }
    }

    /// Only `cmd` (Super/Command) pressed.
    pub const fn cmd() -> Self {
        Self {
            shift: false,
            ctrl: false,
            alt: false,
            cmd: true,
        }
    }

    /// Only `alt` (Option/Alt) pressed.
    pub const fn alt() -> Self {
        Self {
            shift: false,
            ctrl: false,
            alt: true,
            cmd: false,
        }
    }

    /// Only `shift` pressed.
    pub const fn shift() -> Self {
        Self {
            shift: true,
            ctrl: false,
            alt: false,
            cmd: false,
        }
    }
}

/// Mouse button identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Other(u8),
}

/// Normalized HID input events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputEvent {
    KeyDown {
        key: KeyCode,
        modifiers: Modifiers,
    },
    KeyUp {
        key: KeyCode,
        modifiers: Modifiers,
    },
    MouseMove {
        x: i32,
        y: i32,
    },
    MouseDown {
        button: MouseButton,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    MouseUp {
        button: MouseButton,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    Scroll {
        delta_x: i32,
        delta_y: i32,
        modifiers: Modifiers,
    },
}

/// FIFO queue for HID input events.
#[derive(Debug, Clone, Default)]
pub struct InputQueue {
    events: VecDeque<InputEvent>,
}

impl InputQueue {
    /// Creates an empty input queue.
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
        }
    }

    /// Creates an empty input queue with preallocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
        }
    }

    /// Pushes an event to the back of the queue.
    pub fn push(&mut self, event: InputEvent) {
        self.events.push_back(event);
    }

    /// Pops the oldest event from the front of the queue.
    pub fn pop(&mut self) -> Option<InputEvent> {
        self.events.pop_front()
    }

    /// Peeks at the oldest event without removing it.
    pub fn peek(&self) -> Option<&InputEvent> {
        self.events.front()
    }

    /// Returns the number of queued events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Clears all queued events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Drains all queued events in FIFO order.
    pub fn drain_all(&mut self) -> Vec<InputEvent> {
        self.events.drain(..).collect()
    }
}

/// High-level desktop shortcut actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShortcutAction {
    Quit,
    Spotlight,
    Copy,
    Paste,
    SwitchWindow,
}

/// Matches a key + modifier combination against macOS (`cmd`), Linux, and Windows (`ctrl`/`alt`) shortcuts.
pub fn match_shortcut(key: KeyCode, mods: Modifiers) -> Option<ShortcutAction> {
    let primary = (mods.cmd || mods.ctrl) && !mods.alt && !mods.shift;

    match key {
        KeyCode::Q if primary => Some(ShortcutAction::Quit),
        KeyCode::F4 if mods.alt && !mods.ctrl && !mods.cmd && !mods.shift => {
            Some(ShortcutAction::Quit)
        }
        KeyCode::Space if primary => Some(ShortcutAction::Spotlight),
        KeyCode::C if primary => Some(ShortcutAction::Copy),
        KeyCode::V if primary => Some(ShortcutAction::Paste),
        KeyCode::Tab if (mods.cmd || mods.alt) && !mods.ctrl && !mods.shift => {
            Some(ShortcutAction::SwitchWindow)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_queue_preserves_fifo_order() {
        let mut queue = InputQueue::new();
        assert!(queue.is_empty());

        let e1 = InputEvent::KeyDown {
            key: KeyCode::A,
            modifiers: Modifiers::none(),
        };
        let e2 = InputEvent::MouseMove { x: 120, y: 240 };
        let e3 = InputEvent::MouseDown {
            button: MouseButton::Left,
            x: 120,
            y: 240,
            modifiers: Modifiers::shift(),
        };
        let e4 = InputEvent::Scroll {
            delta_x: 0,
            delta_y: -3,
            modifiers: Modifiers::none(),
        };

        queue.push(e1);
        queue.push(e2);
        queue.push(e3);
        queue.push(e4);

        assert_eq!(queue.len(), 4);
        assert_eq!(queue.peek(), Some(&e1));
        assert_eq!(queue.pop(), Some(e1));
        assert_eq!(queue.pop(), Some(e2));
        assert_eq!(queue.pop(), Some(e3));
        assert_eq!(queue.pop(), Some(e4));
        assert_eq!(queue.pop(), None);
        assert!(queue.is_empty());
    }

    #[test]
    fn macos_cmd_shortcuts_match() {
        assert_eq!(
            match_shortcut(KeyCode::Q, Modifiers::cmd()),
            Some(ShortcutAction::Quit)
        );
        assert_eq!(
            match_shortcut(KeyCode::Space, Modifiers::cmd()),
            Some(ShortcutAction::Spotlight)
        );
        assert_eq!(
            match_shortcut(KeyCode::C, Modifiers::cmd()),
            Some(ShortcutAction::Copy)
        );
        assert_eq!(
            match_shortcut(KeyCode::V, Modifiers::cmd()),
            Some(ShortcutAction::Paste)
        );
        assert_eq!(
            match_shortcut(KeyCode::Tab, Modifiers::cmd()),
            Some(ShortcutAction::SwitchWindow)
        );
    }

    #[test]
    fn linux_and_windows_shortcuts_match() {
        assert_eq!(
            match_shortcut(KeyCode::Q, Modifiers::ctrl()),
            Some(ShortcutAction::Quit)
        );
        assert_eq!(
            match_shortcut(KeyCode::F4, Modifiers::alt()),
            Some(ShortcutAction::Quit)
        );
        assert_eq!(
            match_shortcut(KeyCode::Space, Modifiers::ctrl()),
            Some(ShortcutAction::Spotlight)
        );
        assert_eq!(
            match_shortcut(KeyCode::C, Modifiers::ctrl()),
            Some(ShortcutAction::Copy)
        );
        assert_eq!(
            match_shortcut(KeyCode::V, Modifiers::ctrl()),
            Some(ShortcutAction::Paste)
        );
        assert_eq!(
            match_shortcut(KeyCode::Tab, Modifiers::alt()),
            Some(ShortcutAction::SwitchWindow)
        );
    }

    #[test]
    fn unmatched_shortcuts_return_none() {
        assert_eq!(match_shortcut(KeyCode::C, Modifiers::none()), None);
        assert_eq!(
            match_shortcut(
                KeyCode::C,
                Modifiers {
                    shift: true,
                    ctrl: true,
                    alt: false,
                    cmd: false,
                }
            ),
            None
        );
    }
}
