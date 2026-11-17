#![cfg_attr(not(any(test, feature = "std")), no_std)]

//! Clean-room 2D compositor primitives, macOS-inspired desktop chrome,
//! and GPU/DRM command submission traits for Nexus OS.

pub mod chrome;
pub mod color;
pub mod geometry;
pub mod gpu;
pub mod surface;

pub use chrome::{
    DesktopLayout, Dock, DockItemLayout, MenuBar, MenuBarHit, TrafficLightButton, WindowFrame,
    WindowHit,
};
pub use color::Color;
pub use geometry::{Point, Rect, Size};
pub use gpu::{
    BufferDesc, BufferHandle, BufferUsage, CommandBuffer, DrawCommand, FenceHandle, GpuAdapterInfo,
    GpuBackendKind, GpuCapability, GpuDevice, GpuError, GpuVendor, SwapchainHandle, TextureDesc,
    TextureFormat, TextureHandle,
};
pub use surface::{Surface, SurfaceError, SurfaceRef};
