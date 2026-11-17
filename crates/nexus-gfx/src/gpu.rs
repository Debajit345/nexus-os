use crate::color::Color;
use crate::geometry::Rect;

/// Hardware or virtual GPU vendor classification.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Unknown,
    VirtioGpu,
    Amd,
    Intel,
    Nvidia,
    Apple,
    Software,
}

/// Primary driver/submission backend kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GpuBackendKind {
    Vulkan,
    DrmKms,
    VirtioGpu,
    Software,
}

/// Bitmask of GPU/DRM capabilities exposed by an adapter.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct GpuCapability(pub u64);

impl GpuCapability {
    pub const NONE: Self = Self(0);
    pub const VULKAN_1_3: Self = Self(1 << 0);
    pub const DRM_ATOMIC_KMS: Self = Self(1 << 1);
    pub const DMA_BUF: Self = Self(1 << 2);
    pub const TIMELINE_SEMAPHORE: Self = Self(1 << 3);
    pub const COMPUTE_QUEUE: Self = Self(1 << 4);
    pub const TRANSFER_QUEUE: Self = Self(1 << 5);
    pub const HARDWARE_CURSOR: Self = Self(1 << 6);
    pub const VSYNC_TEARING: Self = Self(1 << 7);
    pub const MULTI_PLANE_OVERLAY: Self = Self(1 << 8);
    pub const VIRTIO_3D_VIRGL: Self = Self(1 << 9);

    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Static metadata describing a GPU/DRM adapter.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GpuAdapterInfo {
    pub vendor: GpuVendor,
    pub backend: GpuBackendKind,
    pub vendor_id: u32,
    pub device_id: u32,
    pub vram_bytes: u64,
    pub capabilities: GpuCapability,
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u64);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u64);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FenceHandle(pub u64);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SwapchainHandle(pub u64);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct BufferUsage(pub u32);

impl BufferUsage {
    pub const VERTEX: Self = Self(1 << 0);
    pub const INDEX: Self = Self(1 << 1);
    pub const UNIFORM: Self = Self(1 << 2);
    pub const STORAGE: Self = Self(1 << 3);
    pub const TRANSFER_SRC: Self = Self(1 << 4);
    pub const TRANSFER_DST: Self = Self(1 << 5);

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsage,
    pub host_visible: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    Argb8888,
    Xrgb8888,
    Abgr8888,
    Rgba8888,
    Bgra8888,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub render_target: bool,
    pub scanout: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    OutOfMemory,
    DeviceLost,
    InvalidHandle,
    UnsupportedCapability,
    SubmissionFailed,
    Timeout,
    InvalidArgument,
    CommandBufferFull,
    InvalidState,
}

/// Clean-room 2D/compositor GPU command.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DrawCommand {
    Clear {
        target: TextureHandle,
        color: Color,
    },
    FillRect {
        target: TextureHandle,
        rect: Rect,
        color: Color,
    },
    Blit {
        src: TextureHandle,
        dst: TextureHandle,
        src_rect: Rect,
        dst_x: i32,
        dst_y: i32,
        blend: bool,
    },
    Present {
        swapchain: SwapchainHandle,
        texture: TextureHandle,
    },
    Barrier,
}

/// Recorded GPU/DRM command buffer.
pub trait CommandBuffer {
    fn reset(&mut self) -> Result<(), GpuError>;
    fn begin(&mut self) -> Result<(), GpuError>;
    fn push(&mut self, cmd: DrawCommand) -> Result<(), GpuError>;
    fn end(&mut self) -> Result<(), GpuError>;
    fn command_count(&self) -> usize;
}

/// Clean-room GPU/DRM device abstraction for Vulkan-first, virtio-gpu, AMD, Intel, and NVIDIA backends.
pub trait GpuDevice {
    type CmdBuf: CommandBuffer;

    fn adapter_info(&self) -> &GpuAdapterInfo;

    #[inline]
    fn capabilities(&self) -> GpuCapability {
        self.adapter_info().capabilities
    }

    fn create_buffer(&mut self, desc: BufferDesc) -> Result<BufferHandle, GpuError>;
    fn destroy_buffer(&mut self, handle: BufferHandle) -> Result<(), GpuError>;

    fn create_texture(&mut self, desc: TextureDesc) -> Result<TextureHandle, GpuError>;
    fn destroy_texture(&mut self, handle: TextureHandle) -> Result<(), GpuError>;

    fn create_command_buffer(&mut self) -> Result<Self::CmdBuf, GpuError>;
    fn submit(&mut self, cmd: &Self::CmdBuf) -> Result<FenceHandle, GpuError>;
    fn wait_fence(&mut self, fence: FenceHandle, timeout_ns: u64) -> Result<bool, GpuError>;
    fn present(
        &mut self,
        swapchain: SwapchainHandle,
        texture: TextureHandle,
    ) -> Result<(), GpuError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockCmdBuf {
        recording: bool,
        sealed: bool,
        cmds: [Option<DrawCommand>; 16],
        len: usize,
    }

    impl MockCmdBuf {
        fn new() -> Self {
            Self {
                recording: false,
                sealed: false,
                cmds: [None; 16],
                len: 0,
            }
        }
    }

    impl CommandBuffer for MockCmdBuf {
        fn reset(&mut self) -> Result<(), GpuError> {
            self.recording = false;
            self.sealed = false;
            self.len = 0;
            Ok(())
        }

        fn begin(&mut self) -> Result<(), GpuError> {
            if self.recording {
                return Err(GpuError::InvalidState);
            }
            self.recording = true;
            self.sealed = false;
            self.len = 0;
            Ok(())
        }

        fn push(&mut self, cmd: DrawCommand) -> Result<(), GpuError> {
            if !self.recording {
                return Err(GpuError::InvalidState);
            }
            if self.len >= self.cmds.len() {
                return Err(GpuError::CommandBufferFull);
            }
            self.cmds[self.len] = Some(cmd);
            self.len += 1;
            Ok(())
        }

        fn end(&mut self) -> Result<(), GpuError> {
            if !self.recording {
                return Err(GpuError::InvalidState);
            }
            self.recording = false;
            self.sealed = true;
            Ok(())
        }

        fn command_count(&self) -> usize {
            self.len
        }
    }

    struct MockGpu {
        info: GpuAdapterInfo,
        next_id: u64,
        submitted_cmds: usize,
    }

    impl MockGpu {
        fn new() -> Self {
            Self {
                info: GpuAdapterInfo {
                    vendor: GpuVendor::VirtioGpu,
                    backend: GpuBackendKind::Vulkan,
                    vendor_id: 0x1AF4,
                    device_id: 0x1050,
                    vram_bytes: 256 * 1024 * 1024,
                    capabilities: GpuCapability::VULKAN_1_3
                        .union(GpuCapability::DRM_ATOMIC_KMS)
                        .union(GpuCapability::DMA_BUF),
                },
                next_id: 1,
                submitted_cmds: 0,
            }
        }
    }

    impl GpuDevice for MockGpu {
        type CmdBuf = MockCmdBuf;

        fn adapter_info(&self) -> &GpuAdapterInfo {
            &self.info
        }

        fn create_buffer(&mut self, desc: BufferDesc) -> Result<BufferHandle, GpuError> {
            if desc.size == 0 {
                return Err(GpuError::InvalidArgument);
            }
            let id = self.next_id;
            self.next_id += 1;
            Ok(BufferHandle(id))
        }

        fn destroy_buffer(&mut self, handle: BufferHandle) -> Result<(), GpuError> {
            if handle.0 == 0 {
                return Err(GpuError::InvalidHandle);
            }
            Ok(())
        }

        fn create_texture(&mut self, desc: TextureDesc) -> Result<TextureHandle, GpuError> {
            if desc.width == 0 || desc.height == 0 {
                return Err(GpuError::InvalidArgument);
            }
            let id = self.next_id;
            self.next_id += 1;
            Ok(TextureHandle(id))
        }

        fn destroy_texture(&mut self, handle: TextureHandle) -> Result<(), GpuError> {
            if handle.0 == 0 {
                return Err(GpuError::InvalidHandle);
            }
            Ok(())
        }

        fn create_command_buffer(&mut self) -> Result<Self::CmdBuf, GpuError> {
            Ok(MockCmdBuf::new())
        }

        fn submit(&mut self, cmd: &Self::CmdBuf) -> Result<FenceHandle, GpuError> {
            if !cmd.sealed {
                return Err(GpuError::InvalidState);
            }
            self.submitted_cmds += cmd.command_count();
            let id = self.next_id;
            self.next_id += 1;
            Ok(FenceHandle(id))
        }

        fn wait_fence(&mut self, fence: FenceHandle, _timeout_ns: u64) -> Result<bool, GpuError> {
            if fence.0 == 0 {
                return Err(GpuError::InvalidHandle);
            }
            Ok(true)
        }

        fn present(
            &mut self,
            swapchain: SwapchainHandle,
            texture: TextureHandle,
        ) -> Result<(), GpuError> {
            if swapchain.0 == 0 || texture.0 == 0 {
                return Err(GpuError::InvalidHandle);
            }
            Ok(())
        }
    }

    #[test]
    fn gpu_capability_and_submission_lifecycle() {
        let mut gpu = MockGpu::new();
        assert!(gpu.capabilities().contains(GpuCapability::VULKAN_1_3));
        assert!(gpu.capabilities().contains(GpuCapability::DRM_ATOMIC_KMS));
        assert!(!gpu.capabilities().contains(GpuCapability::VSYNC_TEARING));

        let tex = gpu
            .create_texture(TextureDesc {
                width: 1280,
                height: 800,
                format: TextureFormat::Argb8888,
                render_target: true,
                scanout: true,
            })
            .unwrap();

        let mut cmd = gpu.create_command_buffer().unwrap();
        cmd.begin().unwrap();
        cmd.push(DrawCommand::Clear {
            target: tex,
            color: Color::BLACK,
        })
        .unwrap();
        cmd.push(DrawCommand::FillRect {
            target: tex,
            rect: Rect::new(10, 10, 100, 50),
            color: Color::WHITE,
        })
        .unwrap();
        cmd.end().unwrap();

        let fence = gpu.submit(&cmd).unwrap();
        assert!(gpu.wait_fence(fence, 1_000_000).unwrap());
        assert_eq!(gpu.submitted_cmds, 2);
        gpu.present(SwapchainHandle(1), tex).unwrap();
        gpu.destroy_texture(tex).unwrap();
    }
}
