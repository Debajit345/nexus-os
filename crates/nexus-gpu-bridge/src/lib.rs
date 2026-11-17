#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Clean-room GPU API translation bridge for Nexus OS.
//!
//! Translates Direct3D 11/12 (DXVK/VKD3D-style), Metal (macOS), OpenGL, and
//! native Vulkan draw/compute/present calls into unified [`VulkanCommand`]s
//! and [`nexus_gfx::DrawCommand`]s.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use nexus_gfx::{Color, DrawCommand, Rect, SwapchainHandle, TextureFormat, TextureHandle};

/// Guest graphics APIs supported by the Nexus OS GPU translation bridge.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GraphicsApi {
    Vulkan,
    Direct3D11,
    Direct3D12,
    Metal,
    OpenGL,
}

impl GraphicsApi {
    /// Returns the clean-room translation backend name for this API.
    #[inline]
    pub const fn translation_layer(self) -> &'static str {
        match self {
            Self::Vulkan => "native-vulkan",
            Self::Direct3D11 => "dxvk",
            Self::Direct3D12 => "vkd3d",
            Self::Metal => "metal-vk",
            Self::OpenGL => "zink-gl",
        }
    }

    /// Returns `true` if the guest API is already native Vulkan.
    #[inline]
    pub const fn is_native_vulkan(self) -> bool {
        matches!(self, Self::Vulkan)
    }
}

/// Programmable shader stage in a graphics or compute pipeline.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// Pipeline state description submitted by a guest graphics API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineDesc {
    pub api: GraphicsApi,
    pub shader_hash: u64,
    pub stages: Vec<ShaderStage>,
    pub blend_enabled: bool,
    pub depth_test: bool,
    pub target_format: TextureFormat,
}

impl PipelineDesc {
    /// Creates a new pipeline description with default render state.
    pub fn new(api: GraphicsApi, shader_hash: u64, stages: Vec<ShaderStage>) -> Self {
        Self {
            api,
            shader_hash,
            stages,
            blend_enabled: false,
            depth_test: false,
            target_format: TextureFormat::Bgra8888,
        }
    }

    /// Returns `true` if this pipeline is a compute pipeline.
    pub fn is_compute(&self) -> bool {
        self.stages.contains(&ShaderStage::Compute)
    }

    /// Validates that the pipeline description is well-formed.
    pub fn validate(&self) -> Result<(), BridgeError> {
        if self.shader_hash == 0 {
            return Err(BridgeError::InvalidShaderHash);
        }
        if self.stages.is_empty() {
            return Err(BridgeError::EmptyShaderStages);
        }

        let mut has_vertex = false;
        let mut has_fragment = false;
        let mut has_compute = false;

        for &stage in &self.stages {
            match stage {
                ShaderStage::Vertex => {
                    if has_vertex {
                        return Err(BridgeError::DuplicateShaderStage);
                    }
                    has_vertex = true;
                }
                ShaderStage::Fragment => {
                    if has_fragment {
                        return Err(BridgeError::DuplicateShaderStage);
                    }
                    has_fragment = true;
                }
                ShaderStage::Compute => {
                    if has_compute {
                        return Err(BridgeError::DuplicateShaderStage);
                    }
                    has_compute = true;
                }
            }
        }

        if has_compute && (has_vertex || has_fragment) {
            return Err(BridgeError::IncompatibleShaderStages);
        }

        if !has_compute && !has_vertex {
            return Err(BridgeError::MissingVertexStage);
        }

        Ok(())
    }
}

/// Guest draw call submitted from a translated or native graphics API.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GuestDrawCall {
    pub api: GraphicsApi,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub shader_hash: u64,
}

/// Unified Vulkan-first command emitted by [`GpuBridge`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum VulkanCommand {
    BindPipeline {
        pipeline_id: u64,
        source_api: GraphicsApi,
    },
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    Dispatch {
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    },
    Present {
        swapchain: SwapchainHandle,
        texture: TextureHandle,
    },
}

/// Errors returned during pipeline validation or command translation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BridgeError {
    InvalidShaderHash,
    EmptyShaderStages,
    IncompatibleShaderStages,
    DuplicateShaderStage,
    MissingVertexStage,
    PipelineNotRegistered(u64),
    ZeroVertexCount,
    ZeroInstanceCount,
    ZeroDispatchGroups,
    ComputePipelineForDraw,
    GraphicsPipelineForDispatch,
}

/// Translation counters for diagnostics and telemetry.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BridgeStats {
    pub registered_pipelines: usize,
    pub translated_draws: u64,
    pub translated_dispatches: u64,
    pub translated_presents: u64,
    pub pipeline_binds: u64,
}

#[derive(Clone, Debug)]
struct RegisteredPipeline {
    id: u64,
    desc: PipelineDesc,
}

/// Translates Direct3D 11/12, Metal, OpenGL, and Vulkan calls into unified
/// [`VulkanCommand`]s and [`nexus_gfx::DrawCommand`]s.
#[derive(Debug, Default)]
pub struct GpuBridge {
    pipelines: BTreeMap<(GraphicsApi, u64), RegisteredPipeline>,
    next_pipeline_id: u64,
    bound_pipeline: Option<u64>,
    stats: BridgeStats,
}

impl GpuBridge {
    /// Creates a new empty GPU translation bridge.
    pub fn new() -> Self {
        Self {
            pipelines: BTreeMap::new(),
            next_pipeline_id: 1,
            bound_pipeline: None,
            stats: BridgeStats::default(),
        }
    }

    /// Validates and registers (or returns the existing ID for) a pipeline.
    pub fn register_pipeline(&mut self, desc: PipelineDesc) -> Result<u64, BridgeError> {
        desc.validate()?;
        let key = (desc.api, desc.shader_hash);
        if let Some(existing) = self.pipelines.get(&key) {
            return Ok(existing.id);
        }

        let id = self.next_pipeline_id;
        self.next_pipeline_id = self.next_pipeline_id.saturating_add(1);
        self.pipelines.insert(key, RegisteredPipeline { id, desc });
        self.stats.registered_pipelines = self.pipelines.len();
        Ok(id)
    }

    /// Returns the registered pipeline description for `(api, shader_hash)`, if any.
    pub fn get_pipeline(&self, api: GraphicsApi, shader_hash: u64) -> Option<&PipelineDesc> {
        self.pipelines.get(&(api, shader_hash)).map(|p| &p.desc)
    }

    /// Returns current translation statistics.
    pub fn stats(&self) -> BridgeStats {
        self.stats
    }

    /// Translates a guest draw call into one or more [`VulkanCommand`]s.
    pub fn translate_draw(
        &mut self,
        call: &GuestDrawCall,
    ) -> Result<Vec<VulkanCommand>, BridgeError> {
        if call.shader_hash == 0 {
            return Err(BridgeError::InvalidShaderHash);
        }
        if call.vertex_count == 0 {
            return Err(BridgeError::ZeroVertexCount);
        }
        if call.instance_count == 0 {
            return Err(BridgeError::ZeroInstanceCount);
        }

        let entry = self
            .pipelines
            .get(&(call.api, call.shader_hash))
            .ok_or(BridgeError::PipelineNotRegistered(call.shader_hash))?;

        if entry.desc.is_compute() {
            return Err(BridgeError::ComputePipelineForDraw);
        }

        let pipeline_id = entry.id;
        let mut cmds = Vec::new();

        if self.bound_pipeline != Some(pipeline_id) {
            cmds.push(VulkanCommand::BindPipeline {
                pipeline_id,
                source_api: call.api,
            });
            self.bound_pipeline = Some(pipeline_id);
            self.stats.pipeline_binds += 1;
        }

        cmds.push(VulkanCommand::Draw {
            vertex_count: call.vertex_count,
            instance_count: call.instance_count,
            first_vertex: 0,
            first_instance: 0,
        });
        self.stats.translated_draws += 1;

        Ok(cmds)
    }

    /// Translates a guest compute dispatch into [`VulkanCommand`]s.
    pub fn translate_dispatch(
        &mut self,
        api: GraphicsApi,
        shader_hash: u64,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> Result<Vec<VulkanCommand>, BridgeError> {
        if shader_hash == 0 {
            return Err(BridgeError::InvalidShaderHash);
        }
        if group_count_x == 0 || group_count_y == 0 || group_count_z == 0 {
            return Err(BridgeError::ZeroDispatchGroups);
        }

        let entry = self
            .pipelines
            .get(&(api, shader_hash))
            .ok_or(BridgeError::PipelineNotRegistered(shader_hash))?;

        if !entry.desc.is_compute() {
            return Err(BridgeError::GraphicsPipelineForDispatch);
        }

        let pipeline_id = entry.id;
        let mut cmds = Vec::new();

        if self.bound_pipeline != Some(pipeline_id) {
            cmds.push(VulkanCommand::BindPipeline {
                pipeline_id,
                source_api: api,
            });
            self.bound_pipeline = Some(pipeline_id);
            self.stats.pipeline_binds += 1;
        }

        cmds.push(VulkanCommand::Dispatch {
            group_count_x,
            group_count_y,
            group_count_z,
        });
        self.stats.translated_dispatches += 1;

        Ok(cmds)
    }

    /// Translates a swapchain present call and resets per-frame pipeline binding state.
    pub fn translate_present(
        &mut self,
        _api: GraphicsApi,
        swapchain: SwapchainHandle,
        texture: TextureHandle,
    ) -> VulkanCommand {
        self.bound_pipeline = None;
        self.stats.translated_presents += 1;
        VulkanCommand::Present { swapchain, texture }
    }

    /// Lowers a slice of [`VulkanCommand`]s into [`nexus_gfx::DrawCommand`]s for
    /// submission to a `nexus_gfx::GpuDevice` command buffer.
    pub fn to_gfx_commands(
        &self,
        commands: &[VulkanCommand],
        target: TextureHandle,
        draw_rect: Rect,
        color: Color,
    ) -> Vec<DrawCommand> {
        let mut out = Vec::with_capacity(commands.len());
        for &cmd in commands {
            match cmd {
                VulkanCommand::BindPipeline { .. } | VulkanCommand::Dispatch { .. } => {
                    out.push(DrawCommand::Barrier);
                }
                VulkanCommand::Draw { .. } => {
                    out.push(DrawCommand::FillRect {
                        target,
                        rect: draw_rect,
                        color,
                    });
                }
                VulkanCommand::Present { swapchain, texture } => {
                    out.push(DrawCommand::Present { swapchain, texture });
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn d3d12_draw_and_present_translation() {
        let mut bridge = GpuBridge::new();
        let desc = PipelineDesc::new(
            GraphicsApi::Direct3D12,
            0xD3D1_2001,
            vec![ShaderStage::Vertex, ShaderStage::Fragment],
        );
        let pid = bridge.register_pipeline(desc).unwrap();

        let call = GuestDrawCall {
            api: GraphicsApi::Direct3D12,
            vertex_count: 36,
            instance_count: 4,
            shader_hash: 0xD3D1_2001,
        };
        let cmds = bridge.translate_draw(&call).unwrap();
        assert_eq!(
            cmds,
            vec![
                VulkanCommand::BindPipeline {
                    pipeline_id: pid,
                    source_api: GraphicsApi::Direct3D12,
                },
                VulkanCommand::Draw {
                    vertex_count: 36,
                    instance_count: 4,
                    first_vertex: 0,
                    first_instance: 0,
                },
            ]
        );

        // Second draw with same pipeline skips redundant BindPipeline.
        let cmds2 = bridge.translate_draw(&call).unwrap();
        assert_eq!(
            cmds2,
            vec![VulkanCommand::Draw {
                vertex_count: 36,
                instance_count: 4,
                first_vertex: 0,
                first_instance: 0,
            }]
        );

        let present = bridge.translate_present(
            GraphicsApi::Direct3D12,
            SwapchainHandle(1),
            TextureHandle(7),
        );
        assert_eq!(
            present,
            VulkanCommand::Present {
                swapchain: SwapchainHandle(1),
                texture: TextureHandle(7),
            }
        );

        let gfx = bridge.to_gfx_commands(
            &[cmds[0], cmds[1], present],
            TextureHandle(7),
            Rect::new(0, 0, 1280, 720),
            Color::from_rgba(255, 255, 255, 255),
        );
        assert_eq!(
            gfx,
            vec![
                DrawCommand::Barrier,
                DrawCommand::FillRect {
                    target: TextureHandle(7),
                    rect: Rect::new(0, 0, 1280, 720),
                    color: Color::from_rgba(255, 255, 255, 255),
                },
                DrawCommand::Present {
                    swapchain: SwapchainHandle(1),
                    texture: TextureHandle(7),
                },
            ]
        );
    }

    #[test]
    fn metal_compute_and_graphics_translation() {
        let mut bridge = GpuBridge::new();
        let compute_desc =
            PipelineDesc::new(GraphicsApi::Metal, 0xAE7A_1001, vec![ShaderStage::Compute]);
        let compute_pid = bridge.register_pipeline(compute_desc).unwrap();

        let dispatch_cmds = bridge
            .translate_dispatch(GraphicsApi::Metal, 0xAE7A_1001, 8, 8, 1)
            .unwrap();
        assert_eq!(
            dispatch_cmds,
            vec![
                VulkanCommand::BindPipeline {
                    pipeline_id: compute_pid,
                    source_api: GraphicsApi::Metal,
                },
                VulkanCommand::Dispatch {
                    group_count_x: 8,
                    group_count_y: 8,
                    group_count_z: 1,
                },
            ]
        );

        let gfx_desc = PipelineDesc::new(
            GraphicsApi::Metal,
            0xAE7A_2002,
            vec![ShaderStage::Vertex, ShaderStage::Fragment],
        );
        let gfx_pid = bridge.register_pipeline(gfx_desc).unwrap();
        let draw_cmds = bridge
            .translate_draw(&GuestDrawCall {
                api: GraphicsApi::Metal,
                vertex_count: 3,
                instance_count: 1,
                shader_hash: 0xAE7A_2002,
            })
            .unwrap();
        assert_eq!(
            draw_cmds[0],
            VulkanCommand::BindPipeline {
                pipeline_id: gfx_pid,
                source_api: GraphicsApi::Metal,
            }
        );
        assert_eq!(GraphicsApi::Metal.translation_layer(), "metal-vk");
    }

    #[test]
    fn vulkan_d3d11_and_opengl_translation() {
        let mut bridge = GpuBridge::new();
        for (api, hash, expected_layer) in [
            (GraphicsApi::Vulkan, 0x1111, "native-vulkan"),
            (GraphicsApi::Direct3D11, 0x2222, "dxvk"),
            (GraphicsApi::OpenGL, 0x3333, "zink-gl"),
        ] {
            assert_eq!(api.translation_layer(), expected_layer);
            let desc =
                PipelineDesc::new(api, hash, vec![ShaderStage::Vertex, ShaderStage::Fragment]);
            let pid = bridge.register_pipeline(desc).unwrap();
            let cmds = bridge
                .translate_draw(&GuestDrawCall {
                    api,
                    vertex_count: 6,
                    instance_count: 2,
                    shader_hash: hash,
                })
                .unwrap();
            assert_eq!(
                cmds[0],
                VulkanCommand::BindPipeline {
                    pipeline_id: pid,
                    source_api: api,
                }
            );
        }
        assert!(GraphicsApi::Vulkan.is_native_vulkan());
        assert!(!GraphicsApi::Direct3D12.is_native_vulkan());
        assert_eq!(bridge.stats().translated_draws, 3);
    }

    #[test]
    fn pipeline_validation_and_error_cases() {
        let mut bridge = GpuBridge::new();

        // Zero shader hash
        assert_eq!(
            bridge.register_pipeline(PipelineDesc::new(
                GraphicsApi::Vulkan,
                0,
                vec![ShaderStage::Vertex],
            )),
            Err(BridgeError::InvalidShaderHash)
        );

        // Empty stages
        assert_eq!(
            bridge.register_pipeline(PipelineDesc::new(GraphicsApi::Vulkan, 1, vec![])),
            Err(BridgeError::EmptyShaderStages)
        );

        // Mixing compute and vertex
        assert_eq!(
            bridge.register_pipeline(PipelineDesc::new(
                GraphicsApi::Direct3D12,
                2,
                vec![ShaderStage::Vertex, ShaderStage::Compute],
            )),
            Err(BridgeError::IncompatibleShaderStages)
        );

        // Duplicate stage
        assert_eq!(
            bridge.register_pipeline(PipelineDesc::new(
                GraphicsApi::Metal,
                3,
                vec![ShaderStage::Vertex, ShaderStage::Vertex],
            )),
            Err(BridgeError::DuplicateShaderStage)
        );

        // Fragment without vertex
        assert_eq!(
            bridge.register_pipeline(PipelineDesc::new(
                GraphicsApi::OpenGL,
                4,
                vec![ShaderStage::Fragment],
            )),
            Err(BridgeError::MissingVertexStage)
        );

        // Unregistered pipeline draw
        assert_eq!(
            bridge.translate_draw(&GuestDrawCall {
                api: GraphicsApi::Direct3D12,
                vertex_count: 3,
                instance_count: 1,
                shader_hash: 0xDEAD,
            }),
            Err(BridgeError::PipelineNotRegistered(0xDEAD))
        );

        // Compute pipeline used for draw
        bridge
            .register_pipeline(PipelineDesc::new(
                GraphicsApi::Vulkan,
                0xC0DE,
                vec![ShaderStage::Compute],
            ))
            .unwrap();
        assert_eq!(
            bridge.translate_draw(&GuestDrawCall {
                api: GraphicsApi::Vulkan,
                vertex_count: 3,
                instance_count: 1,
                shader_hash: 0xC0DE,
            }),
            Err(BridgeError::ComputePipelineForDraw)
        );
    }
}
