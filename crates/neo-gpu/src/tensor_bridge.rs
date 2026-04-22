use crate::GpuContext;
use neo_core::{NeoError, NeoResult};
use std::sync::Arc;

const PACK_WGSL: &str = r#"
struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       bgra:   array<u32>;
@group(0) @binding(2) var<storage, read_write> tensor: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dims.width || gid.y >= dims.height) { return; }
    let idx = gid.y * dims.width + gid.x;
    let plane = dims.width * dims.height;
    let p = bgra[idx];
    let b = f32((p >>  0u) & 0xffu) / 255.0;
    let g = f32((p >>  8u) & 0xffu) / 255.0;
    let r = f32((p >> 16u) & 0xffu) / 255.0;
    tensor[idx]              = r;
    tensor[plane + idx]      = g;
    tensor[2u * plane + idx] = b;
}
"#;

const UNPACK_WGSL: &str = r#"
struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       tensor: array<f32>;
@group(0) @binding(2) var<storage, read_write> bgra:   array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dims.width || gid.y >= dims.height) { return; }
    let idx = gid.y * dims.width + gid.x;
    let plane = dims.width * dims.height;
    let r = clamp(tensor[idx],              0.0, 1.0);
    let g = clamp(tensor[plane + idx],      0.0, 1.0);
    let b = clamp(tensor[2u * plane + idx], 0.0, 1.0);
    let ri = u32(r * 255.0);
    let gi = u32(g * 255.0);
    let bi = u32(b * 255.0);
    bgra[idx] = bi | (gi << 8u) | (ri << 16u) | (255u << 24u);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    width: u32,
    height: u32,
}

/// Reusable GPU bridge between Neo's packed BGRA video buffers and
/// model-friendly NCHW `f32` tensors.
///
/// This is the missing link for a real zero-copy path like:
/// `NVDEC -> interop BGRA -> pack -> ONNX/TensorRT -> unpack -> NVENC`.
/// No host bounce is required as long as both buffers are GPU storage
/// buffers (plain wgpu buffers or CUDA↔Vulkan interop buffers).
pub struct BgraTensorBridge {
    ctx: Arc<GpuContext>,
    width: u32,
    height: u32,
    dims_buf: wgpu::Buffer,
    pack_layout: wgpu::BindGroupLayout,
    unpack_layout: wgpu::BindGroupLayout,
    pack_pipeline: wgpu::ComputePipeline,
    unpack_pipeline: wgpu::ComputePipeline,
}

impl BgraTensorBridge {
    pub fn new(ctx: Arc<GpuContext>, width: u32, height: u32) -> NeoResult<Self> {
        if width == 0 || height == 0 {
            return Err(NeoError::InvalidDimensions { width, height });
        }

        let dims = Dims { width, height };
        let dims_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bgra-tensor-dims"),
            size: std::mem::size_of::<Dims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&dims_buf, 0, bytemuck::bytes_of(&dims));

        let pack_layout = create_layout(&ctx.device, "bgra-pack-layout");
        let unpack_layout = create_layout(&ctx.device, "bgra-unpack-layout");
        let pack_pipeline = create_pipeline(
            &ctx.device,
            "bgra-pack-pipeline",
            &pack_layout,
            PACK_WGSL,
        );
        let unpack_pipeline = create_pipeline(
            &ctx.device,
            "bgra-unpack-pipeline",
            &unpack_layout,
            UNPACK_WGSL,
        );

        Ok(Self {
            ctx,
            width,
            height,
            dims_buf,
            pack_layout,
            unpack_layout,
            pack_pipeline,
            unpack_pipeline,
        })
    }

    pub fn tensor_len_f32(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 3
    }

    pub fn tensor_byte_size(&self) -> u64 {
        (self.tensor_len_f32() * std::mem::size_of::<f32>()) as u64
    }

    pub fn pack_into(&self, bgra: &wgpu::Buffer, tensor: &wgpu::Buffer) -> NeoResult<()> {
        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bgra-pack-bind-group"),
            layout: &self.pack_layout,
            entries: &[
                bind_entry(0, self.dims_buf.as_entire_binding()),
                bind_entry(1, bgra.as_entire_binding()),
                bind_entry(2, tensor.as_entire_binding()),
            ],
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bgra-pack-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bgra-pack-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    pub fn unpack_into(&self, tensor: &wgpu::Buffer, bgra: &wgpu::Buffer) -> NeoResult<()> {
        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bgra-unpack-bind-group"),
            layout: &self.unpack_layout,
            entries: &[
                bind_entry(0, self.dims_buf.as_entire_binding()),
                bind_entry(1, tensor.as_entire_binding()),
                bind_entry(2, bgra.as_entire_binding()),
            ],
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bgra-unpack-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bgra-unpack-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.unpack_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

fn create_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    source: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bind_entry<'a>(binding: u32, resource: wgpu::BindingResource<'a>) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry { binding, resource }
}
