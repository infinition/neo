use crate::buffer::VramBuffer;
use crate::context::GpuContext;
use neo_core::NeoResult;

/// Describes the access mode of a buffer binding in a compute shader.
#[derive(Debug, Clone, Copy)]
pub enum BufferAccess {
    /// Read-only storage buffer.
    ReadOnly,
    /// Read-write storage buffer.
    ReadWrite,
    /// Uniform buffer (small, read-only constants).
    Uniform,
}

/// A single binding in a compute pass.
pub struct Binding<'a> {
    pub buffer: &'a VramBuffer,
    pub access: BufferAccess,
}

/// Execute a compute shader on VRAM buffers.
///
/// This is the core GPU execution primitive. Everything in Neo-FFmpeg
/// that touches the GPU goes through this function.
pub fn dispatch_compute(
    ctx: &GpuContext,
    shader_source: &str,
    entry_point: &str,
    bindings: &[Binding<'_>],
    workgroups: [u32; 3],
) -> NeoResult<()> {
    let shader_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry_point),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    // Build bind group layout entries
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: match b.access {
                BufferAccess::ReadOnly => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BufferAccess::ReadWrite => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BufferAccess::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            count: None,
        })
        .collect();

    let bind_group_layout =
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{entry_point}-layout")),
                entries: &layout_entries,
            });

    let bind_group_entries: Vec<wgpu::BindGroupEntry> = bindings
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.buffer.buffer.as_entire_binding(),
        })
        .collect();

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{entry_point}-bind")),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    // wgpu 29: bind_group_layouts is now `&[Option<&BindGroupLayout>]` and
    // push_constant_ranges has been replaced by `immediate_size`.
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{entry_point}-pipeline-layout")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{entry_point}-pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{entry_point}-encoder")),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry_point),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }

    ctx.queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
