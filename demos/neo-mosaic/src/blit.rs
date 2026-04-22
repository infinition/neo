//! Fullscreen blit: renders a BGRA storage buffer onto the winit surface.
//! Adapted from neo-lab's BlitPipeline for the mosaic grid dimensions.

use bytemuck::{Pod, Zeroable};

const SHADER: &str = r#"
struct Dims {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> bgra: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    o.uv  = uvs[vid];
    return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let dst_aspect = f32(dims.dst_w) / f32(dims.dst_h);
    let src_aspect = f32(dims.src_w) / f32(dims.src_h);

    var uv = in.uv;
    if (dst_aspect > src_aspect) {
        let scale = dst_aspect / src_aspect;
        uv.x = (uv.x - 0.5) * scale + 0.5;
    } else {
        let scale = src_aspect / dst_aspect;
        uv.y = (uv.y - 0.5) * scale + 0.5;
    }
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let x = u32(uv.x * f32(dims.src_w));
    let y = u32(uv.y * f32(dims.src_h));
    let idx = min(y, dims.src_h - 1u) * dims.src_w + min(x, dims.src_w - 1u);
    let pix = bgra[idx];
    let b = f32((pix >>  0u) & 0xffu) / 255.0;
    let g = f32((pix >>  8u) & 0xffu) / 255.0;
    let r = f32((pix >> 16u) & 0xffu) / 255.0;
    return vec4<f32>(r, g, b, 1.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Dims {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

pub struct BlitPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    dims_buf: wgpu::Buffer,
    src_w: u32,
    src_h: u32,
}

impl BlitPipeline {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        src_w: u32,
        src_h: u32,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mosaic-blit"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mosaic-blit-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mosaic-blit-pl"),
                bind_group_layouts: &[Some(&bind_layout)],
                immediate_size: 0,
            });

        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mosaic-blit-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mosaic-blit-dims"),
            size: std::mem::size_of::<Dims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &dims_buf,
            0,
            bytemuck::bytes_of(&Dims {
                src_w,
                src_h,
                dst_w: src_w,
                dst_h: src_h,
            }),
        );

        Self {
            pipeline,
            bind_layout,
            dims_buf,
            src_w,
            src_h,
        }
    }

    pub fn update_window_size(&self, queue: &wgpu::Queue, dst_w: u32, dst_h: u32) {
        queue.write_buffer(
            &self.dims_buf,
            0,
            bytemuck::bytes_of(&Dims {
                src_w: self.src_w,
                src_h: self.src_h,
                dst_w,
                dst_h,
            }),
        );
    }

    pub fn make_bind_group(
        &self,
        device: &wgpu::Device,
        src: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mosaic-blit-bg"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.as_entire_binding(),
                },
            ],
        })
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        bind_group: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mosaic-blit-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
