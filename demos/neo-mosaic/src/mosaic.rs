//! Mosaic grid: manages per-tile BGRA buffers, filter pipelines, and the
//! final grid-composition compute shader.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use tracing::info;

use crate::filters;

// ---- Per-tile filter pipeline ------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TileDims {
    width: u32,
    height: u32,
}

struct TileFilter {
    name: String,
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bind_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    dims_buf: wgpu::Buffer,
}

// ---- Grid compositor shader -------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CompositeDims {
    tile_w: u32,
    tile_h: u32,
    grid_stride: u32, // grid width in pixels
    tile_col: u32,
    tile_row: u32,
    _pad: u32,
}

const COMPOSITE_WGSL: &str = r#"
struct Dims {
    tile_w: u32,
    tile_h: u32,
    grid_stride: u32,
    tile_col: u32,
    tile_row: u32,
};
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       tile_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> grid_buf: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lx = gid.x;
    let ly = gid.y;
    if (lx >= dims.tile_w || ly >= dims.tile_h) { return; }
    let src_idx = ly * dims.tile_w + lx;
    let gx = dims.tile_col * dims.tile_w + lx;
    let gy = dims.tile_row * dims.tile_h + ly;
    let dst_idx = gy * dims.grid_stride + gx;
    grid_buf[dst_idx] = tile_buf[src_idx];
}
"#;

// ---- Public API -------------------------------------------------------------

pub struct Tile {
    pub name: String,
    /// Input BGRA buffer (NV12→BGRA converter writes here).
    pub buf_in: wgpu::Buffer,
    /// Output BGRA buffer (filter writes here).
    pub buf_out: wgpu::Buffer,
    filter: TileFilter,
    bind_group: wgpu::BindGroup,
}

pub struct MosaicGrid {
    pub tiles: Vec<Tile>,
    pub grid_buf: wgpu::Buffer,
    pub grid_w: u32,
    pub grid_h: u32,
    pub tile_w: u32,
    pub tile_h: u32,
    #[allow(dead_code)]
    cols: u32,
    #[allow(dead_code)]
    rows: u32,

    composite_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    composite_layout: wgpu::BindGroupLayout,
    /// One (dims_buf, bind_group) per tile for the compositor pass.
    composite_slots: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
}

impl MosaicGrid {
    /// Build a mosaic grid of `cols × rows` tiles, each `tile_w × tile_h`.
    ///
    /// `external_inputs`: if provided, these buffers are used as each tile's
    /// input (e.g. interop BGRA buffers from the zerocopy pipeline). Must
    /// have exactly `cols * rows` entries. If `None`, buffers are allocated.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tile_w: u32,
        tile_h: u32,
        cols: u32,
        rows: u32,
        external_inputs: Option<&[&wgpu::Buffer]>,
    ) -> Self {
        let grid_w = tile_w * cols;
        let grid_h = tile_h * rows;
        let tile_pixels = (tile_w as u64) * (tile_h as u64);
        let grid_pixels = (grid_w as u64) * (grid_h as u64);

        // Allocate the combined grid output buffer.
        let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mosaic-grid"),
            size: grid_pixels * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Build per-tile filter pipelines.
        let filter_defs = filters::builtins();
        let n = (cols * rows) as usize;
        let mut tiles = Vec::with_capacity(n);

        for i in 0..n {
            let (name, wgsl) = &filter_defs[i % filter_defs.len()];
            let label = format!("tile-{i}-{name}");

            let buf_in = if let Some(ext) = &external_inputs {
                ext[i].clone()
            } else {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{label}-in")),
                    size: tile_pixels * 4,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}-out")),
                size: tile_pixels * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}-dims")),
                size: std::mem::size_of::<TileDims>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(
                &dims_buf,
                0,
                bytemuck::bytes_of(&TileDims {
                    width: tile_w,
                    height: tile_h,
                }),
            );

            let bind_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{label}-layout")),
                    entries: &[
                        bgl_entry(0, wgpu::BufferBindingType::Uniform, false),
                        bgl_entry(
                            1,
                            wgpu::BufferBindingType::Storage { read_only: true },
                            false,
                        ),
                        bgl_entry(
                            2,
                            wgpu::BufferBindingType::Storage { read_only: false },
                            false,
                        ),
                    ],
                });

            let pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{label}-pl")),
                    bind_group_layouts: &[Some(&bind_layout)],
                    immediate_size: 0,
                });

            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&label),
                source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
            });
            let pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&label),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&label),
                layout: &bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dims_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_out.as_entire_binding(),
                    },
                ],
            });

            info!(tile = i, filter = *name, "tile ready");
            tiles.push(Tile {
                name: name.to_string(),
                buf_in,
                buf_out,
                filter: TileFilter {
                    name: label,
                    pipeline,
                    bind_layout,
                    dims_buf,
                },
                bind_group,
            });
        }

        // Composite pipeline: copies each tile's output into the grid.
        let composite_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mosaic-composite-layout"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform, false),
                    bgl_entry(
                        1,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                    ),
                    bgl_entry(
                        2,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                ],
            });
        let composite_pl =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mosaic-composite-pl"),
                bind_group_layouts: &[Some(&composite_layout)],
                immediate_size: 0,
            });
        let composite_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mosaic-composite"),
                source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
            });
        let composite_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mosaic-composite"),
                layout: Some(&composite_pl),
                module: &composite_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Pre-build per-tile compositor bind groups.
        let mut composite_slots = Vec::with_capacity(n);
        for (i, tile) in tiles.iter().enumerate() {
            let col = (i as u32) % cols;
            let row = (i as u32) / cols;
            let dims = CompositeDims {
                tile_w,
                tile_h,
                grid_stride: grid_w,
                tile_col: col,
                tile_row: row,
                _pad: 0,
            };
            let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("composite-dims-{i}")),
                size: std::mem::size_of::<CompositeDims>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&dims_buf, 0, bytemuck::bytes_of(&dims));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("composite-bg-{i}")),
                layout: &composite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dims_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tile.buf_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grid_buf.as_entire_binding(),
                    },
                ],
            });
            composite_slots.push((dims_buf, bg));
        }

        info!(cols, rows, grid_w, grid_h, tiles = n, "mosaic grid ready");

        Self {
            tiles,
            grid_buf,
            grid_w,
            grid_h,
            tile_w,
            tile_h,
            cols,
            rows,
            composite_pipeline,
            composite_layout,
            composite_slots,
        }
    }

    /// Record all tile filter dispatches + grid composition into `encoder`.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        let groups_x = (self.tile_w + 7) / 8;
        let groups_y = (self.tile_h + 7) / 8;

        // 1. Run each tile's filter.
        for tile in &self.tiles {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&tile.filter.name),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&tile.filter.pipeline);
            pass.set_bind_group(0, &tile.bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }

        // 2. Composite each tile's output into the grid buffer.
        for (_dims_buf, bg) in &self.composite_slots {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mosaic-composite"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
    }
}

fn bgl_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    _compute_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
