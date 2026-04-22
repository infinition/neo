//! GPU NV12 → BGRA conversion via a wgpu compute shader.
//!
//! This replaces the scalar CPU loop in [`crate::transcode`] with a
//! persistent compute pipeline that is built once and re-used across all
//! frames. The Y / UV / BGRA buffers are allocated once at the target
//! resolution and only their contents are rewritten per frame.
//!
//! Shader details
//! --------------
//!
//! - Input Y plane: one byte per luma sample, packed 4-per-`u32`.
//! - Input UV plane: NV12 interleaved (U, V, U, V, …) at half height,
//!   also packed 4 bytes per `u32`.
//! - Output BGRA: one `u32` per pixel, bytes ordered B, G, R, A in
//!   little-endian (matching NVENC's `NV_ENC_BUFFER_FORMAT_ARGB`).
//! - Workgroup size: 8×8 = 64 threads per workgroup, one thread per
//!   output pixel.
//!
//! The colour math is BT.709 limited-range, identical to the CPU
//! implementation it replaces, so the two paths produce bit-identical
//! (or very near-identical — there's some floating point rounding)
//! output for cross-validation.

use crate::nvdec::DecodedFrame;
use neo_core::{NeoError, NeoResult};
use neo_gpu::GpuContext;
use std::sync::Arc;
use tracing::debug;

const WGSL: &str = r#"
struct Dims {
    width: u32,
    height: u32,
    y_stride_words: u32, // width / 4
    uv_stride_words: u32, // width / 4  (UV row has `width` bytes: w/2 U + w/2 V)
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> y_plane: array<u32>;
@group(0) @binding(2) var<storage, read> uv_plane: array<u32>;
@group(0) @binding(3) var<storage, read_write> bgra: array<u32>;

fn load_byte(buf_index: u32, y_buf: bool) -> u32 {
    let word_index = buf_index / 4u;
    let byte_index = buf_index % 4u;
    var word: u32;
    if (y_buf) {
        word = y_plane[word_index];
    } else {
        word = uv_plane[word_index];
    }
    return (word >> (byte_index * 8u)) & 0xffu;
}

@compute @workgroup_size(8, 8, 1)
fn nv12_to_bgra(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.width || y >= dims.height) {
        return;
    }

    // Luma: one byte per pixel at (y, x).
    let y_byte_index = y * dims.width + x;
    let y_val = f32(load_byte(y_byte_index, true)) - 16.0;

    // Chroma: interleaved UV at (y/2, x & ~1) and +1.
    let uv_row = y / 2u;
    let uv_x = (x / 2u) * 2u;
    let uv_byte_index = uv_row * dims.width + uv_x;
    let u_val = f32(load_byte(uv_byte_index, false)) - 128.0;
    let v_val = f32(load_byte(uv_byte_index + 1u, false)) - 128.0;

    // BT.709 limited-range.
    let c = 1.1644 * y_val;
    let r = clamp(c + 1.7927 * v_val, 0.0, 255.0);
    let g = clamp(c - 0.5329 * v_val - 0.2132 * u_val, 0.0, 255.0);
    let b = clamp(c + 2.1124 * u_val, 0.0, 255.0);

    let ri = u32(r);
    let gi = u32(g);
    let bi = u32(b);
    let ai = 255u;
    // NVENC ARGB buffer is little-endian byte order B, G, R, A.
    let pixel = bi | (gi << 8u) | (ri << 16u) | (ai << 24u);
    bgra[y_byte_index] = pixel;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    width: u32,
    height: u32,
    y_stride_words: u32,
    uv_stride_words: u32,
}

/// Who owns the GPU storage the converter binds into its pipeline.
enum BufferOwnership {
    /// Legacy path: converter allocates + owns Y, UV, BGRA and a
    /// staging download buffer. Used by Stage 4's CPU → upload path.
    Owned {
        y_buf: wgpu::Buffer,
        uv_buf: wgpu::Buffer,
        bgra_buf: wgpu::Buffer,
        staging_buf: wgpu::Buffer,
    },
    /// Stage 6b: buffers are external — they are slices of a CUDA↔Vulkan
    /// interop allocation, and the converter only holds references.
    /// After each dispatch the output is NOT downloaded to CPU; the
    /// caller reads it via CUDA or another wgpu pipeline.
    External {
        /// Kept so Lab-style callers can `queue.write_buffer` into Y/UV
        /// from the CPU before dispatching. In Stage 6b these are aliased
        /// with CUDA dptrs and never touched from the CPU.
        y_buf: wgpu::Buffer,
        uv_buf: wgpu::Buffer,
    },
}

/// Persistent wgpu NV12 → BGRA converter bound to a single resolution.
///
/// Created once per pipeline; reuses all GPU resources across frames.
pub struct Nv12ToBgraConverter {
    ctx: Arc<GpuContext>,
    width: u32,
    height: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    // Holds Dims; bound into `bind_group` via `as_entire_binding` so it
    // must outlive the pipeline but is never read from Rust directly.
    _dims_buf: wgpu::Buffer,
    ownership: BufferOwnership,
    y_size: u64,
    uv_size: u64,
    bgra_size: u64,
}

impl Nv12ToBgraConverter {
    /// Stage 4 constructor: allocate Y/UV/BGRA + staging storage ourselves.
    pub fn new(ctx: Arc<GpuContext>, width: u32, height: u32) -> NeoResult<Self> {
        Self::new_inner(ctx, width, height, None)
    }

    /// Stage 6b constructor: caller-provided buffers. Must be at least
    /// the right byte size for the selected resolution. Used when Y/UV
    /// are aliased into CUDA memory via interop, and BGRA is likewise
    /// aliased so NVENC can pull from it directly.
    pub fn from_external_buffers(
        ctx: Arc<GpuContext>,
        width: u32,
        height: u32,
        y_buf: &wgpu::Buffer,
        uv_buf: &wgpu::Buffer,
        bgra_buf: &wgpu::Buffer,
    ) -> NeoResult<Self> {
        Self::new_inner(ctx, width, height, Some((y_buf, uv_buf, bgra_buf)))
    }

    fn new_inner(
        ctx: Arc<GpuContext>,
        width: u32,
        height: u32,
        external: Option<(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer)>,
    ) -> NeoResult<Self> {
        if width % 8 != 0 || height % 8 != 0 {
            return Err(NeoError::InvalidDimensions { width, height });
        }
        let w = width as u64;
        let h = height as u64;
        let y_size = w * h;
        let uv_size = w * (h / 2);
        let bgra_size = w * h * 4;

        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nv12-to-bgra-shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });

        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12-dims"),
            size: std::mem::size_of::<Dims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dims = Dims {
            width,
            height,
            y_stride_words: width / 4,
            uv_stride_words: width / 4,
        };
        ctx.queue.write_buffer(&dims_buf, 0, bytemuck::bytes_of(&dims));

        // Clone-own the three buffers we'll bind. `wgpu::Buffer` is a
        // thin Arc wrapper, so cloning is cheap and lets us both build
        // the `BindGroup` and stash the buffers in `BufferOwnership`
        // without lifetime gymnastics.
        let (y_buf, uv_buf, bgra_buf, ownership) = match external {
            None => {
                let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("nv12-y"),
                    size: y_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let uv_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("nv12-uv"),
                    size: uv_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bgra_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("nv12-bgra-out"),
                    size: bgra_size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("nv12-bgra-staging"),
                    size: bgra_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let own = BufferOwnership::Owned {
                    y_buf: y_buf.clone(),
                    uv_buf: uv_buf.clone(),
                    bgra_buf: bgra_buf.clone(),
                    staging_buf,
                };
                (y_buf, uv_buf, bgra_buf, own)
            }
            Some((y, uv, bgra)) => (
                y.clone(),
                uv.clone(),
                bgra.clone(),
                BufferOwnership::External {
                    y_buf: y.clone(),
                    uv_buf: uv.clone(),
                },
            ),
        };
        let y_ref = &y_buf;
        let uv_ref = &uv_buf;
        let bgra_ref = &bgra_buf;

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nv12-layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nv12-bind"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uv_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bgra_ref.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nv12-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nv12-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("nv12_to_bgra"),
            compilation_options: Default::default(),
            cache: None,
        });

        debug!(width, height, "Nv12ToBgraConverter initialized");
        Ok(Self {
            ctx,
            width,
            height,
            pipeline,
            bind_group,
            _dims_buf: dims_buf,
            ownership,
            y_size,
            uv_size,
            bgra_size,
        })
    }

    /// Upload `frame` to GPU, run the converter, download BGRA into `out`.
    pub fn convert(&self, frame: &DecodedFrame, out: &mut [u8]) -> NeoResult<()> {
        if frame.width != self.width || frame.height != self.height {
            return Err(NeoError::InvalidDimensions {
                width: frame.width,
                height: frame.height,
            });
        }
        if frame.y.len() as u64 != self.y_size {
            return Err(NeoError::GpuBuffer(format!(
                "Y plane size mismatch: got {}, want {}",
                frame.y.len(),
                self.y_size
            )));
        }
        if frame.uv.len() as u64 != self.uv_size {
            return Err(NeoError::GpuBuffer(format!(
                "UV plane size mismatch: got {}, want {}",
                frame.uv.len(),
                self.uv_size
            )));
        }
        if out.len() as u64 != self.bgra_size {
            return Err(NeoError::GpuBuffer(format!(
                "BGRA out size mismatch: got {}, want {}",
                out.len(),
                self.bgra_size
            )));
        }

        match &self.ownership {
            BufferOwnership::Owned {
                y_buf,
                uv_buf,
                bgra_buf,
                staging_buf,
            } => {
                // Upload Y + UV.
                self.ctx.queue.write_buffer(y_buf, 0, &frame.y);
                self.ctx.queue.write_buffer(uv_buf, 0, &frame.uv);

                // Dispatch compute + copy to staging in one command buffer.
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("nv12-convert-encoder"),
                        });
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("nv12-convert-pass"),
                            timestamp_writes: None,
                        });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    let groups_x = (self.width + 7) / 8;
                    let groups_y = (self.height + 7) / 8;
                    pass.dispatch_workgroups(groups_x, groups_y, 1);
                }
                encoder.copy_buffer_to_buffer(bgra_buf, 0, staging_buf, 0, self.bgra_size);
                self.ctx.queue.submit(std::iter::once(encoder.finish()));

                // Map staging → copy to out → unmap so it can be reused.
                let slice = staging_buf.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                self.ctx
                    .device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .map_err(|e| NeoError::GpuBuffer(format!("device.poll: {e}")))?;
                rx.recv()
                    .map_err(|_| NeoError::GpuBuffer("staging channel closed".into()))?
                    .map_err(|e| NeoError::GpuBuffer(format!("staging map: {e}")))?;

                out.copy_from_slice(&slice.get_mapped_range());
                staging_buf.unmap();
                Ok(())
            }
            BufferOwnership::External { .. } => Err(NeoError::GpuBuffer(
                "convert() is for owned mode only; use dispatch_interop() or upload_and_dispatch() for external buffers"
                    .into(),
            )),
        }
    }

    /// Stage A (Neo Lab): the converter is in External mode, but the
    /// caller wants to drive it from CPU NV12 frames the same way the
    /// owned path does — upload Y + UV from the CPU, dispatch the
    /// compute, then leave the result in the external BGRA buffer for a
    /// downstream shader chain to read. No CPU download.
    pub fn upload_and_dispatch(&self, frame: &DecodedFrame) -> NeoResult<()> {
        if frame.width != self.width || frame.height != self.height {
            return Err(NeoError::InvalidDimensions {
                width: frame.width,
                height: frame.height,
            });
        }
        let (y_buf, uv_buf) = match &self.ownership {
            BufferOwnership::External { y_buf, uv_buf } => (y_buf, uv_buf),
            BufferOwnership::Owned { .. } => {
                return Err(NeoError::GpuBuffer(
                    "upload_and_dispatch() is for external mode only".into(),
                ))
            }
        };
        self.ctx.queue.write_buffer(y_buf, 0, &frame.y);
        self.ctx.queue.write_buffer(uv_buf, 0, &frame.uv);
        let mut encoder =
            self.ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nv12-lab-encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12-lab-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups_x = (self.width + 7) / 8;
            let groups_y = (self.height + 7) / 8;
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Stage 6b dispatch: the Y + UV data is already sitting in the
    /// external Vulkan storage buffers (written by CUDA via interop), so
    /// we only issue the compute pass + wait. The output lands in the
    /// external BGRA storage buffer where CUDA/NVENC can pick it up
    /// without a CPU bounce.
    pub fn dispatch_interop(&self) -> NeoResult<()> {
        match &self.ownership {
            BufferOwnership::External { .. } => {}
            BufferOwnership::Owned { .. } => {
                return Err(NeoError::GpuBuffer(
                    "dispatch_interop() is for external mode only".into(),
                ));
            }
        }
        let mut encoder =
            self.ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nv12-interop-encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12-interop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups_x = (self.width + 7) / 8;
            let groups_y = (self.height + 7) / 8;
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        // Wait for the compute to complete before CUDA/NVENC reads the
        // interop BGRA buffer.
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| NeoError::GpuBuffer(format!("device.poll: {e}")))?;
        Ok(())
    }
}
