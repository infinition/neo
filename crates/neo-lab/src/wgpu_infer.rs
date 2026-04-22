//! GPU inference engine for the supported subset of `WgpuPlanOp`s.
//!
//! Implements the Phase B.2a fast path. Given a
//! [`WgpuPlanDescription`] from `neo-infer-ort`, it builds a fixed
//! chain of compute pipelines and ping-pong storage buffers that
//! turn the chain's BGRA output into NCHW f32, applies the model's
//! ops directly on the GPU, and writes the result back into a BGRA
//! buffer — all without ever touching host memory.
//!
//! ## Layout
//!
//! ```text
//!   chain BGRA  ──pack──▶ tensor_a (NCHW f32) ─┐
//!                                              │
//!                                              ▼ op pipelines
//!                                              │
//!                                          tensor_b
//!                                              │
//!                                              ▼ ... ping-pong ...
//!                                              │
//!                                              ▼
//!                                       final tensor ──unpack──▶ output BGRA
//! ```
//!
//! For an Identity-only plan we collapse the chain to a single
//! pack→unpack with no op dispatches in between (the chain output is
//! literally copied through the f32 tensor and back).
//!
//! All pipelines are built once at construction; per-frame work is
//! pure command-buffer recording.

use bytemuck::{Pod, Zeroable};
use neo_infer_ort::{UnaryKind, WgpuPlanDescription, WgpuPlanOp};
use std::sync::Arc;
use tracing::info;

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
    // NCHW: R plane, G plane, B plane.
    tensor[idx]               = r;
    tensor[plane + idx]       = g;
    tensor[2u * plane + idx]  = b;
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

/// Element-wise unary op with one scalar uniform `cval`. The expression
/// to evaluate is templated in via Rust's `format!` so we get one
/// pipeline per concrete op kind without writing three near-identical
/// shaders.
// 2D dispatch over a 1D tensor.
//
// wgpu (WebGPU spec) caps `dispatch_workgroups` at 65535 per dimension,
// which means a naive `(N/wg_size, 1, 1)` dispatch breaks the moment
// N grows beyond a few million elements. At 4K RGB N = 24.9 M which is
// well past that limit.
//
// Solution: workgroup_size(256, 1, 1) along the first axis and a 2D
// dispatch grid `(STRIDE_GROUPS_X, ceil(N / (256 * STRIDE_GROUPS_X)), 1)`.
// `STRIDE_GROUPS_X = 1024` keeps both dispatch dims well under 65535
// for any 8K image.
//
// Phase B.2b: constants are baked *inline* into the WGSL source as
// `const C: f32 = ...;` declarations, so no uniform buffer is needed.
// The engine only has to bind `(input_tensor, output_tensor)`. Op-
// specific expressions (including non-trivial ones like
// `clamp(x, LO, HI)` or `1.0 / (1.0 + exp(-x))`) are inserted via
// `build_elemwise_wgsl()`.
const ELEMWISE_PROLOGUE: &str = r#"
const STRIDE: u32 = 1024u * 256u; // STRIDE_GROUPS_X * workgroup_size.x

@group(0) @binding(0) var<storage, read>        input:  array<f32>;
@group(0) @binding(1) var<storage, read_write>  output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * STRIDE + gid.x;
    if (i >= arrayLength(&input)) { return; }
    let x: f32 = input[i];
    output[i] = "#;

/// Width of the 2D dispatch grid (in workgroups). Combined with
/// workgroup_size(256, 1, 1) this gives 256 * 1024 = 262 144 elements
/// per "row" of the dispatch grid.
const STRIDE_GROUPS_X: u32 = 1024;
const ELEMS_PER_ROW: u32 = STRIDE_GROUPS_X * 256;

/// Build a complete elemwise WGSL module for one op.
///
/// `consts` is a list of `(name, value)` pairs that become top-of-file
/// `const NAME: f32 = VALUE;` declarations so the op expression can
/// reference them.
fn build_elemwise_wgsl(op_expr: &str, consts: &[(&str, f32)]) -> String {
    let mut src = String::with_capacity(512);
    for (name, value) in consts {
        // WGSL is strict about f32 literals — integers like "1" are
        // interpreted as i32 in `const` contexts. Always emit with a
        // decimal point.
        src.push_str(&format!("const {name}: f32 = {value:?};\n"));
    }
    src.push_str(ELEMWISE_PROLOGUE);
    src.push_str(op_expr);
    src.push_str(";\n}");
    src
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Dims {
    width: u32,
    height: u32,
}

/// One compiled element-wise op (input tensor → output tensor).
///
/// Phase B.2b simplification: op constants are baked into the WGSL
/// source as `const`, so the pipeline needs no uniform buffer and the
/// bind group is pure storage-buffer pingpong.
struct ElemPipeline {
    pipeline: wgpu::ComputePipeline,
    n_elements: u32,
}

pub struct WgpuInferenceEngine {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    width: u32,
    height: u32,
    n_elements: u32,
    plane: u32,

    // Pack/unpack stages.
    pack_pipeline: wgpu::ComputePipeline,
    unpack_pipeline: wgpu::ComputePipeline,
    dims_buf: wgpu::Buffer,

    // Ping-pong float tensors.
    tensor_a: wgpu::Buffer,
    tensor_b: wgpu::Buffer,

    // Bind group layouts (kept around so we can build per-frame bind
    // groups against any caller-supplied src/dst BGRA buffer).
    pack_layout: wgpu::BindGroupLayout,
    unpack_layout: wgpu::BindGroupLayout,
    elem_layout: wgpu::BindGroupLayout,

    // Pre-built bind groups for the parts that don't change frame to
    // frame (the tensor sides of the elemwise pipelines).
    elem_a_to_b: Vec<wgpu::BindGroup>,
    elem_b_to_a: Vec<wgpu::BindGroup>,

    pipelines: Vec<ElemPipeline>,
}

impl WgpuInferenceEngine {
    /// Build a complete inference engine for the given plan + image
    /// resolution. Returns `Err` if the plan is malformed; `Ok(None)`
    /// if the plan contains an op kind we recognize at the description
    /// level but don't have a shader for yet (so callers can fall back
    /// gracefully).
    pub fn build(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        plan: &WgpuPlanDescription,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        if plan.input_shape.len() != 4 {
            return Err(format!(
                "wgpu inference: expected NCHW shape, got {:?}",
                plan.input_shape
            ));
        }
        let (n, c, h, w) = (
            plan.input_shape[0],
            plan.input_shape[1],
            plan.input_shape[2],
            plan.input_shape[3],
        );
        if n != 1 || c != 3 {
            return Err(format!(
                "wgpu inference: expected [1, 3, H, W], got {:?}",
                plan.input_shape
            ));
        }
        if w as u32 != width || h as u32 != height {
            return Err(format!(
                "wgpu inference: model {w}x{h} != image {width}x{height}"
            ));
        }

        let n_elements = (width as u32) * (height as u32) * 3;
        let plane = (width as u32) * (height as u32);
        let tensor_size_bytes = (n_elements as u64) * 4;

        // ---- pack / unpack pipelines ---------------------------------
        let pack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("infer-pack-layout"),
            entries: &[
                bind_uniform(0),
                bind_storage_ro(1),
                bind_storage_rw(2),
            ],
        });
        let pack_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("infer-pack"),
            source: wgpu::ShaderSource::Wgsl(PACK_WGSL.into()),
        });
        let pack_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("infer-pack-pl"),
            bind_group_layouts: &[Some(&pack_layout)],
            immediate_size: 0,
        });
        let pack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("infer-pack-pipeline"),
            layout: Some(&pack_pl),
            module: &pack_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let unpack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("infer-unpack-layout"),
            entries: &[
                bind_uniform(0),
                bind_storage_ro(1),
                bind_storage_rw(2),
            ],
        });
        let unpack_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("infer-unpack"),
            source: wgpu::ShaderSource::Wgsl(UNPACK_WGSL.into()),
        });
        let unpack_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("infer-unpack-pl"),
            bind_group_layouts: &[Some(&unpack_layout)],
            immediate_size: 0,
        });
        let unpack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("infer-unpack-pipeline"),
            layout: Some(&unpack_pl),
            module: &unpack_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("infer-dims"),
            size: std::mem::size_of::<Dims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dims_buf, 0, bytemuck::bytes_of(&Dims { width, height }));

        // ---- ping-pong tensors ---------------------------------------
        let make_tensor = |label: &'static str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: tensor_size_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let tensor_a = make_tensor("infer-tensor-a");
        let tensor_b = make_tensor("infer-tensor-b");

        // ---- elem-wise op layout (input + output storage only) ------
        // Phase B.2b: no uniform buffer — constants are baked into the
        // shader source as WGSL `const`. Bind group is two buffers.
        let elem_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("infer-elem-layout"),
            entries: &[bind_storage_ro(0), bind_storage_rw(1)],
        });
        let elem_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("infer-elem-pl"),
            bind_group_layouts: &[Some(&elem_layout)],
            immediate_size: 0,
        });

        // ---- compile each non-Identity op into its own pipeline ------
        // Identity ops are skipped — they're a no-op.
        let mut pipelines: Vec<ElemPipeline> = Vec::new();
        for op in &plan.ops {
            let wgsl = match op {
                WgpuPlanOp::Identity => continue,
                WgpuPlanOp::SubConst { constant } => {
                    build_elemwise_wgsl("C - x", &[("C", *constant)])
                }
                WgpuPlanOp::AddConst { constant } => {
                    build_elemwise_wgsl("x + C", &[("C", *constant)])
                }
                WgpuPlanOp::MulConst { constant } => {
                    build_elemwise_wgsl("x * C", &[("C", *constant)])
                }
                WgpuPlanOp::DivConstRight { constant } => {
                    build_elemwise_wgsl("x / C", &[("C", *constant)])
                }
                WgpuPlanOp::DivConstLeft { constant } => {
                    build_elemwise_wgsl("C / x", &[("C", *constant)])
                }
                WgpuPlanOp::PowConst { exponent } => {
                    // WGSL `pow` is undefined on negative bases with
                    // non-integer exponents; videos have non-negative
                    // inputs after the pack pass so we clamp defensively.
                    build_elemwise_wgsl("pow(max(x, 0.0), E)", &[("E", *exponent)])
                }
                WgpuPlanOp::MinConst { constant } => {
                    build_elemwise_wgsl("min(x, C)", &[("C", *constant)])
                }
                WgpuPlanOp::MaxConst { constant } => {
                    build_elemwise_wgsl("max(x, C)", &[("C", *constant)])
                }
                WgpuPlanOp::Clip { min, max } => build_elemwise_wgsl(
                    "clamp(x, LO, HI)",
                    &[("LO", *min), ("HI", *max)],
                ),
                WgpuPlanOp::Unary(kind) => {
                    build_elemwise_wgsl(UnaryKind::wgsl_expr(*kind), &[])
                }
            };
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("infer-elem"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("infer-elem-pipeline"),
                layout: Some(&elem_pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            pipelines.push(ElemPipeline {
                pipeline,
                n_elements,
            });
        }

        // Pre-build bind groups for the elemwise stages so per-frame
        // recording is just `set_pipeline + set_bind_group + dispatch`.
        let mut elem_a_to_b = Vec::with_capacity(pipelines.len());
        let mut elem_b_to_a = Vec::with_capacity(pipelines.len());
        for _ in &pipelines {
            elem_a_to_b.push(make_elem_bg(&device, &elem_layout, &tensor_a, &tensor_b));
            elem_b_to_a.push(make_elem_bg(&device, &elem_layout, &tensor_b, &tensor_a));
        }

        info!(
            width, height,
            ops = pipelines.len(),
            tensor_mb = tensor_size_bytes / (1024 * 1024),
            "WgpuInferenceEngine ready"
        );

        Ok(Self {
            device,
            queue,
            width,
            height,
            n_elements,
            plane,
            pack_pipeline,
            unpack_pipeline,
            dims_buf,
            tensor_a,
            tensor_b,
            pack_layout,
            unpack_layout,
            elem_layout,
            elem_a_to_b,
            elem_b_to_a,
            pipelines,
        })
    }

    /// Apply the model in place: read pixels from `bgra`, run pack →
    /// ops → unpack, write the result back to `bgra`. Caller must
    /// ensure `bgra` is at least `width*height*4` bytes and has both
    /// STORAGE and COPY_DST usage.
    pub fn process(&self, bgra: &wgpu::Buffer) -> Result<(), String> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("infer-engine-frame"),
            });

        // ---- 1. pack: BGRA -> tensor_a ------------------------------
        let pack_bg = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("infer-pack-bg"),
                layout: &self.pack_layout,
                entries: &[
                    bg_entry(0, self.dims_buf.as_entire_binding()),
                    bg_entry(1, bgra.as_entire_binding()),
                    bg_entry(2, self.tensor_a.as_entire_binding()),
                ],
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("infer-pack-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &pack_bg, &[]);
            let gx = (self.width + 7) / 8;
            let gy = (self.height + 7) / 8;
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // ---- 2. ops chain (ping-pong tensor_a <-> tensor_b) ---------
        let mut current_is_a = true;
        for (i, p) in self.pipelines.iter().enumerate() {
            let bg = if current_is_a {
                &self.elem_a_to_b[i]
            } else {
                &self.elem_b_to_a[i]
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("infer-elem-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&p.pipeline);
            pass.set_bind_group(0, bg, &[]);
            // 2D dispatch — see ELEMWISE_WGSL_TEMPLATE comment.
            let groups_y = (p.n_elements + ELEMS_PER_ROW - 1) / ELEMS_PER_ROW;
            pass.dispatch_workgroups(STRIDE_GROUPS_X, groups_y, 1);
            current_is_a = !current_is_a;
        }

        // ---- 3. unpack: final tensor -> BGRA ------------------------
        let final_tensor = if current_is_a {
            &self.tensor_a
        } else {
            &self.tensor_b
        };
        let unpack_bg = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("infer-unpack-bg"),
                layout: &self.unpack_layout,
                entries: &[
                    bg_entry(0, self.dims_buf.as_entire_binding()),
                    bg_entry(1, final_tensor.as_entire_binding()),
                    bg_entry(2, bgra.as_entire_binding()),
                ],
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("infer-unpack-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.unpack_pipeline);
            pass.set_bind_group(0, &unpack_bg, &[]);
            let gx = (self.width + 7) / 8;
            let gy = (self.height + 7) / 8;
            pass.dispatch_workgroups(gx, gy, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

// ---- bind layout helpers --------------------------------------------

fn bind_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn make_elem_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("infer-elem-bg"),
        layout,
        entries: &[
            bg_entry(0, src.as_entire_binding()),
            bg_entry(1, dst.as_entire_binding()),
        ],
    })
}
