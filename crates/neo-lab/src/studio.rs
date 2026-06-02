//! neo-studio — standalone egui application for the live video pipeline.
//!
//! Launch the `neo-studio` binary with no arguments: an empty window opens
//! with an **Import video…** button. Pick a clip (raw H.264, or mp4/mkv/avi/
//! mov which are demuxed to Annex-B on the fly via ffmpeg) and the pipeline
//! is built sized to that clip and starts playing. Importing another clip
//! rebuilds the engine live.
//!
//! Render path:
//!
//! ```text
//! StreamSource (NVDEC, loops) ─▶ Nv12ToBgraConverter ─▶ chain.buf_a
//!   ─▶ ShaderChain (hot-reload WGSL) ─▶ final BGRA buffer
//!   ─▶ [optional] BgraTensorBridge ─▶ OnnxModelCuda (zero-copy) ─▶ back to BGRA
//!   ─▶ BlitPipeline ─▶ swapchain  (load = Clear)
//!   ─▶ egui render pass            (load = Load)   ◀── side panel + HUD
//!   ─▶ present()
//! ```
//!
//! Shaders are picked from the side panel (the UI writes the chosen `.wgsl`
//! into the watched `shaders_dir`; `ShaderChain::poll_reload()` swaps it).
//! ONNX models (`[1,3,H,W]` f32 in/out) run zero-copy on the CUDA device when
//! the GPU exposes `VULKAN_EXTERNAL_MEMORY_WIN32` interop.

use crate::blit::BlitPipeline;
use crate::chain::{FinalBuffer, ShaderChain};
use crate::stream::StreamSource;
use neo_core::NeoResult;
use neo_gpu::{BgraTensorBridge, GpuContext};
use neo_hwaccel::{
    interop::{create_interop_buffer, InteropBuffer},
    CudaRuntime, Nv12ToBgraConverter,
};
use neo_infer_ort::OnnxModelCuda;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB, NV_ENC_CODEC_H264_GUID,
        NV_ENC_INPUT_RESOURCE_TYPE,
    },
    Encoder, EncoderInitParams, ErrorKind, Session,
};
use std::{
    ffi::c_void,
    fs::File,
    io::{BufWriter, Write},
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{error, info, warn};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

/// Options for the Studio app. `input` is optional — when `None`, the app
/// opens empty and waits for the user to import a clip from the UI.
pub struct StudioOptions {
    pub input: Option<PathBuf>,
    /// Scratch dir the UI writes the active `.wgsl` into.
    pub shaders_dir: PathBuf,
    pub fps: Option<f64>,
    pub no_vsync: bool,
}

impl Default for StudioOptions {
    fn default() -> Self {
        Self {
            input: None,
            shaders_dir: PathBuf::from("shaders-studio"),
            fps: None,
            no_vsync: false,
        }
    }
}

/// Entry point. Blocks until the window closes.
pub fn run(opts: StudioOptions) -> NeoResult<()> {
    std::fs::create_dir_all(&opts.shaders_dir)
        .map_err(|e| neo_core::NeoError::Pipeline(format!("create shaders dir: {e}")))?;

    let runtime = Arc::new(CudaRuntime::new(0)?);
    runtime
        .ctx
        .bind_to_thread()
        .map_err(|e| neo_core::NeoError::Cuda(format!("bind_to_thread: {e:?}")))?;

    let event_loop = EventLoop::new()
        .map_err(|e| neo_core::NeoError::Pipeline(format!("event loop: {e}")))?;
    let mut app = StudioApp {
        init: StudioInit {
            runtime,
            shaders_dir: opts.shaders_dir,
            pending_input: opts.input,
            fps: opts.fps.unwrap_or(30.0),
            no_vsync: opts.no_vsync,
        },
        window: None,
        state: None,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| neo_core::NeoError::Pipeline(format!("event loop run: {e}")))?;
    Ok(())
}

// ---- Built-in shaders --------------------------------------------------------
// Each body assumes the standard chain layout (see crate::chain docs):
//   @binding(0) uniform Dims { width, height }
//   @binding(1) storage<read>       in_buf:  array<u32>   (packed BGRA)
//   @binding(2) storage<read_write> out_buf: array<u32>

const PREAMBLE: &str = r#"
struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;
fn unpack(p: u32) -> vec4<f32> {
    return vec4<f32>(f32((p>>0u)&0xffu), f32((p>>8u)&0xffu), f32((p>>16u)&0xffu), f32((p>>24u)&0xffu));
}
fn pack(c: vec4<f32>) -> u32 {
    let b=u32(clamp(c.x,0.0,255.0)); let g=u32(clamp(c.y,0.0,255.0));
    let r=u32(clamp(c.z,0.0,255.0)); let a=u32(clamp(c.w,0.0,255.0));
    return b | (g<<8u) | (r<<16u) | (a<<24u);
}
"#;

const PASSTHROUGH: &str = r#"
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x=gid.x; let y=gid.y;
    if (x>=dims.width || y>=dims.height) { return; }
    let idx=y*dims.width+x; out_buf[idx]=in_buf[idx];
}
"#;

const GRAYSCALE: &str = r#"
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x=gid.x; let y=gid.y;
    if (x>=dims.width || y>=dims.height) { return; }
    let idx=y*dims.width+x; let c=unpack(in_buf[idx]);
    let l=0.114*c.x+0.587*c.y+0.299*c.z;
    out_buf[idx]=pack(vec4<f32>(l,l,l,c.w));
}
"#;

const INVERT: &str = r#"
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x=gid.x; let y=gid.y;
    if (x>=dims.width || y>=dims.height) { return; }
    let idx=y*dims.width+x; let c=unpack(in_buf[idx]);
    out_buf[idx]=pack(vec4<f32>(255.0-c.x,255.0-c.y,255.0-c.z,c.w));
}
"#;

const SEPIA: &str = r#"
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x=gid.x; let y=gid.y;
    if (x>=dims.width || y>=dims.height) { return; }
    let idx=y*dims.width+x; let c=unpack(in_buf[idx]);
    let r=c.z; let g=c.y; let b=c.x;
    let sr=clamp(r*0.393+g*0.769+b*0.189,0.0,255.0);
    let sg=clamp(r*0.349+g*0.686+b*0.168,0.0,255.0);
    let sb=clamp(r*0.272+g*0.534+b*0.131,0.0,255.0);
    out_buf[idx]=pack(vec4<f32>(sb,sg,sr,c.w));
}
"#;

const EDGE: &str = r#"
fn luma_at(idx: u32) -> f32 { let c=unpack(in_buf[idx]); return 0.114*c.x+0.587*c.y+0.299*c.z; }
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x=gid.x; let y=gid.y;
    if (x>=dims.width || y>=dims.height) { return; }
    if (x==0u||y==0u||x>=dims.width-1u||y>=dims.height-1u) { out_buf[y*dims.width+x]=pack(vec4<f32>(0.0,0.0,0.0,255.0)); return; }
    let w=dims.width;
    let tl=luma_at((y-1u)*w+x-1u); let tc=luma_at((y-1u)*w+x); let tr=luma_at((y-1u)*w+x+1u);
    let ml=luma_at(y*w+x-1u);                                  let mr=luma_at(y*w+x+1u);
    let bl=luma_at((y+1u)*w+x-1u); let bc=luma_at((y+1u)*w+x); let br=luma_at((y+1u)*w+x+1u);
    let gx=-tl-2.0*ml-bl+tr+2.0*mr+br; let gy=-tl-2.0*tc-tr+bl+2.0*bc+br;
    let m=clamp(sqrt(gx*gx+gy*gy),0.0,255.0);
    out_buf[y*w+x]=pack(vec4<f32>(m,m,m,255.0));
}
"#;

const THERMAL: &str = r#"
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x=gid.x; let y=gid.y;
    if (x>=dims.width || y>=dims.height) { return; }
    let idx=y*dims.width+x; let c=unpack(in_buf[idx]);
    let lum=(0.114*c.x+0.587*c.y+0.299*c.z)/255.0;
    var r:f32; var g:f32; var b:f32;
    if (lum<0.25){ let t=lum/0.25; b=255.0; g=t*255.0; r=0.0; }
    else if (lum<0.5){ let t=(lum-0.25)/0.25; b=(1.0-t)*255.0; g=255.0; r=0.0; }
    else if (lum<0.75){ let t=(lum-0.5)/0.25; b=0.0; g=255.0; r=t*255.0; }
    else { let t=(lum-0.75)/0.25; b=0.0; g=(1.0-t)*255.0; r=255.0; }
    out_buf[idx]=pack(vec4<f32>(b,g,r,255.0));
}
"#;

/// `(display name, wgsl body)` — index 0 is the default (passthrough).
fn builtins() -> &'static [(&'static str, &'static str)] {
    &[
        ("Original", PASSTHROUGH),
        ("Grayscale", GRAYSCALE),
        ("Invert", INVERT),
        ("Sepia", SEPIA),
        ("Edge (Sobel)", EDGE),
        ("Thermal", THERMAL),
    ]
}

/// Replace every `.wgsl` in `dir` with a single active shader so the
/// `ShaderChain` runs exactly one filter. `None` clears the dir (pass-through).
fn write_active_shader(dir: &Path, body: Option<&str>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("wgsl") {
                let _ = std::fs::remove_file(&p);
            }
        }
    }
    if let Some(body) = body {
        let src = format!("{PREAMBLE}\n{body}");
        if let Err(e) = std::fs::write(dir.join("00_active.wgsl"), src) {
            warn!("write active shader: {e}");
        }
    }
}

fn pick_video_file() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Video", &["h264", "264", "mp4", "mkv", "avi", "mov", "m4v", "ts"])
        .add_filter("All files", &["*"])
        .set_title("Import video")
        .pick_file()
}

fn pick_onnx_file() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("ONNX model", &["onnx"])
        .set_title("Load ONNX filter ([1,3,H,W] f32)")
        .pick_file()
}

// ---- Container demuxing ------------------------------------------------------

fn run_ffmpeg(args: &[&str]) -> bool {
    Command::new("ffmpeg")
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Return a raw H.264 Annex-B path for `path`. Raw `.h264` files are used as
/// is; containers (mp4/mkv/avi/mov/…) are demuxed via ffmpeg — losslessly
/// stream-copied when the video is already H.264, otherwise transcoded.
fn prepare_input(path: &Path) -> NeoResult<PathBuf> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if matches!(ext.as_str(), "h264" | "264" | "avc" | "bin") {
        return Ok(path.to_path_buf());
    }

    let dir = std::env::temp_dir().join("neo-studio");
    std::fs::create_dir_all(&dir)
        .map_err(|e| neo_core::NeoError::Pipeline(format!("temp dir: {e}")))?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("clip");
    let out = dir.join(format!("{stem}.h264"));
    let in_s = path.to_string_lossy().to_string();
    let out_s = out.to_string_lossy().to_string();

    let nonempty = |p: &Path| std::fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false);

    info!(file = %path.display(), "demuxing to Annex-B (lossless remux first)…");
    let copied = run_ffmpeg(&[
        "-y", "-i", &in_s, "-map", "0:v:0", "-c:v", "copy",
        "-bsf:v", "h264_mp4toannexb", "-f", "h264", &out_s,
    ]);
    if copied && nonempty(&out) {
        info!("remuxed (stream copy, no re-encode)");
        return Ok(out);
    }

    warn!("stream copy failed — transcoding to H.264 (slower)…");
    let encoded = run_ffmpeg(&[
        "-y", "-i", &in_s, "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-bsf:v", "h264_mp4toannexb", "-f", "h264", &out_s,
    ]);
    if encoded && nonempty(&out) {
        info!("transcoded to H.264");
        return Ok(out);
    }

    Err(neo_core::NeoError::Decode(format!(
        "could not prepare {:?}. Install ffmpeg (on PATH) for mp4/mkv/avi import, \
         or supply a raw .h264 Annex-B file.",
        path
    )))
}

// ---- Resize (nearest) --------------------------------------------------------

const RESIZE_WGSL: &str = r#"
struct D { sw:u32, sh:u32, dw:u32, dh:u32 };
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read>       src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= d.dw || y >= d.dh) { return; }
    let sx = min((x * d.sw) / max(d.dw, 1u), d.sw - 1u);
    let sy = min((y * d.sh) / max(d.dh, 1u), d.sh - 1u);
    dst[y * d.dw + x] = src[sy * d.sw + sx];
}
"#;

/// Nearest-neighbour BGRA (packed u32) resize: src (sw×sh) → dst (dw×dh).
/// Dims are uniform so a single pipeline serves any resolution pair.
struct ResizePipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl ResizePipeline {
    fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("studio-resize"),
            source: wgpu::ShaderSource::Wgsl(RESIZE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("studio-resize-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("studio-resize-pl"), bind_group_layouts: &[Some(&layout)], immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("studio-resize"), layout: Some(&pl), module: &module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        Self { pipeline, layout }
    }

    /// Resize `src` (sw×sh) into `dst` (dw×dh) and submit. When the dims are
    /// equal this is effectively a copy.
    fn run(&self, gpu: &GpuContext, src: &wgpu::Buffer, sw: u32, sh: u32, dst: &wgpu::Buffer, dw: u32, dh: u32) {
        let dims: [u32; 4] = [sw, sh, dw, dh];
        let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("studio-resize-dims"), size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&dims));
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("studio-resize-bg"), layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
            ],
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("studio-resize-enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("studio-resize-pass"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((dw + 7) / 8, (dh + 7) / 8, 1);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
    }
}

/// Extract `(C, H, W)` from an NCHW `[N,C,H,W]` or CHW `[C,H,W]` shape.
/// Returns `None` if the rank is unsupported or any of C/H/W is non-positive.
fn nchw(shape: &[i64]) -> Option<(u32, u32, u32)> {
    let (c, h, w) = match shape.len() {
        4 => (shape[1], shape[2], shape[3]),
        3 => (shape[0], shape[1], shape[2]),
        _ => return None,
    };
    if c <= 0 || h <= 0 || w <= 0 {
        return None;
    }
    Some((c as u32, h as u32, w as u32))
}

// ---- ONNX (zero-copy, lazy) --------------------------------------------------

/// Resolve dynamic dims in a model input shape using the video W/H.
fn resolve_shape(model_shape: &[i64], width: i64, height: i64) -> Vec<i64> {
    let n = model_shape.len();
    model_shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d > 0 {
                return d;
            }
            match (n, i) {
                (4, 0) => 1,
                (4, 1) => 3,
                (4, 2) => height,
                (4, 3) => width,
                (3, 0) => 3,
                (3, 1) => height,
                (3, 2) => width,
                _ => 1,
            }
        })
        .collect()
}

/// CUDA-resident model + its interop tensor buffers + bridge. Sized to the
/// current clip; dropped when the clip changes or the user removes it.
struct OnnxState {
    bridge_in: BgraTensorBridge,
    bridge_out: BgraTensorBridge,
    tensor_in: InteropBuffer,
    tensor_out: InteropBuffer,
    /// BGRA scratch at the model's input / output resolution.
    bgra_in: wgpu::Buffer,
    bgra_out: wgpu::Buffer,
    model: OnnxModelCuda,
    in_shape: Vec<i64>,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    capacity_f32: usize,
    name: String,
}

// ---- Recorder (NVENC → file) -------------------------------------------------

/// Native save dialog. Returns `(path, is_mp4)`.
fn pick_save_target() -> Option<(PathBuf, bool)> {
    let p = rfd::FileDialog::new()
        .add_filter("MP4 (H.264)", &["mp4"])
        .add_filter("H.264 brut (Annex-B)", &["h264"])
        .set_file_name("capture.mp4")
        .set_title("Enregistrer la sortie")
        .save_file()?;
    let mp4 = p
        .extension()
        .and_then(|s| s.to_str())
        .map(|e| e.eq_ignore_ascii_case("mp4"))
        .unwrap_or(false);
    Some((p, mp4))
}

/// Encodes displayed frames with NVENC to an Annex-B file. The CUDA-visible
/// BGRA the encoder reads is an interop buffer we copy each displayed frame
/// into. `Bitstream`/`RegisteredResource` borrow the `Session`, so they're
/// created per frame (no self-referential struct).
struct Recorder {
    session: Session,
    interop: InteropBuffer,
    file: BufWriter<File>,
    h264_path: PathBuf,
    /// When set, remux the `.h264` into this `.mp4` on stop.
    mp4_path: Option<PathBuf>,
    width: u32,
    height: u32,
    fps: u32,
    frames: u64,
}

// ---- Engine (per-video pipeline) --------------------------------------------

struct Engine {
    source: StreamSource,
    converter: Nv12ToBgraConverter,
    chain: ShaderChain,
    blit: BlitPipeline,
    width: u32,
    height: u32,
    name: String,
    selected: usize,
    onnx: Option<OnnxState>,
    /// Video-resolution BGRA target the (possibly upscaled) ONNX output is
    /// resized back into for display/record.
    bgra_display: wgpu::Buffer,
    #[allow(dead_code)]
    y_buf: wgpu::Buffer,
    #[allow(dead_code)]
    uv_buf: wgpu::Buffer,
}

fn build_engine(
    gpu: &Arc<GpuContext>,
    runtime: &Arc<CudaRuntime>,
    surface_format: wgpu::TextureFormat,
    shaders_dir: &Path,
    path: &Path,
    dst_w: u32,
    dst_h: u32,
) -> NeoResult<Engine> {
    let annexb = prepare_input(path)?;
    let bytes = std::fs::read(&annexb)
        .map_err(|e| neo_core::NeoError::Decode(format!("read {:?}: {e}", annexb)))?;
    let source = StreamSource::new(runtime.clone(), bytes)?;
    let w = source.width;
    let h = source.height;
    info!(width = w, height = h, file = %path.display(), "studio: video loaded");

    write_active_shader(shaders_dir, None);
    let chain = ShaderChain::new(gpu.device.clone(), &gpu.queue, w, h, shaders_dir.to_path_buf())
        .map_err(|e| neo_core::NeoError::Pipeline(format!("shader chain: {e}")))?;

    let y_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("studio-y"),
        size: (w as u64) * (h as u64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uv_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("studio-uv"),
        size: (w as u64) * (h as u64) / 2,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let converter =
        Nv12ToBgraConverter::from_external_buffers(gpu.clone(), w, h, &y_buf, &uv_buf, &chain.buf_a)?;

    let blit = BlitPipeline::new(&gpu.device, &gpu.queue, surface_format, w, h);
    blit.update_window_size(&gpu.queue, dst_w, dst_h);

    let bgra_display = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("studio-bgra-display"),
        size: (w as u64) * (h as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Ok(Engine {
        source,
        converter,
        chain,
        blit,
        width: w,
        height: h,
        name: path.file_name().and_then(|s| s.to_str()).unwrap_or("clip").to_string(),
        selected: 0,
        onnx: None,
        bgra_display,
        y_buf,
        uv_buf,
    })
}

// ---- App state ---------------------------------------------------------------

struct StudioInit {
    runtime: Arc<CudaRuntime>,
    shaders_dir: PathBuf,
    pending_input: Option<PathBuf>,
    fps: f64,
    no_vsync: bool,
}

struct StudioApp {
    init: StudioInit,
    window: Option<Arc<Window>>,
    state: Option<StudioState>,
}

struct StudioState {
    gpu: Arc<GpuContext>,
    runtime: Arc<CudaRuntime>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    config: wgpu::SurfaceConfiguration,
    shaders_dir: PathBuf,
    interop_available: bool,
    resizer: ResizePipeline,
    /// Last user-facing status line (errors, model load result…).
    status: Option<String>,
    /// When true, ignore frame pacing and render as fast as the GPU allows.
    uncapped: bool,

    engine: Option<Engine>,

    // egui
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    // pacing / stats
    last_present: Instant,
    target_frame_time: Duration,
    fps_ema: f32,
    last_frame: Instant,
}

impl StudioState {
    fn load_video(&mut self, path: &Path) {
        match build_engine(
            &self.gpu,
            &self.runtime,
            self.surface_format,
            &self.shaders_dir,
            path,
            self.config.width,
            self.config.height,
        ) {
            Ok(engine) => self.engine = Some(engine),
            Err(e) => error!("import failed: {e}"),
        }
    }

    /// Load an ONNX filter for the current clip (zero-copy, lazy interop alloc).
    ///
    /// ONNX Runtime with `load-dynamic` *panics* (it does not return an error)
    /// when the runtime DLLs can't be found. That panic would otherwise unwind
    /// through winit's C callback and abort the whole process, so the load is
    /// wrapped in `catch_unwind` and turned into a status message.
    fn load_model(&mut self, path: &Path) {
        if !self.interop_available {
            self.status = Some("Interop CUDA↔Vulkan indisponible — ONNX impossible.".into());
            return;
        }
        let Some((w, h)) = self.engine.as_ref().map(|e| (e.width, e.height)) else {
            self.status = Some("Importe une vidéo avant un modèle.".into());
            return;
        };

        let ctx = self.runtime.ctx.clone();
        let loaded = catch_unwind(AssertUnwindSafe(|| OnnxModelCuda::load(path, ctx, 0)));
        let model = match loaded {
            Ok(Ok(m)) => m,
            Ok(Err(e)) => {
                let msg = format!("Échec chargement ONNX : {e}");
                error!("{msg}");
                self.status = Some(msg);
                return;
            }
            Err(_) => {
                let msg = "ONNX Runtime a planté au chargement — DLL ONNX Runtime absentes ? \
                           Définis ORT_DYLIB_PATH (voir README).".to_string();
                error!("{msg}");
                self.status = Some(msg);
                return;
            }
        };

        if model.input_count() != 1 {
            self.status = Some(format!("Modèle rejeté : 1 entrée attendue, {} trouvées.", model.input_count()));
            return;
        }
        // Any [N,3,H,W] / [3,H,W] model works: we resize the video to the
        // model's input resolution and resize the output back to video res.
        let in_shape = resolve_shape(model.input_shape(0), w as i64, h as i64);
        let out_shape = resolve_shape(model.output_shape(), w as i64, h as i64);
        let (Some((in_c, in_h, in_w)), Some((out_c, out_h, out_w))) =
            (nchw(&in_shape), nchw(&out_shape))
        else {
            self.status = Some(format!("Modèle rejeté : formes {:?}→{:?} non [N,3,H,W].", in_shape, out_shape));
            return;
        };
        if in_c != 3 || out_c != 3 {
            self.status = Some(format!("Modèle rejeté : 3 canaux requis (in C={in_c}, out C={out_c})."));
            return;
        }

        let in_size = (in_w as u64) * (in_h as u64) * 3 * 4;
        let out_size = (out_w as u64) * (out_h as u64) * 3 * 4;
        let tensor_in = match create_interop_buffer(self.gpu.clone(), &self.runtime, in_size) {
            Ok(b) => b,
            Err(e) => { self.status = Some(format!("interop tensor_in : {e}")); return; }
        };
        let tensor_out = match create_interop_buffer(self.gpu.clone(), &self.runtime, out_size) {
            Ok(b) => b,
            Err(e) => { self.status = Some(format!("interop tensor_out : {e}")); return; }
        };
        let bridge_in = match BgraTensorBridge::new(self.gpu.clone(), in_w, in_h) {
            Ok(b) => b,
            Err(e) => { self.status = Some(format!("bridge_in : {e}")); return; }
        };
        let bridge_out = match BgraTensorBridge::new(self.gpu.clone(), out_w, out_h) {
            Ok(b) => b,
            Err(e) => { self.status = Some(format!("bridge_out : {e}")); return; }
        };
        let bgra_in = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("studio-onnx-bgra-in"),
            size: (in_w as u64) * (in_h as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgra_out = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("studio-onnx-bgra-out"),
            size: (out_w as u64) * (out_h as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("model").to_string();
        info!(model = %name, input = ?(in_w, in_h), output = ?(out_w, out_h), "ONNX filter loaded (zero-copy + resize)");
        if let Some(eng) = self.engine.as_mut() {
            eng.onnx = Some(OnnxState {
                bridge_in,
                bridge_out,
                tensor_in,
                tensor_out,
                bgra_in,
                bgra_out,
                model,
                in_shape,
                in_w,
                in_h,
                out_w,
                out_h,
                capacity_f32: (out_w * out_h * 3) as usize,
                name: name.clone(),
            });
        }
        self.status = Some(format!("Modèle chargé : {name}  ({in_w}×{in_h} → {out_w}×{out_h})"));
    }
}

impl ApplicationHandler for StudioApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Neo Studio")
            .with_inner_size(PhysicalSize::new(1280u32, 720u32));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => { error!("window: {e}"); event_loop.exit(); return; }
        };
        self.window = Some(window.clone());

        let instance = wgpu::Instance::new({
            let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
            d.backends = wgpu::Backends::VULKAN;
            d
        });
        let surface = match instance.create_surface(window.clone()) {
            Ok(s) => s,
            Err(e) => { error!("surface: {e}"); event_loop.exit(); return; }
        };
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })) {
            Ok(a) => a,
            Err(e) => { error!("adapter: {e}"); event_loop.exit(); return; }
        };

        // Request interop if the adapter supports it; fall back gracefully so
        // shader-only use still works on non-NVIDIA / non-Windows GPUs.
        let interop_available = adapter
            .features()
            .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32);
        let required_features = if interop_available {
            wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32
        } else {
            wgpu::Features::empty()
        };

        let limits = adapter.limits();
        let (device, queue) = match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("neo-studio-device"),
            required_features,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::default(),
        })) {
            Ok(p) => p,
            Err(e) => { error!("device: {e}"); event_loop.exit(); return; }
        };
        let gpu = Arc::new(GpuContext {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info: adapter.get_info(),
            limits,
        });

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats.iter().copied()
            .find(|f| matches!(*f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
            .unwrap_or(caps.formats[0]);
        // Prefer a non-vsync present mode so our own frame pacing controls the
        // cadence. Running vsync *and* a manual sleep beats two clocks against
        // each other — that's the micro-stutter. With no-vsync, the pacing
        // sleep alone sets the rhythm (smooth), and "uncapped" just drops it.
        let present_mode = caps.present_modes.iter().copied()
            .find(|m| matches!(*m, wgpu::PresentMode::Mailbox | wgpu::PresentMode::Immediate | wgpu::PresentMode::AutoNoVsync))
            .unwrap_or(wgpu::PresentMode::AutoVsync);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        let resizer = ResizePipeline::new(&gpu.device);

        info!(gpu = %gpu.gpu_name(), interop = interop_available, "Neo Studio ready — import a clip");

        let mut state = StudioState {
            gpu,
            runtime: self.init.runtime.clone(),
            surface,
            surface_format,
            config,
            shaders_dir: self.init.shaders_dir.clone(),
            interop_available,
            resizer,
            status: None,
            uncapped: self.init.no_vsync,
            engine: None,
            egui_ctx,
            egui_state,
            egui_renderer,
            last_present: Instant::now(),
            target_frame_time: Duration::from_secs_f64(1.0 / self.init.fps),
            fps_ema: self.init.fps as f32,
            last_frame: Instant::now(),
        };

        if let Some(path) = self.init.pending_input.take() {
            state.load_video(&path);
        }

        self.state = Some(state);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let (Some(window), Some(state)) = (self.window.as_ref(), self.state.as_mut()) {
            let _ = state.egui_state.on_window_event(window, &event);
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => {
                if let Some(st) = &mut self.state {
                    st.config.width = s.width.max(1);
                    st.config.height = s.height.max(1);
                    st.surface.configure(&st.gpu.device, &st.config);
                    if let Some(eng) = &st.engine {
                        eng.blit.update_window_size(&st.gpu.queue, st.config.width, st.config.height);
                    }
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl StudioApp {
    fn render(&mut self) {
        let Some(window) = self.window.clone() else { return };
        let Some(state) = self.state.as_mut() else { return };

        // Pace to target FPS unless uncapped (then run flat-out).
        if !state.uncapped {
            let now = Instant::now();
            let elapsed = now.duration_since(state.last_present);
            if elapsed < state.target_frame_time {
                std::thread::sleep(state.target_frame_time - elapsed);
            }
        }
        state.last_present = Instant::now();
        let dt = state.last_frame.elapsed().as_secs_f32().max(1e-4);
        state.last_frame = Instant::now();
        state.fps_ema = state.fps_ema * 0.9 + (1.0 / dt) * 0.1;

        // Advance the video, run shader chain + optional ONNX into a BGRA buffer.
        let mut video_src: Option<wgpu::Buffer> = None;
        if let Some(eng) = state.engine.as_mut() {
            if eng.chain.poll_reload() {
                if let Some(err) = eng.chain.last_error() {
                    warn!("shader error: {err}");
                }
            }
            match eng.source.next() {
                Ok(frame) => {
                    if let Err(e) = eng.converter.upload_and_dispatch(&frame) {
                        warn!("convert: {e}");
                    } else {
                        let mut enc = state.gpu.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: Some("studio-chain") });
                        let final_buf = eng.chain.record(&mut enc);
                        state.gpu.queue.submit(std::iter::once(enc.finish()));
                        let src = match final_buf {
                            FinalBuffer::A => eng.chain.buf_a.clone(),
                            FinalBuffer::B => eng.chain.buf_b.clone(),
                        };
                        let (vw, vh) = (eng.width, eng.height);

                        // Optional zero-copy ONNX with resize: video → model res
                        // → infer → model res → video. `display` is what we blit.
                        let mut display = src.clone();
                        let mut drop_model = false;
                        if let Some(onnx) = eng.onnx.as_mut() {
                            state.resizer.run(&state.gpu, &src, vw, vh, &onnx.bgra_in, onnx.in_w, onnx.in_h);
                            if onnx.bridge_in.pack_into(&onnx.bgra_in, onnx.tensor_in.wgpu_buffer()).is_ok() {
                                let in_dptr = onnx.tensor_in.cu_device_ptr;
                                let out_dptr = onnx.tensor_out.cu_device_ptr;
                                let cap = onnx.capacity_f32;
                                let in_shape = onnx.in_shape.clone();
                                let model = &mut onnx.model;
                                // ORT can panic mid-inference; never let it abort the app.
                                let infer = catch_unwind(AssertUnwindSafe(|| {
                                    model.infer_on_device_dynamic(&[in_dptr], &[in_shape.as_slice()], out_dptr, cap)
                                }));
                                match infer {
                                    Ok(Ok(_)) => {
                                        let _ = onnx.bridge_out.unpack_into(onnx.tensor_out.wgpu_buffer(), &onnx.bgra_out);
                                        state.resizer.run(&state.gpu, &onnx.bgra_out, onnx.out_w, onnx.out_h, &eng.bgra_display, vw, vh);
                                        display = eng.bgra_display.clone();
                                    }
                                    Ok(Err(e)) => {
                                        warn!("inference failed, dropping model: {e}");
                                        drop_model = true;
                                    }
                                    Err(_) => {
                                        warn!("inference panicked, dropping model");
                                        drop_model = true;
                                    }
                                }
                            }
                        }
                        if drop_model {
                            eng.onnx = None;
                        }
                        video_src = Some(display);
                    }
                }
                Err(e) => warn!("stream: {e}"),
            }
        }

        // Acquire swapchain.
        let surface_tex = match state.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                state.surface.configure(&state.gpu.device, &state.config);
                return;
            }
            _ => return,
        };
        let view = surface_tex.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Draw video (clears) or just clear when no clip is loaded.
        let mut enc = state.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("studio-blit") });
        if let (Some(eng), Some(src)) = (state.engine.as_ref(), video_src.as_ref()) {
            let bg = eng.blit.make_bind_group(&state.gpu.device, src);
            eng.blit.record(&mut enc, &view, &bg);
        } else {
            let _ = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("studio-clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.06, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        state.gpu.queue.submit(std::iter::once(enc.finish()));

        // ---- egui overlay ----
        let raw_input = state.egui_state.take_egui_input(window.as_ref());
        let fps = state.fps_ema;
        let interop = state.interop_available;
        let video_name = state.engine.as_ref().map(|e| e.name.clone());
        let video_dims = state.engine.as_ref().map(|e| (e.width, e.height));
        let cur_shader = state.engine.as_ref().map(|e| e.selected).unwrap_or(0);
        let model_name = state.engine.as_ref().and_then(|e| e.onnx.as_ref().map(|o| o.name.clone()));
        let status = state.status.clone();

        let mut import_clicked = false;
        let mut newly_selected: Option<usize> = None;
        let mut model_load_clicked = false;
        let mut model_remove_clicked = false;
        let mut uncapped = state.uncapped;

        let full_output = state.egui_ctx.run(raw_input, |ctx| {
            egui::SidePanel::right("controls").default_width(260.0).show(ctx, |ui| {
                ui.heading("Neo Studio");
                ui.separator();
                if ui.button("📂  Import video…").clicked() {
                    import_clicked = true;
                }
                if let Some(s) = &status {
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(255, 170, 60), s);
                }
                ui.separator();
                match (&video_name, video_dims) {
                    (Some(name), Some((w, h))) => {
                        ui.label(format!("Clip: {name}"));
                        ui.label(format!("{w} × {h}   ·   {:.1} FPS", fps));
                        ui.checkbox(&mut uncapped, "Débrider (FPS max, ignore 30)");
                        ui.separator();
                        ui.label("Shader");
                        for (i, (n, _)) in builtins().iter().enumerate() {
                            if ui.selectable_label(cur_shader == i, *n).clicked() {
                                newly_selected = Some(i);
                            }
                        }
                        ui.separator();
                        ui.label("AI model (ONNX · zero-copy)");
                        if interop {
                            if ui.button("＋  Load ONNX…").clicked() {
                                model_load_clicked = true;
                            }
                            match &model_name {
                                Some(m) => {
                                    ui.label(format!("▶ {m}"));
                                    if ui.button("Remove model").clicked() {
                                        model_remove_clicked = true;
                                    }
                                }
                                None => { ui.label("(none — pass-through)"); }
                            }
                        } else {
                            ui.label("⚠ interop unavailable on this GPU");
                        }
                    }
                    _ => {
                        ui.add_space(8.0);
                        ui.label("No clip loaded.");
                        ui.label("Click “Import video…” to load a");
                        ui.label("video file (H.264 / mp4 / mkv / avi).");
                    }
                }
            });
        });

        // Apply UI actions after the closure (borrow checker).
        state.uncapped = uncapped;
        if let Some(i) = newly_selected {
            if let Some(eng) = state.engine.as_mut() {
                if i != eng.selected {
                    eng.selected = i;
                    let body = if i == 0 { None } else { Some(builtins()[i].1) };
                    write_active_shader(&state.shaders_dir, body);
                }
            }
        }
        if model_remove_clicked {
            if let Some(eng) = state.engine.as_mut() {
                eng.onnx = None;
            }
        }

        state.egui_state.handle_platform_output(window.as_ref(), full_output.platform_output);
        let tris = state.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [state.config.width, state.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, delta) in &full_output.textures_delta.set {
            state.egui_renderer.update_texture(&state.gpu.device, &state.gpu.queue, *id, delta);
        }
        let mut enc = state.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("studio-egui") });
        state.egui_renderer.update_buffers(&state.gpu.device, &state.gpu.queue, &mut enc, &tris, &screen);
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("studio-egui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            }).forget_lifetime();
            state.egui_renderer.render(&mut pass, &tris, &screen);
        }
        state.gpu.queue.submit(std::iter::once(enc.finish()));
        for id in &full_output.textures_delta.free {
            state.egui_renderer.free_texture(id);
        }

        surface_tex.present();

        // Open native dialogs *after* presenting (they block this thread).
        if import_clicked {
            if let Some(path) = pick_video_file() {
                state.load_video(&path);
            }
        }
        if model_load_clicked {
            if let Some(path) = pick_onnx_file() {
                state.load_model(&path);
            }
        }
    }
}
