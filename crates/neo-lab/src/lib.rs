//! # neo-lab
//!
//! Neo Lab — live WGSL shader chain editor for video.
//!
//! Decodes a raw H.264 file once into CPU NV12 frames, plays it back in a
//! winit window at the source frame rate, and pipes every frame through a
//! hot-reloadable chain of WGSL compute shaders watched from a directory
//! on disk.
//!
//! ## Pipeline
//!
//! ```text
//! H.264 file ──▶ NVDEC (one-shot decode to RAM, NV12)
//!                 │
//!                 ▼ (per frame)
//!         queue.write_buffer Y, UV
//!                 │
//!                 ▼ wgpu compute NV12 → BGRA  (existing converter)
//!                 │
//!                 ▼  buf_a (BGRA storage buffer)
//!                 │
//!                 ▼ shader chain ping-pong (user-supplied .wgsl files)
//!                 │
//!                 ▼  final_buf (A or B, depending on parity)
//!                 │
//!                 ▼ blit pipeline (fullscreen quad, BGRA → RGBA + letterbox)
//!                 │
//!                 ▼ winit surface present
//! ```
//!
//! ## Hot reload
//!
//! `notify` watches the shaders directory. On file change the chain
//! recompiles every `.wgsl` file (alphabetical order) and atomically
//! swaps the active list. Compilation failures are kept out of the
//! pipeline — the previous successful version stays active and the error
//! is logged + exposed via `ShaderChain::last_error`.

mod blit;
mod chain;
mod model_node;
mod stream;
mod wgpu_infer;

use blit::BlitPipeline;
use chain::{FinalBuffer, ShaderChain};
use model_node::ModelNode;
use stream::StreamSource;
use neo_core::NeoResult;
use neo_gpu::{GpuContext, GpuOptions};
use neo_hwaccel::{CudaRuntime, Nv12ToBgraConverter};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{error, info, warn};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

/// Options accepted by the `lab` subcommand.
pub struct LabOptions {
    pub input: PathBuf,
    pub shaders_dir: PathBuf,
    /// Frames per second for playback. None = use the source rate
    /// inferred from the bitstream metadata, or 30 fps as a fallback.
    pub fps: Option<f64>,
    /// Disable V-sync. Useful for benchmarking the raw GPU throughput
    /// of a shader chain — without it the swap chain caps at the
    /// monitor refresh rate.
    pub no_vsync: bool,
    /// Optional path to an ONNX model run **after** the shader chain.
    /// The model must declare `[1, 3, H, W]` f32 input/output where
    /// H×W matches the video resolution. See `model_node.rs` docs.
    pub model: Option<PathBuf>,
}

/// Run Neo Lab. Blocks until the user closes the window.
pub fn run(opts: LabOptions) -> NeoResult<()> {
    info!(
        input = %opts.input.display(),
        shaders = %opts.shaders_dir.display(),
        "Neo Lab starting"
    );

    // ---- 1. Load the compressed bitstream + create a streaming NVDEC
    //         source. No eager full-clip decode, no 7 GB RAM explosion
    //         on 4K clips; frames are pulled on demand and the parser
    //         loops at EOS.
    let bytes = std::fs::read(&opts.input).map_err(|e| {
        neo_core::NeoError::Decode(format!("read {:?}: {e}", opts.input))
    })?;
    info!(
        size_mb = bytes.len() / (1024 * 1024),
        "loaded compressed bitstream into RAM (the only per-clip buffer Lab keeps)"
    );
    let runtime = Arc::new(CudaRuntime::new(0)?);
    let t0 = Instant::now();
    let source = StreamSource::new(runtime.clone(), bytes)?;
    let width = source.width;
    let height = source.height;
    info!(
        width,
        height,
        probe_ms = t0.elapsed().as_millis(),
        "streaming source ready"
    );

    let fps = opts.fps.unwrap_or(30.0);

    // ---- 2. Spin up the winit event loop with our ApplicationHandler --
    let event_loop = EventLoop::new()
        .map_err(|e| neo_core::NeoError::Pipeline(format!("event loop: {e}")))?;
    let mut app = LabApp::new(LabInit {
        shaders_dir: opts.shaders_dir,
        source: Some(source),
        width,
        height,
        fps,
        no_vsync: opts.no_vsync,
        model: opts.model,
    });
    event_loop
        .run_app(&mut app)
        .map_err(|e| neo_core::NeoError::Pipeline(format!("event loop run: {e}")))?;

    Ok(())
}

struct LabInit {
    shaders_dir: PathBuf,
    /// Moved into `LabState` on the first `resumed()` call. Wrapped in
    /// Option so we can `.take()` it out of the init struct without
    /// needing a `Default` impl on `StreamSource`.
    source: Option<StreamSource>,
    width: u32,
    height: u32,
    fps: f64,
    no_vsync: bool,
    model: Option<PathBuf>,
}

struct LabApp {
    init: LabInit,
    window: Option<Arc<Window>>,
    state: Option<LabState>,
}

impl LabApp {
    fn new(init: LabInit) -> Self {
        Self {
            init,
            window: None,
            state: None,
        }
    }
}

struct LabState {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    gpu: Arc<GpuContext>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    surface_format: wgpu::TextureFormat,

    converter: Nv12ToBgraConverter,
    chain: ShaderChain,
    model: Option<ModelNode>,
    blit: BlitPipeline,

    /// Streaming NVDEC source, moved from `LabInit` once the window
    /// exists. The event loop pulls one frame per render tick.
    source: StreamSource,

    width: u32,
    height: u32,
    last_present: Instant,
    last_log: Instant,
    frames_in_window: u32,
    target_frame_time: Duration,
}

impl ApplicationHandler for LabApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        // ---- Window ----
        // Compute a window size that respects the source aspect ratio
        // and fits comfortably inside any modern desktop monitor
        // (max 1280×720). The compute chain still runs at the source's
        // native resolution; only the displayed surface is scaled.
        let aspect = self.init.width as f64 / self.init.height as f64;
        let (win_w, win_h) = if aspect >= 16.0 / 9.0 {
            (1280u32, (1280.0 / aspect).round() as u32)
        } else {
            ((720.0 * aspect).round() as u32, 720u32)
        };
        let attrs = Window::default_attributes()
            .with_title(format!(
                "Neo Lab — {}×{} source (streaming NVDEC)",
                self.init.width, self.init.height,
            ))
            .with_inner_size(PhysicalSize::new(win_w, win_h));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                error!("window creation: {e}");
                event_loop.exit();
                return;
            }
        };
        self.window = Some(window.clone());

        // ---- wgpu instance + surface ----
        let instance = wgpu::Instance::new({
            let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
            d.backends = wgpu::Backends::VULKAN;
            d
        });
        let surface = match instance.create_surface(window.clone()) {
            Ok(s) => s,
            Err(e) => {
                error!("create_surface: {e}");
                event_loop.exit();
                return;
            }
        };

        // ---- adapter + device + queue ----
        let adapter = match pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        )) {
            Ok(a) => a,
            Err(e) => {
                error!("request_adapter: {e}");
                event_loop.exit();
                return;
            }
        };
        let limits = adapter.limits();
        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("neo-lab-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        )) {
            Ok(p) => p,
            Err(e) => {
                error!("request_device: {e}");
                event_loop.exit();
                return;
            }
        };

        let gpu = Arc::new(GpuContext {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info: adapter.get_info(),
            limits,
        });

        // ---- surface configuration ----
        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| matches!(*f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
            .unwrap_or(surface_caps.formats[0]);
        let present_mode = if self.init.no_vsync {
            // Pick the fastest non-vsync mode the surface supports.
            surface_caps
                .present_modes
                .iter()
                .copied()
                .find(|m| {
                    matches!(
                        *m,
                        wgpu::PresentMode::Immediate
                            | wgpu::PresentMode::Mailbox
                            | wgpu::PresentMode::AutoNoVsync
                    )
                })
                .unwrap_or(wgpu::PresentMode::AutoVsync)
        } else {
            wgpu::PresentMode::AutoVsync
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);

        // ---- converter (NV12 → BGRA) bound to chain.buf_a ----
        // We have to allocate Y/UV/BGRA buffers up-front so we can pass
        // them to the converter in External mode. The chain reuses
        // bgra_buf as its buf_a.
        let w = self.init.width;
        let h = self.init.height;
        let chain = match ShaderChain::new(
            gpu.device.clone(),
            &gpu.queue,
            w,
            h,
            self.init.shaders_dir.clone(),
        ) {
            Ok(c) => c,
            Err(e) => {
                error!("shader chain init: {e}");
                event_loop.exit();
                return;
            }
        };

        // Allocate Y + UV storage buffers. They're separate from the
        // chain's buffers because the chain only deals in BGRA, and the
        // converter binds (dims, y, uv, bgra) — bgra is chain.buf_a.
        let y_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lab-y"),
            size: (w as u64) * (h as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uv_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lab-uv"),
            size: (w as u64) * (h as u64) / 2,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let converter = match Nv12ToBgraConverter::from_external_buffers(
            gpu.clone(),
            w,
            h,
            &y_buf,
            &uv_buf,
            &chain.buf_a,
        ) {
            Ok(c) => c,
            Err(e) => {
                error!("converter init: {e}");
                event_loop.exit();
                return;
            }
        };

        // ---- blit pipeline ----
        let blit = BlitPipeline::new(&gpu.device, &gpu.queue, surface_format, w, h);
        blit.update_window_size(&gpu.queue, config.width, config.height);

        // ---- optional ONNX model (Stage B.1) ----
        let model = match &self.init.model {
            Some(path) => match ModelNode::load(&gpu.device, &gpu.queue, path, w, h) {
                Ok(m) => {
                    info!(
                        model = %path.display(),
                        "inference node ready — chain output will be post-processed by the model"
                    );
                    Some(m)
                }
                Err(e) => {
                    error!(error = %e, path = %path.display(), "failed to load model");
                    event_loop.exit();
                    return;
                }
            },
            None => None,
        };

        let target_frame_time = Duration::from_secs_f64(1.0 / self.init.fps);

        info!(
            gpu = %gpu.gpu_name(),
            backend = ?gpu.backend(),
            window = format!("{}x{}", config.width, config.height),
            target_fps = self.init.fps,
            shaders = chain.nodes(),
            "Neo Lab ready — drop .wgsl files into the shaders dir to filter live"
        );

        let source = self
            .init
            .source
            .take()
            .expect("StreamSource must exist on first resumed()");
        self.state = Some(LabState {
            instance,
            adapter,
            gpu,
            surface,
            config,
            surface_format,
            converter,
            chain,
            model,
            blit,
            source,
            width: w,
            height: h,
            last_present: Instant::now(),
            last_log: Instant::now(),
            frames_in_window: 0,
            target_frame_time,
        });
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                info!("close requested");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(state) = &mut self.state {
                    state.config.width = new_size.width.max(1);
                    state.config.height = new_size.height.max(1);
                    state.surface.configure(&state.gpu.device, &state.config);
                    state.blit.update_window_size(
                        &state.gpu.queue,
                        state.config.width,
                        state.config.height,
                    );
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame(event_loop);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl LabApp {
    fn render_frame(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // 1. Hot-reload check
        if state.chain.poll_reload() {
            if let Some(err) = state.chain.last_error() {
                warn!("shader chain error: {err}");
            } else {
                info!(nodes = state.chain.nodes(), "shader chain reloaded");
            }
        }

        // 2. Frame pacing
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_present);
        if elapsed < state.target_frame_time {
            std::thread::sleep(state.target_frame_time - elapsed);
        }
        state.last_present = Instant::now();

        // 3. Pull the next streamed NV12 frame and run NV12 → BGRA into
        //    chain.buf_a. StreamSource loops the bitstream at EOS by
        //    destroying + recreating the decoder, so this call always
        //    returns a frame unless NVDEC errors out.
        let frame = match state.source.next() {
            Ok(f) => f,
            Err(e) => {
                warn!("stream source next: {e}");
                return;
            }
        };

        if let Err(e) = state.converter.upload_and_dispatch(&frame) {
            warn!("converter dispatch failed: {e}");
            return;
        }

        // 4. Acquire the swapchain texture (wgpu 29 returns the enum
        //    `CurrentSurfaceTexture` directly, not a Result)
        let surface_tex = match state.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                state.surface.configure(&state.gpu.device, &state.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return;
            }
        };
        let view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // 5. Run the user shader chain in its own command buffer so
        //    the optional ONNX post-process can read a fully-flushed
        //    result from the ring-ponged BGRA buffer.
        let mut encoder =
            state
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("lab-frame-chain"),
                });
        let final_buf = state.chain.record(&mut encoder);
        state
            .gpu
            .queue
            .submit(std::iter::once(encoder.finish()));

        // Resolve which chain buffer holds the final pixels.
        let src_buf = match final_buf {
            FinalBuffer::A => state.chain.buf_a.clone(),
            FinalBuffer::B => state.chain.buf_b.clone(),
        };

        // 6. Optional ONNX post-process (Stage B.1: CPU bounce). Skipped
        //    entirely if no --model was passed. The model rewrites
        //    `src_buf` in place so the blit doesn't need to know.
        if let Some(model) = state.model.as_mut() {
            if let Err(e) = model.process(&state.gpu.device, &state.gpu.queue, &src_buf) {
                warn!("model process failed: {e}");
            }
        }

        // 7. Blit the (possibly model-augmented) BGRA buffer onto the
        //    swapchain.
        let mut encoder =
            state
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("lab-frame-blit"),
                });
        let blit_bg = state.blit.make_bind_group(&state.gpu.device, &src_buf);
        state.blit.record(&mut encoder, &view, &blit_bg);
        state
            .gpu
            .queue
            .submit(std::iter::once(encoder.finish()));
        surface_tex.present();

        // 6. FPS log every second
        state.frames_in_window += 1;
        let log_elapsed = state.last_log.elapsed();
        if log_elapsed.as_secs_f64() >= 1.0 {
            let fps = state.frames_in_window as f64 / log_elapsed.as_secs_f64();
            info!(
                fps = format!("{:.1}", fps),
                shaders = state.chain.nodes(),
                "lab"
            );
            state.frames_in_window = 0;
            state.last_log = Instant::now();
        }

        let _ = event_loop;
    }
}

// Force-link the unused fields so dropping the state correctly cleans them up.
#[allow(dead_code)]
fn _force_link(s: &LabState) {
    let _ = &s.instance;
    let _ = &s.adapter;
    let _ = &s.surface_format;
    let _ = &s.width;
    let _ = &s.height;
}
