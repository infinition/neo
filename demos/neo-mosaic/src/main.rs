//! neo-mosaic — Multi-stream GPU video wall (zerocopy).
//!
//! Decodes N copies of the same H.264 file via NVDEC, each with a
//! `CaptureMode::Device` hook that DtoD-copies NV12 planes directly into
//! CUDA↔Vulkan interop buffers — zero CPU bounce. A different compute
//! shader is applied per tile, the results are composited into a single
//! grid, and the grid is blitted to a winit surface.
//!
//! Data flow (per tile, per frame):
//!   NVDEC surface ──DtoD──▶ interop Y/UV ──wgpu compute──▶ BGRA
//!     ──filter compute──▶ tile_out ──compositor──▶ grid_buf ──blit──▶ screen
//!
//! Usage:
//!   neo-mosaic --input clip.h264 [--grid 3x3] [--no-vsync]

mod blit;
mod filters;
mod mosaic;
mod stream;

use blit::BlitPipeline;
use mosaic::MosaicGrid;
use neo_hwaccel::{interop::create_interop_buffer, CudaRuntime, Nv12ToBgraConverter};
use neo_gpu::GpuContext;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use stream::ZerocopyStream;
use tracing::{error, info, warn};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use clap::Parser;

#[derive(Parser)]
#[command(name = "neo-mosaic", about = "Multi-stream GPU video wall (zerocopy)")]
struct Cli {
    /// Path to H.264 Annex-B bitstream.
    #[arg(short, long)]
    input: std::path::PathBuf,

    /// Grid layout, e.g. "2x2", "3x3", "4x4".
    #[arg(short, long, default_value = "3x3")]
    grid: String,

    /// Disable V-sync for raw throughput measurement.
    #[arg(long)]
    no_vsync: bool,

    /// Target playback FPS (default 30).
    #[arg(long, default_value_t = 30.0)]
    fps: f64,
}

fn parse_grid(s: &str) -> (u32, u32) {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() == 2 {
        if let (Ok(c), Ok(r)) = (parts[0].parse(), parts[1].parse()) {
            return (c, r);
        }
    }
    eprintln!("Invalid grid format '{}', using 3x3", s);
    (3, 3)
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let (cols, rows) = parse_grid(&cli.grid);
    let n_tiles = (cols * rows) as usize;

    info!(
        input = %cli.input.display(),
        grid = %cli.grid,
        tiles = n_tiles,
        "neo-mosaic (zerocopy) starting"
    );

    // 1. Load bitstream.
    let bytes = std::fs::read(&cli.input).expect("failed to read input file");
    info!(size_mb = bytes.len() / (1024 * 1024), "bitstream loaded");

    // 2. CUDA runtime.
    let runtime = Arc::new(CudaRuntime::new(0).expect("CUDA init failed"));

    // 3. Probe dimensions from the bitstream (fast SPS probe).
    let probe = neo_hwaccel::nvdec::probe_dimensions(
        runtime.as_ref(),
        neo_hwaccel::NvdecCodec::cudaVideoCodec_H264,
        &bytes,
    )
    .expect("probe failed");
    let tile_w = probe.display_width;
    let tile_h = probe.display_height;
    info!(tile_w, tile_h, "probed dimensions");

    // 4. winit event loop — pass everything needed for deferred init.
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App {
        runtime,
        bytes,
        cols,
        rows,
        tile_w,
        tile_h,
        no_vsync: cli.no_vsync,
        fps: cli.fps,
        window: None,
        state: None,
    };
    event_loop.run_app(&mut app).expect("event loop run");
}

struct App {
    runtime: Arc<CudaRuntime>,
    bytes: Vec<u8>,
    cols: u32,
    rows: u32,
    tile_w: u32,
    tile_h: u32,
    no_vsync: bool,
    fps: f64,
    window: Option<Arc<Window>>,
    state: Option<AppState>,
}

/// Per-tile state: zerocopy stream + interop buffers + converter.
struct TileState {
    stream: ZerocopyStream,
    converter: Nv12ToBgraConverter,
    // Keep interop buffers alive — dropping them invalidates the CUDA dptrs.
    #[allow(dead_code)]
    interop_y: neo_hwaccel::interop::InteropBuffer,
    #[allow(dead_code)]
    interop_uv: neo_hwaccel::interop::InteropBuffer,
    #[allow(dead_code)]
    interop_bgra: neo_hwaccel::interop::InteropBuffer,
}

struct AppState {
    gpu: Arc<GpuContext>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    tiles: Vec<TileState>,
    grid: MosaicGrid,
    blit: BlitPipeline,
    last_present: Instant,
    last_log: Instant,
    frames_rendered: u32,
    target_frame_time: Duration,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let n = (self.cols * self.rows) as usize;
        let grid_w = self.tile_w * self.cols;
        let grid_h = self.tile_h * self.rows;
        let aspect = grid_w as f64 / grid_h as f64;
        let (win_w, win_h) = if aspect >= 16.0 / 9.0 {
            (1600u32, (1600.0 / aspect).round() as u32)
        } else {
            ((900.0 * aspect).round() as u32, 900u32)
        };

        let filter_names: Vec<String> = filters::builtins()
            .iter()
            .map(|(n, _)| n.to_string())
            .collect();
        let title_filters: Vec<&str> = (0..n)
            .map(|i| filter_names[i % filter_names.len()].as_str())
            .collect();

        let attrs = Window::default_attributes()
            .with_title(format!(
                "Neo Mosaic (zerocopy) — {}x{} grid  [{}]",
                self.cols,
                self.rows,
                title_filters.join(", ")
            ))
            .with_inner_size(PhysicalSize::new(win_w, win_h));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                error!("window: {e}");
                event_loop.exit();
                return;
            }
        };
        self.window = Some(window.clone());

        // ---- wgpu with interop features ----
        let instance = wgpu::Instance::new({
            let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
            d.backends = wgpu::Backends::VULKAN;
            d
        });
        let surface = match instance.create_surface(window.clone()) {
            Ok(s) => s,
            Err(e) => {
                error!("surface: {e}");
                event_loop.exit();
                return;
            }
        };
        let adapter = match pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        )) {
            Ok(a) => a,
            Err(e) => {
                error!("adapter: {e}");
                event_loop.exit();
                return;
            }
        };
        let limits = adapter.limits();
        // Request VULKAN_EXTERNAL_MEMORY_WIN32 for CUDA↔Vulkan interop.
        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("neo-mosaic-zerocopy"),
                required_features: wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32,
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        )) {
            Ok(p) => p,
            Err(e) => {
                error!("device: {e}");
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

        // Surface config.
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    *f,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .unwrap_or(caps.formats[0]);
        let present_mode = if self.no_vsync {
            caps.present_modes
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
            format: fmt,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);

        // ---- Per-tile zerocopy setup ----
        let y_size = (self.tile_w as u64) * (self.tile_h as u64);
        let uv_size = y_size / 2;
        let bgra_size = y_size * 4;

        let _tiles_placeholder: Vec<TileState> = Vec::with_capacity(n);

        // We need to build tiles first, then collect bgra buffer refs.
        // Two-pass: create interop buffers + streams, then build the grid.
        struct PreTile {
            stream: ZerocopyStream,
            converter: Nv12ToBgraConverter,
            interop_y: neo_hwaccel::interop::InteropBuffer,
            interop_uv: neo_hwaccel::interop::InteropBuffer,
            interop_bgra: neo_hwaccel::interop::InteropBuffer,
        }

        let mut pre_tiles: Vec<PreTile> = Vec::with_capacity(n);
        for i in 0..n {
            info!(tile = i, "creating interop buffers...");
            let interop_y =
                create_interop_buffer(gpu.clone(), &self.runtime, y_size).expect("interop Y");
            let interop_uv =
                create_interop_buffer(gpu.clone(), &self.runtime, uv_size).expect("interop UV");
            let interop_bgra =
                create_interop_buffer(gpu.clone(), &self.runtime, bgra_size).expect("interop BGRA");

            // Converter bound to interop buffers.
            let converter = Nv12ToBgraConverter::from_external_buffers(
                gpu.clone(),
                self.tile_w,
                self.tile_h,
                interop_y.wgpu_buffer(),
                interop_uv.wgpu_buffer(),
                interop_bgra.wgpu_buffer(),
            )
            .expect("converter init");

            // Zerocopy stream: NVDEC hook writes DtoD to interop Y/UV.
            let mut stream = ZerocopyStream::new(
                self.runtime.clone(),
                self.bytes.clone(),
                interop_y.cu_device_ptr,
                interop_uv.cu_device_ptr,
            )
            .expect("ZerocopyStream init");

            // Advance each stream by a different amount so tiles are out of sync.
            for _ in 0..(i * 5) {
                let _ = stream.next();
                let _ = converter.dispatch_interop();
            }

            pre_tiles.push(PreTile {
                stream,
                converter,
                interop_y,
                interop_uv,
                interop_bgra,
            });
        }

        // Collect refs to BGRA buffers for the mosaic grid.
        let bgra_refs: Vec<&wgpu::Buffer> = pre_tiles
            .iter()
            .map(|t| t.interop_bgra.wgpu_buffer())
            .collect();

        // Mosaic grid with external input buffers (interop BGRA).
        let grid = MosaicGrid::new(
            &gpu.device,
            &gpu.queue,
            self.tile_w,
            self.tile_h,
            self.cols,
            self.rows,
            Some(&bgra_refs),
        );

        // Move pre_tiles into final TileState.
        let tiles: Vec<TileState> = pre_tiles
            .into_iter()
            .map(|t| TileState {
                stream: t.stream,
                converter: t.converter,
                interop_y: t.interop_y,
                interop_uv: t.interop_uv,
                interop_bgra: t.interop_bgra,
            })
            .collect();

        // Blit pipeline.
        let blit = BlitPipeline::new(&gpu.device, &gpu.queue, fmt, grid.grid_w, grid.grid_h);
        blit.update_window_size(&gpu.queue, config.width, config.height);

        info!(
            gpu = %gpu.gpu_name(),
            tiles = tiles.len(),
            grid = format!("{}x{}", grid.grid_w, grid.grid_h),
            "mosaic (zerocopy) ready — zero CPU bounce"
        );

        self.state = Some(AppState {
            gpu,
            surface,
            config,
            tiles,
            grid,
            blit,
            last_present: Instant::now(),
            last_log: Instant::now(),
            frames_rendered: 0,
            target_frame_time: Duration::from_secs_f64(1.0 / self.fps),
        });
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let Some(s) = &mut self.state {
                    s.config.width = new_size.width.max(1);
                    s.config.height = new_size.height.max(1);
                    s.surface.configure(&s.gpu.device, &s.config);
                    s.blit
                        .update_window_size(&s.gpu.queue, s.config.width, s.config.height);
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame(event_loop);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl App {
    fn render_frame(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Frame pacing.
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_present);
        if elapsed < state.target_frame_time {
            std::thread::sleep(state.target_frame_time - elapsed);
        }
        state.last_present = Instant::now();

        // Per tile: decode one frame (DtoD into interop Y/UV) then
        // dispatch the NV12→BGRA compute (interop, no CPU).
        for (i, tile) in state.tiles.iter_mut().enumerate() {
            match tile.stream.next() {
                Ok(true) => {
                    if let Err(e) = tile.converter.dispatch_interop() {
                        warn!(tile = i, "dispatch_interop: {e}");
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    warn!(tile = i, "stream: {e}");
                }
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
        let view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Run tile filters + grid composition.
        let mut encoder =
            state
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mosaic-frame"),
                });
        state.grid.record(&mut encoder);
        state.gpu.queue.submit(std::iter::once(encoder.finish()));

        // Blit grid to screen.
        let mut encoder =
            state
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mosaic-blit"),
                });
        let blit_bg = state
            .blit
            .make_bind_group(&state.gpu.device, &state.grid.grid_buf);
        state.blit.record(&mut encoder, &view, &blit_bg);
        state.gpu.queue.submit(std::iter::once(encoder.finish()));
        surface_tex.present();

        // FPS counter.
        state.frames_rendered += 1;
        let log_elapsed = state.last_log.elapsed();
        if log_elapsed.as_secs_f64() >= 1.0 {
            let fps = state.frames_rendered as f64 / log_elapsed.as_secs_f64();
            info!(
                fps = format!("{fps:.1}"),
                streams = state.tiles.len(),
                grid = format!("{}x{}", self.cols, self.rows),
                mode = "zerocopy",
                "mosaic"
            );
            state.frames_rendered = 0;
            state.last_log = Instant::now();
        }

        let _ = event_loop;
    }
}
