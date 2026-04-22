//! neo-recv — Zerocopy GPU video receiver with latency metrics.
//!
//! Connects to neo-send, receives H.264 over TCP, decodes with NVDEC via
//! CaptureMode::Device (DtoD into interop Y/UV), converts NV12→BGRA via
//! dispatch_interop(), and blits to screen. Zero CPU pixel bounce.
//!
//! Usage:
//!   neo-recv --addr 127.0.0.1:9000

#[path = "protocol.rs"]
mod protocol;
#[path = "metrics.rs"]
mod metrics;

use cudarc::driver::sys::{
    self as cuda_sys, CUmemorytype, CUresult, CUDA_MEMCPY2D_st,
};
use metrics::Metrics;
use neo_core::{NeoError, NeoResult};
use neo_gpu::GpuContext;
use neo_hwaccel::{
    interop::{create_interop_buffer, InteropBuffer},
    nvdec::{CaptureMode, Decoder, DeviceHook},
    CudaRuntime, Nv12ToBgraConverter, NvdecCodec,
};
use protocol::{now_us, FrameHeader};
use std::{
    collections::VecDeque,
    ffi::c_void,
    io::Read,
    net::TcpStream,
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

use bytemuck::{Pod, Zeroable};
use clap::Parser;

#[derive(Parser)]
#[command(name = "neo-recv", about = "Zerocopy GPU video receiver with metrics")]
struct Cli {
    /// Sender address (IP:port).
    #[arg(short, long, default_value = "127.0.0.1:9000")]
    addr: String,

    /// Jitter buffer size in milliseconds (0 = no buffer, direct display).
    /// Higher values = smoother playback, higher latency.
    /// For remote desktop: 0. For video streaming: 50-200.
    #[arg(long, default_value_t = 100)]
    buffer_ms: u64,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    info!(addr = %cli.addr, "connecting to sender...");

    let mut tcp = TcpStream::connect(&cli.addr).expect("failed to connect");
    tcp.set_nodelay(true).ok();
    info!("connected");

    // Read the first frame to probe dimensions via a throwaway CPU decoder.
    let (first_header, first_payload) = read_frame(&mut tcp).expect("first frame");
    let first_recv_us = now_us();

    let runtime = Arc::new(CudaRuntime::new(0).expect("CUDA init"));
    let mut probe_dec = Decoder::new(
        runtime.as_ref(),
        NvdecCodec::cudaVideoCodec_H264,
        CaptureMode::Cpu,
    )
    .expect("probe decoder");
    probe_dec.feed(&first_payload).expect("probe feed");
    probe_dec.flush().expect("probe flush");
    let stats = probe_dec.stats().clone();
    let (width, height) = if stats.display_width > 0 {
        (stats.display_width, stats.display_height)
    } else {
        error!("could not determine frame dimensions");
        std::process::exit(1);
    };
    drop(probe_dec);
    info!(width, height, "stream dimensions");

    let buffer_ms = cli.buffer_ms;
    info!(buffer_ms, "jitter buffer configured");

    let event_loop = EventLoop::new().expect("event loop");
    let mut app = RecvApp {
        tcp,
        runtime,
        width,
        height,
        buffer_ms,
        first_frame_data: Some((first_header, first_payload, first_recv_us)),
        window: None,
        state: None,
    };
    event_loop.run_app(&mut app).expect("event loop");
}

fn read_frame(tcp: &mut TcpStream) -> NeoResult<(FrameHeader, Vec<u8>)> {
    let header = FrameHeader::read_from(tcp)
        .map_err(|e| NeoError::Decode(format!("header: {e}")))?;
    let mut payload = vec![0u8; header.payload_len as usize];
    tcp.read_exact(&mut payload)
        .map_err(|e| NeoError::Decode(format!("payload: {e}")))?;
    Ok((header, payload))
}

struct RecvApp {
    tcp: TcpStream,
    runtime: Arc<CudaRuntime>,
    width: u32,
    height: u32,
    buffer_ms: u64,
    first_frame_data: Option<(FrameHeader, Vec<u8>, u64)>,
    window: Option<Arc<Window>>,
    state: Option<RecvState>,
}

/// A packet sitting in the jitter buffer waiting to be decoded.
struct BufferedPacket {
    header: FrameHeader,
    payload: Vec<u8>,
    recv_us: u64,
}

/// Jitter buffer playback state.
enum PlaybackState {
    /// Accumulating packets until we have buffer_ms worth of data.
    Buffering,
    /// Playing back — first_sender_ts is the timestamp of the first packet,
    /// playback_origin is the Instant when we started playing.
    Playing {
        first_sender_ts: u64,
        playback_origin: Instant,
    },
}

// ---- Zerocopy decode hook ---------------------------------------------------

struct HookState {
    width: u32,
    height: u32,
    y_dptr: u64,
    uv_dptr: u64,
    frame_ready: bool,
}

unsafe extern "C" fn hook_trampoline(
    user: *mut c_void,
    width: u32,
    height: u32,
    src_dptr: u64,
    src_pitch: u32,
) -> i32 {
    let state = &mut *(user as *mut HookState);
    if width != state.width || height != state.height {
        return -1;
    }
    let w = width as usize;
    let h = height as usize;
    let pitch = src_pitch as usize;
    if memcpy2d_dtod(src_dptr, pitch, state.y_dptr, w, w, h).is_err() {
        return -1;
    }
    let uv_src = src_dptr + (pitch as u64) * (h as u64);
    if memcpy2d_dtod(uv_src, pitch, state.uv_dptr, w, w, h / 2).is_err() {
        return -1;
    }
    let r = cuda_sys::cuCtxSynchronize();
    if r != CUresult::CUDA_SUCCESS {
        return -1;
    }
    state.frame_ready = true;
    0
}

unsafe fn memcpy2d_dtod(
    src: u64,
    src_pitch: usize,
    dst: u64,
    dst_pitch: usize,
    width_bytes: usize,
    height: usize,
) -> Result<(), ()> {
    let mut cpy = std::mem::MaybeUninit::<CUDA_MEMCPY2D_st>::uninit();
    std::ptr::write_bytes(cpy.as_mut_ptr(), 0, 1);
    let p = cpy.as_mut_ptr();
    (*p).srcMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
    (*p).srcDevice = src;
    (*p).srcPitch = src_pitch;
    (*p).dstMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
    (*p).dstDevice = dst;
    (*p).dstPitch = dst_pitch;
    (*p).WidthInBytes = width_bytes;
    (*p).Height = height;
    let cpy = cpy.assume_init();
    let r = cuda_sys::cuMemcpy2D_v2(&cpy);
    if r != CUresult::CUDA_SUCCESS {
        Err(())
    } else {
        Ok(())
    }
}

// ---- Receiver state ---------------------------------------------------------

struct RecvState {
    gpu: Arc<GpuContext>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    converter: Nv12ToBgraConverter,
    blit: BlitPipeline,

    // Zerocopy decode state.
    decoder: Decoder,
    hook_state: Box<HookState>,

    // Keep interop buffers alive.
    #[allow(dead_code)]
    interop_y: InteropBuffer,
    #[allow(dead_code)]
    interop_uv: InteropBuffer,
    #[allow(dead_code)]
    interop_bgra: InteropBuffer,

    // Jitter buffer.
    jitter_buffer: VecDeque<BufferedPacket>,
    playback_state: PlaybackState,
    buffer_ms: u64,

    // Latency tracking.
    last_send_us: u64,
    last_recv_us: u64,

    metrics: Metrics,
    last_log: Instant,
}

// ---- Blit pipeline ----------------------------------------------------------

const BLIT_SHADER: &str = r#"
struct Dims { src_w: u32, src_h: u32, dst_w: u32, dst_h: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> bgra: array<u32>;
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vid: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    var u = array<vec2<f32>, 3>(vec2(0.0,1.0), vec2(2.0,1.0), vec2(0.0,-1.0));
    var o: VsOut; o.pos = vec4(p[vid], 0.0, 1.0); o.uv = u[vid]; return o;
}
@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let da = f32(dims.dst_w)/f32(dims.dst_h); let sa = f32(dims.src_w)/f32(dims.src_h);
    var uv = in.uv;
    if (da > sa) { uv.x = (uv.x-0.5)*(da/sa)+0.5; } else { uv.y = (uv.y-0.5)*(sa/da)+0.5; }
    if (uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0) { return vec4(0.0,0.0,0.0,1.0); }
    let x=u32(uv.x*f32(dims.src_w)); let y=u32(uv.y*f32(dims.src_h));
    let idx=min(y,dims.src_h-1u)*dims.src_w+min(x,dims.src_w-1u);
    let px=bgra[idx];
    return vec4(f32((px>>16u)&0xffu)/255.0, f32((px>>8u)&0xffu)/255.0, f32(px&0xffu)/255.0, 1.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlitDims { src_w: u32, src_h: u32, dst_w: u32, dst_h: u32 }

struct BlitPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    dims_buf: wgpu::Buffer,
    src_w: u32, src_h: u32,
}

impl BlitPipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, fmt: wgpu::TextureFormat, w: u32, h: u32) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("recv-blit"), source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[Some(&bind_layout)], immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, layout: Some(&pl),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format: fmt, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default() }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview_mask: None, cache: None,
        });
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        queue.write_buffer(&dims_buf, 0, bytemuck::bytes_of(&BlitDims { src_w: w, src_h: h, dst_w: w, dst_h: h }));
        Self { pipeline, bind_layout, dims_buf, src_w: w, src_h: h }
    }
    fn update_window(&self, queue: &wgpu::Queue, dw: u32, dh: u32) {
        queue.write_buffer(&self.dims_buf, 0, bytemuck::bytes_of(&BlitDims { src_w: self.src_w, src_h: self.src_h, dst_w: dw, dst_h: dh }));
    }
    fn make_bg(&self, device: &wgpu::Device, src: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
            ],
        })
    }
    fn record(&self, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, bg: &wgpu::BindGroup) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view, depth_slice: None, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

// ---- ApplicationHandler -----------------------------------------------------

impl ApplicationHandler for RecvApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let aspect = self.width as f64 / self.height as f64;
        let (ww, wh) = if aspect >= 16.0 / 9.0 {
            (1280u32, (1280.0 / aspect).round() as u32)
        } else {
            ((720.0 * aspect).round() as u32, 720u32)
        };
        let attrs = Window::default_attributes()
            .with_title(format!(
                "Neo Recv (zerocopy, buf={}ms) — {}x{}",
                self.buffer_ms, self.width, self.height
            ))
            .with_inner_size(PhysicalSize::new(ww, wh));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => { error!("window: {e}"); event_loop.exit(); return; }
        };
        self.window = Some(window.clone());

        // wgpu with interop.
        let instance = wgpu::Instance::new({
            let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
            d.backends = wgpu::Backends::VULKAN;
            d
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).expect("adapter");
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("neo-recv-zerocopy"),
            required_features: wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })).expect("device");
        let gpu = Arc::new(GpuContext {
            device: Arc::new(device), queue: Arc::new(queue),
            adapter_info: adapter.get_info(), limits,
        });

        // Surface config.
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats.iter().copied()
            .find(|f| matches!(*f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt,
            width: size.width.max(1), height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync, desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0], view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);

        // Interop buffers.
        let y_size = (self.width as u64) * (self.height as u64);
        let uv_size = y_size / 2;
        let bgra_size = y_size * 4;
        let interop_y = create_interop_buffer(gpu.clone(), &self.runtime, y_size).expect("interop Y");
        let interop_uv = create_interop_buffer(gpu.clone(), &self.runtime, uv_size).expect("interop UV");
        let interop_bgra = create_interop_buffer(gpu.clone(), &self.runtime, bgra_size).expect("interop BGRA");

        // Converter.
        let converter = Nv12ToBgraConverter::from_external_buffers(
            gpu.clone(), self.width, self.height,
            interop_y.wgpu_buffer(), interop_uv.wgpu_buffer(), interop_bgra.wgpu_buffer(),
        ).expect("converter");

        // Blit.
        let blit = BlitPipeline::new(&gpu.device, &gpu.queue, fmt, self.width, self.height);
        blit.update_window(&gpu.queue, config.width, config.height);

        // Zerocopy decoder with device hook.
        let mut hook_state = Box::new(HookState {
            width: self.width,
            height: self.height,
            y_dptr: interop_y.cu_device_ptr,
            uv_dptr: interop_uv.cu_device_ptr,
            frame_ready: false,
        });
        let hook = DeviceHook {
            callback: hook_trampoline,
            user: &mut *hook_state as *mut HookState as *mut c_void,
        };
        let mut decoder = Decoder::new(
            self.runtime.as_ref(), NvdecCodec::cudaVideoCodec_H264, CaptureMode::Device(hook),
        ).expect("decoder");

        // Queue first frame into the jitter buffer instead of immediately decoding.
        let mut jitter_buffer = VecDeque::with_capacity(256);
        let buffer_ms = self.buffer_ms;

        let (playback_state, first_send_us, first_recv_us) = if let Some((hdr, payload, recv_us)) = self.first_frame_data.take() {
            let send_us = hdr.timestamp_us;
            jitter_buffer.push_back(BufferedPacket { header: hdr, payload, recv_us });
            if buffer_ms == 0 {
                // No jitter buffer — start playing immediately (remote desktop mode).
                (PlaybackState::Playing {
                    first_sender_ts: send_us,
                    playback_origin: Instant::now(),
                }, send_us, recv_us)
            } else {
                (PlaybackState::Buffering, send_us, recv_us)
            }
        } else {
            (PlaybackState::Buffering, 0, 0)
        };

        self.tcp.set_nonblocking(true).ok();

        info!(gpu = %gpu.gpu_name(), mode = "zerocopy", buffer_ms, "receiver ready");

        self.state = Some(RecvState {
            gpu, surface, config, converter, blit,
            decoder, hook_state,
            interop_y, interop_uv, interop_bgra,
            jitter_buffer,
            playback_state,
            buffer_ms,
            last_send_us: first_send_us,
            last_recv_us: first_recv_us,
            metrics: Metrics::new(2.0),
            last_log: Instant::now(),
        });
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(KeyCode::Escape), state: ElementState::Pressed, .. }, ..
            } => event_loop.exit(),
            WindowEvent::Resized(s) => {
                if let Some(st) = &mut self.state {
                    st.config.width = s.width.max(1);
                    st.config.height = s.height.max(1);
                    st.surface.configure(&st.gpu.device, &st.config);
                    st.blit.update_window(&st.gpu.queue, st.config.width, st.config.height);
                }
            }
            WindowEvent::RedrawRequested => self.render(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window { w.request_redraw(); }
    }
}

impl RecvApp {
    fn render(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else { return; };

        // 1. Read all available TCP packets into the jitter buffer.
        self.tcp.set_nonblocking(true).ok();
        loop {
            match FrameHeader::read_from(&mut self.tcp) {
                Ok(hdr) => {
                    self.tcp.set_nonblocking(false).ok();
                    self.tcp.set_read_timeout(Some(Duration::from_millis(200))).ok();
                    let recv_us = now_us();
                    let mut payload = vec![0u8; hdr.payload_len as usize];
                    match self.tcp.read_exact(&mut payload) {
                        Ok(()) => {
                            state.jitter_buffer.push_back(BufferedPacket {
                                header: hdr,
                                payload,
                                recv_us,
                            });
                        }
                        Err(_) => break,
                    }
                    self.tcp.set_nonblocking(true).ok();
                }
                Err(_) => break,
            }
        }

        // 2. Jitter buffer state machine.
        match &state.playback_state {
            PlaybackState::Buffering => {
                // Check if we have enough data to start.
                if state.jitter_buffer.len() >= 2 {
                    let first_ts = state.jitter_buffer.front().unwrap().header.timestamp_us;
                    let last_ts = state.jitter_buffer.back().unwrap().header.timestamp_us;
                    let buffered_us = last_ts.saturating_sub(first_ts);
                    if buffered_us >= state.buffer_ms * 1000 || state.buffer_ms == 0 {
                        info!(
                            buffered_ms = buffered_us / 1000,
                            packets = state.jitter_buffer.len(),
                            "jitter buffer primed — starting playback"
                        );
                        state.playback_state = PlaybackState::Playing {
                            first_sender_ts: first_ts,
                            playback_origin: Instant::now(),
                        };
                    }
                }
            }
            PlaybackState::Playing { .. } => {}
        }

        // 3. If playing, decode frames. Two modes:
        //    - buffer_ms == 0 (remote-desktop / live): drain the entire
        //      jitter buffer every tick. H.264 has inter-frame deps so we
        //      must decode them all in order, but we only present the
        //      latest. This keeps glass-to-glass latency bounded by
        //      decode_time × buffered_frames (a few ms at NVDEC speed),
        //      not by accumulated network/setup backlog.
        //    - buffer_ms > 0 (file streaming): PTS-paced playback against
        //      the first-packet timeline, as before.
        let mut frames_decoded = 0u32;
        let mut total_payload_bytes = 0usize;
        if let PlaybackState::Playing { first_sender_ts, playback_origin } = &state.playback_state {
            if state.buffer_ms == 0 {
                while let Some(pkt) = state.jitter_buffer.pop_front() {
                    total_payload_bytes += pkt.payload.len();
                    state.last_send_us = pkt.header.timestamp_us;
                    state.last_recv_us = pkt.recv_us;
                    state.hook_state.frame_ready = false;
                    let _ = state.decoder.feed(&pkt.payload);
                    if state.hook_state.frame_ready {
                        frames_decoded += 1;
                    }
                }
            } else {
                let elapsed_us = playback_origin.elapsed().as_micros() as u64;
                let first_ts = *first_sender_ts;
                while let Some(front) = state.jitter_buffer.front() {
                    let frame_pts_us = front.header.timestamp_us.saturating_sub(first_ts);
                    if frame_pts_us <= elapsed_us {
                        let pkt = state.jitter_buffer.pop_front().unwrap();
                        total_payload_bytes += pkt.payload.len();
                        state.last_send_us = pkt.header.timestamp_us;
                        state.last_recv_us = pkt.recv_us;
                        state.hook_state.frame_ready = false;
                        let _ = state.decoder.feed(&pkt.payload);
                        if state.hook_state.frame_ready {
                            frames_decoded += 1;
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        // 4. Dispatch NV12→BGRA for the last decoded frame.
        if frames_decoded > 0 {
            let decode_start = Instant::now();
            if let Err(e) = state.converter.dispatch_interop() {
                warn!("dispatch_interop: {e}");
            }
            let decode_us = decode_start.elapsed().as_micros() as f64;
            let network_us = (state.last_recv_us as f64) - (state.last_send_us as f64);
            let total_us = (now_us() as f64) - (state.last_send_us as f64);
            state.metrics.record(total_payload_bytes, network_us, decode_us, total_us);
        }

        // 5. Blit interop BGRA to screen.
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
        let mut enc = state.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let bg = state.blit.make_bg(&state.gpu.device, state.interop_bgra.wgpu_buffer());
        state.blit.record(&mut enc, &view, &bg);
        state.gpu.queue.submit(std::iter::once(enc.finish()));
        surface_tex.present();

        // 6. Log.
        if state.last_log.elapsed().as_secs_f64() >= 1.0 {
            let snap = state.metrics.snapshot();
            let buf_depth = state.jitter_buffer.len();
            info!(buf = buf_depth, "{snap}");
            state.last_log = Instant::now();
        }
    }
}
