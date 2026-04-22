//! neo-send — Zerocopy GPU video streamer.
//!
//! NVDEC decode → DtoD → interop Y/UV → wgpu NV12→BGRA → interop BGRA
//! → NVENC encode (reads BGRA from VRAM via registered resource)
//! → TCP send with per-frame timestamps. Zero CPU pixel bounce.
//!
//! Usage:
//!   neo-send --input clip.h264 --addr 0.0.0.0:9000 [--fps 30]

#[path = "protocol.rs"]
mod protocol;
#[path = "zerocopy_stream.rs"]
mod zerocopy_stream;

use neo_core::{NeoError, NeoResult};
use neo_gpu::{GpuContext, GpuOptions};
use neo_hwaccel::{
    interop::create_interop_buffer, CudaRuntime, Nv12ToBgraConverter, NvdecCodec,
};
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB, NV_ENC_CODEC_H264_GUID,
        NV_ENC_INPUT_RESOURCE_TYPE,
    },
    Encoder, EncoderInitParams, ErrorKind,
};
use protocol::FrameHeader;
use std::{
    collections::VecDeque,
    ffi::c_void,
    io::Write,
    net::TcpListener,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{error, info, warn};
use zerocopy_stream::ZerocopyStream;

use clap::Parser;

#[derive(Parser)]
#[command(name = "neo-send", about = "Zerocopy GPU video sender")]
struct Cli {
    /// Path to H.264 Annex-B bitstream.
    #[arg(short, long)]
    input: PathBuf,

    /// Listen address (IP:port).
    #[arg(short, long, default_value = "0.0.0.0:9000")]
    addr: String,

    /// Target FPS for pacing.
    #[arg(long, default_value_t = 30.0)]
    fps: f64,

    /// Loop playback (restart when file ends).
    #[arg(long, default_value_t = true)]
    loop_playback: bool,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        error!("fatal: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> NeoResult<()> {
    // 1. Load bitstream + probe.
    let bytes = std::fs::read(&cli.input)
        .map_err(|e| NeoError::Decode(format!("read: {e}")))?;
    info!(size_mb = bytes.len() / (1024 * 1024), "bitstream loaded");

    let runtime = Arc::new(CudaRuntime::new(0)?);
    let probe = neo_hwaccel::nvdec::probe_dimensions(
        runtime.as_ref(),
        NvdecCodec::cudaVideoCodec_H264,
        &bytes,
    )?;
    let width = probe.display_width;
    let height = probe.display_height;
    info!(width, height, "probed");

    // 2. wgpu with interop features.
    let gpu = Arc::new(
        GpuContext::new_sync(&GpuOptions::interop())
            .map_err(|e| NeoError::HwAccelUnavailable(format!("wgpu interop: {e}")))?,
    );
    info!(gpu = %gpu.gpu_name(), "wgpu interop context");

    // 3. Interop buffers.
    let y_size = (width as u64) * (height as u64);
    let uv_size = y_size / 2;
    let bgra_size = y_size * 4;

    let interop_y = create_interop_buffer(gpu.clone(), &runtime, y_size)?;
    let interop_uv = create_interop_buffer(gpu.clone(), &runtime, uv_size)?;
    let interop_bgra = create_interop_buffer(gpu.clone(), &runtime, bgra_size)?;

    // 4. NV12→BGRA converter bound to interop buffers.
    let converter = Nv12ToBgraConverter::from_external_buffers(
        gpu.clone(),
        width,
        height,
        interop_y.wgpu_buffer(),
        interop_uv.wgpu_buffer(),
        interop_bgra.wgpu_buffer(),
    )?;

    // 5. Zerocopy stream: NVDEC → DtoD → interop Y/UV.
    let mut stream = ZerocopyStream::new(
        runtime.clone(),
        bytes,
        interop_y.cu_device_ptr,
        interop_uv.cu_device_ptr,
    )?;

    // 6. NVENC: register the interop BGRA buffer as input resource.
    let encoder = Encoder::initialize_with_cuda(runtime.ctx.clone())
        .map_err(|e| NeoError::HwAccelUnavailable(format!("NVENC: {e:?}")))?;
    let framerate = cli.fps.round() as u32;
    let mut init = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, width, height);
    init.enable_picture_type_decision().framerate(framerate, 1);
    let session = encoder
        .start_session(NV_ENC_BUFFER_FORMAT_ARGB, init)
        .map_err(|e| NeoError::Encode(format!("session: {e:?}")))?;

    const POOL: usize = 8;
    let mut nvenc_inputs = Vec::with_capacity(POOL);
    for _ in 0..POOL {
        let reg = session
            .register_generic_resource::<()>(
                (),
                NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                interop_bgra.cu_device_ptr as *mut c_void,
                width * 4,
            )
            .map_err(|e| NeoError::Encode(format!("register: {e:?}")))?;
        nvenc_inputs.push(reg);
    }
    let mut bitstreams = Vec::with_capacity(POOL);
    for _ in 0..POOL {
        bitstreams.push(
            session
                .create_output_bitstream()
                .map_err(|e| NeoError::Encode(format!("{e:?}")))?,
        );
    }

    info!("zerocopy encode pipeline ready — zero CPU pixel bounce");

    // 7. Wait for receiver.
    info!(addr = %cli.addr, "waiting for receiver...");
    let listener = TcpListener::bind(&cli.addr)
        .map_err(|e| NeoError::Pipeline(format!("bind: {e}")))?;
    let (mut tcp, peer) = listener
        .accept()
        .map_err(|e| NeoError::Pipeline(format!("accept: {e}")))?;
    tcp.set_nodelay(true).ok();
    info!(peer = %peer, "receiver connected — streaming (zerocopy)");

    // 8. Stream loop — absolute timeline pacing.
    //    Instead of sleeping target_dt after each frame (drifts over time),
    //    we schedule each frame at an absolute time: start + N * target_dt.
    //    This is how real media players pace output.
    let target_dt = Duration::from_secs_f64(1.0 / cli.fps);
    let mut frame_num = 0u32;
    let mut slot = 0usize;
    let mut pending: VecDeque<usize> = VecDeque::new();
    let mut last_log = Instant::now();
    let mut frames_sent = 0u32;
    let mut bytes_sent = 0u64;
    let timeline_start = Instant::now();
    let mut present_index = 0u64;

    loop {
        // Decode next frame (DtoD into interop Y/UV).
        match stream.next() {
            Ok(true) => {}
            Ok(false) => continue,
            Err(e) => {
                warn!("stream: {e}");
                continue;
            }
        }

        // NV12→BGRA via interop (wgpu compute, no CPU).
        converter.dispatch_interop()?;

        // NVENC encode from interop BGRA (VRAM direct).
        if pending.contains(&slot) {
            drain_and_send(
                &mut pending,
                &mut bitstreams,
                &mut tcp,
                &mut frame_num,
                &mut frames_sent,
                &mut bytes_sent,
            )?;
        }
        let result = session.encode_picture(
            &mut nvenc_inputs[slot],
            &mut bitstreams[slot],
            Default::default(),
        );
        pending.push_back(slot);
        match result {
            Ok(()) => drain_and_send(
                &mut pending,
                &mut bitstreams,
                &mut tcp,
                &mut frame_num,
                &mut frames_sent,
                &mut bytes_sent,
            )?,
            Err(e) if e.kind() == ErrorKind::NeedMoreInput => {}
            Err(e) => return Err(NeoError::Encode(format!("{e:?}"))),
        }
        slot = (slot + 1) % POOL;

        // Absolute timeline pacing — sleep until the next scheduled present.
        present_index += 1;
        let next_present = timeline_start + target_dt * present_index as u32;
        let now = Instant::now();
        if next_present > now {
            std::thread::sleep(next_present - now);
        }

        // Log.
        let log_dt = last_log.elapsed();
        if log_dt.as_secs_f64() >= 1.0 {
            let fps = frames_sent as f64 / log_dt.as_secs_f64();
            let mbps = (bytes_sent as f64 * 8.0) / (log_dt.as_secs_f64() * 1_000_000.0);
            info!(
                fps = format!("{fps:.1}"),
                mbps = format!("{mbps:.2}"),
                frame = frame_num,
                mode = "zerocopy",
                "sending"
            );
            frames_sent = 0;
            bytes_sent = 0;
            last_log = Instant::now();
        }
    }
}

fn drain_and_send(
    pending: &mut VecDeque<usize>,
    bitstreams: &mut [nvidia_video_codec_sdk::Bitstream<'_>],
    tcp: &mut std::net::TcpStream,
    frame_num: &mut u32,
    frames_sent: &mut u32,
    bytes_sent: &mut u64,
) -> NeoResult<()> {
    while let Some(i) = pending.pop_front() {
        let lock = bitstreams[i]
            .lock()
            .map_err(|e| NeoError::Encode(format!("lock: {e:?}")))?;
        let data = lock.data();
        if data.is_empty() {
            continue;
        }
        let header = FrameHeader::new(*frame_num, data.len() as u32);
        if let Err(e) = header.write_to(tcp) {
            warn!("send header: {e}");
            std::process::exit(0);
        }
        if let Err(e) = tcp.write_all(data) {
            warn!("send payload: {e}");
            std::process::exit(0);
        }
        *frame_num += 1;
        *frames_sent += 1;
        *bytes_sent += data.len() as u64;
    }
    Ok(())
}
