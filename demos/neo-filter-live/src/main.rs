//! neo-filter-live — Live-swappable ONNX filter streamer.
//!
//! Zero-copy pipeline:
//!   NVDEC → DtoD → interop Y/UV → wgpu NV12→BGRA
//!     → (optional) pack tensor → ONNX filter → unpack → NVENC → TCP
//!
//! Drop any compatible `.onnx` into `--filters-dir` to hot-swap the filter
//! while the stream is running. Delete / clear the dir to pass-through.
//!
//! Compatible filter = 1 input, 1 output, both `[1, 3, H, W]` f32 NCHW
//! (same H/W as the video). Any other arity is rejected without affecting
//! the currently loaded model.

#[path = "../../neo-stream/src/protocol.rs"]
mod protocol;
#[path = "../../neo-stream/src/zerocopy_stream.rs"]
mod zerocopy_stream;

use clap::Parser;
use cudarc::driver::sys::{self as cuda_sys};
use neo_core::{NeoError, NeoResult};
use neo_gpu::{BgraTensorBridge, GpuContext, GpuOptions};
use neo_hwaccel::{
    interop::create_interop_buffer, CudaRuntime, Nv12ToBgraConverter, NvdecCodec,
};
use neo_infer_ort::OnnxModelCuda;
use notify::{Event, EventKind, RecursiveMode, Watcher};
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
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};
use tracing::{error, info, warn};
use zerocopy_stream::ZerocopyStream;

#[derive(Parser)]
#[command(name = "neo-filter-live", about = "Live-swappable ONNX filter streamer")]
struct Cli {
    /// Path to H.264 Annex-B bitstream.
    #[arg(short, long)]
    input: PathBuf,

    /// Directory watched for `.onnx` filters (drop files in to hot-swap).
    #[arg(short, long, default_value = "filters")]
    filters_dir: PathBuf,

    /// Listen address (IP:port).
    #[arg(short, long, default_value = "0.0.0.0:9000")]
    addr: String,

    /// Source FPS.
    #[arg(long, default_value_t = 30.0)]
    fps: f64,
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
    let runtime = Arc::new(CudaRuntime::new(0)?);
    runtime.ctx.bind_to_thread().map_err(|e| NeoError::Cuda(format!("{e:?}")))?;

    let bytes = std::fs::read(&cli.input)
        .map_err(|e| NeoError::Decode(format!("read: {e}")))?;
    info!(size_mb = bytes.len() / (1024 * 1024), "bitstream loaded");

    let probe = neo_hwaccel::nvdec::probe_dimensions(
        runtime.as_ref(),
        NvdecCodec::cudaVideoCodec_H264,
        &bytes,
    )?;
    let width = probe.display_width;
    let height = probe.display_height;
    info!(width, height, "probed video");

    let gpu = Arc::new(
        GpuContext::new_sync(&GpuOptions::interop())
            .map_err(|e| NeoError::HwAccelUnavailable(format!("wgpu interop: {e}")))?,
    );

    let y_size = (width as u64) * (height as u64);
    let uv_size = y_size / 2;
    let bgra_size = y_size * 4;
    let tensor_size = y_size * 3 * 4;

    let interop_y = create_interop_buffer(gpu.clone(), &runtime, y_size)?;
    let interop_uv = create_interop_buffer(gpu.clone(), &runtime, uv_size)?;
    let bgra_buf = create_interop_buffer(gpu.clone(), &runtime, bgra_size)?;
    let tensor_in = create_interop_buffer(gpu.clone(), &runtime, tensor_size)?;
    let tensor_out = create_interop_buffer(gpu.clone(), &runtime, tensor_size)?;

    let converter = Nv12ToBgraConverter::from_external_buffers(
        gpu.clone(),
        width,
        height,
        interop_y.wgpu_buffer(),
        interop_uv.wgpu_buffer(),
        bgra_buf.wgpu_buffer(),
    )?;
    let bridge = BgraTensorBridge::new(gpu.clone(), width, height)?;

    let mut stream = ZerocopyStream::new(
        runtime.clone(),
        bytes,
        interop_y.cu_device_ptr,
        interop_uv.cu_device_ptr,
    )?;

    let encoder = Encoder::initialize_with_cuda(runtime.ctx.clone())
        .map_err(|e| NeoError::HwAccelUnavailable(format!("NVENC: {e:?}")))?;
    let out_fps = cli.fps.round() as u32;
    let mut init = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, width, height);
    init.enable_picture_type_decision().framerate(out_fps, 1);
    let session = encoder
        .start_session(NV_ENC_BUFFER_FORMAT_ARGB, init)
        .map_err(|e| NeoError::Encode(format!("session: {e:?}")))?;

    const POOL: usize = 8;
    let mut nvenc_inputs = Vec::with_capacity(POOL);
    for _ in 0..POOL {
        nvenc_inputs.push(
            session
                .register_generic_resource::<()>(
                    (),
                    NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                    bgra_buf.cu_device_ptr as *mut c_void,
                    width * 4,
                )
                .map_err(|e| NeoError::Encode(format!("register: {e:?}")))?,
        );
    }
    let mut bitstreams = Vec::with_capacity(POOL);
    for _ in 0..POOL {
        bitstreams.push(
            session
                .create_output_bitstream()
                .map_err(|e| NeoError::Encode(format!("{e:?}")))?,
        );
    }

    std::fs::create_dir_all(&cli.filters_dir)
        .map_err(|e| NeoError::Pipeline(format!("create filters dir: {e}")))?;
    info!(dir = %cli.filters_dir.display(), "watching for .onnx filters");

    let (tx, rx) = mpsc::channel::<PathBuf>();
    let tx_w = tx.clone();
    let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        if let Ok(ev) = res {
            if matches!(ev.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                for p in ev.paths {
                    if p.extension().and_then(|s| s.to_str()) == Some("onnx") {
                        let _ = tx_w.send(p);
                    }
                }
            }
        }
    })
    .map_err(|e| NeoError::Pipeline(format!("watcher: {e}")))?;
    watcher
        .watch(&cli.filters_dir, RecursiveMode::NonRecursive)
        .map_err(|e| NeoError::Pipeline(format!("watch: {e}")))?;

    // Prime with any existing .onnx on first run.
    if let Ok(entries) = std::fs::read_dir(&cli.filters_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("onnx") {
                let _ = tx.send(p);
            }
        }
    }

    info!(addr = %cli.addr, "waiting for receiver...");
    let listener = TcpListener::bind(&cli.addr)
        .map_err(|e| NeoError::Pipeline(format!("bind: {e}")))?;
    let (mut tcp, peer) = listener
        .accept()
        .map_err(|e| NeoError::Pipeline(format!("accept: {e}")))?;
    tcp.set_nodelay(true).ok();
    info!(peer = %peer, "receiver connected — streaming");

    let target_dt = Duration::from_secs_f64(1.0 / cli.fps);
    let mut frame_num = 0u32;
    let mut slot = 0usize;
    let mut pending: VecDeque<usize> = VecDeque::new();
    let mut last_log = Instant::now();
    let mut frames_sent = 0u32;
    let mut bytes_sent = 0u64;
    let timeline_start = Instant::now();
    let mut present_index = 0u64;

    let mut model: Option<(OnnxModelCuda, Vec<i64>)> = None;

    loop {
        // Drain pending filter load requests (pick the latest).
        let mut latest: Option<PathBuf> = None;
        while let Ok(p) = rx.try_recv() {
            latest = Some(p);
        }
        if let Some(p) = latest {
            match try_load_filter(&p, runtime.ctx.clone(), width as i64, height as i64) {
                Ok((m, shape)) => {
                    info!(path = %p.display(), shape = ?shape, "filter loaded");
                    model = Some((m, shape));
                }
                Err(e) => warn!(path = %p.display(), "filter rejected: {e}"),
            }
        }

        match stream.next() {
            Ok(true) => {}
            Ok(false) => continue,
            Err(e) => {
                warn!("stream: {e}");
                continue;
            }
        }

        converter.dispatch_interop()?;

        let encode_src_bgra = if let Some((m, in_shape)) = model.as_mut() {
            bridge.pack_into(bgra_buf.wgpu_buffer(), tensor_in.wgpu_buffer())?;
            let r = m.infer_on_device_dynamic(
                &[tensor_in.cu_device_ptr],
                &[in_shape.as_slice()],
                tensor_out.cu_device_ptr,
                (tensor_size / 4) as usize,
            );
            match r {
                Ok(_) => {
                    bridge.unpack_into(tensor_out.wgpu_buffer(), bgra_buf.wgpu_buffer())?;
                }
                Err(e) => {
                    warn!("infer failed, dropping filter: {e}");
                    model = None;
                }
            }
            &bgra_buf
        } else {
            &bgra_buf
        };

        // Suppress unused-var lint — kept for clarity.
        let _ = encode_src_bgra;

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
        let res = session.encode_picture(
            &mut nvenc_inputs[slot],
            &mut bitstreams[slot],
            Default::default(),
        );
        pending.push_back(slot);
        handle_encode_result(
            res,
            &mut pending,
            &mut bitstreams,
            &mut tcp,
            &mut frame_num,
            &mut frames_sent,
            &mut bytes_sent,
        )?;
        slot = (slot + 1) % POOL;

        present_index += 1;
        let next_present = timeline_start + target_dt * present_index as u32;
        let now = Instant::now();
        if next_present > now {
            std::thread::sleep(next_present - now);
        }

        let log_dt = last_log.elapsed();
        if log_dt.as_secs_f64() >= 1.0 {
            let fps = frames_sent as f64 / log_dt.as_secs_f64();
            let mbps = (bytes_sent as f64 * 8.0) / (log_dt.as_secs_f64() * 1_000_000.0);
            let mode = if model.is_some() { "filter" } else { "passthru" };
            info!(mode, fps = format!("{fps:.1}"), mbps = format!("{mbps:.2}"), frame = frame_num, "sending");
            frames_sent = 0;
            bytes_sent = 0;
            last_log = Instant::now();
        }
    }
}

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

/// Load a filter and validate 1-input / 1-output / [1,3,H,W] shape.
fn try_load_filter(
    path: &Path,
    ctx: Arc<cudarc::driver::CudaContext>,
    width: i64,
    height: i64,
) -> std::result::Result<(OnnxModelCuda, Vec<i64>), String> {
    let model = OnnxModelCuda::load(path, ctx, 0).map_err(|e| format!("load: {e}"))?;

    if model.input_count() != 1 {
        return Err(format!(
            "filter must have 1 input, got {}",
            model.input_count()
        ));
    }

    let in_shape = resolve_shape(model.input_shape(0), width, height);
    let expected: i64 = width * height * 3;
    let got: i64 = in_shape.iter().product();
    if got != expected {
        return Err(format!(
            "input shape {:?} has {} elements, expected {} (W*H*3)",
            in_shape, got, expected
        ));
    }

    let out_shape = resolve_shape(model.output_shape(), width, height);
    let got_out: i64 = out_shape.iter().product();
    if got_out != expected {
        return Err(format!(
            "output shape {:?} has {} elements, expected {} (same as input)",
            out_shape, got_out, expected
        ));
    }

    Ok((model, in_shape))
}

fn handle_encode_result(
    res: Result<(), nvidia_video_codec_sdk::EncodeError>,
    pending: &mut VecDeque<usize>,
    bitstreams: &mut [nvidia_video_codec_sdk::Bitstream<'_>],
    tcp: &mut std::net::TcpStream,
    frame_num: &mut u32,
    frames_sent: &mut u32,
    bytes_sent: &mut u64,
) -> NeoResult<()> {
    match res {
        Ok(()) => drain_and_send(pending, bitstreams, tcp, frame_num, frames_sent, bytes_sent),
        Err(e) if e.kind() == ErrorKind::NeedMoreInput => Ok(()),
        Err(e) => Err(NeoError::Encode(format!("{e:?}"))),
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

// Silence cuda_sys unused import when this demo is trimmed further.
#[allow(dead_code)]
fn _unused_keep_cuda_sys() {
    let _ = unsafe { cuda_sys::cuCtxSynchronize() };
}
