//! neo-desktop-send — GPU remote desktop sender (zerocopy).
//!
//! Pipeline (no CPU pixel bounce):
//!
//! ```text
//!   DXGI Desktop Duplication
//!     └─ ID3D11Texture2D (BGRA, GPU)
//!         └─ CopySubresourceRegion → intermediate D3D11 tex (GPU)
//!             └─ CUDA-Graphics interop → CUarray
//!                 └─ cuMemcpy2D (DtoD) → CUdeviceptr (BGRA, pitched)
//!                     └─ NVENC encode (registered as CUDADEVICEPTR/ARGB)
//!                         └─ H.264 NAL → TCP
//! ```
//!
//! Wire protocol matches `neo-stream`'s `FrameHeader` so the existing
//! `neo-recv.exe` works as the receiver:
//!   magic(4 "NEOS") | frame_num(4) | timestamp_us(8) | payload_len(4)
//!
//! Usage:
//!   neo-desktop-send.exe --addr 0.0.0.0:9000 --fps 60 --monitor 0

mod capture;
mod cuda_d3d11;

use capture::ZeroCopyCapture;
use clap::Parser;
use cudarc::driver::sys::{self as cu, CUdeviceptr, CUresult};
use neo_core::{NeoError, NeoResult};
use neo_hwaccel::CudaRuntime;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB, NV_ENC_CODEC_H264_GUID,
        NV_ENC_INPUT_RESOURCE_TYPE, NV_ENC_PRESET_P3_GUID,
        NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_LOW_LATENCY,
    },
    Encoder, EncoderInitParams, ErrorKind,
};
use std::{
    collections::VecDeque,
    ffi::c_void,
    io::Write,
    net::TcpListener,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tracing::{error, info, warn};

const MAGIC: u32 = 0x4E454F53; // "NEOS"
const POOL: usize = 4;

#[derive(Parser, Debug)]
#[command(
    name = "neo-desktop-send",
    about = "GPU remote desktop sender (DXGI + CUDA-D3D11 interop + NVENC, zerocopy)"
)]
struct Cli {
    /// Listen address (IP:port).
    #[arg(short, long, default_value = "0.0.0.0:9000")]
    addr: String,

    /// Target FPS for capture pacing.
    #[arg(long, default_value_t = 60.0)]
    fps: f64,

    /// Monitor index (0 = primary).
    #[arg(long, default_value_t = 0)]
    monitor: u32,

    /// Render the OS cursor into the stream (not yet implemented — DXGI
    /// delivers it out-of-band; the BGRA texture itself is cursor-less).
    #[arg(long, default_value_t = false)]
    show_cursor: bool,

    /// Optional target average bitrate (bits/s). When set, overrides QP.
    #[arg(long)]
    bitrate: Option<u32>,

    /// Optional constant QP (0–51, lower = better quality). Overridden by
    /// `--bitrate` if both are set. Default: encoder's preset default.
    #[arg(long)]
    qp: Option<u32>,

    /// GOP length in frames (distance between IDRs). Default: fps × 2.
    #[arg(long)]
    gop: Option<u32>,

    /// Force the low-latency tuning info (always on for remote desktop).
    #[arg(long, default_value_t = true)]
    low_latency: bool,
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

fn write_header(w: &mut impl Write, frame_num: u32, payload_len: u32) -> std::io::Result<()> {
    let ts = now_us();
    w.write_all(&MAGIC.to_le_bytes())?;
    w.write_all(&frame_num.to_le_bytes())?;
    w.write_all(&ts.to_le_bytes())?;
    w.write_all(&payload_len.to_le_bytes())?;
    Ok(())
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
    if cli.show_cursor {
        warn!(
            "--show-cursor: DXGI delivers the cursor out-of-band; not yet \
             composited onto the stream"
        );
    }

    // 1. CUDA runtime — bind primary context to *this* thread for the entire
    //    capture/encode loop.
    let runtime = Arc::new(CudaRuntime::new(0)?);
    runtime
        .ctx
        .bind_to_thread()
        .map_err(|e| NeoError::Cuda(format!("bind_to_thread: {e:?}")))?;

    // 2. Zero-copy capture (D3D11 device on adapter 0, registered with CUDA).
    let mut cap = ZeroCopyCapture::new(cli.monitor)
        .map_err(|e| NeoError::Pipeline(format!("screen capture: {e}")))?;
    let enc_width = cap.enc_width;
    let enc_height = cap.enc_height;

    // 3. Allocate the destination CUDA pitched buffer (BGRA).
    //    NVENC reads from this buffer every frame, and the capture writes
    //    into it via cuMemcpy2D.
    let (dst_dptr, dst_pitch) = alloc_pitched_bgra(enc_width, enc_height)?;
    info!(
        bytes = dst_pitch * enc_height as usize,
        pitch = dst_pitch,
        "CUDA pitched BGRA buffer allocated"
    );

    // 4. NVENC — encoder + session.
    let encoder = Encoder::initialize_with_cuda(runtime.ctx.clone())
        .map_err(|e| NeoError::HwAccelUnavailable(format!("NVENC: {e:?}")))?;
    let framerate = cli.fps.round().max(1.0) as u32;
    let mut init = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, enc_width, enc_height);
    init.enable_picture_type_decision()
        .framerate(framerate, 1)
        .preset_guid(NV_ENC_PRESET_P3_GUID)
        .tuning_info(NV_ENC_TUNING_INFO_LOW_LATENCY);

    let session = encoder
        .start_session(NV_ENC_BUFFER_FORMAT_ARGB, init)
        .map_err(|e| NeoError::Encode(format!("session: {e:?}")))?;

    if cli.bitrate.is_some() || cli.qp.is_some() || cli.gop.is_some() {
        warn!(
            bitrate = ?cli.bitrate,
            qp = ?cli.qp,
            gop = ?cli.gop,
            "rate-control / GOP CLI flags are accepted but currently routed \
             through preset defaults (P3 + LOW_LATENCY) — fine-grained NV_ENC_CONFIG \
             plumbing is a TODO"
        );
    }

    // 5. Register the CUDA pitched buffer with NVENC POOL times so we can
    //    rotate slots (encoder pipelining).
    let mut nvenc_inputs = Vec::with_capacity(POOL);
    for _ in 0..POOL {
        let reg = session
            .register_generic_resource::<()>(
                (),
                NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                dst_dptr as *mut c_void,
                dst_pitch as u32,
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

    info!(
        enc = format!("{enc_width}x{enc_height}"),
        fps = framerate,
        preset = "P3",
        tuning = "LOW_LATENCY",
        "NVENC ready (zerocopy: D3D11 → CUDA → NVENC)"
    );

    // 6. Wait for receiver.
    info!(addr = %cli.addr, "waiting for receiver...");
    let listener = TcpListener::bind(&cli.addr)
        .map_err(|e| NeoError::Pipeline(format!("bind: {e}")))?;
    let (mut tcp, peer) = listener
        .accept()
        .map_err(|e| NeoError::Pipeline(format!("accept: {e}")))?;
    tcp.set_nodelay(true).ok();
    info!(peer = %peer, "receiver connected — streaming desktop");

    // 7. Capture + encode + send loop, paced on an absolute timeline.
    let target_dt = Duration::from_secs_f64(1.0 / cli.fps);
    let mut frame_num = 0u32;
    let mut slot = 0usize;
    let mut pending: VecDeque<usize> = VecDeque::new();

    let timeline_start = Instant::now();
    let mut present_index: u64 = 0;

    let mut last_log = Instant::now();
    let mut frames_sent = 0u32;
    let mut bytes_sent = 0u64;
    let mut capture_us_total = 0u64;
    let mut encode_us_total = 0u64;
    let mut frames_captured_window = 0u32;
    let mut idle_frames_window = 0u32;

    let mut consecutive_recreates = 0u32;
    loop {
        // -- Capture -------------------------------------------------------
        let cap_start = Instant::now();
        let captured = match cap.capture_into(dst_dptr, dst_pitch, 16) {
            Ok(true) => {
                consecutive_recreates = 0;
                true
            }
            Ok(false) => false, // no new frame within timeout; we still send last
            Err(e) => {
                warn!("capture: {e} — rebuilding duplication");
                // Back off on repeated failures (UAC prompt, fullscreen game,
                // HDR mode toggle…). Each failure waits a bit longer so we
                // don't pin a CPU core spinning on AcquireNextFrame.
                let backoff_ms = (50u64 << consecutive_recreates.min(5)).min(2000);
                std::thread::sleep(Duration::from_millis(backoff_ms));
                consecutive_recreates = consecutive_recreates.saturating_add(1);
                match ZeroCopyCapture::new(cli.monitor) {
                    Ok(c) => {
                        cap = c;
                        if cap.enc_width != enc_width || cap.enc_height != enc_height {
                            return Err(NeoError::Pipeline(format!(
                                "monitor resolution changed: {}x{} → {}x{} \
                                 (NVENC session is fixed-size — restart \
                                 neo-desktop-send to pick up the new mode)",
                                enc_width, enc_height, cap.enc_width, cap.enc_height
                            )));
                        }
                    }
                    Err(e) => {
                        warn!("recapture failed: {e} (retry #{consecutive_recreates})");
                        if consecutive_recreates > 20 {
                            return Err(NeoError::Pipeline(format!(
                                "screen capture unrecoverable after 20 retries: {e}"
                            )));
                        }
                    }
                }
                continue;
            }
        };
        capture_us_total += cap_start.elapsed().as_micros() as u64;
        if captured {
            frames_captured_window += 1;
        } else {
            idle_frames_window += 1;
        }

        // -- Encode --------------------------------------------------------
        let enc_start = Instant::now();
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
        encode_us_total += enc_start.elapsed().as_micros() as u64;
        slot = (slot + 1) % POOL;

        // -- Pace on absolute timeline (no drift) --------------------------
        present_index += 1;
        let next_present = timeline_start + target_dt * present_index as u32;
        let now = Instant::now();
        if next_present > now {
            std::thread::sleep(next_present - now);
        }

        // -- Log every second ---------------------------------------------
        let log_dt = last_log.elapsed();
        if log_dt.as_secs_f64() >= 1.0 {
            let fps = frames_sent as f64 / log_dt.as_secs_f64();
            let mbps = (bytes_sent as f64 * 8.0) / (log_dt.as_secs_f64() * 1_000_000.0);
            let denom = (frames_captured_window + idle_frames_window).max(1) as f64;
            let avg_cap = capture_us_total as f64 / denom / 1000.0;
            let avg_enc = encode_us_total as f64 / denom / 1000.0;
            info!(
                fps = format!("{fps:.1}"),
                mbps = format!("{mbps:.2}"),
                cap_ms = format!("{avg_cap:.2}"),
                enc_ms = format!("{avg_enc:.2}"),
                fresh = frames_captured_window,
                idle = idle_frames_window,
                frame = frame_num,
                "desktop"
            );
            frames_sent = 0;
            bytes_sent = 0;
            capture_us_total = 0;
            encode_us_total = 0;
            frames_captured_window = 0;
            idle_frames_window = 0;
            last_log = Instant::now();
        }
    }
}

fn alloc_pitched_bgra(width: u32, height: u32) -> NeoResult<(CUdeviceptr, usize)> {
    let mut dptr: CUdeviceptr = 0;
    let mut pitch: usize = 0;
    let r = unsafe {
        cu::cuMemAllocPitch_v2(
            &mut dptr,
            &mut pitch,
            (width as usize) * 4,
            height as usize,
            16, // element size in bytes — 16 keeps NVENC happy
        )
    };
    if r != CUresult::CUDA_SUCCESS {
        return Err(NeoError::Cuda(format!("cuMemAllocPitch_v2: {r:?}")));
    }
    if pitch % 4 != 0 {
        return Err(NeoError::Cuda(format!(
            "pitch {pitch} is not a multiple of 4 — NVENC requires multiples of 4"
        )));
    }
    Ok((dptr, pitch))
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
        if let Err(e) = write_header(tcp, *frame_num, data.len() as u32) {
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
