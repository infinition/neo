//! neo-rife-zc — Zerocopy RIFE interpolation streamer.
//!
//! NVDEC decode → DtoD → interop Y/UV → wgpu NV12→BGRA → pack to tensor
//! → ONNX (RIFE) → unpack to BGRA → NVENC encode → TCP send.

#[path = "../../neo-stream/src/protocol.rs"]
mod protocol;
#[path = "../../neo-stream/src/zerocopy_stream.rs"]
mod zerocopy_stream;

use clap::Parser;
use cudarc::driver::sys::{self as cuda_sys, CUresult};
use neo_core::{NeoError, NeoResult};
use neo_gpu::{BgraTensorBridge, GpuContext, GpuOptions};
use neo_hwaccel::{
    interop::create_interop_buffer, CudaRuntime, Nv12ToBgraConverter, NvdecCodec,
};
use neo_infer_ort::OnnxModelCuda;
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

#[derive(Parser)]
#[command(name = "neo-rife-zc", about = "Zerocopy RIFE streamer")]
struct Cli {
    /// Path to ONNX model (RIFE).
    #[arg(short, long)]
    model: PathBuf,

    /// Path to H.264 Annex-B bitstream.
    #[arg(short, long)]
    input: PathBuf,

    /// Listen address (IP:port).
    #[arg(short, long, default_value = "0.0.0.0:9000")]
    addr: String,

    /// Source FPS (output will be 2x this).
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
    // 1. CUDA & ONNX Model
    let runtime = Arc::new(CudaRuntime::new(0)?);
    runtime.ctx.bind_to_thread().map_err(|e| NeoError::Cuda(format!("{e:?}")))?;
    
    let mut model = OnnxModelCuda::load(&cli.model, runtime.ctx.clone(), 0)
        .map_err(|e| NeoError::Pipeline(format!("model load: {e}")))?;
    info!(model = %cli.model.display(), "ONNX model loaded");

    // 2. Load bitstream + probe.
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

    // 3. wgpu with interop features.
    let gpu = Arc::new(
        GpuContext::new_sync(&GpuOptions::interop())
            .map_err(|e| NeoError::HwAccelUnavailable(format!("wgpu interop: {e}")))?,
    );

    // 4. Derive runtime input shapes from the model.
    //
    // RIFE-v4 exports typically declare 3 inputs: (img0, img1, timestep).
    // The timestep tensor's rank varies between exports: sometimes it is
    // a per-pixel mask `[1, 1, H, W]`, sometimes a scalar-ish
    // `[1, 1, 1, 1]` or `[1]`. Any dim left dynamic (-1) is filled from
    // the decoded video dimensions.
    if model.input_count() != 3 {
        return Err(NeoError::Pipeline(format!(
            "neo-rife-zc expects a 3-input RIFE-style model (img0, img1, timestep); got {} inputs",
            model.input_count()
        )));
    }
    let runtime_shapes: Vec<Vec<i64>> = (0..model.input_count())
        .map(|i| resolve_shape(model.input_shape(i), width as i64, height as i64))
        .collect();
    for i in 0..model.input_count() {
        info!(
            idx = i,
            name = %model.input_name(i),
            model_shape = ?model.input_shape(i),
            runtime_shape = ?runtime_shapes[i],
            "input"
        );
    }
    info!(
        name = %model.output_name(),
        shape = ?model.output_shape(),
        "output"
    );

    let img_numel: i64 = (width as i64) * (height as i64) * 3;
    for (i, s) in runtime_shapes.iter().enumerate().take(2) {
        let len: i64 = s.iter().product();
        if len != img_numel {
            return Err(NeoError::Pipeline(format!(
                "input[{i}] runtime shape {:?} has {} elements, expected {} (W*H*3) for an RGB image input",
                s, len, img_numel
            )));
        }
    }

    // 5. Interop buffers.
    let y_size = (width as u64) * (height as u64);
    let uv_size = y_size / 2;
    let bgra_size = y_size * 4;
    let tensor_size = y_size * 3 * 4; // 3 channels, f32

    let interop_y = create_interop_buffer(gpu.clone(), &runtime, y_size)?;
    let interop_uv = create_interop_buffer(gpu.clone(), &runtime, uv_size)?;

    let bgra_current = create_interop_buffer(gpu.clone(), &runtime, bgra_size)?;
    let bgra_interp = create_interop_buffer(gpu.clone(), &runtime, bgra_size)?;

    let tensor_prev = create_interop_buffer(gpu.clone(), &runtime, tensor_size)?;
    let tensor_current = create_interop_buffer(gpu.clone(), &runtime, tensor_size)?;
    let tensor_interp = create_interop_buffer(gpu.clone(), &runtime, tensor_size)?;

    // Timestep tensor, sized from the model's declared 3rd input shape,
    // filled with 0.5f32 (0x3f000000 bits).
    let timestep_len: i64 = runtime_shapes[2].iter().product();
    if timestep_len <= 0 {
        return Err(NeoError::Pipeline(format!(
            "timestep runtime shape {:?} has non-positive element count",
            runtime_shapes[2]
        )));
    }
    let timestep_bytes = (timestep_len as u64) * 4;
    let timestep_buf = create_interop_buffer(gpu.clone(), &runtime, timestep_bytes)?;
    unsafe {
        let r = cuda_sys::cuMemsetD32_v2(
            timestep_buf.cu_device_ptr,
            0x3f000000,
            timestep_len as usize,
        );
        if r != CUresult::CUDA_SUCCESS {
            return Err(NeoError::Cuda(format!("cuMemsetD32: {r:?}")));
        }
    }
    info!(
        shape = ?runtime_shapes[2],
        bytes = timestep_bytes,
        "timestep buffer filled with 0.5"
    );

    // 5. Converters & Bridges.
    let converter = Nv12ToBgraConverter::from_external_buffers(
        gpu.clone(),
        width,
        height,
        interop_y.wgpu_buffer(),
        interop_uv.wgpu_buffer(),
        bgra_current.wgpu_buffer(),
    )?;

    let bridge = BgraTensorBridge::new(gpu.clone(), width, height)?;

    // 6. Zerocopy stream: NVDEC → DtoD → interop Y/UV.
    let mut stream = ZerocopyStream::new(
        runtime.clone(),
        bytes,
        interop_y.cu_device_ptr,
        interop_uv.cu_device_ptr,
    )?;

    // 7. NVENC.
    let encoder = Encoder::initialize_with_cuda(runtime.ctx.clone())
        .map_err(|e| NeoError::HwAccelUnavailable(format!("NVENC: {e:?}")))?;
    let out_fps = (cli.fps * 2.0).round() as u32;
    let mut init = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, width, height);
    init.enable_picture_type_decision().framerate(out_fps, 1);
    let session = encoder
        .start_session(NV_ENC_BUFFER_FORMAT_ARGB, init)
        .map_err(|e| NeoError::Encode(format!("session: {e:?}")))?;

    const POOL: usize = 8;
    let mut nvenc_inputs_current = Vec::with_capacity(POOL);
    let mut nvenc_inputs_interp = Vec::with_capacity(POOL);
    for _ in 0..POOL {
        nvenc_inputs_current.push(
            session.register_generic_resource::<()>(
                (),
                NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                bgra_current.cu_device_ptr as *mut c_void,
                width * 4,
            ).map_err(|e| NeoError::Encode(format!("register: {e:?}")))?
        );
        nvenc_inputs_interp.push(
            session.register_generic_resource::<()>(
                (),
                NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                bgra_interp.cu_device_ptr as *mut c_void,
                width * 4,
            ).map_err(|e| NeoError::Encode(format!("register: {e:?}")))?
        );
    }
    let mut bitstreams = Vec::with_capacity(POOL);
    for _ in 0..POOL {
        bitstreams.push(
            session.create_output_bitstream().map_err(|e| NeoError::Encode(format!("{e:?}")))?
        );
    }

    info!("zerocopy encode pipeline ready");

    // 8. Wait for receiver.
    info!(addr = %cli.addr, "waiting for receiver...");
    let listener = TcpListener::bind(&cli.addr).map_err(|e| NeoError::Pipeline(format!("bind: {e}")))?;
    let (mut tcp, peer) = listener.accept().map_err(|e| NeoError::Pipeline(format!("accept: {e}")))?;
    tcp.set_nodelay(true).ok();
    info!(peer = %peer, "receiver connected — streaming RIFE (zerocopy)");

    // 9. Stream loop.
    let target_dt = Duration::from_secs_f64(1.0 / (cli.fps * 2.0));
    let mut frame_num = 0u32;
    let mut slot = 0usize;
    let mut pending: VecDeque<usize> = VecDeque::new();
    let mut last_log = Instant::now();
    let mut frames_sent = 0u32;
    let mut bytes_sent = 0u64;
    let timeline_start = Instant::now();
    let mut present_index = 0u64;
    let mut has_prev = false;

    let shape_img0: Vec<i64> = runtime_shapes[0].clone();
    let shape_img1: Vec<i64> = runtime_shapes[1].clone();
    let shape_ts: Vec<i64> = runtime_shapes[2].clone();

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

        // NV12→BGRA via interop.
        converter.dispatch_interop()?;

        // BGRA→Tensor.
        bridge.pack_into(bgra_current.wgpu_buffer(), tensor_current.wgpu_buffer())?;

        if !has_prev {
            unsafe { cuda_sys::cuMemcpyDtoD_v2(tensor_prev.cu_device_ptr, tensor_current.cu_device_ptr, tensor_size as usize) };
            has_prev = true;

            // Encode F0
            if pending.contains(&slot) {
                drain_and_send(&mut pending, &mut bitstreams, &mut tcp, &mut frame_num, &mut frames_sent, &mut bytes_sent)?;
            }
            let res = session.encode_picture(&mut nvenc_inputs_current[slot], &mut bitstreams[slot], Default::default());
            pending.push_back(slot);
            handle_encode_result(res, &mut pending, &mut bitstreams, &mut tcp, &mut frame_num, &mut frames_sent, &mut bytes_sent)?;
            slot = (slot + 1) % POOL;

            present_index += 1;
            let next_present = timeline_start + target_dt * present_index as u32;
            let now = Instant::now();
            if next_present > now { std::thread::sleep(next_present - now); }
            continue;
        }

        // RIFE Interpolation
        model.infer_on_device_dynamic(
            &[tensor_prev.cu_device_ptr, tensor_current.cu_device_ptr, timestep_buf.cu_device_ptr],
            &[shape_img0.as_slice(), shape_img1.as_slice(), shape_ts.as_slice()],
            tensor_interp.cu_device_ptr,
            (tensor_size / 4) as usize
        ).map_err(|e| NeoError::Pipeline(format!("infer: {e}")))?;

        // Tensor→BGRA
        bridge.unpack_into(tensor_interp.wgpu_buffer(), bgra_interp.wgpu_buffer())?;

        // Encode Interpolated Frame
        if pending.contains(&slot) { drain_and_send(&mut pending, &mut bitstreams, &mut tcp, &mut frame_num, &mut frames_sent, &mut bytes_sent)?; }
        let res = session.encode_picture(&mut nvenc_inputs_interp[slot], &mut bitstreams[slot], Default::default());
        pending.push_back(slot);
        handle_encode_result(res, &mut pending, &mut bitstreams, &mut tcp, &mut frame_num, &mut frames_sent, &mut bytes_sent)?;
        slot = (slot + 1) % POOL;

        present_index += 1;
        let next_present = timeline_start + target_dt * present_index as u32;
        let now = Instant::now();
        if next_present > now { std::thread::sleep(next_present - now); }

        // Encode Current Frame
        if pending.contains(&slot) { drain_and_send(&mut pending, &mut bitstreams, &mut tcp, &mut frame_num, &mut frames_sent, &mut bytes_sent)?; }
        let res = session.encode_picture(&mut nvenc_inputs_current[slot], &mut bitstreams[slot], Default::default());
        pending.push_back(slot);
        handle_encode_result(res, &mut pending, &mut bitstreams, &mut tcp, &mut frame_num, &mut frames_sent, &mut bytes_sent)?;
        slot = (slot + 1) % POOL;

        present_index += 1;
        let next_present = timeline_start + target_dt * present_index as u32;
        let now = Instant::now();
        if next_present > now { std::thread::sleep(next_present - now); }

        // prev = current
        unsafe { cuda_sys::cuMemcpyDtoD_v2(tensor_prev.cu_device_ptr, tensor_current.cu_device_ptr, tensor_size as usize) };

        // Log.
        let log_dt = last_log.elapsed();
        if log_dt.as_secs_f64() >= 1.0 {
            let fps = frames_sent as f64 / log_dt.as_secs_f64();
            let mbps = (bytes_sent as f64 * 8.0) / (log_dt.as_secs_f64() * 1_000_000.0);
            info!(fps = format!("{fps:.1}"), mbps = format!("{mbps:.2}"), frame = frame_num, "sending");
            frames_sent = 0;
            bytes_sent = 0;
            last_log = Instant::now();
        }
    }
}

/// Resolve a model's declared input shape into a concrete runtime shape.
///
/// Any dynamic dim (`-1`) is filled with a convention:
/// - 4D tensor: `(N=1, C=3, H, W)` (C defaults to 3 only when the model
///   left it dynamic — if the model fixes C to 1 we keep 1)
/// - 3D tensor: `(C=3, H, W)`
/// - 2D tensor: `(H, W)`
/// - 1D / 0D: dynamic dims are replaced with `1`
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
                (2, 0) => height,
                (2, 1) => width,
                _ => 1,
            }
        })
        .collect()
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
