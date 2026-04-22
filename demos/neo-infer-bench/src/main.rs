//! # neo-infer-bench
//!
//! One binary, two jobs:
//!
//! - `bench`: synthetic zero-copy VRAM benchmark for the ort CUDA EP path.
//! - `video`: real video processing harness for ONNX models, including
//!   RIFE-style multi-input interpolation.
//!
//! The bench path stays true zero-copy on input/output buffers. The video
//! path is intentionally pragmatic: decode and encode go through FFmpeg's
//! rawvideo pipes so we can validate real models on real clips today,
//! while inference itself still runs on ONNX Runtime's CUDA Execution
//! Provider. That keeps the "does this model work on video?" loop tight
//! without waiting on the full VRAM-only pre/post chain.

use clap::{Parser, Subcommand};
use cudarc::driver::sys::{self as cuda_sys, CUresult};
use neo_hwaccel::CudaRuntime;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, Command, Stdio};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "neo-infer-bench",
    about = "Zero-copy ONNX bench + practical video runner for CUDA-backed models"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CommandKind>,

    /// Legacy shortcut: if no subcommand is given, these flags run the
    /// synthetic bench mode.
    #[arg(long, global = true)]
    model: Option<PathBuf>,
    #[arg(long, global = true, default_value_t = 100)]
    iters: usize,
    #[arg(long, global = true, default_value_t = 10)]
    warmup: usize,
    #[arg(long, global = true, default_value_t = 0x3F800000)]
    fill_pattern: u32,
    #[arg(long, global = true, default_value_t = 0)]
    device: usize,
}

#[derive(Subcommand, Debug)]
enum CommandKind {
    /// Synthetic interop-buffer bench: CUDA fill -> ORT CUDA EP -> wgpu readback.
    Bench(BenchArgs),
    /// Decode a real video, run an ONNX model on each frame or frame pair, and re-encode it.
    Video(VideoArgs),
}

#[derive(Parser, Debug, Clone)]
struct BenchArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 100)]
    iters: usize,
    #[arg(long, default_value_t = 10)]
    warmup: usize,
    #[arg(long, default_value_t = 0x3F800000)]
    fill_pattern: u32,
    #[arg(long, default_value_t = 0)]
    device: usize,
}

#[derive(Parser, Debug, Clone)]
struct VideoArgs {
    /// ONNX model file.
    #[arg(long)]
    model: PathBuf,

    /// Input video file.
    #[arg(long)]
    input: PathBuf,

    /// Output video file.
    #[arg(long)]
    output: PathBuf,

    /// Optional explicit ffmpeg executable path.
    #[arg(long)]
    ffmpeg: Option<PathBuf>,

    /// Maximum number of decoded frames to process.
    #[arg(long)]
    max_frames: Option<usize>,

    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Value used for extra scalar-ish RIFE inputs (usually 0.5 = halfway frame).
    #[arg(long, default_value_t = 0.5)]
    timestep: f32,

    /// Encoder passed to ffmpeg (`libx264`, `h264_nvenc`, ...).
    #[arg(long, default_value = "libx264")]
    encoder: String,

    /// Override output fps. If omitted, keeps the source fps for 1-input
    /// models and doubles it for RIFE-style multi-input models.
    #[arg(long)]
    fps: Option<f64>,

    /// Force the decode width. Useful for dynamic-shape models like RIFE.
    #[arg(long)]
    width: Option<usize>,

    /// Force the decode height. Useful for dynamic-shape models like RIFE.
    #[arg(long)]
    height: Option<usize>,

    /// Override ffmpeg preset.
    #[arg(long)]
    preset: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorLayout {
    Nchw { height: usize, width: usize, channels: usize },
    Nhwc { height: usize, width: usize, channels: usize },
}

impl TensorLayout {
    fn from_shape(shape: &[i64]) -> Result<Self, String> {
        if shape.len() != 4 {
            return Err(format!(
                "expected a 4D tensor shape, got {:?}; only [1,C,H,W] and [1,H,W,C] are supported",
                shape
            ));
        }
        if shape[0] != 1 {
            return Err(format!("batch size must be 1, got {:?}", shape));
        }
        let dims = [
            shape[0] as usize,
            shape[1] as usize,
            shape[2] as usize,
            shape[3] as usize,
        ];
        if dims[1] == 3 {
            Ok(Self::Nchw {
                channels: dims[1],
                height: dims[2],
                width: dims[3],
            })
        } else if dims[3] == 3 {
            Ok(Self::Nhwc {
                height: dims[1],
                width: dims[2],
                channels: dims[3],
            })
        } else {
            Err(format!(
                "expected RGB tensor with 3 channels, got {:?}; only RGB models are supported here",
                shape
            ))
        }
    }

    fn width(self) -> usize {
        match self {
            Self::Nchw { width, .. } | Self::Nhwc { width, .. } => width,
        }
    }

    fn height(self) -> usize {
        match self {
            Self::Nchw { height, .. } | Self::Nhwc { height, .. } => height,
        }
    }

    fn len(self) -> usize {
        match self {
            Self::Nchw {
                height,
                width,
                channels,
            }
            | Self::Nhwc {
                height,
                width,
                channels,
            } => height * width * channels,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoMode {
    SingleInput,
    RifeLike,
}

#[derive(Debug, Clone)]
struct VideoProbe {
    width: usize,
    height: usize,
    fps: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("build with --features cuda to run this demo");
        std::process::exit(2);
    }

    #[cfg(feature = "cuda")]
    {
        match cli.command {
            Some(CommandKind::Bench(args)) => run_bench(args),
            Some(CommandKind::Video(args)) => run_video(args),
            None => {
                let Some(model) = cli.model else {
                    eprintln!("missing --model; use `neo-infer-bench bench --model ...` or `neo-infer-bench video --model ...`");
                    std::process::exit(2);
                };
                run_bench(BenchArgs {
                    model,
                    iters: cli.iters,
                    warmup: cli.warmup,
                    fill_pattern: cli.fill_pattern,
                    device: cli.device,
                })
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn run_bench(args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    use neo_gpu::{GpuContext, GpuOptions};
    use neo_hwaccel::interop;
    use neo_infer_ort::OnnxModelCuda;

    tracing::info!(
        model = %args.model.display(),
        iters = args.iters,
        warmup = args.warmup,
        "starting synthetic bench"
    );

    let gpu = std::sync::Arc::new(GpuContext::new_sync(&GpuOptions::interop())?);
    let cuda = CudaRuntime::new(args.device)?;
    cuda.ctx
        .bind_to_thread()
        .map_err(|e| format!("bind_to_thread: {e:?}"))?;

    let mut model = OnnxModelCuda::load(&args.model, cuda.ctx.clone(), args.device as i32)?;
    let n_inputs = model.input_count();
    let out_bytes = model.output_byte_size() as u64;

    for i in 0..n_inputs {
        tracing::info!(
            idx = i,
            name = %model.input_name(i),
            shape = ?model.input_shape(i),
            bytes = model.input_byte_size(i),
            "input"
        );
    }
    tracing::info!(
        name = %model.output_name(),
        shape = ?model.output_shape(),
        bytes = out_bytes,
        "output"
    );

    if !model.is_fully_static() {
        return Err(format!(
            "bench mode only supports fully static ONNX shapes. \
model {} has dynamic input/output dims ({:?} -> {:?}). \
Use `video` mode for RIFE-style models, e.g.:\n  .\\demos\\neo-infer-bench.exe video --model {} --input .\\test_input_4k.mp4 --output .\\demos\\neo-infer-bench\\out_rife.mp4 --max-frames 30 --width 640 --height 360",
            args.model.display(),
            model.input_shape(0),
            model.output_shape(),
            args.model.display(),
        ).into());
    }

    let mut in_bufs = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let b = interop::create_interop_buffer(
            gpu.clone(),
            &cuda,
            model.input_byte_size(i) as u64,
        )?;
        in_bufs.push(b);
    }
    let out_buf = interop::create_interop_buffer(gpu.clone(), &cuda, out_bytes)?;

    let mut total_in_bytes: u64 = 0;
    for (i, buf) in in_bufs.iter().enumerate() {
        let bytes = model.input_byte_size(i) as u64;
        total_in_bytes += bytes;
        let n_words = (bytes / 4) as usize;
        unsafe {
            cuda_check(
                cuda_sys::cuMemsetD32_v2(buf.cu_device_ptr, args.fill_pattern, n_words),
                "cuMemsetD32 (input)",
            )?;
        }
    }
    unsafe {
        cuda_check(cuda_sys::cuCtxSynchronize(), "cuCtxSynchronize (fill)")?;
    }

    let input_dptrs: Vec<_> = in_bufs.iter().map(|b| b.cu_device_ptr).collect();

    for _ in 0..args.warmup {
        model.infer_on_device(&input_dptrs, out_buf.cu_device_ptr)?;
    }

    let mut samples = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let t0 = Instant::now();
        model.infer_on_device(&input_dptrs, out_buf.cu_device_ptr)?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = percentile(&samples, 0.50);
    let p90 = percentile(&samples, 0.90);
    let p99 = percentile(&samples, 0.99);
    let mean = samples.iter().copied().sum::<f64>() / samples.len() as f64;
    let throughput_mb_s = (total_in_bytes as f64 / (mean / 1000.0)) / (1024.0 * 1024.0);

    println!("\n=== neo-infer-bench / synthetic ===");
    println!("model:            {}", args.model.display());
    println!("inputs:           {}", n_inputs);
    for i in 0..n_inputs {
        println!(
            "  [{}] {:<24} {:?}",
            i,
            model.input_name(i),
            model.input_shape(i)
        );
    }
    println!("output shape:     {:?}", model.output_shape());
    println!("iters:            {}", args.iters);
    println!("latency ms  mean: {:.3}", mean);
    println!("            p50:  {:.3}", p50);
    println!("            p90:  {:.3}", p90);
    println!("            p99:  {:.3}", p99);
    println!("throughput in:    {:.1} MB/s", throughput_mb_s);

    let output_first_words = download_first_words(&gpu, out_buf.wgpu_buffer(), out_bytes)?;
    println!(
        "output first 4 words: {:#010x} {:#010x} {:#010x} {:#010x}",
        output_first_words[0], output_first_words[1], output_first_words[2], output_first_words[3]
    );

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_video(args: VideoArgs) -> Result<(), Box<dyn std::error::Error>> {
    use neo_infer_ort::OnnxModelCuda;

    let ffmpeg = find_ffmpeg(args.ffmpeg.as_deref())?;
    let probe = probe_video(&ffmpeg, &args.input).unwrap_or(VideoProbe {
        width: 1920,
        height: 1080,
        fps: 30.0,
    });
    let cuda = CudaRuntime::new(args.device)?;
    cuda.ctx
        .bind_to_thread()
        .map_err(|e| format!("bind_to_thread: {e:?}"))?;
    let mut model = OnnxModelCuda::load(&args.model, cuda.ctx.clone(), args.device as i32)?;

    let decode_width = args.width.unwrap_or_else(|| {
        if model.input_is_dynamic(0) {
            probe.width
        } else {
            TensorLayout::from_shape(model.input_shape(0))
                .map(|l| l.width())
                .unwrap_or(probe.width)
        }
    });
    let decode_height = args.height.unwrap_or_else(|| {
        if model.input_is_dynamic(0) {
            probe.height
        } else {
            TensorLayout::from_shape(model.input_shape(0))
                .map(|l| l.height())
                .unwrap_or(probe.height)
        }
    });
    let first_layout = resolve_layout(model.input_shape(0), decode_width, decode_height)
        .map_err(|e| format!("input[0] unsupported: {e}"))?;
    let hinted_out_layout = if model.output_is_dynamic() {
        None
    } else {
        Some(
            resolve_layout(model.output_shape(), decode_width, decode_height)
                .map_err(|e| format!("output unsupported: {e}"))?,
        )
    };
    let mode = if model.input_count() == 1 {
        VideoMode::SingleInput
    } else {
        VideoMode::RifeLike
    };
    let out_fps = args.fps.unwrap_or_else(|| {
        if mode == VideoMode::RifeLike {
            probe.fps * 2.0
        } else {
            probe.fps
        }
    });

    tracing::info!(
        mode = ?mode,
        source_fps = probe.fps,
        output_fps = out_fps,
        input_shape = ?model.input_shape(0),
        output_shape = ?model.output_shape(),
        "starting video run"
    );

    for i in 1..model.input_count() {
        tracing::info!(
            idx = i,
            name = %model.input_name(i),
            shape = ?model.input_shape(i),
            "extra input"
        );
    }

    let mut decoder = spawn_decoder(
        &ffmpeg,
        &args.input,
        decode_width,
        decode_height,
        args.max_frames,
    )?;
    let mut decoder_stdout = decoder
        .stdout
        .take()
        .ok_or("decoder stdout pipe missing")?;
    let mut decoder_stderr = decoder
        .stderr
        .take()
        .ok_or("decoder stderr pipe missing")?;

    let frame_bytes = first_layout.width() * first_layout.height() * 3;
    let mut rgb_buf = vec![0u8; frame_bytes];
    let mut prev_rgb: Option<Vec<u8>> = None;
    let mut prev_tensor: Option<Vec<f32>> = None;
    let mut encoder: Option<Child> = None;
    let mut encoder_stdin: Option<ChildStdinGuard> = None;
    let mut encoder_stderr: Option<ChildStderr> = None;
    let mut final_out_layout = hinted_out_layout;
    let mut latency_ms = Vec::new();
    let mut decoded_frames = 0usize;
    let mut inferred_frames = 0usize;
    let mut written_frames = 0usize;
    let video_start = Instant::now();

    loop {
        match read_exact_or_eof(&mut decoder_stdout, &mut rgb_buf) {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => return Err(format!("decoder read failed: {e}").into()),
        }
        decoded_frames += 1;

        let current_tensor =
            rgb_to_tensor(&rgb_buf, first_layout).map_err(|e| format!("frame {decoded_frames}: {e}"))?;
        let current_shape = runtime_shape(first_layout);

        match mode {
            VideoMode::SingleInput => {
                let t0 = Instant::now();
                let (output, out_shape) =
                    infer_frame(&mut model, &[(&current_tensor, current_shape.as_slice())])?;
                latency_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                inferred_frames += 1;

                let output_layout = resolve_output_layout(
                    &out_shape,
                    hinted_out_layout,
                    decode_width,
                    decode_height,
                )
                    .map_err(|e| format!("output frame {decoded_frames}: {e}"))?;
                ensure_encoder(
                    &mut encoder,
                    &mut encoder_stdin,
                    &mut encoder_stderr,
                    &ffmpeg,
                    &args.output,
                    output_layout.width(),
                    output_layout.height(),
                    out_fps,
                    &args.encoder,
                    args.preset.as_deref(),
                )?;
                final_out_layout = Some(output_layout);
                let output_rgb =
                    tensor_to_rgb(&output, output_layout).map_err(|e| format!("output frame {decoded_frames}: {e}"))?;
                encoder_stdin
                    .as_mut()
                    .ok_or("encoder stdin missing")?
                    .0
                    .write_all(&output_rgb)?;
                written_frames += 1;
            }
            VideoMode::RifeLike => {
                if prev_tensor.is_none() {
                    prev_rgb = Some(rgb_buf.clone());
                    prev_tensor = Some(current_tensor);
                    continue;
                }

                let mut extras = Vec::new();
                let mut extra_shapes = Vec::new();
                for i in 2..model.input_count() {
                    let layout = resolve_runtime_dims(model.input_shape(i), decode_width, decode_height)
                        .map_err(|e| format!("input[{i}] unsupported: {e}"))?;
                    let len = fixed_len_usize(&layout)
                        .ok_or_else(|| format!("input[{i}] still has dynamic dims after specialization: {:?}", layout))?;
                    extras.push(vec![args.timestep; len]);
                    extra_shapes.push(layout);
                }

                let prev_tensor_ref = prev_tensor.as_ref().unwrap();
                let prev_shape = runtime_shape(first_layout);
                let mut input_refs: Vec<(&[f32], &[i64])> = Vec::with_capacity(model.input_count());
                input_refs.push((prev_tensor_ref.as_slice(), prev_shape.as_slice()));
                input_refs.push((current_tensor.as_slice(), current_shape.as_slice()));
                for (extra, shape) in extras.iter().zip(extra_shapes.iter()) {
                    input_refs.push((extra.as_slice(), shape.as_slice()));
                }

                let t0 = Instant::now();
                let (output, out_shape) = infer_frame(&mut model, &input_refs)?;
                latency_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                inferred_frames += 1;

                let interp_layout = resolve_output_layout(
                    &out_shape,
                    hinted_out_layout,
                    decode_width,
                    decode_height,
                )
                    .map_err(|e| format!("interp frame {decoded_frames}: {e}"))?;
                ensure_encoder(
                    &mut encoder,
                    &mut encoder_stdin,
                    &mut encoder_stderr,
                    &ffmpeg,
                    &args.output,
                    interp_layout.width(),
                    interp_layout.height(),
                    out_fps,
                    &args.encoder,
                    args.preset.as_deref(),
                )?;
                final_out_layout = Some(interp_layout);
                encoder_stdin
                    .as_mut()
                    .ok_or("encoder stdin missing")?
                    .0
                    .write_all(prev_rgb.as_ref().unwrap())?;
                written_frames += 1;
                let interp_rgb =
                    tensor_to_rgb(&output, interp_layout).map_err(|e| format!("interp frame {decoded_frames}: {e}"))?;
                encoder_stdin
                    .as_mut()
                    .ok_or("encoder stdin missing")?
                    .0
                    .write_all(&interp_rgb)?;
                written_frames += 1;

                prev_rgb = Some(rgb_buf.clone());
                prev_tensor = Some(current_tensor);
            }
        }
    }

    if mode == VideoMode::RifeLike {
        if let Some(last) = prev_rgb.as_ref() {
            if let Some(stdin) = encoder_stdin.as_mut() {
                stdin.0.write_all(last)?;
            }
            written_frames += 1;
        }
    }

    drop(encoder_stdin);

    let decoder_status = decoder.wait()?;
    let decoder_err = read_stderr(&mut decoder_stderr)?;

    if !decoder_status.success() {
        return Err(format!("ffmpeg decoder failed: {decoder_err}").into());
    }
    if let Some(mut encoder) = encoder {
        let encoder_status = encoder.wait()?;
        let mut stderr = encoder_stderr.ok_or("encoder stderr missing")?;
        let encoder_err = read_stderr(&mut stderr)?;
        if !encoder_status.success() {
            return Err(format!("ffmpeg encoder failed: {encoder_err}").into());
        }
    } else {
        return Err("no output frames were produced; encoder never started".into());
    }

    latency_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if latency_ms.is_empty() {
        0.0
    } else {
        latency_ms.iter().copied().sum::<f64>() / latency_ms.len() as f64
    };
    let elapsed = video_start.elapsed().as_secs_f64();

    println!("\n=== neo-infer-bench / video ===");
    println!("model:            {}", args.model.display());
    println!("input video:      {}", args.input.display());
    println!("output video:     {}", args.output.display());
    println!("mode:             {:?}", mode);
    if let Some(layout) = final_out_layout {
        println!(
            "decode -> model:  {}x{} -> {}x{}",
            first_layout.width(),
            first_layout.height(),
            layout.width(),
            layout.height()
        );
    }
    println!("decoded frames:   {}", decoded_frames);
    println!("model invocations:{}", inferred_frames);
    println!("written frames:   {}", written_frames);
    println!("fps out:          {:.3}", out_fps);
    println!("infer ms mean:    {:.3}", mean);
    println!("infer ms p50:     {:.3}", percentile(&latency_ms, 0.50));
    println!("infer ms p90:     {:.3}", percentile(&latency_ms, 0.90));
    println!("infer ms p99:     {:.3}", percentile(&latency_ms, 0.99));
    println!(
        "wall throughput:  {:.2} frames/s",
        written_frames as f64 / elapsed.max(1e-9)
    );

    Ok(())
}

#[cfg(feature = "cuda")]
fn spawn_decoder(
    ffmpeg: &Path,
    input: &Path,
    width: usize,
    height: usize,
    max_frames: Option<usize>,
) -> Result<Child, Box<dyn std::error::Error>> {
    let mut cmd = Command::new(ffmpeg);
    cmd.arg("-v")
        .arg("error")
        .arg("-hwaccel")
        .arg("auto")
        .arg("-i")
        .arg(input)
        .arg("-vf")
        .arg(format!("scale={width}:{height},format=rgb24"))
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("rgb24");
    if let Some(n) = max_frames {
        cmd.arg("-frames:v").arg(n.to_string());
    }
    cmd.arg("-")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    Ok(cmd.spawn()?)
}

#[cfg(feature = "cuda")]
fn spawn_encoder(
    ffmpeg: &Path,
    output: &Path,
    width: usize,
    height: usize,
    fps: f64,
    encoder: &str,
    preset: Option<&str>,
) -> Result<Child, Box<dyn std::error::Error>> {
    let chosen_preset = preset.unwrap_or_else(|| {
        if encoder.contains("nvenc") {
            "p4"
        } else {
            "fast"
        }
    });
    let mut cmd = Command::new(ffmpeg);
    cmd.arg("-v")
        .arg("error")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("rgb24")
        .arg("-s")
        .arg(format!("{width}x{height}"))
        .arg("-r")
        .arg(format!("{fps:.6}"))
        .arg("-i")
        .arg("-")
        .arg("-an")
        .arg("-c:v")
        .arg(encoder)
        .arg("-preset")
        .arg(chosen_preset)
        .arg("-y")
        .arg(output)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped());
    Ok(cmd.spawn()?)
}

#[cfg(feature = "cuda")]
struct ChildStdinGuard(ChildStdin);

#[cfg(feature = "cuda")]
fn ensure_encoder(
    encoder: &mut Option<Child>,
    encoder_stdin: &mut Option<ChildStdinGuard>,
    encoder_stderr: &mut Option<ChildStderr>,
    ffmpeg: &Path,
    output: &Path,
    width: usize,
    height: usize,
    fps: f64,
    encoder_name: &str,
    preset: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if encoder.is_some() {
        return Ok(());
    }
    let mut child = spawn_encoder(ffmpeg, output, width, height, fps, encoder_name, preset)?;
    let stdin = child.stdin.take().ok_or("encoder stdin pipe missing")?;
    let stderr = child.stderr.take().ok_or("encoder stderr pipe missing")?;
    *encoder_stdin = Some(ChildStdinGuard(stdin));
    *encoder_stderr = Some(stderr);
    *encoder = Some(child);
    Ok(())
}

#[cfg(feature = "cuda")]
fn find_ffmpeg(explicit: Option<&Path>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = explicit {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        return Err(format!("ffmpeg not found at {}", path.display()).into());
    }

    let mut candidates = vec![
        PathBuf::from("ffmpeg.exe"),
        PathBuf::from("ffmpeg"),
        PathBuf::from(r"C:\Users\infinition\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"),
    ];

    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors() {
            candidates.push(ancestor.join("ffmpeg.exe"));
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors() {
            candidates.push(ancestor.join("ffmpeg.exe"));
        }
    }

    for path in candidates {
        if path.exists() {
            return Ok(path);
        }
    }

    if let Ok(output) = Command::new("where").arg("ffmpeg").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = stdout.lines().next() {
                let path = PathBuf::from(line.trim());
                if path.exists() {
                    return Ok(path);
                }
            }
        }
    }

    Err("ffmpeg not found; pass --ffmpeg C:\\path\\to\\ffmpeg.exe".into())
}

#[cfg(feature = "cuda")]
fn probe_video(ffmpeg: &Path, input: &Path) -> Result<VideoProbe, Box<dyn std::error::Error>> {
    let ffprobe = ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
    if ffprobe.exists() {
        let output = Command::new(ffprobe)
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "csv=p=0:s=,",
            ])
            .arg(input)
            .output()?;
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = text.trim().split(',').collect();
            if parts.len() >= 3 {
                let width = parts[0].trim().parse().ok();
                let height = parts[1].trim().parse().ok();
                let fps = parse_ffprobe_rate(parts[2].trim());
                if let (Some(width), Some(height), Some(fps)) = (width, height, fps) {
                    return Ok(VideoProbe { width, height, fps });
                }
            }
        }
    }

    let output = Command::new(ffmpeg)
        .arg("-hide_banner")
        .arg("-i")
        .arg(input)
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if let Some(probe) = parse_ffmpeg_banner(&stderr) {
        return Ok(probe);
    }

    tracing::warn!("ffprobe not found; defaulting source size to 1920x1080 and fps to 30.0");
    Ok(VideoProbe {
        width: 1920,
        height: 1080,
        fps: 30.0,
    })
}

#[cfg(feature = "cuda")]
fn parse_ffprobe_rate(text: &str) -> Option<f64> {
    if let Some((n, d)) = text.split_once('/') {
        let num: f64 = n.parse().ok()?;
        let den: f64 = d.parse().ok()?;
        if den > 0.0 {
            return Some(num / den);
        }
    }
    text.parse().ok()
}

#[cfg(feature = "cuda")]
fn parse_ffmpeg_banner(text: &str) -> Option<VideoProbe> {
    for line in text.lines() {
        if !line.contains("Video:") {
            continue;
        }

        let mut width = None;
        let mut height = None;
        for token in line.split(|c: char| c.is_whitespace() || c == ',') {
            if let Some((w, h)) = token.split_once('x') {
                let w = w.parse::<usize>().ok();
                let h = h.parse::<usize>().ok();
                if w.is_some() && h.is_some() {
                    width = w;
                    height = h;
                    break;
                }
            }
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        let fps = parts
            .windows(2)
            .find_map(|w| if w[1] == "fps" { w[0].parse::<f64>().ok() } else { None });

        if let (Some(width), Some(height), Some(fps)) = (width, height, fps) {
            return Some(VideoProbe { width, height, fps });
        }
    }
    None
}

#[cfg(feature = "cuda")]
fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0usize;
    while filled < buf.len() {
        let n = reader.read(&mut buf[filled..])?;
        if n == 0 {
            if filled == 0 {
                return Ok(false);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "partial frame at end of stream",
            ));
        }
        filled += n;
    }
    Ok(true)
}

#[cfg(feature = "cuda")]
fn read_stderr(stderr: &mut ChildStderr) -> io::Result<String> {
    let mut out = String::new();
    stderr.read_to_string(&mut out)?;
    Ok(out.trim().to_string())
}

#[cfg(feature = "cuda")]
fn rgb_to_tensor(rgb: &[u8], layout: TensorLayout) -> Result<Vec<f32>, String> {
    let expected = layout.width() * layout.height() * 3;
    if rgb.len() != expected {
        return Err(format!("rgb input length {} != expected {}", rgb.len(), expected));
    }

    let mut out = vec![0.0f32; layout.len()];
    match layout {
        TensorLayout::Nchw { height, width, .. } => {
            let plane = height * width;
            for y in 0..height {
                for x in 0..width {
                    let src = (y * width + x) * 3;
                    let dst = y * width + x;
                    out[dst] = rgb[src] as f32 / 255.0;
                    out[plane + dst] = rgb[src + 1] as f32 / 255.0;
                    out[plane * 2 + dst] = rgb[src + 2] as f32 / 255.0;
                }
            }
        }
        TensorLayout::Nhwc { .. } => {
            for (dst, src) in out.iter_mut().zip(rgb.iter()) {
                *dst = *src as f32 / 255.0;
            }
        }
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn tensor_to_rgb(tensor: &[f32], layout: TensorLayout) -> Result<Vec<u8>, String> {
    if tensor.len() != layout.len() {
        return Err(format!(
            "tensor length {} != expected {} for {:?}",
            tensor.len(),
            layout.len(),
            layout
        ));
    }

    let mut out = vec![0u8; layout.width() * layout.height() * 3];
    match layout {
        TensorLayout::Nchw { height, width, .. } => {
            let plane = height * width;
            for y in 0..height {
                for x in 0..width {
                    let src = y * width + x;
                    let dst = (y * width + x) * 3;
                    out[dst] = f32_to_u8(tensor[src]);
                    out[dst + 1] = f32_to_u8(tensor[plane + src]);
                    out[dst + 2] = f32_to_u8(tensor[plane * 2 + src]);
                }
            }
        }
        TensorLayout::Nhwc { .. } => {
            for (dst, src) in out.iter_mut().zip(tensor.iter()) {
                *dst = f32_to_u8(*src);
            }
        }
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn f32_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(feature = "cuda")]
fn infer_frame(
    model: &mut neo_infer_ort::OnnxModelCuda,
    inputs: &[(&[f32], &[i64])],
) -> Result<(Vec<f32>, Vec<i64>), Box<dyn std::error::Error>> {
    if inputs
        .iter()
        .enumerate()
        .any(|(i, _)| model.input_is_dynamic(i))
        || model.output_is_dynamic()
    {
        Ok(model.infer_dynamic(inputs)?)
    } else {
        let refs: Vec<&[f32]> = inputs.iter().map(|(data, _)| *data).collect();
        Ok((model.infer(&refs)?, model.output_shape().to_vec()))
    }
}

#[cfg(feature = "cuda")]
fn resolve_output_layout(
    runtime_shape: &[i64],
    hinted: Option<TensorLayout>,
    fallback_width: usize,
    fallback_height: usize,
) -> Result<TensorLayout, String> {
    if let Ok(layout) = TensorLayout::from_shape(runtime_shape) {
        return Ok(layout);
    }
    if let Some(layout) = hinted {
        return Ok(layout);
    }
    resolve_layout(runtime_shape, fallback_width, fallback_height)
}

#[cfg(feature = "cuda")]
fn runtime_shape(layout: TensorLayout) -> Vec<i64> {
    match layout {
        TensorLayout::Nchw {
            height,
            width,
            channels,
        } => vec![1, channels as i64, height as i64, width as i64],
        TensorLayout::Nhwc {
            height,
            width,
            channels,
        } => vec![1, height as i64, width as i64, channels as i64],
    }
}

#[cfg(feature = "cuda")]
fn resolve_layout(shape: &[i64], fallback_width: usize, fallback_height: usize) -> Result<TensorLayout, String> {
    let resolved = resolve_runtime_dims(shape, fallback_width, fallback_height)?;
    TensorLayout::from_shape(&resolved)
}

#[cfg(feature = "cuda")]
fn resolve_runtime_dims(shape: &[i64], fallback_width: usize, fallback_height: usize) -> Result<Vec<i64>, String> {
    if shape.is_empty() {
        return Err("empty tensor shape".into());
    }
    let mut resolved = shape.to_vec();
    if resolved.len() == 4 {
        if resolved[0] == -1 {
            resolved[0] = 1;
        }
        if resolved[1] == 3 {
            if resolved[2] == -1 {
                resolved[2] = fallback_height as i64;
            }
            if resolved[3] == -1 {
                resolved[3] = fallback_width as i64;
            }
        } else if resolved[3] == 3 {
            if resolved[1] == -1 {
                resolved[1] = fallback_height as i64;
            }
            if resolved[2] == -1 {
                resolved[2] = fallback_width as i64;
            }
        } else {
            for dim in &mut resolved {
                if *dim == -1 {
                    *dim = 1;
                }
            }
        }
    } else {
        for dim in &mut resolved {
            if *dim == -1 {
                *dim = 1;
            }
        }
    }
    Ok(resolved)
}

#[cfg(feature = "cuda")]
fn fixed_len_usize(shape: &[i64]) -> Option<usize> {
    if shape.iter().any(|&d| d <= 0) {
        return None;
    }
    Some(shape.iter().map(|&d| d as usize).product())
}

#[cfg(feature = "cuda")]
fn cuda_check(r: CUresult, ctx: &'static str) -> Result<(), Box<dyn std::error::Error>> {
    if r == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{ctx}: {r:?}").into())
    }
}

#[cfg(feature = "cuda")]
fn download_first_words(
    gpu: &neo_gpu::GpuContext,
    buf: &wgpu::Buffer,
    byte_size: u64,
) -> Result<[u32; 4], Box<dyn std::error::Error>> {
    let read_bytes = 16u64.min(byte_size);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench-staging"),
        size: read_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bench-download"),
        });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, read_bytes);
    gpu.queue.submit(std::iter::once(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    rx.recv()??;
    let data = slice.get_mapped_range();
    let mut out = [0u32; 4];
    for (i, chunk) in data.chunks_exact(4).take(4).enumerate() {
        out[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    drop(data);
    staging.unmap();
    Ok(out)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[allow(dead_code)]
fn _touch(path: &Path) -> io::Result<()> {
    let _ = fs::metadata(path)?;
    Ok(())
}
