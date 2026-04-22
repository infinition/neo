use clap::{Parser, Subcommand};
use neo_gpu::context::{GpuContext, GpuOptions};
use neo_pipeline::PipelineExecutor;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "neo-ffmpeg",
    version,
    about = "Neo-FFmpeg — Zero-copy VRAM video processing for the AI era",
    long_about = r#"
  ███╗   ██╗███████╗ ██████╗       ███████╗███████╗
  ████╗  ██║██╔════╝██╔═══██╗      ██╔════╝██╔════╝
  ██╔██╗ ██║█████╗  ██║   ██║█████╗█████╗  █████╗
  ██║╚██╗██║██╔══╝  ██║   ██║╚════╝██╔══╝  ██╔══╝
  ██║ ╚████║███████╗╚██████╔╝      ██║     ██║
  ╚═╝  ╚═══╝╚══════╝ ╚═════╝       ╚═╝     ╚═╝

  Zero-copy VRAM video processing for the AI era
  NVMe -> GPU -> NVMe -- CPU never touches your frames
"#
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Process an image/video through the GPU pipeline
    Process {
        /// Input file (PNG, JPEG, BMP, TIFF, WebP)
        #[arg(short, long)]
        input: String,

        /// Output file
        #[arg(short, long)]
        output: String,

        /// GPU filters to apply (comma-separated)
        ///
        /// Available filters:
        ///   grayscale        - BT.709 luminance conversion
        ///   invert           - Invert colors
        ///   sepia            - Sepia tone (sepia:0.5 for partial)
        ///   blur             - Gaussian blur 3x3
        ///   sharpen          - Unsharp mask (sharpen:2.0 for strength)
        ///   edge-detect      - Sobel edge detection
        ///   brightness:0.1   - Adjust brightness (-1.0 to 1.0)
        ///   contrast:1.5     - Adjust contrast (0.0 to 3.0)
        ///   upscale-2x       - Bilinear 2x upscale
        ///
        /// Chain multiple: -f grayscale,sharpen,upscale-2x
        #[arg(short, long, value_delimiter = ',')]
        filters: Vec<String>,
    },

    /// Probe a media file
    Probe {
        /// Input file to probe
        input: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// List available GPU devices
    Devices,

    /// Probe NVIDIA CUDA capabilities (NVDEC/NVENC backbone)
    CudaInfo,

    /// End-to-end NVDEC self-test: decode a raw H.264 file via nvcuvid.dll
    NvdecTest {
        /// Input H.264 (Annex-B) file — defaults to the file produced by `nvenc-test`
        #[arg(short, long, default_value = "nvenc-test.h264")]
        input: String,
    },

    /// Stage-6a CUDA↔Vulkan external memory self-test
    InteropTest,

    /// Neo Lab — live WGSL shader chain editor with hot reload.
    /// Decodes a video, plays it in a window, applies every .wgsl in
    /// `--shaders` as a compute filter chain. Edit the files while it's
    /// running and the result updates on the next frame.
    Lab {
        /// Input raw H.264 (Annex-B) file
        #[arg(short, long)]
        input: String,
        /// Directory containing .wgsl shader files. Created if missing.
        #[arg(short, long, default_value = "shaders")]
        shaders: String,
        /// Playback frames per second (default: 30)
        #[arg(long)]
        fps: Option<f64>,
        /// Disable V-sync to measure raw shader chain throughput
        #[arg(long)]
        no_vsync: bool,
        /// Optional ONNX model applied *after* the shader chain.
        /// Must declare [1, 3, H, W] f32 input/output with H×W matching
        /// the source video (Stage B.1 = CPU bounce).
        #[arg(short = 'm', long)]
        model: Option<String>,
    },

    /// Stage-4 end-to-end transcode: NVDEC -> (CPU|wgpu) -> NVENC
    TranscodeTest {
        /// Input raw H.264 Annex-B file
        #[arg(short, long, default_value = "nvenc-test.h264")]
        input: String,
        /// Output raw H.264 Annex-B file
        #[arg(short, long, default_value = "transcode-out.h264")]
        output: String,
        /// Output framerate
        #[arg(long, default_value = "30")]
        fps: u32,
        /// NV12 -> BGRA conversion backend: `cpu` or `wgpu`
        #[arg(short, long, default_value = "wgpu")]
        backend: String,
    },

    /// End-to-end NVENC self-test: encode a synthetic gradient to H.264
    NvencTest {
        /// Output H.264 file (raw Annex-B; play with `ffplay`)
        #[arg(short, long, default_value = "nvenc-test.h264")]
        output: String,
        /// Width
        #[arg(long, default_value = "1920")]
        width: u32,
        /// Height
        #[arg(long, default_value = "1080")]
        height: u32,
        /// Number of frames
        #[arg(short, long, default_value = "120")]
        frames: u32,
        /// Framerate
        #[arg(long, default_value = "30")]
        fps: u32,
    },

    /// Show pipeline graph for a given configuration (dry run)
    Graph {
        /// Input source
        #[arg(short, long)]
        input: String,

        /// Output destination
        #[arg(short, long)]
        output: String,

        /// Filters
        #[arg(short, long, value_delimiter = ',')]
        filters: Vec<String>,
    },

    /// Process a video through the GPU pipeline
    ///
    /// FFmpeg decodes → GPU filters (zero-copy VRAM) → FFmpeg encodes
    Video {
        /// Input video file
        #[arg(short, long)]
        input: String,

        /// Output video file
        #[arg(short, long)]
        output: String,

        /// GPU filters to apply (comma-separated)
        #[arg(short, long, value_delimiter = ',')]
        filters: Vec<String>,

        /// Output codec (h264, h265, av1, nvenc, hevc_nvenc)
        #[arg(short, long, default_value = "h264")]
        codec: String,

        /// CRF quality (lower = better, 0-51)
        #[arg(long, default_value = "18")]
        crf: u32,

        /// Max frames to process (default: all)
        #[arg(long)]
        max_frames: Option<u64>,

        /// Start time in seconds
        #[arg(long)]
        start: Option<f64>,
    },

    /// Benchmark GPU filter throughput
    Bench {
        /// Resolution to benchmark (720, 1080, 4k)
        #[arg(short, long, default_value = "1080")]
        resolution: String,

        /// Filter to benchmark
        #[arg(short, long, default_value = "grayscale")]
        filter: String,

        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: u32,
    },
}

fn main() {
    let cli = Cli::parse();

    let filter = if cli.verbose {
        "neo_ffmpeg=debug,neo_core=debug,neo_gpu=debug,neo_pipeline=debug,neo_hwaccel=debug,neo_lab=debug"
    } else {
        "neo_ffmpeg=info,neo_hwaccel=info,neo_lab=info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(filter))
        .init();

    match cli.command {
        Commands::Process {
            input,
            output,
            filters,
        } => cmd_process(&input, &output, &filters),

        Commands::Video {
            input,
            output,
            filters,
            codec,
            crf,
            max_frames,
            start,
        } => cmd_video(&input, &output, &filters, &codec, crf, max_frames, start),

        Commands::Probe { input, json } => cmd_probe(&input, json),
        Commands::Devices => cmd_devices(),
        Commands::CudaInfo => cmd_cuda_info(),
        Commands::NvencTest {
            output,
            width,
            height,
            frames,
            fps,
        } => cmd_nvenc_test(&output, width, height, frames, fps),
        Commands::NvdecTest { input } => cmd_nvdec_test(&input),
        Commands::InteropTest => cmd_interop_test(),
        Commands::Lab {
            input,
            shaders,
            fps,
            no_vsync,
            model,
        } => cmd_lab(&input, &shaders, fps, no_vsync, model.as_deref()),
        Commands::TranscodeTest {
            input,
            output,
            fps,
            backend,
        } => cmd_transcode_test(&input, &output, fps, &backend),

        Commands::Graph {
            input,
            output,
            filters,
        } => cmd_graph(&input, &output, &filters),

        Commands::Bench {
            resolution,
            filter,
            iterations,
        } => cmd_bench(&resolution, &filter, iterations),
    }
}

fn cmd_process(input: &str, output: &str, filters: &[String]) {
    println!();
    println!("  Neo-FFmpeg -- Processing");
    println!("  ========================");
    println!("  Input:   {input}");
    println!("  Output:  {output}");
    println!(
        "  Filters: {}",
        if filters.is_empty() {
            "none (passthrough)".to_string()
        } else {
            filters.join(" -> ")
        }
    );
    println!();

    // Initialize GPU
    let ctx = match GpuContext::new_sync(&GpuOptions::default()) {
        Ok(ctx) => {
            println!("  GPU:     {} ({:?})", ctx.gpu_name(), ctx.backend());
            Arc::new(ctx)
        }
        Err(e) => {
            eprintln!("  ERROR: No GPU available: {e}");
            return;
        }
    };

    let executor = PipelineExecutor::new(ctx);

    println!();
    println!("  Running pipeline...");
    println!("  -------------------");

    match executor.run_pipeline(Path::new(input), Path::new(output), filters) {
        Ok(stats) => {
            println!();
            println!("  Results");
            println!("  -------");
            println!(
                "  Input:    {}x{}",
                stats.input_width, stats.input_height
            );
            println!(
                "  Output:   {}x{}",
                stats.output_width, stats.output_height
            );
            println!("  Filters:  {}", stats.filters_applied);
            println!();
            println!("  Timing");
            println!("  ------");
            println!("  Load (CPU->VRAM):  {} ms", stats.load_time_ms);
            println!(
                "  Process (GPU):     {} ms  <-- zero-copy in VRAM",
                stats.process_time_ms
            );
            println!("  Save (VRAM->CPU):  {} ms", stats.save_time_ms);
            println!("  Total:             {} ms", stats.total_time_ms);
            println!();
            println!("  Done! Output saved to: {output}");
        }
        Err(e) => {
            eprintln!("  Pipeline error: {e}");
        }
    }
}

fn cmd_video(
    input: &str,
    output: &str,
    filters: &[String],
    codec: &str,
    crf: u32,
    max_frames: Option<u64>,
    start: Option<f64>,
) {
    println!();
    println!("  Neo-FFmpeg -- Video Pipeline");
    println!("  ============================");
    println!("  Input:   {input}");
    println!("  Output:  {output}");
    println!(
        "  Filters: {}",
        if filters.is_empty() {
            "none (passthrough)".to_string()
        } else {
            filters.join(" -> ")
        }
    );
    println!("  Codec:   {codec}");
    println!("  CRF:     {crf}");
    if let Some(n) = max_frames {
        println!("  Frames:  {n} (limited)");
    }
    if let Some(ss) = start {
        println!("  Start:   {ss}s");
    }
    println!();

    // Find ffmpeg
    let ffmpeg = match PipelineExecutor::find_ffmpeg() {
        Ok(p) => {
            println!("  FFmpeg:  {}", p.display());
            p
        }
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return;
        }
    };

    // Initialize GPU
    let ctx = match GpuContext::new_sync(&GpuOptions::default()) {
        Ok(ctx) => {
            println!("  GPU:     {} ({:?})", ctx.gpu_name(), ctx.backend());
            Arc::new(ctx)
        }
        Err(e) => {
            eprintln!("  ERROR: No GPU available: {e}");
            return;
        }
    };

    // Probe video
    let video_info = match PipelineExecutor::probe_video(&ffmpeg, Path::new(input)) {
        Ok(info) => {
            println!(
                "  Video:   {}x{} @ {:.2} FPS, {:.1}s, {} frames",
                info.width, info.height, info.fps, info.duration, info.total_frames
            );
            info
        }
        Err(e) => {
            eprintln!("  ERROR: Failed to probe video: {e}");
            return;
        }
    };

    let _ = video_info; // just for display above

    let executor = PipelineExecutor::new(ctx);

    println!();
    println!("  Processing...");
    println!("  -------------------");

    match executor.run_video_pipeline(
        &ffmpeg,
        Path::new(input),
        Path::new(output),
        filters,
        max_frames,
        start,
        codec,
        crf,
    ) {
        Ok(stats) => {
            println!();
            println!("  Results");
            println!("  -------");
            println!(
                "  Input:          {}x{} @ {:.2} FPS",
                stats.input_width, stats.input_height, stats.fps_in
            );
            println!(
                "  Output:         {}x{}",
                stats.output_width, stats.output_height
            );
            println!("  Frames:         {}", stats.frames_processed);
            println!("  Filters:        {}", stats.filters_applied);
            println!();
            println!("  Timing");
            println!("  ------");
            println!("  Total:          {} ms", stats.total_time_ms);
            println!(
                "  Upload (CPU→GPU):  {:.1} ms  ({:.1} ms/frame)",
                stats.total_upload_us as f64 / 1000.0,
                stats.total_upload_us as f64 / stats.frames_processed as f64 / 1000.0
            );
            println!(
                "  GPU compute:       {:.1} ms  ({:.1} ms/frame)  <-- zero-copy VRAM",
                stats.total_gpu_us as f64 / 1000.0,
                stats.avg_gpu_per_frame_us as f64 / 1000.0
            );
            println!(
                "  Download (GPU→CPU):{:.1} ms  ({:.1} ms/frame)",
                stats.total_download_us as f64 / 1000.0,
                stats.total_download_us as f64 / stats.frames_processed as f64 / 1000.0
            );
            println!(
                "  Effective FPS:  {:.1}",
                stats.effective_fps
            );
            println!();
            println!("  Done! Output saved to: {output}");
        }
        Err(e) => {
            eprintln!("  Pipeline error: {e}");
        }
    }
}

fn cmd_probe(input: &str, json: bool) {
    println!("  Neo-FFmpeg -- Probe");
    println!("  ===================");

    match neo_decode::probe_file(Path::new(input)) {
        Ok(info) => {
            if json {
                match serde_json::to_string_pretty(&info) {
                    Ok(j) => println!("{j}"),
                    Err(e) => eprintln!("JSON error: {e}"),
                }
            } else {
                println!("  File:      {input}");
                println!("  Container: {:?}", info.container);
                if let Some(d) = info.duration {
                    println!("  Duration:  {:.2}s", d.as_secs_f64());
                }
                if let Some(b) = info.bitrate {
                    println!("  Bitrate:   {} kbps", b / 1000);
                }
                println!("  Streams:   {}", info.streams.len());
                for s in &info.streams {
                    println!("    #{}: {:?} ({})", s.index, s.stream_type, s.codec);
                    if let (Some(w), Some(h)) = (s.width, s.height) {
                        println!("        Resolution: {w}x{h}");
                    }
                }
            }
        }
        Err(e) => eprintln!("  Probe error: {e}"),
    }
}

fn cmd_devices() {
    println!();
    println!("  Neo-FFmpeg -- GPU Devices");
    println!("  =========================");

    match GpuContext::new_sync(&GpuOptions::default()) {
        Ok(ctx) => {
            println!("  GPU:         {}", ctx.gpu_name());
            println!("  Backend:     {:?}", ctx.backend());
            println!(
                "  Max buffer:  {} MB",
                ctx.max_buffer_size() / (1024 * 1024)
            );
            println!();
            println!("  Available filters:");
            println!("    grayscale, invert, sepia, blur, sharpen,");
            println!("    edge-detect, brightness, contrast, upscale-2x");
            println!();
            println!("  Status: READY");
        }
        Err(e) => {
            eprintln!("  No GPU available: {e}");
        }
    }
}

fn cmd_cuda_info() {
    use neo_hwaccel::CudaRuntime;

    println!();
    println!("  Neo-FFmpeg -- CUDA / NVDEC / NVENC");
    println!("  ==================================");

    match CudaRuntime::probe() {
        Ok(caps) => {
            if caps.devices.is_empty() {
                println!("  CUDA driver loaded, but no devices reported.");
                return;
            }
            println!("  CUDA devices: {}", caps.devices.len());
            for d in &caps.devices {
                println!();
                println!("  [{}] {}", d.ordinal, d.name);
                println!(
                    "      Compute capability: {}.{}",
                    d.compute_capability.0, d.compute_capability.1
                );
                println!(
                    "      VRAM:               {:.1} GB",
                    d.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                println!("      SMs:                {}", d.multiprocessor_count);
                let nvenc_ok = d.compute_capability.0 >= 5;
                let nvdec_ok = d.compute_capability.0 >= 3;
                println!(
                    "      NVDEC capable:      {}",
                    if nvdec_ok { "yes" } else { "no" }
                );
                println!(
                    "      NVENC capable:      {}",
                    if nvenc_ok { "yes" } else { "no" }
                );
            }
            println!();

            // Try to actually open a context, not just enumerate
            match CudaRuntime::new(0) {
                Ok(rt) => {
                    println!("  Context init:  OK");

                    // Probe NVENC end-to-end via dynamic-loaded nvEncodeAPI64.dll
                    print!("  NVENC probe:   ");
                    match neo_hwaccel::probe_nvenc(&rt) {
                        Ok(caps) => {
                            println!("OK ({} codec GUIDs)", caps.raw_guid_count);
                            println!(
                                "      H.264: {}   HEVC: {}   AV1: {}",
                                if caps.h264 { "yes" } else { "no" },
                                if caps.hevc { "yes" } else { "no" },
                                if caps.av1 { "yes" } else { "no" },
                            );
                            println!();
                            println!("  Status: READY for native NVENC (zero-link-dep dynamic load)");
                        }
                        Err(e) => {
                            println!("FAILED");
                            eprintln!("      {e}");
                        }
                    }
                }
                Err(e) => {
                    println!("  Context init:  FAILED ({e})");
                }
            }
        }
        Err(e) => {
            eprintln!("  CUDA unavailable: {e}");
            eprintln!();
            eprintln!("  Neo-FFmpeg will fall back to the FFmpeg subprocess pipeline.");
        }
    }
}

fn cmd_nvenc_test(output: &str, width: u32, height: u32, frames: u32, fps: u32) {
    use neo_hwaccel::{run_encode_test, CudaRuntime};

    println!();
    println!("  Neo-FFmpeg -- NVENC Self-Test");
    println!("  =============================");
    println!("  Output:    {output}");
    println!("  Size:      {width}x{height}");
    println!("  Frames:    {frames}");
    println!("  Framerate: {fps}");
    println!();

    let runtime = match CudaRuntime::new(0) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("  ERROR: CUDA init failed: {e}");
            return;
        }
    };

    println!("  Encoding...");
    match run_encode_test(&runtime, Path::new(output), width, height, frames, fps) {
        Ok(r) => {
            println!();
            println!("  Results");
            println!("  -------");
            println!("  Frames written: {}", r.frames);
            println!("  Bytes written:  {} ({:.2} MB)", r.bytes_written, r.bytes_written as f64 / 1_048_576.0);
            println!("  Elapsed:        {:.2} s", r.elapsed.as_secs_f64());
            println!("  Encode FPS:     {:.1}", r.fps());
            println!();
            println!("  Done! Play with: ffplay {output}");
        }
        Err(e) => {
            eprintln!();
            eprintln!("  NVENC test FAILED: {e}");
        }
    }
}

fn cmd_nvdec_test(input: &str) {
    use neo_hwaccel::{decode_nvdec, CudaRuntime, NvdecCodec};
    use std::fs;
    use std::time::Instant;

    println!();
    println!("  Neo-FFmpeg -- NVDEC Self-Test");
    println!("  =============================");
    println!("  Input: {input}");
    println!();

    let bytes = match fs::read(input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  ERROR: cannot read {input}: {e}");
            return;
        }
    };
    println!("  Read {} bytes ({:.2} MB)", bytes.len(), bytes.len() as f64 / 1_048_576.0);

    let runtime = match CudaRuntime::new(0) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("  ERROR: CUDA init failed: {e}");
            return;
        }
    };

    println!("  Decoding via dynamic-loaded nvcuvid.dll...");
    let start = Instant::now();
    match decode_nvdec(&runtime, NvdecCodec::cudaVideoCodec_H264, &bytes) {
        Ok(stats) => {
            let elapsed = start.elapsed();
            println!();
            println!("  Results");
            println!("  -------");
            println!("  Coded size:       {}x{}", stats.coded_width, stats.coded_height);
            println!("  Display size:     {}x{}", stats.display_width, stats.display_height);
            println!("  Pictures decoded: {}", stats.pictures_decoded);
            println!("  Pictures shown:   {}", stats.pictures_displayed);
            println!("  Elapsed:          {:.2} s", elapsed.as_secs_f64());
            if stats.pictures_decoded > 0 {
                println!(
                    "  Decode FPS:       {:.1}",
                    stats.pictures_decoded as f64 / elapsed.as_secs_f64()
                );
            }
            println!();
            if stats.pictures_decoded > 0 {
                println!("  SUCCESS - NVDEC dynamic loader works end-to-end.");
            } else {
                println!("  WARNING - parser ran but no pictures were decoded.");
            }
        }
        Err(e) => {
            eprintln!();
            eprintln!("  NVDEC test FAILED: {e}");
        }
    }
}

fn cmd_lab(
    input: &str,
    shaders: &str,
    fps: Option<f64>,
    no_vsync: bool,
    model: Option<&str>,
) {
    use neo_lab::LabOptions;
    use std::path::PathBuf;

    println!();
    println!("  Neo-FFmpeg -- Lab (live shader chain)");
    println!("  =====================================");
    println!("  Input:   {input}");
    println!("  Shaders: {shaders}");
    if let Some(f) = fps {
        println!("  FPS:     {f}");
    }
    if no_vsync {
        println!("  V-sync:  OFF (raw throughput mode)");
    }
    if let Some(m) = model {
        println!("  Model:   {m}  (Stage B.1 CPU bounce)");
    }
    println!();
    println!("  Drop or edit any .wgsl file in '{shaders}' while the");
    println!("  window is open and the chain reloads on the next frame.");
    println!("  Press Esc or close the window to exit.");
    println!();

    let opts = LabOptions {
        input: PathBuf::from(input),
        shaders_dir: PathBuf::from(shaders),
        fps,
        no_vsync,
        model: model.map(PathBuf::from),
    };
    if let Err(e) = neo_lab::run(opts) {
        eprintln!("  Lab failed: {e}");
    }
}

#[cfg(windows)]
fn cmd_transcode_zerocopy(input: &str, output: &str, fps: u32) {
    use neo_hwaccel::{transcode_h264_zerocopy, CudaRuntime};
    use std::fs;

    println!();
    println!("  Neo-FFmpeg -- Stage 6b: TRUE ZERO-COPY Transcode");
    println!("  ================================================");
    println!("  Input:   {input}");
    println!("  Output:  {output}");
    println!("  FPS:     {fps}");
    println!("  Backend: NVDEC -> CUDA↔Vulkan interop -> wgpu compute -> NVENC");
    println!();

    let bytes = match fs::read(input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  ERROR: cannot read {input}: {e}");
            return;
        }
    };
    println!("  Read {} bytes ({:.2} MB)", bytes.len(), bytes.len() as f64 / 1_048_576.0);

    let runtime = match CudaRuntime::new(0) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("  ERROR: CUDA init failed: {e}");
            return;
        }
    };

    match transcode_h264_zerocopy(&runtime, &bytes, Path::new(output), fps) {
        Ok(s) => {
            println!();
            println!("  Results");
            println!("  -------");
            println!("  Size:            {}x{}", s.width, s.height);
            println!("  Frames decoded:  {}", s.frames_decoded);
            println!("  Frames encoded:  {}", s.frames_encoded);
            println!(
                "  Bytes written:   {} ({:.2} MB)",
                s.bytes_written,
                s.bytes_written as f64 / 1_048_576.0
            );
            println!("  Decode:          {:.2} s (includes hook work)", s.decode_elapsed.as_secs_f64());
            println!("  Convert (GPU):   {:.2} s", s.convert_elapsed.as_secs_f64());
            println!("  Encode:          {:.2} s", s.encode_elapsed.as_secs_f64());
            println!("  Total:           {:.2} s", s.total_elapsed.as_secs_f64());
            println!("  Convert FPS:     {:.1}", s.frames_encoded as f64 / s.convert_elapsed.as_secs_f64().max(1e-9));
            println!("  Encode FPS:      {:.1}", s.encode_fps());
            println!("  End-to-end FPS:  {:.1}", s.total_fps());
            println!();
            println!("  SUCCESS — every frame stayed in VRAM from NVDEC to NVENC.");
            println!("  Play with: ffplay {output}");
        }
        Err(e) => {
            eprintln!();
            eprintln!("  Zero-copy transcode FAILED: {e}");
        }
    }
}

#[cfg(not(windows))]
fn cmd_transcode_zerocopy(_input: &str, _output: &str, _fps: u32) {
    eprintln!("  zerocopy backend is Windows-only (OpaqueWin32 handle path).");
}

#[cfg(windows)]
fn cmd_interop_test() {
    use neo_gpu::{GpuContext, GpuOptions};
    use neo_hwaccel::{interop::interop_self_test, CudaRuntime};
    use std::sync::Arc;

    println!();
    println!("  Neo-FFmpeg -- Stage 6a: CUDA↔Vulkan Interop Self-Test");
    println!("  =====================================================");
    println!();

    let cuda = match CudaRuntime::new(0) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("  ERROR: CUDA init failed: {e}");
            return;
        }
    };
    println!("  CUDA:    OK ({})", cuda.capabilities.devices[0].name);

    let gpu = match GpuContext::new_sync(&GpuOptions::interop()) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            eprintln!("  ERROR: wgpu init (interop features) failed: {e}");
            eprintln!("         The Vulkan driver must expose VK_KHR_external_memory_win32.");
            return;
        }
    };
    println!("  wgpu:    {} ({:?})", gpu.gpu_name(), gpu.backend());
    println!();

    match interop_self_test(gpu, &cuda) {
        Ok(s) => {
            println!("  Results");
            println!("  -------");
            println!("  Size:             {} KiB", s.size / 1024);
            println!("  Pattern:          {:#010x}", s.pattern);
            println!("  Mismatches:       {}", s.mismatches);
            println!("  First word seen:  {:#010x}", s.first_word);
            println!("  Last word seen:   {:#010x}", s.last_word);
            println!("  CUDA write:       {:.2} ms", s.cuda_write_ms);
            println!("  wgpu read-back:   {:.2} ms", s.wgpu_read_ms);
            println!();
            if s.ok {
                println!("  SUCCESS - wgpu sees the bytes CUDA wrote. Zero-copy alias is live.");
            } else {
                println!("  FAILURE - views disagree. Check driver + feature support.");
            }
        }
        Err(e) => {
            eprintln!();
            eprintln!("  Interop test FAILED: {e}");
        }
    }
}

#[cfg(not(windows))]
fn cmd_interop_test() {
    eprintln!("  interop-test is Windows-only in Stage 6a (OpaqueWin32 handle path).");
}

fn cmd_transcode_test(input: &str, output: &str, fps: u32, backend: &str) {
    use neo_hwaccel::{transcode_h264, ConvertBackend, CudaRuntime};
    use std::fs;

    // `zerocopy` is handled via a separate code path below.
    if backend.eq_ignore_ascii_case("zerocopy") || backend.eq_ignore_ascii_case("interop") {
        return cmd_transcode_zerocopy(input, output, fps);
    }

    let backend_sel = match backend.to_ascii_lowercase().as_str() {
        "cpu" => ConvertBackend::Cpu,
        "wgpu" | "gpu" => ConvertBackend::WgpuCompute,
        other => {
            eprintln!("  ERROR: unknown backend '{other}' (want cpu|wgpu|zerocopy)");
            return;
        }
    };

    println!();
    println!("  Neo-FFmpeg -- Stage 4 Transcode Self-Test");
    println!("  =========================================");
    println!("  Input:   {input}");
    println!("  Output:  {output}");
    println!("  FPS:     {fps}");
    println!("  Backend: {backend_sel:?}");
    println!();

    let bytes = match fs::read(input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  ERROR: cannot read {input}: {e}");
            return;
        }
    };
    println!("  Read {} bytes ({:.2} MB)", bytes.len(), bytes.len() as f64 / 1_048_576.0);

    let runtime = match CudaRuntime::new(0) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("  ERROR: CUDA init failed: {e}");
            return;
        }
    };

    let pipeline_desc = match backend_sel {
        ConvertBackend::Cpu => "nvcuvid.dll -> CPU NV12 -> CPU BGRA -> nvEncodeAPI64.dll",
        ConvertBackend::WgpuCompute => {
            "nvcuvid.dll -> CPU NV12 -> wgpu compute -> nvEncodeAPI64.dll"
        }
    };
    println!("  Pipeline: {pipeline_desc}");
    match transcode_h264(&runtime, &bytes, Path::new(output), fps, backend_sel) {
        Ok(s) => {
            println!();
            println!("  Results");
            println!("  -------");
            println!("  Size:            {}x{}", s.width, s.height);
            println!("  Frames decoded:  {}", s.frames_decoded);
            println!("  Frames encoded:  {}", s.frames_encoded);
            println!(
                "  Bytes written:   {} ({:.2} MB)",
                s.bytes_written,
                s.bytes_written as f64 / 1_048_576.0
            );
            println!("  Decode:          {:.2} s", s.decode_elapsed.as_secs_f64());
            println!("  NV12->BGRA:      {:.2} s", s.convert_elapsed.as_secs_f64());
            println!("  Encode:          {:.2} s", s.encode_elapsed.as_secs_f64());
            println!("  Total:           {:.2} s", s.total_elapsed.as_secs_f64());
            println!("  Encode FPS:      {:.1}", s.encode_fps());
            println!("  End-to-end FPS:  {:.1}", s.total_fps());
            println!();
            println!("  SUCCESS - full NVDEC -> CPU -> NVENC round trip.");
            println!("  Play with: ffplay {output}");
        }
        Err(e) => {
            eprintln!();
            eprintln!("  Transcode FAILED: {e}");
        }
    }
}

fn cmd_graph(input: &str, output: &str, filters: &[String]) {
    println!();
    println!("  Neo-FFmpeg -- Pipeline Graph");
    println!("  ============================");
    println!();
    print!("  [LOAD {input}]");
    print!(" -> [Upload to VRAM]");
    for f in filters {
        print!(" -> [{f}]");
    }
    print!(" -> [Download from VRAM]");
    println!(" -> [SAVE {output}]");
    println!();
    println!("              |<--- ZERO-COPY GPU ZONE --->|");
    println!();
    println!("  All filter nodes execute entirely in VRAM.");
    println!("  No CPU-GPU round-trips between filters.");
}

fn cmd_bench(resolution: &str, filter: &str, iterations: u32) {
    println!();
    println!("  Neo-FFmpeg -- Benchmark");
    println!("  =======================");

    let (w, h) = match resolution {
        "720" => (1280u32, 720u32),
        "1080" => (1920, 1080),
        "4k" | "2160" => (3840, 2160),
        "8k" | "4320" => (7680, 4320),
        _ => {
            eprintln!("  Unknown resolution: {resolution}");
            return;
        }
    };

    println!("  Resolution:  {w}x{h}");
    println!("  Filter:      {filter}");
    println!("  Iterations:  {iterations}");
    println!();

    let ctx = match GpuContext::new_sync(&GpuOptions::default()) {
        Ok(ctx) => {
            println!("  GPU: {} ({:?})", ctx.gpu_name(), ctx.backend());
            Arc::new(ctx)
        }
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return;
        }
    };

    let executor = PipelineExecutor::new(ctx.clone());

    // Create a synthetic RGBA frame
    let pixel_count = (w * h) as usize;
    let mut rgba_data = vec![0u8; pixel_count * 4];
    for i in 0..pixel_count {
        let x = (i % w as usize) as u8;
        let y = (i / w as usize) as u8;
        rgba_data[i * 4] = x;         // R
        rgba_data[i * 4 + 1] = y;     // G
        rgba_data[i * 4 + 2] = 128;   // B
        rgba_data[i * 4 + 3] = 255;   // A
    }

    let input_buf = match neo_gpu::VramBuffer::from_data(&ctx, &rgba_data, "bench-input") {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  Buffer alloc error: {e}");
            return;
        }
    };

    let frame = neo_pipeline::GpuFrameData {
        buffer: input_buf,
        width: w,
        height: h,
    };

    println!("  Frame uploaded to VRAM ({} MB)", rgba_data.len() / (1024 * 1024));
    println!();
    println!("  Running {iterations} iterations...");

    // Warmup
    let _ = executor.apply_filter(&frame, filter);

    let start = Instant::now();
    for _ in 0..iterations {
        match executor.apply_filter(&frame, filter) {
            Ok(_result) => {}
            Err(e) => {
                eprintln!("  Filter error: {e}");
                return;
            }
        }
    }
    // Make sure all GPU work is done
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    let fps = 1_000_000.0 / avg_us;
    let megapixels = (w as f64 * h as f64) / 1_000_000.0;
    let mpx_per_sec = megapixels * fps;

    println!();
    println!("  Results");
    println!("  -------");
    println!("  Total time:    {:.1} ms", elapsed.as_millis());
    println!("  Avg per frame: {:.1} us", avg_us);
    println!("  Throughput:    {:.0} FPS", fps);
    println!("  Pixel rate:    {:.1} Mpx/s", mpx_per_sec);
    println!();

    if fps >= 60.0 {
        println!("  Verdict: REAL-TIME @ {w}x{h} ({fps:.0} FPS)");
    } else if fps >= 24.0 {
        println!("  Verdict: Cinematic @ {w}x{h} ({fps:.0} FPS)");
    } else {
        println!("  Verdict: Offline processing @ {w}x{h} ({fps:.0} FPS)");
    }
}
