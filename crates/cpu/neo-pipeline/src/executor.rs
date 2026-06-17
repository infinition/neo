use neo_core::{NeoError, NeoResult};
use neo_gpu::buffer::VramBuffer;
use neo_gpu::compute::{dispatch_compute, Binding, BufferAccess};
use neo_gpu::context::GpuContext;
use neo_gpu::shader::builtins;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc as sync_mpsc, Arc};
use std::thread;
use std::time::Instant;
use tracing::info;

/// Represents a frame living in VRAM as packed RGBA u32 pixels.
pub struct GpuFrameData {
    pub buffer: VramBuffer,
    pub width: u32,
    pub height: u32,
}

/// The pipeline executor — loads images, runs GPU filters, saves output.
///
/// This is the beating heart of Neo-FFmpeg. It orchestrates the full
/// zero-copy pipeline: load → upload → process → download → save.
pub struct PipelineExecutor {
    ctx: Arc<GpuContext>,
}

impl PipelineExecutor {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        Self { ctx }
    }

    /// Load an image file into VRAM as packed RGBA u32 pixels.
    pub fn load_image(&self, path: &Path) -> NeoResult<GpuFrameData> {
        let img = image::open(path)
            .map_err(|e| NeoError::Decode(format!("failed to load image: {e}")))?
            .to_rgba8();

        let width = img.width();
        let height = img.height();
        let rgba_bytes = img.into_raw();

        info!(
            width,
            height,
            bytes = rgba_bytes.len(),
            "Image loaded to CPU"
        );

        // Upload to VRAM
        let buffer = VramBuffer::from_data(&self.ctx, &rgba_bytes, "frame-input")?;

        info!("Frame uploaded to VRAM (zero-copy zone entered)");

        Ok(GpuFrameData {
            buffer,
            width,
            height,
        })
    }

    /// Save a VRAM frame to an image file.
    pub fn save_image(&self, frame: &GpuFrameData, path: &Path) -> NeoResult<()> {
        let data = frame.buffer.download_sync(&self.ctx)?;

        info!(
            width = frame.width,
            height = frame.height,
            bytes = data.len(),
            "Frame downloaded from VRAM"
        );

        let img = image::RgbaImage::from_raw(frame.width, frame.height, data)
            .ok_or_else(|| NeoError::Encode("failed to create image from VRAM data".into()))?;

        img.save(path)
            .map_err(|e| NeoError::Encode(format!("failed to save image: {e}")))?;

        info!(path = %path.display(), "Image saved");
        Ok(())
    }

    /// Apply a named filter to a frame, entirely in VRAM.
    ///
    /// Returns a new GpuFrameData with the result (the input buffer is not freed).
    pub fn apply_filter(
        &self,
        frame: &GpuFrameData,
        filter_name: &str,
    ) -> NeoResult<GpuFrameData> {
        let start = Instant::now();

        let result = match filter_name {
            "grayscale" => self.run_same_size_filter(frame, builtins::GRAYSCALE, "grayscale"),
            "invert" => self.run_same_size_filter(frame, builtins::INVERT, "invert"),
            "blur" => self.run_same_size_filter(frame, builtins::BLUR, "blur"),
            "edge-detect" | "edges" => {
                self.run_same_size_filter(frame, builtins::EDGE_DETECT, "edge_detect")
            }
            "sepia" => self.run_sepia(frame, 1.0),
            "sharpen" => self.run_sharpen(frame, 1.5),
            "brightness" => self.run_brightness_contrast(frame, 0.1, 1.0),
            "contrast" => self.run_brightness_contrast(frame, 0.0, 1.5),
            "upscale-2x" | "upscale" => self.run_upscale_2x(frame),
            name if name.starts_with("sepia:") => {
                let intensity: f32 = name[6..].parse().unwrap_or(1.0);
                self.run_sepia(frame, intensity)
            }
            name if name.starts_with("sharpen:") => {
                let strength: f32 = name[8..].parse().unwrap_or(1.5);
                self.run_sharpen(frame, strength)
            }
            name if name.starts_with("brightness:") => {
                let val: f32 = name[11..].parse().unwrap_or(0.1);
                self.run_brightness_contrast(frame, val, 1.0)
            }
            name if name.starts_with("contrast:") => {
                let val: f32 = name[9..].parse().unwrap_or(1.5);
                self.run_brightness_contrast(frame, 0.0, val)
            }
            _ => return Err(NeoError::Pipeline(format!("unknown filter: {filter_name}"))),
        }?;

        let elapsed = start.elapsed();
        info!(
            filter = filter_name,
            elapsed_us = elapsed.as_micros(),
            "Filter applied on GPU"
        );

        Ok(result)
    }

    /// Run a same-size filter (input and output have identical dimensions).
    fn run_same_size_filter(
        &self,
        frame: &GpuFrameData,
        shader_source: &str,
        entry_point: &str,
    ) -> NeoResult<GpuFrameData> {
        let output_buf =
            VramBuffer::new_storage(&self.ctx, frame.buffer.size, &format!("{entry_point}-out"))?;

        // Params: width, height, pad, pad
        let params = [frame.width, frame.height, 0u32, 0u32];
        let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = VramBuffer::new_uniform(&self.ctx, &params_bytes, "params")?;

        dispatch_compute(
            &self.ctx,
            shader_source,
            entry_point,
            &[
                Binding {
                    buffer: &frame.buffer,
                    access: BufferAccess::ReadOnly,
                },
                Binding {
                    buffer: &output_buf,
                    access: BufferAccess::ReadWrite,
                },
                Binding {
                    buffer: &params_buf,
                    access: BufferAccess::Uniform,
                },
            ],
            [
                (frame.width + 7) / 8,
                (frame.height + 7) / 8,
                1,
            ],
        )?;

        Ok(GpuFrameData {
            buffer: output_buf,
            width: frame.width,
            height: frame.height,
        })
    }

    fn run_sepia(&self, frame: &GpuFrameData, intensity: f32) -> NeoResult<GpuFrameData> {
        let output_buf =
            VramBuffer::new_storage(&self.ctx, frame.buffer.size, "sepia-out")?;

        let params_bytes = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&frame.width.to_le_bytes());
            buf.extend_from_slice(&frame.height.to_le_bytes());
            buf.extend_from_slice(&intensity.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes()); // pad
            buf
        };
        let params_buf = VramBuffer::new_uniform(&self.ctx, &params_bytes, "sepia-params")?;

        dispatch_compute(
            &self.ctx,
            builtins::SEPIA,
            "sepia",
            &[
                Binding { buffer: &frame.buffer, access: BufferAccess::ReadOnly },
                Binding { buffer: &output_buf, access: BufferAccess::ReadWrite },
                Binding { buffer: &params_buf, access: BufferAccess::Uniform },
            ],
            [(frame.width + 7) / 8, (frame.height + 7) / 8, 1],
        )?;

        Ok(GpuFrameData { buffer: output_buf, width: frame.width, height: frame.height })
    }

    fn run_sharpen(&self, frame: &GpuFrameData, strength: f32) -> NeoResult<GpuFrameData> {
        let output_buf =
            VramBuffer::new_storage(&self.ctx, frame.buffer.size, "sharpen-out")?;

        let params_bytes = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&frame.width.to_le_bytes());
            buf.extend_from_slice(&frame.height.to_le_bytes());
            buf.extend_from_slice(&strength.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf
        };
        let params_buf = VramBuffer::new_uniform(&self.ctx, &params_bytes, "sharpen-params")?;

        dispatch_compute(
            &self.ctx,
            builtins::SHARPEN,
            "sharpen",
            &[
                Binding { buffer: &frame.buffer, access: BufferAccess::ReadOnly },
                Binding { buffer: &output_buf, access: BufferAccess::ReadWrite },
                Binding { buffer: &params_buf, access: BufferAccess::Uniform },
            ],
            [(frame.width + 7) / 8, (frame.height + 7) / 8, 1],
        )?;

        Ok(GpuFrameData { buffer: output_buf, width: frame.width, height: frame.height })
    }

    fn run_brightness_contrast(
        &self,
        frame: &GpuFrameData,
        brightness: f32,
        contrast: f32,
    ) -> NeoResult<GpuFrameData> {
        let output_buf =
            VramBuffer::new_storage(&self.ctx, frame.buffer.size, "bc-out")?;

        let params_bytes = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&frame.width.to_le_bytes());
            buf.extend_from_slice(&frame.height.to_le_bytes());
            buf.extend_from_slice(&brightness.to_le_bytes());
            buf.extend_from_slice(&contrast.to_le_bytes());
            buf
        };
        let params_buf = VramBuffer::new_uniform(&self.ctx, &params_bytes, "bc-params")?;

        dispatch_compute(
            &self.ctx,
            builtins::BRIGHTNESS_CONTRAST,
            "brightness_contrast",
            &[
                Binding { buffer: &frame.buffer, access: BufferAccess::ReadOnly },
                Binding { buffer: &output_buf, access: BufferAccess::ReadWrite },
                Binding { buffer: &params_buf, access: BufferAccess::Uniform },
            ],
            [(frame.width + 7) / 8, (frame.height + 7) / 8, 1],
        )?;

        Ok(GpuFrameData { buffer: output_buf, width: frame.width, height: frame.height })
    }

    fn run_upscale_2x(&self, frame: &GpuFrameData) -> NeoResult<GpuFrameData> {
        let out_width = frame.width * 2;
        let out_height = frame.height * 2;
        let out_size = (out_width * out_height * 4) as u64;

        let output_buf = VramBuffer::new_storage(&self.ctx, out_size, "upscale-out")?;

        let params_bytes = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&frame.width.to_le_bytes());
            buf.extend_from_slice(&frame.height.to_le_bytes());
            buf.extend_from_slice(&out_width.to_le_bytes());
            buf.extend_from_slice(&out_height.to_le_bytes());
            buf
        };
        let params_buf = VramBuffer::new_uniform(&self.ctx, &params_bytes, "upscale-params")?;

        dispatch_compute(
            &self.ctx,
            builtins::UPSCALE_2X,
            "upscale_2x",
            &[
                Binding { buffer: &frame.buffer, access: BufferAccess::ReadOnly },
                Binding { buffer: &output_buf, access: BufferAccess::ReadWrite },
                Binding { buffer: &params_buf, access: BufferAccess::Uniform },
            ],
            [(out_width + 7) / 8, (out_height + 7) / 8, 1],
        )?;

        info!(
            from = %format!("{}x{}", frame.width, frame.height),
            to = %format!("{out_width}x{out_height}"),
            "Upscale 2x executed on GPU"
        );

        Ok(GpuFrameData {
            buffer: output_buf,
            width: out_width,
            height: out_height,
        })
    }

    /// Upload raw RGBA bytes into VRAM (for video frame processing).
    pub fn upload_raw_frame(
        &self,
        rgba_data: &[u8],
        width: u32,
        height: u32,
    ) -> NeoResult<GpuFrameData> {
        let buffer = VramBuffer::from_data(&self.ctx, rgba_data, "video-frame")?;
        Ok(GpuFrameData {
            buffer,
            width,
            height,
        })
    }

    /// Download a VRAM frame to raw RGBA bytes.
    pub fn download_raw_frame(&self, frame: &GpuFrameData) -> NeoResult<Vec<u8>> {
        frame.buffer.download_sync(&self.ctx)
    }

    /// Get the GPU context (for poll/sync).
    pub fn ctx(&self) -> &GpuContext {
        &self.ctx
    }

    /// Find ffmpeg executable on the system.
    pub fn find_ffmpeg() -> NeoResult<PathBuf> {
        // Check common locations
        let candidates = [
            // Winget install location
            PathBuf::from(r"C:\Users\infinition\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"),
            PathBuf::from("ffmpeg.exe"),
            PathBuf::from("ffmpeg"),
        ];

        for path in &candidates {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Try which/where
        if let Ok(output) = Command::new("where").arg("ffmpeg").output() {
            if output.status.success() {
                let path_str = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = path_str.lines().next() {
                    let p = PathBuf::from(line.trim());
                    if p.exists() {
                        return Ok(p);
                    }
                }
            }
        }

        Err(NeoError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "ffmpeg not found. Install with: winget install Gyan.FFmpeg",
        )))
    }

    /// Probe a video file using ffmpeg to get width, height, fps, duration, frame count.
    pub fn probe_video(ffmpeg: &Path, input: &Path) -> NeoResult<VideoInfo> {
        let ffprobe = ffmpeg.with_file_name(
            if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" }
        );

        let output = Command::new(&ffprobe)
            .args([
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
                "-show_entries", "format=duration",
                "-of", "csv=p=0:s=,",
            ])
            .arg(input)
            .output()
            .map_err(|e| NeoError::Io(e))?;

        let text = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = text.trim().lines().collect();

        // Parse stream line: width,height,fps_num/fps_den,nb_frames,duration
        let stream_parts: Vec<&str> = lines.first().unwrap_or(&"").split(',').collect();

        let width: u32 = stream_parts.first().and_then(|s| s.parse().ok()).unwrap_or(1920);
        let height: u32 = stream_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1080);

        let fps_str = stream_parts.get(2).unwrap_or(&"24/1");
        let fps: f64 = if let Some((n, d)) = fps_str.split_once('/') {
            let num: f64 = n.parse().unwrap_or(24.0);
            let den: f64 = d.parse().unwrap_or(1.0);
            if den > 0.0 { num / den } else { 24.0 }
        } else {
            fps_str.parse().unwrap_or(24.0)
        };

        // Try to get frame count
        let nb_frames: Option<u64> = stream_parts.get(3).and_then(|s| s.parse().ok());

        // Duration from stream or format
        let duration: f64 = stream_parts
            .get(4)
            .and_then(|s| s.parse().ok())
            .or_else(|| lines.get(1).and_then(|s| s.trim().parse().ok()))
            .unwrap_or(0.0);

        let total_frames = nb_frames.unwrap_or_else(|| (duration * fps) as u64);

        Ok(VideoInfo {
            width,
            height,
            fps,
            duration,
            total_frames,
        })
    }

    /// Process a video file with a 3-thread pipeline:
    ///
    /// ```text
    /// Thread 1 (DECODE):  FFmpeg hwaccel decode → pipe read → channel
    /// Thread 2 (GPU):     upload → compute shaders (zero-copy chain) → download
    /// Thread 3 (ENCODE):  channel → pipe write → FFmpeg NVENC encode
    /// ```
    ///
    /// All three stages run in parallel with bounded channels (prefetch).
    /// GPU compute is zero-copy between chained filters in VRAM.
    /// FFmpeg uses hardware acceleration (NVDEC/D3D11VA) when available.
    pub fn run_video_pipeline(
        &self,
        ffmpeg: &Path,
        input: &Path,
        output: &Path,
        filters: &[String],
        max_frames: Option<u64>,
        start_time: Option<f64>,
        output_codec: &str,
        crf: u32,
    ) -> NeoResult<VideoStats> {
        let video_info = Self::probe_video(ffmpeg, input)?;

        info!(
            width = video_info.width,
            height = video_info.height,
            fps = video_info.fps,
            frames = video_info.total_frames,
            "Video probed"
        );

        let frame_limit = max_frames.unwrap_or(video_info.total_frames);

        // Compute output dimensions (upscale filters change size)
        let (mut out_w, mut out_h) = (video_info.width, video_info.height);
        for f in filters {
            if f == "upscale-2x" || f == "upscale" {
                out_w *= 2;
                out_h *= 2;
            }
        }

        let in_w = video_info.width;
        let in_h = video_info.height;
        let fps = video_info.fps;
        let frame_size = (in_w * in_h * 4) as usize; // RGBA

        // ── FFmpeg DECODER with hardware acceleration ──────────────
        let mut decode_args: Vec<String> = Vec::new();

        // Seek before input for fast seeking
        if let Some(ss) = start_time {
            decode_args.extend(["-ss".into(), format!("{ss}")]);
        }

        // Hardware accelerated decode: NVDEC → D3D11VA → auto → fallback CPU
        // The decoded frames still come out as raw RGBA on the pipe,
        // but the actual decode work happens on the GPU's dedicated video engine.
        decode_args.extend([
            "-hwaccel".into(),
            "auto".into(),
            "-i".into(),
            input.to_string_lossy().to_string(),
            "-f".into(),
            "rawvideo".into(),
            "-pix_fmt".into(),
            "rgba".into(),
            "-v".into(),
            "error".into(),
        ]);
        if let Some(n) = max_frames {
            decode_args.extend(["-frames:v".into(), n.to_string()]);
        }
        decode_args.push("-".into());

        let mut decoder = Command::new(ffmpeg)
            .args(&decode_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| NeoError::Decode(format!("failed to start ffmpeg decoder: {e}")))?;

        // ── FFmpeg ENCODER (NVENC when available) ──────────────────
        let fps_str = format!("{fps}");
        let size_str = format!("{out_w}x{out_h}");

        let mut encode_args = vec![
            "-f".to_string(), "rawvideo".into(),
            "-pix_fmt".into(), "rgba".into(),
            "-s".into(), size_str.clone(),
            "-r".into(), fps_str.clone(),
            "-i".into(), "-".into(),
            "-v".into(), "error".into(),
        ];

        // Choose encoder — prefer NVENC (hardware) for GPU-heavy pipeline
        match output_codec {
            "h264" => {
                encode_args.extend(["-c:v".into(), "libx264".into(), "-preset".into(), "fast".into()]);
            }
            "h265" | "hevc" => {
                encode_args.extend(["-c:v".into(), "libx265".into(), "-preset".into(), "fast".into()]);
            }
            "av1" => {
                encode_args.extend(["-c:v".into(), "libsvtav1".into(), "-preset".into(), "6".into()]);
            }
            "nvenc" | "h264_nvenc" => {
                encode_args.extend(["-c:v".into(), "h264_nvenc".into(), "-preset".into(), "p4".into()]);
            }
            "hevc_nvenc" => {
                encode_args.extend(["-c:v".into(), "hevc_nvenc".into(), "-preset".into(), "p4".into()]);
            }
            _ => {
                encode_args.extend(["-c:v".into(), "libx264".into(), "-preset".into(), "fast".into()]);
            }
        }

        if !output_codec.contains("nvenc") {
            encode_args.extend(["-crf".into(), crf.to_string()]);
        }
        encode_args.extend(["-y".into(), output.to_string_lossy().to_string()]);

        let mut encoder = Command::new(ffmpeg)
            .args(&encode_args)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| NeoError::Encode(format!("failed to start ffmpeg encoder: {e}")))?;

        // Take ownership of pipes for threads
        let decoder_stdout = decoder.stdout.take().unwrap();
        let encoder_stdin = encoder.stdin.take().unwrap();

        // ── BOUNDED CHANNELS: 8 frames of prefetch buffer ─────────
        // This lets the decode thread stay ahead of GPU processing,
        // so the GPU never starves waiting for the next frame.
        let (decode_tx, decode_rx) = sync_mpsc::sync_channel::<Vec<u8>>(8);
        let (encode_tx, encode_rx) = sync_mpsc::sync_channel::<Vec<u8>>(8);

        // Shared atomic counters for progress
        let frames_decoded = Arc::new(AtomicU64::new(0));
        let frames_processed = Arc::new(AtomicU64::new(0));
        let frames_encoded = Arc::new(AtomicU64::new(0));
        let decode_error = Arc::new(AtomicBool::new(false));
        let encode_error = Arc::new(AtomicBool::new(false));

        let pipeline_start = Instant::now();

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // THREAD 1: DECODE — reads raw frames from FFmpeg pipe
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let decode_thread = {
            let frames_decoded = frames_decoded.clone();
            let decode_error = decode_error.clone();
            let fl = frame_limit;
            let fs = frame_size;
            thread::Builder::new()
                .name("neo-decode".into())
                .spawn(move || {
                    let mut stdout = decoder_stdout;
                    let mut buf = vec![0u8; fs];
                    let mut count = 0u64;
                    loop {
                        match stdout.read_exact(&mut buf) {
                            Ok(()) => {
                                if decode_tx.send(buf.clone()).is_err() {
                                    break; // GPU thread dropped rx
                                }
                                count += 1;
                                frames_decoded.store(count, Ordering::Relaxed);
                                if count >= fl {
                                    break;
                                }
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                            Err(_) => {
                                decode_error.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                })
                .map_err(|e| NeoError::Pipeline(format!("spawn decode thread: {e}")))?
        };

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // THREAD 3: ENCODE — writes processed frames to FFmpeg pipe
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let encode_thread = {
            let frames_encoded = frames_encoded.clone();
            let encode_error = encode_error.clone();
            thread::Builder::new()
                .name("neo-encode".into())
                .spawn(move || {
                    let mut stdin = encoder_stdin;
                    let mut count = 0u64;
                    while let Ok(data) = encode_rx.recv() {
                        if stdin.write_all(&data).is_err() {
                            encode_error.store(true, Ordering::Relaxed);
                            break;
                        }
                        count += 1;
                        frames_encoded.store(count, Ordering::Relaxed);
                    }
                    drop(stdin); // close pipe → FFmpeg finishes encoding
                })
                .map_err(|e| NeoError::Pipeline(format!("spawn encode thread: {e}")))?
        };

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // THREAD 2 (this thread): GPU — upload → compute → download
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        //
        // Optimizations vs naive loop:
        //  - Pre-allocated staging buffers (no per-frame 28MB allocation)
        //  - Double-buffered download: while CPU reads staging A, GPU copies
        //    next processed frame into staging B (ping-pong)
        //  - No explicit poll() after compute — read_staging() polls once,
        //    waiting for both compute and copy in a single round-trip
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        let out_frame_size = (out_w * out_h * 4) as u64;

        // Two persistent staging buffers for ping-pong download
        let staging_a = VramBuffer::create_staging(&self.ctx, out_frame_size, "staging-a");
        let staging_b = VramBuffer::create_staging(&self.ctx, out_frame_size, "staging-b");
        let stagings = [&staging_a, &staging_b];
        let mut staging_idx = 0usize;

        // Pending download state: which staging is loaded with frame N-1
        let mut pending: Option<usize> = None;

        let mut total_gpu_us: u128 = 0;
        let mut total_upload_us: u128 = 0;
        let mut total_download_us: u128 = 0;
        let mut process_count: u64 = 0;

        loop {
            let raw_frame = match decode_rx.recv() {
                Ok(f) => f,
                Err(_) => break, // decoder finished
            };

            // Upload CPU → VRAM
            let upload_start = Instant::now();
            let frame = self.upload_raw_frame(&raw_frame, in_w, in_h)?;
            total_upload_us += upload_start.elapsed().as_micros();

            // GPU compute: apply all filters in VRAM (zero-copy between filters!)
            let gpu_start = Instant::now();
            let mut processed = frame;
            for filter_name in filters {
                processed = self.apply_filter(&processed, filter_name)?;
            }

            // Submit copy compute → staging (non-blocking, queued behind compute)
            let cur_staging = stagings[staging_idx];
            processed.buffer.copy_to_staging(&self.ctx, cur_staging);
            total_gpu_us += gpu_start.elapsed().as_micros();

            // While the GPU works on this frame's compute+copy, read the
            // PREVIOUS frame's staging buffer (ping-pong: A↔B)
            if let Some(prev_idx) = pending {
                let dl_start = Instant::now();
                let prev_data = VramBuffer::read_staging(
                    &self.ctx,
                    stagings[prev_idx],
                    out_frame_size,
                )?;
                total_download_us += dl_start.elapsed().as_micros();

                if encode_tx.send(prev_data).is_err() {
                    break;
                }

                process_count += 1;
                frames_processed.store(process_count, Ordering::Relaxed);
            }

            // Mark current staging as pending for next iteration
            pending = Some(staging_idx);
            staging_idx = 1 - staging_idx;

            // Progress report every 10 frames
            if process_count > 0 && process_count % 10 == 0 {
                let elapsed = pipeline_start.elapsed().as_secs_f64();
                let eff_fps = process_count as f64 / elapsed;
                let decoded = frames_decoded.load(Ordering::Relaxed);
                let encoded = frames_encoded.load(Ordering::Relaxed);
                eprint!(
                    "\r  [{}/{}] D:{} G:{} E:{} | {:.1} FPS | GPU:{:.1}ms up:{:.1}ms dl:{:.1}ms   ",
                    process_count,
                    frame_limit,
                    decoded,
                    process_count,
                    encoded,
                    eff_fps,
                    total_gpu_us as f64 / process_count as f64 / 1000.0,
                    total_upload_us as f64 / process_count as f64 / 1000.0,
                    total_download_us as f64 / process_count as f64 / 1000.0,
                );
            }

            if process_count >= frame_limit {
                break;
            }
        }

        // Drain the final pending download (frame still in staging buffer)
        if let Some(prev_idx) = pending {
            let dl_start = Instant::now();
            let prev_data = VramBuffer::read_staging(
                &self.ctx,
                stagings[prev_idx],
                out_frame_size,
            )?;
            total_download_us += dl_start.elapsed().as_micros();

            let _ = encode_tx.send(prev_data);
            process_count += 1;
            frames_processed.store(process_count, Ordering::Relaxed);
        }
        eprintln!();

        // ── CLEANUP: join threads, wait for FFmpeg processes ──────
        drop(encode_tx); // close channel → encode thread finishes
        let _ = decode_thread.join();
        let _ = encode_thread.join();
        let _ = decoder.wait();
        let enc_result = encoder.wait();

        if let Ok(status) = enc_result {
            if !status.success() {
                let mut err_out = String::new();
                if let Some(mut stderr) = encoder.stderr.take() {
                    let _ = stderr.read_to_string(&mut err_out);
                }
                return Err(NeoError::Encode(format!(
                    "ffmpeg encoder failed: {err_out}"
                )));
            }
        }

        if decode_error.load(Ordering::Relaxed) {
            return Err(NeoError::Decode("decode thread encountered an error".into()));
        }
        if encode_error.load(Ordering::Relaxed) {
            return Err(NeoError::Encode("encode thread encountered an error".into()));
        }

        let total_time = pipeline_start.elapsed();

        Ok(VideoStats {
            input_width: in_w,
            input_height: in_h,
            output_width: out_w,
            output_height: out_h,
            fps_in: fps,
            frames_processed: process_count,
            filters_applied: filters.len(),
            total_gpu_us: total_gpu_us as u64,
            avg_gpu_per_frame_us: if process_count > 0 {
                total_gpu_us as u64 / process_count
            } else {
                0
            },
            total_time_ms: total_time.as_millis() as u64,
            effective_fps: if total_time.as_secs_f64() > 0.0 {
                process_count as f64 / total_time.as_secs_f64()
            } else {
                0.0
            },
            total_upload_us: total_upload_us as u64,
            total_download_us: total_download_us as u64,
        })
    }

    /// Run a full pipeline: load → apply filters in sequence → save.
    pub fn run_pipeline(
        &self,
        input_path: &Path,
        output_path: &Path,
        filters: &[String],
    ) -> NeoResult<PipelineStats> {
        let total_start = Instant::now();

        // Load
        let load_start = Instant::now();
        let mut frame = self.load_image(input_path)?;
        let load_time = load_start.elapsed();

        let input_width = frame.width;
        let input_height = frame.height;

        // Apply filters in sequence (all in VRAM — zero-copy between filters)
        let process_start = Instant::now();
        for filter_name in filters {
            frame = self.apply_filter(&frame, filter_name)?;
        }
        let process_time = process_start.elapsed();

        // Save
        let save_start = Instant::now();
        self.save_image(&frame, output_path)?;
        let save_time = save_start.elapsed();

        let total_time = total_start.elapsed();

        Ok(PipelineStats {
            input_width,
            input_height,
            output_width: frame.width,
            output_height: frame.height,
            filters_applied: filters.len(),
            load_time_ms: load_time.as_millis() as u64,
            process_time_ms: process_time.as_millis() as u64,
            save_time_ms: save_time.as_millis() as u64,
            total_time_ms: total_time.as_millis() as u64,
        })
    }
}

/// Video file metadata from probing.
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration: f64,
    pub total_frames: u64,
}

/// Statistics from a video pipeline run.
pub struct VideoStats {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub fps_in: f64,
    pub frames_processed: u64,
    pub filters_applied: usize,
    pub total_gpu_us: u64,
    pub avg_gpu_per_frame_us: u64,
    pub total_time_ms: u64,
    pub effective_fps: f64,
    pub total_upload_us: u64,
    pub total_download_us: u64,
}

/// Statistics from a pipeline run.
pub struct PipelineStats {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub filters_applied: usize,
    pub load_time_ms: u64,
    pub process_time_ms: u64,
    pub save_time_ms: u64,
    pub total_time_ms: u64,
}
