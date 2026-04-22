//! Self-test: encode a synthetic gradient to H.264 via NVENC.
//!
//! Validates the entire dynamic-loading + safe-wrapper stack end-to-end.
//! Writes a raw .h264 Annex-B stream that can be played by `ffplay`,
//! VLC, or muxed into an MP4 with `ffmpeg -i out.h264 -c copy out.mp4`.

use crate::cuda::CudaRuntime;
use neo_core::{NeoError, NeoResult};
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB, NV_ENC_CODEC_H264_GUID,
    },
    Encoder, EncoderInitParams, ErrorKind,
};
use std::{
    fs::File,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};
use tracing::{debug, info};

/// Result of an end-to-end NVENC self-test.
#[derive(Debug)]
pub struct EncodeTestResult {
    pub frames: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_written: usize,
    pub elapsed: Duration,
}

impl EncodeTestResult {
    pub fn fps(&self) -> f64 {
        self.frames as f64 / self.elapsed.as_secs_f64()
    }
}

/// Generate `frames` of a moving gradient and encode them to H.264 via NVENC.
///
/// Output is raw Annex-B (`.h264`) — playable directly with `ffplay`.
pub fn run(
    runtime: &CudaRuntime,
    out_path: &Path,
    width: u32,
    height: u32,
    frames: u32,
    framerate: u32,
) -> NeoResult<EncodeTestResult> {
    info!(
        path = %out_path.display(),
        width, height, frames, framerate,
        "starting NVENC self-test"
    );

    // 1. Open NVENC session bound to our CUDA context.
    let encoder = Encoder::initialize_with_cuda(runtime.ctx.clone())
        .map_err(|e| NeoError::HwAccelUnavailable(format!("NVENC init: {e:?}")))?;

    // 2. Configure for H.264 at the requested resolution + framerate.
    let mut init = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, width, height);
    init.enable_picture_type_decision().framerate(framerate, 1);

    let session = encoder
        .start_session(NV_ENC_BUFFER_FORMAT_ARGB, init)
        .map_err(|e| NeoError::Encode(format!("start_session: {e:?}")))?;

    // 3. Allocate one input buffer + one output bitstream. We process
    //    synchronously: fill -> encode -> drain -> repeat. NVENC can return
    //    NeedMoreInput on the first few frames, which is fine since we're
    //    not advancing buffers until we get bytes back via end_of_stream.
    let mut input = session
        .create_input_buffer()
        .map_err(|e| NeoError::Encode(format!("create_input_buffer: {e:?}")))?;
    let mut bitstream = session
        .create_output_bitstream()
        .map_err(|e| NeoError::Encode(format!("create_output_bitstream: {e:?}")))?;

    let mut out_file =
        File::create(out_path).map_err(|e| NeoError::Encode(format!("create file: {e}")))?;

    let frame_size = (width as usize) * (height as usize) * 4;
    let mut frame_buf = vec![0u8; frame_size];

    let start = Instant::now();
    let mut bytes_written = 0usize;

    for i in 0..frames {
        // Synthesize an ARGB gradient that animates with frame index.
        // ARGB layout per NVENC: byte order is B, G, R, A.
        let phase = i as f32 / frames.max(1) as f32;
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) as usize) * 4;
                let r = ((x as f32 / width as f32 + phase) * 255.0) as u8;
                let g = ((y as f32 / height as f32) * 255.0) as u8;
                let b = ((1.0 - phase) * 255.0) as u8;
                frame_buf[idx] = b;
                frame_buf[idx + 1] = g;
                frame_buf[idx + 2] = r;
                frame_buf[idx + 3] = 255;
            }
        }

        // Lock input buffer, copy in our frame, unlock.
        {
            let mut lock = input
                .lock()
                .map_err(|e| NeoError::Encode(format!("input.lock frame {i}: {e:?}")))?;
            unsafe { lock.write(&frame_buf) };
        }

        // Encode this picture. NeedMoreInput means NVENC has buffered the
        // frame but not yet emitted bytes — that's expected for the first
        // few frames depending on lookahead/B-frame settings.
        match session.encode_picture(&mut input, &mut bitstream, Default::default()) {
            Ok(()) => {
                let lock = bitstream
                    .lock()
                    .map_err(|e| NeoError::Encode(format!("bitstream.lock frame {i}: {e:?}")))?;
                let data = lock.data();
                debug!(frame = i, len = data.len(), "encoded picture");
                out_file
                    .write_all(data)
                    .map_err(|e| NeoError::Encode(format!("write frame {i}: {e}")))?;
                bytes_written += data.len();
            }
            Err(e) if e.kind() == ErrorKind::NeedMoreInput => {
                debug!(frame = i, "NVENC buffered (need more input)");
            }
            Err(e) => {
                return Err(NeoError::Encode(format!("encode_picture {i}: {e:?}")));
            }
        }
    }

    // Flush: send EOS and drain any remaining bitstream data.
    session
        .end_of_stream()
        .map_err(|e| NeoError::Encode(format!("end_of_stream: {e:?}")))?;

    // After EOS, lock the output one more time to drain.
    if let Ok(lock) = bitstream.lock() {
        let data = lock.data();
        if !data.is_empty() {
            debug!(len = data.len(), "drained tail bitstream");
            out_file
                .write_all(data)
                .map_err(|e| NeoError::Encode(format!("write tail: {e}")))?;
            bytes_written += data.len();
        }
    }

    let elapsed = start.elapsed();
    info!(bytes_written, elapsed_ms = elapsed.as_millis(), "NVENC self-test complete");

    Ok(EncodeTestResult {
        frames,
        width,
        height,
        bytes_written,
        elapsed,
    })
}
