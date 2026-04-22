//! Stage 4: end-to-end NVDEC → CPU bounce → NVENC transcoder.
//!
//! This is the first full round-trip through the native NVIDIA stack:
//!
//! 1. `nvcuvid.dll` decodes H.264 into NV12 surfaces in VRAM.
//! 2. Each surface is mapped + copied into tightly packed CPU NV12 via
//!    `cuMemcpy2D`. This is the "CPU bounce" — replaced in Stage 6 with
//!    CUDA↔Vulkan external memory interop for true zero-copy.
//! 3. NV12 is converted to ARGB on the CPU (BT.709 limited-range).
//! 4. `nvEncodeAPI64.dll` encodes ARGB → H.264 Annex-B.
//!
//! A wgpu compute filter can be inserted between steps (3) and (4) in a
//! follow-up; the plumbing is ready because the CPU-resident ARGB buffer
//! maps cleanly to a `wgpu::Texture`.
//!
//! This module exists purely so we can exercise the whole pipeline with a
//! single CLI command and prove the codec path is correct before
//! introducing the GPU filter graph.

use crate::{
    cuda::CudaRuntime,
    nvdec::{decode_bytes_capture, DecodedFrame},
    wgpu_convert::Nv12ToBgraConverter,
};
use neo_core::{NeoError, NeoResult};
use neo_gpu::{GpuContext, GpuOptions};
use std::sync::Arc;
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

/// Summary of an end-to-end transcode run.
#[derive(Debug, Clone)]
pub struct TranscodeStats {
    pub width: u32,
    pub height: u32,
    pub frames_decoded: u32,
    pub frames_encoded: u32,
    pub bytes_written: usize,
    pub decode_elapsed: Duration,
    pub convert_elapsed: Duration,
    pub encode_elapsed: Duration,
    pub total_elapsed: Duration,
}

impl TranscodeStats {
    pub fn encode_fps(&self) -> f64 {
        self.frames_encoded as f64 / self.encode_elapsed.as_secs_f64().max(1e-9)
    }
    pub fn total_fps(&self) -> f64 {
        self.frames_encoded as f64 / self.total_elapsed.as_secs_f64().max(1e-9)
    }
}

/// Backend selector for the NV12 → BGRA conversion stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertBackend {
    /// Scalar CPU loop — simple, slow, no GPU dependency. Useful as a
    /// reference / fallback.
    Cpu,
    /// wgpu compute shader — runs on any GPU backend (Vulkan, DX12, Metal,
    /// GL). Uploads Y + UV, dispatches one thread per output pixel,
    /// downloads BGRA. Eliminates the scalar bottleneck and is the
    /// path that a future filter chain will hook into.
    WgpuCompute,
}

/// Decode `input_bytes` (H.264 Annex-B), bounce through CPU as NV12,
/// convert to BGRA on GPU (or CPU, depending on `backend`), and re-encode
/// to `out_path` as H.264 via NVENC.
pub fn transcode_h264(
    runtime: &CudaRuntime,
    input_bytes: &[u8],
    out_path: &Path,
    framerate: u32,
    backend: ConvertBackend,
) -> NeoResult<TranscodeStats> {
    let total_start = Instant::now();

    // ------- 1. Decode (NVDEC) ----------------------------------------------
    let decode_start = Instant::now();
    let (decode_stats, frames) = decode_bytes_capture(
        runtime,
        nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_H264,
        input_bytes,
    )?;
    let decode_elapsed = decode_start.elapsed();

    if frames.is_empty() {
        return Err(NeoError::Decode("NVDEC produced 0 frames".into()));
    }
    let width = frames[0].width;
    let height = frames[0].height;
    info!(
        width, height,
        decoded = decode_stats.pictures_decoded,
        captured = frames.len(),
        "decoded frames captured"
    );

    // ------- 2. Set up NVENC ------------------------------------------------
    //
    // NVENC has internal lookahead / B-frame buffering, so a single
    // input+bitstream pair causes encode_picture to return NeedMoreInput
    // repeatedly and we'd lose most frames at flush time. The documented
    // pattern is to allocate a pool of matching input and bitstream
    // buffers, feed each frame into the next free slot, and once
    // `encode_picture` finally returns Ok, drain ALL accumulated pending
    // slots in order. See `Session::encode_picture` docs in the vendored
    // crate for the contract.
    const POOL: usize = 16;

    let encoder = Encoder::initialize_with_cuda(runtime.ctx.clone())
        .map_err(|e| NeoError::HwAccelUnavailable(format!("NVENC init: {e:?}")))?;

    let mut init = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, width, height);
    init.enable_picture_type_decision().framerate(framerate, 1);

    let session = encoder
        .start_session(NV_ENC_BUFFER_FORMAT_ARGB, init)
        .map_err(|e| NeoError::Encode(format!("start_session: {e:?}")))?;

    let mut inputs = Vec::with_capacity(POOL);
    let mut bitstreams = Vec::with_capacity(POOL);
    for _ in 0..POOL {
        inputs.push(
            session
                .create_input_buffer()
                .map_err(|e| NeoError::Encode(format!("create_input_buffer: {e:?}")))?,
        );
        bitstreams.push(
            session
                .create_output_bitstream()
                .map_err(|e| NeoError::Encode(format!("create_output_bitstream: {e:?}")))?,
        );
    }

    let mut out_file =
        File::create(out_path).map_err(|e| NeoError::Encode(format!("create file: {e}")))?;

    let argb_size = (width as usize) * (height as usize) * 4;
    let mut argb_buf = vec![0u8; argb_size];

    // Bring up the wgpu converter if requested. The GpuContext is created
    // here lazily so the CPU path has no wgpu dependency at runtime.
    let wgpu_converter = match backend {
        ConvertBackend::WgpuCompute => {
            let gpu_ctx =
                Arc::new(GpuContext::new_sync(&GpuOptions::default()).map_err(|e| {
                    NeoError::HwAccelUnavailable(format!("wgpu init: {e}"))
                })?);
            info!(gpu = %gpu_ctx.gpu_name(), backend = ?gpu_ctx.backend(), "wgpu converter selected");
            Some(Nv12ToBgraConverter::new(gpu_ctx, width, height)?)
        }
        ConvertBackend::Cpu => None,
    };

    let mut convert_elapsed = Duration::ZERO;
    let mut encode_elapsed = Duration::ZERO;
    let mut bytes_written = 0usize;
    let mut frames_encoded = 0u32;

    // Queue of bitstream indices that have been submitted but not yet
    // drained. FIFO so drain order matches submission order (required).
    let mut pending: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    // Drain macro: pops all entries from `pending`, locks each matching
    // bitstream, writes the encoded data, and bumps counters. Defined as
    // a macro to sidestep lifetime gymnastics with a closure capturing
    // `&mut bitstreams` — the enclosed items have tangled lifetimes and
    // closures demand a single fixed signature.
    macro_rules! drain {
        () => {{
            while let Some(i) = pending.pop_front() {
                let lock = bitstreams[i]
                    .lock()
                    .map_err(|e| NeoError::Encode(format!("bitstream.lock pool[{i}]: {e:?}")))?;
                let data = lock.data();
                if !data.is_empty() {
                    out_file
                        .write_all(data)
                        .map_err(|e| NeoError::Encode(format!("write pool[{i}]: {e}")))?;
                    bytes_written += data.len();
                    frames_encoded += 1;
                }
            }
        }};
    }

    // ------- 3. Per-frame: convert + encode --------------------------------
    let mut slot = 0usize;
    for (idx, frame) in frames.iter().enumerate() {
        let t0 = Instant::now();
        match &wgpu_converter {
            Some(c) => c.convert(frame, &mut argb_buf)?,
            None => nv12_to_bgra(frame, &mut argb_buf),
        }
        convert_elapsed += t0.elapsed();

        // If we're about to reuse a slot that is still pending, drain
        // everything before writing into its input buffer.
        if pending.contains(&slot) {
            drain!();
        }

        let t1 = Instant::now();
        {
            let mut lock = inputs[slot]
                .lock()
                .map_err(|e| NeoError::Encode(format!("input.lock frame {idx}: {e:?}")))?;
            unsafe { lock.write(&argb_buf) };
        }

        let result = session.encode_picture(
            &mut inputs[slot],
            &mut bitstreams[slot],
            Default::default(),
        );
        pending.push_back(slot);
        match result {
            Ok(()) => {
                debug!(frame = idx, "NVENC ready — draining pending slots");
                drain!();
            }
            Err(e) if e.kind() == ErrorKind::NeedMoreInput => {
                debug!(frame = idx, "NVENC buffered (pool in flight)");
            }
            Err(e) => {
                return Err(NeoError::Encode(format!("encode_picture {idx}: {e:?}")));
            }
        }
        encode_elapsed += t1.elapsed();

        slot = (slot + 1) % POOL;
    }

    // ------- 4. Flush -------------------------------------------------------
    session
        .end_of_stream()
        .map_err(|e| NeoError::Encode(format!("end_of_stream: {e:?}")))?;
    drain!();

    let total_elapsed = total_start.elapsed();
    info!(
        frames_encoded, bytes_written,
        total_ms = total_elapsed.as_millis(),
        "transcode complete"
    );

    Ok(TranscodeStats {
        width,
        height,
        frames_decoded: decode_stats.pictures_decoded,
        frames_encoded,
        bytes_written,
        decode_elapsed,
        convert_elapsed,
        encode_elapsed,
        total_elapsed,
    })
}

/// BT.709 limited-range NV12 → BGRA conversion (NVENC's `ARGB` buffer format
/// is byte order B, G, R, A — hence the name of this function).
///
/// This is a straightforward scalar implementation; SIMD / CUDA / wgpu
/// compute replacements are drop-in in later stages.
fn nv12_to_bgra(frame: &DecodedFrame, out_bgra: &mut [u8]) {
    let w = frame.width as usize;
    let h = frame.height as usize;
    debug_assert_eq!(out_bgra.len(), w * h * 4);

    // BT.709 limited-range coefficients, pre-scaled.
    // R = 1.1644*(Y-16) + 1.7927*(V-128)
    // G = 1.1644*(Y-16) - 0.5329*(V-128) - 0.2132*(U-128)
    // B = 1.1644*(Y-16) + 2.1124*(U-128)
    for y in 0..h {
        let y_row = &frame.y[y * w..y * w + w];
        let uv_row = &frame.uv[(y / 2) * w..(y / 2) * w + w];
        let out_row = &mut out_bgra[y * w * 4..y * w * 4 + w * 4];
        for x in 0..w {
            let yv = y_row[x] as f32 - 16.0;
            let u = uv_row[x & !1] as f32 - 128.0;
            let v = uv_row[(x & !1) + 1] as f32 - 128.0;
            let c = 1.1644 * yv;
            let r = (c + 1.7927 * v).clamp(0.0, 255.0) as u8;
            let g = (c - 0.5329 * v - 0.2132 * u).clamp(0.0, 255.0) as u8;
            let b = (c + 2.1124 * u).clamp(0.0, 255.0) as u8;
            let i = x * 4;
            out_row[i] = b;
            out_row[i + 1] = g;
            out_row[i + 2] = r;
            out_row[i + 3] = 255;
        }
    }
}

// ===========================================================================
// Stage 6b: zero-copy VRAM transcoder
// ===========================================================================
//
// This is the endgame. Every frame stays in VRAM from NVDEC output to NVENC
// input — the only CPU traffic is the compressed H.264 bitstream going to
// disk, which is unavoidable. The data flow is:
//
//   nvcuvid.dll decode
//       └─ mapped NVDEC surface (CUDA device ptr, NV12)
//            └─ cuMemcpy2D DtoD ──▶ interop Y buffer (CUDA dptr ≡ wgpu buf)
//            └─ cuMemcpy2D DtoD ──▶ interop UV buffer
//                                 └─ wgpu compute NV12→BGRA
//                                      └─ interop BGRA buffer (CUDA dptr
//                                         ≡ wgpu buf)
//                                           └─ NVENC register_generic_resource
//                                              + encode_picture
//                                                └─ bitstream to disk
//
// No CPU NV12 upload, no BGRA download, no NVENC input-buffer memcpy.

#[cfg(windows)]
pub mod zerocopy {
    use super::*;
    use crate::interop::{create_interop_buffer, InteropBuffer};
    use crate::nvdec::{CaptureMode, DecodeStats, Decoder, DeviceHook};
    use crate::wgpu_convert::Nv12ToBgraConverter;
    use cudarc::driver::sys::{
        self as cuda_sys, CUmemorytype, CUresult, CUDA_MEMCPY2D_st,
    };
    use nvidia_video_codec_sdk::{
        sys::nvEncodeAPI::NV_ENC_INPUT_RESOURCE_TYPE,
    };
    use std::ffi::c_void;

    /// Runtime state shared between the Rust-side transcoder and the
    /// `unsafe extern "C"` trampoline invoked by NVDEC's display callback.
    ///
    /// Everything is stored via raw pointers because NVENC's
    /// `RegisteredResource<'s, T>` carries a lifetime that infects the
    /// struct otherwise, and the trampoline already needs unsafe anyway.
    /// All pointers originate from locals in
    /// [`transcode_h264_zerocopy`] whose scope strictly outlives the
    /// decoder's `feed`/`flush` calls.
    struct FrameProcessor {
        width: u32,
        height: u32,
        y_dptr: u64,
        uv_dptr: u64,
        converter: *const Nv12ToBgraConverter,
        session: *const nvidia_video_codec_sdk::Session,
        nvenc_inputs:
            *mut Vec<nvidia_video_codec_sdk::RegisteredResource<'static, ()>>,
        nvenc_bitstreams: *mut Vec<nvidia_video_codec_sdk::Bitstream<'static>>,
        pending: *mut std::collections::VecDeque<usize>,
        slot: *mut usize,
        out_file: *mut File,
        bytes_written: *mut usize,
        frames_encoded: *mut u32,
        convert_elapsed: *mut Duration,
        encode_elapsed: *mut Duration,
    }

    /// C trampoline invoked from `handle_display` with the NVDEC surface's
    /// raw device pointer.
    unsafe extern "C" fn trampoline(
        user: *mut c_void,
        width: u32,
        height: u32,
        src_dptr: u64,
        src_pitch: u32,
    ) -> i32 {
        let proc = &mut *(user as *mut FrameProcessor);
        match process_frame(proc, width, height, src_dptr, src_pitch) {
            Ok(()) => 0,
            Err(e) => {
                tracing::error!("zerocopy frame hook: {e}");
                -1
            }
        }
    }

    unsafe fn process_frame(
        proc: &mut FrameProcessor,
        width: u32,
        height: u32,
        src_dptr: u64,
        src_pitch: u32,
    ) -> NeoResult<()> {
        if width != proc.width || height != proc.height {
            return Err(NeoError::InvalidDimensions { width, height });
        }
        let w = width as usize;
        let h = height as usize;
        let pitch = src_pitch as usize;

        // 1. DtoD copy NVDEC Y plane → interop Y buffer (tightly packed).
        let t_conv = Instant::now();
        memcpy2d_dtod(src_dptr, pitch, proc.y_dptr, w, w, h)?;
        memcpy2d_dtod(
            src_dptr + (pitch as u64) * (h as u64),
            pitch,
            proc.uv_dptr,
            w,
            w,
            h / 2,
        )?;
        let r = cuda_sys::cuCtxSynchronize();
        if r != CUresult::CUDA_SUCCESS {
            return Err(NeoError::Cuda(format!("cuCtxSynchronize DtoD: {r:?}")));
        }

        // 2. wgpu compute.
        (*proc.converter).dispatch_interop()?;
        *proc.convert_elapsed += t_conv.elapsed();

        // 3. NVENC direct encode from the interop BGRA buffer.
        let t_enc = Instant::now();
        let slot_ref = &mut *proc.slot;
        let pending_ref = &mut *proc.pending;
        let inputs_ref = &mut *proc.nvenc_inputs;
        let bitstreams_ref = &mut *proc.nvenc_bitstreams;
        if pending_ref.contains(slot_ref) {
            drain_pending_raw(
                pending_ref,
                bitstreams_ref,
                &mut *proc.out_file,
                &mut *proc.bytes_written,
                &mut *proc.frames_encoded,
            )?;
        }
        let idx = *slot_ref;
        let result = (*proc.session).encode_picture(
            &mut inputs_ref[idx],
            &mut bitstreams_ref[idx],
            Default::default(),
        );
        pending_ref.push_back(idx);
        match result {
            Ok(()) => drain_pending_raw(
                pending_ref,
                bitstreams_ref,
                &mut *proc.out_file,
                &mut *proc.bytes_written,
                &mut *proc.frames_encoded,
            )?,
            Err(e) if e.kind() == ErrorKind::NeedMoreInput => {}
            Err(e) => {
                return Err(NeoError::Encode(format!("encode_picture: {e:?}")));
            }
        }
        *proc.encode_elapsed += t_enc.elapsed();

        *slot_ref = (*slot_ref + 1) % inputs_ref.len();
        Ok(())
    }

    fn drain_pending_raw(
        pending: &mut std::collections::VecDeque<usize>,
        bitstreams: &mut Vec<nvidia_video_codec_sdk::Bitstream<'static>>,
        out_file: &mut File,
        bytes_written: &mut usize,
        frames_encoded: &mut u32,
    ) -> NeoResult<()> {
        while let Some(i) = pending.pop_front() {
            let lock = bitstreams[i]
                .lock()
                .map_err(|e| NeoError::Encode(format!("bitstream.lock slot {i}: {e:?}")))?;
            let data = lock.data();
            if !data.is_empty() {
                out_file
                    .write_all(data)
                    .map_err(|e| NeoError::Encode(format!("write slot {i}: {e}")))?;
                *bytes_written += data.len();
                *frames_encoded += 1;
            }
        }
        Ok(())
    }

    /// Issue a cuMemcpy2D device→device copy with a given source pitch,
    /// writing to a tightly packed (pitch == width) destination.
    unsafe fn memcpy2d_dtod(
        src: u64,
        src_pitch: usize,
        dst: u64,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
    ) -> NeoResult<()> {
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
            return Err(NeoError::Cuda(format!("cuMemcpy2D DtoD: {r:?}")));
        }
        Ok(())
    }

    /// Probe the first sequence header to learn the coded frame size.
    /// Uses early-abort so this is typically <10 ms even at 4K.
    fn probe_dimensions_fast(
        runtime: &CudaRuntime,
        input_bytes: &[u8],
    ) -> NeoResult<(u32, u32, DecodeStats)> {
        let stats = crate::nvdec::probe_dimensions(
            runtime,
            nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_H264,
            input_bytes,
        )?;
        Ok((stats.display_width, stats.display_height, stats))
    }

    /// Stage 6b entry point: decode → wgpu compute → encode, all in VRAM.
    pub fn transcode_h264_zerocopy(
        runtime: &CudaRuntime,
        input_bytes: &[u8],
        out_path: &Path,
        framerate: u32,
    ) -> NeoResult<TranscodeStats> {
        let total_start = Instant::now();

        // --- 1. Probe dimensions ------------------------------------------
        let probe_start = Instant::now();
        let (width, height, probe_stats) = probe_dimensions_fast(runtime, input_bytes)?;
        info!(
            width,
            height,
            probe_ms = probe_start.elapsed().as_millis(),
            "probed dimensions"
        );

        // --- 2. Bring up wgpu with interop features -----------------------
        let gpu = Arc::new(
            GpuContext::new_sync(&GpuOptions::interop()).map_err(|e| {
                NeoError::HwAccelUnavailable(format!("wgpu interop init: {e}"))
            })?,
        );
        info!(gpu = %gpu.gpu_name(), backend = ?gpu.backend(), "wgpu interop context");

        // --- 3. Allocate the three interop buffers ------------------------
        let y_size = (width as u64) * (height as u64);
        let uv_size = y_size / 2;
        let bgra_size = y_size * 4;
        let interop_y = create_interop_buffer(gpu.clone(), runtime, y_size)?;
        let interop_uv = create_interop_buffer(gpu.clone(), runtime, uv_size)?;
        let interop_bgra = create_interop_buffer(gpu.clone(), runtime, bgra_size)?;

        // --- 4. Build the wgpu NV12→BGRA converter bound to interop bufs --
        let converter = Nv12ToBgraConverter::from_external_buffers(
            gpu.clone(),
            width,
            height,
            interop_y.wgpu_buffer(),
            interop_uv.wgpu_buffer(),
            interop_bgra.wgpu_buffer(),
        )?;

        // --- 5. NVENC session: POOL input buffers that each point at the
        //       same interop BGRA VRAM block via register_generic_resource
        // --------------------------------------------------------------------
        const POOL: usize = 16;
        let encoder = Encoder::initialize_with_cuda(runtime.ctx.clone())
            .map_err(|e| NeoError::HwAccelUnavailable(format!("NVENC init: {e:?}")))?;
        let mut init = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, width, height);
        init.enable_picture_type_decision().framerate(framerate, 1);
        let session = encoder
            .start_session(NV_ENC_BUFFER_FORMAT_ARGB, init)
            .map_err(|e| NeoError::Encode(format!("start_session: {e:?}")))?;

        let mut nvenc_inputs = Vec::with_capacity(POOL);
        for _ in 0..POOL {
            // All slots alias the same interop BGRA device pointer. NVENC
            // will happily accept multiple registrations of the same
            // underlying memory; it just tracks them as distinct logical
            // input resources.
            let reg = session
                .register_generic_resource::<()>(
                    (),
                    NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                    interop_bgra.cu_device_ptr as *mut c_void,
                    width * 4, // BGRA pitch in bytes
                )
                .map_err(|e| {
                    NeoError::Encode(format!("register_generic_resource: {e:?}"))
                })?;
            nvenc_inputs.push(reg);
        }
        let mut nvenc_bitstreams = Vec::with_capacity(POOL);
        for _ in 0..POOL {
            nvenc_bitstreams.push(
                session
                    .create_output_bitstream()
                    .map_err(|e| {
                        NeoError::Encode(format!("create_output_bitstream: {e:?}"))
                    })?,
            );
        }

        // --- 6. Open output + set up processor state ---------------------
        let mut out_file = File::create(out_path)
            .map_err(|e| NeoError::Encode(format!("create file: {e}")))?;
        let mut bytes_written = 0usize;
        let mut frames_encoded = 0u32;
        let mut convert_elapsed = Duration::ZERO;
        let mut encode_elapsed = Duration::ZERO;
        let mut pending: std::collections::VecDeque<usize> =
            std::collections::VecDeque::new();
        let mut slot: usize = 0;

        // SAFETY: all raw pointers below point into locals that strictly
        // outlive the decoder's feed/flush calls. The trampoline runs
        // synchronously on this thread, so the borrows are effectively
        // scoped to those calls even though we can't express that in the
        // type system because NVENC's RegisteredResource lifetime
        // parameter would otherwise infect FrameProcessor.
        let converter_ptr: *const Nv12ToBgraConverter = &converter;
        let session_ptr: *const nvidia_video_codec_sdk::Session =
            &session as *const _ as *const _;
        let inputs_ptr: *mut Vec<
            nvidia_video_codec_sdk::RegisteredResource<'static, ()>,
        > = &mut nvenc_inputs as *mut _ as *mut _;
        let bitstreams_ptr: *mut Vec<nvidia_video_codec_sdk::Bitstream<'static>> =
            &mut nvenc_bitstreams as *mut _ as *mut _;
        let mut proc = FrameProcessor {
            width,
            height,
            y_dptr: interop_y.cu_device_ptr,
            uv_dptr: interop_uv.cu_device_ptr,
            converter: converter_ptr,
            session: session_ptr,
            nvenc_inputs: inputs_ptr,
            nvenc_bitstreams: bitstreams_ptr,
            pending: &mut pending,
            slot: &mut slot,
            out_file: &mut out_file,
            bytes_written: &mut bytes_written,
            frames_encoded: &mut frames_encoded,
            convert_elapsed: &mut convert_elapsed,
            encode_elapsed: &mut encode_elapsed,
        };

        // --- 7. Run the decoder with our device hook --------------------
        let decode_start = Instant::now();
        let hook = DeviceHook {
            callback: trampoline,
            user: &mut proc as *mut FrameProcessor as *mut c_void,
        };
        {
            let mut dec = Decoder::new(
                runtime,
                nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_H264,
                CaptureMode::Device(hook),
            )?;
            const CHUNK: usize = 1 << 20;
            let mut offset = 0;
            while offset < input_bytes.len() {
                let end = (offset + CHUNK).min(input_bytes.len());
                dec.feed(&input_bytes[offset..end])?;
                offset = end;
            }
            dec.flush()?;
        }
        let decode_elapsed = decode_start.elapsed();

        // Drop `proc` before final flush so the raw pointers it holds
        // don't alias the mutable locals we want to read.
        drop(proc);

        // --- 8. Flush the encoder ----------------------------------------
        session
            .end_of_stream()
            .map_err(|e| NeoError::Encode(format!("end_of_stream: {e:?}")))?;
        while let Some(i) = pending.pop_front() {
            let lock = nvenc_bitstreams[i]
                .lock()
                .map_err(|e| NeoError::Encode(format!("bitstream.lock {i}: {e:?}")))?;
            let data = lock.data();
            if !data.is_empty() {
                out_file
                    .write_all(data)
                    .map_err(|e| NeoError::Encode(format!("write tail {i}: {e}")))?;
                bytes_written += data.len();
                frames_encoded += 1;
            }
        }
        let total_elapsed = total_start.elapsed();
        info!(
            frames_encoded, bytes_written,
            total_ms = total_elapsed.as_millis(),
            "zerocopy transcode complete"
        );

        // In zerocopy mode, every decoded frame is immediately processed,
        // so frames_decoded == frames_encoded; probe_stats may report 0
        // because the early-abort probe doesn't actually decode anything.
        let _ = probe_stats;
        Ok(TranscodeStats {
            width,
            height,
            frames_decoded: frames_encoded,
            frames_encoded,
            bytes_written,
            decode_elapsed,
            convert_elapsed,
            encode_elapsed,
            total_elapsed,
        })
    }
}
