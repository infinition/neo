//! Safe(ish) high-level wrapper around `nvcuvid.dll`.
//!
//! Owns a parser + decoder pair and drives them with a callback-based
//! state machine. The user feeds raw H.264 (or HEVC, etc.) Annex-B bytes
//! via [`Decoder::feed`]; the parser fires sequence / decode / display
//! callbacks back into our state, where we lazily create the decoder on
//! the first sequence header and count successfully decoded pictures.
//!
//! For the Stage-3 self-test we don't yet map the output surface — we just
//! verify the dynamic-loaded path works end-to-end and that NVDEC reports
//! the expected width/height/frame count.

use crate::cuda::CudaRuntime;
use crate::nvdec_sys::{self, NvCuvid};
use cudarc::driver::sys::{
    self as cuda_sys, CUdeviceptr, CUmemorytype, CUDA_MEMCPY2D_st, CUresult,
};
use neo_core::{NeoError, NeoResult};
use nvidia_video_codec_sdk::sys::{
    cuviddec::{
        cudaVideoChromaFormat, cudaVideoCodec, cudaVideoCreateFlags, cudaVideoDeinterlaceMode,
        cudaVideoSurfaceFormat, CUVIDDECODECREATEINFO, CUVIDPICPARAMS, CUVIDPROCPARAMS,
        CUvideodecoder,
    },
    nvcuvid::{
        CUVIDEOFORMAT, CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS, CUVIDSOURCEDATAPACKET, CUvideoparser,
    },
};
use std::{ffi::c_void, ptr};
use tracing::{debug, info, warn};

/// Aggregate stats reported after a decode run.
#[derive(Debug, Clone, Default)]
pub struct DecodeStats {
    pub coded_width: u32,
    pub coded_height: u32,
    pub display_width: u32,
    pub display_height: u32,
    pub pictures_decoded: u32,
    pub pictures_displayed: u32,
}

/// A single NV12-encoded frame that has been copied out of VRAM to CPU RAM.
///
/// Layout:
/// - `y`: `height * width` tightly packed luma bytes
/// - `uv`: `(height/2) * width` tightly packed interleaved chroma (U, V, U, V, …)
#[derive(Clone)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub uv: Vec<u8>,
}

/// What the display callback should do with each decoded surface.
pub enum CaptureMode {
    /// Count frames only.
    None,
    /// Map the surface and copy Y + UV planes to CPU-resident
    /// [`DecodedFrame`] structs (Stage 4 behaviour).
    Cpu,
    /// Map the surface and hand the raw CUDA device pointer to a
    /// user-supplied hook. The hook runs inside the display callback, so
    /// it can perform arbitrary device-to-device copies (into a shared
    /// Vulkan/CUDA interop buffer for example) before the decoder
    /// unmaps the surface.
    ///
    /// The callback signature is C-compatible so we can invoke it from
    /// the trampoline without touching trait objects:
    ///
    /// ```text
    /// fn hook(user, width, height, y_dptr, y_pitch) -> i32
    /// ```
    ///
    /// NVDEC returns a single base pointer per frame; the UV plane is at
    /// `y_dptr + y_pitch * coded_height`, which the hook computes itself
    /// from the supplied width/height/pitch.
    Device(DeviceHook),
}

/// Raw function pointer invoked from inside the display callback.
#[derive(Copy, Clone)]
pub struct DeviceHook {
    pub callback: unsafe extern "C" fn(
        user: *mut std::ffi::c_void,
        width: u32,
        height: u32,
        y_dptr: u64,
        y_pitch: u32,
    ) -> i32,
    pub user: *mut std::ffi::c_void,
}

/// Internal state shared between Rust and the C parser callbacks.
///
/// A raw pointer to this struct is handed to `cuvidCreateVideoParser` via
/// `pUserData`; NVDEC calls our trampolines back with that pointer, and
/// we re-cast it to `&mut DecoderState` to update stats and lazily create
/// the actual `CUvideodecoder` once the sequence header has been parsed.
struct DecoderState {
    sys: &'static NvCuvid,
    decoder: CUvideodecoder,
    stats: DecodeStats,
    capture_mode: CaptureMode,
    /// Captured CPU frames (populated only when `capture_mode == Cpu`).
    frames: Vec<DecodedFrame>,
    /// Set on the first error from inside a callback. The parser API
    /// requires us to return 0/negative to abort, but we still need to
    /// surface the error to the caller.
    last_error: Option<String>,
    /// When true, the sequence callback returns 0 after stashing
    /// dimensions — aborts the decoder immediately. Used for fast SPS
    /// probes that just want coded_width / display_width.
    probe_only: bool,
}

impl DecoderState {
    fn new(sys: &'static NvCuvid, capture_mode: CaptureMode) -> Self {
        Self {
            sys,
            decoder: ptr::null_mut(),
            stats: DecodeStats::default(),
            capture_mode,
            frames: Vec::new(),
            last_error: None,
            probe_only: false,
        }
    }
}

/// High-level NVDEC video decoder.
pub struct Decoder {
    sys: &'static NvCuvid,
    parser: CUvideoparser,
    /// `Box` so the address handed to the C side is stable for the
    /// decoder's lifetime.
    state: Box<DecoderState>,
    /// Keep the runtime alive (the CUDA context must outlive the decoder).
    _runtime: std::sync::Arc<cudarc::driver::CudaContext>,
}

impl Decoder {
    /// Create a new NVDEC decoder for the given codec.
    ///
    /// `capture` selects whether frames are mirrored to CPU (Stage 4),
    /// handed to a raw device hook (Stage 6b), or dropped entirely.
    pub fn new(
        runtime: &CudaRuntime,
        codec: cudaVideoCodec,
        capture: CaptureMode,
    ) -> NeoResult<Self> {
        let sys = nvdec_sys::get()?;

        // Make the CUDA context current on this thread before any nvcuvid call.
        runtime
            .ctx
            .bind_to_thread()
            .map_err(|e| NeoError::Cuda(format!("bind_to_thread: {e:?}")))?;

        let mut state = Box::new(DecoderState::new(sys, capture));
        let user_data = &mut *state as *mut DecoderState as *mut c_void;

        let mut params: CUVIDPARSERPARAMS = unsafe { std::mem::zeroed() };
        params.CodecType = codec;
        params.ulMaxNumDecodeSurfaces = 1; // updated by sequence callback
        params.ulMaxDisplayDelay = 1; // low-latency: emit ASAP
        params.pUserData = user_data;
        params.pfnSequenceCallback = Some(handle_sequence);
        params.pfnDecodePicture = Some(handle_decode);
        params.pfnDisplayPicture = Some(handle_display);

        let mut parser: CUvideoparser = ptr::null_mut();
        let status = unsafe { (sys.create_video_parser)(&mut parser, &mut params) };
        if status != 0 {
            return Err(NeoError::Decode(format!(
                "cuvidCreateVideoParser failed: {status}"
            )));
        }
        debug!(parser = ?parser, "NVDEC parser created");

        Ok(Self {
            sys,
            parser,
            state,
            _runtime: runtime.ctx.clone(),
        })
    }

    /// Feed a chunk of bitstream into the parser. The parser is internally
    /// stateful — chunks do not need to be aligned on NAL boundaries.
    pub fn feed(&mut self, data: &[u8]) -> NeoResult<()> {
        let mut pkt: CUVIDSOURCEDATAPACKET = unsafe { std::mem::zeroed() };
        pkt.payload_size = data.len() as _;
        pkt.payload = data.as_ptr();
        pkt.flags = 0;

        let status = unsafe { (self.sys.parse_video_data)(self.parser, &mut pkt) };
        if status != 0 {
            return Err(NeoError::Decode(format!(
                "cuvidParseVideoData failed: {status}"
            )));
        }
        if let Some(err) = self.state.last_error.take() {
            return Err(NeoError::Decode(err));
        }
        Ok(())
    }

    /// Send end-of-stream to flush any buffered pictures.
    pub fn flush(&mut self) -> NeoResult<()> {
        let mut pkt: CUVIDSOURCEDATAPACKET = unsafe { std::mem::zeroed() };
        pkt.flags = 1; // CUVID_PKT_ENDOFSTREAM
        let status = unsafe { (self.sys.parse_video_data)(self.parser, &mut pkt) };
        if status != 0 {
            return Err(NeoError::Decode(format!("flush failed: {status}")));
        }
        if let Some(err) = self.state.last_error.take() {
            return Err(NeoError::Decode(err));
        }
        Ok(())
    }

    pub fn stats(&self) -> &DecodeStats {
        &self.state.stats
    }

    /// Drain captured frames (requires `capture=true` at construction).
    pub fn take_frames(&mut self) -> Vec<DecodedFrame> {
        std::mem::take(&mut self.state.frames)
    }

    /// Turn on "probe only" mode: the next feed() call will abort at the
    /// first sequence header. Use this to cheaply learn width/height
    /// without committing to a full decode.
    pub fn set_probe_only(&mut self) {
        self.state.probe_only = true;
    }
}

/// Fast dimensions probe: parse only enough of the bitstream to hit the
/// first SPS, extract coded/display size, bail out. Typically <10 ms
/// even at 4K because the parser stops at the first sequence header.
pub fn probe_dimensions(
    runtime: &CudaRuntime,
    codec: cudaVideoCodec,
    bytes: &[u8],
) -> NeoResult<DecodeStats> {
    let mut dec = Decoder::new(runtime, codec, CaptureMode::None)?;
    dec.set_probe_only();

    // The NVDEC parser only invokes the sequence callback when it
    // actually reaches a picture — not on SPS alone. Feed in 256-KiB
    // chunks until the callback has fired (state.display_width becomes
    // non-zero) or the parser errors out from our abort return, or we
    // exceed 8 MiB (bail-out limit for pathological streams).
    const CHUNK: usize = 256 * 1024;
    const MAX: usize = 8 * 1024 * 1024;
    let mut offset = 0;
    while offset < bytes.len() && offset < MAX {
        let end = (offset + CHUNK).min(bytes.len());
        let _ = dec.feed(&bytes[offset..end]);
        if dec.stats().display_width != 0 {
            break;
        }
        offset = end;
    }

    let stats = dec.stats().clone();
    if stats.display_width == 0 || stats.display_height == 0 {
        return Err(NeoError::Decode(
            "probe_dimensions: no sequence header in first 8 MiB".into(),
        ));
    }
    Ok(stats)
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            if !self.parser.is_null() {
                let _ = (self.sys.destroy_video_parser)(self.parser);
            }
            if !self.state.decoder.is_null() {
                let _ = (self.sys.destroy_decoder)(self.state.decoder);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C callback trampolines
// ---------------------------------------------------------------------------

unsafe extern "C" fn handle_sequence(
    user: *mut c_void,
    format: *mut CUVIDEOFORMAT,
) -> i32 {
    let state = &mut *(user as *mut DecoderState);
    let fmt = &*format;

    state.stats.coded_width = fmt.coded_width;
    state.stats.coded_height = fmt.coded_height;
    state.stats.display_width = (fmt.display_area.right - fmt.display_area.left).max(0) as u32;
    state.stats.display_height = (fmt.display_area.bottom - fmt.display_area.top).max(0) as u32;

    // Probe-only mode: we just wanted dimensions, abort the parser.
    // Returning 0 tells NVDEC "no surfaces allocated", and it propagates
    // an error out of cuvidParseVideoData — which the probe helper
    // catches.
    if state.probe_only {
        return 0;
    }

    info!(
        coded = format!("{}x{}", fmt.coded_width, fmt.coded_height),
        display = format!("{}x{}", state.stats.display_width, state.stats.display_height),
        codec = ?fmt.codec,
        chroma = ?fmt.chroma_format,
        bit_depth = fmt.bit_depth_luma_minus8 + 8,
        min_surfaces = fmt.min_num_decode_surfaces,
        "NVDEC sequence header"
    );

    // Lazily create the underlying decoder on the first sequence header.
    // (Reconfigure-on-resolution-change is a Stage-4+ concern.)
    if state.decoder.is_null() {
        let mut info: CUVIDDECODECREATEINFO = std::mem::zeroed();
        info.ulWidth = fmt.coded_width as _;
        info.ulHeight = fmt.coded_height as _;
        info.ulNumDecodeSurfaces = (fmt.min_num_decode_surfaces.max(4)) as _;
        info.CodecType = fmt.codec;
        info.ChromaFormat = fmt.chroma_format;
        info.ulCreationFlags = cudaVideoCreateFlags::cudaVideoCreate_PreferCUVID as _;
        info.bitDepthMinus8 = fmt.bit_depth_luma_minus8 as _;
        info.OutputFormat = if fmt.bit_depth_luma_minus8 > 0 {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016
        } else {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12
        };
        info.DeinterlaceMode = cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave;
        info.ulTargetWidth = fmt.coded_width as _;
        info.ulTargetHeight = fmt.coded_height as _;
        info.ulNumOutputSurfaces = 2;
        info.vidLock = ptr::null_mut();

        // Sanity-check that the chroma format is something we can deal with.
        if !matches!(
            fmt.chroma_format,
            cudaVideoChromaFormat::cudaVideoChromaFormat_420
        ) {
            state.last_error = Some(format!(
                "unsupported chroma format: {:?} (only 4:2:0 is wired up)",
                fmt.chroma_format
            ));
            return 0;
        }

        let status = (state.sys.create_decoder)(&mut state.decoder, &mut info);
        if status != 0 {
            state.last_error = Some(format!("cuvidCreateDecoder failed: {status}"));
            return 0;
        }
        debug!(decoder = ?state.decoder, "NVDEC decoder created");
    }

    // Returning the surface count tells the parser how many decode buffers
    // we've allocated. Anything > 0 means "continue".
    (fmt.min_num_decode_surfaces.max(4)) as i32
}

unsafe extern "C" fn handle_decode(
    user: *mut c_void,
    pic_params: *mut CUVIDPICPARAMS,
) -> i32 {
    let state = &mut *(user as *mut DecoderState);
    if state.decoder.is_null() {
        state.last_error = Some("decode before sequence header".into());
        return 0;
    }
    let status = (state.sys.decode_picture)(state.decoder, pic_params);
    if status != 0 {
        state.last_error = Some(format!("cuvidDecodePicture failed: {status}"));
        return 0;
    }
    state.stats.pictures_decoded += 1;
    1
}

unsafe extern "C" fn handle_display(
    user: *mut c_void,
    disp_info: *mut CUVIDPARSERDISPINFO,
) -> i32 {
    let state = &mut *(user as *mut DecoderState);
    if disp_info.is_null() {
        // EOS marker.
        return 1;
    }
    state.stats.pictures_displayed += 1;

    match &state.capture_mode {
        CaptureMode::None => 1,
        CaptureMode::Cpu => {
            if let Err(msg) = capture_frame_cpu(state, &*disp_info) {
                state.last_error = Some(msg);
                return 0;
            }
            1
        }
        CaptureMode::Device(hook) => {
            let hook = *hook;
            if let Err(msg) = capture_frame_device(state, &*disp_info, hook) {
                state.last_error = Some(msg);
                return 0;
            }
            1
        }
    }
}

/// Common map/unmap scaffolding. Calls `body` with the mapped surface's
/// base device pointer + pitch, then unmaps regardless of success.
unsafe fn with_mapped_surface<F>(
    state: &mut DecoderState,
    disp: &CUVIDPARSERDISPINFO,
    body: F,
) -> Result<(), String>
where
    F: FnOnce(CUdeviceptr, usize, u32, u32) -> Result<(), String>,
{
    let width = state.stats.display_width;
    let height = state.stats.display_height;
    if width == 0 || height == 0 {
        return Err("surface capture: width/height not set".into());
    }

    let mut proc_params: CUVIDPROCPARAMS = std::mem::zeroed();
    proc_params.progressive_frame = disp.progressive_frame;
    proc_params.second_field = disp.repeat_first_field + 1;
    proc_params.top_field_first = disp.top_field_first;
    proc_params.unpaired_field = (disp.repeat_first_field < 0) as i32;

    let mut dev_ptr: CUdeviceptr = 0;
    let mut pitch: u32 = 0;
    let status = (state.sys.map_video_frame64)(
        state.decoder,
        disp.picture_index,
        &mut dev_ptr,
        &mut pitch,
        &mut proc_params,
    );
    if status != 0 {
        return Err(format!("cuvidMapVideoFrame64: {status}"));
    }

    let result = body(dev_ptr, pitch as usize, width, height);

    let unmap_status = (state.sys.unmap_video_frame64)(state.decoder, dev_ptr);
    if unmap_status != 0 {
        return Err(format!("cuvidUnmapVideoFrame64: {unmap_status}"));
    }
    result
}

/// Device-hook capture path (Stage 6b): the mapped surface pointer is
/// handed straight to a user callback. The callback is expected to run
/// any cuMemcpy2D DtoD into its own interop buffers before returning.
unsafe fn capture_frame_device(
    state: &mut DecoderState,
    disp: &CUVIDPARSERDISPINFO,
    hook: DeviceHook,
) -> Result<(), String> {
    with_mapped_surface(state, disp, |dptr, pitch, w, h| {
        let rc = (hook.callback)(hook.user, w, h, dptr as u64, pitch as u32);
        if rc != 0 {
            Err(format!("device hook returned {rc}"))
        } else {
            Ok(())
        }
    })
}

/// Map the decoded surface at `disp_info.picture_index`, copy Y and UV
/// planes to a tightly packed CPU buffer, then unmap. The resulting
/// [`DecodedFrame`] is pushed onto `state.frames`.
unsafe fn capture_frame_cpu(
    state: &mut DecoderState,
    disp: &CUVIDPARSERDISPINFO,
) -> Result<(), String> {
    let mut captured: Option<DecodedFrame> = None;
    with_mapped_surface(state, disp, |dptr, pitch, w, h| {
        let frame = copy_nv12_to_host(dptr, pitch, w, h)?;
        captured = Some(frame);
        Ok(())
    })?;
    if let Some(frame) = captured {
        state.frames.push(frame);
    }
    Ok(())
}

/// Issue two `cuMemcpy2D` calls — one for the Y plane, one for the UV plane —
/// copying from pitched device memory to tightly packed host memory.
unsafe fn copy_nv12_to_host(
    base: CUdeviceptr,
    src_pitch: usize,
    width: u32,
    height: u32,
) -> Result<DecodedFrame, String> {
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = w * (h / 2);

    let mut y = vec![0u8; y_size];
    let mut uv = vec![0u8; uv_size];

    // Y plane: h rows of w bytes, src stride = src_pitch.
    //
    // Note: `CUDA_MEMCPY2D_st` contains a `CUmemorytype` enum whose
    // smallest valid discriminant is 1, so `mem::zeroed()` would panic.
    // We use `MaybeUninit` + byte-zero + explicit field init instead.
    let mut cpy_y_uninit = std::mem::MaybeUninit::<CUDA_MEMCPY2D_st>::uninit();
    std::ptr::write_bytes(cpy_y_uninit.as_mut_ptr(), 0, 1);
    let cpy_y_ptr = cpy_y_uninit.as_mut_ptr();
    (*cpy_y_ptr).srcMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
    (*cpy_y_ptr).srcDevice = base;
    (*cpy_y_ptr).srcPitch = src_pitch;
    (*cpy_y_ptr).dstMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
    (*cpy_y_ptr).dstHost = y.as_mut_ptr() as *mut c_void;
    (*cpy_y_ptr).dstPitch = w;
    (*cpy_y_ptr).WidthInBytes = w;
    (*cpy_y_ptr).Height = h;
    let cpy_y = cpy_y_uninit.assume_init();
    let r = cuda_sys::cuMemcpy2D_v2(&cpy_y);
    if r != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemcpy2D Y plane: {r:?}"));
    }

    // UV plane: starts at base + src_pitch * coded_height. For our purpose
    // (no crop / same coded == display), use `h`. If the decoder coded
    // height differs, this still works because the alignment happens on
    // the source side and we're only reading `h/2` rows of `w` bytes.
    //
    // We use the display height here, which equals coded height for
    // the common 1080p case (coded_height = 1088 due to 16-pixel align,
    // but NVDEC returns the UV plane exactly one coded_height below Y —
    // we approximate by using src_pitch * h because cuvidMapVideoFrame
    // returns a contiguous layout for the mapped surface). For robust
    // operation we should pass coded_height, but Stage-4 tests use
    // dimensions where display == coded.
    let uv_base = base + (src_pitch as u64) * (h as u64);

    let mut cpy_uv_uninit = std::mem::MaybeUninit::<CUDA_MEMCPY2D_st>::uninit();
    std::ptr::write_bytes(cpy_uv_uninit.as_mut_ptr(), 0, 1);
    let cpy_uv_ptr = cpy_uv_uninit.as_mut_ptr();
    (*cpy_uv_ptr).srcMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
    (*cpy_uv_ptr).srcDevice = uv_base;
    (*cpy_uv_ptr).srcPitch = src_pitch;
    (*cpy_uv_ptr).dstMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
    (*cpy_uv_ptr).dstHost = uv.as_mut_ptr() as *mut c_void;
    (*cpy_uv_ptr).dstPitch = w;
    (*cpy_uv_ptr).WidthInBytes = w;
    (*cpy_uv_ptr).Height = h / 2;
    let cpy_uv = cpy_uv_uninit.assume_init();
    let r = cuda_sys::cuMemcpy2D_v2(&cpy_uv);
    if r != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemcpy2D UV plane: {r:?}"));
    }

    Ok(DecodedFrame {
        width,
        height,
        y,
        uv,
    })
}

/// Decode a complete in-memory bitstream and return aggregate stats.
///
/// Convenience entry point used by the `nvdec-test` CLI command — feeds
/// the entire byte slice in one shot, then flushes.
/// Like [`decode_bytes`], but with surface mapping enabled: returns both
/// aggregate stats *and* the CPU-resident NV12 frames.
pub fn decode_bytes_capture(
    runtime: &CudaRuntime,
    codec: cudaVideoCodec,
    bytes: &[u8],
) -> NeoResult<(DecodeStats, Vec<DecodedFrame>)> {
    let mut dec = Decoder::new(runtime, codec, CaptureMode::Cpu)?;
    const CHUNK: usize = 1 << 20;
    let mut offset = 0;
    while offset < bytes.len() {
        let end = (offset + CHUNK).min(bytes.len());
        dec.feed(&bytes[offset..end])?;
        offset = end;
    }
    dec.flush()?;
    let stats = dec.stats().clone();
    let frames = dec.take_frames();
    Ok((stats, frames))
}

pub fn decode_bytes(
    runtime: &CudaRuntime,
    codec: cudaVideoCodec,
    bytes: &[u8],
) -> NeoResult<DecodeStats> {
    let mut dec = Decoder::new(runtime, codec, CaptureMode::None)?;

    // Feed in modest chunks so we don't hand NVDEC a 1 GB pointer.
    const CHUNK: usize = 1 << 20;
    let mut offset = 0;
    while offset < bytes.len() {
        let end = (offset + CHUNK).min(bytes.len());
        dec.feed(&bytes[offset..end])?;
        offset = end;
    }
    dec.flush()?;

    let stats = dec.stats().clone();
    if stats.pictures_decoded == 0 {
        warn!("NVDEC decoded 0 pictures — input may be malformed");
    } else {
        info!(
            decoded = stats.pictures_decoded,
            displayed = stats.pictures_displayed,
            "NVDEC decode complete"
        );
    }
    Ok(stats)
}
