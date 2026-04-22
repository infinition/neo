//! Dynamic-loaded FFI shim for NVIDIA Video Decoder (`nvcuvid.dll`).
//!
//! Like the NVENC shim in the patched vendor crate, this module loads
//! `nvcuvid.dll` at runtime via `libloading` so the resulting binary has no
//! link-time dependency on the NVIDIA Video Codec SDK. Only a recent NVIDIA
//! display driver is required at runtime.
//!
//! The function table is cached in a `OnceLock`; the first call resolves
//! every symbol and subsequent calls are a single atomic load + indirect call.

use cudarc::driver::sys::{CUcontext, CUdeviceptr, CUstream};
use neo_core::{NeoError, NeoResult};
use nvidia_video_codec_sdk::sys::{
    cuviddec::{
        CUVIDDECODECAPS, CUVIDDECODECREATEINFO, CUVIDPICPARAMS, CUVIDPROCPARAMS, CUvideodecoder,
    },
    nvcuvid::{CUVIDPARSERPARAMS, CUVIDSOURCEDATAPACKET, CUvideoparser},
};
use std::sync::OnceLock;

/// Loaded `nvcuvid.dll` function table.
pub struct NvCuvid {
    _lib: libloading::Library,

    // Parser entry points.
    pub create_video_parser:
        unsafe extern "C" fn(*mut CUvideoparser, *mut CUVIDPARSERPARAMS) -> i32,
    pub parse_video_data:
        unsafe extern "C" fn(CUvideoparser, *mut CUVIDSOURCEDATAPACKET) -> i32,
    pub destroy_video_parser: unsafe extern "C" fn(CUvideoparser) -> i32,

    // Decoder entry points.
    pub get_decoder_caps: unsafe extern "C" fn(*mut CUVIDDECODECAPS) -> i32,
    pub create_decoder:
        unsafe extern "C" fn(*mut CUvideodecoder, *mut CUVIDDECODECREATEINFO) -> i32,
    pub destroy_decoder: unsafe extern "C" fn(CUvideodecoder) -> i32,
    pub decode_picture: unsafe extern "C" fn(CUvideodecoder, *mut CUVIDPICPARAMS) -> i32,

    // Surface mapping (64-bit device pointer variants — required for >32-bit VA on x64).
    pub map_video_frame64: unsafe extern "C" fn(
        CUvideodecoder,
        i32,
        *mut CUdeviceptr,
        *mut u32,
        *mut CUVIDPROCPARAMS,
    ) -> i32,
    pub unmap_video_frame64: unsafe extern "C" fn(CUvideodecoder, CUdeviceptr) -> i32,

    // Context lock helpers — needed when sharing one CUcontext across threads.
    pub ctx_lock_create:
        unsafe extern "C" fn(*mut *mut std::ffi::c_void, CUcontext) -> i32,
    pub ctx_lock_destroy: unsafe extern "C" fn(*mut std::ffi::c_void) -> i32,
}

/// Singleton: load + cache `nvcuvid.dll` once per process.
static NVCUVID: OnceLock<Result<NvCuvid, String>> = OnceLock::new();

#[cfg(windows)]
const LIB_NAMES: &[&str] = &["nvcuvid.dll"];
#[cfg(unix)]
const LIB_NAMES: &[&str] = &["libnvcuvid.so.1", "libnvcuvid.so"];

unsafe fn load_inner() -> Result<NvCuvid, String> {
    let mut last_err = String::new();
    for name in LIB_NAMES {
        match libloading::Library::new(name) {
            Ok(lib) => {
                // SAFETY: each `get` matches the C signature declared above.
                let create_video_parser = *lib
                    .get::<unsafe extern "C" fn(*mut CUvideoparser, *mut CUVIDPARSERPARAMS) -> i32>(
                        b"cuvidCreateVideoParser\0",
                    )
                    .map_err(|e| format!("cuvidCreateVideoParser: {e}"))?;
                let parse_video_data = *lib
                    .get::<unsafe extern "C" fn(CUvideoparser, *mut CUVIDSOURCEDATAPACKET) -> i32>(
                        b"cuvidParseVideoData\0",
                    )
                    .map_err(|e| format!("cuvidParseVideoData: {e}"))?;
                let destroy_video_parser = *lib
                    .get::<unsafe extern "C" fn(CUvideoparser) -> i32>(
                        b"cuvidDestroyVideoParser\0",
                    )
                    .map_err(|e| format!("cuvidDestroyVideoParser: {e}"))?;
                let get_decoder_caps = *lib
                    .get::<unsafe extern "C" fn(*mut CUVIDDECODECAPS) -> i32>(
                        b"cuvidGetDecoderCaps\0",
                    )
                    .map_err(|e| format!("cuvidGetDecoderCaps: {e}"))?;
                let create_decoder = *lib
                    .get::<unsafe extern "C" fn(
                        *mut CUvideodecoder,
                        *mut CUVIDDECODECREATEINFO,
                    ) -> i32>(b"cuvidCreateDecoder\0")
                    .map_err(|e| format!("cuvidCreateDecoder: {e}"))?;
                let destroy_decoder = *lib
                    .get::<unsafe extern "C" fn(CUvideodecoder) -> i32>(b"cuvidDestroyDecoder\0")
                    .map_err(|e| format!("cuvidDestroyDecoder: {e}"))?;
                let decode_picture = *lib
                    .get::<unsafe extern "C" fn(CUvideodecoder, *mut CUVIDPICPARAMS) -> i32>(
                        b"cuvidDecodePicture\0",
                    )
                    .map_err(|e| format!("cuvidDecodePicture: {e}"))?;
                let map_video_frame64 = *lib
                    .get::<unsafe extern "C" fn(
                        CUvideodecoder,
                        i32,
                        *mut CUdeviceptr,
                        *mut u32,
                        *mut CUVIDPROCPARAMS,
                    ) -> i32>(b"cuvidMapVideoFrame64\0")
                    .map_err(|e| format!("cuvidMapVideoFrame64: {e}"))?;
                let unmap_video_frame64 = *lib
                    .get::<unsafe extern "C" fn(CUvideodecoder, CUdeviceptr) -> i32>(
                        b"cuvidUnmapVideoFrame64\0",
                    )
                    .map_err(|e| format!("cuvidUnmapVideoFrame64: {e}"))?;
                let ctx_lock_create = *lib
                    .get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, CUcontext) -> i32>(
                        b"cuvidCtxLockCreate\0",
                    )
                    .map_err(|e| format!("cuvidCtxLockCreate: {e}"))?;
                let ctx_lock_destroy = *lib
                    .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32>(
                        b"cuvidCtxLockDestroy\0",
                    )
                    .map_err(|e| format!("cuvidCtxLockDestroy: {e}"))?;

                return Ok(NvCuvid {
                    _lib: lib,
                    create_video_parser,
                    parse_video_data,
                    destroy_video_parser,
                    get_decoder_caps,
                    create_decoder,
                    destroy_decoder,
                    decode_picture,
                    map_video_frame64,
                    unmap_video_frame64,
                    ctx_lock_create,
                    ctx_lock_destroy,
                });
            }
            Err(e) => {
                last_err = format!("LoadLibrary({name}): {e}");
            }
        }
    }
    Err(last_err)
}

/// Get a reference to the cached function table, loading `nvcuvid.dll`
/// on first call.
pub fn get() -> NeoResult<&'static NvCuvid> {
    let slot = NVCUVID.get_or_init(|| unsafe { load_inner() });
    slot.as_ref()
        .map_err(|e| NeoError::HwAccelUnavailable(format!("nvcuvid load failed: {e}")))
}

/// Suppress unused warnings for fields we haven't wired up yet — these are
/// part of the public surface of the loader.
#[allow(dead_code)]
fn _force_link(s: &NvCuvid) {
    let _ = s.unmap_video_frame64;
    let _ = s.map_video_frame64;
    let _ = s.ctx_lock_create;
    let _ = s.ctx_lock_destroy;
    let _ = s.get_decoder_caps;
}

// CUDA stream type re-exported so callers don't need to depend on cudarc::sys.
pub type Stream = CUstream;
