//! Zerocopy streaming NVDEC source.
//!
//! Unlike the CPU-bounce version, this uses `CaptureMode::Device` so that
//! NVDEC's display callback fires a DtoD copy straight into interop Y/UV
//! buffers shared between CUDA and Vulkan. The NV12→BGRA conversion then
//! runs via `dispatch_interop()` — zero CPU involvement.
//!
//! Data flow per frame:
//!   NVDEC surface (CUDA dptr) ──DtoD──▶ interop Y buffer (CUDA ≡ wgpu)
//!                              ──DtoD──▶ interop UV buffer
//!                                        └─ wgpu compute NV12→BGRA
//!                                            └─ interop BGRA (wgpu storage)

use cudarc::driver::sys::{
    self as cuda_sys, CUmemorytype, CUresult, CUDA_MEMCPY2D_st,
};
use neo_core::{NeoError, NeoResult};
use neo_hwaccel::{
    nvdec::{CaptureMode, Decoder, DeviceHook},
    CudaRuntime, NvdecCodec,
};
use std::{ffi::c_void, sync::Arc};
use tracing::{debug, info, warn};

/// State shared between Rust and the `unsafe extern "C"` NVDEC hook.
///
/// A raw pointer to this struct is passed as `user` data. The hook runs
/// synchronously on the same thread inside `Decoder::feed()`, so the
/// borrow is effectively scoped to that call.
struct HookState {
    width: u32,
    height: u32,
    y_dptr: u64,
    uv_dptr: u64,
    frame_ready: bool,
}

/// C trampoline: copies NV12 planes from the mapped NVDEC surface into
/// the interop Y/UV device pointers via `cuMemcpy2D` DtoD.
unsafe extern "C" fn hook_trampoline(
    user: *mut c_void,
    width: u32,
    height: u32,
    src_dptr: u64,
    src_pitch: u32,
) -> i32 {
    let state = &mut *(user as *mut HookState);
    if width != state.width || height != state.height {
        return -1;
    }
    let w = width as usize;
    let h = height as usize;
    let pitch = src_pitch as usize;

    // Y plane: DtoD, tightly packed destination.
    if memcpy2d_dtod(src_dptr, pitch, state.y_dptr, w, w, h).is_err() {
        return -1;
    }
    // UV plane: starts at src_dptr + pitch * coded_height.
    let uv_src = src_dptr + (pitch as u64) * (h as u64);
    if memcpy2d_dtod(uv_src, pitch, state.uv_dptr, w, w, h / 2).is_err() {
        return -1;
    }
    // Synchronize so the wgpu compute sees the data.
    let r = cuda_sys::cuCtxSynchronize();
    if r != CUresult::CUDA_SUCCESS {
        return -1;
    }
    state.frame_ready = true;
    0
}

/// cuMemcpy2D device-to-device with independent src/dst pitches.
unsafe fn memcpy2d_dtod(
    src: u64,
    src_pitch: usize,
    dst: u64,
    dst_pitch: usize,
    width_bytes: usize,
    height: usize,
) -> Result<(), ()> {
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
        Err(())
    } else {
        Ok(())
    }
}

const CHUNK: usize = 256 * 1024;

/// Zerocopy streaming NVDEC source.
///
/// Each call to `next()` feeds bitstream until NVDEC fires the device hook,
/// which copies NV12 planes DtoD into the interop buffers. The caller then
/// calls `converter.dispatch_interop()` to run the NV12→BGRA compute —
/// no CPU memory is ever touched.
pub struct ZerocopyStream {
    runtime: Arc<CudaRuntime>,
    bitstream: Arc<Vec<u8>>,
    decoder: Option<Decoder>,
    hook_state: Box<HookState>,
    feed_offset: usize,
    pub width: u32,
    pub height: u32,
}

impl ZerocopyStream {
    /// Create a new zerocopy stream.
    ///
    /// `y_dptr` and `uv_dptr` are CUDA device pointers from interop buffers
    /// — the NVDEC hook will DtoD-copy each decoded frame into them.
    pub fn new(
        runtime: Arc<CudaRuntime>,
        bitstream: Vec<u8>,
        y_dptr: u64,
        uv_dptr: u64,
    ) -> NeoResult<Self> {
        let probe = neo_hwaccel::nvdec::probe_dimensions(
            runtime.as_ref(),
            NvdecCodec::cudaVideoCodec_H264,
            &bitstream,
        )?;
        let width = probe.display_width;
        let height = probe.display_height;
        info!(width, height, "ZerocopyStream ready");

        let bitstream = Arc::new(bitstream);
        let mut hook_state = Box::new(HookState {
            width,
            height,
            y_dptr,
            uv_dptr,
            frame_ready: false,
        });

        let hook = DeviceHook {
            callback: hook_trampoline,
            user: &mut *hook_state as *mut HookState as *mut c_void,
        };
        let decoder = Some(Decoder::new(
            runtime.as_ref(),
            NvdecCodec::cudaVideoCodec_H264,
            CaptureMode::Device(hook),
        )?);

        Ok(Self {
            runtime,
            bitstream,
            decoder,
            hook_state,
            feed_offset: 0,
            width,
            height,
        })
    }

    /// Feed bitstream until the next frame is DtoD-copied into the interop
    /// Y/UV buffers. Returns `true` if a frame is ready (caller should call
    /// `converter.dispatch_interop()`), or `false` on error.
    pub fn next(&mut self) -> NeoResult<bool> {
        loop {
            self.hook_state.frame_ready = false;
            self.pump()?;
            if self.hook_state.frame_ready {
                return Ok(true);
            }
        }
    }

    fn pump(&mut self) -> NeoResult<()> {
        if self.feed_offset >= self.bitstream.len() {
            // EOS: flush, drop decoder, recreate, rewind.
            if let Some(mut dec) = self.decoder.take() {
                let _ = dec.flush();
                drop(dec);
            }
            self.feed_offset = 0;
            debug!("ZerocopyStream looped");

            let hook = DeviceHook {
                callback: hook_trampoline,
                user: &mut *self.hook_state as *mut HookState as *mut c_void,
            };
            self.decoder = Some(Decoder::new(
                self.runtime.as_ref(),
                NvdecCodec::cudaVideoCodec_H264,
                CaptureMode::Device(hook),
            )?);
        }

        let dec = self.decoder.as_mut().expect("decoder present");
        let end = (self.feed_offset + CHUNK).min(self.bitstream.len());
        dec.feed(&self.bitstream[self.feed_offset..end])?;
        self.feed_offset = end;
        Ok(())
    }
}
