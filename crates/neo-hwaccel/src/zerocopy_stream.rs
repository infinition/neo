//! Zerocopy streaming NVDEC source.
//!
//! `CaptureMode::Device` with a DtoD hook that copies NV12 planes into
//! interop Y/UV buffers. Shared by the `neo` CLI streaming subcommands
//! and the Python bindings.

use crate::nvdec::{CaptureMode, Decoder, DeviceHook};
use crate::{CudaRuntime, NvdecCodec};
use cudarc::driver::sys::{
    self as cuda_sys, CUmemorytype, CUresult, CUDA_MEMCPY2D_st,
};
use neo_core::NeoResult;
use std::{ffi::c_void, sync::Arc};
use tracing::debug;

struct HookState {
    width: u32,
    height: u32,
    y_dptr: u64,
    uv_dptr: u64,
    frame_ready: bool,
}

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

    if memcpy2d_dtod(src_dptr, pitch, state.y_dptr, w, w, h).is_err() {
        return -1;
    }
    let uv_src = src_dptr + (pitch as u64) * (h as u64);
    if memcpy2d_dtod(uv_src, pitch, state.uv_dptr, w, w, h / 2).is_err() {
        return -1;
    }
    let r = cuda_sys::cuCtxSynchronize();
    if r != CUresult::CUDA_SUCCESS {
        return -1;
    }
    state.frame_ready = true;
    0
}

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

/// Default feed size for streaming use (throughput-oriented). For strict
/// frame-by-frame pacing (e.g. the Python bindings, where each `next()` must
/// surface exactly one frame), use `with_chunk` and a small value like 4096:
/// a 256 KiB chunk can hold dozens of frames on highly compressed sources and
/// NVDEC decodes them all in one feed, overwriting the interop buffers before
/// the caller sees the intermediate frames.
const DEFAULT_CHUNK: usize = 256 * 1024;

pub struct ZerocopyStream {
    runtime: Arc<CudaRuntime>,
    bitstream: Arc<Vec<u8>>,
    decoder: Option<Decoder>,
    hook_state: Box<HookState>,
    feed_offset: usize,
    chunk: usize,
    pub width: u32,
    pub height: u32,
}

impl ZerocopyStream {
    pub fn new(
        runtime: Arc<CudaRuntime>,
        bitstream: Vec<u8>,
        y_dptr: u64,
        uv_dptr: u64,
    ) -> NeoResult<Self> {
        Self::with_chunk(runtime, bitstream, y_dptr, uv_dptr, DEFAULT_CHUNK)
    }

    pub fn with_chunk(
        runtime: Arc<CudaRuntime>,
        bitstream: Vec<u8>,
        y_dptr: u64,
        uv_dptr: u64,
        chunk: usize,
    ) -> NeoResult<Self> {
        let probe = crate::nvdec::probe_dimensions(
            runtime.as_ref(),
            NvdecCodec::cudaVideoCodec_H264,
            &bitstream,
        )?;
        let width = probe.display_width;
        let height = probe.display_height;
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
            chunk: chunk.max(1),
            width,
            height,
        })
    }

    /// Feed until next frame is DtoD'd into interop Y/UV.
    /// Returns true when a frame is ready.
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
        let end = (self.feed_offset + self.chunk).min(self.bitstream.len());
        dec.feed(&self.bitstream[self.feed_offset..end])?;
        self.feed_offset = end;
        Ok(())
    }
}
