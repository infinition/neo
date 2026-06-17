//! neo — zero-copy NVDEC video source for Python/PyTorch.
//!
//! NVDEC decode → DtoD → interop Y/UV → wgpu NV12→BGRA → pack to RGB CHW f32
//! → DtoD into a PyTorch-owned CUDA buffer. Pixels never touch host RAM:
//! the host only sees the compressed bitstream that is fed to NVDEC.
//!
//! Python side:
//! ```python
//! import torch, neo
//! src = neo.VideoSource("video.h264")                  # Annex-B H.264
//! t = torch.empty((3, src.height, src.width), dtype=torch.float32, device="cuda")
//! src.next_into(t.data_ptr())                          # t = RGB CHW in [0,1], VRAM only
//! ```

use cudarc::driver::sys::{self as cuda_sys, CUresult};
use neo_gpu::{BgraTensorBridge, GpuContext, GpuOptions};
use neo_hwaccel::{
    interop::{create_interop_buffer, InteropBuffer},
    zerocopy_stream::ZerocopyStream,
    CudaRuntime, Nv12ToBgraConverter, NvdecCodec,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

/// Zero-copy H.264 video source. Decodes on NVDEC, converts NV12→RGB CHW f32
/// on the GPU via wgpu compute, and DtoD-copies the result into a caller
/// provided CUDA buffer (e.g. a PyTorch tensor's data_ptr()).
///
/// `next_into` does NOT block the CPU: it records a CUDA event after the
/// final DtoD. Callers must order their GPU work after it, either with
/// `wait_stream(torch.cuda.current_stream().cuda_stream)` (non-blocking,
/// GPU-side) or `synchronize()` (blocking).
///
/// Loops back to the start of the bitstream at EOF (live-demo friendly).
#[pyclass(unsendable)]
struct VideoSource {
    // Drop order matters: the stream holds raw device pointers into the
    // interop buffers, so it must be dropped first (fields drop in order).
    stream: ZerocopyStream,
    converter: Nv12ToBgraConverter,
    bridge: BgraTensorBridge,
    _interop_y: InteropBuffer,
    _interop_uv: InteropBuffer,
    bgra: InteropBuffer,
    tensor: InteropBuffer,
    runtime: Arc<CudaRuntime>,
    _gpu: Arc<GpuContext>,
    event: cuda_sys::CUevent,
    width: u32,
    height: u32,
    frames: u64,
}

impl Drop for VideoSource {
    fn drop(&mut self) {
        unsafe {
            cuda_sys::cuEventDestroy_v2(self.event);
        }
    }
}

#[pymethods]
impl VideoSource {
    /// Open an Annex-B H.264 bitstream (`.h264` / `.264`).
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(err)?;

        let runtime = Arc::new(CudaRuntime::new(0).map_err(err)?);
        runtime.ctx.bind_to_thread().map_err(|e| err(format!("{e:?}")))?;

        let probe = neo_hwaccel::nvdec::probe_dimensions(
            runtime.as_ref(),
            NvdecCodec::cudaVideoCodec_H264,
            &bytes,
        )
        .map_err(err)?;
        let width = probe.display_width;
        let height = probe.display_height;

        let gpu = Arc::new(GpuContext::new_sync(&GpuOptions::interop()).map_err(err)?);

        let y_size = (width as u64) * (height as u64);
        let interop_y = create_interop_buffer(gpu.clone(), &runtime, y_size).map_err(err)?;
        let interop_uv = create_interop_buffer(gpu.clone(), &runtime, y_size / 2).map_err(err)?;
        let bgra = create_interop_buffer(gpu.clone(), &runtime, y_size * 4).map_err(err)?;
        let tensor = create_interop_buffer(gpu.clone(), &runtime, y_size * 3 * 4).map_err(err)?;

        let converter = Nv12ToBgraConverter::from_external_buffers(
            gpu.clone(),
            width,
            height,
            interop_y.wgpu_buffer(),
            interop_uv.wgpu_buffer(),
            bgra.wgpu_buffer(),
        )
        .map_err(err)?;
        let bridge = BgraTensorBridge::new(gpu.clone(), width, height).map_err(err)?;

        // 4 KiB feed chunks: strict frame-by-frame pacing (one decoded frame
        // surfaced per next(), no chunk-skipping on compressed sources).
        let stream = ZerocopyStream::with_chunk(
            runtime.clone(),
            bytes,
            interop_y.cu_device_ptr,
            interop_uv.cu_device_ptr,
            4096,
        )
        .map_err(err)?;

        let mut event: cuda_sys::CUevent = std::ptr::null_mut();
        unsafe {
            // CU_EVENT_DISABLE_TIMING = 0x2 (sync-only event, cheapest kind)
            let r = cuda_sys::cuEventCreate(&mut event, 0x2);
            if r != CUresult::CUDA_SUCCESS {
                return Err(err(format!("cuEventCreate: {r:?}")));
            }
        }

        Ok(Self {
            stream,
            converter,
            bridge,
            _interop_y: interop_y,
            _interop_uv: interop_uv,
            bgra,
            tensor,
            runtime,
            _gpu: gpu,
            event,
            width,
            height,
            frames: 0,
        })
    }

    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    #[getter]
    fn frames(&self) -> u64 {
        self.frames
    }

    /// Decode the next frame and write it as RGB CHW f32 in [0,1] into
    /// `dst_ptr` (a CUDA device pointer with room for 3*H*W f32, e.g.
    /// `torch.empty((3,H,W), dtype=torch.float32, device='cuda').data_ptr()`).
    ///
    /// The entire path (NVDEC → NV12→BGRA → BGRA→CHW → DtoD) stays in VRAM.
    /// Non-blocking: records a CUDA event after the DtoD. Call `wait_stream`
    /// (or `synchronize`) before consuming the tensor from another stream.
    fn next_into(&mut self, dst_ptr: u64) -> PyResult<bool> {
        self.runtime.ctx.bind_to_thread().map_err(|e| err(format!("{e:?}")))?;

        if !self.stream.next().map_err(err)? {
            return Ok(false);
        }
        self.converter.dispatch_interop().map_err(err)?;
        self.bridge
            .pack_into(self.bgra.wgpu_buffer(), self.tensor.wgpu_buffer())
            .map_err(err)?;

        let size = (self.width as usize) * (self.height as usize) * 3 * 4;
        unsafe {
            let r = cuda_sys::cuMemcpyDtoD_v2(dst_ptr, self.tensor.cu_device_ptr, size);
            if r != CUresult::CUDA_SUCCESS {
                return Err(err(format!("cuMemcpyDtoD: {r:?}")));
            }
            self.record_event()?;
        }
        self.frames += 1;
        Ok(true)
    }

    /// Same as `next_into`, but also writes the decoded BGRA frame (H*W u32)
    /// into `bgra_dst_ptr` for on-GPU visualization (torch uint8 tensor of
    /// shape (H, W, 4), BGRA byte order).
    fn next_into_with_bgra(&mut self, dst_ptr: u64, bgra_dst_ptr: u64) -> PyResult<bool> {
        if !self.next_into(dst_ptr)? {
            return Ok(false);
        }
        let size = (self.width as usize) * (self.height as usize) * 4;
        unsafe {
            let r = cuda_sys::cuMemcpyDtoD_v2(bgra_dst_ptr, self.bgra.cu_device_ptr, size);
            if r != CUresult::CUDA_SUCCESS {
                return Err(err(format!("cuMemcpyDtoD(bgra): {r:?}")));
            }
            self.record_event()?;
        }
        Ok(true)
    }

    /// Make `stream_handle` (e.g. `torch.cuda.current_stream().cuda_stream`)
    /// wait for the last `next_into` to complete. GPU-side ordering only —
    /// does not block the CPU.
    fn wait_stream(&self, stream_handle: u64) -> PyResult<()> {
        unsafe {
            let r = cuda_sys::cuStreamWaitEvent(stream_handle as cuda_sys::CUstream, self.event, 0);
            if r != CUresult::CUDA_SUCCESS {
                return Err(err(format!("cuStreamWaitEvent: {r:?}")));
            }
        }
        Ok(())
    }

    /// Block the CPU until the last `next_into` has fully completed.
    fn synchronize(&self) -> PyResult<()> {
        unsafe {
            let r = cuda_sys::cuEventSynchronize(self.event);
            if r != CUresult::CUDA_SUCCESS {
                return Err(err(format!("cuEventSynchronize: {r:?}")));
            }
        }
        Ok(())
    }
}

impl VideoSource {
    /// Record the completion event on the current (NULL) stream, capturing
    /// all GPU work issued so far for this frame.
    unsafe fn record_event(&self) -> PyResult<()> {
        let r = cuda_sys::cuEventRecord(self.event, std::ptr::null_mut());
        if r != CUresult::CUDA_SUCCESS {
            return Err(err(format!("cuEventRecord: {r:?}")));
        }
        Ok(())
    }
}

#[pymodule]
fn neo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VideoSource>()?;
    Ok(())
}
