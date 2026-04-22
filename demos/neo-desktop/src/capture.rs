//! Zero-copy DXGI Desktop Duplication → CUDA capture.
//!
//! Pipeline (all in VRAM, no CPU pixel bounce):
//!
//! ```text
//! desktop compositor (DXGI)
//!         │  AcquireNextFrame  ── ID3D11Texture2D (BGRA, read-only)
//!         ▼
//! intermediate D3D11 texture (DEFAULT, BGRA, even dims)
//!         │  CopySubresourceRegion  (GPU)
//!         ▼
//! CUDA-Graphics interop
//!         │  cuGraphicsMapResources + cuGraphicsSubResourceGetMappedArray
//!         ▼
//! CUarray (BGRA, opaque tiled)
//!         │  cuMemcpy2D  (DtoD, GPU)
//!         ▼
//! CUdeviceptr (BGRA, pitched, linear) ──> NVENC
//! ```
//!
//! The pixels never touch the CPU. The only bounce is the unavoidable
//! `CopySubresourceRegion` from the duplication texture into our own
//! D3D11 texture (so we can register it with CUDA and crop to even dims).

use crate::cuda_d3d11::{register_d3d11_resource, CU_GRAPHICS_REGISTER_FLAGS_NONE};

use cudarc::driver::sys::{
    self as cu, CUarray, CUdeviceptr, CUgraphicsResource, CUmemorytype, CUresult,
    CUDA_MEMCPY2D_st,
};
use std::ptr;
use tracing::{info, warn};
use windows::{
    core::Interface,
    Win32::{
        Foundation::HMODULE,
        Graphics::{
            Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1},
            Direct3D11::*,
            Dxgi::Common::*,
            Dxgi::*,
        },
    },
};

/// Cropping happens once at the GPU, not per-frame on the CPU.
pub struct ZeroCopyCapture {
    _device: ID3D11Device,
    context: ID3D11DeviceContext,
    duplication: IDXGIOutputDuplication,
    /// Intermediate, GPU-only texture sized to (enc_width, enc_height).
    /// Registered with CUDA so its content can be DtoD-copied to a linear
    /// pitched buffer.
    intermediate: ID3D11Texture2D,
    cu_resource: CUgraphicsResource,

    /// Source desktop dimensions.
    pub src_width: u32,
    pub src_height: u32,
    /// NVENC-friendly even dimensions (cropped, ≤ src).
    pub enc_width: u32,
    pub enc_height: u32,

    monitor: u32,
}

unsafe impl Send for ZeroCopyCapture {}

impl ZeroCopyCapture {
    pub fn new(monitor: u32) -> Result<Self, String> {
        unsafe { Self::new_inner(monitor) }
    }

    unsafe fn new_inner(monitor: u32) -> Result<Self, String> {
        // 1. D3D11 device on the default DXGI adapter. For multi-GPU systems
        //    this assumes adapter 0 matches CUDA device 0 — true on most
        //    desktops with a single discrete NVIDIA GPU.
        let mut device = None;
        let mut context = None;
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
        .map_err(|e| format!("D3D11CreateDevice: {e}"))?;
        let device = device.ok_or("D3D11 device is None")?;
        let context = context.ok_or("D3D11 context is None")?;

        // 2. DXGI adapter / output for the requested monitor.
        let dxgi_device: IDXGIDevice = device.cast().map_err(|e| format!("IDXGIDevice: {e}"))?;
        let adapter = dxgi_device
            .GetAdapter()
            .map_err(|e| format!("GetAdapter: {e}"))?;
        let output: IDXGIOutput = adapter
            .EnumOutputs(monitor)
            .map_err(|e| format!("EnumOutputs({monitor}): {e}"))?;
        let desc = output.GetDesc().map_err(|e| format!("GetDesc: {e}"))?;
        let rect = desc.DesktopCoordinates;
        let src_width = (rect.right - rect.left) as u32;
        let src_height = (rect.bottom - rect.top) as u32;

        let output1: IDXGIOutput1 = output.cast().map_err(|e| format!("IDXGIOutput1: {e}"))?;
        let duplication = output1
            .DuplicateOutput(&device)
            .map_err(|e| format!("DuplicateOutput: {e}"))?;

        // 3. Even dimensions for NVENC.
        let enc_width = src_width & !1;
        let enc_height = src_height & !1;

        // 4. Intermediate D3D11 texture: DEFAULT usage, no CPU access.
        //    BIND_SHADER_RESOURCE is required so CUDA can map it as an array.
        let mid_desc = D3D11_TEXTURE2D_DESC {
            Width: enc_width,
            Height: enc_height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let intermediate = {
            let mut tex = None;
            device
                .CreateTexture2D(&mid_desc, None, Some(&mut tex))
                .map_err(|e| format!("CreateTexture2D intermediate: {e}"))?;
            tex.ok_or("intermediate texture is None")?
        };

        // 5. Register the intermediate texture with CUDA. This is the gate
        //    that fails on multi-GPU systems where the D3D11 device and the
        //    CUDA context live on different adapters.
        let resource_ptr = intermediate.as_raw();
        let cu_resource = register_d3d11_resource(resource_ptr, CU_GRAPHICS_REGISTER_FLAGS_NONE)
            .map_err(|e| format!("CUDA-D3D11 register: {e}"))?;

        info!(
            src = format!("{src_width}x{src_height}"),
            enc = format!("{enc_width}x{enc_height}"),
            monitor,
            "DXGI Desktop Duplication + CUDA interop ready (zerocopy)"
        );
        Ok(Self {
            _device: device,
            context,
            duplication,
            intermediate,
            cu_resource,
            src_width,
            src_height,
            enc_width,
            enc_height,
            monitor,
        })
    }

    /// Capture one frame straight into a CUDA pitched device buffer.
    ///
    /// `dst_dptr` must be at least `dst_pitch * enc_height` bytes and laid
    /// out as BGRA (the format NVENC sees as `NV_ENC_BUFFER_FORMAT_ARGB`).
    /// `dst_pitch` is the row stride in **bytes**, as returned by
    /// `cuMemAllocPitch`.
    ///
    /// Returns `Ok(true)` on a fresh frame, `Ok(false)` on AcquireNextFrame
    /// timeout (= no desktop change), and `Err` on a fatal error.
    /// `Err("DXGI access lost")` means the caller should rebuild the
    /// capture (resolution / mode / lock screen change).
    pub fn capture_into(
        &mut self,
        dst_dptr: CUdeviceptr,
        dst_pitch: usize,
        timeout_ms: u32,
    ) -> Result<bool, String> {
        unsafe { self.capture_into_inner(dst_dptr, dst_pitch, timeout_ms) }
    }

    unsafe fn capture_into_inner(
        &mut self,
        dst_dptr: CUdeviceptr,
        dst_pitch: usize,
        timeout_ms: u32,
    ) -> Result<bool, String> {
        // -- 1. Acquire next desktop frame from the compositor -----------------
        let mut frame_info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource: Option<IDXGIResource> = None;
        let hr = self
            .duplication
            .AcquireNextFrame(timeout_ms, &mut frame_info, &mut resource);
        match hr {
            Err(e) => {
                let code = e.code().0 as u32;
                match code {
                    0x887A0027 => return Ok(false),                    // WAIT_TIMEOUT
                    0x887A0026 => return Err("DXGI access lost".into()), // ACCESS_LOST
                    0x887A0022 => return Err("DXGI access denied".into()), // ACCESS_DENIED (UAC, lock screen)
                    0x887A0005 => return Err("DXGI device removed".into()), // DEVICE_REMOVED
                    _ => return Err(format!("AcquireNextFrame: {e}")),
                }
            }
            Ok(()) => {}
        }

        let resource = match resource {
            Some(r) => r,
            None => {
                let _ = self.duplication.ReleaseFrame();
                return Ok(false);
            }
        };
        let desktop: ID3D11Texture2D = resource
            .cast()
            .map_err(|e| format!("cast desktop tex: {e}"))?;

        // -- 2. Sanity-check the acquired texture -----------------------------
        // Chrome's hardware-accelerated video, fullscreen-exclusive games and
        // HDR mode can hand us a texture whose format / size differs from
        // what we registered with CUDA. CopySubresourceRegion would either
        // silently fail or trip the device into REMOVED. Detect early and
        // ask the caller to rebuild.
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        desktop.GetDesc(&mut desc);
        if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM {
            let _ = self.duplication.ReleaseFrame();
            return Err(format!(
                "DXGI format changed to {:?} (likely HDR / fullscreen overlay)",
                desc.Format
            ));
        }
        if desc.Width < self.enc_width || desc.Height < self.enc_height {
            let _ = self.duplication.ReleaseFrame();
            return Err(format!(
                "DXGI resolution shrank to {}x{}",
                desc.Width, desc.Height
            ));
        }

        // -- 3. GPU-side crop into our intermediate texture --------------------
        let src_box = D3D11_BOX {
            left: 0,
            top: 0,
            front: 0,
            right: self.enc_width,
            bottom: self.enc_height,
            back: 1,
        };
        self.context.CopySubresourceRegion(
            &self.intermediate,
            0,
            0,
            0,
            0,
            &desktop,
            0,
            Some(&src_box),
        );

        // Release the duplication frame ASAP so the compositor isn't blocked.
        if let Err(e) = self.duplication.ReleaseFrame() {
            warn!("ReleaseFrame: {e}");
        }

        // -- 3. CUDA: map the registered D3D11 resource and DtoD copy ----------
        let mut res = self.cu_resource;
        let r = cu::cuGraphicsMapResources(1, &mut res, ptr::null_mut());
        if r != CUresult::CUDA_SUCCESS {
            return Err(format!("cuGraphicsMapResources: {r:?}"));
        }

        let mut cu_array: CUarray = ptr::null_mut();
        let r = cu::cuGraphicsSubResourceGetMappedArray(&mut cu_array, res, 0, 0);
        if r != CUresult::CUDA_SUCCESS {
            let _ = cu::cuGraphicsUnmapResources(1, &mut res, ptr::null_mut());
            return Err(format!("cuGraphicsSubResourceGetMappedArray: {r:?}"));
        }

        // BGRA = 4 bytes per pixel.
        let bytes_per_row = (self.enc_width as usize) * 4;
        let mut copy = std::mem::MaybeUninit::<CUDA_MEMCPY2D_st>::uninit();
        std::ptr::write_bytes(copy.as_mut_ptr(), 0, 1);
        let p = copy.as_mut_ptr();
        (*p).srcMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
        (*p).srcArray = cu_array;
        (*p).dstMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
        (*p).dstDevice = dst_dptr;
        (*p).dstPitch = dst_pitch;
        (*p).WidthInBytes = bytes_per_row;
        (*p).Height = self.enc_height as usize;
        let copy = copy.assume_init();
        let r = cu::cuMemcpy2D_v2(&copy);

        let _ = cu::cuGraphicsUnmapResources(1, &mut res, ptr::null_mut());

        if r != CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemcpy2D_v2 (Array→DevicePtr): {r:?}"));
        }
        Ok(true)
    }

    pub fn monitor(&self) -> u32 {
        self.monitor
    }
}

impl Drop for ZeroCopyCapture {
    fn drop(&mut self) {
        unsafe {
            // Order matters: unregister CUDA resource before the texture goes.
            if !self.cu_resource.is_null() {
                let _ = cu::cuGraphicsUnregisterResource(self.cu_resource);
            }
            let _ = self.duplication.ReleaseFrame();
        }
    }
}
