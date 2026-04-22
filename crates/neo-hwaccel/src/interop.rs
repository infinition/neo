//! Stage 6a: CUDA ↔ Vulkan external memory interop.
//!
//! This module exposes the minimum ceremony required to share a block of
//! VRAM between CUDA and Vulkan/wgpu with no CPU round-trip:
//!
//! 1. Ask wgpu's Vulkan HAL for the raw [`ash::Device`] + physical device.
//! 2. Allocate a `VkBuffer` backed by `vk::DeviceMemory` with the
//!    `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32` handle type exported.
//! 3. Hand the raw `vk::Buffer` to `wgpu::Device::create_buffer_from_hal`
//!    so wgpu's compute shaders can bind it as a storage buffer.
//! 4. Call `vkGetMemoryWin32HandleKHR` to get a Win32 `HANDLE` for the
//!    memory.
//! 5. Import that `HANDLE` into CUDA via `cuImportExternalMemory` +
//!    `cuExternalMemoryGetMappedBuffer`, yielding a `CUdeviceptr` that
//!    aliases the same VRAM pages.
//!
//! Stage 6a provides a self-contained test:
//! - Fill the buffer from CUDA with a known pattern.
//! - Download it through wgpu's staging path.
//! - Verify the wgpu view sees exactly what CUDA wrote.
//!
//! If that works, Stage 6b rewires the NVDEC output copy to write directly
//! into one of these shared buffers, then hands it straight to the wgpu
//! compute converter — no more CPU NV12 bounce.
//!
//! Windows-only for now. Linux would switch the handle type to OpaqueFd
//! and use `vkGetMemoryFdKHR`; the rest is identical.

#![cfg(windows)]

use crate::cuda::CudaRuntime;
use ash::vk;
use cudarc::driver::sys::{
    self as cuda_sys, CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st, CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st,
    CUdeviceptr, CUexternalMemory, CUexternalMemoryHandleType, CUresult,
};
use neo_core::{NeoError, NeoResult};
use neo_gpu::GpuContext;
use std::{ffi::c_void, mem::ManuallyDrop, ptr, sync::Arc};
use tracing::{debug, info, warn};
use windows::Win32::Foundation::HANDLE;

/// A single shared buffer aliased between CUDA and Vulkan/wgpu.
///
/// The backing `vk::DeviceMemory` is owned by this struct and freed on
/// drop after both views have been released.
pub struct InteropBuffer {
    ctx: Arc<GpuContext>,
    size: u64,
    /// Raw Vulkan memory we allocated ourselves — freed manually on drop
    /// after `wgpu_buffer` has destroyed the VkBuffer, because wgpu
    /// doesn't know how to free externally-backed memory.
    vk_memory: vk::DeviceMemory,
    /// wgpu view — used for binding in compute shaders.
    ///
    /// Wrapped in `ManuallyDrop` so we can explicitly drop it before
    /// freeing `vk_memory` (the Vulkan spec requires the buffer be
    /// destroyed while its backing memory is still alive).
    wgpu_buffer: ManuallyDrop<wgpu::Buffer>,
    /// CUDA view — used for `cuMemcpy*` from NVDEC surfaces.
    pub cu_device_ptr: CUdeviceptr,
    cu_external_memory: CUexternalMemory,
    /// The exported Win32 HANDLE we used to import into CUDA. Kept alive
    /// so CUDA's reference count remains positive; closed on drop.
    exported_handle: HANDLE,
}

impl InteropBuffer {
    /// Borrow the wgpu view. Use this to bind the shared buffer into a
    /// compute pipeline — it's a normal `wgpu::Buffer` from wgpu's
    /// perspective.
    pub fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.wgpu_buffer
    }
}

impl InteropBuffer {
    pub fn size(&self) -> u64 {
        self.size
    }
}

impl std::fmt::Debug for InteropBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteropBuffer")
            .field("size", &self.size)
            .field("cu_device_ptr", &self.cu_device_ptr)
            .finish()
    }
}

impl Drop for InteropBuffer {
    fn drop(&mut self) {
        unsafe {
            // 1. Release the CUDA import. This must happen before the
            //    Vulkan memory is freed, otherwise CUDA still holds a live
            //    alias to freed pages.
            let _ = cuda_sys::cuDestroyExternalMemory(self.cu_external_memory);

            // 2. Close the duplicated Win32 handle. CUDA's import
            //    increments the kernel-object refcount; CloseHandle here
            //    releases our own reference.
            let _ = windows::Win32::Foundation::CloseHandle(self.exported_handle);

            // 3. Drop the wgpu buffer — this causes wgpu-hal to call
            //    `vkDestroyBuffer` on our raw VkBuffer. It does NOT free
            //    the backing memory because we built the hal Buffer with
            //    `Buffer::from_raw` (no allocation attached).
            ManuallyDrop::drop(&mut self.wgpu_buffer);

            // 4. Wgpu defers actual resource destruction until the next
            //    device poll. Force the pending queue to drain before we
            //    free the memory the buffer was bound to.
            let _ = self.ctx.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            // 5. Free the raw Vulkan memory via the HAL's ash::Device.
            if let Some(hal_device) = self.ctx.device.as_hal::<wgpu_hal::api::Vulkan>() {
                hal_device.raw_device().free_memory(self.vk_memory, None);
            }
        }
    }
}

/// Allocate a shared Vulkan buffer, wrap it as a wgpu::Buffer, and import
/// it into CUDA. The returned [`InteropBuffer`] gives simultaneous access
/// through both APIs over the same VRAM pages.
///
/// `size_bytes` is rounded up to a 16-byte multiple (Vulkan requires
/// storage-buffer binding alignment).
pub fn create_interop_buffer(
    gpu: Arc<GpuContext>,
    cuda: &CudaRuntime,
    size_bytes: u64,
) -> NeoResult<InteropBuffer> {
    if !gpu
        .device
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32)
    {
        return Err(NeoError::HwAccelUnavailable(
            "wgpu device is missing VULKAN_EXTERNAL_MEMORY_WIN32 feature".into(),
        ));
    }

    // Bind the CUDA context on this thread before importing.
    cuda.ctx
        .bind_to_thread()
        .map_err(|e| NeoError::Cuda(format!("bind_to_thread: {e:?}")))?;

    let size = (size_bytes + 15) & !15u64;

    // Reach into the Vulkan HAL and perform the raw allocation + export.
    let alloc = unsafe {
        let hal_device_opt = gpu
            .device
            .as_hal::<wgpu_hal::api::Vulkan>();
        let hal_device = hal_device_opt
            .as_ref()
            .ok_or_else(|| NeoError::HwAccelUnavailable("wgpu is not on Vulkan".into()))?;
        allocate_exportable_vk_buffer(hal_device, size)?
    };

    // Wrap the raw buffer as a wgpu::Buffer. We use `from_raw`, which tells
    // wgpu-hal "memory is managed externally" — exactly what we want.
    let hal_buffer = unsafe { wgpu_hal::vulkan::Buffer::from_raw(alloc.vk_buffer) };
    let wgpu_buffer = unsafe {
        gpu.device.create_buffer_from_hal::<wgpu_hal::api::Vulkan>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some("cuda-vulkan-interop"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
    };

    // Import the exported handle into CUDA.
    let (cu_external_memory, cu_device_ptr) =
        unsafe { import_into_cuda(alloc.handle, size)? };

    info!(
        size_mb = size / (1024 * 1024),
        vk_memory = ?alloc.vk_memory,
        cu_dptr = cu_device_ptr,
        "CUDA↔Vulkan interop buffer created"
    );

    Ok(InteropBuffer {
        ctx: gpu,
        size,
        vk_memory: alloc.vk_memory,
        wgpu_buffer: ManuallyDrop::new(wgpu_buffer),
        cu_device_ptr,
        cu_external_memory,
        exported_handle: alloc.handle,
    })
}

/// Raw Vulkan objects returned from the HAL allocation step.
struct VkAlloc {
    vk_buffer: vk::Buffer,
    vk_memory: vk::DeviceMemory,
    handle: HANDLE,
}

/// Allocate a VkBuffer + exportable VkDeviceMemory and export the Win32
/// HANDLE. Runs inside `Device::as_hal` so it can see the ash::Device.
unsafe fn allocate_exportable_vk_buffer(
    dev: &wgpu_hal::vulkan::Device,
    size: u64,
) -> NeoResult<VkAlloc> {
    let raw = dev.raw_device();
    let shared = dev.shared_instance();
    let phys = dev.raw_physical_device();

    // 1. Create a vk::Buffer that advertises "external memory is OK".
    let mut external_info = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external_info);
    let vk_buffer = raw
        .create_buffer(&buffer_info, None)
        .map_err(|e| NeoError::HwAccelUnavailable(format!("vkCreateBuffer: {e}")))?;

    // 2. Pick a DEVICE_LOCAL memory type that satisfies the buffer's
    //    requirements.
    let req = raw.get_buffer_memory_requirements(vk_buffer);
    let mem_props = shared
        .raw_instance()
        .get_physical_device_memory_properties(phys);
    let mem_type_index = (0..mem_props.memory_type_count)
        .find(|i| {
            (req.memory_type_bits & (1 << i)) != 0
                && mem_props.memory_types[*i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .ok_or_else(|| {
            NeoError::HwAccelUnavailable(
                "no DEVICE_LOCAL memory type matches buffer requirements".into(),
            )
        })?;

    // 3. Allocate memory that will be exported. `ExportMemoryAllocateInfo`
    //    chains into `MemoryAllocateInfo::push_next`.
    let mut export_info = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type_index)
        .push_next(&mut export_info);
    let vk_memory = raw.allocate_memory(&alloc_info, None).map_err(|e| {
        raw.destroy_buffer(vk_buffer, None);
        NeoError::HwAccelUnavailable(format!("vkAllocateMemory: {e}"))
    })?;

    // 4. Bind buffer ↔ memory.
    raw.bind_buffer_memory(vk_buffer, vk_memory, 0).map_err(|e| {
        raw.destroy_buffer(vk_buffer, None);
        raw.free_memory(vk_memory, None);
        NeoError::HwAccelUnavailable(format!("vkBindBufferMemory: {e}"))
    })?;

    // 5. Export the Win32 handle via vkGetMemoryWin32HandleKHR.
    let ext_mem_win32 =
        ash::khr::external_memory_win32::Device::new(shared.raw_instance(), raw);
    let handle_info = vk::MemoryGetWin32HandleInfoKHR::default()
        .memory(vk_memory)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    let raw_handle = ext_mem_win32
        .get_memory_win32_handle(&handle_info)
        .map_err(|e| {
            raw.destroy_buffer(vk_buffer, None);
            raw.free_memory(vk_memory, None);
            NeoError::HwAccelUnavailable(format!("vkGetMemoryWin32HandleKHR: {e}"))
        })?;
    let handle = HANDLE(raw_handle as *mut _);

    debug!(
        req_size = req.size,
        mem_type = mem_type_index,
        "exported Win32 handle for Vulkan memory"
    );

    Ok(VkAlloc {
        vk_buffer,
        vk_memory,
        handle,
    })
}

/// Import a Win32 OPAQUE_WIN32 handle into CUDA and map it to a
/// CUdeviceptr covering the full `size`.
unsafe fn import_into_cuda(
    handle: HANDLE,
    size: u64,
) -> NeoResult<(CUexternalMemory, CUdeviceptr)> {
    let mut desc: CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st =
        std::mem::MaybeUninit::zeroed().assume_init();
    desc.type_ = CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
    desc.handle.win32.handle = handle.0 as *mut c_void;
    desc.handle.win32.name = ptr::null();
    desc.size = size;
    desc.flags = 0;

    let mut ext_mem: CUexternalMemory = ptr::null_mut();
    let r = cuda_sys::cuImportExternalMemory(&mut ext_mem, &desc);
    if r != CUresult::CUDA_SUCCESS {
        return Err(NeoError::Cuda(format!("cuImportExternalMemory: {r:?}")));
    }

    let mut buf_desc: CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st =
        std::mem::MaybeUninit::zeroed().assume_init();
    buf_desc.offset = 0;
    buf_desc.size = size;
    buf_desc.flags = 0;

    let mut dptr: CUdeviceptr = 0;
    let r = cuda_sys::cuExternalMemoryGetMappedBuffer(&mut dptr, ext_mem, &buf_desc);
    if r != CUresult::CUDA_SUCCESS {
        cuda_sys::cuDestroyExternalMemory(ext_mem);
        return Err(NeoError::Cuda(format!(
            "cuExternalMemoryGetMappedBuffer: {r:?}"
        )));
    }

    Ok((ext_mem, dptr))
}

/// End-to-end smoke test: allocate an [`InteropBuffer`], fill it from
/// CUDA with a known 32-bit pattern, then download it through wgpu's
/// staging path and verify every word matches.
pub fn interop_self_test(
    gpu: Arc<GpuContext>,
    cuda: &CudaRuntime,
) -> NeoResult<InteropStats> {
    const N_WORDS: usize = 1024 * 1024; // 4 MiB
    const PATTERN: u32 = 0xDEAD_BEEFu32;
    let size = (N_WORDS * 4) as u64;

    info!(
        n_words = N_WORDS,
        size_mb = size / (1024 * 1024),
        pattern = format!("{PATTERN:#010x}"),
        "starting CUDA↔Vulkan interop self-test"
    );

    let buf = create_interop_buffer(gpu.clone(), cuda, size)?;

    // 1. Fill from CUDA via cuMemsetD32.
    let t0 = std::time::Instant::now();
    unsafe {
        let r = cuda_sys::cuMemsetD32_v2(buf.cu_device_ptr, PATTERN, N_WORDS);
        if r != CUresult::CUDA_SUCCESS {
            return Err(NeoError::Cuda(format!("cuMemsetD32: {r:?}")));
        }
        // Synchronize so the wgpu side sees the write.
        let r = cuda_sys::cuCtxSynchronize();
        if r != CUresult::CUDA_SUCCESS {
            return Err(NeoError::Cuda(format!("cuCtxSynchronize: {r:?}")));
        }
    }
    let cuda_write_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // 2. Read back through wgpu via a staging buffer.
    let t0 = std::time::Instant::now();
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("interop-staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("interop-download"),
        });
    encoder.copy_buffer_to_buffer(buf.wgpu_buffer(), 0, &staging, 0, size);
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| NeoError::GpuBuffer(format!("poll: {e}")))?;
    rx.recv()
        .map_err(|_| NeoError::GpuBuffer("staging channel closed".into()))?
        .map_err(|e| NeoError::GpuBuffer(format!("map: {e}")))?;

    let data = slice.get_mapped_range();
    let words: &[u32] = bytes_to_u32(&data);
    let mismatches = words.iter().filter(|&&w| w != PATTERN).count();
    let first_word = words[0];
    let last_word = words[words.len() - 1];
    let wgpu_read_ms = t0.elapsed().as_secs_f64() * 1000.0;

    drop(data);
    staging.unmap();

    let ok = mismatches == 0;
    if ok {
        info!(
            mismatches, first_word = format!("{first_word:#010x}"),
            last_word = format!("{last_word:#010x}"),
            "interop self-test PASSED"
        );
    } else {
        warn!(
            mismatches,
            first_word = format!("{first_word:#010x}"),
            "interop self-test FAILED — wgpu view does not match CUDA write"
        );
    }

    Ok(InteropStats {
        size,
        pattern: PATTERN,
        mismatches,
        first_word,
        last_word,
        cuda_write_ms,
        wgpu_read_ms,
        ok,
    })
}

/// Results of [`interop_self_test`].
#[derive(Debug, Clone)]
pub struct InteropStats {
    pub size: u64,
    pub pattern: u32,
    pub mismatches: usize,
    pub first_word: u32,
    pub last_word: u32,
    pub cuda_write_ms: f64,
    pub wgpu_read_ms: f64,
    pub ok: bool,
}

/// Safe transmute of a byte slice into a u32 slice (used only inside this
/// module, so we skip the bytemuck dep dance).
fn bytes_to_u32(bytes: &[u8]) -> &[u32] {
    assert!(bytes.len() % 4 == 0);
    assert!((bytes.as_ptr() as usize) % 4 == 0);
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) }
}
