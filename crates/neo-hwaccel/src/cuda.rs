//! CUDA runtime detection and device enumeration.
//!
//! Uses `cudarc` with dynamic loading so the binary builds and runs on
//! machines without a CUDA toolkit installed — calls just return
//! `HwAccelUnavailable` at runtime.

use cudarc::driver::{result, sys};
use neo_core::{NeoError, NeoResult};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Per-device CUDA capability info.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: (i32, i32),
    pub multiprocessor_count: i32,
}

/// Aggregate CUDA capabilities of the host.
#[derive(Debug, Clone)]
pub struct CudaCapabilities {
    pub devices: Vec<CudaDeviceInfo>,
}

impl CudaCapabilities {
    pub fn primary_device(&self) -> Option<&CudaDeviceInfo> {
        self.devices.first()
    }
}

/// Owned CUDA runtime: holds the active context for the chosen device.
///
/// Created lazily — if CUDA is not present on the machine, construction
/// returns `NeoError::HwAccelUnavailable` and the rest of Neo-FFmpeg
/// transparently falls back to the FFmpeg subprocess pipeline.
pub struct CudaRuntime {
    pub ctx: Arc<cudarc::driver::CudaContext>,
    pub capabilities: CudaCapabilities,
}

impl CudaRuntime {
    /// Probe CUDA without initializing a long-lived context. Returns the
    /// device list or an error explaining why CUDA is not usable.
    pub fn probe() -> NeoResult<CudaCapabilities> {
        result::init()
            .map_err(|e| NeoError::HwAccelUnavailable(format!("cuInit failed: {e:?}")))?;

        let device_count = result::device::get_count()
            .map_err(|e| NeoError::Cuda(format!("cuDeviceGetCount failed: {e:?}")))?;

        let mut devices = Vec::with_capacity(device_count as usize);
        for ordinal in 0..device_count {
            let dev = result::device::get(ordinal)
                .map_err(|e| NeoError::Cuda(format!("cuDeviceGet({ordinal}) failed: {e:?}")))?;

            let name = result::device::get_name(dev)
                .map_err(|e| NeoError::Cuda(format!("cuDeviceGetName failed: {e:?}")))?;

            let total_memory = unsafe { result::device::total_mem(dev) }
                .map_err(|e| NeoError::Cuda(format!("cuDeviceTotalMem failed: {e:?}")))?;

            let major = unsafe {
                result::device::get_attribute(
                    dev,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )
            }
            .map_err(|e| NeoError::Cuda(format!("cc_major failed: {e:?}")))?;

            let minor = unsafe {
                result::device::get_attribute(
                    dev,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                )
            }
            .map_err(|e| NeoError::Cuda(format!("cc_minor failed: {e:?}")))?;

            let sm_count = unsafe {
                result::device::get_attribute(
                    dev,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                )
            }
            .map_err(|e| NeoError::Cuda(format!("sm_count failed: {e:?}")))?;

            debug!(ordinal, %name, total_memory, major, minor, sm_count, "CUDA device");
            devices.push(CudaDeviceInfo {
                ordinal: ordinal as usize,
                name,
                total_memory,
                compute_capability: (major, minor),
                multiprocessor_count: sm_count,
            });
        }

        Ok(CudaCapabilities { devices })
    }

    /// Create a runtime bound to the given device ordinal.
    pub fn new(device_ordinal: usize) -> NeoResult<Self> {
        let capabilities = Self::probe()?;
        if capabilities.devices.is_empty() {
            return Err(NeoError::HwAccelUnavailable(
                "no CUDA devices found".into(),
            ));
        }
        if device_ordinal >= capabilities.devices.len() {
            return Err(NeoError::Cuda(format!(
                "device ordinal {device_ordinal} out of range (have {} devices)",
                capabilities.devices.len()
            )));
        }

        let ctx = cudarc::driver::CudaContext::new(device_ordinal).map_err(|e| {
            NeoError::Cuda(format!(
                "CudaContext::new({device_ordinal}) failed: {e:?}"
            ))
        })?;

        let dev = &capabilities.devices[device_ordinal];
        info!(
            device = %dev.name,
            cc = format!("{}.{}", dev.compute_capability.0, dev.compute_capability.1),
            vram_gb = dev.total_memory / (1024 * 1024 * 1024),
            "CUDA runtime initialized"
        );

        Ok(Self { ctx, capabilities })
    }

    /// Try to initialize CUDA. Logs a warning and returns `None` if
    /// CUDA is unavailable — used by the pipeline to decide between native
    /// and FFmpeg-subprocess paths.
    pub fn try_new() -> Option<Self> {
        match Self::new(0) {
            Ok(rt) => Some(rt),
            Err(e) => {
                warn!("CUDA unavailable, falling back to FFmpeg pipeline: {e}");
                None
            }
        }
    }
}
