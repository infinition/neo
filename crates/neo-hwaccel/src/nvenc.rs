//! NVENC (NVIDIA hardware video encoder) integration.
//!
//! Built on top of the patched `nvidia-video-codec-sdk` vendor crate, which
//! we have modified so that `nvEncodeAPI64.dll` is loaded at runtime via
//! `libloading` — no compile-time `.lib` import library is required, and
//! the binary works on any machine with a recent NVIDIA driver.

use crate::cuda::CudaRuntime;
use neo_core::{NeoError, NeoResult};
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_CODEC_AV1_GUID, NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID,
    },
    Encoder,
};
use tracing::{debug, info};

/// Capability snapshot of the NVENC engine on this device.
#[derive(Debug, Clone)]
pub struct NvencCapabilities {
    pub h264: bool,
    pub hevc: bool,
    pub av1: bool,
    pub raw_guid_count: usize,
}

impl NvencCapabilities {
    pub fn any(&self) -> bool {
        self.h264 || self.hevc || self.av1
    }
}

/// Probe NVENC by opening a real encode session against the supplied
/// CUDA runtime, then querying its supported codec GUIDs.
///
/// This validates the entire dynamic-loading path:
/// `nvEncodeAPI64.dll` -> `NvEncodeAPICreateInstance` -> function table ->
/// `OpenEncodeSessionEx(CUDA)` -> `GetEncodeGUIDs`.
pub fn probe(runtime: &CudaRuntime) -> NeoResult<NvencCapabilities> {
    debug!("opening NVENC session via dynamic-loaded nvEncodeAPI64.dll");
    let encoder = Encoder::initialize_with_cuda(runtime.ctx.clone())
        .map_err(|e| NeoError::HwAccelUnavailable(format!("NVENC init failed: {e:?}")))?;

    let guids = encoder
        .get_encode_guids()
        .map_err(|e| NeoError::HwAccelUnavailable(format!("get_encode_guids failed: {e:?}")))?;

    let h264 = guids.contains(&NV_ENC_CODEC_H264_GUID);
    let hevc = guids.contains(&NV_ENC_CODEC_HEVC_GUID);
    let av1 = guids.contains(&NV_ENC_CODEC_AV1_GUID);

    info!(h264, hevc, av1, count = guids.len(), "NVENC capabilities");

    Ok(NvencCapabilities {
        h264,
        hevc,
        av1,
        raw_guid_count: guids.len(),
    })
}
