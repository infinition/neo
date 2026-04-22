//! # neo-hwaccel
//!
//! Native hardware acceleration for Neo-FFmpeg.
//!
//! This crate provides direct access to NVIDIA hardware codecs (NVDEC/NVENC)
//! and CUDA-Vulkan external memory interop, enabling true zero-copy video
//! processing pipelines: NVMe → NVDEC → CUDA → wgpu compute → NVENC → NVMe
//! without any CPU memory bounce.
//!
//! ## Stages
//!
//! - **Stage 1 (current)**: CUDA device detection via cudarc dynamic loading.
//! - Stage 2: NVDEC FFI bindings (decode H.264 → CUDA memory).
//! - Stage 3: NVENC FFI bindings (encode CUDA memory → H.264).
//! - Stage 4: End-to-end NVDEC → NVENC pipeline (CPU bounce intermediate).
//! - Stage 5: CUDA-Vulkan external memory interop (true zero-copy).

pub mod cuda;
pub mod encode_test;
#[cfg(windows)]
pub mod interop;
pub mod nvdec;
pub mod nvdec_sys;
pub mod nvenc;
pub mod transcode;
pub mod wgpu_convert;

pub use cuda::{CudaCapabilities, CudaDeviceInfo, CudaRuntime};
pub use encode_test::{run as run_encode_test, EncodeTestResult};
pub use nvdec::{
    decode_bytes as decode_nvdec, decode_bytes_capture as decode_nvdec_capture, DecodeStats,
    DecodedFrame, Decoder as NvdecDecoder,
};

/// Re-export the NVDEC codec enum so callers don't need a direct dep on
/// the vendored sys crate.
pub use nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec as NvdecCodec;
pub use nvenc::{probe as probe_nvenc, NvencCapabilities};
pub use transcode::{transcode_h264, ConvertBackend, TranscodeStats};
#[cfg(windows)]
pub use transcode::zerocopy::transcode_h264_zerocopy;
pub use wgpu_convert::Nv12ToBgraConverter;
