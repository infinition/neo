//! # neo-infer
//!
//! AI inference engine for Neo-FFmpeg.
//!
//! Runs neural network models directly on VRAM tensors — no CPU round-trip.
//! The frame stays in GPU memory from decode through inference to encode.
//!
//! ## Supported Backends
//!
//! - **wgpu compute shaders**: Universal, runs everywhere (Vulkan/Metal/DX12/WebGPU)
//! - **ONNX Runtime** (planned): For complex models with optimized kernels
//! - **TensorRT** (planned): Maximum performance on NVIDIA GPUs
//!
//! ## Model Format
//!
//! Neo-FFmpeg uses `.neo` model files — optimized weight bundles that include:
//! - Pre-compiled compute shaders for each target backend
//! - Quantized weights (FP16/INT8)
//! - Input/output tensor descriptors
//! - Pipeline integration metadata

pub mod model;
pub mod runtime;
pub mod session;

pub use model::{ModelInfo, ModelFormat};
pub use runtime::{InferenceRuntime, RuntimeBackend};
pub use session::InferenceSession;
