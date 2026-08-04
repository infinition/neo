//! # neo-moe
//!
//! Zero-copy MoE expert streaming from NVMe → VRAM, designed as a Neo crate.
//!
//! ## Concept
//!
//! For Qwen3-235B-A22B style MoE models only a small fraction of experts is
//! active per token (e.g. 3.6B params out of 35B total for Qwen3.6-35B-A3B).
//! Instead of keeping the full model in VRAM (impossible on 12 GB), we:
//!
//!   1. Parse the GGUF file and build an expert weight map (offset + size per
//!      expert in every layer).
//!   2. Run a lightweight prefetch predictor that guesses which experts will
//!      be needed next, based on the router logits of the current token.
//!   3. Stream those expert tensors NVMe → pinned host memory → VRAM via a
//!      background DMA thread, so compute and I/O fully overlap.
//!   4. Expose the CUDA device pointers to llama.cpp (or any ONNX backend)
//!      through a zero-copy tensor handle — same pattern as Neo's
//!      `neo-hwaccel::zerocopy_stream`.
//!
//! ## Pipeline
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  NVMe (GGUF file, mmap)                                     │
//! │  Expert[layer][id] → raw bytes at known file offset         │
//! └───────────────────────────┬─────────────────────────────────┘
//!                             │  O_DIRECT read or mmap page fault
//!                             ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Pinned host ring-buffer  (2× max_active_experts slots)     │
//! │  cuMemAllocHost  →  page-locked, DMA-able                   │
//! └───────────────────────────┬─────────────────────────────────┘
//!                             │  cudaMemcpyAsync H2D
//!                             ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  VRAM expert pool  (double-buffered)                        │
//! │  cudarc DeviceSlice<u8>                                     │
//! └───────────────────────────┬─────────────────────────────────┘
//!                             │  zero-copy tensor handle
//!                             ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Inference backend  (llama.cpp CUDA tensors / ONNX)         │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod error;
pub mod gguf;
pub mod pool;
pub mod predictor;
pub mod scheduler;
pub mod stream;
pub mod tensor;

// -- OS-specific I/O backend --
#[cfg(target_os = "linux")]
pub mod io_uring;

// -- C FFI bridge for llama.cpp integration --
pub mod ffi;

pub use error::{MoeError, Result};
pub use pool::ExpertPool;
pub use predictor::RouterPredictor;
pub use scheduler::{ExpertRequest, ExpertScheduler};
pub use stream::ExpertStream;
pub use tensor::ExpertTensorHandle;

/// Expert identity: layer index + expert index within that layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertId {
    /// Transformer layer (0-based).
    pub layer: u32,
    /// Expert index within the MoE layer.
    pub expert: u32,
}

impl ExpertId {
    /// Convenience constructor.
    #[inline]
    pub fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

/// Top-level configuration for the MoE streaming engine.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Path to the `.gguf` model file on a fast NVMe.
    pub gguf_path: std::path::PathBuf,

    /// How many experts can sit resident in VRAM simultaneously.
    /// Rule of thumb: set to `top_k_experts * 2` for double-buffering.
    pub vram_resident_experts: usize,

    /// Size of the pinned host ring-buffer in bytes.
    /// Default: 2 × `vram_resident_experts` × expert_byte_size.
    pub pinned_buf_bytes: Option<usize>,

    /// Number of I/O worker threads for NVMe reads.
    pub io_threads: usize,

    /// CUDA device index (0 for single-GPU systems).
    pub cuda_device: usize,

    /// How many layers ahead to prefetch experts for.
    pub prefetch_depth: usize,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            gguf_path: std::path::PathBuf::new(),
            vram_resident_experts: 16,
            pinned_buf_bytes: None,
            io_threads: 4,
            cuda_device: 0,
            prefetch_depth: 2,
        }
    }
}
