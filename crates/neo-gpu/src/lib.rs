//! # neo-gpu
//!
//! GPU abstraction layer for Neo-FFmpeg.
//!
//! Built on `wgpu` for cross-platform GPU access (Vulkan, Metal, DX12, WebGPU).
//! Manages VRAM buffers, compute shaders, and zero-copy transfers.

pub mod buffer;
pub mod compute;
pub mod context;
pub mod shader;
pub mod tensor_bridge;

pub use buffer::VramBuffer;
pub use compute::{dispatch_compute, Binding, BufferAccess};
pub use context::{GpuContext, GpuOptions};
pub use shader::builtins;
pub use tensor_bridge::BgraTensorBridge;
