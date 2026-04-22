//! # neo-core
//!
//! Core types for Neo-FFmpeg: GPU tensor frames, pixel formats, color spaces,
//! and zero-copy buffer abstractions.
//!
//! Everything in Neo-FFmpeg operates on `GpuFrame` — a frame that lives in VRAM
//! and never touches CPU RAM unless explicitly requested.

pub mod color;
pub mod error;
pub mod format;
pub mod frame;
pub mod tensor;
pub mod timestamp;

pub use color::ColorSpace;
pub use error::{NeoError, NeoResult};
pub use format::{CodecId, ContainerFormat, PixelFormat};
pub use frame::{FrameFlags, FramePlane, GpuFrame, GpuFramePool};
pub use tensor::{DataType, TensorDesc, TensorLayout};
pub use timestamp::Timestamp;
