//! # neo-encode
//!
//! Hardware-accelerated video encoding for Neo-FFmpeg.
//!
//! Encodes GPU frames directly from VRAM using NVENC, VideoToolbox, etc.
//! The encoded bitstream goes directly to disk or network — no CPU round-trip.

pub mod encoder;
pub mod muxer;

pub use encoder::{Encoder, EncoderConfig, RateControl};
pub use muxer::{Muxer, MuxerConfig};
