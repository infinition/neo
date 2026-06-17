//! # neo-decode
//!
//! Hardware-accelerated video decoding for Neo-FFmpeg.
//!
//! Abstracts platform-specific hardware decoders (NVDEC, VideoToolbox, VAAPI)
//! behind a unified trait. Decoded frames stay in VRAM — never touch CPU RAM.
//!
//! ## Architecture
//!
//! ```text
//! Compressed bitstream (H.264/HEVC/AV1)
//!         │
//!         ▼
//!   ┌─────────────┐
//!   │  HwDecoder   │  ← Platform-specific (NVDEC / VideoToolbox / VAAPI)
//!   │  (in VRAM)   │
//!   └──────┬───────┘
//!          │
//!          ▼
//!     GpuFrame (NV12/P010 in VRAM)  ← Ready for pipeline processing
//! ```

pub mod decoder;
pub mod demuxer;
pub mod probe;

pub use decoder::{Decoder, DecoderConfig, HwAccelApi};
pub use demuxer::{Demuxer, StreamInfo, MediaInfo};
pub use probe::probe_file;
