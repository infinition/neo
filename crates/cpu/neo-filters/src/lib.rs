//! # neo-filters
//!
//! Built-in neural video filters for Neo-FFmpeg.
//!
//! These are high-level operations that combine `neo-gpu` compute shaders
//! and `neo-infer` models into easy-to-use filter nodes.
//!
//! All filters operate entirely in VRAM — zero CPU involvement.
//!
//! ## Available Filters
//!
//! - **Upscale**: Super-resolution (2x, 4x) using neural networks
//! - **Denoise**: AI-powered noise/grain removal
//! - **Interpolate**: Frame interpolation (24fps → 60fps)
//! - **StyleTransfer**: Apply artistic styles in real-time
//! - **ColorConvert**: GPU-accelerated color space conversion
//! - **Resize**: High-quality Lanczos/bicubic resize on GPU
//! - **Crop**: Zero-copy crop (just adjusts buffer offsets)
//! - **Overlay**: Alpha-composite layers in VRAM

pub mod color_convert;
pub mod crop;
pub mod denoise;
pub mod interpolate;
pub mod overlay;
pub mod resize;
pub mod style_transfer;
pub mod upscale;

use neo_core::frame::GpuFrame;
use neo_core::NeoResult;

/// A filter that processes GpuFrames entirely in VRAM.
pub trait Filter: Send {
    /// Get the filter name.
    fn name(&self) -> &str;

    /// Process a single frame. Returns the processed frame (still in VRAM).
    fn process(&mut self, frame: GpuFrame) -> NeoResult<GpuFrame>;

    /// Process requiring multiple input frames (for temporal filters).
    fn process_temporal(&mut self, frames: &[GpuFrame]) -> NeoResult<GpuFrame> {
        // Default: just process the last frame
        if let Some(frame) = frames.last() {
            self.process(frame.clone())
        } else {
            Err(neo_core::NeoError::Pipeline("no input frames".into()))
        }
    }

    /// Whether this filter needs multiple frames (temporal).
    fn is_temporal(&self) -> bool {
        false
    }

    /// Number of frames needed for temporal processing.
    fn temporal_window(&self) -> usize {
        1
    }

    /// Output dimensions (if different from input).
    fn output_dimensions(&self, input_width: u32, input_height: u32) -> (u32, u32) {
        (input_width, input_height)
    }
}
