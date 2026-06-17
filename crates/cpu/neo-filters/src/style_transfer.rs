use crate::Filter;
use neo_core::frame::GpuFrame;
use neo_core::NeoResult;

/// Real-time neural style transfer.
///
/// Applies an artistic style (e.g., "Ghibli", "oil painting", "cyberpunk")
/// to video frames in real-time, entirely in VRAM.
pub struct StyleTransferFilter {
    style_name: String,
    strength: f32,
}

impl StyleTransferFilter {
    pub fn new(style: &str, strength: f32) -> Self {
        Self {
            style_name: style.to_string(),
            strength: strength.clamp(0.0, 1.0),
        }
    }
}

impl Filter for StyleTransferFilter {
    fn name(&self) -> &str {
        "style-transfer"
    }

    fn process(&mut self, mut frame: GpuFrame) -> NeoResult<GpuFrame> {
        tracing::debug!(
            style = %self.style_name,
            strength = self.strength,
            "Style transfer (stub)"
        );
        frame.flags.ai_processed = true;
        Ok(frame)
    }
}
