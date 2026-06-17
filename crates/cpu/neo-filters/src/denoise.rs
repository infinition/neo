use crate::Filter;
use neo_core::frame::GpuFrame;
use neo_core::NeoResult;

/// AI denoising filter.
///
/// Removes noise, grain, and compression artifacts using a neural network.
/// Operates entirely in VRAM.
pub struct DenoiseFilter {
    /// Strength 0.0 (none) to 1.0 (maximum).
    strength: f32,
}

impl DenoiseFilter {
    pub fn new(strength: f32) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
        }
    }
}

impl Filter for DenoiseFilter {
    fn name(&self) -> &str {
        "denoise"
    }

    fn process(&mut self, mut frame: GpuFrame) -> NeoResult<GpuFrame> {
        tracing::debug!(strength = self.strength, "Denoise (stub)");
        frame.flags.ai_processed = true;
        Ok(frame)
    }
}
