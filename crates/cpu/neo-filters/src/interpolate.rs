use crate::Filter;
use neo_core::frame::GpuFrame;
use neo_core::NeoResult;

/// Frame interpolation filter — generates intermediate frames.
///
/// Takes two consecutive frames and generates N frames in between,
/// enabling smooth slow-motion or frame rate conversion (24fps → 60fps).
pub struct InterpolateFilter {
    /// Number of intermediate frames to generate.
    multiplier: u32,
}

impl InterpolateFilter {
    /// Create a 2x interpolation filter (e.g., 30fps → 60fps).
    pub fn x2() -> Self {
        Self { multiplier: 2 }
    }

    /// Create a 4x interpolation filter (e.g., 30fps → 120fps).
    pub fn x4() -> Self {
        Self { multiplier: 4 }
    }
}

impl Filter for InterpolateFilter {
    fn name(&self) -> &str {
        "interpolate"
    }

    fn process(&mut self, mut frame: GpuFrame) -> NeoResult<GpuFrame> {
        tracing::debug!(multiplier = self.multiplier, "Frame interpolation (stub)");
        frame.flags.ai_processed = true;
        Ok(frame)
    }

    fn is_temporal(&self) -> bool {
        true
    }

    fn temporal_window(&self) -> usize {
        2
    }
}
