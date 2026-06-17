use crate::Filter;
use neo_core::frame::GpuFrame;
use neo_core::NeoResult;

/// GPU alpha-composite overlay.
///
/// Composites a foreground layer on top of the main frame,
/// entirely in VRAM using a compute shader.
pub struct OverlayFilter {
    pub x: i32,
    pub y: i32,
    pub opacity: f32,
}

impl OverlayFilter {
    pub fn new(x: i32, y: i32, opacity: f32) -> Self {
        Self {
            x,
            y,
            opacity: opacity.clamp(0.0, 1.0),
        }
    }
}

impl Filter for OverlayFilter {
    fn name(&self) -> &str {
        "overlay"
    }

    fn process(&mut self, frame: GpuFrame) -> NeoResult<GpuFrame> {
        tracing::debug!(
            x = self.x,
            y = self.y,
            opacity = self.opacity,
            "Overlay (stub)"
        );
        Ok(frame)
    }
}
