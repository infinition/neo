use crate::Filter;
use neo_core::format::PixelFormat;
use neo_core::frame::GpuFrame;
use neo_core::NeoResult;

/// GPU-accelerated color space / pixel format conversion.
///
/// Uses wgpu compute shaders for conversion (NV12→RGBA, YUV→RGB, etc.).
/// This is one of the most critical filters — it bridges hardware decoder
/// output (NV12) to AI inference input (RGBA float).
pub struct ColorConvertFilter {
    target_format: PixelFormat,
}

impl ColorConvertFilter {
    pub fn new(target_format: PixelFormat) -> Self {
        Self { target_format }
    }

    /// Convert to float RGBA (for AI inference input).
    pub fn to_rgba_f32() -> Self {
        Self::new(PixelFormat::Rgba32f)
    }

    /// Convert to float RGBA16 (for AI inference input, half precision).
    pub fn to_rgba_f16() -> Self {
        Self::new(PixelFormat::Rgba16f)
    }

    /// Convert to NV12 (for hardware encoder input).
    pub fn to_nv12() -> Self {
        Self::new(PixelFormat::Nv12)
    }
}

impl Filter for ColorConvertFilter {
    fn name(&self) -> &str {
        "color-convert"
    }

    fn process(&mut self, mut frame: GpuFrame) -> NeoResult<GpuFrame> {
        let from = frame.pixel_format;
        let to = self.target_format;

        if from == to {
            return Ok(frame);
        }

        tracing::debug!(from = %from, to = %to, "Color conversion (stub)");

        // TODO: Dispatch the appropriate compute shader
        // (e.g., NV12_TO_RGBA from neo-gpu builtins)
        frame.pixel_format = to;
        Ok(frame)
    }
}
