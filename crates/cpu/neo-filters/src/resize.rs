use crate::Filter;
use neo_core::frame::GpuFrame;
use neo_core::NeoResult;

/// GPU-accelerated resize filter.
#[derive(Debug, Clone, Copy)]
pub enum ResizeAlgorithm {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos,
}

pub struct ResizeFilter {
    target_width: u32,
    target_height: u32,
    algorithm: ResizeAlgorithm,
}

impl ResizeFilter {
    pub fn new(width: u32, height: u32, algorithm: ResizeAlgorithm) -> Self {
        Self {
            target_width: width,
            target_height: height,
            algorithm,
        }
    }
}

impl Filter for ResizeFilter {
    fn name(&self) -> &str {
        "resize"
    }

    fn process(&mut self, mut frame: GpuFrame) -> NeoResult<GpuFrame> {
        tracing::debug!(
            from = %format!("{}x{}", frame.width, frame.height),
            to = %format!("{}x{}", self.target_width, self.target_height),
            algo = ?self.algorithm,
            "Resize (stub)"
        );
        frame.width = self.target_width;
        frame.height = self.target_height;
        Ok(frame)
    }

    fn output_dimensions(&self, _w: u32, _h: u32) -> (u32, u32) {
        (self.target_width, self.target_height)
    }
}
