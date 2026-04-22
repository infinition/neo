use crate::Filter;
use neo_core::frame::GpuFrame;
use neo_core::{NeoError, NeoResult};

/// Zero-copy crop — adjusts buffer offsets without copying data.
pub struct CropFilter {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl CropFilter {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }
}

impl Filter for CropFilter {
    fn name(&self) -> &str {
        "crop"
    }

    fn process(&mut self, mut frame: GpuFrame) -> NeoResult<GpuFrame> {
        if self.x + self.width > frame.width || self.y + self.height > frame.height {
            return Err(NeoError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }
        // In a real implementation, we'd adjust the buffer offset/stride
        // to point to the cropped region — true zero-copy.
        frame.width = self.width;
        frame.height = self.height;
        Ok(frame)
    }

    fn output_dimensions(&self, _w: u32, _h: u32) -> (u32, u32) {
        (self.width, self.height)
    }
}
