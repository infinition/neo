use serde::{Deserialize, Serialize};

/// Color space for accurate color processing in the GPU pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpace {
    /// Standard dynamic range (sRGB / BT.709)
    Srgb,
    /// BT.709 (HD video standard)
    Bt709,
    /// BT.2020 (4K/8K HDR video)
    Bt2020,
    /// Linear RGB (for compositing and AI inference)
    LinearRgb,
    /// Scene-referred linear (ACES)
    AcesCg,
    /// Display P3 (Apple/cinema)
    DisplayP3,
}

/// Transfer function (gamma curve).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferFunction {
    /// sRGB gamma (~2.2)
    Srgb,
    /// Linear (no gamma)
    Linear,
    /// Perceptual Quantizer (HDR10)
    Pq,
    /// Hybrid Log-Gamma (HLG broadcast HDR)
    Hlg,
}

/// Full color description of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColorDesc {
    pub space: ColorSpace,
    pub transfer: TransferFunction,
    pub range: ColorRange,
}

/// Whether luma/chroma values use full 0-255 or limited 16-235 range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorRange {
    Full,
    Limited,
}

impl Default for ColorDesc {
    fn default() -> Self {
        Self {
            space: ColorSpace::Bt709,
            transfer: TransferFunction::Srgb,
            range: ColorRange::Limited,
        }
    }
}
