use serde::{Deserialize, Serialize};

/// Pixel format — how color data is laid out in memory.
///
/// Neo-FFmpeg treats frames as GPU tensors, so formats map directly
/// to texture formats the GPU understands natively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PixelFormat {
    // 8-bit per channel
    Rgba8,
    Bgra8,
    Nv12,
    Yuv420p,
    Yuv422p,
    Yuv444p,

    // 10-bit (HDR)
    Yuv420p10le,
    Yuv444p10le,
    P010,

    // 16-bit float (native tensor format)
    Rgba16f,

    // 32-bit float (full precision tensor)
    Rgba32f,

    // Grayscale
    Gray8,
    Gray16,
}

impl PixelFormat {
    /// Bytes per pixel (approximate for planar formats).
    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            Self::Gray8 => 1,
            Self::Nv12 | Self::Yuv420p => 2, // 1.5 but we round up
            Self::Gray16 | Self::Yuv422p => 2,
            Self::Yuv444p => 3,
            Self::Rgba8 | Self::Bgra8 => 4,
            Self::Yuv420p10le | Self::P010 => 3,
            Self::Yuv444p10le => 6,
            Self::Rgba16f => 8,
            Self::Rgba32f => 16,
        }
    }

    /// Number of planes for this format.
    pub fn plane_count(&self) -> u32 {
        match self {
            Self::Rgba8 | Self::Bgra8 | Self::Rgba16f | Self::Rgba32f
            | Self::Gray8 | Self::Gray16 => 1,
            Self::Nv12 | Self::P010 => 2,
            Self::Yuv420p | Self::Yuv422p | Self::Yuv444p
            | Self::Yuv420p10le | Self::Yuv444p10le => 3,
        }
    }

    /// Whether this is a float format (native for AI inference).
    pub fn is_float(&self) -> bool {
        matches!(self, Self::Rgba16f | Self::Rgba32f)
    }

    /// Whether this is a hardware-decoder-friendly format.
    pub fn is_hw_decodable(&self) -> bool {
        matches!(self, Self::Nv12 | Self::P010 | Self::Yuv420p)
    }
}

impl std::fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Video codec identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodecId {
    H264,
    H265,
    Av1,
    Vp9,
    ProRes,
    RawRgba,
}

impl std::fmt::Display for CodecId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::H264 => write!(f, "H.264/AVC"),
            Self::H265 => write!(f, "H.265/HEVC"),
            Self::Av1 => write!(f, "AV1"),
            Self::Vp9 => write!(f, "VP9"),
            Self::ProRes => write!(f, "ProRes"),
            Self::RawRgba => write!(f, "Raw RGBA"),
        }
    }
}

/// Container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContainerFormat {
    Mp4,
    Mkv,
    Webm,
    Mov,
    RawStream,
}
