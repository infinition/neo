use serde::{Deserialize, Serialize};

/// Data type for tensor elements.
///
/// Neo-FFmpeg treats video frames as tensors — this is the bridge
/// between "pixel data" and "AI-ready data".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    U8,
    U16,
    F16,
    F32,
    Bf16,
}

impl DataType {
    /// Size in bytes of a single element.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Memory layout for a tensor buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorLayout {
    /// Height x Width x Channels — natural image layout (HWC).
    /// Used by most image processing / GPU texture operations.
    Hwc,
    /// Channels x Height x Width — PyTorch convention (CHW).
    /// Used by most neural networks.
    Chw,
    /// Batch x Channels x Height x Width — batched inference (NCHW).
    Nchw,
    /// Batch x Height x Width x Channels — TensorFlow convention (NHWC).
    Nhwc,
}

/// Describes the shape and layout of a tensor in VRAM.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorDesc {
    /// Dimensions (e.g., [1, 3, 1080, 1920] for NCHW).
    pub shape: Vec<u32>,
    /// Data type of each element.
    pub dtype: DataType,
    /// Memory layout convention.
    pub layout: TensorLayout,
}

impl TensorDesc {
    pub fn new(shape: Vec<u32>, dtype: DataType, layout: TensorLayout) -> Self {
        Self { shape, dtype, layout }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Total size in bytes.
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.byte_size()
    }

    /// Create a standard NCHW tensor desc for a single frame.
    pub fn frame_nchw(channels: u32, height: u32, width: u32, dtype: DataType) -> Self {
        Self {
            shape: vec![1, channels, height, width],
            dtype,
            layout: TensorLayout::Nchw,
        }
    }
}
