use neo_core::tensor::TensorDesc;
use serde::{Deserialize, Serialize};

/// Model format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// Neo-FFmpeg native model (pre-compiled shaders + weights).
    Neo,
    /// ONNX model (requires ONNX Runtime backend).
    Onnx,
    /// SafeTensors weights (used with shader-defined architectures).
    SafeTensors,
}

/// Information about a loaded model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Human-readable model name.
    pub name: String,
    /// Model task (e.g., "super-resolution", "denoise", "style-transfer").
    pub task: ModelTask,
    /// Model format.
    pub format: ModelFormat,
    /// Input tensor descriptors.
    pub inputs: Vec<TensorDesc>,
    /// Output tensor descriptors.
    pub outputs: Vec<TensorDesc>,
    /// Model size in bytes (weights only).
    pub weight_size: u64,
    /// Scale factor (for super-resolution models).
    pub scale_factor: Option<u32>,
}

/// What task the model performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTask {
    /// Upscale (Super Resolution): 1080p → 4K
    SuperResolution,
    /// Denoise: remove noise/grain
    Denoise,
    /// Frame Interpolation: generate intermediate frames
    FrameInterpolation,
    /// Style Transfer: apply artistic style
    StyleTransfer,
    /// Segmentation: detect and mask objects
    Segmentation,
    /// Depth Estimation: generate depth maps
    DepthEstimation,
    /// Colorization: add color to B&W footage
    Colorization,
    /// Face Restoration: enhance/restore faces
    FaceRestoration,
    /// Background Removal: alpha matte generation
    BackgroundRemoval,
    /// Optical Flow: motion estimation between frames
    OpticalFlow,
    /// Generic: user-defined model
    Generic,
}

impl ModelTask {
    /// Whether this task requires multiple input frames.
    pub fn is_temporal(&self) -> bool {
        matches!(self, Self::FrameInterpolation | Self::OpticalFlow)
    }

    /// Whether this task changes the output resolution.
    pub fn changes_resolution(&self) -> bool {
        matches!(self, Self::SuperResolution)
    }
}
