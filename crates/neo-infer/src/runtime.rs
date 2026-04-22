use neo_core::NeoResult;
use neo_core::tensor::TensorDesc;

/// Backend for running inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBackend {
    /// wgpu compute shaders (universal, cross-platform).
    Wgpu,
    /// ONNX Runtime (optimized kernels, wide model support).
    OnnxRuntime,
    /// TensorRT (NVIDIA-only, maximum performance).
    TensorRt,
    /// CoreML (Apple Silicon, on-device).
    CoreMl,
}

/// The inference runtime — executes neural network forward passes on the GPU.
pub trait InferenceRuntime: Send {
    /// Get the backend type.
    fn backend(&self) -> RuntimeBackend;

    /// Run inference on VRAM buffers.
    ///
    /// `input_buffer_ids` and `output_buffer_ids` reference GPU buffers
    /// managed by `neo-gpu`. The runtime reads from inputs and writes
    /// to outputs — all in VRAM.
    fn run(
        &mut self,
        input_buffer_ids: &[u64],
        output_buffer_ids: &[u64],
    ) -> NeoResult<()>;

    /// Get input tensor descriptors.
    fn input_descs(&self) -> &[TensorDesc];

    /// Get output tensor descriptors.
    fn output_descs(&self) -> &[TensorDesc];
}

/// Wgpu-based inference runtime — runs compute shaders for neural network layers.
///
/// This is the universal backend that works on all platforms.
/// Models are expressed as sequences of compute shader dispatches.
pub struct WgpuRuntime {
    backend: RuntimeBackend,
    input_descs: Vec<TensorDesc>,
    output_descs: Vec<TensorDesc>,
}

impl WgpuRuntime {
    pub fn new(input_descs: Vec<TensorDesc>, output_descs: Vec<TensorDesc>) -> Self {
        Self {
            backend: RuntimeBackend::Wgpu,
            input_descs,
            output_descs,
        }
    }
}

impl InferenceRuntime for WgpuRuntime {
    fn backend(&self) -> RuntimeBackend {
        self.backend
    }

    fn run(
        &mut self,
        _input_buffer_ids: &[u64],
        _output_buffer_ids: &[u64],
    ) -> NeoResult<()> {
        // TODO: Execute the model's compute shader graph
        tracing::debug!("Wgpu inference pass (stub)");
        Ok(())
    }

    fn input_descs(&self) -> &[TensorDesc] {
        &self.input_descs
    }

    fn output_descs(&self) -> &[TensorDesc] {
        &self.output_descs
    }
}
