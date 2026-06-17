use crate::model::{ModelFormat, ModelInfo, ModelTask};
use crate::runtime::{InferenceRuntime, RuntimeBackend, WgpuRuntime};
use neo_core::tensor::{DataType, TensorDesc};
use neo_core::{NeoError, NeoResult};
use std::path::Path;

/// An inference session — a loaded model ready to process frames.
///
/// The session holds the model weights in VRAM and the compiled
/// compute pipeline. Call `run()` to process a frame.
pub struct InferenceSession {
    pub info: ModelInfo,
    runtime: Box<dyn InferenceRuntime>,
}

impl InferenceSession {
    /// Load a model from a file.
    pub fn load(path: &Path, backend: RuntimeBackend) -> NeoResult<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let format = match ext {
            "neo" => ModelFormat::Neo,
            "onnx" => ModelFormat::Onnx,
            "safetensors" => ModelFormat::SafeTensors,
            _ => return Err(NeoError::ModelLoad(format!("unknown model format: .{ext}"))),
        };

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // TODO: Actually parse the model file and load weights into VRAM.
        // For now, create a stub session.
        let info = ModelInfo {
            name,
            task: ModelTask::SuperResolution,
            format,
            inputs: vec![TensorDesc::frame_nchw(3, 1080, 1920, DataType::F16)],
            outputs: vec![TensorDesc::frame_nchw(3, 2160, 3840, DataType::F16)],
            weight_size: 0,
            scale_factor: Some(2),
        };

        let runtime: Box<dyn InferenceRuntime> = match backend {
            RuntimeBackend::Wgpu => Box::new(WgpuRuntime::new(
                info.inputs.clone(),
                info.outputs.clone(),
            )),
            other => {
                return Err(NeoError::ModelLoad(format!(
                    "backend {other:?} not yet implemented"
                )))
            }
        };

        tracing::info!(
            model = %info.name,
            task = ?info.task,
            format = ?info.format,
            "Model loaded"
        );

        Ok(Self { info, runtime })
    }

    /// Run inference on GPU buffers (zero-copy).
    pub fn run(
        &mut self,
        input_buffer_ids: &[u64],
        output_buffer_ids: &[u64],
    ) -> NeoResult<()> {
        self.runtime.run(input_buffer_ids, output_buffer_ids)
    }

    /// Get the model task.
    pub fn task(&self) -> ModelTask {
        self.info.task
    }

    /// Get the scale factor (for super-resolution).
    pub fn scale_factor(&self) -> Option<u32> {
        self.info.scale_factor
    }
}
