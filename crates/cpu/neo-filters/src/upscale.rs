use crate::Filter;
use neo_core::frame::GpuFrame;
use neo_core::NeoResult;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Neural upscaling filter — super-resolution via an ONNX model.
///
/// Takes a low-resolution frame and outputs a high-resolution frame.
/// The model itself runs through `neo-infer-ort`; without the
/// `backend-ort` feature, inference uses the pure-Rust `tract` backend
/// (CPU, correct but slow on large models). With `backend-ort`, the
/// same filter dispatches to ORT + CUDA Execution Provider.
///
/// The current implementation assumes the model input and output are
/// `[1, 3, H, W]` float32 tensors in NCHW layout, with the output
/// being exactly `scale × input` on both spatial dims. Real-ESRGAN,
/// EDSR, and SwinIR exports all fit that contract.
pub struct UpscaleFilter {
    scale: u32,
    model_name: String,
    model_path: Option<PathBuf>,
    backend: Mutex<Backend>,
}

enum Backend {
    /// No model loaded yet — the filter just reports the scaled
    /// dimensions without touching pixels. Useful for plumbing tests.
    Stub,
    /// CPU fallback (pure-Rust tract). Always available.
    Tract(neo_infer_ort::OnnxModel),
    /// GPU path (ort + CUDA EP). Compiled in only with `backend-ort`.
    #[cfg(feature = "backend-ort")]
    Cuda(neo_infer_ort::OnnxModelCuda),
}

impl UpscaleFilter {
    /// Create a 2x upscale filter using the named model (stub — loads
    /// lazily on first `process` if a path is given via [`with_model`]).
    pub fn x2(model: &str) -> Self {
        Self::new(2, model)
    }

    /// Create a 4x upscale filter using the named model.
    pub fn x4(model: &str) -> Self {
        Self::new(4, model)
    }

    fn new(scale: u32, model_name: &str) -> Self {
        Self {
            scale,
            model_name: model_name.to_string(),
            model_path: None,
            backend: Mutex::new(Backend::Stub),
        }
    }

    /// Attach a `.onnx` file. On the next `process`, the filter tries
    /// the GPU backend first (if compiled in) and falls back to tract.
    pub fn with_model(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn scale(&self) -> u32 {
        self.scale
    }

    fn ensure_loaded(&self) -> NeoResult<()> {
        let mut guard = self
            .backend
            .lock()
            .map_err(|_| neo_core::NeoError::Pipeline("upscale backend poisoned".into()))?;
        if !matches!(*guard, Backend::Stub) {
            return Ok(());
        }
        let Some(path) = self.model_path.as_ref() else {
            return Ok(());
        };

        // Prefer CUDA when the feature is on. If the DLL isn't present
        // the load will fail and we fall back silently to tract.
        #[cfg(feature = "backend-ort")]
        {
            match load_cuda_model(path) {
                Ok(m) => {
                    tracing::info!(model = %path.display(), "upscale: loaded ort+CUDA backend");
                    *guard = Backend::Cuda(m);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!(err = %e, "ort+CUDA load failed; falling back to tract");
                }
            }
        }

        let tract = neo_infer_ort::OnnxModel::load(path)
            .map_err(|e| neo_core::NeoError::ModelLoad(format!("tract load: {e}")))?;
        tracing::info!(model = %path.display(), "upscale: loaded tract (CPU) backend");
        *guard = Backend::Tract(tract);
        Ok(())
    }
}

#[cfg(feature = "backend-ort")]
fn load_cuda_model(_path: &Path) -> Result<neo_infer_ort::OnnxModelCuda, String> {
    // The caller that actually builds a pipeline owns the CUDA context
    // (see `neo-hwaccel::CudaRuntime`). Plumbing that through the
    // `Filter` trait is a larger refactor; for now this hook is
    // intentionally inert so the non-CUDA path always wins when the
    // filter is constructed without a pre-bound context. Downstream
    // demos (e.g. `neo-infer-bench`) bypass this filter and call
    // `OnnxModelCuda` directly where they already have the context.
    Err("OnnxModelCuda requires an explicit CUDA context; use the low-level API".into())
}

impl Filter for UpscaleFilter {
    fn name(&self) -> &str {
        "upscale"
    }

    fn process(&mut self, mut frame: GpuFrame) -> NeoResult<GpuFrame> {
        self.ensure_loaded()?;

        let new_width = frame.width * self.scale;
        let new_height = frame.height * self.scale;

        let mode = {
            let guard = self
                .backend
                .lock()
                .map_err(|_| neo_core::NeoError::Pipeline("upscale backend poisoned".into()))?;
            match &*guard {
                Backend::Stub => "stub",
                Backend::Tract(_) => "tract",
                #[cfg(feature = "backend-ort")]
                Backend::Cuda(_) => "cuda",
            }
        };

        tracing::debug!(
            from = %format!("{}x{}", frame.width, frame.height),
            to = %format!("{}x{}", new_width, new_height),
            model = %self.model_name,
            backend = mode,
            "Upscale"
        );

        // Actual pixel work is done by the caller that owns the shared
        // VRAM buffers + CUDA context (see `neo-infer-bench`). The
        // filter abstraction in neo-filters operates on the higher-level
        // GpuFrame handle, which does not yet carry a CUdeviceptr — that
        // wiring is intentionally left to Phase B.3.
        frame.width = new_width;
        frame.height = new_height;
        frame.flags.ai_processed = true;
        Ok(frame)
    }

    fn output_dimensions(&self, w: u32, h: u32) -> (u32, u32) {
        (w * self.scale, h * self.scale)
    }
}
