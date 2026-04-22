//! # neo-infer-ort
//!
//! Minimal ONNX inference bridge for Neo-FFmpeg.
//!
//! Despite the crate name (kept stable so callers don't have to change),
//! the actual backend in Phase B.1 is **tract** — a pure-Rust ONNX
//! runtime. The reason: `ort` 2.0.0-rc.12 doesn't ship prebuilt binaries
//! for the `x86_64-pc-windows-gnu` toolchain Neo builds with, and
//! pulling in MSVC just for the bridge would break the
//! "no SDK install" property of the project. tract is slower for large
//! models but it gets us to a working end-to-end inference path **today**
//! with zero native dependencies.
//!
//! Phase B.2 will introduce a feature flag (`backend-ort` /
//! `backend-tract`) so users on MSVC who want CUDA acceleration can
//! switch to ort+CUDA without touching the rest of the chain.
//!
//! ## Public surface
//!
//! - [`OnnxModel::load`] — load a `.onnx` file, validate exactly one
//!   f32 input and one f32 output, freeze a concrete shape.
//! - [`OnnxModel::infer`] — run on a flat `&[f32]` and get a flat
//!   `Vec<f32>` back.
//! - [`OnnxModel::input_shape`] / [`OnnxModel::output_shape`] — for the
//!   chain to size its bridge buffers.
//!
//! Higher-level "image-in image-out" wrappers live in `neo-lab`.

use std::{path::Path, sync::Mutex};
use thiserror::Error;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::DimLike;
use tracing::{debug, info};

pub mod generate;
pub mod wgpu_plan;

#[cfg(feature = "backend-ort")]
pub mod cuda_backend;

#[cfg(feature = "backend-ort")]
pub use cuda_backend::{CudaInferError, OnnxModelCuda};

pub use wgpu_plan::{
    try_compile_from_onnx_bytes, UnaryKind, WgpuPlanDescription, WgpuPlanOp,
};

#[derive(Debug, Error)]
pub enum InferError {
    #[error("tract error: {0}")]
    Tract(String),
    #[error("model has {0} inputs, expected exactly 1")]
    BadInputCount(usize),
    #[error("model has {0} outputs, expected exactly 1")]
    BadOutputCount(usize),
    #[error("input element type is not f32 (got {0})")]
    BadInputDtype(String),
    #[error("input has dynamic dimensions; please specify them at load time: {0:?}")]
    DynamicInput(Vec<TDim>),
    #[error("input length mismatch: got {got}, expected {want}")]
    LengthMismatch { got: usize, want: usize },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, InferError>;

fn map_tract(e: impl std::fmt::Display) -> InferError {
    InferError::Tract(e.to_string())
}

/// Type alias for the optimized typed runnable model that tract produces
/// after `into_runnable()`. We hold one of these per [`OnnxModel`].
type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// A loaded ONNX model with cached input/output metadata.
pub struct OnnxModel {
    /// The runnable graph. tract `run()` takes `&self` so we don't even
    /// need a Mutex around the model itself, but we wrap it anyway in
    /// case we want to add per-call state later.
    runnable: Mutex<RunnableModel>,
    /// Concrete input shape (no symbolic dimensions). Stage B.1 requires
    /// callers to know this in advance.
    input_shape: Vec<usize>,
    /// Resolved on the first successful run.
    output_shape: Mutex<Option<Vec<usize>>>,
    /// Raw ONNX bytes kept around so the wgpu pattern matcher can walk
    /// the original (pre-optimization) graph. ~hundreds of bytes for
    /// the small models we generate; not worth optimising away.
    raw_bytes: Vec<u8>,
}

impl OnnxModel {
    /// Load `path`, validate that the model has exactly one f32 input
    /// and one f32 output, freeze the input shape, and produce a
    /// runnable plan.
    pub fn load(path: &Path) -> Result<Self> {
        info!(model = %path.display(), "loading ONNX model (tract backend)");
        let raw_bytes = std::fs::read(path)?;
        Self::load_from_bytes(raw_bytes)
    }

    /// Load from already-resident ONNX bytes. Used by tests that don't
    /// want to touch the filesystem and by callers who fetched a model
    /// from somewhere else (network, embedded resource, etc.).
    pub fn load_from_bytes(raw_bytes: Vec<u8>) -> Result<Self> {
        // 1. Parse + build an InferenceModel (loose facts).
        let inference = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(&raw_bytes))
            .map_err(|e| InferError::Tract(format!("model_for_read: {e}")))?;

        if inference.input_outlets().map_err(map_tract)?.len() != 1 {
            return Err(InferError::BadInputCount(
                inference.input_outlets().map_err(map_tract)?.len(),
            ));
        }
        if inference.output_outlets().map_err(map_tract)?.len() != 1 {
            return Err(InferError::BadOutputCount(
                inference.output_outlets().map_err(map_tract)?.len(),
            ));
        }

        // 2. Promote to TypedModel — at this point dims are concrete and
        //    facts are TypedFact (not InferenceFact), giving us proper
        //    DatumType + ShapeFact accessors.
        let typed = inference
            .into_typed()
            .map_err(|e| InferError::Tract(format!("into_typed: {e}")))?
            .into_decluttered()
            .map_err(|e| InferError::Tract(format!("into_decluttered: {e}")))?;

        let in_fact = typed.input_fact(0).map_err(map_tract)?;
        if in_fact.datum_type != DatumType::F32 {
            return Err(InferError::BadInputDtype(format!("{:?}", in_fact.datum_type)));
        }

        // 3. Resolve shape to concrete usize. ShapeFact gives `TDim`s.
        let dims: Vec<TDim> = in_fact.shape.dims().to_vec();
        let mut concrete: Vec<usize> = Vec::with_capacity(dims.len());
        for d in &dims {
            match d.to_usize() {
                Ok(v) => concrete.push(v),
                Err(_) => return Err(InferError::DynamicInput(dims.clone())),
            }
        }

        let runnable = typed
            .into_runnable()
            .map_err(|e| InferError::Tract(format!("into_runnable: {e}")))?;

        info!(shape = ?concrete, "ONNX model ready");

        Ok(Self {
            runnable: Mutex::new(runnable),
            input_shape: concrete,
            output_shape: Mutex::new(None),
            raw_bytes,
        })
    }

    /// Try to compile this ONNX graph into a wgpu-friendly plan.
    /// Returns `None` if any operator is outside the supported subset
    /// (in which case callers should fall back to [`infer`]).
    pub fn try_wgpu_plan(&self) -> Option<wgpu_plan::WgpuPlanDescription> {
        wgpu_plan::try_compile_from_onnx_bytes(&self.raw_bytes)
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub fn output_shape(&self) -> Option<Vec<usize>> {
        self.output_shape.lock().unwrap().clone()
    }

    /// Total number of input elements (product of input shape).
    pub fn input_len(&self) -> usize {
        self.input_shape.iter().product()
    }

    /// Run inference on a flat float input. Returns the flat float
    /// output. The shape of the output is cached for the chain layer
    /// after the first call (some models have dynamic output dims that
    /// can only be known post-inference).
    pub fn infer(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_len() {
            return Err(InferError::LengthMismatch {
                got: input.len(),
                want: self.input_len(),
            });
        }

        // Build a tract Tensor from the flat slice + shape.
        let tensor: Tensor = tract_ndarray::Array::from_shape_vec(
            self.input_shape.as_slice(),
            input.to_vec(),
        )
        .map_err(|e| InferError::Tract(format!("ndarray reshape: {e}")))?
        .into();

        let runnable = self
            .runnable
            .lock()
            .map_err(|_| InferError::Tract("model mutex poisoned".into()))?;

        let outputs = runnable
            .run(tvec!(tensor.into()))
            .map_err(|e| InferError::Tract(format!("run: {e}")))?;

        let out = outputs
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Tract("model produced no outputs".into()))?;

        let arr = out
            .to_array_view::<f32>()
            .map_err(|e| InferError::Tract(format!("to_array_view::<f32>: {e}")))?;

        let shape: Vec<usize> = arr.shape().to_vec();
        let flat: Vec<f32> = arr.iter().copied().collect();

        if let Ok(mut out_shape) = self.output_shape.lock() {
            if out_shape.is_none() {
                debug!(?shape, "output shape resolved");
                *out_shape = Some(shape);
            }
        }
        Ok(flat)
    }
}
