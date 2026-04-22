//! # CUDA-resident ONNX inference (ort backend, `backend-ort` feature)
//!
//! Phase B.2: true GPU inference with near-zero CPU round-trip, using
//! the Rust `ort` crate wired to ONNX Runtime's CUDA Execution Provider.
//!
//! Supports N inputs / 1 output — single-input models (Real-ESRGAN,
//! EDSR, SwinIR) and multi-input models (RIFE: img0 + img1 + timestep,
//! IFNet, etc.) go through the same API.
//!
//! ## Two tiers
//!
//! - [`OnnxModelCuda::infer`]: **safe path** — takes host `&[&[f32]]`,
//!   returns a host `Vec<f32>`. Inference runs on GPU; ort handles
//!   HtoD/DtoH internally. Useful for tests and non-pipelined callers.
//!
//! - [`OnnxModelCuda::infer_on_device`]: **zero-copy input, near-zero
//!   copy output** — each input is a raw `CUdeviceptr` (wrapped as a
//!   [`TensorRefMut`] via `TensorRefMut::from_raw`, aliasing the
//!   caller's VRAM directly). Output is allocated by ort on CUDA and
//!   then copied device-to-device into the caller's `out_dptr` (a
//!   single intra-GPU `cuMemcpyDtoD`, typically <200 µs for 1080p RGB
//!   f32). No host bounce.
//!
//! ## Runtime setup
//!
//! `ort` is pulled in with `load-dynamic`; before first use, point at
//! the ONNX Runtime library:
//!
//! ```text
//! # Windows
//! set ORT_DYLIB_PATH=C:\onnxruntime\lib\onnxruntime.dll
//! # (and make sure onnxruntime_providers_cuda.dll is on PATH)
//!
//! # Linux
//! export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
//! ```

#![cfg(feature = "backend-ort")]

use cudarc::driver::{
    sys::{self as cuda_sys, CUdeviceptr, CUresult},
    CudaContext,
};
use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{builder::GraphOptimizationLevel, Session},
    tensor::Shape,
    value::{DynTensorValueType, Tensor, TensorRefMut, ValueType},
};
use std::{path::Path, sync::Arc};
use thiserror::Error;
use tracing::{debug, info};

#[derive(Debug, Error)]
pub enum CudaInferError {
    #[error("ort: {0}")]
    Ort(String),
    #[error("cuda: {0}")]
    Cuda(String),
    #[error("model must have at least one tensor input and exactly one tensor output")]
    BadArity,
    #[error("model input/output shape has unsupported dims: {0:?}")]
    BadShape(Vec<i64>),
    #[error("model input/output tensor is not f32")]
    NotF32,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("expected {expected} input pointers, got {got}")]
    InputArityMismatch { expected: usize, got: usize },
    #[error("operation requires fully static tensor shapes, got {0:?}")]
    DynamicShapeUnsupported(Vec<i64>),
    #[error("output buffer too small: need {need} f32 values, have {have}")]
    OutputBufferTooSmall { need: usize, have: usize },
}

pub type Result<T> = std::result::Result<T, CudaInferError>;

fn map_ort(e: impl std::fmt::Display) -> CudaInferError {
    CudaInferError::Ort(e.to_string())
}

fn cuda_check(r: CUresult, ctx: &'static str) -> Result<()> {
    if r == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaInferError::Cuda(format!("{ctx}: {r:?}")))
    }
}

pub struct OnnxModelCuda {
    session: Session,
    input_names: Vec<String>,
    output_name: String,
    input_shapes: Vec<Vec<i64>>,
    output_shape: Vec<i64>,
    input_lens: Vec<Option<usize>>,
    output_len: Option<usize>,
    cuda_ctx: Arc<CudaContext>,
    device_id: i32,
}

impl OnnxModelCuda {
    /// Load an ONNX model with the CUDA Execution Provider.
    ///
    /// `cuda_ctx` is the driver context Neo owns (see
    /// `neo_hwaccel::CudaRuntime::ctx`). `device_id` is the ordinal —
    /// usually `0`.
    pub fn load(path: &Path, cuda_ctx: Arc<CudaContext>, device_id: i32) -> Result<Self> {
        // `ort::init` is idempotent; first caller wins. Subsequent
        // calls return an error we intentionally ignore so multiple
        // crates can call `load` independently.
        let _ = ort::init().commit();

        // EP priority: TensorRT first (accepts full ONNX, falls back op-by-op
        // to CUDA for unsupported ops), then CUDA EP as safety net. TRT EP
        // is NOT marked `error_on_failure` so ORT cleanly falls back to the
        // CUDA EP when the TensorRT provider DLL or TRT runtime is missing.
        //
        // The TRT EP can be disabled at runtime with `NEO_DISABLE_TRT=1`
        // (useful for debugging or when TRT's long first-run JIT is
        // unwanted). Engines are cached per-model in `NEO_TRT_CACHE_DIR`
        // (defaults to `./trt_cache`) so subsequent loads skip the build.
        let use_trt = std::env::var("NEO_DISABLE_TRT").ok().as_deref() != Some("1");
        let trt_cache = std::env::var("NEO_TRT_CACHE_DIR")
            .unwrap_or_else(|_| "./trt_cache".to_string());
        if use_trt {
            info!(
                model = %path.display(),
                device_id,
                trt_cache = %trt_cache,
                "loading ONNX model (ort + TensorRT EP with CUDA EP fallback)"
            );
        } else {
            info!(
                model = %path.display(),
                device_id,
                "loading ONNX model (ort + CUDA EP only, TRT disabled)"
            );
        }

        let cuda_ep = CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .build()
            .error_on_failure();

        let session_builder = Session::builder()
            .map_err(map_ort)?
            .with_no_environment_execution_providers()
            .map_err(map_ort)?;

        let session_builder = if use_trt {
            let _ = std::fs::create_dir_all(&trt_cache);
            let trt_ep = TensorRTExecutionProvider::default()
                .with_device_id(device_id)
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path(trt_cache.clone())
                .with_timing_cache(true)
                .with_timing_cache_path(trt_cache)
                .with_builder_optimization_level(3)
                .build();
            session_builder
                .with_execution_providers([trt_ep, cuda_ep])
                .map_err(map_ort)?
        } else {
            session_builder
                .with_execution_providers([cuda_ep])
                .map_err(map_ort)?
        };

        let session = session_builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(map_ort)?
            .commit_from_file(path)
            .map_err(map_ort)?;

        if session.inputs.is_empty() || session.outputs.len() != 1 {
            return Err(CudaInferError::BadArity);
        }

        let input_names: Vec<String> = session.inputs.iter().map(|i| i.name.clone()).collect();
        let output_name = session.outputs[0].name.clone();
        let input_shapes: Vec<Vec<i64>> = session
            .inputs
            .iter()
            .map(|i| extract_shape(&i.input_type))
            .collect::<Result<_>>()?;
        let output_shape = extract_shape(&session.outputs[0].output_type)?;

        let input_lens: Vec<Option<usize>> = input_shapes.iter().map(|s| fixed_len(s)).collect();
        let output_len: Option<usize> = fixed_len(&output_shape);

        debug!(
            input_names = ?input_names,
            output_name = %output_name,
            input_shapes = ?input_shapes,
            output_shape = ?output_shape,
            "ort+CUDA session ready"
        );

        Ok(Self {
            session,
            input_names,
            output_name,
            input_shapes,
            output_shape,
            input_lens,
            output_len,
            cuda_ctx,
            device_id,
        })
    }

    pub fn input_count(&self) -> usize {
        self.input_names.len()
    }
    pub fn input_name(&self, i: usize) -> &str {
        &self.input_names[i]
    }
    pub fn input_shape(&self, i: usize) -> &[i64] {
        &self.input_shapes[i]
    }
    pub fn input_is_dynamic(&self, i: usize) -> bool {
        self.input_lens[i].is_none()
    }
    pub fn input_len(&self, i: usize) -> usize {
        self.input_lens[i].unwrap_or(0)
    }
    pub fn input_byte_size(&self, i: usize) -> usize {
        self.input_lens[i].unwrap_or(0) * std::mem::size_of::<f32>()
    }

    pub fn output_name(&self) -> &str {
        &self.output_name
    }
    pub fn output_shape(&self) -> &[i64] {
        &self.output_shape
    }
    pub fn output_is_dynamic(&self) -> bool {
        self.output_len.is_none()
    }
    pub fn output_len(&self) -> usize {
        self.output_len.unwrap_or(0)
    }
    pub fn output_byte_size(&self) -> usize {
        self.output_len.unwrap_or(0) * std::mem::size_of::<f32>()
    }
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
    pub fn is_fully_static(&self) -> bool {
        self.input_lens.iter().all(|l| l.is_some()) && self.output_len.is_some()
    }

    /// Safe-path inference — host slices in, host `Vec<f32>` out.
    /// Inference itself runs on GPU via the CUDA EP; ort performs the
    /// HtoD / DtoH copies.
    pub fn infer(&mut self, inputs: &[&[f32]]) -> Result<Vec<f32>> {
        if !self.input_lens.iter().all(|l| l.is_some()) {
            let shape = self
                .input_shapes
                .iter()
                .find(|s| fixed_len(s).is_none())
                .cloned()
                .unwrap_or_default();
            return Err(CudaInferError::DynamicShapeUnsupported(shape));
        }
        if inputs.len() != self.input_names.len() {
            return Err(CudaInferError::InputArityMismatch {
                expected: self.input_names.len(),
                got: inputs.len(),
            });
        }
        for (i, input) in inputs.iter().enumerate() {
            let want = self.input_lens[i].unwrap();
            if input.len() != want {
                return Err(CudaInferError::Ort(format!(
                    "input[{}] length {} != expected {}",
                    i,
                    input.len(),
                    want
                )));
            }
        }

        let mut session_inputs: Vec<(std::borrow::Cow<'static, str>, Tensor<f32>)> =
            Vec::with_capacity(inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            let shape: Vec<i64> = self.input_shapes[i].clone();
            let tensor = Tensor::<f32>::from_array((shape, input.to_vec())).map_err(map_ort)?;
            session_inputs.push((self.input_names[i].clone().into(), tensor));
        }

        let outputs = self.session.run(session_inputs).map_err(map_ort)?;

        let (_shape, data) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(map_ort)?;
        Ok(data.to_vec())
    }

    /// Host-path inference with explicit runtime shapes.
    ///
    /// This is used for models whose metadata contains dynamic spatial
    /// dims (for example RIFE with `[-1, -1]`). Callers provide the
    /// concrete shape for each input at runtime.
    pub fn infer_dynamic(
        &mut self,
        inputs: &[(&[f32], &[i64])],
    ) -> Result<(Vec<f32>, Vec<i64>)> {
        if inputs.len() != self.input_names.len() {
            return Err(CudaInferError::InputArityMismatch {
                expected: self.input_names.len(),
                got: inputs.len(),
            });
        }

        let mut session_inputs: Vec<(std::borrow::Cow<'static, str>, Tensor<f32>)> =
            Vec::with_capacity(inputs.len());
        for (i, (input, runtime_shape)) in inputs.iter().enumerate() {
            validate_runtime_shape(&self.input_shapes[i], runtime_shape)?;
            let want = fixed_len(runtime_shape).ok_or_else(|| {
                CudaInferError::BadShape(runtime_shape.to_vec())
            })?;
            if input.len() != want {
                return Err(CudaInferError::Ort(format!(
                    "input[{}] length {} != expected {} for runtime shape {:?}",
                    i,
                    input.len(),
                    want,
                    runtime_shape
                )));
            }
            let tensor =
                Tensor::<f32>::from_array((runtime_shape.to_vec(), input.to_vec())).map_err(map_ort)?;
            session_inputs.push((self.input_names[i].clone().into(), tensor));
        }

        let outputs = self.session.run(session_inputs).map_err(map_ort)?;
        let (shape, data) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(map_ort)?;
        let out_shape: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        Ok((data.to_vec(), out_shape))
    }

    /// Zero-copy input, near-zero-copy output.
    ///
    /// The model reads directly from each `input_dptrs[i]` (our VRAM
    /// aliased by `TensorRefMut::from_raw`). Output is allocated by ort
    /// on CUDA and copied device-to-device into `output_dptr` once
    /// inference finishes. Synchronous: both data flows are drained
    /// before return, so any other API aliasing the same VRAM (wgpu,
    /// NVENC) sees the result immediately.
    pub fn infer_on_device(
        &mut self,
        input_dptrs: &[CUdeviceptr],
        output_dptr: CUdeviceptr,
    ) -> Result<()> {
        if !self.is_fully_static() {
            let shape = self
                .input_shapes
                .iter()
                .find(|s| fixed_len(s).is_none())
                .cloned()
                .unwrap_or_else(|| self.output_shape.clone());
            return Err(CudaInferError::DynamicShapeUnsupported(shape));
        }
        if input_dptrs.len() != self.input_names.len() {
            return Err(CudaInferError::InputArityMismatch {
                expected: self.input_names.len(),
                got: input_dptrs.len(),
            });
        }

        self.cuda_ctx
            .bind_to_thread()
            .map_err(|e| CudaInferError::Cuda(format!("bind_to_thread: {e:?}")))?;

        // Capture everything we need as locals BEFORE taking a mutable
        // borrow on `self.session` (the `outputs` value from
        // `run_binding` holds that borrow until it's dropped).
        let in_names = self.input_names.clone();
        let in_shapes = self.input_shapes.clone();
        let out_name = self.output_name.clone();
        let out_bytes = self.output_byte_size();
        let device_id = self.device_id;

        // --- Inputs: wrap each caller VRAM region as a CUDA tensor (no copy).
        let mut input_tensors: Vec<TensorRefMut<'_, f32>> = Vec::with_capacity(in_names.len());
        for (i, &dptr) in input_dptrs.iter().enumerate() {
            let mi = MemoryInfo::new(
                AllocationDevice::CUDA,
                device_id,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(map_ort)?;
            let shape: Shape = in_shapes[i].clone().into();
            let t = unsafe {
                TensorRefMut::<f32>::from_raw(mi, dptr as *mut _, shape).map_err(map_ort)?
            };
            input_tensors.push(t);
        }

        // --- Output: pre-allocate a CUDA tensor in the session's allocator
        // and bind it directly. This avoids `SessionOutputs` re-materializing
        // a value in a way that can trigger an implicit device->CPU copy.
        let out_mi = MemoryInfo::new(
            AllocationDevice::CUDA,
            device_id,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(map_ort)?;
        let out_allocator = Allocator::new(&self.session, out_mi).map_err(map_ort)?;
        let out_tensor =
            Tensor::<f32>::new(&out_allocator, Shape::from(self.output_shape.clone())).map_err(map_ort)?;
        let ort_out_dptr = out_tensor.data_ptr() as u64;

        let mut binding = self.session.create_binding().map_err(map_ort)?;
        for (name, tensor) in in_names.iter().zip(input_tensors.iter()) {
            binding
                .bind_input(name.as_str(), &**tensor)
                .map_err(map_ort)?;
        }
        binding
            .bind_output(out_name.as_str(), out_tensor)
            .map_err(map_ort)?;

        self.session.run_binding(&binding).map_err(map_ort)?;
        unsafe {
            cuda_check(
                cuda_sys::cuMemcpyDtoD_v2(output_dptr, ort_out_dptr, out_bytes),
                "cuMemcpyDtoD(out)",
            )?;
            cuda_check(cuda_sys::cuCtxSynchronize(), "cuCtxSynchronize")?;
        }

        Ok(())
    }

    /// Device-path inference for models with runtime spatial shapes.
    ///
    /// Inputs are read directly from caller-owned CUDA memory via
    /// `TensorRefMut::from_raw`, exactly like [`infer_on_device`], but
    /// each input's concrete runtime shape is provided explicitly.
    ///
    /// The output is allocated by ONNX Runtime on the CUDA device using
    /// `bind_output_to_device`, then copied once device-to-device into
    /// `output_dptr`. Returns the concrete output shape so callers can
    /// interpret the written tensor without any host extraction.
    pub fn infer_on_device_dynamic(
        &mut self,
        input_dptrs: &[CUdeviceptr],
        input_shapes: &[&[i64]],
        output_dptr: CUdeviceptr,
        output_capacity_f32: usize,
    ) -> Result<Vec<i64>> {
        if input_dptrs.len() != self.input_names.len() {
            return Err(CudaInferError::InputArityMismatch {
                expected: self.input_names.len(),
                got: input_dptrs.len(),
            });
        }
        if input_shapes.len() != self.input_names.len() {
            return Err(CudaInferError::InputArityMismatch {
                expected: self.input_names.len(),
                got: input_shapes.len(),
            });
        }

        self.cuda_ctx
            .bind_to_thread()
            .map_err(|e| CudaInferError::Cuda(format!("bind_to_thread: {e:?}")))?;

        let in_names = self.input_names.clone();
        let model_shapes = self.input_shapes.clone();
        let out_name = self.output_name.clone();
        let device_id = self.device_id;

        let mut input_tensors: Vec<TensorRefMut<'_, f32>> = Vec::with_capacity(in_names.len());
        for i in 0..in_names.len() {
            validate_runtime_shape(&model_shapes[i], input_shapes[i])?;
            let mi = MemoryInfo::new(
                AllocationDevice::CUDA,
                device_id,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(map_ort)?;
            let shape: Shape = input_shapes[i].to_vec().into();
            let t = unsafe {
                TensorRefMut::<f32>::from_raw(mi, input_dptrs[i] as *mut _, shape).map_err(map_ort)?
            };
            input_tensors.push(t);
        }

        let out_mi = MemoryInfo::new(
            AllocationDevice::CUDA,
            device_id,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(map_ort)?;

        let mut binding = self.session.create_binding().map_err(map_ort)?;
        for (name, tensor) in in_names.iter().zip(input_tensors.iter()) {
            binding
                .bind_input(name.as_str(), &**tensor)
                .map_err(map_ort)?;
        }
        binding
            .bind_output_to_device(out_name.as_str(), &out_mi)
            .map_err(map_ort)?;

        let outputs = self.session.run_binding(&binding).map_err(map_ort)?;
        let out_value = outputs
            .get(out_name.as_str())
            .ok_or_else(|| CudaInferError::Ort(format!("missing bound output '{}'", out_name)))?;
        let out_tensor = out_value
            .downcast_ref::<DynTensorValueType>()
            .map_err(map_ort)?;
        let out_shape: Vec<i64> = out_tensor.shape().iter().copied().collect();
        let out_len = fixed_len(&out_shape).ok_or_else(|| CudaInferError::BadShape(out_shape.clone()))?;
        if out_len > output_capacity_f32 {
            return Err(CudaInferError::OutputBufferTooSmall {
                need: out_len,
                have: output_capacity_f32,
            });
        }

        unsafe {
            cuda_check(
                cuda_sys::cuMemcpyDtoD_v2(
                    output_dptr,
                    out_tensor.data_ptr() as u64,
                    out_len * std::mem::size_of::<f32>(),
                ),
                "cuMemcpyDtoD(dynamic out)",
            )?;
            cuda_check(cuda_sys::cuCtxSynchronize(), "cuCtxSynchronize")?;
        }

        Ok(out_shape)
    }
}

/// Parse a tensor shape, allowing dynamic dims (`-1`) but still rejecting
/// non-f32 element types and zero dims.
fn extract_shape(t: &ValueType) -> Result<Vec<i64>> {
    match t {
        ValueType::Tensor { ty, shape, .. } => {
            if *ty != ort::tensor::TensorElementType::Float32 {
                return Err(CudaInferError::NotF32);
            }
            let dims: Vec<i64> = shape.iter().copied().collect();
            if dims.is_empty() || dims.iter().any(|&d| d == 0 || d < -1) {
                return Err(CudaInferError::BadShape(dims));
            }
            Ok(dims)
        }
        _ => Err(CudaInferError::Ort("not a tensor type".into())),
    }
}

fn fixed_len(shape: &[i64]) -> Option<usize> {
    if shape.iter().any(|&d| d <= 0) {
        return None;
    }
    Some(shape.iter().map(|&d| d as usize).product())
}

fn validate_runtime_shape(model_shape: &[i64], runtime_shape: &[i64]) -> Result<()> {
    if model_shape.len() != runtime_shape.len() {
        return Err(CudaInferError::BadShape(runtime_shape.to_vec()));
    }
    for (&model_dim, &runtime_dim) in model_shape.iter().zip(runtime_shape.iter()) {
        if runtime_dim <= 0 {
            return Err(CudaInferError::BadShape(runtime_shape.to_vec()));
        }
        if model_dim > 0 && model_dim != runtime_dim {
            return Err(CudaInferError::Ort(format!(
                "runtime shape {:?} does not match model shape {:?}",
                runtime_shape, model_shape
            )));
        }
    }
    Ok(())
}
