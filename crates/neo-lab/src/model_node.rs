//! Post-chain ONNX inference node.
//!
//! Sits *after* the WGSL shader chain: reads the final BGRA buffer
//! (either `buf_a` or `buf_b` depending on parity), runs it through an
//! [`OnnxModel`], and writes the result back into the same buffer so the
//! blit pipeline picks it up transparently.
//!
//! ## Stage B.1 = CPU bounce
//!
//! Per frame:
//!
//! 1. `copy_buffer_to_buffer` BGRA → staging (MAP_READ)
//! 2. Map staging, unpack each `u32` into `(R, G, B)` f32 in [0, 1]
//! 3. Transpose to NCHW planar layout (`[1, 3, H, W]`)
//! 4. Call [`OnnxModel::infer`]
//! 5. Clamp output back to u8, re-pack to BGRA u32
//! 6. `queue.write_buffer` into the same BGRA buffer
//!
//! This eats a pair of PCIe round-trips per frame and is **slow on
//! purpose**: the goal is to prove the end-to-end bridge works with a
//! real ONNX model on Neo's video pipeline. Stage B.2 will replace the
//! staging/write with a CUDA IOBinding so the tensor never leaves VRAM.
//!
//! ## Supported model signature
//!
//! The model must declare:
//!
//! - Exactly one f32 input, shape `[1, 3, H, W]` (NCHW, RGB, 0–1 range)
//! - Exactly one f32 output, same shape
//!
//! `H` and `W` must match the video resolution exactly — Stage B.1 does
//! not resize. If you want super-resolution or stylization at an
//! arbitrary resolution, wrap the model with input/output reshapes first
//! (Netron + onnx-simplifier).

use crate::wgpu_infer::WgpuInferenceEngine;
use neo_core::{NeoError, NeoResult};
use neo_infer_ort::OnnxModel;
use std::{path::Path, sync::Arc, time::Instant};
use tracing::{info, warn};

pub struct ModelNode {
    model: Arc<OnnxModel>,
    width: u32,
    height: u32,

    /// Phase B.2a fast path: if the loaded ONNX model only contains
    /// supported ops, we keep a `WgpuInferenceEngine` here and skip the
    /// CPU bounce entirely. `None` means we fell back to tract.
    gpu_engine: Option<WgpuInferenceEngine>,

    /// Staging buffer for GPU→CPU download of the BGRA pixels.
    /// (Phase B.1 CPU-bounce path; only used when `gpu_engine == None`.)
    download_staging: wgpu::Buffer,
    /// Scratch CPU buffers so we don't reallocate per frame.
    download_scratch: Vec<u8>,
    tensor_in: Vec<f32>,
    tensor_out: Vec<f32>,
    pack_scratch: Vec<u8>,
}

impl ModelNode {
    /// Load `path`, validate the shape against the video resolution,
    /// and pre-allocate all bridge buffers.
    pub fn load(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        model_path: &Path,
        width: u32,
        height: u32,
    ) -> NeoResult<Self> {
        let model = OnnxModel::load(model_path).map_err(|e| {
            NeoError::ModelLoad(format!("onnx load {}: {e}", model_path.display()))
        })?;

        // Validate the input shape: we require [1, 3, H, W].
        let shape = model.input_shape();
        if shape.len() != 4 {
            return Err(NeoError::ModelLoad(format!(
                "model input must be 4D NCHW, got {shape:?}"
            )));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if n != 1 || c != 3 {
            return Err(NeoError::ModelLoad(format!(
                "model input must be [1, 3, H, W]; got {shape:?}"
            )));
        }
        if w != width as usize || h != height as usize {
            return Err(NeoError::ModelLoad(format!(
                "model input {w}×{h} does not match video {width}×{height}. \
                 Re-export the ONNX with matching H/W or wrap it with a reshape."
            )));
        }

        let bgra_size = (width as u64) * (height as u64) * 4;
        let download_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lab-model-staging"),
            size: bgra_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pixel_count = (width as usize) * (height as usize);

        // Try the wgpu fast path first. If the ONNX graph contains only
        // supported ops we get a fully-GPU pipeline; otherwise we keep
        // the CPU bounce path alive and use that as fallback.
        let gpu_engine = match model.try_wgpu_plan() {
            Some(plan) => match WgpuInferenceEngine::build(
                device.clone(),
                queue.clone(),
                &plan,
                width,
                height,
            ) {
                Ok(engine) => {
                    info!(
                        model = %model_path.display(),
                        shape = ?shape,
                        ops = plan.ops.len(),
                        "model node ready (GPU fast path) — Phase B.2a"
                    );
                    Some(engine)
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        "wgpu inference engine build failed; falling back to tract CPU"
                    );
                    None
                }
            },
            None => {
                info!(
                    model = %model_path.display(),
                    shape = ?shape,
                    "model node ready (CPU bounce mode) — graph not in supported wgpu subset"
                );
                None
            }
        };

        Ok(Self {
            model: Arc::new(model),
            width,
            height,
            gpu_engine,
            download_staging,
            download_scratch: vec![0u8; bgra_size as usize],
            tensor_in: vec![0.0f32; 3 * pixel_count],
            tensor_out: vec![0.0f32; 3 * pixel_count],
            pack_scratch: vec![0u8; bgra_size as usize],
        })
    }

    /// Run one inference pass. `src_buf` is the BGRA storage buffer
    /// holding the final shader-chain output; the result is written
    /// back into the same buffer.
    pub fn process(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buf: &wgpu::Buffer,
    ) -> NeoResult<()> {
        // Phase B.2a fast path: GPU-only, no CPU round-trip.
        if let Some(engine) = &self.gpu_engine {
            return engine
                .process(src_buf)
                .map_err(|e| NeoError::Inference(format!("wgpu engine: {e}")));
        }

        // Phase B.1 fallback: CPU bounce through tract.
        let bgra_size = (self.width as u64) * (self.height as u64) * 4;

        // 1. GPU → staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lab-model-download"),
        });
        encoder.copy_buffer_to_buffer(src_buf, 0, &self.download_staging, 0, bgra_size);
        queue.submit(std::iter::once(encoder.finish()));

        // 2. Map staging
        let slice = self.download_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| NeoError::GpuBuffer(format!("poll: {e}")))?;
        rx.recv()
            .map_err(|_| NeoError::GpuBuffer("map channel closed".into()))?
            .map_err(|e| NeoError::GpuBuffer(format!("map: {e}")))?;

        // 3. Copy to scratch so we can unmap ASAP
        let mapped = slice.get_mapped_range();
        self.download_scratch.copy_from_slice(&mapped);
        drop(mapped);
        self.download_staging.unmap();

        // 4. BGRA u32 → NCHW f32 [0,1]
        //    Chain output is BGRA little-endian: byte 0=B, 1=G, 2=R, 3=A.
        let plane = (self.width as usize) * (self.height as usize);
        for i in 0..plane {
            let base = i * 4;
            let b = self.download_scratch[base] as f32 / 255.0;
            let g = self.download_scratch[base + 1] as f32 / 255.0;
            let r = self.download_scratch[base + 2] as f32 / 255.0;
            self.tensor_in[i] = r;               // R plane
            self.tensor_in[plane + i] = g;       // G plane
            self.tensor_in[2 * plane + i] = b;   // B plane
        }

        // 5. Run the model
        let t0 = Instant::now();
        let raw = self
            .model
            .infer(&self.tensor_in)
            .map_err(|e| NeoError::Inference(format!("infer: {e}")))?;
        let infer_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if raw.len() != self.tensor_out.len() {
            return Err(NeoError::Inference(format!(
                "model output len {} != expected {}",
                raw.len(),
                self.tensor_out.len()
            )));
        }
        self.tensor_out.copy_from_slice(&raw);

        // 6. NCHW f32 [0,1] → BGRA u32
        for i in 0..plane {
            let r = (self.tensor_out[i].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (self.tensor_out[plane + i].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (self.tensor_out[2 * plane + i].clamp(0.0, 1.0) * 255.0) as u8;
            let base = i * 4;
            self.pack_scratch[base] = b;
            self.pack_scratch[base + 1] = g;
            self.pack_scratch[base + 2] = r;
            self.pack_scratch[base + 3] = 255;
        }

        // 7. Upload to the same BGRA buffer
        queue.write_buffer(src_buf, 0, &self.pack_scratch);

        // Keep the infer time visible for the HUD/logs.
        if infer_ms > 50.0 {
            warn!(infer_ms, "model inference is the bottleneck (CPU bounce mode)");
        }
        Ok(())
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
}
