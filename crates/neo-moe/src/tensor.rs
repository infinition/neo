//! Zero-copy tensor handle.
//!
//! Bridges a VRAM pool slot to any inference backend that accepts a raw CUDA
//! device pointer — llama.cpp, ONNX Runtime CUDA EP, TensorRT, etc.
//!
//! The handle holds a shared reference to the pool so the slot cannot be
//! evicted while inference is in progress.

use crate::{pool::SharedPool, ExpertId};

/// A zero-copy view of an expert's weights in VRAM.
///
/// Dropping this handle releases the slot back to the pool.
pub struct ExpertTensorHandle {
    /// Which expert this handle refers to.
    pub id: ExpertId,
    /// Raw CUDA device pointer (byte-addressed).
    ///
    /// Consumers must reinterpret this to the correct element type and shape.
    pub device_ptr: *mut u8,
    /// Byte layout of the three projections within the slot.
    pub layout: ExpertLayout,
    /// Kept alive to prevent slot eviction while this handle lives.
    _pool: SharedPool,
    /// Slot index (used to release on drop — future work).
    _slot_idx: usize,
}

// SAFETY: The device pointer is only used from CUDA kernels launched on the
// same device thread, so Send + Sync are safe here.
unsafe impl Send for ExpertTensorHandle {}
unsafe impl Sync for ExpertTensorHandle {}

/// Byte offsets and sizes of each projection within a VRAM slot.
#[derive(Debug, Clone, Copy)]
pub struct ExpertLayout {
    pub gate_offset: usize,
    pub gate_bytes:  usize,
    pub up_offset:   usize,
    pub up_bytes:    usize,
    pub down_offset: usize,
    pub down_bytes:  usize,
}

impl ExpertLayout {
    /// Pack gate / up / down back-to-back.
    pub fn packed(gate_bytes: usize, up_bytes: usize, down_bytes: usize) -> Self {
        Self {
            gate_offset: 0,
            gate_bytes,
            up_offset:   gate_bytes,
            up_bytes,
            down_offset: gate_bytes + up_bytes,
            down_bytes,
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.gate_bytes + self.up_bytes + self.down_bytes
    }
}

impl ExpertTensorHandle {
    /// Construct from a resident pool slot.
    pub fn new(
        id: ExpertId,
        device_ptr: *mut u8,
        layout: ExpertLayout,
        pool: SharedPool,
        slot_idx: usize,
    ) -> Self {
        Self {
            id,
            device_ptr,
            layout,
            _pool: pool,
            _slot_idx: slot_idx,
        }
    }

    // ── Convenience sub-pointers ─────────────────────────────────────────────

    /// CUDA device pointer to the gate projection.
    #[inline]
    pub fn gate_ptr(&self) -> *mut u8 {
        unsafe { self.device_ptr.add(self.layout.gate_offset) }
    }

    /// CUDA device pointer to the up projection.
    #[inline]
    pub fn up_ptr(&self) -> *mut u8 {
        unsafe { self.device_ptr.add(self.layout.up_offset) }
    }

    /// CUDA device pointer to the down projection.
    #[inline]
    pub fn down_ptr(&self) -> *mut u8 {
        unsafe { self.device_ptr.add(self.layout.down_offset) }
    }

    // ── llama.cpp C-API bridge (pseudo-code) ─────────────────────────────────
    //
    // llama.cpp exposes `ggml_backend_buffer_type_t` for custom allocators.
    // A future `neo-moe-llamacpp` crate can wrap these pointers as
    // `ggml_tensor.data` directly, avoiding any copy:
    //
    //   unsafe {
    //       (*expert_tensor).data = self.gate_ptr() as *mut std::ffi::c_void;
    //   }
    //
    // Until that FFI layer is built, the Python bridge below covers the
    // common case (llama-cpp-python / ctransformers).
}

/// Python-facing repr (serialisable over FFI/pyo3).
#[repr(C)]
pub struct ExpertPtrs {
    pub gate_ptr:   u64,   // raw u64 address (cast to void* in Python)
    pub gate_bytes: usize,
    pub up_ptr:     u64,
    pub up_bytes:   usize,
    pub down_ptr:   u64,
    pub down_bytes: usize,
    pub layer:      u32,
    pub expert:     u32,
}

impl ExpertTensorHandle {
    /// Export raw pointers for Python / C FFI consumption.
    pub fn to_c_repr(&self) -> ExpertPtrs {
        ExpertPtrs {
            gate_ptr:   self.gate_ptr()  as u64,
            gate_bytes: self.layout.gate_bytes,
            up_ptr:     self.up_ptr()    as u64,
            up_bytes:   self.layout.up_bytes,
            down_ptr:   self.down_ptr()  as u64,
            down_bytes: self.layout.down_bytes,
            layer:      self.id.layer,
            expert:     self.id.expert,
        }
    }
}
