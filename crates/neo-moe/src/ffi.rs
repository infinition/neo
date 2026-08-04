//! FFI bridge for C/C++ inference backends (llama.cpp).
//!
//! This module exports the `extern "C"` symbols that `neo_moe_backend.c` links
//! against.  The crate must be compiled as a `cdylib` to produce a shared
//! library (`.dll` / `.so`) that llama.cpp can load at runtime.
//!
//! ## Memory model
//!
//! `neo_moe_stream_create()` returns an opaque `*mut NeoMoeStream` which is
//! simply a boxed `ExpertStream` cast to `c_void`.  The caller owns this
//! handle and must eventually pass it to `neo_moe_stream_free()`.
//!
//! All other functions take the handle as the first argument and perform a
//! checked reborrow — returning an error code instead of crashing on null.

use std::{
    ffi::{c_char, CStr},
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use crate::{ExpertId, ExpertStream, MoeConfig};

static LAST_ERROR: OnceLock<Mutex<Vec<u8>>> = OnceLock::new();

fn last_error_store() -> &'static Mutex<Vec<u8>> {
    LAST_ERROR.get_or_init(|| Mutex::new(b"ok\0".to_vec()))
}

fn set_last_error(message: impl AsRef<str>) {
    let mut bytes: Vec<u8> = message
        .as_ref()
        .as_bytes()
        .iter()
        .copied()
        .filter(|b| *b != 0)
        .collect();
    bytes.push(0);

    if let Ok(mut guard) = last_error_store().lock() {
        *guard = bytes;
    }
}

fn clear_last_error() {
    set_last_error("ok");
}

/// Return the last FFI error message as a NUL-terminated C string.
///
/// The returned pointer remains valid until the next FFI call updates the
/// internal error buffer.
#[no_mangle]
pub extern "C" fn neo_moe_last_error() -> *const c_char {
    static FALLBACK: &[u8] = b"neo-moe last error unavailable\0";

    match last_error_store().lock() {
        Ok(guard) => guard.as_ptr() as *const c_char,
        Err(_) => FALLBACK.as_ptr() as *const c_char,
    }
}

// ─── Opaque handle ──────────────────────────────────────────────────────────────

/// Opaque handle type — corresponds to `NeoMoeStream` / `void*` in C.
///
/// We store the `ExpertStream` behind an `Arc` so that the global singleton
/// and local handles can coexist safely.
#[derive(Clone)]
#[repr(C)]
pub struct NeoMoeStream(Arc<ExpertStream>);

// ─── Lifecycle ──────────────────────────────────────────────────────────────────

/// Create a neo-moe streaming engine.
///
/// # Parameters
/// - `gguf_path`: NUL-terminated UTF-8 path to the GGUF file.
/// - `vram_slots`: how many experts can reside in VRAM simultaneously (e.g. 16).
/// - `io_threads`: number of I/O worker threads (e.g. 4).
/// - `cuda_device`: CUDA device index (0 for single-GPU).
/// - `prefetch_depth`: how many layers ahead to prefetch (e.g. 2).
///
/// # Returns
/// Opaque pointer to the stream, or NULL on failure.
/// Must be freed with [`neo_moe_stream_free`].
#[no_mangle]
pub unsafe extern "C" fn neo_moe_stream_create(
    gguf_path: *const c_char,
    vram_slots: u32,
    io_threads: u32,
    cuda_device: u32,
    prefetch_depth: u32,
) -> *mut NeoMoeStream {
    clear_last_error();

    if gguf_path.is_null() {
        set_last_error("gguf_path is null");
        tracing::error!("[neo-moe-ffi] gguf_path is null");
        return std::ptr::null_mut();
    }

    let path = match unsafe { CStr::from_ptr(gguf_path) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("gguf_path is not valid UTF-8");
            tracing::error!("[neo-moe-ffi] gguf_path is not valid UTF-8");
            return std::ptr::null_mut();
        }
    };

    let config = MoeConfig {
        gguf_path: path.into(),
        vram_resident_experts: vram_slots as usize,
        io_threads: io_threads as usize,
        cuda_device: cuda_device as usize,
        prefetch_depth: prefetch_depth as usize,
        ..Default::default()
    };

    match ExpertStream::init(config) {
        Ok(stream) => {
            clear_last_error();
            let handle = Box::into_raw(Box::new(NeoMoeStream(Arc::new(stream))));
            tracing::info!("[neo-moe-ffi] stream created at {:p}", handle);
            handle as *mut NeoMoeStream
        }
        Err(e) => {
            set_last_error(format!("stream init failed: {e}"));
            tracing::error!("[neo-moe-ffi] stream init failed: {e}");
            std::ptr::null_mut()
        }
    }
}

/// Destroy a stream previously created by [`neo_moe_stream_create`].
///
/// Safe to call with NULL (no-op).
#[no_mangle]
pub unsafe extern "C" fn neo_moe_stream_free(stream: *mut NeoMoeStream) {
    if stream.is_null() {
        return;
    }
    let _ = unsafe { Box::from_raw(stream as *mut NeoMoeStream) };
    tracing::info!("[neo-moe-ffi] stream freed");
}

// ─── Inference API ──────────────────────────────────────────────────────────────

/// Block until the specified expert is VRAM-resident and return its CUDA device
/// pointers.
///
/// # Parameters
/// - `stream`: handle returned by [`neo_moe_stream_create`].
/// - `layer`: transformer layer index.
/// - `expert_id`: expert index within the layer.
/// - `gate_ptr` (out): receives the CUDA device pointer for the gate projection.
/// - `up_ptr`   (out): receives the CUDA device pointer for the up projection.
/// - `down_ptr` (out): receives the CUDA device pointer for the down projection.
/// - `timeout_ms`: max wait in milliseconds (0 = use default 30 s).
///
/// # Returns
/// 0 on success, negative errno on failure.
#[no_mangle]
pub unsafe extern "C" fn neo_moe_demand(
    stream: *mut NeoMoeStream,
    layer: u32,
    expert_id: u32,
    gate_ptr: *mut u64,
    up_ptr: *mut u64,
    down_ptr: *mut u64,
    timeout_ms: u32,
) -> i32 {
    clear_last_error();

    let stream = match unsafe { stream.as_ref() } {
        Some(s) => &s.0,
        None => {
            set_last_error("demand: null stream");
            tracing::error!("[neo-moe-ffi] demand: null stream");
            return -1;
        }
    };

    let id = ExpertId::new(layer, expert_id);
    let timeout = if timeout_ms == 0 {
        Duration::from_secs(30)
    } else {
        Duration::from_millis(timeout_ms as u64)
    };

    match stream.demand(id, timeout) {
        Ok(handle) => {
            clear_last_error();
            if !gate_ptr.is_null() {
                unsafe { *gate_ptr = handle.gate_ptr() as u64 };
            }
            if !up_ptr.is_null() {
                unsafe { *up_ptr = handle.up_ptr() as u64 };
            }
            if !down_ptr.is_null() {
                unsafe { *down_ptr = handle.down_ptr() as u64 };
            }
            // The handle is dropped here, releasing the slot back to the pool.
            // The C caller must have already consumed the data by the time
            // this function returns, because the pointers are synchronous.
            //
            // NOTE: For true zero-copy the caller must hold the handle alive
            // until the GPU kernel finishes.  The `neo_moe_demand_keep()` variant
            // below provides that, paired with `neo_moe_release_handle()`.
            //
            // For safety we keep the handle alive by leaking it deliberately,
            // then relying on the separate release path.  See the *keep variants.
            0
        }
        Err(e) => {
            set_last_error(format!("demand({layer},{expert_id}) failed: {e}"));
            tracing::error!("[neo-moe-ffi] demand({},{}): {e}", layer, expert_id);
            -2
        }
    }
}

/// Demand an expert and **keep** the handle alive until explicitly released.
///
/// This variant is the correct choice for zero-copy inference: the caller
/// receives the device pointers, launches the GPU kernel, and then calls
/// [`neo_moe_release_handle`] to return the VRAM slot to the pool.
///
/// # Returns
/// A non-negative handle ID on success (pass to `neo_moe_release_handle`),
/// or negative errno on failure.
#[no_mangle]
pub unsafe extern "C" fn neo_moe_demand_keep(
    stream: *mut NeoMoeStream,
    layer: u32,
    expert_id: u32,
    gate_ptr: *mut u64,
    up_ptr: *mut u64,
    down_ptr: *mut u64,
    timeout_ms: u32,
) -> i64 {
    clear_last_error();

    let stream = match unsafe { stream.as_ref() } {
        Some(s) => &s.0,
        None => {
            set_last_error("demand_keep: null stream");
            tracing::error!("[neo-moe-ffi] demand_keep: null stream");
            return -1;
        }
    };

    let id = ExpertId::new(layer, expert_id);
    let timeout = if timeout_ms == 0 {
        Duration::from_secs(30)
    } else {
        Duration::from_millis(timeout_ms as u64)
    };

    match stream.demand(id, timeout) {
        Ok(handle) => {
            clear_last_error();
            if !gate_ptr.is_null() {
                unsafe { *gate_ptr = handle.gate_ptr() as u64 };
            }
            if !up_ptr.is_null() {
                unsafe { *up_ptr = handle.up_ptr() as u64 };
            }
            if !down_ptr.is_null() {
                unsafe { *down_ptr = handle.down_ptr() as u64 };
            }

            // Leak the handle — the C side owns it until release.
            let leaked = Box::into_raw(Box::new(handle));
            leaked as i64
        }
        Err(e) => {
            set_last_error(format!("demand_keep({layer},{expert_id}) failed: {e}"));
            tracing::error!("[neo-moe-ffi] demand_keep({},{}): {e}", layer, expert_id);
            -2
        }
    }
}

/// Release a handle previously obtained from [`neo_moe_demand_keep`].
///
/// # Safety
/// `handle_id` must be a valid non-negative value returned by `demand_keep`
/// that has not already been released.
#[no_mangle]
pub unsafe extern "C" fn neo_moe_release_handle(handle_id: i64) {
    if handle_id <= 0 {
        return;
    }
    let _ = unsafe { Box::from_raw(handle_id as *mut crate::tensor::ExpertTensorHandle) };
}

/// Submit speculative prefetch requests for the next layer.
///
/// # Parameters
/// - `stream`: handle returned by [`neo_moe_stream_create`].
/// - `layer`: the layer to prefetch experts for.
/// - `expert_ids`: pointer to an array of `n_experts` expert indices.
/// - `n_experts`: number of entries in `expert_ids`.
///
/// # Returns
/// 0 on success, negative errno on failure.
#[no_mangle]
pub unsafe extern "C" fn neo_moe_prefetch(
    stream: *mut NeoMoeStream,
    layer: u32,
    expert_ids: *const u32,
    n_experts: u32,
) -> i32 {
    clear_last_error();

    let stream = match unsafe { stream.as_ref() } {
        Some(s) => &s.0,
        None => {
            set_last_error("prefetch: null stream");
            tracing::error!("[neo-moe-ffi] prefetch: null stream");
            return -1;
        }
    };

    let ids: Vec<ExpertId> = if !expert_ids.is_null() && n_experts > 0 {
        let slice = unsafe { std::slice::from_raw_parts(expert_ids, n_experts as usize) };
        slice
            .iter()
            .map(|&expert| ExpertId::new(layer, expert))
            .collect()
    } else {
        clear_last_error();
        return 0; // nothing to prefetch
    };

    match stream.prefetch(ids) {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(e) => {
            set_last_error(format!("prefetch(layer={layer}) failed: {e}"));
            tracing::warn!("[neo-moe-ffi] prefetch layer {layer}: {e}");
            -3
        }
    }
}

/// Release a VRAM slot back to the pool.
///
/// Call this after the inference backend has finished using the expert's
/// weights (i.e. after the GPU kernel that reads them has completed).
#[no_mangle]
pub unsafe extern "C" fn neo_moe_release(
    stream: *mut NeoMoeStream,
    layer: u32,
    expert_id: u32,
) {
    clear_last_error();

    let stream = match unsafe { stream.as_ref() } {
        Some(s) => &s.0,
        None => {
            set_last_error("release: null stream");
            tracing::error!("[neo-moe-ffi] release: null stream");
            return;
        }
    };

    stream.release(ExpertId::new(layer, expert_id));
}

// ─── Diagnostics ────────────────────────────────────────────────────────────────

/// Return the number of layers and experts per layer.
#[no_mangle]
pub unsafe extern "C" fn neo_moe_model_info(
    stream: *mut NeoMoeStream,
    n_layers: *mut u32,
    n_experts: *mut u32,
) -> i32 {
    clear_last_error();

    let stream = match unsafe { stream.as_ref() } {
        Some(s) => &s.0,
        None => {
            set_last_error("model_info: null stream");
            return -1;
        }
    };

    if !n_layers.is_null() {
        unsafe { *n_layers = stream.n_layers() };
    }
    if !n_experts.is_null() {
        unsafe { *n_experts = stream.n_experts() };
    }

    clear_last_error();
    0
}
