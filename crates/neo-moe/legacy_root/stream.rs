//! Expert stream — NVMe → pinned host → VRAM async I/O engine.
//!
//! Two NVMe read backends are compiled in depending on the target OS:
//!
//! | OS      | Backend                        | Cold read (25 MB) |
//! |---------|--------------------------------|-------------------|
//! | Linux   | O_DIRECT + io_uring (no cache) | ~3.5 ms           |
//! | Windows | mmap (page cache)              | ~8 ms             |
//!
//! On Linux the `io_uring` feature is enabled automatically; on Windows the
//! mmap path is used and the `io_uring` crate is not compiled in.
//!
//! The rest of the pipeline is identical: pinned host staging → cuMemcpyHtoD.

use std::{
    collections::HashMap,
    fs::File,
    sync::Arc,
    thread,
    time::Duration,
};

use crossbeam::channel::Receiver;
use cudarc::driver::{CudaDevice, sys as cuda_sys};
use tracing::{debug, error, info, trace, warn};

use crate::{
    error::{MoeError, Result},
    gguf::{ExpertWeights, GgufWeightMap},
    pool::{make_shared_pool, SharedPool},
    scheduler::{ExpertRequest, ExpertScheduler, Priority},
    tensor::{ExpertLayout, ExpertTensorHandle},
    ExpertId, MoeConfig,
};

// ─── OS-specific imports ──────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
use crate::io_uring::UringReader;

#[cfg(not(target_os = "linux"))]
use memmap2::MmapOptions;

// ─── ExpertStream ─────────────────────────────────────────────────────────────

/// The main streaming engine.
///
/// Call [`ExpertStream::init`] once, then use [`demand`] / [`prefetch`] /
/// [`release`] in your inference loop.
pub struct ExpertStream {
    config:     MoeConfig,
    weight_map: GgufWeightMap,
    pool:       SharedPool,
    scheduler:  ExpertScheduler,
    device:     Arc<CudaDevice>,
}

impl ExpertStream {
    /// Parse the GGUF file, allocate the VRAM pool, and spawn I/O workers.
    pub fn init(config: MoeConfig) -> Result<Self> {
        info!("Initialising ExpertStream for {}", config.gguf_path.display());

        let weight_map = GgufWeightMap::from_file(&config.gguf_path)?;
        let device     = CudaDevice::new(config.cuda_device)?;

        let slot_bytes = weight_map.max_expert_bytes;
        if slot_bytes == 0 {
            return Err(MoeError::GgufParse("No MoE experts found in GGUF file".into()));
        }

        info!(
            "VRAM pool: {} slots × {:.1} MB = {:.1} MB",
            config.vram_resident_experts,
            slot_bytes as f64 / 1e6,
            config.vram_resident_experts * slot_bytes / 1_000_000,
        );

        let pool = make_shared_pool(&device, config.vram_resident_experts, slot_bytes)?;

        let queue_depth = config.vram_resident_experts * config.prefetch_depth * 4;
        let (scheduler, rx) = ExpertScheduler::new(Arc::clone(&pool), queue_depth);

        let experts_map = Arc::new(weight_map.experts.clone());

        // Spawn I/O workers.  Each worker owns its own UringReader (Linux) or
        // holds a reference to the shared mmap (Windows).
        for thread_id in 0..config.io_threads {
            let rx_ref      = rx.clone();
            let experts_ref = Arc::clone(&experts_map);
            let pool_ref    = Arc::clone(&pool);
            let device_ref  = Arc::clone(&device);
            let path        = config.gguf_path.clone();
            let slot_sz     = slot_bytes;

            thread::Builder::new()
                .name(format!("neo-moe-io-{thread_id}"))
                .spawn(move || {
                    run_io_worker(
                        thread_id,
                        path,
                        slot_sz,
                        rx_ref,
                        experts_ref,
                        pool_ref,
                        device_ref,
                    );
                })
                .map_err(MoeError::Io)?;
        }

        info!("ExpertStream ready ({} I/O workers)", config.io_threads);
        Ok(Self { config, weight_map, pool, scheduler, device })
    }

    // ── Public inference API ──────────────────────────────────────────────────

    /// Block until expert `id` is VRAM-resident and return a zero-copy handle.
    ///
    /// If the expert was already prefetched this returns in < 1 µs.
    pub fn demand(&self, id: ExpertId, timeout: Duration) -> Result<ExpertTensorHandle> {
        if let Some(slot_idx) = self.pool.lock().unwrap().find_resident(id) {
            return self.make_handle(id, slot_idx);
        }

        self.scheduler.submit([ExpertRequest { id, priority: Priority::Critical }])?;
        self.scheduler.wait_for(id, timeout)?;

        let slot_idx = self.pool.lock().unwrap()
            .find_resident(id)
            .ok_or(MoeError::PoolExhausted)?;

        self.make_handle(id, slot_idx)
    }

    /// Non-blocking speculative prefetch.  Errors on a full queue are swallowed.
    pub fn prefetch(&self, experts: impl IntoIterator<Item = ExpertId>) -> Result<()> {
        let reqs = experts.into_iter()
            .map(|id| ExpertRequest { id, priority: Priority::Speculative });

        match self.scheduler.submit(reqs) {
            Ok(()) | Err(MoeError::PrefetchQueueFull) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Release the VRAM slot held by `id` so it can be reused.
    pub fn release(&self, id: ExpertId) {
        self.scheduler.mark_consumed(id);
        self.pool.lock().unwrap().evict(id);
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    pub fn n_layers(&self)  -> u32 { self.weight_map.n_layers  }
    pub fn n_experts(&self) -> u32 { self.weight_map.n_experts }

    /// (resident, free) slot counts.
    pub fn vram_stats(&self) -> (usize, usize) {
        let p = self.pool.lock().unwrap();
        (p.resident_count(), p.free_count())
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn make_handle(&self, id: ExpertId, slot_idx: usize) -> Result<ExpertTensorHandle> {
        let pool = self.pool.lock().unwrap();
        let ptr  = pool.device_ptr(slot_idx);
        let w    = self.weight_map.experts.get(&id)
            .ok_or(MoeError::ExpertNotFound { layer: id.layer, expert: id.expert })?;
        let layout = ExpertLayout::packed(w.gate.byte_len, w.up.byte_len, w.down.byte_len);
        Ok(ExpertTensorHandle::new(id, ptr, layout, Arc::clone(&self.pool), slot_idx))
    }
}

// ─── I/O worker dispatch ──────────────────────────────────────────────────────

/// Dispatches to the Linux (io_uring) or Windows (mmap) worker at runtime.
fn run_io_worker(
    thread_id:   usize,
    path:        std::path::PathBuf,
    max_expert:  usize,
    rx:          Receiver<ExpertRequest>,
    experts:     Arc<HashMap<ExpertId, ExpertWeights>>,
    pool:        SharedPool,
    device:      Arc<CudaDevice>,
) {
    #[cfg(target_os = "linux")]
    io_worker_uring(thread_id, &path, max_expert, rx, experts, pool, device);

    #[cfg(not(target_os = "linux"))]
    io_worker_mmap(thread_id, &path, max_expert, rx, experts, pool, device);
}

// ─── Shared: pinned-memory allocation & H2D copy ─────────────────────────────

/// Allocate a page-locked host staging buffer of `size` bytes.
///
/// Returns (ptr, is_pinned).  Falls back to null on failure (caller uses Vec).
unsafe fn alloc_pinned(size: usize) -> (*mut u8, bool) {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let r = cuda_sys::lib().cuMemAllocHost_v2(&mut ptr, size);
    if r == cuda_sys::CUresult::CUDA_SUCCESS && !ptr.is_null() {
        (ptr as *mut u8, true)
    } else {
        (std::ptr::null_mut(), false)
    }
}

/// Copy `src` to the VRAM slot at `dst_ptr` synchronously.
///
/// Returns false on CUDA error.
unsafe fn h2d_copy(dst_ptr: *mut u8, src: &[u8]) -> bool {
    let r = cuda_sys::lib().cuMemcpyHtoD_v2(
        dst_ptr as cuda_sys::CUdeviceptr,
        src.as_ptr() as *const std::ffi::c_void,
        src.len(),
    );
    r == cuda_sys::CUresult::CUDA_SUCCESS
}

// ─── Linux: io_uring worker ───────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn io_worker_uring(
    thread_id:  usize,
    path:       &std::path::Path,
    max_expert: usize,
    rx:         Receiver<ExpertRequest>,
    experts:    Arc<HashMap<ExpertId, ExpertWeights>>,
    pool:       SharedPool,
    device:     Arc<CudaDevice>,
) {
    debug!("io_uring worker {thread_id} starting");

    // Open O_DIRECT fd — fail fast if the FS doesn't support it.
    let file = match crate::io_uring::open_direct(path) {
        Ok(f) => f,
        Err(e) => {
            error!("Worker {thread_id}: O_DIRECT open failed ({e}), exiting");
            return;
        }
    };

    // Build the per-thread io_uring reader.
    let mut uring = match UringReader::new(&file, max_expert) {
        Ok(r) => r,
        Err(e) => {
            error!("Worker {thread_id}: UringReader::new failed ({e}), exiting");
            return;
        }
    };

    // Allocate per-thread pinned staging buffer.
    let (staging_ptr, is_pinned) = unsafe { alloc_pinned(max_expert) };
    if !is_pinned {
        warn!("Worker {thread_id}: cuMemAllocHost failed, using pageable staging");
    }

    for req in rx {
        let id = req.id;
        trace!("Worker {thread_id}: uring load {:?}", id);

        let weights = match experts.get(&id) {
            Some(w) => w,
            None => { error!("Worker {thread_id}: expert {id:?} not in map"); continue; }
        };

        // Acquire VRAM slot.
        let (slot_idx, _evicted) = {
            let mut p = pool.lock().unwrap();
            match p.acquire() {
                Ok(r) => r,
                Err(e) => { error!("Worker {thread_id}: acquire failed: {e}"); continue; }
            }
        };

        let total = weights.total_bytes();

        // Choose staging destination: pinned buffer or fallback Vec.
        let mut fallback: Vec<u8>;
        let staging: &mut [u8] = if !staging_ptr.is_null() {
            unsafe { std::slice::from_raw_parts_mut(staging_ptr, total) }
        } else {
            fallback = vec![0u8; total];
            &mut fallback
        };

        // NVMe → staging (O_DIRECT, no page cache pressure).
        match uring.read_expert_into(weights, staging) {
            Ok(n) if n < total => {
                warn!("Worker {thread_id}: short read {n}/{total} for {id:?}");
            }
            Err(e) => {
                error!("Worker {thread_id}: uring read failed for {id:?}: {e}");
                pool.lock().unwrap().evict(id); // release slot
                continue;
            }
            _ => {}
        }

        // staging → VRAM.
        {
            let p       = pool.lock().unwrap();
            let dst_ptr = p.device_ptr(slot_idx);
            if !unsafe { h2d_copy(dst_ptr, &staging[..total]) } {
                error!("Worker {thread_id}: H2D copy failed for {id:?}");
                drop(p);
                pool.lock().unwrap().evict(id);
                continue;
            }
        }

        pool.lock().unwrap().mark_resident(slot_idx, id);
        trace!("Worker {thread_id}: {id:?} ready (uring) in slot {slot_idx}");
    }

    if !staging_ptr.is_null() {
        unsafe { cuda_sys::lib().cuMemFreeHost(staging_ptr as *mut std::ffi::c_void) };
    }
    debug!("io_uring worker {thread_id} exiting");
}

// ─── Windows / fallback: mmap worker ─────────────────────────────────────────

#[cfg(not(target_os = "linux"))]
fn io_worker_mmap(
    thread_id:  usize,
    path:       &std::path::Path,
    max_expert: usize,
    rx:         Receiver<ExpertRequest>,
    experts:    Arc<HashMap<ExpertId, ExpertWeights>>,
    pool:       SharedPool,
    device:     Arc<CudaDevice>,
) {
    debug!("mmap worker {thread_id} starting");

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => { error!("Worker {thread_id}: open failed: {e}"); return; }
    };
    let mmap = match unsafe { MmapOptions::new().map(&file) } {
        Ok(m) => Arc::new(m),
        Err(e) => { error!("Worker {thread_id}: mmap failed: {e}"); return; }
    };

    let (staging_ptr, is_pinned) = unsafe { alloc_pinned(max_expert) };
    if !is_pinned {
        warn!("Worker {thread_id}: cuMemAllocHost failed, using pageable staging");
    }

    for req in rx {
        let id = req.id;
        trace!("Worker {thread_id}: mmap load {:?}", id);

        let weights = match experts.get(&id) {
            Some(w) => w,
            None => { error!("Worker {thread_id}: expert {id:?} not in map"); continue; }
        };

        let (slot_idx, _) = {
            let mut p = pool.lock().unwrap();
            match p.acquire() {
                Ok(r) => r,
                Err(e) => { error!("Worker {thread_id}: acquire: {e}"); continue; }
            }
        };

        let total = weights.total_bytes();
        let mut fallback: Vec<u8>;
        let staging: &mut [u8] = if !staging_ptr.is_null() {
            unsafe { std::slice::from_raw_parts_mut(staging_ptr, total) }
        } else {
            fallback = vec![0u8; total];
            &mut fallback
        };

        // mmap → staging (page faults drive NVMe reads via OS page cache).
        copy_weights_to_slice(weights, &mmap, staging);

        // staging → VRAM.
        {
            let p       = pool.lock().unwrap();
            let dst_ptr = p.device_ptr(slot_idx);
            if !unsafe { h2d_copy(dst_ptr, &staging[..total]) } {
                error!("Worker {thread_id}: H2D failed for {id:?}");
                drop(p);
                pool.lock().unwrap().evict(id);
                continue;
            }
        }

        pool.lock().unwrap().mark_resident(slot_idx, id);
        trace!("Worker {thread_id}: {id:?} ready (mmap) in slot {slot_idx}");
    }

    if !staging_ptr.is_null() {
        unsafe { cuda_sys::lib().cuMemFreeHost(staging_ptr as *mut std::ffi::c_void) };
    }
    debug!("mmap worker {thread_id} exiting");
}

/// Copy gate + up + down bytes contiguously into `dst` from a mmap.
fn copy_weights_to_slice(weights: &ExpertWeights, mmap: &[u8], dst: &mut [u8]) {
    let mut cursor = 0;
    for loc in [&weights.gate, &weights.up, &weights.down] {
        let s = loc.offset as usize;
        let e = s + loc.byte_len;
        dst[cursor..cursor + loc.byte_len].copy_from_slice(&mmap[s..e]);
        cursor += loc.byte_len;
    }
}
