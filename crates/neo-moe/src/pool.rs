//! VRAM expert pool — double-buffered slot allocator.
//!
//! Maintains a fixed number of CUDA device memory slots, each large enough to
//! hold the largest expert (gate + up + down projections).
//!
//! Uses a simple free-list protected by a mutex — contention is negligible
//! because the number of concurrent requests is bounded by top_k_experts.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DeviceSlice};
use tracing::{debug, trace};

use crate::{
    error::{MoeError, Result},
    ExpertId,
};

/// A single VRAM slot: a pre-allocated device buffer.
pub struct VramSlot {
    /// Device buffer — sized to `max_expert_bytes`.
    pub buf: CudaSlice<u8>,
    /// Which expert is currently loaded (None = free).
    pub resident: Option<ExpertId>,
}

/// The VRAM pool shared between the scheduler and the inference backend.
pub struct ExpertPool {
    slots: Vec<VramSlot>,
    /// Maps ExpertId → slot index for O(1) lookup.
    resident_map: HashMap<ExpertId, usize>,
    /// Indices of free slots.
    free_list: Vec<usize>,
    /// Byte capacity of each slot.
    slot_bytes: usize,
}

impl ExpertPool {
    /// Allocate `n_slots` device buffers of `slot_bytes` each.
    pub fn alloc(device: &Arc<CudaContext>, n_slots: usize, slot_bytes: usize) -> Result<Self> {
        debug!(
            "Allocating VRAM pool: {} slots × {} MB = {} MB",
            n_slots,
            slot_bytes / 1_000_000,
            n_slots * slot_bytes / 1_000_000
        );

        let mut slots = Vec::with_capacity(n_slots);
        let stream = device.default_stream();
        for i in 0..n_slots {
            let buf = unsafe { stream.alloc::<u8>(slot_bytes) }?;
            slots.push(VramSlot { buf, resident: None });
            trace!("Allocated VRAM slot {i}");
        }

        Ok(Self {
            free_list: (0..n_slots).collect(),
            resident_map: HashMap::new(),
            slot_bytes,
            slots,
        })
    }

    /// Check if `id` is already resident; return its slot index if so.
    pub fn find_resident(&self, id: ExpertId) -> Option<usize> {
        self.resident_map.get(&id).copied()
    }

    /// Acquire a free slot (evicting the oldest resident if the pool is full).
    ///
    /// Returns (slot_index, evicted_expert).
    pub fn acquire(&mut self) -> Result<(usize, Option<ExpertId>)> {
        if let Some(idx) = self.free_list.pop() {
            return Ok((idx, None));
        }

        // Pool is full — evict the first resident (simple FIFO for now;
        // a future LRU implementation would be more optimal).
        let (&evict_id, &evict_slot) = self
            .resident_map
            .iter()
            .next()
            .ok_or(MoeError::PoolExhausted)?;

        self.evict(evict_id);
        Ok((evict_slot, Some(evict_id)))
    }

    /// Mark slot `idx` as holding `id`.
    pub fn mark_resident(&mut self, idx: usize, id: ExpertId) {
        self.slots[idx].resident = Some(id);
        self.resident_map.insert(id, idx);
    }

    /// Release an expert back to the free list.
    pub fn evict(&mut self, id: ExpertId) {
        if let Some(idx) = self.resident_map.remove(&id) {
            self.slots[idx].resident = None;
            self.free_list.push(idx);
            trace!("Evicted expert {:?} from slot {idx}", id);
        }
    }

    /// Raw device pointer for a resident slot (for zero-copy tensor binding).
    pub fn device_ptr(&self, slot_idx: usize) -> *mut u8 {
        let buf = &self.slots[slot_idx].buf;
        let (ptr, _sync) = buf.device_ptr(buf.stream().as_ref());
        ptr as *mut u8
    }

    /// Byte capacity of each slot.
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Number of currently free slots.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Number of resident experts.
    pub fn resident_count(&self) -> usize {
        self.resident_map.len()
    }
}

/// Thread-safe wrapper around ExpertPool.
pub type SharedPool = Arc<Mutex<ExpertPool>>;

/// Create a thread-safe expert pool.
pub fn make_shared_pool(
    device: &Arc<CudaContext>,
    n_slots: usize,
    slot_bytes: usize,
) -> Result<SharedPool> {
    let pool = ExpertPool::alloc(device, n_slots, slot_bytes)?;
    Ok(Arc::new(Mutex::new(pool)))
}
