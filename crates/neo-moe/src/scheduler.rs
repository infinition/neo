//! Expert scheduler — orchestrates prefetch requests.
//!
//! Sits between the predictor and the I/O stream. Responsibilities:
//!
//! - Deduplicates requests (don't re-fetch already-resident experts).
//! - Prioritises requests: experts needed NOW > experts needed soon.
//! - Rate-limits the prefetch queue to avoid starving the PCIe bus.
//! - Exposes a synchronous `wait_for` API so the inference thread can
//!   block until a specific expert is available in VRAM.

use std::{
    collections::HashSet,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use crossbeam::channel::{bounded, Receiver, Sender, TrySendError};
use tracing::{debug, trace, warn};

use crate::{
    error::{MoeError, Result},
    pool::SharedPool,
    ExpertId,
};

/// Priority of a prefetch request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Needed for the current token — I/O must complete before compute stalls.
    Critical = 2,
    /// Needed within the next `prefetch_depth` layers.
    Speculative = 1,
}

/// A single prefetch request.
#[derive(Debug, Clone)]
pub struct ExpertRequest {
    pub id: ExpertId,
    pub priority: Priority,
}

/// Internal state shared between the scheduler and the I/O worker.
struct SchedulerState {
    /// Set of experts currently in-flight (requested but not yet VRAM-resident).
    in_flight: HashSet<ExpertId>,
    /// Set of experts ready (VRAM-resident).
    ready: HashSet<ExpertId>,
}

/// Coordinates expert prefetching across the inference thread and I/O workers.
pub struct ExpertScheduler {
    /// Channel to send requests to the I/O worker(s).
    tx: Sender<ExpertRequest>,
    /// Shared state for ready / in-flight tracking.
    state: Arc<(Mutex<SchedulerState>, Condvar)>,
    /// Reference to the VRAM pool (to check residency without locking state).
    pool: SharedPool,
}

impl ExpertScheduler {
    /// Create a scheduler with a bounded prefetch queue.
    ///
    /// `queue_depth` should be at least `n_io_threads × prefetch_depth × top_k`.
    pub fn new(pool: SharedPool, queue_depth: usize) -> (Self, Receiver<ExpertRequest>) {
        let (tx, rx) = bounded(queue_depth);
        let state = Arc::new((
            Mutex::new(SchedulerState {
                in_flight: HashSet::new(),
                ready: HashSet::new(),
            }),
            Condvar::new(),
        ));

        let sched = Self { tx, state, pool };
        (sched, rx)
    }

    /// Submit a batch of prefetch requests.
    ///
    /// Already-resident or already-in-flight experts are silently skipped.
    pub fn submit(&self, requests: impl IntoIterator<Item = ExpertRequest>) -> Result<()> {
        let (lock, _) = &*self.state;

        for req in requests {
            let id = req.id;

            // Skip if already in VRAM.
            if self.pool.lock().unwrap().find_resident(id).is_some() {
                trace!("Expert {:?} already resident, skipping prefetch", id);
                continue;
            }

            let mut state = lock.lock().unwrap();

            // Skip if already in flight.
            if state.in_flight.contains(&id) {
                trace!("Expert {:?} already in-flight, skipping", id);
                continue;
            }

            state.in_flight.insert(id);
            drop(state);

            match self.tx.try_send(req) {
                Ok(()) => {
                    debug!("Scheduled prefetch for expert {:?}", id);
                }
                Err(TrySendError::Full(_)) => {
                    // Queue is full — this means I/O can't keep up.
                    // We drop the speculative request rather than stalling.
                    let mut state = lock.lock().unwrap();
                    state.in_flight.remove(&id);
                    warn!("Prefetch queue full — dropped expert {:?}", id);
                    return Err(MoeError::PrefetchQueueFull);
                }
                Err(TrySendError::Disconnected(_)) => {
                    let mut state = lock.lock().unwrap();
                    state.in_flight.remove(&id);
                    return Err(MoeError::NotInitialised);
                }
            }
        }
        Ok(())
    }

    /// Block until `id` is resident in VRAM, with a timeout.
    pub fn wait_for(&self, id: ExpertId, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        let (lock, cvar) = &*self.state;

        loop {
            // Fast path: check VRAM pool directly.
            if self.pool.lock().unwrap().find_resident(id).is_some() {
                let mut state = lock.lock().unwrap();
                state.in_flight.remove(&id);
                state.ready.insert(id);
                return Ok(());
            }

            let mut state = lock.lock().unwrap();
            if state.ready.contains(&id) {
                state.in_flight.remove(&id);
                return Ok(());
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                state.in_flight.remove(&id);
                return Err(MoeError::Other(format!(
                    "Timeout waiting for expert {:?}",
                    id
                )));
            }

            // Some worker paths only update the VRAM pool and do not always
            // signal the scheduler condvar. Wake up periodically to re-check.
            let sleep_for = remaining.min(Duration::from_millis(10));
            let (_state, _timeout_result) = cvar.wait_timeout(state, sleep_for).unwrap();
        }
    }

    /// Called by the I/O worker after successfully loading an expert into VRAM.
    pub fn notify_ready(&self, id: ExpertId) {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();
        state.in_flight.remove(&id);
        state.ready.insert(id);
        cvar.notify_all();
        debug!("Expert {:?} is now VRAM-resident", id);
    }

    /// Called after the inference backend has consumed an expert
    /// (so it can be evicted from the ready set).
    pub fn mark_consumed(&self, id: ExpertId) {
        let (lock, _) = &*self.state;
        let mut state = lock.lock().unwrap();
        state.ready.remove(&id);
    }
}
