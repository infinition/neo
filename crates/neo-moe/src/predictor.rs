//! Router predictor — guesses which experts will be needed next.
//!
//! The key insight: for a given token, the MoE router emits logits before
//! dispatching to experts. We intercept those logits and use them to prefetch
//! the top-k experts for **the next** layer while the current layer computes.
//!
//! Two strategies are implemented:
//!
//! 1. `TopKPredictor` — deterministic: just take the top-k from the current
//!    router output. Zero false-negatives (we always prefetch the right
//!    experts) but no look-ahead past one layer.
//!
//! 2. `FrequencyPredictor` — statistical: tracks a rolling frequency table
//!    and speculatively prefetches experts that historically co-activate.
//!    Can predict 2–4 layers ahead at the cost of occasional wasted I/O.

use crate::ExpertId;

/// Trait for expert prediction strategies.
pub trait Predict: Send + Sync {
    /// Given router logits for `layer`, predict which experts to prefetch.
    ///
    /// `logits` is a slice of length `n_experts` (one score per expert).
    /// Returns a list of ExpertIds ordered by priority (highest first).
    fn predict(&self, layer: u32, logits: &[f32], top_k: usize) -> Vec<ExpertId>;
}

// ─── TopK (deterministic) ─────────────────────────────────────────────────────

/// Always prefetches the top-k experts by router score.
/// This is zero-waste and zero-latency when the router is deterministic.
pub struct TopKPredictor;

impl Predict for TopKPredictor {
    fn predict(&self, layer: u32, logits: &[f32], top_k: usize) -> Vec<ExpertId> {
        let mut indexed: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        // Partial sort: O(n) for top-k
        indexed.select_nth_unstable_by(top_k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        indexed[..top_k.min(indexed.len())]
            .iter()
            .map(|(expert, _)| ExpertId::new(layer, *expert as u32))
            .collect()
    }
}

// ─── Frequency (speculative) ──────────────────────────────────────────────────

/// Tracks co-activation frequencies and speculatively prefetches likely
/// experts for future layers.
pub struct FrequencyPredictor {
    /// activation_counts[layer][expert] = times activated in recent history.
    activation_counts: Vec<Vec<u32>>,
    /// Rolling window size for frequency decay.
    window: usize,
    /// Step counter for decay scheduling.
    step: usize,
}

impl FrequencyPredictor {
    pub fn new(n_layers: usize, n_experts: usize, window: usize) -> Self {
        Self {
            activation_counts: vec![vec![0u32; n_experts]; n_layers],
            window,
            step: 0,
        }
    }

    /// Record which experts were actually activated (call after each forward pass).
    pub fn record_activations(&mut self, layer: u32, activated: &[u32]) {
        for &expert in activated {
            if let Some(layer_counts) = self.activation_counts.get_mut(layer as usize) {
                if let Some(count) = layer_counts.get_mut(expert as usize) {
                    *count = count.saturating_add(1);
                }
            }
        }

        // Decay every `window` steps.
        self.step += 1;
        if self.step % self.window == 0 {
            for layer_counts in &mut self.activation_counts {
                for count in layer_counts.iter_mut() {
                    *count /= 2; // exponential decay
                }
            }
        }
    }
}

impl Predict for FrequencyPredictor {
    fn predict(&self, layer: u32, logits: &[f32], top_k: usize) -> Vec<ExpertId> {
        let layer_idx = layer as usize;
        let n_experts = logits.len();

        // Combine router logits with historical frequency scores.
        let freq_max = self
            .activation_counts
            .get(layer_idx)
            .and_then(|c| c.iter().max().copied())
            .unwrap_or(1)
            .max(1) as f32;

        let mut scores: Vec<(usize, f32)> = (0..n_experts)
            .map(|e| {
                let freq_score = self
                    .activation_counts
                    .get(layer_idx)
                    .and_then(|c| c.get(e))
                    .copied()
                    .unwrap_or(0) as f32
                    / freq_max;
                let logit_score = *logits.get(e).unwrap_or(&0.0);
                // Weighted blend: 70% router, 30% historical
                (e, 0.7 * logit_score + 0.3 * freq_score)
            })
            .collect();

        scores.select_nth_unstable_by(top_k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        scores[..top_k.min(scores.len())]
            .iter()
            .map(|(expert, _)| ExpertId::new(layer, *expert as u32))
            .collect()
    }
}

// ─── Public handle ────────────────────────────────────────────────────────────

/// Unified predictor handle.
pub struct RouterPredictor {
    inner: Box<dyn Predict>,
}

impl RouterPredictor {
    /// Use deterministic top-k (safe default, no false negatives).
    pub fn top_k() -> Self {
        Self { inner: Box::new(TopKPredictor) }
    }

    /// Use frequency-weighted speculative predictor.
    pub fn frequency(n_layers: usize, n_experts: usize, window: usize) -> Self {
        Self {
            inner: Box::new(FrequencyPredictor::new(n_layers, n_experts, window)),
        }
    }

    /// Predict which experts to prefetch for `layer`.
    pub fn predict(&self, layer: u32, logits: &[f32], top_k: usize) -> Vec<ExpertId> {
        self.inner.predict(layer, logits, top_k)
    }
}
