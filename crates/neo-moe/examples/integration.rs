//! Integration example — simulates a MoE inference loop using neo-moe.
//!
//! This is NOT a functional llama.cpp integration (that requires FFI into
//! ggml tensors), but demonstrates the correct call sequence:
//!
//!   init → for each token:
//!     prefetch(predicted_experts) → compute non-expert layers
//!     → demand(each_active_expert) → run expert matmuls → release
//!
//! Run with:
//!   cargo run --example integration --release -- --model /path/to/qwen3-35b.gguf

use std::{path::PathBuf, time::Duration};

use neo_moe::{ExpertId, ExpertStream, MoeConfig, RouterPredictor};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("model.gguf"));

    // ── 1. Initialise the streaming engine ───────────────────────────────────
    let config = MoeConfig {
        gguf_path:             model_path,
        // For Qwen3.6-35B-A3B Q4_K_XL: each expert ≈ 25 MB
        // 16 slots ≈ 400 MB VRAM for experts (leaves ~11.6 GB for KV + non-expert)
        vram_resident_experts: 16,
        io_threads:            4,
        cuda_device:           0,
        prefetch_depth:        2,
        ..Default::default()
    };

    println!("Initialising ExpertStream...");
    let stream = ExpertStream::init(config)?;

    println!(
        "Model: {} layers, {} experts/layer",
        stream.n_layers(),
        stream.n_experts(),
    );

    // ── 2. Set up predictor ──────────────────────────────────────────────────
    // TopK: zero false-negatives, no speculative waste — good default.
    let predictor = RouterPredictor::top_k();

    // Qwen3.6-35B-A3B activates 4 experts per token per MoE layer.
    let top_k = 4usize;

    // ── 3. Simulated inference loop (one token) ──────────────────────────────
    println!("\nSimulating one token forward pass...\n");

    for layer in 0..stream.n_layers() {
        // In a real integration, you'd:
        //   (a) run attention + norm on GPU
        //   (b) compute router logits via a small linear layer
        // Here we simulate with random logits.
        let fake_logits: Vec<f32> = (0..stream.n_experts())
            .map(|i| (i as f32 * 0.37 + layer as f32 * 0.13).sin())
            .collect();

        // ── Prefetch next layer's experts while we process this layer ────────
        if layer + 1 < stream.n_layers() {
            let next_experts = predictor.predict(layer + 1, &fake_logits, top_k);
            stream.prefetch(next_experts)?;
        }

        // ── Get this layer's active experts ──────────────────────────────────
        let active_ids = predictor.predict(layer, &fake_logits, top_k);

        for id in &active_ids {
            // Demand blocks only if the expert is not yet resident.
            // With prefetch working, this should be near-instant.
            let handle = stream.demand(*id, Duration::from_secs(5))?;

            println!(
                "  Layer {:2} Expert {:3} → gate@{:#x} ({:.1} KB)",
                id.layer,
                id.expert,
                handle.gate_ptr() as u64,
                handle.layout.gate_bytes as f64 / 1024.0,
            );

            // ── HERE: pass handle.gate_ptr() / up_ptr() / down_ptr() to  ────
            // ── llama.cpp ggml_tensor.data or ONNX Runtime CUDA EP.       ────
            // ── The matmul runs entirely in VRAM — no CPU involved.        ────

            // Simulate a tiny compute delay (in reality this is the GPU kernel).
            std::thread::sleep(Duration::from_micros(200));

            // Release the slot so it can be reused.
            stream.release(*id);
        }
    }

    let (resident, free) = stream.vram_stats();
    println!("\nDone. VRAM pool: {resident} resident, {free} free slots.");

    Ok(())
}
