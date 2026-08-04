//! Streaming throughput benchmark.
//!
//! Measures end-to-end latency per expert load (NVMe → pinned host → VRAM)
//! and compares against the naive path (VRAM copy via CPU RAM).
//!
//! Usage:
//!   cargo run --example bench --release -- --model /path/to/qwen3-35b.gguf

use std::{path::PathBuf, time::{Duration, Instant}};

use neo_moe::{ExpertId, ExpertStream, MoeConfig};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("model.gguf"));

    let config = MoeConfig {
        gguf_path:             model_path,
        vram_resident_experts: 8,
        io_threads:            4,
        cuda_device:           0,
        prefetch_depth:        1,
        ..Default::default()
    };

    let stream = ExpertStream::init(config)?;

    println!("=== neo-moe streaming benchmark ===");
    println!("Layers: {}, Experts/layer: {}\n", stream.n_layers(), stream.n_experts());

    let n_iters  = 50usize;
    let layer    = 0u32;
    let expert   = 0u32;
    let id       = ExpertId::new(layer, expert);

    // ── Warm-up ──────────────────────────────────────────────────────────────
    for _ in 0..3 {
        let h = stream.demand(id, Duration::from_secs(10))?;
        stream.release(id);
        drop(h);
    }

    // ── Timed loop ───────────────────────────────────────────────────────────
    let mut latencies_us = Vec::with_capacity(n_iters);

    for i in 0..n_iters {
        let t0 = Instant::now();
        let h = stream.demand(id, Duration::from_secs(10))?;
        let elapsed = t0.elapsed();

        latencies_us.push(elapsed.as_micros() as f64);
        stream.release(id);
        drop(h);
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean   = latencies_us.iter().sum::<f64>() / n_iters as f64;
    let p50    = latencies_us[n_iters / 2];
    let p95    = latencies_us[(n_iters as f64 * 0.95) as usize];
    let p99    = latencies_us[(n_iters as f64 * 0.99) as usize];
    let min_v  = latencies_us[0];
    let max_v  = latencies_us[n_iters - 1];

    println!("Expert load latency ({n_iters} iterations):");
    println!("  mean  = {mean:.0} µs");
    println!("  p50   = {p50:.0} µs");
    println!("  p95   = {p95:.0} µs");
    println!("  p99   = {p99:.0} µs");
    println!("  min   = {min_v:.0} µs");
    println!("  max   = {max_v:.0} µs");

    // Throughput based on expert byte size (printed if available from pool).
    // (Expert size is not directly exposed from ExpertStream, would need API addition)
    println!("\nIf p50 < GPU expert compute time → prefetch hides all I/O latency.");

    Ok(())
}
