//! Real-time latency and throughput metrics.

use std::collections::VecDeque;
use std::time::Instant;

/// Rolling-window metrics tracker.
pub struct Metrics {
    window: VecDeque<Sample>,
    window_secs: f64,
    total_frames: u64,
    total_bytes: u64,
    #[allow(dead_code)]
    start: Instant,
}

struct Sample {
    time: Instant,
    bytes: usize,
    network_us: f64,
    decode_us: f64,
    total_us: f64,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub fps: f64,
    pub bitrate_mbps: f64,
    pub avg_network_ms: f64,
    pub avg_decode_ms: f64,
    pub avg_total_ms: f64,
    pub p99_total_ms: f64,
    pub jitter_ms: f64,
    pub total_frames: u64,
}

impl Metrics {
    pub fn new(window_secs: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(256),
            window_secs,
            total_frames: 0,
            total_bytes: 0,
            start: Instant::now(),
        }
    }

    pub fn record(
        &mut self,
        bytes: usize,
        network_us: f64,
        decode_us: f64,
        total_us: f64,
    ) {
        let now = Instant::now();
        self.total_frames += 1;
        self.total_bytes += bytes as u64;
        self.window.push_back(Sample {
            time: now,
            bytes,
            network_us,
            decode_us,
            total_us,
        });
        // Trim old samples.
        let cutoff = now - std::time::Duration::from_secs_f64(self.window_secs);
        while self.window.front().is_some_and(|s| s.time < cutoff) {
            self.window.pop_front();
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        let n = self.window.len();
        if n == 0 {
            return Snapshot {
                fps: 0.0,
                bitrate_mbps: 0.0,
                avg_network_ms: 0.0,
                avg_decode_ms: 0.0,
                avg_total_ms: 0.0,
                p99_total_ms: 0.0,
                jitter_ms: 0.0,
                total_frames: self.total_frames,
            };
        }

        let dt = if n > 1 {
            self.window.back().unwrap().time
                .duration_since(self.window.front().unwrap().time)
                .as_secs_f64()
                .max(1e-9)
        } else {
            self.window_secs
        };

        let fps = (n as f64 - 1.0).max(1.0) / dt;
        let total_bytes: usize = self.window.iter().map(|s| s.bytes).sum();
        let bitrate_mbps = (total_bytes as f64 * 8.0) / (dt * 1_000_000.0);

        let avg_network_ms: f64 =
            self.window.iter().map(|s| s.network_us).sum::<f64>() / n as f64 / 1000.0;
        let avg_decode_ms: f64 =
            self.window.iter().map(|s| s.decode_us).sum::<f64>() / n as f64 / 1000.0;
        let avg_total_ms: f64 =
            self.window.iter().map(|s| s.total_us).sum::<f64>() / n as f64 / 1000.0;

        // P99 total latency.
        let mut totals: Vec<f64> = self.window.iter().map(|s| s.total_us / 1000.0).collect();
        totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99_idx = ((n as f64) * 0.99).ceil() as usize;
        let p99_total_ms = totals[p99_idx.min(n - 1)];

        // Jitter: std dev of inter-frame intervals.
        let jitter_ms = if n > 2 {
            let samples: Vec<&Sample> = self.window.iter().collect();
            let intervals: Vec<f64> = samples.windows(2)
                .map(|w| w[1].time.duration_since(w[0].time).as_secs_f64() * 1000.0)
                .collect();
            let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let var = intervals.iter().map(|i| (i - mean).powi(2)).sum::<f64>()
                / intervals.len() as f64;
            var.sqrt()
        } else {
            0.0
        };

        Snapshot {
            fps,
            bitrate_mbps,
            avg_network_ms,
            avg_decode_ms,
            avg_total_ms,
            p99_total_ms,
            jitter_ms,
            total_frames: self.total_frames,
        }
    }
}

impl std::fmt::Display for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.1} fps | {:.2} Mbps | net {:.2}ms | dec {:.2}ms | total {:.2}ms (p99 {:.2}ms) | jitter {:.2}ms | frames {}",
            self.fps,
            self.bitrate_mbps,
            self.avg_network_ms,
            self.avg_decode_ms,
            self.avg_total_ms,
            self.p99_total_ms,
            self.jitter_ms,
            self.total_frames,
        )
    }
}
