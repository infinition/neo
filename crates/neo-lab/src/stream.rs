//! Streaming NVDEC source for Neo Lab.
//!
//! The earlier version of Lab was dumb: it called
//! `decode_bytes_capture()` on startup and kept the entire clip in a
//! `Vec<DecodedFrame>` on the CPU heap. At 4K NV12 that's ~12.5 MB per
//! frame, so a 610-frame 20-second clip cost 7.6 GB of RAM. That's
//! obviously not the promise of a "zero-copy VRAM pipeline".
//!
//! This module replaces that caching strategy with a real streaming
//! decoder:
//!
//! - The compressed bitstream (H.264 Annex-B) is held once in RAM
//!   (~100 MB for a 4K 20s clip at 45 Mbps — unavoidable as long as we
//!   don't do file-based streaming, which is a Stage B.2 item).
//! - An `NvdecDecoder` is fed chunk-by-chunk. Each chunk emits some
//!   number of decoded `DecodedFrame`s through the `CaptureMode::Cpu`
//!   hook, which we drain into a small ring (`MAX_BUFFERED`).
//! - `next()` pops from the ring, pulling more bitstream as needed.
//! - When the bitstream is exhausted we flush, drop the decoder, create
//!   a fresh one, reset the feed offset to zero, and keep going. This
//!   is how the Lab loops indefinitely without memory growth.
//!
//! Memory budget for a 4K Lab session: ~100 MB bitstream + up to
//! `MAX_BUFFERED` × 12.5 MB frames. With `MAX_BUFFERED = 4` that's
//! ~150 MB total — **50× less** than the old implementation.

use neo_core::{NeoError, NeoResult};
use neo_hwaccel::{
    nvdec::{CaptureMode, Decoder},
    CudaRuntime, DecodedFrame, NvdecCodec,
};
use std::{collections::VecDeque, sync::Arc};
use tracing::{debug, info};

/// Maximum frames we keep buffered ahead of the render loop. Higher
/// values smooth out jitter at the cost of RAM.
const MAX_BUFFERED: usize = 4;

/// How much bitstream we hand the parser per `pump()` call. Smaller
/// chunks mean tighter control over when decoded frames appear.
const CHUNK: usize = 256 * 1024;

pub struct StreamSource {
    runtime: Arc<CudaRuntime>,
    bitstream: Arc<Vec<u8>>,
    decoder: Option<Decoder>,
    buffered: VecDeque<DecodedFrame>,
    feed_offset: usize,
    pub width: u32,
    pub height: u32,
    frames_emitted: u64,
    loops: u64,
}

impl StreamSource {
    /// Build a new streaming source over the given bitstream.
    ///
    /// Runs a fast probe to learn `width`/`height` without decoding the
    /// whole clip, then creates the initial decoder.
    pub fn new(runtime: Arc<CudaRuntime>, bitstream: Vec<u8>) -> NeoResult<Self> {
        let probe = neo_hwaccel::nvdec::probe_dimensions(
            runtime.as_ref(),
            NvdecCodec::cudaVideoCodec_H264,
            &bitstream,
        )?;
        let width = probe.display_width;
        let height = probe.display_height;
        info!(
            width,
            height,
            bitstream_mb = bitstream.len() / (1024 * 1024),
            "StreamSource ready — streaming decode, no full-clip cache"
        );
        let bitstream = Arc::new(bitstream);
        let decoder = Some(Decoder::new(
            runtime.as_ref(),
            NvdecCodec::cudaVideoCodec_H264,
            CaptureMode::Cpu,
        )?);
        Ok(Self {
            runtime,
            bitstream,
            decoder,
            buffered: VecDeque::with_capacity(MAX_BUFFERED + 2),
            feed_offset: 0,
            width,
            height,
            frames_emitted: 0,
            loops: 0,
        })
    }

    /// Pull the next decoded frame. Feeds the parser just enough to
    /// make at least one frame available, loops back to the start of
    /// the bitstream when exhausted.
    pub fn next(&mut self) -> NeoResult<DecodedFrame> {
        loop {
            if let Some(frame) = self.buffered.pop_front() {
                self.frames_emitted += 1;
                return Ok(frame);
            }
            self.pump()?;
        }
    }

    /// Cheap accessor so the render loop can log without borrowing us
    /// mutably.
    pub fn stats(&self) -> (u64, u64) {
        (self.frames_emitted, self.loops)
    }

    /// Feed one `CHUNK` of bitstream (or reset at EOS) and drain any
    /// frames the parser produced.
    fn pump(&mut self) -> NeoResult<()> {
        if self.feed_offset >= self.bitstream.len() {
            // EOS: flush the parser, then recreate a fresh decoder and
            // rewind the feed offset. We drop the old decoder first so
            // NVDEC releases its surfaces.
            if let Some(mut dec) = self.decoder.take() {
                let _ = dec.flush();
                let tail = dec.take_frames();
                self.buffered.extend(tail);
                drop(dec);
            }
            self.feed_offset = 0;
            self.loops += 1;
            debug!(loops = self.loops, "StreamSource loop");
            self.decoder = Some(Decoder::new(
                self.runtime.as_ref(),
                NvdecCodec::cudaVideoCodec_H264,
                CaptureMode::Cpu,
            )?);
            if !self.buffered.is_empty() {
                return Ok(());
            }
        }

        let dec = self
            .decoder
            .as_mut()
            .expect("decoder present after EOS handling");
        let end = (self.feed_offset + CHUNK).min(self.bitstream.len());
        dec.feed(&self.bitstream[self.feed_offset..end])?;
        self.feed_offset = end;

        let new_frames = dec.take_frames();
        self.buffered.extend(new_frames);

        // Don't let the ring grow unbounded if a single chunk decodes
        // many frames (unlikely but possible with dense I-frame streams).
        while self.buffered.len() > MAX_BUFFERED + 4 {
            // Drop the oldest to keep memory bounded. The render loop
            // will catch up on the next call.
            self.buffered.pop_front();
        }
        Ok(())
    }
}
