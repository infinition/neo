use crate::color::ColorDesc;
use crate::format::PixelFormat;
use crate::tensor::TensorDesc;
use crate::timestamp::Timestamp;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A video frame that conceptually lives in VRAM.
///
/// `GpuFrame` is the central data structure of Neo-FFmpeg.
/// Unlike FFmpeg's `AVFrame` which holds CPU-side pixel data,
/// `GpuFrame` is a handle to a GPU buffer. The actual pixel data
/// never touches CPU RAM in the hot path.
///
/// The frame carries metadata (dimensions, format, timestamps)
/// and a reference to its GPU buffer via `GpuBufferHandle`.
#[derive(Debug, Clone)]
pub struct GpuFrame {
    /// Unique frame ID within this pipeline run.
    pub id: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format of the decoded data.
    pub pixel_format: PixelFormat,
    /// Color space information.
    pub color: ColorDesc,
    /// Presentation timestamp.
    pub pts: Timestamp,
    /// Decode timestamp (can differ from PTS for B-frames).
    pub dts: Timestamp,
    /// Duration of this frame.
    pub duration: Timestamp,
    /// Per-plane GPU buffer handles.
    pub planes: Vec<FramePlane>,
    /// Tensor descriptor — how AI models should interpret this frame.
    pub tensor_desc: Option<TensorDesc>,
    /// Frame flags.
    pub flags: FrameFlags,
}

/// A single plane of a video frame, referencing a GPU buffer region.
#[derive(Debug, Clone)]
pub struct FramePlane {
    /// Handle to the GPU buffer (opaque — managed by neo-gpu).
    pub buffer: GpuBufferHandle,
    /// Byte offset within the buffer.
    pub offset: u64,
    /// Row stride in bytes.
    pub stride: u32,
    /// Height of this plane (may differ from frame height for chroma planes).
    pub height: u32,
}

/// Opaque handle to a GPU buffer.
///
/// The actual `wgpu::Buffer` lives in `neo-gpu`. This handle
/// is an Arc-wrapped ID so frames can be cheaply cloned and
/// the buffer is automatically returned to the pool when all
/// references are dropped.
#[derive(Debug, Clone)]
pub struct GpuBufferHandle {
    pub id: u64,
    pub size: u64,
    /// Drop signal — when all clones are dropped, the buffer
    /// is returned to the pool.
    _drop_guard: Arc<DropGuard>,
}

#[derive(Debug)]
struct DropGuard {
    id: u64,
    pool_signal: Option<async_channel_sender::Sender>,
}

impl Drop for DropGuard {
    fn drop(&mut self) {
        if let Some(ref sender) = self.pool_signal {
            sender.send(self.id);
        }
    }
}

// We use a simple trait-free approach for the drop signal
// to avoid pulling in async_channel in neo-core.
mod async_channel_sender {
    use std::sync::mpsc;

    #[derive(Debug)]
    pub struct Sender {
        tx: mpsc::Sender<u64>,
    }

    impl Sender {
        pub fn new(tx: mpsc::Sender<u64>) -> Self {
            Self { tx }
        }

        pub fn send(&self, id: u64) {
            let _ = self.tx.send(id);
        }
    }
}

pub use async_channel_sender::Sender as BufferReturnSender;

impl GpuBufferHandle {
    /// Create a new handle (typically called by the buffer pool).
    pub fn new(id: u64, size: u64, return_sender: Option<std::sync::mpsc::Sender<u64>>) -> Self {
        Self {
            id,
            size,
            _drop_guard: Arc::new(DropGuard {
                id,
                pool_signal: return_sender.map(BufferReturnSender::new),
            }),
        }
    }

    /// Create a handle with no pool (for testing / one-shot buffers).
    pub fn detached(id: u64, size: u64) -> Self {
        Self::new(id, size, None)
    }
}

/// Frame flags.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FrameFlags {
    /// This is a keyframe (I-frame).
    pub keyframe: bool,
    /// Frame contains errors (partial decode).
    pub corrupt: bool,
    /// Frame has been processed by AI inference.
    pub ai_processed: bool,
}

/// Pre-allocated pool of GPU frame buffers for zero-allocation streaming.
///
/// The pool allocates N buffers up front. When a frame is done being used,
/// its buffer automatically returns to the pool (via the Arc drop guard).
#[derive(Debug)]
pub struct GpuFramePool {
    /// Available buffer IDs.
    rx: std::sync::mpsc::Receiver<u64>,
    /// Sender cloned into each buffer handle.
    tx: std::sync::mpsc::Sender<u64>,
    /// Total pool size.
    pub capacity: usize,
    /// Size of each buffer in bytes.
    pub buffer_size: u64,
    /// Next buffer ID to allocate.
    next_id: u64,
}

impl GpuFramePool {
    /// Create a new pool with `capacity` pre-allocated buffers.
    pub fn new(capacity: usize, buffer_size: u64) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut pool = Self {
            rx,
            tx,
            capacity,
            buffer_size,
            next_id: 0,
        };
        // Pre-fill the pool with available IDs.
        for _ in 0..capacity {
            let id = pool.next_id;
            pool.next_id += 1;
            let _ = pool.tx.send(id);
        }
        pool
    }

    /// Acquire a buffer handle from the pool (blocks if none available).
    pub fn acquire(&self) -> GpuBufferHandle {
        let id = self.rx.recv().expect("frame pool channel closed");
        GpuBufferHandle::new(id, self.buffer_size, Some(self.tx.clone()))
    }

    /// Try to acquire a buffer handle without blocking.
    pub fn try_acquire(&self) -> Option<GpuBufferHandle> {
        self.rx.try_recv().ok().map(|id| {
            GpuBufferHandle::new(id, self.buffer_size, Some(self.tx.clone()))
        })
    }
}
