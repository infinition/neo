use crate::context::GpuContext;
use neo_core::{NeoError, NeoResult};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::trace;

static BUFFER_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A buffer in VRAM.
///
/// This is the fundamental unit of zero-copy processing.
/// Data enters VRAM once and stays there through decode -> inference -> encode.
pub struct VramBuffer {
    pub id: u64,
    pub buffer: wgpu::Buffer,
    pub size: u64,
}

impl std::fmt::Debug for VramBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramBuffer")
            .field("id", &self.id)
            .field("size", &self.size)
            .finish()
    }
}

impl VramBuffer {
    /// Create a new VRAM buffer for storage (writable by GPU, readable by GPU).
    pub fn new_storage(ctx: &GpuContext, size: u64, label: &str) -> NeoResult<Self> {
        let id = BUFFER_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        trace!(id, size, label, "VRAM storage buffer allocated");
        Ok(Self { id, buffer, size })
    }

    /// Create a uniform buffer (for shader parameters).
    pub fn new_uniform(ctx: &GpuContext, data: &[u8], label: &str) -> NeoResult<Self> {
        let id = BUFFER_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        // Uniform buffers must be aligned to 16 bytes
        let aligned_size = ((data.len() + 15) / 16) * 16;
        let mut padded = vec![0u8; aligned_size];
        padded[..data.len()].copy_from_slice(data);

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: aligned_size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(&padded);
        buffer.unmap();
        Ok(Self {
            id,
            buffer,
            size: aligned_size as u64,
        })
    }

    /// Upload CPU data into this VRAM buffer.
    pub fn upload(&self, ctx: &GpuContext, data: &[u8]) {
        ctx.queue.write_buffer(&self.buffer, 0, data);
    }

    /// Create a buffer and immediately upload CPU data into it.
    pub fn from_data(ctx: &GpuContext, data: &[u8], label: &str) -> NeoResult<Self> {
        let id = BUFFER_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(data);
        buffer.unmap();
        trace!(id, size = data.len(), label, "VRAM buffer created from CPU data");
        Ok(Self {
            id,
            buffer,
            size: data.len() as u64,
        })
    }

    /// Download buffer contents back to CPU (synchronous).
    ///
    /// This is the "exit ramp" from VRAM — used when writing final output.
    pub fn download_sync(&self, ctx: &GpuContext) -> NeoResult<Vec<u8>> {
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-download"),
            size: self.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download-encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size);
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| NeoError::GpuBuffer(format!("device.poll: {e}")))?;
        rx.recv()
            .map_err(|_| NeoError::GpuBuffer("map channel closed".into()))?
            .map_err(|e| NeoError::GpuBuffer(format!("map failed: {e}")))?;

        let data = slice.get_mapped_range().to_vec();
        Ok(data)
    }

    /// Copy data between two VRAM buffers (zero-copy within GPU).
    pub fn copy_to(&self, ctx: &GpuContext, dst: &VramBuffer) -> NeoResult<()> {
        let size = self.size.min(dst.size);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("buffer-copy"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &dst.buffer, 0, size);
        ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Submit a GPU→staging copy without waiting (non-blocking).
    /// Call `read_staging()` later to wait and get the data.
    pub fn copy_to_staging(&self, ctx: &GpuContext, staging: &wgpu::Buffer) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("staging-copy"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, staging, 0, self.size);
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Wait for a staging buffer to be readable, then copy data to CPU.
    /// The staging buffer must have had data copied to it via `copy_to_staging()`.
    pub fn read_staging(ctx: &GpuContext, staging: &wgpu::Buffer, size: u64) -> NeoResult<Vec<u8>> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| NeoError::GpuBuffer(format!("device.poll: {e}")))?;
        rx.recv()
            .map_err(|_| NeoError::GpuBuffer("staging map channel closed".into()))?
            .map_err(|e| NeoError::GpuBuffer(format!("staging map failed: {e}")))?;

        let data = slice.get_mapped_range().to_vec();
        staging.unmap(); // unmap so we can reuse for next frame
        Ok(data)
    }

    /// Create a reusable staging buffer for repeated GPU→CPU downloads.
    pub fn create_staging(ctx: &GpuContext, size: u64, label: &str) -> wgpu::Buffer {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}
