//! # neo-io
//!
//! Zero-copy I/O layer for Neo-FFmpeg.
//!
//! Handles the critical first and last miles of the pipeline:
//! - **Input**: NVMe → GPU DirectStorage bypass (no CPU touch)
//! - **Output**: GPU → NVMe / Network direct write
//! - **Network**: Low-latency streaming (RTMP, SRT, WebRTC ingest)
//!
//! On platforms with DirectStorage support (Windows 11, Linux io_uring),
//! data flows directly from disk to GPU memory. On others, we use
//! memory-mapped I/O as the next best thing.

pub mod direct_storage;
pub mod mmap_reader;
pub mod network;

pub use direct_storage::DirectStorageReader;
pub use mmap_reader::MmapReader;
pub use network::NetworkSource;
