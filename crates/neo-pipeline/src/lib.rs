//! # neo-pipeline
//!
//! Graph-based processing pipeline orchestrator for Neo-FFmpeg.
//!
//! The pipeline connects nodes (decode, filter, encode) into a directed graph.
//! Data flows through the graph entirely in VRAM — the pipeline ensures
//! zero-copy transfers between nodes by passing GPU buffer handles.

pub mod builder;
pub mod executor;
pub mod graph;
pub mod node;

pub use builder::PipelineBuilder;
pub use executor::{GpuFrameData, PipelineExecutor, PipelineStats, VideoInfo, VideoStats};
pub use graph::PipelineGraph;
pub use node::{NodeId, NodeKind, PipelineNode};
