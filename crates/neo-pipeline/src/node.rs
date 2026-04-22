/// Unique identifier for a node in the pipeline graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node-{}", self.0)
    }
}

/// What kind of processing a node does.
#[derive(Debug, Clone)]
pub enum NodeKind {
    /// Demuxer — reads compressed packets from a file.
    Demux { path: String },
    /// Decoder — decompresses video into GPU frames.
    Decode,
    /// Filter — processes frames in VRAM (AI or traditional).
    Filter { name: String },
    /// Encoder — compresses GPU frames into a bitstream.
    Encode,
    /// Muxer — writes compressed packets to a file.
    Mux { path: String },
    /// Network source — receives a live stream.
    NetworkSource { url: String },
    /// Network sink — sends to a live stream.
    NetworkSink { url: String },
    /// Tee — duplicates the frame to multiple outputs.
    Tee,
    /// Merge — combines multiple inputs into one.
    Merge,
}

/// A node in the pipeline graph.
#[derive(Debug)]
pub struct PipelineNode {
    pub id: NodeId,
    pub kind: NodeKind,
    pub label: String,
    /// IDs of nodes that feed into this node.
    pub inputs: Vec<NodeId>,
    /// IDs of nodes this node feeds into.
    pub outputs: Vec<NodeId>,
}

impl PipelineNode {
    pub fn new(id: NodeId, kind: NodeKind, label: &str) -> Self {
        Self {
            id,
            kind,
            label: label.to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}
