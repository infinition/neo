use crate::graph::PipelineGraph;
use crate::node::{NodeId, NodeKind};
use neo_core::format::CodecId;
use neo_core::NeoResult;

/// Fluent builder for constructing pipelines.
///
/// ```ignore
/// let pipeline = PipelineBuilder::new()
///     .input("video.mp4")
///     .decode()
///     .filter("upscale-2x")
///     .filter("denoise")
///     .encode(CodecId::H265)
///     .output("output.mp4")
///     .build()?;
/// ```
pub struct PipelineBuilder {
    graph: PipelineGraph,
    last_node: Option<NodeId>,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            graph: PipelineGraph::new(),
            last_node: None,
        }
    }

    /// Add a file input source.
    pub fn input(mut self, path: &str) -> Self {
        let id = self.graph.add_node(
            NodeKind::Demux { path: path.to_string() },
            &format!("input:{}", path),
        );
        self.last_node = Some(id);
        self
    }

    /// Add a network input source.
    pub fn network_input(mut self, url: &str) -> Self {
        let id = self.graph.add_node(
            NodeKind::NetworkSource { url: url.to_string() },
            &format!("net-in:{}", url),
        );
        self.last_node = Some(id);
        self
    }

    /// Add a decode node.
    pub fn decode(mut self) -> Self {
        let id = self.graph.add_node(NodeKind::Decode, "decode");
        if let Some(prev) = self.last_node {
            let _ = self.graph.connect(prev, id);
        }
        self.last_node = Some(id);
        self
    }

    /// Add a filter node.
    pub fn filter(mut self, name: &str) -> Self {
        let id = self.graph.add_node(
            NodeKind::Filter { name: name.to_string() },
            name,
        );
        if let Some(prev) = self.last_node {
            let _ = self.graph.connect(prev, id);
        }
        self.last_node = Some(id);
        self
    }

    /// Add an encode node.
    pub fn encode(mut self, _codec: CodecId) -> Self {
        let id = self.graph.add_node(NodeKind::Encode, "encode");
        if let Some(prev) = self.last_node {
            let _ = self.graph.connect(prev, id);
        }
        self.last_node = Some(id);
        self
    }

    /// Add a file output sink.
    pub fn output(mut self, path: &str) -> Self {
        let id = self.graph.add_node(
            NodeKind::Mux { path: path.to_string() },
            &format!("output:{}", path),
        );
        if let Some(prev) = self.last_node {
            let _ = self.graph.connect(prev, id);
        }
        self.last_node = Some(id);
        self
    }

    /// Add a network output sink.
    pub fn network_output(mut self, url: &str) -> Self {
        let id = self.graph.add_node(
            NodeKind::NetworkSink { url: url.to_string() },
            &format!("net-out:{}", url),
        );
        if let Some(prev) = self.last_node {
            let _ = self.graph.connect(prev, id);
        }
        self.last_node = Some(id);
        self
    }

    /// Build the pipeline graph.
    pub fn build(self) -> NeoResult<PipelineGraph> {
        // Validate the graph
        let _ = self.graph.topological_order()?;
        Ok(self.graph)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
