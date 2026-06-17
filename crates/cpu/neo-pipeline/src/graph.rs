use crate::node::{NodeId, NodeKind, PipelineNode};
use neo_core::{NeoError, NeoResult};
use std::collections::HashMap;

/// The pipeline graph — a DAG of processing nodes.
///
/// Frames flow from source nodes (demux, network) through processing
/// nodes (decode, filter) to sink nodes (encode, mux, network).
pub struct PipelineGraph {
    nodes: HashMap<NodeId, PipelineNode>,
    next_id: u32,
}

impl PipelineGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a node to the graph and return its ID.
    pub fn add_node(&mut self, kind: NodeKind, label: &str) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.insert(id, PipelineNode::new(id, kind, label));
        id
    }

    /// Connect two nodes (from → to).
    pub fn connect(&mut self, from: NodeId, to: NodeId) -> NeoResult<()> {
        // Validate both nodes exist
        if !self.nodes.contains_key(&from) {
            return Err(NeoError::NodeNotFound(from.to_string()));
        }
        if !self.nodes.contains_key(&to) {
            return Err(NeoError::NodeNotFound(to.to_string()));
        }

        self.nodes.get_mut(&from).unwrap().outputs.push(to);
        self.nodes.get_mut(&to).unwrap().inputs.push(from);
        Ok(())
    }

    /// Get a node by ID.
    pub fn node(&self, id: NodeId) -> Option<&PipelineNode> {
        self.nodes.get(&id)
    }

    /// Get all source nodes (no inputs).
    pub fn sources(&self) -> Vec<NodeId> {
        self.nodes
            .values()
            .filter(|n| n.inputs.is_empty())
            .map(|n| n.id)
            .collect()
    }

    /// Get all sink nodes (no outputs).
    pub fn sinks(&self) -> Vec<NodeId> {
        self.nodes
            .values()
            .filter(|n| n.outputs.is_empty())
            .map(|n| n.id)
            .collect()
    }

    /// Topological sort — returns nodes in execution order.
    pub fn topological_order(&self) -> NeoResult<Vec<NodeId>> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for node in self.nodes.values() {
            in_degree.entry(node.id).or_insert(0);
            for &out in &node.outputs {
                *in_degree.entry(out).or_insert(0) += 1;
            }
        }

        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort_by_key(|id| id.0);

        let mut order = Vec::new();
        while let Some(id) = queue.pop() {
            order.push(id);
            if let Some(node) = self.nodes.get(&id) {
                for &out in &node.outputs {
                    if let Some(deg) = in_degree.get_mut(&out) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(out);
                        }
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(NeoError::Pipeline("cycle detected in pipeline graph".into()));
        }

        Ok(order)
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Pretty-print the pipeline graph.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Ok(order) = self.topological_order() {
            for id in order {
                if let Some(node) = self.nodes.get(&id) {
                    parts.push(format!("[{}]", node.label));
                }
            }
        }
        parts.join(" → ")
    }
}

impl Default for PipelineGraph {
    fn default() -> Self {
        Self::new()
    }
}
