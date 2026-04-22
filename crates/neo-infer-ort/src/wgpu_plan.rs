//! Compile a small subset of ONNX graphs into a sequence of wgpu-
//! friendly ops.
//!
//! The idea: Neo Lab's Phase B.1 bridge pays an ~100 ms CPU bounce + a
//! tract pure-Rust inference per frame, which caps model workloads at
//! ~15 fps. For the trivially-simple models Neo generates today
//! (Identity, Sub-const-scalar), running inference through a general
//! ML runtime is massively overkill — each of these "models" is one
//! WGSL line.
//!
//! This module inspects the raw ONNX graph and, if every node is in
//! our supported subset, emits a [`WgpuPlanDescription`] that
//! `neo-lab`'s `WgpuInferenceEngine` can turn into a fixed chain of
//! compute passes. No CPU round-trip, no tract invocation on the hot
//! path, just three compute dispatches per frame (pack → op(s) →
//! unpack) on the same VRAM the shader chain already wrote.
//!
//! ## Supported ops (Phase B.2a)
//!
//! | ONNX op | Requirements | WGSL expression |
//! |---------|--------------|-----------------|
//! | `Identity` | none | `out = in` |
//! | `Sub`      | input A is a scalar initializer, input B is the graph input | `out = const − in` |
//! | `Add`      | symmetric to `Sub` | `out = const + in` |
//! | `Mul`      | symmetric to `Sub` | `out = const × in` |
//!
//! Anything else → [`OnnxModel::try_wgpu_plan`] returns `None` and
//! Neo Lab falls back to the tract CPU path.
//!
//! ## Design note
//!
//! We do NOT walk tract's post-optimization `TypedModel`. After
//! `into_decluttered()` tract may rewrite nodes aggressively (fuse ops,
//! hoist constants, eliminate Identity entirely). That gives a
//! different graph on every tract version and makes pattern matching
//! brittle. Instead we keep the raw ONNX ModelProto around and walk
//! IT — our generated models have known, stable shapes and we control
//! them end-to-end.

use prost::Message;
use tract_onnx::pb;

/// The only data type we produce / consume in Phase B.2a.
/// tract's `DataType::Float = 1`.
const DT_FLOAT: i32 = 1;

/// Element-wise unary ops that take no constants — the whole filter
/// is a single WGSL expression on the input value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryKind {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Relu,
    Floor,
    Ceil,
    Round,
    Reciprocal,
}

impl UnaryKind {
    /// WGSL expression that computes the output from a variable `x`.
    pub fn wgsl_expr(self) -> &'static str {
        match self {
            UnaryKind::Neg => "-x",
            UnaryKind::Abs => "abs(x)",
            UnaryKind::Sqrt => "sqrt(max(x, 0.0))",
            UnaryKind::Exp => "exp(x)",
            UnaryKind::Log => "log(max(x, 1e-6))",
            UnaryKind::Sin => "sin(x)",
            UnaryKind::Cos => "cos(x)",
            UnaryKind::Tanh => "tanh(x)",
            UnaryKind::Sigmoid => "1.0 / (1.0 + exp(-x))",
            UnaryKind::Relu => "max(x, 0.0)",
            UnaryKind::Floor => "floor(x)",
            UnaryKind::Ceil => "ceil(x)",
            UnaryKind::Round => "round(x)",
            UnaryKind::Reciprocal => "1.0 / x",
        }
    }

    /// Parse an ONNX op_type string into a [`UnaryKind`].
    pub fn from_onnx(op_type: &str) -> Option<Self> {
        Some(match op_type {
            "Neg" => UnaryKind::Neg,
            "Abs" => UnaryKind::Abs,
            "Sqrt" => UnaryKind::Sqrt,
            "Exp" => UnaryKind::Exp,
            "Log" => UnaryKind::Log,
            "Sin" => UnaryKind::Sin,
            "Cos" => UnaryKind::Cos,
            "Tanh" => UnaryKind::Tanh,
            "Sigmoid" => UnaryKind::Sigmoid,
            "Relu" => UnaryKind::Relu,
            "Floor" => UnaryKind::Floor,
            "Ceil" => UnaryKind::Ceil,
            "Round" => UnaryKind::Round,
            "Reciprocal" => UnaryKind::Reciprocal,
            _ => return None,
        })
    }
}

/// A single fused operation to execute on a wgpu storage buffer.
#[derive(Debug, Clone, PartialEq)]
pub enum WgpuPlanOp {
    // ---- Phase B.2a (unchanged public API) ---------------------------
    /// `out = in` — no actual dispatch needed, the engine just skips
    /// emitting a pass and reuses the input buffer. Kept explicit so
    /// `WgpuPlanDescription` can represent a full no-op graph.
    Identity,
    /// `out = constant − in`. Used for the invert model.
    SubConst { constant: f32 },
    /// `out = in + constant`.
    AddConst { constant: f32 },
    /// `out = in * constant`.
    MulConst { constant: f32 },

    // ---- Phase B.2b (new unary + binary + ternary) -------------------
    /// `out = <unary>(in)`.
    Unary(UnaryKind),
    /// `out = in / constant` (right-operand const).
    DivConstRight { constant: f32 },
    /// `out = constant / in` (left-operand const). Separate from
    /// `DivConstRight` because division is not commutative.
    DivConstLeft { constant: f32 },
    /// `out = pow(in, exponent)`. Second-operand const only — ONNX
    /// Pow with const-on-left would evaluate to a constant and tract
    /// would already have folded it.
    PowConst { exponent: f32 },
    /// `out = min(in, constant)` (commutative).
    MinConst { constant: f32 },
    /// `out = max(in, constant)` (commutative).
    MaxConst { constant: f32 },
    /// `out = clamp(in, min, max)`.
    Clip { min: f32, max: f32 },
}

/// Fully-specified plan that `neo-lab` can feed to its GPU engine.
#[derive(Debug, Clone)]
pub struct WgpuPlanDescription {
    /// `[N, C, H, W]`, always 4D in Phase B.2a.
    pub input_shape: Vec<usize>,
    /// Ordered list of ops to apply, each writes into the next
    /// ping-pong slot.
    pub ops: Vec<WgpuPlanOp>,
}

impl WgpuPlanDescription {
    pub fn n_elements(&self) -> usize {
        self.input_shape.iter().product()
    }
}

/// Try to compile the raw ONNX bytes into a wgpu plan. Returns `None`
/// if any operator or feature is outside the supported subset; callers
/// should then fall back to the CPU tract path unchanged.
pub fn try_compile_from_onnx_bytes(bytes: &[u8]) -> Option<WgpuPlanDescription> {
    let model = pb::ModelProto::decode(bytes).ok()?;
    let graph = model.graph.as_ref()?;

    // Exactly one input + output, both float NCHW, concrete.
    if graph.input.len() != 1 || graph.output.len() != 1 {
        return None;
    }
    let input_name = graph.input[0].name.clone();
    let output_name = graph.output[0].name.clone();

    let input_shape = concrete_float_nchw_shape(&graph.input[0])?;
    // Output shape is read for sanity checks only — we trust the graph
    // to be shape-consistent since we control its generator.
    let _out_shape = concrete_float_nchw_shape(&graph.output[0]);

    // Build a map of initializer name → scalar f32 (only scalar
    // initializers are supported in B.2a).
    let mut scalars: std::collections::HashMap<&str, f32> = std::collections::HashMap::new();
    for init in &graph.initializer {
        if init.data_type != DT_FLOAT {
            continue;
        }
        // Accept shape [] (true scalar) or [1] or any shape with a
        // single element.
        let numel: usize = init.dims.iter().map(|d| *d as usize).product::<usize>().max(1);
        if numel != 1 {
            continue;
        }
        // Value can live either in `float_data` or `raw_data`.
        let v: f32 = if !init.float_data.is_empty() {
            init.float_data[0]
        } else if init.raw_data.len() == 4 {
            f32::from_le_bytes([
                init.raw_data[0],
                init.raw_data[1],
                init.raw_data[2],
                init.raw_data[3],
            ])
        } else {
            continue;
        };
        scalars.insert(init.name.as_str(), v);
    }

    // Walk nodes. We only support a linear chain: each node consumes
    // either the graph input or the previous node's output, plus at
    // most one scalar initializer.
    let mut current_value = input_name.as_str();
    let mut ops: Vec<WgpuPlanOp> = Vec::with_capacity(graph.node.len());

    for node in &graph.node {
        // Phase B.2b: any element-wise unary (1-input, 0-attribute) op
        // maps directly to a `Unary(kind)` op and bypasses the binary
        // logic below.
        if let Some(unary) = UnaryKind::from_onnx(node.op_type.as_str()) {
            if node.input.len() != 1
                || node.input[0] != current_value
                || node.output.len() != 1
            {
                return None;
            }
            ops.push(WgpuPlanOp::Unary(unary));
            current_value = node.output[0].as_str();
            continue;
        }

        let op = match node.op_type.as_str() {
            "Identity" => {
                if node.input.len() != 1 || node.input[0] != current_value {
                    return None;
                }
                WgpuPlanOp::Identity
            }
            // Clip in opset 11+ takes 1 mandatory input + 2 optional
            // scalar initializer inputs (min, max). We require both to
            // be concrete scalars; any omission aborts the fast path.
            "Clip" => {
                if node.input.len() < 1 || node.input[0] != current_value {
                    return None;
                }
                let lo = node
                    .input
                    .get(1)
                    .and_then(|n| scalars.get(n.as_str()).copied())?;
                let hi = node
                    .input
                    .get(2)
                    .and_then(|n| scalars.get(n.as_str()).copied())?;
                WgpuPlanOp::Clip { min: lo, max: hi }
            }
            "Sub" | "Add" | "Mul" | "Div" | "Pow" | "Min" | "Max" => {
                if node.input.len() != 2 {
                    return None;
                }
                // Decide which operand is the streaming tensor and
                // which is the constant.
                let (scalar, order_scalar_first) =
                    match (scalars.get(node.input[0].as_str()), scalars.get(node.input[1].as_str())) {
                        (Some(v), None) if node.input[1] == current_value => (*v, true),
                        (None, Some(v)) if node.input[0] == current_value => (*v, false),
                        _ => return None,
                    };
                match (node.op_type.as_str(), order_scalar_first) {
                    ("Sub", true) => WgpuPlanOp::SubConst { constant: scalar },
                    // "Sub" with scalar on the right == Add(-const).
                    ("Sub", false) => WgpuPlanOp::AddConst { constant: -scalar },
                    ("Add", _) => WgpuPlanOp::AddConst { constant: scalar },
                    ("Mul", _) => WgpuPlanOp::MulConst { constant: scalar },
                    ("Div", true) => WgpuPlanOp::DivConstLeft { constant: scalar },
                    ("Div", false) => WgpuPlanOp::DivConstRight { constant: scalar },
                    ("Pow", false) => WgpuPlanOp::PowConst { exponent: scalar },
                    ("Pow", true) => return None, // const^input needs log/exp, not worth it
                    ("Min", _) => WgpuPlanOp::MinConst { constant: scalar },
                    ("Max", _) => WgpuPlanOp::MaxConst { constant: scalar },
                    _ => return None,
                }
            }
            _ => return None, // unsupported op, fall back to tract CPU
        };
        ops.push(op);

        // The next node should read from this node's output.
        if node.output.len() != 1 {
            return None;
        }
        current_value = node.output[0].as_str();
    }

    // The last node's output must be the graph output.
    if current_value != output_name {
        return None;
    }

    Some(WgpuPlanDescription { input_shape, ops })
}

/// Extract a concrete `[N, C, H, W]` float shape from a ValueInfoProto,
/// returning `None` if anything is dynamic, wrong rank, or non-float.
fn concrete_float_nchw_shape(info: &pb::ValueInfoProto) -> Option<Vec<usize>> {
    let ty = info.r#type.as_ref()?;
    let pb::type_proto::Value::TensorType(tt) = ty.value.as_ref()? else {
        return None;
    };
    if tt.elem_type != DT_FLOAT {
        return None;
    }
    let shape = tt.shape.as_ref()?;
    if shape.dim.len() != 4 {
        return None;
    }
    let mut dims = Vec::with_capacity(4);
    for d in &shape.dim {
        match d.value.as_ref()? {
            pb::tensor_shape_proto::dimension::Value::DimValue(v) if *v > 0 => {
                dims.push(*v as usize);
            }
            _ => return None,
        }
    }
    Some(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{build_identity_model_bytes, build_invert_model_bytes};

    #[test]
    fn identity_model_compiles_to_identity_op() {
        let bytes = build_identity_model_bytes(1, 3, 4, 4);
        let plan = try_compile_from_onnx_bytes(&bytes).expect("identity compiles");
        assert_eq!(plan.input_shape, vec![1, 3, 4, 4]);
        assert_eq!(plan.ops.len(), 1);
        assert_eq!(plan.ops[0], WgpuPlanOp::Identity);
    }

    #[test]
    fn mul_const_model_compiles_to_mul_const() {
        let bytes = crate::generate::build_scalar_binary_model_bytes("Mul", 0.5, 1, 3, 4, 4);
        let plan = try_compile_from_onnx_bytes(&bytes).expect("mul compiles");
        assert_eq!(plan.ops.len(), 1);
        match plan.ops[0] {
            WgpuPlanOp::MulConst { constant } => {
                assert!((constant - 0.5).abs() < 1e-6);
            }
            ref other => panic!("unexpected op: {other:?}"),
        }
    }

    #[test]
    fn add_const_model_compiles_to_add_const() {
        let bytes = crate::generate::build_scalar_binary_model_bytes("Add", 0.25, 1, 3, 4, 4);
        let plan = try_compile_from_onnx_bytes(&bytes).expect("add compiles");
        assert_eq!(plan.ops.len(), 1);
        match plan.ops[0] {
            WgpuPlanOp::AddConst { constant } => {
                assert!((constant - 0.25).abs() < 1e-6);
            }
            ref other => panic!("unexpected op: {other:?}"),
        }
    }

    #[test]
    fn sqrt_unary_compiles_to_unary_sqrt() {
        let bytes = crate::generate::build_unary_model_bytes("Sqrt", 1, 3, 4, 4);
        let plan = try_compile_from_onnx_bytes(&bytes).expect("sqrt compiles");
        assert_eq!(plan.ops.len(), 1);
        assert_eq!(plan.ops[0], WgpuPlanOp::Unary(UnaryKind::Sqrt));
    }

    #[test]
    fn relu_unary_compiles_to_unary_relu() {
        let bytes = crate::generate::build_unary_model_bytes("Relu", 1, 3, 4, 4);
        let plan = try_compile_from_onnx_bytes(&bytes).expect("relu compiles");
        assert_eq!(plan.ops.len(), 1);
        assert_eq!(plan.ops[0], WgpuPlanOp::Unary(UnaryKind::Relu));
    }

    #[test]
    fn sigmoid_unary_compiles_to_unary_sigmoid() {
        let bytes = crate::generate::build_unary_model_bytes("Sigmoid", 1, 3, 4, 4);
        let plan = try_compile_from_onnx_bytes(&bytes).expect("sigmoid compiles");
        assert_eq!(plan.ops[0], WgpuPlanOp::Unary(UnaryKind::Sigmoid));
    }

    #[test]
    fn clip_compiles_to_clip_op() {
        let bytes = crate::generate::build_clip_model_bytes(0.25, 0.75, 1, 3, 4, 4);
        let plan = try_compile_from_onnx_bytes(&bytes).expect("clip compiles");
        assert_eq!(plan.ops.len(), 1);
        match plan.ops[0] {
            WgpuPlanOp::Clip { min, max } => {
                assert!((min - 0.25).abs() < 1e-6);
                assert!((max - 0.75).abs() < 1e-6);
            }
            ref other => panic!("expected Clip, got {other:?}"),
        }
    }

    #[test]
    fn sub_then_mul_chain_compiles() {
        let bytes = crate::generate::build_sub_then_mul_model_bytes(1.0, 0.5, 1, 3, 4, 4);
        let plan = try_compile_from_onnx_bytes(&bytes).expect("chained compiles");
        assert_eq!(plan.input_shape, vec![1, 3, 4, 4]);
        assert_eq!(plan.ops.len(), 2);
        match plan.ops[0] {
            WgpuPlanOp::SubConst { constant } => assert!((constant - 1.0).abs() < 1e-6),
            ref other => panic!("op[0] should be SubConst, got {other:?}"),
        }
        match plan.ops[1] {
            WgpuPlanOp::MulConst { constant } => assert!((constant - 0.5).abs() < 1e-6),
            ref other => panic!("op[1] should be MulConst, got {other:?}"),
        }
    }

    #[test]
    fn invert_model_compiles_to_sub_const() {
        let bytes = build_invert_model_bytes(1, 3, 8, 16);
        let plan = try_compile_from_onnx_bytes(&bytes).expect("invert compiles");
        assert_eq!(plan.input_shape, vec![1, 3, 8, 16]);
        assert_eq!(plan.ops.len(), 1);
        match plan.ops[0] {
            WgpuPlanOp::SubConst { constant } => {
                assert!((constant - 1.0).abs() < 1e-6, "expected 1.0, got {constant}");
            }
            ref other => panic!("unexpected op: {other:?}"),
        }
    }
}
