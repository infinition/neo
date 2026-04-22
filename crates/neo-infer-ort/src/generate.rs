//! Build minimal ONNX models directly from Rust.
//!
//! Two reasons this module exists:
//!
//! 1. Phase B.1's end-to-end smoke test needs a real `.onnx` file it can
//!    feed through `OnnxModel::load` + `infer`. Depending on Python
//!    + `onnx`/`torch` to generate a test fixture violates Neo's "no
//!    setup" mantra. So we build one in Rust.
//! 2. Neo Lab's model node requires `[1, 3, H, W]` shaped float models.
//!    Real models usually declare `(batch, 3, H, W)` with batch dynamic,
//!    and frequently expect different resolutions than the user's
//!    source video. Being able to **synthesise** a model with exact
//!    dimensions on demand sidesteps model authoring tools entirely.
//!
//! ## What we generate
//!
//! A graph with a single `Identity` node:
//!
//! ```text
//!    input [1, 3, H, W] ──► Identity ──► output [1, 3, H, W]
//! ```
//!
//! Inference runs in a few microseconds so it acts as a true zero-cost
//! passthrough: the chain's output survives the CPU bounce round-trip
//! unchanged (modulo f32→u8 quantisation noise), which is exactly what a
//! smoke test wants to prove.
//!
//! More sophisticated fixtures (e.g. a single Conv with an identity
//! kernel, or a colour-matrix multiplication) can be added next to
//! `identity_model_f32_nchw` using the same `tract_onnx::pb` protobuf
//! types.

use prost::Message;
use std::{io::Write, path::Path};
use tract_onnx::pb;

/// Serialise an `[N, C, H, W]` identity ONNX model to `out_path`.
///
/// The model has exactly one float input (`input`) and one float output
/// (`output`), both of fixed shape `[batch, channels, height, width]`,
/// connected through a single `Identity` node.
///
/// Used by `OnnxModel::load` tests and the `neo-gen-identity-onnx`
/// binary.
pub fn identity_model_f32_nchw(
    out_path: &Path,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(), String> {
    let bytes = build_identity_model_bytes(batch, channels, height, width);
    let mut file =
        std::fs::File::create(out_path).map_err(|e| format!("create {out_path:?}: {e}"))?;
    file.write_all(&bytes)
        .map_err(|e| format!("write {out_path:?}: {e}"))?;
    Ok(())
}

/// Build the raw protobuf bytes for an identity model. Used both by
/// [`identity_model_f32_nchw`] and by unit tests that want to round-trip
/// a model through `tract` without touching the filesystem.
pub fn build_identity_model_bytes(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<u8> {
    // ---- Input + output shape (same) ------------------------------------
    let make_dim = |v: usize| pb::tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(v as i64)),
    };
    let shape = pb::TensorShapeProto {
        dim: vec![
            make_dim(batch),
            make_dim(channels),
            make_dim(height),
            make_dim(width),
        ],
    };
    let float_tensor_type = pb::type_proto::Tensor {
        elem_type: pb::tensor_proto::DataType::Float as i32,
        shape: Some(shape),
    };
    let type_proto = pb::TypeProto {
        denotation: String::new(),
        value: Some(pb::type_proto::Value::TensorType(float_tensor_type)),
    };
    let input = pb::ValueInfoProto {
        name: "input".to_string(),
        r#type: Some(type_proto.clone()),
        doc_string: String::new(),
    };
    let output = pb::ValueInfoProto {
        name: "output".to_string(),
        r#type: Some(type_proto),
        doc_string: String::new(),
    };

    // ---- The one Identity node ------------------------------------------
    let node = pb::NodeProto {
        input: vec!["input".to_string()],
        output: vec!["output".to_string()],
        name: "identity_node".to_string(),
        op_type: "Identity".to_string(),
        domain: String::new(),
        attribute: vec![],
        doc_string: String::new(),
    };

    // ---- Graph ----------------------------------------------------------
    let graph = pb::GraphProto {
        node: vec![node],
        name: "neo_identity".to_string(),
        initializer: vec![],
        sparse_initializer: vec![],
        doc_string: String::new(),
        input: vec![input],
        output: vec![output],
        value_info: vec![],
        quantization_annotation: vec![],
    };

    // ---- Opset: tract-onnx supports up to ~opset 18 happily. Identity is
    //      stable since opset 1 so any version works. --------------------
    let opset = pb::OperatorSetIdProto {
        domain: String::new(),
        version: 13,
    };

    // ---- Final ModelProto ----------------------------------------------
    let model = pb::ModelProto {
        ir_version: 8, // ONNX IR v8 is widely supported
        opset_import: vec![opset],
        producer_name: "neo-infer-ort".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
    };

    // Encode. prost's `encoded_len` + `encode` gives us a minimal copy
    // serialisation.
    let mut buf = Vec::with_capacity(model.encoded_len());
    model
        .encode(&mut buf)
        .expect("prost encode of ModelProto cannot fail on owned buffer");
    buf
}

/// Serialise a photo-negative `[N, C, H, W]` ONNX model to `out_path`.
///
/// The model is:
///
/// ```text
///   const_one [1]  ─┐
///                    ├──► Sub ──► output [1, C, H, W]
///   input [1,C,H,W] ─┘
/// ```
///
/// i.e. `output = 1.0 - input`. With Neo Lab's `[0, 1]` normalised BGRA,
/// this produces a visible photographic negative — a non-trivial proof
/// that tract's graph path (initializer tensors, broadcast rules, Sub
/// op) works end-to-end through the Phase B.1 bridge.
pub fn invert_model_f32_nchw(
    out_path: &Path,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(), String> {
    let bytes = build_invert_model_bytes(batch, channels, height, width);
    let mut file =
        std::fs::File::create(out_path).map_err(|e| format!("create {out_path:?}: {e}"))?;
    file.write_all(&bytes)
        .map_err(|e| format!("write {out_path:?}: {e}"))?;
    Ok(())
}

/// Generate a `Mul(scalar, x)` model — useful as a brightness or
/// gain control. With `factor < 1` it darkens, `> 1` it brightens.
pub fn mul_const_model_f32_nchw(
    out_path: &Path,
    factor: f32,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(), String> {
    let bytes =
        build_scalar_binary_model_bytes("Mul", factor, batch, channels, height, width);
    let mut file =
        std::fs::File::create(out_path).map_err(|e| format!("create {out_path:?}: {e}"))?;
    file.write_all(&bytes)
        .map_err(|e| format!("write {out_path:?}: {e}"))?;
    Ok(())
}

/// Generate an `Add(scalar, x)` model — a brightness offset filter.
pub fn add_const_model_f32_nchw(
    out_path: &Path,
    offset: f32,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(), String> {
    let bytes =
        build_scalar_binary_model_bytes("Add", offset, batch, channels, height, width);
    let mut file =
        std::fs::File::create(out_path).map_err(|e| format!("create {out_path:?}: {e}"))?;
    file.write_all(&bytes)
        .map_err(|e| format!("write {out_path:?}: {e}"))?;
    Ok(())
}

/// Build the bytes of a single-node unary ONNX graph
/// (`op_type` ∈ {Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Tanh, Sigmoid,
/// Relu, Floor, Ceil, Round, Reciprocal}).
pub fn build_unary_model_bytes(
    op_type: &str,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<u8> {
    let make_dim = |v: usize| pb::tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(v as i64)),
    };
    let image_shape = pb::TensorShapeProto {
        dim: vec![
            make_dim(batch),
            make_dim(channels),
            make_dim(height),
            make_dim(width),
        ],
    };
    let float_image_type = pb::type_proto::Tensor {
        elem_type: pb::tensor_proto::DataType::Float as i32,
        shape: Some(image_shape),
    };
    let image_type_proto = pb::TypeProto {
        denotation: String::new(),
        value: Some(pb::type_proto::Value::TensorType(float_image_type)),
    };
    let input = pb::ValueInfoProto {
        name: "input".to_string(),
        r#type: Some(image_type_proto.clone()),
        doc_string: String::new(),
    };
    let output = pb::ValueInfoProto {
        name: "output".to_string(),
        r#type: Some(image_type_proto),
        doc_string: String::new(),
    };
    let node = pb::NodeProto {
        input: vec!["input".to_string()],
        output: vec!["output".to_string()],
        name: format!("neo_unary_{}", op_type.to_lowercase()),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: vec![],
        doc_string: String::new(),
    };
    let graph = pb::GraphProto {
        node: vec![node],
        name: format!("neo_unary_{}", op_type.to_lowercase()),
        initializer: vec![],
        sparse_initializer: vec![],
        doc_string: String::new(),
        input: vec![input],
        output: vec![output],
        value_info: vec![],
        quantization_annotation: vec![],
    };
    let opset = pb::OperatorSetIdProto {
        domain: String::new(),
        version: 13,
    };
    let model = pb::ModelProto {
        ir_version: 8,
        opset_import: vec![opset],
        producer_name: "neo-infer-ort".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
    };
    let mut buf = Vec::with_capacity(model.encoded_len());
    model.encode(&mut buf).unwrap();
    buf
}

/// Convenience: write a unary model to disk.
pub fn unary_model_f32_nchw(
    out_path: &Path,
    op_type: &str,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(), String> {
    let bytes = build_unary_model_bytes(op_type, batch, channels, height, width);
    let mut file =
        std::fs::File::create(out_path).map_err(|e| format!("create {out_path:?}: {e}"))?;
    file.write_all(&bytes)
        .map_err(|e| format!("write {out_path:?}: {e}"))?;
    Ok(())
}

/// Build a Clip(input, min, max) ONNX graph. In opset 11+ Clip takes
/// two optional scalar initializer inputs for its min/max bounds.
pub fn build_clip_model_bytes(
    min_val: f32,
    max_val: f32,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<u8> {
    let make_dim = |v: usize| pb::tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(v as i64)),
    };
    let image_shape = pb::TensorShapeProto {
        dim: vec![
            make_dim(batch),
            make_dim(channels),
            make_dim(height),
            make_dim(width),
        ],
    };
    let float_image_type = pb::type_proto::Tensor {
        elem_type: pb::tensor_proto::DataType::Float as i32,
        shape: Some(image_shape),
    };
    let image_type_proto = pb::TypeProto {
        denotation: String::new(),
        value: Some(pb::type_proto::Value::TensorType(float_image_type)),
    };
    let input = pb::ValueInfoProto {
        name: "input".to_string(),
        r#type: Some(image_type_proto.clone()),
        doc_string: String::new(),
    };
    let output = pb::ValueInfoProto {
        name: "output".to_string(),
        r#type: Some(image_type_proto),
        doc_string: String::new(),
    };
    let scalar_init = |name: &str, v: f32| pb::TensorProto {
        dims: vec![1],
        data_type: pb::tensor_proto::DataType::Float as i32,
        segment: None,
        float_data: vec![v],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: name.to_string(),
        raw_data: vec![],
        external_data: vec![],
        data_location: Some(0),
        double_data: vec![],
        uint64_data: vec![],
        doc_string: String::new(),
    };
    let min_init = scalar_init("clip_min", min_val);
    let max_init = scalar_init("clip_max", max_val);

    let node = pb::NodeProto {
        input: vec![
            "input".to_string(),
            "clip_min".to_string(),
            "clip_max".to_string(),
        ],
        output: vec!["output".to_string()],
        name: "neo_clip".to_string(),
        op_type: "Clip".to_string(),
        domain: String::new(),
        attribute: vec![],
        doc_string: String::new(),
    };
    let graph = pb::GraphProto {
        node: vec![node],
        name: "neo_clip".to_string(),
        initializer: vec![min_init, max_init],
        sparse_initializer: vec![],
        doc_string: String::new(),
        input: vec![input],
        output: vec![output],
        value_info: vec![],
        quantization_annotation: vec![],
    };
    let opset = pb::OperatorSetIdProto {
        domain: String::new(),
        version: 13,
    };
    let model = pb::ModelProto {
        ir_version: 8,
        opset_import: vec![opset],
        producer_name: "neo-infer-ort".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
    };
    let mut buf = Vec::with_capacity(model.encoded_len());
    model.encode(&mut buf).unwrap();
    buf
}

pub fn clip_model_f32_nchw(
    out_path: &Path,
    min_val: f32,
    max_val: f32,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(), String> {
    let bytes = build_clip_model_bytes(min_val, max_val, batch, channels, height, width);
    let mut file =
        std::fs::File::create(out_path).map_err(|e| format!("create {out_path:?}: {e}"))?;
    file.write_all(&bytes)
        .map_err(|e| format!("write {out_path:?}: {e}"))?;
    Ok(())
}

/// Generate a 2-node ONNX graph: `output = mul_factor * (const_a - input)`.
///
/// This is the smallest non-trivial *chained* model. It validates two
/// things in one shot: that the wgpu pattern matcher walks chains and
/// that the inference engine ping-pongs tensors between ops correctly.
pub fn sub_then_mul_model_f32_nchw(
    out_path: &Path,
    const_a: f32,
    mul_factor: f32,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(), String> {
    let bytes = build_sub_then_mul_model_bytes(const_a, mul_factor, batch, channels, height, width);
    let mut file =
        std::fs::File::create(out_path).map_err(|e| format!("create {out_path:?}: {e}"))?;
    file.write_all(&bytes)
        .map_err(|e| format!("write {out_path:?}: {e}"))?;
    Ok(())
}

pub fn build_sub_then_mul_model_bytes(
    const_a: f32,
    mul_factor: f32,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<u8> {
    let make_dim = |v: usize| pb::tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(v as i64)),
    };
    let image_shape = pb::TensorShapeProto {
        dim: vec![
            make_dim(batch),
            make_dim(channels),
            make_dim(height),
            make_dim(width),
        ],
    };
    let float_image_type = pb::type_proto::Tensor {
        elem_type: pb::tensor_proto::DataType::Float as i32,
        shape: Some(image_shape),
    };
    let image_type_proto = pb::TypeProto {
        denotation: String::new(),
        value: Some(pb::type_proto::Value::TensorType(float_image_type)),
    };
    let input = pb::ValueInfoProto {
        name: "input".to_string(),
        r#type: Some(image_type_proto.clone()),
        doc_string: String::new(),
    };
    let output = pb::ValueInfoProto {
        name: "output".to_string(),
        r#type: Some(image_type_proto),
        doc_string: String::new(),
    };

    let scalar_init = |name: &str, v: f32| pb::TensorProto {
        dims: vec![1],
        data_type: pb::tensor_proto::DataType::Float as i32,
        segment: None,
        float_data: vec![v],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: name.to_string(),
        raw_data: vec![],
        external_data: vec![],
        data_location: Some(0),
        double_data: vec![],
        uint64_data: vec![],
        doc_string: String::new(),
    };

    let const_a_init = scalar_init("const_a", const_a);
    let const_mul_init = scalar_init("const_mul", mul_factor);

    // Node 1: tmp = const_a - input  → Sub(const_a, input)
    let sub_node = pb::NodeProto {
        input: vec!["const_a".to_string(), "input".to_string()],
        output: vec!["tmp".to_string()],
        name: "neo_sub_node".to_string(),
        op_type: "Sub".to_string(),
        domain: String::new(),
        attribute: vec![],
        doc_string: String::new(),
    };
    // Node 2: output = const_mul * tmp → Mul(const_mul, tmp)
    let mul_node = pb::NodeProto {
        input: vec!["const_mul".to_string(), "tmp".to_string()],
        output: vec!["output".to_string()],
        name: "neo_mul_node".to_string(),
        op_type: "Mul".to_string(),
        domain: String::new(),
        attribute: vec![],
        doc_string: String::new(),
    };

    let graph = pb::GraphProto {
        node: vec![sub_node, mul_node],
        name: "neo_sub_then_mul".to_string(),
        initializer: vec![const_a_init, const_mul_init],
        sparse_initializer: vec![],
        doc_string: String::new(),
        input: vec![input],
        output: vec![output],
        value_info: vec![],
        quantization_annotation: vec![],
    };

    let opset = pb::OperatorSetIdProto {
        domain: String::new(),
        version: 13,
    };
    let model = pb::ModelProto {
        ir_version: 8,
        opset_import: vec![opset],
        producer_name: "neo-infer-ort".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
    };
    let mut buf = Vec::with_capacity(model.encoded_len());
    model
        .encode(&mut buf)
        .expect("prost encode of ModelProto cannot fail on owned buffer");
    buf
}

/// Shared generator for `op(const, x)` graphs (op ∈ {Sub, Add, Mul}).
/// Used internally by [`build_invert_model_bytes`],
/// [`mul_const_model_f32_nchw`], and friends.
pub fn build_scalar_binary_model_bytes(
    op_type: &str,
    scalar: f32,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<u8> {
    let make_dim = |v: usize| pb::tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(v as i64)),
    };
    let image_shape = pb::TensorShapeProto {
        dim: vec![
            make_dim(batch),
            make_dim(channels),
            make_dim(height),
            make_dim(width),
        ],
    };
    let float_image_type = pb::type_proto::Tensor {
        elem_type: pb::tensor_proto::DataType::Float as i32,
        shape: Some(image_shape),
    };
    let image_type_proto = pb::TypeProto {
        denotation: String::new(),
        value: Some(pb::type_proto::Value::TensorType(float_image_type)),
    };
    let input = pb::ValueInfoProto {
        name: "input".to_string(),
        r#type: Some(image_type_proto.clone()),
        doc_string: String::new(),
    };
    let output = pb::ValueInfoProto {
        name: "output".to_string(),
        r#type: Some(image_type_proto),
        doc_string: String::new(),
    };

    let const_init = pb::TensorProto {
        dims: vec![1],
        data_type: pb::tensor_proto::DataType::Float as i32,
        segment: None,
        float_data: vec![scalar],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: "const_scalar".to_string(),
        raw_data: vec![],
        external_data: vec![],
        data_location: Some(0),
        double_data: vec![],
        uint64_data: vec![],
        doc_string: String::new(),
    };

    let node = pb::NodeProto {
        input: vec!["const_scalar".to_string(), "input".to_string()],
        output: vec!["output".to_string()],
        name: format!("neo_{}_node", op_type.to_lowercase()),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: vec![],
        doc_string: String::new(),
    };

    let graph = pb::GraphProto {
        node: vec![node],
        name: format!("neo_{}_const", op_type.to_lowercase()),
        initializer: vec![const_init],
        sparse_initializer: vec![],
        doc_string: String::new(),
        input: vec![input],
        output: vec![output],
        value_info: vec![],
        quantization_annotation: vec![],
    };

    let opset = pb::OperatorSetIdProto {
        domain: String::new(),
        version: 13,
    };

    let model = pb::ModelProto {
        ir_version: 8,
        opset_import: vec![opset],
        producer_name: "neo-infer-ort".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
    };

    let mut buf = Vec::with_capacity(model.encoded_len());
    model
        .encode(&mut buf)
        .expect("prost encode of ModelProto cannot fail on owned buffer");
    buf
}

pub fn build_invert_model_bytes(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<u8> {
    let make_dim = |v: usize| pb::tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(v as i64)),
    };
    let image_shape = pb::TensorShapeProto {
        dim: vec![
            make_dim(batch),
            make_dim(channels),
            make_dim(height),
            make_dim(width),
        ],
    };
    let float_image_type = pb::type_proto::Tensor {
        elem_type: pb::tensor_proto::DataType::Float as i32,
        shape: Some(image_shape),
    };
    let image_type_proto = pb::TypeProto {
        denotation: String::new(),
        value: Some(pb::type_proto::Value::TensorType(float_image_type)),
    };
    let input = pb::ValueInfoProto {
        name: "input".to_string(),
        r#type: Some(image_type_proto.clone()),
        doc_string: String::new(),
    };
    let output = pb::ValueInfoProto {
        name: "output".to_string(),
        r#type: Some(image_type_proto),
        doc_string: String::new(),
    };

    // Constant `1.0` scalar tensor, broadcast by Sub's NumPy rules
    // across the full [N, C, H, W] input.
    let const_one = pb::TensorProto {
        dims: vec![1],
        data_type: pb::tensor_proto::DataType::Float as i32,
        segment: None,
        float_data: vec![1.0],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: "const_one".to_string(),
        raw_data: vec![],
        external_data: vec![],
        data_location: Some(0),
        double_data: vec![],
        uint64_data: vec![],
        doc_string: String::new(),
    };

    let sub_node = pb::NodeProto {
        input: vec!["const_one".to_string(), "input".to_string()],
        output: vec!["output".to_string()],
        name: "invert_sub".to_string(),
        op_type: "Sub".to_string(),
        domain: String::new(),
        attribute: vec![],
        doc_string: String::new(),
    };

    let graph = pb::GraphProto {
        node: vec![sub_node],
        name: "neo_invert".to_string(),
        initializer: vec![const_one],
        sparse_initializer: vec![],
        doc_string: String::new(),
        input: vec![input],
        output: vec![output],
        value_info: vec![],
        quantization_annotation: vec![],
    };

    let opset = pb::OperatorSetIdProto {
        domain: String::new(),
        version: 13,
    };

    let model = pb::ModelProto {
        ir_version: 8,
        opset_import: vec![opset],
        producer_name: "neo-infer-ort".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
    };

    let mut buf = Vec::with_capacity(model.encoded_len());
    model
        .encode(&mut buf)
        .expect("prost encode of ModelProto cannot fail on owned buffer");
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OnnxModel;

    #[test]
    fn identity_roundtrip_loads_in_tract() {
        // Small dummy shape so the test runs fast.
        let tmp = std::env::temp_dir().join("neo_identity_test.onnx");
        identity_model_f32_nchw(&tmp, 1, 3, 4, 4).expect("generate identity model");

        let model = OnnxModel::load(&tmp).expect("tract loads identity model");
        assert_eq!(model.input_shape(), &[1, 3, 4, 4]);

        // Sanity: running it returns the input unchanged.
        let input: Vec<f32> = (0..48).map(|i| i as f32 / 48.0).collect();
        let output = model.infer(&input).expect("identity inference");
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "identity altered value: in={a} out={b}"
            );
        }

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn mul_const_roundtrip() {
        let tmp = std::env::temp_dir().join("neo_mul_test.onnx");
        mul_const_model_f32_nchw(&tmp, 0.5, 1, 3, 2, 2).expect("generate mul model");
        let model = OnnxModel::load(&tmp).expect("tract loads mul model");
        let input: Vec<f32> = (0..12).map(|i| i as f32 / 12.0).collect();
        let output = model.infer(&input).expect("mul inference");
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a * 0.5 - b).abs() < 1e-5);
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn add_const_roundtrip() {
        let tmp = std::env::temp_dir().join("neo_add_test.onnx");
        add_const_model_f32_nchw(&tmp, 0.25, 1, 3, 2, 2).expect("generate add model");
        let model = OnnxModel::load(&tmp).expect("tract loads add model");
        let input: Vec<f32> = (0..12).map(|i| i as f32 / 12.0).collect();
        let output = model.infer(&input).expect("add inference");
        for (a, b) in input.iter().zip(output.iter()) {
            assert!(((a + 0.25) - b).abs() < 1e-5);
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn sqrt_unary_roundtrip() {
        let tmp = std::env::temp_dir().join("neo_sqrt_test.onnx");
        unary_model_f32_nchw(&tmp, "Sqrt", 1, 3, 2, 2).unwrap();
        let model = OnnxModel::load(&tmp).unwrap();
        let input: Vec<f32> = (0..12).map(|i| (i as f32) / 11.0).collect();
        let output = model.infer(&input).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a.sqrt() - b).abs() < 1e-5);
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn relu_unary_roundtrip() {
        let tmp = std::env::temp_dir().join("neo_relu_test.onnx");
        unary_model_f32_nchw(&tmp, "Relu", 1, 3, 2, 2).unwrap();
        let model = OnnxModel::load(&tmp).unwrap();
        let input: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) / 6.0).collect();
        let output = model.infer(&input).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a.max(0.0) - b).abs() < 1e-5);
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn clip_roundtrip() {
        let tmp = std::env::temp_dir().join("neo_clip_test.onnx");
        clip_model_f32_nchw(&tmp, 0.25, 0.75, 1, 3, 2, 2).unwrap();
        let model = OnnxModel::load(&tmp).unwrap();
        let input: Vec<f32> = (0..12).map(|i| (i as f32) / 11.0).collect();
        let output = model.infer(&input).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a.clamp(0.25, 0.75) - b).abs() < 1e-5);
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn sub_then_mul_roundtrip() {
        // output = 0.5 * (1.0 - input)
        let tmp = std::env::temp_dir().join("neo_sub_then_mul_test.onnx");
        sub_then_mul_model_f32_nchw(&tmp, 1.0, 0.5, 1, 3, 2, 2).expect("generate");
        let model = OnnxModel::load(&tmp).expect("tract loads chained model");
        let input: Vec<f32> = (0..12).map(|i| i as f32 / 12.0).collect();
        let output = model.infer(&input).expect("infer");
        for (a, b) in input.iter().zip(output.iter()) {
            let want = 0.5 * (1.0 - a);
            assert!(
                (want - b).abs() < 1e-5,
                "expected {want}, got {b} (input={a})"
            );
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn invert_roundtrip_produces_1_minus_input() {
        let tmp = std::env::temp_dir().join("neo_invert_test.onnx");
        invert_model_f32_nchw(&tmp, 1, 3, 2, 2).expect("generate invert model");

        let model = OnnxModel::load(&tmp).expect("tract loads invert model");
        assert_eq!(model.input_shape(), &[1, 3, 2, 2]);

        let input: Vec<f32> = vec![
            0.0, 0.1, 0.2, 0.3, // R plane
            0.4, 0.5, 0.6, 0.7, // G plane
            0.8, 0.9, 1.0, 0.25, // B plane
        ];
        let output = model.infer(&input).expect("invert inference");
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!(
                ((1.0 - a) - b).abs() < 1e-5,
                "expected 1-{a} ≈ {b}, diff={}",
                (1.0 - a - b).abs()
            );
        }

        std::fs::remove_file(&tmp).ok();
    }
}
