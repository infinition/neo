//! GGUF weight-map builder.
//!
//! We parse the GGUF header to locate every MoE expert tensor (gate + up + down
//! projections) and record their byte offset + length inside the file.
//! No tensor data is read at this stage — we only build the map.
//!
//! Tensor naming convention for Qwen3 MoE (mirrors llama.cpp):
//!   `blk.{layer}.ffn_gate_exps.{expert}.weight`
//!   `blk.{layer}.ffn_up_exps.{expert}.weight`
//!   `blk.{layer}.ffn_down_exps.{expert}.weight`

use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

use tracing::{debug, info, warn};

use crate::error::{MoeError, Result};
use crate::ExpertId;

/// Byte location of one tensor shard inside the GGUF file.
#[derive(Debug, Clone, Copy)]
pub struct TensorLocation {
    /// Absolute byte offset of the raw data in the file.
    pub offset: u64,
    /// Byte length of the raw (quantised) tensor data.
    pub byte_len: usize,
}

/// Which projection this entry represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Projection {
    Gate,
    Up,
    Down,
}

/// All three projections for one expert.
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    pub gate: TensorLocation,
    pub up: TensorLocation,
    pub down: TensorLocation,
}

impl ExpertWeights {
    /// Total bytes needed to load this expert into VRAM.
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.gate.byte_len + self.up.byte_len + self.down.byte_len
    }
}

/// Complete weight map parsed from a GGUF file.
pub struct GgufWeightMap {
    /// File path (kept for later mmap / O_DIRECT reads).
    pub path: std::path::PathBuf,
    /// Expert weight locations, keyed by ExpertId.
    pub experts: HashMap<ExpertId, ExpertWeights>,
    /// Number of transformer layers detected.
    pub n_layers: u32,
    /// Number of experts per MoE layer.
    pub n_experts: u32,
    /// Largest single expert byte size (used to size the VRAM pool slots).
    pub max_expert_bytes: usize,
}

// ─── GGUF binary format constants ────────────────────────────────────────────
// Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGUF_VERSION_MIN: u32 = 2;
const GGUF_VERSION_MAX: u32 = 3;

/// Minimal GGUF reader — we only need tensor names, offsets, and sizes.
struct GgufReader<R: Read + Seek> {
    inner: R,
    /// Data section start (after the header).
    data_offset: u64,
}

impl<R: Read + Seek> GgufReader<R> {
    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.inner.read_exact(&mut buf).map_err(MoeError::Io)?;
        Ok(buf[0])
    }
    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf).map_err(MoeError::Io)?;
        Ok(u32::from_le_bytes(buf))
    }
    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.inner.read_exact(&mut buf).map_err(MoeError::Io)?;
        Ok(u64::from_le_bytes(buf))
    }
    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }
    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }
    fn read_f32(&mut self) -> Result<f32> {
        let bits = self.read_u32()?;
        Ok(f32::from_bits(bits))
    }
    fn read_f64(&mut self) -> Result<f64> {
        let bits = self.read_u64()?;
        Ok(f64::from_bits(bits))
    }
    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf).map_err(MoeError::Io)?;
        String::from_utf8(buf).map_err(|e| MoeError::GgufParse(e.to_string()))
    }

    /// Skip over a metadata value without storing it.
    fn skip_value(&mut self, vtype: u32) -> Result<()> {
        match vtype {
            0 => { self.read_u8()?; }        // UINT8
            1 => { self.read_i32()?; }       // INT8 (stored as u8, same size)
            2 => { self.read_u32()?; }       // UINT16
            3 => { self.read_i32()?; }       // INT16
            4 => { self.read_u32()?; }       // UINT32
            5 => { self.read_i32()?; }       // INT32
            6 => { self.read_f32()?; }       // FLOAT32
            7 => { self.read_bool()?; }      // BOOL
            8 => { self.read_string()?; }    // STRING
            9 => {                           // ARRAY
                let elem_type = self.read_u32()?;
                let count = self.read_u64()?;
                for _ in 0..count {
                    self.skip_value(elem_type)?;
                }
            }
            10 => { self.read_u64()?; }      // UINT64
            11 => { self.read_i64()?; }      // INT64
            12 => { self.read_f64()?; }      // FLOAT64
            other => {
                return Err(MoeError::GgufParse(format!(
                    "unknown metadata value type {other}"
                )));
            }
        }
        Ok(())
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

impl GgufWeightMap {
    /// Parse a GGUF file and return the complete expert weight map.
    ///
    /// This is a header-only scan — no tensor data is read.
    /// Typical parse time for a 20 GB file: < 100 ms.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        info!("Parsing GGUF weight map: {}", path.display());

        let file = File::open(&path).map_err(MoeError::Io)?;
        let mut r = GgufReader {
            inner: BufReader::with_capacity(1 << 20, file), // 1 MB read-ahead
            data_offset: 0,
        };

        // ── Magic + version ──────────────────────────────────────────────────
        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(MoeError::GgufParse(format!(
                "bad magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}"
            )));
        }
        let version = r.read_u32()?;
        if !(GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&version) {
            warn!("GGUF version {version} is outside tested range {GGUF_VERSION_MIN}–{GGUF_VERSION_MAX}");
        }
        let tensor_count = r.read_u64()?;
        let kv_count = r.read_u64()?;
        debug!("GGUF v{version}: {tensor_count} tensors, {kv_count} KV pairs");

        // ── Skip metadata KV pairs (capture expert_count if present) ────────
        let mut metadata_expert_count: Option<u32> = None;
        for _ in 0..kv_count {
            let key = r.read_string()?;
            let vtype = r.read_u32()?;

            // qwen35moe.expert_count is UINT32 in current GGUFs.
            if key.ends_with(".expert_count") && vtype == 4 {
                let count = r.read_u32()?;
                if count > 0 {
                    metadata_expert_count = Some(count);
                }
                continue;
            }

            r.skip_value(vtype)?;
        }

        // ── Tensor info section ──────────────────────────────────────────────
        // Each entry: name (string), n_dims (u32), dims ([u64; n_dims]),
        //             ggml_type (u32), offset (u64, relative to data section)
        let mut raw_tensors: Vec<(String, u64, usize)> = Vec::with_capacity(tensor_count as usize);

        for _ in 0..tensor_count {
            let name   = r.read_string()?;
            let n_dims = r.read_u32()?;
            let mut n_elems: u64 = 1;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d = r.read_u64()?;
                dims.push(d);
                n_elems *= d;
            }
            let ggml_type = r.read_u32()?;
            let offset    = r.read_u64()?;      // relative to data section

            let byte_len = ggml_type_bytes(ggml_type, n_elems)?;
            raw_tensors.push((name, offset, byte_len));
        }

        // The data section starts at the next 32-byte aligned position after
        // the current seek position (GGUF v3 alignment).
        let header_end = r.inner.seek(SeekFrom::Current(0)).map_err(MoeError::Io)?;
        let data_offset = align_up(header_end, 32);

        // ── Filter expert tensors ─────────────────────────────────────────────
        // Qwen3 MoE tensor names (llama.cpp convention):
        //   blk.{L}.ffn_gate_exps.{E}.weight
        //   blk.{L}.ffn_up_exps.{E}.weight
        //   blk.{L}.ffn_down_exps.{E}.weight
        //
        // Mixtral uses:
        //   blk.{L}.ffn_gate_exp.{E}.weight   (singular "exp")
        //
        // We handle both.

        let mut expert_map: HashMap<ExpertId, (
            Option<TensorLocation>,
            Option<TensorLocation>,
            Option<TensorLocation>,
        )> = HashMap::new();

        let mut max_layer = 0u32;
        let mut max_expert = 0u32;
        let mut fused_map: HashMap<u32, (Option<TensorLocation>, Option<TensorLocation>, Option<TensorLocation>)> = HashMap::new();

        for (name, rel_offset, byte_len) in &raw_tensors {
            let loc = TensorLocation {
                offset:   data_offset + rel_offset,
                byte_len: *byte_len,
            };

            if let Some((layer, maybe_expert, proj)) = parse_expert_tensor_name(name) {
                max_layer = max_layer.max(layer);

                match maybe_expert {
                    Some(expert) => {
                        max_expert = max_expert.max(expert);
                        let entry = expert_map.entry(ExpertId::new(layer, expert)).or_default();
                        match proj {
                            Projection::Gate => entry.0 = Some(loc),
                            Projection::Up => entry.1 = Some(loc),
                            Projection::Down => entry.2 = Some(loc),
                        }
                    }
                    None => {
                        let entry = fused_map.entry(layer).or_default();
                        match proj {
                            Projection::Gate => entry.0 = Some(loc),
                            Projection::Up => entry.1 = Some(loc),
                            Projection::Down => entry.2 = Some(loc),
                        }
                    }
                }
            }
        }

        // Expand fused expert tensors (blk.L.ffn_*_exps.weight) into synthetic
        // per-expert slices when a model packs all experts in one tensor.
        let inferred_expert_count = if max_expert > 0 {
            max_expert + 1
        } else {
            metadata_expert_count.unwrap_or(1)
        };

        if !fused_map.is_empty() {
            for (layer, (gate, up, down)) in fused_map {
                let (gate, up, down) = match (gate, up, down) {
                    (Some(g), Some(u), Some(d)) => (g, u, d),
                    _ => {
                        warn!("Incomplete fused expert tensors for layer={layer}");
                        continue;
                    }
                };

                if inferred_expert_count == 0 {
                    continue;
                }

                let ec = inferred_expert_count as usize;
                if gate.byte_len % ec != 0 || up.byte_len % ec != 0 || down.byte_len % ec != 0 {
                    warn!(
                        "Fused tensor byte size is not divisible by expert_count={} at layer={}",
                        inferred_expert_count,
                        layer
                    );
                    continue;
                }

                let gate_stride = gate.byte_len / ec;
                let up_stride = up.byte_len / ec;
                let down_stride = down.byte_len / ec;

                for expert in 0..inferred_expert_count {
                    let e = expert as usize;
                    let id = ExpertId::new(layer, expert);
                    let entry = expert_map.entry(id).or_default();

                    if entry.0.is_none() {
                        entry.0 = Some(TensorLocation {
                            offset: gate.offset + (e * gate_stride) as u64,
                            byte_len: gate_stride,
                        });
                    }
                    if entry.1.is_none() {
                        entry.1 = Some(TensorLocation {
                            offset: up.offset + (e * up_stride) as u64,
                            byte_len: up_stride,
                        });
                    }
                    if entry.2.is_none() {
                        entry.2 = Some(TensorLocation {
                            offset: down.offset + (e * down_stride) as u64,
                            byte_len: down_stride,
                        });
                    }
                }

                max_expert = max_expert.max(inferred_expert_count.saturating_sub(1));
            }
        }

        // Assemble complete ExpertWeights, warn on incomplete entries.
        let mut experts: HashMap<ExpertId, ExpertWeights> = HashMap::new();
        let mut max_expert_bytes = 0usize;

        for (id, (gate, up, down)) in expert_map {
            match (gate, up, down) {
                (Some(g), Some(u), Some(d)) => {
                    let total = g.byte_len + u.byte_len + d.byte_len;
                    max_expert_bytes = max_expert_bytes.max(total);
                    experts.insert(id, ExpertWeights { gate: g, up: u, down: d });
                }
                _ => warn!("Incomplete expert projections for layer={} expert={}", id.layer, id.expert),
            }
        }

        info!(
            "Found {} experts across {} layers ({:.1} MB each, max)",
            experts.len(),
            max_layer + 1,
            max_expert_bytes as f64 / 1e6
        );

        Ok(GgufWeightMap {
            path,
            experts,
            n_layers:         max_layer + 1,
            n_experts:        max_expert + 1,
            max_expert_bytes,
        })
    }

    /// Lookup the weight locations for a specific expert.
    pub fn get(&self, id: ExpertId) -> Option<&ExpertWeights> {
        self.experts.get(&id)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Parse expert tensor names in two formats:
///
/// **Per-expert** (Mixtral/older Qwen): `blk.{L}.ffn_gate_exps.{E}.weight`
///   → (layer=L, expert=E, Gate)
///
/// **Fused** (Qwen3 UD/XL quants): `blk.{L}.ffn_gate_exps.weight`
///   → (layer=L, expert=0, Gate)  — the whole tensor covers all experts;
///     we register it as expert 0 so the pool slot covers the full tensor.
fn parse_expert_tensor_name(name: &str) -> Option<(u32, Option<u32>, Projection)> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 4 || parts[0] != "blk" {
        return None;
    }

    let layer: u32 = parts[1].parse().ok()?;
    let proj_tag = parts[2];

    // Only handle MoE expert tensors; exclude gate_inp/shexp/shared tensors.
    let proj = match proj_tag {
        "ffn_gate_exps" | "ffn_gate_exp" => Projection::Gate,
        "ffn_up_exps" | "ffn_up_exp" => Projection::Up,
        "ffn_down_exps" | "ffn_down_exp" => Projection::Down,
        _ => return None,
    };

    // Per-expert format: blk.L.ffn_xxx_exps.E.weight (5+ parts, E is u32)
    if parts.len() >= 5 {
        if parts[4] == "weight" {
            if let Ok(expert) = parts[3].parse::<u32>() {
                return Some((layer, Some(expert), proj));
            }
        }
    }

    // Fused format: blk.L.ffn_xxx_exps.weight (4 parts)
    // All experts packed into one tensor.
    if parts.len() == 4 && parts[3] == "weight" {
        return Some((layer, None, proj));
    }

    None
}

/// Compute byte size of a GGML tensor given its quantisation type and element count.
///
/// See: https://github.com/ggerganov/ggml/blob/master/include/ggml/ggml.h
fn ggml_type_bytes(ggml_type: u32, n_elems: u64) -> Result<usize> {
    // (type_id, block_size, bytes_per_block)
    // Only types relevant for Q4_K and related quantisations are listed.
    let (block_size, bytes_per_block): (u64, u64) = match ggml_type {
        0  => (1,    4),    // F32
        1  => (1,    2),    // F16
        2  => (32,   18),   // Q4_0
        3  => (32,   20),   // Q4_1
        6  => (32,   22),   // Q5_0
        7  => (32,   24),   // Q5_1
        8  => (32,   34),   // Q8_0
        9  => (1,    1),    // Q8_1 (handled below)
        10 => (256,  144),  // Q2_K
        11 => (256,  176),  // Q3_K_S
        12 => (256,  128),  // Q4_K_S
        13 => (256,  144),  // Q4_K_M (same block as S but different interleave — size is equal)
        14 => (256,  176),  // Q5_K_S
        15 => (256,  192),  // Q5_K_M
        16 => (256,  256),  // Q6_K
        17 => (256,  298),  // Q8_K (approximate)
        18 => (1,    4),    // IQ2_XXS (approximation)
        19 => (1,    4),    // IQ2_XS
        20 => (1,    4),    // IQ3_XXS
        21 => (1,    2),    // IQ1_S
        22 => (1,    4),    // IQ4_NL
        23 => (1,    4),    // IQ3_S
        24 => (1,    4),    // IQ2_S
        25 => (1,    4),    // IQ4_XS
        26 => (1,    1),    // I8
        27 => (1,    2),    // I16
        28 => (1,    4),    // I32
        29 => (1,    8),    // I64
        30 => (1,    8),    // F64
        31 => (1,    2),    // IQ1_M
        32 => (256,  144),  // BF16 – not a quant but same slot used for BF16
        other => {
            return Err(MoeError::GgufParse(format!(
                "unknown GGML type {other}"
            )));
        }
    };

    let n_blocks = n_elems.div_ceil(block_size);
    Ok((n_blocks * bytes_per_block) as usize)
}

#[inline]
fn align_up(value: u64, align: u64) -> u64 {
    (value + align - 1) & !(align - 1)
}
