//! Built-in WGSL compute shaders — one per mosaic tile.
//!
//! Every shader has the standard neo-lab bind-group layout:
//!   @binding(0) uniform Dims { width, height }
//!   @binding(1) storage<read>       in_buf
//!   @binding(2) storage<read_write> out_buf

/// Returns `(name, wgsl_source)` for each built-in tile filter.
pub fn builtins() -> Vec<(&'static str, String)> {
    vec![
        ("original", shader(PASSTHROUGH)),
        ("grayscale", shader(GRAYSCALE)),
        ("invert", shader(INVERT)),
        ("sepia", shader(SEPIA)),
        ("edge-detect", shader(EDGE_DETECT)),
        ("high-contrast", shader(HIGH_CONTRAST)),
        ("posterize", shader(POSTERIZE)),
        ("thermal", shader(THERMAL)),
        ("emboss", shader(EMBOSS)),
    ]
}

fn shader(body: &str) -> String {
    format!("{PREAMBLE}\n{body}")
}

const PREAMBLE: &str = r#"
struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn unpack(p: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((p >>  0u) & 0xffu),
        f32((p >>  8u) & 0xffu),
        f32((p >> 16u) & 0xffu),
        f32((p >> 24u) & 0xffu),
    );
}
fn pack(c: vec4<f32>) -> u32 {
    let b = u32(clamp(c.x, 0.0, 255.0));
    let g = u32(clamp(c.y, 0.0, 255.0));
    let r = u32(clamp(c.z, 0.0, 255.0));
    let a = u32(clamp(c.w, 0.0, 255.0));
    return b | (g << 8u) | (r << 16u) | (a << 24u);
}
"#;

const PASSTHROUGH: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    out_buf[idx] = in_buf[idx];
}
"#;

const GRAYSCALE: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    let c = unpack(in_buf[idx]);
    let lum = 0.114 * c.x + 0.587 * c.y + 0.299 * c.z;
    out_buf[idx] = pack(vec4<f32>(lum, lum, lum, c.w));
}
"#;

const INVERT: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    let c = unpack(in_buf[idx]);
    out_buf[idx] = pack(vec4<f32>(255.0 - c.x, 255.0 - c.y, 255.0 - c.z, c.w));
}
"#;

const SEPIA: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    let c = unpack(in_buf[idx]);
    let r = c.z; let g = c.y; let b = c.x;
    let sr = clamp(r * 0.393 + g * 0.769 + b * 0.189, 0.0, 255.0);
    let sg = clamp(r * 0.349 + g * 0.686 + b * 0.168, 0.0, 255.0);
    let sb = clamp(r * 0.272 + g * 0.534 + b * 0.131, 0.0, 255.0);
    out_buf[idx] = pack(vec4<f32>(sb, sg, sr, c.w));
}
"#;

const EDGE_DETECT: &str = r#"
fn luma_at(idx: u32) -> f32 {
    let c = unpack(in_buf[idx]);
    return 0.114 * c.x + 0.587 * c.y + 0.299 * c.z;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    if (x == 0u || y == 0u || x >= dims.width - 1u || y >= dims.height - 1u) {
        out_buf[y * dims.width + x] = pack(vec4<f32>(0.0, 0.0, 0.0, 255.0));
        return;
    }
    let w = dims.width;
    let tl = luma_at((y-1u)*w + x-1u); let tc = luma_at((y-1u)*w + x); let tr = luma_at((y-1u)*w + x+1u);
    let ml = luma_at(y*w + x-1u);                                       let mr = luma_at(y*w + x+1u);
    let bl = luma_at((y+1u)*w + x-1u); let bc = luma_at((y+1u)*w + x); let br = luma_at((y+1u)*w + x+1u);
    let gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
    let gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
    let mag = clamp(sqrt(gx*gx + gy*gy), 0.0, 255.0);
    out_buf[y * w + x] = pack(vec4<f32>(mag, mag, mag, 255.0));
}
"#;

const HIGH_CONTRAST: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    let c = unpack(in_buf[idx]);
    let factor = 2.0;
    let b = clamp((c.x - 128.0) * factor + 128.0, 0.0, 255.0);
    let g = clamp((c.y - 128.0) * factor + 128.0, 0.0, 255.0);
    let r = clamp((c.z - 128.0) * factor + 128.0, 0.0, 255.0);
    out_buf[idx] = pack(vec4<f32>(b, g, r, c.w));
}
"#;

const POSTERIZE: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    let c = unpack(in_buf[idx]);
    let levels = 4.0;
    let b = floor(c.x / (256.0 / levels)) * (256.0 / levels);
    let g = floor(c.y / (256.0 / levels)) * (256.0 / levels);
    let r = floor(c.z / (256.0 / levels)) * (256.0 / levels);
    out_buf[idx] = pack(vec4<f32>(b, g, r, c.w));
}
"#;

const THERMAL: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    let c = unpack(in_buf[idx]);
    let lum = (0.114 * c.x + 0.587 * c.y + 0.299 * c.z) / 255.0;
    var r: f32; var g: f32; var b: f32;
    if (lum < 0.25) {
        let t = lum / 0.25;
        b = 255.0; g = t * 255.0; r = 0.0;
    } else if (lum < 0.5) {
        let t = (lum - 0.25) / 0.25;
        b = (1.0 - t) * 255.0; g = 255.0; r = 0.0;
    } else if (lum < 0.75) {
        let t = (lum - 0.5) / 0.25;
        b = 0.0; g = 255.0; r = t * 255.0;
    } else {
        let t = (lum - 0.75) / 0.25;
        b = 0.0; g = (1.0 - t) * 255.0; r = 255.0;
    }
    out_buf[idx] = pack(vec4<f32>(b, g, r, 255.0));
}
"#;

const EMBOSS: &str = r#"
fn luma_at(idx: u32) -> f32 {
    let c = unpack(in_buf[idx]);
    return 0.114 * c.x + 0.587 * c.y + 0.299 * c.z;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    if (x == 0u || y == 0u || x >= dims.width - 1u || y >= dims.height - 1u) {
        out_buf[y * dims.width + x] = pack(vec4<f32>(128.0, 128.0, 128.0, 255.0));
        return;
    }
    let w = dims.width;
    let tl = luma_at((y-1u)*w + x-1u);
    let br = luma_at((y+1u)*w + x+1u);
    let v = clamp((br - tl) + 128.0, 0.0, 255.0);
    out_buf[y * w + x] = pack(vec4<f32>(v, v, v, 255.0));
}
"#;
