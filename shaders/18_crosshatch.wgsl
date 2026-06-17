// Crosshatch - fake pen shading by adding more diagonal ink in darker zones.
//
// Bright areas stay mostly paper white. As luminance drops, more hatch
// layers kick in until shadows become dense black texture.

const SPACING: u32 = 10u;

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn lum_at(idx: u32) -> f32 {
    let pix = in_buf[idx];
    let b = f32((pix >>  0u) & 0xffu);
    let g = f32((pix >>  8u) & 0xffu);
    let r = f32((pix >> 16u) & 0xffu);
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    let lum = lum_at(idx);

    let diag_a = ((x + y) % SPACING) == 0u;
    let diag_b = ((x + (dims.height - 1u - y)) % SPACING) == 0u;
    let diag_c = (((x + 3u) + y) % SPACING) == 0u;
    let diag_d = (((x + 3u) + (dims.height - 1u - y)) % SPACING) == 0u;

    var ink = false;
    if (lum < 220.0 && diag_a) { ink = true; }
    if (lum < 170.0 && diag_b) { ink = true; }
    if (lum < 110.0 && diag_c) { ink = true; }
    if (lum <  70.0 && diag_d) { ink = true; }

    let v: u32 = select(255u, 0u, ink);
    out_buf[idx] = v | (v << 8u) | (v << 16u) | (255u << 24u);
}
