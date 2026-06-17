// Duotone - remap luminance between one shadow colour and one highlight colour.
//
// Great for poster art: the source keeps its structure, but the palette
// collapses into two deliberate tones instead of full RGB realism.

const SHADOW_R: f32 =  20.0;
const SHADOW_G: f32 =  24.0;
const SHADOW_B: f32 =  52.0;
const LIGHT_R:  f32 = 255.0;
const LIGHT_G:  f32 = 214.0;
const LIGHT_B:  f32 =  96.0;

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    let pix = in_buf[idx];

    let b = f32((pix >>  0u) & 0xffu);
    let g = f32((pix >>  8u) & 0xffu);
    let r = f32((pix >> 16u) & 0xffu);
    let lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;

    let out_r = SHADOW_R + (LIGHT_R - SHADOW_R) * lum;
    let out_g = SHADOW_G + (LIGHT_G - SHADOW_G) * lum;
    let out_b = SHADOW_B + (LIGHT_B - SHADOW_B) * lum;

    out_buf[idx] =
        u32(clamp(out_b, 0.0, 255.0)) |
        (u32(clamp(out_g, 0.0, 255.0)) << 8u) |
        (u32(clamp(out_r, 0.0, 255.0)) << 16u) |
        (255u << 24u);
}
