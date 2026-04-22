// Scanlines - fake a CRT by darkening alternating rows.
//
// LINE_DARKEN controls how deep the dark rows go. BLOOM is a tiny lift
// on the bright rows so the picture still feels alive instead of dull.

const LINE_DARKEN: f32 = 0.72;
const BLOOM: f32       = 1.05;

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

    let factor = select(BLOOM, LINE_DARKEN, (y & 1u) == 1u);
    let b = u32(clamp(f32((pix >>  0u) & 0xffu) * factor, 0.0, 255.0));
    let g = u32(clamp(f32((pix >>  8u) & 0xffu) * factor, 0.0, 255.0));
    let r = u32(clamp(f32((pix >> 16u) & 0xffu) * factor, 0.0, 255.0));

    out_buf[idx] = b | (g << 8u) | (r << 16u) | (255u << 24u);
}
