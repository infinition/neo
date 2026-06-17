// Posterize - collapse smooth gradients into a small number of bands.
// Lower LEVELS for harsher comic-book blocks, raise it for a gentler cut.

const LEVELS: f32 = 6.0;

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn quantize(v: f32) -> f32 {
    let steps = max(LEVELS - 1.0, 1.0);
    return floor(v / 255.0 * steps + 0.5) / steps * 255.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    let pix = in_buf[idx];

    let b = quantize(f32((pix >>  0u) & 0xffu));
    let g = quantize(f32((pix >>  8u) & 0xffu));
    let r = quantize(f32((pix >> 16u) & 0xffu));

    out_buf[idx] =
        u32(b) | (u32(g) << 8u) | (u32(r) << 16u) | (255u << 24u);
}
