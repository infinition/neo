// Threshold — high-contrast black/white. Edit `THRESHOLD` to taste.

const THRESHOLD: f32 = 128.0;

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
    let lum = 0.299 * r + 0.587 * g + 0.114 * b;
    let v: u32 = select(0u, 255u, lum >= THRESHOLD);
    out_buf[idx] = v | (v << 8u) | (v << 16u) | (255u << 24u);
}
