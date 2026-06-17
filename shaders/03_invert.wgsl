// Invert — photo negative.

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
    let b = 255u - ((pix >>  0u) & 0xffu);
    let g = 255u - ((pix >>  8u) & 0xffu);
    let r = 255u - ((pix >> 16u) & 0xffu);
    out_buf[idx] = b | (g << 8u) | (r << 16u) | (255u << 24u);
}
