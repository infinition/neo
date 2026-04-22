// Pixelate — replace each block with the color of its top-left pixel.

const BLOCK: u32 = 12u;

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let bx = (x / BLOCK) * BLOCK;
    let by = (y / BLOCK) * BLOCK;
    out_buf[y * dims.width + x] = in_buf[by * dims.width + bx];
}
