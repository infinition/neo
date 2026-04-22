// Sepia — vintage tint via the standard reinhard mix.

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

    let nr = clamp(0.393 * r + 0.769 * g + 0.189 * b, 0.0, 255.0);
    let ng = clamp(0.349 * r + 0.686 * g + 0.168 * b, 0.0, 255.0);
    let nb = clamp(0.272 * r + 0.534 * g + 0.131 * b, 0.0, 255.0);

    let bi = u32(nb);
    let gi = u32(ng);
    let ri = u32(nr);
    out_buf[idx] = bi | (gi << 8u) | (ri << 16u) | (255u << 24u);
}
