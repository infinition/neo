// Vignette — radial darkening from the center.

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

    // Distance from center, normalized so the corner is ~1.
    let cx = f32(dims.width)  * 0.5;
    let cy = f32(dims.height) * 0.5;
    let dx = (f32(x) - cx) / cx;
    let dy = (f32(y) - cy) / cy;
    let d  = sqrt(dx * dx + dy * dy);

    // Soft falloff: full brightness inside ~50% radius, near-black at corners.
    let strength = 0.85;
    let falloff  = 1.5;
    let factor   = clamp(1.0 - strength * pow(d, falloff), 0.0, 1.0);

    let pix = in_buf[idx];
    let b = u32(f32((pix >>  0u) & 0xffu) * factor);
    let g = u32(f32((pix >>  8u) & 0xffu) * factor);
    let r = u32(f32((pix >> 16u) & 0xffu) * factor);
    out_buf[idx] = b | (g << 8u) | (r << 16u) | (255u << 24u);
}
