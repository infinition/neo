// Solarize - invert only the bright half of the tone curve.
// Old darkroom trick: keep shadows natural, flip highlights into surreal ones.

const THRESHOLD: f32 = 132.0;

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn solarize(v: f32) -> f32 {
    if (v > THRESHOLD) {
        return 255.0 - v;
    }
    return v;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let idx = y * dims.width + x;
    let pix = in_buf[idx];

    let b = u32(clamp(solarize(f32((pix >>  0u) & 0xffu)), 0.0, 255.0));
    let g = u32(clamp(solarize(f32((pix >>  8u) & 0xffu)), 0.0, 255.0));
    let r = u32(clamp(solarize(f32((pix >> 16u) & 0xffu)), 0.0, 255.0));

    out_buf[idx] = b | (g << 8u) | (r << 16u) | (255u << 24u);
}
