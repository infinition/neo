// Emboss - turns local contrast into a raised relief.
//
// The trick is just "future pixel minus past pixel" plus a mid-grey
// bias. Flat areas land near 128, while edges tilt bright/dark as if a
// light was raking across the image.

const STRENGTH: f32 = 1.0;
const BIAS: f32     = 128.0;

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn fetch(x: i32, y: i32) -> vec3<f32> {
    let xi = clamp(x, 0, i32(dims.width)  - 1);
    let yi = clamp(y, 0, i32(dims.height) - 1);
    let p = in_buf[u32(yi) * dims.width + u32(xi)];
    return vec3<f32>(
        f32((p >> 16u) & 0xffu),
        f32((p >>  8u) & 0xffu),
        f32((p >>  0u) & 0xffu),
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= dims.width || gid.y >= dims.height) { return; }

    let a = fetch(x - 1, y - 1);
    let b = fetch(x + 1, y + 1);
    let emboss = (b - a) * STRENGTH + vec3<f32>(BIAS, BIAS, BIAS);

    let r = u32(clamp(emboss.r, 0.0, 255.0));
    let g = u32(clamp(emboss.g, 0.0, 255.0));
    let b2 = u32(clamp(emboss.b, 0.0, 255.0));
    out_buf[gid.y * dims.width + gid.x] =
        b2 | (g << 8u) | (r << 16u) | (255u << 24u);
}
