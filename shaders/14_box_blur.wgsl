// Box blur - a cheap 3x3 soften for glow, dream, or de-noise vibes.
// Edit RADIUS to widen the blur without changing the shader shape.

const RADIUS: i32 = 1;

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

    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var taps = 0.0;
    for (var oy = -RADIUS; oy <= RADIUS; oy = oy + 1) {
        for (var ox = -RADIUS; ox <= RADIUS; ox = ox + 1) {
            acc = acc + fetch(x + ox, y + oy);
            taps = taps + 1.0;
        }
    }

    let blur = acc / taps;
    let r = u32(clamp(blur.r, 0.0, 255.0));
    let g = u32(clamp(blur.g, 0.0, 255.0));
    let b = u32(clamp(blur.b, 0.0, 255.0));
    out_buf[gid.y * dims.width + gid.x] =
        b | (g << 8u) | (r << 16u) | (255u << 24u);
}
