// Color grade — film-look LUT-style: lift, gamma, gain + slight teal/orange split.
// Tweak the constants while Lab is running for the demo.

const LIFT:   vec3<f32> = vec3<f32>(0.02, 0.01, 0.04); // shadows push
const GAMMA:  vec3<f32> = vec3<f32>(1.10, 1.00, 0.95); // midtone curve
const GAIN:   vec3<f32> = vec3<f32>(1.05, 1.02, 0.95); // highlights
const SAT:    f32       = 1.10;

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn unpack(p: u32) -> vec3<f32> {
    return vec3<f32>(
        f32((p >> 16u) & 0xffu),
        f32((p >>  8u) & 0xffu),
        f32((p >>  0u) & 0xffu),
    ) / 255.0;
}

fn pack(c: vec3<f32>) -> u32 {
    let q = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0;
    let r = u32(q.r);
    let g = u32(q.g);
    let b = u32(q.b);
    return b | (g << 8u) | (r << 16u) | (255u << 24u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dims.width || gid.y >= dims.height) { return; }
    let idx = gid.y * dims.width + gid.x;
    var c = unpack(in_buf[idx]);

    // Lift / gamma / gain (the standard LGG cinema control set).
    c = c + LIFT * (vec3<f32>(1.0) - c);
    c = pow(c, vec3<f32>(1.0) / GAMMA);
    c = c * GAIN;

    // Saturation around luma.
    let lum = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    c = mix(vec3<f32>(lum), c, SAT);

    out_buf[idx] = pack(c);
}
