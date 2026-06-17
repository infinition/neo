// Unsharp mask — restores perceived sharpness on soft footage.
//
// Standard photographer trick: subtract a blurred version from the
// original, then add the high-frequency residue back. The 5-tap cross
// kernel is the cheapest blur that still gives a visibly different
// result, and `STRENGTH` controls how aggressive the boost is.
//
// Edit STRENGTH while Lab is running. Around 0.6 is "natural", 1.5 is
// "Instagram filter", 3.0 starts to ring.

const STRENGTH: f32 = 1.4;

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

    let centre = fetch(x,     y    );
    let north  = fetch(x,     y - 1);
    let south  = fetch(x,     y + 1);
    let east   = fetch(x + 1, y    );
    let west   = fetch(x - 1, y    );

    let blur = (centre + north + south + east + west) / 5.0;
    let sharp = centre + (centre - blur) * STRENGTH;

    let r = u32(clamp(sharp.r, 0.0, 255.0));
    let g = u32(clamp(sharp.g, 0.0, 255.0));
    let b = u32(clamp(sharp.b, 0.0, 255.0));
    out_buf[gid.y * dims.width + gid.x] =
        b | (g << 8u) | (r << 16u) | (255u << 24u);
}
