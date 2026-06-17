// Pencil sketch — outputs the source as if drawn on white paper.
//
// The trick: take an inverted Sobel magnitude. Where the original has
// strong edges, you want dark pencil lines on a bright background, so
// `255 - mag` flips a high gradient into a dark mark while flat areas
// stay near white.
//
// STRENGTH controls how dark the lines get; LIFT keeps the paper from
// looking grey by clamping the bright areas right at white.

const STRENGTH: f32 = 1.6;
const LIFT: f32     = 230.0;

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn fetch_lum(x: i32, y: i32) -> f32 {
    let xi = clamp(x, 0, i32(dims.width)  - 1);
    let yi = clamp(y, 0, i32(dims.height) - 1);
    let p = in_buf[u32(yi) * dims.width + u32(xi)];
    let b = f32((p >>  0u) & 0xffu);
    let g = f32((p >>  8u) & 0xffu);
    let r = f32((p >> 16u) & 0xffu);
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= dims.width || gid.y >= dims.height) { return; }

    let tl = fetch_lum(x - 1, y - 1);
    let tc = fetch_lum(x,     y - 1);
    let tr = fetch_lum(x + 1, y - 1);
    let ml = fetch_lum(x - 1, y    );
    let mr = fetch_lum(x + 1, y    );
    let bl_ = fetch_lum(x - 1, y + 1);
    let bc = fetch_lum(x,     y + 1);
    let br = fetch_lum(x + 1, y + 1);

    let gxv = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl_);
    let gyv = (bl_ + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    let mag = sqrt(gxv * gxv + gyv * gyv) * STRENGTH;

    let v = u32(clamp(LIFT - mag, 0.0, 255.0));
    let pix = v | (v << 8u) | (v << 16u) | (255u << 24u);
    out_buf[gid.y * dims.width + gid.x] = pix;
}
