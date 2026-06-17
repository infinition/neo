// Sobel edge detect — outputs grayscale gradient magnitude.

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn lum_at(x: i32, y: i32) -> f32 {
    let xi = clamp(x, 0, i32(dims.width)  - 1);
    let yi = clamp(y, 0, i32(dims.height) - 1);
    let pix = in_buf[u32(yi) * dims.width + u32(xi)];
    let b = f32((pix >>  0u) & 0xffu);
    let g = f32((pix >>  8u) & 0xffu);
    let r = f32((pix >> 16u) & 0xffu);
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= dims.width || gid.y >= dims.height) { return; }

    let tl = lum_at(x - 1, y - 1);
    let tc = lum_at(x,     y - 1);
    let tr = lum_at(x + 1, y - 1);
    let ml = lum_at(x - 1, y    );
    let mr = lum_at(x + 1, y    );
    let bl = lum_at(x - 1, y + 1);
    let bc = lum_at(x,     y + 1);
    let br = lum_at(x + 1, y + 1);

    let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    let mag = clamp(sqrt(gx * gx + gy * gy), 0.0, 255.0);
    let v = u32(mag);
    out_buf[gid.y * dims.width + gid.x] =
        v | (v << 8u) | (v << 16u) | (255u << 24u);
}
