// Chromatic aberration — split R/G/B horizontally to fake a cheap lens.
// Tweak `OFFSET` and re-save while Lab is running to feel the hot reload.

const OFFSET: i32 = 6;

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn fetch(x: i32, y: i32) -> u32 {
    let xi = clamp(x, 0, i32(dims.width)  - 1);
    let yi = clamp(y, 0, i32(dims.height) - 1);
    return in_buf[u32(yi) * dims.width + u32(xi)];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= dims.width || gid.y >= dims.height) { return; }

    let pr = fetch(x - OFFSET, y);
    let pg = fetch(x,          y);
    let pb = fetch(x + OFFSET, y);

    let r = (pr >> 16u) & 0xffu;
    let g = (pg >>  8u) & 0xffu;
    let b = (pb >>  0u) & 0xffu;

    out_buf[gid.y * dims.width + gid.x] =
        b | (g << 8u) | (r << 16u) | (255u << 24u);
}
