// Halftone - newspaper dots whose radius grows as the image gets darker.
//
// Each cell samples its centre luminance, then draws a circular dot on
// white paper. Smaller CELL means finer print, larger CELL looks bolder.

const CELL: u32 = 8u;

struct Dims { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn lum_at(x: u32, y: u32) -> f32 {
    let xi = min(x, dims.width  - 1u);
    let yi = min(y, dims.height - 1u);
    let pix = in_buf[yi * dims.width + xi];
    let b = f32((pix >>  0u) & 0xffu);
    let g = f32((pix >>  8u) & 0xffu);
    let r = f32((pix >> 16u) & 0xffu);
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }

    let cell_x = (x / CELL) * CELL;
    let cell_y = (y / CELL) * CELL;
    let sample_x = min(cell_x + CELL / 2u, dims.width  - 1u);
    let sample_y = min(cell_y + CELL / 2u, dims.height - 1u);
    let lum = lum_at(sample_x, sample_y);

    let darkness = 1.0 - lum / 255.0;
    let radius = darkness * f32(CELL) * 0.48;

    let cx = f32(cell_x) + f32(CELL) * 0.5;
    let cy = f32(cell_y) + f32(CELL) * 0.5;
    let dx = f32(x) + 0.5 - cx;
    let dy = f32(y) + 0.5 - cy;
    let dist = sqrt(dx * dx + dy * dy);

    let v: u32 = select(255u, 0u, dist <= radius);
    out_buf[y * dims.width + x] = v | (v << 8u) | (v << 16u) | (255u << 24u);
}
