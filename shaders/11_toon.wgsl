// Toon — cel-shading without any neural network.
//
// Recipe (the same trick studios used before AI cartoonifiers existed):
//   1. Posterize each colour channel into LEVELS bands → flat colour areas.
//   2. Boost saturation a bit to make those flat areas pop.
//   3. Run a Sobel pass on luminance and overlay black ink where the
//      gradient is strong → "drawn" outlines.
//
// Tweak LEVELS, SAT and EDGE_THRESHOLD while Lab is running and the
// frame reacts on the next save.

const LEVELS: f32          = 5.0;
const SATURATION: f32      = 1.45;
const EDGE_THRESHOLD: f32  = 55.0;

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
    let idx = gid.y * dims.width + gid.x;

    // ---- 1. Posterize the centre pixel ---------------------------------
    let p = in_buf[idx];
    var bf = f32((p >>  0u) & 0xffu);
    var gf = f32((p >>  8u) & 0xffu);
    var rf = f32((p >> 16u) & 0xffu);

    rf = floor(rf / 256.0 * LEVELS) / LEVELS * 256.0;
    gf = floor(gf / 256.0 * LEVELS) / LEVELS * 256.0;
    bf = floor(bf / 256.0 * LEVELS) / LEVELS * 256.0;

    // ---- 2. Boost saturation around the local luma ---------------------
    let lum_centre = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    rf = clamp(lum_centre + (rf - lum_centre) * SATURATION, 0.0, 255.0);
    gf = clamp(lum_centre + (gf - lum_centre) * SATURATION, 0.0, 255.0);
    bf = clamp(lum_centre + (bf - lum_centre) * SATURATION, 0.0, 255.0);

    // ---- 3. Sobel edge magnitude on luminance --------------------------
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
    let mag = sqrt(gxv * gxv + gyv * gyv);

    // ---- 4. Ink stroke -------------------------------------------------
    if (mag > EDGE_THRESHOLD) {
        rf = 0.0;
        gf = 0.0;
        bf = 0.0;
    }

    let bi = u32(bf);
    let gi = u32(gf);
    let ri = u32(rf);
    out_buf[idx] = bi | (gi << 8u) | (ri << 16u) | (255u << 24u);
}
