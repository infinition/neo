/// Built-in WGSL compute shaders for video processing operations.
///
/// All shaders operate on packed RGBA u32 pixels:
///   pixel = R | (G << 8) | (B << 16) | (A << 24)
pub mod builtins {

    /// Grayscale conversion — converts RGBA to grayscale (luminance).
    /// Bindings: 0=input(read), 1=output(write), 2=params(uniform)
    pub const GRAYSCALE: &str = r#"
struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn grayscale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let idx = y * params.width + x;
    let pixel = input[idx];
    let r = f32(pixel & 0xFFu) / 255.0;
    let g = f32((pixel >> 8u) & 0xFFu) / 255.0;
    let b = f32((pixel >> 16u) & 0xFFu) / 255.0;
    let a = (pixel >> 24u) & 0xFFu;

    // BT.709 luminance
    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let l = u32(clamp(lum * 255.0, 0.0, 255.0));
    output[idx] = l | (l << 8u) | (l << 16u) | (a << 24u);
}
"#;

    /// Brightness/contrast adjustment.
    /// Bindings: 0=input(read), 1=output(write), 2=params(uniform)
    /// params.brightness: -1.0 to 1.0, params.contrast: 0.0 to 2.0
    pub const BRIGHTNESS_CONTRAST: &str = r#"
struct Params {
    width: u32,
    height: u32,
    brightness: f32,
    contrast: f32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn brightness_contrast(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let idx = y * params.width + x;
    let pixel = input[idx];
    var r = f32(pixel & 0xFFu) / 255.0;
    var g = f32((pixel >> 8u) & 0xFFu) / 255.0;
    var b = f32((pixel >> 16u) & 0xFFu) / 255.0;
    let a = (pixel >> 24u) & 0xFFu;

    // Apply brightness
    r = r + params.brightness;
    g = g + params.brightness;
    b = b + params.brightness;

    // Apply contrast around 0.5
    r = (r - 0.5) * params.contrast + 0.5;
    g = (g - 0.5) * params.contrast + 0.5;
    b = (b - 0.5) * params.contrast + 0.5;

    r = clamp(r, 0.0, 1.0);
    g = clamp(g, 0.0, 1.0);
    b = clamp(b, 0.0, 1.0);

    output[idx] = u32(r * 255.0) | (u32(g * 255.0) << 8u) | (u32(b * 255.0) << 16u) | (a << 24u);
}
"#;

    /// Sepia tone filter.
    /// Bindings: 0=input(read), 1=output(write), 2=params(uniform)
    pub const SEPIA: &str = r#"
struct Params {
    width: u32,
    height: u32,
    intensity: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn sepia(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let idx = y * params.width + x;
    let pixel = input[idx];
    let r = f32(pixel & 0xFFu) / 255.0;
    let g = f32((pixel >> 8u) & 0xFFu) / 255.0;
    let b = f32((pixel >> 16u) & 0xFFu) / 255.0;
    let a = (pixel >> 24u) & 0xFFu;

    // Sepia matrix
    var sr = r * 0.393 + g * 0.769 + b * 0.189;
    var sg = r * 0.349 + g * 0.686 + b * 0.168;
    var sb = r * 0.272 + g * 0.534 + b * 0.131;

    // Blend with original based on intensity
    sr = mix(r, sr, params.intensity);
    sg = mix(g, sg, params.intensity);
    sb = mix(b, sb, params.intensity);

    sr = clamp(sr, 0.0, 1.0);
    sg = clamp(sg, 0.0, 1.0);
    sb = clamp(sb, 0.0, 1.0);

    output[idx] = u32(sr * 255.0) | (u32(sg * 255.0) << 8u) | (u32(sb * 255.0) << 16u) | (a << 24u);
}
"#;

    /// Invert colors.
    /// Bindings: 0=input(read), 1=output(write), 2=params(uniform)
    pub const INVERT: &str = r#"
struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn invert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let idx = y * params.width + x;
    let pixel = input[idx];
    let r = 255u - (pixel & 0xFFu);
    let g = 255u - ((pixel >> 8u) & 0xFFu);
    let b = 255u - ((pixel >> 16u) & 0xFFu);
    let a = (pixel >> 24u) & 0xFFu;
    output[idx] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
"#;

    /// Bilinear upscale 2x.
    /// Input is WxH, output is 2W x 2H.
    /// Bindings: 0=input(read), 1=output(write), 2=params(uniform)
    pub const UPSCALE_2X: &str = r#"
struct Params {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn sample_pixel(x: i32, y: i32) -> vec4<f32> {
    let cx = clamp(x, 0, i32(params.in_width) - 1);
    let cy = clamp(y, 0, i32(params.in_height) - 1);
    let idx = u32(cy) * params.in_width + u32(cx);
    let pixel = input[idx];
    return vec4<f32>(
        f32(pixel & 0xFFu) / 255.0,
        f32((pixel >> 8u) & 0xFFu) / 255.0,
        f32((pixel >> 16u) & 0xFFu) / 255.0,
        f32((pixel >> 24u) & 0xFFu) / 255.0,
    );
}

@compute @workgroup_size(8, 8)
fn upscale_2x(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    if (ox >= params.out_width || oy >= params.out_height) {
        return;
    }

    // Map output pixel to input coordinates
    let fx = (f32(ox) + 0.5) * f32(params.in_width) / f32(params.out_width) - 0.5;
    let fy = (f32(oy) + 0.5) * f32(params.in_height) / f32(params.out_height) - 0.5;

    let ix = i32(floor(fx));
    let iy = i32(floor(fy));
    let dx = fx - f32(ix);
    let dy = fy - f32(iy);

    // Bilinear interpolation
    let p00 = sample_pixel(ix, iy);
    let p10 = sample_pixel(ix + 1, iy);
    let p01 = sample_pixel(ix, iy + 1);
    let p11 = sample_pixel(ix + 1, iy + 1);

    let top = mix(p00, p10, dx);
    let bottom = mix(p01, p11, dx);
    let result = mix(top, bottom, dy);

    let out_idx = oy * params.out_width + ox;
    output[out_idx] = u32(clamp(result.x * 255.0, 0.0, 255.0))
        | (u32(clamp(result.y * 255.0, 0.0, 255.0)) << 8u)
        | (u32(clamp(result.z * 255.0, 0.0, 255.0)) << 16u)
        | (u32(clamp(result.w * 255.0, 0.0, 255.0)) << 24u);
}
"#;

    /// Edge detection (Sobel filter).
    /// Bindings: 0=input(read), 1=output(write), 2=params(uniform)
    pub const EDGE_DETECT: &str = r#"
struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn luminance(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cy) * params.width + u32(cx);
    let pixel = input[idx];
    let r = f32(pixel & 0xFFu) / 255.0;
    let g = f32((pixel >> 8u) & 0xFFu) / 255.0;
    let b = f32((pixel >> 16u) & 0xFFu) / 255.0;
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

@compute @workgroup_size(8, 8)
fn edge_detect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    // Sobel X
    let gx = -luminance(x-1, y-1) + luminance(x+1, y-1)
           - 2.0*luminance(x-1, y) + 2.0*luminance(x+1, y)
           - luminance(x-1, y+1) + luminance(x+1, y+1);

    // Sobel Y
    let gy = -luminance(x-1, y-1) - 2.0*luminance(x, y-1) - luminance(x+1, y-1)
           + luminance(x-1, y+1) + 2.0*luminance(x, y+1) + luminance(x+1, y+1);

    let edge = clamp(sqrt(gx * gx + gy * gy), 0.0, 1.0);
    let e = u32(edge * 255.0);
    let idx = gid.y * params.width + gid.x;
    output[idx] = e | (e << 8u) | (e << 16u) | (255u << 24u);
}
"#;

    /// Gaussian blur (3x3 kernel).
    /// Bindings: 0=input(read), 1=output(write), 2=params(uniform)
    pub const BLUR: &str = r#"
struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn sample(x: i32, y: i32) -> vec3<f32> {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cy) * params.width + u32(cx);
    let pixel = input[idx];
    return vec3<f32>(
        f32(pixel & 0xFFu) / 255.0,
        f32((pixel >> 8u) & 0xFFu) / 255.0,
        f32((pixel >> 16u) & 0xFFu) / 255.0,
    );
}

@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    // 3x3 Gaussian: [1,2,1; 2,4,2; 1,2,1] / 16
    var sum = vec3<f32>(0.0);
    sum += sample(x-1, y-1) * 1.0;
    sum += sample(x,   y-1) * 2.0;
    sum += sample(x+1, y-1) * 1.0;
    sum += sample(x-1, y  ) * 2.0;
    sum += sample(x,   y  ) * 4.0;
    sum += sample(x+1, y  ) * 2.0;
    sum += sample(x-1, y+1) * 1.0;
    sum += sample(x,   y+1) * 2.0;
    sum += sample(x+1, y+1) * 1.0;
    sum = sum / 16.0;

    let idx = gid.y * params.width + gid.x;
    let orig = input[idx];
    let a = (orig >> 24u) & 0xFFu;
    output[idx] = u32(clamp(sum.x * 255.0, 0.0, 255.0))
        | (u32(clamp(sum.y * 255.0, 0.0, 255.0)) << 8u)
        | (u32(clamp(sum.z * 255.0, 0.0, 255.0)) << 16u)
        | (a << 24u);
}
"#;

    /// Sharpen filter (unsharp mask).
    /// Bindings: 0=input(read), 1=output(write), 2=params(uniform)
    pub const SHARPEN: &str = r#"
struct Params {
    width: u32,
    height: u32,
    strength: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn sample(x: i32, y: i32) -> vec3<f32> {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cy) * params.width + u32(cx);
    let pixel = input[idx];
    return vec3<f32>(
        f32(pixel & 0xFFu) / 255.0,
        f32((pixel >> 8u) & 0xFFu) / 255.0,
        f32((pixel >> 16u) & 0xFFu) / 255.0,
    );
}

@compute @workgroup_size(8, 8)
fn sharpen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let center = sample(x, y);
    // 3x3 blur for unsharp mask
    var blur_val = vec3<f32>(0.0);
    blur_val += sample(x-1, y-1) + sample(x, y-1) + sample(x+1, y-1);
    blur_val += sample(x-1, y)   + sample(x, y)   + sample(x+1, y);
    blur_val += sample(x-1, y+1) + sample(x, y+1) + sample(x+1, y+1);
    blur_val = blur_val / 9.0;

    let sharpened = center + (center - blur_val) * params.strength;
    let r = clamp(sharpened.x, 0.0, 1.0);
    let g = clamp(sharpened.y, 0.0, 1.0);
    let b = clamp(sharpened.z, 0.0, 1.0);

    let idx = gid.y * params.width + gid.x;
    let orig = input[idx];
    let a = (orig >> 24u) & 0xFFu;
    output[idx] = u32(r * 255.0) | (u32(g * 255.0) << 8u) | (u32(b * 255.0) << 16u) | (a << 24u);
}
"#;
}
