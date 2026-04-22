//! `neo-gen-brightness-onnx`
//!
//! Generate a `Mul(scale, x)` ONNX model — a brightness/gain control.
//! The wgpu fast path picks this up as `WgpuPlanOp::MulConst` and runs
//! it as a single compute dispatch on the GPU, no CPU bounce.
//!
//! ```text
//! cargo run -p neo-infer-ort --bin neo-gen-brightness-onnx --release -- \
//!     --width 1920 --height 1080 --scale 1.5 --out brighter.onnx
//! ```

use neo_infer_ort::generate::mul_const_model_f32_nchw;
use std::path::PathBuf;

fn main() {
    let mut width: usize = 1920;
    let mut height: usize = 1080;
    let mut scale: f32 = 1.4;
    let mut out = PathBuf::from("brightness.onnx");

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--width" | "-w" => {
                width = args
                    .next()
                    .unwrap_or_else(|| usage_and_exit("missing value for --width"))
                    .parse()
                    .unwrap_or_else(|e| usage_and_exit(&format!("bad --width: {e}")));
            }
            "--height" | "-h" => {
                height = args
                    .next()
                    .unwrap_or_else(|| usage_and_exit("missing value for --height"))
                    .parse()
                    .unwrap_or_else(|e| usage_and_exit(&format!("bad --height: {e}")));
            }
            "--scale" | "-s" => {
                scale = args
                    .next()
                    .unwrap_or_else(|| usage_and_exit("missing value for --scale"))
                    .parse()
                    .unwrap_or_else(|e| usage_and_exit(&format!("bad --scale: {e}")));
            }
            "--out" | "-o" => {
                out = PathBuf::from(
                    args.next()
                        .unwrap_or_else(|| usage_and_exit("missing value for --out")),
                );
            }
            "--help" => usage_and_exit(""),
            other => usage_and_exit(&format!("unknown flag: {other}")),
        }
    }

    match mul_const_model_f32_nchw(&out, scale, 1, 3, height, width) {
        Ok(()) => {
            println!(
                "wrote {out:?}: brightness model (Mul x*{scale}), input [1, 3, {height}, {width}] f32",
                out = out,
                scale = scale,
                height = height,
                width = width
            );
        }
        Err(e) => {
            eprintln!("failed to generate brightness model: {e}");
            std::process::exit(1);
        }
    }
}

fn usage_and_exit(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("error: {msg}");
    }
    eprintln!(
        "usage: neo-gen-brightness-onnx \
         [--width W] [--height H] [--scale F] [--out path.onnx]\n\
         defaults: --width 1920 --height 1080 --scale 1.4 --out brightness.onnx"
    );
    std::process::exit(if msg.is_empty() { 0 } else { 2 });
}
