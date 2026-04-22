//! `neo-gen-invert-onnx`
//!
//! Generate a single-op `Sub(1, x)` ONNX model on disk — a photographic
//! negative filter — so you can visually confirm Neo Lab's `--model`
//! path with a real, non-trivial graph.
//!
//! Usage:
//!
//! ```text
//! cargo run -p neo-infer-ort --bin neo-gen-invert-onnx --release -- \
//!     --width 1920 --height 1080 --out invert_1080p.onnx
//!
//! ./target/release/neo-ffmpeg.exe lab \
//!     -i benchmarks/src_1080p.h264 \
//!     -s shaders \
//!     -m invert_1080p.onnx
//! ```
//!
//! Unlike the identity generator, this exercises tract's full path:
//! constant initializers, binary element-wise ops with numpy broadcast,
//! and a result that's visually obvious on screen.

use neo_infer_ort::generate::invert_model_f32_nchw;
use std::path::PathBuf;

fn main() {
    let mut width: usize = 1920;
    let mut height: usize = 1080;
    let mut out = PathBuf::from("invert.onnx");

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

    match invert_model_f32_nchw(&out, 1, 3, height, width) {
        Ok(()) => {
            println!(
                "wrote {out:?}: invert (photo negative) model, input [1, 3, {height}, {width}] f32",
                out = out,
                height = height,
                width = width
            );
        }
        Err(e) => {
            eprintln!("failed to generate invert model: {e}");
            std::process::exit(1);
        }
    }
}

fn usage_and_exit(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("error: {msg}");
    }
    eprintln!(
        "usage: neo-gen-invert-onnx \
         [--width W] [--height H] [--out path.onnx]\n\
         defaults: --width 1920 --height 1080 --out invert.onnx"
    );
    std::process::exit(if msg.is_empty() { 0 } else { 2 });
}
