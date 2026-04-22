//! `neo-gen-identity-onnx`
//!
//! Generate a minimal `[1, 3, H, W]` identity ONNX model on disk so you
//! can smoke-test Neo Lab's `--model` path without downloading or
//! authoring anything.
//!
//! Usage:
//! ```text
//! cargo run -p neo-infer-ort --bin neo-gen-identity-onnx --release -- \
//!     --width 1920 --height 1080 --out identity_1080p.onnx
//!
//! ./target/release/neo-ffmpeg.exe lab \
//!     -i benchmarks/src_1080p.h264 \
//!     -s shaders \
//!     -m identity_1080p.onnx
//! ```
//!
//! The model is a single `Identity` op — the resulting Lab run is a
//! visual no-op at the pixel level but it exercises the entire B.1
//! bridge: GPU → staging → host → tract inference → host → GPU.

use neo_infer_ort::generate::identity_model_f32_nchw;
use std::path::PathBuf;

fn main() {
    let mut width: usize = 1920;
    let mut height: usize = 1080;
    let mut out = PathBuf::from("identity.onnx");

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

    match identity_model_f32_nchw(&out, 1, 3, height, width) {
        Ok(()) => {
            println!(
                "wrote {out:?}: identity model, input [1, 3, {height}, {width}] f32",
                out = out,
                height = height,
                width = width
            );
        }
        Err(e) => {
            eprintln!("failed to generate identity model: {e}");
            std::process::exit(1);
        }
    }
}

fn usage_and_exit(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("error: {msg}");
    }
    eprintln!(
        "usage: neo-gen-identity-onnx \
         [--width W] [--height H] [--out path.onnx]\n\
         defaults: --width 1920 --height 1080 --out identity.onnx"
    );
    std::process::exit(if msg.is_empty() { 0 } else { 2 });
}
