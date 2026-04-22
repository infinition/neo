//! `neo-gen-clip-onnx`
//!
//! Generate a single `Clip(input, min, max)` ONNX model. On [0, 1]
//! BGRA-normalised video this is a posteriser: values below `min` get
//! raised, values above `max` get crushed, creating hard bands.
//!
//! ```text
//! neo-gen-clip-onnx --min 0.3 --max 0.7 --width 1920 --height 1080 --out clip.onnx
//! ```

use neo_infer_ort::generate::clip_model_f32_nchw;
use std::path::PathBuf;

fn main() {
    let mut width: usize = 1920;
    let mut height: usize = 1080;
    let mut min_val: f32 = 0.2;
    let mut max_val: f32 = 0.8;
    let mut out = PathBuf::from("clip.onnx");

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
            "--min" => {
                min_val = args
                    .next()
                    .unwrap_or_else(|| usage_and_exit("missing value for --min"))
                    .parse()
                    .unwrap_or_else(|e| usage_and_exit(&format!("bad --min: {e}")));
            }
            "--max" => {
                max_val = args
                    .next()
                    .unwrap_or_else(|| usage_and_exit("missing value for --max"))
                    .parse()
                    .unwrap_or_else(|e| usage_and_exit(&format!("bad --max: {e}")));
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

    if min_val >= max_val {
        eprintln!(
            "error: --min ({min_val}) must be strictly less than --max ({max_val})"
        );
        std::process::exit(2);
    }

    match clip_model_f32_nchw(&out, min_val, max_val, 1, 3, height, width) {
        Ok(()) => {
            println!(
                "wrote {out:?}: clip model [{min_val}, {max_val}], input [1, 3, {height}, {width}] f32",
                out = out,
                min_val = min_val,
                max_val = max_val,
                height = height,
                width = width
            );
        }
        Err(e) => {
            eprintln!("failed to generate clip model: {e}");
            std::process::exit(1);
        }
    }
}

fn usage_and_exit(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("error: {msg}");
    }
    eprintln!(
        "usage: neo-gen-clip-onnx \
         [--min F] [--max F] [--width W] [--height H] [--out path.onnx]\n\
         defaults: --min 0.2 --max 0.8 --width 1920 --height 1080 --out clip.onnx"
    );
    std::process::exit(if msg.is_empty() { 0 } else { 2 });
}
