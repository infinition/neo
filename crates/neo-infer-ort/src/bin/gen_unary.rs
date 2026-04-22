//! `neo-gen-unary-onnx`
//!
//! Generate a single-op unary ONNX model for any of the 14 element-wise
//! functions the wgpu fast path supports: Neg, Abs, Sqrt, Exp, Log,
//! Sin, Cos, Tanh, Sigmoid, Relu, Floor, Ceil, Round, Reciprocal.
//!
//! Useful visual picks on [0, 1] BGRA-normalised video:
//!
//! | Op    | Effect                                           |
//! |-------|--------------------------------------------------|
//! | Sqrt  | Lift shadows, compress highlights (gamma-ish)   |
//! | Log   | Extreme highlight compression (film-scan look)  |
//! | Sin   | Periodic tone remap, very psychedelic           |
//! | Tanh  | Soft rolloff in midtones                        |
//! | Sigmoid | Squishes everything toward gray (~[0.5, 0.73]) |
//! | Relu  | No-op on already non-negative input             |
//! | Floor | Posterise to 1-level (output is mostly 0)       |
//!
//! ```text
//! neo-gen-unary-onnx --op Sqrt --width 1920 --height 1080 --out sqrt.onnx
//! ```

use neo_infer_ort::generate::unary_model_f32_nchw;
use std::path::PathBuf;

fn main() {
    let mut width: usize = 1920;
    let mut height: usize = 1080;
    let mut op = String::from("Sqrt");
    let mut out = PathBuf::from("unary.onnx");

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
            "--op" => {
                op = args
                    .next()
                    .unwrap_or_else(|| usage_and_exit("missing value for --op"));
            }
            "--out" | "-o" => {
                out = PathBuf::from(
                    args.next()
                        .unwrap_or_else(|| usage_and_exit("missing value for --out")),
                );
            }
            "--list" => {
                println!(
                    "Supported --op values:\n  Neg, Abs, Sqrt, Exp, Log, Sin, Cos,\n  \
                     Tanh, Sigmoid, Relu, Floor, Ceil, Round, Reciprocal"
                );
                std::process::exit(0);
            }
            "--help" => usage_and_exit(""),
            other => usage_and_exit(&format!("unknown flag: {other}")),
        }
    }

    match unary_model_f32_nchw(&out, &op, 1, 3, height, width) {
        Ok(()) => {
            println!(
                "wrote {out:?}: unary `{op}` model, input [1, 3, {height}, {width}] f32",
                out = out,
                op = op,
                height = height,
                width = width
            );
        }
        Err(e) => {
            eprintln!("failed to generate unary model: {e}");
            std::process::exit(1);
        }
    }
}

fn usage_and_exit(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("error: {msg}");
    }
    eprintln!(
        "usage: neo-gen-unary-onnx \
         [--op NAME] [--width W] [--height H] [--out path.onnx]\n\
         --op defaults to Sqrt. Run with --list to see the 14 supported names.\n\
         defaults: --width 1920 --height 1080 --out unary.onnx"
    );
    std::process::exit(if msg.is_empty() { 0 } else { 2 });
}
