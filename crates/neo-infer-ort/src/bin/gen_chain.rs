//! `neo-gen-chain-onnx`
//!
//! Generate the smallest non-trivial *chained* ONNX model:
//! `output = mul * (sub - input)`. Two nodes (`Sub` then `Mul`),
//! two scalar initialisers. The wgpu fast path picks this up as
//! `[SubConst, MulConst]` and runs it as two compute dispatches with
//! ping-pong storage buffers — proving the inference engine handles
//! N>1 op chains end-to-end.
//!
//! ```text
//! ./target/release/neo-gen-chain-onnx.exe \
//!     --width 1920 --height 1080 \
//!     --sub 1.0 --mul 0.5 \
//!     --out chain.onnx
//! ```
//!
//! With `--sub 1.0 --mul 0.5` the model fuses photo-negative + half
//! brightness into a single ONNX file.

use neo_infer_ort::generate::sub_then_mul_model_f32_nchw;
use std::path::PathBuf;

fn main() {
    let mut width: usize = 1920;
    let mut height: usize = 1080;
    let mut sub: f32 = 1.0;
    let mut mul: f32 = 0.5;
    let mut out = PathBuf::from("chain.onnx");

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
            "--sub" => {
                sub = args
                    .next()
                    .unwrap_or_else(|| usage_and_exit("missing value for --sub"))
                    .parse()
                    .unwrap_or_else(|e| usage_and_exit(&format!("bad --sub: {e}")));
            }
            "--mul" => {
                mul = args
                    .next()
                    .unwrap_or_else(|| usage_and_exit("missing value for --mul"))
                    .parse()
                    .unwrap_or_else(|e| usage_and_exit(&format!("bad --mul: {e}")));
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

    match sub_then_mul_model_f32_nchw(&out, sub, mul, 1, 3, height, width) {
        Ok(()) => {
            println!(
                "wrote {out:?}: chained model output = {mul} * ({sub} - input), \
                 input [1, 3, {height}, {width}] f32",
                out = out,
                mul = mul,
                sub = sub,
                height = height,
                width = width
            );
        }
        Err(e) => {
            eprintln!("failed to generate chain model: {e}");
            std::process::exit(1);
        }
    }
}

fn usage_and_exit(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("error: {msg}");
    }
    eprintln!(
        "usage: neo-gen-chain-onnx \
         [--width W] [--height H] [--sub F] [--mul F] [--out path.onnx]\n\
         defaults: --width 1920 --height 1080 --sub 1.0 --mul 0.5 --out chain.onnx"
    );
    std::process::exit(if msg.is_empty() { 0 } else { 2 });
}
