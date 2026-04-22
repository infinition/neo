# Neo

Zero-copy GPU video pipeline in Rust. Decode, filter, run neural nets, encode
and stream - frame data never leaves VRAM.


## Origin

The project started as a ground-up rewrite of FFmpeg in safe Rust. It quickly
became clear that the CPU side of a modern pipeline is the bottleneck: decoded
frames round-trip through host RAM for every filter, shader, or model pass.
Neo pivoted to a zero-copy architecture where the entire pipeline lives on the
GPU - disk reads hit NVMe, land in NVDEC, stay on the device through every
filter/shader/ONNX pass, and exit through NVENC. The host never sees raw
pixels.

## Architecture

```
  NVMe                 GPU (VRAM)                           NVMe / Net
  ────                 ──────────                           ─────────
                ┌──────────────────────────┐
 H.264/H.265 ──▶│ NVDEC (nvcuvid)          │
  bitstream     │ → CUDA device ptr (NV12) │
                └──────────────┬───────────┘
                               │ DtoD memcpy (Y + UV planes)
                               ▼
                ┌──────────────────────────┐
                │ Interop Y / UV           │   CUDA ptr  ◀─┐
                │ VK_KHR_external_memory   │◀──────────────┤ aliased, same memory
                │ _win32                   │   wgpu buf  ◀─┘
                └──────────────┬───────────┘
                               │ wgpu compute shader
                               ▼
                ┌──────────────────────────┐
                │ NV12 → BGRA              │
                │  + user WGSL shader chain│   ← hot-reloadable .wgsl
                │  (programmable filters)  │
                └──────────────┬───────────┘
                               │ wgpu compute shader
                               ▼
                ┌──────────────────────────┐
                │ BGRA → NCHW f32 pack     │
                └──────────────┬───────────┘
                               │ zero-copy tensor bind
                               ▼
                ┌──────────────────────────┐
                │ ONNX Runtime             │
                │  TensorRT EP (FP16)      │   ← RIFE / ESRGAN / deepfake
                │  CUDA EP   (fallback)    │     any 1-in / 1-out NCHW model
                │  TensorRefMut::from_raw  │
                └──────────────┬───────────┘
                               │ wgpu compute shader
                               ▼
                ┌──────────────────────────┐
                │ NCHW → BGRA unpack       │
                └──────────────┬───────────┘
                               │ CUDA ptr alias
                               ▼
                ┌──────────────────────────┐
                │ NVENC                    │
                │ NvEncRegisterResource    │──▶ H.264 Annex-B
                │ (CUDADEVICEPTR)          │     ▼
                └──────────────────────────┘    disk / TCP socket

  Host RAM sees only compressed bitstream in / out. No raw pixel ever crosses
  the PCIe bus a second time.
```

## Repo layout

```
neo/
  crates/                 Library crates (workspace members)
    neo-core/             Frame/tensor primitives, error types
    neo-gpu/              wgpu context, compute, BGRA↔NCHW tensor bridge
    neo-hwaccel/          NVDEC, NVENC, CUDA↔Vulkan interop buffers
    neo-infer/            Inference abstractions
    neo-infer-ort/        ONNX Runtime backend (CUDA EP + TensorRT EP)
    neo-decode/           CPU-side demuxing / probing
    neo-encode/           CPU-side muxing
    neo-filters/          Fixed-function filter primitives
    neo-pipeline/         Graph builder + executor
    neo-io/               mmap, direct storage, network readers
    neo-cli/              neo CLI binary
    neo-lab/              Experimental bits
  demos/                  Standalone demo binaries
    neo-stream/           TCP frame protocol + receiver (neo-recv)
    neo-desktop/          Desktop capture → NVENC → TCP
    neo-mosaic/           Live WGSL shader mosaic over an H.264 source
    neo-rife-zc/          Zero-copy RIFE frame interpolation (2x FPS)
    neo-filter-live/      Hot-swappable ONNX filter (drop .onnx in a folder)
    neo-infer-bench/      Inference micro-benchmark
  shaders/                WGSL example shaders (21 filters)
  benchmarks/             run.sh, results.csv, RESULTS.md
  scripts/                benchmark.sh (ffmpeg vs neo comparison)
  vendor/
    nvidia-video-codec-sdk/   NVENC/NVDEC bindings (vendored, Apache-2.0)
    oidn-wgpu-interop/        OIDN+wgpu external-memory bridge
  assets/                 Runtime assets - git-ignored, see below
    videos/  models/  filters/  trt_cache/  bin/
```

## Requirements

- Windows 10/11 x64 (Linux path is planned; NVENC/NVDEC code is Windows-only today)
- NVIDIA GPU with driver 535+ (Vulkan 1.3, CUDA 12, NVENC, NVDEC)
- Rust stable 1.78+

For the ONNX demos, Neo uses ONNX Runtime 1.22.x GPU via dynamic loading.
The first run takes a few minutes the first time TensorRT builds its engine
cache; subsequent runs are instant.

### ONNX Runtime + CUDA DLLs

Download **ONNX Runtime 1.22.x GPU** for Windows x64 and place the DLLs in a
directory of your choice (example: `C:\onnxruntime\lib\`). That directory must
also contain:

- cuDNN 9 for CUDA 12 (`cudnn64_9.dll`, `cudnn_ops64_9.dll`,
  `cudnn_cnn64_9.dll`, `cudnn_adv64_9.dll`, `cudnn_graph64_9.dll`, engines,
  heuristic DLLs)
- CUDA Toolkit 12 runtime (`cudart64_12.dll`, `cublas64_12.dll`,
  `cublasLt64_12.dll`, `cufft64_11.dll`, `curand64_10.dll`)
- TensorRT 10.x (optional, for the TRT EP): `nvinfer_10.dll`,
  `nvinfer_builder_resource_10.dll`, `nvonnxparser_10.dll`

Set these environment variables once per terminal (PowerShell syntax):

```powershell
$env:ORT_DYLIB_PATH = "C:\onnxruntime\lib\onnxruntime.dll"
$env:PATH           = "C:\onnxruntime\lib;$env:PATH"
$env:NEO_TRT_CACHE_DIR = "C:\neo-trt-cache"   # TensorRT engine cache
# $env:NEO_DISABLE_TRT = "1"                  # to force CUDA EP only
```

## Build

All library crates build from the repo root:

```bash
cargo build --release
```

Each demo under `demos/` is its own Cargo workspace. Build them independently:

```bash
# Release builds
cd demos/neo-stream       && cargo build --release   # → neo-recv / neo-send
cd demos/neo-mosaic       && cargo build --release
cd demos/neo-desktop      && cargo build --release
cd demos/neo-rife-zc      && cargo build --release
cd demos/neo-filter-live  && cargo build --release
cd demos/neo-infer-bench  && cargo build --release
```

```bash
# Dev / debug build
cargo build
# Check only (fast)
cargo check
# Test
cargo test
# Lint
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
```

## Demos

All demos assume a receiver is listening on `127.0.0.1:9000`. The receiver is
the shared `neo-recv` binary from `demos/neo-stream`.

### Receiver (shared by every streaming demo)

```bash
cd demos/neo-stream/target/release
./neo-recv.exe 127.0.0.1:9000
```

### neo-mosaic - live WGSL shader mosaic

Decodes an H.264 clip, applies a user-selected WGSL shader per tile, encodes
and streams. Shaders in `shaders/examples/` hot-reload on save.

```bash
cd demos/neo-mosaic/target/release
./neo-mosaic.exe \
  --input   ../../../assets/videos/in-4k.h264 \
  --shader  ../../../shaders/examples/08_edge_sobel.wgsl \
  --addr    0.0.0.0:9000 \
  --fps     30
```

### neo-desktop - desktop capture → NVENC → TCP

Captures the desktop with D3D11, shares the texture with CUDA, encodes with
NVENC, sends over TCP.

```bash
cd demos/neo-desktop/target/release
./neo-desktop-send.exe --addr 0.0.0.0:9000 --fps 60
```

### neo-rife-zc - zero-copy RIFE frame interpolation (2x FPS)

Interpolates one intermediate frame per decoded pair. Output FPS is `--fps * 2`.
Model must be a 3-input RIFE export: `(img0, img1, timestep)`.

```bash
cd demos/neo-rife-zc/target/release
./neo-rife-zc.exe \
  --model ../../../assets/models/rife_v4.onnx \
  --input ../../../assets/videos/in-4k.h264 \
  --addr  0.0.0.0:9000 \
  --fps   30
```

### neo-filter-live - hot-swappable ONNX filter

Watches a directory; drop any `.onnx` file with shape `[1,3,H,W]` → `[1,3,H,W]`
and the pipeline swaps to it live. Remove the file to return to pass-through.

```bash
cd demos/neo-filter-live/target/release
./neo-filter-live.exe \
  --input        ../../../assets/videos/in-4k.h264 \
  --filters-dir  ../../../assets/filters \
  --addr         0.0.0.0:9000 \
  --fps          30
```

```bash
# Hot-swap, from anywhere:
cp some_model.onnx assets/filters/
rm assets/filters/some_model.onnx   # back to pass-through
```

### neo-infer-bench - inference benchmark

Measures CUDA↔Vulkan interop + ORT inference throughput without encoder
overhead.

```bash
cd demos/neo-infer-bench/target/release
./neo-infer-bench.exe \
  --model ../../../assets/models/alexnet_Opset17.onnx \
  --runs  100
```

### neo CLI

The legacy CPU/GPU CLI (kept for comparison). Build from the root workspace:

```bash
cargo build --release -p neo-cli
./target/release/neo.exe process \
  --input  assets/videos/in-4k.h264 \
  --output out.h264 \
  --filter grayscale
```

## Benchmarks

```bash
# Full benchmark sweep (requires ffmpeg + neo on PATH or at the
# default paths in the script)
./benchmarks/run.sh

# FFmpeg vs Neo one-shot comparison (4K image filter passes)
./scripts/benchmark.sh
```

Results are written to `benchmarks/results.csv` and summarised in
`benchmarks/RESULTS.md`.

## Assets

`assets/` is git-ignored. Populate it with your own clips and models:

```
assets/
  videos/    *.h264 / *.mp4 test clips
  models/    *.onnx (RIFE, ESRGAN, classification, etc.)
  filters/   drop *.onnx here for neo-filter-live hot-swap
  trt_cache/ generated by the TensorRT EP on first run
  bin/       local copies of ffmpeg.exe and similar helpers
```

Any public clip works. `Tears of Steel` (CC-BY, Blender Foundation) and
`Big Buck Bunny` are commonly used.

## Environment variables

| Variable              | Purpose                                           |
|-----------------------|---------------------------------------------------|
| `ORT_DYLIB_PATH`      | Full path to `onnxruntime.dll` (required for ORT) |
| `NEO_TRT_CACHE_DIR`   | Directory for TensorRT engine + timing caches     |
| `NEO_DISABLE_TRT`     | Set to `1` to force the CUDA EP only              |
| `RUST_LOG`            | `info`, `debug`, etc. Drives `tracing_subscriber` |

## Releases

A GitHub Actions workflow at `.github/workflows/release.yml` builds every
demo and the `neo` CLI on `windows-latest` and attaches the `.exe`
bundle to a GitHub Release whenever a `v*` tag is pushed:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow can also be triggered manually from the Actions tab
(`workflow_dispatch`), in which case the binaries are uploaded as
build artifacts instead of a release.

## Roadmap

- Linux support for NVENC/NVDEC + CUDA/Vulkan external memory (`VK_KHR_external_memory_fd`).
- AV1 NVENC encode path (currently H.264 only in the demos).
- HDR / 10-bit end-to-end (P010 NV12 variant on the NVDEC side, P010 on NVENC).
- Audio track passthrough and muxing (MP4 container on the output side).
- Multi-GPU device selection and round-robin scheduling.
- ONNX models with multiple outputs and non-image tensors in the filter-live demo.
- Replace remaining CPU fallbacks in `neo-filters` with wgpu compute equivalents.
- AMD (VCN/AMF) and Intel (QSV) hardware paths via a thin provider trait over `neo-hwaccel`.
- Package the demos as a single distributable with bundled runtime DLLs.
- Integration tests that run the full zero-copy pipeline headless in CI.

## License

Dual-licensed under either of

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in the work by you, as defined in
the Apache-2.0 license, shall be dual-licensed as above, without any
additional terms or conditions.

Vendored crates under `vendor/` keep their own licenses (MIT, see
`vendor/*/LICENSE`).

## Star History

<a href="https://www.star-history.com/?repos=infinition%2Fneo&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=infinition/neo&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=infinition/neo&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=infinition/neo&type=date&legend=top-left" />
 </picture>
</a>
