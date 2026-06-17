# Neo

[![License](https://img.shields.io/badge/License-MIT%20%2F%20Apache%202.0-blue.svg)](LICENSE-MIT) ![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white) [![Release](https://img.shields.io/github/v/release/infinition/neo?style=flat)](https://github.com/infinition/neo/releases) [![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/infinition)

Zero-copy GPU video pipeline in Rust. Decode, filter, run neural nets, encode
and stream - frame data never leaves VRAM.

The project started as a ground-up rewrite of FFmpeg in safe Rust, then
pivoted when it became clear the CPU side is the bottleneck of modern
pipelines: NVMe → NVDEC → wgpu compute → ONNX Runtime → NVENC, with the host
only ever touching compressed bitstream. Full picture in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Three deliverables

| | What | Where |
|---|---|---|
| **`neo`** | CLI - every pipeline as a subcommand | [`apps/neo`](apps/neo) |
| **`neo-studio`** | GUI - live WGSL filter chain + hot-swappable ONNX models over a playing video | [`apps/neo-studio`](apps/neo-studio) |
| **`neo` (Python)** | NVDEC → PyTorch CUDA tensors, zero-copy | [`python/neo-py`](python/neo-py) |

## Quickstart

```powershell
# 1. Build everything (MSVC toolchain is pinned by rust-toolchain.toml)
cargo build --release

# 2. Grab a test clip + basic weights into the git-ignored assets/ dir
.\scripts\fetch-assets.ps1

# 3. Play
.\target\release\neo.exe --help
.\target\release\neo-studio.exe
```

For the ONNX-backed subcommands (`rife`, `filter-live`, `infer-bench`, Studio
models), set up ONNX Runtime once per terminal - see
[docs/SETUP.md](docs/SETUP.md):

```powershell
.\scripts\setup-ort.ps1 -OrtDir "C:\onnxruntime\lib"
```

For Python (bindings + examples in one go):

```powershell
.\scripts\dev.ps1
```

## The `neo` CLI

```text
Pipelines
  neo process         Image through the GPU filter pipeline (CPU reference path)
  neo video           Video transcode through GPU filters (FFmpeg in/out)
  neo lab             Windowed live WGSL shader chain with hot reload (+ optional ONNX model)
  neo studio          Same as neo-studio.exe (egui UI)

Streaming (zero-copy NVDEC→NVENC→TCP)
  neo send            Stream an H.264 file - pacing, looping
  neo recv            Receive + display with jitter buffer and latency metrics
  neo desktop         Desktop capture (DXGI) → NVENC → TCP (remote desktop)
  neo mosaic          N×N video wall, a different GPU filter per tile
  neo rife            RIFE frame interpolation streamer (2× FPS)
  neo filter-live     Hot-swappable ONNX filter (drop .onnx in a watched dir)

Diagnostics & benches
  neo probe           Probe a media file (container, streams, codecs)
  neo devices         List GPU devices
  neo cuda-info       CUDA / NVDEC / NVENC capability probe
  neo nvenc-test      NVENC self-test (encode a synthetic gradient)
  neo nvdec-test      NVDEC self-test (decode via nvcuvid.dll)
  neo interop-test    CUDA↔Vulkan external-memory self-test
  neo transcode-test  NVDEC → (cpu|wgpu|zerocopy) → NVENC end-to-end test
  neo graph           Print the pipeline graph for a config (dry run)
  neo bench           GPU filter throughput bench
  neo infer-bench     ONNX inference micro-bench + practical video runner

ONNX filter generators (make .onnx filters for lab / studio / filter-live)
  neo-gen-identity-onnx    pass-through        neo-gen-invert-onnx      negative
  neo-gen-brightness-onnx  gain (Mul)          neo-gen-clip-onnx        posterize (Clip)
  neo-gen-unary-onnx       14 unary ops        neo-gen-chain-onnx       chained Sub+Mul
  (cargo build --release -p neo-infer-ort builds them; they hit the wgpu fast path, no ORT needed)
```

Typical streaming session (two terminals):

```powershell
.\target\release\neo.exe recv 127.0.0.1:9000
.\target\release\neo.exe mosaic --input assets\videos\bunny.h264 --grid 3x3
```

## Python in 10 lines

```python
import torch, neo

src = neo.VideoSource("assets/videos/bunny.h264")
t = torch.empty((3, src.height, src.width), dtype=torch.float32, device="cuda")

while src.next_into(t.data_ptr()):                  # frame lands in VRAM only
    src.wait_stream(torch.cuda.current_stream().cuda_stream)
    ...                                              # CLIP, YOLO-World, RIFE...
```

Examples - open-vocabulary detection, CLIP semantic video search (Streamlit),
LocateAnything-3B grounding, matting, face swap:
[`python/examples/`](python/examples/).

## Repo layout

```
crates/        Library crates (GPU core + legacy cpu/ reference path)
apps/          neo (CLI) and neo-studio (GUI)
python/        neo-py bindings, examples/, tests/
shaders/       21 WGSL example filters (hot-reloadable)
benchmarks/    rust/ + python/ suites, results/ (curated numbers)
scripts/       dev.ps1, fetch-assets.ps1, setup-ort.ps1
docs/          ARCHITECTURE, SETUP, PYTHON, BENCHMARKS
vendor/        Vendored NVENC/NVDEC bindings
assets/        Git-ignored runtime assets (clips, models, weights)
```

## Requirements

Windows 10/11 x64 · NVIDIA driver 535+ (Vulkan 1.3, CUDA 12, NVENC/NVDEC) ·
Rust stable MSVC · details and troubleshooting in [docs/SETUP.md](docs/SETUP.md).

## Roadmap

- Linux support (`VK_KHR_external_memory_fd`) for NVENC/NVDEC + interop.
- AV1 NVENC encode path; HDR / 10-bit end-to-end (P010).
- Audio passthrough and MP4 muxing on the output side.
- Multi-GPU device selection; AMD (VCN/AMF) and Intel (QSV) provider trait.
- neo-studio: TensorRT/CUDA EP toggle, .wgsl file browser (today: built-in shader list), finish the in-app NVENC recorder.
- Headless zero-copy integration tests in CI.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual-licensed as above, without any additional
terms or conditions. Vendored crates under `vendor/` keep their own licenses.

## Star History

<a href="https://www.star-history.com/?repos=infinition%2Fneo&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=infinition/neo&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=infinition/neo&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=infinition/neo&type=date&legend=top-left" />
 </picture>
</a>
