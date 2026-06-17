# Setup

## Requirements

- Windows 10/11 x64
- NVIDIA GPU with driver 535+ (Vulkan 1.3, CUDA 12, NVENC, NVDEC)
- Rust stable — the MSVC toolchain is pinned by `rust-toolchain.toml`
  (requires Visual Studio Build Tools / "Desktop development with C++")
- ffmpeg + ffprobe on PATH (asset preparation, benchmarks, clip extraction)
- Python 3.10+ (only for the Python bindings and examples)

## Build

```powershell
cargo build --release          # everything: neo.exe, neo-studio.exe, libs
```

Or run `.\scripts\dev.ps1` to build the workspace **and** set up the Python
venv + bindings in one go.

## Assets

`assets/` is git-ignored. `.\scripts\fetch-assets.ps1` downloads a test clip
(Big Buck Bunny) and YOLO-World weights. Layout:

```
assets/
  videos/    *.h264 (Annex-B) / *.mp4 test clips
  models/    *.onnx / *.pt (RIFE, ESRGAN, RVM, inswapper, YOLO-World...)
  weights/   raw weights (e.g. CLIP ViT-B-32.pt)
  filters/   drop *.onnx here for `neo filter-live` hot-swap
  indexes/   *.neo indexes built by the perception-engine example
  images/    sample images for the face-swap / matting examples
```

HuggingFace-hosted models (CLIP, LocateAnything-3B) download automatically on
first run. ONNX exports (RIFE v4.14, RealESRGAN, RVM, inswapper_128) come from
their respective project releases — put them under `assets/models/`.

## ONNX Runtime + CUDA DLLs (for the ONNX subcommands and neo-studio models)

Neo loads ONNX Runtime 1.22.x GPU dynamically. Download it for Windows x64 and
place the DLLs in a directory of your choice (example: `C:\onnxruntime\lib\`).
That directory must also contain:

- cuDNN 9 for CUDA 12 (`cudnn64_9.dll`, `cudnn_ops64_9.dll`,
  `cudnn_cnn64_9.dll`, `cudnn_adv64_9.dll`, `cudnn_graph64_9.dll`, engines,
  heuristic DLLs)
- CUDA Toolkit 12 runtime (`cudart64_12.dll`, `cublas64_12.dll`,
  `cublasLt64_12.dll`, `cufft64_11.dll`, `curand64_10.dll`)
- TensorRT 10.x (optional, for the TRT EP): `nvinfer_10.dll`,
  `nvinfer_builder_resource_10.dll`, `nvonnxparser_10.dll`

Then, once per terminal:

```powershell
.\scripts\setup-ort.ps1 -OrtDir "C:\onnxruntime\lib"
```

The first TensorRT run takes a few minutes while the engine cache builds;
subsequent runs are instant.

## Environment variables

| Variable              | Purpose                                           |
|-----------------------|---------------------------------------------------|
| `ORT_DYLIB_PATH`      | Full path to `onnxruntime.dll` (required for ORT) |
| `NEO_TRT_CACHE_DIR`   | Directory for TensorRT engine + timing caches     |
| `NEO_DISABLE_TRT`     | Set to `1` to force the CUDA EP only              |
| `RUST_LOG`            | `info`, `debug`, etc. Drives `tracing_subscriber` |
| `NEO_STUDIO_SHADERS`  | neo-studio: scratch dir for the active shader     |
| `NEO_STUDIO_FPS`      | neo-studio: playback fps override                 |
| `NEO_STUDIO_NO_VSYNC` | neo-studio: set to disable v-sync                 |

## Troubleshooting

- **`dlltool.exe: program not found` during build** — you are on the GNU
  toolchain. The repo pins MSVC via `rust-toolchain.toml`; make sure rustup
  honors it (`rustup show` inside the repo should say `*-msvc`).
- **`onnxruntime.dll not found`** — run `scripts/setup-ort.ps1` in this
  terminal; the env vars do not persist across sessions.
- **NVENC/NVDEC init fails** — check `neo cuda-info`; driver must be 535+.
- **wgpu interop init fails** — the Vulkan driver must expose
  `VK_KHR_external_memory_win32` (any recent NVIDIA driver does).
