# Architecture

Neo is a zero-copy GPU video pipeline: decode, filter, run neural nets, encode
and stream — frame data never leaves VRAM. Disk reads hit NVMe, land in NVDEC,
stay on the device through every filter/shader/ONNX pass, and exit through
NVENC. The host never sees raw pixels.

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

## Crate map

```
crates/
  neo-core        Frame/tensor primitives, error types. No deps on other crates.
  neo-gpu         wgpu context, compute dispatch, BGRA↔NCHW tensor bridge.
  neo-hwaccel     NVDEC, NVENC, CUDA↔Vulkan interop buffers, zerocopy stream.
  neo-infer-ort   ONNX Runtime backend (CUDA EP + TensorRT EP), dynamic loading.
  neo-lab         Windowed lab: live WGSL shader chain + egui Studio UI.
  cpu/            Legacy CPU reference path — kept as the benchmark baseline.
    neo-decode    CPU-side demuxing / probing.
    neo-encode    CPU-side muxing.
    neo-filters   Fixed-function filter primitives.
    neo-pipeline  Graph builder + executor (drives `neo process` / `neo video`).
    neo-infer     CPU inference abstractions.
    neo-io        mmap, direct storage, network readers.

apps/
  neo             The CLI. One binary, all pipelines as subcommands.
  neo-studio      The GUI (egui): import a clip, pick WGSL filters live,
                  load/remove ONNX models (NVENC recorder is WIP).

python/
  neo-py          pyo3 cdylib → Python module `neo` (maturin).
                  NVDEC → CUDA tensor handoff via `data_ptr()`.

vendor/
  nvidia-video-codec-sdk   Vendored NVENC/NVDEC bindings (Apache-2.0).
  oidn-wgpu-interop        OIDN + wgpu external-memory bridge.
```

## Dependency rules

- `neo-core` depends on nothing internal; everything depends on it.
- The GPU path is `neo-gpu` + `neo-hwaccel` (+ `neo-infer-ort` for models).
- The `cpu/` crates form the legacy reference path; only `apps/neo` links both
  worlds (the `process`, `video`, `bench` subcommands use the CPU path, all
  streaming subcommands use the zero-copy GPU path).
- `neo-hwaccel::zerocopy_stream` is the shared NVDEC→interop streaming source
  used by the CLI streaming subcommands and the Python bindings.

## Platform

Windows 10/11 x64 only today (NVDEC/NVENC + `VK_KHR_external_memory_win32`).
The Linux path (`VK_KHR_external_memory_fd`) is on the roadmap.
