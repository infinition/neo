# Phase B.2 — zero-copy inference bridge (planned)

This document is the implementation plan for the next step of Neo Lab's
inference path. **Phase B.1 is landed** (see below). Phase B.2 is not
yet written — this file is here so the design choices don't get lost
when we come back to it.

## Where we stand after Phase B.1 ✅

```
NVDEC → StreamSource → NV12 frame on CPU
   → wgpu upload (Y + UV)
   → wgpu compute (shader chain)
   → final BGRA in wgpu storage buffer
   → [ModelNode, Phase B.1]
       → copy_buffer_to_buffer     ┐
       → device.poll(Wait)          │  CPU bounce
       → map_async + memcpy         │
       → unpack BGRA u32 → NCHW f32 ┘
       → tract::OnnxModel::infer
       → repack NCHW f32 → BGRA u32 ┐
       → queue.write_buffer          ┘  CPU bounce again
   → blit
```

Measured cost of the Phase B.1 bridge for a `[1, 3, 1080, 1920]` f32
identity/invert model on an RTX 4070 Ti:

| Path component | Time per frame |
|---|---|
| GPU→staging copy + map + memcpy | ~2–4 ms |
| BGRA u32 → NCHW f32 (6.2M floats) | ~8 ms |
| tract inference (invert / Sub) | 50–125 ms |
| NCHW f32 → BGRA u32 | ~8 ms |
| queue.write_buffer upload | ~3 ms |
| **Total overhead vs "no model"** | **~75–150 ms** |

At 1080p with a trivial model the full Lab pipeline drops from ~30 fps
to ~15 fps. Phase B.2 must cut that overhead to <1 ms/frame so
non-trivial models can run at 30 fps.

## Phase B.2 goal

Replace the CPU bounce with a path that lets the inference engine read
**the same device memory** the wgpu chain just wrote into, and write
its output back into **another device memory** that blit / NVENC can
read directly.

The hard constraint: no change to Neo Lab's user-facing API. `ModelNode`
stays, `--model` stays, `.onnx` files stay. Only the guts of `process()`
change.

## Three viable execution backends

| Backend | Pros | Cons |
|---|---|---|
| **ort + CUDA EP** (ONNX Runtime with CUDA execution provider) | Wide model coverage, mature, has `IoBinding` for zero-copy on CUDA pointers | ort 2.0.0-rc12 has no prebuilts for x86_64-pc-windows-gnu; dynamic-load feature is buggy (VitisAI API mismatch on current API version). Needs MSVC toolchain or upstream fix. |
| **candle** (HuggingFace, pure Rust) | Native Rust, CUDA backend, no native dep hunting, small | Narrower model coverage than ort/tract; conv kernels still maturing; no direct DLPack story |
| **TensorRT** (direct FFI via `cudarc::nvinfer`) | Fastest on NVIDIA hardware, native CUDA device pointer API | Heavy SDK install, breaks Neo's "no SDK install" promise, Linux-first tooling |

**Recommended path**: ort + CUDA EP, using `IoBinding`. This gets us to
zero-copy fastest *if* we accept the toolchain requirement. The MSVC
toolchain is a one-time install on Windows; the user pain is bounded.

The Neo identity is "no NVIDIA SDK install required" — ort+CUDA-EP is
NOT the NVIDIA SDK, it bundles prebuilt libraries that speak CUDA to
the driver. This is a defensible line.

## Concrete steps

### Step 1 — Extract a backend trait in `neo-infer-ort`

```rust
pub trait InferenceBackend {
    fn input_shape(&self) -> &[usize];

    /// Run on flat f32 (Phase B.1 legacy path). Stays, used by unit tests.
    fn infer(&self, input: &[f32]) -> Result<Vec<f32>>;

    /// Phase B.2 fast path: the backend reads from and writes to
    /// device memory directly. Implementations that can't do this
    /// should return `None` so `ModelNode` falls back to `infer()`.
    fn infer_device(
        &self,
        in_dptr: u64, in_pitch_bytes: usize,
        out_dptr: u64, out_pitch_bytes: usize,
    ) -> Option<Result<()>> {
        let _ = (in_dptr, in_pitch_bytes, out_dptr, out_pitch_bytes);
        None
    }
}
```

`TractBackend` already exists (rename the current `OnnxModel` to
`TractBackend: InferenceBackend`). Add a feature flag
`backend-tract` (default on).

### Step 2 — Add `ort` behind a feature flag

```toml
[features]
default = ["backend-tract"]
backend-tract = ["dep:tract-onnx"]
backend-ort = ["dep:ort"]
```

Use `ort = { version = "2", features = ["cuda", "load-dynamic"] }`.
The `load-dynamic` dodges the MinGW binary-unavailable problem by
loading the DLL at runtime — but we also need to document that the
user must have onnxruntime.dll on PATH or in `ORT_DYLIB_PATH`.

### Step 3 — Implement `OrtBackend::infer_device`

```rust
impl InferenceBackend for OrtBackend {
    fn infer_device(
        &self,
        in_dptr: u64, in_pitch_bytes: usize,
        out_dptr: u64, out_pitch_bytes: usize,
    ) -> Option<Result<()>> {
        let mut io = ort::IoBinding::new(&self.session)?;
        unsafe {
            io.bind_input_tensor_from_device(
                self.input_name.as_str(),
                OrtMemoryInfo::cuda(),
                in_dptr as *mut c_void,
                in_pitch_bytes,
                // shape
            )?;
            io.bind_output_tensor_to_device(
                self.output_name.as_str(),
                OrtMemoryInfo::cuda(),
                out_dptr as *mut c_void,
                out_pitch_bytes,
            )?;
        }
        self.session.run_with_binding(&io)?;
        Some(Ok(()))
    }
    // ...
}
```

### Step 4 — Wire `ModelNode` to use `infer_device` when available

```rust
// neo-lab/src/model_node.rs
pub fn process(&mut self, ...) -> NeoResult<()> {
    if let Some(result) = self.backend.infer_device(
        self.in_interop.cu_dptr(),
        self.in_interop.pitch_bytes(),
        self.out_interop.cu_dptr(),
        self.out_interop.pitch_bytes(),
    ) {
        return result.map_err(...);
    }
    // Fallback to CPU bounce (Phase B.1 path, unchanged)
    self.process_cpu_bounce(...)
}
```

`self.in_interop` and `self.out_interop` are `InteropBuffer`s (from
`neo-hwaccel::interop`, Stage 6a). We already know how to share VRAM
between CUDA and Vulkan — we just need to do it for the model's
input/output buffers too.

### Step 5 — Conversion at the edges

The chain emits BGRA u32, but the model wants NCHW f32 in [0, 1]. This
can't be done for free on CUDA pointers — we need a small CUDA kernel
(or a dedicated wgpu compute pass, same memory) that does
`u32 → f32 plane + normalize` on the GPU.

Two ways to do it:

**(a) Extra wgpu compute pass** — write a "unpack_bgra_to_nchw_f32"
shader in the chain, have it write into an `InteropBuffer` that CUDA
can then read. This keeps everything in wgpu land and doesn't require a
CUDA toolchain.

**(b) CUDA kernel** — slightly faster, but requires NVRTC or a .ptx
fixture. Against the "no SDK install" goal.

Go with (a). Same story for the output side: a "pack_nchw_f32_to_bgra"
wgpu pass that reads the interop buffer the model wrote into.

### Step 6 — Measure

Target numbers at 1080p with an invert model (same baseline as B.1):

- Phase B.1: 15 fps → 66 ms per frame with model
- Phase B.2 target: **>25 fps → <40 ms per frame** — that's ~30% of a
  frame budget spent in the bridge + inference. Good enough to run
  non-trivial models at real-time 1080p.
- Stretch target: **>50 fps** — the invert model is a single Sub, it
  should cost <1 ms on the GPU once we stop round-tripping to host.

## What doesn't need to change

- `neo-infer-ort`'s current API (`OnnxModel`, `generate::*`) is
  preserved. tract stays as the default backend for maximum
  portability.
- Neo Lab's CLI: `--model path.onnx` means the same thing.
- The generator binaries (`neo-gen-identity-onnx`,
  `neo-gen-invert-onnx`) don't move.

## Risks / open questions

1. **ort DLL hunting on Windows**: if we can't ship a prebuilt
   `onnxruntime.dll`, the user experience regresses. Solution: bundle
   the DLL next to `neo-ffmpeg.exe` in release artifacts.
2. **IoBinding only works on tensors with known shape at bind time**.
   We already require concrete shapes on load — matches.
3. **f32 vs f16**: many modern models expect f16. ort supports it, our
   wgpu pack/unpack passes will need an f16 variant. Shelved until
   Phase B.3 (first model that actually needs f16).
4. **wgpu ↔ CUDA synchronisation**: the wgpu pass writing the model's
   input must complete before CUDA reads it, and vice versa for the
   output. We already use `device.poll(Wait)` for sync elsewhere; the
   same pattern works here but adds latency. A Vulkan timeline semaphore
   imported into CUDA via `cuImportExternalSemaphore` is the proper
   answer — see Stage 6b interop for the handle-export pattern.

## Estimated effort

- Step 1 (trait extraction): ~half a day
- Step 2 (ort feature flag): ~half a day
- Step 3 (IoBinding wrapper): 1-2 days, bulk of the unsafe FFI
- Step 4 (ModelNode wire-up): half a day
- Step 5 (wgpu pack/unpack shaders): 1 day
- Step 6 (measurement + tuning): 1 day

Total: **~5-6 days of focused work**. Phase B.2 is a real sprint, not a
session. That's why Phase B.1 shipped first.
