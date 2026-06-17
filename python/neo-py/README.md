# neo (Python bindings)

Zero-copy NVDEC video source for PyTorch. Decodes H.264 on the GPU and hands
frames to PyTorch as CUDA tensors — pixels never touch host RAM; the host only
ever sees the compressed bitstream.

```
            ┌──────────────────────── one Python process ───────────────────────┐
  .h264     │  neo.pyd (Rust)                             PyTorch                │
  bitstream │  NVDEC → DtoD → interop Y/UV                                       │
 ───────────▶  → wgpu NV12→BGRA → BGRA→RGB CHW f32  ──DtoD──▶ CUDA tensor        │
            │                                             → any model (CLIP,    │
            │                                               YOLO-World, VLM...) │
            └────────────────────────────────────────────────────────────────────┘
```

## Install (development)

```powershell
# from the repo root, inside your venv
pip install maturin
maturin develop --release -m python\neo-py\Cargo.toml
```

## Usage

```python
import torch, neo

src = neo.VideoSource("assets/videos/bunny.h264")   # Annex-B H.264
t = torch.empty((3, src.height, src.width), dtype=torch.float32, device="cuda")

while src.next_into(t.data_ptr()):                  # RGB CHW in [0,1], VRAM only
    src.wait_stream(torch.cuda.current_stream().cuda_stream)
    ...                                              # run any model on `t`
```

API:

- `VideoSource(path)` — open an Annex-B H.264 bitstream (`.h264` / `.264`).
  Loops back to the start at EOF.
- `.width` / `.height` / `.frames` — stream properties.
- `.next_into(dst_ptr)` — decode the next frame into a caller-provided CUDA
  buffer (`tensor.data_ptr()`). Non-blocking: records a CUDA event.
- `.next_into_with_bgra(dst_ptr, bgra_ptr)` — also writes the BGRA frame for
  on-GPU visualization (uint8 tensor of shape `(H, W, 4)`).
- `.wait_stream(stream_handle)` — GPU-side ordering against a torch stream.
- `.synchronize()` — block the CPU until the last frame is fully decoded.

`.mp4` sources must be converted to Annex-B first (the examples do it
automatically via ffmpeg `-c copy`).

See [`python/examples/`](../examples/) for full demos (YOLO-World, CLIP
semantic search, RIFE, matting, LocateAnything-3B grounding).
