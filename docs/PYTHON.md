# Python bindings & examples

The `neo` Python module exposes the zero-copy NVDEC source to PyTorch: frames
are decoded on the GPU and written straight into CUDA tensors. The host only
ever touches the compressed bitstream.

## Install

```powershell
# one-shot (creates .venv, installs deps, builds the bindings):
.\scripts\dev.ps1

# or manually, inside your venv:
pip install maturin -r python\requirements.txt
maturin develop --release -m python\neo-py\Cargo.toml
```

## Minimal example

```python
import torch, neo

src = neo.VideoSource("assets/videos/bunny.h264")    # Annex-B H.264
t = torch.empty((3, src.height, src.width), dtype=torch.float32, device="cuda")

while src.next_into(t.data_ptr()):                   # RGB CHW in [0,1], VRAM only
    src.wait_stream(torch.cuda.current_stream().cuda_stream)
    # ... run any model on t ...
```

Full API reference: [`python/neo-py/README.md`](../python/neo-py/README.md).

## Examples

See [`python/examples/README.md`](../python/examples/README.md) for the
catalog (YOLO-World, CLIP semantic search, LocateAnything-3B grounding, RIFE,
matting, face swap). **All examples are run from the repo root** and read
their inputs from `assets/`.

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
python python\tests\test_neo_module.py        # decode smoke test
python python\tests\check_preprocess.py       # GPU preprocessing parity vs HF processor
python python\tests\check_parity.py           # YOLO-World neo vs baseline parity
```

## Notes & limits

- `VideoSource` accepts raw Annex-B H.264 only; the examples convert `.mp4`
  automatically with `ffmpeg -c copy -bsf:v h264_mp4toannexb`.
- `next_into` is non-blocking — order GPU work with `wait_stream(...)` or use
  `synchronize()` when in doubt.
- Webcam input delivers raw pixels to RAM by construction; the zero-copy path
  only applies to compressed file/stream sources.
