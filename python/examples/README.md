# Python examples

All examples consume the `neo` Python module (zero-copy NVDEC → PyTorch CUDA
tensors, see [`python/neo-py/`](../neo-py/)) and compare against a CPU
baseline (`--mode baseline`, OpenCV decode + CPU preprocessing).

**Run everything from the repo root.** Test clips, models and weights live in
the git-ignored `assets/` directory (`scripts/fetch-assets.ps1` downloads the
basics).

| Example | What it does |
|---|---|
| `yoloworld_live.py` | Open-vocabulary detection (YOLO-World) on a zero-copy stream |
| `yoloworld_multistream.py` | Same, N parallel streams |
| `locate_live.py` | LocateAnything-3B visual grounding, GPU preprocessing |
| `matting_live.py` | Robust Video Matting (RVM) background replacement |
| `rife_live.py` | RIFE frame interpolation (2× FPS) |
| `face_anon_live.py` | Face detection + anonymization |
| `face_swap_live.py` / `face_swap_enhanced.py` | InsightFace swap (+ RealESRGAN enhance) |
| `neo_filter_live.py` | Generic ONNX filter runner on the zero-copy stream |
| `gpu_preprocess.py` / `gpu_preprocess_yolo.py` | GPU-side resize/normalize/patchify helpers |
| `perception-engine/` | CLIP semantic video search: index in VRAM, Streamlit dashboard |

Typical invocations:

```powershell
# Open-vocabulary detection, live window
python python\examples\yoloworld_live.py --source assets\videos\demo.h264 --mode neo --classes "person, car" --display

# LocateAnything-3B grounding
python python\examples\locate_live.py --source assets\videos\bunny.h264 --mode neo --prompt "the rabbit" --display

# CLIP semantic search dashboard (index a video first with main.py ingest)
python python\examples\perception-engine\main.py ingest --video assets\videos\demo.h264 --index-out assets\indexes\demo.neo
streamlit run python\examples\perception-engine\app.py
```

Benchmarks comparing neo vs baseline live in [`benchmarks/python/`](../../benchmarks/python/).
