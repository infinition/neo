# Benchmarks

Two suites, results checked into `benchmarks/results/`.

## Rust — transcode backends (`benchmarks/rust/`)

Compares the Neo backends (`zerocopy`, `wgpu`, `cpu`) against FFmpeg
(`h264_nvenc` and `libx264`) on fixed 1080p/2160p clips:

```bash
./benchmarks/rust/run.sh          # writes benchmarks/results/results.csv
```

A one-shot 4K image-filter comparison also exists:

```bash
./benchmarks/rust/ffmpeg_compare.sh
```

Summary tables: [`benchmarks/results/RESULTS.md`](../benchmarks/results/RESULTS.md).

## Python — zero-copy vs CPU baseline (`benchmarks/python/`)

Each script runs the same model in `--mode neo` (NVDEC→CUDA tensor, GPU
preprocessing) and `--mode baseline` (OpenCV decode + CPU preprocessing) and
reports FPS, CPU usage and host pixel traffic:

```powershell
.\benchmarks\python\run_compare_locate.ps1    # LocateAnything-3B grounding
.\benchmarks\python\run_compare_yolo.ps1      # YOLO-World detection
python benchmarks\python\bench_full.py        # full sweep (yolo, rife, matting, filters)
```

Raw JSON output lands in the repo root (`bench_*.json`, git-ignored); curated
numbers live in
[`benchmarks/results/PYTHON_BENCH_RESULTS.md`](../benchmarks/results/PYTHON_BENCH_RESULTS.md).

## In-binary micro-benches

```powershell
neo bench --resolution 4k --filter grayscale --iterations 100   # GPU filter throughput
neo infer-bench bench --model assets\models\some_model.onnx     # ORT interop bench
```
