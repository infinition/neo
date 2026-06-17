"""Full benchmark suite: neo zero-copy vs CPU baseline, run sequentially.

Runs each scenario in its own process (clean CUDA/GPU state every time):
  1. parity check (proves both modes produce identical detections)
  2. mono-stream neo        (100 frames, synchronized inference)
  3. mono-stream baseline   (100 frames)
  4. multi-stream 3x neo    (40 batches = 120 frames)
  5. multi-stream 3x baseline
  6. multi-stream 6x neo    (40 batches = 240 frames)
  7. multi-stream 6x baseline

Then aggregates the JSON outputs into a final comparison table (also written
to BENCH_RESULTS.md, ready to paste into the README).

Usage: python bench_full.py [--frames 100] [--quick]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
H264 = str(HERE / ".." / ".." / "assets" / "videos" / "demo.h264")
MP4 = str(HERE / ".." / ".." / "assets" / "videos" / "demo.mp4")
CLASSES = "person, car"


def run(name, cmd):
    print(f"\n=== [{name}] {' '.join(c for c in cmd if c != PY)}", flush=True)
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-12:])
    print(tail, flush=True)
    if r.returncode not in (0, 255):  # multistream exits 255 via PS quirk
        print(f"[warn] {name} exited with {r.returncode}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100, help="mono-stream frames")
    ap.add_argument("--batches", type=int, default=40, help="multi-stream batch iterations")
    ap.add_argument("--quick", action="store_true", help="frames=30, batches=15")
    ap.add_argument("--report-only", action="store_true",
                    help="skip runs, aggregate existing bench_*.json files")
    args = ap.parse_args()
    if args.quick:
        args.frames, args.batches = 30, 15

    live = str(HERE / "yoloworld_live.py")
    multi = str(HERE / "yoloworld_multistream.py")

    if args.report_only:
        report()
        return

    run("parity", [PY, str(HERE / "check_parity.py"), H264, CLASSES, "400"])
    run("mono neo", [PY, live, "--source", H264, "--mode", "neo",
                     "--classes", CLASSES, "--bench", str(args.frames)])
    run("mono baseline", [PY, live, "--source", MP4, "--mode", "baseline",
                          "--classes", CLASSES, "--bench", str(args.frames)])
    for n in (3, 6):
        run(f"multi{n} neo", [PY, multi, "--source", H264, "--mode", "neo",
                              "--streams", str(n), "--classes", CLASSES,
                              "--frames", str(args.batches)])
        run(f"multi{n} baseline", [PY, multi, "--source", MP4, "--mode", "baseline",
                                   "--streams", str(n), "--classes", CLASSES,
                                   "--frames", str(args.batches)])

    report()


def report():
    def j(p):
        f = HERE / p
        return json.loads(f.read_text()) if f.exists() else None

    mono_n, mono_b = j("bench_yolo_neo.json"), j("bench_yolo_baseline.json")
    rows = []

    def stage(d, key):
        v = d["stages_ms"].get(key)
        return f"{v[0]:.2f} ms" if v else "—"

    if mono_n and mono_b:
        rows += [
            ("**Mono-flux** end-to-end", f"{mono_b['fps']:.1f} FPS", f"{mono_n['fps']:.1f} FPS",
             f"×{mono_n['fps'] / mono_b['fps']:.2f}"),
            ("CPU (process)", f"{mono_b['cpu_process_pct']:.1f}%", f"{mono_n['cpu_process_pct']:.1f}%",
             f"÷{mono_b['cpu_process_pct'] / max(mono_n['cpu_process_pct'], 0.1):.1f}"),
            ("Pixels sur PCIe / frame", f"{mono_b['host_bytes_per_frame'] / 1e6:.2f} MB", "0 MB", "zero-copy"),
            ("decode / frame", stage(mono_b, "decode (CPU)"), stage(mono_n, "decode+to_vram (GPU)"), ""),
            ("préprocessing / frame", stage(mono_b, "preprocess (CPU)"), stage(mono_n, "preprocess (GPU)"), ""),
            ("upload PCIe / frame", stage(mono_b, "upload PCIe (HtoD)"), "0 ms", ""),
            ("inférence / frame", stage(mono_b, "inference (GPU)"), stage(mono_n, "inference (GPU)"), ""),
        ]
    for n in (3, 6):
        mn, mb = j(f"bench_multi_neo_{n}.json"), j(f"bench_multi_baseline_{n}.json")
        if mn and mb:
            rows += [
                (f"**Multi-flux {n}x** cumulé", f"{mb['fps_total']:.1f} FPS", f"{mn['fps_total']:.1f} FPS",
                 f"×{mn['fps_total'] / mb['fps_total']:.2f}"),
                (f"CPU (process, {n}x)", f"{mb['cpu_process_pct']:.1f}%", f"{mn['cpu_process_pct']:.1f}%",
                 f"÷{mb['cpu_process_pct'] / max(mn['cpu_process_pct'], 0.1):.1f}"),
            ]

    lines = ["| Métrique | Baseline (CPU) | Neo (zero-copy) | Gain |",
             "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    table = "\n".join(lines)
    (HERE / "BENCH_RESULTS.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + "=" * 70)
    print(table.encode(sys.stdout.encoding or "utf-8", errors="replace")
               .decode(sys.stdout.encoding or "utf-8"))
    print(f"\n[saved] {HERE / 'BENCH_RESULTS.md'}")


if __name__ == "__main__":
    main()
