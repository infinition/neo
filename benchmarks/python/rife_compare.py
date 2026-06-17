"""RIFE frame interpolation (2x FPS) — neo zero-copy vs CPU baseline.

Single-script, single-process, no TCP — same architecture as yoloworld_live:

--mode neo       NVDEC decode (neo.pyd) -> torch CUDA frames ->
                 RIFE via ONNX Runtime CUDA EP with IOBinding on raw device
                 pointers. Pixels never touch host RAM.
--mode baseline  OpenCV CPU decode -> numpy preprocess -> ort.run() with CPU
                 tensors (PCIe up AND down every frame) -> numpy output.

Model: vs-mlrt "v2" RIFE export — single input (1, 7, H, W):
channels 0-2 = img0, 3-5 = img1, 6 = timestep map (0.5 = midpoint).
Padding is handled inside the graph. Output = (1, 3, H, W).

Examples
--------
  python rife_compare.py --source assets\\videos\\demo.h264 --mode neo --bench 200
  python rife_compare.py --source assets\\videos\\demo.mp4 --mode baseline --bench 200
  python rife_compare.py --source assets\\videos\\demo.h264 --mode neo --display
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# onnxruntime-gpu does not bundle cuDNN on Windows; torch does. Make ORT's
# CUDA EP find cudnn64_*.dll inside torch\lib before the session is created.
os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
import onnxruntime as ort  # noqa: E402


def open_session(model_path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        model_path, so,
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    )
    inp = sess.get_inputs()[0]
    if len(sess.get_inputs()) != 1 or inp.shape[1] != 7:
        sys.exit(f"expected a vs-mlrt v2 RIFE export (1 input, 7 channels), "
                 f"got {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"[init] {Path(model_path).name} input={inp.shape} "
          f"provider={sess.get_providers()[0]}")
    return sess, inp.name, sess.get_outputs()[0].name


# --------------------------------------------------------------------------- neo

def run_neo(args):
    import neo

    src = neo.VideoSource(str(args.source))
    w, h = src.width, src.height
    print(f"[init] source {w}x{h}, zero-copy NVDEC")
    sess, in_name, out_name = open_session(args.model)

    frame = torch.empty((3, h, w), dtype=torch.float32, device="cuda")
    inp = torch.empty((1, 7, h, w), dtype=torch.float32, device="cuda")
    inp[0, 6].fill_(0.5)  # timestep map: midpoint
    out = torch.empty((1, 3, h, w), dtype=torch.float32, device="cuda")

    io = sess.io_binding()
    io.bind_input(in_name, "cuda", 0, np.float32, tuple(inp.shape), inp.data_ptr())
    io.bind_output(out_name, "cuda", 0, np.float32, tuple(out.shape), out.data_ptr())

    def decode_next():
        """img1 -> img0 slot, decode a fresh frame into the img1 slot (VRAM)."""
        inp[0, 0:3].copy_(inp[0, 3:6])
        if not src.next_into(frame.data_ptr()):
            return False
        src.wait_stream(torch.cuda.current_stream().cuda_stream)
        inp[0, 3:6].copy_(frame)
        return True

    def infer():
        torch.cuda.synchronize()  # tensor writes must land before ORT reads
        sess.run_with_iobinding(io)
        return out  # stays in VRAM

    # prime img1 so the first decode_next() gives (frame0, frame1)
    assert src.next_into(frame.data_ptr()), "empty source"
    src.wait_stream(torch.cuda.current_stream().cuda_stream)
    inp[0, 3:6].copy_(frame)

    rife_loop(args, w, h, decode_next, infer, gpu_output=True)


# ---------------------------------------------------------------------- baseline

def run_baseline(args):
    import cv2

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        sys.exit(f"cannot open {args.source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[init] source {w}x{h}, OpenCV CPU decode")
    sess, in_name, out_name = open_session(args.model)

    inp = np.empty((1, 7, h, w), dtype=np.float32)
    inp[0, 6].fill(0.5)

    def read_frame():
        ok, bgr = cap.read()
        if not ok:  # loop like the neo source
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = cap.read()
            if not ok:
                return None
        return bgr[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

    def decode_next():
        inp[0, 0:3] = inp[0, 3:6]
        f = read_frame()
        if f is None:
            return False
        inp[0, 3:6] = f
        return True

    def infer():
        # ort.run with numpy: implicit HtoD upload + DtoH download every call
        return sess.run([out_name], {in_name: inp})[0]

    f = read_frame()
    assert f is not None, "empty source"
    inp[0, 3:6] = f

    rife_loop(args, w, h, decode_next, infer, gpu_output=False)


# -------------------------------------------------------------------------- loop

def rife_loop(args, w, h, decode_next, infer, gpu_output):
    import psutil
    proc = psutil.Process()

    print("[init] warmup ...", flush=True)
    assert decode_next(), "empty source"
    for _ in range(3):
        infer()

    disp = None
    if args.display:
        import cv2
        disp = cv2

    t_dec, t_inf = [], []
    proc.cpu_percent()
    psutil.cpu_percent()
    n = args.bench if not args.display else 10 ** 9
    produced = 0
    t_start = time.perf_counter()

    for i in range(n):
        t0 = time.perf_counter()
        if not decode_next():
            break
        t_dec.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        result = infer()
        if gpu_output:
            torch.cuda.synchronize()
        t_inf.append(time.perf_counter() - t0)
        produced += 2  # interpolated mid-frame + the new source frame

        if disp is not None:
            mid = result[0].cpu().numpy() if gpu_output else result[0]  # viz only
            bgr = (np.clip(mid.transpose(1, 2, 0)[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
            bgr = np.ascontiguousarray(bgr)
            disp.putText(bgr, f"RIFE x2 [{args.mode}] interp #{i}", (15, 30),
                         disp.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            disp.imshow("rife_compare (q to quit)", bgr)
            if disp.waitKey(1) & 0xFF == ord("q"):
                break

    wall = time.perf_counter() - t_start
    cpu_p = proc.cpu_percent() / psutil.cpu_count()
    cpu_s = psutil.cpu_percent()

    fps_out = produced / wall
    print("\n========== RIFE RESULTS ==========")
    print(f"mode           : {args.mode}")
    print(f"source         : {w}x{h}  ({args.source})")
    print(f"output frames  : {produced} in {wall:.2f}s")
    print(f"OUTPUT FPS     : {fps_out:.1f}  (soutient une source a {fps_out / 2:.1f} fps)")
    print(f"realtime 30->60 : {'YES' if fps_out >= 60 else 'NO'}")
    print(f"realtime 60->120: {'YES' if fps_out >= 120 else 'NO'}")
    print(f"decode+prep ms : mean {1000 * np.mean(t_dec):.2f} / median {1000 * np.median(t_dec):.2f}")
    print(f"inference ms   : mean {1000 * np.mean(t_inf):.2f} / median {1000 * np.median(t_inf):.2f}")
    print(f"CPU            : process {cpu_p:.1f}%  system {cpu_s:.1f}%")

    res = {
        "mode": args.mode, "size": [w, h], "output_fps": fps_out,
        "decode_ms": float(np.mean(t_dec)), "infer_ms": float(np.mean(t_inf)),
        "cpu_process_pct": cpu_p, "cpu_system_pct": cpu_s,
    }
    p = Path(__file__).parent / f"bench_rife_{args.mode}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"[saved] {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="assets/models/rife_v2/rife_v4.14.onnx")
    ap.add_argument("--mode", choices=["neo", "baseline"], default="neo")
    ap.add_argument("--bench", type=int, default=200, help="source frames to process")
    ap.add_argument("--display", action="store_true", help="show interpolated frames")
    args = ap.parse_args()

    if args.mode == "neo":
        run_neo(args)
    else:
        run_baseline(args)


if __name__ == "__main__":
    main()
