"""YOLO-World on a live video stream — neo zero-copy vs CPU baseline.

Modes
-----
--mode neo       neo zero-copy path: NVDEC decode -> wgpu NV12->RGB CHW f32
                 -> GPU preprocess (resize to 640x640) -> model.
                 Pixels never touch host RAM (except for visual display).
--mode baseline  Classic path: OpenCV CPU decode -> CPU resize/normalize
                 -> PCIe upload -> model.

Examples
--------
  python yoloworld_live.py --source assets/videos/bunny.h264 --mode neo --classes "rabbit, butterfly, flower" --display
  python yoloworld_live.py --source webcam --mode neo --classes "person, dog, phone" --display
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld

import gpu_preprocess_yolo


# --------------------------------------------------------------------------- model

def load_model(model_name: str, classes: list):
    print(f"[init] loading YOLO-World model ({model_name}) ...", flush=True)
    t0 = time.perf_counter()
    model = YOLOWorld(model_name)
    model.to("cuda")
    print(f"[init] setting vocabulary classes: {classes} ...", flush=True)
    model.set_classes(classes)
    print(f"[init] model ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return model


# --------------------------------------------------------------------------- sources

def ensure_annexb(path: Path) -> Path:
    if path.suffix.lower() in (".h264", ".264", ".annexb"):
        return path
    import tempfile
    import subprocess
    out = Path(tempfile.gettempdir()) / (path.stem + ".h264")
    if not out.exists():
        print(f"[init] extracting Annex-B bitstream -> {out}")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path),
             "-c:v", "copy", "-bsf:v", "h264_mp4toannexb", "-an", str(out)],
            check=True,
        )
    return out


class NeoSource:
    """neo zero-copy: NVDEC -> VRAM RGB CHW f32. Loops at EOF."""

    def __init__(self, path: Path):
        import neo

        self.src = neo.VideoSource(str(ensure_annexb(path)))
        self.w, self.h = self.src.width, self.src.height
        self.frame = torch.empty((3, self.h, self.w), dtype=torch.float32, device="cuda")
        self.bgra = torch.empty((self.h, self.w, 4), dtype=torch.uint8, device="cuda")

    def next(self, with_display=False):
        if with_display:
            ok = self.src.next_into_with_bgra(self.frame.data_ptr(), self.bgra.data_ptr())
        else:
            ok = self.src.next_into(self.frame.data_ptr())
        if not ok:
            return None
        # GPU-side ordering: torch's current stream waits for the Rust-side
        # decode/convert/DtoD (CUDA event) without blocking the CPU.
        self.src.wait_stream(torch.cuda.current_stream().cuda_stream)
        return self.frame

    def display_frame(self):
        # GPU->CPU download for visualization ONLY (not part of the pipeline).
        return self.bgra[:, :, :3].cpu().numpy()


class CvSource:
    """OpenCV capture: file (CPU decode) or webcam."""

    def __init__(self, spec, loop=True):
        self.loop = loop
        self.is_cam = isinstance(spec, int)
        self.cap = cv2.VideoCapture(spec, cv2.CAP_DSHOW) if self.is_cam else cv2.VideoCapture(str(spec))
        if not self.cap.isOpened():
            sys.exit(f"cannot open source: {spec}")
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def next_bgr(self):
        ok, frame = self.cap.read()
        if not ok and self.loop and not self.is_cam:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None


# --------------------------------------------------------------------------- bench

class StageTimer:
    def __init__(self):
        self.acc = {}

    def add(self, name, dt):
        self.acc.setdefault(name, []).append(dt * 1000.0)

    def summary(self):
        return {k: (float(np.mean(v)), float(np.median(v))) for k, v in self.acc.items()}


def bench(args, model):
    import psutil

    timer = StageTimer()
    proc = psutil.Process()
    host_bytes_per_frame = 0
    n = args.bench

    if args.mode == "neo":
        src = NeoSource(Path(args.source))
        w, h = src.w, src.h
    else:
        spec = (int(args.source.split(":")[1]) if ":" in args.source else 0) if args.source.startswith("webcam") else Path(args.source)
        src = CvSource(spec)
        w, h = src.w, src.h

    print(f"[init] source {w}x{h}, warming up ...")
    # Warmup
    dummy = torch.zeros((1, 3, 640, 640), device="cuda")
    for _ in range(5):
        model.predict(dummy, verbose=False)

    proc.cpu_percent()
    t_start = time.perf_counter()
    last_boxes = []

    for i in range(n):
        if args.mode == "neo":
            t0 = time.perf_counter()
            frame = src.next()
            timer.add("decode+to_vram (GPU)", time.perf_counter() - t0)
            if frame is None:
                break

            t0 = time.perf_counter()
            pv, _meta = gpu_preprocess_yolo.preprocess_gpu(frame, letterbox=not args.no_letterbox)
            torch.cuda.synchronize()
            timer.add("preprocess (GPU)", time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            bgr = src.next_bgr()
            timer.add("decode (CPU)", time.perf_counter() - t0)
            if bgr is None:
                break

            t0 = time.perf_counter()
            pv_cpu, _meta = gpu_preprocess_yolo.preprocess_cpu(bgr, letterbox=not args.no_letterbox)
            timer.add("preprocess (CPU)", time.perf_counter() - t0)

            t0 = time.perf_counter()
            pv = pv_cpu.to("cuda")
            torch.cuda.synchronize()
            timer.add("upload PCIe (HtoD)", time.perf_counter() - t0)
            host_bytes_per_frame = pv_cpu.numel() * pv_cpu.element_size()

        t0 = time.perf_counter()
        results = model.predict(pv, verbose=False, conf=args.conf)
        torch.cuda.synchronize()
        timer.add("inference (GPU)", time.perf_counter() - t0)
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            last_boxes = boxes.xyxy.cpu().numpy().tolist()
        else:
            last_boxes = []
            
        print(f"  frame {i + 1}/{n}: {len(last_boxes)} box(es)", flush=True)

    wall = time.perf_counter() - t_start
    cpu_proc = proc.cpu_percent() / psutil.cpu_count()
    cpu_sys = psutil.cpu_percent()

    print("\n========== RESULTS ==========")
    print(f"mode            : {args.mode}")
    print(f"source          : {args.source} ({w}x{h})")
    print(f"frames          : {n}   wall {wall:.3f}s   {n / wall:.2f} FPS end-to-end")
    print(f"CPU avg         : process {cpu_proc:.1f}%  system {cpu_sys:.1f}%")
    print(f"host pixel bytes: {host_bytes_per_frame / 1e6:.2f} MB/frame "
          f"({'PCIe upload per frame' if host_bytes_per_frame else 'ZERO — pixels never leave VRAM'})")
    print("stage timings (mean / median ms):")
    for k, (mean, med) in timer.summary().items():
        print(f"  {k:24s} {mean:8.2f} / {med:.2f}")

    out = {
        "mode": args.mode, "source": str(args.source), "size": [w, h],
        "frames": n, "wall_s": wall, "fps": n / wall,
        "cpu_process_pct": cpu_proc, "cpu_system_pct": cpu_sys,
        "host_bytes_per_frame": host_bytes_per_frame,
        "stages_ms": timer.summary(),
    }
    res = Path(__file__).parent / f"bench_yolo_{args.mode}.json"
    res.write_text(json.dumps(out, indent=2))
    print(f"[saved] {res}")


# --------------------------------------------------------------------------- live display

def display(args, model, classes):
    is_cam = args.source.startswith("webcam")
    if args.mode == "neo" and not is_cam:
        src = NeoSource(Path(args.source))
        w, h = src.w, src.h
    else:
        spec = (int(args.source.split(":")[1]) if ":" in args.source else 0) if is_cam else Path(args.source)
        src = CvSource(spec)
        w, h = src.w, src.h

    vlm = VlmOnDemand(args, w, h) if args.vlm else None

    state = {"boxes": [], "cls": [], "conf": [], "infer_ms": 0.0, "pv": None, "stop": False}
    lock = threading.Lock()

    def worker():
        stream = torch.cuda.Stream()
        while not state["stop"]:
            # While the VLM is generating (~2-3s of GPU kernels), pause YOLO:
            # running both saturates the GPU and the display loop starves.
            if vlm and vlm.is_busy():
                time.sleep(0.01)
                continue
            with lock:
                pv = state["pv"]
                state["pv"] = None
            if pv is None:
                time.sleep(0.002)
                continue
            
            t0 = time.perf_counter()
            stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(stream):
                # Predict on the separate stream
                results = model.predict(pv, verbose=False, conf=args.conf)
            
            boxes_obj = results[0].boxes
            if len(boxes_obj) > 0:
                bx = boxes_obj.xyxy.cpu().numpy()
                cl = boxes_obj.cls.cpu().numpy().astype(int)
                cf = boxes_obj.conf.cpu().numpy()
            else:
                bx, cl, cf = [], [], []

            with lock:
                state["boxes"] = bx
                state["cls"] = cl
                state["conf"] = cf
                state["infer_ms"] = (time.perf_counter() - t0) * 1000.0

    threading.Thread(target=worker, daemon=True).start()
    title = f"YOLO-World [{args.mode}] — {classes} (q to quit)"
    t_last, fps = time.perf_counter(), 0.0

    while True:
        t0 = time.perf_counter()
        frame_for_vlm = None
        if isinstance(src, NeoSource):
            frame_t = src.next(with_display=True)
            if frame_t is None:
                break
            pv, meta = gpu_preprocess_yolo.preprocess_gpu(frame_t, letterbox=not args.no_letterbox)
            frame_for_vlm = frame_t
            disp = np.ascontiguousarray(src.display_frame())
        else:
            bgr = src.next_bgr()
            if bgr is None:
                break
            if args.mode == "neo":
                # preprocess CPU frame on GPU
                t = torch.from_numpy(bgr[:, :, ::-1].copy()).to("cuda")
                t = t.permute(2, 0, 1).float().div_(255.0)
                pv, meta = gpu_preprocess_yolo.preprocess_gpu(t, letterbox=not args.no_letterbox)
                frame_for_vlm = t
            else:
                pv_cpu, meta = gpu_preprocess_yolo.preprocess_cpu(bgr, letterbox=not args.no_letterbox)
                pv = pv_cpu.to("cuda")
                frame_for_vlm = bgr
            disp = bgr

        with lock:
            state["pv"] = pv
            boxes = list(state["boxes"])
            cls_ids = list(state["cls"])
            confs = list(state["conf"])
            infer_ms = state["infer_ms"]

        mapped = gpu_preprocess_yolo.unmap_boxes(boxes, meta, w, h)
        for i, (p1x, p1y, p2x, p2y) in enumerate(mapped):
            label = f"{classes[cls_ids[i]]} {confs[i]:.2f}"
            cv2.rectangle(disp, (p1x, p1y), (p2x, p2y), (0, 255, 0), 2)
            cv2.putText(disp, label, (p1x, p1y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if vlm:
            vlm.draw(disp, w, h)

        dt = time.perf_counter() - t_last
        t_last = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        hud = f"{args.mode.upper()} | video {fps:.1f} fps | infer {infer_ms:.0f} ms | {len(boxes)} obj"
        if vlm:
            hud += f" | [l] VLM {vlm.status()}"
        cv2.putText(disp, hud, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(title, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if vlm and key == ord("l"):
            vlm.trigger(frame_for_vlm)
        if vlm and key == ord("c"):
            vlm.clear()
        if not is_cam and not args.no_pace:
            time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t0)))

    state["stop"] = True
    if vlm:
        vlm.stop()
    cv2.destroyAllWindows()


# --------------------------------------------------------------- VLM on demand

class VlmOnDemand:
    """LocateAnything-3B grounding on the current frame, triggered by key 'l'.

    YOLO-World tracks every frame; the VLM answers a rich text query
    (--vlm-prompt) in ~1-3s on the frame captured at trigger time. In neo
    mode the captured frame is already in VRAM and the VLM preprocessing
    (resize/normalize/patchify) runs entirely on GPU.
    """

    def __init__(self, args, w, h):
        import sys as _sys
        # gpu_preprocess / locate_live live next to this script
        _sys.path.insert(0, str(Path(__file__).parent))
        import gpu_preprocess as locate_pre
        import locate_live

        self._pre = locate_pre
        self._ll = locate_live
        self.prompt = args.vlm_prompt
        self.model, self.processor = locate_live.load_model()
        self.ids_inputs = locate_live.build_inputs(self.processor, self.prompt, w, h)
        self.boxes, self.busy, self.last_ms = [], False, 0.0
        self._lock = threading.Lock()
        # Dedicated CUDA stream: without it, the VLM's ~2-3s of kernels are
        # enqueued on the default stream and every display-loop GPU op (frame
        # download, preprocess) stalls behind them — the window freezes.
        self._stream = torch.cuda.Stream()

    def is_busy(self):
        with self._lock:
            return self.busy

    def trigger(self, frame):
        with self._lock:
            if self.busy:
                return
            self.busy = True
        if isinstance(frame, torch.Tensor):
            snap = frame.clone()  # frame buffer is reused by the decoder
        else:  # baseline: BGR numpy -> GPU tensor
            snap = torch.from_numpy(frame[:, :, ::-1].copy()).to("cuda")
            snap = snap.permute(2, 0, 1).float().div_(255.0)
        threading.Thread(target=self._run, args=(snap,), daemon=True).start()

    def _run(self, snap):
        t0 = time.perf_counter()
        try:
            self._stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(self._stream):
                pv, grid = self._pre.preprocess_gpu(snap)
                grid_t = torch.as_tensor(np.array([list(grid)]), device="cuda")
                txt = self._ll.run_inference(self.model, self.processor, self.ids_inputs,
                                             pv, grid_t, 256)
            with self._lock:
                self.boxes = self._ll.parse_boxes(txt)
                self.last_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            print(f"[vlm] error: {e}", flush=True)
        finally:
            with self._lock:
                self.busy = False

    def draw(self, disp, w, h):
        with self._lock:
            boxes = list(self.boxes)
        for (x1, y1, x2, y2) in boxes:
            p1 = (int(x1 / 1000 * w), int(y1 / 1000 * h))
            p2 = (int(x2 / 1000 * w), int(y2 / 1000 * h))
            cv2.rectangle(disp, p1, p2, (0, 0, 255), 2)
            cv2.putText(disp, self.prompt[:40], (p1[0], p1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def status(self):
        with self._lock:
            if self.busy:
                return "..."
            return f"{len(self.boxes)} box ({self.last_ms:.0f} ms)" if self.last_ms else "ready"

    def clear(self):
        with self._lock:
            self.boxes = []

    def stop(self):
        pass


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="YOLO-World Live Object Grounding")
    ap.add_argument("--source", required=True, help="Path to video or webcam[:index]")
    ap.add_argument("--mode", choices=["neo", "baseline"], default="neo")
    ap.add_argument("--classes", default="person, dog", help="comma-separated classes to detect")
    ap.add_argument("--model", default="assets/models/yolov8s-worldv2.pt", help="yolov8[n/s/m/l]-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--bench", type=int, default=0, help="headless benchmark over N frames")
    ap.add_argument("--display", action="store_true", help="live window with bounding boxes")
    ap.add_argument("--no-pace", action="store_true", help="run display loop as fast as possible without 30 FPS cap")
    ap.add_argument("--no-letterbox", action="store_true",
                    help="aspect-squash resize instead of letterbox (legacy behavior)")
    ap.add_argument("--vlm", action="store_true",
                    help="load LocateAnything-3B for on-demand grounding (key 'l', clear 'c')")
    ap.add_argument("--vlm-prompt", default="the main subject of the scene",
                    help="text query for the on-demand VLM grounding")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    model = load_model(args.model, classes)

    if not args.bench and not args.display:
        args.bench = 10

    if args.bench:
        bench(args, model)
    if args.display:
        display(args, model, classes)


if __name__ == "__main__":
    main()
