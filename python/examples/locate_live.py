"""LocateAnything-3B on a live video stream — neo zero-copy vs CPU baseline.

Modes
-----
--mode neo       neo zero-copy path: NVDEC decode -> wgpu NV12->RGB CHW f32
                 -> GPU preprocess (resize/normalize/patchify) -> model.
                 Pixels never touch host RAM (display is the only exception,
                 and only with --display).
--mode baseline  Classic path: OpenCV CPU decode -> PIL -> HF processor (CPU
                 resize/normalize/patchify) -> PCIe upload -> model.

Sources: a .h264/.264 Annex-B file, a .mp4 (auto-extracted with ffmpeg for
neo), or "webcam" / "webcam:1" (USB capture is host-side by nature; the neo
mode still does all preprocessing on GPU).

Examples
--------
  python locate_live.py --source assets/videos/bunny.h264 --mode neo --prompt "the rabbit" --bench 20
  python locate_live.py --source assets/videos/bunny.mp4 --mode baseline --prompt "the rabbit" --bench 20
  python locate_live.py --source webcam --mode neo --prompt "a person" --display
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import gpu_preprocess

MODEL_ID = "nvidia/LocateAnything-3B"
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


# --------------------------------------------------------------------------- model

def load_model():
    from transformers import AutoModel, AutoProcessor

    print(f"[init] loading {MODEL_ID} (bf16) ...", flush=True)
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
        .to("cuda")
        .eval()
    )
    print(f"[init] model ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return model, processor


def build_inputs(processor, prompt: str, width: int, height: int):
    """One-time warmup: tokenized inputs for a frame of (width, height).

    The video resolution is constant, so the patch grid — and therefore the
    expanded <image pad> token sequence — is identical for every frame. Only
    pixel_values changes per frame.
    """
    dummy = Image.new("RGB", (width, height))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": dummy},
            {"type": "text", "text": f"Locate all the instances that matches the following description: {prompt}."},
        ],
    }]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt")
    return inputs


def decode_response(out, tokenizer, input_len: int) -> str:
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], str):
        return out[0]
    if torch.is_tensor(out):
        seq = out[0] if out.dim() == 2 else out
        return tokenizer.decode(seq[input_len:], skip_special_tokens=False)
    return str(out)


def run_inference(model, processor, ids_inputs, pixel_values, image_grid_hws, max_new_tokens):
    input_ids = ids_inputs["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(
            pixel_values=pixel_values.to(torch.bfloat16),
            input_ids=input_ids,
            attention_mask=ids_inputs["attention_mask"].to("cuda"),
            image_grid_hws=image_grid_hws,
            tokenizer=processor.tokenizer,
            max_new_tokens=max_new_tokens,
            generation_mode="hybrid",
            use_cache=True,
        )
    return decode_response(out, processor.tokenizer, input_ids.shape[1])


def parse_boxes(text: str):
    """-> list of (x1, y1, x2, y2) normalized to [0, 1000].

    Dedupes repeated boxes and drops degenerate slivers (<0.1% of the image):
    the model sometimes loops on a tiny box at the end of long generations.
    """
    seen, out = set(), []
    for m in BOX_RE.findall(text):
        box = tuple(int(v) for v in m)
        x1, y1, x2, y2 = box
        if box in seen or (x2 - x1) * (y2 - y1) < 1000:
            continue
        seen.add(box)
        out.append(box)
    return out


# --------------------------------------------------------------------------- sources

def ensure_annexb(path: Path) -> Path:
    if path.suffix.lower() in (".h264", ".264", ".annexb"):
        return path
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
        return self.frame if ok else None

    def display_frame(self):
        # GPU->CPU download for visualization ONLY (not part of the pipeline).
        return self.bgra[:, :, :3].cpu().numpy()


class CvSource:
    """OpenCV capture: file (CPU decode) or webcam."""

    def __init__(self, spec, loop=True):
        import cv2

        self.cv2 = cv2
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
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
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


def bench(args, model, processor):
    import psutil

    timer = StageTimer()
    proc = psutil.Process()
    host_bytes_per_frame = 0
    n = args.bench

    if args.mode == "neo":
        src = NeoSource(Path(args.source))
        w, h = src.w, src.h
    else:
        spec = parse_webcam(args.source) if args.source.startswith("webcam") else Path(args.source)
        src = CvSource(spec)
        w, h = src.w, src.h

    print(f"[init] source {w}x{h}, warmup inputs ...")
    ids_inputs = build_inputs(processor, args.prompt, w, h)
    ref_grid = ids_inputs["image_grid_hws"]
    print(f"[init] grid_hws={ref_grid.tolist() if hasattr(ref_grid, 'tolist') else ref_grid}")

    # warmup inference (CUDA graphs / kernels / first-token compile)
    print("[init] warmup inference ...")
    txt = run_inference(model, processor, ids_inputs,
                        ids_inputs["pixel_values"].to("cuda"),
                        to_grid_tensor(ref_grid), args.max_new_tokens)
    print(f"[init] warmup response: {txt[:200]!r}")

    proc.cpu_percent()  # reset counter
    psutil.cpu_percent()
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
            pv, grid = gpu_preprocess.preprocess_gpu(frame)
            torch.cuda.synchronize()
            timer.add("preprocess (GPU)", time.perf_counter() - t0)
            grid_t = to_grid_tensor(np.array([list(grid)]))
        else:
            t0 = time.perf_counter()
            bgr = src.next_bgr()
            timer.add("decode (CPU)", time.perf_counter() - t0)
            if bgr is None:
                break

            t0 = time.perf_counter()
            pil = Image.fromarray(bgr[:, :, ::-1])
            feat = processor.image_processor.preprocess(pil, return_tensors="pt")
            pv_cpu = feat["pixel_values"]
            timer.add("preprocess (CPU)", time.perf_counter() - t0)

            t0 = time.perf_counter()
            pv = pv_cpu.to("cuda")
            torch.cuda.synchronize()
            timer.add("upload PCIe (HtoD)", time.perf_counter() - t0)
            host_bytes_per_frame = pv_cpu.numel() * pv_cpu.element_size()
            grid_t = to_grid_tensor(feat["image_grid_hws"])

        t0 = time.perf_counter()
        txt = run_inference(model, processor, ids_inputs, pv, grid_t, args.max_new_tokens)
        torch.cuda.synchronize()
        timer.add("inference (GPU)", time.perf_counter() - t0)
        last_boxes = parse_boxes(txt)
        print(f"  frame {i + 1}/{n}: {len(last_boxes)} box(es)", flush=True)

    wall = time.perf_counter() - t_start
    cpu_proc = proc.cpu_percent() / psutil.cpu_count()
    cpu_sys = psutil.cpu_percent()

    print("\n========== RESULTS ==========")
    print(f"mode            : {args.mode}")
    print(f"source          : {args.source} ({w}x{h})")
    print(f"frames          : {n}   wall {wall:.1f}s   {n / wall:.2f} FPS end-to-end")
    print(f"CPU avg         : process {cpu_proc:.1f}%  system {cpu_sys:.1f}%")
    print(f"host pixel bytes: {host_bytes_per_frame / 1e6:.2f} MB/frame "
          f"({'PCIe upload per frame' if host_bytes_per_frame else 'ZERO — pixels never leave VRAM'})")
    print("stage timings (mean / median ms):")
    for k, (mean, med) in timer.summary().items():
        print(f"  {k:24s} {mean:8.2f} / {med:.2f}")
    print(f"last boxes      : {last_boxes}")

    out = {
        "mode": args.mode, "source": str(args.source), "size": [w, h],
        "frames": n, "wall_s": wall, "fps": n / wall,
        "cpu_process_pct": cpu_proc, "cpu_system_pct": cpu_sys,
        "host_bytes_per_frame": host_bytes_per_frame,
        "stages_ms": timer.summary(),
    }
    res = Path(__file__).parent / f"bench_{args.mode}.json"
    res.write_text(json.dumps(out, indent=2))
    print(f"[saved] {res}")


def to_grid_tensor(grid):
    if torch.is_tensor(grid):
        return grid.to("cuda")
    return torch.as_tensor(np.asarray(grid), device="cuda")


def parse_webcam(spec: str) -> int:
    return int(spec.split(":", 1)[1]) if ":" in spec else 0


# --------------------------------------------------------------------------- live display

def display(args, model, processor):
    import cv2

    is_cam = args.source.startswith("webcam")
    if args.mode == "neo" and not is_cam:
        src = NeoSource(Path(args.source))
        w, h = src.w, src.h
    else:
        spec = parse_webcam(args.source) if is_cam else Path(args.source)
        src = CvSource(spec)
        w, h = src.w, src.h

    ids_inputs = build_inputs(processor, args.prompt, w, h)

    state = {"boxes": [], "infer_ms": 0.0, "pv": None, "grid": None, "stop": False}
    lock = threading.Lock()

    def worker():
        stream = torch.cuda.Stream()
        while not state["stop"]:
            with lock:
                pv, grid = state["pv"], state["grid"]
                state["pv"] = None
            if pv is None:
                time.sleep(0.002)
                continue
            t0 = time.perf_counter()
            stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(stream):
                txt = run_inference(model, processor, ids_inputs, pv, grid, args.max_new_tokens)
            with lock:
                state["boxes"] = parse_boxes(txt)
                state["infer_ms"] = (time.perf_counter() - t0) * 1000.0

    threading.Thread(target=worker, daemon=True).start()
    title = f"LocateAnything [{args.mode}] — {args.prompt!r} (q to quit)"
    t_last, fps = time.perf_counter(), 0.0

    while True:
        t0 = time.perf_counter()
        if isinstance(src, NeoSource):
            frame_t = src.next(with_display=True)
            if frame_t is None:
                break
            pv, grid = gpu_preprocess.preprocess_gpu(frame_t)
            grid_t = to_grid_tensor(np.array([list(grid)]))
            disp = np.ascontiguousarray(src.display_frame())  # viz-only download
        else:
            bgr = src.next_bgr()
            if bgr is None:
                break
            if args.mode == "neo":
                # webcam: capture is host-side by nature; preprocess on GPU
                t = torch.from_numpy(bgr[:, :, ::-1].copy()).to("cuda")
                t = t.permute(2, 0, 1).float().div_(255.0)
                pv, grid = gpu_preprocess.preprocess_gpu(t)
                grid_t = to_grid_tensor(np.array([list(grid)]))
            else:
                pil = Image.fromarray(bgr[:, :, ::-1])
                feat = processor.image_processor.preprocess(pil, return_tensors="pt")
                pv = feat["pixel_values"].to("cuda")
                grid_t = to_grid_tensor(feat["image_grid_hws"])
            disp = bgr

        with lock:
            state["pv"], state["grid"] = pv, grid_t
            boxes, infer_ms = state["boxes"], state["infer_ms"]

        for (x1, y1, x2, y2) in boxes:
            p1 = (int(x1 / 1000 * w), int(y1 / 1000 * h))
            p2 = (int(x2 / 1000 * w), int(y2 / 1000 * h))
            cv2.rectangle(disp, p1, p2, (0, 255, 0), 2)
        dt = time.perf_counter() - t_last
        t_last = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
        cv2.putText(disp, f"{args.mode} | video {fps:.1f} fps | infer {infer_ms:.0f} ms | {len(boxes)} obj",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(title, disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # pace file playback roughly to real time
        if not is_cam:
            time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t0)))

    state["stop"] = True
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help=".h264/.mp4 path or webcam[:index]")
    ap.add_argument("--mode", choices=["neo", "baseline"], default="neo")
    ap.add_argument("--prompt", default="a person", help="what to locate")
    ap.add_argument("--bench", type=int, default=0, help="headless benchmark over N frames")
    ap.add_argument("--display", action="store_true", help="live window with boxes")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    if not args.bench and not args.display:
        args.bench = 10

    model, processor = load_model()
    if args.bench:
        bench(args, model, processor)
    if args.display:
        display(args, model, processor)


if __name__ == "__main__":
    main()
