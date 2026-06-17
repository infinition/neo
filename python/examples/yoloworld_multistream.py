import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLOWorld

import gpu_preprocess_yolo
import neo


# --------------------------------------------------------------------------- helpers

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
    def __init__(self, path: Path, with_display=False):
        self.src = neo.VideoSource(str(ensure_annexb(path)))
        self.w, self.h = self.src.width, self.src.height
        self.frame = torch.empty((3, self.h, self.w), dtype=torch.float32, device="cuda")
        self.with_display = with_display
        if with_display:
            self.bgra = torch.empty((self.h, self.w, 4), dtype=torch.uint8, device="cuda")

    def next(self):
        if self.with_display:
            ok = self.src.next_into_with_bgra(self.frame.data_ptr(), self.bgra.data_ptr())
        else:
            ok = self.src.next_into(self.frame.data_ptr())
        if not ok:
            return None
        # GPU-side ordering: torch waits for the Rust-side decode/DtoD event.
        self.src.wait_stream(torch.cuda.current_stream().cuda_stream)
        return self.frame

    def display_frame(self):
        return self.bgra[:, :, :3].cpu().numpy()


class CvSource:
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            sys.exit(f"cannot open source: {path}")
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def next_bgr(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None


# --------------------------------------------------------------------------- benchmark

def run_benchmark(args):
    # Load YOLO-World
    classes = [c.strip() for c in args.classes.split(",")]
    print(f"[init] loading YOLO-World model ...")
    model = YOLOWorld(args.model)
    model.to("cuda")
    model.set_classes(classes)
    
    n_streams = args.streams
    print(f"[init] initializing {n_streams} parallel stream(s) in {args.mode.upper()} mode ...")
    
    # Warmup input batch
    warmup_batch = torch.zeros((n_streams, 3, 640, 640), device="cuda")
    for _ in range(5):
        model.predict(warmup_batch, verbose=False)
        
    h264_path = Path(args.source)
    mp4_path = Path(args.source)
    if mp4_path.suffix.lower() == ".h264":
        mp4_path = mp4_path.with_suffix(".mp4")

    # Initialize sources
    sources = []
    if args.mode == "neo":
        if not h264_path.exists():
            sys.exit(f"Source file not found: {h264_path}")
        try:
            for i in range(n_streams):
                sources.append(NeoSource(h264_path, with_display=args.display))
        except Exception as e:
            print(f"\n[ERROR] Failed to open {n_streams} NVDEC streams.")
            print("This is likely due to the NVIDIA GeForce driver concurrent session limit (5 streams).")
            print(f"Details: {e}")
            sys.exit(1)
    else:
        if not mp4_path.exists():
            sys.exit(f"Source file not found: {mp4_path}")
        for i in range(n_streams):
            sources.append(CvSource(mp4_path))
            
    w, h = sources[0].w, sources[0].h
    print(f"[init] streams ready: {n_streams}x ({w}x{h}).")
    
    if args.display:
        title = f"YOLO-World Mosaic [{args.mode.upper()}] — {n_streams} streams (q to quit)"
        t_last, fps = time.perf_counter(), 0.0
        
        cols = int(np.ceil(np.sqrt(n_streams)))
        rows = int(np.ceil(n_streams / cols))
        tile_disp_w, tile_disp_h = 640, 360
        
        print(f"[live] opening mosaic display grid {cols * tile_disp_w}x{rows * tile_disp_h} ...")
        
        while True:
            t0 = time.perf_counter()
            batch_tensors = []
            display_tiles = []
            
            if args.mode == "neo":
                for src in sources:
                    frame = src.next()
                    if frame is None:
                        break
                    pv, meta = gpu_preprocess_yolo.preprocess_gpu(frame, letterbox=not args.no_letterbox)
                    batch_tensors.append(pv)
                    display_tiles.append(np.ascontiguousarray(src.display_frame()))
            else:
                for src in sources:
                    bgr = src.next_bgr()
                    if bgr is None:
                        break
                    pv_cpu, meta = gpu_preprocess_yolo.preprocess_cpu(bgr, letterbox=not args.no_letterbox)
                    pv = pv_cpu.to("cuda")
                    batch_tensors.append(pv)
                    display_tiles.append(bgr)
                    
            if len(batch_tensors) < n_streams:
                break
                
            batched = torch.cat(batch_tensors, dim=0)
            
            # Synchronous batch inference
            results = model.predict(batched, verbose=False, conf=args.conf)
            
            # Render grid
            grid_img = np.zeros((rows * tile_disp_h, cols * tile_disp_w, 3), dtype=np.uint8)
            
            for i in range(n_streams):
                tile_bgr = display_tiles[i]
                tile_h, tile_w, _ = tile_bgr.shape
                
                # Draw detections on tile
                boxes = results[i].boxes
                if len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()

                    mapped = gpu_preprocess_yolo.unmap_boxes(xyxy, meta, tile_w, tile_h)
                    for j, (p1x, p1y, p2x, p2y) in enumerate(mapped):
                        label = f"{classes[cls_ids[j]]} {confs[j]:.2f}"
                        cv2.rectangle(tile_bgr, (p1x, p1y), (p2x, p2y), (0, 255, 0), 2)
                        cv2.putText(tile_bgr, label, (p1x, p1y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Resize tile to grid cell dimensions
                disp_tile = cv2.resize(tile_bgr, (tile_disp_w, tile_disp_h))
                
                r = i // cols
                c = i % cols
                grid_img[r * tile_disp_h:(r + 1) * tile_disp_h, c * tile_disp_w:(c + 1) * tile_disp_w] = disp_tile
                
            # Compute FPS
            dt = time.perf_counter() - t_last
            t_last = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
            
            cv2.putText(grid_img, f"{args.mode.upper()} | mosaic {fps:.1f} fps | {n_streams} streams",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            
            cv2.imshow(title, grid_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
            if not args.no_pace:
                time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t0)))
                
        cv2.destroyAllWindows()
        
    else:
        # Headless benchmark mode
        print(f"[bench] running {args.frames} iterations...")
        proc = psutil.Process()
        proc.cpu_percent()
        psutil.cpu_percent()
        
        t_start = time.perf_counter()
        
        for f in range(args.frames):
            batch_tensors = []
            
            if args.mode == "neo":
                for src in sources:
                    frame = src.next()
                    if frame is None:
                        sys.exit("EOF reached during benchmark")
                    pv, _meta = gpu_preprocess_yolo.preprocess_gpu(frame, letterbox=not args.no_letterbox)
                    batch_tensors.append(pv)
                batched = torch.cat(batch_tensors, dim=0)
            else:
                for src in sources:
                    bgr = src.next_bgr()
                    if bgr is None:
                        sys.exit("EOF reached during benchmark")
                    pv_cpu, _meta = gpu_preprocess_yolo.preprocess_cpu(bgr, letterbox=not args.no_letterbox)
                    pv = pv_cpu.to("cuda")
                    batch_tensors.append(pv)
                batched = torch.cat(batch_tensors, dim=0)
                
            model.predict(batched, verbose=False, conf=args.conf)
            if (f + 1) % 10 == 0:
                print(f"  progress: {f + 1}/{args.frames} batches done", flush=True)

        torch.cuda.synchronize()
        wall = time.perf_counter() - t_start
        cpu_proc = proc.cpu_percent() / psutil.cpu_count()
        cpu_sys = psutil.cpu_percent()
        
        total_frames = n_streams * args.frames
        print("\n========== MULTI-STREAM RESULTS ==========")
        print(f"Mode                : {args.mode.upper()}")
        print(f"Simultaneous streams: {n_streams}")
        print(f"Total frames processed: {total_frames}")
        print(f"Total wall time     : {wall:.3f}s")
        print(f"Throughput          : {total_frames / wall:.2f} FPS (total)")
        print(f"Throughput per stream: {(total_frames / wall) / n_streams:.2f} FPS")
        print(f"CPU usage (process) : {cpu_proc:.1f}%")
        print(f"CPU usage (system)  : {cpu_sys:.1f}%")
        print("==========================================")

        res = Path(__file__).parent / f"bench_multi_{args.mode}_{n_streams}.json"
        res.write_text(json.dumps({
            "mode": args.mode, "streams": n_streams, "frames_total": total_frames,
            "wall_s": wall, "fps_total": total_frames / wall,
            "fps_per_stream": (total_frames / wall) / n_streams,
            "cpu_process_pct": cpu_proc, "cpu_system_pct": cpu_sys,
        }, indent=2))
        print(f"[saved] {res}")


def main():
    ap = argparse.ArgumentParser(description="Multi-stream YOLO-World Benchmark")
    ap.add_argument("--source", required=True, help="Path to video file")
    ap.add_argument("--mode", choices=["neo", "baseline"], default="neo")
    ap.add_argument("--streams", type=int, default=1, help="number of parallel streams to decode")
    ap.add_argument("--classes", default="person, car", help="comma-separated classes to detect")
    ap.add_argument("--frames", type=int, default=50, help="number of batch iterations")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--model", default="assets/models/yolov8s-worldv2.pt", help="model name")
    ap.add_argument("--display", action="store_true", help="show live mosaic window")
    ap.add_argument("--no-pace", action="store_true", help="run display loop at full speed without 30 FPS cap")
    ap.add_argument("--no-letterbox", action="store_true",
                    help="aspect-squash resize instead of letterbox (legacy behavior)")
    args = ap.parse_args()
    
    run_benchmark(args)


if __name__ == "__main__":
    main()
