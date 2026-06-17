"""Parity check: neo GPU preprocessing vs baseline CPU preprocessing.

Takes ONE frame decoded by neo, pushes it through both preprocessing paths
(GPU letterbox vs cv2 letterbox), runs YOLO-World on both, and compares the
detections. They must match (same count/classes, IoU ~1.0) for the FPS
benchmarks to be a fair apples-to-apples comparison.

Usage: python check_parity.py [video.h264] [classes] [skip_frames]
"""

import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLOWorld

import gpu_preprocess_yolo as pre
import neo

video = sys.argv[1] if len(sys.argv) > 1 else "assets/videos/bunny.h264"
classes = [c.strip() for c in (sys.argv[2] if len(sys.argv) > 2 else "rabbit, butterfly, bird").split(",")]

src = neo.VideoSource(video)
frame = torch.empty((3, src.height, src.width), dtype=torch.float32, device="cuda")
bgra = torch.empty((src.height, src.width, 4), dtype=torch.uint8, device="cuda")
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 30
for _ in range(skip):
    assert src.next_into_with_bgra(frame.data_ptr(), bgra.data_ptr())
src.synchronize()

# Path A: neo GPU letterbox on the VRAM frame
pv_gpu, meta_a = pre.preprocess_gpu(frame)

# Path B: baseline CPU letterbox on the SAME frame (BGRA download)
bgr = bgra[:, :, :3].cpu().numpy()
pv_cpu, meta_b = pre.preprocess_cpu(bgr)

diff = (pv_gpu.cpu() - pv_cpu).abs()
print(f"pixel_values: shape gpu={tuple(pv_gpu.shape)} cpu={tuple(pv_cpu.shape)}  "
      f"diff max={diff.max():.4f} mean={diff.mean():.5f}")

model = YOLOWorld("assets/models/yolov8s-worldv2.pt")
model.to("cuda")
model.set_classes(classes)

ra = model.predict(pv_gpu, verbose=False, conf=0.25)[0].boxes
rb = model.predict(pv_cpu.to("cuda"), verbose=False, conf=0.25)[0].boxes


def fmt(b):
    return [(classes[int(c)], round(float(cf), 3), [round(float(v), 1) for v in xy])
            for xy, c, cf in zip(b.xyxy.cpu(), b.cls.cpu(), b.conf.cpu())]


print(f"neo GPU path : {len(ra)} box(es) {fmt(ra)}")
print(f"baseline path: {len(rb)} box(es) {fmt(rb)}")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


if len(ra) == len(rb) and len(ra) > 0:
    ious = [max(iou(x.tolist(), y.tolist()) for y in rb.xyxy.cpu()) for x in ra.xyxy.cpu()]
    print(f"IoU (matched): min={min(ious):.3f} mean={np.mean(ious):.3f}")
    print("PARITY OK" if min(ious) > 0.9 else "PARITY WEAK — investigate")
elif len(ra) == len(rb):
    print("PARITY OK (no detections on this frame — try other classes/frame)")
else:
    print("PARITY MISMATCH — different box counts")
