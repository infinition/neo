"""Sanity check: GPU preprocess (neo path) vs HF CPU processor on one frame.

Decodes one frame with neo, runs both preprocessing paths, compares
pixel_values numerically. Small differences are expected (the CPU path
quantizes through uint8/PIL and uses a different YUV->RGB rounding), large
ones indicate a layout bug.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import gpu_preprocess
import neo

src = neo.VideoSource(sys.argv[1] if len(sys.argv) > 1 else "assets/videos/bunny.h264")
frame = torch.empty((3, src.height, src.width), dtype=torch.float32, device="cuda")
bgra = torch.empty((src.height, src.width, 4), dtype=torch.uint8, device="cuda")
assert src.next_into_with_bgra(frame.data_ptr(), bgra.data_ptr())

# GPU path
pv_gpu, grid_gpu = gpu_preprocess.preprocess_gpu(frame, dtype=torch.float32)

# CPU reference path on the SAME decoded frame (via the BGRA download)
from transformers import AutoImageProcessor

ip = AutoImageProcessor.from_pretrained("nvidia/LocateAnything-3B", trust_remote_code=True)
rgb = bgra[:, :, :3].cpu().numpy()[:, :, ::-1]
feat = ip.preprocess(Image.fromarray(rgb), return_tensors="pt")
pv_cpu, grid_cpu = feat["pixel_values"], feat["image_grid_hws"]

print(f"grid  gpu={grid_gpu} cpu={np.asarray(grid_cpu).tolist()}")
print(f"shape gpu={tuple(pv_gpu.shape)} cpu={tuple(pv_cpu.shape)}")
diff = (pv_gpu.cpu().float() - pv_cpu.float()).abs()
print(f"diff  max={diff.max():.4f} mean={diff.mean():.5f}  (values span [-1,1])")
if list(pv_gpu.shape) == list(pv_cpu.shape) and diff.mean() < 0.05:
    print("OK — GPU preprocess matches the HF processor")
else:
    print("MISMATCH — check layout/resize")
