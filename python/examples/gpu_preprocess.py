"""GPU replica of LocateAnythingImageProcessor.

Mirrors image_processing_locateanything.py exactly, but takes an RGB CHW f32
tensor already resident in VRAM (e.g. produced by neo.VideoSource) and
never touches host memory:

    rescale (bicubic, multiples of merge*patch) -> normalize ((x-.5)/.5)
    -> patchify (N, 3, 14, 14) + grid (H/14, W/14)

The CPU reference path is: PIL bicubic resize -> to_tensor -> TF.normalize
-> patchify. Differences here are sub-quantization (PIL works on uint8).
"""

import math

import torch
import torch.nn.functional as F

PATCH = 14
MERGE = (2, 2)
IN_TOKEN_LIMIT = 25600  # from preprocessor_config.json


def target_size(w: int, h: int) -> tuple[int, int]:
    """Final (W, H) after the processor's rescale step."""
    if (w // PATCH) * (h // PATCH) > IN_TOKEN_LIMIT:
        scale = math.sqrt(IN_TOKEN_LIMIT / ((w // PATCH) * (h // PATCH)))
        w, h = int(w * scale), int(h * scale)
    pad_w = MERGE[1] * PATCH
    pad_h = MERGE[0] * PATCH
    tw = math.ceil(w / pad_w) * pad_w
    th = math.ceil(h / pad_h) * pad_h
    if tw // PATCH >= 512 or th // PATCH >= 512:
        raise ValueError("Exceed pos emb")
    return tw, th


@torch.no_grad()
def preprocess_gpu(frame: torch.Tensor, dtype=torch.bfloat16):
    """frame: (3, H, W) f32 RGB in [0,1], CUDA. Returns (pixel_values, grid_hw).

    pixel_values: (N, 3, 14, 14) `dtype` CUDA tensor, N = grid_h * grid_w.
    Everything runs on the GPU; no host transfer.
    """
    _, h, w = frame.shape
    tw, th = target_size(w, h)
    x = frame.unsqueeze(0)
    if (tw, th) != (w, h):
        x = F.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False, antialias=True)
        x = x.clamp_(0.0, 1.0)
    x = x.squeeze(0) * 2.0 - 1.0  # normalize mean=std=0.5

    c, hh, ww = x.shape
    gh, gw = hh // PATCH, ww // PATCH
    patches = x.reshape(c, gh, PATCH, gw, PATCH).permute(1, 3, 0, 2, 4)
    patches = patches.contiguous().view(-1, c, PATCH, PATCH).to(dtype)
    return patches, (gh, gw)
