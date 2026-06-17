"""GPU preprocessing for YOLO-World, mirroring ultralytics' LetterBox.

Ultralytics letterboxes (aspect-preserving resize + gray 114 padding) when it
preprocesses images itself. When you pass a raw torch tensor to
`model.predict`, it is used as-is — so we must letterbox ourselves, otherwise
the model sees aspect-squashed frames and accuracy silently degrades.

`preprocess_gpu` (VRAM in, VRAM out) and `preprocess_cpu` (baseline,
cv2/numpy) produce the same geometry; both return the mapping needed to
project detections back to original frame coordinates via `unmap_boxes`.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

YOLO_SIZE = 640
PAD_VALUE = 114 / 255.0


@torch.no_grad()
def preprocess_gpu(frame: torch.Tensor, size: int = YOLO_SIZE, letterbox: bool = True,
                   dtype=torch.float32):
    """frame: (3, H, W) f32 RGB in [0,1], CUDA -> ((1,3,size,size), meta).

    meta = (gain, pad_x, pad_y) for unmap_boxes. Everything stays in VRAM.
    """
    _, h, w = frame.shape
    x = frame.unsqueeze(0)
    if letterbox:
        gain = min(size / w, size / h)
        new_w, new_h = round(w * gain), round(h * gain)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        x = F.pad(x, (pad_x, size - new_w - pad_x, pad_y, size - new_h - pad_y),
                  value=PAD_VALUE)
        meta = (gain, pad_x, pad_y)
    else:
        if (h, w) != (size, size):
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        meta = (None, 0, 0)  # plain squash: unmap uses per-axis scale
    return x.to(dtype), meta


def preprocess_cpu(bgr: np.ndarray, size: int = YOLO_SIZE, letterbox: bool = True):
    """Baseline path: BGR uint8 HWC (CPU) -> ((1,3,size,size) f32 CPU, meta).

    Same geometry as preprocess_gpu so both modes are directly comparable.
    """
    h, w = bgr.shape[:2]
    if letterbox:
        gain = min(size / w, size / h)
        new_w, new_h = round(w * gain), round(h * gain)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        resized = cv2.resize(bgr, (new_w, new_h))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        meta = (gain, pad_x, pad_y)
    else:
        canvas = cv2.resize(bgr, (size, size))
        meta = (None, 0, 0)
    rgb = canvas[:, :, ::-1].copy()
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    return t, meta


def unmap_boxes(xyxy, meta, orig_w: int, orig_h: int, size: int = YOLO_SIZE):
    """Map (N,4) boxes from model input space back to original frame pixels."""
    gain, pad_x, pad_y = meta
    out = []
    for x1, y1, x2, y2 in xyxy:
        if gain is not None:
            x1, x2 = (x1 - pad_x) / gain, (x2 - pad_x) / gain
            y1, y2 = (y1 - pad_y) / gain, (y2 - pad_y) / gain
        else:
            x1, x2 = x1 * orig_w / size, x2 * orig_w / size
            y1, y2 = y1 * orig_h / size, y2 * orig_h / size
        out.append((max(0, int(x1)), max(0, int(y1)),
                    min(orig_w - 1, int(x2)), min(orig_h - 1, int(y2))))
    return out
