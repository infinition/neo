"""Smoke test: load model, run one inference on a frame extracted from the
sample clip, print the raw response and parsed boxes."""

import sys
import time

import cv2
import torch
from PIL import Image

from locate_live import load_model, parse_boxes

cap = cv2.VideoCapture("assets/videos/bunny.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ok, bgr = cap.read()
assert ok
pil = Image.fromarray(bgr[:, :, ::-1])
print(f"frame {pil.size}")

model, processor = load_model()
prompt = sys.argv[1] if len(sys.argv) > 1 else "the rabbit"

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": f"Locate all the instances that matches the following description: {prompt}."},
    ],
}]
text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = processor.process_vision_info(messages)
inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
print("keys:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in inputs.items()})

t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(
        pixel_values=inputs["pixel_values"].to(torch.bfloat16),
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs["image_grid_hws"],
        tokenizer=processor.tokenizer,
        max_new_tokens=256,
        generation_mode="hybrid",
        use_cache=True,
    )
print(f"inference: {time.perf_counter() - t0:.2f}s")
print("raw type:", type(out))
if torch.is_tensor(out):
    txt = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
elif isinstance(out, (list, tuple)):
    txt = out[0] if isinstance(out[0], str) else processor.tokenizer.decode(out[0], skip_special_tokens=False)
else:
    txt = str(out)
print("response:", txt[:500])
print("boxes:", parse_boxes(txt))
print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
