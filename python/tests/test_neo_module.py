import torch, neo
src = neo.VideoSource("assets/videos/bunny.h264")
print("video", src.width, "x", src.height)
t = torch.empty((3, src.height, src.width), dtype=torch.float32, device="cuda")
for i in range(5):
    assert src.next_into(t.data_ptr())
    print(f"frame {i}: mean={t.mean().item():.4f} min={t.min().item():.3f} max={t.max().item():.3f}")
print("OK - zero-copy decode works")
