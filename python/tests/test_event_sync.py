import torch, neo
src = neo.VideoSource("assets/videos/bunny.h264")
t = torch.empty((3, src.height, src.width), dtype=torch.float32, device="cuda")
s = torch.cuda.Stream()
for i in range(3):
    assert src.next_into(t.data_ptr())
    src.wait_stream(torch.cuda.current_stream().cuda_stream)
    src.wait_stream(s.cuda_stream)
    with torch.cuda.stream(s):
        m = t.mean().item()
    src.synchronize()
    print(f"frame {i}: mean={m:.4f}")
print("OK - event sync API works (wait_stream + synchronize)")
