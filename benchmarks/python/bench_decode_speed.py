import time
import torch
import neo

src = neo.VideoSource("assets/videos/demo.h264")
print(f"Video resolution: {src.width}x{src.height}")

frame = torch.empty((3, src.height, src.width), dtype=torch.float32, device="cuda")
bgra = torch.empty((src.height, src.width, 4), dtype=torch.uint8, device="cuda")

t_decode_accum = 0
t_download_accum = 0

for i in range(20):
    t0 = time.perf_counter()
    ok = src.next_into_with_bgra(frame.data_ptr(), bgra.data_ptr())
    t1 = time.perf_counter()
    if not ok:
        print("EOF")
        break
    
    # Simulate display download
    t2 = time.perf_counter()
    disp = bgra[:, :, :3].cpu().numpy()
    t3 = time.perf_counter()
    
    print(f"Frame {i}: decode+copy={ (t1-t0)*1000 :.2f}ms download={ (t3-t2)*1000 :.2f}ms")
    if i > 0: # Skip first frame compile overhead
        t_decode_accum += (t1-t0)
        t_download_accum += (t3-t2)

print(f"Average (excluding first): decode={t_decode_accum/19*1000:.2f}ms download={t_download_accum/19*1000:.2f}ms")
