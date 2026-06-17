import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

def ensure_annexb(path: Path) -> Path:
    if path.suffix.lower() in (".h264", ".264", ".annexb"):
        return path
    import tempfile
    import subprocess
    out = Path(tempfile.gettempdir()) / (path.stem + ".h264")
    if not out.exists():
        print(f"[init] Extraction du flux Annex-B H.264 -> {out}")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path),
             "-c:v", "copy", "-bsf:v", "h264_mp4toannexb", "-an", str(out)],
            check=True,
        )
    return out


class NeoSource:
    """Decodes video frames directly into CUDA memory (Zero-Copy VRAM)."""
    def __init__(self, path):
        import neo
        annexb_path = ensure_annexb(Path(path))
        self.src = neo.VideoSource(str(annexb_path))
        self.w, self.h = self.src.width, self.src.height
        self.frame = torch.empty((3, self.h, self.w), dtype=torch.float32, device="cuda")

    def next(self):
        ok = self.src.next_into(self.frame.data_ptr())
        if not ok:
            return None
        self.src.wait_stream(torch.cuda.current_stream().cuda_stream)
        return self.frame


class CvSource:
    """Decodes video frames using CPU OpenCV."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            sys.exit(f"Source video inaccessible : {path}")
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def next_bgr(self):
        ok, bgr = self.cap.read()
        return bgr if ok else None


def ingest_video(video_path, encoder, mode="neo", sample_fps=1.0, batch_size=8):
    """
    Decodes the video and extracts embeddings at a specified sampling rate.
    All tensors are stored and accumulated in VRAM.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        sys.exit(f"Fichier vidéo introuvable : {video_path}")

    # Read video properties with OpenCV first
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If the video format doesn't provide a frame count (e.g. raw H.264)
    if total_frames <= 0:
        total_frames = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total_frames += 1
        # Re-open capture since we read it to the end
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        
    duration = total_frames / video_fps if total_frames > 0 else 0
    cap.release()

    print(f"[ingest] Vidéo : {video_path.name}")
    print(f"[ingest] Résolution native : {video_w}x{video_h} | native FPS : {video_fps:.2f} | Durée : {duration:.1f}s")
    print(f"[ingest] Mode : {mode} | Taux d'échantillonnage : {sample_fps} FPS | Total Frames : {total_frames}")

    if mode == "neo":
        src = NeoSource(video_path)
    else:
        src = CvSource(video_path)

    # Frame step to match target sample_fps
    frame_step = int(round(video_fps / sample_fps))
    if frame_step < 1:
        frame_step = 1

    embeddings = []
    timestamps = []

    batch_frames = []
    batch_timestamps = []

    frame_idx = 0
    processed_count = 0
    t_start = time.perf_counter()

    while True:
        if frame_idx >= total_frames:
            break

        if mode == "neo":
            frame_t = src.next()
            if frame_t is None:
                break
            if frame_idx % frame_step == 0:
                # Must clone since the native ring buffer gets overwritten
                batch_frames.append(frame_t.clone())
                batch_timestamps.append(frame_idx / video_fps)
        else:
            bgr = src.next_bgr()
            if bgr is None:
                break
            if frame_idx % frame_step == 0:
                t = torch.from_numpy(bgr[:, :, ::-1].copy()).to("cuda")
                t = t.permute(2, 0, 1).float().div_(255.0)
                batch_frames.append(t)
                batch_timestamps.append(frame_idx / video_fps)

        frame_idx += 1

        # Run inference in batches to maximize GPU occupancy
        if len(batch_frames) >= batch_size:
            batch_tensor = torch.stack(batch_frames)
            emb = encoder.encode_images(batch_tensor)  # (B, 512)
            embeddings.append(emb)
            timestamps.extend(batch_timestamps)
            processed_count += len(batch_frames)
            
            # Reset buffers
            batch_frames = []
            batch_timestamps = []
            
            elapsed = time.perf_counter() - t_start
            print(f"[ingest] Indexé {processed_count} frames... ({processed_count / elapsed:.1f} FPS)", end="\r", flush=True)

    # Process remaining frames
    if batch_frames:
        batch_tensor = torch.stack(batch_frames)
        emb = encoder.encode_images(batch_tensor)
        embeddings.append(emb)
        timestamps.extend(batch_timestamps)
        processed_count += len(batch_frames)

    total_time = time.perf_counter() - t_start
    print(f"\n[ingest] Indexation terminée : {processed_count} frames en {total_time:.2f}s ({processed_count / total_time:.1f} FPS)")

    if embeddings:
        embeddings_tensor = torch.cat(embeddings, dim=0)
    else:
        embeddings_tensor = torch.empty((0, 512), device="cuda")

    return embeddings_tensor, torch.tensor(timestamps, dtype=torch.float32, device="cuda")
