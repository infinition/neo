import subprocess
from pathlib import Path

def extract_clip(video_path, start_time, end_time, output_path):
    """
    Extracts a sub-clip from the source video using GPU-accelerated NVENC encoder.
    If NVENC is not available or fails, falls back to CPU encoding.
    """
    duration = end_time - start_time
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. GPU accelerated command (h264_nvenc)
    cmd_gpu = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-c:v", "h264_nvenc",
        "-preset", "p1", # fastest preset
        "-an", # ignore audio
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd_gpu, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[extract] Clip extrait (VRAM -> NVENC) : {output_path.name} [{start_time:.1f}s - {end_time:.1f}s]")
        return True
    except Exception:
        # 2. CPU fallback command (libx264)
        cmd_cpu = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", f"{start_time:.3f}",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-an",
            str(output_path)
        ]
        try:
            subprocess.run(cmd_cpu, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[extract] Clip extrait (CPU Fallback) : {output_path.name} [{start_time:.1f}s - {end_time:.1f}s]")
            return True
        except Exception as e:
            print(f"[extract] Erreur lors de l'extraction du clip : {e}")
            return False
