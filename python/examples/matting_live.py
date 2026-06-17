"""matting_live — Background removal temps réel (RVM) optimisé.

Démontre la puissance de l'architecture neo (zero-copy) sur un modèle
Image-to-Image lourd : Robust Video Matting (RVM).

En mode neo, l'image est décodée en VRAM, transférée dans PyTorch/ONNX via IOBinding
et le blending (incrustation) est fait sur le GPU pour éviter de
déplacer de gros tenseurs (1080p).
"""

import argparse
import sys
import time
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import psutil

os_add_dll_directory = getattr(os, "add_dll_directory", None)
if os_add_dll_directory is not None:
    os_add_dll_directory(str(Path(torch.__file__).parent / "lib"))
import onnxruntime as ort


def ensure_annexb(path: Path) -> Path:
    if path.suffix.lower() in (".h264", ".264", ".annexb"):
        return path
    import tempfile
    import subprocess
    out = Path(tempfile.gettempdir()) / (path.stem + ".h264")
    if not out.exists():
        print(f"[init] extraction du flux Annex-B H.264 -> {out}")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path),
             "-c:v", "copy", "-bsf:v", "h264_mp4toannexb", "-an", str(out)],
            check=True,
        )
    return out


class NeoSource:
    """Décode en VRAM (zero-copy)."""

    def __init__(self, path):
        import neo
        annexb_path = ensure_annexb(Path(path))
        self.src = neo.VideoSource(str(annexb_path))
        self.w, self.h = self.src.width, self.src.height
        self.frame = torch.empty((3, self.h, self.w), dtype=torch.float32, device="cuda")
        self.bgra = torch.empty((self.h, self.w, 4), dtype=torch.uint8, device="cuda")

    def next(self, with_display=False):
        if with_display:
            ok = self.src.next_into_with_bgra(self.frame.data_ptr(), self.bgra.data_ptr())
        else:
            ok = self.src.next_into(self.frame.data_ptr())
        if not ok:
            return None
        self.src.wait_stream(torch.cuda.current_stream().cuda_stream)
        return self.frame


class CvSource:
    def __init__(self, spec, is_cam):
        self.cap = cv2.VideoCapture(spec, cv2.CAP_DSHOW) if is_cam else cv2.VideoCapture(str(spec))
        if not self.cap.isOpened():
            sys.exit(f"source inaccessible : {spec}")
        if is_cam:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.is_cam = is_cam
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def next_bgr(self):
        ok, bgr = self.cap.read()
        if not ok:
            if self.is_cam:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self.cap.read()
        return bgr if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help=".h264/.mp4 ou webcam[:index]")
    ap.add_argument("--mode", choices=["neo", "baseline"], default="neo")
    ap.add_argument("--model", default="assets/models/rvm/rvm_mobilenetv3_fp32.onnx")
    ap.add_argument("--bg-color", type=str, default="0,255,0", help="R,G,B")
    ap.add_argument("--bg-image", type=str, default=None, help="Chemin vers une image d'arrière-plan (.jpg/.png)")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()
    if not args.display and not args.bench:
        args.bench = 100

    print("[init] Chargement de Robust Video Matting (RVM) ...")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        args.model, so,
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    )

    is_cam = args.source.startswith("webcam")
    if args.mode == "neo" and not is_cam:
        src = NeoSource(str(args.source))
    else:
        spec = int(args.source.split(":", 1)[1]) if (is_cam and ":" in args.source) else (
            0 if is_cam else str(args.source))
        src = CvSource(spec, is_cam)
    w, h = src.w, src.h
    print(f"[init] source {w}x{h}, mode {args.mode}")

    bg_color = tuple(map(int, args.bg_color.split(',')))
    bg_color_bgr = (bg_color[2], bg_color[1], bg_color[0])

    t_dec, t_inf, t_post = [], [], []
    pcie_bytes = 0
    produced = 0
    proc = psutil.Process()
    proc.cpu_percent()
    psutil.cpu_percent()

    if args.mode == "neo":
        # Découverte des formes des états récurrents par dry-run
        print("[init] Découverte des dimensions des états recurrents RVM ...")
        dummy_src = np.zeros((1, 3, h, w), dtype=np.float32)
        dummy_r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        dummy_r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        dummy_r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        dummy_r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        dummy_ratio = np.array([0.25], dtype=np.float32)

        _, _, d_r1, d_r2, d_r3, d_r4 = sess.run(
            ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
            {
                "src": dummy_src,
                "r1i": dummy_r1, "r2i": dummy_r2, "r3i": dummy_r3, "r4i": dummy_r4,
                "downsample_ratio": dummy_ratio
            }
        )
        print(f"[init] Dimensions détectées : r1={d_r1.shape}, r2={d_r2.shape}, r3={d_r3.shape}, r4={d_r4.shape}")

        # Allocation des tenseurs CUDA (zero-copy)
        src_t = torch.empty((1, 3, h, w), dtype=torch.float32, device="cuda")
        ratio_t = torch.tensor([0.25], dtype=torch.float32, device="cuda")

        r1i_t = torch.zeros(d_r1.shape, dtype=torch.float32, device="cuda")
        r2i_t = torch.zeros(d_r2.shape, dtype=torch.float32, device="cuda")
        r3i_t = torch.zeros(d_r3.shape, dtype=torch.float32, device="cuda")
        r4i_t = torch.zeros(d_r4.shape, dtype=torch.float32, device="cuda")

        r1o_t = torch.empty(d_r1.shape, dtype=torch.float32, device="cuda")
        r2o_t = torch.empty(d_r2.shape, dtype=torch.float32, device="cuda")
        r3o_t = torch.empty(d_r3.shape, dtype=torch.float32, device="cuda")
        r4o_t = torch.empty(d_r4.shape, dtype=torch.float32, device="cuda")

        fgr_t = torch.empty((1, 3, h, w), dtype=torch.float32, device="cuda")
        pha_t = torch.empty((1, 1, h, w), dtype=torch.float32, device="cuda")

        # Configuration IOBinding
        io = sess.io_binding()
        io.bind_input("src", "cuda", 0, np.float32, tuple(src_t.shape), src_t.data_ptr())
        io.bind_input("r1i", "cuda", 0, np.float32, tuple(r1i_t.shape), r1i_t.data_ptr())
        io.bind_input("r2i", "cuda", 0, np.float32, tuple(r2i_t.shape), r2i_t.data_ptr())
        io.bind_input("r3i", "cuda", 0, np.float32, tuple(r3i_t.shape), r3i_t.data_ptr())
        io.bind_input("r4i", "cuda", 0, np.float32, tuple(r4i_t.shape), r4i_t.data_ptr())
        io.bind_input("downsample_ratio", "cuda", 0, np.float32, tuple(ratio_t.shape), ratio_t.data_ptr())

        io.bind_output("fgr", "cuda", 0, np.float32, tuple(fgr_t.shape), fgr_t.data_ptr())
        io.bind_output("pha", "cuda", 0, np.float32, tuple(pha_t.shape), pha_t.data_ptr())
        io.bind_output("r1o", "cuda", 0, np.float32, tuple(r1o_t.shape), r1o_t.data_ptr())
        io.bind_output("r2o", "cuda", 0, np.float32, tuple(r2o_t.shape), r2o_t.data_ptr())
        io.bind_output("r3o", "cuda", 0, np.float32, tuple(r3o_t.shape), r3o_t.data_ptr())
        io.bind_output("r4o", "cuda", 0, np.float32, tuple(r4o_t.shape), r4o_t.data_ptr())

        # Warmup GPU
        print("[init] Warmup GPU ...")
        for _ in range(3):
            sess.run_with_iobinding(io)
        torch.cuda.synchronize()

        # Fond en VRAM (RGB float32 [0,1])
        if args.bg_image:
            if not Path(args.bg_image).exists():
                sys.exit(f"Image de fond introuvable : {args.bg_image}")
            bg_img = cv2.imread(args.bg_image)
            if bg_img is None:
                sys.exit(f"Impossible de lire l'image de fond : {args.bg_image}")
            bg_img = cv2.resize(bg_img, (w, h))
            bg_tensor = torch.from_numpy(bg_img[:, :, ::-1].copy()).to("cuda")
            bg_tensor = bg_tensor.permute(2, 0, 1).float().div_(255.0)
        else:
            bg_tensor = torch.tensor([bg_color[0], bg_color[1], bg_color[2]], dtype=torch.float32, device="cuda").view(3, 1, 1) / 255.0

    else:
        # Initialisation baseline NumPy
        if args.bg_image:
            if not Path(args.bg_image).exists():
                sys.exit(f"Image de fond introuvable : {args.bg_image}")
            bg_img = cv2.imread(args.bg_image)
            if bg_img is None:
                sys.exit(f"Impossible de lire l'image de fond : {args.bg_image}")
            bg_image = cv2.resize(bg_img, (w, h))
        else:
            bg_image = np.full((h, w, 3), bg_color_bgr, dtype=np.uint8)
        r1i = np.zeros((1, 1, 1, 1), dtype=np.float32)
        r2i = np.zeros((1, 1, 1, 1), dtype=np.float32)
        r3i = np.zeros((1, 1, 1, 1), dtype=np.float32)
        r4i = np.zeros((1, 1, 1, 1), dtype=np.float32)
        downsample_ratio = np.array([0.25], dtype=np.float32)

        # Warmup baseline
        print("[init] Warmup CPU/GPU baseline ...")
        dummy_src = np.zeros((1, 3, h, w), dtype=np.float32)
        for _ in range(3):
            sess.run(
                ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
                {
                    "src": dummy_src,
                    "r1i": r1i, "r2i": r2i, "r3i": r3i, "r4i": r4i,
                    "downsample_ratio": downsample_ratio
                }
            )

    print("[init] Démarrage de la boucle principale ...")
    t0 = time.perf_counter()
    n = args.bench if not args.display else 10 ** 9
    disp_fps, t_last = 0.0, time.perf_counter()

    for i in range(n):
        # -------------------------------------------------- DECODE & PREPARE
        t_dec_start = time.perf_counter()
        if args.mode == "neo":
            if is_cam:
                bgr = src.next_bgr()
                if bgr is None:
                    break
                t_dec.append(time.perf_counter() - t_dec_start)
                
                t_prep_start = time.perf_counter()
                t = torch.from_numpy(bgr[:, :, ::-1].copy()).to("cuda")
                t = t.permute(2, 0, 1).float().div_(255.0)
                src_t.copy_(t.unsqueeze(0))
                # On cumule le temps de prep au décodage GPU
                t_dec[-1] += (time.perf_counter() - t_prep_start)
            else:
                frame_t = src.next(with_display=args.display)
                if frame_t is None:
                    break
                src_t.copy_(frame_t.unsqueeze(0))
                t_dec.append(time.perf_counter() - t_dec_start)
        else:
            bgr = src.next_bgr()
            if bgr is None:
                break
            rgb = bgr[:, :, ::-1]
            tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
            tensor = np.expand_dims(tensor, 0)
            pcie_bytes += tensor.nbytes
            t_dec.append(time.perf_counter() - t_dec_start)

        # -------------------------------------------------- INFERENCE
        t_inf_start = time.perf_counter()
        if args.mode == "neo":
            sess.run_with_iobinding(io)
            torch.cuda.synchronize()
            t_inf.append(time.perf_counter() - t_inf_start)

            # Copy recurrent states GPU -> GPU
            r1i_t.copy_(r1o_t)
            r2i_t.copy_(r2o_t)
            r3i_t.copy_(r3o_t)
            r4i_t.copy_(r4o_t)
        else:
            fgr, pha, r1o, r2o, r3o, r4o = sess.run(
                ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
                {
                    "src": tensor,
                    "r1i": r1i, "r2i": r2i, "r3i": r3i, "r4i": r4i,
                    "downsample_ratio": downsample_ratio
                }
            )
            pcie_bytes += fgr.nbytes + pha.nbytes
            t_inf.append(time.perf_counter() - t_inf_start)
            r1i, r2i, r3i, r4i = r1o, r2o, r3o, r4o

        # -------------------------------------------------- COMPOSITING / POST-PROCESS
        t_post_start = time.perf_counter()
        if args.mode == "neo":
            compo_rgb = fgr_t[0] * pha_t[0] + bg_tensor * (1.0 - pha_t[0])
            compo_bgr = compo_rgb.flip(0)
            result_t = (compo_bgr.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0)
            if args.display:
                result = result_t.contiguous().cpu().numpy()
            t_post.append(time.perf_counter() - t_post_start)
        else:
            pha_img = np.squeeze(pha)
            pha_img = np.expand_dims(pha_img, axis=-1)
            fgr_img = np.squeeze(fgr).transpose(1, 2, 0)[:, :, ::-1] * 255.0
            compo = fgr_img * pha_img + bg_image * (1 - pha_img)
            result = compo.astype(np.uint8)
            t_post.append(time.perf_counter() - t_post_start)

        produced += 1

        # -------------------------------------------------- DISPLAY
        if args.display:
            dt = time.perf_counter() - t_last
            t_last = time.perf_counter()
            disp_fps = 0.9 * disp_fps + 0.1 * (1.0 / max(dt, 1e-6))
            
            infer_ms = t_inf[-1] * 1000.0
            cv2.putText(result, f"MATTING [{args.mode}] | {disp_fps:.0f} fps | "
                                f"infer {infer_ms:.1f} ms",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("matting_live (q quit)", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    wall = time.perf_counter() - t0
    cpu_p = proc.cpu_percent() / psutil.cpu_count()
    cpu_s = psutil.cpu_percent()

    print("\n========== MATTING RESULTS ==========")
    print(f"mode           : {args.mode}   source {w}x{h}")
    print(f"frames         : {produced} in {wall:.2f}s  =>  {produced / wall:.1f} FPS")
    print(f"decode+prep ms : mean {1000 * np.mean(t_dec):.2f}")
    print(f"inference ms   : mean {1000 * np.mean(t_inf):.2f}")
    print(f"postprocess ms : mean {1000 * np.mean(t_post):.2f}")
    print(f"PCIe trafic    : {pcie_bytes / 1024 / 1024:.1f} MB total ({pcie_bytes / 1024 / 1024 / max(1, produced):.2f} MB/frame)")
    print(f"CPU            : process {cpu_p:.1f}%  system {cpu_s:.1f}%")

    res = {"mode": args.mode, "in": [w, h], "fps": produced / wall,
           "decode_ms": float(np.mean(t_dec)), "infer_ms": float(np.mean(t_inf)),
           "postprocess_ms": float(np.mean(t_post)), "pcie_bytes_per_frame": pcie_bytes / max(1, produced),
           "cpu_process_pct": cpu_p, "cpu_system_pct": cpu_s}
    
    p = Path(__file__).parent / f"bench_matting_{args.mode}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"[saved] {p}")

    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
