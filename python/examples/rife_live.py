"""RIFE live player — interpolation temps réel sur fichier H.264 ou webcam.

Décode la source, interpole (factor-1) frames intermédiaires entre chaque
paire (RIFE v4, timestep arbitraire) et affiche le flux à fps*factor.

Architecture : un thread producteur (décode -> RIFE -> téléchargement
affichage) alimente une petite file ; le thread principal présente les
frames à cadence STRICTEMENT régulière — pas de rafales, c'est ce qui fait
la fluidité. Si le GPU ne tient pas la cible, la cadence s'adapte à ce qui
est réellement produit (affiché dans le HUD).

- Fichier .h264/.mp4 : décodage NVDEC zero-copy (neo.pyd).
- Webcam : capture USB (host par nature), upload GPU, inférence VRAM.
  À 640x480 l'inférence coûte ~8-10 ms -> 30->60/90 fluide.

Latence inhérente : ~1 frame source + inférence (RIFE a besoin de la frame
suivante) + la file (~factor frames d'affichage).

Exemples
--------
  python rife_live.py --source assets\\videos\\demo.h264 --fps 24 --factor 2
  python rife_live.py --source webcam --factor 2 --zoom 2
  python rife_live.py --source webcam:0 --cam-width 1280 --cam-height 720 --factor 3

Touches : q quitter, i n'afficher que les frames interpolées.
"""

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
import onnxruntime as ort  # noqa: E402


# ------------------------------------------------------------------------ rife

class GpuRife:
    """RIFE v4 (export vs-mlrt v2, 7 canaux) via ONNX Runtime CUDA EP,
    IOBinding sur pointeurs CUDA : entrées/sorties 100% VRAM."""

    def __init__(self, model_path: str, w: int, h: int):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            model_path, so,
            providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )
        meta = self.sess.get_inputs()[0]
        if len(self.sess.get_inputs()) != 1 or meta.shape[1] != 7:
            sys.exit(f"export RIFE 'v2' attendu (1 entrée, 7 canaux), reçu {meta.shape}")
        self.inp = torch.empty((1, 7, h, w), dtype=torch.float32, device="cuda")
        self.out = torch.empty((1, 3, h, w), dtype=torch.float32, device="cuda")
        io = self.sess.io_binding()
        io.bind_input(meta.name, "cuda", 0, np.float32, tuple(self.inp.shape), self.inp.data_ptr())
        io.bind_output(self.sess.get_outputs()[0].name, "cuda", 0, np.float32,
                       tuple(self.out.shape), self.out.data_ptr())
        self.io = io
        print(f"[init] {Path(model_path).name} {w}x{h} provider={self.sess.get_providers()[0]}")

    def push(self, frame: torch.Tensor):
        """img1 -> img0, nouvelle frame (3,H,W) f32 [0,1] RGB cuda -> img1."""
        self.inp[0, 0:3].copy_(self.inp[0, 3:6])
        self.inp[0, 3:6].copy_(frame)

    def interpolate(self, t: float) -> torch.Tensor:
        self.inp[0, 6].fill_(t)
        torch.cuda.synchronize()
        self.sess.run_with_iobinding(self.io)
        return self.out


# --------------------------------------------------------------------- sources

class NeoFile:
    def __init__(self, path: str):
        import neo
        self.src = neo.VideoSource(path)
        self.w, self.h = self.src.width, self.src.height
        self.frame = torch.empty((3, self.h, self.w), dtype=torch.float32, device="cuda")

    def next(self):
        if not self.src.next_into(self.frame.data_ptr()):
            return None
        self.src.wait_stream(torch.cuda.current_stream().cuda_stream)
        return self.frame


class Webcam:
    def __init__(self, idx: int, w: int, h: int, fps: float):
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            sys.exit(f"webcam {idx} introuvable")
        if w:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[init] webcam {self.w}x{self.h} @ {self.cap.get(cv2.CAP_PROP_FPS):.0f} fps annoncés")

    def next(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None
        t = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).to("cuda", non_blocking=True)
        return t.permute(2, 0, 1).float().div_(255.0)


# ---------------------------------------------------------------------- player

def to_bgr_u8(rgb_chw: torch.Tensor) -> np.ndarray:
    """(3,H,W) f32 [0,1] cuda -> BGR u8 HWC numpy (téléchargement affichage)."""
    u8 = (rgb_chw.clamp(0, 1) * 255).byte()
    bgr = u8.flip(0).permute(1, 2, 0).contiguous()
    return bgr.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help=".h264/.mp4 ou webcam[:index]")
    ap.add_argument("--factor", type=int, default=2, choices=[2, 3, 4],
                    help="multiplication des FPS")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="cadence source (fichier) ou demandée (webcam)")
    ap.add_argument("--model", default="assets/models/rife_v2/rife_v4.14.onnx")
    ap.add_argument("--cam-width", type=int, default=0)
    ap.add_argument("--cam-height", type=int, default=0)
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="agrandissement fenêtre (ex: 2 pour une webcam 640x480)")
    ap.add_argument("--duration", type=float, default=0, help="auto-stop après N s (0 = infini)")
    args = ap.parse_args()

    is_cam = args.source.startswith("webcam")
    path = args.source
    if not is_cam and not path.lower().endswith((".h264", ".264")):
        import subprocess
        import tempfile
        out = Path(tempfile.gettempdir()) / (Path(path).stem + ".h264")
        if not out.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-c:v", "copy",
                            "-bsf:v", "h264_mp4toannexb", "-an", str(out)], check=True)
        path = str(out)

    factor = args.factor

    # Le producteur possède TOUT le pipeline GPU : le VideoSource pyo3 est
    # `unsendable` (il doit vivre et mourir dans le thread qui l'a créé).
    q: "queue.Queue[tuple[np.ndarray, bool]]" = queue.Queue(maxsize=2 * factor)
    state = {"stop": False, "prod_dt": 1.0 / (args.fps * factor), "infer_ms": 0.0}
    ready = threading.Event()

    def producer():
        try:
            if is_cam:
                idx = int(args.source.split(":", 1)[1]) if ":" in args.source else 0
                src = Webcam(idx, args.cam_width, args.cam_height, args.fps)
            else:
                src = NeoFile(path)
            rife = GpuRife(args.model, src.w, src.h)
            f = src.next()
            assert f is not None, "source vide"
            rife.inp[0, 3:6].copy_(f)
            for _ in range(2):
                rife.interpolate(0.5)
        except Exception as e:
            print(f"[producer] init: {e}", flush=True)
            state["stop"] = True
            ready.set()
            return
        ready.set()

        ema = state["prod_dt"]
        while not state["stop"]:
            t0 = time.perf_counter()
            fr = src.next()
            if fr is None:
                state["stop"] = True
                break
            rife.push(fr)
            outs = []
            ti = time.perf_counter()
            for i in range(1, factor):
                outs.append((to_bgr_u8(rife.interpolate(i / factor)[0]), True))
            state["infer_ms"] = (time.perf_counter() - ti) * 1000 / max(1, factor - 1)
            outs.append((to_bgr_u8(fr), False))
            ema = 0.9 * ema + 0.1 * ((time.perf_counter() - t0) / factor)
            state["prod_dt"] = ema
            for item in outs:
                while not state["stop"]:
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        pass

    threading.Thread(target=producer, daemon=True).start()
    ready.wait()
    print(f"[live] cible {args.fps:.0f} -> {args.fps * factor:.0f} fps, fenêtre ouverte ...")

    # -------- présentation : cadence régulière, adaptée à ce qui est produit
    title = f"RIFE live x{factor} [{'webcam' if is_cam else 'neo zero-copy'}] (q quit)"
    target_dt = 1.0 / (args.fps * factor)
    interp_only = False
    shown, win_t0, fps_disp = 0, time.perf_counter(), 0.0
    next_present = time.perf_counter()
    t_start = time.perf_counter()

    while not state["stop"]:
        try:
            bgr, is_interp = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if interp_only and not is_interp:
            continue

        # cadence effective : la cible, ou le débit réel de production si le
        # GPU ne suit pas — espacement régulier garanti dans les deux cas.
        eff_dt = max(target_dt, state["prod_dt"])
        next_present = max(next_present + eff_dt, time.perf_counter() - eff_dt)
        sleep = next_present - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)

        if args.zoom != 1.0:
            bgr = cv2.resize(bgr, None, fx=args.zoom, fy=args.zoom,
                             interpolation=cv2.INTER_LINEAR)
        shown += 1
        el = time.perf_counter() - win_t0
        if el >= 1.0:
            fps_disp = shown / el
            shown, win_t0 = 0, time.perf_counter()
        cv2.putText(bgr, f"x{factor} | {fps_disp:.1f} fps affichés (cible {1 / target_dt:.0f},"
                         f" soutenable {1 / state['prod_dt']:.0f})"
                         f" | infer {state['infer_ms']:.1f} ms | {'interp' if is_interp else 'source'}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(title, bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("i"):
            interp_only = not interp_only
        if args.duration and time.perf_counter() - t_start > args.duration:
            break

    state["stop"] = True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
