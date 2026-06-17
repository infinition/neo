"""face_swap_live — deepfake temps réel (sensibilisation).

Échange un visage source avec les visages détectés dans le flux vidéo, pour
DÉMONTRER qu'un deepfake live tourne sur une carte grand public.

Architecture (corrigée pour la fluidité)
----------------------------------------
- Détection UNIQUE par frame via InsightFace (SCRFD) — pas de double
  détection. `FaceAnalysis.get()` fait déjà détection + landmarks + embedding ;
  un détecteur séparé (YOLO) serait redondant, on l'a supprimé.
- Worker asynchrone « latest-frame-wins » : le thread d'affichage ne gèle
  jamais, le swap tourne aussi vite que le GPU le permet et l'overlay se
  rafraîchit à son propre rythme.
- `--det-size` (def. 320) : SCRFD en 320² au lieu de 640² ≈ 2× plus rapide.
- `--detect-every N` : ne redétecte que toutes les N frames, réutilise les
  visages entre-temps (un visage ne se téléporte pas) — encore du gain.

Honnêteté zero-copy : le face-swap touche de PETITS crops, le coût est
l'inférence (SCRFD + ArcFace + inswapper), pas le mouvement de données. Le
décode neo garde le CPU libre et évite le transfert de la frame complète,
mais ce workload n'est PAS la vitrine du zero-copy (voir neo_filter_live.py
pour la super-résolution, là où ça compte). InsightFace travaille en numpy
BGR : on télécharge la frame UNE fois (pas un aller-retour par visage comme
avant).

Modes
-----
--mode neo       NVDEC zero-copy decode (CPU libre) -> 1 download -> swap.
--mode baseline  OpenCV CPU decode -> swap.

Exemples
--------
  python face_swap_live.py --source webcam --source-face source_face.jpg --display
  python face_swap_live.py --source assets\\videos\\demo.h264 --mode neo --display
  python face_swap_live.py --source assets\\videos\\demo.h264 --mode neo --bench 100

Touches : q quitter, s afficher/masquer le swap (montre l'original).
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
from insightface.app import FaceAnalysis  # noqa: E402
from insightface.model_zoo import get_model  # noqa: E402


class NeoSource:
    """Décode en VRAM (zero-copy) puis télécharge la frame BGR UNE fois
    (InsightFace travaille en numpy)."""

    def __init__(self, path):
        import neo
        self.src = neo.VideoSource(path)
        self.w, self.h = self.src.width, self.src.height
        self.frame = torch.empty((3, self.h, self.w), dtype=torch.float32, device="cuda")
        self.bgra = torch.empty((self.h, self.w, 4), dtype=torch.uint8, device="cuda")

    def next_bgr(self):
        if not self.src.next_into_with_bgra(self.frame.data_ptr(), self.bgra.data_ptr()):
            return None
        self.src.wait_stream(torch.cuda.current_stream().cuda_stream)
        return np.ascontiguousarray(self.bgra[:, :, :3].cpu().numpy())


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help=".h264/.mp4 ou webcam[:index]")
    ap.add_argument("--source-face", default="assets/images/source_face.jpg")
    ap.add_argument("--mode", choices=["neo", "baseline"], default="neo")
    ap.add_argument("--swapper", default="assets/models/inswapper/inswapper_128.onnx")
    ap.add_argument("--det-size", type=int, default=320, help="taille SCRFD (320 rapide, 640 précis)")
    ap.add_argument("--detect-every", type=int, default=2, help="redétecte toutes les N frames")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()
    if not args.display and not args.bench:
        args.bench = 100

    print(f"[init] InsightFace (SCRFD det_size={args.det_size}) ...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    if not Path(args.swapper).exists():
        sys.exit(f"inswapper introuvable : {args.swapper}")
    swapper = get_model(args.swapper, download=False,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    print(f"[init] visage source : {args.source_face}")
    simg = cv2.imread(args.source_face)
    if simg is None:
        sys.exit(f"image source illisible : {args.source_face}")
    sfaces = app.get(simg)
    if not sfaces:
        sys.exit("aucun visage dans l'image source.")
    source_face = max(sfaces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    is_cam = args.source.startswith("webcam")
    if args.mode == "neo" and not is_cam:
        src = NeoSource(str(args.source))
    else:
        spec = int(args.source.split(":", 1)[1]) if (is_cam and ":" in args.source) else (
            0 if is_cam else str(args.source))
        src = CvSource(spec, is_cam)
    w, h = src.w, src.h
    print(f"[init] source {w}x{h}, mode {args.mode}")

    # ---- worker async : prend la dernière frame, détecte + swappe ----
    shared = {"frame": None, "result": None, "faces": [], "n": 0,
              "swap_ms": 0.0, "n_faces": 0, "stop": False, "show": True}
    lock = threading.Lock()

    def worker():
        fcount = 0
        last_faces = []
        while not shared["stop"]:
            with lock:
                bgr = shared["frame"]
                shared["frame"] = None
            if bgr is None:
                time.sleep(0.001)
                continue
            t0 = time.perf_counter()
            if fcount % max(1, args.detect_every) == 0:
                last_faces = app.get(bgr)  # détection unique (SCRFD + embed)
            fcount += 1
            for f in last_faces:
                try:
                    bgr = swapper.get(bgr, f, source_face, paste_back=True)
                except Exception:
                    pass
            with lock:
                shared["result"] = bgr
                shared["faces"] = last_faces
                shared["swap_ms"] = (time.perf_counter() - t0) * 1000.0
                shared["n_faces"] = len(last_faces)
                shared["n"] += 1

    threading.Thread(target=worker, daemon=True).start()

    import psutil
    proc = psutil.Process(); proc.cpu_percent(); psutil.cpu_percent()
    t0 = time.perf_counter()
    n = args.bench if not args.display else 10 ** 9
    disp_fps, t_last = 0.0, time.perf_counter()
    produced = 0

    for i in range(n):
        bgr = src.next_bgr()
        if bgr is None:
            break
        produced += 1
        with lock:
            shared["frame"] = bgr                 # dernière frame pour le worker
            result = shared["result"]
            swap_ms, n_faces, swaps = shared["swap_ms"], shared["n_faces"], shared["n"]

        if args.display:
            show = result if (result is not None and shared["show"]) else bgr
            show = np.ascontiguousarray(show)
            dt = time.perf_counter() - t_last; t_last = time.perf_counter()
            disp_fps = 0.9 * disp_fps + 0.1 * (1.0 / max(dt, 1e-6))
            cv2.putText(show, f"DEEPFAKE [{args.mode}] | affichage {disp_fps:.0f} fps | "
                              f"swap {swap_ms:.0f} ms ({1000 / max(swap_ms, 1):.0f} fps) | "
                              f"{n_faces} visage(s) | s:original",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("face_swap_live (q quit, s original)", show)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                shared["show"] = not shared["show"]
        elif i % 20 == 0:
            print(f"  frame {i}: swaps réalisés {swaps}, dernier {swap_ms:.0f} ms", flush=True)

    shared["stop"] = True
    time.sleep(0.05)
    wall = time.perf_counter() - t0
    cpu_p = proc.cpu_percent() / psutil.cpu_count()
    print("\n========== DEEPFAKE RESULTS ==========")
    print(f"mode            : {args.mode}   source {w}x{h}")
    print(f"frames décodées : {produced} in {wall:.2f}s  =>  {produced / wall:.1f} FPS affichage")
    print(f"swaps réalisés  : {shared['n']}  (worker {shared['n'] / wall:.1f} fps, "
          f"~{shared['swap_ms']:.0f} ms/swap)")
    print(f"CPU             : process {cpu_p:.1f}%")
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
