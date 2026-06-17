"""face_anon_live — anonymisation faciale temps réel (sensibilisation deepfake).

Détecte les visages/personnes sur un flux vidéo (YOLO-World open-vocabulary,
décodage NVDEC zero-copy) et les pixelise en direct.

Pourquoi cette démo : la localisation de visage en temps réel sur GPU grand
public est exactement le PREMIER étage de tout pipeline deepfake (détecter ->
aligner -> remplacer). Cette démo s'arrête à l'étage défendable : elle
ANONYMISE (pixelise) au lieu d'usurper. Le message de sensibilisation : ce
qui tourne ici à >100 FPS sur une carte de gamer, c'est la moitié du chemin
d'un deepfake — la barrière technique a disparu, seul le choix éthique reste.

Le pipeline lourd (frames en VRAM, détection batchée, 0 octet sur PCIe) est
celui prouvé par les benchmarks neo : un face-swap utiliserait la MÊME
plomberie zero-copy, en remplaçant la pixelisation par un modèle de swap.

Modes
-----
--mode neo       NVDEC zero-copy -> détection VRAM. 0 pixel sur PCIe.
--mode baseline  OpenCV CPU decode -> détection.

Exemples
--------
  python face_anon_live.py --source webcam --display
  python face_anon_live.py --source assets\\videos\\demo.h264 --mode neo --display
  python face_anon_live.py --source assets\\videos\\demo.h264 --mode neo --bench 100

Touches : q quitter, p basculer pixelisation / boîtes simples.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld

import gpu_preprocess_yolo as pre


def pixelate(img, x1, y1, x2, y2, blocks=12):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (blocks, max(1, blocks * (y2 - y1) // max(1, x2 - x1))),
                       interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


class NeoSource:
    def __init__(self, path):
        import neo
        self.src = neo.VideoSource(path)
        self.w, self.h = self.src.width, self.src.height
        self.frame = torch.empty((3, self.h, self.w), dtype=torch.float32, device="cuda")
        self.bgra = torch.empty((self.h, self.w, 4), dtype=torch.uint8, device="cuda")

    def next(self):
        if not self.src.next_into_with_bgra(self.frame.data_ptr(), self.bgra.data_ptr()):
            return None
        self.src.wait_stream(torch.cuda.current_stream().cuda_stream)
        return self.frame

    def display_bgr(self):
        return np.ascontiguousarray(self.bgra[:, :, :3].cpu().numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help=".h264/.mp4 ou webcam[:index]")
    ap.add_argument("--mode", choices=["neo", "baseline"], default="neo")
    ap.add_argument("--classes", default="human face, head",
                    help="cibles à anonymiser (open-vocabulary)")
    ap.add_argument("--model", default="assets/models/yolov8s-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.02, help="seuil (visages = bas)")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()
    if not args.display and not args.bench:
        args.bench = 100

    classes = [c.strip() for c in args.classes.split(",")]
    print(f"[init] YOLO-World, cibles={classes}")
    model = YOLOWorld(args.model)
    model.to("cuda")
    model.set_classes(classes)

    is_cam = args.source.startswith("webcam")
    if args.mode == "neo" and not is_cam:
        src = NeoSource(str(args.source))
        w, h = src.w, src.h
    else:
        idx = int(args.source.split(":", 1)[1]) if ":" in args.source else 0
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if is_cam else cv2.VideoCapture(str(args.source))
        if not cap.isOpened():
            sys.exit(f"source inaccessible : {args.source}")
        if is_cam:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FPS, 30)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # warmup
    for _ in range(5):
        model.predict(torch.zeros((1, 3, 640, 640), device="cuda"), verbose=False)

    import psutil
    proc = psutil.Process(); proc.cpu_percent(); psutil.cpu_percent()
    t0 = time.perf_counter()
    n = args.bench if not args.display else 10 ** 9
    fps, t_last = 0.0, time.perf_counter()
    do_pix, done, total_faces = True, 0, 0

    for i in range(n):
        if args.mode == "neo" and not is_cam:
            ft = src.next()
            if ft is None:
                break
            pv, meta = pre.preprocess_gpu(ft, letterbox=True)
            disp = src.display_bgr() if args.display else None
        else:
            ok, bgr = cap.read()
            if not ok:
                if is_cam:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            if args.mode == "neo":
                t = torch.from_numpy(bgr[:, :, ::-1].copy()).to("cuda").permute(2, 0, 1).float().div_(255.0)
                pv, meta = pre.preprocess_gpu(t, letterbox=True)
            else:
                pv_cpu, meta = pre.preprocess_cpu(bgr, letterbox=True)
                pv = pv_cpu.to("cuda")
            disp = bgr if args.display else None

        res = model.predict(pv, verbose=False, conf=args.conf)[0].boxes
        done += 1
        boxes = pre.unmap_boxes(res.xyxy.cpu().numpy(), meta, w, h) if len(res) else []
        total_faces += len(boxes)

        if args.display and disp is not None:
            for (x1, y1, x2, y2) in boxes:
                if do_pix:
                    pixelate(disp, x1, y1, x2, y2)
                else:
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            dt = time.perf_counter() - t_last; t_last = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
            cv2.putText(disp, f"ANON [{args.mode}] | {fps:.0f} fps | {len(boxes)} visage(s) "
                              f"| {'pixelise' if do_pix else 'boites'} (p)",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("face_anon_live (q quit)", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("p"):
                do_pix = not do_pix

    wall = time.perf_counter() - t0
    cpu_p = proc.cpu_percent() / psutil.cpu_count()
    print("\n========== ANON RESULTS ==========")
    print(f"mode      : {args.mode}   source {w}x{h}")
    print(f"frames    : {done} in {wall:.2f}s  =>  {done / wall:.1f} FPS")
    print(f"détections: {total_faces} boîtes ({total_faces / max(1, done):.1f}/frame)")
    print(f"CPU       : process {cpu_p:.1f}%")

    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
