"""neo_filter_live — n'importe quel modèle ONNX 1-in/1-out NCHW, zero-copy.

Décode une source (fichier H.264 NVDEC zero-copy, ou webcam), passe chaque
frame dans un modèle de restauration/super-résolution via ONNX Runtime CUDA
EP avec IOBinding sur pointeurs CUDA (entrée ET sortie en VRAM), et affiche.

C'est LA démo où le zero-copy compte le plus : en super-résolution ×4 la
sortie fait 16× la taille de l'entrée — en mode baseline ce tenseur géant
traverse le PCIe à chaque frame (aller pour l'entrée, retour pour la sortie).

Modèles testés (export vs-mlrt, entrée (1,3,H,W)) :
  RealESRGANv3 (super-res ×4), cugan (×2), waifu2x. Voir --model.

Modes
-----
--mode neo       NVDEC zero-copy -> modèle (VRAM in/out). 0 pixel sur PCIe.
--mode baseline  OpenCV CPU -> ort.run numpy (PCIe entrée + sortie / frame).

Exemples
--------
  # Super-res live : on décode, on réduit à 360p (low-res simulé), on ×4 -> 1440p
  python neo_filter_live.py --source assets\\videos\\demo.h264 --mode neo \\
      --model assets\\models\\RealESRGANv3\\RealESRGANv3.onnx --in-height 360 --display

  # Benchmark comparatif (PCIe, FPS, CPU)
  python neo_filter_live.py --source assets\\videos\\demo.h264 --mode neo \\
      --model assets\\models\\RealESRGANv3\\RealESRGANv3.onnx --in-height 360 --bench 100
  python neo_filter_live.py --source assets\\videos\\demo.mp4 --mode baseline \\
      --model assets\\models\\RealESRGANv3\\RealESRGANv3.onnx --in-height 360 --bench 100

Touches : q quitter.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
import onnxruntime as ort  # noqa: E402


def open_session(model_path: str):
    if not Path(model_path).exists():
        sys.exit(f"modèle introuvable : {model_path}")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        model_path, so,
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    )
    ins = sess.get_inputs()
    if len(ins) != 1:
        sys.exit(f"modèle à 1 entrée attendu, reçu {[i.name for i in ins]}")
    return sess, ins[0].name, sess.get_outputs()[0].name


def probe_output_hw(sess, in_name, out_name, h, w):
    """Découvre la taille de sortie (facteur d'upscale) avec une passe à blanc."""
    dummy = np.zeros((1, 3, h, w), dtype=np.float32)
    out = sess.run([out_name], {in_name: dummy})[0]
    return out.shape[2], out.shape[3]


class NeoSource:
    def __init__(self, path):
        import neo
        self.src = neo.VideoSource(str(path))
        self.w, self.h = self.src.width, self.src.height
        self.frame = torch.empty((3, self.h, self.w), dtype=torch.float32, device="cuda")

    def next(self):
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


# --------------------------------------------------------------------------- neo

def run_neo(args, infer_h, infer_w):
    is_cam = args.source.startswith("webcam")
    if is_cam:
        spec = int(args.source.split(":", 1)[1]) if ":" in args.source else 0
        src = CvSource(spec, is_cam)
    else:
        src = NeoSource(str(args.source))
    sw, sh = src.w, src.h

    sess, in_name, out_name = open_session(args.model)
    oh, ow = probe_output_hw(sess, in_name, out_name, infer_h, infer_w)
    print(f"[init] source {sw}x{sh} -> entrée modèle {infer_w}x{infer_h} -> sortie {ow}x{oh} "
          f"(provider {sess.get_providers()[0]})")

    inp = torch.empty((1, 3, infer_h, infer_w), dtype=torch.float32, device="cuda")
    out = torch.empty((1, 3, oh, ow), dtype=torch.float32, device="cuda")

    io = sess.io_binding()
    io.bind_input(in_name, "cuda", 0, np.float32, tuple(inp.shape), inp.data_ptr())
    io.bind_output(out_name, "cuda", 0, np.float32, tuple(out.shape), out.data_ptr())

    need_resize = (infer_h, infer_w) != (sw, sh)

    def decode_into():
        if is_cam:
            bgr = src.next_bgr()
            if bgr is None:
                return False
            t = torch.from_numpy(bgr[:, :, ::-1].copy()).to("cuda")
            if "deoldify" in args.model.lower():
                t = t.permute(2, 0, 1).float()
                if need_resize:
                    small = torch.nn.functional.interpolate(
                        t.unsqueeze(0), size=(infer_h, infer_w),
                        mode="bilinear", align_corners=False, antialias=True).clamp_(0, 255)
                else:
                    small = t.unsqueeze(0)
                inp.copy_(small)
            else:
                t = t.permute(2, 0, 1).float().div_(255.0)
                if need_resize:
                    small = torch.nn.functional.interpolate(
                        t.unsqueeze(0), size=(infer_h, infer_w),
                        mode="bilinear", align_corners=False, antialias=True).clamp_(0, 1)
                else:
                    small = t.unsqueeze(0)
                inp.copy_(small)
            return True
        else:
            frame = src.next()
            if frame is None:
                return False
            if "deoldify" in args.model.lower():
                t = frame * 255.0
                if need_resize:
                    small = torch.nn.functional.interpolate(
                        t.unsqueeze(0), size=(infer_h, infer_w),
                        mode="bilinear", align_corners=False, antialias=True).clamp_(0, 255)
                else:
                    small = t.unsqueeze(0)
                inp.copy_(small)
            else:
                if need_resize:
                    small = torch.nn.functional.interpolate(
                        frame.unsqueeze(0), size=(infer_h, infer_w),
                        mode="bilinear", align_corners=False, antialias=True).clamp_(0, 1)
                else:
                    small = frame.unsqueeze(0)
                inp.copy_(small)
            return True

    def infer():
        torch.cuda.synchronize()
        sess.run_with_iobinding(io)
        return out

    loop(args, infer_w, infer_h, ow, oh, decode_into, infer, gpu=True)


# ---------------------------------------------------------------------- baseline

def run_baseline(args, infer_h, infer_w):
    is_cam = args.source.startswith("webcam")
    if is_cam:
        spec = int(args.source.split(":", 1)[1]) if ":" in args.source else 0
    else:
        spec = str(args.source)
    src = CvSource(spec, is_cam)
    sw, sh = src.w, src.h
    sess, in_name, out_name = open_session(args.model)
    oh, ow = probe_output_hw(sess, in_name, out_name, infer_h, infer_w)
    print(f"[init] source {sw}x{sh} -> entrée {infer_w}x{infer_h} -> sortie {ow}x{oh} (CPU decode)")

    holder = {"inp": np.zeros((1, 3, infer_h, infer_w), dtype=np.float32)}

    def decode_into():
        bgr = src.next_bgr()
        if bgr is None:
            return False
        small = cv2.resize(bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        if "deoldify" in args.model.lower():
            rgb = small[:, :, ::-1].astype(np.float32)
            chw = rgb.transpose(2, 0, 1)
            holder["inp"][0] = chw
        else:
            rgb = small[:, :, ::-1].astype(np.float32) / 255.0
            chw = rgb.transpose(2, 0, 1)
            holder["inp"][0] = chw
        return True

    def infer():
        return sess.run([out_name], {in_name: holder["inp"]})[0]  # PCIe in + out

    loop(args, infer_w, infer_h, ow, oh, decode_into, infer, gpu=False)


# -------------------------------------------------------------------------- loop

def loop(args, iw, ih, ow, oh, decode_into, infer, gpu):
    import psutil
    proc = psutil.Process()

    print("[init] warmup ...", flush=True)
    assert decode_into(), "source vide"
    for _ in range(3):
        infer()

    in_bytes = 3 * ih * iw * 4
    out_bytes = 3 * oh * ow * 4
    pcie = 0 if gpu else (in_bytes + out_bytes)

    t_dec, t_inf = [], []
    proc.cpu_percent()
    psutil.cpu_percent()
    n = args.bench if not args.display else 10 ** 9
    t_start = time.perf_counter()
    done = 0

    for i in range(n):
        t0 = time.perf_counter()
        if not decode_into():
            break
        t_dec.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        res = infer()
        if gpu:
            torch.cuda.synchronize()
        t_inf.append(time.perf_counter() - t0)
        done += 1

        if args.display:
            if gpu:
                out_tensor = res[0]
                if "deoldify" in args.model.lower():
                    u8 = out_tensor.clamp(0, 255).byte().permute(1, 2, 0).contiguous()
                else:
                    u8 = (out_tensor.clamp(0, 1) * 255).byte().flip(0).permute(1, 2, 0).contiguous()
                bgr = u8.cpu().numpy()  # download affichage uniquement
            else:
                out_arr = res[0]
                if "deoldify" in args.model.lower():
                    bgr = np.clip(out_arr.transpose(1, 2, 0), 0, 255).astype(np.uint8)
                else:
                    bgr = (np.clip(out_arr.transpose(1, 2, 0)[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
            bgr = np.ascontiguousarray(bgr)
            cv2.putText(bgr, f"[{args.mode}] {iw}x{ih} -> {ow}x{oh} | "
                             f"infer {1000 * np.mean(t_inf[-30:]):.0f} ms | "
                             f"PCIe {pcie / 1e6:.1f} MB/f",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(f"neo_filter_live [{args.mode}] (q quit)", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    wall = time.perf_counter() - t_start
    cpu_p = proc.cpu_percent() / psutil.cpu_count()
    cpu_s = psutil.cpu_percent()
    fps = done / wall if wall else 0

    print("\n========== FILTER RESULTS ==========")
    print(f"mode           : {args.mode}")
    print(f"pipeline       : {iw}x{ih} -> {ow}x{oh}  ({Path(args.model).stem})")
    print(f"frames         : {done} in {wall:.2f}s  =>  {fps:.1f} FPS")
    print(f"decode+prep ms : mean {1000 * np.mean(t_dec):.2f}")
    print(f"inference ms   : mean {1000 * np.mean(t_inf):.2f}")
    print(f"PCIe pixels/f  : {pcie / 1e6:.2f} MB  ({'ZERO — VRAM' if pcie == 0 else 'in+out chaque frame'})")
    print(f"CPU            : process {cpu_p:.1f}%  system {cpu_s:.1f}%")

    res = {"mode": args.mode, "in": [iw, ih], "out": [ow, oh], "fps": fps,
           "decode_ms": float(np.mean(t_dec)), "infer_ms": float(np.mean(t_inf)),
           "pcie_bytes_per_frame": pcie, "cpu_process_pct": cpu_p, "cpu_system_pct": cpu_s}
    p = Path(__file__).parent / f"bench_filter_{args.mode}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"[saved] {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True, help="ONNX 1-in/1-out NCHW (1,3,H,W)")
    ap.add_argument("--mode", choices=["neo", "baseline"], default="neo")
    ap.add_argument("--in-height", type=int, default=0,
                    help="réduit l'entrée à cette hauteur (16:9 déduit) — 0 = taille source")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()
    if not args.bench and not args.display:
        args.bench = 100

    if args.in_height:
        ih = args.in_height
        if "deoldify" in args.model.lower():
            # Pour DeOldify (architecture U-Net), la hauteur et la largeur doivent être des multiples de 32
            ih = int(round(ih / 32) * 32)
            iw = int(round(ih * 16 / 9 / 32) * 32)
        else:
            iw = int(round(ih * 16 / 9 / 8) * 8)
    else:
        # taille source : récupérée dans run_* via la source
        ih = iw = 0

    is_cam = args.source.startswith("webcam")
    if not ih:
        if is_cam:
            spec = int(args.source.split(":", 1)[1]) if ":" in args.source else 0
            c = cv2.VideoCapture(spec, cv2.CAP_DSHOW)
            iw = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            ih = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            c.release()
        else:
            if args.mode == "neo":
                import neo
                s = neo.VideoSource(str(args.source)); ih, iw = s.height, s.width; del s
            else:
                c = cv2.VideoCapture(str(args.source))
                iw = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                ih = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                c.release()

    if args.mode == "neo":
        run_neo(args, ih, iw)
    else:
        run_baseline(args, ih, iw)


if __name__ == "__main__":
    main()
