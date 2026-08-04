#!/usr/bin/env python3
"""
qwen_neo_moe.py — Passerelle Python pour neo-moe + llama.cpp.

Deux modes :
  1. DIRECT — charge neo_moe.dll et pilote le prefetch MoE depuis Python.
  2. WRAPPER — lance llama-server.exe avec les flags optimaux et injecte
     neo-moe via la lib (nécessite un build custom de llama.cpp).

Prérequis :
  cargo build --release  (produit target/release/neo_moe.dll)

Usage :
  python qwen_neo_moe.py --model .models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import c_uint32, c_uint64, c_char_p, c_void_p, c_int32, c_int64
from pathlib import Path

# ─── FFI bindings ─────────────────────────────────────────────────────────────────

class NeoMoeFFI:
    """Charge neo_moe.dll et expose les symboles C."""

    def __init__(self, dll_path: str = None):
        if dll_path is None:
            # Cherche dans target/release/
            dll_path = (
                Path(__file__).parent / "target" / "release" / "neo_moe.dll"
            )
            if not dll_path.exists():
                dll_path = (
                    Path(__file__).parent.parent
                    / "target"
                    / "release"
                    / "neo_moe.dll"
                )

        dll_path = Path(dll_path)
        if not dll_path.exists():
            print(f"[neo-moe] DLL introuvable: {dll_path}")
            print("[neo-moe] Lance d'abord : cargo build --release")
            self.available = False
            return

        self.lib = ctypes.CDLL(str(dll_path))
        self.available = True
        self._bind_functions()

    def _bind_functions(self):
        lib = self.lib

        # neo_moe_stream_create
        lib.neo_moe_stream_create.restype = c_void_p
        lib.neo_moe_stream_create.argtypes = [
            c_char_p, c_uint32, c_uint32, c_uint32, c_uint32,
        ]

        # neo_moe_stream_free
        lib.neo_moe_stream_free.argtypes = [c_void_p]

        # neo_moe_demand
        lib.neo_moe_demand.restype = c_int32
        lib.neo_moe_demand.argtypes = [
            c_void_p, c_uint32, c_uint32,
            ctypes.POINTER(c_uint64),  # gate_ptr
            ctypes.POINTER(c_uint64),  # up_ptr
            ctypes.POINTER(c_uint64),  # down_ptr
            c_uint32,
        ]

        # neo_moe_prefetch
        lib.neo_moe_prefetch.restype = c_int32
        lib.neo_moe_prefetch.argtypes = [
            c_void_p, c_uint32,
            ctypes.POINTER(c_uint32), c_uint32,
        ]

        # neo_moe_release
        lib.neo_moe_release.argtypes = [c_void_p, c_uint32, c_uint32]

        # neo_moe_model_info
        lib.neo_moe_model_info.restype = c_int32
        lib.neo_moe_model_info.argtypes = [
            c_void_p,
            ctypes.POINTER(c_uint32),
            ctypes.POINTER(c_uint32),
        ]

        # neo_moe_demand_keep
        lib.neo_moe_demand_keep.restype = c_int64
        lib.neo_moe_demand_keep.argtypes = [
            c_void_p, c_uint32, c_uint32,
            ctypes.POINTER(c_uint64),
            ctypes.POINTER(c_uint64),
            ctypes.POINTER(c_uint64),
            c_uint32,
        ]

        # neo_moe_release_handle
        lib.neo_moe_release_handle.argtypes = [c_int64]

    def create_stream(
        self, gguf_path: str, vram_slots=16, io_threads=4, cuda_device=0, depth=2
    ) -> int:
        """Crée le streaming engine. Retourne le pointeur (handle)."""
        if not self.available:
            return 0
        ptr = self.lib.neo_moe_stream_create(
            gguf_path.encode("utf-8"),
            vram_slots, io_threads, cuda_device, depth,
        )
        return ptr

    def free_stream(self, ptr: int):
        if ptr:
            self.lib.neo_moe_stream_free(ptr)

    def model_info(self, ptr: int):
        """Retourne (n_layers, n_experts) ou (0, 0)."""
        if not ptr:
            return (0, 0)
        n_layers = c_uint32(0)
        n_experts = c_uint32(0)
        rc = self.lib.neo_moe_model_info(
            ptr, ctypes.byref(n_layers), ctypes.byref(n_experts)
        )
        if rc == 0:
            return (n_layers.value, n_experts.value)
        return (0, 0)

    def demand(self, ptr: int, layer: int, expert: int, timeout_ms=30000):
        """Demande un expert. Retourne (gate_ptr, up_ptr, down_ptr) ou None."""
        if not ptr:
            return None
        gate = c_uint64(0)
        up = c_uint64(0)
        down = c_uint64(0)
        rc = self.lib.neo_moe_demand(
            ptr, layer, expert,
            ctypes.byref(gate), ctypes.byref(up), ctypes.byref(down),
            timeout_ms,
        )
        if rc == 0:
            return (gate.value, up.value, down.value)
        return None

    def prefetch(self, ptr: int, layer: int, expert_ids: list):
        """Prefetch spéculatif non-bloquant."""
        if not ptr or not expert_ids:
            return
        arr = (c_uint32 * len(expert_ids))(*expert_ids)
        self.lib.neo_moe_prefetch(ptr, layer, arr, len(expert_ids))

    def release(self, ptr: int, layer: int, expert: int):
        """Libère le slot VRAM."""
        if ptr:
            self.lib.neo_moe_release(ptr, layer, expert)


# ─── LLama-server launcher ────────────────────────────────────────────────────────

def launch_llama_server(args):
    """Lance llama-server.exe avec les flags optimisés neo-moe."""

    root = Path(args.model).parent.parent
    llama_dir = root / "llama-b9721-bin-win-cuda-12.4-x64"
    cuda_dir = root / "cudart-llama-bin-win-cuda-12.4-x64"

    # Construire la commande
    cmd = [
        str(llama_dir / "llama-server.exe"),
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--ctx-size", str(args.ctx_size),
        "--predict", str(args.predict),
        "--temp", str(args.temperature),
        "--top-p", str(args.top_p),
        "--top-k", str(args.top_k),
        "--repeat-penalty", str(args.repeat_penalty),
        "--flash-attn",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
    ]

    if args.cpu_moe:
        cmd.append("--cpu-moe")
        print("[launcher] Mode CPU-MoE (experts sur CPU, safe pour 12GB VRAM)")
    else:
        print("[launcher] Mode GPU-MoE (experts sur GPU — necessite assez de VRAM)")

    # Alias + chat template
    cmd.extend(["--alias", "qwen3.6-35b-a3b"])
    cmd.extend(['--chat-template-kwargs', '{"preserve_thinking": true}'])

    # PATH
    env = os.environ.copy()
    env["PATH"] = f"{cuda_dir};{llama_dir};{env['PATH']}"

    print(f"[launcher] Demarrage de llama-server...")
    print(f"[launcher] Port: {args.host}:{args.port}")
    print(f"[launcher] Commande: {' '.join(str(c) for c in cmd)}")
    print()

    proc = subprocess.Popen(cmd, env=env)
    return proc


# ─── Neo-moe integration demo ──────────────────────────────────────────────────────

def run_moe_demo(ffi: NeoMoeFFI, dll_path: str, model_path: str):
    """Demo du streaming MoE : parse le GGUF, prefetch, demand, release."""

    if not ffi.available:
        print("\n[WARN] neo_moe.dll non disponible — demo ignorée.")
        print(f"[WARN] Build : cargo build --release dans {dll_path}")
        return

    print(f"\n[neo-moe] Initialisation du streaming engine...")
    stream = ffi.create_stream(model_path, vram_slots=16, io_threads=4)

    if not stream:
        print("[neo-moe] ERREUR: echec de l'initialisation.")
        return

    n_layers, n_experts = ffi.model_info(stream)
    print(f"[neo-moe] Modele : {n_layers} couches × {n_experts} experts")
    print(f"[neo-moe] Streaming, prefetch depth=2, 16 slots VRAM")

    # Demo : un token sur la couche 0
    layer = 0
    top_k = 4

    print(f"\n[neo-moe] Prefetch des {top_k} premiers experts (couche {layer})...")
    ffi.prefetch(stream, layer, list(range(top_k)))

    print(f"[neo-moe] Demand des experts...")
    for expert in range(top_k):
        ptrs = ffi.demand(stream, layer, expert, timeout_ms=5000)
        if ptrs:
            gate, up, down = ptrs
            print(
                f"  Expert {expert:2d} → gate@{gate:#010x}  "
                f"up@{up:#010x}  down@{down:#010x}"
            )
        else:
            print(f"  Expert {expert:2d} → TIMEOUT")
        ffi.release(stream, layer, expert)

    ffi.free_stream(stream)
    print(f"\n[neo-moe] Demo terminee.")


# ─── CLI ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="neo-moe : MoE streaming engine pour Qwen3.6-35B"
    )
    parser.add_argument(
        "--model", default=None,
        help="Chemin vers le .gguf (defaut: cherche dans .models/)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--predict", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--cpu-moe", action="store_true", default=True,
                        help="Utilise --cpu-moe (defaut: True pour 12GB VRAM)")
    parser.add_argument("--dll", default=None,
                        help="Chemin vers neo_moe.dll")
    parser.add_argument("--demo", action="store_true",
                        help="Lance la demo neo-moe uniquement (sans serveur)")

    args = parser.parse_args()

    # Détection auto du modèle
    if args.model is None:
        candidates = [
            Path(__file__).parent / ".models" / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            Path.cwd() / ".models" / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        ]
        for c in candidates:
            if c.exists():
                args.model = str(c)
                break
        if args.model is None:
            print("[ERREUR] Modele introuvable. Utilise --model <path>")
            sys.exit(1)

    print(f"[config] Modele: {args.model}")

    neo_moe_path = Path(args.model).parent.parent
    dll_candidates = [
        args.dll,
        str(neo_moe_path / "neo-moe" / "target" / "release" / "neo_moe.dll"),
        str(Path(__file__).parent / "target" / "release" / "neo_moe.dll"),
        str(Path.cwd() / "target" / "release" / "neo_moe.dll"),
    ]

    ffi = None
    for dll in dll_candidates:
        if dll and Path(dll).exists():
            ffi = NeoMoeFFI(dll)
            break
    if ffi is None:
        ffi = NeoMoeFFI()  # dernière chance

    if args.demo:
        run_moe_demo(ffi, str(neo_moe_path / "neo-moe"), args.model)
        return

    # Mode serveur
    proc = launch_llama_server(args)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[launcher] Arret du serveur...")
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
