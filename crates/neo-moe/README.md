# neo-moe

Zero-copy MoE expert streaming — **NVMe → pinned host → VRAM** — as a Neo crate.

Enables running models like **Qwen3.6-35B-A3B-UD-Q4\_K\_XL** (~20 GB) on a
**RTX 4070 Ti (12 GB VRAM)** by streaming only the active experts into VRAM
on-demand, rather than requiring the full model to be resident.

---

## Architecture

```
NVMe (GGUF)
  Expert[L][E] bytes at known file offset
        │
        │  ┌─ Linux:   O_DIRECT + io_uring (no page cache, ~3.5 ms)
        │  └─ Windows:  mmap (page cache, ~8 ms)
        ▼
Pinned host staging     ← cuMemAllocHost (page-locked, DMA-able)
  per-thread buffer
        │
        │  cuMemcpyHtoD (synchronous H2D, no CPU involvement)
        ▼
VRAM expert pool        ← cudarc DeviceSlice<u8>
  vram_resident_experts slots  (double-buffered, 16 × ~25 MB)
        │
        │  raw *mut u8 device pointer — zero-copy
        ▼
Inference backend
  llama.cpp / ONNX Runtime CUDA EP / TensorRT
  (ggml_tensor.data = handle.gate_ptr())
```

### Key insight

Qwen3.6-35B-A3B activates only **3.6B parameters per token** out of 35B total.
That's ~4 experts × 3 projections × ~25 MB each = ~300 MB needed per MoE layer.
The 4070 Ti has 12 GB — more than enough for all active experts
**plus** KV cache and non-expert weights, as long as we stream lazily.

---

## Quick start

```bash
# Build the shared library (cdylib)
cd neo-moe && cargo build --release

# Windows: produces target/release/neo_moe.dll
# Linux:   produces target/release/libneo_moe.so

# Run the integration demo
cargo run --example integration --release -- --model path/to/model.gguf

# Run the streaming benchmark
cargo run --example bench --release -- --model path/to/model.gguf

# Or use the Python launcher
python qwen_neo_moe.py --model path/to/model.gguf --demo
```

## Launcher Qwen

Le dossier `Qwen_3.6_35b/` contient trois façons de lancer le modèle :

| Fichier | Description |
|---------|-------------|
| `Qwen3.6-35B-A3B-UD-Q4_K_XL.bat` | Launcher original — CPU-MoE (stable, recommandé) |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL.neo-moe.bat` | Launcher avec choix de stratégie MoE (menu interactif) |
| `qwen_neo_moe.py` | Passerelle Python avec FFI neo-moe + contrôle fin |

### Stratégies de streaming

1. **CPU-MoE** — `--cpu-moe` : experts MoE sur CPU, attention + KV sur GPU. ~4 GB VRAM.
   Safe pour 12 GB. C'est le mode par défaut et le plus stable.

2. **GPU Streaming** — via `neo_moe.dll` : les experts sont chargés dynamiquement
   depuis le NVMe vers la VRAM. Nécessite un build custom de llama.cpp avec le
   backend `neo_moe_backend.c` lié à `-lneo_moe`.

3. **Hybride** — llama-server avec `neo_moe.dll` dans le PATH. Sans build custom,
   les experts restent sur CPU mais la lib est prête à être injectée.

---

## C FFI — llama.cpp integration

`ffi.rs` exports a C-compatible API consumed by `neo_moe_backend.c`:

| C function | Rust impl | Purpose |
|------------|-----------|---------|
| `neo_moe_stream_create()` | `ffi::neo_moe_stream_create()` | Initialise le streaming engine |
| `neo_moe_demand()` | `ffi::neo_moe_demand()` | Bloque jusqu'à ce que l'expert soit en VRAM |
| `neo_moe_demand_keep()` | `ffi::neo_moe_demand_keep()` | Idem + garde le slot verrouillé |
| `neo_moe_release_handle()` | `ffi::neo_moe_release_handle()` | Libère un handle `demand_keep` |
| `neo_moe_prefetch()` | `ffi::neo_moe_prefetch()` | Prefetch spéculatif non-bloquant |
| `neo_moe_release()` | `ffi::neo_moe_release()` | Rend le slot VRAM au pool |
| `neo_moe_model_info()` | `ffi::neo_moe_model_info()` | Retourne n_layers / n_experts |

Pour intégrer avec llama.cpp :

```bash
# 1. Build neo-moe
cargo build --release

# 2. Compiler llama.cpp avec le backend
cd llama.cpp
gcc -I../neo-moe -L../neo-moe/target/release \
    -lneo_moe -o llama-server \
    examples/server/server.cpp ... neo_moe_backend.c

# 3. Lancer
./llama-server --model model.gguf --host 0.0.0.0 --port 8001
```

---

## Performance expectations on RTX 4070 Ti + NVMe Gen4

| Metric | Expected |
|--------|----------|
| NVMe sequential read | ~7 GB/s |
| PCIe H2D bandwidth | ~15 GB/s |
| VRAM bandwidth | ~504 GB/s |
| Expert load (25 MB, cold NVMe) | ~4 ms |
| Expert load (pinned host cached) | ~1.7 ms |
| Expert load (already VRAM-resident) | ~0 µs |
| Expert GPU compute (matmul, 3.6B active) | ~5–15 ms/layer |

**Conclusion**: when the prefetch predictor is correct, NVMe I/O (~4 ms) is fully
hidden behind GPU compute (~10 ms).

---

## Files

```
neo-moe/
├── Cargo.toml           # cdylib + lib targets
├── Makefile             # build / test / install
├── build.bat            # Windows build script
├── neo_moe_backend.h    # C API header
├── neo_moe_backend.c    # ggml_backend impl (llama.cpp)
├── qwen_neo_moe.py      # Python bridge + launcher
├── .cargo/config.toml   # cargo aliases
├── examples/
│   ├── integration.rs   # Full inference loop demo
│   └── bench.rs         # NVMe→VRAM latency benchmark
└── src/
    ├── lib.rs           # Public API, ExpertId, MoeConfig
    ├── error.rs         # MoeError, Result
    ├── gguf.rs          # GGUF header parser → ExpertWeights map
    ├── pool.rs          # VRAM slot allocator (SharedPool)
    ├── predictor.rs     # TopK + Frequency expert predictors
    ├── scheduler.rs     # Prefetch queue + wait_for() API
    ├── tensor.rs        # ExpertTensorHandle (zero-copy device ptr)
    ├── stream.rs        # Core engine: dual-backend (mmap / io_uring)
    ├── io_uring.rs      # [Linux] O_DIRECT + io_uring NVMe reader
    └── ffi.rs           # extern "C" bridge for llama.cpp
```

## Status

- [x] GGUF parser — header-only scan, <100ms
- [x] VRAM pool — double-buffered, slot-based
- [x] Predictor — TopK (deterministic) + Frequency (speculative)
- [x] Scheduler — priority queue with dedup
- [x] Stream engine — mmap (Windows) + io_uring (Linux)
- [x] C FFI — cdylib with full C API
- [x] C backend — ggml_backend for llama.cpp injection
- [x] Python bridge — ctypes launcher
- [x] Launcher — menu interactif avec 4 stratégies
- [ ] Custom llama.cpp build — lien avec `-lneo_moe`
- [ ] Windows `FILE_FLAG_NO_BUFFERING` — alternative à O_DIRECT
- [ ] LRU eviction — remplacer FIFO dans le pool
