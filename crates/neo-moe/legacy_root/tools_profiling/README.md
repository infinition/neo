# Étude de localité des experts MoE — Qwen3.6-35B-A3B

Objectif : décider **factuellement** si un cache VRAM d'experts (+ streaming depuis
la RAM) peut faire tourner Qwen3.6-35B-A3B vite sur un RTX 4070 Ti — ou si la
physique le borne au niveau actuel. Tout dépend d'**une** inconnue mesurable :
le routage des experts est-il *concentré* (peu d'experts chauds → cache gagne)
ou *uniforme* (→ le cache n'aide quasi pas).

## Géométrie réelle du modèle (mesurée via `gguf_expert_info.py`)

| | |
|---|---|
| arch | qwen35moe, 40 layers |
| experts | 256/layer, 8 actifs (sparsité 1/32) |
| 1 expert (gate+up+down) | 1.83 MiB |
| pool experts total | 18.32 GiB |
| non-experts (résident VRAM) | 2.49 GiB |
| actif / token | 573 MiB |

## Pipeline

1. **Capturer la vraie trace** (routage = identique CPU/GPU, donc build CPU-only suffit) :
   ```
   build_trace_cpu.bat                 # produit bld_cpu_trace\bin\llama-neo-moe-trace.exe
   ```
   ```
   set NEO_TRACE_OUT=trace_qwen.csv
   llama-neo-moe-trace.exe -m ..\Qwen_3.6_35b\.models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ^
       -ngl 0 -c 4096 -n 800 -p "Explain how a turbojet engine works, step by step."
   ```
   Sortie CSV : `step,layer,e0,...,e7` (un row par token×layer).
   Lance-le sur 2-3 prompts de domaines différents (code, prose, maths) — la
   localité dépend du domaine.

2. **Analyser** :
   ```
   python analyze_expert_locality.py trace_qwen.csv
   ```
   Donne : working-set (Go), hit-rate LRU/LFU vs taille de cache, skew, et le
   plafond t/s selon le tier de service des miss (RAM/PCIe vs NVMe).

## Lecture du verdict

- **working-set ≤ 8 GiB** → tout devient résident après warmup : vitesse VRAM pure,
  streaming inutile (fix = juste FIFO→LRU + dimensionner le pool).
- **hit-rate @8GiB élevé (>75 %)** → miss servis depuis la RAM (15 GB/s) cachés
  derrière le compute → ~30-45 t/s atteignable. **Ton archi a un avenir, redirigée
  RAM→VRAM (pas NVMe).**
- **hit-rate @8GiB faible (~40 %) + skew quasi-uniforme** → le cache n'arbitre rien ;
  sur ce modèle, pas de gain au-dessus de l'offload natif `--n-cpu-moe`. Le levier
  bascule alors vers : quant non-uniforme (Q2 cold / Q4 hot) ou speculative decoding.

## Démo sans trace (pour voir la forme du résultat)

```
python analyze_expert_locality.py --synthetic skewed
python analyze_expert_locality.py --synthetic uniform
```
