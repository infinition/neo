#!/usr/bin/env python3
"""
analyze_expert_locality.py  --  Le "cerveau" de l'étude de localité MoE.

Ingère une trace réelle des experts activés (CSV: step,layer,expert) capturée
par neo_moe_trace pendant une génération, et répond à LA question décisive :

    "Avec un cache VRAM de N Go, quel hit-rate atteint-on, et donc à quelle
     vitesse peut-on faire tourner Qwen3.6-35B-A3B sur un 4070 Ti ?"

Constantes mesurées sur le GGUF réel (Qwen3.6-35B-A3B-UD-Q4_K_XL) :
    n_layer=40, n_expert=256, n_expert_used=8
    1 expert (gate+up+down) = 1.832 MiB ; pool experts = 18.32 GiB
Override possible via --unit-mb / --n-expert.

Usage:
    python analyze_expert_locality.py trace.csv
    python analyze_expert_locality.py --synthetic skewed     # démo sans trace
    python analyze_expert_locality.py --synthetic uniform
"""
import sys, argparse, random
from collections import OrderedDict, Counter

# --- constantes mesurées (voir gguf_expert_info.py) -------------------------
N_LAYER   = 40
N_EXPERT  = 256
N_USED    = 8
UNIT_MIB  = 1.832            # un expert = gate+up+down pour 1 layer
POOL_GIB  = N_LAYER * N_EXPERT * UNIT_MIB / 1024     # 18.32 GiB

# bandes passantes (RTX 4070 Ti + PCIe4 + DDR4 dual)
BW = {"VRAM": 504.0, "PCIe": 15.0, "NVMe": 7.0}      # GiB/s


def uid(layer, expert):
    return layer * N_EXPERT + expert


def lru_hitrate(accesses, cache_units):
    """Simule un cache LRU sur le flux temporel d'accès aux unités d'expert."""
    cache = OrderedDict()
    hits = 0
    for u in accesses:
        if u in cache:
            cache.move_to_end(u)
            hits += 1
        else:
            cache[u] = True
            if len(cache) > cache_units:
                cache.popitem(last=False)
    return hits / len(accesses) if accesses else 0.0


def belady_upper_bound(accesses, cache_units, sample=200_000):
    """Borne haute optionnelle (LFU global = proxy d'oracle) pour cadrer le LRU."""
    freq = Counter(accesses)
    hot = set(u for u, _ in freq.most_common(cache_units))
    hits = sum(1 for u in accesses if u in hot)
    return hits / len(accesses) if accesses else 0.0


def tps_ceiling(hitrate, tier_for_miss):
    """Plafond t/s imposé par le transfert des miss depuis `tier_for_miss`."""
    active_mib = N_LAYER * N_USED * UNIT_MIB        # 573 MiB
    miss_mib = (1.0 - hitrate) * active_mib
    if miss_mib <= 0:
        return float("inf")
    secs = (miss_mib / 1024.0) / BW[tier_for_miss]
    return 1.0 / secs


def analyze(accesses, n_steps, label):
    total = len(accesses)
    distinct = len(set(accesses))
    ws_gib = distinct * UNIT_MIB / 1024

    print(f"\n================  {label}  ================")
    print(f"trace : {n_steps} tokens décodés, {total} accès d'experts "
          f"({total/max(n_steps,1):.0f}/token, attendu {N_LAYER*N_USED})")
    print(f"working-set : {distinct} / {N_LAYER*N_EXPERT} unités distinctes "
          f"touchées  ({100*distinct/(N_LAYER*N_EXPERT):.1f} %)  = {ws_gib:.2f} GiB")
    print(f"pool total experts : {POOL_GIB:.2f} GiB   |   actif/token : "
          f"{N_LAYER*N_USED*UNIT_MIB:.0f} MiB")

    if ws_gib <= 8.0:
        print(f"  >> VERDICT : le working-set ({ws_gib:.2f} GiB) TIENT dans ~8 GiB VRAM.")
        print(f"     Après warmup -> 100% résident, ZÉRO transfert, vitesse VRAM pure.")
    else:
        print(f"  >> working-set > 8 GiB : pas de pleine résidence, le cache arbitre.")

    # skew : concentration des accès (Gini-like via top-décile)
    freq = Counter(accesses)
    ranked = sorted(freq.values(), reverse=True)
    top10 = sum(ranked[:max(1, len(ranked)//10)]) / total
    print(f"skew : le décile d'experts le plus chaud capte {100*top10:.1f} % des accès "
          f"({'forte localité' if top10 > 0.30 else 'quasi-uniforme'})")

    print(f"\n  cache VRAM   |  hit-rate(LRU)  hit-rate(LFU*)  |  plafond t/s si miss depuis…")
    print(f"  (GiB / units)|                                 |   RAM(PCIe)   NVMe")
    print(f"  " + "-"*72)
    for gib in (4, 6, 7, 8, 10):
        units = int(gib * 1024 / UNIT_MIB)
        hl = lru_hitrate(accesses, units)
        hf = belady_upper_bound(accesses, units)
        print(f"  {gib:>2} GiB /{units:>5} | {100*hl:>7.1f} %     {100*hf:>7.1f} %    |"
              f"   {tps_ceiling(hl,'PCIe'):>6.0f}    {tps_ceiling(hl,'NVMe'):>5.0f}")
    print(f"  (LFU* = top-K experts globaux = proxy d'oracle, borne haute atteignable)")
    print(f"  Rappel : ces plafonds bornent le TRANSFERT ; le compute GPU plafonne ~30-45 t/s.")


# ---------------------------------------------------------------------------
def gen_synthetic(mode, n_steps=1500, seed=1):
    rng = random.Random(seed)
    accesses = []
    # par layer : une distribution de sélection des 256 experts
    if mode == "skewed":
        # Zipf doux par layer : ~30-40 experts chauds dominent
        layer_w = []
        for L in range(N_LAYER):
            w = [1.0/((i+1)**1.1) for i in range(N_EXPERT)]
            perm = list(range(N_EXPERT)); rng.shuffle(perm)
            layer_w.append((perm, w))
    for step in range(n_steps):
        for L in range(N_LAYER):
            if mode == "uniform":
                chosen = rng.sample(range(N_EXPERT), N_USED)
            else:
                perm, w = layer_w[L]
                # tirage pondéré sans remise (approché)
                chosen = set()
                while len(chosen) < N_USED:
                    r = rng.random() * sum(w)
                    acc = 0.0
                    for i, wi in enumerate(w):
                        acc += wi
                        if acc >= r:
                            chosen.add(perm[i]); break
                chosen = list(chosen)
            for e in chosen:
                accesses.append(uid(L, e))
    return accesses, n_steps


def load_csv(path):
    accesses = []
    steps = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "#s":   # skip header/comments
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            step, layer = int(parts[0]), int(parts[1])
            steps.add(step)
            for e in parts[2:]:
                accesses.append(uid(layer, int(e)))
    return accesses, len(steps)


def summary_line(accesses, n_steps, label):
    total = len(accesses)
    distinct = len(set(accesses))
    ws = 100*distinct/(N_LAYER*N_EXPERT)
    freq = Counter(accesses)
    ranked = sorted(freq.values(), reverse=True)
    top10 = 100*sum(ranked[:max(1,len(ranked)//10)])/total
    h7 = 100*lru_hitrate(accesses, int(7*1024/UNIT_MIB))
    h8 = 100*lru_hitrate(accesses, int(8*1024/UNIT_MIB))
    print(f"  {label:<14} {n_steps:>5}    {ws:>5.1f} %   {top10:>5.1f} %    "
          f"{h7:>5.1f} %    {h8:>5.1f} %")
    return h8


def main():
    global UNIT_MIB
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", help="CSV step,layer,e0,e1,...")
    ap.add_argument("--synthetic", choices=["skewed", "uniform"])
    ap.add_argument("--summary", nargs="+", metavar="LABEL=FILE",
                    help="résumé comparatif multi-traces")
    ap.add_argument("--unit-mb", type=float, default=UNIT_MIB)
    args = ap.parse_args()

    UNIT_MIB = args.unit_mb

    if args.summary:
        print(f"\n  {'domaine':<14} {'tokens':>5}   {'ws':>6}    {'skew':>6}    "
              f"{'hit@7G':>6}   {'hit@8G':>6}")
        print("  " + "-"*64)
        for spec in args.summary:
            label, path = spec.split("=", 1)
            acc, n = load_csv(path)
            summary_line(acc, n, label)
        print("\n  (ws=working-set touché ; skew=part du décile chaud ; hit=LRU)")
        return

    if args.synthetic:
        acc, n = gen_synthetic(args.synthetic)
        analyze(acc, n, f"SYNTHÉTIQUE ({args.synthetic})")
    elif args.trace:
        acc, n = load_csv(args.trace)
        analyze(acc, n, f"TRACE RÉELLE : {args.trace}")
    else:
        print("Donne une trace CSV, ou --synthetic skewed|uniform pour une démo.")
        sys.exit(1)


if __name__ == "__main__":
    main()
