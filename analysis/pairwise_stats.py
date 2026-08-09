"""
analysis/pairwise_stats.py — Paired comparisons for the V4 campaign.

Methods (all comm_limited, R=FOV, 30 paired seeds):
  Phase 1: Frontier, Least-Visited Greedy, Chao-U (bias_cap), Entropy,
           Entropy-Frac.
  Phase 2: Frontier+Entropy, Frontier+Richness.

Research comparisons:
  Entropy      vs Entropy-Frac    (count vs fractional informational signal)
  Entropy      vs Frontier
  Entropy-Frac vs Frontier
  Entropy      vs Chao-U
  Entropy-Frac vs Chao-U
  Frontier     vs Chao-U          (baseline sanity)
  Frontier+Entropy  vs Entropy-Frac   (does geometry add to pure entropy?)
  Frontier+Richness vs Frontier+Entropy (richness vs entropy, same architecture)
  Frontier+Richness vs Frontier         (does richness beat pure frontier?)
  Frontier+Richness vs Chao-U           (richness target-selection vs global gate)
  Frontier+Richness vs Entropy-Frac

Protocol: paired Wilcoxon on steps_90 (censored at MAX_STEPS) + Holm-Bonferroni
across all comparisons. Delta = matched-pairs statistic m/n^2 (positive = A
tends to finish sooner). No narrative without p-values.

Usage:
  python analysis/pairwise_stats.py [--dir results/baseline]
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from config import (MAX_STEPS, METHOD_FRONTIER, METHOD_GREEDY, METHOD_CHAO_U,
                    METHOD_ENTROPY, METHOD_ENTROPY_FRAC,
                    METHOD_FRONTIER_ENTROPY, METHOD_FRONTIER_RICHNESS)

METHODS = [METHOD_FRONTIER, METHOD_GREEDY, METHOD_CHAO_U,
           METHOD_ENTROPY, METHOD_ENTROPY_FRAC,
           METHOD_FRONTIER_ENTROPY, METHOD_FRONTIER_RICHNESS]

# Research comparisons (A vs B: negative delta means B is faster).
PAIRS = [
    (METHOD_ENTROPY, METHOD_ENTROPY_FRAC),
    (METHOD_ENTROPY, METHOD_FRONTIER),
    (METHOD_ENTROPY_FRAC, METHOD_FRONTIER),
    (METHOD_ENTROPY, METHOD_CHAO_U),
    (METHOD_ENTROPY_FRAC, METHOD_CHAO_U),
    (METHOD_FRONTIER, METHOD_CHAO_U),
    (METHOD_FRONTIER_ENTROPY, METHOD_ENTROPY_FRAC),
    (METHOD_FRONTIER_RICHNESS, METHOD_FRONTIER_ENTROPY),
    (METHOD_FRONTIER_RICHNESS, METHOD_FRONTIER),
    (METHOD_FRONTIER_RICHNESS, METHOD_CHAO_U),
    (METHOD_FRONTIER_RICHNESS, METHOD_ENTROPY_FRAC),
]


def sanitize(method):
    import re
    return re.sub(r"[^\w\-]", "_", method)


def file_for(out_dir, method):
    variant = "bias_cap" if method == METHOD_CHAO_U else None
    safe = sanitize(method)
    if variant is not None:
        safe = f"{safe}@{variant}"
    return os.path.join(out_dir, f"raw_comm_limited__{safe}.csv")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r.get("run", 0)))
    steps = np.array([MAX_STEPS if r["steps_90"] in (None, "")
                      else float(r["steps_90"]) for r in rows])
    cens = sum(1 for r in rows if str(r.get("censored")).lower() in ("1", "true"))
    return steps, cens


def paired_stats(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if np.all(np.abs(d) < 1e-12):
        return 1.0, 0.0
    try:
        _, p = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        return 1.0, 0.0
    n = len(a)
    m = 0.0
    for i in range(n):
        m += np.sum(a[i] > b) - np.sum(a[i] < b)
    return float(p), m / (n * n)


def holm_bonferroni(ps):
    n = len(ps)
    order = np.argsort(ps)
    out = [None] * n
    for rank, idx in enumerate(order):
        out[idx] = min(1.0, ps[idx] * (n - rank))
    for i in range(n - 1, 0, -1):
        out[order[i - 1]] = min(out[order[i - 1]], out[order[i]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/baseline")
    args = ap.parse_args()

    data = {}
    missing = []
    for m in METHODS:
        r = load(file_for(args.dir, m))
        if r is None:
            missing.append(m)
            continue
        data[m] = r

    if missing:
        print("WARNING missing:", ", ".join(missing))

    print("# V4 Phase 1 — medians/IQR of steps_90 (censored=MAX_STEPS=10000)\n")
    for m in METHODS:
        if m not in data:
            continue
        s = data[m][0]
        print(f"{m:>20} med={np.median(s):.0f} "
              f"[{np.percentile(s, 25):.0f}-{np.percentile(s, 75):.0f}] "
              f"mean={np.mean(s):.0f} cens={data[m][1]}")
    print()

    print("# Paired Wilcoxon + Holm-Bonferroni\n")
    table = []
    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        p, d = paired_stats(data[a][0], data[b][0])
        table.append((a, b, p, d))

    ps = [t[2] for t in table]
    corrected = holm_bonferroni(ps)
    print(f"{'A':<18} {'B':<18} {'p':>9} {'p_holm':>9} {'delta':>7} {'sig':>5}")
    print("-" * 70)
    for (a, b, p, d), pc in zip(table, corrected):
        sig = "YES" if pc < 0.05 else "no"
        print(f"{a:<18} {b:<18} {p:>9.4f} {pc:>9.4f} {d:>+7.3f} {sig:>5}")


if __name__ == "__main__":
    main()
