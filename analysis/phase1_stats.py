"""
analysis/phase1_stats.py — Phase-1 A/B statistics (Project08).

Compares the three Phase-1 methods (Random, Frontier-Bounded,
Richness-Angular) on the LOCALIZATION-quality metrics recorded by
run_phase1.py (quality_auc, time_to_quality, quality_final), using the
paired design (same env seed per run index across methods).

Protocol (locked):
  - Paired Wilcoxon (zero_method="wilcox") per (metric, pair).
  - Holm-Bonferroni across ALL tests in the run (global).
  - Delta = matched-pairs statistic m/n^2 (positive = A better).
  - Median relative gain of A over B per run (sign flipped for metrics
    where lower is better).

Usage:
  python analysis/phase1_stats.py --dir results/phase1_S1_fov5 --max-steps 4500
  python analysis/phase1_stats.py --dir results/phase1_S1prime_fov3 --max-steps 4500
"""

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from config import (METHOD_RANDOM, METHOD_FRONTIER_BOUNDED,
                    METHOD_FRONTIER_RICHNESS_ANGULAR)

METHODS = [METHOD_RANDOM, METHOD_FRONTIER_BOUNDED,
           METHOD_FRONTIER_RICHNESS_ANGULAR]

HIGHER_BETTER = {
    "quality_auc": True,
    "quality_final": True,
    "time_to_quality": False,
}

# Research comparisons (A vs B). The key test is Richness-Angular vs
# Frontier-Bounded (does the localization signal beat the geometric frame?).
PAIRS = [
    (METHOD_FRONTIER_RICHNESS_ANGULAR, METHOD_FRONTIER_BOUNDED),
    (METHOD_FRONTIER_RICHNESS_ANGULAR, METHOD_RANDOM),
    (METHOD_FRONTIER_BOUNDED, METHOD_RANDOM),
]


def sanitize(method):
    return re.sub(r"[^\w\-]", "_", method)


def load(dir_path, method):
    path = os.path.join(dir_path, f"raw_comm_limited__{sanitize(method)}.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r.get("run", 0)))
    return rows


def _num(r, key, cap=None):
    v = r.get(key)
    if v in (None, ""):
        return cap
    try:
        return float(v)
    except ValueError:
        return cap


def metric_vector(rows, metric, max_steps):
    return np.array([_num(r, metric, max_steps) for r in rows], dtype=float)


def paired_stats(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float("nan"), 0.0
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
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-steps", type=int, default=4500,
                    help="Censoring cap for time_to_quality.")
    args = ap.parse_args()

    data = {}
    missing = []
    for m in METHODS:
        rows = load(args.dir, m)
        if rows is None or not rows:
            missing.append(m)
            continue
        data[m] = rows

    if missing:
        print("WARNING missing:", ", ".join(missing))

    print(f"# Phase-1 A/B - {args.dir}")
    print(f"# runs per method: {len(next(iter(data.values()))) if data else 0}, "
          f"censoring cap = {args.max_steps}\n")

    print("# Per-method medians [IQR]\n")
    for m in METHODS:
        if m not in data:
            continue
        parts = []
        for metric in HIGHER_BETTER:
            v = metric_vector(data[m], metric, args.max_steps)
            parts.append(f"{metric}={np.median(v):.3f} "
                         f"[{np.percentile(v, 25):.3f}-"
                         f"{np.percentile(v, 75):.3f}]")
        und = np.median([_num(r, "undetermined_final", 0.0)
                         for r in data[m]])
        parts.append(f"undetermined_final={und:.4f}")
        print(f"  {m:<18} {'  '.join(parts)}")
    print()

    tests = []
    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        for metric, higher in HIGHER_BETTER.items():
            va = metric_vector(data[a], metric, args.max_steps)
            vb = metric_vector(data[b], metric, args.max_steps)
            p, d = paired_stats(va, vb)
            med_a, med_b = np.median(va), np.median(vb)
            gain = np.nanmedian((va - vb) / np.where(vb == 0, np.nan, vb))
            if not higher:
                gain = -gain
            tests.append((a, b, metric, med_a, med_b, gain * 100.0, p, d))

    ps = [t[6] for t in tests]
    corrected = holm_bonferroni(ps)

    print("# Paired Wilcoxon + Holm-Bonferroni (global)\n")
    print(f"{'A':<16} {'B':<16} {'metric':<14} {'med_A':>7} "
          f"{'med_B':>7} {'gain%':>7} {'p':>8} {'p_holm':>8} "
          f"{'delta':>7} {'sig':>5}")
    print("-" * 100)
    for (a, b, metric, med_a, med_b, gain, p, d), pc in zip(tests, corrected):
        sig = "YES" if pc < 0.05 else "no"
        print(f"{a:<16} {b:<16} {metric:<14} {med_a:>7.3f} {med_b:>7.3f} "
              f"{gain:>+7.1f} {p:>8.4f} {pc:>8.4f} {d:>+7.3f} {sig:>5}")


if __name__ == "__main__":
    main()
