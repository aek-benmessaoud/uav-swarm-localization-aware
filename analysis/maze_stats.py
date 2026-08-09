"""
analysis/maze_stats.py — Maze-campaign A/B statistics (post-submission).

Campaign: A6 agents, 40 paired maze maps (topology="maze", Kruskal perfect
maze, LOS-occluded FOV), budget T = 0.7 x FB median steps_90 (measured by
probe_budget_maze.py = 1323 steps), methods Frontier-Bounded (control),
Coverage-U, Richness-Angular. Same protocol family as budget_stats.

Protocol:
  - Paired Wilcoxon (zero_method="wilcox"), Holm-Bonferroni across the 3
    pairwise comparisons.
  - Delta = matched-pairs statistic m/n^2 (positive = A better).
  - Median relative gain of A over B (sign flipped for lower-is-better).
  - Primary: quality_auc @ T (integrated localization quality under the
    budget; higher better). Guard: final_coverage @ T must NOT be
    significantly worse (Pareto-clean check).
  - Directional hypothesis (MAexp-coherent, NOT cross-validation):
    Richness-Angular >= Coverage-U in the maze topology.

Usage:
  python analysis/maze_stats.py [--dir results/budget_A6_maze__maze]
"""

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from config import (METHOD_FRONTIER_BOUNDED, METHOD_COVERAGE_U,
                    METHOD_FRONTIER_RICHNESS_ANGULAR)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

PRIMARY = "quality_auc"
GUARD = "final_coverage"
SECONDARY = ["coverage_auc", "mean_bound_final", "undetermined_final",
             "steps_dual", "time_to_quality"]
LOWER_BETTER = {"mean_bound_final", "undetermined_final", "steps_dual",
                "time_to_quality"}

# Research comparisons (A vs B; positive delta = A better). RA >= CU is the
# directional MAexp-coherent hypothesis in the maze topology.
PAIRS = [
    (METHOD_COVERAGE_U, METHOD_FRONTIER_BOUNDED),
    (METHOD_FRONTIER_RICHNESS_ANGULAR, METHOD_FRONTIER_BOUNDED),
    (METHOD_FRONTIER_RICHNESS_ANGULAR, METHOD_COVERAGE_U),
]

METHODS = [METHOD_FRONTIER_BOUNDED, METHOD_COVERAGE_U,
           METHOD_FRONTIER_RICHNESS_ANGULAR]


def sanitize(method):
    return re.sub(r"[^\w\-]", "_", method)


def load(dir_path, method, tag):
    path = os.path.join(dir_path,
                        f"raw_comm_limited__{sanitize(method)}{tag}.csv")
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


def metric_vector(rows, metric, cap):
    return np.array([_num(r, metric, cap) for r in rows], dtype=float)


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


def gain_pct(va, vb, lower_better):
    """Median relative gain of A over B; positive = A better.

    Denominator is ALWAYS the baseline (B = vb), matching the paper's
    rel-red convention (g = 100*(vb - va)/vb, e.g. undetermined_table /
    e3_table in paper_build.py). lower_better=True -> num = vb - va
    (reduction in A relative to B); higher_better -> num = va - vb.
    """
    if lower_better:
        num = vb - va
    else:
        num = va - vb
    return float(np.nanmedian(num / np.where(vb == 0, np.nan, vb)) * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(RESULTS_DIR,
                                                  "budget_A6_maze__maze"))
    ap.add_argument("--tag", default="__maze",
                    help="filename suffix (default __maze)")
    ap.add_argument("--label", default="maze (perfect, 0 loops)",
                    help="human label for the report header")
    ap.add_argument("--budget", type=int, default=1323,
                    help="budget T in the report header (cosmetic)")
    args = ap.parse_args()

    data = {}
    for m in METHODS:
        r = load(args.dir, m, args.tag)
        if r is None or not r:
            print(f"FATAL: no rows for {m} in {args.dir}")
            return 2
        data[m] = r
    n = len(data[METHOD_FRONTIER_BOUNDED])
    print(f"# Maze campaign — {n} paired runs, budget T={args.budget} "
          f"(0.7 x FB median steps_90), topology={args.label}\n")

    print(f"# PRIMARY: {PRIMARY} @ budget (higher better)\n")
    prow = []
    for a, b in PAIRS:
        va = metric_vector(data[a], PRIMARY, -1.0)
        vb = metric_vector(data[b], PRIMARY, -1.0)
        p, d = paired_stats(va, vb)
        g = gain_pct(va, vb, lower_better=False)
        prow.append((a, b, np.median(va), np.median(vb), g, p, d))
    pcorr = holm_bonferroni([r[5] for r in prow])
    print(f"{'A':<18} {'B':<18} {'med_A':>8} {'med_B':>8} {'gain%':>7} "
          f"{'p':>8} {'p_holm':>8} {'delta':>7} {'sig':>5}")
    print("-" * 78)
    for (a, b, ma, mb, g, p, d), pc in zip(prow, pcorr):
        sig = "YES" if pc < 0.05 else "no"
        print(f"{a:<18} {b:<18} {ma:>8.3f} {mb:>8.3f} {g:>+7.1f} "
              f"{p:>8.4f} {pc:>8.4f} {d:>+7.3f} {sig:>5}")

    print(f"\n# GUARD: {GUARD} @ budget (candidate must not be significantly "
          f"worse)")
    grow = []
    for a, b in PAIRS:
        va = metric_vector(data[a], GUARD, -1.0)
        vb = metric_vector(data[b], GUARD, -1.0)
        p, _ = paired_stats(va, vb)
        grow.append((a, b, np.median(va), np.median(vb), p))
    gcorr = holm_bonferroni([r[4] for r in grow])
    print(f"{'A':<18} {'B':<18} {'med_A':>8} {'med_B':>8} {'p':>8} "
          f"{'p_holm':>8} {'regression?':>12}")
    print("-" * 70)
    regress = False
    for (a, b, ma, mb, p), pc in zip(grow, gcorr):
        is_r = pc < 0.05 and ma < mb
        regress |= is_r
        print(f"{a:<18} {b:<18} {ma:>8.1f} {mb:>8.1f} {p:>8.4f} {pc:>8.4f} "
              f"{'YES' if is_r else 'no':>12}")

    print(f"\n# Secondary metrics (gain% = A better)")
    print(f"{'A':<18} {'metric':<18} {'med_A':>8} {'med_B':>8} {'gain%':>7}")
    print("-" * 60)
    for a, b in PAIRS:
        for metric in SECONDARY:
            lb = metric in LOWER_BETTER
            va = metric_vector(data[a], metric, None)
            vb = metric_vector(data[b], metric, None)
            if np.all(np.isnan(va)) or np.all(np.isnan(vb)):
                continue
            g = gain_pct(va, vb, lb)
            print(f"{a:<18} {metric:<18} {np.nanmedian(va):>8.3f} "
                  f"{np.nanmedian(vb):>8.3f} {g:>+7.1f}")

    print("\n# Directional check (MAexp-coherent, NOT cross-validation): "
          "RA >= CU on quality_auc")
    va = metric_vector(data[METHOD_FRONTIER_RICHNESS_ANGULAR], PRIMARY, -1.0)
    vb = metric_vector(data[METHOD_COVERAGE_U], PRIMARY, -1.0)
    p, d = paired_stats(va, vb)
    pc = holm_bonferroni([p])[0]
    print(f"  quality_auc med RA={np.median(va):.3f} vs CU={np.median(vb):.3f}, "
          f"p={p:.4f} (Holm {pc:.4f}), delta={d:+.3f} -> "
          f"{'coherent (RA>=CU)' if pc < 0.05 and np.median(va) >= np.median(vb) else 'not coherent'}")


if __name__ == "__main__":
    main()
