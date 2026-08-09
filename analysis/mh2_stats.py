"""
analysis/mh2_stats.py — MH #2: GDOP/FIM baseline policy statistics.

Question: how does a classical GDOP/FIM baseline (unit-weight direction-only
GDOP, greedy simulated bound-reduction in the SAME Frontier-Bounded frame as
Coverage-U) perform against the proposed config-count family?

Comparisons (n=40 paired, same env seeds per run index): GDOP vs FB, GDOP vs
CU, GDOP vs RA on
  - mean_bound_final @ T (lower better; rel-red positive = GDOP better),
  - final_coverage @ T (guard: no Holm-sig regression),
  - quality_auc / undetermined_final (context).

Protocol identical to budget_stats.py / mh1_stats.py: paired Wilcoxon
zero_method="wilcox", Holm-Bonferroni across the 2 regimes per comparison
family, delta = matched-pairs m/n^2, rel-red by ratio-of-medians.

Usage: python analysis/mh2_stats.py
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from config import (METHOD_FRONTIER_BOUNDED, METHOD_COVERAGE_U,
                    METHOD_FRONTIER_RICHNESS_ANGULAR, METHOD_GDOP)

REGIMES = ["A3_obs005", "A6_obs005"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")


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


def rel_red(va, vb):
    """(med(B) - med(A)) / med(B) * 100; positive = A better (lower better)."""
    ma = float(np.nanmedian(va))
    mb = float(np.nanmedian(vb))
    if mb == 0 or np.isnan(mb):
        return float("nan")
    return (mb - ma) / mb * 100.0


def main():
    data = {}
    for rg in REGIMES:
        d = os.path.join(RESULTS_DIR, f"budget_{rg}")
        rows = {}
        for m in (METHOD_FRONTIER_BOUNDED, METHOD_COVERAGE_U,
                  METHOD_FRONTIER_RICHNESS_ANGULAR, METHOD_GDOP):
            r = load(d, m)
            if r is None or not r:
                print(f"WARNING incomplete regime {rg}, method {m}: "
                      f"{0 if r is None else len(r)} runs")
                r = []
            rows[m] = r
        if all(rows[m] for m in rows):
            data[rg] = rows

    baselines = [
        (METHOD_FRONTIER_BOUNDED, "FB"),
        (METHOD_COVERAGE_U, "CU"),
        (METHOD_FRONTIER_RICHNESS_ANGULAR, "RA"),
    ]

    print("# MH #2 — GDOP/FIM baseline vs FB / CU / RA "
          "(n=40 paired, mean_bound_final @ T, lower better)\n")

    # ---------- A. accuracy ----------
    print("## A. mean_bound_final @ T: GDOP vs baseline "
          "(rel-red% positive = GDOP better)")
    for meth, label in baselines:
        rows = []
        for rg in REGIMES:
            a = data[rg][METHOD_GDOP]
            b = data[rg][meth]
            va = metric_vector(a, "mean_bound_final", None)
            vb = metric_vector(b, "mean_bound_final", None)
            p, d = paired_stats(va, vb)
            rows.append((rg, np.nanmedian(va), np.nanmedian(vb),
                         rel_red(va, vb), p, d))
        corr = holm_bonferroni([r[4] for r in rows])
        print(f"\n  GDOP vs {label}:")
        print(f"    {'regime':<12} {'med_' + label:>10} {'med_GDOP':>10} "
              f"{'rel-red%':>9} {'p':>8} {'p_holm':>8} {'delta':>7} {'sig':>5}")
        for (rg, ma, mb, rel, p, d), pc in zip(rows, corr):
            print(f"    {rg:<12} {mb:>10.4f} {ma:>10.4f} {rel:>+9.1f} "
                  f"{p:>8.4f} {pc:>8.4f} {d:>+7.3f} "
                  f"{'YES' if pc < 0.05 else 'no':>5}")

    # ---------- B. coverage guard ----------
    print("\n## B. GUARD: final_coverage @ T (regression = Holm-sig lower)")
    for meth, label in baselines:
        grow = []
        for rg in REGIMES:
            a = data[rg][METHOD_GDOP]
            b = data[rg][meth]
            va = metric_vector(a, "final_coverage", None)
            vb = metric_vector(b, "final_coverage", None)
            p, _ = paired_stats(va, vb)
            grow.append((rg, np.nanmedian(va), np.nanmedian(vb), p))
        gcorr = holm_bonferroni([r[3] for r in grow])
        print(f"  GDOP vs {label}:")
        print(f"    {'regime':<12} {'med_' + label:>10} {'med_GDOP':>10} "
              f"{'p':>8} {'p_holm':>8} {'regression?':>12}")
        for (rg, ma, mb, p), pc in zip(grow, gcorr):
            is_r = pc < 0.05 and ma < mb
            print(f"    {rg:<12} {mb:>10.1f} {ma:>10.1f} {p:>8.4f} "
                  f"{pc:>8.4f} {'YES' if is_r else 'no':>12}")

    # ---------- C. context ----------
    print("\n## C. Context (GDOP vs baseline)")
    for metric, lower in (("quality_auc", False), ("undetermined_final", True)):
        for meth, label in baselines:
            print(f"  {metric} GDOP vs {label} "
                  f"({'lower' if lower else 'higher'} better):")
            for rg in REGIMES:
                a = data[rg][METHOD_GDOP]
                b = data[rg][meth]
                va = metric_vector(a, metric, None)
                vb = metric_vector(b, metric, None)
                if np.all(np.isnan(va)) or np.all(np.isnan(vb)):
                    continue
                if lower:
                    g = rel_red(va, vb)
                else:
                    mb = float(np.nanmedian(vb))
                    g = (float(np.nanmedian(va)) - mb) / mb * 100.0 \
                        if mb else float("nan")
                print(f"    {rg:<12} med_b={np.nanmedian(vb):>8.4f} "
                      f"med_gdop={np.nanmedian(va):>8.4f} gain%={g:>+7.1f}")

    print("\n# n = 40 pairs per regime; paired Wilcoxon (zero_method='wilcox'); "
          "Holm-Bonferroni across the 2 regimes per comparison family.")


if __name__ == "__main__":
    main()
