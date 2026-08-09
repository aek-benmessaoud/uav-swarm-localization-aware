"""
analysis/mh1_stats.py — MH #1: dynamic normalization of the Coverage-U score.

Question: does replacing the constant FOV_area denominator with the
traversable-cell count in the FOV window (free_count_FOV) recover the
A6_obs020 dilution failure without hurting the healthy 5%-obs regime?

Comparisons (n=40 paired, same env seeds per run index):
  A. CU-norm vs FB      — primary accuracy comparison (mean_bound_final @ T,
                          lower better; rel-reduction positive = CU-norm better).
  B. CU-norm vs CU-old  — the before/after paired A/B of the normalization
                          itself (mean_bound_final @ T).
  C. Coverage guard     — final_coverage @ T must NOT be Holm-sig worse, for
                          CU-norm vs FB and CU-norm vs CU-old.
  D. Context            — quality_auc (higher better, historically null) and
                          undetermined_final (lower better).

Protocol identical to budget_stats.py / sweep_stats.py: paired Wilcoxon
zero_method="wilcox", Holm-Bonferroni across the regimes of each comparison,
delta = matched-pairs m/n^2.

Usage: python analysis/mh1_stats.py
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from config import METHOD_FRONTIER_BOUNDED, METHOD_COVERAGE_U, \
    METHOD_COVERAGE_U_NORM

REGIMES = ["A6_obs020", "A6_obs005"]
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
    """Median relative reduction of A vs B, ratio-of-medians convention used
    by the paper's E4-CONFIRM tables: (med(B) - med(A)) / med(B) * 100.
    Positive = A better (lower is better on the metric)."""
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
                  METHOD_COVERAGE_U_NORM):
            r = load(d, m)
            if r is None or not r:
                print(f"WARNING incomplete regime {rg}, method {m}: "
                      f"{0 if r is None else len(r)} runs")
                r = []
            rows[m] = r
        if rows[METHOD_FRONTIER_BOUNDED] and rows[METHOD_COVERAGE_U] \
                and rows[METHOD_COVERAGE_U_NORM]:
            data[rg] = rows

    print("# MH #1 — dynamic normalization of the Coverage-U score "
          "(n=40 paired, mean_bound_final @ T, lower better)")
    print("# CU-norm = Coverage-U-norm (free-count denominator); "
          "CU = Coverage-U (constant FOV_area); FB = Frontier-Bounded\n")

    # ---------- A. CU-norm vs FB (accuracy) ----------
    print("## A. mean_bound_final @ T: CU-norm vs FB "
          "(rel-red% positive = CU-norm better)")
    rows = []
    for rg in REGIMES:
        a = data[rg][METHOD_COVERAGE_U_NORM]
        b = data[rg][METHOD_FRONTIER_BOUNDED]
        va = metric_vector(a, "mean_bound_final", None)
        vb = metric_vector(b, "mean_bound_final", None)
        p, d = paired_stats(va, vb)
        rows.append((rg, np.nanmedian(va), np.nanmedian(vb),
                     rel_red(va, vb), p, d))
    corr = holm_bonferroni([r[4] for r in rows])
    print(f"  {'regime':<12} {'med_FB':>9} {'med_CUnorm':>10} "
          f"{'rel-red%':>9} {'p':>8} {'p_holm':>8} {'delta':>7} {'sig':>5}")
    print("  " + "-" * 70)
    for (rg, ma, mb, rel, p, d), pc in zip(rows, corr):
        print(f"  {rg:<12} {mb:>9.4f} {ma:>10.4f} {rel:>+9.1f} {p:>8.4f} "
              f"{pc:>8.4f} {d:>+7.3f} {'YES' if pc < 0.05 else 'no':>5}")

    # ---------- B. CU-norm vs CU-old (accuracy) ----------
    print("\n## B. mean_bound_final @ T: CU-norm vs CU-old "
          "(rel-change% positive = normalization better)")
    rows2 = []
    for rg in REGIMES:
        a = data[rg][METHOD_COVERAGE_U_NORM]
        b = data[rg][METHOD_COVERAGE_U]
        va = metric_vector(a, "mean_bound_final", None)
        vb = metric_vector(b, "mean_bound_final", None)
        p, d = paired_stats(va, vb)
        rows2.append((rg, np.nanmedian(va), np.nanmedian(vb),
                      rel_red(va, vb), p, d))
    corr2 = holm_bonferroni([r[4] for r in rows2])
    print(f"  {'regime':<12} {'med_CU':>9} {'med_CUnorm':>10} "
          f"{'rel-change%':>11} {'p':>8} {'p_holm':>8} {'delta':>7} {'sig':>5}")
    print("  " + "-" * 72)
    for (rg, ma, mb, rel, p, d), pc in zip(rows2, corr2):
        print(f"  {rg:<12} {mb:>9.4f} {ma:>10.4f} {rel:>+11.1f} {p:>8.4f} "
              f"{pc:>8.4f} {d:>+7.3f} {'YES' if pc < 0.05 else 'no':>5}")

    # ---------- C. coverage guard ----------
    print("\n## C. GUARD: final_coverage @ T (regression = Holm-sig lower)")
    for meth_b, label in ((METHOD_FRONTIER_BOUNDED, "FB"),
                          (METHOD_COVERAGE_U, "CU")):
        grow = []
        for rg in REGIMES:
            a = data[rg][METHOD_COVERAGE_U_NORM]
            b = data[rg][meth_b]
            va = metric_vector(a, "final_coverage", None)
            vb = metric_vector(b, "final_coverage", None)
            p, _ = paired_stats(va, vb)
            grow.append((rg, np.nanmedian(va), np.nanmedian(vb), p))
        gcorr = holm_bonferroni([r[3] for r in grow])
        print(f"  CU-norm vs {label}:")
        print(f"    {'regime':<12} {'med_' + label:>10} {'med_CUnorm':>11} "
              f"{'p':>8} {'p_holm':>8} {'regression?':>12}")
        for (rg, ma, mb, p), pc in zip(grow, gcorr):
            is_r = pc < 0.05 and ma < mb
            print(f"    {rg:<12} {mb:>10.1f} {ma:>11.1f} {p:>8.4f} "
                  f"{pc:>8.4f} {'YES' if is_r else 'no':>12}")

    # ---------- D. context ----------
    print("\n## D. Context metrics (CU-norm vs FB)")
    for metric, lower in (("quality_auc", False), ("undetermined_final", True)):
        print(f"  {metric} ({'lower' if lower else 'higher'} better):")
        print(f"    {'regime':<12} {'med_FB':>9} {'med_CUnorm':>10} "
              f"{'gain%':>8}")
        for rg in REGIMES:
            a = data[rg][METHOD_COVERAGE_U_NORM]
            b = data[rg][METHOD_FRONTIER_BOUNDED]
            va = metric_vector(a, metric, None)
            vb = metric_vector(b, metric, None)
            if np.all(np.isnan(va)) or np.all(np.isnan(vb)):
                continue
            g = rel_red(va, vb) if lower else \
                float((np.nanmedian(va) - np.nanmedian(vb))
                      / np.where(np.nanmedian(vb) == 0, np.nan,
                                 np.nanmedian(vb)) * 100.0)
            print(f"    {rg:<12} {np.nanmedian(vb):>9.4f} "
                  f"{np.nanmedian(va):>10.4f} {g:>+8.1f}")

    # ---------- E. context: CU-old vs FB (same metric) ----------
    print("\n## E. Context: CU-old vs FB on mean_bound_final @ T "
          "(the pre-normalization effect size)")
    print(f"    {'regime':<12} {'med_FB':>9} {'med_CU':>9} {'rel-red%':>9} "
          f"{'p':>8} {'p_holm':>8} {'sig':>5}")
    erows = []
    for rg in REGIMES:
        a = data[rg][METHOD_COVERAGE_U]
        b = data[rg][METHOD_FRONTIER_BOUNDED]
        va = metric_vector(a, "mean_bound_final", None)
        vb = metric_vector(b, "mean_bound_final", None)
        p, d = paired_stats(va, vb)
        erows.append((rg, np.nanmedian(va), np.nanmedian(vb),
                      rel_red(va, vb), p, d))
    ecorr = holm_bonferroni([r[4] for r in erows])
    for (rg, ma, mb, rel, p, d), pc in zip(erows, ecorr):
        print(f"    {rg:<12} {mb:>9.4f} {ma:>9.4f} {rel:>+9.1f} {p:>8.4f} "
              f"{pc:>8.4f} {'YES' if pc < 0.05 else 'no':>5}")

    print("\n# n = 40 pairs per regime; paired Wilcoxon (zero_method='wilcox'); "
          "Holm-Bonferroni across the 2 regimes per comparison family.")


if __name__ == "__main__":
    main()
