"""
analysis/ra_sweep_budget_range_stats.py — RA through the budget (C) and
fusion-range (D) sweeps.

The paper's sweeps C (budget-frac, Table 13) and D (comm-range, Table 14)
report only Coverage-U vs Frontier-Bounded. RA was ALREADY run in the same
campaigns (paired runs/seeds, 40 rows per cell) but never analyzed. RA is now
the accuracy-optimal member of the config-count family (MH #2/MH #3), so its
robustness along these two axes is a reporting gap, not a new experiment.

Protocol identical to sweep_stats.py / ra_sweep_stats.py: paired Wilcoxon
(zero_method="wilcox"), Holm-Bonferroni across the 2 regimes per variant
family, delta = m/n^2, rel-red by median relative change (lower_better).

Reports, per sweep and variant, per regime:
  - RA vs FB  (does RA's edge over the movement frame survive?)
  - RA vs CU  (does RA's advantage over Coverage-U survive?)
plus the coverage guard (RA vs FB) and quality_auc context.

Usage: python analysis/ra_sweep_budget_range_stats.py
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

REGIMES = ["A3_obs005", "A6_obs005"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

PRIMARY = "mean_bound_final"
GUARD = "final_coverage"
CTX = "quality_auc"

# label -> (variant -> human label, variant list)
SWEEPS = {
    "C budget-frac":   ({"b03": "0.3", "b05": "0.5", "b09": "0.9"},
                        ["b03", "b05", "b09"]),
    "D comm-range":    ({"r25": "2.5", "r125": "1.25"},
                        ["r25", "r125"]),
    "E bearing-noise": ({"be2": "2.0 deg"}, ["be2"]),
}


def sanitize(method):
    return re.sub(r"[^\w\-]", "_", method)


def load(dir_path, method, variant=None):
    fname = f"raw_comm_limited__{sanitize(method)}.csv"
    if variant:
        fname = f"raw_comm_limited__{sanitize(method)}__{variant}.csv"
    path = os.path.join(dir_path, fname)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r.get("run", 0)))
    return rows


def _num(r, key):
    v = r.get(key)
    if v in (None, ""):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def vec(rows, key):
    return np.array([_num(r, key) for r in rows or []], dtype=float)


def paired_stats(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a) & np.isfinite(b)], b[np.isfinite(a) & np.isfinite(b)]
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


def fmt_p(p):
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def report(variants, labels):
    print("# RA through the sweep "
          "(mean_bound_final @ T, lower better)\n")
    for variant in variants:
        print(f"{'=' * 78}\n## variant = {labels[variant]}\n{'=' * 78}")
        for a_name, b_name, title in (
                ("Richness-Angular", "Frontier-Bounded", "RA vs FB"),
                ("Richness-Angular", "Coverage-U", "RA vs CU")):
            rows = []
            for rg in REGIMES:
                d = os.path.join(RESULTS_DIR, f"budget_{rg}__{variant}")
                va = vec(load(d, a_name, variant), PRIMARY)
                vb = vec(load(d, b_name, variant), PRIMARY)
                if len(va) == 0 or len(vb) == 0:
                    print(f"  WARNING {d}: missing data")
                    continue
                p, dlt = paired_stats(va, vb)
                rows.append((rg, np.nanmedian(vb), np.nanmedian(va),
                             rel_red(va, vb), p, dlt))
            if not rows:
                continue
            corr = holm_bonferroni([r[4] for r in rows])
            print(f"\n  {title}  (rel-red% positive = RA better)")
            print(f"    {'regime':<12} {'med_' + b_name:>12} {'med_RA':>10} "
                  f"{'rel-red%':>9} {'p':>9} {'p_holm':>9} {'delta':>7} "
                  f"{'sig':>4}")
            for (rg, mb, ma, g, p, dlt), pc in zip(rows, corr):
                print(f"    {rg:<12} {mb:>12.4f} {ma:>10.4f} {g:>+9.1f} "
                      f"{fmt_p(p):>9} {fmt_p(pc):>9} {dlt:>+7.3f} "
                      f"{'YES' if pc < 0.05 else 'no':>4}")

        # coverage guard: RA vs FB
        gprint = []
        for rg in REGIMES:
            d = os.path.join(RESULTS_DIR, f"budget_{rg}__{variant}")
            va = vec(load(d, "Richness-Angular", variant), GUARD)
            vb = vec(load(d, "Frontier-Bounded", variant), GUARD)
            if len(va) == 0 or len(vb) == 0:
                continue
            p, _ = paired_stats(va, vb)
            gprint.append((rg, np.nanmedian(vb), np.nanmedian(va),
                           rel_red(vb, va), p))
        gcorr = holm_bonferroni([r[4] for r in gprint])
        print(f"\n  GUARD {GUARD} RA vs FB (higher better)")
        print(f"    {'regime':<12} {'med_FB':>12} {'med_RA':>10} "
              f"{'gain%':>9} {'p':>9} {'p_holm':>9} {'regress':>8}")
        for (rg, mb, ma, g, p), pc in zip(gprint, gcorr):
            is_r = pc < 0.05 and ma < mb
            print(f"    {rg:<12} {mb:>12.1f} {ma:>10.1f} {g:>+9.1f} "
                  f"{fmt_p(p):>9} {fmt_p(pc):>9} {'YES' if is_r else 'no':>8}")

        # context
        for rg in REGIMES:
            d = os.path.join(RESULTS_DIR, f"budget_{rg}__{variant}")
            va = vec(load(d, "Richness-Angular", variant), CTX)
            vb = vec(load(d, "Coverage-U", variant), CTX)
            if len(va) and len(vb):
                print(f"    [ctx] {rg} quality_auc med RA={np.nanmedian(va):.4f}"
                      f" CU={np.nanmedian(vb):.4f}")


def main():
    for name, (labels, variants) in SWEEPS.items():
        print(f"{'#' * 78}\n# SWEEP {name}\n{'#' * 78}")
        report(variants, labels)


if __name__ == "__main__":
    main()
