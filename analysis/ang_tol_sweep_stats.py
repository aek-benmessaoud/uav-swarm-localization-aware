"""
analysis/ang_tol_sweep_stats.py — ANG_TOL sensitivity sweep statistics.

The config-count family re-run under ANG_TOL_DEG in {5, 10, 20, 30} (dirs
budget_{regime}__at{tol}) on the flagship A3/A6 5%-obstacle regimes, n = 40
paired; ANG_TOL = 15 deg and the Frontier-Bounded (FB) control are reused from
the confirmed results/budget_{A3,A6}_obs005 (FB does not read angular
configurations, so it is ANG_TOL-invariant).

Protocol identical to the confirmed sweeps: paired Wilcoxon
(zero_method="wilcox"), Holm-Bonferroni across the ANG_TOL levels per family,
rel-red by median relative change (lower_better).

Reports, per regime and ANG_TOL level:
  - RA vs FB  (does RA's edge over the movement frame survive the threshold?)
  - CU vs FB  (companion)
  - RA vs CU  (direction RA >= CU, the confirmed claim)
plus the coverage guard (RA vs FB) and quality_auc context.

Usage: python analysis/ang_tol_sweep_stats.py
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

REGIMES = ["A3_obs005", "A6_obs005"]
TOL_LEVELS = [5.0, 10.0, 15.0, 20.0, 30.0]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

PRIMARY = "mean_bound_final"
GUARD = "final_coverage"
CTX = "quality_auc"


def sanitize(method):
    return re.sub(r"[^\w\-]", "_", method)


def load(dir_path, method, tag=None):
    fname = f"raw_comm_limited__{sanitize(method)}"
    if tag:
        fname += f"__{tag}"
    fname += ".csv"
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


def aligned(rows_a, rows_b, key):
    """Return (va, vb) aligned by run index (paired seeds)."""
    ma = {int(r.get("run", -1)): _num(r, key) for r in rows_a or []}
    mb = {int(r.get("run", -1)): _num(r, key) for r in rows_b or []}
    runs = sorted(set(ma) & set(mb))
    va = np.array([ma[r] for r in runs], dtype=float)
    vb = np.array([mb[r] for r in runs], dtype=float)
    return va, vb


def paired_stats(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    fin = np.isfinite(a) & np.isfinite(b)
    a, b = a[fin], b[fin]
    if len(a) == 0:
        return float("nan"), 0.0
    if np.all(np.abs(a - b) < 1e-12):
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


def rel_of_medians(ma, mb, lower_better):
    # Ratio of medians, matching the paper's rel-b convention (tab:base).
    if lower_better:
        g = 100.0 * (mb - ma) / mb
    else:
        g = 100.0 * (ma - mb) / mb
    return float(g) if mb else 0.0


def fmt_p(p):
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def tol_dir(rg, tol):
    if tol == 15.0:
        return os.path.join(RESULTS_DIR, f"budget_{rg}")
    return os.path.join(RESULTS_DIR, f"budget_{rg}__at{int(tol)}")


def tol_tag(tol):
    return None if tol == 15.0 else f"at{int(tol)}"


def main():
    print("# ANG_TOL sensitivity sweep "
          "(mean_bound_final @ T, lower better, n=40 paired)\n")
    for rg in REGIMES:
        print(f"{'=' * 84}\n## regime {rg}\n{'=' * 84}")
        fb_dir = os.path.join(RESULTS_DIR, f"budget_{rg}")
        fb_rows = load(fb_dir, "Frontier-Bounded")
        if fb_rows is None:
            print("  MISSING FB baseline")
            continue
        for a_name, b_name, title, order in (
                ("Richness-Angular", "Frontier-Bounded", "RA vs FB", "RA"),
                ("Coverage-U", "Frontier-Bounded", "CU vs FB", "CU"),
                ("Richness-Angular", "Coverage-U", "RA vs CU", "RA")):
            rows = []
            for tol in TOL_LEVELS:
                d = tol_dir(rg, tol)
                if b_name == "Frontier-Bounded":
                    rb = fb_rows
                else:
                    rb = load(d, b_name, tol_tag(tol))
                ra = load(d, a_name, tol_tag(tol))
                if ra is None or rb is None or len(ra) == 0:
                    print(f"  WARNING {d}: {a_name}={len(ra or [])} "
                          f"{b_name}={len(rb or [])}")
                    continue
                va, vb = aligned(ra, rb, PRIMARY)
                if len(va) == 0:
                    print(f"  WARNING {d}: no shared runs ({a_name} vs {b_name})")
                    continue
                p, dlt = paired_stats(va, vb)
                mb, ma = float(np.nanmedian(vb)), float(np.nanmedian(va))
                g = rel_of_medians(ma, mb, True)
                rows.append((int(tol), mb, ma, g, p, dlt))
            if not rows:
                continue
            corr = holm_bonferroni([r[4] for r in rows])
            print(f"\n  {title}  (rel-red% positive = {order} better)")
            print(f"    {'tol':>4} {'med_' + b_name:>12} {'med_' + order:>10} "
                  f"{'rel-red%':>9} {'p':>9} {'p_holm':>9} {'delta':>7} "
                  f"{'sig':>4}")
            for (t, mb, ma, g, p, dlt), pc in zip(rows, corr):
                print(f"    {t:>4} {mb:>12.4f} {ma:>10.4f} {g:>+9.1f} "
                      f"{fmt_p(p):>9} {fmt_p(pc):>9} {dlt:>+7.3f} "
                      f"{'YES' if pc < 0.05 else 'no':>4}")

        gprint = []
        for tol in TOL_LEVELS:
            d = tol_dir(rg, tol)
            ra = load(d, "Richness-Angular", tol_tag(tol))
            if ra is None or len(ra) == 0:
                continue
            va, vb = aligned(ra, fb_rows, GUARD)
            if len(va) == 0:
                continue
            p, _ = paired_stats(va, vb)
            mb, ma = float(np.nanmedian(vb)), float(np.nanmedian(va))
            gprint.append((int(tol), mb, ma, rel_of_medians(ma, mb, False), p))
        gcorr = holm_bonferroni([r[4] for r in gprint])
        print(f"\n  GUARD {GUARD} RA vs FB (higher better)")
        print(f"    {'tol':>4} {'med_FB':>12} {'med_RA':>10} "
              f"{'gain%':>9} {'p':>9} {'p_holm':>9} {'regress':>8}")
        for (t, mb, ma, g, p), pc in zip(gprint, gcorr):
            is_r = pc < 0.05 and ma < mb
            print(f"    {t:>4} {mb:>12.1f} {ma:>10.1f} {g:>+9.1f} "
                  f"{fmt_p(p):>9} {fmt_p(pc):>9} {'YES' if is_r else 'no':>8}")

        for tol in TOL_LEVELS:
            d = tol_dir(rg, tol)
            ra = vec(load(d, "Richness-Angular", tol_tag(tol)), CTX)
            cu = vec(load(d, "Coverage-U", tol_tag(tol)), CTX)
            if len(ra) and len(cu):
                print(f"    [ctx] tol={int(tol):>2} quality_auc med "
                      f"RA={np.nanmedian(ra):.4f} CU={np.nanmedian(cu):.4f}")


if __name__ == "__main__":
    main()
