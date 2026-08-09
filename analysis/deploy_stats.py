"""
analysis/deploy_stats.py — E3 A/B statistics: Deploy-U vs Frontier-Bounded.

Compares the localization-aware deployment strategy (Deploy-U) against the
validated movement control (Frontier-Bounded) across the four E3 regimes
(A2/A3/A6 at obs=5%, A6 at obs=20%), paired by env seed (run index).

Primary (pre-registered): steps_dual = first step where BOTH coverage >= 90%
AND quality >= 0.9 (the title's "Accuracy AND Coverage"). Censored at
--max-steps.

Protocol (locked, mirrors phase1_stats):
  - Paired Wilcoxon (zero_method="wilcox") per regime on the primary metric.
  - Holm-Bonferroni across the 4 regimes (global).
  - Delta = matched-pairs statistic m/n^2 (positive = Deploy-U better).
  - Median relative gain of Deploy-U over Frontier-Bounded (sign flipped so
    positive = Deploy-U better for lower-is-better metrics).

A4 verdict: median gain >= 8% AND global Holm p < 0.05 on the primary metric.

Usage:
  python analysis/deploy_stats.py --max-steps 8000
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from config import METHOD_FRONTIER_BOUNDED, METHOD_DEPLOY

REGIMES = ["A2_obs005", "A3_obs005", "A6_obs005", "A6_obs020"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

PRIMARY = "steps_dual"
SECONDARY = ["steps_90", "quality_auc", "time_to_quality", "quality_final",
             "mean_bound_final", "deploy_frac", "orbit_frac"]
# lower-is-better metrics (positive gain = Deploy-U better when smaller)
LOWER_BETTER = {"steps_dual", "steps_90", "time_to_quality", "mean_bound_final"}
MIN_GAIN_PCT = 8.0  # A4 preregistered threshold


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=8000,
                    help="Censoring cap for steps_dual / steps_90.")
    args = ap.parse_args()

    data = {}
    for rg in REGIMES:
        d = os.path.join(RESULTS_DIR, f"deploy_{rg}")
        fb = load(d, METHOD_FRONTIER_BOUNDED)
        du = load(d, METHOD_DEPLOY)
        if fb is None or du is None or not fb or not du:
            print(f"WARNING incomplete regime {rg}: "
                  f"FB={0 if fb is None else len(fb)} "
                  f"DU={0 if du is None else len(du)}")
            continue
        data[rg] = (fb, du)

    print(f"# E3 A/B: {METHOD_DEPLOY} vs {METHOD_FRONTIER_BOUNDED} "
          f"(paired, censored at {args.max_steps})")
    print("# PRIMARY metric: steps_dual (coverage>=90% AND quality>=0.9)\n")

    print("# Per-regime medians [IQR]\n")
    for rg in REGIMES:
        if rg not in data:
            continue
        fb, du = data[rg]
        parts = []
        for metric in [PRIMARY] + SECONDARY:
            vf = metric_vector(fb, metric, args.max_steps)
            vd = metric_vector(du, metric, args.max_steps)
            parts.append(f"{metric}="
                         f"FB {np.median(vf):.3f} / DU {np.median(vd):.3f}")
        print(f"  {rg:<12} {'  '.join(parts)}")
    print()

    # --- PRIMARY: paired Wilcoxon per regime + Holm across regimes ---
    rows = []
    for rg in REGIMES:
        if rg not in data:
            continue
        fb, du = data[rg]
        va = metric_vector(fb, PRIMARY, args.max_steps)
        vb = metric_vector(du, PRIMARY, args.max_steps)
        p, d = paired_stats(va, vb)
        med_a, med_b = np.median(va), np.median(vb)
        gain = np.nanmedian((va - vb) / np.where(va == 0, np.nan, va)) * 100.0
        rows.append((rg, med_a, med_b, gain, p, d))

    ps = [r[4] for r in rows]
    corrected = holm_bonferroni(ps)

    print("# PRIMARY paired Wilcoxon + Holm (global across regimes)\n")
    print(f"{'regime':<12} {'med_FB':>8} {'med_DU':>8} {'gain%':>7} "
          f"{'p':>8} {'p_holm':>8} {'delta':>7} {'sig':>5}")
    print("-" * 72)
    for (rg, med_a, med_b, gain, p, d), pc in zip(rows, corrected):
        sig = "YES" if pc < 0.05 else "no"
        print(f"{rg:<12} {med_a:>8.0f} {med_b:>8.0f} {gain:>+7.1f} "
              f"{p:>8.4f} {pc:>8.4f} {d:>+7.3f} {sig:>5}")

    # --- A4 verdict on the primary metric ---
    if rows:
        gains = [r[3] for r in rows]
        med_gain = float(np.median(gains))
        sig_all = [pc < 0.05 for pc in corrected]
        any_sig = any(sig_all)
        passed = med_gain >= MIN_GAIN_PCT and any_sig
        print()
        print(f"# A4 verdict (primary=steps_dual): median gain = "
              f"{med_gain:+.1f}% (threshold {MIN_GAIN_PCT:+.0f}%), "
              f"any Holm-sig = {any_sig}")
        print(f"#   => {'PASS' if passed else 'FAIL'}")

    # --- secondary metrics across regimes (informational) ---
    print("\n# Secondary metrics (medians, gain% = Deploy-U better)")
    print(f"{'metric':<18} {'regime':<12} {'med_FB':>8} {'med_DU':>8} "
          f"{'gain%':>7}")
    print("-" * 60)
    for metric in SECONDARY:
        for rg in REGIMES:
            if rg not in data:
                continue
            fb, du = data[rg]
            va = metric_vector(fb, metric, args.max_steps)
            vb = metric_vector(du, metric, args.max_steps)
            med_a, med_b = np.median(va), np.median(vb)
            # Denominator is ALWAYS the baseline (FB = va). gain positive =
            # Deploy-U better: for lower-better, DU < FB is good -> (FB-DU)/FB;
            # for higher-better, DU > FB is good -> (DU-FB)/FB.
            if metric in LOWER_BETTER:
                gain = np.nanmedian(
                    (va - vb) / np.where(va == 0, np.nan, va)) * 100.0
            else:
                gain = np.nanmedian(
                    (vb - va) / np.where(va == 0, np.nan, va)) * 100.0
            print(f"{metric:<18} {rg:<12} {med_a:>8.3f} {med_b:>8.3f} "
                  f"{gain:>+7.1f}")


if __name__ == "__main__":
    main()
