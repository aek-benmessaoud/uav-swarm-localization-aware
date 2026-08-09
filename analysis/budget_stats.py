"""
analysis/budget_stats.py — E4 A/B statistics: Coverage-U vs Frontier-Bounded
under a finite mission budget (plus Deploy-U context row).

Primary (pre-registered): quality_auc @ budget T (integrated accuracy under the
mission, higher better). Guard: final_coverage @ T must NOT be significantly
worse (Pareto-clean check). Secondary: mean_bound_final @ T (lower better),
undetermined_final @ T (lower better), steps_dual (lower better), coverage_auc
(higher better).

Protocol (locked, mirrors deploy_stats):
  - Paired Wilcoxon (zero_method="wilcox") per regime.
  - Holm-Bonferroni across the 4 regimes (global).
  - Delta = matched-pairs statistic m/n^2 (positive = A better).
  - Median relative gain of A over B (sign flipped for lower-is-better).

A4 verdict (E4): median quality_auc gain >= 8% AND any Holm-sig on quality_auc
AND no Holm-sig coverage regression.

Usage:
  python analysis/budget_stats.py
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from config import (METHOD_FRONTIER_BOUNDED, METHOD_DEPLOY, METHOD_COVERAGE_U)

REGIMES = ["A2_obs005", "A3_obs005", "A6_obs005", "A6_obs020"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

PRIMARY = "quality_auc"
GUARD = "final_coverage"
SECONDARY = ["coverage_auc", "mean_bound_final", "undetermined_final",
             "steps_dual", "time_to_quality"]
# lower-is-better metrics (positive gain = A better when smaller)
LOWER_BETTER = {"mean_bound_final", "undetermined_final", "steps_dual",
                "time_to_quality"}
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

    Matches the paper's rel-red convention EXACTLY (paper_build.py inline:
    g = 100*(vb - va)/vb element-wise, then np.nanmedian), with vb = B =
    baseline ALWAYS the denominator. Fixed 2026-08-09: the lower_better branch
    previously divided by the candidate (va), inflating small quantities.
    """
    if lower_better:
        g = 100.0 * (vb - va) / vb
    else:
        g = 100.0 * (va - vb) / vb
    return float(np.nanmedian(g))


def main():
    data = {}
    for rg in REGIMES:
        d = os.path.join(RESULTS_DIR, f"budget_{rg}")
        rows = {}
        for m in (METHOD_FRONTIER_BOUNDED, METHOD_DEPLOY, METHOD_COVERAGE_U):
            r = load(d, m)
            if r is None or not r:
                print(f"WARNING incomplete regime {rg}, method {m}: "
                      f"{0 if r is None else len(r)} runs")
                r = []
            rows[m] = r
        if rows[METHOD_FRONTIER_BOUNDED] and rows[METHOD_COVERAGE_U]:
            data[rg] = rows

    print(f"# E4 A/B: {METHOD_COVERAGE_U} vs {METHOD_FRONTIER_BOUNDED} "
          f"(paired, budget-limited; {METHOD_DEPLOY} as context)")
    print(f"# PRIMARY: {PRIMARY} @ budget (higher better)\n")

    # --- primary table (Coverage-U vs FB) ---
    rows = []
    for rg in REGIMES:
        if rg not in data:
            continue
        a = data[rg][METHOD_COVERAGE_U]
        b = data[rg][METHOD_FRONTIER_BOUNDED]
        va = metric_vector(a, PRIMARY, -1.0)
        vb = metric_vector(b, PRIMARY, -1.0)
        p, d = paired_stats(va, vb)
        med_a, med_b = np.median(va), np.median(vb)
        g = gain_pct(va, vb, lower_better=False)
        rows.append((rg, med_a, med_b, g, p, d))
    ps = [r[4] for r in rows]
    corrected = holm_bonferroni(ps)

    print(f"# PRIMARY {PRIMARY} (median), Coverage-U vs FB")
    print(f"{'regime':<12} {'med_FB':>8} {'med_CU':>8} {'gain%':>7} "
          f"{'p':>8} {'p_holm':>8} {'delta':>7} {'sig':>5}")
    print("-" * 72)
    for (rg, med_a, med_b, g, p, d), pc in zip(rows, corrected):
        sig = "YES" if pc < 0.05 else "no"
        print(f"{rg:<12} {med_b:>8.3f} {med_a:>8.3f} {g:>+7.1f} "
              f"{p:>8.4f} {pc:>8.4f} {d:>+7.3f} {sig:>5}")

    # --- coverage guard (Pareto-clean check) ---
    grow = []
    for rg in REGIMES:
        if rg not in data:
            continue
        a = data[rg][METHOD_COVERAGE_U]
        b = data[rg][METHOD_FRONTIER_BOUNDED]
        va = metric_vector(a, GUARD, -1.0)
        vb = metric_vector(b, GUARD, -1.0)
        p, _ = paired_stats(va, vb)
        med_a, med_b = np.median(va), np.median(vb)
        grow.append((rg, med_a, med_b, p))
    gps = [r[3] for r in grow]
    gcorr = holm_bonferroni(gps)
    print(f"\n# GUARD: {GUARD} @ budget (Coverage-U must not be significantly "
          f"worse)")
    print(f"{'regime':<12} {'med_FB':>8} {'med_CU':>8} {'p':>8} "
          f"{'p_holm':>8} {'regression?':>12}")
    print("-" * 60)
    regress = False
    for (rg, med_a, med_b, p), pc in zip(grow, gcorr):
        is_r = pc < 0.05 and med_a < med_b
        regress |= is_r
        print(f"{rg:<12} {med_b:>8.1f} {med_a:>8.1f} {p:>8.4f} {pc:>8.4f} "
              f"{'YES' if is_r else 'no':>12}")

    # --- A4 verdict ---
    if rows:
        gains = [r[3] for r in rows]
        med_gain = float(np.median(gains))
        any_sig = any(pc < 0.05 for pc in corrected)
        passed = med_gain >= MIN_GAIN_PCT and any_sig and not regress
        print()
        print(f"# A4 verdict (primary={PRIMARY}): median gain = {med_gain:+.1f}% "
              f"(threshold {MIN_GAIN_PCT:+.0f}%), any Holm-sig = {any_sig}, "
              f"coverage regression = {regress}")
        print(f"#   => {'PASS' if passed else 'FAIL'}")

    # --- secondary metrics (Coverage-U vs FB) ---
    print("\n# Secondary metrics, Coverage-U vs FB (gain% = Coverage-U better)")
    print(f"{'metric':<18} {'regime':<12} {'med_FB':>8} {'med_CU':>8} "
          f"{'gain%':>7}")
    print("-" * 60)
    for metric in SECONDARY:
        lb = metric in LOWER_BETTER
        for rg in REGIMES:
            if rg not in data:
                continue
            a = data[rg][METHOD_COVERAGE_U]
            b = data[rg][METHOD_FRONTIER_BOUNDED]
            va = metric_vector(a, metric, None)
            vb = metric_vector(b, metric, None)
            if np.all(np.isnan(va)) or np.all(np.isnan(vb)):
                continue
            med_a, med_b = np.nanmedian(va), np.nanmedian(vb)
            g = gain_pct(va, vb, lb)
            print(f"{metric:<18} {rg:<12} {med_b:>8.3f} {med_a:>8.3f} "
                  f"{g:>+7.1f}")

    # --- accuracy secondaries with paired tests ---
    # quality_auc is a coarse binary (>=2 configs) saturation measure; the
    # continuous accuracy metrics (mean_bound_final, undetermined_final) carry
    # the precision signal directly. Report their paired tests explicitly.
    print("\n# Accuracy secondaries, paired (Coverage-U vs FB; gain = "
          "Coverage-U better)")
    for metric in ("mean_bound_final", "undetermined_final"):
        print(f"\n  {metric} (lower better)")
        arows = []
        for rg in REGIMES:
            if rg not in data:
                continue
            a = data[rg][METHOD_COVERAGE_U]
            b = data[rg][METHOD_FRONTIER_BOUNDED]
            va = metric_vector(a, metric, None)
            vb = metric_vector(b, metric, None)
            if np.all(np.isnan(va)) or np.all(np.isnan(vb)):
                continue
            p, d = paired_stats(va, vb)
            g = gain_pct(va, vb, lower_better=True)
            arows.append((rg, np.nanmedian(va), np.nanmedian(vb), g, p, d))
        aps = [r[4] for r in arows]
        acorr = holm_bonferroni(aps)
        print(f"    {'regime':<12} {'med_FB':>9} {'med_CU':>9} {'gain%':>8} "
              f"{'p':>8} {'p_holm':>8} {'delta':>7} {'sig':>5}")
        for (rg, ma, mb, g, p, d), pc in zip(arows, acorr):
            print(f"    {rg:<12} {mb:>9.4f} {ma:>9.4f} {g:>+8.1f} {p:>8.4f} "
                  f"{pc:>8.4f} {d:>+7.3f} {'YES' if pc < 0.05 else 'no':>5}")

    # --- E4-CONFIRM: pre-registered higher-power verdict on the accuracy
    # primary. E4's preregistered primary (quality_auc) failed (saturated binary
    # fraction); the coherent directional effect was on mean_bound_final @ T.
    # Confirmation campaign: n=40 pairs/regime (30 new seed-pairs appended),
    # methods FB + Coverage-U only, same budgets T. Pre-registered success:
    #   (i) median relative reduction of mean_bound_final rel. FB >= 10%,
    #   (ii) any Holm-sig on mean_bound_final @ T across the 4 regimes,
    #   (iii) no Holm-sig coverage regression.
    # Fisher combined p across the 4 regimes reported as a global statement.
    print("\n# E4-CONFIRM: higher-power verdict on mean_bound_final @ T "
          "(rel-FB reduction)")
    for metric in ("mean_bound_final",):
        prow = []
        for rg in REGIMES:
            if rg not in data:
                continue
            a = data[rg][METHOD_COVERAGE_U]
            b = data[rg][METHOD_FRONTIER_BOUNDED]
            va = metric_vector(a, metric, None)
            vb = metric_vector(b, metric, None)
            if np.all(np.isnan(va)) or np.all(np.isnan(vb)):
                continue
            p, d = paired_stats(va, vb)
            med_a, med_b = np.nanmedian(va), np.nanmedian(vb)
            rel = (med_b - med_a) / med_b * 100.0
            prow.append((rg, med_a, med_b, rel, p, d))
        cps = [r[4] for r in prow]
        cholm = holm_bonferroni(cps)
        print(f"    {'regime':<12} {'med_FB':>9} {'med_CU':>9} "
              f"{'rel-red%':>9} {'p':>8} {'p_holm':>8} {'delta':>7} {'sig':>5}")
        for (rg, ma, mb, rel, p, d), pc in zip(prow, cholm):
            print(f"    {rg:<12} {mb:>9.4f} {ma:>9.4f} {rel:>+9.1f} {p:>8.4f} "
                  f"{pc:>8.4f} {d:>+7.3f} {'YES' if pc < 0.05 else 'no':>5}")
        rels = [r[3] for r in prow]
        med_rel = float(np.nanmedian(rels))
        any_sig = any(pc < 0.05 for pc in cholm)
        if len(cps) >= 2 and all(p > 0 for p in cps):
            chi2 = -2.0 * np.sum(np.log(np.maximum(cps, 1e-300)))
            f_p = float(1.0 - stats.chi2.cdf(chi2, 2 * len(cps)))
        else:
            f_p = float("nan")
        # guard: coverage regression on the full (appended) sample
        cov_reg = False
        for rg in REGIMES:
            if rg not in data:
                continue
            va = metric_vector(data[rg][METHOD_COVERAGE_U], GUARD, -1.0)
            vb = metric_vector(data[rg][METHOD_FRONTIER_BOUNDED], GUARD, -1.0)
            p, _ = paired_stats(va, vb)
            if p < 0.05 and np.nanmedian(va) < np.nanmedian(vb):
                cov_reg = True
        passed = med_rel >= 10.0 and any_sig and not cov_reg
        print(f"\n    median rel-reduction = {med_rel:+.1f}% (threshold +10%), "
              f"any Holm-sig = {any_sig}, coverage regression = {cov_reg}, "
              f"Fisher combined p = {f_p:.4f}")
        print(f"    => {'PASS' if passed else 'FAIL'} (E4-CONFIRM)")

    # --- context: Deploy-U vs FB on primary ---
    print("\n# Context: Deploy-U vs FB on PRIMARY (from the same budget data)")
    print(f"{'regime':<12} {'med_FB':>8} {'med_DU':>8} {'gain%':>7}")
    print("-" * 40)
    for rg in REGIMES:
        if rg not in data or not data[rg][METHOD_DEPLOY]:
            continue
        a = data[rg][METHOD_DEPLOY]
        b = data[rg][METHOD_FRONTIER_BOUNDED]
        # Deploy-U exists only on the original 10 seed-pairs (0-9); restrict FB
        # to the same run indices so the context comparison stays paired.
        du_runs = {int(r.get("run", -1)) for r in a}
        fb = [r for r in b if int(r.get("run", -1)) in du_runs]
        va = metric_vector(a, PRIMARY, -1.0)
        vb = metric_vector(fb, PRIMARY, -1.0)
        g = gain_pct(va, vb, lower_better=False)
        print(f"{rg:<12} {np.median(vb):>8.3f} {np.median(va):>8.3f} "
              f"{g:>+7.1f}")


if __name__ == "__main__":
    main()
