"""
analysis/sweep_stats.py — post-preregistration robustness sweeps (A-D).

Three confirmatory follow-up sweeps (FUTURE_WORK_NOTES.md phases B/C/D), all
paired by env_seed with the same Wilcoxon + Holm-Bonferroni protocol:

  B. sigma-loc sweep   — self-localization noise std in grid cells applied to
                         the LOCAL decision signal only (oracle keeps true
                         geometry). Does the E4-CONFIRM Coverage-U effect
                         survive observation noise?  variants: s05, s10
  C. budget sweep      — mission budget as fraction of FB median steps_90.
                         Core "budget effect" claim: CU advantage should be
                         largest at tight budgets, vanish at high budget.
                         variants: b03, b05, b09 (b07 = confirmed data)
  D. comm-range sweep  — fusion range in grid cells (default = FOV = 5).
                         Locality stress test. variants: r25, r125
  E. bearing-noise sweep — bearing-measurement noise (std in degrees) applied
                         to the LOCAL angular configurations only; oracle CRLB
                         and global clusters keep true geometry.
                         variant: be2 (sigma = 2 deg)

Primary metric per sweep: mean_bound_final @ T (lower better). Guard:
final_coverage @ T (Coverage-U must not be significantly worse). Context:
quality_auc @ T.

Usage:
  python analysis/sweep_stats.py
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

# label -> (flag that identifies the dir suffix, dict of variant labels)
SWEEPS = {
    "B sigma-loc":   ("__s", ["s05", "s10"]),
    "C budget-frac": ("__b", ["b03", "b05", "b09"]),
    "D comm-range":  ("__r", ["r25", "r125"]),
    "E bearing-noise": ("__be", ["be2"]),
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


def rel_gain(va, vb, lower_better):
    """Median relative change of a vs b; positive = a better.

    Matches the paper's convention exactly (paper_build.py: g = 100*(vb - va)/vb
    element-wise, then np.nanmedian), with vb = baseline always the denominator.
    Fixed 2026-08-09: lower_better previously divided by the candidate (va).
    """
    if lower_better:
        g = 100.0 * (vb - va) / vb
    else:
        g = 100.0 * (va - vb) / vb
    return float(np.nanmedian(g))


def fmt_p(p):
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def base_dir(rg):
    return os.path.join(RESULTS_DIR, f"budget_{rg}")


def main():
    # baseline (confirmed) numbers for reference
    for rg in REGIMES:
        fb = vec(load(base_dir(rg), "Frontier-Bounded"), PRIMARY)
        cu = vec(load(base_dir(rg), "Coverage-U"), PRIMARY)
        p, _ = paired_stats(cu, fb)
        g = rel_gain(cu, fb, lower_better=True)
        print(f"[baseline] {rg}: CU vs FB rel-red {g:+.1f}% (p={p:.4f})")

    for sweep, (_suffix, variants) in SWEEPS.items():
        print(f"\n{'=' * 78}\n# Sweep {sweep} — Coverage-U vs Frontier-Bounded\n"
              f"{'=' * 78}")
        for variant in variants:
            print(f"\n## variant {variant}")
            rows = []
            for rg in REGIMES:
                dirname = os.path.join(RESULTS_DIR,
                                       f"budget_{rg}__{variant}")
                fb = vec(load(dirname, "Frontier-Bounded", variant), PRIMARY)
                cu = vec(load(dirname, "Coverage-U", variant), PRIMARY)
                if len(fb) == 0 or len(cu) == 0:
                    print(f"  WARNING {dirname}: missing data")
                    continue
                p, dlt = paired_stats(cu, fb)
                rows.append((rg, np.nanmedian(fb), np.nanmedian(cu),
                             rel_gain(cu, fb, True), p, dlt))
            if not rows:
                continue
            ps = [r[4] for r in rows]
            corr = holm_bonferroni(ps)
            print(f"  PRIMARY {PRIMARY} (lower better)")
            print(f"  {'regime':<12} {'med_FB':>9} {'med_CU':>9} "
                  f"{'rel-red%':>9} {'p':>9} {'p_holm':>9} {'delta':>7} "
                  f"{'sig':>4}")
            for (rg, mfb, mcu, g, p, dlt), pc in zip(rows, corr):
                print(f"  {rg:<12} {mfb:>9.4f} {mcu:>9.4f} {g:>+9.1f} "
                      f"{fmt_p(p):>9} {fmt_p(pc):>9} {dlt:>+7.3f} "
                      f"{'YES' if pc < 0.05 else 'no':>4}")
            # guard + context
            gprint = []
            for rg in REGIMES:
                dirname = os.path.join(RESULTS_DIR,
                                       f"budget_{rg}__{variant}")
                fb = vec(load(dirname, "Frontier-Bounded", variant), GUARD)
                cu = vec(load(dirname, "Coverage-U", variant), GUARD)
                if len(fb) == 0 or len(cu) == 0:
                    continue
                p, _ = paired_stats(cu, fb)
                gprint.append((rg, np.nanmedian(fb), np.nanmedian(cu),
                               rel_gain(cu, fb, False), p))
            gps = [r[4] for r in gprint]
            gcorr = holm_bonferroni(gps)
            print(f"  GUARD  {GUARD} (higher better)")
            print(f"  {'regime':<12} {'med_FB':>9} {'med_CU':>9} "
                  f"{'gain%':>9} {'p':>9} {'p_holm':>9} {'regress':>8}")
            for (rg, mfb, mcu, g, p), pc in zip(gprint, gcorr):
                is_r = pc < 0.05 and mcu < mfb
                print(f"  {rg:<12} {mfb:>9.1f} {mcu:>9.1f} {g:>+9.1f} "
                      f"{fmt_p(p):>9} {fmt_p(pc):>9} "
                      f"{'YES' if is_r else 'no':>8}")


if __name__ == "__main__":
    main()
