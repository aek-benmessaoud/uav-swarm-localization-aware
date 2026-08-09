"""
analysis/baseline_budget_stats.py — Baseline budget table (reviewer #2/#3).

One table over the newly completed budget baselines on the two 40-run regimes
(A3_obs005, A6_obs005): Random, Frontier-Bounded, Richness-Angular,
Entropy-Frac, Frontier+Entropy, Coverage-U. Metrics @ budget T:
  - mean_bound_final (lower better) — continuous accuracy signal
  - final_coverage (higher better)  — coverage at T
  - quality_auc (higher better)     — integrated quality
Each method is paired (same env_seed per run) against Random (the floor) and
against Frontier-Bounded (the movement-frame control). Wilcoxon signed-rank +
Holm-Bonferroni per metric.

Usage:
  python analysis/baseline_budget_stats.py
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

METHODS = ["Random", "Frontier-Bounded", "Richness-Angular",
           "Entropy-Frac", "Frontier+Entropy", "Coverage-U"]
FLOOR = "Random"
CONTROL = "Frontier-Bounded"

METRICS = [
    ("mean_bound_final", "lower"),
    ("final_coverage", "higher"),
    ("quality_auc", "higher"),
]
LOWER_BETTER = {"mean_bound_final"}


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


def _num(r, key):
    v = r.get(key)
    if v in (None, ""):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def metric_vector(rows, metric):
    return np.array([_num(r, metric) for r in rows], dtype=float)


def paired_stats(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    d = a - b
    if np.all(np.abs(d[np.isfinite(d)]) < 1e-12):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        return 1.0
    return float(p)


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
    data = {}
    for rg in REGIMES:
        d = os.path.join(RESULTS_DIR, f"budget_{rg}")
        rows = {}
        for m in METHODS:
            r = load(d, m)
            if r is None or not r:
                print(f"WARNING incomplete {rg} / {m}: "
                      f"{0 if r is None else len(r)} runs")
                r = []
            rows[m] = r
        data[rg] = rows

    for rg in REGIMES:
        print(f"\n{'=' * 80}\n# Regime {rg} (n per method: "
              f"{min(len(data[rg][m]) for m in METHODS)})\n{'=' * 80}")

        # --- medians table ---
        print(f"\n# Medians @ budget T")
        print(f"{'method':<18} {'mean_bound':>11} {'coverage':>10} "
              f"{'qual_auc':>9}")
        for m in METHODS:
            mb = np.nanmedian(metric_vector(data[rg][m], "mean_bound_final"))
            cov = np.nanmedian(metric_vector(data[rg][m], "final_coverage"))
            qa = np.nanmedian(metric_vector(data[rg][m], "quality_auc"))
            print(f"{m:<18} {mb:>11.4f} {cov:>10.1f} {qa:>9.4f}")

        # --- paired vs Random (floor) ---
        for metric, direction in METRICS:
            print(f"\n# {metric} ({direction} better), paired vs {FLOOR}")
            print(f"{'method':<18} {'med':>10} {'med_floor':>10} {'p':>8} "
                  f"{'p_holm':>8} {'sig':>5}")
            ref = metric_vector(data[rg][FLOOR], metric)
            ps = []
            for m in METHODS:
                if m == FLOOR:
                    continue
                v = metric_vector(data[rg][m], metric)
                ps.append(paired_stats(v, ref))
            corr = holm_bonferroni(ps)
            idx = 0
            for m in METHODS:
                if m == FLOOR:
                    continue
                v = metric_vector(data[rg][m], metric)
                p, pc = ps[idx], corr[idx]
                idx += 1
                better = (np.nanmedian(v) > np.nanmedian(ref)
                          if direction == "higher"
                          else np.nanmedian(v) < np.nanmedian(ref))
                sig = "YES" if (pc < 0.05 and better) else "no"
                print(f"{m:<18} {np.nanmedian(v):>10.4f} "
                      f"{np.nanmedian(ref):>10.4f} {p:>8.4f} {pc:>8.4f} "
                      f"{sig:>5}")

        # --- paired vs Frontier-Bounded (control) ---
        print(f"\n# Paired vs {CONTROL} (movement-frame control)")
        print(f"{'method':<18} {'metric':<18} {'med':>10} {'med_ctrl':>10} "
              f"{'p':>8} {'p_holm':>8} {'sig':>5}")
        ctrl = metric_vector(data[rg][CONTROL], None) if False else None
        for metric, direction in METRICS:
            ref = metric_vector(data[rg][CONTROL], metric)
            ps = []
            for m in METHODS:
                if m == CONTROL:
                    continue
                ps.append(paired_stats(metric_vector(data[rg][m], metric),
                                       ref))
            corr = holm_bonferroni(ps)
            idx = 0
            for m in METHODS:
                if m == CONTROL:
                    continue
                v = metric_vector(data[rg][m], metric)
                p, pc = ps[idx], corr[idx]
                idx += 1
                better = (np.nanmedian(v) > np.nanmedian(ref)
                          if direction == "higher"
                          else np.nanmedian(v) < np.nanmedian(ref))
                sig = "YES" if (pc < 0.05 and better) else "no"
                print(f"{m:<18} {metric:<18} {np.nanmedian(v):>10.4f} "
                      f"{np.nanmedian(ref):>10.4f} {p:>8.4f} {pc:>8.4f} "
                      f"{sig:>5}")


if __name__ == "__main__":
    main()
