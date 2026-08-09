"""
analysis/pareto_stats.py — E4-PARETO: lambda sensitivity / accuracy-coverage
Pareto frontier for Coverage-U under the fixed budget T.

Data provenance (all seed-paired, runs 0-39):
  - Frontier-Bounded (lambda = 0) and Coverage-U at lambda = 0.5:
      results/budget_{regime}/raw_comm_limited__{method}.csv   (E4 + E4-CONFIRM)
  - Coverage-U at lambda in {0.25, 1.0, 2.0}:
      results/pareto_{regime}/raw_comm_limited__Coverage-U__lam{lam}.csv

Per regime and lambda (vs FB control):
  - accuracy : med mean_bound_final @ T (lower better), rel-FB reduction %,
               paired Wilcoxon + Holm across the 4 nonzero lambdas (within
               regime).
  - guard    : med final_coverage @ T, paired Wilcoxon + Holm across lambdas
               (no significant regression allowed).
  - support  : undetermined_final @ T (lower better), descriptive.

Pre-registered verdict (E4-PARETO):
  PASS iff, in at least one of the two regimes, >= 2 of the 4 lambda values
  satisfy BOTH (rel-FB reduction of mean_bound >= 10% AND Holm-sig on
  mean_bound_final) AND none of those lambdas has a Holm-sig coverage
  regression. Secondary reading: plateau vs knife-edge of lambda = 0.5.

Usage:
  python analysis/pareto_stats.py
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from config import (METHOD_FRONTIER_BOUNDED, METHOD_COVERAGE_U,
                    COVERAGE_U_LAMBDA)

REGIMES = ["A3_obs005", "A6_obs005"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
LAMBDAS = [0.25, COVERAGE_U_LAMBDA, 1.0, 2.0]

ACC = "mean_bound_final"
COV = "final_coverage"
UND = "undetermined_final"
MIN_RED_PCT = 10.0


def sanitize(method):
    return re.sub(r"[^\w\-]", "_", method)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r.get("run", 0)))
    return rows


def vec(rows, metric):
    out = []
    for r in rows:
        v = r.get(metric)
        try:
            out.append(float(v) if v not in (None, "") else np.nan)
        except ValueError:
            out.append(np.nan)
    return np.array(out, dtype=float)


def paired_p(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return 1.0
    if np.all(np.abs(a - b) < 1e-12):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        return 1.0
    return p


def holm(ps):
    n = len(ps)
    order = np.argsort(ps)
    out = [None] * n
    for rank, idx in enumerate(order):
        out[idx] = min(1.0, ps[idx] * (n - rank))
    for i in range(n - 1, 0, -1):
        out[order[i - 1]] = min(out[order[i - 1]], out[order[i]])
    return out


def load_fb_cu(rg):
    d = os.path.join(RESULTS_DIR, f"budget_{rg}")
    fb = load(os.path.join(d, f"raw_comm_limited__{sanitize(METHOD_FRONTIER_BOUNDED)}.csv"))
    cu = load(os.path.join(d, f"raw_comm_limited__{sanitize(METHOD_COVERAGE_U)}.csv"))
    return fb, cu


def load_lam(rg, lam):
    d = os.path.join(RESULTS_DIR, f"pareto_{rg}")
    return load(os.path.join(d, f"raw_comm_limited__{sanitize(METHOD_COVERAGE_U)}__lam{lam}.csv"))


def main():
    print(f"# E4-PARETO: Coverage-U lambda sensitivity under budget T "
          f"(vs Frontier-Bounded, paired, n per cell)")
    print(f"# regimes: {REGIMES}; lambdas: {LAMBDAS} "
          f"(0 = FB control); accuracy={ACC}, guard={COV}, support={UND}\n")

    all_ok = True
    for rg in REGIMES:
        fb, cu = load_fb_cu(rg)
        if fb is None or cu is None:
            print(f"WARNING {rg}: missing FB or CU(lam=0.5) in budget_*")
            all_ok = False
            continue
        n = len(fb)
        print(f"===== {rg}  (n={n} pairs) =====")

        rows = []  # (lam, med_acc, med_cov, rel_red, p_acc, p_cov, med_und)
        for lam in LAMBDAS:
            if lam == COVERAGE_U_LAMBDA:
                cu_lam = cu
            else:
                cu_lam = load_lam(rg, lam)
                if cu_lam is None:
                    print(f"WARNING {rg} lam={lam}: missing pareto_* CSV")
                    all_ok = False
                    continue
            va = vec(cu_lam, ACC)
            vb = vec(fb, ACC)
            cv_a = vec(cu_lam, COV)
            cv_b = vec(fb, COV)
            un_a = vec(cu_lam, UND)
            med_a, med_b = np.nanmedian(va), np.nanmedian(vb)
            rel = (med_b - med_a) / med_b * 100.0 if med_b > 0 else 0.0
            p_acc = paired_p(va, vb)
            p_cov = paired_p(cv_a, cv_b)
            rows.append((lam, med_a, np.nanmedian(cv_a), rel, p_acc, p_cov,
                         np.nanmedian(un_a)))
        # FB control row (lam = 0)
        rows.insert(0, (0.0, np.nanmedian(vec(fb, ACC)),
                        np.nanmedian(vec(fb, COV)), 0.0, 1.0, 1.0,
                        np.nanmedian(vec(fb, UND))))

        # Holm across the 4 nonzero-lambda tests only (lam=0 is the control).
        non_ctrl = [r for r in rows if r[0] > 0]
        acc_ps = [r[4] for r in non_ctrl]
        cov_ps = [r[5] for r in non_ctrl]
        acc_holm = holm(acc_ps)
        cov_holm = holm(cov_ps)
        holm_map = {}
        for (lam, *_r), ah, ch in zip(non_ctrl, acc_holm, cov_holm):
            holm_map[lam] = (ah, ch)

        print(f"\n  {'lam':>5} {'med_bound_FB':>13} {'med_bound_CU':>13} "
              f"{'rel-red%':>9} {'p_acc':>8} {'p_acc_holm':>10} "
              f"{'med_cov%':>9} {'p_cov_holm':>10} {'med_und':>9}")
        print("  " + "-" * 96)
        for lam, ma, mc, rel, pa, pc, mu in rows:
            ah, ch = holm_map.get(lam, (float("nan"), float("nan")))
            print(f"  {lam:>5} {rows[0][1]:>13.4f} {ma:>13.4f} {rel:>+9.1f} "
                  f"{pa:>8.4f} {ah:>10.4f} {mc:>9.1f} {ch:>10.4f} {mu:>9.4f}")

        passing = [(lam, float(rel)) for (lam, _, _, rel, _, _, _), ah in
                   zip(non_ctrl, acc_holm) if rel >= MIN_RED_PCT and ah < 0.05]
        no_regress = all(ch >= 0.05 for (_, ch) in holm_map.values())
        print(f"\n  lambdas meeting (rel-red>={MIN_RED_PCT:.0f}% AND "
              f"Holm-sig): {passing}")
        print(f"  coverage guard (no Holm-sig regression at any lam>0): "
              f"{'OK' if no_regress else 'FAIL'}")
        all_ok = all_ok and len(passing) >= 2 and no_regress

    print(f"\n# E4-PARETO verdict: >=2 passing lambdas in >=1 regime and "
          f"no coverage regression => {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
