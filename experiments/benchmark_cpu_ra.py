"""
experiments/benchmark_cpu_ra.py — RA-only CPU/decision benchmark (10 runs).

Re-measures Richness-Angular ms_per_decision on A6_obs005 (6 agents, 5%
obstacles, 1600 steps) with n=10 to beat the noise of the earlier 4-run
batches. Writes results/ra_cpu_10runs.csv in the ra_cpu.csv format
(median + IQR) and prints the run-level series.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import METHOD_FRONTIER_RICHNESS_ANGULAR
from experiments._runner import run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(ROOT, "results", "ra_cpu_10runs.csv")

KW = dict(grid_size=100, num_agents=6, obstacle_ratio=0.05, max_steps=1600)


def _q(vals, p):
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    import csv
    vals_ms = []
    vals_cpu = []
    t_start = time.perf_counter()
    for r in range(1, args.runs + 1):
        t0 = time.perf_counter()
        res = run_episode(METHOD_FRONTIER_RICHNESS_ANGULAR, "comm_limited", r,
                          env_seed=1000 + r, timing=True, **KW)
        ms = res.get("ms_per_decision")
        vals_ms.append(ms if ms is not None else float("nan"))
        vals_cpu.append(res.get("policy_cpu_s"))
        wall = time.perf_counter() - t0
        print(f"[Richness-Angular] run {r}/{args.runs} done in {wall:.1f}s "
              f"ms/dec={ms:.4f}", flush=True)

    med = statistics.median(vals_ms)
    iqr_lo = _q(vals_ms, 0.25)
    iqr_hi = _q(vals_ms, 0.75)
    cpu_med = statistics.median(vals_cpu)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "method", "runs",
                    "ms_per_decision_median", "ms_per_decision_iqr_lo",
                    "ms_per_decision_iqr_hi", "policy_cpu_s_median"])
        w.writerow(["A6_obs005", METHOD_FRONTIER_RICHNESS_ANGULAR, args.runs,
                    f"{med:.4f}", f"{iqr_lo:.4f}", f"{iqr_hi:.4f}",
                    f"{cpu_med:.4f}"])

    print(f"\n== Richness-Angular ms/decision: median = {med:.4f} "
          f"(IQR [{iqr_lo:.4f}, {iqr_hi:.4f}], n={args.runs})")
    print(f"policy_cpu_s median = {cpu_med:.4f}")
    print(f"Wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
