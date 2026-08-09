"""
experiments/benchmark_cpu.py — CPU/decision benchmark (reviewer point #4).

Serial: run_episode(..., timing=True) on a fixed budget regime (A6_obs005:
6 agents, 5% obstacles, 1600 steps) for Frontier-Bounded / Coverage-U /
Entropy-Frac / Random, 10 runs each. Aggregates ms_per_decision and
policy_cpu_s per method. Writes results/benchmark_cpu.csv + prints the table.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._runner import run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(ROOT, "results", "benchmark_cpu.csv")

METHODS = ["Frontier-Bounded", "Coverage-U", "Entropy-Frac", "Random",
           "GDOP", "Hybrid"]

# A6_obs005 regime
KW = dict(grid_size=100, num_agents=6, obstacle_ratio=0.05, max_steps=1600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    import csv
    rows = []
    t_start = time.perf_counter()
    for method in METHODS:
        vals_ms = []
        vals_cpu = []
        for r in range(1, args.runs + 1):
            t0 = time.perf_counter()
            res = run_episode(method, "comm_limited", r, env_seed=1000 + r,
                              timing=True, **KW)
            ms = res.get("ms_per_decision")
            cpu = res.get("policy_cpu_s")
            vals_ms.append(ms if ms is not None else float("nan"))
            vals_cpu.append(cpu if cpu is not None else float("nan"))
            wall = time.perf_counter() - t0
            print(f"[{method}] run {r}/{args.runs} done in {wall:.1f}s "
                  f"ms/dec={ms:.3f}" if ms is not None else
                  f"[{method}] run {r}/{args.runs} done in {wall:.1f}s "
                  f"ms/dec=N/A", flush=True)
        n = len(vals_ms)
        import statistics
        med_ms = statistics.median(vals_ms)
        med_cpu = statistics.median(vals_cpu)
        rows.append({"method": method, "runs": n,
                     "ms_per_decision_median": round(med_ms, 4),
                     "policy_cpu_s_median": round(med_cpu, 4)})
        print(f"== {method}: ms_per_decision median = {med_ms:.4f} "
              f"(n={n})", flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nCPU benchmark done in {time.perf_counter() - t_start:.0f}s")
    print(f"Wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
