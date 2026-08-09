"""
experiments/benchmark_cpu_ra_cu.py — Interleaved RA/CU CPU benchmark.

Alternates Coverage-U and Richness-Angular runs (same env seeds, same
session, same background load), then reports the same-batch per-run ratio
RA/CU (load-immune) plus the medians. Writes results/ra_cu_cpu_interleaved.csv
and prints the paired series.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import METHOD_COVERAGE_U, METHOD_FRONTIER_RICHNESS_ANGULAR
from experiments._runner import run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(ROOT, "results", "ra_cu_cpu_interleaved.csv")

KW = dict(grid_size=100, num_agents=6, obstacle_ratio=0.05, max_steps=1600)


def _run(method, seed):
    res = run_episode(method, "comm_limited", seed - 1000, env_seed=seed,
                      timing=True, **KW)
    return res.get("ms_per_decision"), res.get("policy_cpu_s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--order", type=str, default="CU,RA",
                    help="comma-separated: first method each pair is CU or RA")
    args = ap.parse_args()

    order = [x.strip() for x in args.order.split(",")]
    assert set(order) == {"CU", "RA"} and len(order) == 2, order
    methods = {"CU": METHOD_COVERAGE_U, "RA": METHOD_FRONTIER_RICHNESS_ANGULAR}

    import csv
    rows = []
    pairs_ms = []
    pairs_cpu = []
    t_start = time.perf_counter()
    for p in range(1, args.pairs + 1):
        pair = {}
        for m in order:
            seed = 1000 + (p - 1) * 2 + (0 if m == order[0] else 1)
            t0 = time.perf_counter()
            ms, cpu = _run(methods[m], seed)
            wall = time.perf_counter() - t0
            pair[m] = (ms, cpu)
            print(f"[pair {p}] {m} (seed {seed}) done in {wall:.1f}s "
                  f"ms/dec={ms:.4f}", flush=True)
        if pair["RA"][0] and pair["CU"][0]:
            rows.append((p, pair["CU"][0], pair["RA"][0],
                         pair["RA"][0] / pair["CU"][0],
                         pair["CU"][1], pair["RA"][1]))
            pairs_ms.append((pair["CU"][0], pair["RA"][0]))
            pairs_cpu.append(pair["RA"][1] / pair["CU"][1])

    ratios = [r[3] for r in rows]
    med_ratio = statistics.median(ratios)
    cu_ms = statistics.median(r[1] for r in rows)
    ra_ms = statistics.median(r[2] for r in rows)
    med_ratio_cpu = statistics.median(pairs_cpu)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "CU_ms_per_decision", "RA_ms_per_decision",
                    "ratio_RA_over_CU", "CU_policy_cpu_s", "RA_policy_cpu_s"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["MEDIAN", f"{cu_ms:.4f}", f"{ra_ms:.4f}",
                    f"{med_ratio:.4f}", "", f"{med_ratio_cpu:.4f}"])

    print("\n== Same-batch interleaved (load-immune) ==")
    print(f"CU median: {cu_ms:.4f} ms/dec | RA median: {ra_ms:.4f} ms/dec")
    print(f"RA/CU per-run ratio median: {med_ratio:.3f}  "
          f"(policy_cpu ratio: {med_ratio_cpu:.3f})")
    print(f"Wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
