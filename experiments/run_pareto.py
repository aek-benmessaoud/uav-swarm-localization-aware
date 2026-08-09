"""
experiments/run_pareto.py — E4-PARETO: lambda sensitivity / accuracy-coverage
Pareto frontier for Coverage-U under the fixed budget T.

Only the two regimes with the strongest confirmed effect are run (A3_obs005,
A6_obs005); FB (lambda = 0) and Coverage-U at lambda = 0.5 already exist at
seeds 0-39 in results/budget_{regime} and are NOT re-run. New runs:
lambda in {0.25, 1.0, 2.0} x 2 regimes x 40 seeds = 240 episodes, written to
results/pareto_{regime}/raw_comm_limited__Coverage-U__lam{lam}.csv.

Pairing: same env_seed per run index as budget_* (env_seed_for_run in the
shared runner), so the whole lambda grid shares one seed ladder.

REFUSES TO RUN unless gates/phase1_GO.txt exists.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (INFO_MODEL, GRID_SIZE, DEFAULT_FOV_RADIUS,
                    QUALITY_SAMPLE_K, QUALITY_TARGET, METHOD_COVERAGE_U,
                    NUM_WORKERS)
from experiments._runner import run_experiment_set

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATES_DIR = os.path.join(ROOT, "gates")
RESULTS_DIR = os.path.join(ROOT, "results")

LAMBDAS = [0.25, 1.0, 2.0]

# Budgets T = 0.7 x FB median steps_90 (E3), identical to run_budget.py.
REGIMES = [
    {"label": "A3_obs005", "num_agents": 3, "obstacle_ratio": 0.05,
     "budget": 3200},
    {"label": "A6_obs005", "num_agents": 6, "obstacle_ratio": 0.05,
     "budget": 1600},
]


def gate_ok():
    go = os.path.join(GATES_DIR, "phase1_GO.txt")
    if not os.path.exists(go):
        print("HARD GATE BLOCKED: campaign requires Phase 1a validation first.")
        print(f"  missing: {go}")
        return False
    print(f"Gate OK: {go}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", nargs="+", type=float, default=LAMBDAS)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--grid", type=int, default=GRID_SIZE)
    ap.add_argument("--fov", type=int, default=DEFAULT_FOV_RADIUS)
    ap.add_argument("--regimes", nargs="+",
                    default=[r["label"] for r in REGIMES])
    ap.add_argument("--quality-sample-k", type=int, default=QUALITY_SAMPLE_K)
    ap.add_argument("--quality-target", type=float, default=QUALITY_TARGET)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    if not gate_ok():
        return 1

    selected = [r for r in REGIMES if r["label"] in args.regimes]
    t_start = time.perf_counter()
    for rg in selected:
        kwargs = dict(grid_size=args.grid,
                      num_agents=rg["num_agents"],
                      fov_radius=args.fov,
                      obstacle_ratio=rg["obstacle_ratio"],
                      max_steps=rg["budget"],
                      quality_sample_k=args.quality_sample_k,
                      quality_target=args.quality_target)
        print(f"\n===== regime {rg['label']} (agents={rg['num_agents']}, "
              f"obs={rg['obstacle_ratio']}, budget={rg['budget']}) =====",
              flush=True)
        for lam in args.lambdas:
            out_dir = os.path.join(RESULTS_DIR, f"pareto_{rg['label']}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"=== {METHOD_COVERAGE_U} lam={lam} ({args.runs} runs) ===",
                  flush=True)
            run_experiment_set(METHOD_COVERAGE_U, INFO_MODEL, args.runs,
                               out_dir, tag=f"__lam{lam}", lam=lam,
                               num_workers=args.workers, **kwargs)

    print(f"\nPareto campaign done in {time.perf_counter() - t_start:.0f}s")
    print("Results in results/pareto_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
