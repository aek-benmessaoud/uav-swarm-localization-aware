"""
experiments/run_deploy.py — E3: Deploy-U vs Frontier-Bounded (paired A/B).

Research idea 1+3 (flanking formations + station-keeping) tested against the
validated movement control Frontier-Bounded, on the angular-localization
scenario. Runs paired episodes per regime (agents x obstacle ratio), writing
raw results incrementally to results/deploy_{label}/raw_{info_model}__{method}.csv
(resumable, like run_phase1).

Primary metric (pre-registered): steps_dual = first step where BOTH coverage
>= 90% AND quality >= QUALITY_TARGET (the title's "Accuracy AND Coverage").
Secondary: quality_auc, time_to_quality, coverage_auc, mean_bound_final.

REFUSES TO RUN unless gates/phase1_GO.txt exists (Phase 1a barrier).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (INFO_MODEL, GRID_SIZE, MAX_STEPS, DEFAULT_FOV_RADIUS,
                    DEFAULT_OBSTACLE_RATIO, QUALITY_SAMPLE_K, QUALITY_TARGET,
                    METHOD_FRONTIER_BOUNDED, METHOD_DEPLOY)
from experiments._runner import run_experiment_set

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATES_DIR = os.path.join(ROOT, "gates")
RESULTS_DIR = os.path.join(ROOT, "results")

REGIMES = [
    {"label": "A2_obs005", "num_agents": 2, "obstacle_ratio": 0.05},
    {"label": "A3_obs005", "num_agents": 3, "obstacle_ratio": 0.05},
    {"label": "A6_obs005", "num_agents": 6, "obstacle_ratio": 0.05},
    {"label": "A6_obs020", "num_agents": 6, "obstacle_ratio": 0.20},
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
    ap.add_argument("--methods", nargs="+",
                    default=[METHOD_FRONTIER_BOUNDED, METHOD_DEPLOY])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--grid", type=int, default=GRID_SIZE)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--fov", type=int, default=DEFAULT_FOV_RADIUS)
    ap.add_argument("--regimes", nargs="+",
                    default=[r["label"] for r in REGIMES])
    ap.add_argument("--quality-sample-k", type=int, default=QUALITY_SAMPLE_K)
    ap.add_argument("--quality-target", type=float, default=QUALITY_TARGET)
    args = ap.parse_args()

    if not gate_ok():
        return 1

    selected = [r for r in REGIMES if r["label"] in args.regimes]
    t_start = time.perf_counter()
    for rg in selected:
        out_dir = os.path.join(RESULTS_DIR, f"deploy_{rg['label']}")
        os.makedirs(out_dir, exist_ok=True)
        kwargs = dict(grid_size=args.grid,
                      num_agents=rg["num_agents"],
                      fov_radius=args.fov,
                      obstacle_ratio=rg["obstacle_ratio"],
                      max_steps=args.steps,
                      quality_sample_k=args.quality_sample_k,
                      quality_target=args.quality_target)
        print(f"\n===== regime {rg['label']} "
              f"(agents={rg['num_agents']}, obs={rg['obstacle_ratio']}) =====",
              flush=True)
        for method in args.methods:
            print(f"=== {method} ({args.runs} runs) ===", flush=True)
            run_experiment_set(method, INFO_MODEL, args.runs, out_dir,
                               **kwargs)

    print(f"\nDeploy campaign done in {time.perf_counter() - t_start:.0f}s")
    print("Results in results/deploy_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
