"""
experiments/run_phase1.py — Phase-1 campaign (localization-quality A/B).

REFUSES TO RUN unless gates/phase1_GO.txt exists (Phase 1a must validate the
model first — hard barrier). Runs NUM_RUNS paired episodes per Phase-1 method
(Random, Frontier-Bounded, Richness-Angular), writing each raw result
incrementally to results/raw_{info_model}__{method}.csv (resumable).

Each episode evaluates BOTH the movement frame (coverage) and the
localization quality: quality(t) sampled every K steps, normalized AUC,
time-to-threshold, cluster_cap_hit_frac (decision-signal saturation) and
undetermined_final (oracle unobserved fraction).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (PHASE1_METHODS, NUM_RUNS, INFO_MODEL, GRID_SIZE,
                    MAX_STEPS, NUM_AGENTS, DEFAULT_FOV_RADIUS,
                    DEFAULT_OBSTACLE_RATIO, QUALITY_SAMPLE_K, QUALITY_TARGET)
from experiments._runner import run_experiment_set

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATES_DIR = os.path.join(ROOT, "gates")
RESULTS_DIR = os.path.join(ROOT, "results")


def gate_ok():
    go = os.path.join(GATES_DIR, "phase1_GO.txt")
    if not os.path.exists(go):
        print("HARD GATE BLOCKED: Phase-1 campaign requires Phase 1a "
              "validation first.")
        print(f"  missing: {go}")
        print("  run:  python experiments/run_phase1a.py")
        return False
    print(f"Gate OK: {go}")
    with open(go, "r", encoding="utf-8") as f:
        for line in f:
            print("   ", line.rstrip())
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=PHASE1_METHODS)
    ap.add_argument("--runs", type=int, default=NUM_RUNS)
    ap.add_argument("--grid", type=int, default=GRID_SIZE)
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--agents", type=int, default=NUM_AGENTS)
    ap.add_argument("--fov", type=int, default=DEFAULT_FOV_RADIUS)
    ap.add_argument("--obstacle-ratio", type=float,
                    default=DEFAULT_OBSTACLE_RATIO)
    ap.add_argument("--quality-sample-k", type=int, default=QUALITY_SAMPLE_K)
    ap.add_argument("--quality-target", type=float, default=QUALITY_TARGET)
    ap.add_argument("--out-dir", default=RESULTS_DIR,
                    help="Results directory (separate per scenario so raw CSVs "
                         "are not resumed across scenarios).")
    args = ap.parse_args()

    if not gate_ok():
        return 1
    os.makedirs(args.out_dir, exist_ok=True)

    kwargs = dict(grid_size=args.grid, num_agents=args.agents,
                  fov_radius=args.fov, obstacle_ratio=args.obstacle_ratio,
                  max_steps=args.steps,
                  quality_sample_k=args.quality_sample_k,
                  quality_target=args.quality_target)

    t_start = time.perf_counter()
    for method in args.methods:
        print(f"\n=== Phase 1: {method} ({args.runs} runs) ===", flush=True)
        run_experiment_set(method, INFO_MODEL, args.runs, args.out_dir,
                           **kwargs)

    print(f"\nPhase 1 done in {time.perf_counter() - t_start:.0f}s")
    print(f"Results in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
