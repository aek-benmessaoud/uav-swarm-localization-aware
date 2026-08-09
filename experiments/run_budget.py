"""
experiments/run_budget.py — E4: U-prioritized coverage under a finite budget.

Budget-limited missions: episode stops at T = 0.7 x FB median steps_90 (from
E3), a region where coverage is partial (66-76%) and quality is still climbing,
so accuracy and coverage genuinely compete. Paired A/B (same env seed per run).

Methods:
  Frontier-Bounded   — control (validated movement frame).
  Deploy-U           — E3 method (context row).
  Coverage-U         — NEW: continuous under-observed bonus in target selection
                       (lam = 0.5); lam = 0 is exactly Frontier-Bounded.
  CentralOracle-CRLB / -Config — E5 CENTRALIZED ORACLE control rows (global
                       perfect knowledge, same scoring form). Run the same way,
                       same budgets, same paired seeds; results land in the
                       same results/budget_* dirs as raw_comm_limited__*.csv.

Follow-up sweeps (FUTURE_WORK_NOTES.md, phases A-D) — same runner, extra flags:
  --sigma-loc FLOAT    self-localization noise (std in grid cells) applied to
                       the LOCAL decision signal only; the oracle CRLB keeps
                       TRUE geometry (env.py:438). Default 0.0.
  --sigma-bearing FLOAT
                       bearing-measurement noise (std in degrees) applied to
                       the LOCAL angular configurations (env.py:470) only; the
                       oracle CRLB and global clusters keep TRUE geometry.
                       Default 0.0.
  --comm-range FLOAT   communication/fusion range in grid cells (default:
                       config COMM_RANGE = FOV). Range sweep for the locality
                       stress test (phase D).
  --budget-frac FLOAT  budget T as a fraction of FB median steps_90
                       (default 0.7). Budget sweep (phase C).
  --variant NAME       appends __{variant} to the output dir and __{variant}
                       to every raw CSV filename, so sweep variants never
                       collide with the confirmed budget_* data.

Metrics (already in the runner): quality_auc @ T (primary), final_coverage @ T,
mean_bound_final @ T, undetermined_final @ T, steps_dual, coverage_auc.

REFUSES TO RUN unless gates/phase1_GO.txt exists.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (INFO_MODEL, GRID_SIZE, DEFAULT_FOV_RADIUS,
                    QUALITY_SAMPLE_K, QUALITY_TARGET,
                    METHOD_FRONTIER_BOUNDED, METHOD_DEPLOY, METHOD_COVERAGE_U,
                    NUM_WORKERS)
from experiments._runner import run_experiment_set

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATES_DIR = os.path.join(ROOT, "gates")
RESULTS_DIR = os.path.join(ROOT, "results")

# Budget T per regime = 0.7 x FB median steps_90 measured in E3 (preregistered).
REGIMES = [
    {"label": "A2_obs005", "num_agents": 2, "obstacle_ratio": 0.05,
     "budget": 4200},
    {"label": "A3_obs005", "num_agents": 3, "obstacle_ratio": 0.05,
     "budget": 3200},
    {"label": "A6_obs005", "num_agents": 6, "obstacle_ratio": 0.05,
     "budget": 1600},
    {"label": "A6_obs020", "num_agents": 6, "obstacle_ratio": 0.20,
     "budget": 1750},
    # Maze topology: obstacle_ratio is meaningless (walls ~50% by construction),
    # kept for the schema. Budget MUST be recalibrated per map type (probe:
    # 0.7 x FB median steps_90 on maze maps) before any campaign.
    {"label": "A6_maze", "num_agents": 6, "obstacle_ratio": 0.50,
     "budget": None, "topology": "maze"},
    # Cluster topology: contiguous obstacle blocks at the given density
    # (real LOS occlusion, routing preserved). Budget MUST be recalibrated
    # per map type (probe: 0.7 x FB median steps_90) before any campaign.
    {"label": "A6_cluster020", "num_agents": 6, "obstacle_ratio": 0.20,
     "budget": None, "topology": "cluster"},
    {"label": "A3_cluster020", "num_agents": 3, "obstacle_ratio": 0.20,
     "budget": None, "topology": "cluster"},
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
                    default=[METHOD_FRONTIER_BOUNDED, METHOD_DEPLOY,
                             METHOD_COVERAGE_U])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--grid", type=int, default=GRID_SIZE)
    ap.add_argument("--fov", type=int, default=DEFAULT_FOV_RADIUS)
    ap.add_argument("--regimes", nargs="+",
                    default=[r["label"] for r in REGIMES])
    ap.add_argument("--quality-sample-k", type=int, default=QUALITY_SAMPLE_K)
    ap.add_argument("--quality-target", type=float, default=QUALITY_TARGET)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--sigma-loc", type=float, default=0.0)
    ap.add_argument("--sigma-bearing", type=float, default=0.0)
    ap.add_argument("--comm-range", type=float, default=None)
    ap.add_argument("--budget-frac", type=float, default=0.7)
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--budget", type=int, default=None,
                    help="Budget T override (steps). Required for the A6_maze "
                         "regime whose E3 budget does not exist.")
    ap.add_argument("--loop-density", type=float, default=0.0,
                    help="Maze only: fraction of non-tree walls reopened after "
                         "Kruskal (loops). 0.0 = perfect maze (default).")
    ap.add_argument("--horizon", type=int, default=8,
                    help="bounded-BFS lookahead horizon (ENTROPY_HORIZON, "
                         "default 8).")
    args = ap.parse_args()

    if not gate_ok():
        return 1

    num_workers = args.workers if args.workers is not None else NUM_WORKERS

    selected = [r for r in REGIMES if r["label"] in args.regimes]
    t_start = time.perf_counter()
    for rg in selected:
        variant = args.variant or ""
        out_dir = os.path.join(
            RESULTS_DIR,
            f"budget_{rg['label']}" + (f"__{variant}" if variant else ""))
        os.makedirs(out_dir, exist_ok=True)
        if rg.get("budget") is None:
            if args.budget is None:
                print(f"ABORT: regime {rg['label']} needs --budget "
                      f"(recalibrate via probe: 0.7 x FB median steps_90).")
                return 2
            budget = int(args.budget)
        else:
            budget = int(round(rg["budget"] * args.budget_frac / 0.7))
        kwargs = dict(grid_size=args.grid,
                      num_agents=rg["num_agents"],
                      fov_radius=args.fov,
                      obstacle_ratio=rg["obstacle_ratio"],
                      max_steps=budget,
                      sigma_loc=args.sigma_loc,
                      sigma_bearing=args.sigma_bearing,
                      comm_range=args.comm_range,
                      quality_sample_k=args.quality_sample_k,
                      quality_target=args.quality_target,
                      topology=rg.get("topology", "random"),
                      maze_loop_density=args.loop_density,
                      horizon=args.horizon)
        print(f"\n===== regime {rg['label']} (agents={rg['num_agents']}, "
              f"obs={rg['obstacle_ratio']}, topology={rg.get('topology', 'random')}, "
              f"loops={args.loop_density}, "
              f"budget={budget} "
              f"= {args.budget_frac:.2f}xFB, sigma_loc={args.sigma_loc}, "
              f"sigma_bearing={args.sigma_bearing}, "
              f"comm={args.comm_range}) =====", flush=True)
        for method in args.methods:
            print(f"=== {method} ({args.runs} runs, "
                  f"{num_workers} workers) ===", flush=True)
            run_experiment_set(method, INFO_MODEL, args.runs, out_dir,
                               tag=f"__{variant}" if variant else "",
                               num_workers=num_workers, **kwargs)

    print(f"\nBudget campaign done in {time.perf_counter() - t_start:.0f}s")
    print("Results in results/budget_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
