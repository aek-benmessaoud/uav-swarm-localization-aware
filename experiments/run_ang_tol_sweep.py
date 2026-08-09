"""
experiments/run_ang_tol_sweep.py — ANG_TOL sensitivity sweep (friend-review P1).

Re-runs the config-count members Richness-Angular (RA) and Coverage-U (CU)
under ANG_TOL_DEG in {5, 10, 20, 30} on the flagship 5%-obstacle regimes
A3_obs005 and A6_obs005, n = 40 paired (same env seeds as budget_*). The
ANG_TOL = 15 deg baseline and the Frontier-Bounded (FB) control are reused
from the confirmed `results/budget_{A3,A6}_obs005` (FB does not read angular
configurations, so it is ANG_TOL-invariant).

ANG_TOL_DEG is injected through the ANG_TOL_DEG environment variable, which
config.py reads at import time; spawned workers inherit it and re-import
config -> estimators/angular -> env with the overridden threshold.

Output dirs: results/budget_{regime}__at{tol} with raw CSVs tagged __at{tol}.

Protocol (analysis/ang_tol_sweep_stats.py): paired Wilcoxon
(zero_method="wilcox"), Holm-Bonferroni across the ANG_TOL levels per family.

Usage: python experiments/run_ang_tol_sweep.py [--workers N] [--runs 40]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (INFO_MODEL, METHOD_FRONTIER_BOUNDED,
                    METHOD_COVERAGE_U, METHOD_FRONTIER_RICHNESS_ANGULAR,
                    NUM_WORKERS)
from experiments._runner import run_experiment_set
from experiments.run_budget import REGIMES, gate_ok

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")

ANG_TOL_LEVELS = [5.0, 10.0, 20.0, 30.0]
SWEEP_REGIMES = ["A3_obs005", "A6_obs005"]
METHODS_ANG_TOL = [METHOD_FRONTIER_RICHNESS_ANGULAR, METHOD_COVERAGE_U]


def _tol_tag(tol):
    return f"at{int(tol)}"


def _kwargs(rg):
    return dict(grid_size=100, num_agents=rg["num_agents"],
                fov_radius=5, obstacle_ratio=rg["obstacle_ratio"],
                max_steps=int(rg["budget"]), sigma_loc=0.0,
                sigma_bearing=0.0, comm_range=None,
                quality_sample_k=None, quality_target=None,
                topology="random", maze_loop_density=0.0, horizon=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--tols", nargs="+", type=float, default=ANG_TOL_LEVELS)
    args = ap.parse_args()

    if not gate_ok():
        return 1
    num_workers = args.workers if args.workers is not None else NUM_WORKERS
    regimes = [r for r in REGIMES if r["label"] in SWEEP_REGIMES]

    t_start = time.perf_counter()
    for tol in args.tols:
        os.environ["ANG_TOL_DEG"] = str(tol)
        tag = f"__{_tol_tag(tol)}"
        for rg in regimes:
            out_dir = os.path.join(RESULTS_DIR,
                                   f"budget_{rg['label']}__{_tol_tag(tol)}")
            for method in METHODS_ANG_TOL:
                print(f"\n=== ANG_TOL={tol} {method} {rg['label']} "
                      f"({args.runs} runs, {num_workers} workers) ===",
                      flush=True)
                run_experiment_set(method, INFO_MODEL, args.runs, out_dir,
                                   tag=tag, num_workers=num_workers,
                                   **_kwargs(rg))

    print(f"\nANG_TOL sweep done in {time.perf_counter() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
