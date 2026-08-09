"""
experiments/run_phase1a.py — Empirical validation of the localization model
(Phase 1a). RUN THIS BEFORE the Phase-1 campaign.

For each of PHASE1A_RUNS paired episodes (Frontier-Bounded movement frame,
the geometric baseline), record every true bearing observation, then run the
Gauss-Newton estimator under angular noise sigma_ref on each traversable cell
and correlate, cell by cell:
    rho(bound, error)   : CRLB oracle vs empirical error (localizable cells)
    rho(U_local, error) : config-count decision signal vs empirical error
                          (all traversable cells)

HARD GATE: if the pooled criteria pass (see constants below), writes
    gates/phase1_GO.txt
and the Phase-1 campaign refuses to run without it. On failure it writes
    gates/phase1_NOGO.txt  (with the numbers) and exits non-zero.
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env import GridEnv
from metrics import coverage_percent
from policies.factory import build_policy
from utils.seed_manager import env_seed_for_run
from config import (QUALITY_SIGMA_BEARING_DEG, METHOD_FRONTIER_BOUNDED,
                    GRID_SIZE, NUM_AGENTS, DEFAULT_FOV_RADIUS,
                    DEFAULT_OBSTACLE_RATIO)

from estimators.validation import validate_phase1a, cell_data, spearman

# -------------------- Phase 1a configuration --------------------
PHASE1A_RUNS = 10
PHASE1A_STEPS = 1500
PHASE1A_POLICY = METHOD_FRONTIER_BOUNDED

# Gate criteria (pooled across runs). The CRLB must track empirical error
# among localizable cells; the local config count must track error across the
# whole map (more configurations -> lower error).
GATE_RHO_BOUND_ERROR_MIN = 0.5
GATE_RHO_U_ERROR_MAX = -0.4
GATE_ALPHA = 0.05

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATES_DIR = os.path.join(ROOT, "gates")
RESULTS_DIR = os.path.join(ROOT, "results")


def run_one(run_index, steps, grid_size, num_agents, fov_radius,
            obstacle_ratio):
    """One Phase-1a episode. Returns (env, wall_time_s)."""
    env_seed = env_seed_for_run(run_index)
    env = GridEnv(grid_size=grid_size, num_agents=num_agents,
                  obstacle_ratio=obstacle_ratio, seed=env_seed,
                  info_model="comm_limited")
    env.record_global_raw = True

    policies = [
        build_policy(PHASE1A_POLICY, seed=env_seed * 1000 + i + 1,
                     fov_radius=fov_radius, horizon=8)
        for i in range(num_agents)
    ]

    t0 = time.perf_counter()
    for _ in range(steps):
        env.check_and_merge()
        actions = [policies[i].select_action(env, i)[0]
                   for i in range(num_agents)]
        env.step(actions)
        if coverage_percent(env) >= 100.0:
            break
    wall = time.perf_counter() - t0
    return env, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=PHASE1A_RUNS)
    ap.add_argument("--steps", type=int, default=PHASE1A_STEPS)
    ap.add_argument("--grid", type=int, default=GRID_SIZE)
    ap.add_argument("--agents", type=int, default=NUM_AGENTS)
    ap.add_argument("--fov", type=int, default=DEFAULT_FOV_RADIUS)
    ap.add_argument("--obstacle-ratio", type=float,
                    default=DEFAULT_OBSTACLE_RATIO)
    args = ap.parse_args()

    os.makedirs(GATES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Per-run rows + pooled cell data across ALL runs (one episode per seed).
    pool_bound, pool_u, pool_ug, pool_err = [], [], [], []
    done = set()
    csv_path = os.path.join(RESULTS_DIR, "phase1a_validation.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(int(row["run"]))

    for run in range(args.runs):
        if run in done:
            continue
        env, wall = run_one(run, args.steps, args.grid, args.agents, args.fov,
                            args.obstacle_ratio)
        rng = np.random.default_rng(1000 + run)
        stats = validate_phase1a(env, rng)
        b, u, ug, e = cell_data(env, rng, include_undetermined=True)
        pool_bound.extend(b.tolist())
        pool_u.extend(u.tolist())
        pool_ug.extend(ug.tolist())
        pool_err.extend(e.tolist())

        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "run", "n_cells", "n_localizable",
                "rho_bound_error", "rho_bound_error_p",
                "rho_U_local_error", "rho_U_local_error_p",
                "rho_U_global_error", "rho_U_global_error_p", "wall_time_s"])
            if new_file:
                w.writeheader()
            w.writerow({k: stats.get(k, "") for k in [
                "run", "n_cells", "n_localizable",
                "rho_bound_error", "rho_bound_error_p",
                "rho_U_local_error", "rho_U_local_error_p",
                "rho_U_global_error", "rho_U_global_error_p"]} |
                {"run": run, "wall_time_s": wall})
        print(f"    run {run + 1}/{args.runs} done in {wall:.1f}s  "
              f"rho_bE={stats['rho_bound_error']:.3f} "
              f"rho_UlE={stats['rho_U_local_error']:.3f} "
              f"rho_UgE={stats['rho_U_global_error']:.3f}", flush=True)

    if not pool_bound:
        print("No new runs computed; nothing to gate on. "
              f"Remove {csv_path} to recompute.")
        return 1

    n_cells = len(pool_err)
    n_loc = int(sum(np.isfinite(pool_err)))
    rho_be, p_be = spearman(pool_bound, pool_err)
    rho_ule, p_ule = spearman(pool_u, pool_err)
    rho_uge, p_uge = spearman(pool_ug, pool_err)

    # Localizable-only bound/error correlation (honest CRLB check).
    fin = np.isfinite(pool_err)
    if int(fin.sum()) >= 3:
        rho_be_loc, p_be_loc = spearman(
            np.array(pool_bound)[fin], np.array(pool_err)[fin])
    else:
        rho_be_loc, p_be_loc = float("nan"), 1.0

    print("\n=== Phase 1a pooled validation ===")
    print(f"  cells tested : {n_cells}  (localizable: {n_loc})")
    print(f"  rho(bound, error)           = {rho_be:.3f} (p={p_be:.2e})")
    print(f"  rho(bound, error) |localiz  = {rho_be_loc:.3f} "
          f"(p={p_be_loc:.2e})")
    print(f"  rho(U_local, error)         = {rho_ule:.3f} (p={p_ule:.2e})")
    print(f"  rho(U_global, error)        = {rho_uge:.3f} (p={p_uge:.2e})")

    ok_be = (np.isfinite(rho_be_loc) and rho_be_loc >= GATE_RHO_BOUND_ERROR_MIN
             and p_be_loc < GATE_ALPHA)
    ok_ue = (np.isfinite(rho_ule) and rho_ule <= GATE_RHO_U_ERROR_MAX
             and p_ule < GATE_ALPHA)

    gate_path = os.path.join(GATES_DIR, "phase1_GO.txt")
    nogo_path = os.path.join(GATES_DIR, "phase1_NOGO.txt")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Phase 1a validation gate",
        f"generated : {stamp}",
        f"runs      : {args.runs} x {args.steps} steps (grid {args.grid})",
        f"policy    : {PHASE1A_POLICY}",
        f"cells     : {n_cells}  localizable: {n_loc}",
        f"sigma_bearing = {QUALITY_SIGMA_BEARING_DEG} deg",
        f"rho(bound,error)|localiz = {rho_be_loc:.3f} "
        f"(min {GATE_RHO_BOUND_ERROR_MIN}, p<{GATE_ALPHA})",
        f"rho(U_local,error)       = {rho_ule:.3f} "
        f"(max {GATE_RHO_U_ERROR_MAX}, p<{GATE_ALPHA})",
        f"status    : {'GO' if ok_be and ok_ue else 'NOGO'}",
    ]
    if ok_be and ok_ue:
        with open(gate_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        if os.path.exists(nogo_path):
            os.remove(nogo_path)
        print(f"\nGATE PASSED -> wrote {gate_path}")
        return 0
    with open(nogo_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nGATE FAILED -> wrote {nogo_path}")
    print("Fix the model/estimator and re-run before Phase 1.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
