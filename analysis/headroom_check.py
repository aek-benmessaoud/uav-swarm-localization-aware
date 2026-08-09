"""
analysis/headroom_check.py — Step-0 diagnostic for the richness re-observation
controller (Project08).

Question: is there a regime where Frontier-Bounded leaves localization work on
the table at the end of an episode? If yes, the re-observation controller has
headroom to exploit. If no (F1 -> 0 everywhere), the premise fails.

Measures per (agents, obstacle_ratio) variant, over NUM_RUNS paired episodes:
  coverage      : final coverage %
  quality_final : fraction of traversable cells well-localized (CRLB <= 1.5)
  f0_frac       : traversable cells with 0 global configurations (unseen)
  f1_frac       : traversable cells with exactly 1 configuration (SEEN but not
                  localizable -> the headroom the controller would target)
  f2plus_frac   : traversable cells with >= 2 configurations
  cap_hit_frac  : cluster-cap saturation on the local decision signal

Usage:
  python analysis/headroom_check.py [--runs 10] [--steps 4500]
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
from config import (METHOD_FRONTIER_BOUNDED, GRID_SIZE, DEFAULT_FOV_RADIUS,
                    DEFAULT_OBSTACLE_RATIO, QUALITY_THRESHOLD)

VARIANT_DEFS = [
    {"label": "A2_obs005", "num_agents": 2, "obstacle_ratio": 0.05},
    {"label": "A3_obs005", "num_agents": 3, "obstacle_ratio": 0.05},
    {"label": "A6_obs020", "num_agents": 6, "obstacle_ratio": 0.20},
]


def run_one(run_index, num_agents, obstacle_ratio, steps, fov_radius):
    env_seed = env_seed_for_run(run_index)
    env = GridEnv(grid_size=GRID_SIZE, num_agents=num_agents,
                  obstacle_ratio=obstacle_ratio, seed=env_seed,
                  info_model="comm_limited")
    env.quality_threshold = QUALITY_THRESHOLD

    policies = [
        build_policy(METHOD_FRONTIER_BOUNDED, seed=env_seed * 1000 + i + 1,
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

    counts = env.global_config_count_grid()
    tr = ~env.obstacle_map
    n_trav = int(tr.sum()) or 1
    f0 = int(np.sum((counts == 0) & tr)) / n_trav
    f1 = int(np.sum((counts == 1) & tr)) / n_trav
    f2p = int(np.sum((counts >= 2) & tr)) / n_trav

    return {
        "coverage": coverage_percent(env),
        "quality_final": env.quality_well_localized(),
        "f0_frac": f0,
        "f1_frac": f1,
        "f2plus_frac": f2p,
        "cap_hit_frac": env.cluster_cap_hit_frac(),
        "wall_time_s": wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=4500)
    ap.add_argument("--fov", type=int, default=DEFAULT_FOV_RADIUS)
    ap.add_argument("--out", default="results/headroom_variants.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["variant", "num_agents", "obstacle_ratio", "run",
                  "coverage", "quality_final", "f0_frac", "f1_frac",
                  "f2plus_frac", "cap_hit_frac", "wall_time_s"]
    new_file = not os.path.exists(args.out)
    done = set()
    if not new_file:
        with open(args.out, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["variant"], int(row["run"])))
    fh = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()

    for vd in VARIANT_DEFS:
        print(f"=== {vd['label']}  agents={vd['num_agents']} "
              f"obstacle={vd['obstacle_ratio']} ===", flush=True)
        rows = []
        for run in range(args.runs):
            if (vd["label"], run) in done:
                continue
            r = run_one(run, vd["num_agents"], vd["obstacle_ratio"],
                        args.steps, args.fov)
            row = {"variant": vd["label"], "num_agents": vd["num_agents"],
                   "obstacle_ratio": vd["obstacle_ratio"], "run": run, **r}
            out_row = dict(row)
            for k in ("coverage", "quality_final", "f0_frac", "f1_frac",
                      "f2plus_frac", "cap_hit_frac", "wall_time_s"):
                v = row[k]
                out_row[k] = "" if v is None else round(float(v), 6)
            writer.writerow(out_row)
            fh.flush()
            rows.append(r)
            print(f"    run {run + 1}/{args.runs}: cov={r['coverage']:.1f} "
                  f"q={r['quality_final']:.3f} f1={r['f1_frac']:.3f} "
                  f"in {r['wall_time_s']:.0f}s", flush=True)
        if rows:
            print(f"  MEAN over {args.runs}: cov={np.mean([r['coverage'] for r in rows]):.1f} "
                  f"quality_final={np.mean([r['quality_final'] for r in rows]):.3f} "
                  f"f1={np.mean([r['f1_frac'] for r in rows]):.3f} "
                  f"f0={np.mean([r['f0_frac'] for r in rows]):.3f} "
                  f"f2plus={np.mean([r['f2plus_frac'] for r in rows]):.3f}", flush=True)
        else:
            print(f"  (all runs already in {args.out}, skipped)", flush=True)
    fh.close()
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
