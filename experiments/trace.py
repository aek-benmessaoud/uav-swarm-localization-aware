"""
experiments/trace.py — replay one episode and RECORD what the raw CSV cannot
hold, for the qualitative figures (Figure A: trajectories + F1 overlay,
Figure B: config-count heatmaps, Figure C: F1-fraction(t)):

  - per-step agent positions (steps+1, num_agents, 2),
  - global config-count grids at sample steps (0, T/4, T/2, 3T/4, T),
  - ambiguous-fraction trace (fraction of traversable cells with exactly 1
    global angular configuration — observed but rank-deficient) at every step.

Deterministic: identical env_seed / policy seeds reproduce the campaign run.

Usage:
  python experiments/trace.py --method "Coverage-U" --run 0 --regime A3_obs005
                              --out-dir results/traces
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import (INFO_MODEL, GRID_SIZE, DEFAULT_FOV_RADIUS,
                    QUALITY_SAMPLE_K, QUALITY_TARGET)
from experiments._runner import run_episode
from utils.seed_manager import env_seed_for_run

REGIMES = {
    "A2_obs005": {"num_agents": 2, "obstacle_ratio": 0.05, "budget": 4200},
    "A3_obs005": {"num_agents": 3, "obstacle_ratio": 0.05, "budget": 3200},
    "A6_obs005": {"num_agents": 6, "obstacle_ratio": 0.05, "budget": 1600},
    "A6_obs020": {"num_agents": 6, "obstacle_ratio": 0.20, "budget": 1750},
    # Maze topology regime (post-submission campaign): budget = campaign T
    # (0.7 x FB median steps_90 measured by probe_budget_maze.py = 1323).
    "A6_maze": {"num_agents": 6, "obstacle_ratio": 0.50, "budget": 1323,
                "topology": "maze"},
}


class TraceRecorder:
    def __init__(self, sample_steps):
        self.sample_steps = sorted(set(sample_steps))
        self.positions = []
        self.steps = []
        self.ambiguous_frac = []
        self.snap_steps = []
        self.snap_grids = []

    def on_step(self, env, step):
        self.positions.append(
            np.asarray(env.agent_positions, dtype=np.int32).copy())
        self.steps.append(int(step))
        tr = ~env.obstacle_map
        cfg = env.global_config_count_grid()
        if int(step) in self.sample_steps:
            self.snap_steps.append(int(step))
            self.snap_grids.append(cfg.copy())
        ntr = env.traversable
        self.ambiguous_frac.append(
            float(np.sum((cfg == 1) & tr)) / ntr if ntr else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--regime", required=True, choices=sorted(REGIMES))
    ap.add_argument("--out-dir", default=os.path.join("results", "traces"))
    ap.add_argument("--grid", type=int, default=GRID_SIZE)
    ap.add_argument("--fov", type=int, default=DEFAULT_FOV_RADIUS)
    ap.add_argument("--topology", type=str, default=None,
                    help="override topology (e.g. 'maze'); defaults to the "
                         "regime's topology field, else 'random'")
    args = ap.parse_args()

    rg = REGIMES[args.regime]
    T = rg["budget"]
    topology = args.topology or rg.get("topology", "random")
    sample_steps = [0, T // 4, T // 2, 3 * T // 4, T]

    rec = TraceRecorder(sample_steps)
    env_seed = env_seed_for_run(args.run)
    res = run_episode(args.method, INFO_MODEL, args.run, env_seed,
                      grid_size=args.grid, num_agents=rg["num_agents"],
                      fov_radius=args.fov, obstacle_ratio=rg["obstacle_ratio"],
                      max_steps=T, quality_sample_k=QUALITY_SAMPLE_K,
                      quality_target=QUALITY_TARGET, on_step=rec.on_step,
                      topology=topology)

    out = os.path.join(args.out_dir, args.regime,
                       f"{args.method.replace(' ', '_').replace('-', '_')}"
                       f"_run{args.run:02d}")
    os.makedirs(out, exist_ok=True)

    np.save(os.path.join(out, "positions.npy"),
            np.stack(rec.positions))
    np.savez(os.path.join(out, "config_snapshots.npz"),
             steps=np.asarray(rec.snap_steps),
             grids=np.stack(rec.snap_grids))
    with open(os.path.join(out, "f1_trace.csv"), "w") as fh:
        fh.write("step,ambiguous_frac\n")
        for s, f in zip(rec.steps, rec.ambiguous_frac):
            fh.write(f"{s},{f:.8f}\n")

    meta = {
        "method": args.method,
        "regime": args.regime,
        "run": args.run,
        "env_seed": env_seed,
        "budget": T,
        "num_agents": rg["num_agents"],
        "obstacle_ratio": rg["obstacle_ratio"],
        "topology": topology,
        "grid_size": args.grid,
        "fov_radius": args.fov,
        "sample_steps": sample_steps,
        "final_coverage": res["final_coverage"],
        "mean_bound_final": res["mean_bound_final"],
        "quality_final": res["quality_final"],
        "quality_auc": res["quality_auc"],
        "undetermined_final": res["undetermined_final"],
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[trace] {args.method} run {args.run} {args.regime} -> {out}")
    print(f"  cov={res['final_coverage']:.1f}% mb={res['mean_bound_final']:.4f} "
          f"qfin={res['quality_final']:.3f} und={res['undetermined_final']:.3f}")


if __name__ == "__main__":
    main()
