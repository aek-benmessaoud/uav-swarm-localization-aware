"""
analysis/probe_hybrid_theta.py — PRE-RUN probe for the Hybrid policy threshold.

Measures the per-agent per-step window free-fraction
    free_frac = _box_sum(~obs_local, fov_radius, agent_r, agent_c) / FOV_area
over a few representative episodes, for both candidate regimes, and reports
the empirical trigger rates P(free_frac < theta) at theta = 0.5 and 0.8.
Used ONLY to freeze theta BEFORE the hybrid campaign (documented pre-run).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import METHOD_COVERAGE_U_NORM
from experiments._runner import run_episode
from policies.hybrid import _window_free_frac


def free_frac_hook(env, step, fov_radius, out):
    out["steps"] += 1
    for aid in range(env.num_agents):
        r, c = env.agent_positions[aid]
        env.update_local_memory(aid, fov_radius)
        info = env.get_local_info(aid)
        out["fracs"].append(
            _window_free_frac(info["obs"], int(r), int(c), fov_radius))


def probe(regime, episodes=4, fov_radius=5, horizon=8):
    obr, n_agents, steps = {
        "A3_obs005": (0.05, 3, 3200),
        "A6_obs005": (0.05, 6, 1600),
        "A6_obs020": (0.20, 6, 1600),
    }[regime]
    acc = {"steps": 0, "fracs": []}
    for run in range(1, episodes + 1):
        run_episode(
            METHOD_COVERAGE_U_NORM, "comm_limited", run, env_seed=1000 + run,
            fov_radius=fov_radius, horizon=horizon, grid_size=100,
            num_agents=n_agents, obstacle_ratio=obr, max_steps=steps,
            on_step=lambda env, s, acc=acc:
                free_frac_hook(env, s, fov_radius, acc),
        )
    f = np.array(acc["fracs"], dtype=float)
    q = np.percentile(f, [0, 5, 10, 25, 50, 75, 90, 100])
    print(f"[{regime}] episodes={episodes} steps={acc['steps']} "
          f"samples={f.size}")
    print(f"  free_frac q[0,5,10,25,50,75,90,100] = "
          f"{', '.join(f'{x:.3f}' for x in q)}")
    for th in (0.5, 0.8):
        p = float(np.mean(f < th))
        print(f"  P(free_frac < {th}) = {p:.4f} "
              f"({p * f.size:.0f}/{f.size} samples) -> RA mode")
    return f


if __name__ == "__main__":
    print("# Hybrid theta probe (PRE-RUN; freeze theta after this)\n")
    probe("A6_obs005")
    probe("A6_obs020")
