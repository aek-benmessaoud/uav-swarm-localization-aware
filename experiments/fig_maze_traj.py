"""
experiments/fig_maze_traj.py — render RA trajectories in the maze topology to
check the over-lingering hypothesis (colleague review).

RA has MORE fusion events per rendezvous than FB/CU (706 vs 588/562 at equal
rendezvous_steps) while ending with the LOWEST coverage -> the hypothesis is
that agents linger/loop near each other in corridors without progressing to
new areas. This figure overlays the 6 agent trajectories on the maze map.

Requires traces already produced by experiments/trace.py --regime A6_maze.

Outputs (results/figures):
  fig_maze_traj_RA_run17.png (median RA run)
  fig_maze_traj_RA_run15.png (p75 RA run)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import GridEnv
from config import GRID_SIZE
from utils.seed_manager import env_seed_for_run

TRACE_ROOT = os.path.join("results", "traces")
FIG_OUT = os.path.join("results", "figures")
METHOD = "Richness_Angular"
RUNS = [17, 15]


def json_load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def maze_env(seed):
    return GridEnv(grid_size=GRID_SIZE, num_agents=6, obstacle_ratio=0.5,
                   seed=seed, topology="maze")


def fig_trajectories():
    for run in RUNS:
        base = os.path.join(TRACE_ROOT, "A6_maze", f"{METHOD}_run{run:02d}")
        meta = json_load(os.path.join(base, "meta.json"))
        positions = np.load(os.path.join(base, "positions.npy"))
        obs = maze_env(env_seed_for_run(run)).obstacle_map

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.imshow(obs, cmap="Greys", vmin=0, vmax=1, alpha=0.45,
                  interpolation="nearest")
        colors = plt.cm.turbo(np.linspace(0, 1, positions.shape[1]))
        for a in range(positions.shape[1]):
            ax.plot(positions[:, a, 1], positions[:, a, 0], lw=1.0,
                    alpha=0.8, color=colors[a])
            ax.plot(positions[-1, a, 1], positions[-1, a, 0], "o", ms=6,
                    mfc=colors[a], mec="black", mew=0.8)
        ax.set_xlim(-0.5, obs.shape[1] - 0.5)
        ax.set_ylim(obs.shape[0] - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{METHOD} run {run} (maze)  cov={meta['final_coverage']:.1f}%  "
            f"und={meta['undetermined_final']:.3f}  "
            f"budget={meta['budget']}", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG_OUT,
                                     f"fig_maze_traj_{METHOD}_run{run}.{ext}"),
                        dpi=200)
        plt.close(fig)
        print(f"[fig] fig_maze_traj_{METHOD}_run{run}.png")


if __name__ == "__main__":
    os.makedirs(FIG_OUT, exist_ok=True)
    fig_trajectories()
