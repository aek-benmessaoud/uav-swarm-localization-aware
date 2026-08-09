"""
experiments/fig_cluster.py — render the cluster-topology maps for the
post-submission cluster campaign (topology="cluster" in GridEnv).

Cluster topology = contiguous obstacle BLOCKS (random rectangles 1..5 wide)
at 20% density, outer edge ring free. Real line-of-sight occlusion (unlike
i.i.d. single cells) while the free space keeps multiple routing paths.

Outputs (results/figures):
  fig_cluster_maps.png   4 seeds x 4 -> 16 cluster maps (obstacle occupancy).
  fig_cluster_los.png    one cluster map with an agent FOV: square FOV vs
                         LOS-occluded FOV, to document the occlusion rule.

Maps are deterministic per env_seed, so no episode rerun is needed.
"""

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

FIG_OUT = os.path.join("results", "figures")

FOV_R = 5


def cluster_env(seed):
    return GridEnv(grid_size=GRID_SIZE, num_agents=1, obstacle_ratio=0.20,
                   seed=seed, topology="cluster")


def fig_maps():
    seeds = [env_seed_for_run(r) for r in range(16)]
    examples = (seeds[0], seeds[15])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, seed in zip(axes, examples):
        env = cluster_env(seed)
        ax.imshow(env.obstacle_map, cmap="Greys", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"seed {seed}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Cluster maps — 100x100, 20% contiguous blocks "
                 "(topology=cluster)", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_OUT, f"fig_cluster_maps.{ext}"), dpi=200)
    plt.close(fig)
    print("[fig] fig_cluster_maps.png")


def fig_los():
    env = cluster_env(env_seed_for_run(0))
    om = env.obstacle_map
    gs = env.grid_size
    free = np.argwhere(~om)
    dist = (free[:, 0] - gs // 2) ** 2 + (free[:, 1] - gs // 2) ** 2
    r, c = free[dist.argmin()]

    r0, r1 = max(0, r - FOV_R), min(gs, r + FOV_R + 1)
    c0, c1 = max(0, c - FOV_R), min(gs, c + FOV_R + 1)
    square = np.zeros((gs, gs), dtype=bool)
    square[r0:r1, c0:c1] = True
    los = env._los_visible(r, c, FOV_R, r0, r1, c0, c1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, mask, title in (
            (axes[0], square, "square FOV (no occlusion)"),
            (axes[1], los, "line-of-sight FOV (occluded)")):
        shown = np.zeros((gs, gs), dtype=np.float32)
        shown[om] = 0.15
        shown[mask] = 0.9
        ax.imshow(shown, cmap="Greys", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].plot(c, r, "o", ms=7, mfc="#2c7bb6", mec="white", mew=0.8)
    axes[1].plot(c, r, "o", ms=7, mfc="#2c7bb6", mec="white", mew=0.8)
    fig.suptitle(f"LOS occlusion on cluster map, agent at ({r},{c}), FOV r={FOV_R} "
                 f"({int(square.sum())} -> {int(los.sum())} visible cells)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_OUT, f"fig_cluster_los.{ext}"), dpi=200)
    plt.close(fig)
    print("[fig] fig_cluster_los.png")


if __name__ == "__main__":
    os.makedirs(FIG_OUT, exist_ok=True)
    fig_maps()
    fig_los()
