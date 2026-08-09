"""
experiments/fig_qualitative.py — render the FB-vs-CU qualitative figures from
the data recorded by experiments/trace.py.

Outputs (results/figures):
  fig_qualitative_traj.png   agent trajectories (rows: regime x percentile)
  fig_qualitative_config_A3_obs005.png  config-count heatmaps at t in {0,T/4,T/2,T}
  fig_qualitative_config_A6_obs005.png  (same for A6)
  fig_qualitative_f1.png     F1-fraction vs step

Requires the trace outputs already produced. Obstacle maps are rebuilt from
env_seed (deterministic), so no rerun of the episodes is needed here.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import GridEnv
from config import INFO_MODEL, GRID_SIZE
from utils.seed_manager import env_seed_for_run

TRACE_ROOT = os.path.join("results", "traces")
FIG_OUT = os.path.join("results", "figures")
REGIMES = {
    "A3_obs005": {"num_agents": 3, "obstacle_ratio": 0.05,
                  "median_run": 26, "p75_run": 25},
    "A6_obs005": {"num_agents": 6, "obstacle_ratio": 0.05,
                  "median_run": 21, "p75_run": 12},
}
METHODS = ["Frontier-Bounded", "Coverage-U"]
FB_C = "#2c7bb6"
CU_C = "#d7191c"


def method_dir(method):
    return method.replace(" ", "_").replace("-", "_")


def load_trace(regime, method, run):
    base = os.path.join(TRACE_ROOT, regime, f"{method_dir(method)}_run{run:02d}")
    meta = json_load(os.path.join(base, "meta.json"))
    positions = np.load(os.path.join(base, "positions.npy"))
    snap = np.load(os.path.join(base, "config_snapshots.npz"))
    f1 = list(csv.DictReader(open(os.path.join(base, "f1_trace.csv"))))
    return (meta, positions, snap["steps"], snap["grids"],
            np.array([float(r["ambiguous_frac"]) for r in f1]))


def json_load(path):
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def obstacle_map(regime, run, grid=GRID_SIZE):
    rg = REGIMES[regime]
    env = GridEnv(grid_size=grid, num_agents=rg["num_agents"],
                  obstacle_ratio=rg["obstacle_ratio"],
                  seed=env_seed_for_run(run), info_model=INFO_MODEL)
    return env.obstacle_map


def draw_obstacles(ax, obs):
    ax.imshow(obs, cmap="Greys", vmin=0, vmax=1, alpha=0.35,
              interpolation="nearest")
    ax.set_xlim(-0.5, obs.shape[1] - 0.5)
    ax.set_ylim(obs.shape[0] - 0.5, -0.5)


def fig_trajectories():
    n_rows = len(REGIMES) * 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4.6 * n_rows))
    r_i = 0
    for regime, rg in REGIMES.items():
        for lab, run in (("median", rg["median_run"]),
                         ("p75", rg["p75_run"])):
            obs = obstacle_map(regime, run)
            for c_i, method in enumerate(METHODS):
                meta, pos, _, _, _ = load_trace(regime, method, run)
                ax = axes[r_i, c_i]
                draw_obstacles(ax, obs)
                for a in range(pos.shape[1]):
                    ax.plot(pos[:, a, 1], pos[:, a, 0], lw=1.1, alpha=0.85)
                    ax.plot(pos[-1, a, 1], pos[-1, a, 0], "o", ms=6,
                            mfc="black", mec="white", mew=0.8)
                ax.set_title(
                    f"{method}\n{regime} run {run} ({lab})  "
                    f"m_bound={meta['mean_bound_final']:.4f}  "
                    f"cov={meta['final_coverage']:.0f}%", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            r_i += 1
    fig.suptitle("Trajectories — Frontier-Bounded vs Coverage-U (comm-limited)",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_OUT, f"fig_qualitative_traj.{ext}"),
                    dpi=200)
    plt.close(fig)
    print("[fig] fig_qualitative_traj.png")


def fig_config_heatmaps():
    steps_all = {}
    grids_all = {}
    for regime in REGIMES:
        steps_all[regime] = None
        grids_all[regime] = []
        for method in METHODS:
            run = REGIMES[regime]["median_run"]
            _, _, steps, grids, _ = load_trace(regime, method, run)
            steps_all[regime] = steps
            grids_all[regime].append(grids)
        vmax = max(int(g.max()) for g in grids_all[regime])
        vmax = min(vmax, 8)
        obs = obstacle_map(regime, REGIMES[regime]["median_run"])
        fig, axes = plt.subplots(len(METHODS), len(steps),
                                 figsize=(4.2 * len(steps), 4.6 * len(METHODS)))
        for r_i, method in enumerate(METHODS):
            for c_i, s in enumerate(steps):
                ax = axes[r_i, c_i]
                g = np.ma.masked_where(obs > 0, grids_all[regime][r_i][c_i])
                im = ax.imshow(g, cmap="viridis", vmin=0, vmax=vmax,
                               interpolation="nearest")
                ax.set_title(f"t={s}/{steps[-1]}", fontsize=10)
                if c_i == 0:
                    ax.set_ylabel(method, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025,
                          pad=0.02, label="global config count")
        cb.ax.tick_params(labelsize=8)
        fig.suptitle(f"Config-count snapshots — {regime} "
                     f"(run {REGIMES[regime]['median_run']})",
                     fontsize=13, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(
                FIG_OUT, f"fig_qualitative_config_{regime}.{ext}"), dpi=200)
        plt.close(fig)
        print(f"[fig] fig_qualitative_config_{regime}.png")


def fig_f1():
    n_rows = len(REGIMES)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.5, 3.1 * n_rows))
    for r_i, (regime, rg) in enumerate(REGIMES.items()):
        ax = axes[r_i]
        for method, col in ((METHODS[0], FB_C), (METHODS[1], CU_C)):
            for lab, run, ls in (("median", rg["median_run"], "-"),
                                 ("p75", rg["p75_run"], "--")):
                _, _, _, _, f1 = load_trace(regime, method, run)
                T = len(f1) - 1
                ax.plot(np.arange(len(f1)), f1, ls, color=col, lw=1.4,
                        label=f"{method} ({lab})")
        ax.set_title(f"{regime}", fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("ambiguous fraction")
        ax.set_ylim(0, 0.10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2, loc="center right")
    fig.suptitle("Fraction of traversable cells with exactly one global "
                 "angular configuration (observed but rank-deficient)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_OUT, f"fig_qualitative_f1.{ext}"), dpi=200)
    plt.close(fig)
    print("[fig] fig_qualitative_f1.png")


if __name__ == "__main__":
    os.makedirs(FIG_OUT, exist_ok=True)
    fig_trajectories()
    fig_config_heatmaps()
    fig_f1()
