"""
analysis/validate_u_gap.py — E1: is the local richness/config-count signal a
reliable indicator of localization need in the swarm?

Question (gate for the deployment strategy): can an agent's LOCAL config-count
map (the only signal the deployment policy can use) tell it where localization
work remains?

For each regime (agents x obstacle) and 10 paired episodes (Frontier-Bounded),
sample at several steps and pool per-agent known-cell samples:

  rho_count   : Spearman(local config count, global config count) over the
                cells the agent knows -> does the local map reflect reality?
  rho_bound   : Spearman(local config count, global CRLB bound) over known
                cells -> does the local map indicate under-localization?
  precision   : among known cells with local count <= 1, fraction that are
                truly not well-localized (global bound > QUALITY_THRESHOLD)
  recall      : fraction of all truly not-well-localized cells detected
  rho_U_gap   : Spearman(agent scalar richness U_local, true gap fraction)
                over (run, step, agent) points -> does the estimator track
                the swarm's remaining localization work?

E1 verdict (pre-registered):
  local map usable if  rho_count >= 0.6, rho_bound <= -0.30,
                       precision >= 0.50, recall >= 0.25
  scalar U tracks gap if rho_U_gap >= 0.6
  (reported per regime; a regime failing these is documented, not silently kept)

Usage:
  python analysis/validate_u_gap.py [--runs 10] [--steps 4500]
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
from estimators.richness import chao_u
from estimators.validation import spearman
from config import (METHOD_FRONTIER_BOUNDED, GRID_SIZE, DEFAULT_FOV_RADIUS,
                    QUALITY_THRESHOLD)

REGIMES = [
    {"label": "A2_obs005", "num_agents": 2, "obstacle_ratio": 0.05},
    {"label": "A3_obs005", "num_agents": 3, "obstacle_ratio": 0.05},
    {"label": "A6_obs005", "num_agents": 6, "obstacle_ratio": 0.05},
    {"label": "A6_obs020", "num_agents": 6, "obstacle_ratio": 0.20},
]

SAMPLE_STEPS = [1000, 2500, 4500]

# E1 pre-registered thresholds.
THR_RHO_COUNT = 0.6
THR_RHO_BOUND = -0.30
THR_PRECISION = 0.50
THR_RECALL = 0.25
THR_RHO_U_GAP = 0.6


def sample(env, step):
    bound = env.global_bound_grid()
    gc = env.global_config_count_grid()
    tr = ~env.obstacle_map
    gap = (bound > QUALITY_THRESHOLD) & tr
    gap_frac = float(np.sum(gap)) / (int(np.sum(tr)) or 1)
    rows = []
    for i in range(env.num_agents):
        lc = env.get_config_count_grid(i)
        seen = env.local_seen_mask[i]
        obs = env.local_obstacle_map[i]
        known = seen & ~obs
        n_known = int(np.sum(known))
        rec = {"run": None, "step": step, "agent": i, "n_known": n_known,
               "gap_frac": gap_frac}
        if n_known >= 20:
            kc = known & tr
            lcv = lc[kc].astype(np.float64)
            gcv = gc[kc].astype(np.float64)
            bv = bound[kc]
            rc, _ = spearman(lcv, gcv)
            rb, _ = spearman(lcv, bv)
            pred = known & (lc <= 1)
            hit = pred & gap
            n_pred = int(np.sum(pred))
            n_gap = int(np.sum(gap))
            rec.update({
                "rho_count": float(rc),
                "rho_bound": float(rb),
                "precision": (float(np.sum(hit)) / n_pred if n_pred else 0.0),
                "recall": (float(np.sum(hit)) / n_gap if n_gap else 0.0),
                "u_local": chao_u(lc, known, obs,
                                  total_unknown=env.get_total_undetermined(i),
                                  variant="bias_cap"),
            })
        rows.append(rec)
    return rows


def run_one(run_index, num_agents, obstacle_ratio, steps, fov_radius,
            sample_steps):
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
    samples = []
    done_steps = 0
    for step in range(1, steps + 1):
        env.check_and_merge()
        actions = [policies[i].select_action(env, i)[0]
                   for i in range(num_agents)]
        env.step(actions)
        done_steps = step
        if step in sample_steps:
            samples.extend(sample(env, step))
        if coverage_percent(env) >= 100.0:
            break
    if done_steps not in sample_steps and samples:
        pass
    wall = time.perf_counter() - t0
    return env, samples, done_steps, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=4500)
    ap.add_argument("--fov", type=int, default=DEFAULT_FOV_RADIUS)
    ap.add_argument("--out", default="results/validate_u_gap.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["regime", "run", "step", "agent", "n_known", "gap_frac",
                  "rho_count", "rho_bound", "precision", "recall", "u_local"]
    new_file = not os.path.exists(args.out)
    done = set()
    if not new_file:
        with open(args.out, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["regime"], int(row["run"])))
    fh = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()

    print("# E1 - local config-count signal vs true localization gap\n")
    summary = []
    for rg in REGIMES:
        label = rg["label"]
        print(f"=== {label}  agents={rg['num_agents']} "
              f"obstacle={rg['obstacle_ratio']} ===", flush=True)
        rows = []
        for run in range(args.runs):
            if (label, run) in done:
                continue
            env, samples, nsteps, wall = run_one(
                run, rg["num_agents"], rg["obstacle_ratio"], args.steps,
                args.fov, SAMPLE_STEPS)
            for s in samples:
                row = {"regime": label, **s}
                row["run"] = run
                writer.writerow({k: ("" if row[k] is None else round(float(row[k]), 6))
                                 if k not in ("regime", "run", "step", "agent")
                                 else row[k] for k in fieldnames})
            fh.flush()
            rows.extend(samples)
            print(f"    run {run + 1}/{args.runs}: steps={nsteps} "
                  f"in {wall:.0f}s", flush=True)

        if not rows:
            with open(args.out, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["regime"] == label:
                        rows.append(row)
            for r in rows:
                for k in ("rho_count", "rho_bound", "precision", "recall",
                          "u_local", "gap_frac"):
                    r[k] = float(r[k]) if r[k] not in ("",) else float("nan")
                r["n_known"] = int(r["n_known"])

        valid = [r for r in rows if r["rho_count"] == r["rho_count"]]
        if not valid:
            continue
        lc = np.array([r["u_local"] for r in valid], dtype=float)
        gf = np.array([r["gap_frac"] for r in valid], dtype=float)
        rho_ug, p_ug = spearman(lc, gf)
        rho_c = np.mean([r["rho_count"] for r in valid])
        rho_b = np.mean([r["rho_bound"] for r in valid])
        prec = np.mean([r["precision"] for r in valid])
        rec = np.mean([r["recall"] for r in valid])
        med_gap = np.median([r["gap_frac"] for r in valid])
        print(f"  gap_frac med={med_gap:.4f}")
        print(f"  rho_count (local vs global counts, mean over samples) = "
              f"{rho_c:.3f}")
        print(f"  rho_bound (local count vs bound, mean)                = "
              f"{rho_b:.3f}")
        print(f"  precision={prec:.3f}  recall={rec:.3f}")
        print(f"  rho(U_local, gap_frac) pooled = {rho_ug:.3f} (p={p_ug:.2e})")
        print()
        summary.append({
            "regime": label, "gap_frac": med_gap,
            "rho_count": rho_c, "rho_bound": rho_b,
            "precision": prec, "recall": rec,
            "rho_U_gap": rho_ug, "rho_U_gap_p": p_ug,
        })

    fh.close()
    print("# E1 summary")
    print(f"{'regime':<12} {'gap':>6} {'rho_count':>9} {'rho_bound':>9} "
          f"{'prec':>5} {'rec':>5} {'rho_Ugap':>8}")
    print("-" * 60)
    for s in summary:
        print(f"{s['regime']:<12} {s['gap_frac']:>6.4f} {s['rho_count']:>9.3f} "
              f"{s['rho_bound']:>9.3f} {s['precision']:>5.2f} "
              f"{s['recall']:>5.2f} {s['rho_U_gap']:>8.3f}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
