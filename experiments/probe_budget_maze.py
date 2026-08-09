"""
experiments/probe_budget_maze.py — recalibrate the campaign budget for a
structured topology (maze / cluster).

Protocol (same convention as E3): run Frontier-Bounded to completion
(max_steps=MAX_STEPS, breaks at 100% coverage) on the given map type, record
steps_90 per run, then set the campaign budget T = 0.7 x FB median steps_90.

Because structured maps change both wall fraction and LOS occlusion, E3
budgets measured on 5% open maps DO NOT transfer (they were measured for open
space). This probe is the prerequisite before run_budget.py --regimes
A6_maze / A6_cluster020.

Run seeds: paired — the same env seeds as the 40-run campaign will use
(env_seed_for_run(r) = BASE_SEED + r * SEED_STRIDE).
"""

import argparse
import csv
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (GRID_SIZE, MAX_STEPS, DEFAULT_FOV_RADIUS, NUM_WORKERS,
                    METHOD_FRONTIER_BOUNDED, INFO_MODEL)
from experiments._runner import run_episode
from utils.seed_manager import env_seed_for_run

OUT_DIR = os.path.join("results", "budget_A6_maze__probe")
CSV_PATH = os.path.join(OUT_DIR, "probe_fb_steps90.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--grid", type=int, default=GRID_SIZE)
    ap.add_argument("--fov", type=int, default=DEFAULT_FOV_RADIUS)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--budget-frac", type=float, default=0.7)
    ap.add_argument("--topology", type=str, default="maze",
                    choices=["maze", "cluster"])
    ap.add_argument("--obstacle-ratio", type=float, default=0.5)
    ap.add_argument("--num-agents", type=int, default=6)
    ap.add_argument("--tag", type=str, default=None,
                    help="suffix for the output dir (e.g. cluster020).")
    args = ap.parse_args()
    workers = args.workers if args.workers is not None else NUM_WORKERS

    tag = args.tag or args.topology
    out_dir = os.path.join("results", f"budget_A6_{tag}__probe")
    csv_path = os.path.join(out_dir, "probe_fb_steps90.csv")
    os.makedirs(out_dir, exist_ok=True)
    print(f"probe: Frontier-Bounded on {args.topology} maps "
          f"(obs={args.obstacle_ratio}), {args.runs} runs, "
          f"max_steps={args.max_steps}, workers={workers}", flush=True)

    seeds = [env_seed_for_run(r) for r in range(args.runs)]
    jobs = [(m, r, s) for r, s in enumerate(seeds)
            for m in (METHOD_FRONTIER_BOUNDED,)]

    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, m, r, s, args): (m, r, s)
                for m, r, s in jobs}
        for fut in as_completed(futs):
            m, r, s = futs[fut]
            res = fut.result()
            rows.append(res)
            print(f"  run {r}: steps_90={res['steps_90']} "
                  f"(censored={res['censored']}) cov={res['final_coverage']:.1f} "
                  f"t={res['wall_time_s']:.1f}s", flush=True)

    rows.sort(key=lambda x: x["run"])
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    steps = [r["steps_90"] for r in rows if not r["censored"]]
    n_censored = sum(r["censored"] for r in rows)
    if not steps:
        print(f"FATAL: all probe runs censored (steps_90 == None). "
              f"{args.topology} maps may be too hard for FB at this max_steps.")
        return 2
    median = float(statistics.median(steps))
    budget = int(round(args.budget_frac * median))
    print(f"\nprobe done in {time.perf_counter() - t0:.0f}s")
    print(f"FB steps_90 (uncensored): n={len(steps)}, "
          f"median={median:.0f}, censored={n_censored}/{args.runs}")
    print(f"PROBE RESULT: {args.topology} budget T = {args.budget_frac} x median "
          f"= {budget} steps")
    print(f"-> run: python experiments/run_budget.py --regimes A6_{tag} "
          f"--methods Frontier-Bounded Coverage-U Richness-Angular "
          f"--runs 40 --budget {budget} --variant {tag}")
    print(f"(raw rows saved in {csv_path})")
    return 0


def _run_one(m, r, s, args):
    return run_episode(m, INFO_MODEL, r, s, fov_radius=args.fov,
                       grid_size=args.grid, num_agents=args.num_agents,
                       obstacle_ratio=args.obstacle_ratio,
                       max_steps=args.max_steps,
                       topology=args.topology)


if __name__ == "__main__":
    sys.exit(main())
