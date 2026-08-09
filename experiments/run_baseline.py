"""
run_baseline.py — V4 campaign: comm_limited, 3 signal families.

Protocol (locked decisions):
  - Grid 100x100, 6 agents, FOV=5, obstacles 5%, no noise, MAX_STEPS=10000.
  - NUM_RUNS=30 paired seeds (stride 1000).
  - Info model: comm_limited, R = FOV (comm_range follows fov).
  - Phase 1 methods: Frontier, Least-Visited Greedy, Chao-U (bias_cap),
    Entropy, Entropy-Frac. Phase 2 adds Frontier+Entropy, Frontier+Richness.
  - Movement: bounded-BFS (horizon R=8) through non-visited cells, identical
    frame for all new policies.

Usage:
  python experiments/run_baseline.py --methods Entropy --limit 1   # smoke
  python experiments/run_baseline.py                               # Phase 1
  python experiments/run_baseline.py --cpu-timing --methods Frontier,Entropy
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GRID_SIZE, NUM_AGENTS, MAX_STEPS, NUM_RUNS, DEFAULT_FOV_RADIUS,
    DEFAULT_OBSTACLE_RATIO, ADAPTIVE_K, NUM_WORKERS,
    METHOD_FRONTIER, METHOD_GREEDY, METHOD_CHAO_U,
    METHOD_ENTROPY, METHOD_ENTROPY_FRAC,
    METHOD_FRONTIER_ENTROPY, METHOD_FRONTIER_RICHNESS,
    ENTROPY_HORIZON, ENTROPY_WINDOW, SENSOR_RELIABILITY,
    UTILITY_EPSILON, TIE_BREAK_EPS, INFO_MODEL,
)
from experiments._runner import run_episode, RAW_CSV_FIELDS, TIMING_FIELDS
from utils.paths import raw_csv_path

OUT_DIR = "results/baseline"

# Phase 1 (validated individually before any hybrid is enabled).
PHASE1_METHODS = [
    METHOD_FRONTIER, METHOD_GREEDY, METHOD_CHAO_U,
    METHOD_ENTROPY, METHOD_ENTROPY_FRAC,
]

# Phase 2 (hybrids, enabled only after Phase 1 validation).
PHASE2_METHODS = [
    METHOD_FRONTIER_ENTROPY, METHOD_FRONTIER_RICHNESS,
]

CAMPAIGN_METHODS = PHASE1_METHODS + PHASE2_METHODS

# V2/V3 tuning (locked): Chao-U corrected variant.
CHAO_DEFAULT_VARIANT = "bias_cap"


def _cell_tag(info_model, fov_radius, obstacle_ratio, num_agents, comm_range,
              horizon=ENTROPY_HORIZON, window=ENTROPY_WINDOW, eps=UTILITY_EPSILON,
              p_known=SENSOR_RELIABILITY):
    """Filename tag encoding all NON-default dimensions of the cell."""
    if info_model != "comm_limited":
        return info_model
    parts = [info_model]
    if abs(float(fov_radius) - float(DEFAULT_FOV_RADIUS)) > 1e-9:
        parts.append(f"fov{float(fov_radius):g}")
    if abs(float(obstacle_ratio) - float(DEFAULT_OBSTACLE_RATIO)) > 1e-9:
        parts.append(f"obs{float(obstacle_ratio):g}")
    if num_agents != NUM_AGENTS:
        parts.append(f"agents{num_agents}")
    if (comm_range is not None
            and abs(float(comm_range) - float(fov_radius)) > 1e-9):
        parts.append(f"{float(comm_range):g}")
    if horizon != ENTROPY_HORIZON:
        parts.append(f"R{horizon}")
    if window is not None and window != DEFAULT_FOV_RADIUS:
        parts.append(f"w{window}")
    if abs(float(eps) - float(UTILITY_EPSILON)) > 1e-9:
        parts.append(f"eps{float(eps):g}")
    if abs(float(p_known) - float(SENSOR_RELIABILITY)) > 1e-9:
        parts.append(f"p{float(p_known):g}")
    return "@".join(parts)


def _csv_path(tag, method, variant=None):
    return raw_csv_path(OUT_DIR, tag, method, None, variant)


def _load_done(path):
    import csv
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done[int(row["run"])] = row
    return done


def _fields(timing):
    return RAW_CSV_FIELDS + (TIMING_FIELDS if timing else [])


def _append_row(path, row, timing=False):
    fields = _fields(timing)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        import csv as _c
        w = _c.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--info-model", type=str, default=INFO_MODEL)
    ap.add_argument("--comm-range", type=float, default=None,
                    help="comm_limited range in grid cells (default = fov)")
    ap.add_argument("--fov", type=int, default=DEFAULT_FOV_RADIUS)
    ap.add_argument("--obstacles", type=float, default=DEFAULT_OBSTACLE_RATIO)
    ap.add_argument("--agents", type=int, default=NUM_AGENTS)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--methods", type=str, default=None,
                    help="comma-separated subset of campaign methods")
    ap.add_argument("--phase2", action="store_true",
                    help="include the Phase 2 hybrid methods")
    ap.add_argument("--chao-cap", action="store_true")
    ap.add_argument("--chao-bias", action="store_true")
    ap.add_argument("--chao-bias-cap", action="store_true")
    ap.add_argument("--horizon", type=int, default=ENTROPY_HORIZON,
                    help="bounded-BFS horizon R for the new policies")
    ap.add_argument("--window", type=int, default=ENTROPY_WINDOW,
                    help="signal window radius (default = FOV)")
    ap.add_argument("--p-known", type=float, default=SENSOR_RELIABILITY)
    ap.add_argument("--eps", type=float, default=UTILITY_EPSILON)
    ap.add_argument("--tie-eps", type=float, default=TIE_BREAK_EPS)
    ap.add_argument("--limit", type=int, default=None,
                    help="only run first N runs per cell (smoke test)")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--cpu-timing", action="store_true",
                    help="CPU-timing phase: measure policy decision cost "
                         "(serial, results in results/cpu/)")
    args = ap.parse_args()

    timing = args.cpu_timing
    OUT_DIR = args.out_dir
    if timing:
        OUT_DIR = os.path.join(OUT_DIR, "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    num_runs = args.limit if args.limit else NUM_RUNS
    t0 = time.time()

    info_model = args.info_model
    fov_radius = args.fov
    obstacle_ratio = args.obstacles
    num_agents = args.agents
    comm_range = args.comm_range
    if info_model == "comm_limited":
        comm_range = fov_radius if comm_range is None else comm_range
    else:
        comm_range = None

    if args.chao_cap:
        chao_variant = "cap"
    elif args.chao_bias:
        chao_variant = "bias"
    elif args.chao_bias_cap:
        chao_variant = "bias_cap"
    else:
        chao_variant = CHAO_DEFAULT_VARIANT

    candidates = CAMPAIGN_METHODS if args.phase2 else PHASE1_METHODS
    methods = (args.methods.split(",") if args.methods else list(candidates))

    tag = _cell_tag(info_model, fov_radius, obstacle_ratio, num_agents,
                    comm_range, args.horizon, args.window, args.eps,
                    args.p_known)
    if timing:
        tag = "cpu_" + tag
    workers = 1 if timing else args.workers
    print(f"=== V4 {'cpu-timing' if timing else 'campaign'}: {info_model} "
          f"[{tag}] FOV={fov_radius} obs={obstacle_ratio} agents={num_agents} "
          f"R={comm_range} horizon={args.horizon} window={args.window} ===",
          flush=True)
    for method in methods:
        variant = chao_variant if method == METHOD_CHAO_U else None
        path = _csv_path(tag, method, variant)
        done = _load_done(path)
        todo = [r for r in range(num_runs) if r not in done]
        if not todo:
            print(f"  {method}: already complete", flush=True)
            continue
        print(f"  {method}: {len(todo)} runs", flush=True)
        _run_parallel([(method, info_model, r, variant, comm_range,
                        fov_radius, obstacle_ratio, num_agents) for r in todo],
                      path, workers, timing, args.horizon, args.window,
                      args.p_known, args.eps, args.tie_eps)

    print(f"\nV4 {'cpu-timing' if timing else 'campaign'} done in "
          f"{time.time() - t0:.1f}s -> results in {OUT_DIR}", flush=True)


def _run_parallel(jobs, path, workers, timing=False, horizon=8, window=None,
                  p_known=0.9, eps=1.0, tie_eps=1e-3):
    """Run jobs (method, info_model, run, chao_variant, comm_range, fov_radius,
    obstacle_ratio, num_agents) in parallel, appending results incrementally
    from the main process. workers=1 runs inline (serial) so per-decision CPU
    timings are free of ProcessPool contention."""
    import csv
    from utils.seed_manager import env_seed_for_run

    queued = [(m, im, r, env_seed_for_run(r), cv, cr, fv, ob, na)
              for m, im, r, cv, cr, fv, ob, na in jobs]

    n = 0
    if workers == 1:
        for tup in queued:
            res = _run_one(tup, timing, horizon, window, p_known, eps, tie_eps)
            m, im, r = tup[0], tup[1], tup[2]
            n += 1
            _append_row(path, res, timing)
            print(f"    [{im} | {m}] run {r}: s90={res['steps_90']}, "
                  f"cov={res['final_coverage']:.1f}%, "
                  f"fus={res.get('fusion_events', 0)}, "
                  f"t={res['wall_time_s']:.1f}s, "
                  f"ms/d={res.get('ms_per_decision')} "
                  f"({n}/{len(jobs)} in this batch)", flush=True)
        return

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, tup, timing, horizon, window, p_known,
                          eps, tie_eps): (tup[0], tup[1], tup[2]) for tup in queued}
        for fut in as_completed(futs):
            res = fut.result()
            m, im, r = futs[fut]
            n += 1
            _append_row(path, res, timing)
            print(f"    [{im} | {m}] run {r}: s90={res['steps_90']}, "
                  f"cov={res['final_coverage']:.1f}%, "
                  f"fus={res.get('fusion_events', 0)}, "
                  f"t={res['wall_time_s']:.1f}s "
                  f"({n}/{len(jobs)} in this batch)", flush=True)


def _run_one(tup, timing=False, horizon=8, window=None, p_known=0.9,
             eps=1.0, tie_eps=1e-3):
    """Module-level picklable worker: one episode."""
    m, im, r, es, cv, cr, fv, ob, na = tup
    return run_episode(m, im, r, es, fov_radius=fv,
                       K=ADAPTIVE_K, grid_size=GRID_SIZE, num_agents=na,
                       obstacle_ratio=ob, max_steps=MAX_STEPS,
                       chao_variant=cv, comm_range=cr, timing=timing,
                       horizon=horizon, window=window, p_known=p_known,
                       eps=eps, tie_eps=tie_eps)


if __name__ == "__main__":
    main()
