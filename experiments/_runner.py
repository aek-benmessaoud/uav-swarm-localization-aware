"""
experiments/_runner.py — Shared single-episode runner + parallel driver.

INCREMENTAL SAVING (locked decision): each completed run is appended
immediately to a per-(method, info_model) raw CSV. The script is RESUMABLE:
on restart it skips runs already present in the CSV.

TIMING (locked decision): policy CPU time is measured with
`time.process_time()` (CPU time of this process, immune to OS scheduling),
never `time.time()`.
"""

import os
import sys
import time
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env import GridEnv
from metrics import (coverage_percent, steps_to_threshold, overlap_ratio,
                     unseen_estimator)
from policies.factory import build_policy
from utils.seed_manager import env_seed_for_run, policy_seed_for

RAW_CSV_FIELDS = [
    "method", "info_model", "run", "env_seed",
    "steps_90", "steps_95", "steps_99", "censored", "success_90",
    "overlap", "final_U", "final_coverage", "alpha_mean",
    "chao_version", "fov_radius", "obstacle_ratio", "num_agents",
    "comm_range", "fusion_events", "rendezvous_steps",
    "topology", "maze_loop_density",
    "alpha_sat_frac", "fallback_frac", "random_walk_frac",
    "quality_auc", "time_to_quality", "quality_final",
    "coverage_auc", "steps_dual", "mean_bound_final",
    "cluster_cap_hit_frac", "undetermined_final",
    "deploy_frac", "orbit_frac", "hybrid_ra_frac",
    "wall_time_s",
]

# Optional CPU-timing columns (only written when --cpu-timing is active, into a
# separate results dir; campaign CSVs keep the base schema).
TIMING_FIELDS = [
    "policy_cpu_s", "ms_per_decision",
]


def run_episode(method_name, info_model, run_index, env_seed, fov_radius=5,
                K=0.5, grid_size=100, num_agents=6,
                obstacle_ratio=0.05, p_miss=0.0, sigma_loc=0.0,
                sigma_bearing=0.0,
                max_steps=10000, num_workers=None, task_id=None,
                chao_variant=None, comm_range=None, timing=False,
                horizon=8, window=None, p_known=0.9, eps=1.0,
                 tie_eps=1e-3, quality_sample_k=None, quality_target=None,
                 quality_threshold=None, lam=None, on_step=None,
                 topology="random", maze_loop_density=0.0):
    """Run ONE episode. Returns a result dict (pure function, picklable).

    on_step(env, step) is an optional hook called with the env at step 0 and
    after every step (steps 1..max_steps). Used ONLY by trace.py for the
    qualitative figures; campaigns pass None (no overhead).
    """
    from config import (QUALITY_SAMPLE_K, QUALITY_TARGET)
    if quality_sample_k is None:
        quality_sample_k = QUALITY_SAMPLE_K
    if quality_target is None:
        quality_target = QUALITY_TARGET

    env = GridEnv(grid_size=grid_size, num_agents=num_agents,
                  obstacle_ratio=obstacle_ratio, seed=env_seed,
                  info_model=info_model, p_miss=p_miss, sigma_loc=sigma_loc,
                  sigma_bearing=sigma_bearing,
                  comm_range=comm_range, topology=topology,
                  maze_loop_density=maze_loop_density)
    if quality_threshold is not None:
        env.quality_threshold = quality_threshold

    policies = [
        build_policy(method_name,
                     seed=policy_seed_for(env_seed, i),
                     fov_radius=fov_radius, K=K,
                     chao_variant=chao_variant, horizon=horizon,
                     window=window, p_known=p_known, eps=eps, tie_eps=tie_eps,
                     lam=lam)
        for i in range(num_agents)
    ]

    coverage_history = [coverage_percent(env)]
    alphas = []
    fusion_events = 0
    rendezvous_steps = 0
    quality_times = [0]
    quality_vals = [env.quality_well_localized(quality_threshold)]

    if on_step is not None:
        on_step(env, 0)

    policy_cpu = 0.0
    decisions = 0
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        if info_model == "comm_limited":
            if env.check_and_merge() > 0:
                rendezvous_steps += 1
        actions = []
        for i in range(num_agents):
            t_p = time.process_time()
            act, _, _, a = policies[i].select_action(env, i)
            policy_cpu += time.process_time() - t_p
            decisions += 1
            actions.append(act)
            if a is not None:
                alphas.append(float(a))
        env.step(actions)
        cov = coverage_percent(env)
        coverage_history.append(cov)
        if on_step is not None:
            on_step(env, step)
        if step % quality_sample_k == 0:
            quality_times.append(step)
            quality_vals.append(env.quality_well_localized(quality_threshold))
        if cov >= 100.0:
            break

    fusion_events = env.fusion_events_count
    wall = time.perf_counter() - t0

    s90 = steps_to_threshold(coverage_history, 90)
    s95 = steps_to_threshold(coverage_history, 95)
    s99 = steps_to_threshold(coverage_history, 99)
    censored = s90 is None

    alpha_mean = float(np.mean(alphas)) if alphas else None
    alpha_sat_frac = None
    if alphas and method_name == "Chao-U":
        alpha_max = 1.0 / (1.0 + K)
        thr = 0.9 * alpha_max
        alpha_sat_frac = float(np.mean(np.array(alphas) >= thr))

    fallback_total = sum(getattr(p, "fallback", 0) for p in policies)
    rw_total = sum(getattr(p, "random_walk", 0) for p in policies)
    fallback_frac = (fallback_total / decisions if decisions else None)
    random_walk_frac = (rw_total / decisions if decisions else None)

    quality_auc = _auc_trapezoid(quality_times, quality_vals)
    time_to_quality = _first_reach(quality_times, quality_vals, quality_target)
    quality_final = quality_vals[-1]
    coverage_auc = _auc_trapezoid(list(range(len(coverage_history))),
                                  coverage_history)
    steps_dual = None
    for t, q in zip(quality_times, quality_vals):
        if (q >= quality_target and t < len(coverage_history)
                and coverage_history[t] >= 90.0):
            steps_dual = t
            break
    bound = env.global_bound_grid()
    finite = bound[np.isfinite(bound) & ~env.obstacle_map]
    mean_bound_final = float(np.mean(finite)) if finite.size else float("nan")
    cluster_cap_hit_frac = env.cluster_cap_hit_frac()
    undetermined_final = env.global_undetermined_fraction()
    deploy_total = sum(getattr(p, "deploy", 0) for p in policies)
    orbit_total = sum(getattr(p, "orbit", 0) for p in policies)
    hybrid_ra_total = sum(getattr(p, "ra_steps", 0) for p in policies)
    hybrid_mode_total = sum(getattr(p, "mode_steps", 0) for p in policies)
    hybrid_ra_frac = (hybrid_ra_total / hybrid_mode_total
                      if hybrid_mode_total else None)

    return {
        "method": method_name,
        "info_model": info_model,
        "run": run_index,
        "env_seed": env_seed,
        "steps_90": s90,
        "steps_95": s95,
        "steps_99": s99,
        "censored": censored,
        "success_90": 100.0 if not censored else 0.0,
        "overlap": overlap_ratio(env),
        "final_U": unseen_estimator(env),
        "final_coverage": coverage_history[-1],
        "alpha_mean": alpha_mean,
        "alpha_sat_frac": alpha_sat_frac,
        "chao_version": chao_variant or "original",
        "fov_radius": float(fov_radius),
        "obstacle_ratio": float(obstacle_ratio),
        "num_agents": num_agents,
        "comm_range": comm_range if comm_range is not None else "",
        "topology": topology,
        "maze_loop_density": float(maze_loop_density),
        "fusion_events": fusion_events,
        "rendezvous_steps": rendezvous_steps,
        "fallback_frac": fallback_frac,
        "random_walk_frac": random_walk_frac,
        "quality_auc": quality_auc,
        "time_to_quality": time_to_quality,
        "quality_final": quality_final,
        "coverage_auc": coverage_auc,
        "steps_dual": steps_dual,
        "mean_bound_final": mean_bound_final,
        "cluster_cap_hit_frac": cluster_cap_hit_frac,
        "undetermined_final": undetermined_final,
        "deploy_frac": (deploy_total / decisions if decisions else None),
        "orbit_frac": (orbit_total / decisions if decisions else None),
        "hybrid_ra_frac": hybrid_ra_frac,
        "wall_time_s": wall,
        "policy_cpu_s": policy_cpu if timing else None,
        "ms_per_decision": (1000.0 * policy_cpu / decisions
                            if timing and decisions else None),
    }


def _auc_trapezoid(times, vals):
    """Normalized AUC by trapezoids: (1/T) * integral quality(t) dt, T = last
    sample time. 0 if no samples."""
    if len(times) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(times)):
        area += (vals[i - 1] + vals[i]) / 2.0 * (times[i] - times[i - 1])
    return area / times[-1] if times[-1] > 0 else 0.0


def _first_reach(times, vals, target):
    """First sample time where quality >= target (mirror of steps_to_threshold).
    None if never reached."""
    for t, v in zip(times, vals):
        if v >= target:
            return t
    return None


def _row_for_csv(res):
    def _num(v, fmt="int"):
        if v is None:
            return ""
        if fmt == "int":
            return int(v) if v == v else ""
        return round(float(v), 6)

    row = {
        "method": res["method"],
        "info_model": res["info_model"],
        "run": res["run"],
        "env_seed": res["env_seed"],
        "steps_90": _num(res["steps_90"]),
        "steps_95": _num(res["steps_95"]),
        "steps_99": _num(res["steps_99"]),
        "censored": 1 if res["censored"] else 0,
        "success_90": round(res["success_90"], 1),
        "overlap": _num(res["overlap"], "f"),
        "final_U": _num(res["final_U"], "f"),
        "final_coverage": _num(res["final_coverage"], "f"),
        "alpha_mean": _num(res["alpha_mean"], "f"),
        "alpha_sat_frac": _num(res.get("alpha_sat_frac"), "f"),
        "chao_version": res.get("chao_version", "") or "",
        "fov_radius": _num(res.get("fov_radius"), "f"),
        "obstacle_ratio": _num(res.get("obstacle_ratio"), "f"),
        "num_agents": res.get("num_agents", ""),
        "comm_range": str(res.get("comm_range", "") or ""),
        "topology": str(res.get("topology", "") or ""),
        "maze_loop_density": _num(res.get("maze_loop_density"), "f"),
        "fusion_events": int(res.get("fusion_events", 0) or 0),
        "rendezvous_steps": int(res.get("rendezvous_steps", 0) or 0),
        "fallback_frac": _num(res.get("fallback_frac"), "f"),
        "random_walk_frac": _num(res.get("random_walk_frac"), "f"),
        "quality_auc": _num(res.get("quality_auc"), "f"),
        "time_to_quality": _num(res.get("time_to_quality")),
        "quality_final": _num(res.get("quality_final"), "f"),
        "coverage_auc": _num(res.get("coverage_auc"), "f"),
        "steps_dual": _num(res.get("steps_dual")),
        "mean_bound_final": _num(res.get("mean_bound_final"), "f"),
        "cluster_cap_hit_frac": _num(res.get("cluster_cap_hit_frac"), "f"),
        "undetermined_final": _num(res.get("undetermined_final"), "f"),
        "deploy_frac": _num(res.get("deploy_frac"), "f"),
        "orbit_frac": _num(res.get("orbit_frac"), "f"),
        "hybrid_ra_frac": _num(res.get("hybrid_ra_frac"), "f"),
        "wall_time_s": _num(res["wall_time_s"], "f"),
    }
    if res.get("policy_cpu_s") is not None:
        row["policy_cpu_s"] = _num(res["policy_cpu_s"], "f")
        row["ms_per_decision"] = _num(res.get("ms_per_decision"), "f")
    return row


def _run_one(run, method_name, info_model, kwargs):
    """Picklable module-level worker for ProcessPoolExecutor (Windows spawn)."""
    env_seed = env_seed_for_run(run)
    return run, run_episode(method_name, info_model, run, env_seed, **kwargs)


def run_experiment_set(method_name, info_model, num_runs, out_dir, tag="",
                       num_workers=1, **kwargs):
    """
    Run NUM_RUNS paired episodes for one (method, info_model), writing each
    raw result incrementally to out_dir/raw_{info_model}__{method}{tag}.csv.
    Returns the list of results. `tag` disambiguates variants of one method
    (e.g. Coverage-U at different lambda values). With num_workers > 1 the
    pending runs are distributed across a process pool; the PARENT process is
    the sole CSV writer, so incremental append + resume stay race-free.
    """
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", method_name)
    csv_path = os.path.join(out_dir, f"raw_{info_model}__{safe}{tag}.csv")

    # Load already-completed runs for resume
    done = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done[int(row["run"])] = row

    pending = [r for r in range(num_runs) if r not in done]
    results = [done[r] for r in range(num_runs) if r in done]
    results_map = {int(r["run"]): r for r in results}
    header_pending = not (os.path.exists(csv_path)
                          and os.path.getsize(csv_path) > 0)

    def _write(res):
        nonlocal header_pending
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
            if header_pending:
                writer.writeheader()
                header_pending = False
            writer.writerow(_row_for_csv(res))

    def _log(res):
        run = int(res["run"])
        print(f"    [{info_model} | {method_name}] run {run + 1}/{num_runs} "
              f"done in {res['wall_time_s']:.1f}s "
              f"(s90={res['steps_90']}, cov={res['final_coverage']:.1f}%)",
              flush=True)

    use_pool = num_workers > 1 and len(pending) > 1
    if use_pool:
        with ProcessPoolExecutor(
                max_workers=min(num_workers, len(pending))) as ex:
            futs = {ex.submit(_run_one, r, method_name, info_model, kwargs): r
                    for r in pending}
            for fut in as_completed(futs):
                run, res = fut.result()
                _write(res)
                _log(res)
                results_map[run] = res
    else:
        for run in pending:
            _, res = _run_one(run, method_name, info_model, kwargs)
            _write(res)
            _log(res)
            results_map[run] = res

    return [results_map[r] for r in range(num_runs)]
