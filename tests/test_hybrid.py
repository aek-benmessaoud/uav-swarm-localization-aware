"""
tests/test_hybrid.py — MH #3 Hybrid policy (CU-norm <-> RA routing, theta=0.8).

Checks:
  - factory builds HybridPolicy (lam = 0.5, theta = 0.8),
  - theta = 0.0 forces the CU-norm branch everywhere -> with lam=0 the hybrid
    is ACTION-IDENTICAL to Frontier-Bounded (same frame, same target set),
  - a hard obstacle world triggers RA mode (ra_steps > 0) while an open world
    never does (ra_steps == 0) — the switch is exercised, not inert,
  - NO ORACLE LEAKAGE: scrambling the global oracle accumulators between
    steps must not change a single action,
  - episode metrics well-formed, hybrid_ra_frac in [0, 1], deterministic.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env import GridEnv
from policies.factory import build_policy
from policies.frontier_bounded import FrontierBoundedPolicy
from policies.hybrid import HybridPolicy
from experiments._runner import run_episode
from config import METHOD_HYBRID, METHOD_FRONTIER_BOUNDED


def test_builds():
    p = build_policy(METHOD_HYBRID, seed=0, fov_radius=5, horizon=8)
    assert isinstance(p, HybridPolicy)
    assert p.lam == 0.5
    assert p.theta == 0.8
    assert p.fov_radius == 5
    assert hasattr(p, "ra_steps")
    assert hasattr(p, "mode_steps")


def test_theta0_lam0_actions_identical_to_frontier_bounded():
    for env_seed in (0, 7, 42):
        env_fb = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                         seed=env_seed, info_model="comm_limited")
        env_hy = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                         seed=env_seed, info_model="comm_limited")
        pols_fb = [FrontierBoundedPolicy(seed=i, fov_radius=5, horizon=8)
                   for i in range(2)]
        pols_hy = [HybridPolicy(seed=i, fov_radius=5, horizon=8, lam=0.0,
                                theta=0.0) for i in range(2)]
        for _ in range(150):
            acts_fb = [p.select_action(env_fb, i) for i, p in enumerate(pols_fb)]
            acts_hy = [p.select_action(env_hy, i) for i, p in enumerate(pols_hy)]
            assert [a[0] for a in acts_fb] == [a[0] for a in acts_hy], \
                (env_seed, "action mismatch")
            env_fb.step([a[0] for a in acts_fb])
            env_hy.step([a[0] for a in acts_hy])


def _drive(pols, env, steps):
    for _ in range(steps):
        acts = [p.select_action(env, i) for i, p in enumerate(pols)]
        env.step([a[0] for a in acts])
    return sum(p.mode_steps for p in pols), sum(p.ra_steps for p in pols)


def test_switch_is_exercised_in_obstacle_world_not_in_open_world():
    # Open world: window free fraction ~1.0 >> theta -> pure CU-norm mode.
    env_open = GridEnv(grid_size=50, num_agents=3, obstacle_ratio=0.0,
                       seed=5, info_model="comm_limited")
    pols_open = [HybridPolicy(seed=i, fov_radius=5, horizon=8)
                 for i in range(3)]
    mode_open, ra_open = _drive(pols_open, env_open, 120)
    assert mode_open > 0
    assert ra_open == 0, "RA mode must never trigger in an open world"

    # Obstacle-heavy world: many windows below theta=0.8 -> RA mode fires.
    env_dense = GridEnv(grid_size=50, num_agents=3, obstacle_ratio=0.4,
                        seed=5, info_model="comm_limited")
    pols_dense = [HybridPolicy(seed=i, fov_radius=5, horizon=8)
                  for i in range(3)]
    mode_dense, ra_dense = _drive(pols_dense, env_dense, 120)
    assert mode_dense > 0
    assert ra_dense > 0, "RA mode must trigger in an obstacle-heavy world"
    assert ra_dense <= mode_dense


def test_no_oracle_leakage():
    env_a = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.2,
                    seed=9, info_model="comm_limited")
    env_b = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.2,
                    seed=9, info_model="comm_limited")
    pa = [HybridPolicy(seed=i, fov_radius=5, horizon=8) for i in range(2)]
    pb = [HybridPolicy(seed=i, fov_radius=5, horizon=8) for i in range(2)]
    scram = np.random.default_rng(123)
    for _ in range(120):
        aa = [p.select_action(env_a, i) for i, p in enumerate(pa)]
        env_b.global_info[:] = scram.standard_normal(env_b.global_info.shape)
        env_b.global_obs_count[:] = 999
        ab = [p.select_action(env_b, i) for i, p in enumerate(pb)]
        assert [x[0] for x in aa] == [x[0] for x in ab], "oracle leak detected"
        env_a.step([x[0] for x in aa])
        env_b.step([x[0] for x in ab])


def test_episode_metrics_well_formed():
    r = run_episode(METHOD_HYBRID, "comm_limited", 0, env_seed=0,
                    max_steps=300, grid_size=40, num_agents=3,
                    obstacle_ratio=0.2, fov_radius=5, horizon=8)
    assert r["final_coverage"] > 0
    assert 0.0 <= r["fallback_frac"] <= 1.0
    assert 0.0 <= r["random_walk_frac"] <= 1.0
    assert r["mean_bound_final"] == r["mean_bound_final"]  # not NaN
    assert r["quality_final"] > 0.0
    assert r["undetermined_final"] < 1.0
    assert r["hybrid_ra_frac"] is not None
    assert 0.0 <= r["hybrid_ra_frac"] <= 1.0


def test_episode_deterministic():
    kw = dict(max_steps=200, grid_size=40, num_agents=3, obstacle_ratio=0.2,
              fov_radius=5, horizon=8)
    r1 = run_episode(METHOD_HYBRID, "comm_limited", 1, env_seed=1000, **kw)
    r2 = run_episode(METHOD_HYBRID, "comm_limited", 1, env_seed=1000, **kw)
    assert r1["steps_90"] == r2["steps_90"]
    assert r1["final_coverage"] == r2["final_coverage"]
    assert r1["hybrid_ra_frac"] == r2["hybrid_ra_frac"]
