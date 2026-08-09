"""
tests/test_gdop.py — MH #2 GDOP/FIM baseline policy.

Checks:
  - factory builds GdopPolicy (lam = 0.5, bound_cap = 20, normalize = "free"),
  - direction-only bound math: two orthogonal bearings -> sqrt(2); two
    near-collinear bearings -> large finite bound (geometry, not just count);
    < 2 bearings -> bound cap,
  - lam = 0 is ACTION-IDENTICAL to Frontier-Bounded (same frame, same target
    set, same rng stream) — the strongest "GDOP uses the same movement frame"
    check,
  - NO ORACLE LEAKAGE: scrambling the global oracle accumulators between
    steps must not change a single action,
  - episode metrics well-formed, deterministic, observations actually recorded.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env import GridEnv
from policies.factory import build_policy
from policies.frontier_bounded import FrontierBoundedPolicy
from policies.gdop import GdopPolicy
from experiments._runner import run_episode
from config import METHOD_GDOP, METHOD_FRONTIER_BOUNDED


def test_builds():
    p = build_policy(METHOD_GDOP, seed=0, fov_radius=5, horizon=8)
    assert isinstance(p, GdopPolicy)
    assert p.lam == 0.5
    assert p.bound_cap == 20.0
    assert p.normalize == "free"
    assert hasattr(p, "fallback")
    assert hasattr(p, "random_walk")


def _bound_for(env, pol, cell, angles):
    env.local_angle_clusters[0][cell] = tuple(angles)
    G00, G01, G11, bound = pol._local_gdop_bound(env, 0, 5, 5)
    r, c = divmod(cell, env.grid_size)
    return G00[r, c], G01[r, c], G11[r, c], float(bound[r, c])


def test_direction_only_bound_math():
    env = GridEnv(grid_size=30, num_agents=1, obstacle_ratio=0.0,
                  seed=3, info_model="comm_limited")
    env.agent_positions[0] = [5, 5]
    pol = GdopPolicy(seed=0)
    gs = 30
    cell = 10 * gs + 10

    # two orthogonal bearings -> G = I, bound = sqrt(trace(I)/det(I)) = sqrt(2)
    _, _, _, b_ortho = _bound_for(env, pol, cell, (0.0, math.pi / 2.0))
    assert abs(b_ortho - math.sqrt(2.0)) < 1e-9

    # two near-collinear bearings (0.1 rad): geometry is bad, bound is large
    # but finite (distinguishes GDOP from a mere config count of 2).
    _, _, _, b_collin = _bound_for(env, pol, cell, (0.0, 0.1))
    assert b_collin < pol.bound_cap - 1e-9
    assert b_collin > 10.0

    # single bearing -> rank-deficient -> cap
    _, _, _, b_one = _bound_for(env, pol, cell, (0.0,))
    assert b_one == pol.bound_cap

    # never observed (no clusters) -> cap
    cell2 = 8 * gs + 12
    _, _, _, b_none = _bound_for(env, pol, cell2, ())
    assert b_none == pol.bound_cap


def test_lam0_actions_identical_to_frontier_bounded():
    for env_seed in (0, 7, 42):
        env_fb = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                         seed=env_seed, info_model="comm_limited")
        env_gd = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                         seed=env_seed, info_model="comm_limited")
        pols_fb = [FrontierBoundedPolicy(seed=i, fov_radius=5, horizon=8)
                   for i in range(2)]
        pols_gd = [GdopPolicy(seed=i, fov_radius=5, horizon=8, lam=0.0)
                   for i in range(2)]
        for _ in range(150):
            acts_fb = [p.select_action(env_fb, i) for i, p in enumerate(pols_fb)]
            acts_gd = [p.select_action(env_gd, i) for i, p in enumerate(pols_gd)]
            assert [a[0] for a in acts_fb] == [a[0] for a in acts_gd], \
                (env_seed, "action mismatch")
            env_fb.step([a[0] for a in acts_fb])
            env_gd.step([a[0] for a in acts_gd])


def test_no_oracle_leakage():
    env_a = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                    seed=9, info_model="comm_limited")
    env_b = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                    seed=9, info_model="comm_limited")
    pa = [GdopPolicy(seed=i, fov_radius=5, horizon=8) for i in range(2)]
    pb = [GdopPolicy(seed=i, fov_radius=5, horizon=8) for i in range(2)]
    scram = np.random.default_rng(123)
    for _ in range(120):
        aa = [p.select_action(env_a, i) for i, p in enumerate(pa)]
        # corrupt the global oracle accumulators (the only inputs the CRLB
        # oracle / CentralOracle read); a leaking GDOP policy would diverge.
        env_b.global_info[:] = scram.standard_normal(env_b.global_info.shape)
        env_b.global_obs_count[:] = 999
        ab = [p.select_action(env_b, i) for i, p in enumerate(pb)]
        assert [x[0] for x in aa] == [x[0] for x in ab], "oracle leak detected"
        env_a.step([x[0] for x in aa])
        env_b.step([x[0] for x in ab])


def test_episode_metrics_well_formed():
    r = run_episode(METHOD_GDOP, "comm_limited", 0, env_seed=0, max_steps=300,
                    grid_size=40, num_agents=3, obstacle_ratio=0.05,
                    fov_radius=5, horizon=8)
    assert r["final_coverage"] > 0
    assert 0.0 <= r["fallback_frac"] <= 1.0
    assert 0.0 <= r["random_walk_frac"] <= 1.0
    assert r["mean_bound_final"] == r["mean_bound_final"]  # not NaN
    assert r["quality_final"] > 0.0
    assert r["undetermined_final"] < 1.0


def test_episode_deterministic():
    kw = dict(max_steps=200, grid_size=40, num_agents=3, obstacle_ratio=0.05,
              fov_radius=5, horizon=8)
    r1 = run_episode(METHOD_GDOP, "comm_limited", 1, env_seed=1000, **kw)
    r2 = run_episode(METHOD_GDOP, "comm_limited", 1, env_seed=1000, **kw)
    assert r1["steps_90"] == r2["steps_90"]
    assert r1["final_coverage"] == r2["final_coverage"]
