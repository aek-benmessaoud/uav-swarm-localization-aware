"""
tests/test_coverage_u.py — Coverage-U policy (budget-limited U-prioritized
coverage).

Checks:
  - factory build (default lam = 0.5),
  - lam = 0 produces ACTION-IDENTICAL trajectories to Frontier-Bounded on two
    identical-seed envs (same target set, same rng stream) — the strongest
    form of "lam=0 is exactly the control",
  - the bonus mechanism is ACTIVE at lam = 0.5 (trajectories diverge from FB),
  - _box_sum matches a naive box-count reference,
  - episode metrics well-formed, deterministic.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env import GridEnv
from metrics import coverage_percent
from policies.factory import build_policy
from policies.frontier_bounded import FrontierBoundedPolicy
from policies.coverage_u import CoverageUPolicy, _box_sum
from experiments._runner import run_episode
from config import (METHOD_COVERAGE_U, METHOD_COVERAGE_U_NORM,
                    METHOD_FRONTIER_BOUNDED)


def test_builds():
    p = build_policy(METHOD_COVERAGE_U, seed=0, fov_radius=5, horizon=8)
    assert isinstance(p, CoverageUPolicy)
    assert hasattr(p, "fallback")
    assert hasattr(p, "random_walk")


def test_builds_norm_variant():
    p_area = build_policy(METHOD_COVERAGE_U, seed=0)
    p_free = build_policy(METHOD_COVERAGE_U_NORM, seed=0)
    assert isinstance(p_free, CoverageUPolicy)
    assert p_area.normalize == "area"
    assert p_free.normalize == "free"


def test_box_sum_matches_naive():
    rng = np.random.default_rng(0)
    for gs in (20, 40):
        grid = rng.random((gs, gs)) < 0.3
        rs, cs = np.meshgrid(np.arange(gs), np.arange(gs), indexing="ij")
        rs = rs.ravel()[:500]
        cs = cs.ravel()[:500]
        got = _box_sum(grid, 3, rs, cs)
        for (r, c), g in zip(zip(rs, cs), got):
            r0, r1 = max(0, r - 3), min(gs, r + 4)
            c0, c1 = max(0, c - 3), min(gs, c + 4)
            exp = int(np.sum(grid[r0:r1, c0:c1]))
            assert g == exp, (r, c, g, exp)


def test_lam0_actions_identical_to_frontier_bounded():
    for env_seed in (0, 7, 42):
        env_fb = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                         seed=env_seed, info_model="comm_limited")
        env_cu = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                         seed=env_seed, info_model="comm_limited")
        pols_fb = [FrontierBoundedPolicy(seed=i, fov_radius=5, horizon=8)
                   for i in range(2)]
        pols_cu = [CoverageUPolicy(seed=i, fov_radius=5, horizon=8, lam=0.0)
                   for i in range(2)]
        for _ in range(150):
            acts_fb = [p.select_action(env_fb, i) for i, p in enumerate(pols_fb)]
            acts_cu = [p.select_action(env_cu, i) for i, p in enumerate(pols_cu)]
            assert [a[0] for a in acts_fb] == [a[0] for a in acts_cu], \
                (env_seed, "action mismatch")
            env_fb.step([a[0] for a in acts_fb])
            env_cu.step([a[0] for a in acts_cu])
        assert coverage_percent(env_fb) == coverage_percent(env_cu)


def test_normalize_free_lam0_identical_to_frontier_bounded():
    # The dynamic-normalization variant must collapse to Frontier-Bounded at
    # lam = 0 exactly like the constant-area variant (bonus term is zero).
    for env_seed in (0, 7, 42):
        env_fb = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                         seed=env_seed, info_model="comm_limited")
        env_norm = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                           seed=env_seed, info_model="comm_limited")
        pols_fb = [FrontierBoundedPolicy(seed=i, fov_radius=5, horizon=8)
                   for i in range(2)]
        pols_norm = [CoverageUPolicy(seed=i, fov_radius=5, horizon=8, lam=0.0,
                                     normalize="free") for i in range(2)]
        for _ in range(150):
            acts_fb = [p.select_action(env_fb, i) for i, p in enumerate(pols_fb)]
            acts_norm = [p.select_action(env_norm, i)
                         for i, p in enumerate(pols_norm)]
            assert [a[0] for a in acts_fb] == [a[0] for a in acts_norm], \
                (env_seed, "action mismatch")
            env_fb.step([a[0] for a in acts_fb])
            env_norm.step([a[0] for a in acts_norm])


def test_normalize_free_diverges_on_dense_obstacles():
    # At 20% obstacles the free-count denominator is materially smaller than
    # the constant FOV_area, so the bonus term is stronger: the two variants
    # must produce different trajectories on the dilution regime.
    kw = dict(max_steps=200, grid_size=40, num_agents=2, obstacle_ratio=0.20,
              fov_radius=5, horizon=8)
    r_area = run_episode(METHOD_COVERAGE_U, "comm_limited", 0, env_seed=7,
                         **kw)
    r_free = run_episode(METHOD_COVERAGE_U_NORM, "comm_limited", 0,
                         env_seed=7, **kw)
    assert (r_area["steps_90"] != r_free["steps_90"]
            or r_area["final_coverage"] != r_free["final_coverage"])


def test_lam0_5_diverges_from_frontier_bounded():
    kw = dict(max_steps=200, grid_size=40, num_agents=2, obstacle_ratio=0.05,
              fov_radius=5, horizon=8)
    r1 = run_episode(METHOD_FRONTIER_BOUNDED, "comm_limited", 0, env_seed=7,
                     **kw)
    r2 = run_episode(METHOD_COVERAGE_U, "comm_limited", 0, env_seed=7, **kw)
    assert r1["steps_90"] != r2["steps_90"] or \
        r1["final_coverage"] != r2["final_coverage"]


def test_episode_metrics_well_formed():
    r = run_episode(METHOD_COVERAGE_U, "comm_limited", 0, env_seed=0,
                    max_steps=120, grid_size=40, num_agents=3,
                    obstacle_ratio=0.05, fov_radius=5, horizon=8)
    assert r["final_coverage"] > 0
    assert 0.0 <= r["fallback_frac"] <= 1.0
    assert 0.0 <= r["random_walk_frac"] <= 1.0
    for key in ("coverage_auc", "quality_auc", "steps_dual",
                "mean_bound_final"):
        assert key in r


def test_episode_deterministic():
    kw = dict(max_steps=120, grid_size=40, num_agents=3, obstacle_ratio=0.05,
              fov_radius=5, horizon=8)
    r1 = run_episode(METHOD_COVERAGE_U, "comm_limited", 1, env_seed=1000, **kw)
    r2 = run_episode(METHOD_COVERAGE_U, "comm_limited", 1, env_seed=1000, **kw)
    assert r1["steps_90"] == r2["steps_90"]
    assert r1["final_coverage"] == r2["final_coverage"]
