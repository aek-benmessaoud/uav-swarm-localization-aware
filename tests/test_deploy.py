"""
tests/test_deploy.py — Deploy-U policy (localization-aware deployment).

Research idea 1+3: coverage (Frontier-Bounded) until the known map is mostly
well-localized, then DEPLOY: orbit the worst known under-localized cells
(<= 1 angular configuration) to force angular diversity.

Tested: factory build, no-deploy under high under-localized fraction,
deploy trigger + approach + orbit mechanics (unit-level, fabricated maps),
episode-level well-formed metrics, determinism.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env import GridEnv
from policies.factory import build_policy
from experiments._runner import run_episode
from config import METHOD_DEPLOY


def _env():
    return GridEnv(grid_size=20, num_agents=1, obstacle_ratio=0.0, seed=0,
                   info_model="comm_limited")


def test_deploy_builds():
    p = build_policy(METHOD_DEPLOY, seed=0, fov_radius=5, horizon=8)
    assert hasattr(p, "fallback")
    assert hasattr(p, "random_walk")
    assert hasattr(p, "deploy")
    assert hasattr(p, "orbit")


def test_deploy_no_trigger_when_mostly_underlocalized():
    env = _env()
    p = build_policy(METHOD_DEPLOY, seed=1, fov_radius=5, horizon=8)
    config = np.ones((20, 20), dtype=np.int8)  # everything under-localized
    known = np.ones((20, 20), dtype=bool)
    obs = np.zeros((20, 20), dtype=bool)
    pos = env.agent_positions[0]
    # under_frac = 1.0 > 0.30 -> no deploy
    assert p._maybe_trigger(env, 0, tuple(pos), config, known, obs, 20) is None


def test_deploy_triggers_on_low_underlocalized_fraction():
    env = _env()
    p = build_policy(METHOD_DEPLOY, seed=1, fov_radius=5, horizon=8)
    config = np.full((20, 20), 2, dtype=np.int8)
    for rr, cc in ((2, 2), (2, 17), (17, 2)):
        config[rr, cc] = 0
    known = np.ones((20, 20), dtype=bool)
    obs = np.zeros((20, 20), dtype=bool)
    pos = env.agent_positions[0]
    action = p._maybe_trigger(env, 0, tuple(pos), config, known, obs, 20)
    assert action is not None
    assert action[1] == "deploy"
    assert action[0] in (0, 1, 2, 3, 4)


def test_deploy_orbit_around_adjacent_target():
    env = _env()
    p = build_policy(METHOD_DEPLOY, seed=1, fov_radius=5, horizon=8)
    pos = tuple(env.agent_positions[0])
    target = (pos[0], pos[1] + 1)  # adjacent -> orbit, not approach
    p._active = True
    p._target = target
    p._station_left = 2
    config = np.full((20, 20), 2, dtype=np.int8)
    config[target] = 1
    known = np.ones((20, 20), dtype=bool)
    obs = np.zeros((20, 20), dtype=bool)
    action = p._deploy_step(env, 0, pos, config, known, obs, 20)
    assert action is not None
    assert action[1] == "deploy"
    assert action[2] in ("approach", "orbit")
    assert action[0] in (0, 1, 2, 3, 4)


def test_deploy_episode_well_formed_metrics():
    r = run_episode(METHOD_DEPLOY, "comm_limited", 0, env_seed=0,
                    max_steps=150, grid_size=40, num_agents=3,
                    obstacle_ratio=0.05, fov_radius=2, horizon=8)
    for key in ("deploy_frac", "orbit_frac"):
        v = r[key]
        assert v is None or 0.0 <= v <= 1.0, key
    for key in ("coverage_auc", "mean_bound_final"):
        assert key in r, key
    assert r["steps_dual"] is None or r["steps_dual"] >= 0
    assert 0.0 <= r["fallback_frac"] <= 1.0
    assert 0.0 <= r["random_walk_frac"] <= 1.0


def test_deploy_episode_deterministic():
    kw = dict(max_steps=120, grid_size=40, num_agents=3, obstacle_ratio=0.05,
              fov_radius=5, horizon=8)
    r1 = run_episode(METHOD_DEPLOY, "comm_limited", 1, env_seed=1000, **kw)
    r2 = run_episode(METHOD_DEPLOY, "comm_limited", 1, env_seed=1000, **kw)
    assert r1["steps_90"] == r2["steps_90"]
    assert r1["final_coverage"] == r2["final_coverage"]
    assert r1["deploy_frac"] == r2["deploy_frac"]
