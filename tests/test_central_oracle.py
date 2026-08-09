"""
tests/test_central_oracle.py — E5 centralized-oracle control rows.

Checks:
  - factory builds both modes (CentralOracle-CRLB / -Config) with lam = 0.5,
  - actions are valid moves in the TRUE free space (never into obstacles),
  - mode="config" vs mode="crlb" under-sets match the intended definitions on
    a crafted env,
  - lam = 0 makes the score independent of the mode (target set + D identical
    -> same action), the strongest "movement frame is mode-neutral" check,
  - episode metrics well-formed, deterministic.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env import GridEnv
from policies.factory import build_policy
from policies.central_oracle import CentralOraclePolicy, _global_bfs
from experiments._runner import run_episode
from config import METHOD_CENTRAL_CRLB, METHOD_CENTRAL_CONFIG


def test_builds():
    for name, mode in ((METHOD_CENTRAL_CRLB, "crlb"),
                       (METHOD_CENTRAL_CONFIG, "config")):
        p = build_policy(name, seed=0, fov_radius=5, horizon=8)
        assert isinstance(p, CentralOraclePolicy)
        assert p.mode == mode
        assert p.lam == 0.5
        assert hasattr(p, "fallback")
        assert hasattr(p, "random_walk")


def test_actions_never_into_obstacles():
    for mode in ("config", "crlb"):
        env = GridEnv(grid_size=60, num_agents=3, obstacle_ratio=0.20,
                      seed=5, info_model="comm_limited")
        pols = [CentralOraclePolicy(seed=i, fov_radius=5, horizon=8, mode=mode)
                for i in range(3)]
        for _ in range(120):
            acts = [p.select_action(env, i) for i, p in enumerate(pols)]
            assert all(a[0] in (0, 1, 2, 3, 4) for a in acts)
            for i, a in enumerate(acts):
                r, c = env.agent_positions[i]
                if a[0] == 0:
                    r -= 1
                elif a[0] == 1:
                    r += 1
                elif a[0] == 2:
                    c -= 1
                elif a[0] == 3:
                    c += 1
                if not (0 <= r < 60 and 0 <= c < 60):
                    continue
                assert not env.obstacle_map[r, c], (mode, i, a[0])
            env.step([a[0] for a in acts])


def test_under_sets_match_definitions():
    env = GridEnv(grid_size=30, num_agents=1, obstacle_ratio=0.05,
                  seed=3, info_model="comm_limited")
    pol_c = CentralOraclePolicy(seed=0, mode="config")
    pol_r = CentralOraclePolicy(seed=0, mode="crlb")

    obs = env.obstacle_map
    seen = env.global_obs_count > 0
    assert np.array_equal(
        pol_c._under_set(env),
        seen & ~obs & (env.global_config_count_grid() <= 1))
    assert np.array_equal(
        pol_r._under_set(env),
        seen & ~obs & (env.global_bound_grid() > 1.5))


def test_under_sets_track_bottleneck_semantics():
    # Cell (15,15): one bearing -> rank-deficient (bound inf) AND config==1,
    # so it is in BOTH under sets. A second independent bearing makes it
    # well-localized: config==2 and finite bound < 1.5 -> leaves BOTH sets.
    env = GridEnv(grid_size=30, num_agents=1, obstacle_ratio=0.0,
                  seed=3, info_model="comm_limited")
    env.agent_positions[0] = [5, 5]
    env._record_fov_observations(0, 12)
    pol_c = CentralOraclePolicy(seed=0, mode="config")
    pol_r = CentralOraclePolicy(seed=0, mode="crlb")
    assert pol_c._under_set(env)[15, 15]
    assert pol_r._under_set(env)[15, 15]

    env.agent_positions[0] = [25, 15]
    env._record_fov_observations(0, 12)
    assert not pol_c._under_set(env)[15, 15]
    assert not pol_r._under_set(env)[15, 15]
    # a cell observed only once stays in both sets
    once = np.argwhere((env.global_obs_count == 1) & ~env.obstacle_map)
    assert once.size > 0
    assert np.all(pol_c._under_set(env)[once[:, 0], once[:, 1]])
    assert np.all(pol_r._under_set(env)[once[:, 0], once[:, 1]])


def test_lam0_movement_frame_mode_neutral():
    # With lam = 0 the under-set never enters the score, so config and crlb
    # modes must be ACTION-IDENTICAL (same target set, same D, same rng).
    for env_seed in (0, 11, 77):
        env_a = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                        seed=env_seed, info_model="comm_limited")
        env_b = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                        seed=env_seed, info_model="comm_limited")
        pa = [CentralOraclePolicy(seed=i, fov_radius=5, horizon=8, lam=0.0,
                                  mode="config") for i in range(2)]
        pb = [CentralOraclePolicy(seed=i, fov_radius=5, horizon=8, lam=0.0,
                                  mode="crlb") for i in range(2)]
        for _ in range(150):
            aa = [p.select_action(env_a, i) for i, p in enumerate(pa)]
            ab = [p.select_action(env_b, i) for i, p in enumerate(pb)]
            assert [x[0] for x in aa] == [x[0] for x in ab], \
                (env_seed, "mode-dependent action at lam=0")
            env_a.step([x[0] for x in aa])
            env_b.step([x[0] for x in ab])


def test_global_bfs_matches_reference():
    # Mirror of test_bfs.py on the global frame: distances = Manhattan on an
    # open grid, first-step directions correct, horizon cap, obstacle wall
    # blocks traversal.
    env = GridEnv(grid_size=15, num_agents=1, obstacle_ratio=0.0,
                  seed=0, info_model="comm_limited")
    env.agent_positions[0] = [5, 5]
    obs = env.obstacle_map
    visit = env.visit_count
    free_path = ~obs & (visit == 0)
    D, curdir = _global_bfs(5, 5, 15, free_path, 10)
    assert D[5, 5] == 0
    assert int(curdir[5, 5]) == 4
    assert D[7, 5] == 2
    assert D[5, 8] == 3
    assert int(curdir[7, 5]) == 1
    assert int(curdir[5, 7]) == 3
    assert int(curdir[6, 6]) == 1

    D3, _ = _global_bfs(5, 5, 15, free_path, 3)
    assert D3[5, 8] == 3
    assert D3[5, 9] == -1
    assert D3[9, 9] == -1

    # obstacle wall at col 3 blocks the left region
    obs[:, 3] = True
    free_path2 = ~obs & (visit == 0)
    D, _ = _global_bfs(5, 5, 15, free_path2, 10)
    assert D[5, 4] == 1
    assert D[5, 2] == -1
    assert D[5, 0] == -1
    assert D[5, 7] == 2


def test_episode_metrics_well_formed():
    for name in (METHOD_CENTRAL_CRLB, METHOD_CENTRAL_CONFIG):
        r = run_episode(name, "comm_limited", 0, env_seed=0, max_steps=300,
                        grid_size=40, num_agents=3, obstacle_ratio=0.05,
                        fov_radius=5, horizon=8)
        assert r["final_coverage"] > 0
        assert 0.0 <= r["fallback_frac"] <= 1.0
        assert 0.0 <= r["random_walk_frac"] <= 1.0
        # the sensor hook must fire: observations recorded -> finite CRLB
        # bounds exist and some cells become well-localized.
        assert r["mean_bound_final"] == r["mean_bound_final"]  # not NaN
        assert r["quality_final"] > 0.0
        assert r["undetermined_final"] < 1.0


def test_observations_recorded():
    # Guard against the oracle skipping the sensor hook: a run must produce
    # real observations (global_obs_count > 0, finite bounds, quality > 0).
    env = GridEnv(grid_size=40, num_agents=2, obstacle_ratio=0.05,
                  seed=4, info_model="comm_limited")
    pols = [CentralOraclePolicy(seed=i, fov_radius=5, horizon=8)
            for i in range(2)]
    for _ in range(60):
        acts = [p.select_action(env, i) for i, p in enumerate(pols)]
        env.step([a[0] for a in acts])
    assert np.sum(env.global_obs_count > 0) > 0
    bound = env.global_bound_grid()
    assert np.sum(np.isfinite(bound) & ~env.obstacle_map) > 0


def test_episode_deterministic():
    kw = dict(max_steps=120, grid_size=40, num_agents=3, obstacle_ratio=0.05,
              fov_radius=5, horizon=8)
    for name in (METHOD_CENTRAL_CRLB, METHOD_CENTRAL_CONFIG):
        r1 = run_episode(name, "comm_limited", 1, env_seed=1000, **kw)
        r2 = run_episode(name, "comm_limited", 1, env_seed=1000, **kw)
        assert r1["steps_90"] == r2["steps_90"]
        assert r1["final_coverage"] == r2["final_coverage"]
