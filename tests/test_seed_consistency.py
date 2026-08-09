"""
tests/test_seed_consistency.py — Paired-seed reproducibility (V4, comm_limited).
Same env seed => identical obstacle map + start positions; the episode is
deterministic given the env + per-agent policy seeds.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env import GridEnv
from experiments._runner import run_episode
from utils.seed_manager import env_seed_for_run, policy_seed_for


def test_same_env_seed_same_map():
    e1 = GridEnv(seed=1234, info_model="comm_limited")
    e2 = GridEnv(seed=1234, info_model="comm_limited")
    assert np.array_equal(e1.obstacle_map, e2.obstacle_map)
    assert e1.agent_positions == e2.agent_positions


def test_different_env_seed_different_map():
    e1 = GridEnv(seed=1234, info_model="comm_limited")
    e2 = GridEnv(seed=2234, info_model="comm_limited")
    assert not np.array_equal(e1.obstacle_map, e2.obstacle_map)


def test_env_seed_stride():
    assert env_seed_for_run(0) == 0
    assert env_seed_for_run(1) == 1000
    assert env_seed_for_run(29) == 29000


def test_policy_seed_unique_per_agent():
    s = {policy_seed_for(0, i) for i in range(6)}
    assert len(s) == 6


def test_episode_deterministic_entropy():
    r1 = run_episode("Entropy", "comm_limited", 0, env_seed=0, max_steps=200,
                     grid_size=50, num_agents=4, obstacle_ratio=0.05)
    r2 = run_episode("Entropy", "comm_limited", 0, env_seed=0, max_steps=200,
                     grid_size=50, num_agents=4, obstacle_ratio=0.05)
    assert r1["steps_90"] == r2["steps_90"]
    assert r1["final_coverage"] == r2["final_coverage"]
    assert r1["overlap"] == r2["overlap"]


def test_episode_deterministic_frontier_richness():
    r1 = run_episode("Frontier+Richness", "comm_limited", 1, env_seed=1000,
                     max_steps=150, grid_size=40, num_agents=3,
                     obstacle_ratio=0.05)
    r2 = run_episode("Frontier+Richness", "comm_limited", 1, env_seed=1000,
                     max_steps=150, grid_size=40, num_agents=3,
                     obstacle_ratio=0.05)
    assert r1["steps_90"] == r2["steps_90"]
    assert r1["final_coverage"] == r2["final_coverage"]


def test_paired_env_seed_across_methods():
    obs_maps = {}
    for method in ["Frontier", "Chao-U", "Entropy", "Entropy-Frac"]:
        env = GridEnv(seed=env_seed_for_run(3), info_model="comm_limited")
        obs_maps[method] = env.obstacle_map.copy()
    for m in obs_maps:
        assert np.array_equal(obs_maps[m], obs_maps["Frontier"])
