"""
tests/test_control.py — Movement-control (Frontier-Bounded) + instrumentation.

Frontier-Bounded is the TEMPORARY movement control: Frontier's selection rule
(nearest reachable unknown cell) inside the entropy family's bounded-BFS frame
(horizon R, fallback to explore_action). It isolates the movement frame from
the signal. NOT in the paper.

Also checks that the fallback / random-walk counters exported by the runner
are well-formed for the bounded-BFS policies.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from policies.factory import build_policy
from experiments._runner import run_episode


def test_control_policy_builds():
    for seed in [0, 42]:
        p = build_policy("Frontier-Bounded", seed=seed, fov_radius=2,
                         horizon=8)
        assert hasattr(p, "fallback")
        assert hasattr(p, "random_walk")


def test_control_policy_episode_runs():
    r = run_episode("Frontier-Bounded", "comm_limited", 0, env_seed=0,
                    max_steps=100, grid_size=40, num_agents=3,
                    obstacle_ratio=0.05, fov_radius=2, horizon=8)
    assert r["final_coverage"] > 0
    assert 0.0 <= r["fallback_frac"] <= 1.0
    assert 0.0 <= r["random_walk_frac"] <= 1.0


def test_control_episode_deterministic():
    kw = dict(max_steps=120, grid_size=40, num_agents=3, obstacle_ratio=0.05,
              fov_radius=7, horizon=8)
    r1 = run_episode("Frontier-Bounded", "comm_limited", 1, env_seed=1000, **kw)
    r2 = run_episode("Frontier-Bounded", "comm_limited", 1, env_seed=1000, **kw)
    assert r1["steps_90"] == r2["steps_90"]
    assert r1["final_coverage"] == r2["final_coverage"]


def test_instrumented_fractions_bounded_policies():
    for method in ["Entropy", "Entropy-Frac", "Frontier+Entropy",
                   "Frontier+Richness", "Frontier-Bounded"]:
        r = run_episode(method, "comm_limited", 2, env_seed=2000,
                        max_steps=80, grid_size=40, num_agents=3,
                        obstacle_ratio=0.05)
        assert 0.0 <= r["fallback_frac"] <= 1.0, method
        assert 0.0 <= r["random_walk_frac"] <= 1.0, method


def test_random_walk_frac_well_formed_frontier():
    r = run_episode("Frontier", "comm_limited", 2, env_seed=2000,
                    max_steps=80, grid_size=40, num_agents=3,
                    obstacle_ratio=0.05)
    assert 0.0 <= r["random_walk_frac"] <= 1.0
    # Frontier has no select_target, so no fallback counter.
    assert r["fallback_frac"] is None or 0.0 <= r["fallback_frac"] <= 1.0
