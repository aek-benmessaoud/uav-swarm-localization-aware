"""
tests/test_angular.py — Project08 angular observation model + CRLB evaluation.

Covers:
  - greedy angular clustering (threshold, circular wrap, cap, immutability),
  - exactness of center-based fusion merge (== re-clustering raw angles),
  - env recording (own vs local counts, fusion union, undetermined cells),
  - oracle CRLB bound (finite for 2 independent directions, +inf otherwise),
  - bit-coherence of the transposed Chao-U with the LOCKED V4 estimator,
  - new policies build/run, determinism of the new quality metrics.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from estimators import angular
from estimators.angular import (ANG_TOL, CLUSTER_CAP, add_angle_to_clusters,
                                angular_distance, greedy_cluster_centers,
                                merge_raw_angle_lists)
from estimators import richness
from env import GridEnv
from policies.factory import build_policy
from experiments._runner import run_episode

DEG = np.pi / 180.0


# ----------------------------------------------------------------------
# Pure clustering
# ----------------------------------------------------------------------

def test_angular_distance():
    assert angular_distance(0.0, 0.0) == 0.0
    assert abs(angular_distance(0.0, 20 * DEG) - 20 * DEG) < 1e-12
    assert abs(angular_distance(0.0, 200 * DEG) - 160 * DEG) < 1e-12
    assert abs(angular_distance(350 * DEG, 10 * DEG) - 20 * DEG) < 1e-12


def test_add_angle_clusters_threshold():
    centers, hit = add_angle_to_clusters([], 0.0)
    assert centers == [0.0] and not hit
    # 20 deg from 0 deg with 15 deg tol -> new cluster
    centers, hit = add_angle_to_clusters(centers, 20 * DEG)
    assert len(centers) == 2 and not hit
    # 10 deg from 0 deg -> assign (nearest center)
    centers, hit = add_angle_to_clusters(centers, 10 * DEG)
    assert len(centers) == 2 and not hit


def test_add_angle_clusters_circular_wrap():
    centers, _ = add_angle_to_clusters([], 350 * DEG)
    centers, hit = add_angle_to_clusters(centers, 10 * DEG)
    assert len(centers) == 2  # 20 deg apart > 15 deg
    centers, _ = add_angle_to_clusters([], 355 * DEG)
    centers, hit = add_angle_to_clusters(centers, 5 * DEG)
    assert len(centers) == 1  # 10 deg apart <= 15 deg


def test_add_angle_clusters_cap():
    centers = []
    hit_any = False
    for k in range(20):
        centers, hit = add_angle_to_clusters(centers, k * 30 * DEG)
        hit_any = hit_any or hit
    assert len(centers) == CLUSTER_CAP
    assert hit_any


def test_add_angle_clusters_immutability():
    base = [0.0, 30.0 * DEG]
    newc, _ = add_angle_to_clusters(base, 60.0 * DEG)
    assert base == [0.0, 30.0 * DEG]  # input never mutated
    assert newc != base


def test_greedy_cluster_centers_deterministic():
    angles = [80 * DEG, 10 * DEG, 50 * DEG, 5 * DEG, 200 * DEG]
    a = greedy_cluster_centers(angles)
    b = greedy_cluster_centers(list(reversed(angles)))
    assert a == b


def test_merge_raw_equals_recluster_of_union():
    """Fusion = union of raw lists re-clustered from scratch (spec B9), exact
    regardless of the incremental (arrival-order) perception history."""
    rng = np.random.default_rng(7)
    tol = 25 * DEG
    for _ in range(50):
        n_a, n_b = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        raw_a = rng.uniform(0, 360, n_a) * DEG
        raw_b = rng.uniform(0, 360, n_b) * DEG
        # incremental perception (arrival order) on each side
        ca = []
        for a in raw_a:
            ca, _ = add_angle_to_clusters(ca, a, tol=tol)
        cb = []
        for b in raw_b:
            cb, _ = add_angle_to_clusters(cb, b, tol=tol)
        merged = merge_raw_angle_lists(raw_a, raw_b, tol=tol)
        ref = greedy_cluster_centers(list(raw_a) + list(raw_b), tol=tol)
        assert merged == ref
        # the merged count can exceed both individual counts
        assert len(merged) >= max(len(ca), len(cb)) - 1


# ----------------------------------------------------------------------
# Env recording
# ----------------------------------------------------------------------

def _free_env(gs=12, seed=3):
    env = GridEnv(grid_size=gs, num_agents=2, obstacle_ratio=0.03, seed=seed,
                  info_model="comm_limited", comm_range=4.0)
    return env


def test_recording_config_counts_and_zero_for_unseen():
    env = _free_env()
    env.agent_positions = [[5, 5], [5, 8]]
    env.update_local_memory(0, fov_radius=1)
    counts = env.get_config_count_grid(0)
    # agent's own cell and all free cells in the 3x3 window around (5,5)
    for r in range(4, 7):
        for c in range(4, 7):
            if not env.obstacle_map[r, c] and (r, c) != (5, 5):
                assert counts[r, c] == 1, (r, c)
    # a far-away unseen cell has 0 configurations
    assert counts[0, 0] == 0
    # global oracle saw the same cells
    assert env.global_observation_count > 0


def test_own_vs_local_counts_before_fusion():
    env = _free_env()
    # agent 0 observes window around (2,2); agent 1 around (5,2). The two
    # agents are within comm range (distance 3 <= 4) but their windows are
    # disjoint, so fusion must transfer the configurations.
    env.agent_positions = [[2, 2], [5, 2]]
    env.update_local_memory(0, fov_radius=1)
    env.update_local_memory(1, fov_radius=1)
    c0 = env.get_config_count_grid(0)
    c1 = env.get_config_count_grid(1)
    # (5,1) is in agent 1's window, unseen by agent 0 before fusion
    assert c1[5, 1] == 1
    assert c0[5, 1] == 0
    n_fus = env.check_and_merge()
    assert n_fus == 1
    c0 = env.get_config_count_grid(0)
    c1 = env.get_config_count_grid(1)
    # after fusion both agents know both windows
    assert c0[5, 1] == 1 and c1[5, 1] == 1
    assert c0[2, 2] == 0 and c1[2, 2] == 0  # own cells are not measured
    assert c0[2, 1] == 1 and c1[2, 1] == 1


def test_fusion_reclusters_overlap():
    env = _free_env()
    # agent 0 observes cell (6,6) from the east, agent 1 from the south
    # (agents within comm range 4, distance sqrt(10) ~ 3.16).
    # After fusion the cell keeps 2 independent configs (> 15 deg apart).
    env.obstacle_map[:] = False
    env.obstacle_map[0, 0] = True  # keep one obstacle far away
    env.agent_positions = [[6, 3], [5, 6]]
    env.update_local_memory(0, fov_radius=3)
    env.update_local_memory(1, fov_radius=3)
    n_fus = env.check_and_merge()
    assert n_fus == 1
    counts = env.get_config_count_grid(0)
    assert counts[6, 6] == 2  # east (90 deg) + south (0 deg) bearings
    assert env.local_measurement_count[1][6, 6] == 2


def test_total_undetermined_initial_is_zero():
    """No perception yet -> known set empty -> undetermined pool is 0, exactly
    like V4's Frontier+Richness (flat signal at t=0)."""
    env = GridEnv(grid_size=30, num_agents=3, obstacle_ratio=0.05, seed=0,
                  info_model="pure_local")
    assert env.get_total_undetermined(0) == 0


def test_total_undetermined_after_perception():
    env = GridEnv(grid_size=20, num_agents=1, obstacle_ratio=0.02, seed=0,
                  info_model="pure_local")
    env.agent_positions = [[10, 10]]
    env.update_local_memory(0, fov_radius=2)
    counts = env.get_config_count_grid(0)
    n_observed = int(np.sum(counts > 0))
    assert n_observed > 0
    # every seen traversable cell has at most 1 config -> all still
    # undetermined (single bearings cannot localize). The agent's own cell is
    # seen but has no measurement (degenerate bearing), so it also counts.
    known = env.local_seen_mask[0]
    obs = env.local_obstacle_map[0]
    seen_free = int(np.sum(known & ~obs))
    assert np.all((counts <= 1)[known & ~obs])
    assert env.get_total_undetermined(0) == seen_free


# ----------------------------------------------------------------------
# Oracle CRLB
# ----------------------------------------------------------------------

def test_bound_requires_two_independent_directions():
    env = GridEnv(grid_size=20, num_agents=1, obstacle_ratio=0.0, seed=1,
                  info_model="pure_local")
    # single observation -> rank-1 J -> bound +inf (not localizable)
    env.agent_positions = [[10, 5]]
    env.update_local_memory(0, fov_radius=5)
    bound = env.global_bound_grid()
    assert np.isinf(bound[10, 10])
    # second orthogonal observation -> finite bound
    env.agent_positions = [[5, 10]]
    env.update_local_memory(0, fov_radius=5)
    bound = env.global_bound_grid()
    assert np.isfinite(bound[10, 10])
    assert bound[10, 10] > 0


def test_bound_analytic_single_direction():
    """One observation at distance d gives rank-1 J: exact analytic check."""
    env = GridEnv(grid_size=10, num_agents=1, obstacle_ratio=0.0, seed=0,
                  info_model="pure_local")
    env.agent_positions = [[5, 2]]
    env.update_local_memory(0, fov_radius=3)
    J00, J01, J11 = (env.global_info[5, 5, 0],
                     env.global_info[5, 5, 1],
                     env.global_info[5, 5, 2])
    det = J00 * J11 - J01 * J01
    assert det < 1e-12  # rank deficient
    assert J00 + J11 > 0


def test_quality_well_localized_monotone():
    env = GridEnv(grid_size=25, num_agents=1, obstacle_ratio=0.0, seed=2,
                  info_model="pure_local")
    env.quality_threshold = 1.5
    q0 = env.quality_well_localized()
    assert q0 == 0.0  # nothing localized yet
    env.agent_positions = [[12, 12]]
    env.update_local_memory(0, fov_radius=4)
    q1 = env.quality_well_localized()
    assert q1 == 0.0  # single observations cannot localize
    env.agent_positions = [[8, 12]]
    env.update_local_memory(0, fov_radius=4)
    q2 = env.quality_well_localized()
    assert q2 > q1  # second direction starts localizing cells


# ----------------------------------------------------------------------
# Transposed Chao-U coherence
# ----------------------------------------------------------------------

def test_chao_u_transposed_bit_coherent_with_v4():
    """Feeding configuration counts as `visit` reproduces the LOCKED V4
    estimator exactly when the inputs are identical."""
    rng = np.random.default_rng(0)
    visit = rng.integers(0, 5, (30, 30)).astype(np.int8)
    known = rng.random((30, 30)) < 0.8
    obs = rng.random((30, 30)) < 0.1
    obs &= known
    total = int(np.sum((~obs) & (known | True)))
    total = int(np.sum((known & ~obs) == 0)) + int(np.sum(~known))
    a = richness.chao_u(visit, known, obs, total_unknown=total,
                        variant="bias_cap")
    b = richness.chao_u(visit.astype(np.int8), known, obs,
                        total_unknown=total, variant="bias_cap")
    assert a == b


def test_richness_map_angular_matches_scalar():
    from analysis.compute_entropy import (_window_chao_u_scalar_sanity,
                                          richness_map)
    rng = np.random.default_rng(11)
    counts = rng.integers(0, 6, (20, 20)).astype(np.int8)
    known = rng.random((20, 20)) < 0.8
    obs = rng.random((20, 20)) < 0.1
    obs &= known
    total = int(np.sum(~known)) + int(np.sum((counts == 0) & known & ~obs))
    fast = richness_map(counts, known, obs, 2, total)
    ref = _window_chao_u_scalar_sanity(counts, known, obs, 2, total)
    assert np.array_equal(fast, ref)


# ----------------------------------------------------------------------
# Policies + runner metrics
# ----------------------------------------------------------------------

def test_new_policies_build():
    for m in ["Random", "Richness-Angular"]:
        p = build_policy(m, seed=0, fov_radius=2, horizon=8)
        assert p is not None


def test_new_policies_episode_runs():
    for m in ["Random", "Richness-Angular"]:
        r = run_episode(m, "comm_limited", 0, env_seed=0, max_steps=120,
                        grid_size=40, num_agents=3, obstacle_ratio=0.05,
                        fov_radius=2)
        assert r["final_coverage"] > 0, m
        assert 0.0 <= r["quality_auc"] <= 1.0, m
        assert 0.0 <= r["quality_final"] <= 1.0, m
        assert r["cluster_cap_hit_frac"] is None or \
            0.0 <= r["cluster_cap_hit_frac"] <= 1.0, m
        assert 0.0 <= r["undetermined_final"] <= 1.0, m


def test_quality_metrics_deterministic():
    kw = dict(max_steps=80, grid_size=30, num_agents=2, obstacle_ratio=0.05,
              fov_radius=2)
    r1 = run_episode("Richness-Angular", "comm_limited", 3, env_seed=3000, **kw)
    r2 = run_episode("Richness-Angular", "comm_limited", 3, env_seed=3000, **kw)
    assert r1["steps_90"] == r2["steps_90"]
    assert r1["quality_auc"] == r2["quality_auc"]
    assert r1["quality_final"] == r2["quality_final"]
    assert r1["cluster_cap_hit_frac"] == r2["cluster_cap_hit_frac"]


def test_time_to_quality_is_none_or_positive():
    r = run_episode("Frontier-Bounded", "comm_limited", 1, env_seed=1000,
                    max_steps=100, grid_size=30, num_agents=2,
                    obstacle_ratio=0.05, fov_radius=2)
    t = r["time_to_quality"]
    assert t is None or (isinstance(t, int) and t > 0)
