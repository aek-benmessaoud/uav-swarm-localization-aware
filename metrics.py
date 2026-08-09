"""
metrics.py — Global coverage metrics (GLOBAL ground truth only; these are
measurements for the report, never inputs to any policy).
"""

import numpy as np


def coverage_percent(env):
    visited = int(np.sum((env.visit_count > 0) & ~env.obstacle_map))
    return 100.0 * visited / env.traversable if env.traversable else 0.0


def steps_to_threshold(coverage_history, threshold=90.0):
    """First 1-based step index where coverage reached threshold; None if never."""
    for step, cov in enumerate(coverage_history, start=1):
        if cov >= threshold:
            return step
    return None


def overlap_ratio(env):
    """Fraction of total visits that are redundant (revisits)."""
    vc = env.visit_count[~env.obstacle_map]
    total = int(vc.sum())
    if total == 0:
        return 0.0
    unique = int(np.sum(vc > 0))
    return (total - unique) / total


def unseen_estimator(env):
    """Bounded global Chao-style estimate of remaining unseen cells (diagnostic)."""
    vc = env.visit_count[~env.obstacle_map]
    F1 = int(np.sum(vc == 1))
    F2 = int(np.sum(vc == 2))
    actual_unseen = int(np.sum(vc == 0))
    U = float(F1) if F2 == 0 else (F1 ** 2) / (2.0 * F2)
    return min(U, float(actual_unseen))


def mean_nearest_neighbor_distance(agent_positions):
    pos = np.array(agent_positions).astype(float)
    if len(pos) <= 1:
        return 0.0
    dists = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return float(np.mean(np.min(dists, axis=1)))
