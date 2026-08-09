"""
policies/coverage_u.py — "Coverage-U": U-prioritized coverage under a budget.

Continuous localization-aware target selection inside the validated
Frontier-Bounded movement frame. NO estimator, NO mode switch:

    score(target) = D/horizon - lam * (under_count_FOV(target) / FOV_area)

where D = BFS layer of the target and under_count_FOV(target) = number of
KNOWN-FREE cells with <= 1 independent angular configuration (the accuracy
bottleneck) inside the target's FOV square. The FOV square matches the env's
get_fov_mask footprint (Chebyshev radius fov_radius).

A positive lam steers each agent toward unknown territory adjacent to
under-localized known cells: en-route and destination observations re-observe
those cells from new bearings, adding independent configurations (env angular
model) while coverage proceeds. This is the FIRST test of the config-count
signal on a budget-limited DUAL objective (accuracy AND coverage genuinely
compete): Phase 1 / E3 ran to 100% coverage, which already reaches quality 1.0
and leaves the signal no margin. Here the episode stops at a budget T before
full coverage, so early accuracy has value.

lam = 0 reduces EXACTLY to Frontier-Bounded (identical target set, identical
rng stream) — used as the control-in-the-policy sanity test.

MH #1 (dynamic normalization): `normalize="free"` replaces the constant
FOV_area denominator with the number of TRAVERSABLE cells in the target's FOV
window (free_count_FOV = (known & ~obs) | unknown = ~obs under the codebase
convention obs subseteq known). In obstacle-dominated windows the constant
denominator dilutes the under-set signal (A6_obs020 failure); the dynamic
denominator restores its strength. The default `normalize="area"` reproduces
the original scoring bit-for-bit (paired A/B control).
"""

import numpy as np

from policies._common import bounded_bfs, explore_action


def _box_sum(grid, radius, rs, cs):
    """Integer-image box counts: for each (r, c) in (rs, cs), the number of
    True cells inside the axis-aligned square of half-width `radius` centered
    on it (matches the env's get_fov_mask Chebyshev footprint)."""
    gs = grid.shape[0]
    S = np.zeros((gs + 1, gs + 1), dtype=np.int32)
    S[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
    r0 = np.maximum(0, rs - radius)
    r1 = np.minimum(gs, rs + radius + 1)
    c0 = np.maximum(0, cs - radius)
    c1 = np.minimum(gs, cs + radius + 1)
    return S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]


class CoverageUPolicy:
    def __init__(self, seed=None, fov_radius=5, horizon=8, tie_eps=1e-3,
                 lam=0.5, normalize="area"):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.tie_eps = tie_eps
        self.lam = lam
        self.normalize = normalize
        self.fallback = 0
        self.random_walk = 0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        known = info["known"]
        obs = info["obs"]
        unknown = (~known).astype(bool)
        config = env.get_config_count_grid(agent_id)
        under = known & ~obs & (config <= 1)

        D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)
        reachable = np.argwhere((D > 0) & unknown)
        if reachable.size == 0:
            self.fallback += 1
            action, mode = explore_action(env, agent_id, self.rng)
            if mode == "explore_random":
                self.random_walk += 1
            return action, "coverage_u", mode, None

        fov_area = float((2 * self.fov_radius + 1) ** 2)
        rs = reachable[:, 0]
        cs = reachable[:, 1]
        bonus_term = np.zeros(reachable.shape[0], dtype=np.float64)
        if self.lam > 0:
            bonus = _box_sum(under, self.fov_radius, rs, cs)
            if self.normalize == "free":
                free_count = _box_sum(~obs, self.fov_radius, rs, cs)
                denom = np.where(free_count > 0, free_count, fov_area)
            else:
                denom = fov_area
            bonus_term = self.lam * bonus / denom
        score = (D[rs, cs] / self.horizon - bonus_term)
        m = np.min(score)
        ties = reachable[score == m]
        pick = ties[self.rng.integers(0, ties.shape[0])]
        tr, tc = int(pick[0]), int(pick[1])
        return int(curdir[tr, tc]), "coverage_u", "explore", None
