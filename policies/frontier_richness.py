"""
policies/frontier_richness.py — Geometric + statistical-richness hybrid (V4).

Candidates = FRONTIER cells reachable within a bounded-BFS horizon R.
Utility = window Chao-U (bias_cap) richness of the candidate's neighborhood,
/ (BFS layer + eps). The first BFS step is the action.
"""

import numpy as np

from analysis.compute_entropy import (frontier_mask, richness_map,
                                      select_target)
from policies._common import bounded_bfs, explore_action


class FrontierRichnessPolicy:
    def __init__(self, seed=None, fov_radius=5, horizon=8, window=None,
                 eps=1.0, tie_eps=1e-3):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.window = window if window is not None else fov_radius
        self.eps = eps
        self.tie_eps = tie_eps
        self.fallback = 0
        self.random_walk = 0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        unknown = (~info["known"]).astype(np.float64)
        frontier = frontier_mask(info["known"], info["obs"], unknown)

        D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)
        gain = richness_map(info["visit"], info["known"], info["obs"],
                            self.window, env.get_total_unknown(agent_id))

        utility = np.full_like(D, -np.inf, dtype=np.float64)
        pos = D > 0
        utility[pos] = gain[pos] / (D[pos] + self.eps)

        action = select_target(utility, D, curdir, frontier, self.rng,
                               tie_eps=self.tie_eps)
        if action is None:
            self.fallback += 1
            action, mode = explore_action(env, agent_id, self.rng)
            if mode == "explore_random":
                self.random_walk += 1
            return action, "Frontier+Richness", mode, None
        return int(action), "Frontier+Richness", "frontier_richness", None
