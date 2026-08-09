"""
policies/frontier_bounded.py — MOVEMENT CONTROL (temporary, NOT in the paper).

Isolates the effect of the movement frame from the effect of the signal.
Frontier-Bounded uses the entropy family's EXACT movement machinery
(bounded_bfs horizon R, fallback to explore_action when no candidate in the
horizon) but Frontier's selection rule: nearest reachable UNKNOWN cell
(min BFS layer), random tie-break. No utility, no window, no statistics.

If Frontier-Bounded ~ Frontier  : the bounded movement does not explain the
                                  family's gains -> signal-driven.
If Frontier-Bounded ~ Entropy   : a large part of the FOV=2/3 gain is a
                                  movement artifact, not the signal.
"""

import numpy as np

from policies._common import bounded_bfs, explore_action


class FrontierBoundedPolicy:
    def __init__(self, seed=None, fov_radius=5, horizon=8, tie_eps=1e-3):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.tie_eps = tie_eps
        self.fallback = 0
        self.random_walk = 0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        unknown = (~info["known"]).astype(np.float64)

        D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)

        reachable = np.argwhere((D > 0) & unknown.astype(bool))
        if reachable.size == 0:
            self.fallback += 1
            action, mode = explore_action(env, agent_id, self.rng)
            if mode == "explore_random":
                self.random_walk += 1
            return action, "frontier_bounded", mode, None

        depth_min = D[reachable[:, 0], reachable[:, 1]].min()
        ties = reachable[D[reachable[:, 0], reachable[:, 1]] == depth_min]
        pick = ties[self.rng.integers(0, ties.shape[0])]
        tr, tc = int(pick[0]), int(pick[1])
        return int(curdir[tr, tc]), "frontier_bounded", "explore", None
