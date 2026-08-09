"""
policies/frontier.py — Frontier exploration: BFS to nearest locally-unvisited cell.
"""

import numpy as np
from policies._common import explore_action


class FrontierPolicy:
    def __init__(self, seed=None, fov_radius=5):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.random_walk = 0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        action, mode = explore_action(env, agent_id, self.rng)
        if mode == "explore_random":
            self.random_walk += 1
        return action, "frontier", mode, None
