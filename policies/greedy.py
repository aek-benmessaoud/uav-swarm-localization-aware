"""
policies/greedy.py — "Least-Visited Greedy": always exploit the known cell
with the lowest OWN visit count. (V1's "Voronoi" — renamed, real Voronoi dropped.)
"""

import numpy as np
from policies._common import exploit_action


class GreedyPolicy:
    def __init__(self, seed=None, fov_radius=5):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        action, mode = exploit_action(env, agent_id, self.rng)
        return action, "greedy", mode, None
