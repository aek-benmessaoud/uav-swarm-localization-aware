"""
policies/random_walk.py — Random policy with local repulsion (Project08).

Uniformly random valid move (unknown cells are walkable). If another agent is
closer than `repulsion_dist` (grid cells), the agent instead moves to the valid
neighbor that maximizes the distance to the nearest other agent (tie-broken
randomly). Repulsion uses the position-only channel (like collision resolution),
independent of COMM_RANGE/fusion.
"""

import numpy as np

from policies._common import valid_neighbors_local


class RandomPolicy:
    def __init__(self, seed=None, fov_radius=5, repulsion_dist=2.0):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.repulsion_dist = repulsion_dist
        self.random_walk = 0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        r, c = env.agent_positions[agent_id]
        others = [p for i, p in enumerate(env.agent_positions) if i != agent_id]
        if others:
            nearest = min(_dist2((r, c), p) for p in others)
            if nearest < self.repulsion_dist ** 2:
                return self._repulse(env, agent_id, r, c, others)
        neigh = valid_neighbors_local(env, agent_id, r, c, include_stay=True)
        if not neigh:
            return 4, "random", "explore_stay", None
        return int(self.rng.choice([a for a, _, _ in neigh])), \
            "random", "explore_random", None

    def _repulse(self, env, agent_id, r, c, others):
        best_actions = []
        best_score = -1.0
        neigh = valid_neighbors_local(env, agent_id, r, c, include_stay=True)
        for a, nr, nc in neigh:
            score = min(_dist2((nr, nc), p) for p in others)
            if score > best_score:
                best_score = score
                best_actions = [a]
            elif abs(score - best_score) < 1e-9:
                best_actions.append(a)
        if best_actions:
            self.random_walk += 1
            return int(self.rng.choice(best_actions)), "random", "repulse", None
        return 4, "random", "explore_stay", None


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
