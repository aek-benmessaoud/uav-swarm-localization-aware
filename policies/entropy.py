"""
policies/entropy.py — Information-driven exploration policies (V4).

Two flavours of the SAME target-selection rule (locked experiment design):
  Entropy       : utility = count of unknown cells in a radius-w window around
                  the candidate, divided by (BFS layer + eps).
  Entropy-Frac  : utility = sum of fractional H(p) over the window (sensor
                  reliability model: p=0.9 free / 0.1 obstacle / 0.5 unknown),
                  divided by (BFS layer + eps).

Both pick the argmax among UNKNOWN cells reachable within a bounded-BFS
horizon R (movement frame: paths through non-visited cells only). The first
BFS step is the action. Fallback to explore_action when no target is within
the horizon.
"""

import numpy as np

from analysis.compute_entropy import (H_map, info_gain_count_map,
                                      info_gain_frac_map, p_map, select_target)
from policies._common import bounded_bfs, explore_action


class _EntropyBase:
    def __init__(self, seed=None, fov_radius=5, horizon=8, window=None,
                 p_known=0.9, eps=1.0, tie_eps=1e-3):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.window = window if window is not None else fov_radius
        self.p_known = p_known
        self.eps = eps
        self.tie_eps = tie_eps
        self.fallback = 0
        self.random_walk = 0

    def _gain(self, info):
        raise NotImplementedError

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        unknown = (~info["known"]).astype(np.float64)

        D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)
        gain = self._gain(info)

        utility = np.full_like(D, -np.inf, dtype=np.float64)
        pos = D > 0
        utility[pos] = gain[pos] / (D[pos] + self.eps)

        action = select_target(utility, D, curdir, unknown, self.rng,
                               tie_eps=self.tie_eps)
        if action is None:
            self.fallback += 1
            action, mode = explore_action(env, agent_id, self.rng)
            if mode == "explore_random":
                self.random_walk += 1
            return action, self.tag, mode, None
        return int(action), self.tag, "entropy", None


class EntropyPolicy(_EntropyBase):
    """Count signal: number of unknown cells in the window (H=1 per unknown)."""

    tag = "Entropy"

    def _gain(self, info):
        unknown = (~info["known"]).astype(np.float64)
        return info_gain_count_map(unknown, self.window)


class EntropyFracPolicy(_EntropyBase):
    """Fractional signal: sum of binary entropy H(p) over the window."""

    tag = "Entropy-Frac"

    def _gain(self, info):
        return info_gain_frac_map(info["known"], info["obs"], self.window,
                                  p_known=self.p_known)
