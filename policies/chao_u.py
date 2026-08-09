"""
policies/chao_u.py — Adaptive hybrid driven by LOCAL Chao-U.

variant: "original" | "cap" | "bias" | "bias_cap" (see estimators.richness.chao_u).
The decision is the locked logistic gate alpha = U_norm/(U_norm + K).
"""

import numpy as np
from estimators import richness
from policies._common import explore_action, exploit_action


class ChaoUPolicy:
    def __init__(self, K=0.5, seed=None, fov_radius=5, variant="original"):
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.variant = variant
        self.random_walk = 0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)

        total_unknown = env.get_total_unknown(agent_id)
        U = richness.chao_u(info["visit"], info["known"], info["obs"],
                            total_unknown=total_unknown, variant=self.variant)
        total_known = richness.total_known(info["known"], info["obs"])
        alpha, norm = richness.alpha_from_U(U, total_known, self.K)

        if self.rng.random() < alpha:
            action, mode = explore_action(env, agent_id, self.rng)
            if mode == "explore_random":
                self.random_walk += 1
        else:
            action, mode = exploit_action(env, agent_id, self.rng)
        return action, "chao_u", mode, alpha
