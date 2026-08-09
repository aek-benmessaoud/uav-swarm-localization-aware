"""
policies/hybrid.py — "Hybrid": one policy, two local scorings, one frame.

Routes between the two validated signal families by the agent's CURRENT
FOV-window free fraction (per agent, per decision step — NOT per candidate
target, so no cross-scale score mixing on the same candidate list):

    free_frac >= THETA  -> Coverage-U-norm score  (open / sparse windows)
    free_frac <  THETA  -> Richness-Angular utility (dense / fragmented)

The Frontier-Bounded movement frame is the NON-NEGOTIABLE invariant
(E5-CORRECTED lesson): same bounded_bfs, same horizon H=8, and the SAME
candidate set for both modes (reachable unknown cells) — only the score
changes, never the frame or the candidate list.

- CU-norm branch (policies.coverage_u, normalize="free"):
      score(t) = D/H - lam * under_count_FOV(t) / free_count_FOV(t)   (min)
- RA branch (policies.frontier_richness_angular):
      utility(t) = richness_map(t) / (D + eps)                        (max)
  evaluated on the SAME reachable-unknown candidates.

THETA was frozen BEFORE the campaign from probe_hybrid_theta.py (theta=0.8 =
1 - obs_ratio of the dense regime). `ra_steps` / `mode_steps` counters feed
the runner's `hybrid_ra_frac` column (fraction of scored decisions in RA
mode) so the trigger actually being exercised is verifiable per campaign.
"""

import numpy as np

from analysis.compute_entropy import richness_map
from policies._common import bounded_bfs, explore_action
from policies.coverage_u import _box_sum


def _window_free_frac(obs, r, c, radius):
    """Free fraction of the agent's CURRENT FOV window: number of traversable
    (non-obstacle-by-local-belief) cells over the ACTUAL clamped window area
    (the env's get_fov_mask Chebyshev footprint shrinks at the grid border, so
    the denominator must be the real window, not the constant FOV_area)."""
    gs = obs.shape[0]
    r0 = max(0, r - radius)
    r1 = min(gs, r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(gs, c + radius + 1)
    area = (r1 - r0) * (c1 - c0)
    if area <= 0:
        return 0.0
    free = _box_sum(~obs, radius, np.array([r]), np.array([c]))[0]
    return float(free) / area


class HybridPolicy:
    def __init__(self, seed=None, fov_radius=5, horizon=8, tie_eps=1e-3,
                 lam=0.5, theta=0.8, window=None, eps=1.0):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.tie_eps = tie_eps
        self.lam = lam
        self.theta = theta
        self.window = window if window is not None else fov_radius
        self.eps = eps
        self.fallback = 0
        self.random_walk = 0
        self.ra_steps = 0
        self.mode_steps = 0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        known = info["known"]
        obs = info["obs"]
        unknown = (~known).astype(bool)
        config = env.get_config_count_grid(agent_id)

        D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)
        reachable = np.argwhere((D > 0) & unknown)
        if reachable.size == 0:
            self.fallback += 1
            action, mode = explore_action(env, agent_id, self.rng)
            if mode == "explore_random":
                self.random_walk += 1
            return action, "Hybrid", mode, None

        rs = reachable[:, 0]
        cs = reachable[:, 1]

        ra, ca = env.agent_positions[agent_id]
        free_frac = _window_free_frac(obs, int(ra), int(ca), self.fov_radius)

        self.mode_steps += 1
        if free_frac >= self.theta:
            fov_area = float((2 * self.fov_radius + 1) ** 2)
            bonus = _box_sum(known & ~obs & (config <= 1),
                             self.fov_radius, rs, cs)
            target_free = _box_sum(~obs, self.fov_radius, rs, cs)
            denom = np.where(target_free > 0, target_free, fov_area)
            score = (D[rs, cs] / self.horizon
                     - self.lam * bonus / denom)
            m = np.min(score)
            ties = reachable[score == m]
            pick = ties[self.rng.integers(0, ties.shape[0])]
            tr, tc = int(pick[0]), int(pick[1])
            return int(curdir[tr, tc]), "Hybrid", "hybrid_cunorm", None

        self.ra_steps += 1
        total_und = env.get_total_undetermined(agent_id)
        gain = richness_map(config, known, obs, self.window, total_und)
        utility = gain[rs, cs] / (D[rs, cs] + self.eps)
        m = np.max(utility)
        ties = reachable[utility == m]
        pick = ties[self.rng.integers(0, ties.shape[0])]
        tr, tc = int(pick[0]), int(pick[1])
        return int(curdir[tr, tc]), "Hybrid", "hybrid_ra", None
