"""
policies/central_oracle.py — CENTRALIZED ORACLE upper bound (E5, control row).

The pre-registered E5 question: the proposed methods use ONLY the local
config-count signal (Coverage-U). How much of the accuracy gain would a
centralized policy with GLOBAL perfect knowledge achieve on the same
dual objective? The oracle maximizes

    score(target) = D/horizon - lam * (under_count_FOV(target) / FOV_area)

with the SAME scoring form as Coverage-U, but with global knowledge instead of
the comm-limited local frame:

  - movement frame: bounded BFS through true free / unvisited cells (perfect
    map), target set = cells never observed by ANY agent (global unknown);
  - under-set signal (chosen by `mode`):
      mode="config" : globally-observed-free cells with GLOBAL config count
                      <= 1  (perfect-fusion version of Coverage-U's signal);
      mode="crlb"   : globally-observed-free cells whose ORACLE CRLB bound
                      sqrt(trace(J^-1)) > QUALITY_THRESHOLD (1.5 cells) —
                      the true accuracy bottleneck, impossible to know locally.

The ladder Coverage-U -> CentralConfig -> CentralCRLB isolates (i) the value
of perfect fusion and (ii) the proxy gap config-count vs true CRLB. The oracle
is a CONTROL row, never a proposed method: it is infeasible in the deployed
system (needs the true map + oracle CRLB). It only anchors the "perfect
centralized" ceiling against which the transposition ratio
reduction_CU / reduction_CentralCRLB is measured.
"""

import numpy as np

from policies.coverage_u import _box_sum
from policies._common import bounded_bfs


def _global_bfs(r, c, gs, free_path, max_depth):
    """Bounded BFS on a GLOBAL free-path mask (same algorithm as
    policies/_common.bounded_bfs but masks are supplied by the caller).

    Returns (D, curdir): D[i, j] = BFS layer (1..max_depth) or -1;
    curdir[i, j] = first-step action toward (i, j) (4 = stay/unset).
    """
    D = np.full((gs, gs), -1, dtype=np.int16)
    curdir = np.full((gs, gs), 4, dtype=np.int8)
    D[r, c] = 0
    visited = np.zeros((gs, gs), dtype=bool)
    visited[r, c] = True
    cur = np.zeros((gs, gs), dtype=bool)
    cur[r, c] = True
    depth = 0

    while True:
        depth += 1
        if depth > max_depth:
            break
        nxt = np.zeros((gs, gs), dtype=bool)
        nxt[:-1, :] |= cur[1:, :]   # up
        nxt[1:, :] |= cur[:-1, :]   # down
        nxt[:, :-1] |= cur[:, 1:]   # left
        nxt[:, 1:] |= cur[:, :-1]   # right
        nxt &= free_path & ~visited
        if not nxt.any():
            break

        mu = np.zeros((gs, gs), dtype=bool); mu[:-1, :] = cur[1:, :]; mu &= nxt
        md = np.zeros((gs, gs), dtype=bool); md[1:, :] = cur[:-1, :]; md &= nxt
        ml = np.zeros((gs, gs), dtype=bool); ml[:, :-1] = cur[:, 1:]; ml &= nxt
        mr = np.zeros((gs, gs), dtype=bool); mr[:, 1:] = cur[:, :-1]; mr &= nxt

        if depth == 1:
            cd = np.full((gs, gs), 4, dtype=np.int8)
            cd[mu] = 0
            cd[md] = np.where(cd[md] == 4, 1, cd[md])
            cd[ml] = np.where(cd[ml] == 4, 2, cd[ml])
            cd[mr] = np.where(cd[mr] == 4, 3, cd[mr])
            curdir = cd
        else:
            nd = curdir.copy()
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[:-1, :] = np.where(mu[:-1, :], curdir[1:, :], 4)
            nd = np.minimum(nd, cand)
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[1:, :] = np.where(md[1:, :], curdir[:-1, :], 4)
            nd = np.minimum(nd, cand)
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[:, :-1] = np.where(ml[:, :-1], curdir[:, 1:], 4)
            nd = np.minimum(nd, cand)
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[:, 1:] = np.where(mr[:, 1:], curdir[:, :-1], 4)
            nd = np.minimum(nd, cand)
            curdir = nd

        D[nxt] = depth
        visited |= nxt
        cur = nxt

    return D, curdir


def _valid_global(env, r, c, obs, include_stay=False):
    """Valid moves in the true free space (no obstacle knowledge limit)."""
    gs = env.grid_size
    moves = [(0, -1, 0), (1, 1, 0), (2, 0, -1), (3, 0, 1)]
    if include_stay:
        moves.append((4, 0, 0))
    out = []
    for action, dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < gs and 0 <= nc < gs and not obs[nr, nc]:
            out.append((action, nr, nc))
    return out


class CentralOraclePolicy:
    def __init__(self, seed=None, fov_radius=5, horizon=8, tie_eps=1e-3,
                 lam=0.5, mode="crlb", coverage_guard_eps=None, frame="global"):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.tie_eps = tie_eps
        self.lam = lam
        self.mode = mode
        self.coverage_guard_eps = coverage_guard_eps
        self.frame = frame
        self.fallback = 0
        self.random_walk = 0

    def _under_set(self, env):
        obs = env.obstacle_map
        seen = env.global_obs_count > 0
        if self.mode == "config":
            cfg = env.global_config_count_grid()
            return seen & ~obs & (cfg <= 1)
        bound = env.global_bound_grid()
        thr = getattr(env, "quality_threshold", None)
        if thr is None:
            from config import QUALITY_THRESHOLD
            thr = QUALITY_THRESHOLD
        return seen & ~obs & (bound > thr)

    def _explore(self, env, agent_id, obs, visit):
        """Global explore: BFS to the nearest unvisited free cell in the true
        map; fallback random valid move. Returns a single action int."""
        r, c = env.agent_positions[agent_id]
        gs = env.grid_size
        target = ~obs & (visit == 0)
        if target[r, c]:
            return 4
        D, curdir = _global_bfs(r, c, gs, ~obs, self.horizon)
        hits = np.argwhere((D > 0) & target)
        if hits.size:
            tr, tc = int(hits[0][0]), int(hits[0][1])
            return int(curdir[tr, tc])
        neigh = _valid_global(env, r, c, obs, include_stay=True)
        if neigh:
            self.random_walk += 1
            return int(self.rng.choice([a for a, _, _ in neigh]))
        return 4

    def select_action(self, env, agent_id):
        # Generate the sensor observation for this step (same hook every policy
        # uses). This feeds the env's global angular / Fisher state
        # (global_obs_count, global_info, global_angle_clusters) that the oracle
        # reads; WITHOUT it no observations ever happen and the env stalls.
        env.update_local_memory(agent_id, self.fov_radius)

        obs = env.obstacle_map
        visit = env.visit_count

        # Movement frame: either the global oracle frame (no path through any
        # visited cell, true map) or the LOCAL bounded_bfs frame identical to
        # Coverage-U/FB. The frame="local" variant isolates the frame effect:
        # the movement machinery is byte-identical to the decentralized
        # methods and ONLY the under-set signal is global.
        if self.frame == "local":
            info = env.get_local_info(agent_id)
            unknown = (~info["known"]).astype(bool)
            D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)
        else:
            seen = env.global_obs_count > 0
            known = seen & ~obs
            unknown = ~seen
            free_path = ~obs & ~(known & (visit > 0))
            r, c = env.agent_positions[agent_id]
            D, curdir = _global_bfs(r, c, env.grid_size, free_path,
                                    self.horizon)
        reachable = np.argwhere((D > 0) & unknown)
        if reachable.size == 0:
            self.fallback += 1
            return self._explore(env, agent_id, obs, visit), \
                "central_oracle", "explore", None

        under = self._under_set(env)
        fov_area = float((2 * self.fov_radius + 1) ** 2)
        rs = reachable[:, 0]
        cs = reachable[:, 1]
        bonus = np.zeros(reachable.shape[0], dtype=np.int32)
        if self.lam > 0:
            bonus = _box_sum(under, self.fov_radius, rs, cs)
        cov_term = D[rs, cs] / self.horizon
        bonus_term = self.lam * bonus / fov_area
        if self.coverage_guard_eps is not None:
            bonus_term = np.minimum(bonus_term, (1.0 - self.coverage_guard_eps)
                                    * cov_term)
        score = cov_term - bonus_term
        m = np.min(score)
        ties = reachable[score == m]
        pick = ties[self.rng.integers(0, ties.shape[0])]
        tr, tc = int(pick[0]), int(pick[1])
        return int(curdir[tr, tc]), "central_oracle", "explore", None
