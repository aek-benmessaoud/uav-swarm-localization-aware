"""
policies/frontier_richness_deploy.py — "Deploy-U": localization-aware deployment.

Research idea 1+3 (flanking formations + station-keeping), built on the
validated Project08 signal: the per-cell independent ANGULAR configuration
count (get_config_count_grid). A cell with <= 1 configuration is under-
localized (rank-deficient bearing-only FIM).

Mode selection (self-gated on the local signal):
  * COVERAGE  — the Frontier-Bounded movement frame (nearest reachable
    unknown cell within the horizon, BFS, fallback explore). While most known
    cells are still under-localized (under_frac > threshold) the best move is
    to keep opening new territory: exploration itself adds configurations.
  * DEPLOY    — once the known map is mostly well-localized but specific known
    cells remain at <= 1 configuration (under_frac <= threshold), the agent
    goes to the worst reachable cell and ORBITS it: each step it moves to the
    free neighbor maximizing the change in bearing to the target, sweeping
    orthogonal viewing angles. Every new bearing far enough from existing
    clusters adds an independent configuration (env angular model), i.e. the
    orbit guarantees angular diversity (flanking emerges naturally when two
    agents converge on the same high-urgency cell from different sides).

Counters (leak-free, all read only the local knowledge interface):
  fallback / random_walk  -> runner fallback_frac / random_walk_frac
  deploy / orbit          -> runner deploy_frac / orbit_frac
"""

import numpy as np

from policies._common import bounded_bfs, explore_action


def _circular_dist(a, b):
    d = abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)
    return d


def _bearing(r_from, c_from, r_to, c_to):
    return np.arctan2(c_to - c_from, r_to - r_from)


class FrontierRichnessDeployPolicy:
    def __init__(self, seed=None, fov_radius=5, horizon=8, tie_eps=1e-3,
                 cooldown=12, under_frac_max=0.30,
                 min_under_cells=3, orbit_radius=2, station_steps=6,
                 approach_depth=12):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.tie_eps = tie_eps
        self.cooldown_max = cooldown
        self.under_frac_max = under_frac_max
        self.min_under_cells = min_under_cells
        self.orbit_radius = orbit_radius
        self.station_steps = station_steps
        self.approach_depth = approach_depth

        self.fallback = 0
        self.random_walk = 0
        self.deploy = 0
        self.orbit = 0

        self._step = 0
        self._cooldown = 0
        self._active = False
        self._target = None
        self._station_left = 0
        self._prev_pos = None

    # ------------------------------------------------------------------
    def select_action(self, env, agent_id):
        self._step += 1
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        config = env.get_config_count_grid(agent_id)
        gs = env.grid_size
        pos = env.agent_positions[agent_id]
        known = info["known"]
        obs = info["obs"]

        if self._active:
            action = self._deploy_step(env, agent_id, pos, config, known, obs, gs)
            if action is not None:
                return action

        self._cooldown = max(0, self._cooldown - 1)
        if self._cooldown == 0 and not self._active:
            action = self._maybe_trigger(env, agent_id, pos, config, known, obs, gs)
            if action is not None:
                return action

        return self._coverage(env, agent_id)

    # ------------------------------------------------------------------
    def _deploy_step(self, env, agent_id, pos, config, known, obs, gs):
        """One step of an active deploy: approach, then orbit the target."""
        tr, tc = self._target
        if not (0 <= tr < gs and 0 <= tc < gs):
            return self._finish()
        if not (known[tr, tc] and not obs[tr, tc]):
            return self._finish()
        if config[tr, tc] >= 2:
            return self._finish()

        r, c = pos
        cheb = max(abs(tr - r), abs(tc - c))
        if cheb > self.orbit_radius:
            # --- approach: bounded BFS to the target (crosses anything free) ---
            act = self._bfs_to_target(env, agent_id, tr, tc)
            if act is not None:
                self.deploy += 1
                self._prev_pos = (r, c)
                return act, "deploy", "approach", None
            return self._finish()

        # --- arrive / orbit ---
        self.deploy += 1
        self.orbit += 1
        if self._station_left <= 0:
            self._station_left = self.station_steps
        self._station_left -= 1

        act = self._orbit_step(env, agent_id, pos, config, obs, gs, tr, tc)
        if self._station_left <= 0:
            self._active = False
            self._cooldown = self.cooldown_max
            self._target = None
        return act, "deploy", "orbit", None


    def _orbit_step(self, env, agent_id, pos, config, obs, gs, tr, tc):
        r, c = pos
        current_angle = _bearing(r, c, tr, tc)
        neigh = []
        for a, nr, nc in _neighbors(r, c, gs):
            if not obs[nr, nc]:
                neigh.append((a, nr, nc))
        if not neigh:
            return 4

        best_a, best_score = None, -1e9
        for a, nr, nc in neigh:
            cand_angle = _bearing(nr, nc, tr, tc)
            s = float(_circular_dist(cand_angle, current_angle))
            if (nr, nc) == (r, c):
                s -= 10.0
            if self._prev_pos is not None and (nr, nc) == self._prev_pos \
                    and len(neigh) > 1:
                s -= 5.0
            if config[nr, nc] > 0:
                s += 0.05 / (1.0 + float(config[nr, nc]))
            if s > best_score:
                best_score = s
                best_a = a
        self._prev_pos = (r, c)
        return best_a if best_a is not None else 4

    def _finish(self):
        self._active = False
        self._cooldown = self.cooldown_max
        self._target = None
        return None

    # ------------------------------------------------------------------
    def _maybe_trigger(self, env, agent_id, pos, config, known, obs, gs):
        free = known & ~obs
        total_free = int(np.sum(free))
        under = free & (config <= 1)
        n_under = int(np.sum(under))
        if n_under < self.min_under_cells:
            return None
        under_frac = n_under / total_free if total_free else 0.0
        if under_frac > self.under_frac_max:
            return None

        # one bounded BFS pass (obstacle-blocked only) -> depth + first-step dir
        D, curdir = self._reach_map(env, agent_id, gs)
        reach = under & (D >= 0) & (D <= self.approach_depth)
        if not reach.any():
            return None

        # pick the worst reachable under-localized cell (0-config first, then
        # nearest, then rng)
        cand = np.argwhere(reach)
        best = None
        best_score = None
        for rt, rc in cand:
            rt, rc = int(rt), int(rc)
            score = (0 if config[rt, rc] == 0 else 1, int(D[rt, rc]),
                     self.rng.random())
            if best is None or score < best_score:
                best_score = score
                best = (rt, rc)
        if best is None:
            return None

        self._active = True
        self._target = best
        self._station_left = 0
        self._prev_pos = tuple(pos)
        tr, tc = best
        act = int(curdir[tr, tc])
        self.deploy += 1
        return act, "deploy", "approach", None

    # ------------------------------------------------------------------
    def _coverage(self, env, agent_id):
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

    # ------------------------------------------------------------------
    # BFS helpers (leak-free: local obstacle knowledge only)
    # ------------------------------------------------------------------
    def _reach_map(self, env, agent_id, gs):
        """One bounded BFS pass from the agent (obstacle-blocked only, unknown
        and known-visited cells are traversable) producing:
          D[i, j]      = BFS layer (0..approach_depth) or -1 if beyond
          curdir[i, j] = first-step action toward (i, j)
        """
        r0, c0 = env.agent_positions[agent_id]
        obs = env.get_obstacle_knowledge(agent_id)
        D = np.full((gs, gs), -1, dtype=np.int16)
        curdir = np.full((gs, gs), 4, dtype=np.int8)
        D[r0, c0] = 0
        cur = np.zeros((gs, gs), dtype=bool)
        cur[r0, c0] = True
        for depth in range(1, self.approach_depth + 1):
            nxt = np.zeros((gs, gs), dtype=bool)
            nxt[:-1, :] |= cur[1:, :]
            nxt[1:, :] |= cur[:-1, :]
            nxt[:, :-1] |= cur[:, 1:]
            nxt[:, 1:] |= cur[:, :-1]
            nxt &= ~obs & (D < 0)
            if not nxt.any():
                break
            mu = np.zeros((gs, gs), dtype=bool); mu[:-1, :] = cur[1:, :]; mu &= nxt
            md = np.zeros((gs, gs), dtype=bool); md[1:, :] = cur[:-1, :]; md &= nxt
            ml = np.zeros((gs, gs), dtype=bool); ml[:, :-1] = cur[:, 1:]; ml &= nxt
            mr = np.zeros((gs, gs), dtype=bool); mr[:, 1:] = cur[:, :-1]; mr &= nxt
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
            cur = nxt
        return D, curdir

    def _bfs_to_target(self, env, agent_id, tr, tc):
        """First action along the shortest path to (tr, tc), bounded by
        approach_depth, crossing only free cells (local obstacle knowledge).
        Returns action 0..4 or None if unreachable."""
        r0, c0 = env.agent_positions[agent_id]
        if (r0, c0) == (tr, tc):
            return 4
        obs = env.get_obstacle_knowledge(agent_id)
        gs = env.grid_size
        visited = np.zeros((gs, gs), dtype=bool)
        visited[r0, c0] = True
        curdir = np.full((gs, gs), 4, dtype=np.int8)
        cur = np.zeros((gs, gs), dtype=bool)
        cur[r0, c0] = True
        for depth in range(1, self.approach_depth + 1):
            nxt = np.zeros((gs, gs), dtype=bool)
            nxt[:-1, :] |= cur[1:, :]
            nxt[1:, :] |= cur[:-1, :]
            nxt[:, :-1] |= cur[:, 1:]
            nxt[:, 1:] |= cur[:, :-1]
            nxt &= ~obs & ~visited
            if not nxt.any():
                return None
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
            if nxt[tr, tc]:
                return int(curdir[tr, tc])
            visited |= nxt
            cur = nxt
        return None


def _neighbors(r, c, gs):
    out = []
    for a, dr, dc in ((0, -1, 0), (1, 1, 0), (2, 0, -1), (3, 0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < gs and 0 <= nc < gs:
            out.append((a, nr, nc))
    return out
