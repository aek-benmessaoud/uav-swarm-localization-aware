"""
policies/gdop.py — "GDOP": classical GDOP/FIM baseline policy (MH #2).

Target selection by simulated LOCAL bound improvement, inside the SAME
Frontier-Bounded movement frame as Coverage-U (bounded_bfs horizon R, same
candidate set = unknown cells reachable within the horizon, same fallback).
Only the SCORING SIGNAL differs — this is the apples-to-apples comparison the
reviewer asked for (the E5-CORRECTED lesson: do not conflate signal with
movement frame).

Local GDOP signal (unit-weight direction-only):
  Per cell, the agent knows its local independent angular configurations
  (cluster-center bearings, own + fused). G_x = sum_k u_k u_k^T over the
  unit vectors of those bearings, local bound = sqrt(trace(G^{-1})), capped at
  `bound_cap` when rank-deficient (< 2 independent directions). This is the
  classical geometric dilution of precision: it scores not only the COUNT of
  directions but their GEOMETRY (two near-collinear directions give a huge
  bound despite a "healthy" count of 2) — a genuinely different signal from
  Coverage-U's config-count <= 1.
  The oracle CRLB uses 1/(sigma^2 d^2) weighting, but per-direction observer
  distances are destroyed by fusion (only cluster centers survive), so
  unit-weight GDOP is the defensible LOCAL approximation.

Greedy simulated reduction (standard FIM/CRLB-driven planning, cf. the GDOP /
FIM literature cited in Section 2.3):
  For each candidate target t, simulate standing at t: every traversable cell
  x in FOV(t) (x != t) receives one new bearing from t, so
  G_after = G_x + u(t->x) u(t->x)^T. gain_x = bound_before - bound_after >= 0
  (information-monotone). bonus(t) = sum_x gain_x, and
  score(t) = D(t)/H - lam * bonus(t) / free_count_FOV(t),
  i.e. the SAME scoring form as Coverage-U with the dynamic free-count
  normalization (MH #1). The rank-deficient -> finite transition (first
  independent second bearing) is the dominant gain, matching the oracle metric.

NO ORACLE LEAKAGE: reads only env.local_angle_clusters[agent_id],
get_local_info (local maps) and the agent's own position (the established
position-only channel, used identically by Frontier-Bounded / Coverage-U).
Never touches global_info / global_obs_count / true obstacle_map.
"""

import numpy as np

from policies._common import bounded_bfs, explore_action


class GdopPolicy:
    def __init__(self, seed=None, fov_radius=5, horizon=8, tie_eps=1e-3,
                 lam=0.5, bound_cap=20.0, normalize="free"):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.tie_eps = tie_eps
        self.lam = lam
        self.bound_cap = float(bound_cap)
        self.normalize = normalize
        self.fallback = 0
        self.random_walk = 0
        self._tiny = 1e-12

    def _local_gdop_bound(self, env, agent_id, ra, ca):
        """Direction-only Gram matrices + capped GDOP bound grid on the patch
        of cells that can lie in any candidate's FOV window (targets within
        horizon layers, windows extend by fov_radius). Cells without clusters
        (never observed) keep the bound cap."""
        gs = env.grid_size
        pad = self.horizon + self.fov_radius
        r_lo = max(0, ra - pad)
        r_hi = min(gs, ra + pad + 1)
        c_lo = max(0, ca - pad)
        c_hi = min(gs, ca + pad + 1)

        G00 = np.zeros((gs, gs), dtype=np.float64)
        G01 = np.zeros((gs, gs), dtype=np.float64)
        G11 = np.zeros((gs, gs), dtype=np.float64)
        clusters = env.local_angle_clusters[agent_id]
        for cell, centers in clusters.items():
            r, c = divmod(cell, gs)
            if not (r_lo <= r < r_hi and c_lo <= c < c_hi):
                continue
            if not centers:
                continue
            ang = np.asarray(centers, dtype=np.float64)
            ux = np.cos(ang)
            uy = np.sin(ang)
            G00[r, c] = float(np.sum(ux * ux))
            G01[r, c] = float(np.sum(ux * uy))
            G11[r, c] = float(np.sum(uy * uy))

        det = G00 * G11 - G01 * G01
        ok = det > self._tiny
        bound = np.full((gs, gs), self.bound_cap, dtype=np.float64)
        tr_ = G00 + G11
        bound[ok] = np.sqrt(tr_[ok] / det[ok])
        return G00, G01, G11, bound

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        obs = info["obs"]
        unknown = (~info["known"]).astype(bool)

        D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)
        reachable = np.argwhere((D > 0) & unknown)
        if reachable.size == 0:
            self.fallback += 1
            action, mode = explore_action(env, agent_id, self.rng)
            if mode == "explore_random":
                self.random_walk += 1
            return action, "GDOP", mode, None

        ra, ca = env.agent_positions[agent_id]
        gs = env.grid_size
        G00, G01, G11, bound = self._local_gdop_bound(env, agent_id, ra, ca)
        fov = self.fov_radius
        fov_area = float((2 * fov + 1) ** 2)

        scores = np.full(reachable.shape[0], np.inf, dtype=np.float64)
        for i, (tr, tc) in enumerate(reachable):
            r0 = max(0, tr - fov)
            r1 = min(gs, tr + fov + 1)
            c0 = max(0, tc - fov)
            c1 = min(gs, tc + fov + 1)
            rows = slice(r0, r1)
            cols = slice(c0, c1)

            free_w = ~obs[rows, cols]
            free_w[tr - r0, tc - c0] = False  # no bearing to own cell
            n_free = int(np.sum(free_w))
            if n_free == 0:
                continue

            gy, gx = np.mgrid[r0:r1, c0:c1]
            dy = gy - tr   # row diff (target -> cell)
            dx = gx - tc   # col diff (target -> cell)
            d = np.sqrt(dy * dy + dx * dx)
            # Unit vector target -> cell in (row, col) basis, matching the
            # cluster-center angle convention (env atan2(dcol, drow)):
            # u = (dy/d, dx/d). Own cell (d=0) contributes nothing.
            inv = np.zeros_like(d)
            np.divide(1.0, d, out=inv, where=d > 0)
            ux = dy * inv
            uy = dx * inv

            a00 = G00[rows, cols] + ux * ux
            a01 = G01[rows, cols] + ux * uy
            a11 = G11[rows, cols] + uy * uy
            det_a = a00 * a11 - a01 * a01
            ok_a = det_a > self._tiny
            b_after = np.full_like(a00, self.bound_cap)
            b_after[ok_a] = np.sqrt((a00[ok_a] + a11[ok_a]) / det_a[ok_a])

            gain = bound[rows, cols] - b_after
            gain = np.where(free_w, np.maximum(gain, 0.0), 0.0)
            bonus = float(np.sum(gain))
            denom = float(n_free) if n_free > 0 else fov_area
            scores[i] = D[tr, tc] / self.horizon - self.lam * bonus / denom

        m = np.min(scores)
        ties = reachable[scores == m]
        pick = ties[self.rng.integers(0, ties.shape[0])]
        tr, tc = int(pick[0]), int(pick[1])
        return int(curdir[tr, tc]), "GDOP", "explore", None
