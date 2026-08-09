"""
analysis/compute_entropy.py — Vectorized per-cell signal maps for V4 policies.

All functions operate on the agent's local knowledge bundle {visit, known, obs}
(no global truth), consistent with the comm_limited info model.

Signal conventions (locked):
  - Probability map p: unknown cell  -> 0.5; known free -> P_known;
    known obstacle -> 1 - P_known. Deterministic, no environmental noise.
  - H(p) = binary entropy in bits (H(0.5)=1, H(0.9)=H(0.1)~0.469).
  - info_gain (count) = number of UNKNOWN cells in the window.
  - info_gain (frac)  = sum of H over the window.
  - richness          = Chao-U bias_cap evaluated on the window.
  - frontier_mask     = known free cells with >= 1 unknown 4-neighbor.
"""

import numpy as np

from estimators.richness import chao_u
from policies._common import box_sum

LOG2 = np.log(2.0)


def p_map(known, obs, p_known=0.9):
    """Deterministic probability map under the sensor-reliability model."""
    p = np.full(known.shape, 0.5, dtype=np.float64)
    known_free = known & ~obs
    known_obs = known & obs
    p[known_free] = p_known
    p[known_obs] = 1.0 - p_known
    return p


def H_map(p):
    """Binary entropy in bits, vectorized; H(0)=H(1)=0."""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p)) / LOG2


def info_gain_count_map(unknown, w):
    """Per-cell count of unknown cells in the radius-w window (count signal)."""
    return box_sum(unknown, w)


def info_gain_frac_map(known, obs, w, p_known=0.9):
    """Per-cell sum of H(p) over the radius-w window (fractional signal)."""
    return box_sum(H_map(p_map(known, obs, p_known)), w)


def richness_map(visit, known, obs, w, total_unknown):
    """Per-cell Chao-U bias_cap on the radius-w window.

    For every cell c: F1(c) = # traversable cells with visit==1 inside the
    window, F2(c) = # with visit==2. U_bias = F1(F1-1)/(2(F2+1)) is then
    capped at `total_unknown` and floored at 1, matching richness.chao_u
    with variant="bias_cap" evaluated window-wise.
    """
    traversable = known & ~obs
    ones = traversable & (visit == 1)
    twos = traversable & (visit == 2)
    F1 = box_sum(ones.astype(np.float64), w)
    F2 = box_sum(twos.astype(np.float64), w)
    U = (F1 * (F1 - 1.0)) / (2.0 * (F2 + 1.0))
    U = np.minimum(U, float(max(int(total_unknown), 0)))
    return np.maximum(U, 1.0)


def frontier_mask(known, obs, unknown):
    """Known free cells adjacent (4-neighborhood) to >= 1 unknown cell."""
    unknown = np.asarray(unknown, dtype=bool)
    gs = known.shape[0]
    adj = np.zeros_like(unknown)
    adj[:-1, :] |= unknown[1:, :]
    adj[1:, :] |= unknown[:-1, :]
    adj[:, :-1] |= unknown[:, 1:]
    adj[:, 1:] |= unknown[:, :-1]
    return (known & ~obs & adj).astype(np.float64)


def select_target(utility, D, curdir, candidate_mask, rng, tie_eps=1e-3):
    """Pick the argmax utility cell among candidates with D>0 (within the BFS
    horizon), tie-broken by a small per-agent scatter. Returns the first-step
    action (int) or None if no candidate is reachable within the horizon."""
    reachable = (np.asarray(candidate_mask, dtype=bool)) & (D > 0)
    if not reachable.any():
        return None
    vals = utility[reachable].copy()
    if tie_eps > 0:
        vals = vals + rng.uniform(-tie_eps, tie_eps, size=vals.shape)
    j = int(np.argmax(vals))
    idx = np.argwhere(reachable)[j]
    tr, tc = int(idx[0]), int(idx[1])
    return int(curdir[tr, tc])


def window_radius(default, configured):
    """Resolve ENTROPY_WINDOW: None -> default (FOV radius)."""
    return default if configured is None else configured


def _window_chao_u_scalar_sanity(visit, known, obs, w, total_unknown):
    """Reference scalar implementation (tests only): brute-force window loop,
    equivalent to richness_map but O(N^2) — used to validate the vectorized
    version."""
    gs = visit.shape[0]
    out = np.zeros((gs, gs), dtype=np.float64)
    for r in range(gs):
        for c in range(gs):
            r0, r1 = max(0, r - w), min(gs, r + w + 1)
            c0, c1 = max(0, c - w), min(gs, c + w + 1)
            sub_visit = visit[r0:r1, c0:c1]
            sub_known = known[r0:r1, c0:c1]
            sub_obs = obs[r0:r1, c0:c1]
            out[r, c] = chao_u(sub_visit, sub_known, sub_obs,
                               total_unknown=total_unknown, variant="bias_cap")
    return out
