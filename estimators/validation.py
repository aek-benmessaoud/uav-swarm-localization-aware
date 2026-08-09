"""
estimators/validation.py — Empirical validation of the localization model
(Phase 1a).

Purpose: confirm that (i) the bearing observations collected during an episode
actually localize cells (Gauss-Newton triangulation under angular noise), and
(ii) the decision/evaluation signals are valid proxies:
  - CRLB bound  (global oracle, env.global_bound_grid)  vs empirical error:
        positive Spearman correlation (bound grows with error);
  - local config count U (agent's fused view) vs empirical error:
        negative Spearman correlation (more configurations -> lower error).

The Gauss-Newton estimator inlines the exact geometric convention of env.py:
a cell at grid position (r, c) is observed from an agent at (or, oc) with true
bearing  atan2(c - oc, r - or). Noisy bearings are drawn N(0, sigma_deg).
"""

import numpy as np
from scipy.stats import spearmanr

from config import QUALITY_SIGMA_BEARING_DEG


def true_bearing(observer, cell):
    """Bearing from observer (or, oc) to cell (r, c), radians in (-pi, pi]."""
    or_, oc = observer
    r, c = cell
    return np.arctan2(c - oc, r - or_)


def gauss_newton_localize(observers, noisy_bearings, init=None, iters=12,
                          tol=1e-9):
    """Gauss-Newton bearing-only triangulation.

    observers      : list of (or, oc) true observer positions (grid cells).
    noisy_bearings : measured bearings (radians), same length.
    Returns (x, y) estimate in grid-cell coordinates, or (nan, nan) if the
    information matrix is rank deficient (< 2 independent directions).
    """
    obs = np.asarray(observers, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[0] < 2:
        return (float("nan"), float("nan"))
    n = obs.shape[0]
    th = np.asarray(noisy_bearings, dtype=np.float64).reshape(n)

    if init is None:
        p = obs.mean(axis=0).copy()
    else:
        p = np.asarray(init, dtype=np.float64).copy().reshape(2)

    for _ in range(iters):
        dx = p[0] - obs[:, 0]
        dy = p[1] - obs[:, 1]
        d2 = dx * dx + dy * dy
        if np.any(d2 < 1e-12):
            return (float("nan"), float("nan"))  # observer on the target
        J = np.empty((n, 2))
        J[:, 0] = -dy / d2
        J[:, 1] = dx / d2
        residual = (np.arctan2(dy, dx) - th) % (2.0 * np.pi)
        residual = np.where(residual > np.pi, residual - 2.0 * np.pi, residual)

        H = J.T @ J
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if det < 1e-12:
            return (float("nan"), float("nan"))  # rank deficient
        g = J.T @ residual
        step = np.linalg.solve(H, g)
        p = p - step
        if float(np.max(np.abs(step))) < tol:
            break
    return float(p[0]), float(p[1])


def empirical_errors(env, rng, sigma_deg=None, sample_cells=None):
    """Per-cell empirical localization error from the recorded observations.

    Uses env.global_raw_obs (must be enabled during the episode). For every
    traversable cell, triangulate from its TRUE observer positions + Gaussian
    bearing noise; unobservable cells get +inf error. Returns an (gs, gs)
    float grid (inf where not localizable) and the dict cell->error.
    """
    gs = env.grid_size
    if sigma_deg is None:
        sigma_deg = QUALITY_SIGMA_BEARING_DEG
    sigma = np.deg2rad(sigma_deg)

    errors = np.full((gs, gs), np.inf, dtype=np.float64)
    error_map = {}
    tr = ~env.obstacle_map
    cells = (np.argwhere(tr) if sample_cells is None else sample_cells)
    for r, c in cells:
        cell = int(r * gs + c)
        obs_list = env.global_raw_obs.get(cell)
        if not obs_list:
            continue
        th_true = np.array([true_bearing(o, (int(r), int(c)))
                            for o in obs_list])
        noise = rng.normal(0.0, sigma, size=th_true.size)
        x, y = gauss_newton_localize(obs_list, th_true + noise)
        if not np.isfinite(x):
            continue  # remains +inf (not localizable)
        err = np.hypot(x - float(r), y - float(c))
        errors[r, c] = err
        error_map[cell] = err
    return errors, error_map


def spearman(x, y):
    """(rho, p) Spearman rank correlation; (nan, 1.0) if degenerate."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 3:
        return float("nan"), 1.0
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def cell_data(env, rng, include_undetermined=True, sigma_deg=None):
    """Per-cell (bound, u_local, u_global, error) arrays.

    include_undetermined=True  -> all traversable cells (unobserved get +inf);
    include_undetermined=False -> only cells with finite GN error (localizable).
    Returns (bound, u_local, u_global, error) numpy arrays, same cell order.
    """
    errors, _ = empirical_errors(env, rng, sigma_deg=sigma_deg)
    bound = env.global_bound_grid()
    u_local = env.get_config_count_grid(0).astype(np.float64)
    u_global = env.global_config_count_grid().astype(np.float64)
    tr = ~env.obstacle_map
    if not include_undetermined:
        tr = tr & np.isfinite(errors)
    idx = np.argwhere(tr)
    return (bound[idx[:, 0], idx[:, 1]],
            u_local[idx[:, 0], idx[:, 1]],
            u_global[idx[:, 0], idx[:, 1]],
            errors[idx[:, 0], idx[:, 1]])


def validate_phase1a(env, rng, include_undetermined=True):
    """Full Phase-1a validation on a finished episode.

    Returns a dict with, over the selected traversable cells:
      n_cells                  : number of cells tested,
      n_localizable            : cells with finite GN error,
      rho_bound_error          : Spearman(CRLB bound, empirical error),
      rho_U_local_error        : Spearman(local config count, error),
      rho_U_global_error       : Spearman(global config count, error),
      *_p                      : corresponding p-values.
    """
    bound, u_local, u_global, errors = cell_data(
        env, rng, include_undetermined=include_undetermined)
    if bound.size == 0:
        return {"n_cells": 0, "n_localizable": 0,
                "rho_bound_error": float("nan"),
                "rho_U_local_error": float("nan"),
                "rho_U_global_error": float("nan"),
                "rho_bound_error_p": 1.0,
                "rho_U_local_error_p": 1.0,
                "rho_U_global_error_p": 1.0}

    rho_be, p_be = spearman(bound, errors)
    rho_ule, p_ule = spearman(u_local, errors)
    rho_uge, p_uge = spearman(u_global, errors)
    return {
        "n_cells": int(bound.size),
        "n_localizable": int(np.sum(np.isfinite(errors))),
        "rho_bound_error": rho_be,
        "rho_U_local_error": rho_ule,
        "rho_U_global_error": rho_uge,
        "rho_bound_error_p": p_be,
        "rho_U_local_error_p": p_ule,
        "rho_U_global_error_p": p_uge,
    }
