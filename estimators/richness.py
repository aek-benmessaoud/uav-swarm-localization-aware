"""
estimators/richness.py — Richness estimators for adaptive exploration.

All estimators operate EXCLUSIVELY on the agent's knowledge bundle
{visit, known, obs} produced by env.get_local_info(). They never touch
the environment's global state directly (leak-free by construction).

Shared logic:
  U_norm = min(U / total_known, 1.0)
  alpha  = U_norm / (U_norm + K), K_DEFAULT = 0.5
"""

import numpy as np

# =============================================================================
# Frequency helpers
# =============================================================================

def _frequency_counts(visit, known, obs, max_count=10):
    """freq[i] = number of known traversable cells visited exactly i times."""
    tr = known & ~obs
    vc = visit[tr]
    freqs = np.zeros(max_count + 1, dtype=np.float64)
    for i in range(1, max_count + 1):
        freqs[i] = float(np.sum(vc == i))
    return freqs


def _traversable_known(known, obs):
    return known & ~obs


# =============================================================================
# Estimators — return U (estimated number of still-unseen cells)
# =============================================================================

def chao_u(visit, known, obs, total_unknown=None, variant="original"):
    """
    Chao-U with selectable formula/cap (used to fix the pure_local saturation).

    variant:
      "original" : U = F1^2/(2F2); if F2 == 0, U = F1. No cap.
      "cap"      : original U, then U = min(U, total_unknown).
      "bias"     : bias-corrected U = F1(F1-1)/(2(F2+1)); the "+1" handles
                   F2=0 natively (U = F1(F1-1)/2), floored at 1.
      "bias_cap" : bias-corrected U, then capped at total_unknown.

    All variants floor U at 1. total_unknown must be provided for cap variants.
    """
    freqs = _frequency_counts(visit, known, obs)
    F1 = freqs[1]
    F2 = freqs[2]
    if variant == "bias" or variant == "bias_cap":
        U = (F1 * (F1 - 1)) / (2.0 * (F2 + 1))
    else:
        if F2 > 0:
            U = (F1 ** 2) / (2.0 * F2)
        else:
            U = F1
    if variant == "cap" or variant == "bias_cap":
        U = min(U, float(max(total_unknown, 0)) if total_unknown is not None else U)
    return max(float(U), 1.0)


def chao_u_components(visit, known, obs):
    """Return (F1, F2) for trace/diagnostics (same counts as chao_u)."""
    freqs = _frequency_counts(visit, known, obs)
    return float(freqs[1]), float(freqs[2])


def jackknife_u(visit, known, obs):
    """Jackknife-U: U = F1 (singleton count). Floor at 1."""
    freqs = _frequency_counts(visit, known, obs)
    return max(float(freqs[1]), 1.0)


def ace_u(visit, known, obs, total_unknown):
    """
    Abundance-based Coverage Estimator.
    U_ACE = S_ACE - S_obs, then capped at min(U_ACE, total_unknown).
    The cap prevents C_ACE ~ 0 during swarm exploration from exploding
    U_ACE beyond the grid; uses the agent's own unknown count.
    """
    tr = _traversable_known(known, obs)
    vc = visit[tr]
    S_obs = float(np.sum(vc > 0))

    freqs = _frequency_counts(visit, known, obs, max_count=10)

    F1 = freqs[1]
    F2 = freqs[2]
    S_rare = float(np.sum(freqs[1:11]))
    N_rare = float(np.sum(np.arange(1, 11) * freqs[1:11]))

    if N_rare <= 1.0:
        return max(float(F1), 1.0)

    C_ACE = 1.0 - F1 / N_rare
    if C_ACE < 1e-10:
        return max(float(F1), 1.0)

    sum_i_im1 = float(np.sum(np.arange(1, 11) * np.arange(0, 10) * freqs[1:11]))
    gamma2 = (S_rare / C_ACE) * sum_i_im1 / (N_rare * (N_rare - 1.0)) - 1.0
    gamma2 = max(gamma2, 0.0)

    S_abund = float(np.sum(vc > 10))
    S_ACE = S_abund + S_rare / C_ACE + (F1 / C_ACE) * gamma2
    U = max(S_ACE - S_obs, 0.0)
    return max(min(U, float(max(total_unknown, 0))), 1.0)


def total_known(known, obs):
    return int(np.sum(_traversable_known(known, obs)))


# =============================================================================
# Normalisation + alpha
# =============================================================================

def u_norm(U, total_known):
    return min(float(U) / max(float(total_known), 1.0), 1.0)


def alpha_from_U(U, total_known, K=0.5):
    """U_norm / (U_norm + K), clipped to [0, 1]."""
    norm = u_norm(U, total_known)
    den = norm + K
    if den <= 1e-12:
        return 0.0, norm
    return float(np.clip(norm / den, 0.0, 1.0)), norm


def alpha_threshold_from_U(U, total_known, K=0.5, tau=0.1, U_max=1.0):
    """Threshold + linear decision strategy.

    alpha = 0                         if U_norm < tau   (pure exploit)
    alpha = (U_norm - tau)/(U_max - tau)               otherwise, clipped to [0,1].

    tau filters noisy low-richness states (drops exploration pressure once the
    local map looks "saturated"); late in an episode this degenerates the
    adaptive estimator into the myopic Greedy exploit, which wins at low FOV.
    K is accepted for signature parity with alpha_from_U (unused here).
    """
    norm = u_norm(U, total_known)
    if norm < tau:
        return 0.0, norm
    span = max(float(U_max) - float(tau), 1e-9)
    return float(np.clip((norm - tau) / span, 0.0, 1.0)), norm


def estimator_from_name(name):
    """Map estimator name -> callable (visit, known, obs[, total_unknown])."""
    return {
        "chao": chao_u,
        "jackknife": jackknife_u,
        "ace": ace_u,
    }[name]
