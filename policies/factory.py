"""
policies/factory.py — Build any policy from a canonical method name.
Kept in ONE place so run_*.py and tests share identical construction.
"""

from policies.frontier import FrontierPolicy
from policies.greedy import GreedyPolicy
from policies.chao_u import ChaoUPolicy
from policies.entropy import EntropyPolicy, EntropyFracPolicy
from policies.frontier_entropy import FrontierEntropyPolicy
from policies.frontier_richness import FrontierRichnessPolicy
from policies.frontier_bounded import FrontierBoundedPolicy
from policies.random_walk import RandomPolicy
from policies.frontier_richness_angular import FrontierRichnessAngularPolicy
from policies.frontier_richness_deploy import FrontierRichnessDeployPolicy
from policies.coverage_u import CoverageUPolicy
from policies.gdop import GdopPolicy
from policies.hybrid import HybridPolicy
from policies.central_oracle import CentralOraclePolicy


def build_policy(method_name, seed=None, fov_radius=5, K=0.5,
                 chao_variant=None, horizon=8, window=None, p_known=0.9,
                 eps=1.0, tie_eps=1e-3, lam=None):
    """
    method_name: one of the canonical names in config.py.
    Returns a fresh policy instance (never shared between agents).
    """
    from config import (METHOD_CHAO_U, METHOD_FRONTIER, METHOD_GREEDY,
                        METHOD_ENTROPY, METHOD_ENTROPY_FRAC,
                        METHOD_FRONTIER_ENTROPY, METHOD_FRONTIER_RICHNESS,
                        METHOD_FRONTIER_BOUNDED, METHOD_RANDOM,
                        METHOD_FRONTIER_RICHNESS_ANGULAR, METHOD_DEPLOY,
                        METHOD_COVERAGE_U,
                        METHOD_COVERAGE_U_NORM, METHOD_GDOP,
                        METHOD_HYBRID, HYBRID_THETA,
                        METHOD_CENTRAL_CRLB, METHOD_CENTRAL_CONFIG,
                        METHOD_CENTRAL_CRLB_COV, METHOD_CENTRAL_CRLB_COV2,
                        METHOD_CENTRAL_CRLB_LOCAL, METHOD_CENTRAL_CONFIG_LOCAL,
                        DEPLOY_COOLDOWN, DEPLOY_UNDER_FRAC_MAX,
                        DEPLOY_MIN_UNDER_CELLS, DEPLOY_ORBIT_RADIUS,
                        DEPLOY_STATION_STEPS, DEPLOY_APPROACH_DEPTH,
                        COVERAGE_U_LAMBDA, CENTRAL_ORACLE_LAMBDA,
                        GDOP_LAMBDA, GDOP_BOUND_CAP)

    common = dict(seed=seed, fov_radius=fov_radius, horizon=horizon,
                  window=window, p_known=p_known, eps=eps, tie_eps=tie_eps)
    richness_kw = {k: v for k, v in common.items() if k != "p_known"}

    if method_name == METHOD_FRONTIER:
        return FrontierPolicy(seed=seed, fov_radius=fov_radius)
    if method_name == METHOD_GREEDY:
        return GreedyPolicy(seed=seed, fov_radius=fov_radius)
    if method_name == METHOD_CHAO_U:
        return ChaoUPolicy(K=K, seed=seed, fov_radius=fov_radius,
                           variant=chao_variant or "original")
    if method_name == METHOD_ENTROPY:
        return EntropyPolicy(**common)
    if method_name == METHOD_ENTROPY_FRAC:
        return EntropyFracPolicy(**common)
    if method_name == METHOD_FRONTIER_ENTROPY:
        return FrontierEntropyPolicy(**common)
    if method_name == METHOD_FRONTIER_RICHNESS:
        return FrontierRichnessPolicy(**richness_kw)
    if method_name == METHOD_FRONTIER_BOUNDED:
        return FrontierBoundedPolicy(seed=seed, fov_radius=fov_radius,
                                     horizon=horizon, tie_eps=tie_eps)
    if method_name == METHOD_RANDOM:
        from config import RANDOM_REPULSION_DIST
        return RandomPolicy(seed=seed, fov_radius=fov_radius,
                            repulsion_dist=RANDOM_REPULSION_DIST)
    if method_name == METHOD_FRONTIER_RICHNESS_ANGULAR:
        return FrontierRichnessAngularPolicy(**richness_kw)
    if method_name == METHOD_DEPLOY:
        return FrontierRichnessDeployPolicy(
            seed=seed, fov_radius=fov_radius, horizon=horizon,
            cooldown=DEPLOY_COOLDOWN, under_frac_max=DEPLOY_UNDER_FRAC_MAX,
            min_under_cells=DEPLOY_MIN_UNDER_CELLS,
            orbit_radius=DEPLOY_ORBIT_RADIUS,
            station_steps=DEPLOY_STATION_STEPS,
            approach_depth=DEPLOY_APPROACH_DEPTH)
    if method_name == METHOD_COVERAGE_U:
        return CoverageUPolicy(seed=seed, fov_radius=fov_radius,
                               horizon=horizon, tie_eps=tie_eps,
                               lam=COVERAGE_U_LAMBDA if lam is None else lam)
    if method_name == METHOD_COVERAGE_U_NORM:
        return CoverageUPolicy(seed=seed, fov_radius=fov_radius,
                               horizon=horizon, tie_eps=tie_eps,
                               lam=COVERAGE_U_LAMBDA if lam is None else lam,
                               normalize="free")
    if method_name == METHOD_GDOP:
        return GdopPolicy(seed=seed, fov_radius=fov_radius,
                          horizon=horizon, tie_eps=tie_eps,
                          lam=GDOP_LAMBDA if lam is None else lam,
                          bound_cap=GDOP_BOUND_CAP, normalize="free")
    if method_name == METHOD_HYBRID:
        return HybridPolicy(seed=seed, fov_radius=fov_radius,
                            horizon=horizon, tie_eps=tie_eps,
                            lam=COVERAGE_U_LAMBDA if lam is None else lam,
                            theta=HYBRID_THETA)
    if method_name == METHOD_CENTRAL_CRLB:
        return CentralOraclePolicy(seed=seed, fov_radius=fov_radius,
                                   horizon=horizon, tie_eps=tie_eps,
                                   lam=CENTRAL_ORACLE_LAMBDA, mode="crlb")
    if method_name == METHOD_CENTRAL_CONFIG:
        return CentralOraclePolicy(seed=seed, fov_radius=fov_radius,
                                   horizon=horizon, tie_eps=tie_eps,
                                   lam=CENTRAL_ORACLE_LAMBDA, mode="config")
    if method_name == METHOD_CENTRAL_CRLB_COV:
        from config import CENTRAL_ORACLE_COV_GUARD_EPS
        return CentralOraclePolicy(seed=seed, fov_radius=fov_radius,
                                   horizon=horizon, tie_eps=tie_eps,
                                   lam=CENTRAL_ORACLE_LAMBDA, mode="crlb",
                                   coverage_guard_eps=CENTRAL_ORACLE_COV_GUARD_EPS)
    if method_name == METHOD_CENTRAL_CRLB_COV2:
        from config import CENTRAL_ORACLE_COV_GUARD_EPS2
        return CentralOraclePolicy(seed=seed, fov_radius=fov_radius,
                                   horizon=horizon, tie_eps=tie_eps,
                                   lam=CENTRAL_ORACLE_LAMBDA, mode="crlb",
                                   coverage_guard_eps=CENTRAL_ORACLE_COV_GUARD_EPS2)
    if method_name == METHOD_CENTRAL_CRLB_LOCAL:
        return CentralOraclePolicy(seed=seed, fov_radius=fov_radius,
                                   horizon=horizon, tie_eps=tie_eps,
                                   lam=CENTRAL_ORACLE_LAMBDA, mode="crlb",
                                   frame="local")
    if method_name == METHOD_CENTRAL_CONFIG_LOCAL:
        return CentralOraclePolicy(seed=seed, fov_radius=fov_radius,
                                   horizon=horizon, tie_eps=tie_eps,
                                   lam=CENTRAL_ORACLE_LAMBDA, mode="config",
                                   frame="local")

    raise ValueError(f"Unknown method: {method_name}")
