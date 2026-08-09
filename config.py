"""
config.py — Single Source of Truth for Project08 (localization-driven
exploration).

Domain: multi-agent bearing-only localization. Each cell accumulates
"independent angular configurations" (bearings > ANG_TOL apart, greedy
clustering). The Chao-U richness signal is transposed from visit counts
to configuration counts (F1/F2 = cells with exactly 1/2 configurations).
Evaluation is a GLOBAL oracle CRLB bound per cell (decoupled from the
local decision signal).

All run_*.py, analysis and tests must import constants from here.
"""

import numpy as np

# ==================== ENVIRONMENT DEFAULTS ====================
GRID_SIZE = 100
DEFAULT_FOV_RADIUS = 5
DEFAULT_OBSTACLE_RATIO = 0.05
NUM_AGENTS = 6

# ==================== EXPERIMENT RIGOR ====================
MAX_STEPS = 10000
NUM_RUNS = 30
BASE_SEED = 0
# Seed stride: run r uses env_seed = BASE_SEED + r * SEED_STRIDE.
# All methods share the SAME env seed per run index (paired at map level).
SEED_STRIDE = 1000

# ==================== INFO MODEL ====================
# comm_limited: limited-range communication, rendezvous (proximity) fusion.
#   Agents share their maps ONLY when within COMM_RANGE. Only production model.
# pure_local / fov_perfect: kept ONLY as optional ablations (never defaults).
INFO_MODEL = "comm_limited"
INFO_MODEL_PURE_LOCAL = "pure_local"
INFO_MODEL_ABLATION = "fov_perfect"

# ==================== COMMUNICATION (comm_limited) ====================
# Proximity-triggered fusion: two agents within COMM_RANGE (Euclidean, in
# grid cells) exchange maps each step. R = FOV by default.
COMM_RANGE = float(DEFAULT_FOV_RADIUS)

# ==================== NOISE DEFAULTS (ablation only) ====================
DEFAULT_P_MISS = 0.0
DEFAULT_SIGMA_LOC = 0.0

# ==================== TARGET-SELECTION POLICIES ====================
# Bounded-BFS horizon R: candidates (unknown / frontier) and movement are
# restricted to cells reachable in <= HORIZON BFS layers through non-visited
# cells (unknown + known-unvisited free; already-visited cells excluded).
ENTROPY_HORIZON = 8
# Square window radius for info_gain / richness (defaults to FOV radius).
ENTROPY_WINDOW = None  # None -> DEFAULT_FOV_RADIUS
# Fractional-entropy sensor reliability: a seen cell gets p=0.9 (free) /
# p=0.1 (obstacle); unknown cells keep p=0.5. H(p) is then fractional.
SENSOR_RELIABILITY = 0.9
# Denominator floor in utility = gain / (distance + EPSILON).
UTILITY_EPSILON = 1.0
# Tie-break noise added to utility (per-agent scatter; v3 lesson: shared
# argmax makes all agents converge on the same target).
TIE_BREAK_EPS = 1e-3

# ==================== POLICY HYPERPARAMETERS ====================
ADAPTIVE_K = 0.5

# ==================== PARALLELIZATION ====================
NUM_WORKERS = 4

# ==================== CANONICAL METHOD NAMES ====================
METHOD_FRONTIER = "Frontier"
METHOD_GREEDY = "Least-Visited Greedy"
METHOD_CHAO_U = "Chao-U"
METHOD_ENTROPY = "Entropy"
METHOD_ENTROPY_FRAC = "Entropy-Frac"
METHOD_FRONTIER_ENTROPY = "Frontier+Entropy"
METHOD_FRONTIER_RICHNESS = "Frontier+Richness"

# TEMPORARY movement-control (NOT in the paper): Frontier selection rule but
# the entropy family's bounded-BFS movement frame (horizon R, fallback).
METHOD_FRONTIER_BOUNDED = "Frontier-Bounded"

# Order used in paper tables
METHOD_ORDER = [
    METHOD_FRONTIER,
    METHOD_GREEDY,
    METHOD_CHAO_U,
    METHOD_ENTROPY,
    METHOD_ENTROPY_FRAC,
    METHOD_FRONTIER_ENTROPY,
    METHOD_FRONTIER_RICHNESS,
]

# ==================== CHAO-U LOCKED CONFIG ====================
# V2/V3 tuning (locked): Chao-U corrected variant (saturation fix).
CHAO_DEFAULT_VARIANT = "bias_cap"

# ==================== PROJECT08: ANGULAR LOCALIZATION MODEL =================
# Two observations of a cell are "independent configurations" when their
# bearing directions are separated by more than ANG_TOL_DEG (circular).
# Greedy clustering: assign to the nearest existing center, otherwise start a
# new cluster, bounded by CLUSTER_CAP per cell. Cluster-count saturation is
# monitored via cluster_cap_hit_frac (decision signal, mirrors alpha_sat_frac).
ANG_TOL_DEG = 15.0
CLUSTER_CAP = 8

# ==================== PROJECT08: ORACLE CRLB EVALUATION =====================
# Bearing-only Cramer-Rao bound per cell: J = sum_k u_k u_k^T / (sigma^2 d_k^2),
# bound = sqrt(trace(J^-1)) in grid-cell units. sigma_ref is the NOMINAL
# bearing precision used to scale the bound (fixed reference, documented).
# A cell is "well-localized" when its bound <= QUALITY_THRESHOLD cells.
# quality(t) = fraction of traversable cells well-localized, sampled every
# QUALITY_SAMPLE_K steps; AUC by trapezoids + time-to-threshold (mirror of
# steps_90) with target QUALITY_TARGET.
QUALITY_SIGMA_BEARING_DEG = 1.0
QUALITY_THRESHOLD = 1.5
QUALITY_SAMPLE_K = 25
QUALITY_TARGET = 0.9

# ==================== PROJECT08: POLICIES ====================
METHOD_RANDOM = "Random"
METHOD_FRONTIER_RICHNESS_ANGULAR = "Richness-Angular"
# Random-policy repulsion radius (grid cells): a random agent that finds
# another agent closer than this distance moves away instead of blindly.
RANDOM_REPULSION_DIST = 2.0

# ==================== PROJECT08: DEPLOY-U (localization-aware deployment) ====
# Research idea 1+3: Frontier-Bounded movement, then DEPLOY mode that orbits
# the worst known under-localized cells (<= 1 angular configuration) to force
# angular diversity. Self-gated on the local signal:
#   under_frac_max     : deploy only once most KNOWN cells are well-localized
#                        (fraction of under-localized known cells <= threshold).
#   min_under_cells    : ignore noise-level residual sets below this size.
#   orbit_radius       : Chebyshev distance to the target that triggers orbit.
#   station_steps      : max steps per orbit bout.
#   cooldown / approach_depth : re-check cadence / bounded BFS reach.
METHOD_DEPLOY = "Deploy-U"
DEPLOY_COOLDOWN = 12
DEPLOY_UNDER_FRAC_MAX = 0.30
DEPLOY_MIN_UNDER_CELLS = 3
DEPLOY_ORBIT_RADIUS = 2
DEPLOY_STATION_STEPS = 6
DEPLOY_APPROACH_DEPTH = 12

# Phase-1 method order (Project08).
PHASE1_METHODS = [
    METHOD_RANDOM,
    METHOD_FRONTIER_BOUNDED,
    METHOD_FRONTIER_RICHNESS_ANGULAR,
]

# E4: U-prioritized coverage under a finite budget ("Coverage-U").
# Continuous target scoring inside the Frontier-Bounded frame:
#   score(target) = D/horizon - lam * (under_count_FOV(target) / FOV_area),
# where under_count_FOV counts known-free cells with <= 1 angular
# configuration inside the target's FOV square. lam = 0 == Frontier-Bounded.
METHOD_COVERAGE_U = "Coverage-U"
COVERAGE_U_LAMBDA = 0.5

# MH #1: dynamic-normalization variant of Coverage-U (same signal, free-count
# denominator instead of the constant FOV_area). Method name carries the
# variant so the factory selects the normalize flag with zero runner changes.
# score(t) = D/horizon - lam * under_count_FOV(t) / free_count_FOV(t),
# free_count_FOV = number of traversable (by local belief) cells in the FOV
# window. Restores the under-set signal strength in obstacle-dominated windows
# (A6_obs020 dilution failure).
METHOD_COVERAGE_U_NORM = "Coverage-U-norm"

# MH #2: GDOP/FIM baseline policy — target selection by simulated LOCAL
# bound/FIM improvement (unit-weight direction-only GDOP from the locally-known
# independent configuration centers), in the SAME Frontier-Bounded frame as
# Coverage-U. Classical-literature baseline; never a proposed method.
METHOD_GDOP = "GDOP"
GDOP_LAMBDA = 0.5
# Bound cap (grid cells) for rank-deficient cells (< 2 independent directions):
# bounds above this are collapsed to the cap for the simulated-gain computation.
# 20 cells is far above any achievable finite bearing-only bound here
# (near-collinear two-direction cells ~10.8) so inf->finite localization is the
# dominant gain, as it should be.
GDOP_BOUND_CAP = 20.0

# MH #3: HYBRID policy — single policy routing between the two validated
# local scorings by the agent's CURRENT FOV-window free fraction, inside the
# SAME Frontier-Bounded frame (same bounded_bfs, same H=8, same candidate set
# = reachable-unknown cells; only the score changes):
#   free_frac = free_count_FOV(agent pos) / FOV_area  >= THETA  -> CU-norm score
#   free_frac = free_count_FOV(agent pos) / FOV_area  <  THETA  -> RA utility
# free_count_FOV = traversable (free + unknown, i.e. ~obs_local) cells in the
# window (O(1) via the MH#1 integral image), over the ACTUAL clamped window
# area (env FOV footprint shrinks at the grid border). THETA is FROZEN pre-run
# from the probe (probe_hybrid_theta.py): theta = 0.8 = 1 - obs_ratio of the
# dense regime = "flip to RA when the window is more obstacle-dominated than
# the regime's nominal density". Probe (4 ep/regime, N=38424 windows): theta=0.5
# is FULLY inert (P(free_frac<0.5)=0 in both regimes); theta=0.8 gives 0% RA in
# A6_obs005 (sparse control stays pure CU-norm) vs ~47% RA in A6_obs020 (dense
# test actually exercises the RA branch).
METHOD_HYBRID = "Hybrid"
HYBRID_THETA = 0.8

# E5: CENTRALIZED ORACLE upper bound (control row, EVALUATION ONLY — never a
# proposed method, infeasible in the deployed system). Same scoring form as
# Coverage-U (score = D/horizon - lam * under_count_FOV/FOV_area) but with
# GLOBAL perfect knowledge. Two signal modes for the under-set:
#   "config": globally-observed-free cells with global config count <= 1
#             (perfect-fusion version of the Coverage-U signal);
#   "crlb"  : globally-observed-free cells whose oracle CRLB bound
#             sqrt(trace(J^-1)) > QUALITY_THRESHOLD (the true accuracy
#             bottleneck). Anchors the transposition ratio
#             reduction_CoverageU / reduction_CentralCRLB.
METHOD_CENTRAL_CRLB = "CentralOracle-CRLB"
METHOD_CENTRAL_CONFIG = "CentralOracle-Config"
METHOD_CENTRAL_CRLB_COV = "CentralOracle-CRLB-cov"
METHOD_CENTRAL_CRLB_COV2 = "CentralOracle-CRLB-cov2"
# E5-CORRECTED: same oracle signals but in the LOCAL bounded_bfs movement
# frame (byte-identical to Coverage-U/FB). Isolates the global-vs-local
# SIGNAL effect from the global-frame artifact that confounded E5.
METHOD_CENTRAL_CRLB_LOCAL = "CentralOracle-CRLB-local"
METHOD_CENTRAL_CONFIG_LOCAL = "CentralOracle-Config-local"
CENTRAL_ORACLE_LAMBDA = 0.5
CENTRAL_ORACLE_COV_GUARD_EPS = 0.05
CENTRAL_ORACLE_COV_GUARD_EPS2 = 0.3
