"""
estimators/angular.py — Independent angular configuration model (Project08).

A cell accumulates a bearing observation every time it lies inside an agent's
FOV window. Two observations are "independent configurations" when their
bearing directions are separated by more than ANG_TOL (default 15 deg),
measured circularly. Clustering is greedy:

  - PERCEPTION (arrival order, spec A5): for a new angle, compute the circular
    distance to every existing center; if the nearest center is within ANG_TOL
    -> assign to it (no new cluster), otherwise start a new cluster, bounded
    by CLUSTER_CAP per cell. This is order-dependent by design (clusters are
    built as measurements stream in).
  - FUSION (spec B9): the two agents' RAW angle lists are unioned and
    re-clustered from scratch in sorted order (deterministic, exact).

The decision signal transposes V4's Chao-U exactly (see estimators/richness):
  - "visit"       := per-cell configuration count,
  - F1            := known traversable cells with exactly 1 configuration,
  - F2            := known traversable cells with exactly 2 configurations,
  - cap           := total undetermined cells = known traversable cells with
                     <= 1 configuration (not yet localizable; a single bearing
                     is rank-deficient and cannot localize a 2-D position).
"""

import numpy as np

from config import ANG_TOL_DEG, CLUSTER_CAP

ANG_TOL = np.deg2rad(ANG_TOL_DEG)
TWO_PI = 2.0 * np.pi


def angular_distance(a, b):
    """Circular distance between two angles, in radians, in [0, pi]."""
    d = (a - b) % TWO_PI
    return d if d <= np.pi else TWO_PI - d


def add_angle_to_clusters(centers, angle, tol=ANG_TOL, cap=CLUSTER_CAP):
    """Add one angle to an existing cluster-center list (immutable).

    Returns (new_centers, cap_hit):
      new_centers : a NEW list (the input is never mutated);
      cap_hit     : True if the angle would have started a new cluster but the
                    cap is already full (the observation is then discarded
                    from the count — mirrored by the cluster_cap_hit_frac).
    """
    new_centers = list(centers)
    for c in new_centers:
        d = (angle - c) % TWO_PI
        if d > np.pi:
            d = TWO_PI - d
        if d <= tol:
            return new_centers, False
    if len(new_centers) >= cap:
        return new_centers, True
    new_centers.append(angle)
    return new_centers, False


def greedy_cluster_centers(angles, tol=ANG_TOL, cap=CLUSTER_CAP):
    """Cluster a sequence of angles from scratch (deterministic).

    Sort the angles, then apply the greedy nearest-center rule in sorted order.
    Returns the list of cluster centers (length <= cap).
    """
    centers = []
    for a in sorted(angles):
        matched = False
        for c in centers:
            d = (a - c) % TWO_PI
            if d > np.pi:
                d = TWO_PI - d
            if d <= tol:
                matched = True
                break
        if not matched and len(centers) < cap:
            centers.append(a)
    return centers


def merge_raw_angle_lists(a, b, tol=ANG_TOL, cap=CLUSTER_CAP):
    """Fusion of two RAW angle lists: union + re-cluster from scratch.

    Spec B9 ("union des listes d'angles par cellule + re-clustering"):
    the union of the two agents' raw bearings is re-clustered in sorted
    order. This is exact by construction and independent of arrival order.
    """
    a = list(a)
    b = list(b)
    if not b:
        return a
    if not a:
        return b
    return greedy_cluster_centers(a + b, tol=tol, cap=cap)
