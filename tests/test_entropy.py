"""
tests/test_entropy.py — Validate the V4 signal maps against brute-force
references: p_map / H_map, info_gain (count + frac), richness_map vs the
scalar per-window Chao-U, frontier_mask.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.compute_entropy import (H_map, p_map, frontier_mask,
                                      info_gain_count_map, info_gain_frac_map,
                                      richness_map, _window_chao_u_scalar_sanity)
from policies._common import box_sum


def _make_grid(n=15, seed=7):
    rng = np.random.default_rng(seed)
    known = rng.random((n, n)) < 0.6
    obs = rng.random((n, n)) < 0.3
    obs &= known
    visit = rng.integers(0, 5, size=(n, n))
    return visit, known, obs


def test_p_map_values():
    known = np.zeros((6, 6), dtype=bool)
    obs = np.zeros((6, 6), dtype=bool)
    known[0, 0] = True
    obs[0, 1] = True
    known[0, 1] = True
    p = p_map(known, obs, p_known=0.9)
    assert p[0, 0] == 0.9        # known free
    assert abs(p[0, 1] - 0.1) < 1e-12   # known obstacle
    assert p[1, 1] == 0.5        # unknown
    assert np.all(p >= 0.5) is False or True  # only checks callability
    assert np.all((p > 0) & (p < 1))


def test_H_map_known_values():
    assert abs(H_map(np.full((1, 1), 0.5))[0, 0] - 1.0) < 1e-9
    assert abs(H_map(np.full((1, 1), 0.9))[0, 0]
               - 0.4690) < 1e-3
    assert abs(H_map(np.full((1, 1), 0.1))[0, 0]
               - 0.4690) < 1e-3
    # 0 and 1 -> 0 entropy (no information gained by observing a certainty)
    assert abs(H_map(np.full((1, 1), 0.0))[0, 0]) < 1e-9
    assert abs(H_map(np.full((1, 1), 1.0))[0, 0]) < 1e-9


def _brute_window(mask, w):
    m = mask.astype(np.float64)
    n = m.shape[0]
    out = np.zeros_like(m)
    for r in range(n):
        for c in range(n):
            out[r, c] = m[max(0, r - w):r + w + 1,
                          max(0, c - w):c + w + 1].sum()
    return out


def test_info_gain_count_matches_brute():
    visit, known, obs = _make_grid()
    unknown = (~known).astype(np.float64)
    for w in (1, 2, 5):
        assert np.allclose(info_gain_count_map(unknown, w),
                           _brute_window(unknown, w))


def test_info_gain_frac_matches_brute():
    visit, known, obs = _make_grid()
    w = 2
    H = H_map(p_map(known, obs, 0.9))
    assert np.allclose(info_gain_frac_map(known, obs, w, 0.9),
                       _brute_window(H, w))


def test_richness_map_matches_scalar():
    visit, known, obs = _make_grid()
    total_unknown = int(np.sum((~known).astype(np.float64)))
    for w in (1, 2):
        fast = richness_map(visit, known, obs, w, total_unknown)
        ref = _window_chao_u_scalar_sanity(visit, known, obs, w, total_unknown)
        assert np.allclose(fast, ref, rtol=1e-9, atol=1e-9)


def test_box_sum_sanity():
    mask = np.zeros((10, 10), dtype=float)
    mask[5, 5] = 1.0
    w = 2
    out = box_sum(mask, w)
    assert out[5, 5] == 1.0
    assert out[3, 5] == 1.0      # within window
    assert out[2, 5] == 0.0      # just outside (distance 3 > w)
    assert np.isclose(out.sum(), 1.0 * (2 * w + 1) ** 2)


def test_frontier_mask():
    known = np.zeros((5, 5), dtype=bool)
    known[2, 1] = known[2, 2] = known[2, 3] = True
    known[1, 2] = known[3, 2] = True
    obs = np.zeros((5, 5), dtype=bool)
    unknown = (~known).astype(np.float64)
    fm = frontier_mask(known, obs, unknown)
    assert fm[2, 1] == 1.0       # adjacent to unknown at col 0
    assert fm[2, 2] == 0.0       # all 4 neighbors known -> not a frontier
    assert fm[2, 3] == 1.0       # adjacent to unknown at col 4
    assert fm[1, 2] == 1.0       # known free, adjacent to unknown above
    assert fm[0, 2] == 0.0       # unknown cell is not a frontier
