"""
tests/test_bfs.py — Validate bounded_bfs (V4 movement frame):
  - BFS layers == graph distance on an open grid (within max_depth),
  - first-step direction correctness,
  - horizon cap: cells beyond max_depth -> -1,
  - no path through already-visited cells: a visited ring fully encloses
    the agent, everything beyond it is -1.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env import GridEnv
from policies._common import bounded_bfs


def _env_open(n=15):
    env = GridEnv(grid_size=n, num_agents=1, obstacle_ratio=0.0, seed=0,
                  info_model="comm_limited")
    return env


def test_open_grid_distances():
    env = _env_open()
    env.agent_positions[0] = (5, 5)
    D, curdir = bounded_bfs(env, 0, max_depth=10)
    assert D[5, 5] == 0
    assert curdir[5, 5] == 4
    assert D[7, 5] == 2          # down x2
    assert D[5, 8] == 3          # right x3
    assert D[2, 5] == 3          # up x3
    assert D[3, 7] == 4          # manhattan 4


def test_open_grid_curdir():
    env = _env_open()
    env.agent_positions[0] = (5, 5)
    D, curdir = bounded_bfs(env, 0, max_depth=10)
    assert int(curdir[7, 5]) == 1      # down
    assert int(curdir[5, 7]) == 3      # right
    assert int(curdir[3, 5]) == 0      # up
    assert int(curdir[5, 3]) == 2      # left
    assert int(curdir[6, 6]) == 1      # first step to a diagonal is down/right


def test_horizon_cap():
    env = _env_open()
    env.agent_positions[0] = (5, 5)
    D, _ = bounded_bfs(env, 0, max_depth=3)
    assert D[5, 8] == 3          # exactly at horizon
    assert D[5, 9] == -1         # beyond horizon
    assert D[9, 9] == -1


def test_no_path_through_visited():
    env = _env_open()
    env.agent_positions[0] = (5, 5)
    ai = 0
    known = env.local_seen_mask[ai]
    visit = env.local_visit_count[ai]
    # visited ring at manhattan distance exactly 2, fully enclosing the agent
    ring = [
        (3, 5), (4, 4), (5, 3), (6, 4),
        (7, 5), (6, 6), (5, 7), (4, 6),
    ]
    for r, c in ring:
        known[r, c] = True
        visit[r, c] = 1

    D, _ = bounded_bfs(env, 0, max_depth=10)
    assert D[5, 5] == 0
    # layer-1 neighbors (unknown) still reachable
    assert D[4, 5] == 1
    assert D[5, 4] == 1
    # the visited ring itself is not traversable -> -1
    assert D[5, 7] == -1
    assert D[7, 5] == -1
    # and it blocks everything beyond it
    assert D[5, 8] == -1
    assert D[8, 5] == -1


def test_obstacle_block():
    env = _env_open(n=11)
    env.agent_positions[0] = (5, 5)
    ai = 0
    # full-height wall at col 3: left region is completely separated
    for r in range(env.grid_size):
        env.local_obstacle_map[ai][r, 3] = True
        env.local_seen_mask[ai][r, 3] = True
    D, _ = bounded_bfs(env, 0, max_depth=10)
    assert D[5, 5] == 0
    assert D[5, 4] == 1          # right of wall reachable
    assert D[5, 2] == -1         # left of wall unreachable
    assert D[5, 0] == -1
    assert D[5, 7] == 2
