"""
tests/test_maze.py — Maze topology: generator correctness + LOS occlusion.

Covers the preregistered maze caveats (FUTURE_WORK_NOTES #1):
  - Kruskal maze free space is ONE connected component.
  - The maze is a tree: no closed loops (E = V - 1).
  - Spawn is always on a free cell.
  - A maze episode is playable by the bounded-BFS policies.
  - Line-of-sight occlusion blocks perception through walls (only maze).
  - topology="random" keeps the historical square-FOV behaviour (non-regression).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import deque

from env import GridEnv
from experiments._runner import run_episode


def _free_connectivity_edges(env):
    """Returns (n_free, n_edges) of the free-cell 4-neighbour graph."""
    om = env.obstacle_map
    free = ~om
    n_free = int(free.sum())
    n_edges = 0
    r, c = np.nonzero(free)
    for rr, cc in zip(r, c):
        if rr + 1 < env.grid_size and free[rr + 1, cc]:
            n_edges += 1
        if cc + 1 < env.grid_size and free[rr, cc + 1]:
            n_edges += 1
    return n_free, n_edges


def _bfs_component(env):
    """Size of the free-space component reachable from the first free cell."""
    om = env.obstacle_map
    gs = env.grid_size
    free = ~om
    start = tuple(np.argwhere(free)[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < gs and 0 <= nc < gs and free[nr, nc] \
                    and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen)


def test_maze_connected_single_component():
    for seed in [0, 1, 42, 1000]:
        env = GridEnv(grid_size=100, seed=seed, topology="maze")
        n_free, _ = _free_connectivity_edges(env)
        assert _bfs_component(env) == n_free, f"seed {seed}: disconnected maze"


def test_maze_tree_no_cycles():
    for seed in [0, 1, 42, 1000]:
        env = GridEnv(grid_size=100, seed=seed, topology="maze")
        n_free, n_edges = _free_connectivity_edges(env)
        assert n_edges == n_free - 1, \
            f"seed {seed}: {n_edges} edges for {n_free} free cells (loops?)"


def test_maze_wall_fraction_around_half():
    for seed in [0, 1, 42]:
        env = GridEnv(grid_size=100, seed=seed, topology="maze")
        frac = float(env.obstacle_map.mean())
        assert 0.45 <= frac <= 0.60, f"seed {seed}: wall fraction {frac:.3f}"


def test_maze_deterministic_per_seed():
    e1 = GridEnv(grid_size=100, seed=7, topology="maze")
    e2 = GridEnv(grid_size=100, seed=7, topology="maze")
    e3 = GridEnv(grid_size=100, seed=8, topology="maze")
    assert np.array_equal(e1.obstacle_map, e2.obstacle_map)
    assert not np.array_equal(e1.obstacle_map, e3.obstacle_map)


def test_maze_spawn_on_free():
    env = GridEnv(grid_size=100, num_agents=6, seed=0, topology="maze")
    for r, c in env.agent_positions:
        assert not env.obstacle_map[r, c]


def test_maze_episode_playable():
    r = run_episode("Frontier-Bounded", "comm_limited", 0, env_seed=0,
                    max_steps=300, grid_size=100, num_agents=6,
                    obstacle_ratio=0.5, fov_radius=5, horizon=8,
                    topology="maze")
    assert r["final_coverage"] > 0
    assert r["topology"] == "maze"


def test_maze_richness_angular_playable():
    r = run_episode("Richness-Angular", "comm_limited", 0, env_seed=0,
                    max_steps=300, grid_size=100, num_agents=6,
                    obstacle_ratio=0.5, fov_radius=5, horizon=8,
                    topology="maze")
    assert r["final_coverage"] > 0


def test_los_blocks_perception_through_walls():
    env = GridEnv(grid_size=100, seed=0, topology="maze")
    om = env.obstacle_map
    gs = 100
    free = np.argwhere(~om)
    # central free cell
    dist = (free[:, 0] - 50) ** 2 + (free[:, 1] - 50) ** 2
    r, c = free[dist.argmin()]
    square = np.zeros((gs, gs), dtype=bool)
    square[max(0, r - 5):min(gs, r + 6),
           max(0, c - 5):min(gs, c + 6)] = True
    occ = env._los_visible(r, c, 5, max(0, r - 5), min(gs, r + 6),
                           max(0, c - 5), min(gs, c + 6))
    assert int(occ.sum()) < int(square.sum()), \
        "LOS should hide cells behind walls"
    # every visible cell has NO wall strictly between source and target
    for tr, tc in np.argwhere(occ):
        from env import _supercover_rel
        line = _supercover_rel(tr - r, tc - c, 5)
        assert not any(om[r + sr, c + sc] for sr, sc in line)


def test_los_full_visibility_on_open_grid():
    env = GridEnv(grid_size=100, seed=0, topology="random",
                  obstacle_ratio=0.0)
    r, c = 50, 50
    square = np.zeros((100, 100), dtype=bool)
    square[45:56, 45:56] = True
    occ = env._los_visible(r, c, 5, 45, 56, 45, 56)
    assert np.array_equal(occ, square), \
        "no walls -> full FOV must be visible"


def test_random_topology_occlusion_off():
    env = GridEnv(grid_size=100, seed=0, topology="random",
                  obstacle_ratio=0.05)
    assert env.occlude is False
    fov = env.get_fov_mask(50, 50, 5)
    assert int(fov.sum()) == 121, "random topology keeps square FOV"


def test_maze_occlusion_on():
    env = GridEnv(grid_size=100, seed=0, topology="maze")
    assert env.occlude is True


def test_maze_loops_restore_cycles_keep_connectivity():
    """maze_loop_density>0 reopens non-tree walls: cycles appear (loops>0)
    but free space stays ONE connected component."""
    for seed in [0, 1, 42]:
        env = GridEnv(grid_size=100, seed=seed, topology="maze",
                      maze_loop_density=0.10)
        n_free, n_edges = _free_connectivity_edges(env)
        assert _bfs_component(env) == n_free, f"seed {seed}: disconnected"
        assert n_edges >= n_free, \
            f"seed {seed}: loops expected (edges {n_edges} >= free {n_free})"


def test_maze_loops_zero_density_identical_perfect():
    """Default (maze_loop_density=0.0) is bit-identical to the perfect maze."""
    a = GridEnv(grid_size=100, seed=7, topology="maze",
                maze_loop_density=0.0)
    b = GridEnv(grid_size=100, seed=7, topology="maze")
    assert np.array_equal(a.obstacle_map, b.obstacle_map)
    n_free, n_edges = _free_connectivity_edges(b)
    assert n_edges == n_free - 1, "perfect maze must stay a tree"


def test_cluster_density_target():
    """Cluster topology reaches the requested obstacle density."""
    for seed in [0, 1, 42, 1000]:
        env = GridEnv(grid_size=100, seed=seed, topology="cluster",
                      obstacle_ratio=0.20)
        frac = float(env.obstacle_map.mean())
        assert 0.19 <= frac <= 0.21, \
            f"seed {seed}: density {frac:.3f} != 0.20"


def test_cluster_deterministic_per_seed():
    e1 = GridEnv(grid_size=100, seed=7, topology="cluster",
                 obstacle_ratio=0.20)
    e2 = GridEnv(grid_size=100, seed=7, topology="cluster",
                 obstacle_ratio=0.20)
    e3 = GridEnv(grid_size=100, seed=8, topology="cluster",
                 obstacle_ratio=0.20)
    assert np.array_equal(e1.obstacle_map, e2.obstacle_map)
    assert not np.array_equal(e1.obstacle_map, e3.obstacle_map)


def test_cluster_free_space_large_component():
    """Free space stays essentially connected (routing preserved)."""
    for seed in [0, 1, 42, 1000]:
        env = GridEnv(grid_size=100, seed=seed, topology="cluster",
                      obstacle_ratio=0.20)
        n_free = int((~env.obstacle_map).sum())
        assert _bfs_component(env) >= 0.95 * n_free, \
            f"seed {seed}: free space fragmented"


def test_cluster_occlusion_on_and_blocks_sight():
    """Cluster topology: LOS occlusion ON; blocks create real occlusion."""
    env = GridEnv(grid_size=100, seed=0, topology="cluster",
                  obstacle_ratio=0.20)
    assert env.occlude is True
    om = env.obstacle_map
    gs = 100
    free = np.argwhere(~om)
    dist = (free[:, 0] - 50) ** 2 + (free[:, 1] - 50) ** 2
    r, c = free[dist.argmin()]
    square = np.zeros((gs, gs), dtype=bool)
    square[max(0, r - 5):min(gs, r + 6),
           max(0, c - 5):min(gs, c + 6)] = True
    occ = env._los_visible(r, c, 5, max(0, r - 5), min(gs, r + 6),
                           max(0, c - 5), min(gs, c + 6))
    assert int(occ.sum()) < int(square.sum()), \
        "cluster blocks should hide some cells behind them"
    for tr, tc in np.argwhere(occ):
        from env import _supercover_rel
        line = _supercover_rel(tr - r, tc - c, 5)
        assert not any(om[r + sr, c + sc] for sr, sc in line)


def test_cluster_spawn_on_free_and_playable():
    env = GridEnv(grid_size=100, num_agents=6, seed=0, topology="cluster",
                  obstacle_ratio=0.20)
    for r, c in env.agent_positions:
        assert not env.obstacle_map[r, c]
    res = run_episode("Frontier-Bounded", "comm_limited", 0, env_seed=0,
                      max_steps=300, grid_size=100, num_agents=6,
                      obstacle_ratio=0.20, fov_radius=5, horizon=8,
                      topology="cluster")
    assert res["final_coverage"] > 0
    assert res["topology"] == "cluster"
