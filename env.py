"""
env.py — Grid environment for UAV swarm coverage. V3 (comm_limited info model).

INFO MODEL
----------
comm_limited  (V3 default): limited-range communication with proximity-triggered
  (rendezvous) fusion. Each agent keeps a private memory:
  - agent_own_visit_count[agent_id]: counts only cells THIS agent visited.
  - local_visit_count[agent_id]    : what the agent "knows" — initialized to
    its own counts, updated by FUSION when agents meet within COMM_RANGE.
  - local_seen_mask[agent_id]      : cells ever seen inside its FOV (or a
    neighbor's FOV via fusion) — occupancy mask.
  - local_obstacle_map[agent_id]   : obstacles ever seen (FOV + fusion).
  Fusion is symmetric: both agents keep the combined knowledge (visit = MAX,
  seen/obstacle = union). No global visit_count / obstacle_map is ever fed to
  a policy. Collision resolution is handled by the environment from REAL agent
  positions (position-only channel), fully independent of COMM_RANGE/fusion.

pure_local  (V2 reference): zero communication — each agent perceives ONLY its
  own history. Same structure, no fusion ever fires.

fov_perfect (ablation): perfect knowledge of the whole grid (global obstacle
  map + global visit counts). Used as the upper bound of the spectrum.

The canonical spectrum: pure_local <= comm_limited <= fov_perfect.
"""

import math

import numpy as np

from estimators.angular import (ANG_TOL, CLUSTER_CAP,
                                add_angle_to_clusters, greedy_cluster_centers)


# ---------------------------------------------------------------------------
# Supercover line cache (module scope, immutable): for every (dr, dc) offset
# inside the FOV square, the relative grid cells STRICTLY BETWEEN (0,0) and
# (dr, dc) that the segment crosses. Used by the maze line-of-sight test.
# ---------------------------------------------------------------------------
_SUPERCOVER_CACHE = {}


def _supercover_rel(dr, dc, max_radius):
    """Relative supercover cells strictly between (0,0) and (dr, dc).

    Returns a tuple of (r, c) offsets. A cell (i, j) lies on the supercover
    line if the open segment from the source to the target intersects the
    cell's CLOSED square [i-0.5, i+0.5] x [j-0.5, j+0.5]. Endpoints (source
    and target cells) are excluded so the target is checked separately by the
    caller. Exact geometric test (no DDA corner ambiguity), computed once and
    cached per (dr, dc) for the whole FOV square.
    """
    key = (dr, dc)
    cached = _SUPERCOVER_CACHE.get(key)
    if cached is not None:
        return cached

    R = max_radius
    line = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            if i == 0 and j == 0:
                continue
            if i == dr and j == dc:
                # Target cell: checked by the caller, never a blocker.
                continue
            # P(t) = (t*dr, t*dc), t in [0, 1]. Need t in [0,1] with
            # |t*dr - i| <= 0.5 and |t*dc - j| <= 0.5.
            lo, hi = 0.0, 1.0
            if dr == 0:
                if i != 0:
                    continue
            else:
                t0 = (i - 0.5) / dr
                t1 = (i + 0.5) / dr
                lo = max(lo, min(t0, t1))
                hi = min(hi, max(t0, t1))
            if dc == 0:
                if j != 0:
                    continue
            else:
                t0 = (j - 0.5) / dc
                t1 = (j + 0.5) / dc
                lo = max(lo, min(t0, t1))
                hi = min(hi, max(t0, t1))
            if lo <= hi and 0.0 < lo <= 1.0:
                # Cell strictly between endpoints (target t=1 excluded).
                line.append((i, j))
    _SUPERCOVER_CACHE[key] = tuple(line)
    return _SUPERCOVER_CACHE[key]


class GridEnv:
    """
    2-D grid world shared by all UAV agents.

    Action encoding: 0=up, 1=down, 2=left, 3=right, 4=stay

    ANGULAR OBSERVATION MODEL (Project08): every time a traversable cell lies
    inside an agent's FOV, the agent records the bearing (atan2) to that cell.
    Per (agent, cell) the bearings are greedily clustered into independent
    angular configurations (> ANG_TOL apart), capped at CLUSTER_CAP. The local
    configuration-count grid mirrors local_visit_count (decision signal);
    a GLOBAL oracle accumulator (metrics only, never fed to policies) records
    the bearing-only Fisher information per cell for the CRLB evaluation.
    """

    def __init__(self, grid_size=100, num_agents=6, obstacle_ratio=0.05,
                 seed=0, info_model="pure_local", p_miss=0.0, sigma_loc=0.0,
                 sigma_bearing=0.0, comm_range=None, topology="random",
                 maze_loop_density=0.0):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obstacle_ratio = obstacle_ratio
        self.seed = seed
        self.info_model = info_model
        self.p_miss = p_miss
        self.sigma_loc = sigma_loc
        self.sigma_bearing = sigma_bearing
        self.topology = topology
        self.maze_loop_density = maze_loop_density
        # Occlusion (line-of-sight) applies to the maze AND cluster topologies:
        # walls/blocks block perception exactly like the MAexp radar filter.
        # The random topology keeps the historical square-FOV-through-walls
        # model. Cluster topology = contiguous obstacle blocks (real LOS
        # occlusion) while the free space keeps multiple routing paths.
        self.occlude = topology in ("maze", "cluster")
        if info_model == "comm_limited" and comm_range is None:
            from config import COMM_RANGE
            comm_range = COMM_RANGE
        self.comm_range = comm_range

        self.rng = np.random.default_rng(seed)
        self._build_obstacle_map()

        # --- GLOBAL ground truth (metrics ONLY, never fed to policies) ---
        self.visit_count = np.zeros((grid_size, grid_size), dtype=np.int32)

        # --- PER-AGENT OWN MEMORY ---
        # Only incremented when the agent itself steps onto the cell.
        self.agent_own_visit_count = [
            np.zeros((grid_size, grid_size), dtype=np.int32)
            for _ in range(num_agents)
        ]
        # What the agent "knows": own visits, augmented by rendezvous fusion
        # (comm_limited). For pure_local it stays == own counts.
        self.local_visit_count = [
            np.zeros((grid_size, grid_size), dtype=np.int32)
            for _ in range(num_agents)
        ]
        # Cells ever seen inside FOV (and, for comm_limited, via fusion).
        self.local_seen_mask = [
            np.zeros((grid_size, grid_size), dtype=bool)
            for _ in range(num_agents)
        ]
        # Obstacles ever seen inside FOV (unknown cells are treated as free).
        self.local_obstacle_map = [
            np.zeros((grid_size, grid_size), dtype=bool)
            for _ in range(num_agents)
        ]

        # Number of symmetric pair-merges performed (comm_limited only).
        self.fusion_events_count = 0
        self.last_rendezvous_pairs = 0

        # --- ANGULAR MEASUREMENT MODEL (Project08) ---
        # Per (agent, cell):
        #   local_angle_clusters : independent angular configuration CENTERS
        #       (greedy nearest-center, > ANG_TOL, capped at CLUSTER_CAP).
        #   local_raw_angles : deduplicated SET of raw bearings per cell, kept
        #       ONLY for exact re-clustering at fusion time (spec B9: union of
        #       raw angle lists -> sorted greedy re-cluster). Sets bound the
        #       merge cost: a cell's distinct bearings <= #distinct observer
        #       positions in FOV range (~121 for FOV=5), and set union never
        #       self-multiplies shared lists.
        #   local_*     : own + fused (mirrors local_visit_count).
        # Config-count grids (int8) are maintained incrementally so policies
        # read O(1) per cell. Shared local raw sets are copy-on-write: every
        # perception replaces the set with a new one, so a partner keeps its
        # own reference after fusion.
        self.local_angle_clusters = [{} for _ in range(num_agents)]
        self.local_raw_angles = [{} for _ in range(num_agents)]
        self.local_measurement_count = [
            np.zeros((grid_size, grid_size), dtype=np.int8)
            for _ in range(num_agents)
        ]
        # local_dirty[i][j]: set of cells whose data for agent i may differ from
        # agent j's since their last merge. Perception re-adds observed cells;
        # a merge (i,j) only visits dirty[i][j] U dirty[j][i] and then clears
        # both, propagating the merged cells to the other partners' dirty sets.
        # This keeps fusion cost proportional to CHANGE, not map size (the
        # whole-map iteration of V4-style merges explodes once agents travel
        # together: O(steps x map_size)).
        self.local_dirty = [
            [set() for _ in range(num_agents)] for _ in range(num_agents)
        ]

        # --- ANGULAR ORACLE (metrics ONLY, never fed to policies) ---
        # GLOBAL observation record: all agents' bearings, clustered globally.
        # global_info[cells, (0,1,2)] = accumulated bearing-only Fisher
        # information J00, J01, J11 per cell -> CRLB bound = sqrt(trace(J^-1)).
        # global_obs_count = per-cell count of observations (vectorized).
        self.global_angle_clusters = {}
        self.global_info = np.zeros((grid_size, grid_size, 3),
                                    dtype=np.float64)
        self.global_obs_count = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.global_observation_count = 0
        self.global_cap_hit_count = 0
        # Optional raw recording (Phase 1a validation only): when enabled, the
        # TRUE observer positions behind each global observation are stored per
        # cell so a Gauss-Newton estimator can be run against noisy bearings.
        self.record_global_raw = False
        self.global_raw_obs = {}

        # Frontier BFS exhaustion cache (per-agent). When a policy's BFS
        # exhausts (no locally-unvisited cell reachable), the reachable
        # component can only shrink while the unvisited/obstacle knowledge is
        # unchanged, so the "no reachable target" result stays valid. The
        # masks are snapshots used to invalidate the cache. Behavior-identical.
        self.explore_exhausted = [False] * num_agents
        self.explore_exhausted_unvisited = [None] * num_agents
        self.explore_exhausted_obs = [None] * num_agents

        # --- Noise bookkeeping ---
        self.noise_rngs = [
            np.random.default_rng(seed + 1000 + i) for i in range(num_agents)
        ]
        self.loc_error_sum = 0.0
        self.loc_error_count = 0

        self.traversable = int(np.sum(~self.obstacle_map))
        self._place_agents()
        self.last_collision_count = 0
        self.sigma_bearing_ref = None  # lazy-loaded from config
        self.quality_threshold = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_obstacle_map(self):
        """Random obstacles. Outer edge ring left free for spawn safety."""
        if self.topology == "maze":
            self._build_maze_map()
            return
        if self.topology == "cluster":
            self._build_cluster_map()
            return
        total = self.grid_size * self.grid_size
        inner = [(r, c) for r in range(1, self.grid_size - 1)
                 for c in range(1, self.grid_size - 1)]
        n_obs = int(total * self.obstacle_ratio)
        flat = np.zeros(total, dtype=bool)
        cand = self.rng.choice(len(inner), size=n_obs, replace=False)
        for idx in cand:
            r, c = inner[idx]
            flat[r * self.grid_size + c] = True
        self.obstacle_map = flat.reshape(self.grid_size, self.grid_size)

    def _build_maze_map(self):
        """Kruskal randomized maze (MAexp-style), optionally with loops.

        Construction on a C x C grid of cells, one cell per (odd, odd)
        grid position, walls on the even rows/cols between them. Kruskal's
        algorithm removes walls in random order whenever the two cells they
        separate belong to different components -> the removed walls form a
        spanning tree: the whole free space is ONE connected component and
        there is exactly one path between any two free cells (no closed
        loops). Guarantees the connectivity/no-loop caveats of the maze item.

        maze_loop_density > 0: after the spanning tree, additionally reopen
        that fraction of the NON-TREE candidate walls (uniformly at random).
        Each extra opening creates a second path between two free cells (a
        loop) -> restores routing choice while never disconnecting the free
        space (nothing is ever closed). maze_loop_density=0.0 (default) is
        the perfect maze, bit-identical to before.

        Free cells: C^2 cells + (C^2 - 1) spanning-tree edges. Wall fraction
        ~ 50% at C=49 on a 100 grid.
        """
        gs = self.grid_size
        # Number of maze cells per axis: cell i occupies grid row 2i+1.
        # C = (gs-1)//2 -> cells reach 2C-1 <= gs-2, keeping a free margin.
        C = (gs - 1) // 2
        n = C * C

        # Union-Find over cells (row-major index i*C+j).
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[ra] = rb
            return True

        # Walls: True = wall. Everything starts as wall except the cells.
        wall = np.ones((gs, gs), dtype=bool)
        for i in range(C):
            for j in range(C):
                wall[2 * i + 1, 2 * j + 1] = False

        # Candidate walls: horizontal + vertical walls between adjacent cells.
        # Shuffle once, then Kruskal.
        edges = []
        for i in range(C):
            for j in range(C):
                if i + 1 < C:
                    edges.append((i * C + j, (i + 1) * C + j,
                                  2 * i + 2, 2 * j + 1))
                if j + 1 < C:
                    edges.append((i * C + j, i * C + (j + 1),
                                  2 * i + 1, 2 * j + 2))
        perm = self.rng.permutation(len(edges))
        removed = 0
        for idx in perm:
            a, b, r, c = edges[idx]
            if union(a, b):
                wall[r, c] = False
                removed += 1
                if removed == n - 1:
                    break

        # Loops: reopen a fraction of the remaining (non-tree) candidate walls.
        loop_density = getattr(self, "maze_loop_density", 0.0) or 0.0
        if loop_density > 0.0:
            tree_edges = set(edges[i] for i in perm[:removed])
            non_tree = [e for e in edges if e not in tree_edges]
            n_open = int(round(len(non_tree) * loop_density))
            opened = self.rng.choice(len(non_tree), size=n_open, replace=False)
            for k in opened:
                _, _, r, c = non_tree[k]
                wall[r, c] = False

        self.obstacle_map = wall

    def _build_cluster_map(self):
        """Cluster/blob obstacles: contiguous blocks at the target density.

        Obstacles are placed as axis-aligned rectangles of random size
        (width/height drawn in 1..CLUSTER_MAX_EXTENT, centred anywhere in the
        inner area, outer edge ring left free for spawn safety). Contiguous
        blocks create REAL line-of-sight occlusion (unlike i.i.d. single
        cells, which barely block a supercover line) while the free space
        keeps multiple routing paths between any two cells (unlike the
        perfect maze's spanning tree). This is the occlusion-vs-routing
        decomposition test: same LOS filter as the maze, no routing collapse.

        Deterministic per env seed (draws come from self.rng in a fixed
        order). The loop stops when the placed obstacle count reaches the
        requested obstacle_ratio (a partially-filled final rectangle may stop
        short of the exact count; the density is a lower-bound target).
        """
        gs = self.grid_size
        target = int(gs * gs * self.obstacle_ratio)
        obs = np.zeros((gs, gs), dtype=bool)
        placed = 0
        guard = 0
        while placed < target and guard < 100000:
            guard += 1
            h = int(self.rng.integers(1, 6))
            w = int(self.rng.integers(1, 6))
            r = int(self.rng.integers(1, gs - h - 1))
            c = int(self.rng.integers(1, gs - w - 1))
            for rr in range(r, r + h):
                for cc in range(c, c + w):
                    if placed >= target:
                        break
                    if not obs[rr, cc]:
                        obs[rr, cc] = True
                        placed += 1

        self.obstacle_map = obs

    def _place_agents(self):
        free = list(zip(*np.where(~self.obstacle_map)))
        replace_flag = len(free) < self.num_agents
        chosen = self.rng.choice(len(free), size=self.num_agents,
                                 replace=replace_flag)
        self.agent_positions = [list(free[i]) for i in chosen]
        for aid, (r, c) in enumerate(self.agent_positions):
            self.visit_count[r, c] += 1
            self.agent_own_visit_count[aid][r, c] += 1
            self.local_visit_count[aid][r, c] += 1

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def step(self, actions):
        """Move each agent. No stacking: contested cells keep one random
        winner, losers stay in place."""
        intended = []
        for agent_id, action in enumerate(actions):
            dr, dc = self.DELTAS[action]
            r, c = self.agent_positions[agent_id]
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                    and not self.obstacle_map[nr, nc]):
                intended.append((nr, nc))
            else:
                intended.append((r, c))

        from collections import defaultdict
        claimants = defaultdict(list)
        for agent_id, cell in enumerate(intended):
            claimants[cell].append(agent_id)

        self.last_collision_count = 0
        for cell, agents in claimants.items():
            if len(agents) > 1:
                self.last_collision_count += len(agents) - 1
                winner = int(self.rng.choice(agents))
                for loser in agents:
                    if loser != winner:
                        intended[loser] = self.agent_positions[loser][:]

        for agent_id in range(self.num_agents):
            nr, nc = intended[agent_id]
            self.agent_positions[agent_id] = [nr, nc]
            self.visit_count[nr, nc] += 1
            self.agent_own_visit_count[agent_id][nr, nc] += 1
            self.local_visit_count[agent_id][nr, nc] += 1

        return [pos[:] for pos in self.agent_positions]

    # ------------------------------------------------------------------
    # Communication (comm_limited): proximity-triggered fusion
    # ------------------------------------------------------------------

    def distance(self, i, j):
        """Euclidean distance between agents i and j (units = grid cells)."""
        return float(np.linalg.norm(
            np.asarray(self.agent_positions[i], dtype=float)
            - np.asarray(self.agent_positions[j], dtype=float)))

    def _within_comm_range(self, i, j):
        if self.comm_range is None:
            return False
        return self.distance(i, j) <= float(self.comm_range)

    def check_and_merge(self, agent_id_list=None):
        """Check all pairs of agents within COMM_RANGE and fuse their maps.
        Called ONCE per decision step, BEFORE FOV perception (§5.3 order).
        No-op for info models other than comm_limited."""
        if self.info_model != "comm_limited":
            return 0
        import itertools
        if agent_id_list is None:
            agent_id_list = list(range(self.num_agents))
        n_fusions = 0
        for i, j in itertools.combinations(agent_id_list, 2):
            if self._within_comm_range(i, j):
                self.merge_maps(i, j)
                n_fusions += 1
        self.last_rendezvous_pairs = n_fusions
        return n_fusions

    def merge_maps(self, i, j):
        """
        Symmetric merge: both agents leave with the COMBINED knowledge.
        - Visits  : element-wise MAX (never a sum, to avoid inflating F1/F2).
        - Seen    : boolean union.
        - Obstacles: boolean union (occupancy is shared geometry).
        - Angles  : union of independent configuration clusters per cell
          (re-clustered greedily; exact because centers are fixed reps).
        The merge is idempotent per pair; the counter counts pair-merges with
        new information. If neither agent changed anything since their last
        merge (dirty sets empty), the whole merge is a no-op.
        """
        if self.info_model == "comm_limited":
            if not self.local_dirty[i][j] and not self.local_dirty[j][i]:
                return
        merged_visit = np.maximum(self.local_visit_count[i],
                                  self.local_visit_count[j])
        merged_seen = self.local_seen_mask[i] | self.local_seen_mask[j]
        merged_obs = self.local_obstacle_map[i] | self.local_obstacle_map[j]

        self.local_visit_count[i] = merged_visit.copy()
        self.local_visit_count[j] = merged_visit.copy()
        self.local_seen_mask[i] = merged_seen
        self.local_seen_mask[j] = merged_seen.copy()
        self.local_obstacle_map[i] = merged_obs
        self.local_obstacle_map[j] = merged_obs.copy()

        self._merge_angle_maps(i, j)

        self.fusion_events_count += 1

    def _merge_angle_maps(self, i, j):
        """Union of the two agents' local configuration clusters (comm_limited).

        ONLY the cells in dirty[i][j] U dirty[j][i] are visited (the cells
        that may differ since the pair's last merge) — spec B9 "union des
        listes d'angles par cellule + re-clustering" applied exactly on those.
        For each such cell, merge the DEDUPLICATED raw angle sets and
        re-cluster from scratch in sorted order; both agents receive the SAME
        new raw set and center list. The merged cells are propagated to the
        other partners' dirty sets so multi-hop fusion stays exact.
        """
        di = self.local_dirty[i]
        dj = self.local_dirty[j]
        to_merge = di[j] | dj[i]
        if not to_merge:
            return

        ci = self.local_angle_clusters[i]
        cj = self.local_angle_clusters[j]
        ri = self.local_raw_angles[i]
        rj = self.local_raw_angles[j]
        count_i = self.local_measurement_count[i]
        count_j = self.local_measurement_count[j]
        gs = self.grid_size
        for cell in to_merge:
            ri_c = ri.get(cell)
            rj_c = rj.get(cell)
            if ri_c is None:
                # i has no data for this cell: adopt j's exactly (shared refs).
                ri[cell] = rj_c
                ci[cell] = cj[cell]
            elif rj_c is None:
                rj[cell] = ri_c
                cj[cell] = ci[cell]
            elif ri_c is rj_c:
                continue  # identical shared set -> nothing new to fuse
            elif ri_c <= rj_c:
                # union == rj_c -> exact re-cluster is j's current clustering
                ri[cell] = rj_c
                ci[cell] = cj[cell]
            elif rj_c <= ri_c:
                rj[cell] = ri_c
                cj[cell] = ci[cell]
            else:
                merged_raw = ri_c | rj_c
                merged_centers = greedy_cluster_centers(merged_raw)
                ri[cell] = merged_raw
                rj[cell] = merged_raw
                ci[cell] = merged_centers
                cj[cell] = merged_centers
            n = len(ci[cell])
            r, c = divmod(cell, gs)
            count_i[r, c] = n
            count_j[r, c] = n

        di[j].clear()
        dj[i].clear()
        for k in range(self.num_agents):
            if k == i or k == j:
                continue
            di[k].update(to_merge)
            dj[k].update(to_merge)

    # ------------------------------------------------------------------
    # FOV / memory
    # ------------------------------------------------------------------

    def get_fov_mask(self, r, c, fov_radius):
        """Square FOV window centred at (r, c), clamped to grid bounds.

        Maze topology: the FOV is further occluded by line-of-sight — a cell
        is visible only if the supercover line from the agent to that cell
        does not cross any wall cell. This mirrors the MAexp radar filter
        (obstructed points are discarded) and makes the maze test the
        topology of INFORMATION, not just of movement.
        """
        gs = self.grid_size
        mask = np.zeros((gs, gs), dtype=bool)
        r0 = max(0, r - fov_radius)
        r1 = min(gs, r + fov_radius + 1)
        c0 = max(0, c - fov_radius)
        c1 = min(gs, c + fov_radius + 1)
        mask[r0:r1, c0:c1] = True
        if self.occlude:
            mask = mask & self._los_visible(r, c, fov_radius, r0, r1, c0, c1)
        return mask

    def _los_visible(self, r, c, fov_radius, r0, r1, c0, c1):
        """Boolean grid of cells with unobstructed line-of-sight from (r, c).

        For every candidate cell inside the FOV square we trace the supercover
        line (grid cells touched by the segment) from the agent to the target.
        If any wall cell lies on the line BEFORE the target, the target is
        occluded. The target cell itself may be a wall (visible as a wall).
        """
        obs = self.obstacle_map
        visible = np.zeros_like(obs)
        for tr in range(r0, r1):
            for tc in range(c0, c1):
                if tr == r and tc == c:
                    visible[tr, tc] = True
                    continue
                dr, dc = tr - r, tc - c
                blocked = False
                for sr, sc in _supercover_rel(dr, dc, fov_radius):
                    rr, cc = r + sr, c + sc
                    if obs[rr, cc]:
                        blocked = True
                        break
                visible[tr, tc] = not blocked
        return visible

    def update_local_memory(self, agent_id, fov_radius):
        """
        Merge visible cells into the agent's local memory.
        pure_local: only marks seen + obstacles (NEVER copies visit counts).
        fov_perfect: full knowledge (ablation).
        Every visible traversable cell also records a bearing observation
        (angular model). For fov_perfect, observations cover the whole grid.
        """
        if self.info_model == "fov_perfect":
            self.local_seen_mask[agent_id][:] = ~self.obstacle_map
            self.local_obstacle_map[agent_id][:] = self.obstacle_map
            self._record_fov_observations(agent_id, fov_radius, whole_grid=True)
            return

        x_obs, y_obs = self._observed_position(agent_id)
        fov = self.get_fov_mask(x_obs, y_obs, fov_radius)

        if self.p_miss > 0:
            rng = self.noise_rngs[agent_id]
            noise = rng.random(fov.shape) > self.p_miss
            fov = fov & noise

        self.local_seen_mask[agent_id][fov] = True
        self.local_obstacle_map[agent_id][fov] = self.obstacle_map[fov]
        self._record_fov_observations(agent_id, fov_radius, fov=fov,
                                      obs_pos=(x_obs, y_obs))

    def _record_fov_observations(self, agent_id, fov_radius, fov=None,
                                 whole_grid=False, obs_pos=None):
        """Record a bearing to every visible traversable cell (angular model).

        Local: per (agent, cell) independent angular configurations (greedy,
        > ANG_TOL, capped at CLUSTER_CAP). Own and local dicts/counts are
        updated separately (local may already contain fused clusters). The
        LOCAL bearing is measured from the observed position `obs_pos` (true
        position when None — identical in Phase 1).
        Global (metrics only): accumulates the bearing-only Fisher information
        per cell from the agent's TRUE position (oracle geometry) and the
        globally clustered configuration set. cap_hit is counted on the LOCAL
        decision signal (mirrors alpha_sat_frac).
        """
        if whole_grid:
            fov = np.ones(self.obstacle_map.shape, dtype=bool)
        elif fov is None:
            fov = self.get_fov_mask(*self.agent_positions[agent_id], fov_radius)
        mask = fov & ~self.obstacle_map
        rows, cols = np.nonzero(mask)
        n = int(rows.size)
        if n == 0:
            return

        if self.sigma_bearing_ref is None:
            from config import QUALITY_SIGMA_BEARING_DEG
            self.sigma_bearing_ref = math.radians(QUALITY_SIGMA_BEARING_DEG)

        # --- Oracle Fisher accumulation (true geometry, metrics only) ---
        r_true, c_true = self.agent_positions[agent_id]
        dy = rows - r_true
        dx = cols - c_true
        d2 = dy * dy + dx * dx
        keep = d2 > 0  # the agent never takes a bearing to its own cell
        rows, cols, dy, dx, d2 = (rows[keep], cols[keep],
                                  dy[keep], dx[keep], d2[keep])
        n = int(rows.size)
        if n == 0:
            return
        d = np.sqrt(d2)
        ux = -dy / d
        uy = dx / d
        w = 1.0 / (self.sigma_bearing_ref ** 2 * d2)
        info = self.global_info
        info[rows, cols, 0] += w * ux * ux
        info[rows, cols, 1] += w * ux * uy
        info[rows, cols, 2] += w * uy * uy

        # --- Per-cell local clustering (decision signal) ---
        if obs_pos is None:
            obs_pos = tuple(self.agent_positions[agent_id])
        ox, oy = obs_pos
        loc_centers = self.local_angle_clusters[agent_id]
        loc_raw = self.local_raw_angles[agent_id]
        gcl = self.global_angle_clusters
        loc_cnt = self.local_measurement_count[agent_id]
        gs = self.grid_size

        cap_hits = 0
        noise_rng = self.noise_rngs[agent_id] if self.sigma_bearing > 0 else None
        for r, c in zip(rows, cols):
            cell = int(r * gs + c)
            angle = math.atan2(c - oy, r - ox)
            loc_angle = angle

            # Bearing-noise injection (robustness sweep): corrupt ONLY the
            # LOCAL decision signal (config-count clustering). The oracle
            # Fisher accumulation (above) and the global cluster record below
            # keep TRUE geometry, so the CRLB metric stays decoupled from the
            # injected measurement error.
            if noise_rng is not None:
                loc_angle = loc_angle + noise_rng.normal(0.0, self.sigma_bearing)
                loc_angle = math.atan2(math.sin(loc_angle), math.cos(loc_angle))

            # local record: raw SET (dedup, copy-on-write) + incremental
            # centers. cap_hit is counted on this LOCAL decision signal.
            lraw = loc_raw.get(cell)
            loc_raw[cell] = (lraw | {loc_angle} if lraw is not None else {loc_angle})
            newc_loc, hit = add_angle_to_clusters(loc_centers.get(cell, ()),
                                                  loc_angle)
            loc_centers[cell] = newc_loc
            loc_cnt[r, c] = len(newc_loc)

            # global oracle: incremental centers only (no fusion at global level)
            newc_gl, _ = add_angle_to_clusters(gcl.get(cell, ()), angle)
            gcl[cell] = newc_gl
            cap_hits += int(hit)

        self.global_obs_count[rows, cols] += 1
        self.global_observation_count += n
        self.global_cap_hit_count += cap_hits

        if self.record_global_raw:
            gro = self.global_raw_obs
            for r, c, in zip(rows, cols):
                cell = int(r * gs + c)
                lst = gro.get(cell)
                if lst is None:
                    gro[cell] = [(ox, oy)]
                else:
                    lst.append((ox, oy))

        if self.info_model == "comm_limited":
            obs_cells = set((rows * gs + cols).tolist())
            di = self.local_dirty[agent_id]
            for j in range(self.num_agents):
                if j != agent_id:
                    di[j].update(obs_cells)

    def _observed_position(self, agent_id):
        """True position possibly corrupted by localisation noise."""
        r, c = self.agent_positions[agent_id]
        if self.sigma_loc <= 0:
            return int(r), int(c)
        rng = self.noise_rngs[agent_id]
        x_f = r + rng.normal(0, self.sigma_loc)
        y_f = c + rng.normal(0, self.sigma_loc)
        self.loc_error_sum += (x_f - r) ** 2 + (y_f - c) ** 2
        self.loc_error_count += 1
        return int(round(x_f)), int(round(y_f))

    def get_rmse_2d(self):
        if self.loc_error_count == 0:
            return 0.0
        return np.sqrt(self.loc_error_sum / self.loc_error_count)

    # ------------------------------------------------------------------
    # Knowledge interface (single access point for policies)
    # ------------------------------------------------------------------

    def get_visit_knowledge(self, agent_id):
        """Visit counts the agent may legitimately use."""
        if self.info_model == "fov_perfect":
            return self.visit_count
        if self.info_model == "comm_limited":
            return self.local_visit_count[agent_id]
        return self.agent_own_visit_count[agent_id]

    def get_known_mask(self, agent_id):
        """Cells the agent knows about (traversable-candidate set)."""
        if self.info_model == "fov_perfect":
            return ~self.obstacle_map
        return self.local_seen_mask[agent_id]

    def get_obstacle_knowledge(self, agent_id):
        """Obstacles the agent knows about."""
        if self.info_model == "fov_perfect":
            return self.obstacle_map
        return self.local_obstacle_map[agent_id]

    def get_total_unknown(self, agent_id):
        """Number of cells the agent believes are still to be covered.
        Used to cap ACE-U. Always expressed in the agent's frame."""
        if self.info_model == "fov_perfect":
            return int(np.sum((self.visit_count == 0) & ~self.obstacle_map))
        known = self.local_seen_mask[agent_id]
        obs = self.local_obstacle_map[agent_id]
        own = self.get_visit_knowledge(agent_id)
        traversable_known = known & ~obs
        return int(np.sum(traversable_known & (own == 0)))

    def get_local_info(self, agent_id):
        """Convenience bundle used by all estimators/policies."""
        return {
            "visit": self.get_visit_knowledge(agent_id),
            "known": self.get_known_mask(agent_id),
            "obs": self.get_obstacle_knowledge(agent_id),
        }

    def get_local_unvisited_mask(self, agent_id):
        """Targets for frontier exploration: known traversable cells the
        agent has not itself visited."""
        info = self.get_local_info(agent_id)
        return info["known"] & ~info["obs"] & (info["visit"] == 0)

    # ------------------------------------------------------------------
    # Angular knowledge interface (single access point for policies)
    # ------------------------------------------------------------------

    def get_config_count_grid(self, agent_id):
        """Per-cell local configuration count (int8 grid). This is the angular
        analog of local_visit_count and is the `visit` input of the transposed
        Chao-U richness signal. For fov_perfect (ablation), global counts."""
        if self.info_model == "fov_perfect":
            out = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
            for cell, centers in self.global_angle_clusters.items():
                r, c = divmod(cell, self.grid_size)
                out[r, c] = len(centers)
            return out
        return self.local_measurement_count[agent_id]

    def global_config_count_grid(self):
        """Per-cell global configuration count (all agents, true geometry).
        Used ONLY for validation diagnostics (Phase 1a), never by policies."""
        out = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for cell, centers in self.global_angle_clusters.items():
            r, c = divmod(cell, self.grid_size)
            out[r, c] = len(centers)
        return out

    def get_total_undetermined(self, agent_id):
        """Number of cells the agent believes are still unlocalized: KNOWN
        traversable cells with fewer than 2 independent angular configurations
        (a single bearing is rank-deficient / not localizable). This is the
        angular analog of get_total_unknown in V4 and caps the transposed
        Chao-U. Cells never seen are not counted (initial pool is 0, so the
        signal starts flat exactly like V4's Frontier+Richness)."""
        if self.info_model == "fov_perfect":
            count = self.get_config_count_grid(agent_id)
            return int(np.sum((count <= 1) & ~self.obstacle_map))
        count = self.local_measurement_count[agent_id]
        known = self.local_seen_mask[agent_id]
        obs = self.local_obstacle_map[agent_id]
        return int(np.sum(known & ~obs & (count <= 1)))

    # ------------------------------------------------------------------
    # Oracle CRLB (metrics ONLY, never fed to policies)
    # ------------------------------------------------------------------

    def global_bound_grid(self):
        """Per-cell bearing-only Cramer-Rao bound sqrt(trace(J^-1)) in grid
        cells, from the TRUE observation geometry. Cells with < 2 independent
        directions (rank-deficient J) or never observed get +inf."""
        J00 = self.global_info[..., 0]
        J01 = self.global_info[..., 1]
        J11 = self.global_info[..., 2]
        det = J00 * J11 - J01 * J01
        trace = J00 + J11
        bound = np.full_like(det, np.inf)
        ok = det > 1e-12
        bound[ok] = np.sqrt(trace[ok] / det[ok])
        return bound

    def global_undetermined_fraction(self):
        """Fraction of traversable cells with 0 global configurations
        (never observed), i.e. still unexplored/unlocalized space."""
        tr = ~self.obstacle_map
        if not self.traversable:
            return 0.0
        return float(np.sum((self.global_obs_count == 0) & tr)) / \
            self.traversable

    def cluster_cap_hit_frac(self):
        """Fraction of traversable observations blocked by the CLUSTER_CAP on
        the LOCAL decision signal (mirrors alpha_sat_frac). None if no
        observations yet."""
        if self.global_observation_count == 0:
            return None
        return float(self.global_cap_hit_count) / self.global_observation_count

    def quality_well_localized(self, threshold=None):
        """quality(t) = fraction of traversable cells whose CRLB bound
        <= threshold (grid cells). threshold defaults to config."""
        if threshold is None:
            if self.quality_threshold is None:
                from config import QUALITY_THRESHOLD
                self.quality_threshold = QUALITY_THRESHOLD
            threshold = self.quality_threshold
        bound = self.global_bound_grid()
        tr = ~self.obstacle_map
        if not self.traversable:
            return 0.0
        return float(np.sum((bound <= threshold) & tr)) / self.traversable

    # ------------------------------------------------------------------
    # Coverage helpers (metrics use GLOBAL truth)
    # ------------------------------------------------------------------

    def global_coverage(self):
        visited = int(np.sum((self.visit_count > 0) & ~self.obstacle_map))
        return 100.0 * visited / self.traversable if self.traversable else 0.0

    def local_coverage(self, agent_id):
        """Fraction of the agent's known traversable cells it has visited."""
        info = self.get_local_info(agent_id)
        tr = info["known"] & ~info["obs"]
        total = int(np.sum(tr))
        if total == 0:
            return 0.0
        visited = int(np.sum((info["visit"][tr] > 0)))
        return 100.0 * visited / total

    def reset(self):
        self.visit_count[:] = 0
        for i in range(self.num_agents):
            self.agent_own_visit_count[i][:] = 0
            self.local_visit_count[i][:] = 0
            self.local_seen_mask[i][:] = False
            self.local_obstacle_map[i][:] = False
        self.fusion_events_count = 0
        self.last_rendezvous_pairs = 0
        self.explore_exhausted = [False] * self.num_agents
        self.explore_exhausted_unvisited = [None] * self.num_agents
        self.explore_exhausted_obs = [None] * self.num_agents
        self.loc_error_sum = 0.0
        self.loc_error_count = 0
        for i in range(self.num_agents):
            self.local_angle_clusters[i].clear()
            self.local_raw_angles[i].clear()
            self.local_measurement_count[i][:] = 0
            for j in range(self.num_agents):
                self.local_dirty[i][j].clear()
        self.global_angle_clusters.clear()
        self.global_info[:] = 0.0
        self.global_obs_count[:] = 0
        self.global_observation_count = 0
        self.global_cap_hit_count = 0
        self.global_raw_obs.clear()
        self._place_agents()


class NoisyGridEnv(GridEnv):
    """Alias kept for backward compatibility with run scripts."""
