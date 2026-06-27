"""
continuous_map.py - Map generator for the continuous 100x100 environment.

Adapted from v3's map_generator. The key differences are:
    - Wall/fire cells are still aligned to the integer grid (cells are 1x1
      squares in continuous coordinates)
    - Agent, item, victim positions are CONTINUOUS — they are placed at the
      CENTER of their assigned cell (so position = (i + 0.5, j + 0.5))
    - The BFS connectivity check works on cell-level (same as v3)

The map_dict produced is the same shape as v3, with positions now being
floats instead of integers.
"""

import numpy as np
from collections import deque


def generate_continuous_map(size=100, num_fires=15, wall_density=0.08,
                            random_walls=True, rng=None):
    """
    Generates a random valid continuous map.

    Args:
        size:         grid dimension for wall/fire placement
        num_fires:    number of fire cells
        wall_density: fraction of cells used as walls
        random_walls: if True, walls are random
        rng:          numpy RandomState

    Returns:
        dict with:
            wall_cells:   list of (i, j) integer cells that are walls
            fire_cells:   list of (i, j) integer cells that are fires
            agent_starts: list of 2 continuous positions np.array([r, c])
            item_pos:     continuous position np.array([r, c])
            victim_pos:   continuous position np.array([r, c])
            size:         the grid dimension
    """
    if rng is None:
        rng = np.random.RandomState()

    actual_walls = int(size * size * wall_density) if random_walls else 0
    actual_fires = num_fires

    min_fire_dist = max(2, size // 20)
    min_iv_dist = max(4, size // 6)
    agent_zone_height = max(1, size // 6)
    item_zone_start = size // 2

    for attempt in range(2000):
        all_cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(all_cells)

        wall_set = set(all_cells[:actual_walls]) if actual_walls > 0 else set()
        free_cells = [c for c in all_cells if c not in wall_set]
        rng.shuffle(free_cells)

        if len(free_cells) < 4 + actual_fires:
            continue

        # Place agents in the top zone, well-separated
        top_left = [c for c in free_cells
                    if c[0] < agent_zone_height and c[1] < size // 2]
        top_right = [c for c in free_cells
                     if c[0] < agent_zone_height and c[1] >= size // 2]
        if not top_left or not top_right:
            continue
        agent_cell_0 = top_left[rng.randint(len(top_left))]
        agent_cell_1 = top_right[rng.randint(len(top_right))]
        if agent_cell_0 == agent_cell_1:
            continue
        occupied = {agent_cell_0, agent_cell_1}

        # Item in lower zone
        item_candidates = [c for c in free_cells
                           if c[0] >= item_zone_start and c not in occupied]
        if not item_candidates:
            continue
        item_cell = item_candidates[rng.randint(len(item_candidates))]
        occupied.add(item_cell)

        # Victim distant from item
        victim_candidates = [
            c for c in free_cells
            if c not in occupied
            and abs(c[0] - item_cell[0]) + abs(c[1] - item_cell[1]) >= min_iv_dist
        ]
        if not victim_candidates:
            continue
        victim_cell = victim_candidates[rng.randint(len(victim_candidates))]
        occupied.add(victim_cell)

        # Fires far from agents
        if actual_fires > 0:
            fire_candidates = [
                c for c in free_cells
                if c not in occupied
                and all(abs(c[0] - s[0]) + abs(c[1] - s[1]) >= min_fire_dist
                        for s in [agent_cell_0, agent_cell_1])
            ]
            n_fires = min(actual_fires, len(fire_candidates))
            if n_fires > 0:
                fire_indices = rng.choice(len(fire_candidates), size=n_fires,
                                          replace=False)
                fire_cells = [fire_candidates[i] for i in fire_indices]
            else:
                fire_cells = []
        else:
            fire_cells = []

        # Connectivity check (cell-level, treating fires as passable)
        # We check that walls don't completely block paths. Fires are passable
        # so we don't include them in `blocked` for the BFS.
        blocked = wall_set
        valid = True
        for start in [agent_cell_0, agent_cell_1]:
            if not _bfs_reachable(start, item_cell, blocked, size):
                valid = False
                break
        if valid:
            if not _bfs_reachable(item_cell, victim_cell, blocked, size):
                valid = False

        if valid:
            # Convert cell positions to continuous (cell center)
            def cell_center(cell):
                return np.array([cell[0] + 0.5, cell[1] + 0.5], dtype=np.float32)

            return {
                "wall_cells": list(wall_set),
                "fire_cells": fire_cells,
                "agent_starts": [cell_center(agent_cell_0),
                                  cell_center(agent_cell_1)],
                "item_pos": cell_center(item_cell),
                "victim_pos": cell_center(victim_cell),
                "size": size,
            }

    raise RuntimeError(
        f"Could not generate valid continuous map after 2000 attempts "
        f"(size={size}, walls={actual_walls}, fires={actual_fires})"
    )


def _bfs_reachable(start, goal, blocked, size):
    """BFS reachability check on cell-level integer grid."""
    if start == goal:
        return True
    if start in blocked or goal in blocked:
        return False
    visited = {start}
    queue = deque([start])
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if (nr, nc) not in visited and (nr, nc) not in blocked:
                    if (nr, nc) == goal:
                        return True
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False
