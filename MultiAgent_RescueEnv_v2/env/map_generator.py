"""
map_generator.py - Universal procedural map generator + universal projection.

NEW IN v3:
    - Parametric grid size N (works for 6, 12, 24, 48, 96, ... any N)
    - Random walls with configurable density (no more hardcoded FIXED_WALLS)
    - Universal phi function: project_pos(r, c, source, target) for any
      integer ratio. Auto-detects projection rule based on ratio:
        - ratio == 2: any-is-blocked  (one blocked cell makes coarse blocked)
        - ratio >  2: majority-is-blocked (>= 50% blocked makes coarse blocked)
    - Same Option A philosophy: episode K of any experiment uses the SAME
      underlying concrete map seed; coarser representations are projections.
"""

import numpy as np
from collections import deque


# ── Universal projection function ─────────────────────────────────────

def project_pos(r, c, source_size, target_size):
    """
    Maps a position from a source NxN grid to a target MxM grid.

    Works for any integer ratio. For source=24, target=12 (ratio 2),
    each target cell collapses a 2x2 block from source. For source=24,
    target=6 (ratio 4), each target cell collapses a 4x4 block.
    """
    return (r * target_size // source_size,
            c * target_size // source_size)


def get_phi(source_level, target_level):
    """
    Returns a projection function from source to target level (closure).

    Usage:
        phi = get_phi(24, 12)
        phi(r, c)  # returns (r_target, c_target)
    """
    if source_level == target_level:
        return lambda r, c: (r, c)
    if source_level < target_level:
        raise ValueError(f"Cannot project from smaller to larger: "
                         f"{source_level} -> {target_level}")
    return lambda r, c: project_pos(r, c, source_level, target_level)


# ── Concrete N×N map generation ───────────────────────────────────────

def generate_concrete_map(size, num_fires=2, wall_density=0.10,
                          random_walls=True, rng=None):
    """
    Generates a random valid N×N concrete map.

    Args:
        size:         grid dimension (any N >= 4 makes sense; 3 too small for fires)
        num_fires:    number of fire cells (capped if grid is small)
        wall_density: fraction of cells to use as walls (e.g. 0.10 = 10%)
        random_walls: if True, walls are randomly placed; if False, no walls
                      (useful for the base case at level 3)
        rng:          numpy RandomState for reproducibility

    Returns:
        dict with map configuration
    """
    if rng is None:
        rng = np.random.RandomState()

    # Auto-scale parameters for very small grids
    if size <= 3:
        actual_walls = 0
        actual_fires = 0   # too cramped
    elif size <= 6:
        actual_walls = int(size * size * wall_density * 0.5)  # less dense
        actual_fires = min(num_fires, 1)
    else:
        actual_walls = int(size * size * wall_density) if random_walls else 0
        actual_fires = num_fires

    # Minimum fire distance scales with grid size
    min_fire_dist = max(2, size // 5)
    # Minimum item-victim distance scales similarly
    min_iv_dist = max(2, size // 3)
    # Agent placement area (top of grid)
    agent_zone_height = max(1, size // 4)
    # Item placement area (bottom of grid)
    item_zone_start = size // 2 if size >= 6 else 0

    for attempt in range(2000):
        all_cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(all_cells)

        # 1. Place walls randomly
        if actual_walls > 0:
            wall_set = set(all_cells[:actual_walls])
        else:
            wall_set = set()

        free_cells = [c for c in all_cells if c not in wall_set]
        rng.shuffle(free_cells)

        if len(free_cells) < 4 + actual_fires:
            continue   # too few free cells, retry

        # 2. Place two agents in the top zone
        top_left_zone = [c for c in free_cells
                         if c[0] < agent_zone_height and c[1] < size // 2]
        top_right_zone = [c for c in free_cells
                          if c[0] < agent_zone_height and c[1] >= size // 2]

        if not top_left_zone or not top_right_zone:
            # fallback: any two free cells
            if len(free_cells) < 2:
                continue
            agent_starts = [free_cells[0], free_cells[1]]
        else:
            agent_starts = [
                top_left_zone[rng.randint(len(top_left_zone))],
                top_right_zone[rng.randint(len(top_right_zone))],
            ]

        if agent_starts[0] == agent_starts[1]:
            continue

        occupied = set(agent_starts)

        # 3. Place item in lower zone
        item_candidates = [c for c in free_cells
                           if c[0] >= item_zone_start and c not in occupied]
        if not item_candidates:
            item_candidates = [c for c in free_cells if c not in occupied]
        if not item_candidates:
            continue
        item_pos = item_candidates[rng.randint(len(item_candidates))]
        occupied.add(item_pos)

        # 4. Place victim distant from item
        victim_candidates = [
            c for c in free_cells
            if c not in occupied
            and abs(c[0] - item_pos[0]) + abs(c[1] - item_pos[1]) >= min_iv_dist
        ]
        if not victim_candidates:
            victim_candidates = [c for c in free_cells if c not in occupied]
        if not victim_candidates:
            continue
        victim_pos = victim_candidates[rng.randint(len(victim_candidates))]
        occupied.add(victim_pos)

        # 5. Place fires far from agents
        if actual_fires > 0:
            fire_candidates = [
                c for c in free_cells
                if c not in occupied
                and all(abs(c[0] - s[0]) + abs(c[1] - s[1]) >= min_fire_dist
                        for s in agent_starts)
            ]
            actual_n = min(actual_fires, len(fire_candidates))
            if actual_n > 0:
                fire_indices = rng.choice(
                    len(fire_candidates), size=actual_n, replace=False
                )
                fire_cells = [fire_candidates[i] for i in fire_indices]
            else:
                fire_cells = []
        else:
            fire_cells = []

        # 6. Connectivity check via BFS
        blocked = wall_set | set(fire_cells)
        valid = True
        for start in agent_starts:
            if not _bfs_reachable(start, item_pos, blocked, size):
                valid = False
                break
        if valid:
            if not _bfs_reachable(item_pos, victim_pos, blocked, size):
                valid = False

        if valid:
            return {
                "wall_cells": list(wall_set),
                "fire_cells": fire_cells,
                "agent_starts": agent_starts,
                "item_pos": item_pos,
                "victim_pos": victim_pos,
                "size": size,
            }

    raise RuntimeError(
        f"Could not generate valid {size}x{size} map after 2000 attempts "
        f"(walls={actual_walls}, fires={actual_fires})"
    )


# ── Universal projection of a concrete map to a coarser level ─────────

def project_map(concrete_map, target_size, rule="auto"):
    """
    Projects a concrete map to a coarser level. Universal: works for any
    source->target with target evenly dividing source.

    Args:
        concrete_map: dict from generate_concrete_map() at some source_size
        target_size:  must be <= source_size, ideally source_size % target_size == 0
        rule:         "auto", "any", or "majority"
                      - "auto": chooses based on ratio (any if ratio<=2, else majority)
                      - "any":  coarse blocked if >=1 concrete cell blocked
                      - "majority": coarse blocked if >=50% concrete cells blocked

    Returns:
        dict with the projected map at target_size
    """
    source_size = concrete_map["size"]
    if target_size == source_size:
        return dict(concrete_map)
    if target_size > source_size:
        raise ValueError(f"Cannot project up: {source_size} -> {target_size}")

    phi = get_phi(source_size, target_size)
    ratio = source_size / target_size

    # Auto-select rule based on ratio
    if rule == "auto":
        rule = "density"

    # Project positions
    item_pos = phi(*concrete_map["item_pos"])
    victim_pos = phi(*concrete_map["victim_pos"])
    agent_starts = [phi(*p) for p in concrete_map["agent_starts"]]

    # Count blocked cells in each coarse cell
    concrete_blocked = set(concrete_map["wall_cells"]) | set(concrete_map["fire_cells"])
    coarse_total = {}
    coarse_blocked = {}

    for r in range(source_size):
        for c in range(source_size):
            cr, cc = phi(r, c)
            key = (cr, cc)
            coarse_total[key] = coarse_total.get(key, 0) + 1
            if (r, c) in concrete_blocked:
                coarse_blocked[key] = coarse_blocked.get(key, 0) + 1

    # Apply the chosen rule
    blocked_coarse = set()
    for key, total in coarse_total.items():
        blocked_count = coarse_blocked.get(key, 0)
        if rule == "any" and blocked_count >= 1:
            blocked_coarse.add(key)
        elif rule == "majority" and blocked_count >= total * 0.5:
            blocked_coarse.add(key)
        elif rule == "density" and blocked_count >= total * 0.33:
            # Coarse cell blocked if >= 33% of sub-cells are blocked
            blocked_coarse.add(key)

    # Never block critical positions
    forbidden = {item_pos, victim_pos} | set(agent_starts)
    blocked_coarse = blocked_coarse - forbidden

    return {
        "wall_cells": list(blocked_coarse),
        "fire_cells": [],   # fires absorbed into walls at coarse levels
        "agent_starts": agent_starts,
        "item_pos": item_pos,
        "victim_pos": victim_pos,
        "size": target_size,
    }


# ── BFS reachability ──────────────────────────────────────────────────

def _bfs_reachable(start, goal, blocked, size):
    """Standard BFS reachability check on an NxN grid."""
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
