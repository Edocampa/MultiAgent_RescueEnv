"""
Procedural map generation + projection to coarser levels

The fundamental unit is a 10x10 concrete map with fixed walls (results_1) and randomized
positions of agents, item, victim and fires. 
This concrete map is generated once per episode.

To get the same map at level 5 or 3, with the function project_map() applies the
appropriate phi function to every position and computes the coarse obstacle map
"""

import numpy as np
from collections import deque


# Fixed walls for the 10x10 concrete level
FIXED_WALLS_10 = [
    (3, 2), (3, 3), (3, 4), (3, 5),
    (6, 4), (6, 5), (6, 6), (6, 7),
]


# Phi mapping functions

def phi_10_to_5(r, c):
    """10x10 cell -> 5x5 cell (uniform 2x2 blocking)"""
    return (r // 2, c // 2)


def phi_5_to_3(r, c):
    """5x5 cell -> 3x3 cell (non-uniform: row mapping 0,0,1,2,2)"""
    row_map = [0, 0, 1, 2, 2]
    col_map = [0, 0, 1, 2, 2]
    return (row_map[r], col_map[c])


def phi_10_to_3(r, c):
    """Composition: 10x10 -> 5x5 -> 3x3"""
    r5, c5 = phi_10_to_5(r, c)
    return phi_5_to_3(r5, c5)


def get_phi(source_level, target_level):
    """Returns the projection function from source to target level"""
    if source_level == target_level:
        return lambda r, c: (r, c)
    if source_level == 10 and target_level == 5:
        return phi_10_to_5
    if source_level == 10 and target_level == 3:
        return phi_10_to_3
    if source_level == 5 and target_level == 3:
        return phi_5_to_3
    raise ValueError(f"No projection {source_level} -> {target_level}")


# Concrete 10x10 map generation
def generate_concrete_map(num_fires=5, random_walls=False, num_random_walls=8, rng=None):
    """
    Universal generator for 10x10 maps.
    
    Args:
        num_fires: Number of dynamic fire obstacles.
        random_walls: If True, generates walls randomly (Result Difficult). 
                      If False, uses FIXED_WALLS_10 (Result 1).
        num_random_walls: How many walls to place if random_walls is True.
        rng: NumPy RandomState.
    """
    if rng is None:
        rng = np.random.RandomState()

    size = 10
    
    # Attempt to generate a valid configuration
    for attempt in range(1000): 
        all_cells = [(r, c) for r in range(size) for c in range(size)]
        
        # Wall logic switch
        if random_walls:
            # Case: Procedural Geometry (Difficult)
            # We shuffle all cells and pick the first N as walls
            rng.shuffle(all_cells)
            wall_set = set(all_cells[:num_random_walls])
        else:
            # Case: Static Geometry (Result 1)
            # We use the predefined fixed walls
            wall_set = set(FIXED_WALLS_10)

        # Identify cells not occupied by walls
        free_cells = [c for c in all_cells if c not in wall_set]
        rng.shuffle(free_cells)

        # 1. Place Agents (taking the first two available shuffled cells)
        agent_starts = [free_cells[0], free_cells[1]]
        occupied = {agent_starts[0], agent_starts[1]}

        # 2. Place Item (Medical Kit) - constrained to the lower half
        item_candidates = [c for c in free_cells if c[0] >= size // 2 and c not in occupied]
        if not item_candidates: continue
        item_pos = item_candidates[rng.randint(len(item_candidates))]
        occupied.add(item_pos)

        # 3. Place Victim - distant from the item
        victim_candidates = [
            c for c in free_cells if c not in occupied 
            and abs(c[0] - item_pos[0]) + abs(c[1] - item_pos[1]) >= 4
        ]
        if not victim_candidates: continue
        victim_pos = victim_candidates[rng.randint(len(victim_candidates))]
        occupied.add(victim_pos)

        # 4. Place Fires - dynamic obstacles away from agents
        fire_candidates = [
            c for c in free_cells if c not in occupied
            and all(abs(c[0] - s[0]) + abs(c[1] - s[1]) >= 2 for s in agent_starts)
        ]
        if len(fire_candidates) < num_fires: continue
        fire_indices = rng.choice(len(fire_candidates), size=num_fires, replace=False)
        fire_cells = [fire_candidates[i] for i in fire_indices]

        # 5. Connectivity Validation via BFS
        blocked = wall_set | set(fire_cells)
        valid = True
        for start in agent_starts:
            if not _bfs_reachable(start, item_pos, blocked, size):
                valid = False; break
        if valid and not _bfs_reachable(item_pos, victim_pos, blocked, size):
            valid = False

        if valid:
            return {
                "wall_cells": list(wall_set),
                "fire_cells": fire_cells,
                "agent_starts": agent_starts,
                "item_pos": item_pos,
                "victim_pos": victim_pos,
                "size": 10,
            }

    raise RuntimeError("Failed to generate a valid map after 1000 attempts")



# Projection of a concrete map to a coarser level

def project_map(concrete_map, target_level):
    """
    Projects a concrete 10x10 map to a coarser level

    HYBRID PROJECTION RULE (revised after first training results):
        - 10 -> 5: ANY-IS-BLOCKED. A 5x5 cell is blocked if at least ONE of
          its 4 concrete cells is a wall or fire. This makes V*_5 aware of
          obstacles, so the shaping signal at level 10 can guide agents AROUND
          obstacles instead of straight through them.
        - 10 -> 3 / 5 -> 3: MAJORITY-IS-BLOCKED. A 3x3 cell is blocked if at
          least 50% of its concrete cells are blocked. Pure "any-is-blocked"
          would over-block the 3x3 (corner cells span up to 16 concrete cells)
          and make the base case unsolvable.

    Fires are treated AS obstacles in the projection (they contribute to the
    blocked count). This way V*_abstract is aware of fire zones and shaping
    pulls agents away from them.

    Critical positions (item, victim, agents) are never blocked: if projection
    would block them, the block is removed for that cell

    Args:
        concrete_map: dict from generate_concrete_map()
        target_level: 10, 5, or 3

    Returns:
        dict with the projected map at target_level
    """
    if target_level == 10:
        return dict(concrete_map)

    phi = get_phi(10, target_level)

    # Project positions
    item_pos = phi(*concrete_map["item_pos"])
    victim_pos = phi(*concrete_map["victim_pos"])
    agent_starts = [
        phi(*concrete_map["agent_starts"][0]),
        phi(*concrete_map["agent_starts"][1]),
    ]

    # Count blocked vs total concrete cells per coarse cell
    concrete_blocked = set(concrete_map["wall_cells"]) | set(concrete_map["fire_cells"])
    coarse_total = {}
    coarse_blocked = {}

    for r in range(10):
        for c in range(10):
            cr, cc = phi(r, c)
            key = (cr, cc)
            coarse_total[key] = coarse_total.get(key, 0) + 1
            if (r, c) in concrete_blocked:
                coarse_blocked[key] = coarse_blocked.get(key, 0) + 1

    # Apply the appropriate rule based on target level
    blocked_coarse = set()
    if target_level == 5:
        # ANY-IS-BLOCKED: blocked if >= 1 concrete cell blocked
        for key, _ in coarse_total.items():
            if coarse_blocked.get(key, 0) >= 1:
                blocked_coarse.add(key)
    elif target_level == 3:
        # MAJORITY-IS-BLOCKED: blocked if >= 50% concrete cells blocked
        for key, total in coarse_total.items():
            if coarse_blocked.get(key, 0) >= total * 0.5:
                blocked_coarse.add(key)

    # never block critical positions
    forbidden = {item_pos, victim_pos, agent_starts[0], agent_starts[1]}
    blocked_coarse = blocked_coarse - forbidden

    # All blocked cells are reported as walls. Fires are not separately tracked
    # at coarse levels - they're absorbed into the wall set
    return {
        "wall_cells": list(blocked_coarse),
        "fire_cells": [],
        "agent_starts": agent_starts,
        "item_pos": item_pos,
        "victim_pos": victim_pos,
        "size": target_level,
    }


# BFS utility

def _bfs_reachable(start, goal, blocked, size):
    """Standard BFS reachability check."""
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
