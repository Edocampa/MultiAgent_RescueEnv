"""
map_generator.py - From v3 with density-based projection rule.
Used by env_discrete for levels 3, 6, 12, 25, 50 (the discrete pyramid).
"""

import numpy as np
from collections import deque


def project_pos(r, c, source_size, target_size):
    return (r * target_size // source_size,
            c * target_size // source_size)


def get_phi(source_level, target_level):
    if source_level == target_level:
        return lambda r, c: (r, c)
    if source_level < target_level:
        raise ValueError(f"Cannot project up: {source_level} -> {target_level}")
    return lambda r, c: project_pos(r, c, source_level, target_level)


def generate_concrete_map(size, num_fires=2, wall_density=0.10,
                          random_walls=True, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    if size <= 3:
        actual_walls = 0
        actual_fires = 0
    elif size <= 6:
        actual_walls = int(size * size * wall_density * 0.5)
        actual_fires = min(num_fires, 1)
    else:
        actual_walls = int(size * size * wall_density) if random_walls else 0
        actual_fires = num_fires

    min_fire_dist = max(2, size // 5)
    min_iv_dist = max(2, size // 3)
    agent_zone_height = max(1, size // 4)
    item_zone_start = size // 2 if size >= 6 else 0

    for attempt in range(2000):
        all_cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(all_cells)

        wall_set = set(all_cells[:actual_walls]) if actual_walls > 0 else set()
        free_cells = [c for c in all_cells if c not in wall_set]
        rng.shuffle(free_cells)

        if len(free_cells) < 4 + actual_fires:
            continue

        top_left_zone = [c for c in free_cells
                         if c[0] < agent_zone_height and c[1] < size // 2]
        top_right_zone = [c for c in free_cells
                          if c[0] < agent_zone_height and c[1] >= size // 2]

        if not top_left_zone or not top_right_zone:
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

        item_candidates = [c for c in free_cells
                           if c[0] >= item_zone_start and c not in occupied]
        if not item_candidates:
            item_candidates = [c for c in free_cells if c not in occupied]
        if not item_candidates:
            continue
        item_pos = item_candidates[rng.randint(len(item_candidates))]
        occupied.add(item_pos)

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

        if actual_fires > 0:
            fire_candidates = [
                c for c in free_cells
                if c not in occupied
                and all(abs(c[0] - s[0]) + abs(c[1] - s[1]) >= min_fire_dist
                        for s in agent_starts)
            ]
            actual_n = min(actual_fires, len(fire_candidates))
            if actual_n > 0:
                fire_indices = rng.choice(len(fire_candidates),
                                           size=actual_n, replace=False)
                fire_cells = [fire_candidates[i] for i in fire_indices]
            else:
                fire_cells = []
        else:
            fire_cells = []

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

    raise RuntimeError(f"Could not generate valid {size}x{size} map.")


def project_map(concrete_map, target_size, rule="auto"):
    source_size = concrete_map["size"]
    if target_size == source_size:
        return dict(concrete_map)
    if target_size > source_size:
        raise ValueError(f"Cannot project up: {source_size} -> {target_size}")

    phi = get_phi(source_size, target_size)
    ratio = source_size / target_size

    if rule == "auto":
        rule = "density"

    item_pos = phi(*concrete_map["item_pos"])
    victim_pos = phi(*concrete_map["victim_pos"])
    agent_starts = [phi(*p) for p in concrete_map["agent_starts"]]

    concrete_blocked = set(concrete_map["wall_cells"]) | set(concrete_map["fire_cells"])
    coarse_total, coarse_blocked = {}, {}

    for r in range(source_size):
        for c in range(source_size):
            cr, cc = phi(r, c)
            key = (cr, cc)
            coarse_total[key] = coarse_total.get(key, 0) + 1
            if (r, c) in concrete_blocked:
                coarse_blocked[key] = coarse_blocked.get(key, 0) + 1

    blocked_coarse = set()
    for key, total in coarse_total.items():
        blocked_count = coarse_blocked.get(key, 0)
        if rule == "any" and blocked_count >= 1:
            blocked_coarse.add(key)
        elif rule == "majority" and blocked_count >= total * 0.5:
            blocked_coarse.add(key)
        elif rule == "density" and blocked_count >= total * 0.33:
            blocked_coarse.add(key)

    forbidden = {item_pos, victim_pos} | set(agent_starts)
    blocked_coarse = blocked_coarse - forbidden

    return {
        "wall_cells": list(blocked_coarse),
        "fire_cells": [],
        "agent_starts": agent_starts,
        "item_pos": item_pos,
        "victim_pos": victim_pos,
        "size": target_size,
    }


def _bfs_reachable(start, goal, blocked, size):
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
