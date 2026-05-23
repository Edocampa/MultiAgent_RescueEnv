"""
Hierarchical abstraction layer

For each episode (defined by a concrete 10x10 map), this class computes V*
analytically via Value Iteration at every level (3, 5, 10). VI is exact and
instantaneous on the small abstract grids

The V* at the upper level is then used as the shaping signal for training
at the lower level:
    Training at 5  -> shaping = V*_3(phi_5_to_3(pos))
    Training at 10 -> shaping = V*_5(phi_10_to_5(pos))
    Training at 3  -> NO shaping (base case, sparse reward)

V* is single-agent: same V* applied to both agents independently as a spatial heuristic
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "env"))
from map_generator import (
    project_map, phi_10_to_5, phi_5_to_3, phi_10_to_3, get_phi
)


def compute_V_via_VI(goal_cell, blocked_cells, grid_size, gamma=0.8):
    """
    Solves single-agent shortest-path MDP via Value Iteration

    V*(goal) = 1.0
    V*(blocked) = 0.0
    V*(s) = gamma * max_{neighbor s'} V*(s')
    """
    V = np.zeros((grid_size, grid_size), dtype=np.float64)
    if goal_cell not in blocked_cells:
        V[goal_cell[0], goal_cell[1]] = 1.0

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(500):
        V_old = V.copy()
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) == goal_cell:
                    continue
                if (r, c) in blocked_cells:
                    V[r, c] = 0.0
                    continue
                best = 0.0
                for dr, dc in moves:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        if (nr, nc) not in blocked_cells:
                            best = max(best, V_old[nr, nc])
                V[r, c] = gamma * best
        if np.max(np.abs(V - V_old)) < 1e-6:
            break

    return V


class HierarchicalAbstraction:
    """
    Manages V* maps at all three levels for the current episode

    Per-episode workflow:
        1. abstraction.update(concrete_map): receives the 10x10 map,
           projects it to all levels, computes V* at each level via VI.
        2. During training, abstraction.compute_shaping(pos, training_level, ...)
           returns the shaping signal for an agent at the given position
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.V_to_item = {3: None, 5: None, 10: None}
        self.V_to_victim = {3: None, 5: None, 10: None}

    def update(self, concrete_map):
        """
        Recompute V* at all levels using the projected versions of concrete_map

        Args:
            concrete_map: dict from generate_concrete_map() (the 10x10 map)
        """

        gamma_shaping = 0.8

        for level in [3, 5, 10]:
            level_map = project_map(concrete_map, level)
            blocked = set(level_map["wall_cells"]) | set(level_map["fire_cells"])

            self.V_to_item[level] = compute_V_via_VI(
                level_map["item_pos"], blocked, level, gamma_shaping
            )
            self.V_to_victim[level] = compute_V_via_VI(
                level_map["victim_pos"], blocked, level, gamma_shaping
            )

    def compute_shaping(
        self,
        pos,
        training_level,
        has_item,
        differential=True,
        prev_pos=None,
        prev_has_item=None,
        scale=1.0,
    ):
        """
        Computes shaping signal for an agent at training_level

        Shaping comes from V* at the level UPPER than training_level
        (i.e., one level coarser, with smaller grid and less info)


        The discount factor gamma is included for theoretical correctness:
        without it, the shaping accumulates a non-zero bias even on optimal
        paths and confuses the policy gradient. With gamma=0.99 the practical
        difference is small, but it makes the formula consistent with the
        standard PBRS theory.
        
        """
        if training_level == 3:
            return 0.0   # base case: no shaping

        upper_level = 5 if training_level == 10 else 3
        phi = get_phi(training_level, upper_level)

        upper_pos = phi(*pos)
        V_upper = (self.V_to_victim[upper_level] if has_item
                   else self.V_to_item[upper_level])
        if V_upper is None:
            return 0.0

        F_new = float(V_upper[upper_pos[0], upper_pos[1]])

        if not differential:
            return scale * F_new

        if prev_pos is None:
            return 0.0

        had_item = prev_has_item if prev_has_item is not None else has_item
        V_prev = (self.V_to_victim[upper_level] if had_item
                  else self.V_to_item[upper_level])
        prev_upper = phi(*prev_pos)
        F_old = float(V_prev[prev_upper[0], prev_upper[1]])

        # Canonical PBRS: F = scale * (gamma * V(new) - V(old))
        return scale * (F_new - F_old)

    def print_V(self, level, has_item=False):
        V = self.V_to_victim[level] if has_item else self.V_to_item[level]
        if V is None:
            print(f"V at level {level}: not computed")
            return
        target = "victim" if has_item else "item"
        print(f"V at level {level}, target = {target}:")
        for r in range(V.shape[0]):
            print("  " + " ".join(f"{V[r, c]:.3f}" for c in range(V.shape[1])))
