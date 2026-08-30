"""
abstract_layer.py - Generalized hierarchical abstraction for N levels.

For each episode (defined by a concrete map at the largest level), computes V*
at every level of the pyramid via Value Iteration. The V* at the level above
is used as the shaping signal for training at the current level.

Generalization in v3:
    - Accepts an arbitrary list of levels, e.g. [3, 6, 12, 24]
    - Shaping at level L uses V* at the level immediately above (coarser)
      in the pyramid, found via list indexing.
    - V* computed analytically on the projected grids.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "env"))
from map_generator import project_map, get_phi


def compute_V_via_VI(goal_cell, blocked_cells, grid_size, gamma=0.8,
                     max_iters=500, tol=1e-6):
    """
    Solves single-agent shortest-path MDP via Value Iteration.

    V*(goal) = 1.0
    V*(blocked) = 0.0
    V*(s) = gamma * max_{neighbor s'} V*(s')

    Lower gamma_VI (e.g. 0.85) yields a steeper spatial gradient, which makes
    the shaping signal more informative. This gamma is INDEPENDENT of the
    MAPPO gamma (which stays 0.99 for the RL task).
    """
    V = np.zeros((grid_size, grid_size), dtype=np.float64)
    if goal_cell not in blocked_cells:
        V[goal_cell[0], goal_cell[1]] = 1.0

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(max_iters):
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
        if np.max(np.abs(V - V_old)) < tol:
            break

    return V


class HierarchicalAbstraction:
    """
    Manages V* maps for an N-level pyramid.

    Per-episode workflow:
        1. update(concrete_map): projects to all levels, computes V*.
        2. compute_shaping(pos, training_level, ...): returns shaping signal
           for an agent operating at training_level, using V* at the level
           immediately above in the pyramid.

    Args:
        levels: list of levels in ascending order, e.g. [3, 6, 12, 24]
        gamma_VI: discount for VI (default 0.8 for steep gradient)
    """

    def __init__(self, levels, gamma_VI=0.8):
        self.levels = sorted(levels)   # ascending
        self.gamma_VI = gamma_VI
        self.V_to_item = {L: None for L in self.levels}
        self.V_to_victim = {L: None for L in self.levels}

    def update(self, concrete_map):
        """
        Recompute V* at all levels using projections of concrete_map.

        concrete_map should be at the LARGEST level (self.levels[-1]).
        """
        source_size = concrete_map["size"]
        assert source_size == self.levels[-1], (
            f"concrete_map size {source_size} should match top level "
            f"{self.levels[-1]}"
        )

        for level in self.levels:
            if level == source_size:
                level_map = concrete_map
            else:
                level_map = project_map(concrete_map, level, rule="any")

            blocked = set(level_map["wall_cells"]) | set(level_map["fire_cells"])

            self.V_to_item[level] = compute_V_via_VI(
                level_map["item_pos"], blocked, level, self.gamma_VI
            )
            self.V_to_victim[level] = compute_V_via_VI(
                level_map["victim_pos"], blocked, level, self.gamma_VI
            )

    def get_upper_level(self, training_level):
        """
        Returns the level immediately above training_level in the pyramid.
        Returns None for the base case (no level above).
        """
        idx = self.levels.index(training_level)
        if idx == 0:
            return None   # base case
        return self.levels[idx - 1]

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
        Computes shaping signal for an agent at training_level.
        Uses V* at the upper (coarser) level in the pyramid.

        Formula: F = scale * (V_upper(s') - V_upper(s))
                 (NO gamma in this formula - empirically this telescopic
                 form works better than the canonical PBRS with gamma)
        """
        upper_level = self.get_upper_level(training_level)
        if upper_level is None:
            return 0.0   # base case, no shaping

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
