"""
abstract_layer.py - Hierarchical abstraction for the new pyramid.

Adapted from v3 with two additions:
    - Supports arbitrary pyramid like [3, 6, 12, 25, 50, 100]
    - Has compute_shaping_continuous() for the L100 continuous level:
      takes a continuous (r, c) position, floors to cell, then computes
      shaping as usual using V*_upper.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "env_discrete"))
from map_generator import project_map, get_phi

"""
def compute_V_via_VI(goal_cell, blocked_cells, grid_size, gamma=0.80,
                     max_iters=500, tol=1e-6):
    Solve single-agent shortest-path MDP via Value Iteration.
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
    """

def compute_V_via_VI(goal_cell, blocked_cells, grid_size, gamma=0.80,
                     max_iters=500, tol=1e-6):
    """Solve single-agent shortest-path MDP via Value Iteration (vectorized).
    
    Numpy-vectorized version: ~100x faster than the pure-Python loop,
    produces identical output. Critical for the new pyramid because
    VI at L=100 with gamma=0.97 is otherwise the training bottleneck.
    """
    V = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    # Build mask of blocked cells
    blocked_mask = np.zeros((grid_size, grid_size), dtype=bool)
    if blocked_cells:
        rows, cols = zip(*blocked_cells)
        blocked_mask[list(rows), list(cols)] = True
    
    # Goal initialization (skip if goal is somehow blocked)
    if blocked_mask[goal_cell[0], goal_cell[1]]:
        return V
    V[goal_cell[0], goal_cell[1]] = 1.0

    for _ in range(max_iters):
        V_old = V.copy()
        # Zero out blocked cells before propagation (can't transition INTO a wall)
        V_unblocked = np.where(blocked_mask, 0.0, V_old)

        # Build 4 shifted views = value of each direction's neighbor.
        # Cells at the boundary see "0" (as if there's a wall outside).
        V_up = np.zeros_like(V_old)
        V_up[:-1, :] = V_unblocked[1:, :]    # value of cell BELOW (move up to reach goal)
        V_down = np.zeros_like(V_old)
        V_down[1:, :] = V_unblocked[:-1, :]  # value of cell ABOVE
        V_left = np.zeros_like(V_old)
        V_left[:, :-1] = V_unblocked[:, 1:]  # value of cell to the RIGHT
        V_right = np.zeros_like(V_old)
        V_right[:, 1:] = V_unblocked[:, :-1] # value of cell to the LEFT

        # Bellman max over 4 neighbors
        V_max = np.maximum.reduce([V_up, V_down, V_left, V_right])
        V_new = gamma * V_max

        # Enforce constraints: blocked stay 0, goal stays 1.0
        V_new[blocked_mask] = 0.0
        V_new[goal_cell[0], goal_cell[1]] = 1.0

        if np.max(np.abs(V_new - V_old)) < tol:
            V = V_new
            break
        V = V_new
    return V


class HierarchicalAbstraction:
    """Manages V* maps for an N-level pyramid (e.g. [3, 6, 12, 25, 50, 100])."""

    def __init__(self, levels, gamma_VI=0.80):
        self.levels = sorted(levels)
        self.gamma_VI = gamma_VI
        self.V_to_item = {L: None for L in self.levels}
        self.V_to_victim = {L: None for L in self.levels}

    def update(self, concrete_map):
        source_size = concrete_map["size"]
        assert source_size == self.levels[-1], \
            f"concrete_map size {source_size} must match top level {self.levels[-1]}"
        for level in self.levels:
            if level == source_size:
                level_map = concrete_map
            else:
                level_map = project_map(concrete_map, level)
            blocked = set(level_map["wall_cells"]) | set(level_map["fire_cells"])
            self.V_to_item[level] = compute_V_via_VI(
                level_map["item_pos"], blocked, level, self.gamma_VI)
            self.V_to_victim[level] = compute_V_via_VI(
                level_map["victim_pos"], blocked, level, self.gamma_VI)

    def get_upper_level(self, training_level):
        idx = self.levels.index(training_level)
        if idx == 0:
            return None
        return self.levels[idx - 1]

    def compute_shaping(self, pos, training_level, has_item,
                        differential=True, prev_pos=None,
                        prev_has_item=None, scale=1.0):
        """Discrete-level shaping (for L3-L50). pos is integer (r, c)."""
        upper_level = self.get_upper_level(training_level)
        if upper_level is None:
            return 0.0
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

        # ── Pickup transition guard ──────────────────────────────────
        # When has_item changes between steps, the V function used to
        # compute the potential switches (V_to_item -> V_to_victim).
        # The differential V_new - V_old then compares two DIFFERENT
        # potentials, producing a spurious negative signal that
        # PPO interprets as "pickup is bad". We return 0 here so the
        # transition is governed only by the env's pickup reward.
        if had_item != has_item:
            return 0.0


        V_prev = (self.V_to_victim[upper_level] if had_item
                  else self.V_to_item[upper_level])
        prev_upper = phi(*prev_pos)
        F_old = float(V_prev[prev_upper[0], prev_upper[1]])
        return scale * (F_new - F_old)

    """
        def compute_shaping(self, pos, training_level, has_item,
                        differential=True, prev_pos=None,
                        prev_has_item=None, scale=1.0):
        Discrete-level shaping (for L3-L50). pos is integer (r, c).
        upper_level = self.get_upper_level(training_level)
        if upper_level is None:
            return 0.0
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
    """

    def compute_shaping_continuous(self, pos_continuous, training_level,
                                    has_item, differential=True,
                                    prev_pos_continuous=None,
                                    prev_has_item=None, scale=1.0):
        """
        Continuous-level shaping (for L100). pos_continuous is np.array of floats.

        Floors the continuous position to a cell, then uses the standard
        cell-aligned projection to the upper level.
        """
        cell = (int(np.floor(pos_continuous[0])),
                int(np.floor(pos_continuous[1])))
        if prev_pos_continuous is not None:
            prev_cell = (int(np.floor(prev_pos_continuous[0])),
                         int(np.floor(prev_pos_continuous[1])))
        else:
            prev_cell = None
        return self.compute_shaping(
            cell, training_level, has_item,
            differential=differential,
            prev_pos=prev_cell, prev_has_item=prev_has_item,
            scale=scale,
        )

    def print_V(self, level, has_item=False):
        V = self.V_to_victim[level] if has_item else self.V_to_item[level]
        if V is None:
            print(f"V at level {level}: not computed")
            return
        target = "victim" if has_item else "item"
        print(f"V at level {level}, target = {target}:")
        for r in range(min(V.shape[0], 20)):
            print("  " + " ".join(f"{V[r, c]:.3f}" for c in range(min(V.shape[1], 20))))
