"""
continuous_world.py - Continuous-position simulator with cell-aligned walls.

GEOMETRY:
    World is [0, size] x [0, size] continuous. Walls and fires occupy entire
    cells (1x1 squares aligned with the integer grid). Agents have continuous
    (r, c) positions.

DYNAMICS:
    Each step takes a continuous action (delta_r, delta_c) and:
        1. Computes new position
        2. Clamps to world boundaries
        3. Checks collision with wall cells -> rejects movement (agent stays)
        4. Checks if new cell is a fire cell -> applies penalty
        5. Updates discrete state (item pickup, rescue)

CELL-LEVEL EVENTS:
    Item pickup: agent enters the cell containing the item
    Rescue: an item-holding agent's cell is Manhattan-adjacent (dist 1) to
            the victim's cell

OBSERVATION:
    Two observations per agent (for now we keep it minimal — when integrated
    with options, the OPTION will use its own local observation):
        Macro observation (for the MAPPO macro policy):
            10 base features + local OBSTACLE view (size*size grid of cells
            around the agent's CELL, e.g. 11x11). Plus continuous offset
            (r_frac, c_frac) = position inside current cell.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from continuous_map import generate_continuous_map


# Cell type constants (for the discrete obstacle grid)
EMPTY = 0
WALL  = 1
FIRE  = 2


class ContinuousWorld:
    """
    Continuous-position grid environment.

    Args:
        size:         grid dimension (default 100)
        num_fires:    number of fire cells
        wall_density: fraction of cells that are walls
        view_size:    local view radius (cells) for the agent's observation
        d_max:        max continuous movement per step
        seed:         RNG seed
    """

    def __init__(self, size=100, num_fires=15, wall_density=0.08,
                 view_size=11, d_max=0.2, seed=None):
        self.size = size
        self.num_fires = num_fires
        self.wall_density = wall_density
        self.view_size = view_size
        self.view_radius = view_size // 2
        self.d_max = d_max
        self._rng = np.random.RandomState(seed)

        # Observation dim: 10 base + view_size^2 local cells
        #self.obs_dim = 10 + view_size * view_size
        #self.global_dim = 10 + size * size

        # Observation dim: 8 base + 4 one-hot has_item + view_size^2 local cells
        # (was 10 base with has_item as 2 scalars; now 8 base + 4 one-hot)
        self.obs_dim = 10 + view_size * view_size
        self.global_dim = 10 + size * size

        # State (set in reset)
        self.cell_grid = None        # NxN int array (EMPTY/WALL/FIRE)
        self.wall_cells = None
        self.fire_cells = None
        self.agent_pos = [None, None]    # continuous positions
        self.item_pos = None
        self.victim_pos = None
        self.agent_has_item = [False, False]
        self.done = False
        self.outcome = "timeout"

    def reset(self, concrete_map=None):
        if concrete_map is None:
            concrete_map = generate_continuous_map(
                size=self.size,
                num_fires=self.num_fires,
                wall_density=self.wall_density,
                random_walls=True,
                rng=self._rng,
            )

        self.wall_cells = list(concrete_map["wall_cells"])
        self.fire_cells = list(concrete_map["fire_cells"])
        self.agent_pos[0] = np.array(concrete_map["agent_starts"][0],
                                      dtype=np.float32)
        self.agent_pos[1] = np.array(concrete_map["agent_starts"][1],
                                      dtype=np.float32)
        self.item_pos = np.array(concrete_map["item_pos"], dtype=np.float32)
        self.victim_pos = np.array(concrete_map["victim_pos"], dtype=np.float32)
        self.agent_has_item = [False, False]
        self.done = False
        self.outcome = "timeout"

        # Build the cell grid
        self.cell_grid = np.zeros((self.size, self.size), dtype=np.int32)
        for r, c in self.wall_cells:
            self.cell_grid[r, c] = WALL
        for r, c in self.fire_cells:
            self.cell_grid[r, c] = FIRE

        return self.get_observations()

    def cell_of(self, pos):
        """Returns the integer cell (i, j) containing the continuous position."""
        return (int(np.floor(pos[0])), int(np.floor(pos[1])))

    def step(self, actions):
        """
        Args:
            actions: list of 2 np.array, each (delta_r, delta_c) in [-1, 1]
                     (will be scaled by d_max)

        Returns:
            obs_list, reward_list, done, info
        """
        if self.done:
            return self.get_observations(), [0.0, 0.0], True, {"outcome": self.outcome}

        rewards = [0.0, 0.0]

        # Apply movement for each agent
        for k in range(2):
            action = np.clip(np.asarray(actions[k], dtype=np.float32), -1.0, 1.0)
            delta = action * self.d_max
            new_pos = self.agent_pos[k] + delta

            # Clamp to world boundaries
            new_pos[0] = np.clip(new_pos[0], 0.0, self.size - 1e-3)
            new_pos[1] = np.clip(new_pos[1], 0.0, self.size - 1e-3)

            # Collision with wall: reject the move
            new_cell = self.cell_of(new_pos)
            if self.cell_grid[new_cell[0], new_cell[1]] == WALL:
                rewards[k] += -0.3
                # Agent stays in place
                continue

            # Fire penalty if new cell is fire
            if self.cell_grid[new_cell[0], new_cell[1]] == FIRE:
                # Penalty only the FIRST step we enter, not every step inside
                old_cell = self.cell_of(self.agent_pos[k])
                if old_cell != new_cell:
                    rewards[k] += -5.0
                    rewards[1 - k] += -2.0

            self.agent_pos[k] = new_pos

        # Inter-agent collision (same cell) -> small penalty
        cell_0 = self.cell_of(self.agent_pos[0])
        cell_1 = self.cell_of(self.agent_pos[1])
        if cell_0 == cell_1:
            rewards[0] += -0.2
            rewards[1] += -0.2

        # Item pickup
        if not self.agent_has_item[0] and not self.agent_has_item[1]:
            item_cell = self.cell_of(self.item_pos)
            if cell_0 == item_cell:
                self.agent_has_item[0] = True
                rewards[0] += 5.0
                rewards[1] += 2.0
            elif cell_1 == item_cell:
                self.agent_has_item[1] = True
                rewards[1] += 5.0
                rewards[0] += 2.0

        # Rescue check
        if self.agent_has_item[0] or self.agent_has_item[1]:
            holder_cell = cell_0 if self.agent_has_item[0] else cell_1
            victim_cell = self.cell_of(self.victim_pos)
            if self._cells_adjacent(holder_cell, victim_cell):
                rewards[0] += 10.0
                rewards[1] += 10.0
                self.done = True
                self.outcome = "rescued"
                return self.get_observations(), rewards, True, {"outcome": "rescued"}

        return self.get_observations(), rewards, False, {"outcome": "ongoing"}

    def _cells_adjacent(self, a, b):
        """Manhattan distance 1 (including same cell)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1

    
    def get_observations(self):
        #Returns observations for both agents.

        #Each observation is 10 base features + view_size^2 local cells.
        
        N = float(self.size - 1)
        view_count = self.view_size * self.view_size

        obs0 = np.zeros(self.obs_dim, dtype=np.float32)
        obs1 = np.zeros(self.obs_dim, dtype=np.float32)

        # Base features (normalized continuous positions in [0, 1])
        for k, (obs, my_idx) in enumerate([(obs0, 0), (obs1, 1)]):
            mine = self.agent_pos[my_idx]
            other = self.agent_pos[1 - my_idx]
            obs[0] = mine[0] / N
            obs[1] = mine[1] / N
            obs[2] = other[0] / N
            obs[3] = other[1] / N
            obs[4] = self.item_pos[0] / N
            obs[5] = self.item_pos[1] / N
            obs[6] = self.victim_pos[0] / N
            obs[7] = self.victim_pos[1] / N
            obs[8] = float(self.agent_has_item[my_idx])
            obs[9] = float(self.agent_has_item[1 - my_idx])
            obs[10:10 + view_count] = self._local_view(self.cell_of(mine))

        return [obs0, obs1]
        
    
    # Function for one-hot encoding
    
    """def get_observations(self):
        Returns observations for both agents.

        Each observation is 12 base features + view_size^2 local cells.
        Base features (12 dim):
            [0:2]   my (r, c) normalized
            [2:4]   other's (r, c) normalized
            [4:6]   item (r, c) normalized
            [6:8]   victim (r, c) normalized
            [8:10]  my has_item one-hot: [1,0]=has, [0,1]=not
            [10:12] other has_item one-hot: [1,0]=has, [0,1]=not
        
        N = float(self.size - 1)
        view_count = self.view_size * self.view_size

        obs0 = np.zeros(self.obs_dim, dtype=np.float32)
        obs1 = np.zeros(self.obs_dim, dtype=np.float32)

        for k, (obs, my_idx) in enumerate([(obs0, 0), (obs1, 1)]):
            mine = self.agent_pos[my_idx]
            other = self.agent_pos[1 - my_idx]
            obs[0] = mine[0] / N
            obs[1] = mine[1] / N
            obs[2] = other[0] / N
            obs[3] = other[1] / N
            obs[4] = self.item_pos[0] / N
            obs[5] = self.item_pos[1] / N
            obs[6] = self.victim_pos[0] / N
            obs[7] = self.victim_pos[1] / N

            # One-hot encoding for my has_item
            if self.agent_has_item[my_idx]:
                obs[8] = 1.0     # has item
                obs[9] = 0.0
            else:
                obs[8] = 0.0
                obs[9] = 1.0     # does NOT have item

            # One-hot encoding for other's has_item
            if self.agent_has_item[1 - my_idx]:
                obs[10] = 1.0    # has item
                obs[11] = 0.0
            else:
                obs[10] = 0.0
                obs[11] = 1.0    # does NOT have item

            obs[12:12 + view_count] = self._local_view(self.cell_of(mine))

        return [obs0, obs1] """
    

    def _local_view(self, center_cell):
        """Returns a view_size x view_size flat array around center_cell."""
        view = np.zeros(self.view_size * self.view_size, dtype=np.float32)
        cr, cc = center_cell
        idx = 0
        radius = self.view_radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = cr + dr, cc + dc
                if r < 0 or r >= self.size or c < 0 or c >= self.size:
                    view[idx] = 1.0      # out of bounds = blocked
                elif self.cell_grid[r, c] == WALL:
                    view[idx] = 1.0
                elif self.cell_grid[r, c] == FIRE:
                    view[idx] = 0.5
                idx += 1
        return view
    
    
    def get_global_state(self):
        #Global state for centralized critic: continuous positions + full obstacle map.
        N = float(self.size - 1)
        gs = np.zeros(self.global_dim, dtype=np.float32)

        gs[0] = self.agent_pos[0][0] / N
        gs[1] = self.agent_pos[0][1] / N
        gs[2] = self.agent_pos[1][0] / N
        gs[3] = self.agent_pos[1][1] / N
        gs[4] = self.item_pos[0] / N
        gs[5] = self.item_pos[1] / N
        gs[6] = self.victim_pos[0] / N
        gs[7] = self.victim_pos[1] / N
        gs[8] = float(self.agent_has_item[0])
        gs[9] = float(self.agent_has_item[1])

        for r in range(self.size):
            for c in range(self.size):
                idx = 10 + r * self.size + c
                if self.cell_grid[r, c] == WALL:
                    gs[idx] = 1.0
                elif self.cell_grid[r, c] == FIRE:
                    gs[idx] = 0.5

        return gs
        
    
    
    # Function for one-hot encoding

    """def get_global_state(self):
        #Global state for centralized critic: continuous positions + one-hot has_item + full obstacle map.
        N = float(self.size - 1)
        gs = np.zeros(self.global_dim, dtype=np.float32)

        gs[0] = self.agent_pos[0][0] / N
        gs[1] = self.agent_pos[0][1] / N
        gs[2] = self.agent_pos[1][0] / N
        gs[3] = self.agent_pos[1][1] / N
        gs[4] = self.item_pos[0] / N
        gs[5] = self.item_pos[1] / N
        gs[6] = self.victim_pos[0] / N
        gs[7] = self.victim_pos[1] / N

        # One-hot has_item for agent 0
        if self.agent_has_item[0]:
            gs[8] = 1.0
            gs[9] = 0.0
        else:
            gs[8] = 0.0
            gs[9] = 1.0

        # One-hot has_item for agent 1
        if self.agent_has_item[1]:
            gs[10] = 1.0
            gs[11] = 0.0
        else:
            gs[10] = 0.0
            gs[11] = 1.0

        # Obstacle map shifted by 2 to accommodate the extra 2 dims
        for r in range(self.size):
            for c in range(self.size):
                idx = 12 + r * self.size + c
                if self.cell_grid[r, c] == WALL:
                    gs[idx] = 1.0
                elif self.cell_grid[r, c] == FIRE:
                    gs[idx] = 0.5

        return gs"""

    # ── Helper for the option dispatcher (sub-project 3) ──────────────

    def get_option_obs(self, agent_idx):
        """
        Returns the 18-dim observation that the fire-aware options expect:
            [0:2]    (r_rel, c_rel) inside current cell, in [0, 1]
            [2:10]   8 wall flags for the 8 neighbors
            [10:18]  8 fire flags for the 8 neighbors
        """
        pos = self.agent_pos[agent_idx]
        cell = self.cell_of(pos)
        r_rel = pos[0] - cell[0]
        c_rel = pos[1] - cell[1]

        neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),           (0, 1),
                            (1, -1),  (1, 0),  (1, 1)]
        obs = np.zeros(18, dtype=np.float32)
        obs[0] = r_rel
        obs[1] = c_rel
        for k, (dr, dc) in enumerate(neighbors_offsets):
            nr, nc = cell[0] + dr, cell[1] + dc
            if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
                obs[2 + k] = 1.0     # out of bounds = blocked (wall)
            elif self.cell_grid[nr, nc] == WALL:
                obs[2 + k] = 1.0
            elif self.cell_grid[nr, nc] == FIRE:
                obs[10 + k] = 1.0    # ← fire flag
        return obs
