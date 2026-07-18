"""
grid_world.py - Parametric grid environment (from v3).
Used for discrete pyramid levels 3, 6, 12, 25, 50.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from map_generator import generate_concrete_map, project_map


EMPTY, WALL, FIRE, ITEM, VICTIM = 0, 1, 2, 3, 4


def get_view_size_for(grid_size, max_view=11):
    return min(max_view, grid_size)


def get_obs_dim(size, view_size=None):
    if view_size is None:
        view_size = get_view_size_for(size)
    view_size = min(view_size, size)
    # 8 positions + 4 one-hot has_item + view_size^2 local view
    return 12 + view_size * view_size, view_size


def get_global_dim(size):
    # 8 positions + 4 one-hot has_item + size^2 obstacle map
    return 12 + size * size


class SimpleGridWorld:
    def __init__(self, size, num_fires=2, wall_density=0.10,
                 random_walls=True, view_size=None, seed=None):
        self.size = size
        self.num_fires = num_fires
        self.wall_density = wall_density
        self.random_walls = random_walls
        self._rng = np.random.RandomState(seed)
        self.view_size = view_size if view_size else get_view_size_for(size)
        self.view_size = min(self.view_size, size)
        self.view_radius = self.view_size // 2
        self.obs_dim = 12 + self.view_size * self.view_size
        self.global_dim = get_global_dim(size)
        self.grid = None
        self.wall_cells = []
        self.fire_cells = []
        self.agent1_pos = None
        self.agent2_pos = None
        self.item_pos = None
        self.victim_pos = None
        self.agent1_has_item = False
        self.agent2_has_item = False
        self.done = False
        self.outcome = "timeout"
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

    def reset(self, concrete_map=None):
        if concrete_map is None:
            map_data = generate_concrete_map(
                size=self.size, num_fires=self.num_fires,
                wall_density=self.wall_density,
                random_walls=self.random_walls, rng=self._rng,
            )
        else:
            if concrete_map["size"] == self.size:
                map_data = concrete_map
            else:
                map_data = project_map(concrete_map, self.size)
        self.wall_cells = list(map_data["wall_cells"])
        self.fire_cells = list(map_data["fire_cells"])
        self.agent1_pos = tuple(map_data["agent_starts"][0])
        self.agent2_pos = tuple(map_data["agent_starts"][1])
        self.item_pos = tuple(map_data["item_pos"])
        self.victim_pos = tuple(map_data["victim_pos"])
        self.agent1_has_item = False
        self.agent2_has_item = False
        self.done = False
        self.outcome = "timeout"
        if self.agent1_pos == self.item_pos:
            self.agent1_has_item = True
        elif self.agent2_pos == self.item_pos:
            self.agent2_has_item = True
        self._build_grid()
        return self.get_observations()

    def _build_grid(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        for r, c in self.wall_cells:
            self.grid[r, c] = WALL
        for r, c in self.fire_cells:
            self.grid[r, c] = FIRE
        if not self.agent1_has_item and not self.agent2_has_item:
            self.grid[self.item_pos[0], self.item_pos[1]] = ITEM
        self.grid[self.victim_pos[0], self.victim_pos[1]] = VICTIM

    def step(self, actions):
        if self.done:
            return self.get_observations(), [0.0, 0.0], True, {"outcome": self.outcome}
        a1, a2 = actions
        r1, r2 = 0.0, 0.0
        new1 = self._try_move(self.agent1_pos, a1)
        if new1 == self.agent1_pos and a1 != 4:
            r1 -= 0.3
        self.agent1_pos = new1
        new2 = self._try_move(self.agent2_pos, a2)
        if new2 == self.agent2_pos and a2 != 4:
            r2 -= 0.3
        self.agent2_pos = new2
        if self.agent1_pos == self.agent2_pos:
            r1 -= 0.2
            r2 -= 0.2
        if self.agent1_pos in self.fire_cells:
            r1 -= 5.0
            r2 -= 2.0
        if self.agent2_pos in self.fire_cells:
            r2 -= 5.0
            r1 -= 2.0
        if not self.agent1_has_item and not self.agent2_has_item:
            if self.agent1_pos == self.item_pos:
                self.agent1_has_item = True
                r1 += 5.0
                r2 += 2.0
            elif self.agent2_pos == self.item_pos:
                self.agent2_has_item = True
                r2 += 5.0
                r1 += 2.0
        has_item = self.agent1_has_item or self.agent2_has_item
        if has_item:
            holder = self.agent1_pos if self.agent1_has_item else self.agent2_pos
            if self._is_adjacent(holder, self.victim_pos):
                r1 += 10.0
                r2 += 10.0
                self.done = True
                self.outcome = "rescued"
                return self.get_observations(), [r1, r2], True, {"outcome": "rescued"}
        self._build_grid()
        return self.get_observations(), [r1, r2], False, {"outcome": "ongoing"}

    def _try_move(self, pos, action):
        dr, dc = self.action_map[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
            return pos
        if self.grid[nr, nc] == WALL:
            return pos
        return (nr, nc)

    def _is_adjacent(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1
    
    """

    def get_observations(self):
        N = max(1, self.size - 1)
        view_count = self.view_size * self.view_size
        obs0 = np.zeros(self.obs_dim, dtype=np.float32)
        obs1 = np.zeros(self.obs_dim, dtype=np.float32)
        obs0[0] = self.agent1_pos[0] / N
        obs0[1] = self.agent1_pos[1] / N
        obs0[2] = self.agent2_pos[0] / N
        obs0[3] = self.agent2_pos[1] / N
        obs0[4] = self.item_pos[0] / N
        obs0[5] = self.item_pos[1] / N
        obs0[6] = self.victim_pos[0] / N
        obs0[7] = self.victim_pos[1] / N
        obs0[8] = float(self.agent1_has_item)
        obs0[9] = float(self.agent2_has_item)
        obs0[10:10 + view_count] = self._local_view(self.agent1_pos)
        obs1[0] = self.agent2_pos[0] / N
        obs1[1] = self.agent2_pos[1] / N
        obs1[2] = self.agent1_pos[0] / N
        obs1[3] = self.agent1_pos[1] / N
        obs1[4] = self.item_pos[0] / N
        obs1[5] = self.item_pos[1] / N
        obs1[6] = self.victim_pos[0] / N
        obs1[7] = self.victim_pos[1] / N
        obs1[8] = float(self.agent2_has_item)
        obs1[9] = float(self.agent1_has_item)
        obs1[10:10 + view_count] = self._local_view(self.agent2_pos)
        return [obs0, obs1]
        """
    
    # Function for one-hot encoding

    def get_observations(self):
        N = max(1, self.size - 1)
        view_count = self.view_size * self.view_size
        obs0 = np.zeros(self.obs_dim, dtype=np.float32)
        obs1 = np.zeros(self.obs_dim, dtype=np.float32)

        # Agent 0's observation (egocentric: "mine" first, "other" second)
        obs0[0] = self.agent1_pos[0] / N
        obs0[1] = self.agent1_pos[1] / N
        obs0[2] = self.agent2_pos[0] / N
        obs0[3] = self.agent2_pos[1] / N
        obs0[4] = self.item_pos[0] / N
        obs0[5] = self.item_pos[1] / N
        obs0[6] = self.victim_pos[0] / N
        obs0[7] = self.victim_pos[1] / N
        # My has_item one-hot
        if self.agent1_has_item:
            obs0[8], obs0[9] = 1.0, 0.0
        else:
            obs0[8], obs0[9] = 0.0, 1.0
        # Other has_item one-hot
        if self.agent2_has_item:
            obs0[10], obs0[11] = 1.0, 0.0
        else:
            obs0[10], obs0[11] = 0.0, 1.0
        obs0[12:12 + view_count] = self._local_view(self.agent1_pos)

        # Agent 1's observation
        obs1[0] = self.agent2_pos[0] / N
        obs1[1] = self.agent2_pos[1] / N
        obs1[2] = self.agent1_pos[0] / N
        obs1[3] = self.agent1_pos[1] / N
        obs1[4] = self.item_pos[0] / N
        obs1[5] = self.item_pos[1] / N
        obs1[6] = self.victim_pos[0] / N
        obs1[7] = self.victim_pos[1] / N
        # My has_item one-hot
        if self.agent2_has_item:
            obs1[8], obs1[9] = 1.0, 0.0
        else:
            obs1[8], obs1[9] = 0.0, 1.0
        # Other has_item one-hot
        if self.agent1_has_item:
            obs1[10], obs1[11] = 1.0, 0.0
        else:
            obs1[10], obs1[11] = 0.0, 1.0
        obs1[12:12 + view_count] = self._local_view(self.agent2_pos)

        return [obs0, obs1]

    def _local_view(self, center):
        view = np.zeros(self.view_size * self.view_size, dtype=np.float32)
        cr, cc = center
        idx = 0
        radius = self.view_radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = cr + dr, cc + dc
                if r < 0 or r >= self.size or c < 0 or c >= self.size:
                    view[idx] = 1.0
                elif self.grid[r, c] == WALL:
                    view[idx] = 1.0
                elif self.grid[r, c] == FIRE:
                    view[idx] = 0.5
                idx += 1
        return view
    
    """

    def get_global_state(self):
        N = max(1, self.size - 1)
        gs = np.zeros(self.global_dim, dtype=np.float32)
        gs[0] = self.agent1_pos[0] / N
        gs[1] = self.agent1_pos[1] / N
        gs[2] = self.agent2_pos[0] / N
        gs[3] = self.agent2_pos[1] / N
        gs[4] = self.item_pos[0] / N
        gs[5] = self.item_pos[1] / N
        gs[6] = self.victim_pos[0] / N
        gs[7] = self.victim_pos[1] / N
        gs[8] = float(self.agent1_has_item)
        gs[9] = float(self.agent2_has_item)
        for r in range(self.size):
            for c in range(self.size):
                idx = 10 + r * self.size + c
                if self.grid[r, c] == WALL:
                    gs[idx] = 1.0
                elif self.grid[r, c] == FIRE:
                    gs[idx] = 0.5
        return gs
        """
    
     # Function for one-hot encoding

    def get_global_state(self):
        N = max(1, self.size - 1)
        gs = np.zeros(self.global_dim, dtype=np.float32)
        gs[0] = self.agent1_pos[0] / N
        gs[1] = self.agent1_pos[1] / N
        gs[2] = self.agent2_pos[0] / N
        gs[3] = self.agent2_pos[1] / N
        gs[4] = self.item_pos[0] / N
        gs[5] = self.item_pos[1] / N
        gs[6] = self.victim_pos[0] / N
        gs[7] = self.victim_pos[1] / N
        # Agent 0 has_item one-hot
        if self.agent1_has_item:
            gs[8], gs[9] = 1.0, 0.0
        else:
            gs[8], gs[9] = 0.0, 1.0
        # Agent 1 has_item one-hot
        if self.agent2_has_item:
            gs[10], gs[11] = 1.0, 0.0
        else:
            gs[10], gs[11] = 0.0, 1.0
        # Obstacle map (shifted by 2 from before)
        for r in range(self.size):
            for c in range(self.size):
                idx = 12 + r * self.size + c
                if self.grid[r, c] == WALL:
                    gs[idx] = 1.0
                elif self.grid[r, c] == FIRE:
                    gs[idx] = 0.5
        return gs
