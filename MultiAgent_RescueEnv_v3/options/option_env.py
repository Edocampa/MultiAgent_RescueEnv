"""
option_env.py - 3x3 continuous mini-environment for option training.

GEOMETRY:
    The world is a 3x3 grid of unit cells. Continuous coordinates (r, c)
    in [0, 3] x [0, 3]. Cell (i, j) covers [i, i+1) x [j, j+1).
    The agent starts in the CENTER cell (1, 1), at a random position
    (r_init, c_init) in [1, 2] x [1, 2].

DIRECTION OPTIONS:
    Each option specializes in one direction:
        "U" (Up):    target cell (0, 1), means row decreases
        "R" (Right): target cell (1, 2), means col increases
        "D" (Down):  target cell (2, 1), means row increases
        "L" (Left):  target cell (1, 0), means col decreases

REWARDS:
    +100  if the agent enters the target cell  -> terminal "success"
     -1   if the agent enters a non-target adjacent cell -> terminal "wrong_exit"
     -0.5 when a movement is rejected by a wall
     -0.05 per step (small step penalty to encourage decisive movement)

WALLS:
    Walls occupy entire cells (1x1 squares aligned with the grid).
    The center cell (1, 1) is ALWAYS free.
    The target cell is ALWAYS free (must be reachable).
    The 7 remaining cells are walls with probability `wall_prob`.

ACTIONS:
    Continuous 2D vector (dr, dc) in [-d_max, +d_max]^2 with d_max = 0.2.
    Each step the agent moves by this amount, subject to collision with walls
    and grid boundaries (clamped to [0, 3]).

OBSERVATION:
    10-dim vector:
        [0:2]  (r_rel, c_rel): position INSIDE the center cell, in [0, 1]
        [2:10] 8 booleans for walls in the 8 cells surrounding the center:
               (0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)
               (the center (1,1) is excluded since it's always free)
"""

import numpy as np


# Direction constants
DIRECTIONS = ["U", "R", "D", "L"]

# Target cell for each direction (in the 3x3 abstract grid)
TARGET_CELL = {
    "U": (0, 1),
    "R": (1, 2),
    "D": (2, 1),
    "L": (1, 0),
}

# Center cell and the 8 neighbors (order is fixed and matches the observation)
CENTER = (1, 1)
NEIGHBORS = [(0, 0), (0, 1), (0, 2),
             (1, 0),         (1, 2),
             (2, 0), (2, 1), (2, 2)]


class OptionEnv:
    """
    Continuous 3x3 mini-environment for training a single direction option.

    Args:
        direction:  "U", "R", "D", or "L" — the target direction this env trains
        d_max:      maximum movement per step (default 0.2)
        wall_prob:  probability that a non-essential cell is a wall
        max_steps:  episode timeout (default 50)
        seed:       RNG seed
    """

    def __init__(self, direction, d_max=0.2, wall_prob=0.0, fire_prob=0.0,
             max_steps=50, seed=None):
        assert direction in DIRECTIONS
        self.direction = direction
        self.target = TARGET_CELL[direction]
        self.d_max = d_max
        self.wall_prob = wall_prob
        self.fire_prob = fire_prob      
        self.max_steps = max_steps
        self._rng = np.random.RandomState(seed)
        self.pos = None
        self.walls = None
        self.fires = None                
        self.steps = 0

    def set_wall_prob(self, p):
        """Curriculum: change wall_prob between episodes."""
        self.wall_prob = float(p)
    
    def set_fire_prob(self, p):
        """Curriculum: change fire_prob between episodes."""
        self.fire_prob = float(p)

    def reset(self):
        # 1. Generate walls AND fires.
        # Walls can't be at center or target (target must always be reachable).
        # Fires can be anywhere except center (target CAN be fire, this teaches
        # the option how to handle "macro chose this direction but it's a fire").
        forbidden_walls = {CENTER, self.target}
        self.walls = set()
        self.fires = set()
        for cell in NEIGHBORS:
            if cell == CENTER:
                continue
            r = self._rng.rand()
            if cell in forbidden_walls:
                # Can be fire but not wall
                if r < self.fire_prob:
                    self.fires.add(cell)
            else:
                # Can be wall, fire, or empty
                if r < self.wall_prob:
                    self.walls.add(cell)
                elif r < self.wall_prob + self.fire_prob:
                    self.fires.add(cell)

        # 2. Random starting position inside the center cell
        margin = 0.05
        r_init = CENTER[0] + margin + self._rng.rand() * (1 - 2 * margin)
        c_init = CENTER[1] + margin + self._rng.rand() * (1 - 2 * margin)
        self.pos = np.array([r_init, c_init], dtype=np.float32)

        self.steps = 0
        return self._get_obs()

    def step(self, action):
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        delta = action * self.d_max
        new_pos = self.pos + delta
        reward = -0.05   # step penalty

        new_pos[0] = np.clip(new_pos[0], 0.0, 2.999)
        new_pos[1] = np.clip(new_pos[1], 0.0, 2.999)
        new_cell = (int(np.floor(new_pos[0])), int(np.floor(new_pos[1])))

        # Wall collision: reject movement
        if new_cell in self.walls:
            reward += -0.5
            new_pos = self.pos.copy()
            new_cell = (int(np.floor(new_pos[0])), int(np.floor(new_pos[1])))

        self.pos = new_pos

        # Direction-aware shaping (unchanged)
        if new_cell == CENTER:
            r_rel = self.pos[0] - CENTER[0]
            c_rel = self.pos[1] - CENTER[1]
            if self.direction == "R":
                reward += 0.1 * c_rel
            elif self.direction == "L":
                reward += 0.1 * (1.0 - c_rel)
            elif self.direction == "D":
                reward += 0.1 * r_rel
            elif self.direction == "U":
                reward += 0.1 * (1.0 - r_rel)

        # Termination logic with FIRE PENALTY
        done = False
        info = {"outcome": "ongoing"}

        if new_cell == self.target:
            # Target reached. If target is fire, big penalty applied.
            fire_penalty = -50.0 if new_cell in self.fires else 0.0
            reward += 100.0 + fire_penalty
            done = True
            info["outcome"] = "success_via_fire" if fire_penalty < 0 else "success"

        elif new_cell != CENTER:
            # Wrong exit. If also fire: even worse.
            fire_penalty = -50.0 if new_cell in self.fires else 0.0
            reward += -1.0 + fire_penalty
            done = True
            info["outcome"] = "wrong_fire" if fire_penalty < 0 else "wrong_exit"

        elif self.steps >= self.max_steps:
            done = True
            info["outcome"] = "timeout"

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """18-dim obs: 2 position + 8 wall flags + 8 fire flags."""
        obs = np.zeros(18, dtype=np.float32)
        obs[0] = self.pos[0] - CENTER[0]
        obs[1] = self.pos[1] - CENTER[1]
        for k, cell in enumerate(NEIGHBORS):
            obs[2 + k] = 1.0 if cell in self.walls else 0.0
            obs[10 + k] = 1.0 if cell in self.fires else 0.0
        return obs

    def render_ascii(self):
        """Quick ASCII visualization for debugging."""
        lines = []
        for i in range(3):
            row = []
            for j in range(3):
                if (i, j) in self.walls:
                    row.append("##")
                elif (i, j) == self.target:
                    row.append("TT")
                elif (i, j) == CENTER:
                    # Show approximate agent position if inside
                    ar = int(np.floor(self.pos[0]))
                    ac = int(np.floor(self.pos[1]))
                    if (ar, ac) == (i, j):
                        row.append(" A")
                    else:
                        row.append(" .")
                else:
                    row.append(" .")
            lines.append(" ".join(row))
        return "\n".join(lines)
