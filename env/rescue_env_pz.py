"""
PettingZoo wrapper, parameterized by size

Accepts a pre-generated concrete_map at reset time
"""

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

import sys
import os
sys.path.append(os.path.dirname(__file__))
from grid_world import SimpleGridWorld


class RescueEnvPZ(ParallelEnv):
    metadata = {"render_modes": [None], "name": "rescue_v2"}

    def __init__(self, size=10, max_cycles=200, num_fires=2, random_walls=False,num_random_walls=8,seed=None):
        self.size = size
        self.max_cycles = max_cycles
        self._step_count = 0
        self._env = SimpleGridWorld(size=size, num_fires=num_fires,random_walls=random_walls, 
            num_random_walls=num_random_walls, seed=seed)

        self.possible_agents = ["agent_0", "agent_1"]
        self._observation_spaces = {
            a: spaces.Box(low=0.0, high=1.0,
                          shape=(self._env.obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._action_spaces = {
            a: spaces.Discrete(5)
            for a in self.possible_agents
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None, concrete_map=None):
        """
        Reset. If concrete_map is provided, the env uses that (projected if needed)
        """
        self.agents = self.possible_agents[:]
        self._step_count = 0
        obs_list = self._env.reset(concrete_map=concrete_map)
        observations = {"agent_0": obs_list[0], "agent_1": obs_list[1]}
        infos = {"agent_0": {}, "agent_1": {}}
        return observations, infos

    def step(self, actions):
        self._step_count += 1
        action_list = [actions["agent_0"], actions["agent_1"]]
        obs_list, reward_list, done, info = self._env.step(action_list)

        truncated = self._step_count >= self.max_cycles and not done

        observations = {"agent_0": obs_list[0], "agent_1": obs_list[1]}
        rewards = {"agent_0": reward_list[0], "agent_1": reward_list[1]}
        terminations = {"agent_0": done, "agent_1": done}
        truncations = {"agent_0": truncated, "agent_1": truncated}
        infos = {"agent_0": info, "agent_1": info}

        if done or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
