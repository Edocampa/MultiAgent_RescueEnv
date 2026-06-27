"""
continuous_pz.py - PettingZoo wrapper for the continuous environment.

For now this is a thin wrapper around ContinuousWorld. The action_space is
prepared for the future option-based macro policy (5 discrete actions:
U, R, D, L, NOP), but step() still expects CONTINUOUS actions because the
option dispatcher (sub-project 3) hasn't been built yet.

When the dispatcher is added, this wrapper will translate the discrete
macro_action into multiple continuous steps via the corresponding option.
"""

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

import sys
import os
sys.path.append(os.path.dirname(__file__))
from continuous_world import ContinuousWorld


class ContinuousRescueEnvPZ(ParallelEnv):
    metadata = {"render_modes": [None], "name": "continuous_rescue_v4"}

    def __init__(self, size=100, max_cycles=2000, num_fires=15,
                 wall_density=0.08, view_size=11, d_max=0.2, seed=None):
        self.size = size
        self.max_cycles = max_cycles
        self._step_count = 0

        self._env = ContinuousWorld(
            size=size, num_fires=num_fires, wall_density=wall_density,
            view_size=view_size, d_max=d_max, seed=seed,
        )

        self.possible_agents = ["agent_0", "agent_1"]
        self._observation_spaces = {
            a: spaces.Box(low=0.0, high=1.0,
                          shape=(self._env.obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        # Action space: continuous (delta_r, delta_c) in [-1, 1]^2
        # When integrated with options (sub-project 3), this will be replaced
        # by a Discrete(5) and the dispatcher will run the option internally.
        self._action_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for a in self.possible_agents
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None, concrete_map=None):
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
