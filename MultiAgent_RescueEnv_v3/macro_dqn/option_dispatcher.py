"""
option_dispatcher.py - Macro-level wrapper for continuous env with options.

USAGE:
    The macro-policy chooses a DISCRETE action in {0:U, 1:R, 2:D, 3:L, 4:NOP}
    for each agent. The dispatcher then:
        1. For each agent, loads the corresponding pre-trained option policy
        2. Runs continuous micro-steps until both agents exit their cells
           (or option timeout reached)
        3. Accumulates rewards and returns the resulting state

This is a Semi-MDP wrapper: one "macro step" = many continuous "micro steps".

INTEGRATION WITH PYRAMID:
    The continuous env operates at level 100. The macro_action's effect is
    to move the agent toward a different cell. Shaping comes from V*_50
    (the level above in the pyramid).
"""

import sys
import os
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "continuous_env"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "options"))

from continuous_world import ContinuousWorld
from option_policy import OptionActor


# Macro action indices
ACTION_U = 0
ACTION_R = 1
ACTION_D = 2
ACTION_L = 3
ACTION_NOP = 4
ACTION_NAMES = {0: "U", 1: "R", 2: "D", 3: "L", 4: "NOP"}
ACTION_DIRS = {0: "U", 1: "R", 2: "D", 3: "L"}


class OptionDispatcher:
    """
    Wraps a ContinuousWorld and provides a discrete-action interface
    powered by pre-trained options.

    Args:
        continuous_env:  a ContinuousWorld instance
        options_dir:     directory containing pi_U.pt, pi_R.pt, pi_D.pt, pi_L.pt
        option_timeout:  max micro-steps per option execution (default 30)
        device:          torch device for option inference
    """

    def __init__(self, continuous_env, options_dir, option_timeout=30,
                 device="cpu"):
        self.env = continuous_env
        self.option_timeout = option_timeout
        self.device = device

        # Load the 4 pre-trained option policies
        self.option_actors = {}
        for direction in ["U", "R", "D", "L"]:
            ckpt_path = os.path.join(options_dir, f"pi_{direction}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Option checkpoint not found: {ckpt_path}. "
                    f"Train the options first via train_options.py."
                )
            actor = OptionActor(obs_dim=18, action_dim=2, hidden_dim=64).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            actor.load_state_dict(ckpt["actor"])
            actor.eval()
            self.option_actors[direction] = actor

        # State tracking
        self.last_macro_outcomes = None

    def reset(self, concrete_map=None):
        """Reset the underlying continuous env."""
        return self.env.reset(concrete_map=concrete_map)

    def step(self, macro_actions):
        """
        Executes one MACRO step: each agent runs the option corresponding to
        its macro_action until exiting its current cell or timing out.

        Args:
            macro_actions: list of 2 ints in {0,1,2,3,4} (action indices)

        Returns:
            obs_list, reward_list, done, info
                - reward_list: accumulated rewards over all micro-steps
                - info["macro_outcomes"]: list of 2 strings, "exited" or "timeout"
                - info["micro_steps"]: number of micro-steps executed
        """
        # Snapshot starting cells (option terminates when agent exits THIS cell)
        start_cells = [self.env.cell_of(self.env.agent_pos[k]) for k in range(2)]

        accumulated_rewards = [0.0, 0.0]
        macro_outcomes = ["exited", "exited"]
        agent_done = [False, False]   # has this agent finished its option?

        # Handle NOP: agent stays still during this macro step
        for k in range(2):
            if macro_actions[k] == ACTION_NOP:
                agent_done[k] = True
                macro_outcomes[k] = "nop"

        # Run micro-steps until both agents finish (or env terminates)
        micro_step = 0
        done = False
        info = {}

        while micro_step < self.option_timeout and not all(agent_done) and not done:
            micro_step += 1
            actions = [np.zeros(2, dtype=np.float32),
                       np.zeros(2, dtype=np.float32)]

            for k in range(2):
                if agent_done[k]:
                    continue
                direction = ACTION_DIRS[macro_actions[k]]
                actor = self.option_actors[direction]
                opt_obs = self.env.get_option_obs(k)
                with torch.no_grad():
                    opt_obs_t = torch.tensor(opt_obs, dtype=torch.float32,
                                              device=self.device)
                    delta = actor.get_deterministic_action(opt_obs_t)
                actions[k] = delta

            obs_list, rewards, env_done, info = self.env.step(actions)
            accumulated_rewards[0] += rewards[0]
            accumulated_rewards[1] += rewards[1]

            if env_done:
                done = True
                break

            # Check if each agent has exited its starting cell
            for k in range(2):
                if agent_done[k]:
                    continue
                current_cell = self.env.cell_of(self.env.agent_pos[k])
                if current_cell != start_cells[k]:
                    agent_done[k] = True

        # Handle timeouts (agents that haven't exited by now)
        for k in range(2):
            if not agent_done[k]:
                macro_outcomes[k] = "timeout"

        info["macro_outcomes"] = macro_outcomes
        info["micro_steps"] = micro_step

        return self.env.get_observations(), accumulated_rewards, done, info
    
    """
    def get_global_state_compact(self):
        
        Returns a compact global state for the centralized critic.
        Avoids the full 10010-dim obstacle map. Instead:
            10 base features (positions, has_item flags)
            121 local view around agent 0 (11x11)
            121 local view around agent 1 (11x11)
        Total: 252 dims.
        
        N = float(self.env.size - 1)
        gs = np.zeros(252, dtype=np.float32)
        gs[0] = self.env.agent_pos[0][0] / N
        gs[1] = self.env.agent_pos[0][1] / N
        gs[2] = self.env.agent_pos[1][0] / N
        gs[3] = self.env.agent_pos[1][1] / N
        gs[4] = self.env.item_pos[0] / N
        gs[5] = self.env.item_pos[1] / N
        gs[6] = self.env.victim_pos[0] / N
        gs[7] = self.env.victim_pos[1] / N
        gs[8] = float(self.env.agent_has_item[0])
        gs[9] = float(self.env.agent_has_item[1])
        gs[10:131] = self.env._local_view(self.env.cell_of(self.env.agent_pos[0]))
        gs[131:252] = self.env._local_view(self.env.cell_of(self.env.agent_pos[1]))
        return gs
        """
    
    # Function for one-hot encoding
    
    def get_global_state_compact(self):
        
        """Returns a compact global state for the centralized critic.
        Avoids the full obstacle map. Instead:
            8 base features (positions)
            4 one-hot has_item (2 per agent)
            121 local view around agent 0 (11x11)
            121 local view around agent 1 (11x11)
        Total: 254 dims."""
        
        N = float(self.env.size - 1)
        gs = np.zeros(254, dtype=np.float32)
        gs[0] = self.env.agent_pos[0][0] / N
        gs[1] = self.env.agent_pos[0][1] / N
        gs[2] = self.env.agent_pos[1][0] / N
        gs[3] = self.env.agent_pos[1][1] / N
        gs[4] = self.env.item_pos[0] / N
        gs[5] = self.env.item_pos[1] / N
        gs[6] = self.env.victim_pos[0] / N
        gs[7] = self.env.victim_pos[1] / N

        # One-hot has_item for agent 0
        if self.env.agent_has_item[0]:
            gs[8] = 1.0
            gs[9] = 0.0
        else:
            gs[8] = 0.0
            gs[9] = 1.0

        # One-hot has_item for agent 1
        if self.env.agent_has_item[1]:
            gs[10] = 1.0
            gs[11] = 0.0
        else:
            gs[10] = 0.0
            gs[11] = 1.0

        gs[12:133] = self.env._local_view(self.env.cell_of(self.env.agent_pos[0]))
        gs[133:254] = self.env._local_view(self.env.cell_of(self.env.agent_pos[1]))
        return gs

    @property
    def obs_dim(self):
        return self.env.obs_dim

    @property
    def global_dim(self):
        return 254 # it was 252 without one-hot encoding 
        
    
