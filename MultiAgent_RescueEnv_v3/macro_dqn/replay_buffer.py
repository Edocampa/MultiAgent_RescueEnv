"""
replay_buffer.py — Experience replay buffer for DQN.

Stores transitions (s, a, r, s', done) and samples random mini-batches.
This is THE key mechanism that makes DQN off-policy: it allows the
same transition to be used for many gradient updates.

Uses a circular buffer: when full, oldest transitions are overwritten.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Simple circular buffer. Stores each transition as separate arrays
    (SoA layout) for efficient sampling with numpy.

    Since we share the Q-net between agents, we store transitions from
    BOTH agents in the same buffer. Effectively 2x sample rate.
    """

    def __init__(self, capacity, obs_dim, device="cpu"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = device

        # Pre-allocate all arrays for speed
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros(capacity,            dtype=np.int64)
        self.rewards  = np.zeros(capacity,            dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones    = np.zeros(capacity,            dtype=np.float32)

        self.pos = 0        # next insertion index
        self.size = 0       # how many transitions currently stored

    def add(self, obs, action, reward, next_obs, done):
        """Add a single transition. Overwrites oldest when full."""
        self.obs[self.pos]      = obs
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos]    = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a random mini-batch of transitions."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.from_numpy(self.obs[idx]).to(self.device),
            "actions":  torch.from_numpy(self.actions[idx]).to(self.device),
            "rewards":  torch.from_numpy(self.rewards[idx]).to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(self.device),
            "dones":    torch.from_numpy(self.dones[idx]).to(self.device),
        }

    def __len__(self):
        return self.size