"""
q_network.py — Q-network for DQN.

Architecture identical shape to your MAPPO Actor, so the comparison
PPO vs DQN is fair. The only difference is the output:
    Actor (PPO): outputs LOGITS over 5 actions (for a categorical distribution)
    Q-network (DQN): outputs Q-VALUES for 5 actions (one scalar per action)

Same MLP: obs_dim → 128 → 128 → 5
Same initialization (orthogonal).
"""

import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim=5, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        # Last layer: small gain to keep initial Q-values near zero
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)

    def forward(self, obs):
        """Returns Q(s, a) for all actions a. Shape: [batch, action_dim]."""
        return self.network(obs)