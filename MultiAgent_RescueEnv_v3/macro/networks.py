"""networks.py - Actor and Critic networks (from v3, parametric dims)."""

import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
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
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)

    def forward(self, obs):
        return self.network(obs)

    def get_action(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze()

    def evaluate_actions(self, obs, actions):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


class Critic(nn.Module):
    def __init__(self, global_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0)

    def forward(self, global_state):
        return self.network(global_state)
