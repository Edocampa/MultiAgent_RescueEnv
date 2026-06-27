"""
option_policy.py - Continuous Gaussian policy + value critic for option training.

Two networks:
    OptionActor: outputs a 2D Gaussian distribution over actions.
        - mean is squashed by tanh into [-1, 1] (matching the env's clipping)
        - log_std is a learnable parameter (state-independent, common for PPO)
    OptionCritic: outputs a scalar value V(s) for advantage estimation.

Both networks use a small MLP (64x64 hidden) - the task is simple, no need to overparameterize.
"""

import torch
import torch.nn as nn
import numpy as np


class OptionActor(nn.Module):
    def __init__(self, obs_dim=18, action_dim=2, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # State-independent log_std (standard PPO trick)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        self._init_weights()

    def _init_weights(self):
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)

    def forward(self, obs):
        h = self.shared(obs)
        mean = torch.tanh(self.mean_head(h))   # squash to [-1, 1]
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, obs):
        """Sample an action from the Gaussian. Returns (action, log_prob)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        # Clip the sampled action to [-1, 1] (the env will clip too, but
        # this keeps the log_prob calculation consistent)
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1).squeeze()
        return action.squeeze(0).cpu().numpy(), log_prob

    def get_deterministic_action(self, obs):
        """Return the mean (no sampling), used for evaluation/visualization."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mean, _ = self.forward(obs)
        return mean.detach().squeeze(0).cpu().numpy()

    def evaluate_actions(self, obs, actions):
        """Used during PPO update: compute log_prob and entropy of given actions."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy


class OptionCritic(nn.Module):
    def __init__(self, obs_dim=18, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0)

    def forward(self, obs):
        return self.network(obs).squeeze(-1)
