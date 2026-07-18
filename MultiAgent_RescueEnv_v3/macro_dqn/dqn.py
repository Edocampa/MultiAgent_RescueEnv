"""
dqn.py — Shared-Q DQN agent for the multi-agent cooperative task.

Design choices:
    - SHARED Q-network: both agents use the same Q-net. Both their
      transitions go into the same replay buffer. This effectively
      doubles the sample rate and enforces policy consistency.
    - TARGET network: separate frozen copy of Q updated every N steps,
      used to compute the Bellman targets. Prevents "moving target"
      instability.
    - EPSILON-GREEDY exploration: linear decay from eps_start to eps_end
      over the first `eps_decay_steps` environment steps.
    - HUBER LOSS (smooth L1) instead of MSE: more robust to reward outliers.
      Given the reward variance in your task (fires -5, catastrophes -100+),
      this stabilizes training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from q_network import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Independent Q-Learning with SHARED Q-network across 2 agents.

    Args:
        obs_dim:               observation dim (131 without one-hot)
        action_dim:            5 (U, R, D, L, NOP)
        hidden_dim:            MLP hidden size
        lr:                    optimizer learning rate
        gamma:                 discount factor for TD target
        buffer_capacity:       max replay buffer size
        batch_size:            minibatch size for updates
        target_update_freq:    how often to sync target net (in env steps)
        eps_start / eps_end:   epsilon-greedy schedule endpoints
        eps_decay_steps:       linear anneal over this many env steps
        min_buffer_before_train: don't train until buffer has this many
        device:                'cuda' or 'cpu'
    """

    def __init__(
        self,
        obs_dim,
        action_dim=5,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        buffer_capacity=200_000,
        batch_size=128,
        target_update_freq=1000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=100_000,
        min_buffer_before_train=2000,
        max_grad_norm=10.0,
        device="cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.min_buffer_before_train = min_buffer_before_train
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Q-network and target network
        self.q_net = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)
        for p in self.target_q_net.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=lr, eps=1e-5
        )

        # Shared replay buffer for both agents
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim, device=device)

        # Counters
        self.env_steps = 0             # total env steps seen
        self.updates_done = 0          # total gradient updates

    def epsilon(self):
        """Linear decay from eps_start to eps_end."""
        frac = min(1.0, self.env_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, obs, deterministic=False):
        """
        Choose an action via epsilon-greedy.
            obs: numpy array of shape (obs_dim,)
        Returns: int in [0, action_dim)
        """
        eps = 0.0 if deterministic else self.epsilon()
        if np.random.rand() < eps:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q_values = self.q_net(obs_t)          # shape [1, action_dim]
            return int(q_values.argmax(dim=1).item())

    def add_transition(self, obs, action, reward, next_obs, done):
        """Store a single (s, a, r, s', done) tuple in the buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.env_steps += 1

    def update(self):
        """Perform ONE gradient step on a mini-batch. Returns metrics or None."""
        if len(self.buffer) < self.min_buffer_before_train:
            return None
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        obs      = batch["obs"]
        actions  = batch["actions"]
        rewards  = batch["rewards"]
        next_obs = batch["next_obs"]
        dones    = batch["dones"]

        # Current Q(s, a) — one scalar per transition
        q_all = self.q_net(obs)                                 # [B, A]
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1) # [B]

        # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_q_net(next_obs)                # [B, A]
            q_next_max = q_next.max(dim=1)[0]                   # [B]
            targets = rewards + self.gamma * q_next_max * (1.0 - dones)

        # Huber loss (smoother than MSE for large TD errors)
        loss = F.smooth_l1_loss(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.updates_done += 1

        # Periodic sync of target network
        if self.updates_done % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "target_mean": float(targets.mean().item()),
        }

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "env_steps": self.env_steps,
            "updates_done": self.updates_done,
        }, path)

    def load(self, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_q_net.load_state_dict(ckpt["target_q_net"])
        self.env_steps = ckpt.get("env_steps", 0)
        self.updates_done = ckpt.get("updates_done", 0)