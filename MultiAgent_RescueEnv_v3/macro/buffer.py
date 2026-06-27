"""buffer.py - Rollout buffer with GAE (from v3)."""

import torch
import numpy as np


class RolloutBuffer:
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.clear()

    def clear(self):
        self.observations = []
        self.global_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    @property
    def size(self):
        return len(self.rewards)

    def add(self, obs, global_state, actions, log_probs, rewards, value, done):
        self.observations.append(np.array(obs, dtype=np.float32))
        self.global_states.append(np.array(global_state, dtype=np.float32))
        self.actions.append(np.array(actions, dtype=np.int64))
        lp = np.array([lp.detach().item() if torch.is_tensor(lp) else lp
                        for lp in log_probs], dtype=np.float32)
        self.log_probs.append(lp)
        self.rewards.append(np.array(rewards, dtype=np.float32))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def compute_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        T = self.size
        N = self.num_agents
        advantages = np.zeros((T, N), dtype=np.float32)
        returns = np.zeros((T, N), dtype=np.float32)
        for agent_idx in range(N):
            last_gae = 0.0
            for t in reversed(range(T)):
                next_value = last_value if t == T - 1 else self.values[t + 1]
                mask = 0.0 if self.dones[t] else 1.0
                delta = (self.rewards[t][agent_idx]
                         + gamma * next_value * mask
                         - self.values[t])
                last_gae = delta + gamma * gae_lambda * mask * last_gae
                advantages[t, agent_idx] = last_gae
            returns[:, agent_idx] = advantages[:, agent_idx] + np.array(self.values)
        return advantages, returns

    def get_tensors(self, advantages, returns, device="cpu"):
        return {
            "obs": torch.FloatTensor(np.array(self.observations)).to(device),
            "global_state": torch.FloatTensor(np.array(self.global_states)).to(device),
            "actions": torch.LongTensor(np.array(self.actions)).to(device),
            "old_log_probs": torch.FloatTensor(np.array(self.log_probs)).to(device),
            "advantages": torch.FloatTensor(advantages).to(device),
            "returns": torch.FloatTensor(returns).to(device),
        }
