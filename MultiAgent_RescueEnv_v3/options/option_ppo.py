"""
option_ppo.py - Simple single-agent PPO trainer for an option.

This is a stripped-down version of MAPPO:
    - Single agent (not multi-agent, no shared/centralized state)
    - Single learner (no dual learner — option training doesn't use shaping)
    - Continuous actions (Gaussian policy)

Workflow per episode:
    1. Reset env, collect rollout
    2. Compute GAE advantages
    3. Run K epochs of PPO updates with minibatches
    4. Clear buffer, next episode
"""

import torch
import torch.nn as nn
import numpy as np
from option_policy import OptionActor, OptionCritic


class PPOBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    @property
    def size(self):
        return len(self.rewards)

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        lp = log_prob.detach().item() if torch.is_tensor(log_prob) else float(log_prob)
        self.log_probs.append(lp)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        T = self.size
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else self.values[t + 1]
            mask = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def to_tensors(self, advantages, returns, device):
        return {
            "obs":           torch.tensor(np.array(self.obs), device=device),
            "actions":       torch.tensor(np.array(self.actions), device=device),
            "old_log_probs": torch.tensor(np.array(self.log_probs, dtype=np.float32), device=device),
            "advantages":    torch.tensor(advantages, device=device),
            "returns":       torch.tensor(returns, device=device),
        }


class OptionPPO:
    def __init__(self, obs_dim=10, action_dim=2, hidden_dim=64,
                 lr_actor=3e-4, lr_critic=5e-4,
                 gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, epochs=4, batch_size=64,
                 entropy_coef=0.01, max_grad_norm=0.5,
                 device="cpu"):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.actor = OptionActor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = OptionCritic(obs_dim, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        self.buffer = PPOBuffer()

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(obs_t)
            value = self.critic(obs_t.unsqueeze(0)).item()
        return action, log_prob, value

    def add_to_buffer(self, obs, action, log_prob, reward, value, done):
        self.buffer.add(obs, action, log_prob, reward, value, done)

    def update(self):
        if self.buffer.size < 2:
            self.buffer.clear()
            return {}

        # Compute last value (0 if last step was terminal)
        if self.buffer.dones[-1]:
            last_value = 0.0
        else:
            last_obs = torch.tensor(self.buffer.obs[-1], device=self.device).unsqueeze(0)
            with torch.no_grad():
                last_value = self.critic(last_obs).item()

        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )
        data = self.buffer.to_tensors(advantages, returns, self.device)

        T = self.buffer.size
        metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.epochs):
            idx = np.random.permutation(T)
            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                bi = idx[start:end]

                b_obs = data["obs"][bi]
                b_act = data["actions"][bi]
                b_old_lp = data["old_log_probs"][bi]
                b_adv = data["advantages"][bi]
                b_ret = data["returns"][bi]

                # Normalize advantages within the batch
                if len(b_adv) > 1:
                    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                # Actor loss
                new_lp, entropy = self.actor.evaluate_actions(b_obs, b_act)
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coef * entropy.mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                # Critic loss
                pred = self.critic(b_obs)
                critic_loss = nn.functional.mse_loss(pred, b_ret)
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                metrics["actor_loss"] += actor_loss.item()
                metrics["critic_loss"] += critic_loss.item()
                metrics["entropy"] += entropy.mean().item()
                n_updates += 1

        self.buffer.clear()
        for k in metrics:
            metrics[k] /= max(1, n_updates)
        return metrics

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
