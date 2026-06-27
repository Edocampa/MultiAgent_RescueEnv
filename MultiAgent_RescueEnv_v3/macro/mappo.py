"""mappo.py - Multi-Agent PPO with optional dual learner (from v3)."""

import torch
import torch.nn as nn
import numpy as np
from networks import Actor, Critic
from buffer import RolloutBuffer


class MAPPO:
    def __init__(self, num_agents=2, obs_dim=35, global_dim=110, action_dim=5,
                 hidden_dim=128, lr_actor=3e-4, lr_critic=5e-4, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.2, epochs=10, batch_size=64,
                 entropy_coef=0.01, max_grad_norm=0.5, device="cpu",
                 dual_learner=False):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.dual_learner = dual_learner

        self.actors_biased = [Actor(obs_dim, action_dim, hidden_dim).to(device)
                               for _ in range(num_agents)]
        self.critic_biased = Critic(global_dim, hidden_dim).to(device)
        self.actor_opts_biased = [
            torch.optim.Adam(a.parameters(), lr=lr_actor, eps=1e-5)
            for a in self.actors_biased
        ]
        self.critic_opt_biased = torch.optim.Adam(
            self.critic_biased.parameters(), lr=lr_critic, eps=1e-5)

        if dual_learner:
            self.actors_unbiased = [Actor(obs_dim, action_dim, hidden_dim).to(device)
                                     for _ in range(num_agents)]
            self.critic_unbiased = Critic(global_dim, hidden_dim).to(device)
            self.actor_opts_unbiased = [
                torch.optim.Adam(a.parameters(), lr=lr_actor, eps=1e-5)
                for a in self.actors_unbiased
            ]
            self.critic_opt_unbiased = torch.optim.Adam(
                self.critic_unbiased.parameters(), lr=lr_critic, eps=1e-5)

        self.buffer_biased = RolloutBuffer(num_agents)
        if dual_learner:
            self.buffer_unbiased = RolloutBuffer(num_agents)

    def select_actions(self, obs_list, global_state):
        actions = []
        log_probs = []
        for i, actor in enumerate(self.actors_biased):
            obs_tensor = torch.FloatTensor(obs_list[i]).to(self.device)
            with torch.no_grad():
                a, lp = actor.get_action(obs_tensor)
            actions.append(a)
            log_probs.append(lp)
        gs = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v_biased = self.critic_biased(gs).item()
        if self.dual_learner:
            with torch.no_grad():
                v_unbiased = self.critic_unbiased(gs).item()
            return actions, log_probs, v_biased, v_unbiased
        return actions, log_probs, v_biased

    def add_to_buffer(self, obs, global_state, actions, log_probs,
                      rewards_biased, value_biased, done,
                      rewards_unbiased=None, value_unbiased=None):
        self.buffer_biased.add(obs, global_state, actions, log_probs,
                                rewards_biased, value_biased, done)
        if self.dual_learner:
            self.buffer_unbiased.add(obs, global_state, actions, log_probs,
                                      rewards_unbiased, value_unbiased, done)

    def update(self):
        m = {"biased": self._update_one(
            self.buffer_biased, self.actors_biased,
            self.actor_opts_biased, self.critic_biased,
            self.critic_opt_biased)}
        if self.dual_learner:
            m["unbiased"] = self._update_one(
                self.buffer_unbiased, self.actors_unbiased,
                self.actor_opts_unbiased, self.critic_unbiased,
                self.critic_opt_unbiased)
        return m

    def _update_one(self, buffer, actors, actor_opts, critic, critic_opt):
        if buffer.size < 2:
            buffer.clear()
            return {}
        if buffer.dones[-1]:
            last_value = 0.0
        else:
            last_gs = torch.FloatTensor(buffer.global_states[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_value = critic(last_gs).item()
        advantages, returns = buffer.compute_advantages(
            last_value, self.gamma, self.gae_lambda)
        data = buffer.get_tensors(advantages, returns, self.device)
        T = buffer.size
        metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        for _ in range(self.epochs):
            indices = np.random.permutation(T)
            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                idx = indices[start:end]
                b_obs = data["obs"][idx]
                b_gs = data["global_state"][idx]
                b_actions = data["actions"][idx]
                b_old_lp = data["old_log_probs"][idx]
                b_adv = data["advantages"][idx]
                b_ret = data["returns"][idx]
                for i, (actor, opt) in enumerate(zip(actors, actor_opts)):
                    new_lp, entropy = actor.evaluate_actions(b_obs[:, i, :],
                                                              b_actions[:, i])
                    adv_i = b_adv[:, i]
                    if len(adv_i) > 1:
                        adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)
                    ratio = torch.exp(new_lp - b_old_lp[:, i])
                    surr1 = ratio * adv_i
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                        1 + self.clip_eps) * adv_i
                    actor_loss = (-torch.min(surr1, surr2).mean()
                                 - self.entropy_coef * entropy.mean())
                    opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    opt.step()
                    metrics["actor_loss"] += actor_loss.item()
                    metrics["entropy"] += entropy.mean().item()
                pred = critic(b_gs).squeeze(-1)
                target = b_ret.mean(dim=1)
                critic_loss = nn.functional.mse_loss(pred, target)
                critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
                critic_opt.step()
                metrics["critic_loss"] += critic_loss.item()
        buffer.clear()
        n = max(1, self.epochs * max(1, T // self.batch_size))
        for k in metrics:
            metrics[k] /= n
        return metrics

    def save(self, path):
        ckpt = {
            "dual_learner": self.dual_learner,
            "obs_dim": self.obs_dim,
            "global_dim": self.global_dim,
            "actors_biased": [a.state_dict() for a in self.actors_biased],
            "critic_biased": self.critic_biased.state_dict(),
        }
        if self.dual_learner:
            ckpt["actors_unbiased"] = [a.state_dict() for a in self.actors_unbiased]
            ckpt["critic_unbiased"] = self.critic_unbiased.state_dict()
        torch.save(ckpt, path)
