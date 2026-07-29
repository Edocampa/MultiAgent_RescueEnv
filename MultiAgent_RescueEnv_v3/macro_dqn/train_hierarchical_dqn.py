"""
train_hierarchical_dqn.py — Training orchestrator for DQN (Exp 11 only).

Kept parallel to macro/train_hierarchical.py for clean comparison.
Only implements run_only_exp11 (L100 hier). If we want L50 hier DQN
sanity check later, we can add run_only_exp10.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "env_discrete"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "continuous_env"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "options"))

from map_generator import generate_concrete_map, project_map
from continuous_world import ContinuousWorld
from continuous_map import generate_continuous_map
from option_dispatcher import OptionDispatcher

from dqn import DQNAgent
from abstract_layer import HierarchicalAbstraction
from config import (ENV, DQN_BASE, HIERARCHY, TRAIN, CONVERGENCE,
                    LEVELS, CONTINUOUS_LEVEL)


def generate_map_pool(num_maps, base_seed):
    rngs = []
    for k in range(num_maps):
        rngs.append(np.random.RandomState(base_seed * 100000 + k))
    return rngs


def train_exp11_dqn(seed, save_dir, max_wall_time_seconds=None):
    """
    Train Exp 11 (L100 hier with V*_50 shaping) using shared-Q DQN.
    """
    os.makedirs(save_dir, exist_ok=True)
    level = 100
    num_episodes = TRAIN["episodes_per_level"][level]
    max_steps = ENV["max_steps_per_level"][level]
    view_size = ENV["view_size_per_level"][level]
    num_fires = ENV["num_fires_per_level"][level]

    print(f"\n{'═' * 72}")
    print(f"  EXPERIMENT 11 (DQN): L{level} - HIERARCHICAL (V*_50)")
    print(f"  Seed: {seed} | Episodes: {num_episodes} | Max macro-steps: {max_steps}")
    print(f"  Algorithm: SHARED-Q DQN with replay buffer")
    print(f"  Buffer: {DQN_BASE['buffer_capacity']} | Batch: {DQN_BASE['batch_size']}")
    print(f"  Eps: {DQN_BASE['eps_start']}->{DQN_BASE['eps_end']} in {DQN_BASE['eps_decay_steps']} steps")
    if max_wall_time_seconds:
        print(f"  Wall time budget: {max_wall_time_seconds}s ({max_wall_time_seconds/3600:.1f}h)")
    print(f"{'═' * 72}\n")

    torch.manual_seed(seed + 11)
    np.random.seed(seed + 11)

    # Env + dispatcher
    cont_env = ContinuousWorld(
        size=level, num_fires=num_fires,
        wall_density=ENV["wall_density"],
        view_size=view_size, d_max=ENV["d_max"],
        seed=seed + 11,
    )
    options_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ENV["options_dir"]))
    dispatcher = OptionDispatcher(
        cont_env, options_dir=options_dir,
        option_timeout=ENV["option_timeout"], device=DQN_BASE["device"],
    )

    # DQN agent (shared between two players)
    obs_dim = dispatcher.obs_dim
    agent = DQNAgent(obs_dim=obs_dim, **DQN_BASE)

    # Abstraction for shaping
    abstraction = HierarchicalAbstraction(
        levels=LEVELS, gamma_VI=HIERARCHY["gamma_VI"]
    )
    map_rngs = generate_map_pool(num_episodes, seed)

    diff_shaping = HIERARCHY["differential_shaping"]
    scale = HIERARCHY["shaping_scale"]
    updates_per_env_step = TRAIN["updates_per_env_step"]
    log_every = TRAIN["log_every"]

    rewards_hist = []
    rewards_biased_hist = []
    sr_window = []
    sr_log = []
    total_successes = 0
    best_sr = 0.0 

    wall_start = time.time()

    for episode in range(1, num_episodes + 1):
        # Wall-time check
        if max_wall_time_seconds is not None:
            elapsed = time.time() - wall_start
            if elapsed > max_wall_time_seconds:
                print(f"\n  *** WALL-TIME TIMEOUT at episode {episode-1} "
                      f"({elapsed:.0f}s / {max_wall_time_seconds}s) ***")
                break

        # Generate concrete map at top level (L100 continuous)
        concrete_map = generate_continuous_map(
            size=100,
            num_fires=ENV["num_fires_per_level"][100],
            wall_density=ENV["wall_density"],
            random_walls=ENV["random_walls"],
            rng=map_rngs[episode - 1],
        )

        # Discrete version for abstraction (floor continuous positions)
        concrete_map_discrete = {
            "wall_cells": concrete_map["wall_cells"],
            "fire_cells": concrete_map["fire_cells"],
            "agent_starts": [
                (int(np.floor(concrete_map["agent_starts"][0][0])),
                 int(np.floor(concrete_map["agent_starts"][0][1]))),
                (int(np.floor(concrete_map["agent_starts"][1][0])),
                 int(np.floor(concrete_map["agent_starts"][1][1]))),
            ],
            "item_pos": (int(np.floor(concrete_map["item_pos"][0])),
                         int(np.floor(concrete_map["item_pos"][1]))),
            "victim_pos": (int(np.floor(concrete_map["victim_pos"][0])),
                           int(np.floor(concrete_map["victim_pos"][1]))),
            "size": 100,
        }
        abstraction.update(concrete_map_discrete)

        dispatcher.reset(concrete_map=concrete_map)
        obs_list = cont_env.get_observations()

        ep_reward = 0.0
        ep_reward_biased = 0.0
        outcome = "timeout"

        for step_idx in range(max_steps):
            # Store PRE-step positions and has_item for shaping
            old_pos = [cont_env.agent_pos[0].copy(),
                       cont_env.agent_pos[1].copy()]
            had_item_before = (cont_env.agent_has_item[0]
                               or cont_env.agent_has_item[1])

            # Both agents choose actions via shared Q with eps-greedy
            actions = [
                agent.select_action(obs_list[0]),
                agent.select_action(obs_list[1]),
            ]

            _, base_rewards, env_done, info = dispatcher.step(actions)

            # New state
            new_obs_list = cont_env.get_observations()
            new_pos = [cont_env.agent_pos[0].copy(),
                       cont_env.agent_pos[1].copy()]
            has_item_after = (cont_env.agent_has_item[0]
                              or cont_env.agent_has_item[1])

            # Compute shaping for each agent (continuous version)
            s0 = abstraction.compute_shaping_continuous(
                new_pos[0], level, has_item_after,
                differential=diff_shaping,
                prev_pos_continuous=old_pos[0],
                prev_has_item=had_item_before,
                scale=scale,
            )
            s1 = abstraction.compute_shaping_continuous(
                new_pos[1], level, has_item_after,
                differential=diff_shaping,
                prev_pos_continuous=old_pos[1],
                prev_has_item=had_item_before,
                scale=scale,
            )

            r0_biased = base_rewards[0] + s0
            r1_biased = base_rewards[1] + s1

            # Store BOTH agents' transitions in the shared buffer
            # Using BIASED reward (shaping-augmented) so DQN learns
            # with the same guidance PPO had
            agent.add_transition(
                obs_list[0], actions[0], r0_biased,
                new_obs_list[0], env_done,
            )
            agent.add_transition(
                obs_list[1], actions[1], r1_biased,
                new_obs_list[1], env_done,
            )

            # One or more gradient updates
            for _ in range(updates_per_env_step):
                agent.update()

            ep_reward += (base_rewards[0] + base_rewards[1]) / 2
            ep_reward_biased += (r0_biased + r1_biased) / 2

            obs_list = new_obs_list

            if info.get("outcome") == "rescued":
                outcome = "rescued"
            if env_done:
                break

        # Bookkeeping
        success = 1 if outcome == "rescued" else 0
        sr_window.append(success)
        if len(sr_window) > 100:
            sr_window.pop(0)
        if success:
            total_successes += 1

        rewards_hist.append(ep_reward)
        rewards_biased_hist.append(ep_reward_biased)

        if episode % log_every == 0:
            sr_last100 = sum(sr_window) / len(sr_window)
            sr_log.append(sr_last100)
            sr_cum = total_successes / episode
            print(f"  DQN L{level} Ep {episode:5d} | "
                  f"R: {ep_reward:7.2f} | "
                  f"R_bias: {ep_reward_biased:7.2f} | "
                  f"SR_cum: {sr_cum:5.1%} | SR_last100: {sr_last100:5.1%} | "
                  f"eps: {agent.epsilon():.3f} | buf: {len(agent.buffer)} | "
                  f"out: {outcome}")
            # Salvataggio del best model quando SR_last100 supera il record
            if sr_last100 > best_sr and len(sr_window) >= 100:
                best_sr = sr_last100
                agent.save(os.path.join(save_dir, "best_policy.pt"))
                print(f"    [NEW BEST SR: {best_sr:.1%}, best_policy.pt saved]")

    # Save results
    np.save(os.path.join(save_dir, "sr.npy"), np.array(sr_log))
    np.save(os.path.join(save_dir, "rewards.npy"), np.array(rewards_hist))
    np.save(os.path.join(save_dir, "rewards_biased.npy"),
            np.array(rewards_biased_hist))
    agent.save(os.path.join(save_dir, "policy.pt"))

    plot_experiment(rewards_hist, rewards_biased_hist, sr_log,
                    log_every, save_dir)

    final_sr_window = (np.mean(sr_log[-10:]) if len(sr_log) >= 10
                       else (sr_log[-1] if sr_log else 0))
    print(f"\n  DQN Experiment 11 complete.")
    print(f"  Total episodes: {episode}")
    print(f"  Final SR (last 10 logs): {final_sr_window:.1%}")
    verdict = ("SUCCESS" if final_sr_window >= CONVERGENCE["success_threshold"]
               else "FAILURE" if final_sr_window <= CONVERGENCE["failure_threshold"]
               else "PARTIAL")
    print(f"  Verdict: {verdict}\n")


def plot_experiment(rew, rew_b, sr_log, log_every, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exp 11 (DQN): L100 - HIERARCHICAL (V*_50)",
                 fontsize=13, fontweight="bold")
    window = min(100, len(rew))
    if window > 0:
        sm = np.convolve(rew, np.ones(window)/window, mode="valid")
        axes[0].plot(sm, color="steelblue", linewidth=1.2, label="base reward")
        if len(rew_b) > 0:
            sm_b = np.convolve(rew_b, np.ones(window)/window, mode="valid")
            axes[0].plot(sm_b, color="coral", linewidth=1.2,
                         label="base + shaping")
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_title("Reward (smoothed, w=100)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    if sr_log:
        x = np.arange(log_every, len(sr_log) * log_every + 1, log_every)
        axes[1].plot(x, sr_log, color="green", marker="o", markersize=3)
    axes[1].set_title("Success Rate (last 100 episodes)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    SAVE_DIR = os.path.join(PROJECT_ROOT, TRAIN["save_dir"])
    out_dir = os.path.join(SAVE_DIR, "seed_42", "exp_11_L100_hier_dqn_4run")

    MAX_HOURS = 20   

    train_exp11_dqn(
        seed=42,
        save_dir=out_dir,
        max_wall_time_seconds=MAX_HOURS * 3600,
    )