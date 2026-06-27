"""
train_options.py - Train all 4 direction options sequentially.

For each direction (U, R, D, L):
    1. Create OptionEnv with that direction
    2. Train OptionPPO for N episodes with wall curriculum
    3. Save checkpoint as pi_<dir>.pt
    4. Generate training plot (rewards + success rate)

Wall curriculum (configurable):
    - Episodes 0 to CURRICULUM_STAGES[0]:  wall_prob = 0%
    - Episodes CURRICULUM_STAGES[0] to [1]: wall_prob = 15%
    - Episodes CURRICULUM_STAGES[1] to end: wall_prob = 30%

The agent should converge to a near-100% success rate at wall_prob=0
within ~1000 episodes. With walls, expect 80-95% success.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from option_env import OptionEnv, DIRECTIONS
from option_ppo import OptionPPO


# ── CONFIG ────────────────────────────────────────────────────────────

CONFIG = {
    "episodes_per_option": 7000,
    "log_every": 100,
    "max_steps_per_episode": 50,
    "d_max": 0.2,

    # Wall curriculum: (cutoff_episode, wall_prob)
    "curriculum": [
    # (cutoff_episode, wall_prob, fire_prob)
    (1500, 0.00, 0.00),    # base direction
    (3000, 0.15, 0.00),    # walls only
    (5000, 0.15, 0.15),    # walls + light fires
    (7000, 0.25, 0.20),    # walls + heavier fires
    ],

    # PPO hyperparameters
    "ppo": {
        "hidden_dim": 64,
        "lr_actor": 3e-4,
        "lr_critic": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "epochs": 4,
        "batch_size": 64,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },

    "seed": 42,
}


def get_curriculum_probs(episode, curriculum):
    """Returns (wall_prob, fire_prob) for the given episode."""
    for cutoff, wp, fp in curriculum:
        if episode < cutoff:
            return wp, fp
    return curriculum[-1][1], curriculum[-1][2]


def train_single_option(direction, save_dir, config):
    """Train one option (one direction) and save checkpoint + plots."""
    print(f"\n{'═' * 64}")
    print(f"  TRAINING OPTION: pi_{direction}")
    print(f"  Direction: {direction} | Episodes: {config['episodes_per_option']}")
    print(f"  Device: {config['ppo']['device']}")
    print(f"{'═' * 64}\n")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    env = OptionEnv(
        direction=direction,
        d_max=config["d_max"],
        wall_prob=0.0,                       # set per episode via curriculum
        max_steps=config["max_steps_per_episode"],
        seed=config["seed"],
    )

    ppo = OptionPPO(
        obs_dim=18,
        action_dim=2,
        **config["ppo"],
    )

    rewards_hist = []
    sr_window = []   # rolling success window (last 100 episodes)
    sr_log = []      # logged every `log_every` episodes
    success_counts = {"success": 0, "wrong_exit": 0, "timeout": 0}

    for ep in range(1, config["episodes_per_option"] + 1):
        # Set wall_prob based on curriculum
        wp, fp = get_curriculum_probs(ep, config["curriculum"])
        env.set_wall_prob(wp)
        env.set_fire_prob(fp)

        obs = env.reset()
        ep_reward = 0.0
        outcome = "timeout"

        while True:
            action, log_prob, value = ppo.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            ppo.add_to_buffer(obs, action, log_prob, reward, value, done)
            ep_reward += reward
            obs = next_obs
            if done:
                outcome = info["outcome"]
                break

        ppo.update()

        rewards_hist.append(ep_reward)
        success = 1 if outcome in ("success", "success_via_fire") else 0
        sr_window.append(success)
        if len(sr_window) > 100:
            sr_window.pop(0)
        success_counts[outcome] = success_counts.get(outcome, 0) + 1

        if ep % config["log_every"] == 0:
            sr_last100 = sum(sr_window) / len(sr_window)
            sr_log.append(sr_last100)
            print(f"  pi_{direction} Ep {ep:5d} | "
                  f"R: {ep_reward:7.2f} | "
                  f"SR_last100: {sr_last100:5.1%} | "
                  f"wall_p: {wp:.2f}, fire_p: {fp:.2f} | "
                  f"out: {outcome}")

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, f"pi_{direction}.pt")
    ppo.save(ckpt_path)

    # Save numpy histories
    np.save(os.path.join(save_dir, f"pi_{direction}_rewards.npy"),
            np.array(rewards_hist))
    np.save(os.path.join(save_dir, f"pi_{direction}_sr.npy"),
            np.array(sr_log))

    # Plot
    plot_training(direction, rewards_hist, sr_log,
                  config["log_every"], save_dir)

    print(f"\n  Option pi_{direction} saved -> {ckpt_path}")
    print(f"  Final SR_last100: {sr_log[-1]:.1%}")
    print(f"  Outcome breakdown over {ep} episodes:")
    for k, v in success_counts.items():
        print(f"    {k:12s}: {v:6d} ({v/ep:5.1%})")

    return {
        "direction": direction,
        "rewards": rewards_hist,
        "sr_log": sr_log,
        "final_sr": sr_log[-1] if sr_log else 0,
    }


def plot_training(direction, rewards, sr_log, log_every, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Option pi_{direction}: training curves",
                 fontsize=13, fontweight="bold")

    window = min(100, len(rewards))
    if window > 0:
        sm = np.convolve(rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(sm, color="steelblue", linewidth=1.2)
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_title("Reward (smoothed)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
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
    plt.savefig(os.path.join(save_dir, f"pi_{direction}_training.png"), dpi=140)
    plt.close()


def plot_comparison(results, save_dir):
    """Single plot with all 4 options' success-rate curves."""
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"U": "#1f77b4", "R": "#2ca02c", "D": "#d62728", "L": "#9467bd"}
    log_every = CONFIG["log_every"]

    for r in results:
        sr = r["sr_log"]
        if not sr:
            continue
        x = np.arange(log_every, len(sr) * log_every + 1, log_every)
        ax.plot(x, sr, color=colors[r["direction"]], linewidth=2,
                marker="o", markersize=3,
                label=f"pi_{r['direction']} (final SR: {r['final_sr']:.1%})")

    # Mark curriculum transitions
    for cutoff, wp, fp in CONFIG["curriculum"][:-1]:
        ax.axvline(cutoff, color="gray", linestyle=":", alpha=0.4)
        ax.text(cutoff, 0.05, f" wp:{wp:.0%}, fp:{fp:.0%}",
                fontsize=8, color="gray", rotation=90, va="bottom")

    ax.set_title("Training comparison — all 4 direction options",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (last 100 episodes)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "options_comparison.png"), dpi=160)
    plt.close()
    print(f"\n  Comparison plot saved -> {os.path.join(save_dir, 'options_comparison.png')}")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    SAVE_DIR = os.path.join(PROJECT_ROOT, "trained_options_2run")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Saving trained options to: {SAVE_DIR}\n")

    all_results = []
    for direction in DIRECTIONS:
        result = train_single_option(direction, SAVE_DIR, CONFIG)
        all_results.append(result)

    plot_comparison(all_results, SAVE_DIR)

    print("\n" + "█" * 64)
    print("  ALL OPTIONS TRAINED")
    print("█" * 64)
    print(f"\n  {'Direction':<12} {'Final SR':>10}")
    print(f"  {'-' * 24}")
    for r in all_results:
        print(f"  pi_{r['direction']:<10} {r['final_sr']:>9.1%}")
    print(f"\n  Saved in: {SAVE_DIR}\n")
