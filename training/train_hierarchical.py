"""
Main orchestrator for the 5-experiment narrative

Narrative (descent for failure, ascent for success):
    Experiment 1: Level 10 SPARSE         -> expected to fail
    Experiment 2: Level  5 SPARSE         -> expected to fail
    Experiment 3: Level  3 SPARSE         -> expected to succeed (base case)
    Experiment 4: Level  5 + V*_3 shaping -> expected to succeed
    Experiment 5: Level 10 + V*_5 shaping -> expected to succeed (final goal)

Same physical map at three resolutions:
    Before training, pre-generate a list of map seeds [s_1, s_2, ..., s_N]
    The same list is used across all 5 experiments. So episode K of any
    experiment starts from the same 10x10 concrete map (generated with seed s_K),
    just projected to the level being trained

For sparse experiments (1, 2, 3): single learner, no shaping
For hierarchical experiments (4, 5): dual learner

Outputs:
    results/seed_42/exp_1_level10_sparse/        plot.png, sr.npy, rewards.npy
    results/seed_42/exp_2_level5_sparse/         
    results/seed_42/exp_3_level3_sparse/         
    results/seed_42/exp_4_level5_hier/           
    results/seed_42/exp_5_level10_hier/          
    results/seed_42/comparison.png               
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
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "env"))

from map_generator import generate_concrete_map
from rescue_env_pz import RescueEnvPZ
from grid_world import get_obs_dim, get_global_dim
from mappo import MAPPO
from abstract_layer import HierarchicalAbstraction
from config import ENV, MAPPO_BASE, HIERARCHY, TRAIN, CONVERGENCE


# Pre-generate map seeds

def generate_map_pool(num_maps, base_seed):
    """
    Pre-generates a pool of map seeds, one per episode

    All 5 experiments will use these same seeds in the same order, so
    episode K always starts from the same physical concrete map

    Returns:
        list of np.random.RandomState objects, one per episode
    """
    rngs = []
    for k in range(num_maps):
        # Each episode gets a unique seed derived from base_seed
        # Using a deterministic combination so seed lists are reproducible
        rngs.append(np.random.RandomState(base_seed * 100000 + k))
    return rngs


# Single-experiment training function

def train_experiment(
    exp_id,
    exp_label,
    level,
    use_shaping,
    map_rngs,
    seed,
    abstraction,
    save_dir,
    log_every=100,
):
    """
    Runs one experiment of the 5 (one combination of level + sparse/hierarchical)

    Args:
        exp_id:       1-5
        exp_label:    descriptive string for plots/logs
        level:        3, 5, or 10
        use_shaping:  False for sparse, True for hierarchical (with shaping)
        map_rngs:     list of pre-generated RandomStates (one per episode)
        seed:         master random seed (for torch.manual_seed and policy init)
        abstraction:  HierarchicalAbstraction instance (computes V* per episode)
        save_dir:     output directory for this experiment
        log_every:    log frequency
    """
    os.makedirs(save_dir, exist_ok=True)

    obs_dim, _ = get_obs_dim(level)
    global_dim = get_global_dim(level)
    num_episodes = len(map_rngs)
    max_steps = ENV["max_steps_per_level"][level]

    # Reset torch seed per experiment so each starts with the same network init
    # (this isolates the effect of shaping vs no shaping on the same task)
    torch.manual_seed(seed + exp_id)
    np.random.seed(seed + exp_id)

    print(f"\n{'═' * 72}")
    print(f"  EXPERIMENT {exp_id}: {exp_label}")
    print(f"  Level: {level} | Episodes: {num_episodes} | Max steps: {max_steps}")
    print(f"  Shaping: {'YES (dual learner)' if use_shaping else 'NO (single learner)'}")
    print(f"  obs_dim: {obs_dim} | global_dim: {global_dim}")
    print(f"{'═' * 72}\n")

    # Create env at this level
    env = RescueEnvPZ(size=level, max_cycles=max_steps,
                       num_fires=ENV["num_fires"], seed=seed + exp_id)

    # Create MAPPO
    mappo_cfg = dict(MAPPO_BASE)
    mappo_cfg["obs_dim"] = obs_dim
    mappo_cfg["global_dim"] = global_dim
    agent = MAPPO(**mappo_cfg, dual_learner=use_shaping)

    diff_shaping = HIERARCHY["differential_shaping"]
    scale = HIERARCHY["shaping_scale"]

    rewards_hist = []
    rewards_biased_hist = []
    sr_window = []
    sr_log = []
    total_successes = 0

    for episode in range(1, num_episodes + 1):
        # Generate the concrete 10x10 map for this episode
        # Same rng as all other experiments at this episode index
        concrete_map = generate_concrete_map(
            num_fires=ENV["num_fires"],
            random_walls=ENV.get("random_walls", False),      
            num_random_walls=ENV.get("num_random_walls", 8),
            rng=map_rngs[episode - 1],
        )

        # Compute V* at all levels for this episode's concrete map
        abstraction.update(concrete_map)

        # Reset env (the env will project concrete_map to its level if needed)
        obs, _ = env.reset(concrete_map=concrete_map)

        ep_reward = 0.0
        ep_reward_biased = 0.0
        outcome = "timeout"

        # Edge case: at coarse levels, agent might start on victim with item
        # Check if episode is already done from reset
        had_item_init = (env._env.agent1_has_item or env._env.agent2_has_item)
        if had_item_init:
            holder = env._env.agent1_pos if env._env.agent1_has_item else env._env.agent2_pos
            if env._env._is_adjacent(holder, env._env.victim_pos):
                # Trivial success, no rollout needed
                env._env.done = True
                env._env.outcome = "rescued"
                env.agents = []
                outcome = "rescued"

        while env.agents:
            obs_list = [obs["agent_0"], obs["agent_1"]]
            global_state = env._env.get_global_state()

            if use_shaping:
                actions, log_probs, val_b, val_u = agent.select_actions(
                    obs_list, global_state
                )
            else:
                actions, log_probs, val_b = agent.select_actions(
                    obs_list, global_state
                )
                val_u = None

            actions_dict = {"agent_0": actions[0], "agent_1": actions[1]}

            old_pos_0 = env._env.agent1_pos
            old_pos_1 = env._env.agent2_pos
            had_item_before = (env._env.agent1_has_item
                               or env._env.agent2_has_item)

            next_obs, rewards, terms, truncs, infos = env.step(actions_dict)

            new_pos_0 = env._env.agent1_pos
            new_pos_1 = env._env.agent2_pos
            has_item_after = (env._env.agent1_has_item
                              or env._env.agent2_has_item)

            done = terms.get("agent_0", False)
            info = infos.get("agent_0", {})
            if info.get("outcome") == "rescued":
                outcome = "rescued"

            r0_base = rewards.get("agent_0", 0.0)
            r1_base = rewards.get("agent_1", 0.0)

            if use_shaping:
                s0 = abstraction.compute_shaping(
                    new_pos_0, level, has_item_after,
                    differential=diff_shaping,
                    prev_pos=old_pos_0, prev_has_item=had_item_before,
                    scale=scale,
                )
                s1 = abstraction.compute_shaping(
                    new_pos_1, level, has_item_after,
                    differential=diff_shaping,
                    prev_pos=old_pos_1, prev_has_item=had_item_before,
                    scale=scale,
                )
                r0_b = r0_base + s0
                r1_b = r1_base + s1

                agent.add_to_buffer(
                    obs=obs_list, global_state=global_state,
                    actions=actions, log_probs=log_probs,
                    rewards_biased=[r0_b, r1_b],
                    value_biased=val_b, done=done,
                    rewards_unbiased=[r0_base, r1_base],
                    value_unbiased=val_u,
                )
                ep_reward_biased += (r0_b + r1_b) / 2
                ep_reward += (r0_base + r1_base) / 2
            else:
                agent.add_to_buffer(
                    obs=obs_list, global_state=global_state,
                    actions=actions, log_probs=log_probs,
                    rewards_biased=[r0_base, r1_base],
                    value_biased=val_b, done=done,
                )
                ep_reward += (r0_base + r1_base) / 2
                ep_reward_biased = ep_reward

            obs = next_obs

        if agent.buffer_biased.size > 0:
            agent.update()

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
            extra = f" | R_bias: {ep_reward_biased:6.2f}" if use_shaping else ""
            print(
                f"  Exp{exp_id} L{level} Ep {episode:5d} | "
                f"R: {ep_reward:6.2f}{extra} | "
                f"SR_cum: {sr_cum:5.1%} | "
                f"SR_last100: {sr_last100:5.1%} | "
                f"out: {outcome}"
            )

    # Save outputs
    np.save(os.path.join(save_dir, "sr.npy"), np.array(sr_log))
    np.save(os.path.join(save_dir, "rewards.npy"), np.array(rewards_hist))
    if use_shaping:
        np.save(os.path.join(save_dir, "rewards_biased.npy"),
                np.array(rewards_biased_hist))
    agent.save(os.path.join(save_dir, "policy.pt"))

    plot_single_experiment(
        exp_id, exp_label, level, use_shaping,
        rewards_hist, rewards_biased_hist, sr_log,
        save_dir, log_every,
    )

    final_sr = total_successes / num_episodes
    sr_last_window = (np.mean(sr_log[-10:]) if len(sr_log) >= 10
                      else (sr_log[-1] if sr_log else 0))
    print(f"\n  Experiment {exp_id} complete.")
    print(f"  Cumulative SR: {final_sr:.1%}")
    print(f"  Final SR (last 10 logs): {sr_last_window:.1%}")

    # Convergence verdict
    if sr_last_window >= CONVERGENCE["success_threshold"]:
        verdict = "SUCCESS"
    elif sr_last_window <= CONVERGENCE["failure_threshold"]:
        verdict = "FAILURE"
    else:
        verdict = "PARTIAL"
    print(f"  Verdict: {verdict}\n")

    return {
        "exp_id": exp_id,
        "label": exp_label,
        "level": level,
        "use_shaping": use_shaping,
        "rewards": rewards_hist,
        "rewards_biased": rewards_biased_hist,
        "sr_log": sr_log,
        "final_sr": final_sr,
        "final_sr_window": sr_last_window,
        "verdict": verdict,
    }


def plot_single_experiment(exp_id, exp_label, level, dual,
                            rew, rew_b, sr_log, save_dir, log_every):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Experiment {exp_id}: {exp_label}",
                 fontsize=13, fontweight="bold")

    window = min(100, len(rew))
    if window > 0:
        sm = np.convolve(rew, np.ones(window) / window, mode="valid")
        axes[0].plot(sm, color="steelblue", linewidth=1.2,
                     label="base reward")
        if dual and len(rew_b) > 0:
            sm_b = np.convolve(rew_b, np.ones(window) / window, mode="valid")
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


# Main pipeline

def run_full_pipeline(seed, save_dir):
    """Runs all 5 experiments for one seed."""
    out_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'█' * 72}")
    print(f"  HIERARCHICAL TRAINING PIPELINE")
    print(f"  Seed: {seed}")
    print(f"  5 experiments: descent for failure, ascent for success")
    print(f"{'█' * 72}")

    abstraction = HierarchicalAbstraction(gamma=MAPPO_BASE["gamma"])

    # Pre-generate map seed pool: enough episodes for the longest experiment
    # Different experiments use different prefixes of this same pool, so
    # episode K of any experiment uses the same underlying concrete map
    max_episodes = max(TRAIN["episodes_per_level"].values())
    map_pool_full = generate_map_pool(max_episodes, seed)
    print(f"\n  Pre-generated {max_episodes} concrete map seeds.")
    print(f"  All 5 experiments will share these in order (Option A).\n")

    results = []

    # Experiment 1: Level 10 SPARSE


    n10 = TRAIN["episodes_per_level"][10]
    res = train_experiment(
        exp_id=1, exp_label="Level 10 - SPARSE (no shaping)",
        level=10, use_shaping=False,
        map_rngs=map_pool_full[:n10],
        seed=seed, abstraction=abstraction,
        save_dir=os.path.join(out_dir, "exp_1_level10_sparse"),
        log_every=TRAIN["log_every"],
    )
    results.append(res)
    

    # Experiment 2: Level 5 SPARSE
    n5 = TRAIN["episodes_per_level"][5]
    res = train_experiment(
        exp_id=2, exp_label="Level 5 - SPARSE (no shaping)",
        level=5, use_shaping=False,
        map_rngs=map_pool_full[:n5],
        seed=seed, abstraction=abstraction,
        save_dir=os.path.join(out_dir, "exp_2_level5_sparse"),
        log_every=TRAIN["log_every"],
    )
    results.append(res)
    

    # Experiment 3: Level 3 SPARSE (base case)
    n3 = TRAIN["episodes_per_level"][3]
    res = train_experiment(
        exp_id=3, exp_label="Level 3 - SPARSE (base case)",
        level=3, use_shaping=False,
        map_rngs=map_pool_full[:n3],
        seed=seed, abstraction=abstraction,
        save_dir=os.path.join(out_dir, "exp_3_level3_sparse"),
        log_every=TRAIN["log_every"],
    )
    results.append(res)

    # Experiment 4: Level 5 + shaping V*_3
    res = train_experiment(
        exp_id=4, exp_label="Level 5 - HIERARCHICAL (shaping from V*_3)",
        level=5, use_shaping=True,
        map_rngs=map_pool_full[:n5],
        seed=seed, abstraction=abstraction,
        save_dir=os.path.join(out_dir, "exp_4_level5_hier"),
        log_every=TRAIN["log_every"],
    )
    results.append(res)


    # Experiment 5: Level 10 + shaping V*_5
    res = train_experiment(
        exp_id=5, exp_label="Level 10 - HIERARCHICAL (shaping from V*_5)",
        level=10, use_shaping=True,
        map_rngs=map_pool_full[:n10],
        seed=seed, abstraction=abstraction,
        save_dir=os.path.join(out_dir, "exp_5_level10_hier"),
        log_every=TRAIN["log_every"],
    )
    results.append(res)


    # Generate the comparison plot
    plot_comparison(results, out_dir, seed)

    # Print summary
    print(f"\n{'█' * 72}")
    print(f"  PIPELINE COMPLETE for seed {seed}")
    print(f"{'█' * 72}")
    print(f"  {'Experiment':<55} {'Final SR':>10} {'Verdict':>10}")
    print(f"  {'-' * 75}")
    for r in results:
        print(f"  Exp {r['exp_id']}: {r['label']:<48} "
              f"{r['final_sr_window']:>9.1%} {r['verdict']:>10}")
    print(f"\n  Results saved in: {out_dir}\n")


def plot_comparison(results, out_dir, seed):
    """The narrative plot: all 5 SR curves side by side."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        1: ("#d62728", "--"),   # red dashed - level 10 sparse (failure)
        2: ("#ff7f0e", "--"),   # orange dashed - level 5 sparse (failure)
        3: ("#2ca02c", "-"),    # green solid - level 3 sparse (base success)
        4: ("#1f77b4", "-"),    # blue solid - level 5 hierarchical
        5: ("#9467bd", "-"),    # purple solid - level 10 hierarchical (final goal)
    }

    log_every = TRAIN["log_every"]

    for r in results:
        sr = r["sr_log"]
        if not sr:
            continue
        x = np.arange(log_every, len(sr) * log_every + 1, log_every)
        color, ls = colors[r["exp_id"]]
        ax.plot(x, sr, color=color, linestyle=ls, linewidth=2.0,
                marker="o", markersize=3,
                label=f"Exp {r['exp_id']}: {r['label']}")

    # Reference thresholds
    ax.axhline(CONVERGENCE["success_threshold"], color="green",
               linewidth=0.8, linestyle=":", alpha=0.6,
               label=f"Success threshold ({CONVERGENCE['success_threshold']:.0%})")
    ax.axhline(CONVERGENCE["failure_threshold"], color="red",
               linewidth=0.8, linestyle=":", alpha=0.6,
               label=f"Failure threshold ({CONVERGENCE['failure_threshold']:.0%})")

    ax.set_title(f"Hierarchical Training Comparison (seed {seed})\n"
                 "Descent: sparse fails at high levels (10, 5), succeeds at base (3)\n"
                 "Ascent: shaping from V*_3 enables 5, V*_5 enables 10",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (last 100 episodes)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison.png"), dpi=180)
    plt.close()
    print(f"  Comparison plot saved -> {os.path.join(out_dir, 'comparison.png')}")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    SAVE_DIR = os.path.join(PROJECT_ROOT, "results_1_gamma")

    for seed in TRAIN["seeds"]:
        run_full_pipeline(seed=seed, save_dir=SAVE_DIR)

    print("\nAll seeds complete.")
