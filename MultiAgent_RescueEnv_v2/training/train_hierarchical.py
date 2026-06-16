"""
train_hierarchical.py - Generalized orchestrator for N-level pyramid.

For a pyramid like [3, 6, 12, 24], generates 2N-1 = 7 experiments:

    Sparse experiments (top-down, the "descent"):
        Exp 1: L24 SPARSE  -> expected fail
        Exp 2: L12 SPARSE  -> expected fail or partial
        Exp 3: L6  SPARSE  -> partial
        Exp 4: L3  SPARSE  -> success (base case)

    Hierarchical experiments (bottom-up, the "ascent"):
        Exp 5: L6  + V*_3 shaping   -> success
        Exp 6: L12 + V*_6 shaping   -> success
        Exp 7: L24 + V*_12 shaping  -> success (final goal)

OPTION A: episode K of any experiment uses the same underlying concrete map
(generated at the largest level and projected to smaller levels as needed).

Output structure:
    results_24x24/seed_42/
        exp_1_L24_sparse/   (training.png, sr.npy, rewards.npy, policy.pt)
        exp_2_L12_sparse/
        ...
        comparison.png      <- the money figure with all 2N-1 curves
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
from config import ENV, MAPPO_BASE, HIERARCHY, TRAIN, CONVERGENCE, LEVELS


# ── Pre-generate map seeds ────────────────────────────────────────────

def generate_map_pool(num_maps, base_seed):
    """
    Pre-generates a pool of RandomState objects, one per episode.
    All experiments share these in order (Option A).
    """
    rngs = []
    for k in range(num_maps):
        rngs.append(np.random.RandomState(base_seed * 100000 + k))
    return rngs


# ── Single-experiment training ────────────────────────────────────────

def train_experiment(exp_id, exp_label, level, use_shaping, map_rngs,
                     seed, abstraction, save_dir, log_every=100):
    """
    Trains one experiment (one combination of level + sparse/hierarchical).
    """
    os.makedirs(save_dir, exist_ok=True)

    view_size = ENV["view_size_per_level"][level]
    obs_dim, _ = get_obs_dim(level, view_size=view_size)
    global_dim = get_global_dim(level)
    num_episodes = len(map_rngs)
    max_steps = ENV["max_steps_per_level"][level]
    num_fires_level = ENV["num_fires_per_level"][level]

    torch.manual_seed(seed + exp_id)
    np.random.seed(seed + exp_id)

    print(f"\n{'═' * 72}")
    print(f"  EXPERIMENT {exp_id}: {exp_label}")
    print(f"  Level: {level} | Episodes: {num_episodes} | Max steps: {max_steps}")
    print(f"  View: {view_size}x{view_size} | obs_dim: {obs_dim} | global_dim: {global_dim}")
    print(f"  Shaping: {'YES (dual learner)' if use_shaping else 'NO (single learner)'}")
    print(f"{'═' * 72}\n")

    # Environment at this level
    env = RescueEnvPZ(
        size=level, max_cycles=max_steps,
        num_fires=num_fires_level,
        wall_density=ENV["wall_density"],
        random_walls=ENV["random_walls"],
        view_size=view_size,
        seed=seed + exp_id,
    )

    # MAPPO agent
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

    top_level = max(abstraction.levels)

    for episode in range(1, num_episodes + 1):
        # Generate concrete map at TOP LEVEL using shared map RNG
        # (ensures Option A: same physical map across experiments)
        concrete_map = generate_concrete_map(
            size=top_level,
            num_fires=ENV["num_fires_per_level"][top_level],
            wall_density=ENV["wall_density"],
            random_walls=ENV["random_walls"],
            rng=map_rngs[episode - 1],
        )

        # Compute V* at all levels for this episode
        abstraction.update(concrete_map)

        # Reset env: it will auto-project concrete_map to its own size
        obs, _ = env.reset(concrete_map=concrete_map)

        ep_reward = 0.0
        ep_reward_biased = 0.0
        outcome = "timeout"

        # Edge case: agent may start at item due to projection collapse
        had_item_init = (env._env.agent1_has_item or env._env.agent2_has_item)
        if had_item_init:
            holder = env._env.agent1_pos if env._env.agent1_has_item else env._env.agent2_pos
            if env._env._is_adjacent(holder, env._env.victim_pos):
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
            had_item_before = (env._env.agent1_has_item or env._env.agent2_has_item)

            next_obs, rewards, terms, truncs, infos = env.step(actions_dict)

            new_pos_0 = env._env.agent1_pos
            new_pos_1 = env._env.agent2_pos
            has_item_after = (env._env.agent1_has_item or env._env.agent2_has_item)

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
    fig.suptitle(f"Exp {exp_id}: {exp_label}", fontsize=13, fontweight="bold")

    window = min(100, len(rew))
    if window > 0:
        sm = np.convolve(rew, np.ones(window) / window, mode="valid")
        axes[0].plot(sm, color="steelblue", linewidth=1.2, label="base reward")
        if dual and len(rew_b) > 0:
            sm_b = np.convolve(rew_b, np.ones(window) / window, mode="valid")
            axes[0].plot(sm_b, color="coral", linewidth=1.2, label="base + shaping")
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


# ── Main pipeline ─────────────────────────────────────────────────────

def run_full_pipeline(seed, save_dir):
    """
    Runs all 2N-1 experiments for one seed (N levels).
    """
    out_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'█' * 72}")
    print(f"  HIERARCHICAL TRAINING PIPELINE — v3 (N-level generalization)")
    print(f"  Seed: {seed} | Pyramid: {LEVELS}")
    print(f"  {2*len(LEVELS)-1} experiments: descent for failure, ascent for success")
    print(f"{'█' * 72}")

    abstraction = HierarchicalAbstraction(
        levels=LEVELS, gamma_VI=HIERARCHY["gamma_VI"]
    )

    # Pre-generate map pool sized to the longest experiment
    max_episodes = max(TRAIN["episodes_per_level"].values())
    map_pool_full = generate_map_pool(max_episodes, seed)
    print(f"\n  Pre-generated {max_episodes} concrete map seeds.")
    print(f"  All experiments share these in order (Option A).\n")

    results = []
    exp_id = 1

    # PHASE 1: SPARSE experiments (descent, top to base)
    for level in sorted(LEVELS, reverse=True):
        n_eps = TRAIN["episodes_per_level"][level]
        res = train_experiment(
            exp_id=exp_id,
            exp_label=f"L{level} - SPARSE (no shaping)",
            level=level, use_shaping=False,
            map_rngs=map_pool_full[:n_eps],
            seed=seed, abstraction=abstraction,
            save_dir=os.path.join(out_dir, f"exp_{exp_id}_L{level}_sparse"),
            log_every=TRAIN["log_every"],
        )
        results.append(res)
        exp_id += 1

    # PHASE 2: HIERARCHICAL experiments (ascent, base+1 to top)
    for level in sorted(LEVELS)[1:]:
        n_eps = TRAIN["episodes_per_level"][level]
        upper = abstraction.get_upper_level(level)
        res = train_experiment(
            exp_id=exp_id,
            exp_label=f"L{level} - HIERARCHICAL (shaping from V*_{upper})",
            level=level, use_shaping=True,
            map_rngs=map_pool_full[:n_eps],
            seed=seed, abstraction=abstraction,
            save_dir=os.path.join(out_dir, f"exp_{exp_id}_L{level}_hier"),
            log_every=TRAIN["log_every"],
        )
        results.append(res)
        exp_id += 1

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
    """All 2N-1 SR curves on a single plot (the money figure)."""
    fig, ax = plt.subplots(figsize=(14, 8))

    n_exp = len(results)
    n_levels = len(LEVELS)
    sparse_count = n_levels   # first N exp are sparse
    # Color palette: warm for sparse (red->yellow->green), cool for hierarchical (blue->purple)
    sparse_cmap = plt.cm.YlOrRd_r
    hier_cmap = plt.cm.cool

    log_every = TRAIN["log_every"]

    for i, r in enumerate(results):
        sr = r["sr_log"]
        if not sr:
            continue
        x = np.arange(log_every, len(sr) * log_every + 1, log_every)

        if i < sparse_count:
            # Sparse: dashed, warm color
            color = sparse_cmap(0.2 + 0.6 * i / max(1, sparse_count - 1))
            linestyle = "--"
        else:
            # Hierarchical: solid, cool color
            j = i - sparse_count
            color = hier_cmap(0.2 + 0.6 * j / max(1, n_levels - 2))
            linestyle = "-"

        ax.plot(x, sr, color=color, linestyle=linestyle, linewidth=2.0,
                marker="o", markersize=3,
                label=f"Exp {r['exp_id']}: {r['label']}")

    ax.axhline(CONVERGENCE["success_threshold"], color="green",
               linewidth=0.8, linestyle=":", alpha=0.6,
               label=f"Success threshold ({CONVERGENCE['success_threshold']:.0%})")
    ax.axhline(CONVERGENCE["failure_threshold"], color="red",
               linewidth=0.8, linestyle=":", alpha=0.6,
               label=f"Failure threshold ({CONVERGENCE['failure_threshold']:.0%})")

    ax.set_title(
        f"Hierarchical Training Comparison — Pyramid {LEVELS} (seed {seed})\n"
        f"Descent: sparse fails at high levels, succeeds at base ({LEVELS[0]})\n"
        f"Ascent: shaping from V*_lower enables higher levels",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (last 100 episodes)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison.png"), dpi=180)
    plt.close()
    print(f"  Comparison plot saved -> {os.path.join(out_dir, 'comparison.png')}")

"""

Main per runnare tutti gli experiments

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    SAVE_DIR = os.path.join(PROJECT_ROOT, TRAIN["save_dir"])

    for seed in TRAIN["seeds"]:
        run_full_pipeline(seed=seed, save_dir=SAVE_DIR)

    print("\nAll seeds complete.")
"""

def run_experiments_by_ids(seed, exp_ids_to_run, save_dir):
    """
    Runs only the specified experiment IDs. Loads completed ones from disk.
    Generates comparison plot only when all 7 experiments are available.
    """
    out_dir = os.path.join(save_dir, f"seed_{seed}")

    abstraction = HierarchicalAbstraction(
        levels=LEVELS, gamma_VI=HIERARCHY["gamma_VI"]
    )
    max_episodes = max(TRAIN["episodes_per_level"].values())
    map_pool_full = generate_map_pool(max_episodes, seed)

    # Build full experiment list in the same order as run_full_pipeline
    all_experiments = []
    exp_id = 1
    for level in sorted(LEVELS, reverse=True):
        all_experiments.append({
            "exp_id": exp_id, "level": level, "use_shaping": False,
            "label": f"L{level} - SPARSE (no shaping)",
            "folder": f"exp_{exp_id}_L{level}_sparse",
        })
        exp_id += 1
    for level in sorted(LEVELS)[1:]:
        upper = abstraction.get_upper_level(level)
        all_experiments.append({
            "exp_id": exp_id, "level": level, "use_shaping": True,
            "label": f"L{level} - HIERARCHICAL (shaping from V*_{upper})",
            "folder": f"exp_{exp_id}_L{level}_hier",
        })
        exp_id += 1

    results = []

    for exp_info in all_experiments:
        exp_dir = os.path.join(out_dir, exp_info["folder"])

        if exp_info["exp_id"] in exp_ids_to_run:
            # Run this experiment fresh
            n_eps = TRAIN["episodes_per_level"][exp_info["level"]]
            res = train_experiment(
                exp_id=exp_info["exp_id"],
                exp_label=exp_info["label"],
                level=exp_info["level"],
                use_shaping=exp_info["use_shaping"],
                map_rngs=map_pool_full[:n_eps],
                seed=seed, abstraction=abstraction,
                save_dir=exp_dir,
                log_every=TRAIN["log_every"],
            )
            results.append(res)
        else:
            # Load from already-completed experiment on disk
            sr_path = os.path.join(exp_dir, "sr.npy")
            rew_path = os.path.join(exp_dir, "rewards.npy")
            if not os.path.exists(sr_path):
                print(f"  WARNING: Exp {exp_info['exp_id']} not found at {sr_path}")
                print(f"  Add {exp_info['exp_id']} to exp_ids_to_run to train it.")
                continue
            sr_log = list(np.load(sr_path))
            rewards  = list(np.load(rew_path))
            sr_last = (float(np.mean(sr_log[-10:])) if len(sr_log) >= 10
                       else (sr_log[-1] if sr_log else 0))
            verdict = ("SUCCESS" if sr_last >= CONVERGENCE["success_threshold"]
                       else "FAILURE" if sr_last <= CONVERGENCE["failure_threshold"]
                       else "PARTIAL")
            results.append({
                "exp_id": exp_info["exp_id"], "label": exp_info["label"],
                "level": exp_info["level"], "use_shaping": exp_info["use_shaping"],
                "rewards": rewards, "rewards_biased": [],
                "sr_log": sr_log,
                "final_sr": sum(sr_log) / len(sr_log) if sr_log else 0,
                "final_sr_window": sr_last, "verdict": verdict,
            })
            print(f"  Loaded Exp {exp_info['exp_id']}: {exp_info['label']}"
                  f" -> SR={sr_last:.1%} ({verdict})")

    # Generate comparison plot only if all experiments are present
    if len(results) == len(all_experiments):
        plot_comparison(results, out_dir, seed)
        print(f"\n  Comparison plot saved.")
    else:
        n_missing = len(all_experiments) - len(results)
        print(f"\n  {n_missing} experiment(s) missing — comparison plot not generated yet.")
        print(f"  Run the missing experiments and then call this function again.")

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    SAVE_DIR = os.path.join(PROJECT_ROOT, TRAIN["save_dir"])

    # ── Opzioni disponibili ────────────────────────────────────────────
    # Pipeline completa (tutti i seed, tutti gli esperimenti):
    # for seed in TRAIN["seeds"]:
    #     run_full_pipeline(seed=seed, save_dir=SAVE_DIR)

    # Solo Exp 6 e 7 (carica 1-5 da disco, riesegue 6 e 7):
    run_experiments_by_ids(seed=42, exp_ids_to_run=[1], save_dir=SAVE_DIR)

    print("\nDone.")