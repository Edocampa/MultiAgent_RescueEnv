"""
train_hierarchical.py - Orchestrator for the 11 experiments of Phase 2.

Pyramid: [3, 6, 12, 25, 50, 100]

11 experiments:
    Sparse (descent):  L100, L50, L25, L12, L6, L3
    Hierarchical (ascent): L6, L12, L25, L50, L100

For levels 3-50 we use the DISCRETE env (env_discrete.RescueEnvPZ).
For level 100 we use the CONTINUOUS env + OptionDispatcher (macro actions).

OPTION A IS PRESERVED: episode K of any experiment uses the SAME concrete
map seed, projected to the appropriate level. The concrete map is generated
at the TOP level (100), then projected down for lower-level experiments.
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
from rescue_env_pz import RescueEnvPZ
from grid_world import get_obs_dim as discrete_get_obs_dim, get_global_dim as discrete_get_global_dim
from continuous_world import ContinuousWorld
from continuous_map import generate_continuous_map
from option_dispatcher import OptionDispatcher

from mappo import MAPPO
from abstract_layer import HierarchicalAbstraction
from config import (ENV, MAPPO_BASE, HIERARCHY, TRAIN, CONVERGENCE,
                     LEVELS, CONTINUOUS_LEVEL)


# ── Map pool ──────────────────────────────────────────────────────────

def generate_map_pool(num_maps, base_seed):
    rngs = []
    for k in range(num_maps):
        rngs.append(np.random.RandomState(base_seed * 100000 + k))
    return rngs


################# train experiment with timer  ###############

def train_experiment(exp_id, exp_label, level, use_shaping, map_rngs,
                     seed, abstraction, save_dir, log_every=100, max_wall_time_seconds=None, start_wall_time=None):
    os.makedirs(save_dir, exist_ok=True)
    is_continuous = (level == CONTINUOUS_LEVEL)
    view_size = ENV["view_size_per_level"][level]
    num_episodes = len(map_rngs)
    max_steps = ENV["max_steps_per_level"][level]
    num_fires_level = ENV["num_fires_per_level"][level]
    top_level = max(abstraction.levels)

    torch.manual_seed(seed + exp_id)
    np.random.seed(seed + exp_id)

    print(f"\n{'═' * 72}")
    print(f"  EXPERIMENT {exp_id}: {exp_label}")
    print(f"  Level: {level} | Episodes: {num_episodes} | Max steps: {max_steps}")
    print(f"  Type: {'CONTINUOUS+DISPATCHER' if is_continuous else 'DISCRETE'}")
    print(f"  Shaping: {'YES (dual learner)' if use_shaping else 'NO'}")
    print(f"{'═' * 72}\n")

    # ── Create environment (discrete or continuous+dispatcher) ────────
    if is_continuous:
        cont_env = ContinuousWorld(
            size=level, num_fires=num_fires_level,
            wall_density=ENV["wall_density"],
            view_size=view_size, d_max=ENV["d_max"],
            seed=seed + exp_id,
        )
        options_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ENV["options_dir"]))
        device = MAPPO_BASE["device"]
        dispatcher = OptionDispatcher(
            cont_env, options_dir=options_dir,
            option_timeout=ENV["option_timeout"], device=device,
        )
        obs_dim = dispatcher.obs_dim
        global_dim = dispatcher.global_dim
        env = dispatcher
    else:
        obs_dim, _ = discrete_get_obs_dim(level, view_size=view_size)
        global_dim = discrete_get_global_dim(level)
        env = RescueEnvPZ(
            size=level, max_cycles=max_steps,
            num_fires=num_fires_level,
            wall_density=ENV["wall_density"],
            random_walls=ENV["random_walls"],
            view_size=view_size,
            seed=seed + exp_id,
        )

    # ── Create MAPPO ──────────────────────────────────────────────────
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

    
    start_wall_time = time.time()   # se non già definito sopra

    for episode in range(1, num_episodes + 1):
        # Timer check
        if max_wall_time_seconds is not None:
            elapsed = time.time() - start_wall_time
            if elapsed > max_wall_time_seconds:
                print(f"\n  *** WALL-TIME TIMEOUT reached at episode {episode-1} "
                    f"({elapsed:.0f}s / {max_wall_time_seconds}s) ***")
                print(f"  Stopping training and saving current state.")
                break
        # Generate concrete map at TOP level (always 100) for Option A
        if top_level == 100:
            concrete_map = generate_continuous_map(
                size=top_level,
                num_fires=ENV["num_fires_per_level"][top_level],
                wall_density=ENV["wall_density"],
                random_walls=ENV["random_walls"],
                rng=map_rngs[episode - 1],
            )
            # Continuous map has float positions; convert to integer cells
            # for the abstract_layer (which expects discrete cells at the top)
            concrete_map_discrete = {
                "wall_cells": concrete_map["wall_cells"],
                "fire_cells": concrete_map["fire_cells"],
                "agent_starts": [tuple(int(np.floor(p[0])) for p in [concrete_map["agent_starts"][0]]) + tuple([int(np.floor(concrete_map["agent_starts"][0][1]))]),
                                  tuple([int(np.floor(concrete_map["agent_starts"][1][0]))]) + tuple([int(np.floor(concrete_map["agent_starts"][1][1]))])],
                "item_pos": (int(np.floor(concrete_map["item_pos"][0])),
                             int(np.floor(concrete_map["item_pos"][1]))),
                "victim_pos": (int(np.floor(concrete_map["victim_pos"][0])),
                               int(np.floor(concrete_map["victim_pos"][1]))),
                "size": top_level,
            }
            # Re-construct agent_starts cleanly
            concrete_map_discrete["agent_starts"] = [
                (int(np.floor(concrete_map["agent_starts"][0][0])),
                 int(np.floor(concrete_map["agent_starts"][0][1]))),
                (int(np.floor(concrete_map["agent_starts"][1][0])),
                 int(np.floor(concrete_map["agent_starts"][1][1]))),
            ]
            concrete_for_abstraction = concrete_map_discrete
        else:
            concrete_for_abstraction = generate_concrete_map(
                size=top_level,
                num_fires=ENV["num_fires_per_level"][top_level],
                wall_density=ENV["wall_density"],
                random_walls=ENV["random_walls"],
                rng=map_rngs[episode - 1],
            )
            concrete_map = concrete_for_abstraction

        abstraction.update(concrete_for_abstraction)

        # Reset env with the appropriate map
        if is_continuous:
            obs_list = env.reset(concrete_map=concrete_map)
            obs = {"agent_0": obs_list[0], "agent_1": obs_list[1]}
        else:
            obs, _ = env.reset(concrete_map=concrete_for_abstraction)

        ep_reward = 0.0
        ep_reward_biased = 0.0
        outcome = "timeout"
        steps_count = 0

        # Trivial-success check for very coarse levels
        if not is_continuous:
            had_item_init = (env._env.agent1_has_item or env._env.agent2_has_item)
            if had_item_init:
                holder = env._env.agent1_pos if env._env.agent1_has_item else env._env.agent2_pos
                if env._env._is_adjacent(holder, env._env.victim_pos):
                    env._env.done = True
                    env._env.outcome = "rescued"
                    env.agents = []
                    outcome = "rescued"

        # ── Rollout loop ─────────────────────────────────────────────
        while True:
            if is_continuous:
                env_active = not env.env.done
            else:
                env_active = bool(env.agents)
            if not env_active:
                break
            if steps_count >= max_steps:
                break
            steps_count += 1

            if is_continuous:
                obs_list = list(env.env.get_observations())
                global_state = env.get_global_state_compact()
                old_pos = [env.env.agent_pos[0].copy(),
                           env.env.agent_pos[1].copy()]
                had_item_before = (env.env.agent_has_item[0]
                                    or env.env.agent_has_item[1])
            else:
                obs_list = [obs["agent_0"], obs["agent_1"]]
                global_state = env._env.get_global_state()
                old_pos_0 = env._env.agent1_pos
                old_pos_1 = env._env.agent2_pos
                had_item_before = (env._env.agent1_has_item
                                    or env._env.agent2_has_item)

            if use_shaping:
                actions, log_probs, val_b, val_u = agent.select_actions(
                    obs_list, global_state)
            else:
                actions, log_probs, val_b = agent.select_actions(
                    obs_list, global_state)
                val_u = None

            # ── Apply actions ───────────────────────────────────────
            if is_continuous:
                _, rewards_pair, env_done, info = env.step(actions)
                new_pos = [env.env.agent_pos[0].copy(),
                           env.env.agent_pos[1].copy()]
                has_item_after = (env.env.agent_has_item[0]
                                   or env.env.agent_has_item[1])
                if info.get("outcome") == "rescued":
                    outcome = "rescued"
                done = env_done
                r0_base, r1_base = rewards_pair[0], rewards_pair[1]
            else:
                actions_dict = {"agent_0": actions[0], "agent_1": actions[1]}
                next_obs, rewards, terms, truncs, infos = env.step(actions_dict)
                obs = next_obs
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

            # ── Compute shaping if needed ───────────────────────────
            if use_shaping:
                if is_continuous:
                    s0 = abstraction.compute_shaping_continuous(
                        new_pos[0], level, has_item_after,
                        differential=diff_shaping,
                        prev_pos_continuous=old_pos[0],
                        prev_has_item=had_item_before,
                        scale=scale)
                    s1 = abstraction.compute_shaping_continuous(
                        new_pos[1], level, has_item_after,
                        differential=diff_shaping,
                        prev_pos_continuous=old_pos[1],
                        prev_has_item=had_item_before,
                        scale=scale)
                else:
                    s0 = abstraction.compute_shaping(
                        new_pos_0, level, has_item_after,
                        differential=diff_shaping,
                        prev_pos=old_pos_0, prev_has_item=had_item_before,
                        scale=scale)
                    s1 = abstraction.compute_shaping(
                        new_pos_1, level, has_item_after,
                        differential=diff_shaping,
                        prev_pos=old_pos_1, prev_has_item=had_item_before,
                        scale=scale)
                r0_b = r0_base + s0
                r1_b = r1_base + s1
                agent.add_to_buffer(
                    obs=obs_list, global_state=global_state,
                    actions=actions, log_probs=log_probs,
                    rewards_biased=[r0_b, r1_b],
                    value_biased=val_b, done=done,
                    rewards_unbiased=[r0_base, r1_base],
                    value_unbiased=val_u)
                ep_reward_biased += (r0_b + r1_b) / 2
                ep_reward += (r0_base + r1_base) / 2
            else:
                agent.add_to_buffer(
                    obs=obs_list, global_state=global_state,
                    actions=actions, log_probs=log_probs,
                    rewards_biased=[r0_base, r1_base],
                    value_biased=val_b, done=done)
                ep_reward += (r0_base + r1_base) / 2
                ep_reward_biased = ep_reward

            if done:
                break

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
            extra = f" | R_bias: {ep_reward_biased:7.2f}" if use_shaping else ""
            print(f"  Exp{exp_id} L{level} Ep {episode:5d} | "
                  f"R: {ep_reward:7.2f}{extra} | "
                  f"SR_cum: {sr_cum:5.1%} | "
                  f"SR_last100: {sr_last100:5.1%} | "
                  f"out: {outcome}")

    # Save outputs
    np.save(os.path.join(save_dir, "sr.npy"), np.array(sr_log))
    np.save(os.path.join(save_dir, "rewards.npy"), np.array(rewards_hist))
    if use_shaping:
        np.save(os.path.join(save_dir, "rewards_biased.npy"),
                np.array(rewards_biased_hist))
    agent.save(os.path.join(save_dir, "policy.pt"))

    plot_single_experiment(exp_id, exp_label, level, use_shaping,
                            rewards_hist, rewards_biased_hist, sr_log,
                            save_dir, log_every)

    final_sr = total_successes / num_episodes
    sr_last_window = (np.mean(sr_log[-10:]) if len(sr_log) >= 10
                      else (sr_log[-1] if sr_log else 0))
    print(f"\n  Experiment {exp_id} complete.")
    print(f"  Final SR (last 10 logs): {sr_last_window:.1%}")
    verdict = ("SUCCESS" if sr_last_window >= CONVERGENCE["success_threshold"]
               else "FAILURE" if sr_last_window <= CONVERGENCE["failure_threshold"]
               else "PARTIAL")
    print(f"  Verdict: {verdict}\n")
    return {
        "exp_id": exp_id, "label": exp_label, "level": level,
        "use_shaping": use_shaping,
        "rewards": rewards_hist, "rewards_biased": rewards_biased_hist,
        "sr_log": sr_log, "final_sr": final_sr,
        "final_sr_window": sr_last_window, "verdict": verdict,
    }


def plot_single_experiment(exp_id, exp_label, level, dual,
                            rew, rew_b, sr_log, save_dir, log_every):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Exp {exp_id}: {exp_label}", fontsize=13, fontweight="bold")
    window = min(100, len(rew))
    if window > 0:
        sm = np.convolve(rew, np.ones(window)/window, mode="valid")
        axes[0].plot(sm, color="steelblue", linewidth=1.2, label="base reward")
        if dual and len(rew_b) > 0:
            sm_b = np.convolve(rew_b, np.ones(window)/window, mode="valid")
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
    out_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'█' * 72}")
    print(f"  HIERARCHICAL TRAINING — v4 PHASE 2")
    print(f"  Seed: {seed} | Pyramid: {LEVELS}")
    print(f"  {2*len(LEVELS)-1} experiments")
    print(f"{'█' * 72}")

    abstraction = HierarchicalAbstraction(
        levels=LEVELS, gamma_VI=HIERARCHY["gamma_VI"])
    max_episodes = max(TRAIN["episodes_per_level"].values())
    map_pool_full = generate_map_pool(max_episodes, seed)
    print(f"\n  Pre-generated {max_episodes} concrete map seeds.\n")

    results = []
    exp_id = 1

    # Sparse experiments (descent: top to base)
    for level in sorted(LEVELS, reverse=True):
        n_eps = TRAIN["episodes_per_level"][level]
        res = train_experiment(
            exp_id=exp_id,
            exp_label=f"L{level} - SPARSE (no shaping)",
            level=level, use_shaping=False,
            map_rngs=map_pool_full[:n_eps],
            seed=seed, abstraction=abstraction,
            save_dir=os.path.join(out_dir, f"exp_{exp_id}_L{level}_sparse"),
            log_every=TRAIN["log_every"])
        results.append(res)
        exp_id += 1

    # Hierarchical experiments (ascent: base+1 to top)
    for level in sorted(LEVELS)[1:]:
        n_eps = TRAIN["episodes_per_level"][level]
        upper = abstraction.get_upper_level(level)
        res = train_experiment(
            exp_id=exp_id,
            exp_label=f"L{level} - HIERARCHICAL (V*_{upper})",
            level=level, use_shaping=True,
            map_rngs=map_pool_full[:n_eps],
            seed=seed, abstraction=abstraction,
            save_dir=os.path.join(out_dir, f"exp_{exp_id}_L{level}_hier"),
            log_every=TRAIN["log_every"])
        results.append(res)
        exp_id += 1

    plot_comparison(results, out_dir, seed)
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
    fig, ax = plt.subplots(figsize=(14, 8))
    n_levels = len(LEVELS)
    sparse_count = n_levels
    sparse_cmap = plt.cm.YlOrRd_r
    hier_cmap = plt.cm.cool
    log_every = TRAIN["log_every"]
    for i, r in enumerate(results):
        sr = r["sr_log"]
        if not sr:
            continue
        x = np.arange(log_every, len(sr) * log_every + 1, log_every)
        if i < sparse_count:
            color = sparse_cmap(0.2 + 0.6 * i / max(1, sparse_count - 1))
            linestyle = "--"
        else:
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
        f"Phase 2 — Pyramid {LEVELS} (seed {seed})\n"
        f"L3-L50 discrete | L100 continuous via options",
        fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (last 100 episodes)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison.png"), dpi=180)
    plt.close()
    print(f"  Comparison plot saved -> {os.path.join(out_dir, 'comparison.png')}")

def run_only_exp11(seed, save_dir, max_wall_time_seconds=None):
    """Lancia direttamente Exp 11 (L100 hier con shaping da V*_50)."""
    out_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    abstraction = HierarchicalAbstraction(
        levels=LEVELS, gamma_VI=HIERARCHY["gamma_VI"])
    max_episodes = max(TRAIN["episodes_per_level"].values())
    map_pool_full = generate_map_pool(max_episodes, seed)

    level = 100
    n_eps = TRAIN["episodes_per_level"][level]
    upper = abstraction.get_upper_level(level)   # = 50

    print(f"\nLaunching Exp 11 (L{level} hier with V*_{upper} shaping)")
    print(f"Seed: {seed} | Episodes: {n_eps}")
    if max_wall_time_seconds:
        print(f"Wall-time timer: {max_wall_time_seconds}s "
              f"({max_wall_time_seconds/3600:.1f}h)\n")

    res = train_experiment(
        exp_id=11,
        exp_label=f"L{level} - HIERARCHICAL (V*_{upper})",
        level=level, use_shaping=True,
        map_rngs=map_pool_full[:n_eps],
        seed=seed, abstraction=abstraction,
        save_dir=os.path.join(out_dir, f"exp_11_L{level}_hier"),
        log_every=TRAIN["log_every"],
        max_wall_time_seconds=max_wall_time_seconds,
    )

    print(f"\nDone. Final SR: {res['final_sr_window']:.1%} ({res['verdict']})")
    print(f"Final SR: {res['final_sr_window']:.1%} ({res['verdict']})")

def run_only_exp10(seed, save_dir, max_wall_time_seconds=None,
                   override_episodes=None):
    """Lancia Exp 10 (L50 hier con shaping da V*_25).
    
    Sanity check: se Exp 10 (discreto, più piccolo) converge bene,
    il framework gerarchico funziona e Exp 11 (continuo) ha solo
    bisogno di più tempo.
    """
    out_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    abstraction = HierarchicalAbstraction(
        levels=LEVELS, gamma_VI=HIERARCHY["gamma_VI"])
    max_episodes = max(TRAIN["episodes_per_level"].values())
    map_pool_full = generate_map_pool(max_episodes, seed)

    level = 50
    n_eps = override_episodes if override_episodes else TRAIN["episodes_per_level"][level]
    upper = abstraction.get_upper_level(level)   # = 25

    print(f"\nLaunching Exp 10 (L{level} hier with V*_{upper} shaping) — SANITY CHECK")
    print(f"Seed: {seed} | Episodes: {n_eps}")
    if max_wall_time_seconds:
        print(f"Wall-time timer: {max_wall_time_seconds}s "
              f"({max_wall_time_seconds/3600:.1f}h)\n")

    res = train_experiment(
        exp_id=10,
        exp_label=f"L{level} - HIERARCHICAL (V*_{upper})",
        level=level, use_shaping=True,
        map_rngs=map_pool_full[:n_eps],
        seed=seed, abstraction=abstraction,
        save_dir=os.path.join(out_dir, f"exp_10_L{level}_hier"),
        log_every=TRAIN["log_every"],
        max_wall_time_seconds=max_wall_time_seconds,
    )

    print(f"\nDone. Final SR: {res['final_sr_window']:.1%} ({res['verdict']})")

def run_selected_experiments(seed, save_dir, exp_ids_to_run,
                              skip_completed=True,
                              max_wall_time_per_exp=None,
                              max_wall_time_total=None):
    """
    Run only the specified experiments by ID, in order.

    Args:
        seed: RNG seed
        save_dir: base directory
        exp_ids_to_run: iterable of experiment IDs to run (e.g. [10, 11])
        skip_completed: skip experiments whose folder already has training.png
        max_wall_time_per_exp: seconds; each experiment will be interrupted
                               if it exceeds this. Passed to train_experiment
                               (requires that train_experiment supports it).
        max_wall_time_total: seconds; the overall loop will stop if this is
                             exceeded, even if experiments remain.

    Experiment ID mapping for pyramid [3, 6, 12, 25, 50, 100]:
      1: L100 sparse    5: L6  sparse    9: L25  hier (V*_12)
      2: L50  sparse    6: L3  sparse   10: L50  hier (V*_25)
      3: L25  sparse    7: L6  hier (V*_3)   11: L100 hier (V*_50)
      4: L12  sparse    8: L12 hier (V*_6)
    """
    out_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'█' * 72}")
    print(f"  SELECTED EXPERIMENTS MODE")
    print(f"  Seed: {seed}")
    print(f"  Experiments requested: {sorted(exp_ids_to_run)}")
    if max_wall_time_per_exp:
        print(f"  Per-exp wall-time budget: {max_wall_time_per_exp}s "
              f"({max_wall_time_per_exp/3600:.1f}h)")
    if max_wall_time_total:
        print(f"  Total wall-time budget:   {max_wall_time_total}s "
              f"({max_wall_time_total/3600:.1f}h)")
    print(f"{'█' * 72}\n")

    exp_ids_set = set(exp_ids_to_run)

    # Build the full experiment table (must match run_full_pipeline order)
    all_exps = []
    exp_id = 1
    for level in sorted(LEVELS, reverse=True):
        all_exps.append({
            "exp_id": exp_id, "level": level, "use_shaping": False,
            "label": f"L{level} - SPARSE (no shaping)",
            "folder": f"exp_{exp_id}_L{level}_sparse",
        })
        exp_id += 1
    for level in sorted(LEVELS)[1:]:
        abstraction_tmp = HierarchicalAbstraction(
            levels=LEVELS, gamma_VI=HIERARCHY["gamma_VI"])
        upper = abstraction_tmp.get_upper_level(level)
        all_exps.append({
            "exp_id": exp_id, "level": level, "use_shaping": True,
            "label": f"L{level} - HIERARCHICAL (V*_{upper})",
            "folder": f"exp_{exp_id}_L{level}_hier",
        })
        exp_id += 1

    invalid = exp_ids_set - {e["exp_id"] for e in all_exps}
    if invalid:
        print(f"ERROR: Invalid experiment IDs: {sorted(invalid)}")
        print(f"Valid IDs: 1-{len(all_exps)}")
        return

    selected = [e for e in all_exps if e["exp_id"] in exp_ids_set]

    abstraction = HierarchicalAbstraction(
        levels=LEVELS, gamma_VI=HIERARCHY["gamma_VI"])
    max_episodes = max(TRAIN["episodes_per_level"].values())
    map_pool_full = generate_map_pool(max_episodes, seed)
    log_every = TRAIN["log_every"]

    loop_start = time.time()

    for exp in selected:
        # Global timer check
        if max_wall_time_total is not None:
            elapsed = time.time() - loop_start
            if elapsed > max_wall_time_total:
                print(f"\n  *** TOTAL WALL-TIME BUDGET EXHAUSTED "
                      f"({elapsed:.0f}s / {max_wall_time_total}s) ***")
                print(f"  Stopping loop. Remaining experiments not executed.")
                break

        n_eps = TRAIN["episodes_per_level"][exp["level"]]
        save_subdir = os.path.join(out_dir, exp["folder"])

        if skip_completed and os.path.exists(
                os.path.join(save_subdir, "training.png")):
            print(f"\n  [SKIP] Exp {exp['exp_id']} ({exp['label']}) "
                  f"— already completed")
            continue

        # Compute remaining time budget for this specific experiment
        exp_time_budget = max_wall_time_per_exp
        if max_wall_time_total is not None:
            remaining_total = max_wall_time_total - (time.time() - loop_start)
            if exp_time_budget is None:
                exp_time_budget = remaining_total
            else:
                exp_time_budget = min(exp_time_budget, remaining_total)

        print(f"\n{'▓' * 72}")
        print(f"  Running Exp {exp['exp_id']}: {exp['label']}")
        print(f"  Episodes: {n_eps} | Output: {exp['folder']}")
        if exp_time_budget is not None:
            print(f"  This exp budget: {exp_time_budget:.0f}s "
                  f"({exp_time_budget/3600:.1f}h)")
        print(f"{'▓' * 72}")

        train_experiment(
            exp_id=exp["exp_id"],
            exp_label=exp["label"],
            level=exp["level"],
            use_shaping=exp["use_shaping"],
            map_rngs=map_pool_full[:n_eps],
            seed=seed, abstraction=abstraction,
            save_dir=save_subdir,
            log_every=log_every,
            max_wall_time_seconds=exp_time_budget,   # ← nuovo parametro
        )

    total_elapsed = time.time() - loop_start
    print(f"\n{'█' * 72}")
    print(f"  Selected experiments loop complete for seed {seed}")
    print(f"  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/3600:.2f}h)")
    print(f"{'█' * 72}\n")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    SAVE_DIR = os.path.join(PROJECT_ROOT, TRAIN["save_dir"])

    # ═══════════════════════════════════════════════════════════════
    #  MODE: choose one of:
    #    "full"     — run all 11 experiments in the pipeline
    #    "exp11"    — run only Experiment 11 (L100 hier)
    #    "selected" — run a chosen subset of experiments
    # ═══════════════════════════════════════════════════════════════
    MODE = "selected"

    if MODE == "full":
        for seed in TRAIN["seeds"]:
            run_full_pipeline(seed=seed, save_dir=SAVE_DIR)

    elif MODE == "exp11":
        for seed in TRAIN["seeds"]:
            run_only_exp11(
                seed=seed, save_dir=SAVE_DIR,
                max_wall_time_seconds=22 * 3600,
            )

    elif MODE == "selected":
        # Customize which experiments and which seeds
        SEEDS_TO_RUN = [42]          # ← change as needed
        EXPS_TO_RUN  = [10]   # ← all except Exp 11
        # Some other examples:
        # EXPS_TO_RUN = [11]                    # only Exp 11
        # EXPS_TO_RUN = [7, 8, 9, 10, 11]       # only the hierarchical ones
        # EXPS_TO_RUN = [10, 11]                # only L50 and L100 hier
        # EXPS_TO_RUN = [1, 2, 3]               # only the top sparse ones

        for seed in SEEDS_TO_RUN:
            run_selected_experiments(
                seed=seed,
                save_dir=SAVE_DIR,
                exp_ids_to_run=EXPS_TO_RUN,
                skip_completed=True,  # skip if training.png exists
                max_wall_time_per_exp=14 * 3600,   # 18h per exp max
                max_wall_time_total=1000 * 3600,     # 40h totali per seed
            )

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    print("\nAll runs complete.")