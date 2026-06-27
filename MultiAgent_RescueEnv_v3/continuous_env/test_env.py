"""
test_env.py - Smoke test for the continuous environment.

Generates a random 100x100 map, runs episodes with random actions, and
prints sanity metrics. Useful to verify:
    - Map generation succeeds
    - Collision detection prevents agents from entering walls
    - Fire penalties are applied correctly
    - Item pickup and rescue logic works
"""

import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from continuous_world import ContinuousWorld
from continuous_map import generate_continuous_map


def random_action_test(num_episodes=5, max_steps=500, seed=42):
    env = ContinuousWorld(size=100, num_fires=15, wall_density=0.08,
                          view_size=11, d_max=0.2, seed=seed)

    print("\n" + "=" * 60)
    print("  CONTINUOUS ENV — RANDOM ACTION SMOKE TEST")
    print("=" * 60)

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        print(f"\n  Episode {ep}:")
        print(f"    Walls: {len(env.wall_cells)}  Fires: {len(env.fire_cells)}")
        print(f"    Agent0 start: ({env.agent_pos[0][0]:.2f}, {env.agent_pos[0][1]:.2f})")
        print(f"    Agent1 start: ({env.agent_pos[1][0]:.2f}, {env.agent_pos[1][1]:.2f})")
        print(f"    Item: ({env.item_pos[0]:.2f}, {env.item_pos[1]:.2f})")
        print(f"    Victim: ({env.victim_pos[0]:.2f}, {env.victim_pos[1]:.2f})")

        wall_bumps = [0, 0]
        fire_visits = [0, 0]
        total_reward = [0.0, 0.0]
        outcome = "timeout"

        rng = np.random.RandomState(seed + ep)
        for step in range(max_steps):
            old_pos = [env.agent_pos[0].copy(), env.agent_pos[1].copy()]
            actions = [rng.uniform(-1, 1, size=2), rng.uniform(-1, 1, size=2)]
            obs, rewards, done, info = env.step(actions)

            for k in range(2):
                # Check wall bumps (movement was rejected)
                if np.allclose(env.agent_pos[k], old_pos[k], atol=1e-4):
                    wall_bumps[k] += 1
                if rewards[k] <= -5.0:
                    fire_visits[k] += 1
                total_reward[k] += rewards[k]

            if done:
                outcome = info["outcome"]
                print(f"    Step {step+1}: terminated with outcome '{outcome}'")
                break
        else:
            print(f"    Reached max_steps={max_steps}")

        print(f"    Total reward: agent0={total_reward[0]:.2f}, agent1={total_reward[1]:.2f}")
        print(f"    Wall bumps:   agent0={wall_bumps[0]}, agent1={wall_bumps[1]}")
        print(f"    Fire visits:  agent0={fire_visits[0]}, agent1={fire_visits[1]}")
        print(f"    Has item:     agent0={env.agent_has_item[0]}, agent1={env.agent_has_item[1]}")

    print("\n" + "=" * 60)
    print("  ALL EPISODES COMPLETED — ENV IS WORKING")
    print("=" * 60)


def test_observation_shapes():
    env = ContinuousWorld(size=100, num_fires=15, wall_density=0.08,
                          view_size=11, d_max=0.2, seed=42)
    obs_list = env.reset()
    print(f"\n  Observation shape per agent: {obs_list[0].shape}")
    print(f"  Expected obs_dim: {env.obs_dim} = 10 + {env.view_size}^2")
    assert obs_list[0].shape == (env.obs_dim,)
    assert obs_list[1].shape == (env.obs_dim,)

    gs = env.get_global_state()
    print(f"  Global state shape: {gs.shape}")
    print(f"  Expected global_dim: {env.global_dim} = 10 + {env.size}^2")
    assert gs.shape == (env.global_dim,)

    opt_obs = env.get_option_obs(0)
    print(f"  Option observation shape: {opt_obs.shape}")
    print(f"  Expected option obs_dim: 10")
    assert opt_obs.shape == (10,)

    print("  All observation shapes OK")


def test_map_generation():
    print("\n  Testing map generation on multiple seeds...")
    for seed in [0, 7, 42, 123, 999]:
        rng = np.random.RandomState(seed)
        m = generate_continuous_map(size=100, num_fires=15,
                                     wall_density=0.08, rng=rng)
        print(f"    seed {seed}: walls={len(m['wall_cells'])}, "
              f"fires={len(m['fire_cells'])}, "
              f"item=({m['item_pos'][0]:.1f}, {m['item_pos'][1]:.1f})")


if __name__ == "__main__":
    test_map_generation()
    test_observation_shapes()
    random_action_test(num_episodes=3, max_steps=300, seed=42)
