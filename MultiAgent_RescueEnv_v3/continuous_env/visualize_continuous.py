"""
visualize_continuous.py - Pygame viewer for the continuous environment.

Shows the 100x100 grid with walls (gray), fires (red), item (gold square),
victim (gold circle), and agents as moving balls. Runs episodes with random
actions so you can see the dynamics.

Keyboard:
  SPACE pause
  N next episode
  ESC quit
"""

import sys
import os
import numpy as np
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from continuous_world import ContinuousWorld, WALL, FIRE


# Display settings — auto-scale based on grid size
def get_cell_size(grid_size):
    target_window = 800
    return max(4, target_window // grid_size)


PANEL_HEIGHT = 100
BG       = (245, 240, 225)
GRID_COL = (220, 215, 200)
WALL_COL = (70, 70, 70)
FIRE_COL = (220, 60, 30)
ITEM_COL = (240, 180, 50)
VICTIM_COL = (240, 200, 30)
AGENT_COLORS = [(40, 110, 220), (220, 70, 70)]
TRAJ_COLORS = [(40, 110, 220, 100), (220, 70, 70, 100)]
PANEL_BG = (28, 28, 48)
PANEL_TXT = (235, 235, 245)


def run_visualization(num_episodes=5, max_steps=1000, frame_delay_ms=20,
                      size=100, seed=42):
    env = ContinuousWorld(size=size, num_fires=15, wall_density=0.08,
                          view_size=11, d_max=0.2, seed=seed)
    cell_size = get_cell_size(size)
    grid_pixels = size * cell_size

    pygame.init()
    screen_w = grid_pixels
    screen_h = grid_pixels + PANEL_HEIGHT
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(f"Continuous Rescue — Size {size}")
    font_big = pygame.font.SysFont("Arial", 20, bold=True)
    font_small = pygame.font.SysFont("Arial", 12)

    running = True
    ep_num = 0

    while running and ep_num < num_episodes:
        ep_num += 1
        env.reset()
        trajectories = [[tuple(env.agent_pos[0])], [tuple(env.agent_pos[1])]]
        total_reward = [0.0, 0.0]
        outcome = None
        paused = False

        rng = np.random.RandomState(seed + ep_num)

        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_n:
                        step = max_steps   # exit inner loop

            if not running or step >= max_steps:
                break

            if paused:
                draw(screen, env, trajectories, ep_num, step, total_reward,
                     outcome, paused, font_big, font_small, cell_size, grid_pixels)
                pygame.time.wait(50)
                continue

            actions = [rng.uniform(-1, 1, size=2),
                       rng.uniform(-1, 1, size=2)]
            obs, rewards, done, info = env.step(actions)
            trajectories[0].append(tuple(env.agent_pos[0]))
            trajectories[1].append(tuple(env.agent_pos[1]))
            total_reward[0] += rewards[0]
            total_reward[1] += rewards[1]

            if done:
                outcome = info["outcome"]

            draw(screen, env, trajectories, ep_num, step, total_reward,
                 outcome, paused, font_big, font_small, cell_size, grid_pixels)
            pygame.time.wait(frame_delay_ms)

            if done:
                # Pause briefly to show terminal state
                for _ in range(40):
                    draw(screen, env, trajectories, ep_num, step,
                         total_reward, outcome, paused, font_big,
                         font_small, cell_size, grid_pixels)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                    if not running:
                        break
                    pygame.time.wait(40)
                break

    pygame.quit()


def draw(screen, env, trajectories, ep_num, step, total_reward, outcome,
         paused, font_big, font_small, cell_size, grid_pixels):
    screen.fill(BG)

    # Draw walls and fires
    for r, c in env.wall_cells:
        pygame.draw.rect(screen, WALL_COL,
                         (c * cell_size, r * cell_size, cell_size, cell_size))
    for r, c in env.fire_cells:
        pygame.draw.rect(screen, FIRE_COL,
                         (c * cell_size, r * cell_size, cell_size, cell_size))

    # Item (gold square)
    ir = int(env.item_pos[0] * cell_size)
    ic = int(env.item_pos[1] * cell_size)
    s = max(4, cell_size)
    if not (env.agent_has_item[0] or env.agent_has_item[1]):
        pygame.draw.rect(screen, ITEM_COL,
                         (ic - s // 2, ir - s // 2, s, s))

    # Victim (gold circle)
    vr = int(env.victim_pos[0] * cell_size)
    vc = int(env.victim_pos[1] * cell_size)
    pygame.draw.circle(screen, VICTIM_COL, (vc, vr), max(4, cell_size // 2))

    # Trajectories (thin)
    for k in range(2):
        if len(trajectories[k]) > 1:
            pts = [(int(c * cell_size), int(r * cell_size))
                   for r, c in trajectories[k]]
            pygame.draw.lines(screen, AGENT_COLORS[k], False, pts, 1)

    # Agents
    for k in range(2):
        ar = int(env.agent_pos[k][0] * cell_size)
        ac = int(env.agent_pos[k][1] * cell_size)
        rad = max(4, cell_size)
        pygame.draw.circle(screen, AGENT_COLORS[k], (ac, ar), rad)
        pygame.draw.circle(screen, (255, 255, 255), (ac, ar), rad, 2)
        if env.agent_has_item[k]:
            # Star indicator
            pygame.draw.circle(screen, ITEM_COL, (ac, ar - rad - 2),
                                max(3, cell_size // 2))

    # Panel
    py = grid_pixels
    pygame.draw.rect(screen, PANEL_BG, (0, py, grid_pixels, PANEL_HEIGHT))
    col = PANEL_TXT
    if outcome == "rescued":
        col = (100, 220, 120)
    line = (f"Ep {ep_num}  Step {step}  "
            f"R: agent0={total_reward[0]:+.1f}, agent1={total_reward[1]:+.1f}")
    if outcome:
        line += f"   >>> {outcome.upper()} <<<"
    screen.blit(font_big.render(line, True, col), (16, py + 14))
    if paused:
        screen.blit(font_big.render("PAUSED", True, (100, 220, 120)),
                    (grid_pixels - 110, py + 14))
    ctrl = "SPACE pause  N next  ESC quit"
    screen.blit(font_small.render(ctrl, True, (160, 160, 175)),
                (16, py + PANEL_HEIGHT - 22))
    pygame.display.flip()


if __name__ == "__main__":
    # Visualize a smaller grid by default (faster, easier to see).
    # Set SIZE = 100 to see the full target scale.
    SIZE = 50
    run_visualization(num_episodes=5, max_steps=2000,
                      frame_delay_ms=10, size=SIZE, seed=42)
