"""
visualize_option.py - Pygame viewer for a trained option.

Loads a saved option checkpoint and runs episodes with random wall
configurations and starting positions, rendering the agent's trajectory.

Keyboard:
  SPACE   pause
  N       next episode
  R       restart current episode
  +/-     speed up/slow down
  ESC     quit

Edit CHECKPOINT_PATH at the bottom to visualize different options.
"""

import sys
import os
import numpy as np
import torch
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from option_env import OptionEnv, DIRECTIONS, TARGET_CELL, CENTER
from option_policy import OptionActor


CELL_PIXELS = 200       # each cell is 200x200 pixels
GRID_SIZE = 3
PANEL_HEIGHT = 100

# Colors
BG       = (245, 240, 225)
GRID_COL = (180, 175, 165)
WALL_COL = (60, 60, 60)
CENTER_COL = (220, 230, 245)
TARGET_COL = (180, 240, 180)
AGENT_COL = (40, 100, 200)
TRAJ_COL = (40, 100, 200, 90)
PANEL_BG = (30, 30, 50)
PANEL_TXT = (235, 235, 245)


def cell_to_pixel(r, c):
    return (c * CELL_PIXELS, r * CELL_PIXELS)


def pos_to_pixel(r, c):
    """Continuous (r, c) to pixel (x, y) on screen."""
    return (c * CELL_PIXELS, r * CELL_PIXELS)


def run_visualization(checkpoint_path, direction, num_episodes=20,
                     wall_prob=0.2, frame_delay_ms=150, seed=None):
    print(f"Loading option pi_{direction} from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = OptionActor(obs_dim=10, action_dim=2, hidden_dim=64)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env = OptionEnv(direction=direction, d_max=0.2,
                    wall_prob=wall_prob, max_steps=80, seed=seed)

    pygame.init()
    screen_w = CELL_PIXELS * GRID_SIZE
    screen_h = CELL_PIXELS * GRID_SIZE + PANEL_HEIGHT
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(f"Option pi_{direction}")
    font_big = pygame.font.SysFont("Arial", 22, bold=True)
    font_small = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()

    running = True
    ep_num = 0
    paused = False

    while running and ep_num < num_episodes:
        ep_num += 1
        obs = env.reset()
        trajectory = [tuple(env.pos)]
        steps = 0
        total_reward = 0.0
        outcome = None
        ep_done = False

        while running and not ep_done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_n:
                        ep_done = True
                    elif event.key == pygame.K_r:
                        ep_done = True
                        ep_num -= 1
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        frame_delay_ms = max(20, frame_delay_ms - 50)
                    elif event.key == pygame.K_MINUS:
                        frame_delay_ms = min(2000, frame_delay_ms + 50)

            if paused:
                draw(screen, env, trajectory, ep_num, steps, total_reward,
                     outcome, paused, font_big, font_small, direction)
                pygame.time.wait(50)
                continue

            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = actor.get_deterministic_action(obs_t)
            obs, reward, done, info = env.step(action)
            trajectory.append(tuple(env.pos))
            steps += 1
            total_reward += reward
            if done:
                outcome = info["outcome"]
                ep_done = True

            draw(screen, env, trajectory, ep_num, steps, total_reward,
                 outcome, paused, font_big, font_small, direction)
            pygame.time.wait(frame_delay_ms)

        # End-of-episode pause
        if running and outcome is not None:
            print(f"Episode {ep_num}: {outcome}, reward={total_reward:.2f}, steps={steps}")
            for _ in range(20):
                draw(screen, env, trajectory, ep_num, steps, total_reward,
                     outcome, paused, font_big, font_small, direction)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                if not running:
                    break
                pygame.time.wait(50)

    pygame.quit()


def draw(screen, env, trajectory, ep_num, steps, total_reward, outcome,
         paused, font_big, font_small, direction):
    screen.fill(BG)

    # Draw cells
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            px, py = cell_to_pixel(i, j)
            color = BG
            if (i, j) in env.walls:
                color = WALL_COL
            elif (i, j) == env.target:
                color = TARGET_COL
            elif (i, j) == CENTER:
                color = CENTER_COL
            pygame.draw.rect(screen, color, (px, py, CELL_PIXELS, CELL_PIXELS))

    # Grid lines
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(screen, GRID_COL,
                         (0, i * CELL_PIXELS),
                         (CELL_PIXELS * GRID_SIZE, i * CELL_PIXELS), 2)
        pygame.draw.line(screen, GRID_COL,
                         (i * CELL_PIXELS, 0),
                         (i * CELL_PIXELS, CELL_PIXELS * GRID_SIZE), 2)

    # Draw trajectory as polyline
    if len(trajectory) > 1:
        points = [(c * CELL_PIXELS, r * CELL_PIXELS) for r, c in trajectory]
        pygame.draw.lines(screen, (40, 100, 200), False, points, 3)

    # Draw agent
    px, py = trajectory[-1][1] * CELL_PIXELS, trajectory[-1][0] * CELL_PIXELS
    pygame.draw.circle(screen, AGENT_COL, (int(px), int(py)), 12)
    pygame.draw.circle(screen, (255, 255, 255), (int(px), int(py)), 12, 2)

    # Draw target arrow on target cell
    target_px = env.target[1] * CELL_PIXELS + CELL_PIXELS // 2
    target_py = env.target[0] * CELL_PIXELS + CELL_PIXELS // 2
    pygame.draw.circle(screen, (50, 150, 50), (target_px, target_py), 20, 4)

    # Panel
    py = CELL_PIXELS * GRID_SIZE
    pygame.draw.rect(screen, PANEL_BG, (0, py, CELL_PIXELS * GRID_SIZE, PANEL_HEIGHT))
    info_line = f"pi_{direction}  Ep {ep_num}  Step {steps}  R: {total_reward:+.2f}"
    if outcome:
        col = (100, 220, 120) if outcome == "success" else (255, 100, 100)
        info_line += f"   >>> {outcome.upper()} <<<"
    else:
        col = PANEL_TXT
    screen.blit(font_big.render(info_line, True, col), (16, py + 14))
    if paused:
        screen.blit(font_big.render("PAUSED", True, (100, 220, 120)),
                    (CELL_PIXELS * GRID_SIZE - 130, py + 14))
    ctrl = "SPACE pause  N next  R restart  +/- speed  ESC quit"
    screen.blit(font_small.render(ctrl, True, (160, 160, 175)),
                (16, py + PANEL_HEIGHT - 24))
    pygame.display.flip()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    # Configure here
    DIRECTION = "R"      # one of "U", "R", "D", "L"
    WALL_PROB = 0.2      # wall probability during visualization
    NUM_EPISODES = 20    # episodes to show
    FRAME_MS = 100       # speed (lower = faster)

    CKPT_PATH = os.path.join(PROJECT_ROOT, "trained_options", f"pi_{DIRECTION}.pt")

    if not os.path.exists(CKPT_PATH):
        print(f"ERROR: Checkpoint not found at {CKPT_PATH}")
        print("Train the options first by running:")
        print("  python -m options.train_options")
        sys.exit(1)

    run_visualization(
        checkpoint_path=CKPT_PATH,
        direction=DIRECTION,
        num_episodes=NUM_EPISODES,
        wall_prob=WALL_PROB,
        frame_delay_ms=FRAME_MS,
        seed=None,
    )
