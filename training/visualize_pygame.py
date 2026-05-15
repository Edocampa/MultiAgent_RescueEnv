"""
Pygame demo of a trained policy

Auto-detects grid size from the checkpoint dimensions
Loads the unbiased policy by default

Keyboard:
  SPACE pause   N next   R restart   +/- speed   ESC quit

Default: visualizes the level-10 hierarchical policy 
Edit CHECKPOINT_PATH at the bottom to visualize other experiments
"""

import sys
import os
import numpy as np
import torch
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "env"))

from rescue_env_pz import RescueEnvPZ
from networks import Actor
from config import ENV, MAPPO_BASE, HIERARCHY


def get_cell_size(size):
    return {3: 130, 5: 100, 10: 64}.get(size, 64)


PANEL_HEIGHT = 120
BG_COLOR        = (250, 246, 232)
GRID_LINE       = (200, 195, 180)
WALL_COLOR      = (90, 90, 90)
FIRE_COLOR      = (255, 69, 0)
ITEM_COLOR      = (220, 20, 60)
VICTIM_COLOR    = (255, 215, 0)
ROBOT1_COLOR    = (70, 130, 180)
ROBOT2_COLOR    = (200, 50, 50)
PANEL_BG        = (30, 30, 50)
PANEL_TEXT      = (235, 235, 245)
HIGHLIGHT_GREEN = (100, 220, 120)
HIGHLIGHT_RED   = (255, 100, 100)
HAS_ITEM_GLOW   = (255, 215, 0)


def load_sprite(name, asset_dir, target_size):
    path = os.path.join(asset_dir, name)
    if not os.path.exists(path):
        return None
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, target_size)
    except Exception:
        return None


class PygameRenderer:
    def __init__(self, env, asset_dir):
        self.env = env
        self.size = env._env.size
        self.cell_size = get_cell_size(self.size)
        self.grid_pixels = self.size * self.cell_size
        self.window_w = self.grid_pixels
        self.window_h = self.grid_pixels + PANEL_HEIGHT

        pygame.init()
        pygame.display.set_caption(f"Multi-Agent Rescue - Level {self.size}")
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        self.font_big = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_med = pygame.font.SysFont("Arial", 18)
        self.font_small = pygame.font.SysFont("Arial", 14)

        sprite_size = int(self.cell_size * 0.85)
        self.sprites = {
            "robot1": load_sprite("robot.png", asset_dir, (sprite_size, sprite_size)),
            "robot2": load_sprite("robot.png", asset_dir, (sprite_size, sprite_size)),
            "kit":    load_sprite("kit.png",    asset_dir, (int(sprite_size * 0.8), int(sprite_size * 0.8))),
            "victim": load_sprite("victim.png", asset_dir, (sprite_size, sprite_size)),
            "fire":   load_sprite("fire.png",   asset_dir, (sprite_size, sprite_size)),
            "wall":   load_sprite("wall.png",   asset_dir, (self.cell_size, self.cell_size)),
        }

    def cell_center(self, r, c):
        return (c * self.cell_size + self.cell_size // 2,
                r * self.cell_size + self.cell_size // 2)

    def cell_pixel(self, r, c):
        return (c * self.cell_size, r * self.cell_size)

    def blit(self, sprite, r, c):
        if sprite is None:
            return False
        rect = sprite.get_rect()
        rect.center = self.cell_center(r, c)
        self.screen.blit(sprite, rect)
        return True

    def render(self, step, total_reward, paused=False, info_text=""):
        env = self.env._env
        self.screen.fill(BG_COLOR)
        for i in range(self.size + 1):
            pygame.draw.line(self.screen, GRID_LINE,
                             (i * self.cell_size, 0),
                             (i * self.cell_size, self.grid_pixels), 1)
            pygame.draw.line(self.screen, GRID_LINE,
                             (0, i * self.cell_size),
                             (self.grid_pixels, i * self.cell_size), 1)
        for (r, c) in env.wall_cells:
            if not self.blit(self.sprites["wall"], r, c):
                px, py = self.cell_pixel(r, c)
                pygame.draw.rect(self.screen, WALL_COLOR,
                                 (px, py, self.cell_size, self.cell_size))
        for (r, c) in env.fire_cells:
            if not self.blit(self.sprites["fire"], r, c):
                pygame.draw.circle(self.screen, FIRE_COLOR,
                                   self.cell_center(r, c), self.cell_size // 3)
        if not env.agent1_has_item and not env.agent2_has_item:
            r, c = env.item_pos
            if not self.blit(self.sprites["kit"], r, c):
                cx, cy = self.cell_center(r, c)
                pygame.draw.rect(self.screen, ITEM_COLOR,
                                 (cx - 18, cy - 18, 36, 36))
        r, c = env.victim_pos
        if not self.blit(self.sprites["victim"], r, c):
            pygame.draw.circle(self.screen, VICTIM_COLOR,
                               self.cell_center(r, c), self.cell_size // 3)
        r, c = env.agent1_pos
        if not self.blit(self.sprites["robot1"], r, c):
            pygame.draw.circle(self.screen, ROBOT1_COLOR,
                               self.cell_center(r, c), self.cell_size // 3)
        if env.agent1_has_item:
            cx, cy = self.cell_center(r, c)
            self._star(cx, cy - self.cell_size // 2 + 8, 8)
        r, c = env.agent2_pos
        if not self.blit(self.sprites["robot2"], r, c):
            pygame.draw.circle(self.screen, ROBOT2_COLOR,
                               self.cell_center(r, c), self.cell_size // 3)
        if env.agent2_has_item:
            cx, cy = self.cell_center(r, c)
            self._star(cx, cy - self.cell_size // 2 + 8, 8)
        self._panel(step, total_reward, paused, info_text)
        pygame.display.flip()

    def _star(self, cx, cy, radius):
        import math
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = radius if i % 2 == 0 else radius * 0.5
            x = cx + r * math.cos(angle)
            y = cy - r * math.sin(angle)
            points.append((x, y))
        pygame.draw.polygon(self.screen, HAS_ITEM_GLOW, points)
        pygame.draw.polygon(self.screen, (0, 0, 0), points, 1)

    def _panel(self, step, total_reward, paused, info_text):
        py = self.grid_pixels
        pygame.draw.rect(self.screen, PANEL_BG,
                         (0, py, self.window_w, PANEL_HEIGHT))
        col = PANEL_TEXT
        if "RESCUED" in info_text.upper():
            col = HIGHLIGHT_GREEN
        elif "FIRE" in info_text.upper() or "TIMEOUT" in info_text.upper():
            col = HIGHLIGHT_RED
        line = f"Step: {step}    Reward: {total_reward:+.2f}"
        if info_text:
            line += f"    >>> {info_text} <<<"
        self.screen.blit(self.font_big.render(line, True, col), (16, py + 12))
        if paused:
            self.screen.blit(self.font_med.render("PAUSED", True, HIGHLIGHT_GREEN),
                             (self.window_w - 100, py + 14))
        controls = "SPACE pause  N next  R restart  +/- speed  ESC quit"
        self.screen.blit(self.font_small.render(controls, True, (160, 160, 175)),
                         (16, py + PANEL_HEIGHT - 24))

    def quit(self):
        pygame.quit()


def load_actors(checkpoint_path, prefer_unbiased=True):
    print(f"Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "actors_unbiased" in ckpt and prefer_unbiased:
        actor_dicts = ckpt["actors_unbiased"]
        print("  Loaded UNBIASED policy (Algorithm 1 final)")
    elif "actors_biased" in ckpt:
        actor_dicts = ckpt["actors_biased"]
        print("  Loaded BIASED policy")
    elif "actor_0" in ckpt:
        actor_dicts = [ckpt["actor_0"], ckpt["actor_1"]]
        print("  Loaded v1-style policy")
    else:
        raise ValueError("Unknown checkpoint format")

    first_layer = actor_dicts[0]["network.0.weight"]
    hidden_dim, obs_dim = first_layer.shape
    if obs_dim == 35:
        gd = ckpt.get("global_dim", 110)
        size = 10 if gd == 110 else 5
    elif obs_dim == 19:
        size = 3
    else:
        size = 10
    print(f"  Detected: obs_dim={obs_dim}, hidden_dim={hidden_dim}, size={size}")

    a0 = Actor(obs_dim=obs_dim, action_dim=5, hidden_dim=hidden_dim)
    a1 = Actor(obs_dim=obs_dim, action_dim=5, hidden_dim=hidden_dim)
    a0.load_state_dict(actor_dicts[0])
    a1.load_state_dict(actor_dicts[1])
    a0.eval()
    a1.eval()
    return a0, a1, size


def run_demo(checkpoint_path, asset_dir, frame_delay_ms=300, seed=None):
    a0, a1, size = load_actors(checkpoint_path)
    max_cycles = {10: 200, 5: 100, 3: 50}.get(size, 200)
    env = RescueEnvPZ(  size=size,
                        max_cycles=max_cycles,
                        num_fires=ENV["num_fires"],
                        random_walls=ENV.get("random_walls", False),     
                        num_random_walls=ENV.get("num_random_walls", 8),
                        seed=seed)
    renderer = PygameRenderer(env, asset_dir)

    ep_num = 0
    running = True

    while running:
        ep_num += 1
        print(f"\n--- Episode {ep_num} ---")
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0
        paused = False
        outcome = None
        ep_done = False

        renderer.render(0, 0.0, info_text="START")
        pygame.time.wait(800)

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
                        frame_delay_ms = max(50, frame_delay_ms - 50)
                    elif event.key == pygame.K_MINUS:
                        frame_delay_ms = min(2000, frame_delay_ms + 50)

            if not running:
                break

            if not paused and env.agents:
                step += 1
                o0 = torch.FloatTensor(obs["agent_0"])
                o1 = torch.FloatTensor(obs["agent_1"])
                with torch.no_grad():
                    act0 = torch.argmax(a0(o0.unsqueeze(0)), dim=-1).item()
                    act1 = torch.argmax(a1(o1.unsqueeze(0)), dim=-1).item()
                obs, rewards, terms, truncs, infos = env.step({
                    "agent_0": act0, "agent_1": act1
                })
                total_reward += (rewards.get("agent_0", 0) + rewards.get("agent_1", 0)) / 2
                info = infos.get("agent_0", {})
                if info.get("outcome") in ("rescued", "fire"):
                    outcome = info["outcome"]
                if not env.agents and outcome is None:
                    outcome = "timeout"
                renderer.render(step, total_reward, paused=paused)
                pygame.time.wait(frame_delay_ms)
            else:
                renderer.render(step, total_reward, paused=paused)
                pygame.time.wait(50)

        if running and outcome is not None:
            msg = {"rescued": "RESCUED!", "fire": "FIRE",
                   "timeout": "TIMEOUT"}.get(outcome, outcome)
            print(f"Episode {ep_num} ended: {msg}, reward={total_reward:.2f}, steps={step}")
            for _ in range(40):
                renderer.render(step, total_reward, info_text=msg)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                if not running:
                    break
                pygame.time.wait(50)

    renderer.quit()
    print("\nDemo ended.")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    # Default: experiment 5 (level 10 hierarchical)
    SEED = 42
    EXPERIMENT = "exp_5_level10_hier"

    CHECKPOINT_PATH = os.path.join(
        PROJECT_ROOT, "results", f"seed_{SEED}", EXPERIMENT, "policy.pt"
    )
    ASSET_DIR = os.path.join(PROJECT_ROOT, "assets")

    print(f"Project: {PROJECT_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print()

    run_demo(checkpoint_path=CHECKPOINT_PATH, asset_dir=ASSET_DIR,
             frame_delay_ms=300, seed=42)
