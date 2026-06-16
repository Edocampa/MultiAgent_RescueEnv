"""
config.py - Phase 1 configuration (24×24 with 4 levels).

The pyramid is defined by LEVELS. To experiment with different scales,
change LEVELS and adjust the per-level parameters.

The levels must be in ASCENDING order. levels[0] is the base case
(no shaping), levels[-1] is the final concrete level.
"""

# ── Pyramid of levels ─────────────────────────────────────────────────
# Phase 1: 24x24 with 4 levels (rapporti 2x ad ogni step)
# Phase 2 (future): [3, 6, 12, 24, 48, 96] for the full scale
LEVELS = [3, 6, 12, 24]


# ── Environment settings ──────────────────────────────────────────────
ENV = {
    "levels": LEVELS,

    # Max steps per level. Scale with grid diameter (~2 * N steps minimum
    # for an agent to traverse).
    "max_steps_per_level": {
        3:   50,    # tiny grid
        6:   150,
        12:  300,
        24:  700,   # full task
    },

    # View size per level. Capped at the grid size for small grids.
    # On 24x24, view 11x11 covers 21% of the map (similar density to 5x5
    # on 10x10). Without this, agent is blind on large grids.
    "view_size_per_level": {
        3:  3,
        6:  5,
        12: 7,
        24: 11,
    },

    # Fires per level (scaled to grid)
    "num_fires_per_level": {
        3:   0,
        6:   1,
        12:  3,
        24:  7,
    },

    # Wall density (fraction of cells)
    "wall_density": 0.10,
    "random_walls": True,
}


# ── MAPPO base hyperparameters ────────────────────────────────────────
MAPPO_BASE = {
    "num_agents":   2,
    "action_dim":   5,
    "hidden_dim": 128,
    "lr_actor":  3e-4,
    "lr_critic": 5e-4,
    "gamma":      0.99,
    "gae_lambda": 0.95,
    "clip_eps":   0.2,
    "epochs":     10,
    "batch_size": 64,
    "entropy_coef": 0.08,     # high entropy, same as v2 final iteration
    "max_grad_norm": 0.5,
    "device": "cuda",         # GPU enabled (change to "cpu" if no GPU)
}


# ── Hierarchy / shaping configuration ─────────────────────────────────
HIERARCHY = {
    "shaping_scale": 20.0,        # large scale to overcome reward signal
    "differential_shaping": True,
    "gamma_VI": 0.8,             # discount for V* computation (NOT MAPPO gamma)
                                   # Lower than MAPPO gamma to make V* gradient steeper
}


# ── Training settings ─────────────────────────────────────────────────
TRAIN = {
    # Episodes per level. Larger grids need much more training.
    # These are starting estimates; tune after first run.
    "episodes_per_level": {
        3:   2000,
        6:   10000,
        12:  20000,
        24:  30000,    # the big one
    },

    "seeds": [42],
    "log_every": 100,
    "save_dir": "results_24x24_2run/",
}


# ── Convergence thresholds ────────────────────────────────────────────
CONVERGENCE = {
    "success_threshold": 0.80,
    "failure_threshold": 0.30,
}
