"""
config.py — DQN hyperparameters for the L100 hierarchical experiment.

Kept parallel to macro/config.py for clean PPO vs DQN comparison.
"""

LEVELS = [3, 6, 12, 25, 50, 100]
CONTINUOUS_LEVEL = 100


ENV = {
    "levels": LEVELS,
    "max_steps_per_level": {
        3:   30,
        6:   80,
        12:  250,
        25:  500,
        50:  500,
        100: 300,
    },
    "view_size_per_level": {
        3:  3,
        6:  5,
        12: 7,
        25: 9,
        50: 11,
        100: 11,
    },
    "num_fires_per_level": {
        3:   0,
        6:   1,
        12:  3,
        25:  5,
        50:  10,
        100: 15,
    },
    "wall_density": 0.08,
    "random_walls": True,

    "d_max": 0.2,
    "option_timeout": 20,
    "options_dir": "../trained_options",
}


DQN_BASE = {
    "action_dim":           5,
    "hidden_dim":           128,
    "lr":                   3e-4,
    "gamma":                0.99,
    "buffer_capacity":      500_000,
    "batch_size":           128,
    "target_update_freq":   1000,
    "eps_start":            1.0,
    "eps_end":              0.15,
    "eps_decay_steps":      10_000_000,
    "min_buffer_before_train": 2000,
    "max_grad_norm":        10.0,
    "device":               "cuda",
}


HIERARCHY = {
    "shaping_scale":       25.0,
    "differential_shaping": True,
    "gamma_VI":            0.97,
}


TRAIN = {
    "episodes_per_level": {
        3:    2000,
        6:    5000,
        12:   10000,
        25:   15000,
        50:   20000,
        100:  30000,
    },
    "seeds":       [42],
    "log_every":   100,
    "updates_per_env_step": 1,   # how many gradient updates per env step
    "save_dir":    "results_dqn/",
}


CONVERGENCE = {
    "success_threshold": 0.80,
    "failure_threshold": 0.30,
}