"""
config.py - Phase 2 configuration: pyramid [3, 6, 12, 25, 50, 100].

The pyramid has SIX levels:
    - L3 to L50: DISCRETE (uses env_discrete/RescueEnvPZ, 5 grid actions)
    - L100: CONTINUOUS (uses continuous_env + option_dispatcher,
            5 macro actions = pre-trained options U/R/D/L/NOP)

11 experiments total (6 sparse + 5 hierarchical, excluding L3 hier).
"""

LEVELS = [3, 6, 12, 25, 50, 100]
CONTINUOUS_LEVEL = 100   # the level that uses the option dispatcher


ENV = {
    "levels": LEVELS,
    "max_steps_per_level": {
        3:   30,
        6:   80,
        12:  250,
        25:  500,
        50:  500,
        100: 300,    # macro-steps (each is a full option execution, ~5-30 micro)
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

    # Continuous-specific
    "d_max": 0.2,
    "option_timeout": 20,
    "options_dir": "../trained_options",
}


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
    "entropy_coef": 0.08,
    "max_grad_norm": 0.5,
    "device": "cuda",
}


HIERARCHY = {
    "shaping_scale": 25.0,
    "differential_shaping": True,
    "gamma_VI": 0.97,
}


TRAIN = {
    "episodes_per_level": {
        3:    2000,
        6:    5000,
        12:   10000,
        25:   15000,
        50:   20000,
        100:  30000,    # macro-episodes; conservative initial value
    },
    "seeds": [42],   # start with one seed; add more later if Phase 1 succeeds
    "log_every": 100,
    "save_dir": "results_v3_1run_L100_one_hot/",
}


CONVERGENCE = {
    "success_threshold": 0.80,
    "failure_threshold": 0.30,
}
