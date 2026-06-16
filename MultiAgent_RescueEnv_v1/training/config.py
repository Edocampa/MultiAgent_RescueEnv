"""
Single source of truth for the hierarchical training

The narrative:
    Exp 1: Level 10 SPARSE         -> expected to fail
    Exp 2: Level  5 SPARSE         -> expected to fail
    Exp 3: Level  3 SPARSE         -> expected to succeed (base case)
    Exp 4: Level  5 + V*_3 shaping -> expected to succeed
    Exp 5: Level 10 + V*_5 shaping -> expected to succeed (final goal)

All 5 experiments share the same map seed list, so episode K of any
experiment uses the same underlying 10x10 physical map
"""

# Per-level environment settings
ENV = {
    "max_steps_per_level": {
        3:  50,    # 3x3 quick episodes
        5:  100,   # 5x5 mid-length
        10: 200,   # full task
    },
    "num_fires": 2,    #in the concrete 10x10; auto-handled at coarse levels (result_1)
    #"num_fires": 3, #(results_difficult)

    # Wall Logic
    "random_walls": False,    # False = Result 1 (Fissi), True = Difficult (Casuali)
    "num_random_walls": 8,    # Numero di muri se random_walls è True
}

# MAPPO base hyperparameters
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
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "device": "cuda",
}

# Hierarchy configuration
HIERARCHY = {
    # Multiplier for the V* shaping signal.
    # IMPORTANT: V values are in [0, 1] so raw differences are tiny (~0.01-0.05
    # per step). Task rewards are +5 (pickup), +10 (rescue), -5 (fire).
    # Without scaling up, the shaping is invisible noise to the optimizer.
    # Empirical sweet spot: ~5.0 makes per-step shaping comparable to wall
    # penalties (-0.3) without dominating pickup/rescue.
    "shaping_scale": 10.0,           # was 1.0, increased after first iteration
    # shaping_scale: 15.0 (results_difficult)

    # Use canonical PBRS: F = scale * (gamma * V(s') - V(s))
    "differential_shaping": True,
}

# Training settings
TRAIN = {
    # Episodes per level. Same number across sparse/hierarchical runs of
    # the same level so the final comparison plot has aligned x-axes
    "episodes_per_level": {
        3:  2000,    # base case: small grid, converges fast
        5:  10000,    # intermediate
        10: 15000,   # full task: needs many episodes
    },

    "seeds": [123],   # one seed for now; add more after verifying convergence

    "log_every": 100,
    "save_dir": "results_baseline/results_4/",    # result 1 ---> top (entropia 0.08 + gamma 0.8), result 2: entropia 0.05, result 3: entropia 0.01, reslt 4: 0.01 entropia + 0.85 gamma
}

# Convergence criteria
CONVERGENCE = {
    # SR_last100 averaged over the last K episodes:
    "success_threshold": 0.80,   # >= 80% sustained = "solved"
    "failure_threshold": 0.30,   # <= 30% throughout = "failed"
    # Anything in between is reported as "partial"
}
