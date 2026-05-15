# Multi-Agent Hierarchical RL — Thesis Framework

Implementation of the hierarchical abstraction layer based on a paper for single-agent, extended to a Multi-Agent Reinforcement Learning (MARL) setting. 
The project implements **Dual Learner**: training MAPPO across three levels of abstraction, where the $V^*$ (Optimal Value Function) of a coarser level provides a dense **Reward Shaping** signal to the level below.

---

## 1. Core Concept: The Hierarchy of Truth
Training MARL with sparse rewards on a $10 \times 10$ grid often fails due to the "curse of dimensionality" and signal sparsity. I solve this by creating a hierarchy of the same physical environment:

* **Level 10 (Concrete):** The $10 \times 10$ reality. High complexity, sparse rewards.
* **Level 5 (Intermediate):** A $5 \times 5$ abstraction.
* **Level 3 (Coarse/Base Case):** A $3 \times 3$ macro-view. Computationally easy; MAPPO converges quickly here.

**Shaping Flow:** The $V^*$ computed at Level 3 guides Level 5. The $V^*$ computed at Level 5 guides the final Level 10.

---

## 2. Technical Implementation Details

### A. Geometric Mapping & Projection Rules
The environment uses a hybrid projection logic to transform the $10 \times 10$ map into coarser representations:
* **$10 \to 5$ (Uniform):** Each $2 \times 2$ block of concrete cells maps to 1 abstract cell.
    * *Rule:* **Any-is-blocked**. If even one concrete cell contains a wall or fire, the macro-cell is blocked. (Safety-first approach).
* **$5 \to 3$ (Non-Uniform):** Uses a center-focused mapping `[0, 0, 1, 2, 2]`.
    * *Rule:* **Majority-is-blocked**. A macro-cell is blocked only if $\geq 50\%$ of its underlying cells are obstacles. (Prevents "clogging" the coarse map).

### B. Reward Shaping Formula (PBRS)
We use the **Potential-Based Reward Shaping (PBRS)** framework (Ng et al., 1999):
$$F(s, s') = \text{scale} \cdot (\gamma \cdot \Phi(s') - \Phi(s))$$

* **Dual Gamma Logic:**
    * $\gamma_{Agent} = 0.99$: Used in the shaping formula for mathematical policy invariance.
    * $\gamma_{VI} = 0.80$: Used within **Value Iteration** to generate a steep, clear potential gradient (the "compass").
* **Scaling:** We use a `shaping_scale = 10.0` (Static) or `15.0` (Procedural) to ensure the signal is numerically significant for the Neural Network.

### C. The Dual Learner 
To ensure theoretical convergence while benefiting from "biased" guidance:
* **Biased Learner:** Learns from *Environment Reward + Shaping*. It drives exploration.
* **Unbiased Learner:** Learns only from *Environment Reward*. It acts as a "guarantor," ensuring the final policy is optimal even if the abstract map was slightly inaccurate.

---

## 3. The Thesis Narrative: 5 Experiments
The script `train_hierarchical.py` runs a sequential pipeline to demonstrate the "Enabling Hierarchy":

1.  **Exp 1: Level 10 SPARSE** $\to$ **Fails** (~40-50% SR). Sparse reward is insufficient.
2.  **Exp 2: Level 5 SPARSE** $\to$ **Fails** (~60% SR). Still too complex for pure exploration.
3.  **Exp 3: Level 3 SPARSE** $\to$ **Succeeds** (~95% SR). The base case is solved.
4.  **Exp 4: Level 5 + $V^*_3$ Shaping** $\to$ **Succeeds** (~85% SR). Guided by the Level 3 base case.
5.  **Exp 5: Level 10 + $V^*_5$ Shaping** $\to$ **Succeeds** (~75-90% SR). The final goal.

---

## 4. Environment Modes
We test the robustness of the system in two distinct scenarios:

| Mode | Wall Geometry | Success Rate | Scientific Meaning |
| :--- | :--- | :--- | :--- |
| **Result 1** | **Static** (Fixed) | ~90% | Navigation in a known environment with dynamic hazards. |
| **Difficult** | **Procedural** (Random) | ~72% | Pure Generalization. Solving unseen labyrinths on the fly. |

---

## 5. File Structure
* `env/map_generator.py`: Generates maps and handles BFS connectivity/projections.
* `env/grid_world.py`: The core environment with local/global observations.
* `training/mappo.py`: Multi-Agent PPO implementation with Dual Learner logic.
* `training/abstract_layer.py`: Computes $V^*$ via Value Iteration for shaping.
* `training/config.py`: The "Single Source of Truth" for all hyperparameters.
* `visualize_pygame.py`: Real-time demo of trained policies.

---
