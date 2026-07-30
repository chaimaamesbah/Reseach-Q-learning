# Constraint Programming-Guided Reinforcement Learning in the Stochastic FrozenLake Environment

## Project Description

This project investigates the effectiveness of using Constraint Programming (CP) and Belief Propagation (BP) to generate reward shaping signals for Q-learning agents operating in the stochastic FrozenLake-v1 environment. The primary objective is to enhance learning efficiency and final policy quality in scenarios characterized by sparse rewards and uncertain state transitions.

Two novel CP-based reward shaping methods are introduced and evaluated:
* **CP-Marginal Strength (CP-MS) Shaping:** Leverages action marginal probabilities derived from the CP model.
* **CP-Expected Total Reward (CP-ETR) Shaping:** Uses the CP model's estimate of the expected total future reward.

These methods are systematically compared against several baselines:
* Standard Q-learning (no shaping)
* Q-learning with classic potential-based shaping (Manhattan distance)
* A CP-MS Greedy heuristic agent (a non-learning, planning-based approach)

The evaluation is conducted across various 4x4 and 8x8 slippery FrozenLake instances, designed to offer a range of complexities.

A **budget mode** further extends the CP-ETR approach by introducing deterministic (no-slip) actions alongside the standard stochastic ones, subject to a per-episode budget constraint managed by a curriculum.

## Repository Contents

* **`main.py`**: The main script to configure and run a single experiment.
* **`run_experiment.py`**: Batch experiment runner — launches `main.py` across all (instance, method, seed) combinations, caches results, and generates plots and summary tables.
* **`environment.py`**: Handles creation and management of FrozenLake environment instances, including the 8-action budget wrapper.
* **`q_learning_standard.py`**: Implements standard Q-learning and Q-learning with classic potential-based shaping.
* **`q_learning_cp.py`**: Implements Q-learning agents with CP-MS and CP-ETR reward shaping, including the socket client for communication with the CP server, the augmented state mapper, and no-slip curriculum strategies.
* **`heuristic_agents.py`**: Implements the CP-MS Greedy agent and an Optimal policy agent.
* **`config.py`**: Stores hyperparameters for Q-learning algorithms and evaluation parameters.
* **`utils.py`**: Contains utility functions for saving/loading results, agent evaluation, and plotting learning curves.
* **`instances.json`**: Defines configurations for different FrozenLake instances (e.g., size, hole patterns, custom maps, CP model parameters).
* **`MiniCPBP/`**: Directory containing the Java-based Constraint Programming server (utilizing MiniCPBP) that provides marginal probabilities and expected total reward values to the Python agents. Includes a `pom.xml` for Maven build.
* **`experiment_results/`**: Learning curves and summary CSV from reported experiments.

## How to Run

### Prerequisites

* Python 3.9+
* Java Development Kit (JDK 11 or compatible)
* Apache Maven (`mvn` in your PATH)

```bash
pip install -r requirements.txt
```

### Compile the Java CP server (once)

```bash
cd MiniCPBP && mvn compile && cd ..
```

### Running a single experiment

```bash
python main.py --instance <INSTANCE_ID> --agent <AGENT_TYPE> --shaping <SHAPING_METHOD> --episodes <NUM_EPISODES> --seed <SEED_VALUE>
```

#### Key arguments

* `--instance <INSTANCE_ID>`: Required. Instance ID from `instances.json` (e.g., `4s`, `4medium`, `4hard`, `8s`, `8medium`, `8hard`).
* `--agent <AGENT_TYPE>`:
    * `q`: Q-learning agent. Must be accompanied by `--shaping`.
    * `cp_greedy`: CP-MS Greedy heuristic agent (no learning).
    * `optimal`: Agent following a pre-defined optimal policy from `instances.json`.
* `--shaping <SHAPING_METHOD>`: Required if `--agent q`.
    * `none`: Standard Q-learning (no shaping).
    * `classic`: Classic potential-based shaping (Manhattan distance).
    * `cp-ms`: CP-Marginal Strength shaping.
    * `cp-etr`: CP-Expected Total Reward shaping.
* `--budget <N>`: Max no-slip actions per episode (0 = disabled, standard 4-action mode).
* `--noslip-strategy`: `fail` (curriculum, default) or `full-budget`.
* `--episodes <N>`: Number of training episodes (default: 500). Ignored for non-learning agents.
* `--seed <N>`: Random seed for reproducibility.

#### Example commands

```bash
# Q-learning with CP-ETR shaping
python main.py --instance 4hard --agent q --shaping cp-etr --episodes 5000 --seed 1

# Q-learning with CP-ETR shaping + budget mode (3 no-slip actions max)
python main.py --instance 4s --agent q --shaping cp-etr --budget 3 --episodes 5000 --seed 1

# CP-MS Greedy heuristic agent
python main.py --instance 4s --agent cp_greedy --seed 1

# Standard Q-learning baseline
python main.py --instance 8s --agent q --shaping none --episodes 5000 --seed 1
```

### Running batch experiments

```bash
# Full experiment (default: 6 instances × 5 methods × 20 seeds × 5000 episodes)
python run_experiment.py

# Custom run
python run_experiment.py \
  --instances 4s 4hard \
  --methods q-none q-cp-etr cp-greedy \
  --seeds 5 \
  --episodes 2000 \
  --workers 4

# Regenerate plots and summary table from cached results without re-running
python run_experiment.py --plots-only
```

Results are cached in `experiment_results/cache/` after each run. The summary CSV and learning curve plots are written to `experiment_results/`.

## Key Findings (Summary)

* **CP-ETR Shaping for Q-learning:** Generally proves to be the most robust and effective learning method among those tested, achieving stable convergence to near-optimal policies on several 4x4 and 8x8 instances.
* **CP-MS Greedy Heuristic:** This non-learning agent often demonstrates near-optimal performance, highlighting the strong guidance potential directly extractable from the CP-BP model at each step. It serves as a powerful benchmark.
* **CP-MS Shaping for Q-learning:** Can accelerate initial learning but sometimes converges to suboptimal policies, particularly on more complex maps.
* **Classic Potential-Based Shaping:** Shows unreliable and mixed performance. The simplistic nature of the Manhattan distance heuristic limits its general applicability in complex layouts.
* **Standard Q-Learning:** Performs poorly across all tested instances within the training budget, underscoring the difficulty of learning in sparse and stochastic environments without effective guidance.
* **Computational Considerations:** CP-based methods exhibit good sample efficiency but incur higher wall-clock time per episode due to the overhead of communication with the Java CP server and CP-BP computation.

This project demonstrates that integrating lookahead information from CP-BP models, particularly through the CP-ETR shaping mechanism, offers a promising approach to guide Q-learning agents effectively in stochastic grid-world environments.
