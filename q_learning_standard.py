# FL/q_learning_standard.py
"""
Q-learning agents: Standard (no shaping) and Classic Potential-Based Shaping.
"""
import numpy as np
import random
import logging
import datetime
import os
from tqdm import tqdm

import config
import utils


def _get_hyperparameters():
    return (
        config.EPSILON,
        config.LEARNING_RATE,
        config.DISCOUNT_FACTOR,
        config.EPSILON_DECAY,
        config.EPSILON_MIN,
    )


def _log(msg: str, verbose: int, min_level: int = 1, logger: logging.Logger = None):
    if verbose >= min_level:
        tqdm.write(msg)
    if logger:
        logger.info(msg)


def _maybe_save_qtable(q_table, shaping_method_name, instance_id, total_episodes, episode,
                       verbose: int, run_dir: str):
    """Sauvegarde la Q-table CSV uniquement si verbose >= 2."""
    if verbose < 2:
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    qtables_dir = os.path.join(run_dir, "qtables")
    os.makedirs(qtables_dir, exist_ok=True)
    q_filename = os.path.join(
        qtables_dir,
        f"q_{shaping_method_name}_{instance_id}_{total_episodes}eps_{episode}eval_{timestamp}.csv"
    )
    utils.save_q_table_csv(q_table, q_filename)


def train_q_learning(env, total_episodes, max_steps, size, holes, goal,
                     shaping_method_name, instance_id,
                     verbose: int = 0, run_dir: str = "results", logger: logging.Logger = None):
    """Standard Q-learning."""
    _log(f"Starting standard Q-learning: {total_episodes} episodes", verbose, logger=logger)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.full((state_size, action_size), config.Q_INIT_VALUE_NONE)

    epsilon, lr, gamma_discount, eps_decay, eps_min = _get_hyperparameters()
    episode_log, evaluation_log = [], []

    episode_iter = tqdm(range(total_episodes), desc="Training", leave=False, dynamic_ncols=True) \
        if verbose >= 1 else range(total_episodes)

    for episode in episode_iter:
        state, info = env.reset()
        episode_steps = 0
        env_reward_sum = 0.0
        shaped_reward_sum = 0.0
        final_reward = 0.0
        done = False
        terminated = False
        truncated = False

        for step_idx in range(max_steps):
            episode_steps += 1
            current_state_for_update = state
            if not (0 <= state < state_size):
                _log(f"ERROR: Invalid state {state} Ep {episode + 1}/St {step_idx}.", verbose, logger=logger)
                break

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                max_val = np.max(q_table[state])
                best_actions = np.where(q_table[state] == max_val)[0]
                action = int(np.random.choice(best_actions))

            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as err:
                if hasattr(env, "goal_state_idx"):
                    raise
                _log(f"ERROR: env.step exception: {err}.", verbose, logger=logger)
                break

            if not (0 <= next_state < state_size):
                _log(f"ERROR: Invalid next_state {next_state} Ep {episode + 1}/St {step_idx}.", verbose, logger=logger)
                break

            reward_used_for_update = reward
            old_q = q_table[current_state_for_update, action]

            best_next_action_q_value = np.max(q_table[next_state])
            td_target = reward_used_for_update + gamma_discount * best_next_action_q_value * (1 - int(done))
            td_error = td_target - q_table[current_state_for_update, action]
            q_table[current_state_for_update, action] += lr * td_error

            # if episode + 1 in {1, 2, 390, 400, 430}:
            #     action_names = ["Attendre", "Est", "Nord", "Sud", "Ouest"]

            #     print(f"\nPolitique à l'épisode {episode + 1}")

            #     for s in range(state_size):
            #         best_action = int(np.argmax(q_table[s]))
            #         print(
            #             f"État {s:2d} : "
            #             f"{action_names[best_action]:8s} "
            #             f"Q = {q_table[s, best_action]:.4f}"
            #         )

                # tqdm.write(f"\nÉpisode {episode + 1} | Étape {step_idx + 1}")
                # tqdm.write(f"État : {current_state_for_update} → {next_state}")
                # tqdm.write(f"Action : {action_names[action]}")
                # tqdm.write(f"Récompense : {reward}")
                # tqdm.write(
                #     f"Q[{current_state_for_update}, {action}] : "
                #     f"{old_q:.3f} → {q_table[current_state_for_update, action]:.3f}"
                # )

                # tqdm.write("\n          Attendre     Est     Nord      Sud    Ouest")
                # for s in range(state_size):
                #     values = " ".join(f"{v:8.3f}" for v in q_table[s])
                #     tqdm.write(f"État {s:2d} : {values}")

                # tqdm.write("-" * 65)

            shaped_reward_sum += reward_used_for_update
            env_reward_sum += info.get("raw_reward", reward)
            state = next_state
            if done:
                final_reward = reward
                break

        success = int(utils.is_success(env, state, terminated, final_reward))
        episode_log.append({
            'episode': episode + 1, 'steps': episode_steps,
            'env_reward': env_reward_sum, 'shaped_reward': shaped_reward_sum,
            'success': success
        })
        epsilon = config.decay_epsilon(epsilon)

        if (episode + 1) % config.EVAL_FREQUENCY == 0 or (episode + 1) == total_episodes:
            # Evaluate CP-trained policies using the original RDDL rewards.
            if hasattr(env, "set_evaluation_mode"):
                env.set_evaluation_mode(True)
            try:
                sr, ar_undisc, ar_disc, avg_ss, avg_sf = utils.evaluate_agent(
                    env, q_table, max_steps, config.EVAL_EPISODES)
            finally:
                if hasattr(env, "set_evaluation_mode"):
                    env.set_evaluation_mode(False)
            evaluation_log.append({
                'training_episode': episode + 1, 'eval_success_rate': sr,
                'eval_avg_return': ar_undisc, 'eval_avg_discounted_return': ar_disc,
                'avg_steps_success': avg_ss, 'avg_steps_failure': avg_sf
            })
            _log(f"\n--- Evaluation at Episode {episode + 1}/{total_episodes} (Standard Q) ---", verbose, logger=logger)
            _log(f"  Success Rate:              {sr:.2%}", verbose, logger=logger)
            _log(f"  Avg Return (Undiscounted): {ar_undisc:.4f}", verbose, logger=logger)
            _log(f"  Avg Discounted Return:     {ar_disc:.4f}", verbose, logger=logger)
            _log(f"  Avg Steps (Success/Fail):  {avg_ss:.1f} / {avg_sf:.1f}", verbose, logger=logger)
            _log(f"  Current Epsilon:           {epsilon:.4f}", verbose, logger=logger)
            _log("  Current Greedy Policy:", verbose, logger=logger)
            # L'affichage 2D existant est specifique a FrozenLake (grille carree).
            if size is not None and not hasattr(env, "dead_state_idx"):
                policy_grid = utils.get_policy_grid_from_q_table(q_table, size, holes, goal)
                if policy_grid:
                    for row_str in policy_grid:
                        _log(f"    {row_str}", verbose, logger=logger)
            _log("-" * (4 + size + 1) if size is not None else "-" * 40,
                 verbose, logger=logger)
            _maybe_save_qtable(q_table, shaping_method_name, instance_id, total_episodes,
                               episode + 1, verbose, run_dir)

    _log("Standard Q-learning complete.", verbose, logger=logger)
    return q_table, episode_log, evaluation_log


def shaped_reward_classic(state, next_state, reward, done, hole_positions, goal_position, grid_size):
    """Classic potential shaping: R_env + gamma*Phi(s') - Phi(s)."""

    def potential(s):
        if s == goal_position:
            return 0
        if s in hole_positions:
            return -grid_size * 2
        row, col = divmod(s, grid_size)
        goal_row, goal_col = divmod(goal_position, grid_size)
        manhattan_dist = abs(row - goal_row) + abs(col - goal_col)
        return -manhattan_dist

    phi_s = potential(state)
    phi_s_prime = potential(next_state)
    gamma_potential = config.DISCOUNT_FACTOR
    potential_shaping_signal = gamma_potential * phi_s_prime - phi_s
    return reward + potential_shaping_signal


def train_q_learning_with_classic_shaping(env, total_episodes, max_steps, grid_size, hole_positions, goal_position,
                                          shaping_method_name, instance_id,
                                          verbose: int = 0, run_dir: str = "results",
                                          logger: logging.Logger = None):
    """Q-learning with Classic Potential-Based Reward Shaping."""
    _log(f"Starting classic shaped training: {total_episodes} episodes", verbose, logger=logger)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.full((state_size, action_size), config.Q_INIT_VALUE_CLASSIC)

    epsilon, lr, gamma_discount, eps_decay, eps_min = _get_hyperparameters()
    episode_log, evaluation_log = [], []
    hole_set = set(hole_positions)

    episode_iter = tqdm(range(total_episodes), desc="Training", leave=False, dynamic_ncols=True) \
        if verbose >= 1 else range(total_episodes)

    for episode in episode_iter:
        state, info = env.reset()
        episode_steps = 0
        env_reward_sum = 0.0
        shaped_reward_sum = 0.0
        final_reward = 0.0
        done = False
        terminated = False
        truncated = False

        for step_idx in range(max_steps):
            episode_steps += 1
            current_state_for_update = state
            if not (0 <= state < state_size):
                _log(f"ERROR: Invalid state {state} Ep {episode + 1}/St {step_idx}.", verbose, logger=logger)
                break

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            try:
                next_state, base_env_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as err:
                _log(f"ERROR: env.step exception: {err}.", verbose, logger=logger)
                break

            if not (0 <= next_state < state_size):
                _log(f"ERROR: Invalid next_state {next_state} Ep {episode + 1}/St {step_idx}.", verbose, logger=logger)
                break

            reward_used_for_update = shaped_reward_classic(
                state, next_state, base_env_reward, done, hole_set, goal_position, grid_size
            )

            best_next_action_q_value = np.max(q_table[next_state])
            td_target = reward_used_for_update + gamma_discount * best_next_action_q_value * (1 - int(done))
            td_error = td_target - q_table[current_state_for_update, action]
            q_table[current_state_for_update, action] += lr * td_error

            shaped_reward_sum += reward_used_for_update
            env_reward_sum += base_env_reward
            state = next_state
            if done:
                final_reward = base_env_reward
                break

        success = int(final_reward == 1.0 and terminated)
        episode_log.append({
            'episode': episode + 1, 'steps': episode_steps,
            'env_reward': env_reward_sum, 'shaped_reward': shaped_reward_sum,
            'success': success
        })
        epsilon = config.decay_epsilon(epsilon)

        if (episode + 1) % config.EVAL_FREQUENCY == 0 or (episode + 1) == total_episodes:
            sr, ar_undisc, ar_disc, avg_ss, avg_sf = utils.evaluate_agent(env, q_table, max_steps, config.EVAL_EPISODES)
            evaluation_log.append({
                'training_episode': episode + 1, 'eval_success_rate': sr,
                'eval_avg_return': ar_undisc, 'eval_avg_discounted_return': ar_disc,
                'avg_steps_success': avg_ss, 'avg_steps_failure': avg_sf
            })
            _log(f"\n--- Evaluation at Episode {episode + 1}/{total_episodes} (Classic Shaping) ---",
                 verbose, logger=logger)
            _log(f"  Success Rate:              {sr:.2%}", verbose, logger=logger)
            _log(f"  Avg Return (Undiscounted): {ar_undisc:.4f}", verbose, logger=logger)
            _log(f"  Avg Discounted Return:     {ar_disc:.4f}", verbose, logger=logger)
            _log(f"  Avg Steps (Success/Fail):  {avg_ss:.1f} / {avg_sf:.1f}", verbose, logger=logger)
            _log(f"  Current Epsilon:           {epsilon:.4f}", verbose, logger=logger)
            _log("  Current Greedy Policy:", verbose, logger=logger)
            policy_grid = utils.get_policy_grid_from_q_table(q_table, grid_size, hole_positions, goal_position)
            if policy_grid:
                for row_str in policy_grid:
                    _log(f"    {row_str}", verbose, logger=logger)
            _log("-" * (4 + grid_size + 1), verbose, logger=logger)
            _maybe_save_qtable(q_table, shaping_method_name, instance_id, total_episodes,
                               episode + 1, verbose, run_dir)

    _log("Classic shaped training complete.", verbose, logger=logger)
    return q_table, episode_log, evaluation_log
