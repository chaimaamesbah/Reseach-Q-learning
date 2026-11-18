import math
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import q_learning_standard
from crossingtraffic.crossingtraffic_cp_etr import (
    CrossingTrafficCPClient,
    CrossingTrafficCPEnv,
    Q_TO_JAVA_ACTION,
)
from crossingtraffic.crossingtraffic_environment import CrossingTrafficEnv
from reward_shaping import cp_etr_reward
from run_experiment import _cache_path, _domain_horizons, _model_fingerprint


class CrossingTrafficEnvironmentTests(unittest.TestCase):
    def test_declared_state_spaces_and_initial_encoding(self):
        expected = {
            "2": (74, 50, 16),
            "4": (4098, 3180, 1728),
        }
        for instance_id, (n_states, initial_state, initial_mask) in expected.items():
            env = CrossingTrafficEnv(instance_id)
            try:
                state, _ = env.reset(seed=1)
                self.assertEqual(env.observation_space.n, n_states)
                self.assertEqual(state, initial_state)
                self.assertEqual(env._last_obs_mask, initial_mask)
                self.assertEqual(env.action_space.n, 5)
                self.assertEqual(env.horizon, 40)
                self.assertEqual(env.cp_nb_steps, 45)
            finally:
                env.close()

    def test_java_mask_uses_x_plus_width_times_y(self):
        env = CrossingTrafficEnv("4")
        try:
            obs = {"obstacle-at___x2__y3": True}
            self.assertEqual(env._encode_obs_mask(obs), 1 << (1 + 4 * 2))
        finally:
            env.close()

    def test_actions_keep_q_and_java_orders_distinct(self):
        self.assertEqual(Q_TO_JAVA_ACTION, {0: 4, 1: 2, 2: 0, 3: 1, 4: 3})
        env = CrossingTrafficEnv("2")
        try:
            self.assertEqual(
                [set(combo) for combo in env._combos],
                [set(), {"move-east"}, {"move-north"},
                 {"move-south"}, {"move-west"}],
            )
        finally:
            env.close()


class CrossingTrafficShapingTests(unittest.TestCase):
    def test_signed_etr_and_raw_reward_use_equation_8(self):
        value = cp_etr_reward(-1.0, -3.0, -5.0)
        self.assertAlmostEqual(value, -1.0 + config.DISCOUNT_FACTOR * -5.0 + 3.0)
        self.assertTrue(math.isfinite(value))

    def test_goal_or_dead_state_is_shaped_after_the_transition(self):
        class FakeClient:
            def __init__(self):
                self.steps = []
                self.queries = 0

            def step(self, step_idx, java_move, robot_idx, obs_mask):
                self.steps.append((step_idx, java_move, robot_idx, obs_mask))

            def query_etr(self):
                self.queries += 1
                return -2.0

        env = object.__new__(CrossingTrafficCPEnv)
        env.client = FakeClient()
        env.alpha = 1.0
        env.gamma = config.DISCOUNT_FACTOR
        env._evaluation_mode = False
        env._etr_before = -4.0
        env._cp_step_idx = 0
        env._terminal_reached = False

        with patch(
            "crossingtraffic.crossingtraffic_environment.CrossingTrafficEnv.step",
            return_value=(73, -1.0, False, False, {
                "next_robot_idx": 8,
                "previous_obs_mask": 16,
                "state_terminal": True,
            }),
        ):
            state, reward, terminated, truncated, info = env.step(2)

        self.assertEqual(state, 73)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(len(env.client.steps), 1)
        self.assertEqual(env.client.steps[0], (0, 0, 8, 16))
        self.assertEqual(env.client.queries, 0)
        self.assertAlmostEqual(reward, -1.0 + 4.0)
        self.assertAlmostEqual(info["etr_after"], 0.0)

    def test_terminal_bellman_update_does_not_bootstrap(self):
        class OneStepEnv:
            observation_space = type("Space", (), {"n": 2})()
            action_space = type(
                "ActionSpace", (),
                {"n": 1, "sample": lambda self: 0},
            )()

            def reset(self, **kwargs):
                return 0, {}

            def step(self, action):
                return 1, -2.0, True, False, {"raw_reward": -2.0}

        env = OneStepEnv()
        with patch.object(config, "EVAL_FREQUENCY", 1), patch.object(
            config, "EVAL_EPISODES", 1
        ):
            q_table, _, _ = q_learning_standard.train_q_learning(
                env, 1, 1, None, [], None, "none", "test", verbose=0
            )
        self.assertAlmostEqual(q_table[0, 0], config.LEARNING_RATE * -2.0)


class CrossingTrafficExperimentTests(unittest.TestCase):
    def test_horizons_are_part_of_crossing_cache_identity(self):
        self.assertEqual(_domain_horizons("2", "crossingtraffic"), (40, 45))
        self.assertIsNotNone(_model_fingerprint("2", "crossingtraffic"))
        a = _cache_path(
            Path("/tmp"), "2", "q-none", 1, 2,
            domain="crossingtraffic", python_horizon=40, cp_nb_steps=45,
        )
        b = _cache_path(
            Path("/tmp"), "2", "q-none", 1, 2,
            domain="crossingtraffic", python_horizon=41, cp_nb_steps=45,
        )
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
