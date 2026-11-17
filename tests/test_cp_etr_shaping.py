import unittest
from unittest.mock import patch

import config
from navigation.navigation_cp_etr import NavigationCPEnv
from reward_shaping import cp_etr_reward


class _FakeClient:
    def __init__(self):
        self.steps = []

    def step(self, step_idx, action, next_state):
        self.steps.append((step_idx, action, next_state))


class CPETRShapingTests(unittest.TestCase):
    def test_equation_8_keeps_raw_reward_and_gamma(self):
        self.assertAlmostEqual(
            cp_etr_reward(2.0, 3.0, 5.0),
            2.0 + config.DISCOUNT_FACTOR * 5.0 - 3.0,
        )

    def test_terminal_potential_is_zero(self):
        self.assertAlmostEqual(cp_etr_reward(1.0, 0.4, 0.0), 0.6)

    def test_navigation_applies_terminal_shaping_once(self):
        env = object.__new__(NavigationCPEnv)
        env.client = _FakeClient()
        env.alpha = config.CP_ETR_SHAPING_COEFF
        env.gamma = config.DISCOUNT_FACTOR
        env._evaluation_mode = False
        env._etr_before = 0.4
        env._cp_step_idx = 0
        env._terminal_reached = False
        env.shaped_steps = 0
        env.goal_state_idx = 2
        env.dead_state_idx = 3

        with patch(
            "navigation.navigation_cp_etr.NavigationEnv.step",
            return_value=(2, 1.0, False, False, {}),
        ):
            state, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(state, 2)
        self.assertAlmostEqual(reward, 1.0 - 0.4)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(len(env.client.steps), 1)
        self.assertEqual(info["etr_after"], 0.0)

    def test_epsilon_schedule_is_shared(self):
        epsilon = config.EPSILON
        for _ in range(1000):
            epsilon = config.decay_epsilon(epsilon)
        self.assertAlmostEqual(config.decay_epsilon(1.0), 0.99)
        self.assertEqual(epsilon, config.EPSILON_MIN)


if __name__ == "__main__":
    unittest.main()
