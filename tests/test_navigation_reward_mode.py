import unittest
from pathlib import Path
import tempfile

from gymnasium import spaces

from navigation.navigation_environment import NavigationEnv
from run_experiment import _build_cmd, _cache_path, _load_cache, _save_cache
from utils import is_success


class _FakeRDDL:
    def __init__(self, transitions):
        self.transitions = iter(transitions)

    def step(self, action):
        next_state, reward = next(self.transitions)
        return {"state": next_state}, reward, False, False, {}


def _make_env(mode, transitions):
    env = object.__new__(NavigationEnv)
    env.reward_mode = mode
    env.goal_state_idx = 2
    env.dead_state_idx = 3
    env._last_state = 1
    env._goal_reached = False
    env._last_obs_dict = None
    env._env = _FakeRDDL(transitions)
    env._encode_state = lambda obs: obs["state"]
    env._noop = {}
    env._combos = [frozenset()]
    env.action_space = spaces.Discrete(1)
    return env


class NavigationRewardModeTests(unittest.TestCase):
    def test_frozenlake_rewards_first_goal_entry_only(self):
        env = _make_env("frozenlake", [(2, -1), (2, 0), (3, -1)])

        _, reward, _, _, info = env.step(0)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["rddl_reward"], -1.0)
        self.assertEqual(info["raw_reward"], 1.0)

        _, reward, _, _, info = env.step(0)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["rddl_reward"], 0.0)

        _, reward, _, _, info = env.step(0)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["rddl_reward"], -1.0)

        dead_env = _make_env("frozenlake", [(3, -1)])
        _, reward, _, _, _ = dead_env.step(0)
        self.assertEqual(reward, 0.0)

    def test_original_mode_preserves_rddl_reward(self):
        env = _make_env("original", [(2, -1)])
        _, reward, _, _, info = env.step(0)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info["raw_reward"], -1.0)
        self.assertEqual(info["reward_mode"], "original")

    def test_navigation_command_and_cache_include_reward_mode(self):
        cmd = _build_cmd(
            "q-none", "1", 5, 7, "/tmp/run",
            domain="navigation", reward_mode="frozenlake",
        )
        self.assertEqual(cmd[cmd.index("--reward-mode") + 1], "frozenlake")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            original = _cache_path(
                output_dir, "1", "q-none", 7, 5,
                reward_mode="original", domain="navigation",
            )
            frozenlake = _cache_path(
                output_dir, "1", "q-none", 7, 5,
                reward_mode="frozenlake", domain="navigation",
            )
            self.assertNotEqual(original, frozenlake)
            self.assertNotEqual(
                frozenlake,
                _cache_path(
                    output_dir, "1", "q-none", 7, 5,
                    reward_mode="frozenlake", domain="navigation",
                    alpha=0.5,
                ),
            )
            self.assertNotEqual(
                frozenlake,
                _cache_path(
                    output_dir, "1", "q-none", 7, 5,
                    reward_mode="frozenlake", domain="navigation",
                    bp_iter=2,
                ),
            )
            entry = {
                "domain": "navigation",
                "reward_mode": "frozenlake",
                "alpha": 1.0,
                "bp_iter": 1,
                "python_horizon": 40,
                "cp_nb_steps": 45,
            }
            _save_cache(
                output_dir, "1", "q-none", 7, 5, entry,
                reward_mode="frozenlake", domain="navigation",
            )
            self.assertEqual(
                _load_cache(
                    output_dir, "1", "q-none", 7, 5,
                    reward_mode="frozenlake", domain="navigation",
                ),
                entry,
            )
            self.assertIsNone(
                _load_cache(
                    output_dir, "1", "q-none", 7, 5,
                    reward_mode="original", domain="navigation",
                )
            )
            incomplete = _cache_path(
                output_dir, "1", "q-none", 7, 5,
                reward_mode="frozenlake", domain="navigation",
                alpha=0.5,
            )
            incomplete.write_text('{"domain":"navigation",'
                                  '"reward_mode":"frozenlake"}')
            self.assertIsNone(
                _load_cache(
                    output_dir, "1", "q-none", 7, 5,
                    reward_mode="frozenlake", domain="navigation",
                    alpha=0.5,
                )
            )

    def test_success_rate_uses_goal_state_even_without_termination_flag(self):
        env = type("NavigationLike", (), {"goal_state_idx": 2})()
        self.assertTrue(is_success(env, state=2, terminated=False, reward=0.0))
        self.assertFalse(is_success(env, state=1, terminated=False, reward=0.0))


if __name__ == "__main__":
    unittest.main()
