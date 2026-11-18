"""Gymnasium adapter for the IPPC 2014 CrossingTraffic MDP.

The RDDL environment remains the source of transitions and rewards.  The
integer observation is a deterministic tabular-Q index containing the robot
position and the moving-obstacle configuration.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyRDDLGym


def _resolve_crossing_rddl_paths(instance_id: str) -> tuple[Path, Path]:
    """Resolve packaged RDDL files without using rddlrepository's manifest."""
    spec = importlib.util.find_spec("rddlrepository")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("rddlrepository est requis pour CrossingTraffic")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    directory = (
        package_root / "archive" / "competitions" / "IPPC2014"
        / "CrossingTraffic" / "MDP"
    )
    domain = directory / "domain.rddl"
    instance = directory / f"instance{instance_id}.rddl"
    if not domain.is_file() or not instance.is_file():
        raise FileNotFoundError(
            f"RDDL CrossingTraffic introuvable : {domain} / {instance}"
        )
    return domain, instance


def enumerate_valid_combos(action_keys: List[str]) -> List[frozenset]:
    """Index 0 is noop, followed by one action per sorted RDDL key."""
    return [frozenset()] + [frozenset([key]) for key in action_keys]


def decode_action(action_index: int, combos: List[frozenset], noop: Dict) -> Dict:
    active = combos[action_index]
    return {key: (key in active) for key in noop}


class CrossingTrafficEnv(gym.Env):
    """Tabular-Q wrapper for a CrossingTraffic RDDL instance.

    The Q-table state layout follows the reference encoder used by
    Research-Deep-RL: ``robot_pos * 2**(W*(H-2)) + compact_obstacle_bits``.
    Robot positions use ``x * H + y`` in the Q-table, while CP protocol cell
    indices use ``x + W * y``; both representations are exposed in ``info``.
    """

    metadata = {"render_modes": []}

    def __init__(self, instance_id: str = "2"):
        super().__init__()
        instance_id = str(instance_id)
        metadata_file = Path(__file__).resolve().parent / "crossingtraffic_mdp_ippc2011.json"
        with metadata_file.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if instance_id not in data:
            raise ValueError(f"Instance CrossingTraffic inconnue : {instance_id}")
        instance = data[instance_id]

        self.instance_id = instance_id
        self.domain = "crossingtraffic"
        self.nX = int(instance["grid_x"])
        self.nY = int(instance["grid_y"])
        self.horizon = int(instance["horizon"])
        self.cp_nb_steps = int(instance["cp_nbSteps"])
        self.goal_state_idx = int(instance["goal_state_idx"])
        self.dead_state_idx = int(instance["dead_state_idx"])
        self.init_state_idx = int(instance["init_state_idx"])
        self.n_obs_configs = 1 << (self.nX * max(0, self.nY - 2))
        expected_states = self.nX * self.nY * self.n_obs_configs + 2
        if int(instance["n_states"]) != expected_states:
            raise ValueError(
                f"JSON incohérent : n_states={instance['n_states']} != {expected_states}"
            )

        domain_path, instance_path = _resolve_crossing_rddl_paths(instance_id)
        self._env = pyRDDLGym.make(str(domain_path), str(instance_path))
        self.horizon = int(self._env.horizon)
        self.max_allowed_actions = int(self._env.max_allowed_actions)
        self._noop = dict(self._env.sampler.grounded_noop_actions)
        self._action_keys = sorted(self._env.action_space.spaces.keys())
        expected_actions = ["move-east", "move-north", "move-south", "move-west"]
        if self._action_keys != expected_actions:
            raise ValueError(f"Actions RDDL inattendues : {self._action_keys}")
        if self.max_allowed_actions != 1:
            raise ValueError("CrossingTraffic suppose une seule action par étape")
        self._combos = enumerate_valid_combos(self._action_keys)
        self.action_space = spaces.Discrete(len(self._combos))
        self.observation_space = spaces.Discrete(expected_states)

        self.goal_robot_idx = self._cell_idx(self.nX - 1, self.nY - 1)
        self._last_obs_dict = None
        self._last_state = None
        self._last_robot_idx = None
        self._last_obs_mask = 0

    def _cell_idx(self, x: int, y: int) -> int:
        """Java/CP cell index: x + width*y."""
        return x + self.nX * y

    def _robot_coordinates(self, obs_dict: Dict) -> tuple[int, int] | None:
        for x in range(self.nX):
            for y in range(self.nY):
                if obs_dict.get(f"robot-at___x{x + 1}__y{y + 1}", False):
                    return x, y
        return None

    def _robot_q_position(self, obs_dict: Dict) -> int:
        coords = self._robot_coordinates(obs_dict)
        return -1 if coords is None else coords[0] * self.nY + coords[1]

    def _robot_java_index(self, obs_dict: Dict) -> int:
        coords = self._robot_coordinates(obs_dict)
        return -1 if coords is None else self._cell_idx(*coords)

    def _encode_obs_mask(self, obs_dict: Dict) -> int:
        """Encode all obstacles with the Java convention idx(x,y)=x+W*y."""
        mask = 0
        for x in range(self.nX):
            for y in range(self.nY):
                if obs_dict.get(f"obstacle-at___x{x + 1}__y{y + 1}", False):
                    mask |= 1 << self._cell_idx(x, y)
        return mask

    def _encode_state(self, obs_dict: Dict) -> int:
        robot_q_pos = self._robot_q_position(obs_dict)
        if robot_q_pos < 0:
            return self.dead_state_idx
        if self._robot_java_index(obs_dict) == self.goal_robot_idx:
            return self.goal_state_idx

        compact_bits = 0
        bit = 0
        for y in range(1, self.nY - 1):
            for x in range(self.nX):
                key = f"obstacle-at___x{x + 1}__y{y + 1}"
                if obs_dict.get(key, False):
                    compact_bits |= 1 << bit
                bit += 1
        return robot_q_pos * self.n_obs_configs + compact_bits

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._env.seed(seed)
            np.random.seed(seed)
        obs_dict, info = self._env.reset()
        self._last_obs_dict = dict(obs_dict)
        self._last_state = self._encode_state(self._last_obs_dict)
        self._last_robot_idx = self._robot_java_index(self._last_obs_dict)
        self._last_obs_mask = self._encode_obs_mask(self._last_obs_dict)
        return self._last_state, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide : {action}")
        previous_obs = dict(self._last_obs_dict)
        previous_robot_idx = self._robot_java_index(previous_obs)
        previous_obs_mask = self._encode_obs_mask(previous_obs)
        rddl_action = decode_action(int(action), self._combos, self._noop)
        obs_dict, reward, rddl_terminated, truncated, info = self._env.step(rddl_action)
        self._last_obs_dict = dict(obs_dict)
        state = self._encode_state(self._last_obs_dict)
        next_robot_idx = self._robot_java_index(self._last_obs_dict)
        next_obs_mask = self._encode_obs_mask(self._last_obs_dict)
        goal_reached = next_robot_idx == self.goal_robot_idx
        robot_dead = next_robot_idx == -1
        state_terminal = goal_reached or robot_dead

        # RDDL keeps both the goal and the dead robot absorbing until the
        # horizon.  Preserve that episode semantics: only a termination
        # explicitly reported by RDDL can set ``terminated``; reaching the
        # goal/dead state is exposed in ``info`` for evaluation and CP shaping.
        terminated = bool(rddl_terminated)
        info = dict(info)
        info.update({
            "raw_reward": float(reward),
            "rddl_reward": float(reward),
            "previous_robot_idx": previous_robot_idx,
            "next_robot_idx": next_robot_idx,
            "previous_obs_mask": previous_obs_mask,
            "obs_mask": next_obs_mask,
            "goal_reached": goal_reached,
            "robot_dead": robot_dead,
            "state_terminal": state_terminal,
            "rddl_terminated": bool(rddl_terminated),
            "termination": terminated,
            "truncated": bool(truncated),
        })
        self._last_state = state
        self._last_robot_idx = next_robot_idx
        self._last_obs_mask = next_obs_mask
        return state, float(reward), terminated, bool(truncated), info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
