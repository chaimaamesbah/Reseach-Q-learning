"""Navigation pour le Q-learning tabulaire, sans dépendre de Research-Deep-RL.

Le code utile de Research-Deep-RL est recopié dans CE fichier :
  - envs/action_space.py : enumerate_valid_combos et decode_action
  - envs/cp_state_encoder.py : make_navigation_encoder
  - envs/rddl_env.py : création pyRDDLGym, gestion reset/step, seed, render, close

On adapte uniquement l'observation : au lieu d'un vecteur PPO, renvoyer
un indice entier entre 0 et n_states - 1 pour la Q-table. Le modèle RDDL reste
la source des transitions et de la récompense originale ; le mode
``frozenlake`` ne transforme que la récompense retournée au Q-learning.
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Dict, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pyRDDLGym


# --- Copie de Research-Deep-RL/envs/action_space.py ---
def enumerate_valid_combos(action_keys: List[str]) -> List[frozenset]:
    """Index 0 = ne rien faire; ensuite une action par clé RDDL."""
    combos: List[frozenset] = [frozenset()]
    for key in action_keys:
        combos.append(frozenset([key]))
    return combos


def decode_action(action_index: int, combos: List[frozenset], noop: Dict) -> Dict:
    """Convertit un entier de Q-learning en dictionnaire d'actions RDDL."""
    active_keys = combos[action_index]
    return {k: (k in active_keys) for k in noop}


# --- Copie de Research-Deep-RL/envs/cp_state_encoder.py ---
def make_navigation_encoder(nX: int, nY: int,
                            xpos_sorted: list, ypos_sorted: list,
                            dead_state_idx: int):
    """Encode obs_dict en state_idx = xi * nY + yj; état mort si aucun robot-at."""
    def encode(obs_dict: Dict) -> int:
        for xi, xl in enumerate(xpos_sorted):
            for yj, yl in enumerate(ypos_sorted):
                if obs_dict.get(f"robot-at___{xl}__{yl}", False):
                    return xi * nY + yj
        return dead_state_idx

    return encode


def _resolve_navigation_rddl_paths(instance_id: str) -> tuple[Path, Path]:
    """Resolve Navigation resources without relying on rddlrepository's manifest.

    rddlrepository 2.2 stores absolute paths in its generated manifest.  Those
    paths are not portable (and can become stale after moving the repository),
    so resolve the packaged RDDL files relative to the installed Python package
    instead.  The RDDL files themselves remain untouched.
    """
    spec = importlib.util.find_spec("rddlrepository")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "rddlrepository est requis pour l'environnement Navigation"
        )

    package_root = Path(next(iter(spec.submodule_search_locations)))
    navigation_dir = (
        package_root / "archive" / "competitions" / "IPPC2011"
        / "Navigation" / "MDP"
    )
    domain_path = navigation_dir / "domain.rddl"
    instance_path = navigation_dir / f"instance{instance_id}.rddl"
    if not domain_path.is_file() or not instance_path.is_file():
        raise FileNotFoundError(
            "RDDL Navigation introuvable dans l'installation rddlrepository : "
            f"{domain_path} / {instance_path}"
        )
    return domain_path, instance_path


class NavigationEnv(gym.Env):
    """Adaptation locale du wrapper PPO pour les observations entières de Q-learning."""

    metadata = {"render_modes": []}

    def __init__(self, instance_id="1", reward_mode="original"):
        super().__init__()
        instance_id = str(instance_id)
        if reward_mode not in ("original", "frozenlake"):
            raise ValueError(
                "reward_mode doit être 'original' ou 'frozenlake'"
            )
        self.reward_mode = reward_mode

        # Les labels x/y sont ceux du JSON du projet, dans le même ordre que Java.
        metadata_file = Path(__file__).resolve().parent / "navigation_mdp_ippc2011.json"
        with metadata_file.open(encoding="utf-8") as f:
            instance_data = json.load(f)[instance_id]

        self.nX = int(instance_data["grid_x"])
        self.nY = int(instance_data["grid_y"])
        self.init_state_idx = int(instance_data["init_state_idx"])
        self.goal_state_idx = int(instance_data["goal_state_idx"])
        self.dead_state_idx = int(instance_data["dead_state_idx"])
        n_states = int(instance_data["n_states"])
        self.cp_nb_steps = int(instance_data["cp_nbSteps"])

        if n_states != self.nX * self.nY + 1:
            raise ValueError("JSON incohérent : n_states doit compter les positions + l'état mort")

        self._encode_state = make_navigation_encoder(
            self.nX, self.nY,
            instance_data["xpos"], instance_data["ypos"],
            self.dead_state_idx,
        )

        # Même moteur et instance que le wrapper PPO de Deep-RL.  Les chemins
        # sont résolus dans le paquet installé pour éviter le manifeste
        # rddlrepository généré avec un chemin absolu obsolète.
        domain_path, instance_path = _resolve_navigation_rddl_paths(instance_id)
        self._env = pyRDDLGym.make(str(domain_path), str(instance_path))
        self.horizon = int(self._env.horizon)
        self.max_allowed_actions = int(self._env.max_allowed_actions)

        # Même décodage que Deep-RL. L'ordre trié donne :
        # 0=attendre, 1=est, 2=nord, 3=sud, 4=ouest.
        self._noop = dict(self._env.sampler.grounded_noop_actions)
        self._action_keys = sorted(self._env.action_space.spaces.keys())
        self._combos = enumerate_valid_combos(self._action_keys)
        self.action_space = spaces.Discrete(len(self._combos))
        self.observation_space = spaces.Discrete(n_states)

        expected_keys = ["move-east", "move-north", "move-south", "move-west"]
        if self._action_keys != expected_keys or self.action_space.n != 5:
            raise ValueError(f"Actions RDDL inattendues : {self._action_keys}")
        if self.max_allowed_actions != 1:
            raise ValueError("Cette adaptation suppose une seule action de déplacement par étape")

        self._last_obs_dict = None
        self._last_state = None
        self._goal_reached = False

    def reset(self, *, seed=None, options=None):
        # Copié du reset du wrapper PPO, sans conversion en vecteur NumPy.
        if seed is not None:
            self._env.seed(seed)
            np.random.seed(seed)

        obs_dict, info = self._env.reset()
        self._last_obs_dict = obs_dict
        state = self._encode_state(obs_dict)
        self._last_state = state
        self._goal_reached = state == self.goal_state_idx
        return state, info

    def step(self, action):
        # Copié du step PPO sans RewardShaper : même dict RDDL et env.step().
        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide : {action}")

        previous_state = self._last_state
        rddl_action = decode_action(int(action), self._combos, self._noop)
        obs_dict, reward, terminated, truncated, info = self._env.step(rddl_action)
        self._last_obs_dict = obs_dict

        state = self._encode_state(obs_dict)
        rddl_reward = float(reward)
        if self.reward_mode == "frozenlake":
            # FrozenLake gives +1 only on the first transition entering the
            # goal. Goal and dead states are absorbing and therefore give 0.
            raw_reward = float(
                previous_state != self.goal_state_idx
                and state == self.goal_state_idx
                and not self._goal_reached
            )
        else:
            raw_reward = rddl_reward
        if state == self.goal_state_idx:
            self._goal_reached = True
        self._last_state = state
        info = dict(info)
        info["rddl_reward"] = rddl_reward
        info["raw_reward"] = raw_reward
        info["reward_mode"] = self.reward_mode
        return state, raw_reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
