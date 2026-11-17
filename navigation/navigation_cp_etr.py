"""CP-ETR Navigation pour Q-learning, sans importer Research-Deep-RL.

Reproduit le protocole NavigationService Java et la regle de shaping du projet PPO-CP.
IMPORTANT: l'ETR est une esperance de reward total et peut etre negative.
"""
from __future__ import annotations

import math
import socket
from typing import Optional

import config
from reward_shaping import cp_etr_reward
from .navigation_environment import NavigationEnv


# Ordre des actions Q/Gym: 0=noop, 1=est, 2=nord, 3=sud, 4=ouest.
# Ordre attendu par NavigationService Java: 0=nord, 1=sud, 2=est, 3=ouest, 4=noop.
Q_TO_JAVA_ACTION = {0: 4, 1: 2, 2: 0, 3: 1, 4: 3}


class NavigationCPClient:
    """Protocole TCP avec une reponse complete par ligne."""

    def __init__(self, host="127.0.0.1", port=12350, timeout=120):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.reader = None

    def connect(self, instance_id: str):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reader = self.sock.makefile("r", encoding="utf-8")
        welcome = self.reader.readline().strip()
        if not welcome.startswith("OK Welcome"):
            raise RuntimeError(f"Accueil du serveur CP inattendu : {welcome!r}")
        answer = self.command(f"INIT {instance_id}")
        if not answer.startswith("OK INIT"):
            raise RuntimeError(f"INIT CP a echoue : {answer}")
        return True

    def command(self, request: str) -> str:
        if self.sock is None or self.reader is None:
            raise RuntimeError("Client CP non connecte")
        self.sock.sendall((request + "\n").encode("utf-8"))
        reply = self.reader.readline().strip()
        if not reply:
            raise RuntimeError(f"Reponse vide du serveur CP pour {request}")
        if reply.startswith("ERROR"):
            raise RuntimeError(f"CP {request} : {reply}")
        return reply

    def reset(self):
        answer = self.command("RESET")
        if not answer.startswith("OK RESET"):
            raise RuntimeError(f"RESET CP a echoue : {answer}")

    def query_etr(self) -> float:
        answer = self.command("QUERY_ETR")
        if not answer.startswith("ETR_VALUE "):
            raise RuntimeError(f"QUERY_ETR reponse inattendue : {answer}")
        value = float(answer.split()[1])
        if not math.isfinite(value):
            raise RuntimeError(f"ETR invalide : {value}")
        # NE PAS limiter [0, 1] : ETR Navigation peut etre -40, -20, etc.
        return value

    def step(self, step_idx: int, java_action: int, next_state: int):
        answer = self.command(f"STEP {step_idx} {java_action} {next_state}")
        if not answer.startswith("OK STEP"):
            raise RuntimeError(f"STEP CP a echoue : {answer}")

    def close(self):
        try:
            if self.reader is not None:
                self.reader.close()
        finally:
            if self.sock is not None:
                self.sock.close()
        self.reader = None
        self.sock = None


class NavigationCPEnv(NavigationEnv):
    """Meme RDDL + meme Q-learning, mais la recompense d'entrainement est shapee.

    `set_evaluation_mode(True)` desactive CP pour mesurer la performance sur
    la vraie recompense de l'environnement, comme pour Q sans CP.
    """

    def __init__(self, instance_id: str, client: NavigationCPClient,
                 alpha: float = config.CP_ETR_SHAPING_COEFF,
                 gamma: float = config.DISCOUNT_FACTOR):
        super().__init__(instance_id=instance_id)
        self.client = client
        self.alpha = alpha
        self.gamma = gamma
        self._evaluation_mode = False
        self._etr_before = 0.0
        self._cp_step_idx = 0
        self._terminal_reached = False
        self.shaped_steps = 0

    def set_evaluation_mode(self, enabled: bool):
        self._evaluation_mode = bool(enabled)

    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        self._cp_step_idx = 0
        self._terminal_reached = False
        if not self._evaluation_mode:
            self.client.reset()
            self._etr_before = self.client.query_etr()
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        info["raw_reward"] = float(reward)
        if self._evaluation_mode:
            return state, reward, terminated, truncated, info

        # Once a terminal state was already observed, preserve the existing
        # environment behavior without issuing further CP requests.
        if self._terminal_reached:
            return state, reward, terminated, truncated, info

        self.client.step(self._cp_step_idx, Q_TO_JAVA_ACTION[int(action)], state)
        self._cp_step_idx += 1
        terminal_transition = (
            terminated
            or truncated
            or state in (self.goal_state_idx, self.dead_state_idx)
        )
        if terminal_transition:
            # The CP server's terminal ETR is not the potential used by
            # finite-horizon PBRS: Phi(s_terminal) is defined as zero.
            etr_after = 0.0
            shaping = cp_etr_reward(
                reward, self._etr_before, etr_after,
                alpha=self.alpha, gamma=self.gamma,
            ) - float(reward)
            self._terminal_reached = True
            self._etr_before = 0.0
        else:
            etr_after = self.client.query_etr()
            shaping = cp_etr_reward(
                reward, self._etr_before, etr_after,
                alpha=self.alpha, gamma=self.gamma,
            ) - float(reward)
            self._etr_before = etr_after
        self.shaped_steps += 1
        info["etr_after"] = etr_after
        info["cp_shaping"] = shaping
        return state, float(reward) + shaping, terminated, truncated, info
