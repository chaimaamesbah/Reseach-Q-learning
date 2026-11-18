"""CrossingTraffic CP-ETR client and Gymnasium wrapper."""
from __future__ import annotations

import math
import socket
from typing import Optional

import config
from reward_shaping import cp_etr_reward
from .crossingtraffic_environment import CrossingTrafficEnv


# Q-learning order: noop, east, north, south, west.
# Java order: north, south, east, west, noop.
Q_TO_JAVA_ACTION = {0: 4, 1: 2, 2: 0, 3: 1, 4: 3}


class CrossingTrafficCPClient:
    """Strict client for the four-field CrossingTraffic STEP protocol."""

    def __init__(self, host="127.0.0.1", port=12346, timeout=120):
        self.host = host
        self.port = int(port)
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
            self.close()
            raise RuntimeError(f"Accueil CP CrossingTraffic inattendu : {welcome!r}")
        answer = self.command(f"INIT {instance_id}")
        if not answer.startswith("OK INIT"):
            raise RuntimeError(f"INIT CrossingTraffic échoué : {answer}")

    def command(self, request: str) -> str:
        if self.sock is None or self.reader is None:
            raise RuntimeError("Client CP CrossingTraffic non connecté")
        try:
            self.sock.sendall((request + "\n").encode("utf-8"))
            reply = self.reader.readline().strip()
        except (OSError, socket.timeout) as exc:
            raise RuntimeError(f"Erreur TCP pour {request}: {exc}") from exc
        if not reply:
            raise RuntimeError(f"Réponse vide du serveur CP pour {request}")
        if reply.startswith("ERROR"):
            raise RuntimeError(f"CP {request}: {reply}")
        return reply

    def reset(self):
        answer = self.command("RESET")
        if not answer.startswith("OK RESET"):
            raise RuntimeError(f"RESET CP échoué : {answer}")

    def query_etr(self) -> float:
        answer = self.command("QUERY_ETR")
        if not answer.startswith("ETR_VALUE "):
            raise RuntimeError(f"QUERY_ETR inattendu : {answer}")
        try:
            value = float(answer.split()[1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"ETR non numérique : {answer}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"ETR non fini : {value}")
        return value

    def step(self, step_idx: int, java_move: int,
              robot_idx: int, obs_mask: int):
        answer = self.command(
            f"STEP {int(step_idx)} {int(java_move)} "
            f"{int(robot_idx)} {int(obs_mask)}"
        )
        if not answer.startswith("OK STEP"):
            raise RuntimeError(f"STEP CP échoué : {answer}")

    def close(self):
        if self.sock is not None:
            try:
                if self.reader is not None:
                    self.sock.sendall(b"QUIT\n")
                    self.reader.readline()
            except OSError:
                pass
        try:
            if self.reader is not None:
                self.reader.close()
        finally:
            if self.sock is not None:
                self.sock.close()
        self.reader = None
        self.sock = None


class CrossingTrafficCPEnv(CrossingTrafficEnv):
    """CrossingTraffic environment with strict CP-ETR shaping."""

    def __init__(self, instance_id: str, client: CrossingTrafficCPClient,
                 alpha: float = config.CP_ETR_SHAPING_COEFF,
                 gamma: float = config.DISCOUNT_FACTOR):
        super().__init__(instance_id=instance_id)
        self.client = client
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self._evaluation_mode = False
        self._etr_before = 0.0
        self._cp_step_idx = 0
        self._terminal_reached = False

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
        if self._evaluation_mode:
            return super().step(action)

        state, reward, terminated, truncated, info = super().step(action)
        if self._terminal_reached:
            info = dict(info)
            info.update({
                "etr_before": 0.0,
                "etr_after": 0.0,
                "cp_shaping": 0.0,
                "reward_used_for_update": float(reward),
            })
            return state, float(reward), terminated, truncated, info
        self.client.step(
            self._cp_step_idx,
            Q_TO_JAVA_ACTION[int(action)],
            info["next_robot_idx"],
            info["previous_obs_mask"],
        )
        self._cp_step_idx += 1

        # Goal/dead are absorbing RDDL states, so the Python episode still
        # reaches its configured horizon.  They nevertheless are terminal for
        # potential shaping: the transition is sent once to keep CP in sync,
        # then Phi(s_terminal)=0 is used and no second terminal shaping is
        # applied on the remaining absorbing steps.
        terminal_transition = bool(
            terminated or truncated or info.get("state_terminal", False)
        )
        if terminal_transition:
            # The CP model may have a negative dead-state ETR; PBRS defines
            # the potential of the terminal state itself as zero.
            etr_after = 0.0
            self._terminal_reached = bool(
                info.get("state_terminal", False) or terminated or truncated
            )
        else:
            etr_after = self.client.query_etr()

        shaped = cp_etr_reward(
            reward,
            self._etr_before,
            etr_after,
            alpha=self.alpha,
            gamma=self.gamma,
        )
        info = dict(info)
        info.update({
            "etr_before": self._etr_before,
            "etr_after": etr_after,
            "cp_shaping": shaped - float(reward),
            "reward_used_for_update": shaped,
        })
        self._etr_before = 0.0 if terminal_transition else etr_after
        return state, float(shaped), terminated, truncated, info
