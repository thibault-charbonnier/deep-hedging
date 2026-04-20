from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .hedging_env import HedgingEnv


class GymHedgingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.env = HedgingEnv(config)
        action_low = float(config.get("hedging_agent", {}).get("action_low", 0.0))
        action_high = float(config.get("hedging_agent", {}).get("action_high", 1.0))

        self.action_space = spaces.Box(
            low=np.array([action_low], dtype=np.float32),
            high=np.array([action_high], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

    @property
    def times(self):
        return self.env.times

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        path_data = options["path_data"] if options is not None else None
        state = self.env.setup_env(path_data)
        return np.asarray(state, dtype=np.float32), {}

    def step(self, action):
        raise NotImplementedError(
            "Orchestrator bypasses gym step and uses env.env.apply_action() "
            "to preserve delayed-reward structure from the paper."
        )

