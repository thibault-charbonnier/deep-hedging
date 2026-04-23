from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AbstractHedgingAgent(ABC):
    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        self.gamma = float(agent_cfg.get("discount_factor", 1.0))

    @abstractmethod
    def act(self, state: np.ndarray, eval_mode: bool = False) -> Any:
        raise NotImplementedError

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def learn(self) -> float | None:
        raise NotImplementedError

    @abstractmethod
    def set_train_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_eval_mode(self) -> None:
        raise NotImplementedError

