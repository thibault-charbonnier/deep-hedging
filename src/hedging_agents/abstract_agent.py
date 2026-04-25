from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AbstractHedgingAgent(ABC):
    """Interface that every hedging agent must implement."""

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        self.gamma = float(agent_cfg.get("discount_factor", 1.0))

    @abstractmethod
    def act(self, state: np.ndarray, eval_mode: bool = False) -> Any:
        """Return an action for ``state``. When ``eval_mode`` is True, no exploration noise."""
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
        """Push a transition into the replay buffer."""
        raise NotImplementedError

    @abstractmethod
    def learn(self) -> float | None:
        """Perform one gradient update; return a scalar training loss or None if skipped."""
        raise NotImplementedError

    @abstractmethod
    def set_train_mode(self) -> None:
        """Enable training mode (dropout/batchnorm active, exploration on)."""
        raise NotImplementedError

    @abstractmethod
    def set_eval_mode(self) -> None:
        """Switch every network to evaluation mode and disable exploration."""
        raise NotImplementedError

