import numpy as np
from typing import Any
from abc import ABC, abstractmethod


class AbstractHedgingAgent(ABC):
    """
    Abstract class for hedging agents.

    This module defines the mechanics of the hedging agent :
        - Choose an hedging action from a state given by the environment.
        - Store the transition for learning.
        - Learn from the stored transitions.
    """

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        self.cfg = agent_cfg
        self.gamma = float(agent_cfg.get("discount_factor"))

    @abstractmethod
    def act(self, state: np.ndarray, eval_mode: bool = False) -> Any:
        """
        Choose an action (hedging position) given the current state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.
        eval_mode : bool, optional
            Whether the agent is in evaluation mode (default: False).
            This can be used to disable exploration during evaluation.

        Returns
        -------
        Any
            The action chosen by the agent.
        """
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
        """
        Store the transition (state, action, reward, next_state, done) for learning.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.
        action : Any
            The action taken by the agent.
        reward : float
            The reward received after taking the action.
        next_state : np.ndarray
            The next state of the environment after taking the action.
        done : bool
            Whether the episode has ended after taking the action.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self) -> float | None:
        """
        Perform one learning step.
        Return a loss if relevant, otherwise None.
        """
        raise NotImplementedError

    @abstractmethod
    def set_train_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_eval_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError