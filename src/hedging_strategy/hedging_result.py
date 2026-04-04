import numpy as np
import pandas as pd
from typing import Any, Literal
from dataclasses import dataclass, field

SplitType = Literal["train", "eval_agent", "eval_benchmark"]


@dataclass
class EpisodeResult:
    """
    Stores all relevant data for a single episode of interaction with the environment.
    """

    split: SplitType
    episode_idx: int
    path_data: dict[str, np.ndarray]
    times: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    actions: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    trade_costs: list[float] = field(default_factory=list)
    liquidation_costs: list[float] = field(default_factory=list)
    losses: list[float | None] = field(default_factory=list)
    agent_infos: list[dict[str, Any]] = field(default_factory=list)

    def add_step(
        self,
        action: float,
        info: dict[str, Any],
        loss: float | None = None,
        agent_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Store all relevant data from a single step in this episode.

        Parameters
        ----------
        action : float
            The action taken by the agent at this step.
        info : dict[str, Any]
            The info dictionary returned by the environment at this step.
        loss : float | None, optional
            The loss value from the agent's learning step at this step, if applicable.
        agent_info : dict[str, Any] | None, optional
            Any additional info from the agent at this step, if applicable.
        """
        self.actions.append(float(action))
        self.rewards.append(float(info.get("reward", np.nan)))
        self.costs.append(float(info.get("cost", np.nan)))
        self.trade_costs.append(float(info.get("trade_cost", 0.0)))
        self.liquidation_costs.append(float(info.get("liquidation_cost", 0.0)))
        self.losses.append(None if loss is None else float(loss))
        self.agent_infos.append({} if agent_info is None else agent_info)

    def step_frame(self) -> pd.DataFrame:
        """
        Build a summary DataFrame with one row per step in this episode, including all logged data.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all step-level data for this episode.
        """
        n_steps = len(self.actions)
        data: dict[str, Any] = {
            "split": [self.split] * n_steps,
            "episode_idx": [self.episode_idx] * n_steps,
            "step_idx": list(range(n_steps)),
            "time": self.times[:-1],
            "time_next": self.times[1:],
            "spot": self.path_data["S"][:-1],
            "spot_next": self.path_data["S"][1:],
            "action": self.actions,
            "reward": self.rewards,
            "cost": self.costs,
            "trade_cost": self.trade_costs,
            "liquidation_cost": self.liquidation_costs,
            "loss": self.losses,
        }
        if "sigma" in self.path_data:
            data["sigma"] = self.path_data["sigma"][:-1]
            data["sigma_next"] = self.path_data["sigma"][1:]
        if "variance" in self.path_data:
            data["variance"] = self.path_data["variance"][:-1]
            data["variance_next"] = self.path_data["variance"][1:]

        return pd.DataFrame(data)


class HedgingResult:
    """
    Stores all relevant data from an entire run of the hedging experiment, including multiple episodes of training and evaluation.
    """

    def __init__(self) -> None:
        self.episodes: dict[SplitType, list[EpisodeResult]] = {
            "train": [],
            "eval_agent": [],
            "eval_benchmark": [],
        }
        
    def add_episode(self, episode_result: EpisodeResult, type: SplitType) -> None:
        """
        Store the results from a single episode in the appropriate split.

        Parameters
        ----------
        episode_result : EpisodeResult
            The result object containing all data from this episode.
        type : SplitType
            The split type ("train", "eval_agent", or "eval_benchmark") to which this episode belongs.
        """
        self.episodes[type].append(episode_result)
