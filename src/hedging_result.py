from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

SplitType = Literal["train", "eval_agent", "eval_benchmark"]


@dataclass
class EpisodeResult:
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
        self.actions.append(float(action))
        self.rewards.append(float(info.get("reward", np.nan)))
        self.costs.append(float(info.get("cost", np.nan)))
        self.trade_costs.append(float(info.get("trade_cost", 0.0)))
        self.liquidation_costs.append(float(info.get("liquidation_cost", 0.0)))
        self.losses.append(None if loss is None else float(loss))
        self.agent_infos.append({} if agent_info is None else agent_info)

    def step_frame(self) -> pd.DataFrame:
        n_steps = len(self.actions)
        data: dict[str, Any] = {
            "split": [self.split] * n_steps,
            "episode_idx": [self.episode_idx] * n_steps,
            "step_idx": list(range(n_steps)),
            "time": self.times[:n_steps],
            "time_next": self.times[1 : n_steps + 1],
            "spot": self.path_data["S"][:n_steps],
            "spot_next": self.path_data["S"][1 : n_steps + 1],
            "action": self.actions,
            "reward": self.rewards,
            "cost": self.costs,
            "trade_cost": self.trade_costs,
            "liquidation_cost": self.liquidation_costs,
            "loss": self.losses,
        }
        for extra_key in ("sigma", "variance"):
            if extra_key in self.path_data:
                data[extra_key] = self.path_data[extra_key][:n_steps]
                data[f"{extra_key}_next"] = self.path_data[extra_key][1 : n_steps + 1]
        return pd.DataFrame(data)


class HedgingResult:
    def __init__(self) -> None:
        self.episodes: dict[SplitType, list[EpisodeResult]] = {
            "train": [],
            "eval_agent": [],
            "eval_benchmark": [],
        }

    def add_episode(self, episode_result: EpisodeResult, type: SplitType) -> None:
        self.episodes[type].append(episode_result)
