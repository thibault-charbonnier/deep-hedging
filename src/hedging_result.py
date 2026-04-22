from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

SplitType = Literal["train", "eval_agent", "eval_benchmark"]
ALL_SPLITS: tuple[SplitType, SplitType, SplitType] = ("train", "eval_agent", "eval_benchmark")


def _nanmean(values: list[float]) -> float:
    return float(np.nanmean(values)) if values else np.nan


def _nanstd(values: list[float]) -> float:
    return float(np.nanstd(values)) if values else np.nan


def _nansum(values: list[float]) -> float:
    return float(np.nansum(values)) if values else np.nan


def _nanskewness(values: list[float]) -> float:
    """Calculate skewness (3rd moment / std^3), handling NaN values."""
    if not values:
        return np.nan
    arr = np.asarray(values, dtype=float)
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) < 3:
        return np.nan
    mean = float(np.mean(finite_vals))
    std = float(np.std(finite_vals, ddof=0))
    if std == 0:
        return np.nan
    skew = float(np.mean(((finite_vals - mean) / std) ** 3))
    return skew


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
        if n_steps == len(self.times):
            time = np.concatenate(([self.times[0]], self.times[:-1]))
            time_next = np.concatenate(([self.times[0]], self.times[1:]))
            spot = np.concatenate(([self.path_data["S"][0]], self.path_data["S"][:-1]))
            spot_next = np.concatenate(([self.path_data["S"][0]], self.path_data["S"][1:]))
        else:
            time = self.times[:n_steps]
            time_next = self.times[1 : n_steps + 1]
            spot = self.path_data["S"][:n_steps]
            spot_next = self.path_data["S"][1 : n_steps + 1]
        data: dict[str, Any] = {
            "split": [self.split] * n_steps,
            "episode_idx": [self.episode_idx] * n_steps,
            "step_idx": list(range(n_steps)),
            "time": time,
            "time_next": time_next,
            "spot": spot,
            "spot_next": spot_next,
            "action": self.actions,
            "reward": self.rewards,
            "cost": self.costs,
            "trade_cost": self.trade_costs,
            "liquidation_cost": self.liquidation_costs,
            "loss": self.losses,
        }
        for extra_key in ("sigma", "variance"):
            if extra_key in self.path_data:
                if n_steps == len(self.times):
                    data[extra_key] = np.concatenate(([self.path_data[extra_key][0]], self.path_data[extra_key][:-1]))
                    data[f"{extra_key}_next"] = np.concatenate(([self.path_data[extra_key][0]], self.path_data[extra_key][1:]))
                else:
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

    def step_frame(self, split: SplitType | None = None) -> pd.DataFrame:
        selected: list[SplitType] = [split] if split else list(ALL_SPLITS)
        frames: list[pd.DataFrame] = []
        for key in selected:
            for ep in self.episodes[cast(SplitType, key)]:
                frames.append(ep.step_frame())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def episode_table(self, split: SplitType | None = None) -> pd.DataFrame:
        selected: list[SplitType] = [split] if split else list(ALL_SPLITS)
        rows: list[dict[str, Any]] = []
        for key in selected:
            for ep in self.episodes[cast(SplitType, key)]:
                costs = ep.costs
                trade_costs = ep.trade_costs
                liquidation_costs = ep.liquidation_costs
                total_cost = _nansum(costs)
                loss_values = (
                    np.asarray([np.nan if x is None else float(x) for x in ep.losses], dtype=float)
                    if ep.losses
                    else np.asarray([], dtype=float)
                )
                finite_losses = loss_values[np.isfinite(loss_values)] if loss_values.size > 0 else np.asarray([], dtype=float)
                rows.append(
                    {
                        "split": key,
                        "episode_idx": ep.episode_idx,
                        "n_steps": len(ep.actions),
                        "total_cost": total_cost,
                        "mean_step_cost": _nanmean(costs),
                        "std_step_cost": _nanstd(costs),
                        "total_trade_cost": _nansum(trade_costs),
                        "total_liquidation_cost": _nansum(liquidation_costs),
                        "mean_loss": float(finite_losses.mean()) if finite_losses.size > 0 else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    def split_summary(self, split: SplitType | None = None, risk_lambda: float = 1.5) -> pd.DataFrame:
        ep = self.episode_table(split=split)
        if ep.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for split_name, g in ep.groupby("split", sort=False):
            mean_cost = float(g["total_cost"].mean())
            std_cost = float(g["total_cost"].std(ddof=0))
            skew_cost = float(_nanskewness(g["total_cost"].tolist()))
            rows.append(
                {
                    "split": split_name,
                    "episodes": int(len(g)),
                    "mean_total_cost": mean_cost,
                    "std_total_cost": std_cost,
                    "skew_total_cost": skew_cost,
                    "y_objective": mean_cost + risk_lambda * std_cost,
                }
            )
        return pd.DataFrame(rows)

