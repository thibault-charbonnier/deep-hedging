from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

SplitType = Literal["train", "eval_agent", "eval_benchmark"]


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
    actions: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    trade_costs: list[float] = field(default_factory=list)
    liquidation_costs: list[float] = field(default_factory=list)
    losses: list[float | None] = field(default_factory=list)
    setup_action: float | None = None
    setup_reward: float | None = None
    setup_cost: float = 0.0
    setup_trade_cost: float = 0.0
    setup_loss: float | None = None

    def set_setup(self, action: float, info: dict[str, Any], loss: float | None = None) -> None:
        self.setup_action = float(action)
        self.setup_reward = float(info.get("reward", np.nan))
        self.setup_cost = float(info.get("cost", 0.0))
        self.setup_trade_cost = float(info.get("trade_cost", 0.0))
        self.setup_loss = None if loss is None else float(loss)

    def add_step(
        self,
        action: float,
        info: dict[str, Any],
        loss: float | None = None,
    ) -> None:
        self.actions.append(float(action))
        self.rewards.append(float(info.get("reward", np.nan)))
        self.costs.append(float(info.get("cost", np.nan)))
        self.trade_costs.append(float(info.get("trade_cost", 0.0)))
        self.liquidation_costs.append(float(info.get("liquidation_cost", 0.0)))
        self.losses.append(None if loss is None else float(loss))

    def step_frame(self) -> pd.DataFrame:
        n_steps = len(self.actions)
        has_setup = self.setup_action is not None
        row_count = n_steps + (1 if has_setup else 0)

        step_idx = list(range(n_steps))
        time = list(self.times[:n_steps])
        time_next = list(self.times[1 : n_steps + 1])
        spot = list(self.path_data["S"][:n_steps])
        spot_next = list(self.path_data["S"][1 : n_steps + 1])
        action = list(self.actions)
        reward = list(self.rewards)
        cost = list(self.costs)
        trade_cost = list(self.trade_costs)
        liquidation_cost = list(self.liquidation_costs)
        loss = list(self.losses)

        if has_setup:
            step_idx = [-1] + step_idx
            time = [self.times[0]] + time
            time_next = [self.times[0]] + time_next
            spot = [self.path_data["S"][0]] + spot
            spot_next = [self.path_data["S"][0]] + spot_next
            action = [float(self.setup_action)] + action
            reward = [float(self.setup_reward) if self.setup_reward is not None else np.nan] + reward
            cost = [float(self.setup_cost)] + cost
            trade_cost = [float(self.setup_trade_cost)] + trade_cost
            liquidation_cost = [0.0] + liquidation_cost
            loss = [self.setup_loss] + loss

        data: dict[str, Any] = {
            "split": [self.split] * row_count,
            "episode_idx": [self.episode_idx] * row_count,
            "step_idx": step_idx,
            "time": time,
            "time_next": time_next,
            "spot": spot,
            "spot_next": spot_next,
            "action": action,
            "reward": reward,
            "cost": cost,
            "trade_cost": trade_cost,
            "liquidation_cost": liquidation_cost,
            "loss": loss,
        }
        for extra_key in ("sigma", "variance"):
            if extra_key in self.path_data:
                extra = list(self.path_data[extra_key][:n_steps])
                extra_next = list(self.path_data[extra_key][1 : n_steps + 1])
                if has_setup:
                    extra = [self.path_data[extra_key][0]] + extra
                    extra_next = [self.path_data[extra_key][0]] + extra_next
                data[extra_key] = extra
                data[f"{extra_key}_next"] = extra_next
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

    def step_frame(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for eps in self.episodes.values():
            for ep in eps:
                frames.append(ep.step_frame())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def episode_table(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for split_key, eps in self.episodes.items():
            for ep in eps:
                step_cost_sum = float(np.nansum(ep.costs)) if ep.costs else 0.0
                total_cost = step_cost_sum + float(ep.setup_cost)
                loss_entries = [np.nan if x is None else float(x) for x in ep.losses]
                if ep.setup_loss is not None:
                    loss_entries = [float(ep.setup_loss)] + loss_entries
                loss_values = np.asarray(loss_entries, dtype=float) if loss_entries else np.asarray([], dtype=float)
                finite_losses = loss_values[np.isfinite(loss_values)] if loss_values.size > 0 else np.asarray([], dtype=float)
                rows.append(
                    {
                        "split": split_key,
                        "episode_idx": ep.episode_idx,
                        "n_steps": len(ep.actions),
                        "total_cost": total_cost,
                        "mean_step_cost": float(np.nanmean(ep.costs)) if ep.costs else float("nan"),
                        "std_step_cost": float(np.nanstd(ep.costs)) if ep.costs else float("nan"),
                        "total_trade_cost": (float(np.nansum(ep.trade_costs)) if ep.trade_costs else 0.0) + float(ep.setup_trade_cost),
                        "total_liquidation_cost": float(np.nansum(ep.liquidation_costs)) if ep.liquidation_costs else 0.0,
                        "mean_loss": float(finite_losses.mean()) if finite_losses.size > 0 else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    def split_summary(self, risk_lambda: float = 1.5) -> pd.DataFrame:
        ep = self.episode_table()
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

