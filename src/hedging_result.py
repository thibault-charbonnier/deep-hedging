from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .utils.helpers import nanskewness

SplitType = Literal["train", "eval_agent", "eval_benchmark"]
ALL_SPLITS: tuple[SplitType, SplitType, SplitType] = ("train", "eval_agent", "eval_benchmark")


def _safe(op, values) -> float:
    """Apply ``op`` (np.nanmean / nanstd / nansum) to ``values``; return NaN if empty."""
    return float(op(values)) if values else np.nan


@dataclass
class EpisodeResult:
    """Per-step log for a single episode (one simulated path)."""

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
    agent_infos: list[dict[str, Any]] = field(default_factory=list)

    def add_step(
        self,
        action: float,
        info: dict[str, Any],
        loss: float | None = None,
        agent_info: dict[str, Any] | None = None,
    ) -> None:
        """Append one step: action, env info (reward/cost/trade/liq), optional loss and agent info."""
        self.actions.append(float(action))
        self.rewards.append(float(info.get("reward", np.nan)))
        self.costs.append(float(info.get("cost", np.nan)))
        self.trade_costs.append(float(info.get("trade_cost", 0.0)))
        self.liquidation_costs.append(float(info.get("liquidation_cost", 0.0)))
        self.losses.append(None if loss is None else float(loss))
        self.agent_infos.append({} if agent_info is None else agent_info)

    def step_frame(self) -> pd.DataFrame:
        """Return a long-format DataFrame with one row per recorded step.

        The first row is the setup step: its ``*`` and ``*_next`` values
        are both taken at t=0 (no transition yet). For i >= 1, row i
        pairs values at index ``i-1`` (current) with index ``i`` (next).
        Extra path fields (``sigma``, ``variance``) are included when
        present.
        """
        n_steps = len(self.actions)

        def _pair(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            first = series[0]
            return (np.concatenate(([first], series[:-1])),
                    np.concatenate(([first], series[1:])))

        time, time_next = _pair(self.times)
        spot, spot_next = _pair(self.path_data["S"])
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
                cur, nxt = _pair(self.path_data[extra_key])
                data[extra_key] = cur
                data[f"{extra_key}_next"] = nxt
        return pd.DataFrame(data)


class HedgingResult:
    """Container holding all ``EpisodeResult`` instances across train/eval splits."""

    def __init__(self) -> None:
        self.episodes: dict[SplitType, list[EpisodeResult]] = {
            "train": [],
            "eval_agent": [],
            "eval_benchmark": [],
        }

    def add_episode(self, episode_result: EpisodeResult, type: SplitType) -> None:
        """Register ``episode_result`` under the given split."""
        self.episodes[type].append(episode_result)

    def step_frame(self, split: SplitType | None = None) -> pd.DataFrame:
        """Concatenate per-step frames over the selected split(s). All splits if None."""
        selected: list[SplitType] = [split] if split else list(ALL_SPLITS)
        frames: list[pd.DataFrame] = []
        for key in selected:
            for ep in self.episodes[key]:
                frames.append(ep.step_frame())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def episode_table(self, split: SplitType | None = None) -> pd.DataFrame:
        """Return one row per episode with aggregate cost/loss statistics."""
        selected: list[SplitType] = [split] if split else list(ALL_SPLITS)
        rows: list[dict[str, Any]] = []
        for key in selected:
            for ep in self.episodes[key]:
                costs = ep.costs
                trade_costs = ep.trade_costs
                liquidation_costs = ep.liquidation_costs
                total_cost = _safe(np.nansum, costs)
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
                        "mean_step_cost": _safe(np.nanmean, costs),
                        "std_step_cost": _safe(np.nanstd, costs),
                        "total_trade_cost": _safe(np.nansum, trade_costs),
                        "total_liquidation_cost": _safe(np.nansum, liquidation_costs),
                        "mean_loss": float(finite_losses.mean()) if finite_losses.size > 0 else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    def split_summary(self, split: SplitType | None = None, risk_lambda: float = 1.5) -> pd.DataFrame:
        """Aggregate per-split stats of ``total_cost`` across episodes.

        For each split: episode count, mean, std (ddof=0), skew, and the
        mean-variance objective ``Y = mean + risk_lambda * std`` used as
        the main comparison metric.
        """
        ep = self.episode_table(split=split)
        if ep.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for split_name, g in ep.groupby("split", sort=False):
            mean_cost = float(g["total_cost"].mean())
            std_cost = float(g["total_cost"].std(ddof=0))
            skew_cost = float(nanskewness(g["total_cost"].tolist()))
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

