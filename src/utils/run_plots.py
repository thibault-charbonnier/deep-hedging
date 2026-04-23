from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..hedging_result import _nanskewness

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

COLOR_RL = "#2E86AB"
COLOR_BM = "#E07A5F"
COLOR_NEUTRAL = "#6C757D"
COLOR_POSITIVE = "#52B788"
COLOR_NEGATIVE = "#C1121F"


def _load_csv_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _load_run_artifacts(run_id: str, outputs_dir: str | Path | None) -> dict[str, object]:
    run_dir = (Path(outputs_dir) if outputs_dir is not None else Path(__file__).resolve().parents[2] / "outputs") / run_id
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    artifacts: dict[str, object] = {
        "cfg": cfg,
        "rl_steps": pd.read_csv(run_dir / "data" / "eval_agent_steps.csv"),
        "bm_steps": pd.read_csv(run_dir / "data" / "eval_benchmark_steps.csv"),
        "rl_episodes": pd.read_csv(run_dir / "tables" / "eval_agent_episodes.csv"),
        "bm_episodes": pd.read_csv(run_dir / "tables" / "eval_benchmark_episodes.csv"),
        "rl_summary": pd.read_csv(run_dir / "tables" / "eval_agent_summary.csv"),
        "bm_summary": pd.read_csv(run_dir / "tables" / "eval_benchmark_summary.csv"),
        "train_steps": _load_csv_optional(run_dir / "data" / "train_steps.csv"),
        "train_episodes": _load_csv_optional(run_dir / "tables" / "train_episodes.csv"),
    }
    return artifacts


def _get_scalar(df: pd.DataFrame | None, column: str) -> float:
    if df is None or df.empty or column not in df.columns:
        return float("nan")
    return float(df.iloc[0][column])


def _bar_with_labels(ax: plt.Axes, values: list[float], title: str, ylabel: str) -> None:
    bars = ax.bar([0, 1], values, width=0.5, color=[COLOR_RL, COLOR_BM], edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks([0, 1], ["RL", "Benchmark"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _plot_training_loss(ax: plt.Axes, train_steps: pd.DataFrame | None) -> None:
    ax.set_title("Training Loss")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")
    if train_steps is None or train_steps.empty or "loss" not in train_steps.columns:
        ax.text(0.5, 0.5, "No training data available", ha="center", va="center", color=COLOR_NEUTRAL, fontsize=10)
        ax.set_axis_off()
        return
    loss_values = train_steps["loss"].dropna().to_numpy(dtype=float)
    if loss_values.size == 0:
        _hide_training_panel(ax)
        return
    x = np.arange(len(loss_values), dtype=int)
    smoothed = pd.Series(loss_values).rolling(200, min_periods=1).mean().to_numpy()
    ax.plot(x, loss_values, color=COLOR_NEUTRAL, alpha=0.25, label="raw")
    ax.plot(x, smoothed, color=COLOR_RL, label="smoothed")
    ax.set_yscale("log")
    ax.legend()


def _plot_training_episode_cost(ax: plt.Axes, train_episodes: pd.DataFrame | None) -> None:
    ax.set_title("Training Episode Cost")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total cost")
    if train_episodes is None or train_episodes.empty or "total_cost" not in train_episodes.columns:
        ax.text(0.5, 0.5, "No training data available", ha="center", va="center", color=COLOR_NEUTRAL, fontsize=10)
        ax.set_axis_off()
        return
    x = train_episodes["episode_idx"].to_numpy()
    y = train_episodes["total_cost"].astype(float).to_numpy()
    series = pd.Series(y)
    rolling_mean = series.rolling(100, min_periods=1).mean().to_numpy()
    rolling_std = series.rolling(100, min_periods=1).std().fillna(0.0).to_numpy()
    ax.plot(x, y, color=COLOR_NEUTRAL, alpha=0.2, label="raw")
    ax.fill_between(x, rolling_mean - rolling_std, rolling_mean + rolling_std, color=COLOR_RL, alpha=0.2)
    ax.plot(x, rolling_mean, color=COLOR_RL, linewidth=2, label="mean ± 1σ")
    ax.legend()


def plot_run(run_id: str, outputs_dir: str | Path | None = None) -> None:
    artifacts = _load_run_artifacts(run_id, outputs_dir)
    cfg = artifacts["cfg"]
    rl_steps = artifacts["rl_steps"]
    bm_steps = artifacts["bm_steps"]
    rl_episodes = artifacts["rl_episodes"]
    bm_episodes = artifacts["bm_episodes"]
    rl_summary = artifacts["rl_summary"]
    bm_summary = artifacts["bm_summary"]
    train_steps = artifacts["train_steps"]
    train_episodes = artifacts["train_episodes"]

    y_rl = _get_scalar(rl_summary, "y_objective")
    y_bm = _get_scalar(bm_summary, "y_objective")
    mean_rl = _get_scalar(rl_summary, "mean_total_cost")
    mean_bm = _get_scalar(bm_summary, "mean_total_cost")
    std_rl = _get_scalar(rl_summary, "std_total_cost")
    std_bm = _get_scalar(bm_summary, "std_total_cost")
    improvement_pct = 100.0 * (y_bm - y_rl) / y_bm if y_bm not in (0.0, np.nan) and np.isfinite(y_bm) and y_bm != 0 else float("nan")

    skew_rl = _nanskewness(rl_episodes["total_cost"].tolist()) if not rl_episodes.empty and "total_cost" in rl_episodes.columns else float("nan")
    skew_bm = _nanskewness(bm_episodes["total_cost"].tolist()) if not bm_episodes.empty and "total_cost" in bm_episodes.columns else float("nan")

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = np.asarray(axes)

    ax = axes[0, 0]
    ax.set_title("RL Holding vs Benchmark Holding")
    ax.hexbin(bm_steps["action"], rl_steps["action"], gridsize=40, cmap="Blues", mincnt=1)
    lims = [
        float(min(bm_steps["action"].min(), rl_steps["action"].min())),
        float(max(bm_steps["action"].max(), rl_steps["action"].max())),
    ]
    ax.plot(lims, lims, linestyle="--", color=COLOR_NEUTRAL, linewidth=1.0)
    ax.text(0.05, 0.95, "Under-hedge", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=COLOR_NEUTRAL, alpha=0.8)
    ax.text(0.95, 0.05, "Over-hedge", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=COLOR_NEUTRAL, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Benchmark holding")
    ax.set_ylabel("RL holding")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax = axes[0, 1]
    ax.set_title("Total Cost Distribution")
    rl_costs = rl_episodes["total_cost"].astype(float).to_numpy() if not rl_episodes.empty and "total_cost" in rl_episodes.columns else np.asarray([], dtype=float)
    bm_costs = bm_episodes["total_cost"].astype(float).to_numpy() if not bm_episodes.empty and "total_cost" in bm_episodes.columns else np.asarray([], dtype=float)
    if rl_costs.size and bm_costs.size:
        ax.hist(rl_costs, bins=30, density=True, alpha=0.55, edgecolor="white", color=COLOR_RL, label="RL")
        ax.hist(bm_costs, bins=30, density=True, alpha=0.55, edgecolor="white", color=COLOR_BM, label="Benchmark")
        ax.axvline(float(np.mean(rl_costs)), color=COLOR_RL, linestyle="--")
        ax.axvline(float(np.mean(bm_costs)), color=COLOR_BM, linestyle="--")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No evaluation data available", ha="center", va="center", color=COLOR_NEUTRAL, fontsize=10)
    ax.set_xlabel("Total hedging cost")
    ax.set_ylabel("Density")

    _bar_with_labels(axes[1, 0], [y_rl, y_bm], "Y Objective", "Y objective")
    _bar_with_labels(axes[1, 1], [mean_rl, mean_bm], "Mean Total Cost", "Mean total cost")
    _bar_with_labels(axes[2, 0], [std_rl, std_bm], "Std Total Cost", "Std total cost")
    _bar_with_labels(axes[2, 1], [skew_rl, skew_bm], "Skewness", "Skewness")
    axes[2, 1].axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)

    _plot_training_loss(axes[3, 0], train_steps if isinstance(train_steps, pd.DataFrame) else None)
    _plot_training_episode_cost(axes[3, 1], train_episodes if isinstance(train_episodes, pd.DataFrame) else None)

    fig.suptitle(f"Hedging Run — {run_id}", fontsize=14, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.975,
        f"Process={cfg['run']['process']} | Agent={cfg['run']['agent']} | Benchmark={cfg['run']['benchmark']} | "
        f"κ={cfg['hedging_env']['transaction_cost']:.1%} | T={cfg['simulation']['maturity']:.4f}y | "
        f"Y improvement={improvement_pct:+.2f}%",
        ha="center",
        fontsize=10,
        alpha=0.7,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


