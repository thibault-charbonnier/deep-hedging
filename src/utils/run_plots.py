from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import display


def _load_train_data(run_id: str, outputs_dir: str | Path | None = None) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    root = Path(outputs_dir) if outputs_dir is not None else Path(__file__).resolve().parents[2] / "outputs"
    run_dir = root / run_id
    train_steps = pd.read_csv(run_dir / "data" / "train_steps.csv")
    train_ep = pd.read_csv(run_dir / "tables" / "train_episodes.csv")
    return run_dir, train_steps, train_ep


def _plot_train_loss_core(run_id: str, train_steps: pd.DataFrame, smooth_window: int, show_raw: bool) -> pd.DataFrame:
    loss_df = train_steps.loc[train_steps["loss"].notna(), ["loss"]].copy()
    loss_df["update_idx"] = np.arange(len(loss_df), dtype=int)
    loss_df["loss_smooth"] = loss_df["loss"].rolling(smooth_window, min_periods=1).mean()

    plt.figure(figsize=(9, 4))
    if show_raw:
        plt.plot(loss_df["update_idx"], loss_df["loss"], alpha=0.25, label="loss raw")
    plt.plot(
        loss_df["update_idx"],
        loss_df["loss_smooth"],
        linewidth=2,
        label=f"loss smooth ({smooth_window})",
    )
    plt.title(f"Training loss - {run_id}")
    plt.xlabel("Update index")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return loss_df


def plot_train_loss(
    run_id: str,
    outputs_dir: str | Path | None = None,
    smooth_window: int = 200,
    show_raw: bool = True,
) -> pd.DataFrame:
    """Plot train loss for one run."""
    run_dir, train_steps, _ = _load_train_data(run_id, outputs_dir)
    loss_df = _plot_train_loss_core(run_id, train_steps, smooth_window, show_raw)

    print(f"Run utilise: {run_id}")
    print(f"Source: {run_dir / 'data' / 'train_steps.csv'}")
    return loss_df


def plot_run(run_id: str, outputs_dir: str | Path | None = None) -> None:
    from src.hedging_result import _nanskewness

    root = Path(outputs_dir) if outputs_dir is not None else Path(__file__).resolve().parents[2] / "outputs"
    run_dir = root / run_id

    rl_summary = pd.read_csv(run_dir / "tables" / "eval_agent_summary.csv")
    bm_summary = pd.read_csv(run_dir / "tables" / "eval_benchmark_summary.csv")
    rl_steps = pd.read_csv(run_dir / "data" / "eval_agent_steps.csv")
    bm_steps = pd.read_csv(run_dir / "data" / "eval_benchmark_steps.csv")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    # Load episode data for skewness calculation
    rl_episodes = pd.read_csv(run_dir / "tables" / "eval_agent_episodes.csv")
    bm_episodes = pd.read_csv(run_dir / "tables" / "eval_benchmark_episodes.csv")

    y_rl = float(rl_summary.loc[0, "y_objective"])
    y_bm = float(bm_summary.loc[0, "y_objective"])
    mean_rl = float(rl_summary.loc[0, "mean_total_cost"])
    std_rl = float(rl_summary.loc[0, "std_total_cost"])
    mean_bm = float(bm_summary.loc[0, "mean_total_cost"])
    std_bm = float(bm_summary.loc[0, "std_total_cost"])
    improvement_pct = 100.0 * (y_bm - y_rl) / y_bm if y_bm != 0 else float("nan")

    # Calculate skewness from episode data
    skew_rl = _nanskewness(rl_episodes["total_cost"].tolist())
    skew_bm = _nanskewness(bm_episodes["total_cost"].tolist())

    cmp = pd.DataFrame(
        [
            {"method": "RL (DeepDPG)", "mean_total_cost": mean_rl, "std_total_cost": std_rl, "skew_total_cost": skew_rl, "y_objective": y_rl},
            {"method": "Benchmark", "mean_total_cost": mean_bm, "std_total_cost": std_bm, "skew_total_cost": skew_bm, "y_objective": y_bm},
        ]
    )
    print(f"Run utilise: {run_id}")
    print(f"Improvement RL vs Benchmark = {improvement_pct:.2f}%")
    display(cmp)

    # Additional skewness statistics
    print("\nSkewness Analysis:")
    print(f"  RL Agent skewness:    {skew_rl:>8.4f}")
    print(f"  Benchmark skewness:   {skew_bm:>8.4f}")
    print(f"  Difference (RL-BM):   {skew_rl - skew_bm:>8.4f}")

    plt.figure(figsize=(6, 4))
    plt.bar(cmp["method"], cmp["y_objective"])
    plt.ylabel("Y objective")
    plt.title(f"RL vs Benchmark | Improvement = {improvement_pct:.2f}%")
    plt.tight_layout()
    plt.show()

    K = float(cfg["derivative"]["strike"])
    r = float(cfg["derivative"].get("rf_rate", 0.0))
    q = float(cfg["derivative"].get("div_rate", 0.0))
    sigma = float(cfg["simulation"]["gbm"]["sigma"])
    maturity = float(cfg["simulation"]["maturity"])

    def bs_delta_call(spot, strike, ttm, rate, div, vol):
        if ttm <= 1e-12:
            return 1.0 if spot > strike else 0.0
        d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol**2) * ttm) / (vol * np.sqrt(ttm))
        return np.exp(-div * ttm) * norm.cdf(d1)

    ttm_rl = np.maximum(maturity - rl_steps["time"].to_numpy(), 0.0)
    delta_bm = np.array([bs_delta_call(s, K, t, r, q, sigma) for s, t in zip(bm_steps["spot"].to_numpy(), np.maximum(maturity - bm_steps["time"].to_numpy(), 0.0))])
    delta_rl = np.array([bs_delta_call(s, K, t, r, q, sigma) for s, t in zip(rl_steps["spot"].to_numpy(), ttm_rl)])
    hold_rl = rl_steps["action"].to_numpy()
    hold_bm = bm_steps["action"].to_numpy()

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(delta_rl, hold_rl, s=4, alpha=0.25, label="RL policy")
    lo = min(delta_rl.min(), hold_rl.min())
    hi = max(delta_rl.max(), hold_rl.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="delta line (y=x)")
    plt.xlabel("Delta hedge (%)")
    plt.ylabel("Current holding (%)")
    plt.title("Under-hedge / Over-hedge vs Delta")
    plt.legend()
    plt.tight_layout()
    plt.show()

    rl_vs_bm = pd.DataFrame(
        {
            "episode_idx": rl_steps["episode_idx"],
            "step_idx": rl_steps["step_idx"],
            "delta_rl": hold_rl,
            "delta_benchmark": hold_bm,
            "trade_cost_rl": rl_steps["trade_cost"],
            "trade_cost_benchmark": bm_steps["trade_cost"],
        }
    )
    rl_vs_bm["delta_diff"] = rl_vs_bm["delta_rl"] - rl_vs_bm["delta_benchmark"]
    rl_vs_bm["trade_cost_total_rl"] = rl_steps["trade_cost"] + rl_steps["liquidation_cost"]
    rl_vs_bm["trade_cost_total_bm"] = bm_steps["trade_cost"] + bm_steps["liquidation_cost"]

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(rl_vs_bm["delta_benchmark"], rl_vs_bm["delta_rl"], s=4, alpha=0.25)
    lo = min(rl_vs_bm["delta_benchmark"].min(), rl_vs_bm["delta_rl"].min())
    hi = max(rl_vs_bm["delta_benchmark"].max(), rl_vs_bm["delta_rl"].max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("Delta benchmark")
    plt.ylabel("Delta RL")
    plt.title("RL vs Benchmark Delta")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.hist(rl_vs_bm["delta_diff"], bins=60, alpha=0.85)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Delta RL - Delta benchmark")
    plt.ylabel("Count")
    plt.title("Distribution de l'ecart de hedge")
    plt.tight_layout()
    plt.show()

    tc = pd.DataFrame(
        [
            {"method": "RL", "total_trade_cost": rl_vs_bm["trade_cost_total_rl"].sum()},
            {"method": "Benchmark", "total_trade_cost": rl_vs_bm["trade_cost_total_bm"].sum()},
        ]
    )
    plt.figure(figsize=(6, 4))
    plt.bar(tc["method"], tc["total_trade_cost"])
    plt.ylabel("Total transaction cost")
    plt.title("RL vs Benchmark - Cost de trade complet")
    plt.tight_layout()
    plt.show()

    # Skewness comparison
    plt.figure(figsize=(6, 4))
    plt.bar(cmp["method"], cmp["skew_total_cost"])
    plt.ylabel("Skewness of total cost")
    plt.title("RL vs Benchmark - Skewness")
    plt.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Distribution of total costs with skewness visualization
    plt.figure(figsize=(10, 5))
    plt.hist(rl_episodes["total_cost"], bins=30, alpha=0.6, label=f"RL (skew={skew_rl:.3f})", color="blue")
    plt.hist(bm_episodes["total_cost"], bins=30, alpha=0.6, label=f"Benchmark (skew={skew_bm:.3f})", color="orange")
    plt.xlabel("Total hedging cost")
    plt.ylabel("Count")
    plt.title("Distribution of total cost - RL vs Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_run2(
    run_id: str,
    outputs_dir: str | Path | None = None,
    smooth_window: int = 200,
    show_raw: bool = True,
) -> None:
    """Simple learning diagnostics for one run."""
    _, train_steps, train_ep = _load_train_data(run_id, outputs_dir)

    # 1) Loss over updates (raw + smooth)
    _plot_train_loss_core(run_id, train_steps, smooth_window, show_raw)

    # 2) Total cost per episode (raw + smooth)
    ep_idx = train_ep["episode_idx"]
    total_cost = train_ep["total_cost"]
    total_cost_smooth = total_cost.rolling(100, min_periods=1).mean()
    plt.figure(figsize=(9, 4))
    plt.plot(ep_idx, total_cost, alpha=0.25, label="total_cost raw")
    plt.plot(ep_idx, total_cost_smooth, linewidth=2, label="total_cost smooth (100)")
    plt.title(f"Training episode cost - {run_id}")
    plt.xlabel("Episode")
    plt.ylabel("Total cost")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Early vs late total-cost distributions
    split_idx = len(train_ep) // 2
    early = train_ep.iloc[:split_idx]["total_cost"]
    late = train_ep.iloc[split_idx:]["total_cost"]
    plt.figure(figsize=(8, 4))
    plt.hist(early, bins=30, alpha=0.6, label="early")
    plt.hist(late, bins=30, alpha=0.6, label="late")
    plt.title(f"Early vs late total cost - {run_id}")
    plt.xlabel("Total cost")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) Policy stabilization proxy: rolling std of actions
    action_std = train_steps["action"].rolling(500, min_periods=1).std().fillna(0.0)
    plt.figure(figsize=(9, 4))
    plt.plot(action_std, linewidth=2)
    plt.title(f"Rolling std(action) - {run_id}")
    plt.xlabel("Train step")
    plt.ylabel("Rolling std")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all_graphs(
    run_id: str,
    outputs_dir: str | Path | None = None,
    smooth_window: int = 200,
    show_raw: bool = True,
) -> None:
    """Run all available plots for one run (learning + RL vs benchmark)."""
    plot_run2(
        run_id=run_id,
        outputs_dir=outputs_dir,
        smooth_window=smooth_window,
        show_raw=show_raw,
    )
    plot_run(run_id=run_id, outputs_dir=outputs_dir)
