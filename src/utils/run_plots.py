from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import display


def plot_run(run_id: str, outputs_dir: str | Path | None = None) -> None:
    root = Path(outputs_dir) if outputs_dir is not None else Path(__file__).resolve().parents[2] / "outputs"
    run_dir = root / run_id

    rl_summary = pd.read_csv(run_dir / "tables" / "eval_agent_summary.csv")
    bm_summary = pd.read_csv(run_dir / "tables" / "eval_benchmark_summary.csv")
    rl_steps = pd.read_csv(run_dir / "data" / "eval_agent_steps.csv")
    bm_steps = pd.read_csv(run_dir / "data" / "eval_benchmark_steps.csv")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    y_rl = float(rl_summary.loc[0, "y_objective"])
    y_bm = float(bm_summary.loc[0, "y_objective"])
    mean_rl = float(rl_summary.loc[0, "mean_total_cost"])
    std_rl = float(rl_summary.loc[0, "std_total_cost"])
    mean_bm = float(bm_summary.loc[0, "mean_total_cost"])
    std_bm = float(bm_summary.loc[0, "std_total_cost"])
    improvement_pct = 100.0 * (y_bm - y_rl) / y_bm if y_bm != 0 else float("nan")

    cmp = pd.DataFrame(
        [
            {"method": "RL (DeepDPG)", "mean_total_cost": mean_rl, "std_total_cost": std_rl, "y_objective": y_rl},
            {"method": "Benchmark", "mean_total_cost": mean_bm, "std_total_cost": std_bm, "y_objective": y_bm},
        ]
    )
    print(f"Run utilise: {run_id}")
    print(f"Improvement RL vs Benchmark = {improvement_pct:.2f}%")
    display(cmp)

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


