from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..hedging_result import _nanskewness
from ..valuation.bs_valuation import BSValuation


def _option_price_t0(cfg: dict) -> float:
    """BS price of the option at t=0 from cfg (matches main.py._option_price_t0)."""
    maturity = float(cfg["simulation"]["maturity"])
    spot = float(cfg["simulation"]["S0"])
    sigma = float(cfg["simulation"]["gbm"]["sigma"])
    engine = BSValuation(
        strike=cfg["derivative"]["strike"],
        maturity=maturity,
        rate=cfg["derivative"].get("rf_rate", 0.0),
        dividend=cfg["derivative"].get("div_rate", 0.0),
        option_type=cfg.get("derivative", {}).get("option_type", "call"),
    )
    p, _ = engine.price_and_delta(spot=spot, t=0.0, sigma=sigma)
    return abs(float(p))


def _cbrt_m3_proxy(values: list[float] | np.ndarray, eps: float = 1e-8) -> float:
    """Signed cube root of the third central moment.

    Matches the actor-loss proxy used in SkewDDPG. Lives on the same
    scale as `std` (and thus as `mean_total_cost`, `std_total_cost`),
    not the dimensionless statistical skewness.
    """
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 3:
        return float("nan")
    m3 = float(np.mean((finite - finite.mean()) ** 3))
    return float(np.sign(m3) * (abs(m3) + eps) ** (1.0 / 3.0))

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


def _bar_with_labels(
    ax: plt.Axes,
    values: list[float],
    title: str,
    ylabel: str,
    usd_scale: float | None = None,
) -> None:
    bars = ax.bar([0, 1], values, width=0.5, color=[COLOR_RL, COLOR_BM], edgecolor="white")
    if usd_scale is not None and np.isfinite(usd_scale):
        labels = [f"{v:.2f}%\n(${v * usd_scale:.3f})" for v in values]
    else:
        labels = [f"{v:.2f}" for v in values]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    ax.set_xticks([0, 1], ["RL", "Benchmark"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _stacked_y_bar(
    ax: plt.Axes,
    title: str,
    *,
    mean_rl: float, mean_bm: float,
    std_rl: float, std_bm: float,
    pen_rl: float, pen_bm: float,
    risk_lambda: float,
    skew_lambda: float,
    skew_penalty: str,
    include_skew: bool,
    usd_scale: float | None = None,
) -> None:
    """Render Y(0) as a stacked bar showing mean / λ_std·std / λ_skew·penalty(skew_proxy)."""
    segments = [
        ("Mean", mean_rl, mean_bm, "#4A90E2"),
        (f"λ_std·std  (λ={risk_lambda:g})", risk_lambda * std_rl, risk_lambda * std_bm, "#F5A623"),
    ]
    if include_skew:
        segments.append(
            (f"λ_skew·{skew_penalty}(skew_proxy)  (λ={skew_lambda:g})",
             skew_lambda * pen_rl, skew_lambda * pen_bm, "#D0021B")
        )

    x = np.array([0, 1])
    bot = np.zeros(2, dtype=float)
    for label, vrl, vbm, color in segments:
        vals = np.array([vrl, vbm], dtype=float)
        ax.bar(x, vals, bottom=bot, width=0.5, color=color, edgecolor="white", label=label)
        for i, v in enumerate(vals):
            if abs(v) > 1.0:
                ax.text(x[i], bot[i] + v / 2.0, f"{v:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        bot = bot + vals

    for i, total in enumerate(bot):
        if usd_scale is not None and np.isfinite(usd_scale):
            label = f"{total:.2f}%\n(${total * usd_scale:.3f})"
        else:
            label = f"{total:.2f}"
        ax.text(x[i], total, label, ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks([0, 1], ["RL", "Benchmark"])
    ax.set_ylabel("Y(0)")
    ax.set_title(title)
    ax.axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8)


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


def plot_run(
    run_id: str,
    outputs_dir: str | Path | None = None,
    include_skew_in_y: bool = False,
) -> None:
    """Render the standard run dashboard.

    If `include_skew_in_y` is True, Y(0) is recomputed locally as
        Y = mean + λ_std · std + λ_skew · skew_proxy
    using the cube-root-of-m₃ proxy and the λ's from cfg["hedging_agent"].
    Otherwise Y(0) = mean + λ_std · std is read from *_summary.csv
    (the value written at run time).
    """
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

    mean_rl = _get_scalar(rl_summary, "mean_total_cost")
    mean_bm = _get_scalar(bm_summary, "mean_total_cost")
    std_rl = _get_scalar(rl_summary, "std_total_cost")
    std_bm = _get_scalar(bm_summary, "std_total_cost")

    skew_rl = _nanskewness(rl_episodes["total_cost"].tolist()) if not rl_episodes.empty and "total_cost" in rl_episodes.columns else float("nan")
    skew_bm = _nanskewness(bm_episodes["total_cost"].tolist()) if not bm_episodes.empty and "total_cost" in bm_episodes.columns else float("nan")

    # Cube-root-of-m3 proxy on the rescaled total_cost (% option price).
    # Same formula as the SkewDDPG actor-loss penalty, so it lives on the
    # same scale as mean/std in the plots above.
    cbrt_rl = _cbrt_m3_proxy(rl_episodes["total_cost"].tolist()) if not rl_episodes.empty and "total_cost" in rl_episodes.columns else float("nan")
    cbrt_bm = _cbrt_m3_proxy(bm_episodes["total_cost"].tolist()) if not bm_episodes.empty and "total_cost" in bm_episodes.columns else float("nan")

    risk_lambda = float(cfg.get("hedging_agent", {}).get("risk_lambda", 1.5))
    skew_lambda = float(cfg.get("hedging_agent", {}).get("skew_lambda", 0.0))
    skew_penalty = str(cfg.get("hedging_agent", {}).get("skew_penalty", "positive")).lower()

    # All *_total_cost columns in the CSVs are rescaled ×100/option_price_t0,
    # so multiplying a plotted value by (option_price_t0 / 100) gives back
    # the raw currency amount (in units of S).
    try:
        option_price_t0 = _option_price_t0(cfg)
        usd_scale = option_price_t0 / 100.0
    except Exception:
        option_price_t0 = float("nan")
        usd_scale = None

    # Real transaction cost in $ (mean over episodes).  `total_trade_cost`
    # is the per-episode sum of all κ·|S·ΔH| fees INCLUDING setup and
    # terminal liquidation — it's what the strategy actually pays out.
    def _mean_trade_cost_usd(df: pd.DataFrame) -> float:
        if df.empty or "total_trade_cost" not in df.columns or usd_scale is None:
            return float("nan")
        return float(df["total_trade_cost"].mean() * usd_scale)

    trade_rl_usd = _mean_trade_cost_usd(rl_episodes)
    trade_bm_usd = _mean_trade_cost_usd(bm_episodes)

    # Apply the SAME penalty function the actor uses on skew_proxy, so the
    # stacked Y(0) bar matches what the agent actually optimises.
    def _penalty(x: float) -> float:
        if not np.isfinite(x):
            return float("nan")
        if skew_penalty == "absolute":
            return abs(x)
        if skew_penalty == "signed":
            return x
        return max(0.0, x)  # default "positive" → ReLU, right-tail only

    pen_rl = _penalty(cbrt_rl)
    pen_bm = _penalty(cbrt_bm)

    if include_skew_in_y:
        y_rl = mean_rl + risk_lambda * std_rl + skew_lambda * pen_rl
        y_bm = mean_bm + risk_lambda * std_bm + skew_lambda * pen_bm
        y_title = f"Y(0) = mean + {risk_lambda:g}·std + {skew_lambda:g}·{skew_penalty}(skew_proxy)"
    else:
        y_rl = _get_scalar(rl_summary, "y_objective")
        y_bm = _get_scalar(bm_summary, "y_objective")
        y_title = f"Y(0) = mean + {risk_lambda:g}·std"
    improvement_pct = 100.0 * (y_bm - y_rl) / y_bm if y_bm not in (0.0, np.nan) and np.isfinite(y_bm) and y_bm != 0 else float("nan")

    fig, axes = plt.subplots(5, 2, figsize=(14, 22))
    axes = np.asarray(axes)

    ax = axes[0, 0]
    ax.set_title("RL Holding vs Benchmark Holding")
    ax.hexbin(bm_steps["action"], rl_steps["action"], gridsize=40, cmap="Blues", mincnt=1)
    lims = [
        float(min(bm_steps["action"].min(), rl_steps["action"].min())),
        float(max(bm_steps["action"].max(), rl_steps["action"].max())),
    ]
    ax.plot(lims, lims, linestyle="--", color=COLOR_NEUTRAL, linewidth=1.0)
    ax.text(0.05, 0.95, "Over-hedge", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=COLOR_NEUTRAL,alpha=0.8)
    ax.text(0.95, 0.05, "Under-hedge", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=COLOR_NEUTRAL,alpha=0.8)
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

    _stacked_y_bar(
        axes[1, 0],
        y_title,
        mean_rl=mean_rl, mean_bm=mean_bm,
        std_rl=std_rl, std_bm=std_bm,
        pen_rl=pen_rl, pen_bm=pen_bm,
        risk_lambda=risk_lambda,
        skew_lambda=skew_lambda,
        skew_penalty=skew_penalty,
        include_skew=include_skew_in_y,
    )
    _bar_with_labels(axes[1, 1], [mean_rl, mean_bm], "Mean Total Cost", "Mean total cost")
    _bar_with_labels(axes[2, 0], [std_rl, std_bm], "Std Total Cost", "Std total cost")
    _bar_with_labels(axes[2, 1], [skew_rl, skew_bm], "Skewness (statistical, dimensionless)", "Skewness")
    axes[2, 1].axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)

    _bar_with_labels(
        axes[3, 0],
        [pen_rl, pen_bm],
        f"Skew Penalty — {skew_penalty}(skew_proxy) (% option price)",
        f"{skew_penalty}(skew_proxy)",
    )
    axes[3, 0].axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)

    # Real transaction cost per episode, in $ (setup + per-step rebal + terminal liquidation).
    ax_tc = axes[3, 1]
    bars = ax_tc.bar([0, 1], [trade_rl_usd, trade_bm_usd], width=0.5,
                     color=[COLOR_RL, COLOR_BM], edgecolor="white")
    ax_tc.bar_label(bars, labels=[f"${v:.4f}" for v in (trade_rl_usd, trade_bm_usd)],
                    padding=3, fontsize=9)
    ax_tc.set_xticks([0, 1], ["RL", "Benchmark"])
    ax_tc.set_ylabel("Transaction cost ($ / option)")
    ax_tc.set_title("Avg transaction cost paid per episode ($)")

    _plot_training_loss(axes[4, 0], train_steps if isinstance(train_steps, pd.DataFrame) else None)
    _plot_training_episode_cost(axes[4, 1], train_episodes if isinstance(train_episodes, pd.DataFrame) else None)

    fig.suptitle(f"Hedging Run — {run_id}", fontsize=14, fontweight="bold", y=0.995)
    opt_str = f"Opt(t=0)=${option_price_t0:.3f}" if np.isfinite(option_price_t0) else ""
    fig.text(
        0.5,
        0.975,
        f"Process={cfg['run']['process']} | Agent={cfg['run']['agent']} | Benchmark={cfg['run']['benchmark']} | "
        f"κ={cfg['hedging_env']['transaction_cost']:.1%} | T={cfg['simulation']['maturity']:.4f}y | "
        f"{opt_str} | Y improvement={improvement_pct:+.2f}%",
        ha="center",
        fontsize=10,
        alpha=0.7,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


