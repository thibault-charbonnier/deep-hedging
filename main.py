"""
Deep Hedging of Derivatives Using Reinforcement Learning
=========================================================
Reproduction and extension of Cao, Chen, Hull & Poulos (2021).

Project: ML for Finance — M2MO / ENSAE / Mastère FGR
"""

import logging
import copy
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as sp_stats
from rich.logging import RichHandler

from src.orchestrator import Orchestrator
from src.valuation import BSValuation
from src.utils.enums import AgentType, BenchmarkType, ProcessType
from src.utils.helpers import json_to_dict

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42

def reset_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

reset_seed()

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    force=True,
)
logger = logging.getLogger(__name__)

FIGURE_DIR = Path("figures")


# ======================================================================
# Helpers
# ======================================================================

def total_costs(result, split_name: str) -> list[float]:
    return [sum(ep.costs) for ep in result.episodes[split_name]]


def compute_metrics(costs: list[float], lam: float = 1.5,
                    confidence: float = 0.95) -> dict:
    arr = np.asarray(costs)
    n = len(arr)
    m, s = float(arr.mean()), float(arr.std(ddof=1))
    y = m + lam * s
    t_crit = float(sp_stats.t.ppf((1 + confidence) / 2, df=n - 1))
    mean_ci = t_crit * s / np.sqrt(n)
    B = 5_000
    rng = np.random.RandomState(0)
    boot_y = np.empty(B)
    for b in range(B):
        sample = arr[rng.randint(0, n, size=n)]
        boot_y[b] = sample.mean() + lam * sample.std(ddof=1)
    alpha = (1 - confidence) / 2
    return {
        "mean": m, "std": s, "Y": y, "mean_ci": mean_ci,
        "Y_ci_lo": float(np.percentile(boot_y, 100 * alpha)),
        "Y_ci_hi": float(np.percentile(boot_y, 100 * (1 - alpha))),
    }


def bs_option_price(config: dict) -> float:
    bs = BSValuation(
        strike=config["derivative"]["strike"],
        maturity=config["simulation"]["maturity"],
        rate=config["derivative"].get("rf_rate", 0.0),
        dividend=config["derivative"].get("div_rate", 0.0),
        option_type=config["derivative"].get("option_type", "call"),
    )
    price, _ = bs.price_and_delta(
        spot=config["simulation"]["S0"], t=0.0,
        sigma=config["simulation"]["gbm"]["sigma"],
    )
    return abs(price)


# ── Description strings for titles ───────────────────────────────────

def derivative_desc(cfg: dict) -> str:
    """e.g. 'Short European Call, K=100, T=1.0y' """
    sign = "Short" if cfg["hedging_env"]["position_sign"] < 0 else "Long"
    otype = cfg["derivative"].get("option_type", "call").capitalize()
    K = cfg["derivative"]["strike"]
    T = cfg["simulation"]["maturity"]
    if T < 1:
        t_str = f"T={T*12:.0f}m"
    else:
        t_str = f"T={T:.1f}y"
    return f"{sign} European {otype}, K={K:.0f}, {t_str}"


def market_desc(cfg: dict, process_name: str = "GBM") -> str:
    """e.g. 'GBM(μ=5%, σ=20%), S₀=100' """
    S0 = cfg["simulation"]["S0"]
    if process_name == "GBM":
        mu = cfg["simulation"]["gbm"].get("mu", 0.0)
        sig = cfg["simulation"]["gbm"]["sigma"]
        return f"GBM(μ={mu:.0%}, σ={sig:.0%}), S₀={S0:.0f}"
    elif process_name == "SABR":
        p = cfg["simulation"]["sabr"]
        return (f"SABR(β=1, σ₀={p['sigma0']:.0%}, ν={p['nu']:.1f}, "
                f"ρ={p['rho']:.1f}), S₀={S0:.0f}")
    elif process_name == "SVJ":
        p = cfg["simulation"]["svj"]
        return (f"SVJ(v₀={p['v0']:.2f}, κ={p['kappa']:.1f}, θ={p['theta']:.2f}, "
                f"ξ={p['xi']:.1f}, ρ={p['rho']:.1f}, λ_J={p['jump_intensity']:.1f}), "
                f"S₀={S0:.0f}")
    return f"S₀={S0:.0f}"


def hedging_desc(cfg: dict, agent_name: str = "DeepDPG") -> str:
    """e.g. 'DeepDPG, κ=1%, Δt=daily (n=60), λ=1.5' """
    kappa = cfg["hedging_env"]["transaction_cost"]
    n = cfg["simulation"]["n_steps"]
    T = cfg["simulation"]["maturity"]
    dt_days = T * 252 / n
    if dt_days < 1.5:
        freq = "daily"
    elif dt_days < 2.5:
        freq = "2-day"
    elif dt_days < 4:
        freq = "3-day"
    else:
        freq = "weekly"
    lam = cfg["hedging_agent"].get("risk_lambda", 1.5)
    return f"{agent_name}, κ={kappa:.0%}, rebal={freq} (n={n}), λ={lam}"


def full_title(cfg: dict, experiment: str,
               agent_name: str = "DeepDPG",
               process_name: str = "GBM") -> str:
    """Multi-line title for plots."""
    return (f"{experiment}\n"
            f"{derivative_desc(cfg)} | {market_desc(cfg, process_name)}\n"
            f"{hedging_desc(cfg, agent_name)}")


def subtitle(cfg: dict, process_name: str = "GBM",
             agent_name: str = "DeepDPG") -> str:
    """Single-line subtitle (for smaller plots)."""
    return (f"{derivative_desc(cfg)} | {market_desc(cfg, process_name)} | "
            f"{hedging_desc(cfg, agent_name)}")


# ── Convergence ──────────────────────────────────────────────────────

def check_convergence(train_res, window: int = 500,
                      threshold: float = 0.05) -> dict:
    costs = [sum(ep.costs) for ep in train_res.episodes["train"]]
    n = len(costs)
    if n < 2 * window:
        window = max(n // 4, 10)
    prev = np.mean(costs[-2 * window : -window])
    last = np.mean(costs[-window:])
    rel_change = abs(last - prev) / (abs(prev) + 1e-12)
    return {
        "converged": rel_change < threshold,
        "rel_change": rel_change,
        "prev_window_mean": float(prev),
        "last_window_mean": float(last),
        "window": window, "threshold": threshold,
    }


def log_convergence(conv: dict) -> None:
    status = "✅ CONVERGED" if conv["converged"] else "⚠️  NOT CONVERGED"
    logger.info(
        f"  Convergence: {status}  "
        f"(Δ={conv['rel_change']:.3%}, "
        f"last {conv['window']} mean={conv['last_window_mean']:.4f}, "
        f"prev {conv['window']} mean={conv['prev_window_mean']:.4f})")


def welch_t_test(a: list[float], b: list[float]) -> dict:
    t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


# ── run_experiment ───────────────────────────────────────────────────

def run_experiment(config: dict, process_type, agent_type, benchmark_type,
                   process_name: str = "GBM",
                   agent_name: str = "DeepDPG") -> dict:
    reset_seed()
    runner = Orchestrator(config=config, process_type=process_type,
                          agent_type=agent_type, benchmark_type=benchmark_type)
    train_res      = runner.train()
    eval_agent_res = runner.test()
    eval_bench_res = runner.test_benchmark()

    lam = float(config["hedging_agent"].get("risk_lambda", 1.5))
    agent_costs = total_costs(eval_agent_res, "eval_agent")
    bench_costs = total_costs(eval_bench_res, "eval_benchmark")
    a = compute_metrics(agent_costs, lam)
    b = compute_metrics(bench_costs, lam)
    improvement = 100.0 * (b["Y"] - a["Y"]) / b["Y"] if b["Y"] != 0 else 0.0

    conv = check_convergence(train_res)
    log_convergence(conv)
    ttest = welch_t_test(agent_costs, bench_costs)
    sig = ("***" if ttest["p_value"] < 0.001 else
           "**"  if ttest["p_value"] < 0.01  else
           "*"   if ttest["p_value"] < 0.05  else "ns")
    logger.info(f"  Welch t-test: t={ttest['t_stat']:.3f}, "
                f"p={ttest['p_value']:.2e} ({sig})")

    return {
        "agent": a, "benchmark": b, "improvement": improvement,
        "train_res": train_res,
        "eval_agent_res": eval_agent_res,
        "eval_benchmark_res": eval_bench_res,
        "agent_costs": agent_costs, "bench_costs": bench_costs,
        "convergence": conv, "ttest": ttest,
        "process_name": process_name, "agent_name": agent_name,
    }


def print_paper_table(rows: list[dict], title: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    logger.info(f"\n{'─'*80}")
    logger.info(f"  {title}")
    logger.info(f"{'─'*80}")
    logger.info(df.to_string(index=False, float_format="%.2f"))
    logger.info(f"{'─'*80}\n")
    return df


# ======================================================================
# GRAPHIQUES
# ======================================================================

def _save(fig, name: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  📊 {path}")


def plot_training_curve(train_res, cfg: dict, process_name: str = "GBM",
                        agent_name: str = "DeepDPG",
                        filename: str = "training_curve.png"):
    costs = [sum(ep.costs) for ep in train_res.episodes["train"]]
    w = max(len(costs) // 50, 10)
    smoothed = pd.Series(costs).rolling(w, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(costs, alpha=0.12, color="steelblue", label="Episode cost")
    ax.plot(smoothed, color="steelblue", lw=2, label=f"Rolling avg (w={w})")
    ax.set(xlabel="Episode", ylabel="Total hedging cost")
    ax.set_title(full_title(cfg, "Training curve — Episode hedging cost",
                            agent_name, process_name), fontsize=10)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


def plot_convergence(train_res, conv: dict, cfg: dict,
                     process_name: str = "GBM",
                     agent_name: str = "DeepDPG",
                     filename: str = "convergence_check.png"):
    costs = [sum(ep.costs) for ep in train_res.episodes["train"]]
    n = len(costs)
    w = conv["window"]
    smoothed = pd.Series(costs).rolling(w, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(costs, alpha=0.08, color="steelblue")
    ax.plot(smoothed, color="steelblue", lw=2, label=f"Rolling avg (w={w})")
    ax.axvspan(n-2*w, n-w, alpha=0.15, color="orange", label="Previous window")
    ax.axvspan(n-w, n,     alpha=0.15, color="green",  label="Last window")
    ax.axhline(conv["prev_window_mean"], color="orange", ls="--", lw=1.5)
    ax.axhline(conv["last_window_mean"], color="green",  ls="--", lw=1.5)

    status = "CONVERGED" if conv["converged"] else "NOT CONVERGED"
    ax.set_title(
        f"Convergence check — {status} (Δ={conv['rel_change']:.2%}, "
        f"threshold={conv['threshold']:.0%})\n"
        f"{subtitle(cfg, process_name, agent_name)}", fontsize=10)
    ax.set(xlabel="Episode", ylabel="Total hedging cost")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


def plot_cost_distribution(agent_costs, bench_costs, cfg: dict,
                           process_name: str = "GBM",
                           agent_name: str = "DeepDPG",
                           filename: str = "cost_distribution.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    lo = min(min(agent_costs), min(bench_costs))
    hi = max(max(agent_costs), max(bench_costs))
    bins = np.linspace(lo, hi, 60)
    ax.hist(bench_costs, bins=bins, alpha=0.5, label="Delta hedging (BS)",
            color="salmon", edgecolor="white")
    ax.hist(agent_costs, bins=bins, alpha=0.5, label=f"RL ({agent_name})",
            color="steelblue", edgecolor="white")
    ax.axvline(np.mean(bench_costs), color="salmon", ls="--", lw=2,
               label=f"Delta E[C]={np.mean(bench_costs):.4f}")
    ax.axvline(np.mean(agent_costs), color="steelblue", ls="--", lw=2,
               label=f"RL E[C]={np.mean(agent_costs):.4f}")
    ax.set(xlabel="Total hedging cost (per option)", ylabel="Frequency")
    ax.set_title(full_title(cfg,
        f"Distribution of hedging costs — {agent_name} vs BS Delta",
        agent_name, process_name), fontsize=10)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


def plot_over_under_hedging(eval_agent_res, eval_bench_res, cfg: dict,
                            process_name: str = "GBM",
                            agent_name: str = "DeepDPG",
                            filename: str = "over_under_hedge.png"):
    ag, dl = [], []
    for ea, eb in zip(eval_agent_res.episodes["eval_agent"],
                      eval_bench_res.episodes["eval_benchmark"]):
        ag.extend(ea.actions); dl.extend(eb.actions)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(dl, ag, alpha=0.02, s=3, color="steelblue")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=1, label="H_RL = Δ_BS (no adjustment)")
    ax.fill_between(lims, lims, [lims[1], lims[1]], alpha=0.04,
                    color="green", label="Over-hedged zone")
    ax.fill_between(lims, [lims[0], lims[0]], lims, alpha=0.04,
                    color="red", label="Under-hedged zone")
    ax.set(xlabel="BS Delta hedge position (H_Δ)",
           ylabel=f"RL agent position (H_RL, {agent_name})")
    ax.set_title(
        f"Over/Under-hedging — {agent_name} vs BS Delta (cf. Exhibit 5)\n"
        f"{subtitle(cfg, process_name, agent_name)}", fontsize=10)
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


def plot_rebalancing_comparison(df: pd.DataFrame, cfg: dict,
                                filename: str = "rebalancing_comparison.png"):
    lam = cfg["hedging_agent"].get("risk_lambda", 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(df)); w = 0.35

    # Left: Y(0)
    ax = axes[0]
    ax.bar(x-w/2, df["Delta Mean (%)"]+lam*df["Delta Std (%)"], w,
           label="Delta Y(0)", color="salmon", edgecolor="white")
    ax.bar(x+w/2, df["RL Mean (%)"]+lam*df["RL Std (%)"], w,
           label="RL Y(0)", color="steelblue", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["Rebal Freq"])
    ax.set(ylabel="Y(0) = E[C]+λσ(C) as % of V₀",
           title=f"Y(0) by rebalancing frequency\n{derivative_desc(cfg)} | "
                 f"{market_desc(cfg, 'GBM')} | κ={cfg['hedging_env']['transaction_cost']:.0%}, λ={lam}")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # Right: improvement %
    ax = axes[1]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["Y improv (%)"]]
    ax.bar(x, df["Y improv (%)"], 0.5, color=colors, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["Rebal Freq"])
    ax.set(ylabel="Y(0) improvement (%)",
           title="RL improvement vs Delta hedging\n"
                 "(positive = RL outperforms)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, filename)


def plot_process_comparison(df: pd.DataFrame, cfg: dict,
                            filename: str = "process_comparison.png"):
    lam = cfg["hedging_agent"].get("risk_lambda", 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df)); w = 0.35

    ax = axes[0]
    ax.bar(x-w/2, df["Delta Mean (%)"]+lam*df["Delta Std (%)"], w,
           label="Delta Y(0)", color="salmon", edgecolor="white")
    ax.bar(x+w/2, df["RL Mean (%)"]+lam*df["RL Std (%)"], w,
           label="RL Y(0)", color="steelblue", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["Process"])
    ax.set(ylabel="Y(0) as % of V₀",
           title=f"Y(0) across asset-price processes\n{derivative_desc(cfg)} | "
                 f"DeepDPG, κ={cfg['hedging_env']['transaction_cost']:.0%}, "
                 f"Hybrid BS valuation")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["Y improv (%)"]]
    ax.bar(x, df["Y improv (%)"], 0.5, color=colors, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["Process"])
    ax.set(ylabel="Y(0) improvement (%)",
           title="RL improvement vs Delta\n(hybrid BS valuation for P&L)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, filename)


def plot_maturity_comparison(df: pd.DataFrame, cfg: dict,
                             filename: str = "maturity_comparison.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(df)); w = 0.35

    for ax, cols, ylabel in zip(
        axes[:2],
        [("Delta Mean (%)", "RL Mean (%)"), ("Delta Std (%)", "RL Std (%)")],
        ["Mean cost (% of V₀)", "Std cost (% of V₀)"],
    ):
        ax.bar(x-w/2, df[cols[0]], w, label="Delta", color="salmon", edgecolor="white")
        ax.bar(x+w/2, df[cols[1]], w, label="RL",    color="steelblue", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(df["Maturity"])
        ax.set(ylabel=ylabel)
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_title(f"Mean hedging cost by maturity\nGBM(σ=20%), DeepDPG, "
                      f"κ={cfg['hedging_env']['transaction_cost']:.0%}, daily rebal")
    axes[1].set_title(f"Std of hedging cost by maturity\n(same parameters)")

    ax = axes[2]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["Y improv (%)"]]
    ax.bar(x, df["Y improv (%)"], 0.5, color=colors, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["Maturity"])
    ax.set(ylabel="Y(0) improvement (%)",
           title="RL improvement vs Delta\nby option maturity")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, filename)


def plot_agent_comparison(df: pd.DataFrame, cfg: dict,
                          filename: str = "agent_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df))
    colors = ["#e74c3c", "#f39c12", "#2980b9"][:len(df)]

    ax = axes[0]
    ax.bar(x, df["Y(0) (%)"], 0.45, color=colors, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["Agent"])
    ax.set(ylabel="Y(0) as % of V₀",
           title=f"Agent comparison — Y(0) = E[C]+λσ(C)\n"
                 f"{derivative_desc(cfg)} | GBM(σ=20%) | "
                 f"κ={cfg['hedging_env']['transaction_cost']:.0%}\n"
                 f"All agents: dual Q-function, Y=E[C]+{cfg['hedging_agent'].get('risk_lambda',1.5)}·σ(C)")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    improv = df["Y improv vs Δ (%)"]
    bar_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in improv]
    ax.bar(x, improv, 0.45, color=bar_colors, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["Agent"])
    ax.set(ylabel="Y(0) improvement vs Delta (%)",
           title="Improvement over BS Delta hedging\n"
                 "DQN/DoubleDQN: discrete actions (21-grid)\n"
                 "DeepDPG: continuous actions (actor-critic)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, filename)


# ======================================================================
# EXPERIMENTS
# ======================================================================

def experiment_baseline(config: dict) -> dict:
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1 — Baseline: GBM + DeepDPG")
    logger.info("=" * 60)
    res = run_experiment(config, ProcessType.GBM, AgentType.DeepDPG,
                         BenchmarkType.BsDelta, "GBM", "DeepDPG")
    V0 = bs_option_price(config)
    print_paper_table([{
        "Method": "Delta Hedging",
        "Mean (%)":   res["benchmark"]["mean"]/V0*100,
        "±CI (%)":    res["benchmark"]["mean_ci"]/V0*100,
        "Std (%)":    res["benchmark"]["std"]/V0*100,
        "Y(0) (%)":   res["benchmark"]["Y"]/V0*100,
        "Y 95%CI":    f"[{res['benchmark']['Y_ci_lo']/V0*100:.1f}, "
                      f"{res['benchmark']['Y_ci_hi']/V0*100:.1f}]",
    }, {
        "Method": "RL (DeepDPG)",
        "Mean (%)":   res["agent"]["mean"]/V0*100,
        "±CI (%)":    res["agent"]["mean_ci"]/V0*100,
        "Std (%)":    res["agent"]["std"]/V0*100,
        "Y(0) (%)":   res["agent"]["Y"]/V0*100,
        "Y 95%CI":    f"[{res['agent']['Y_ci_lo']/V0*100:.1f}, "
                      f"{res['agent']['Y_ci_hi']/V0*100:.1f}]",
    }], title=(f"Baseline — {derivative_desc(config)} | "
               f"{market_desc(config, 'GBM')} | "
               f"κ={config['hedging_env']['transaction_cost']:.0%}"))
    logger.info(f"  Y(0) improvement: {res['improvement']:.2f}%")
    return res


def experiment_rebalancing_frequencies(base_config: dict) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2 — Rebalancing frequencies (cf. Exhibits 3-4)")
    logger.info("=" * 60)
    maturity = base_config["simulation"]["maturity"]
    V0 = bs_option_price(base_config)
    freq_map = {
        "weekly": int(round(52*maturity)),
        "3-day":  int(round(84*maturity)),
        "2-day":  int(round(126*maturity)),
        "daily":  int(round(252*maturity)),
    }
    rows = []
    for label, n_steps in freq_map.items():
        logger.info(f"  ► {label} (n_steps={n_steps})")
        cfg = copy.deepcopy(base_config)
        cfg["simulation"]["n_steps"] = max(n_steps, 5)
        res = run_experiment(cfg, ProcessType.GBM, AgentType.DeepDPG,
                             BenchmarkType.BsDelta, "GBM", "DeepDPG")
        rows.append({
            "Rebal Freq":     label,
            "Delta Mean (%)": res["benchmark"]["mean"]/V0*100,
            "Delta Std (%)":  res["benchmark"]["std"]/V0*100,
            "RL Mean (%)":    res["agent"]["mean"]/V0*100,
            "RL Std (%)":     res["agent"]["std"]/V0*100,
            "Y improv (%)":   res["improvement"],
            "p-value":        res["ttest"]["p_value"],
            "converged":      res["convergence"]["converged"],
        })
    return print_paper_table(rows,
        title=f"Rebalancing frequency — {derivative_desc(base_config)} | "
              f"{market_desc(base_config, 'GBM')}")


def experiment_stochastic_vol(base_config: dict) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3 — SABR & SVJ (cf. Exhibits 6-7 + extension)")
    logger.info("=" * 60)
    V0 = bs_option_price(base_config)
    rows = []
    for proc_name, proc_type in [("GBM", ProcessType.GBM),
                                  ("SABR", ProcessType.SABR),
                                  ("SVJ", ProcessType.SVJ)]:
        logger.info(f"  ► {proc_name}: {market_desc(base_config, proc_name)}")
        cfg = copy.deepcopy(base_config)
        res = run_experiment(cfg, proc_type, AgentType.DeepDPG,
                             BenchmarkType.BsDelta, proc_name, "DeepDPG")
        rows.append({
            "Process":        proc_name,
            "Delta Mean (%)": res["benchmark"]["mean"]/V0*100,
            "Delta Std (%)":  res["benchmark"]["std"]/V0*100,
            "RL Mean (%)":    res["agent"]["mean"]/V0*100,
            "RL Std (%)":     res["agent"]["std"]/V0*100,
            "Y improv (%)":   res["improvement"],
            "p-value":        res["ttest"]["p_value"],
            "converged":      res["convergence"]["converged"],
        })
    return print_paper_table(rows,
        title=f"Stochastic vol — {derivative_desc(base_config)} | "
              f"Hybrid BS valuation (σ_BS={base_config['simulation']['gbm']['sigma']:.0%})")


def experiment_maturities(base_config: dict) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4 — Varying maturities (cf. Exhibits 3 vs 4)")
    logger.info("=" * 60)
    maturities = {"1 month": 1/12, "3 months": 3/12,
                  "6 months": 6/12, "1 year": 1.0}
    rows = []
    for label, T in maturities.items():
        n_steps = max(int(round(252*T)), 5)
        logger.info(f"  ► {label} (T={T:.4f}, n={n_steps})")
        cfg = copy.deepcopy(base_config)
        cfg["simulation"]["maturity"] = T
        cfg["simulation"]["n_steps"] = n_steps
        V0 = bs_option_price(cfg)
        res = run_experiment(cfg, ProcessType.GBM, AgentType.DeepDPG,
                             BenchmarkType.BsDelta, "GBM", "DeepDPG")
        rows.append({
            "Maturity":       label,
            "Delta Mean (%)": res["benchmark"]["mean"]/V0*100,
            "Delta Std (%)":  res["benchmark"]["std"]/V0*100,
            "RL Mean (%)":    res["agent"]["mean"]/V0*100,
            "RL Std (%)":     res["agent"]["std"]/V0*100,
            "Y improv (%)":   res["improvement"],
            "p-value":        res["ttest"]["p_value"],
            "converged":      res["convergence"]["converged"],
        })
    return print_paper_table(rows,
        title=f"Maturity comparison — Short Call ATM, "
              f"GBM(σ=20%), DeepDPG, κ={base_config['hedging_env']['transaction_cost']:.0%}")


def experiment_compare_agents(base_config: dict) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5 — DQN vs DoubleDQN vs DDPG")
    logger.info("=" * 60)
    V0 = bs_option_price(base_config)
    agents = [
        ("DQN",       AgentType.DQN),
        ("DoubleDQN", AgentType.DoubleDQN),
        ("DeepDPG",   AgentType.DeepDPG),
    ]
    rows = []
    for agent_name, agent_type in agents:
        logger.info(f"  ► {agent_name}")
        cfg = copy.deepcopy(base_config)
        res = run_experiment(cfg, ProcessType.GBM, agent_type,
                             BenchmarkType.BsDelta, "GBM", agent_name)
        rows.append({
            "Agent":            agent_name,
            "Mean (%)":         res["agent"]["mean"]/V0*100,
            "±CI (%)":          res["agent"]["mean_ci"]/V0*100,
            "Std (%)":          res["agent"]["std"]/V0*100,
            "Y(0) (%)":         res["agent"]["Y"]/V0*100,
            "Y 95%CI":          f"[{res['agent']['Y_ci_lo']/V0*100:.1f}, "
                                f"{res['agent']['Y_ci_hi']/V0*100:.1f}]",
            "Y improv vs Δ (%)": res["improvement"],
            "converged":         res["convergence"]["converged"],
        })
    return print_paper_table(rows,
        title=f"Agent comparison — {derivative_desc(base_config)} | "
              f"GBM(σ=20%) | All: dual Q-function, Y=E[C]+λσ(C)")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   Deep Hedging — Full Experiment Suite                  ║")
    logger.info("║   Cao, Chen, Hull & Poulos (2021)                      ║")
    logger.info("╚══════════════════════════════════════════════════════════╝\n")

    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError("config.json introuvable.")
    config = json_to_dict(str(config_path))

    # ── 1. Baseline ──────────────────────────────────────────────────
    res = experiment_baseline(config)
    logger.info("\nGenerating baseline plots...")
    plot_training_curve(res["train_res"], config, "GBM", "DeepDPG")
    plot_convergence(res["train_res"], res["convergence"], config, "GBM", "DeepDPG")
    plot_cost_distribution(res["agent_costs"], res["bench_costs"], config, "GBM", "DeepDPG")
    plot_over_under_hedging(res["eval_agent_res"], res["eval_benchmark_res"],
                            config, "GBM", "DeepDPG")

    # ── 2. Fréquences de rebalancement ───────────────────────────────
    df_freq = experiment_rebalancing_frequencies(config)
    plot_rebalancing_comparison(df_freq, config)

    # ── 3. Processus stochastiques : SABR & SVJ ─────────────────────
    df_proc = experiment_stochastic_vol(config)
    plot_process_comparison(df_proc, config)

    # ── 4. Variation de maturité ─────────────────────────────────────
    df_mat = experiment_maturities(config)
    plot_maturity_comparison(df_mat, config)

    # ── 5. Comparaison DQN vs DoubleDQN vs DDPG ─────────────────────
    df_agents = experiment_compare_agents(config)
    plot_agent_comparison(df_agents, config)

    logger.info("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
