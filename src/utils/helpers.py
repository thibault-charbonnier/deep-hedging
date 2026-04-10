import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats

from src.orchestrator import Orchestrator
from src.utils.enums import BenchmarkType
from src.valuation import BSValuation

SEED = 42
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)


def json_to_dict(json_file: str) -> dict:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def reset_seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def total_costs(res, split):
    return [sum(e.costs) for e in res.episodes[split]]


def metrics(costs, lam=1.5):
    a = np.asarray(costs)
    n = len(a)
    m, s = float(a.mean()), float(a.std(ddof=1))
    t_c = float(sp_stats.t.ppf(0.975, df=n - 1))
    rng = np.random.RandomState(0)
    by = np.array([
        a[rng.randint(0, n, n)].mean() + lam * a[rng.randint(0, n, n)].std(ddof=1)
        for _ in range(5000)
    ])
    return {
        "mean": m,
        "std": s,
        "Y": m + lam * s,
        "mean_ci": t_c * s / np.sqrt(n),
        "Y_lo": float(np.percentile(by, 2.5)),
        "Y_hi": float(np.percentile(by, 97.5)),
    }


def bs_price(cfg):
    bs = BSValuation(
        cfg["derivative"]["strike"],
        cfg["simulation"]["maturity"],
        cfg["derivative"].get("rf_rate", 0),
        cfg["derivative"].get("div_rate", 0),
        cfg["derivative"].get("option_type", "call"),
    )
    p, _ = bs.price_and_delta(cfg["simulation"]["S0"], 0, cfg["simulation"]["gbm"]["sigma"])
    return abs(p)


def convergence(train_res, w=500, thr=0.05):
    c = [sum(e.costs) for e in train_res.episodes["train"]]
    n = len(c)
    if n < 2 * w:
        w = max(n // 4, 10)
    p, l = np.mean(c[-2 * w:-w]), np.mean(c[-w:])
    rc = abs(l - p) / (abs(p) + 1e-12)
    return {"ok": rc < thr, "rc": rc, "prev": float(p), "last": float(l), "w": w}


def welch(a, b):
    t, p = sp_stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p)}


def run(cfg, proc, agent, bench):
    reset_seed()
    o = Orchestrator(cfg, proc, agent, bench)
    tr = o.train()
    ea = o.test()
    eb = o.test_benchmark()
    lam = float(cfg["hedging_agent"].get("risk_lambda", 1.5))
    ac, bc = total_costs(ea, "eval_agent"), total_costs(eb, "eval_benchmark")
    am, bm = metrics(ac, lam), metrics(bc, lam)
    imp = 100 * (bm["Y"] - am["Y"]) / bm["Y"] if bm["Y"] != 0 else 0
    cv = convergence(tr)
    tt = welch(ac, bc)
    s = "✅" if cv["ok"] else "⚠️"
    sig = "***" if tt["p"] < .001 else "**" if tt["p"] < .01 else "*" if tt["p"] < .05 else "ns"
    logger.info(f"  Conv: {s} Δ={cv['rc']:.3%} | Welch p={tt['p']:.2e}({sig})")
    return {
        "agent": am,
        "bench": bm,
        "imp": imp,
        "tr": tr,
        "ea": ea,
        "eb": eb,
        "ac": ac,
        "bc": bc,
        "cv": cv,
        "tt": tt,
        "runner": o,
    }


def run_multi_bench(cfg, proc, agent):
    """Run agent once, then evaluate against multiple benchmarks on same paths."""
    reset_seed()
    o = Orchestrator(cfg, proc, agent, BenchmarkType.BsDelta)
    tr = o.train()
    ea = o.test()
    lam = float(cfg["hedging_agent"].get("risk_lambda", 1.5))
    ac = total_costs(ea, "eval_agent")
    am = metrics(ac, lam)
    cv = convergence(tr)
    s = "✅" if cv["ok"] else "⚠️"
    logger.info(f"  Conv: {s} Δ={cv['rc']:.3%}")

    # Evaluate each benchmark on same eval paths
    benchmarks = {}
    for bname, btype in [
        ("Practitioner Δ", BenchmarkType.SABRPractitionerDelta),
        ("Bartlett Δ", BenchmarkType.BartlettDelta),
    ]:
        bench_inst = btype.value(cfg)
        eb = o.test_benchmark(benchmark_override=bench_inst)
        bc = total_costs(eb, "eval_benchmark")
        bm = metrics(bc, lam)
        imp = 100 * (bm["Y"] - am["Y"]) / bm["Y"] if bm["Y"] != 0 else 0
        tt = welch(ac, bc)
        benchmarks[bname] = {"metrics": bm, "imp": imp, "tt": tt, "costs": bc}

    return {"agent": am, "benchmarks": benchmarks, "tr": tr, "ea": ea, "ac": ac, "cv": cv}


def ptable(rows, title):
    df = pd.DataFrame(rows)
    logger.info(f"\n{'─' * 85}\n  {title}\n{'─' * 85}")
    logger.info(df.to_string(index=False, float_format="%.1f"))
    logger.info(f"{'─' * 85}\n")
    return df


# Frequency map from maturity to number of rebalancing steps.
def freq_map(T):
    return {
        "weekly": max(int(round(52 * T)), 3),
        "3-day": max(int(round(84 * T)), 5),
        "2-day": max(int(round(126 * T)), 5),
        "daily": max(int(round(252 * T)), 5),
    }


def _sv(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  📊 {FIG / name}")


def plot_training(tr, title, fn="training_curve.png"):
    c = [sum(e.costs) for e in tr.episodes["train"]]
    w = max(len(c) // 50, 10)
    sm = pd.Series(c).rolling(w, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(c, alpha=.12, color="steelblue")
    ax.plot(sm, color="steelblue", lw=2, label=f"Rolling avg (w={w})")
    ax.set(xlabel="Episode", ylabel="Hedging cost", title=title)
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    _sv(fig, fn)


def plot_costs(ac, bc, title, fn="cost_dist.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(min(min(ac), min(bc)), max(max(ac), max(bc)), 60)
    ax.hist(bc, bins=bins, alpha=.5, label="Delta", color="salmon", edgecolor="white")
    ax.hist(ac, bins=bins, alpha=.5, label="RL", color="steelblue", edgecolor="white")
    ax.axvline(np.mean(bc), color="salmon", ls="--", lw=2)
    ax.axvline(np.mean(ac), color="steelblue", ls="--", lw=2)
    ax.set(xlabel="Total cost", ylabel="Count", title=title)
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    _sv(fig, fn)


def plot_scatter(ea, eb, title, fn="scatter.png"):
    ag, dl = [], []
    for a, b in zip(ea.episodes["eval_agent"], eb.episodes["eval_benchmark"]):
        ag.extend(a.actions)
        dl.extend(b.actions)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(dl, ag, alpha=.02, s=3, color="steelblue")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=1, label="H_RL = Δ_BS")
    ax.fill_between(lims, lims, [lims[1], lims[1]], alpha=.04, color="green", label="Over-hedged")
    ax.fill_between(lims, [lims[0], lims[0]], lims, alpha=.04, color="red", label="Under-hedged")
    ax.set(xlabel="BS Delta position", ylabel="RL position", title=title)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    _sv(fig, fn)


def plot_freq_bars(df, title, fn="freq_bars.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df))
    w = .35
    lam = 1.5
    ax = axes[0]
    ax.bar(x - w / 2, df["Δ Mean%"] + lam * df["Δ Std%"], w, label="Delta Y(0)", color="salmon", edgecolor="white")
    ax.bar(x + w / 2, df["RL Mean%"] + lam * df["RL Std%"], w, label="RL Y(0)", color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Rebal"])
    ax.set(ylabel="Y(0) % of V₀")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=.3, axis="y")
    ax = axes[1]
    cols = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["Y improv%"]]
    ax.bar(x, df["Y improv%"], .5, color=cols, edgecolor="white")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Rebal"])
    ax.set(ylabel="Y(0) improvement %", title="RL improvement vs Delta")
    ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout()
    _sv(fig, fn)


def plot_agents(df, title, fn="agents.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    colors = ["#e74c3c", "#f39c12", "#2980b9"][:len(df)]
    ax.bar(x, df["Y(0)%"], .45, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Agent"])
    ax.set(ylabel="Y(0) % of V₀", title=title)
    ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout()
    _sv(fig, fn)


def plot_sabr_benchmarks(df, title, fn="sabr_benchmarks.png"):
    """Compare RL, practitioner delta and Bartlett delta across rebalancing frequencies."""
    x = np.arange(len(df))
    w = 0.25
    lam = 1.5
    rl_y = df["RL Mean%"] + lam * df["RL Std%"]
    pr_y = df["Practitioner Δ Mean%"] + lam * df["Practitioner Δ Std%"]
    ba_y = df["Bartlett Δ Mean%"] + lam * df["Bartlett Δ Std%"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, pr_y, w, label="Practitioner Δ Y(0)", color="salmon", edgecolor="white")
    ax.bar(x, ba_y, w, label="Bartlett Δ Y(0)", color="#f39c12", edgecolor="white")
    ax.bar(x + w, rl_y, w, label="RL Y(0)", color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Rebal"])
    ax.set(ylabel="Y(0) % of V₀", title=title)
    ax.legend()
    ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout()
    _sv(fig, fn)


def plot_process_comparison(df, title, fn="process_comparison.png"):
    """Compare RL vs Delta across processes and maturities."""
    lam = 1.5
    labels = [f"{t} — {p}" for t, p in zip(df["T"], df["Process"])]
    x = np.arange(len(df))
    w = 0.35
    delta_y = df["Δ Mean%"] + lam * df["Δ Std%"]
    rl_y = df["RL Mean%"] + lam * df["RL Std%"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, delta_y, w, label="Delta Y(0)", color="salmon", edgecolor="white")
    ax.bar(x + w / 2, rl_y, w, label="RL Y(0)", color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set(ylabel="Y(0) % of V₀", title=title)
    ax.legend()
    ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout()
    _sv(fig, fn)


def plot_maturity_comparison(df, title, fn="maturity_comparison.png"):
    """Mean/std comparison across maturities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df))
    w = 0.35
    for ax, cols, sub in zip(
        axes,
        [("Δ Mean%", "RL Mean%"), ("Δ Std%", "RL Std%")],
        ["Mean cost (% of V₀)", "Std cost (% of V₀)"],
    ):
        ax.bar(x - w / 2, df[cols[0]], w, label="Delta", color="salmon", edgecolor="white")
        ax.bar(x + w / 2, df[cols[1]], w, label="RL", color="steelblue", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Maturity"])
        ax.set(ylabel="% of V₀", title=f"{title}\n{sub}")
        ax.legend()
        ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout()
    _sv(fig, fn)
