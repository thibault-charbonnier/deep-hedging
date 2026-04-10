"""
Deep Hedging of Derivatives Using Reinforcement Learning
=========================================================
Reproduction and extension of Cao, Chen, Hull & Poulos (2021).
ML for Finance — M2MO / ENSAE / Mastère FGR, 2025-2026.

Structure:
  PART A — Faithful replication of the paper's results
    A.1  GBM,  T=1 month, varying rebalancing frequency    (Exhibit 3)
    A.2  GBM,  T=3 months, varying rebalancing frequency   (Exhibit 4)
    A.3  SABR, T=1 month, 3 benchmarks, varying frequency  (Exhibit 6)
    A.4  SABR, T=3 months, 3 benchmarks, varying frequency (Exhibit 7)

  PART B — Extensions beyond the paper
    B.1  SVJ stochastic volatility with jumps
    B.2  Longer maturities (6 months, 1 year)
    B.3  Agent comparison: DQN vs DoubleDQN vs DeepDPG
"""
import logging, copy
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as sp_stats
from rich.logging import RichHandler

from src.orchestrator import Orchestrator
from src.valuation import BSValuation
from src.utils.enums import AgentType, BenchmarkType, ProcessType
from src.utils import json_to_dict

SEED = 42
def reset_seed():
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
reset_seed()

logging.basicConfig(level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)], force=True)
logger = logging.getLogger(__name__)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

# ======================================================================
# HELPERS
# ======================================================================
def total_costs(res, split): return [sum(e.costs) for e in res.episodes[split]]

def metrics(costs, lam=1.5):
    a = np.asarray(costs); n = len(a)
    m, s = float(a.mean()), float(a.std(ddof=1))
    t_c = float(sp_stats.t.ppf(0.975, df=n-1))
    rng = np.random.RandomState(0)
    by = np.array([a[rng.randint(0,n,n)].mean()+lam*a[rng.randint(0,n,n)].std(ddof=1) for _ in range(5000)])
    return {"mean":m,"std":s,"Y":m+lam*s,"mean_ci":t_c*s/np.sqrt(n),
            "Y_lo":float(np.percentile(by,2.5)),"Y_hi":float(np.percentile(by,97.5))}

def bs_price(cfg):
    bs = BSValuation(cfg["derivative"]["strike"], cfg["simulation"]["maturity"],
        cfg["derivative"].get("rf_rate",0), cfg["derivative"].get("div_rate",0),
        cfg["derivative"].get("option_type","call"))
    p,_ = bs.price_and_delta(cfg["simulation"]["S0"], 0, cfg["simulation"]["gbm"]["sigma"])
    return abs(p)

def convergence(train_res, w=500, thr=0.05):
    c=[sum(e.costs) for e in train_res.episodes["train"]]; n=len(c)
    if n<2*w: w=max(n//4,10)
    p,l = np.mean(c[-2*w:-w]), np.mean(c[-w:])
    rc = abs(l-p)/(abs(p)+1e-12)
    return {"ok":rc<thr,"rc":rc,"prev":float(p),"last":float(l),"w":w}

def welch(a,b):
    t,p=sp_stats.ttest_ind(a,b,equal_var=False)
    return {"t":float(t),"p":float(p)}

def run(cfg, proc, agent, bench):
    reset_seed()
    o=Orchestrator(cfg, proc, agent, bench)
    tr=o.train(); ea=o.test(); eb=o.test_benchmark()
    lam=float(cfg["hedging_agent"].get("risk_lambda",1.5))
    ac,bc=total_costs(ea,"eval_agent"),total_costs(eb,"eval_benchmark")
    am,bm=metrics(ac,lam),metrics(bc,lam)
    imp=100*(bm["Y"]-am["Y"])/bm["Y"] if bm["Y"]!=0 else 0
    cv=convergence(tr); tt=welch(ac,bc)
    s="✅" if cv["ok"] else "⚠️"
    sig="***" if tt["p"]<.001 else "**" if tt["p"]<.01 else "*" if tt["p"]<.05 else "ns"
    logger.info(f"  Conv: {s} Δ={cv['rc']:.3%} | Welch p={tt['p']:.2e}({sig})")
    return {"agent":am,"bench":bm,"imp":imp,"tr":tr,"ea":ea,"eb":eb,
            "ac":ac,"bc":bc,"cv":cv,"tt":tt,"runner":o}

def run_multi_bench(cfg, proc, agent):
    """Run agent once, then evaluate against multiple benchmarks on same paths."""
    reset_seed()
    o = Orchestrator(cfg, proc, agent, BenchmarkType.BsDelta)
    tr = o.train(); ea = o.test()
    lam = float(cfg["hedging_agent"].get("risk_lambda", 1.5))
    ac = total_costs(ea, "eval_agent")
    am = metrics(ac, lam)
    cv = convergence(tr)
    s = "✅" if cv["ok"] else "⚠️"
    logger.info(f"  Conv: {s} Δ={cv['rc']:.3%}")

    # Evaluate each benchmark on same eval paths
    benchmarks = {}
    for bname, btype in [("Practitioner Δ", BenchmarkType.SABRPractitionerDelta),
                         ("Bartlett Δ",     BenchmarkType.BartlettDelta)]:
        bench_inst = btype.value(cfg)
        eb = o.test_benchmark(benchmark_override=bench_inst)
        bc = total_costs(eb, "eval_benchmark")
        bm = metrics(bc, lam)
        imp = 100*(bm["Y"]-am["Y"])/bm["Y"] if bm["Y"]!=0 else 0
        tt = welch(ac, bc)
        benchmarks[bname] = {"metrics": bm, "imp": imp, "tt": tt, "costs": bc}

    return {"agent": am, "benchmarks": benchmarks, "tr": tr, "ea": ea,
            "ac": ac, "cv": cv}

def ptable(rows, title):
    df=pd.DataFrame(rows)
    logger.info(f"\n{'─'*85}\n  {title}\n{'─'*85}")
    logger.info(df.to_string(index=False, float_format="%.1f"))
    logger.info(f"{'─'*85}\n"); return df

# ── Frequency map ────────────────────────────────────────────────────
def freq_map(T):
    return {"weekly":max(int(round(52*T)),3), "3-day":max(int(round(84*T)),5),
            "2-day":max(int(round(126*T)),5), "daily":max(int(round(252*T)),5)}

# ======================================================================
# PART A — PAPER REPLICATION
# ======================================================================

def part_A1_A2(base_cfg, T, exhibit_name):
    """GBM, varying rebalancing freq — Exhibits 3 & 4."""
    logger.info("="*65)
    logger.info(f"PART A — {exhibit_name}: GBM, T={T*12:.0f}m, Short ATM Call")
    logger.info("="*65)
    V0 = None; rows = []
    for label, ns in freq_map(T).items():
        logger.info(f"  ► {label} (n={ns})")
        cfg = copy.deepcopy(base_cfg)
        cfg["simulation"]["maturity"] = T
        cfg["simulation"]["n_steps"] = ns
        if V0 is None: V0 = bs_price(cfg)
        r = run(cfg, ProcessType.GBM, AgentType.DeepDPG, BenchmarkType.BsDelta)
        rows.append({"Rebal":label,
            "Δ Mean%":r["bench"]["mean"]/V0*100, "Δ Std%":r["bench"]["std"]/V0*100,
            "RL Mean%":r["agent"]["mean"]/V0*100, "RL Std%":r["agent"]["std"]/V0*100,
            "Y improv%":r["imp"], "p":r["tt"]["p"], "conv":r["cv"]["ok"]})
    return ptable(rows, f"{exhibit_name} — GBM, T={T*12:.0f}m, κ=1%, μ=5%, σ=20%")


def part_A3_A4(base_cfg, T, exhibit_name):
    """SABR, varying rebalancing freq, 3 benchmarks — Exhibits 6 & 7."""
    logger.info("="*65)
    logger.info(f"PART A — {exhibit_name}: SABR β=1, T={T*12:.0f}m, ρ=-0.4, ν=0.6")
    logger.info("="*65)
    V0 = None; rows = []
    for label, ns in freq_map(T).items():
        logger.info(f"  ► {label} (n={ns})")
        cfg = copy.deepcopy(base_cfg)
        cfg["simulation"]["maturity"] = T
        cfg["simulation"]["n_steps"] = ns
        if V0 is None: V0 = bs_price(cfg)

        res = run_multi_bench(cfg, ProcessType.SABR, AgentType.DeepDPG)
        am = res["agent"]

        row = {"Rebal": label,
               "RL Mean%": am["mean"]/V0*100,
               "RL Std%":  am["std"]/V0*100}

        for bname, bdata in res["benchmarks"].items():
            bm = bdata["metrics"]
            row[f"{bname} Mean%"] = bm["mean"]/V0*100
            row[f"{bname} Std%"]  = bm["std"]/V0*100
            row[f"Y improv vs {bname}%"] = bdata["imp"]

        rows.append(row)

    return ptable(rows, f"{exhibit_name} — SABR(β=1,ρ=-0.4,ν=0.6), T={T*12:.0f}m, κ=1%")


# ======================================================================
# PART B — EXTENSIONS
# ======================================================================

def part_B1_svj(base_cfg):
    """Extension: SVJ process."""
    logger.info("="*65)
    logger.info("PART B.1 — Extension: SVJ (Heston + jumps)")
    logger.info("="*65)
    rows = []
    for T_label, T in [("1m", 1/12), ("3m", 3/12)]:
        cfg = copy.deepcopy(base_cfg)
        cfg["simulation"]["maturity"] = T
        cfg["simulation"]["n_steps"] = max(int(round(252*T)), 5)
        V0 = bs_price(cfg)
        for proc_name, proc in [("GBM",ProcessType.GBM), ("SVJ",ProcessType.SVJ)]:
            logger.info(f"  ► {proc_name}, T={T_label}")
            r = run(cfg, proc, AgentType.DeepDPG, BenchmarkType.BsDelta)
            rows.append({"T":T_label, "Process":proc_name,
                "Δ Mean%":r["bench"]["mean"]/V0*100,"Δ Std%":r["bench"]["std"]/V0*100,
                "RL Mean%":r["agent"]["mean"]/V0*100,"RL Std%":r["agent"]["std"]/V0*100,
                "Y improv%":r["imp"], "conv":r["cv"]["ok"]})
    return ptable(rows, "Extension B.1 — SVJ vs GBM, daily rebal, κ=1%")


def part_B2_maturities(base_cfg):
    """Extension: longer maturities."""
    logger.info("="*65)
    logger.info("PART B.2 — Extension: longer maturities (6m, 1y)")
    logger.info("="*65)
    rows = []
    for label, T in [("1m",1/12),("3m",3/12),("6m",6/12),("1y",1.0)]:
        ns = max(int(round(252*T)),5)
        logger.info(f"  ► {label} (n={ns})")
        cfg = copy.deepcopy(base_cfg)
        cfg["simulation"]["maturity"] = T; cfg["simulation"]["n_steps"] = ns
        V0 = bs_price(cfg)
        r = run(cfg, ProcessType.GBM, AgentType.DeepDPG, BenchmarkType.BsDelta)
        rows.append({"Maturity":label,
            "Δ Mean%":r["bench"]["mean"]/V0*100,"Δ Std%":r["bench"]["std"]/V0*100,
            "RL Mean%":r["agent"]["mean"]/V0*100,"RL Std%":r["agent"]["std"]/V0*100,
            "Y improv%":r["imp"], "conv":r["cv"]["ok"]})
    return ptable(rows, "Extension B.2 — Maturity comparison, GBM daily, κ=1%")


def part_B3_agents(base_cfg):
    """Extension: DQN vs DoubleDQN vs DeepDPG."""
    logger.info("="*65)
    logger.info("PART B.3 — Extension: DQN vs DoubleDQN vs DeepDPG")
    logger.info("="*65)
    cfg = copy.deepcopy(base_cfg)
    cfg["simulation"]["maturity"] = 3/12
    cfg["simulation"]["n_steps"] = 63
    V0 = bs_price(cfg)
    rows = []
    for aname, atype in [("DQN",AgentType.DQN),("DoubleDQN",AgentType.DoubleDQN),
                          ("DeepDPG",AgentType.DeepDPG)]:
        logger.info(f"  ► {aname}")
        r = run(cfg, ProcessType.GBM, atype, BenchmarkType.BsDelta)
        rows.append({"Agent":aname,
            "Mean%":r["agent"]["mean"]/V0*100, "±CI%":r["agent"]["mean_ci"]/V0*100,
            "Std%":r["agent"]["std"]/V0*100, "Y(0)%":r["agent"]["Y"]/V0*100,
            "Y improv vs Δ%":r["imp"], "conv":r["cv"]["ok"]})
    return ptable(rows, "Extension B.3 — Agent comparison, GBM T=3m daily, κ=1%\n"
                        "  All agents: dual Q-function, Y=E[C]+λσ(C)")


# ======================================================================
# PLOTS
# ======================================================================
def _sv(fig, name):
    fig.savefig(FIG/name, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  📊 {FIG/name}")

def plot_training(tr, title, fn="training_curve.png"):
    c=[sum(e.costs) for e in tr.episodes["train"]]
    w=max(len(c)//50,10); sm=pd.Series(c).rolling(w,min_periods=1).mean()
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(c,alpha=.12,color="steelblue"); ax.plot(sm,color="steelblue",lw=2,label=f"Rolling avg (w={w})")
    ax.set(xlabel="Episode",ylabel="Hedging cost",title=title); ax.legend(); ax.grid(True,alpha=.3)
    fig.tight_layout(); _sv(fig,fn)

def plot_costs(ac,bc,title,fn="cost_dist.png"):
    fig,ax=plt.subplots(figsize=(10,5))
    bins=np.linspace(min(min(ac),min(bc)),max(max(ac),max(bc)),60)
    ax.hist(bc,bins=bins,alpha=.5,label="Delta",color="salmon",edgecolor="white")
    ax.hist(ac,bins=bins,alpha=.5,label="RL",color="steelblue",edgecolor="white")
    ax.axvline(np.mean(bc),color="salmon",ls="--",lw=2)
    ax.axvline(np.mean(ac),color="steelblue",ls="--",lw=2)
    ax.set(xlabel="Total cost",ylabel="Count",title=title); ax.legend(); ax.grid(True,alpha=.3)
    fig.tight_layout(); _sv(fig,fn)

def plot_scatter(ea,eb,title,fn="scatter.png"):
    ag,dl=[],[]
    for a,b in zip(ea.episodes["eval_agent"],eb.episodes["eval_benchmark"]):
        ag.extend(a.actions); dl.extend(b.actions)
    fig,ax=plt.subplots(figsize=(7,7))
    ax.scatter(dl,ag,alpha=.02,s=3,color="steelblue")
    lims=[min(ax.get_xlim()[0],ax.get_ylim()[0]),max(ax.get_xlim()[1],ax.get_ylim()[1])]
    ax.plot(lims,lims,"k--",lw=1,label="H_RL = Δ_BS")
    ax.fill_between(lims,lims,[lims[1],lims[1]],alpha=.04,color="green",label="Over-hedged")
    ax.fill_between(lims,[lims[0],lims[0]],lims,alpha=.04,color="red",label="Under-hedged")
    ax.set(xlabel="BS Delta position",ylabel="RL position",title=title)
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True,alpha=.3)
    fig.tight_layout(); _sv(fig,fn)

def plot_freq_bars(df, title, fn="freq_bars.png"):
    fig,axes=plt.subplots(1,2,figsize=(14,5)); x=np.arange(len(df)); w=.35; lam=1.5
    ax=axes[0]
    ax.bar(x-w/2, df["Δ Mean%"]+lam*df["Δ Std%"], w, label="Delta Y(0)", color="salmon", edgecolor="white")
    ax.bar(x+w/2, df["RL Mean%"]+lam*df["RL Std%"], w, label="RL Y(0)", color="steelblue", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["Rebal"]); ax.set(ylabel="Y(0) % of V₀")
    ax.set_title(title); ax.legend(); ax.grid(True,alpha=.3,axis="y")
    ax=axes[1]
    cols=["#2ecc71" if v>0 else "#e74c3c" for v in df["Y improv%"]]
    ax.bar(x, df["Y improv%"], .5, color=cols, edgecolor="white")
    ax.axhline(0,color="k",lw=.8); ax.set_xticks(x); ax.set_xticklabels(df["Rebal"])
    ax.set(ylabel="Y(0) improvement %",title="RL improvement vs Delta"); ax.grid(True,alpha=.3,axis="y")
    fig.tight_layout(); _sv(fig,fn)

def plot_agents(df, title, fn="agents.png"):
    fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(len(df))
    colors=["#e74c3c","#f39c12","#2980b9"][:len(df)]
    ax.bar(x, df["Y(0)%"], .45, color=colors, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["Agent"])
    ax.set(ylabel="Y(0) % of V₀",title=title); ax.grid(True,alpha=.3,axis="y")
    fig.tight_layout(); _sv(fig,fn)


def plot_sabr_benchmarks(df, title, fn="sabr_benchmarks.png"):
    """Compare RL, practitioner delta and Bartlett delta across rebalancing frequencies."""
    x = np.arange(len(df)); w = 0.25; lam = 1.5
    rl_y = df["RL Mean%"] + lam * df["RL Std%"]
    pr_y = df["Practitioner Δ Mean%"] + lam * df["Practitioner Δ Std%"]
    ba_y = df["Bartlett Δ Mean%"] + lam * df["Bartlett Δ Std%"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, pr_y, w, label="Practitioner Δ Y(0)", color="salmon", edgecolor="white")
    ax.bar(x,     ba_y, w, label="Bartlett Δ Y(0)",    color="#f39c12", edgecolor="white")
    ax.bar(x + w, rl_y, w, label="RL Y(0)",            color="steelblue", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["Rebal"])
    ax.set(ylabel="Y(0) % of V₀", title=title)
    ax.legend(); ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout(); _sv(fig, fn)


def plot_process_comparison(df, title, fn="process_comparison.png"):
    """Compare RL vs Delta across processes and maturities."""
    lam = 1.5
    labels = [f"{t} — {p}" for t, p in zip(df["T"], df["Process"])]
    x = np.arange(len(df)); w = 0.35
    delta_y = df["Δ Mean%"] + lam * df["Δ Std%"]
    rl_y = df["RL Mean%"] + lam * df["RL Std%"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, delta_y, w, label="Delta Y(0)", color="salmon", edgecolor="white")
    ax.bar(x + w/2, rl_y,    w, label="RL Y(0)",    color="steelblue", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set(ylabel="Y(0) % of V₀", title=title)
    ax.legend(); ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout(); _sv(fig, fn)


def plot_maturity_comparison(df, title, fn="maturity_comparison.png"):
    """Mean/std comparison across maturities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df)); w = 0.35
    for ax, cols, sub in zip(
        axes,
        [("Δ Mean%", "RL Mean%"), ("Δ Std%", "RL Std%")],
        ["Mean cost (% of V₀)", "Std cost (% of V₀)"]
    ):
        ax.bar(x - w/2, df[cols[0]], w, label="Delta", color="salmon", edgecolor="white")
        ax.bar(x + w/2, df[cols[1]], w, label="RL", color="steelblue", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(df["Maturity"])
        ax.set(ylabel="% of V₀", title=f"{title}\n{sub}")
        ax.legend(); ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout(); _sv(fig, fn)


# ======================================================================
# MAIN
# ======================================================================
def main():
    logger.info("╔═══════════════════════════════════════════════════════════════╗")
    logger.info("║  Deep Hedging — Cao, Chen, Hull & Poulos (2021)              ║")
    logger.info("║  PART A: Paper Replication  |  PART B: Extensions            ║")
    logger.info("╚═══════════════════════════════════════════════════════════════╝\n")

    cfg = json_to_dict("config.json")

    # ══════════════════════════════════════════════════════════════════
    #  PART A — PAPER REPLICATION
    # ══════════════════════════════════════════════════════════════════

    # A.1 — GBM, T=1 month (Exhibit 3)
    df_a1 = part_A1_A2(cfg, T=1/12, exhibit_name="Exhibit 3")
    plot_freq_bars(df_a1,
        "Exhibit 3 — GBM, T=1m, Short ATM Call\nS₀=100, K=100, σ=20%, μ=5%, r=0, q=0, κ=1%",
        "A1_exhibit3_gbm_1m.png")

    # A.2 — GBM, T=3 months (Exhibit 4)
    df_a2 = part_A1_A2(cfg, T=3/12, exhibit_name="Exhibit 4")
    plot_freq_bars(df_a2,
        "Exhibit 4 — GBM, T=3m, Short ATM Call\nS₀=100, K=100, σ=20%, μ=5%, r=0, q=0, κ=1%",
        "A2_exhibit4_gbm_3m.png")

    # A.3 — SABR, T=1 month (Exhibit 6)
    df_a3 = part_A3_A4(cfg, T=1/12, exhibit_name="Exhibit 6")
    plot_sabr_benchmarks(df_a3,
        "Exhibit 6 — SABR, T=1m, Short ATM Call\nPractitioner Δ vs Bartlett Δ vs RL | ρ=-0.4, ν=0.6, κ=1%",
        "A3_exhibit6_sabr_1m.png")

    # A.4 — SABR, T=3 months (Exhibit 7)
    df_a4 = part_A3_A4(cfg, T=3/12, exhibit_name="Exhibit 7")
    plot_sabr_benchmarks(df_a4,
        "Exhibit 7 — SABR, T=3m, Short ATM Call\nPractitioner Δ vs Bartlett Δ vs RL | ρ=-0.4, ν=0.6, κ=1%",
        "A4_exhibit7_sabr_3m.png")

    # Baseline plots (from A.1 daily run)
    cfg_1m = copy.deepcopy(cfg); cfg_1m["simulation"]["maturity"]=1/12; cfg_1m["simulation"]["n_steps"]=21
    r = run(cfg_1m, ProcessType.GBM, AgentType.DeepDPG, BenchmarkType.BsDelta)
    plot_training(r["tr"],
        "Training curve — DeepDPG, GBM, T=1m, daily\n"
        "Short ATM Call, S₀=100, σ=20%, κ=1%, PER enabled",
        "A_training_curve.png")
    plot_costs(r["ac"], r["bc"],
        "Cost distribution — DeepDPG vs BS Delta\n"
        "GBM, T=1m, daily rebal, S₀=100, σ=20%, κ=1%",
        "A_cost_distribution.png")
    plot_scatter(r["ea"], r["eb"],
        "Over/Under-hedging — DeepDPG vs BS Delta (Exhibit 5)\n"
        "GBM, T=1m, daily, S₀=100, K=100, σ=20%, κ=1%",
        "A_over_under_hedge.png")

    # ══════════════════════════════════════════════════════════════════
    #  PART B — EXTENSIONS
    # ══════════════════════════════════════════════════════════════════

    # B.1 — SVJ process
    df_b1 = part_B1_svj(cfg)
    plot_process_comparison(df_b1,
        "Extension B.1 — Process comparison\nGBM vs SVJ | DeepDPG vs BS Delta | daily rebalancing, κ=1%",
        "B1_process_comparison.png")

    # B.2 — Longer maturities
    df_b2 = part_B2_maturities(cfg)
    plot_maturity_comparison(df_b2,
        "Extension B.2 — Maturity comparison\nGBM | DeepDPG vs BS Delta | daily rebalancing, κ=1%",
        "B2_maturity_comparison.png")

    # B.3 — Agent comparison
    df_b3 = part_B3_agents(cfg)
    plot_agents(df_b3,
        "Agent comparison — GBM T=3m daily, κ=1%\n"
        "All: dual Q-function, Y=E[C]+1.5·σ(C)\n"
        "DQN/DoubleDQN: discrete (21-grid) | DeepDPG: continuous",
        "B3_agent_comparison.png")

    logger.info("\n=== ALL DONE ===")

if __name__ == "__main__":
    main()
