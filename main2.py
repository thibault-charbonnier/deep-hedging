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
from rich.logging import RichHandler

from src.utils.enums import AgentType, BenchmarkType, ProcessType
from src.utils.helpers import (
    bs_price,
    freq_map,
    json_to_dict,
    plot_agents,
    plot_costs,
    plot_freq_bars,
    plot_maturity_comparison,
    plot_process_comparison,
    plot_sabr_benchmarks,
    plot_scatter,
    plot_training,
    ptable,
    reset_seed,
    run,
    run_multi_bench,
)

reset_seed()

logging.basicConfig(level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)], force=True)
logger = logging.getLogger(__name__)

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
