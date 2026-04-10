import logging
from copy import deepcopy
from rich.logging import RichHandler

from src.utils.enums import AgentType, BenchmarkType, ProcessType
from src.utils.helpers import (
    bs_price, freq_map, json_to_dict, plot_costs, plot_freq_bars,
    plot_sabr_benchmarks, plot_scatter, plot_training, ptable,
    reset_seed, run, run_multi_bench
)

reset_seed()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    force=True
)
logger = logging.getLogger(__name__)


def run_gbm(cfg, T, name):
    logger.info(f"\n[ {name} : GBM, T={T * 12:.0f}m ]")
    rows, v0 = [], None

    for label, steps in freq_map(T).items():
        c = deepcopy(cfg)
        c["simulation"].update({"maturity": T, "n_steps": steps})
        if not v0: v0 = bs_price(c)

        res = run(c, ProcessType.GBM, AgentType.DeepDPG, BenchmarkType.BsDelta)
        rows.append({
            "Rebal": label,
            "Δ Mean%": res["bench"]["mean"] / v0 * 100,
            "Δ Std%": res["bench"]["std"] / v0 * 100,
            "RL Mean%": res["agent"]["mean"] / v0 * 100,
            "RL Std%": res["agent"]["std"] / v0 * 100,
            "Y improv%": res["imp"],
            "conv": res["cv"]["ok"]
        })
    return ptable(rows, f"{name} (GBM)")


def run_sabr(cfg, T, name):
    logger.info(f"\n[ {name} : SABR, T={T * 12:.0f}m ]")
    rows, v0 = [], None

    for label, steps in freq_map(T).items():
        c = deepcopy(cfg)
        c["simulation"].update({"maturity": T, "n_steps": steps})
        if not v0: v0 = bs_price(c)

        res = run_multi_bench(c, ProcessType.SABR, AgentType.DeepDPG)
        row = {
            "Rebal": label,
            "RL Mean%": res["agent"]["mean"] / v0 * 100,
            "RL Std%": res["agent"]["std"] / v0 * 100
        }

        for bname, bdata in res["benchmarks"].items():
            row[f"{bname} Mean%"] = bdata["metrics"]["mean"] / v0 * 100
            row[f"{bname} Std%"] = bdata["metrics"]["std"] / v0 * 100
            row[f"Y imp vs {bname}%"] = bdata["imp"]

        rows.append(row)
    return ptable(rows, f"{name} (SABR)")


def main():
    cfg = json_to_dict("config.json")

    # --- PART A: REPLICATION ---
    df_e3 = run_gbm(cfg, 1 / 12, "Exhibit 3")
    plot_freq_bars(df_e3, "Exhibit 3 (GBM 1m)", "exhibit3_gbm_1m.png")

    df_e4 = run_gbm(cfg, 3 / 12, "Exhibit 4")
    plot_freq_bars(df_e4, "Exhibit 4 (GBM 3m)", "exhibit4_gbm_3m.png")

    df_e6 = run_sabr(cfg, 1 / 12, "Exhibit 6")
    plot_sabr_benchmarks(df_e6, "Exhibit 6 (SABR 1m)", "exhibit6_sabr_1m.png")

    df_e7 = run_sabr(cfg, 3 / 12, "Exhibit 7")
    plot_sabr_benchmarks(df_e7, "Exhibit 7 (SABR 3m)", "exhibit7_sabr_3m.png")

    # --- BASE PLOTS (Daily GBM 1m) ---
    logger.info("\n[ Generating Base Plots... ]")
    c_base = deepcopy(cfg)
    c_base["simulation"].update({"maturity": 1 / 12, "n_steps": 21})
    res_base = run(c_base, ProcessType.GBM, AgentType.DeepDPG, BenchmarkType.BsDelta)

    plot_training(res_base["tr"], "Training (GBM 1m)", "training_gbm.png")
    plot_costs(res_base["ac"], res_base["bc"], "Cost Dist (RL vs BS)", "cost_dist.png")
    plot_scatter(res_base["ea"], res_base["eb"], "Over/Under-hedging", "exhibit5_scatter.png")


if __name__ == "__main__":
    main()