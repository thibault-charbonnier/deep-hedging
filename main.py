import logging
import cProfile
import io
import pstats
import argparse
from pathlib import Path
from rich.logging import RichHandler
from src.utils.helpers import json_to_dict, set_global_seed
from src.orchestrator import Orchestrator
from src.utils.enums import AgentType, ProcessType, BenchmarkType
from src.persistence import RunStore
from src.persistence.run_store import RunContext
from src.valuation.bs_valuation import BSValuation


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("--- Start ---")


REBALANCE_FREQ_DAYS = {
    "daily": 1,
    "2d": 2,
    "3d": 3,
    "weekly": 5,
    "biweekly": 10,
    "monthly": 21,
}


def _enum_from_name(enum_cls, name: str, default_name: str):
    key = name or default_name
    if key in enum_cls.__members__:
        return enum_cls[key]
    lowered = {k.lower(): k for k in enum_cls.__members__}
    if key.lower() in lowered:
        return enum_cls[lowered[key.lower()]]
    valid = ", ".join(enum_cls.__members__.keys())
    raise ValueError(f"Invalid {enum_cls.__name__}: '{key}'. Valid values: {valid}")


def _resolve_rebalance_steps(*, maturity: float, rebalancing: str, trading_days_per_year: int) -> int:
    if maturity <= 0:
        raise ValueError("maturity must be > 0")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be > 0")
    if rebalancing not in REBALANCE_FREQ_DAYS:
        valid = ", ".join(REBALANCE_FREQ_DAYS.keys())
        raise ValueError(f"Invalid rebalancing: '{rebalancing}'. Valid values: {valid}")
    freq_days = REBALANCE_FREQ_DAYS[rebalancing]
    return max(1, int(round(maturity * trading_days_per_year / freq_days)))


def _apply_time_grid_overrides(config: dict, args: argparse.Namespace) -> None:
    run_cfg = config.get("run", {})

    maturity = float(args.maturity) if args.maturity is not None else float(run_cfg.get("maturity", config["simulation"]["maturity"]))
    trading_days_per_year = int(args.trading_days_per_year) if args.trading_days_per_year is not None else int(run_cfg.get("trading_days_per_year", 252))

    config["simulation"]["maturity"] = maturity

    if args.n_steps is not None:
        config["simulation"]["n_steps"] = int(args.n_steps)
        return

    rebalancing = args.rebalancing or run_cfg.get("rebalancing")
    if rebalancing:
        config["simulation"]["n_steps"] = _resolve_rebalance_steps(
            maturity=maturity,
            rebalancing=str(rebalancing),
            trading_days_per_year=trading_days_per_year,
        )


def _option_price_t0(config: dict) -> float:
    maturity = float(config["simulation"]["maturity"])
    spot = float(config["simulation"]["S0"])
    sigma = float(config["simulation"]["gbm"]["sigma"])
    engine = BSValuation(
        strike=config["derivative"]["strike"],
        maturity=maturity,
        rate=config["derivative"].get("rf_rate", 0.0),
        dividend=config["derivative"].get("div_rate", 0.0),
        option_type=config.get("derivative", {}).get("option_type", "call"),
    )
    p, _ = engine.price_and_delta(spot=spot, t=0.0, sigma=sigma)
    return abs(float(p))


def _run_pipeline(
    config: dict,
    run_mode: str,
    process_name: str | None = None,
    agent_name: str | None = None,
    benchmark_name: str | None = None,
    seed: int | None = None,
) -> tuple[RunStore, RunContext]:
    run_cfg = config.get("run", {})
    process_type = _enum_from_name(ProcessType, process_name or run_cfg.get("process", "GBM"), "GBM")
    agent_type = _enum_from_name(AgentType, agent_name or run_cfg.get("agent", "DeepDPG"), "DeepDPG")
    benchmark_type = _enum_from_name(BenchmarkType, benchmark_name or run_cfg.get("benchmark", "BsDelta"), "BsDelta")

    logger.info(
        "Run setup: mode=%s process=%s agent=%s benchmark=%s",
        run_mode,
        process_type.name,
        agent_type.name,
        benchmark_type.name,
    )

    runner = Orchestrator(
        config=config,
        process_type=process_type,
        agent_type=agent_type,
        benchmark_type=benchmark_type,
    )

    run_tag = f"main_{run_mode}_{process_type.name}_{agent_type.name}_{benchmark_type.name}"
    store = RunStore(base_dir="outputs")
    extra_meta = {"seed": seed} if seed is not None else None
    ctx = store.start_run(script=run_tag, config=config, extra_meta=extra_meta)
    option_price_t0 = _option_price_t0(config)
    ok = False
    try:
        risk_lambda = float(config.get("hedging_agent", {}).get("risk_lambda", 1.5))
        save_figures = bool(run_cfg.get("save_figures", True))

        if run_mode in {"train", "full", "smoke"}:
            train_res = runner.train()
            store.save_result(
                ctx=ctx,
                result=train_res,
                label="train",
                risk_lambda=risk_lambda,
                save_figures=save_figures,
                option_price_t0=option_price_t0,
            )

        if run_mode in {"eval_agent", "full", "smoke"}:
            eval_agent_res = runner.test()
            store.save_result(
                ctx=ctx,
                result=eval_agent_res,
                label="eval_agent",
                risk_lambda=risk_lambda,
                save_figures=save_figures,
                option_price_t0=option_price_t0,
            )

        if run_mode in {"eval_benchmark", "full", "smoke"}:
            eval_benchmark_res = runner.test_benchmark()
            store.save_result(
                ctx=ctx,
                result=eval_benchmark_res,
                label="eval_benchmark",
                risk_lambda=risk_lambda,
                save_figures=save_figures,
                option_price_t0=option_price_t0,
            )

        ok = True
        logger.info("Run saved in outputs/%s", ctx.run_id)
    finally:
        store.finalize(ctx=ctx, ok=ok)
    return store, ctx


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-hedging runner")
    parser.add_argument("--config", default="config.json", help="Path to config json")
    parser.add_argument(
        "--mode",
        choices=["train", "eval_agent", "eval_benchmark", "full", "smoke"],
        default=None,
        help="Execution mode",
    )
    parser.add_argument("--process", default=None, help=f"Process type ({', '.join(ProcessType.__members__.keys())})")
    parser.add_argument("--agent", default=None, help=f"Agent type ({', '.join(AgentType.__members__.keys())})")
    parser.add_argument("--benchmark", default=None, help=f"Benchmark type ({', '.join(BenchmarkType.__members__.keys())})")
    parser.add_argument("--maturity", type=float, default=None, help="Override maturity in years (e.g. 1.0, 0.25)")
    parser.add_argument("--n-steps", type=int, default=None, help="Override exact number of rebalancing steps")
    parser.add_argument(
        "--rebalancing",
        choices=list(REBALANCE_FREQ_DAYS.keys()),
        default=None,
        help="Set rebalancing frequency (maps to n_steps using trading days per year)",
    )
    parser.add_argument("--trading-days-per-year", type=int, default=None, help="Trading days per year for rebalancing mapping (default 252)")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed for reproducible runs")
    args = parser.parse_args()

    config = json_to_dict(args.config)
    cfg_mode = config.get("run", {}).get("mode", "full")
    run_mode = args.mode or cfg_mode

    profiling_enabled = bool(config.get("run", {}).get("enable_cprofile", True))
    profile_top_n = int(config.get("run", {}).get("profile_top_n", 60))

    seed = args.seed if args.seed is not None else config.get("run", {}).get("seed")
    if seed is not None:
        seed = int(seed)
        set_global_seed(seed)
        logger.info("Seed set to %d", seed)

    if run_mode == "smoke":
        config["training_schedule"]["train_episodes"] = int(config.get("run", {}).get("smoke_train_episodes", 5))
        config["training_schedule"]["eval_episodes"] = int(config.get("run", {}).get("smoke_eval_episodes", 5))
        config["simulation"]["n_steps"] = int(config.get("run", {}).get("smoke_n_steps", 20))

    # Apply maturity/n_steps/rebalancing overrides after smoke so CLI stays the final authority.
    _apply_time_grid_overrides(config, args)

    logger.info(
        "Time grid: maturity=%.6f years, n_steps=%d",
        float(config["simulation"]["maturity"]),
        int(config["simulation"]["n_steps"]),
    )

    profiler = cProfile.Profile() if profiling_enabled else None
    if profiler is not None:
        profiler.enable()
    store, ctx = _run_pipeline(
        config=config,
        run_mode=run_mode,
        process_name=args.process,
        agent_name=args.agent,
        benchmark_name=args.benchmark,
        seed=seed,
    )
    if profiler is not None:
        profiler.disable()
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        stats.print_stats(profile_top_n)
        report = s.getvalue()
        print(report)
        store.save_profile_text(ctx=ctx, stats_text=report)
        prof_path = Path(ctx.profile_dir) / "cprofile.prof"
        profiler.dump_stats(str(prof_path))

    logger.info("--- END ---")


if __name__ == "__main__":
    main()

