import logging
import argparse
from rich.logging import RichHandler
from src.utils.helpers import json_to_dict, set_global_seed
from src.orchestrator import Orchestrator
from src.utils.enums import AgentType, ProcessType, BenchmarkType
from src.persistence import RunStore
from src.persistence.run_store import RunContext
from src.valuation.bs_valuation import option_price_t0


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("--- Start ---")


TRADING_DAYS_PER_YEAR = 252


def _run_pipeline(config: dict) -> tuple[RunStore, RunContext]:
    """Run complete pipeline: train + eval_agent + eval_benchmark."""
    run_cfg = config["run"]
    process_type = ProcessType[run_cfg["process"]]
    agent_type = AgentType[run_cfg["agent"]]
    benchmark_type = BenchmarkType[run_cfg["benchmark"]]

    logger.info(
        "Run setup: process=%s agent=%s benchmark=%s",
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

    run_tag = f"{process_type.name}_{agent_type.name}_{benchmark_type.name}"
    store = RunStore(base_dir="outputs")
    seed = run_cfg.get("seed")
    extra_meta = {"seed": seed} if seed is not None else None
    ctx = store.start_run(script=run_tag, config=config, extra_meta=extra_meta)
    option_price_t0_value = option_price_t0(config)
    ok = False
    try:
        risk_lambda = float(config["hedging_agent"]["risk_lambda"])

        for label, run_step in (
            ("train", runner.train),
            ("eval_agent", runner.test),
            ("eval_benchmark", runner.test_benchmark),
        ):
            result = run_step()
            store.save_result(
                ctx=ctx,
                result=result,
                label=label,
                risk_lambda=risk_lambda,
                option_price_t0=option_price_t0_value,
            )

        ok = True
        logger.info("Run saved in outputs/%s", ctx.run_id)
    finally:
        store.finalize(ctx=ctx, ok=ok)
    return store, ctx


def main() -> None:
    """Entry point: parse CLI args, load config, seed, and run the pipeline."""
    parser = argparse.ArgumentParser(description="Deep-hedging runner - configure everything in config.json")
    parser.add_argument("--config", default="config.json", help="Path to config file (default: config.json)")
    args = parser.parse_args()

    config = json_to_dict(args.config)
    run_cfg = config.get("run", {})

    seed = run_cfg.get("seed")
    if seed is not None:
        seed = int(seed)
        set_global_seed(seed)
        logger.info("Seed set to %d", seed)

    maturity_years = float(run_cfg.get("maturity", 0.25))
    config["simulation"]["maturity"] = maturity_years

    rebalancing = int(run_cfg.get("rebalancing", 1))
    n_steps = max(1, int(round(maturity_years * TRADING_DAYS_PER_YEAR / rebalancing)))
    config["simulation"]["n_steps"] = n_steps

    logger.info(
        "Configuration: maturity=%.4f years, rebalancing=%s days, n_steps=%d, seed=%s",
        maturity_years,
        rebalancing,
        n_steps,
        seed or "default",
    )

    _run_pipeline(config=config)

    logger.info("--- END ---")


if __name__ == "__main__":
    main()
