import logging
from rich.logging import RichHandler
from src.utils.helpers import json_to_dict
from src.orchestrator import Orchestrator
from src.utils.enums import AgentType, ProcessType, BenchmarkType


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("--- Start ---")

config = json_to_dict("config.json")
runner = Orchestrator(
    config=config,
    process_type=ProcessType.GBM,
    agent_type=AgentType.DQN,
    benchmark_type=BenchmarkType.BsDelta
)

# runner.train()
# runner.test()
runner.test_benchmark()

logger.info("--- END ---")
