from __future__ import annotations
import logging
from typing import Any
from .hedging_strategy.hedging_env import HedgingEnv
from .hedging_result import HedgingResult, EpisodeResult
from .utils.enums import AgentType, ProcessType, BenchmarkType

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config, process_type, agent_type, benchmark_type):
        self.config = config
        self.env = HedgingEnv(config)
        self.process = process_type.value(config["simulation"])
        self.agent = agent_type.value(config["hedging_agent"])
        self.benchmark = benchmark_type.value(config)
        self.train_episodes = int(config["training_schedule"]["train_episodes"])
        self.eval_episodes  = int(config["training_schedule"]["eval_episodes"])
        logger.info("Simulating paths...")
        self.training_paths = self.process.simulate_paths(self.train_episodes)
        self.eval_paths     = self.process.simulate_paths(self.eval_episodes)

    def _ep_path(self, paths, ep):
        return {k: v[ep] for k, v in paths.items()}

    def train(self):
        self.agent.set_train_mode()
        res = HedgingResult()
        for ep in range(self.train_episodes):
            path = self._ep_path(self.training_paths, ep)
            state = self.env.setup_env(path)
            done = False
            er = EpisodeResult(split="train", episode_idx=ep,
                               times=self.env.times, path_data=path)
            while not done:
                action = self.agent.act(state, eval_mode=False)
                ns, reward, done, info = self.env.step(action)
                self.agent.store_transition(state, action, reward, ns, done)
                loss = self.agent.learn()
                er.add_step(action=action, info=info, loss=loss)
                state = ns
            res.add_episode(er, type="train")
        return res

    def test(self):
        self.agent.set_eval_mode()
        res = HedgingResult()
        for ep in range(self.eval_episodes):
            path = self._ep_path(self.eval_paths, ep)
            state = self.env.setup_env(path)
            done = False
            er = EpisodeResult(split="eval_agent", episode_idx=ep,
                               times=self.env.times, path_data=path)
            while not done:
                action = self.agent.act(state, eval_mode=True)
                state, _, done, info = self.env.step(action)
                er.add_step(action=action, info=info)
            res.add_episode(er, type="eval_agent")
        return res

    def test_benchmark(self, benchmark_override=None):
        bench = benchmark_override or self.benchmark
        res = HedgingResult()
        for ep in range(self.eval_episodes):
            path = self._ep_path(self.eval_paths, ep)
            state = self.env.setup_env(path)
            done = False
            er = EpisodeResult(split="eval_benchmark", episode_idx=ep,
                               times=self.env.times, path_data=path)
            while not done:
                action = bench(state)
                state, _, done, info = self.env.step(action)
                er.add_step(action=action, info=info)
            res.add_episode(er, type="eval_benchmark")
        return res
