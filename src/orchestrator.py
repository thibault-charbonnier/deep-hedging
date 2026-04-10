from __future__ import annotations

import logging
from typing import Any

from .hedging_strategy.hedging_env import HedgingEnv
from .hedging_result import HedgingResult, EpisodeResult
from .utils.enums import AgentType, ProcessType, BenchmarkType


logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        config: dict[str, Any],
        process_type: ProcessType,
        agent_type: AgentType,
        benchmark_type: BenchmarkType,
    ) -> None:
        self.config = config
        self.env = HedgingEnv(config)
        self.process = process_type.value(config["simulation"])
        self.agent = agent_type.value(config["hedging_agent"])
        self.benchmark = benchmark_type.value(config)

        self.train_episodes = int(config["training_schedule"]["train_episodes"])
        self.eval_episodes = int(config["training_schedule"]["eval_episodes"])

        logger.info("Simulating training and evaluation paths...")
        self.training_paths = self.process.simulate_paths(self.train_episodes)
        self.eval_paths = self.process.simulate_paths(self.eval_episodes)

    def _episode_path(self, paths: dict[str, Any], episode: int) -> dict[str, Any]:
        return {key: value[episode] for key, value in paths.items()}

    def train(self) -> HedgingResult:
        self.agent.set_train_mode()
        res = HedgingResult()
        for episode in range(self.train_episodes):
            episode_path = self._episode_path(self.training_paths, episode)
            state = self.env.setup_env(episode_path)
            done = False

            episode_res = EpisodeResult(
                split="train",
                episode_idx=episode,
                times=self.env.times,
                path_data=episode_path,
            )

            while not done:
                action = self.agent.act(state, eval_mode=False)
                next_state, reward, done, info = self.env.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                loss = self.agent.learn()
                episode_res.add_step(action=action, info=info, loss=loss)
                state = next_state

            res.add_episode(episode_res, type="train")
        return res

    def test(self) -> HedgingResult:
        self.agent.set_eval_mode()
        res = HedgingResult()
        for episode in range(self.eval_episodes):
            episode_path = self._episode_path(self.eval_paths, episode)
            state = self.env.setup_env(episode_path)
            done = False

            episode_res = EpisodeResult(
                split="eval_agent",
                episode_idx=episode,
                times=self.env.times,
                path_data=episode_path,
            )

            while not done:
                action = self.agent.act(state, eval_mode=True)
                state, _, done, info = self.env.step(action)
                episode_res.add_step(action=action, info=info)

            res.add_episode(episode_res, type="eval_agent")
        return res

    def test_benchmark(self) -> HedgingResult:
        res = HedgingResult()
        for episode in range(self.eval_episodes):
            episode_path = self._episode_path(self.eval_paths, episode)
            state = self.env.setup_env(episode_path)
            done = False

            episode_res = EpisodeResult(
                split="eval_benchmark",
                episode_idx=episode,
                times=self.env.times,
                path_data=episode_path,
            )

            while not done:
                action = self.benchmark(state)
                state, _, done, info = self.env.step(action)
                episode_res.add_step(action=action, info=info)

            res.add_episode(episode_res, type="eval_benchmark")
        return res
