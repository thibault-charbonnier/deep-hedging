from __future__ import annotations
import logging
from .hedging_strategy.hedging_env import HedgingEnv
from .hedging_result import HedgingResult, EpisodeResult

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
        self.update_frequency = max(1, int(config["training_schedule"].get("update_frequency", 1)))
        self.training_paths = None
        self.eval_paths = None

    def _ep_path(self, paths, ep):
        return {k: v[ep] for k, v in paths.items()}

    def _ensure_training_paths(self):
        if self.training_paths is None:
            logger.info("Simulating training paths...")
            self.training_paths = self.process.simulate_paths(self.train_episodes)

    def _ensure_eval_paths(self):
        if self.eval_paths is None:
            logger.info("Simulating evaluation paths...")
            self.eval_paths = self.process.simulate_paths(self.eval_episodes)

    def train(self):
        self._ensure_training_paths()
        self.agent.set_train_mode()
        res = HedgingResult()
        step_count = 0
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
                step_count += 1
                loss = self.agent.learn() if (step_count % self.update_frequency == 0) else None
                er.add_step(action=action, info=info, loss=loss)
                state = ns
            res.add_episode(er, type="train")
        return res

    def test(self):
        self._ensure_eval_paths()
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
        self._ensure_eval_paths()
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
