from __future__ import annotations
import logging
import numpy as np
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
        self.eval_episodes = int(config["training_schedule"]["eval_episodes"])
        self.update_frequency = max(1, int(config["training_schedule"].get("update_frequency", 1)))
        self.kappa = float(config["hedging_env"]["transaction_cost"])
        self.maturity = float(config["simulation"]["maturity"])
        self.training_paths = None
        self.eval_paths = None
        self.train_step_count = 0

    def _episode_times(self, path):
        return np.linspace(0.0, self.maturity, len(path["S"]))

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

    def _run_episode(self, path, policy_fn, learn, record_fn):
        state = self.env.setup_env(path)
        n = self.env.n_steps

        a0 = float(policy_fn(state))
        state_next, raw0 = self.env.apply_action(a0)
        setup_cost = self.kappa * raw0["S_i"] * abs(raw0["H_new"] - raw0["H_prev"])
        pending_setup_reward = -setup_cost

        prev_state = state
        prev_action = a0
        prev_raw = raw0
        state = state_next

        total_reward = 0.0
        for step_idx in range(1, n + 1):
            is_terminal_action = step_idx == n
            if is_terminal_action:
                ai = 0.0
            else:
                ai = float(policy_fn(state))
            state_next, raw_i = self.env.apply_action(ai)

            v_curr = raw_i["V_i"]
            s_curr = raw_i["S_i"]
            v_prev = prev_raw["V_i"]
            s_prev = prev_raw["S_i"]
            h_prev = prev_raw["H_new"]
            h_curr = raw_i["H_new"]

            trade_cost_i = self.kappa * s_curr * abs(h_curr - h_prev)
            reward_i = (v_curr - v_prev) + h_prev * (s_curr - s_prev) - trade_cost_i

            if pending_setup_reward != 0.0:
                reward_i += pending_setup_reward
                pending_setup_reward = 0.0

            liquidation_cost = 0.0
            if is_terminal_action:
                liquidation_cost = self.kappa * s_curr * abs(h_curr)
                reward_i -= liquidation_cost

            total_reward += reward_i
            done = is_terminal_action
            current_loss = None

            if learn:
                self.agent.store_transition(prev_state, prev_action, reward_i, state, done)
                self.train_step_count += 1
                if self.train_step_count % self.update_frequency == 0:
                    current_loss = self.agent.learn()

            info = {
                "spot_t": s_prev,
                "spot_next": s_curr,
                "hedge": h_curr,
                "trade_cost": trade_cost_i,
                "liquidation_cost": liquidation_cost,
                "reward": reward_i,
                "cost": -reward_i,
            }
            record_fn(step_idx - 1, prev_action, reward_i, info, current_loss)

            prev_state = state
            prev_action = ai
            prev_raw = raw_i
            state = state_next

        return total_reward

    def train(self):
        self._ensure_training_paths()
        self.agent.set_train_mode()
        res = HedgingResult()
        for ep in range(self.train_episodes):
            path = self._ep_path(self.training_paths, ep)
            er = EpisodeResult(split="train", episode_idx=ep, times=self._episode_times(path), path_data=path)

            def record(_step_idx, action, _reward, info, loss):
                er.add_step(action=action, info=info, loss=loss)

            policy = lambda s: self.agent.act(s, eval_mode=False)
            self._run_episode(path, policy_fn=policy, learn=True, record_fn=record)
            res.add_episode(er, type="train")
        return res

    def test(self):
        self._ensure_eval_paths()
        self.agent.set_eval_mode()
        res = HedgingResult()
        for ep in range(self.eval_episodes):
            path = self._ep_path(self.eval_paths, ep)
            er = EpisodeResult(split="eval_agent", episode_idx=ep, times=self._episode_times(path), path_data=path)

            def record(_step_idx, action, _reward, info, _loss):
                er.add_step(action=action, info=info)

            policy = lambda s: self.agent.act(s, eval_mode=True)
            self._run_episode(path, policy_fn=policy, learn=False, record_fn=record)
            res.add_episode(er, type="eval_agent")
        return res

    def test_benchmark(self, benchmark_override=None):
        self._ensure_eval_paths()
        bench = benchmark_override or self.benchmark
        res = HedgingResult()
        for ep in range(self.eval_episodes):
            path = self._ep_path(self.eval_paths, ep)
            er = EpisodeResult(split="eval_benchmark", episode_idx=ep, times=self._episode_times(path), path_data=path)

            def record(_step_idx, action, _reward, info, _loss):
                er.add_step(action=action, info=info)

            policy = lambda s: bench(s)
            self._run_episode(path, policy_fn=policy, learn=False, record_fn=record)
            res.add_episode(er, type="eval_benchmark")
        return res
