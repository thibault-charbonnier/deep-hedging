from __future__ import annotations
import inspect
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

        # Setup transition: (s_0, a_0, R_setup, s_1, done=False)
        a0 = float(policy_fn(state, 0))
        next_state, raw0 = self.env.apply_action(a0)
        setup_cost = self.kappa * raw0["S_i"] * abs(raw0["H_new"] - raw0["H_prev"])
        setup_reward = -setup_cost

        setup_loss = None
        if learn:
            self.agent.store_transition(state, a0, setup_reward, next_state, False)
            self.train_step_count += 1
            if self.train_step_count % self.update_frequency == 0:
                setup_loss = self.agent.learn()

        setup_info = {
            "reward": setup_reward,
            "cost": setup_cost,
            "trade_cost": setup_cost,
            "liquidation_cost": 0.0,
            "spot_t": raw0["S_i"],
            "spot_next": raw0["S_i"],
            "hedge": raw0["H_new"],
        }
        record_fn(-1, a0, setup_reward, setup_info, setup_loss)

        total_reward = setup_reward
        state, prev_raw = next_state, raw0

        for step_idx in range(1, n + 1):
            is_terminal = step_idx == n
            if is_terminal:
                # Contractual close-out: no policy decision at terminal step.
                ai = 0.0
            else:
                ai = float(policy_fn(state, step_idx))
            next_state, raw_i = self.env.apply_action(ai)

            v_curr = raw_i["V_i"]
            s_curr = raw_i["S_i"]
            v_prev = prev_raw["V_i"]
            s_prev = prev_raw["S_i"]
            h_prev = raw_i["H_prev"]
            h_curr = raw_i["H_new"]

            if is_terminal:
                trade_cost = 0.0
                liquidation_cost = self.kappa * s_curr * abs(h_prev)
                reward = (v_curr - v_prev) + h_prev * (s_curr - s_prev) - liquidation_cost
            else:
                trade_cost = self.kappa * s_curr * abs(h_curr - h_prev)
                liquidation_cost = 0.0
                reward = (v_curr - v_prev) + h_prev * (s_curr - s_prev) - trade_cost

            total_reward += reward
            loss = None

            if learn:
                buffer_next = next_state if next_state is not None else state
                self.agent.store_transition(state, ai, reward, buffer_next, is_terminal)
                self.train_step_count += 1
                if self.train_step_count % self.update_frequency == 0:
                    loss = self.agent.learn()

            info = {
                "spot_t": s_prev,
                "spot_next": s_curr,
                "hedge": h_curr,
                "trade_cost": trade_cost,
                "liquidation_cost": liquidation_cost,
                "reward": reward,
                "cost": -reward,
            }
            record_fn(step_idx - 1, ai, reward, info, loss)

            prev_raw = raw_i
            state = next_state

        return total_reward

    def train(self):
        self._ensure_training_paths()
        self.agent.set_train_mode()
        res = HedgingResult()
        for ep in range(self.train_episodes):
            path = self._ep_path(self.training_paths, ep)
            er = EpisodeResult(split="train", episode_idx=ep, times=self._episode_times(path), path_data=path)

            def record(_step_idx, action, _reward, info, loss):
                if _step_idx == -1:
                    er.set_setup(action=action, info=info, loss=loss)
                else:
                    er.add_step(action=action, info=info, loss=loss)

            policy = lambda s, _step_idx: self.agent.act(s, eval_mode=False)
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
                if _step_idx == -1:
                    er.set_setup(action=action, info=info, loss=None)
                else:
                    er.add_step(action=action, info=info)

            policy = lambda s, _step_idx: self.agent.act(s, eval_mode=True)
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

            sigma_path = None
            if "sigma" in path:
                sigma_path = np.asarray(path["sigma"], dtype=float)
            elif "variance" in path:
                sigma_path = np.sqrt(np.maximum(np.asarray(path["variance"], dtype=float), 1e-10))

            bench_call_arity = len(inspect.signature(bench.__call__).parameters)

            def record(_step_idx, action, _reward, info, _loss):
                if _step_idx == -1:
                    er.set_setup(action=action, info=info, loss=None)
                else:
                    er.add_step(action=action, info=info)

            def policy(s, step_idx):
                if bench_call_arity >= 2:
                    sigma_t = None
                    if sigma_path is not None:
                        sigma_t = float(sigma_path[min(step_idx, len(sigma_path) - 1)])
                    return bench(s, sigma_t)
                return bench(s)

            self._run_episode(path, policy_fn=policy, learn=False, record_fn=record)
            res.add_episode(er, type="eval_benchmark")
        return res
