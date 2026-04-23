from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
from .hedging_strategy.hedging_env import HedgingEnv
from .hedging_result import HedgingResult, EpisodeResult

logger = logging.getLogger(__name__)


def _paths_cache_key(process_name: str, sim_cfg: dict, n_paths: int, seed) -> str:
    """Deterministic key from (process, sim params, n_paths, seed)."""
    relevant = {"process": process_name, "sim": sim_cfg, "n_paths": int(n_paths), "seed": seed}
    blob = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _load_paths_npz(path: Path) -> dict[str, np.ndarray] | None:
    """Load a ``{name: array}`` dict from an ``.npz`` file, or None if missing."""
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k].copy() for k in data.files}


def _save_paths_npz(path: Path, paths: dict[str, np.ndarray]) -> None:
    """Save a ``{name: array}`` dict to ``path`` as an ``.npz`` archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **paths)


class Orchestrator:
    """Coordinate path simulation, agent training/evaluation, and benchmark evaluation.

    Responsibilities:
    - Simulate (and optionally cache) training and evaluation paths.
    - Run the episode loops for training, agent eval, and benchmark eval.
    - Hand each step to the HedgingEnv and feed transitions to the agent.
    """

    def __init__(self, config, process_type, agent_type, benchmark_type):
        self.config = config
        self.process_name = process_type.name
        self.env = HedgingEnv(config)
        self.process = process_type.value(config["simulation"])
        self.agent = agent_type.value(config["hedging_agent"])
        self.benchmark = benchmark_type.value(config)
        self.train_episodes = int(config["training_schedule"]["train_episodes"])
        self.eval_episodes = int(config["training_schedule"]["eval_episodes"])
        self.update_frequency = max(1, int(config["training_schedule"].get("update_frequency", 1)))
        self.training_paths = None
        self.eval_paths = None

        # Optional path cache (opt-in via run.paths_cache_dir).
        # If set, paths are loaded from / saved to <cache_dir>/<hash>_{train,eval}.npz
        # so multiple runs sharing the same (process, sim config, seed, n_paths)
        # hit the exact same paths without re-simulating.
        cache_dir = config.get("run", {}).get("paths_cache_dir")
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._seed = config.get("run", {}).get("seed")

    def _ep_path(self, paths, ep):
        """Slice the ``{name: array}`` batch into a single-episode ``{name: row}`` dict."""
        return {k: v[ep] for k, v in paths.items()}

    def _cache_path(self, kind: str, n_paths: int) -> Path | None:
        """Return the cache file path for ``kind`` in {'train','eval'} or None if caching is disabled."""
        if self._cache_dir is None:
            return None
        key = _paths_cache_key(self.process_name, self.config["simulation"], n_paths, self._seed)
        return self._cache_dir / f"{key}_{kind}.npz"

    def _ensure_training_paths(self):
        """Load training paths from cache if available, otherwise simulate and cache them."""
        if self.training_paths is not None:
            return
        cache_path = self._cache_path("train", self.train_episodes)
        if cache_path is not None:
            loaded = _load_paths_npz(cache_path)
            if loaded is not None:
                logger.info("Loaded cached training paths from %s", cache_path)
                self.training_paths = loaded
                return
        logger.info("Simulating training paths...")
        self.training_paths = self.process.simulate_paths(self.train_episodes)
        if cache_path is not None:
            _save_paths_npz(cache_path, self.training_paths)
            logger.info("Cached training paths to %s", cache_path)

    def _ensure_eval_paths(self):
        """Load evaluation paths from cache if available, otherwise simulate and cache them."""
        if self.eval_paths is not None:
            return
        cache_path = self._cache_path("eval", self.eval_episodes)
        if cache_path is not None:
            loaded = _load_paths_npz(cache_path)
            if loaded is not None:
                logger.info("Loaded cached evaluation paths from %s", cache_path)
                self.eval_paths = loaded
                return
        logger.info("Simulating evaluation paths...")
        self.eval_paths = self.process.simulate_paths(self.eval_episodes)
        if cache_path is not None:
            _save_paths_npz(cache_path, self.eval_paths)
            logger.info("Cached evaluation paths to %s", cache_path)

    def train(self):
        """Train the agent over ``train_episodes`` simulated paths.

        For each episode: pick an initial hedge H0, step through the path
        while collecting (s, a, r, s', done) transitions into the agent's
        replay buffer, periodically calling ``agent.learn()``. Uses a
        buffered commit so the terminal liquidation reward is folded into
        the second-to-last transition (avoids bootstrapping through a
        dummy terminal step).
        """
        self._ensure_training_paths()
        self.agent.set_train_mode()
        res = HedgingResult()
        step_count = 0
        for ep in range(self.train_episodes):
            path = self._ep_path(self.training_paths, ep)
            state_init = self.env.setup_env(path)
            H0 = self.agent.act(state_init, eval_mode=False)
            self.env.set_initial_hedge(H0)
            setup_cost = self.env.transac_cost * abs(float(path["S"][0]) * float(H0))
            er = EpisodeResult(split="train", episode_idx=ep, times=self.env.times, path_data=path)
            state = np.asarray(self.env._build_state(self.env.i, self.env.h_prev), dtype=np.float32)
            self.agent.store_transition(state_init, H0, -float(setup_cost), state, False)
            step_count += 1
            setup_loss = self.agent.learn() if (step_count % self.update_frequency == 0) else None
            er.add_step(
                action=H0,
                info={
                    "spot_t": float(path["S"][0]),
                    "spot_next": float(path["S"][0]),
                    "hedge": float(H0),
                    "trade_cost": float(setup_cost),
                    "liquidation_cost": 0.0,
                    "reward": -float(setup_cost),
                    "cost": float(setup_cost),
                },
                loss=setup_loss,
                agent_info={"is_setup_step": True},
            )
            done = False
            # Buffered-commit: a non-terminal transition is held one iteration
            # before being stored, so that if the NEXT step is terminal we can
            # fold the liquidation reward into it and mark done=True — instead
            # of storing a separate terminal transition (s_{n-1}, a=0, r_liq)
            # that would leave Q(s_{n-1}, ·) uncalibrated off-zero and poison
            # the bootstrap target at i=n-2 via target-policy extrapolation.
            prev_s = prev_a = prev_r = None
            while not done:
                is_terminal = (self.env.i == self.env.n_steps - 1)
                action = 0.0 if is_terminal else self.agent.act(state, eval_mode=False)
                ns, reward, done, info = self.env.step(action)
                if is_terminal:
                    if prev_s is not None:
                        # Fold r_liq into the prior transition (second-to-last
                        # decision) and terminate there: no bootstrap on s_{n-1}.
                        self.agent.store_transition(prev_s, prev_a, prev_r + reward, ns, True)
                    else:
                        # Degenerate case n_steps == 1: no prior transition
                        # exists to absorb the liquidation reward.
                        self.agent.store_transition(state, action, reward, ns, True)
                    prev_s = prev_a = prev_r = None
                else:
                    if prev_s is not None:
                        self.agent.store_transition(prev_s, prev_a, prev_r, state, False)
                    prev_s, prev_a, prev_r = state, action, reward
                step_count += 1
                loss = self.agent.learn() if (step_count % self.update_frequency == 0) else None
                er.add_step(action=action, info=info, loss=loss)
                state = ns
            res.add_episode(er, type="train")
        return res

    def _run_eval_episodes(self, policy_fn, split: str) -> HedgingResult:
        """Run ``eval_episodes`` episodes with ``policy_fn(state) -> action``, no learning.

        Used for both the trained-agent evaluation and the analytical
        benchmark evaluation — they differ only in the policy.
        """
        self._ensure_eval_paths()
        res = HedgingResult()
        for ep in range(self.eval_episodes):
            path = self._ep_path(self.eval_paths, ep)
            state = self.env.setup_env(path)
            H0 = policy_fn(state)
            self.env.set_initial_hedge(H0)
            setup_cost = self.env.transac_cost * abs(float(path["S"][0]) * float(H0))
            er = EpisodeResult(split=split, episode_idx=ep, times=self.env.times, path_data=path)
            er.add_step(
                action=H0,
                info={
                    "spot_t": float(path["S"][0]),
                    "spot_next": float(path["S"][0]),
                    "hedge": float(H0),
                    "trade_cost": float(setup_cost),
                    "liquidation_cost": 0.0,
                    "reward": -float(setup_cost),
                    "cost": float(setup_cost),
                },
                agent_info={"is_setup_step": True},
            )
            state = np.asarray(self.env._build_state(self.env.i, self.env.h_prev), dtype=np.float32)
            done = False
            while not done:
                # No policy call at terminal: env forces liquidation at T.
                is_terminal = (self.env.i == self.env.n_steps - 1)
                action = 0.0 if is_terminal else policy_fn(state)
                state, _, done, info = self.env.step(action)
                er.add_step(action=action, info=info)
            res.add_episode(er, type=split)
        return res

    def test(self):
        """Evaluate the trained agent over ``eval_episodes`` paths (no learning, no exploration)."""
        self.agent.set_eval_mode()
        return self._run_eval_episodes(
            policy_fn=lambda s: self.agent.act(s, eval_mode=True),
            split="eval_agent",
        )

    def test_benchmark(self):
        """Evaluate the analytical benchmark (BS / Bartlett / SABR practitioner) on the eval paths."""
        return self._run_eval_episodes(
            policy_fn=self.benchmark,
            split="eval_benchmark",
        )