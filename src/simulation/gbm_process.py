from __future__ import annotations

import numpy as np


class GBMProcess:
    def __init__(self, simulation_cfg: dict) -> None:
        self.n_steps = int(simulation_cfg["n_steps"])
        self.maturity = float(simulation_cfg["maturity"])
        self.S0 = float(simulation_cfg["S0"])
        self.mu = float(simulation_cfg["gbm"].get("mu", 0.0))
        self.sigma = float(simulation_cfg["gbm"]["sigma"])
        self.dt = self.maturity / self.n_steps
        self.sqrt_dt = np.sqrt(self.dt)

    def simulate_paths(self, n_paths: int) -> dict[str, np.ndarray]:
        n_paths = int(n_paths)
        z = np.random.normal(size=(n_paths, self.n_steps))
        log_returns = (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * self.sqrt_dt * z
        log_prices = np.cumsum(log_returns, axis=1)
        S = np.empty((n_paths, self.n_steps + 1), dtype=float)
        S[:, 0] = self.S0
        S[:, 1:] = self.S0 * np.exp(log_prices)
        return {"S": S}
