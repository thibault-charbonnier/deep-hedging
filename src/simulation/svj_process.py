from __future__ import annotations

import numpy as np


class SVJProcess:
    """
    Stochastic volatility with jumps.

    Variance follows a full-truncation Heston-like Euler scheme.
    Jumps act on log-returns with Poisson arrivals and Gaussian jump sizes.
    """

    def __init__(self, simulation_cfg: dict) -> None:
        self.n_steps = int(simulation_cfg["n_steps"])
        self.maturity = float(simulation_cfg["maturity"])
        self.S0 = float(simulation_cfg["S0"])
        params = simulation_cfg["svj"]
        self.mu = float(params.get("mu", 0.0))
        self.v0 = float(params.get("v0", 0.04))
        self.kappa = float(params.get("kappa", 1.0))
        self.theta = float(params.get("theta", 0.04))
        self.xi = float(params.get("xi", 0.5))
        self.rho = float(params.get("rho", -0.5))
        self.jump_intensity = float(params.get("jump_intensity", 1.0))
        self.jump_mean = float(params.get("jump_mean", -0.02))
        self.jump_std = float(params.get("jump_std", 0.05))
        self.dt = self.maturity / self.n_steps
        self.sqrt_dt = np.sqrt(self.dt)

    def simulate_paths(self, n_paths: int) -> dict[str, np.ndarray]:
        """Simulate ``n_paths`` (S, variance) trajectories under Heston + jumps.
        """
        n_paths = int(n_paths)
        z1 = np.random.normal(size=(n_paths, self.n_steps))
        z2_indep = np.random.normal(size=(n_paths, self.n_steps))
        z2 = self.rho * z1 + np.sqrt(max(1.0 - self.rho ** 2, 0.0)) * z2_indep

        N = np.random.poisson(self.jump_intensity * self.dt, size=(n_paths, self.n_steps))
        J = np.random.normal(self.jump_mean, self.jump_std, size=(n_paths, self.n_steps))
        jump_term = N * J

        S = np.empty((n_paths, self.n_steps + 1), dtype=float)
        variance = np.empty((n_paths, self.n_steps + 1), dtype=float)
        S[:, 0] = self.S0
        variance[:, 0] = self.v0

        for t in range(self.n_steps):
            v_t = np.maximum(variance[:, t], 0.0)
            sqrt_v_t = np.sqrt(np.maximum(v_t, 1e-10))
            variance[:, t + 1] = np.maximum(
                variance[:, t] + self.kappa * (self.theta - v_t) * self.dt + self.xi * sqrt_v_t * self.sqrt_dt * z2[:, t],
                1e-10,
            )
            S[:, t + 1] = S[:, t] * np.exp(
                (self.mu - 0.5 * v_t) * self.dt + sqrt_v_t * self.sqrt_dt * z1[:, t] + jump_term[:, t]
            )

        return {"S": S, "variance": variance}
