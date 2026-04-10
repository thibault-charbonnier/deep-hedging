from __future__ import annotations

import numpy as np


class SABRProcess:
    """
    Simplified SABR with beta = 1:
        dS = mu S dt + sigma S dW1
        d sigma = nu sigma dW2
    with corr(dW1, dW2) = rho.
    """

    def __init__(self, simulation_cfg: dict) -> None:
        self.n_steps = int(simulation_cfg["n_steps"])
        self.maturity = float(simulation_cfg["maturity"])
        self.S0 = float(simulation_cfg["S0"])
        self.mu = float(simulation_cfg["sabr"].get("mu", 0.0))
        self.sigma0 = float(simulation_cfg["sabr"].get("sigma0", 0.2))
        self.nu = float(simulation_cfg["sabr"].get("nu", 0.3))
        self.rho = float(simulation_cfg["sabr"].get("rho", -0.5))
        self.dt = self.maturity / self.n_steps
        self.sqrt_dt = np.sqrt(self.dt)

    def simulate_paths(self, n_paths: int) -> dict[str, np.ndarray]:
        n_paths = int(n_paths)
        z1 = np.random.normal(size=(n_paths, self.n_steps))
        z2_indep = np.random.normal(size=(n_paths, self.n_steps))
        z2 = self.rho * z1 + np.sqrt(max(1.0 - self.rho ** 2, 0.0)) * z2_indep

        S = np.empty((n_paths, self.n_steps + 1), dtype=float)
        sigma = np.empty((n_paths, self.n_steps + 1), dtype=float)
        S[:, 0] = self.S0
        sigma[:, 0] = self.sigma0

        for t in range(self.n_steps):
            sigma_t = np.maximum(sigma[:, t], 1e-8)
            sigma[:, t + 1] = sigma_t * np.exp((-0.5 * self.nu ** 2) * self.dt + self.nu * self.sqrt_dt * z2[:, t])
            S[:, t + 1] = S[:, t] * np.exp((self.mu - 0.5 * sigma_t ** 2) * self.dt + sigma_t * self.sqrt_dt * z1[:, t])

        return {"S": S, "sigma": sigma}
