import numpy as np
from typing import Any
from .base_process import BaseProcess


class SVJProcess(BaseProcess):
    """
    Implementation of the BaseProcess abstract class for a Stochastic Volatility with Jumps (SVJ) process.

    We simulate the discretized version of this SDE:
        dS_t / S_t = (mu - lambda * kJ) dt + sqrt(v_t) dW1_t + dJ_t
        dv_t = kappa * (theta - v_t) dt + xi * sqrt(v_t) dW2_t
        corr(dW1_t, dW2_t) = rho
    Where Jumps are modeled as a compound Poisson process:
        N_t ~ Poisson(lambda * dt)
        log-jump sizes ~ Normal(jump_mean, jump_std)

    Model-specific parameters:
        - mu : drift of the process
        - v0 : initial variance
        - kappa : mean reversion speed of the variance
        - theta : long-term mean of the variance
        - xi : volatility of volatility
        - rho : correlation between the Brownian motions
        - jump_intensity : intensity (lambda) of the Poisson jump process
        - jump_mean : mean of the normal distribution for log-jump sizes
        - jump_std : standard deviation of the normal distribution for log-jump sizes
    """

    def __init__(self, simulation_cfg: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        simulation_cfg : dict
            Configuration dictionary containing the base information for the simulation and model-specific parameters.
        """
        super().__init__(simulation_cfg)

        self.mu = float(simulation_cfg["mu"])
        self.v0 = float(simulation_cfg["v0"])
        self.kappa = float(simulation_cfg["kappa"])
        self.theta = float(simulation_cfg["theta"])
        self.xi = float(simulation_cfg["xi"])
        self.rho = float(simulation_cfg["rho"])

        self.jump_intensity = float(simulation_cfg["jump_intensity"])
        self.jump_mean = float(simulation_cfg["jump_mean"])
        self.jump_std = float(simulation_cfg["jump_std"])

        self.kJ = np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1.0

    def simulate_one_path(self) -> dict[str, np.ndarray]:
        """
        Simulate one path of the SVJ process.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary containing:
            - "S": the simulated price path of the SVJ process
            - "variance": the simulated variance path of the SVJ process
            - "jump_count": the simulated jump count path of the SVJ process
        """
        z1, z2 = self._correlated_normals(self.rng, self.rho)

        S = self._init_1d_path(self.S0)
        variance = self._init_1d_path(self.v0)
        jump_count = self._init_1d_path(0.0)

        for i in range(self.n_steps):
            v_i = max(variance[i], 0.0)

            variance[i + 1] = max(
                variance[i]
                + self.kappa * (self.theta - v_i) * self.dt
                + self.xi * np.sqrt(v_i) * self.sqrt_dt * z2[i],
                1e-12,
            )

            n_jumps = self.rng.poisson(self.jump_intensity * self.dt)
            jump_count[i + 1] = n_jumps

            if n_jumps > 0:
                jump_sum = self.rng.normal(
                    loc=self.jump_mean,
                    scale=self.jump_std,
                    size=n_jumps,
                ).sum()
            else:
                jump_sum = 0.0

            S[i + 1] = S[i] * np.exp(
                (self.mu - self.jump_intensity * self.kJ - 0.5 * v_i) * self.dt
                + np.sqrt(v_i) * self.sqrt_dt * z1[i]
                + jump_sum
            )

        return {
            "S": S,
            "variance": variance,
            "jump_count": jump_count,
        }