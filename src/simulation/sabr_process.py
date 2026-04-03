import numpy as np
from typing import Any
from .base_process import BaseProcess


class SABRProcess(BaseProcess):
    """
    Implementation of the BaseProcess abstract class for a simplified SABR stochastic volatility process (beta = 1).

    We simulate the discretized version of the simplified SABR SDE:
        dS_t     = mu * S_t * dt + sigma_t * S_t * dW1_t
        dσ_t     = nu * sigma_t * dW2_t
        corr(dW1_t, dW2_t) = rho

    Model-specific parameters:
        - mu : drift of the process
        - sigma_0 : initial volatility
        - nu : volatility of volatility
        - rho : correlation between the Brownian motions
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
        self.sigma0 = float(simulation_cfg["sigma0"])
        self.nu = float(simulation_cfg["nu"])
        self.rho = float(simulation_cfg["rho"])

    def simulate_one_path(self) -> dict[str, np.ndarray]:
        """
        Simulate one path of the simplified SABR process.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary containing:
            - "S": the simulated price path of the SABR process
            - "sigma": the simulated volatility path of the SABR process
        """
        z1, z2 = self._correlated_normals(self.rng, self.rho)

        S = self._init_1d_path(self.S0)
        sigma = self._init_1d_path(self.sigma0)

        for i in range(self.n_steps):
            sigma_i = max(sigma[i], 1e-12)

            sigma[i + 1] = sigma_i * np.exp(
                -0.5 * self.nu**2 * self.dt
                + self.nu * self.sqrt_dt * z2[i]
            )
            
            S[i + 1] = S[i] * np.exp(
                (self.mu - 0.5 * sigma_i**2) * self.dt
                + sigma_i * self.sqrt_dt * z1[i]
            )

        return {
            "S": S,
            "sigma": sigma,
        }