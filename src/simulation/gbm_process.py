
import numpy as np
from typing import Any
from .base_process import BaseProcess


class GBMProcess(BaseProcess):
    """
    Implementation of the BaseProcess abstract class for the Geometric Brownian Motion (GBM) process.

    We simulate the discretized version of the GBM SDE:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t

    Model-specific parameters:
        - mu : drift of the process
        - sigma : (constant) volatility of the process
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
        self.sigma = float(simulation_cfg["sigma"])

    def _simulate_one_path(self) -> dict[str, np.ndarray]:
        """
        Simulate one path of the GBM process.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary containing:
            - "S": the simulated price path of the GBM process
        """
        z = self.rng.standard_normal(self.n_steps)

        log_returns = (
            (self.mu - 0.5 * self.sigma**2) * self.dt
            + self.sigma * self.sqrt_dt * z
        )

        S = self._init_1d_path(self.S0)
        S[1:] = self.S0 * np.exp(np.cumsum(log_returns))

        return {"S": S}