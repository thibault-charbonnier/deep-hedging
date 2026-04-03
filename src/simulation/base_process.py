import numpy as np
from typing import Any
from abc import ABC, abstractmethod


class BaseProcess(ABC):
    """
    Abstract base class for stochastic processes simulation.
    Every subclass must implement the simulate_one_path method.
    
    The structure of the class is the following :
        - Input : a configuration dictionnary with
            * Base information for the simulation (maturity, number of steps, initial price, seed)
            * Model specific parameters (volatility, mean reversion speed, jump intensity, etc.)
        - Output : a dict of 1D arrays of length n_steps + 1 containing
            * The simulated paths of the underlying asset price
            * If applicable the simulated paths of the volatility or variance
            * If applicable the simulated jump count process
    """ 

    def __init__(self, simulation_cfg: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        simulation_cfg : dict
            Configuration dictionary containing the base information for the simulation and model-specific parameters.
        """
        self.cfg = simulation_cfg

        seed = int(simulation_cfg.get("seed", 42))
        self.rng = np.random.default_rng(seed)

        self.T = float(simulation_cfg["maturity"])
        self.n_steps = int(simulation_cfg["n_steps"])
        self.S0 = float(simulation_cfg["S0"])

        self.dt = self.T / self.n_steps
        self.sqrt_dt = np.sqrt(self.dt)
        self.times = np.linspace(0.0, self.T, self.n_steps + 1)

    @abstractmethod
    def simulate_one_path(self) -> dict[str, np.ndarray]:
        """
        Simulate one path of the selected process.

        Returns:
            dict[str, np.ndarray]: A dictionary containing the simulated paths. The keys depend on the process (e.g., "S", "sigma", "v", "jump_count").
        """
        raise NotImplementedError("Subclasses must implement the simulate_one_path method.")

    def simulate_paths(self, n_paths: int) -> dict[str, np.ndarray]:
        """
        Simulate multiple paths and stack outputs.

        Returns
        -------
        dict[str, np.ndarray]
            Each value has shape (n_paths, n_steps + 1).
        """
        first_path = self.simulate_one_path()
        output = {
            key: np.empty((n_paths, value.shape[0]), dtype=float)
            for key, value in first_path.items()
        }

        for key, value in first_path.items():
            output[key][0] = value

        for i in range(1, n_paths):
            path_i = self.simulate_one_path()
            for key, value in path_i.items():
                output[key][i] = value

        return output

    def _init_1d_path(self, x0: float) -> np.ndarray:
        """
        Initialize a 1D path array with the initial value.

        Parameters
        ----------
        x0 : float
            Initial value of the path.

        Returns
        -------
        np.ndarray
            A 1D array of shape (n_steps + 1,) initialized with x0 at the first position.
        """
        path = np.empty(self.n_steps + 1, dtype=float)
        path[0] = float(x0)
        return path

    def _correlated_normals(
        self,
        rng: np.random.Generator,
        rho: float,
        size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate two correlated standard normal series.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator instance.
        rho : float
            Correlation coefficient between the two series.
        size : int, optional
            Number of samples to generate (default is None, which uses self.n_steps).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two arrays of shape (size,) containing the correlated standard normal samples.
        """
        size = self.n_steps if size is None else size
        z1 = rng.standard_normal(size)
        z_indep = rng.standard_normal(size)
        z2 = rho * z1 + np.sqrt(1.0 - rho**2) * z_indep
        return z1, z2