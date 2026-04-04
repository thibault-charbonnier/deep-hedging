import numpy as np
from typing import Any
from ..valuation import BSValuation


class HedgingEnv:
    """
    Module responsible for handling the "hedging environment" in which the agent operates.

    This includes :
        - Provide to the agent the state (previous hedge, spot, time to maturity) at each step.
        - Compute the reward of a given hedge action (through the given valuation logic).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        config : dict[str, Any]
            Global configuration dictionary.
        """
        self.transac_cost = config.get("hedging_env").get("transaction_cost")
        self.position_sign = config.get("hedging_env").get("position_sign")

        self.valuation_sigma = config.get("simulation").get("gbm").get("sigma")
        self.maturity = config.get("simulation").get("maturity")

        self.valuation_engine = BSValuation(
            strike=config.get("derivative").get("strike"),
            maturity=self.maturity,
            rate=config.get("derivative").get("rf_rate"),
            dividend=config.get("derivative").get("div_rate"),
            option_type="call" if self.position_sign > 0 else "put"
        )

    def setup_env(self, path_data: np.ndarray) -> np.ndarray:
        """
        Setup the environment for a new episode.

        Parameters
        ----------
        path_data : np.ndarray
            The simulated path data for the episode.

        Returns
        -------
        np.ndarray
            The initial state of the environment.
        """
        self.path_data = path_data
        self.n_steps = len(path_data) - 1
        self.times = np.linspace(0, self.maturity, len(path_data) + 1)

        self.i = 0
        self.v_prev, self.h_prev = self._derivative_value(0)

        self.episode_reward = 0.0
        self.episode_cost = 0.0

        return self._build_state(self.i, self.h_prev)
    
    def step(self, hedge: float) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        """
        Execute a step in the environment given a hedge action from the agent.

        Parameters
        ----------
        hedge : float
            The hedge action taken by the agent at the current step.
        
        Returns
        -------
        tuple[np.ndarray, float, bool, dict[str, float]]
            - The next state of the environment after executing the action.
            - The reward obtained from executing the action.
            - A boolean indicating if the episode is done.
            - An info dictionary with additional details about the step to debug or analyse.
        """
        i = self.i
        spot_t = self.path_data[i]
        spot_next = self.path_data[i + 1]

        trade_cost = self.transac_cost * spot_t * abs(hedge - self.h_prev)

        v_next = self._derivative_value(i + 1)[0]

        reward = (v_next - self.v_prev) + hedge * (spot_next - spot_t) - trade_cost

        done = i == self.n_steps - 1

        liquidation_cost = 0.0
        if done:
            liquidation_cost = self.transac_cost * spot_next * abs(hedge)
            reward -= liquidation_cost

        self.episode_reward += reward
        self.episode_cost += -reward

        self.i += 1
        self.h_prev = 0.0 if done else hedge
        self.v_prev = v_next

        next_state = self._build_state(self.i, self.h_prev)

        info = {
            "spot_t": spot_t,
            "spot_next": spot_next,
            "hedge": hedge,
            "trade_cost": trade_cost,
            "liquidation_cost": liquidation_cost,
            "reward": reward,
            "cost": -reward,
            "episode_reward": self.episode_reward,
            "episode_cost": self.episode_cost,
        }

        return next_state, reward, done, info

    def _build_state(self, process_step: int, hedge_pos: float) -> np.ndarray:
        """
        Build the state vector for the agent at a given process step and hedge position.

        The state vector contains :
            - The current hedge position of the agent.
            - The current spot price of the underlying.
            - The time to maturity of the derivative (T - t).

        Parameters
        ----------
        process_step : int
            The index of the process step for which to build the state.
        hedge_pos : float
            The current hedge position of the agent.

        Returns
        -------
        np.ndarray
            The state vector for the agent.
        """
        return np.asarray([hedge_pos, self.path_data[process_step], max(self.maturity - self.times[process_step], 0.0)], dtype=float)

    def _derivative_value(self, process_step: int) -> tuple[float, float]:
        """
        Call the valuation engine to compute the derivative value at a given step.

        Parameters
        ----------
        process_step : tuple[float, float]
            The index of the process step for which to compute the derivative value.
        """
        price, delta = self.valuation_engine.price_and_delta(
            spot=self.path_data[process_step],
            t=self.times[process_step],
            sigma=self.valuation_sigma,
        )
        return self.position_sign * float(price), self.position_sign * float(delta)