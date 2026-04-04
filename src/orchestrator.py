import logging
from typing import Any
from .hedging_strategy import HedgingEnv, HedgingResult, EpisodeResult
from .utils.enums import AgentType, ProcessType, BenchmarkType


logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Module responsible for orchestrating the training and evaluation of the hedging agents.

    It implements the following functionalities:
        - Initialize the main modules (agent, environment).
        - Orchestrate the interaction between :
            * The simulated market data
            * The valuation logic
            * The hedging agent (ML logic)
            * The deterministic benchmark(s)
            * The environment (building the state, computing the reward)
    """
    def __init__(
        self, 
        config: dict[str, Any],
        process_type: ProcessType,
        agent_type: AgentType,
        benchmark_type: BenchmarkType
    ) -> None:
        """
        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary for the experiment runner.
        process_type : ProcessType
            The type of process to simulate the market data.
        agent_type : AgentType
            The type of hedging agent to train and evaluate.
        benchmark_type : BenchmarkType
            The type of benchmark strategy to evaluate.
        """
        logger.info("Initializing Orchestrator...")
        self.config = config
        self.env = HedgingEnv(config)

        self.process = process_type.value(config.get("simulation"))
        self.agent = agent_type.value(config.get("hedging_agent"))
        self.benchmark = benchmark_type.value(config)

        self.train_episodes = config.get("training_schedule").get("train_episodes")
        self.eval_episodes = config.get("training_schedule").get("eval_episodes")
        self.target_update_freq = int(config.get("training_schedule").get("update_frequency"))

        logger.info(f"Simulating {self.train_episodes} training paths and {self.eval_episodes} evaluation paths...")
        self.training_paths = self.process.simulate_paths(self.train_episodes)
        self.eval_paths = self.process.simulate_paths(self.eval_episodes)

    def train(self) -> tuple[list[float], list[float]]:
        """
        Train the agent on the simulated data.

        Returns
        -------
        tuple[list[float], list[float]]
            A tuple containing the list of training costs and the list of training losses.
        """
        logger.info("Starting training...")
        self.agent.set_train_mode()

        res = HedgingResult()
        for episode in range(self.train_episodes):
            logger.info(f"\tAgent training episode {episode + 1}/{self.train_episodes}...")

            path_data = self.training_paths["S"][episode]
            state = self.env.setup_env(path_data=path_data)
            done = False

            episode_res = EpisodeResult(split="train", episode_idx=episode, times=self.env.times,
                                        path_data={key: value[episode] for key, value in self.training_paths.items()})
            
            while not done:
                action = self.agent.act(state, eval_mode=False)

                next_state, reward, done, info = self.env.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                
                loss = self.agent.learn()
                state = next_state

                episode_res.add_step(action=action, info=info, loss=loss)

            res.add_episode(episode_res, type="train")

        return res
                

    def test(self) -> list[float]:
        """
        Evaluate the trained agent on new simulated data.

        Returns
        -------
        list[float]
            A list containing the costs obtained by the agent on the evaluation episodes.
        """
        logger.info("Starting evaluation...")

        self.agent.set_eval_mode()

        res = HedgingResult()
        for episode in range(self.eval_episodes):
            logger.info(f"\tAgent evaluation episode {episode + 1}/{self.eval_episodes}...")
            
            path_data = self.eval_paths["S"][episode]
            state = self.env.setup_env(path_data=path_data)
            done = False
            
            episode_res = EpisodeResult(split="eval_agent", episode_idx=episode, times=self.env.times,
                                        path_data={key: value[episode] for key, value in self.eval_paths.items()})
            
            while not done:
                action = self.agent.act(state, eval_mode=True)
                state, _, done, info = self.env.step(action)
                episode_res.add_step(action=action, info=info)

            res.add_episode(episode_res, type="eval_agent")

        return res

    def test_benchmark(self) -> HedgingResult:
        """
        Evaluate the benchmark strategy on new simulated data.

        Returns
        -------
        HedgingResult
            A HedgingResult object containing the results from the benchmark evaluation.
        """
        logger.info("Starting benchmark evaluation...")

        res = HedgingResult()
        for episode in range(self.eval_episodes):
            logger.info(f"\tBenchmark evaluation episode {episode + 1}/{self.eval_episodes}...")
            
            path_data = self.eval_paths["S"][episode]
            state = self.env.setup_env(path_data=path_data)
            done = False

            episode_res = EpisodeResult(split="eval_benchmark", episode_idx=episode, times=self.env.times,
                                        path_data={key: value[episode] for key, value in self.eval_paths.items()})
            
            while not done:
                action = self.benchmark(state)
                state, _, done, info = self.env.step(action)
                episode_res.add_step(action=action, info=info)
            
            res.add_episode(episode_res, type="eval_benchmark")

        return res