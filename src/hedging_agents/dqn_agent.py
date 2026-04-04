from typing import Any
from .abstract_agent import AbstractHedgingAgent


class DQNHedgingAgent(AbstractHedgingAgent):
    
    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        super().__init__(agent_cfg)

    def act(self, state: Any, eval_mode: bool = False) -> float:
        ...

    def learn(self, state: Any, action: float, reward: float, next_state: Any, done: bool) -> None:
        ...

    def save(self, path: str) -> None:
        ...
    
    def load(self, path: str) -> None:
        ...

    def set_eval_mode(self):
        return super().set_eval_mode()
    
    def set_train_mode(self):
        return super().set_train_mode()

    def store_transition(self, state, action, reward, next_state, done):
        return super().store_transition(state, action, reward, next_state, done)