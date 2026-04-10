from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int] = (128, 128),
        output_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


@dataclass
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.buffer: deque[tuple[np.ndarray, float, float, np.ndarray, float]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                float(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                float(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return TransitionBatch(
            states=torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device).unsqueeze(-1),
            rewards=torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device).unsqueeze(-1),
            next_states=torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device),
            dones=torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=device).unsqueeze(-1),
        )


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
