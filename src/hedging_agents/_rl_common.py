"""
Shared RL building blocks.

Includes:
- MLP / CriticMLP network helpers
- Hard target-network updates
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def get_device() -> torch.device:
    # For this project workload (small step-wise tensors), CPU is faster than MPS transfer overhead.
    return torch.device("cpu")


# ── Networks ─────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: Iterable[int] = (128, 128),
                 output_activation: nn.Module | None = None) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], output_dim))
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Iterable[int] = (128, 128)) -> None:
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        return self.net(torch.cat([state, action], dim=-1))


class QuantileCriticMLP(nn.Module):
    """Critic that predicts N quantiles of the return distribution (QR-DQN style)."""

    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int,
                 hidden_dims: Iterable[int] = (128, 128)) -> None:
        super().__init__()
        self.n_quantiles = int(n_quantiles)
        self.net = MLP(state_dim + action_dim, self.n_quantiles, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        return self.net(torch.cat([state, action], dim=-1))  # [B, N]



# ── Target network updates ──────────────────────────────────────────


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
