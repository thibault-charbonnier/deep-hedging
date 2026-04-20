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

DEVICE = torch.device("cpu")


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
        return self.net(torch.cat([state, action], dim=-1))



# ── Target network updates ──────────────────────────────────────────


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """One-shot copy θ' ← θ. Used for initial target network synchronisation."""
    target.load_state_dict(source.state_dict())


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: θ' ← τ·θ + (1-τ)·θ'  (Lillicrap et al. 2016, Section 3).

    Applied at every learn step with small tau (typically 0.001-0.01)
    to provide a slowly-tracking target that stabilises critic training.
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

