"""
Shared RL building blocks.

Includes:
- MLP / CriticMLP network helpers
- **Prioritized** replay buffer (Schaul et al., 2015)
- Hard target-network updates
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
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


# ── Transition batch ─────────────────────────────────────────────────

@dataclass
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor          # importance-sampling weights (1.0 for uniform)
    indices: np.ndarray | None     # buffer indices (None for uniform)


# ── Prioritized Experience Replay (Schaul et al., 2015) ──────────────
# Used by Cao et al. (2021) Section 4: "we also implement the
# prioritized experience replay method"

class SumTree:
    """Binary sum-tree for O(log n) proportional sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        while left < len(self.tree):
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
            left = 2 * idx + 1
            right = left + 1

        return idx

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Proportional prioritization (Schaul et al., 2015).
    
    Priorities are based on TD error: transitions with higher error
    are replayed more often, improving data efficiency.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1,
                 alpha: float = 0.6, beta_start: float = 0.4,
                 beta_frames: int = 100_000, epsilon: float = 1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha           # prioritization exponent
        self.beta = beta_start       # IS correction exponent (annealed → 1)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.frame = 0

    def __len__(self) -> int:
        return self.tree.n_entries

    def push(self, state, action, reward, next_state, done) -> None:
        transition = (
            np.asarray(state, dtype=np.float32).copy(),
            float(action), float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            float(done),
        )
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        indices = []
        priorities = []
        transitions = []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, prio, data = self.tree.get(s)
            if data is None:
                # Edge case: sample again
                s = random.uniform(0, self.tree.total)
                idx, prio, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(prio)
            transitions.append(data)

        priorities_arr = np.array(priorities, dtype=np.float64) + self.epsilon
        sampling_probs = priorities_arr / self.tree.total
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*transitions)

        return TransitionBatch(
            states=torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device).unsqueeze(-1),
            rewards=torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device).unsqueeze(-1),
            next_states=torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device),
            dones=torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=device).unsqueeze(-1),
            weights=torch.as_tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1),
            indices=np.array(indices, dtype=np.int64),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            p = (float(prio) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, p)
            self.tree.update(int(idx), p)


# ── Target network updates ──────────────────────────────────────────


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
