"""
Shared RL building blocks.

Contents:
- ``DEVICE`` — single source of truth for the torch device.
- ``MLP`` / ``CriticMLP`` / ``QuantileCriticMLP`` / ``DeterministicActor``
  network helpers.
- ``hard_update`` for periodic target-network copies.
- ``PERActorCriticAgent`` — base class for DDPG-style agents with
  Prioritized Experience Replay and ε-greedy exploration.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from cpprb import PrioritizedReplayBuffer

from .abstract_agent import AbstractHedgingAgent


# Single source of truth for the torch device used by every network in the
# project. For this workload (small step-wise tensors), CPU is faster than the
# MPS transfer overhead.
DEVICE = torch.device("cpu")


# ── Networks ─────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Standard fully-connected ReLU MLP with an optional output activation."""

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
        """Run the stacked Linear/ReLU layers on ``x``."""
        return self.net(x)


class CriticMLP(nn.Module):
    """Scalar critic ``Q(s, a)`` implemented as an MLP on the concatenated (s, a) vector."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Iterable[int] = (128, 128)) -> None:
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return ``Q(s, a)``; ``action`` is auto-reshaped to 2D if needed."""
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
        """Return ``[B, N]`` tensor of predicted quantile values for each input pair."""
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        return self.net(torch.cat([state, action], dim=-1))  # [B, N]


class DeterministicActor(nn.Module):
    """Deterministic policy network: tanh output rescaled to ``[action_low, action_high]``."""

    def __init__(self, state_dim: int, hidden_dims: Iterable[int],
                 action_low: float, action_high: float) -> None:
        super().__init__()
        self.backbone = MLP(state_dim, 1, hidden_dims, output_activation=nn.Tanh())
        self.register_buffer("a_lo", torch.tensor([action_low], dtype=torch.float32))
        self.register_buffer("a_hi", torch.tensor([action_high], dtype=torch.float32))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action in ``[a_lo, a_hi]`` for state ``s``."""
        mid = 0.5 * (self.a_hi + self.a_lo)
        half = 0.5 * (self.a_hi - self.a_lo)
        return mid + half * self.backbone(s)


# ── Target network updates ──────────────────────────────────────────


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy every parameter from ``source`` into ``target`` (periodic hard target update)."""
    target.load_state_dict(source.state_dict())


# ── PER + ε-greedy base class ──────────────────────────────────────


class PERActorCriticAgent(AbstractHedgingAgent):
    """Base class for DDPG-style agents (deterministic actor + PER buffer + ε-greedy).

    Handles: config parsing, actor networks, PER replay buffer,
    epsilon-greedy exploration, target-network periodic hard copies,
    and the train/eval mode toggle. Subclasses only need to:

      - build their critic network(s) and optimiser(s) in ``__init__``
      - override ``_networks``, ``_critic_nets``, ``_sync_targets``
      - implement ``learn``
    """

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        """Parse common hyperparameters and build actor, actor target, and PER buffer."""
        super().__init__(agent_cfg)
        self.device = DEVICE
        self.state_dim = int(agent_cfg.get("state_dim", 4))
        self.hidden_dims = tuple(agent_cfg.get("hidden_dims", [128, 128]))
        self.lr_actor = float(agent_cfg.get("actor_learning_rate", 1e-4))
        self.lr_critic = float(agent_cfg.get("critic_learning_rate", 1e-3))
        self.batch_size = int(agent_cfg.get("learning_batch_size", 128))
        self.buffer_size = int(agent_cfg.get("replay_capacity", 100_000))
        self.min_buffer = int(agent_cfg.get("min_buffer_size", self.batch_size))
        self.action_low = float(agent_cfg.get("action_low", 0.0))
        self.action_high = float(agent_cfg.get("action_high", 1.0))

        # ε-greedy exploration schedule.
        self.epsilon = float(agent_cfg.get("exploration_rate_start", 1.0))
        self.epsilon_min = float(agent_cfg.get("exploration_rate_end", 0.05))
        self.epsilon_decay = float(agent_cfg.get("exploration_rate_decay", 0.995))

        # Periodic hard target update.
        self.target_update_freq = int(agent_cfg.get("target_update_freq", 100))

        # Actor + target.
        self.actor = DeterministicActor(
            self.state_dim, self.hidden_dims, self.action_low, self.action_high
        ).to(self.device)
        self.actor_target = DeterministicActor(
            self.state_dim, self.hidden_dims, self.action_low, self.action_high
        ).to(self.device)
        hard_update(self.actor_target, self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Prioritized Experience Replay with IS-beta annealing.
        self.per_alpha = float(agent_cfg.get("per_alpha", 0.6))
        self.per_beta_start = float(agent_cfg.get("per_beta_start", 0.4))
        self.per_beta_frames = int(agent_cfg.get("per_beta_frames", 100_000))
        self.per_eps = float(agent_cfg.get("per_eps", 1e-6))
        self.per_frame = 0
        env_dict = {
            "obs": {"shape": (self.state_dim,)},
            "act": {"shape": (1,)},
            "rew": {},
            "next_obs": {"shape": (self.state_dim,)},
            "done": {},
        }
        self.replay_buffer = PrioritizedReplayBuffer(
            size=self.buffer_size,
            env_dict=env_dict,
            alpha=self.per_alpha,
            beta=self.per_beta_start,
            eps=self.per_eps,
        )

        self.train_mode_enabled = True
        self.learn_steps = 0

    # ── Helpers ──────────────────────────────────────────────────────

    def _st(self, state) -> torch.Tensor:
        """Convert a numpy state to a ``[1, state_dim]`` tensor on the agent device."""
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    def _replay_size(self) -> int:
        """Return the current number of transitions stored in the replay buffer."""
        return int(self.replay_buffer.get_stored_size())

    def _update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update the PER priorities for the given sample indices."""
        self.replay_buffer.update_priorities(indices, priorities)

    def _sample_batch_tensors(self) -> dict[str, Any]:
        """Sample a PER batch, advance the IS-beta schedule, and return tensors on the device."""
        self.per_frame += 1
        beta = min(
            1.0,
            self.per_beta_start + self.per_frame * (1.0 - self.per_beta_start) / self.per_beta_frames,
        )
        batch = self.replay_buffer.sample(self.batch_size, beta=beta)
        return {
            "states": torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device).reshape(-1, 1),
            "rewards": torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).reshape(-1),
            "next_states": torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).reshape(-1),
            "weights": torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device).reshape(-1),
            "indexes": np.asarray(batch["indexes"], dtype=np.int64).reshape(-1),
        }

    def _freeze_critics(self) -> None:
        """Disable gradient flow through every critic network (used during actor step)."""
        for net in self._critic_nets():
            for p in net.parameters():
                p.requires_grad_(False)

    def _unfreeze_critics(self) -> None:
        """Re-enable gradient flow through every critic network after the actor step."""
        for net in self._critic_nets():
            for p in net.parameters():
                p.requires_grad_(True)

    def _post_learn_step(self) -> None:
        """Bump counter, copy online into targets at target_update_freq, and decay epsilon."""
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self._sync_targets()
        if self.train_mode_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Subclass hooks ───────────────────────────────────────────────

    def _networks(self) -> list[nn.Module]:
        """Return every network (actor/target + critics/targets) — used by train/eval toggles."""
        raise NotImplementedError

    def _critic_nets(self) -> list[nn.Module]:
        """Return online critic networks that must be frozen during the actor update."""
        raise NotImplementedError

    def _sync_targets(self) -> None:
        """Hard-copy every online network into its paired target network."""
        raise NotImplementedError

    # ── Public API ───────────────────────────────────────────────────

    def act(self, state, eval_mode: bool = False) -> float:
        """Return the hedge action for ``state`` (epsilon-greedy during training)."""
        if (not eval_mode) and self.train_mode_enabled:
            if np.random.rand() < self.epsilon:
                return float(np.random.uniform(self.action_low, self.action_high))
        with torch.no_grad():
            a = self.actor(self._st(state)).squeeze(0).cpu().numpy()[0]
        return float(np.clip(a, self.action_low, self.action_high))

    def store_transition(self, state, action, reward, next_state, done) -> None:
        """Push the transition into the prioritized replay buffer."""
        self.replay_buffer.add(
            obs=np.asarray(state, dtype=np.float32),
            act=np.asarray([float(action)], dtype=np.float32),
            rew=float(reward),
            next_obs=np.asarray(next_state, dtype=np.float32),
            done=float(done),
        )

    def set_train_mode(self) -> None:
        """Enable exploration and switch every network to ``train()`` mode."""
        self.train_mode_enabled = True
        for m in self._networks():
            m.train()

    def set_eval_mode(self) -> None:
        """Disable exploration and switch every network to ``eval()`` mode."""
        self.train_mode_enabled = False
        for m in self._networks():
            m.eval()
