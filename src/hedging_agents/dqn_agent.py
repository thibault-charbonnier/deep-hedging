"""
DQN with dual Q-functions for risk-aware hedging.

Follows Cao et al. (2021) Section 3.4:
  - Q1(s, a) estimates E[C_t | s, a]       (expected future cost)
  - Q2(s, a) estimates E[C_t^2 | s, a]     (second non-central moment)
  - Greedy action = argmin_a  F(s, a)
    where  F(s, a) = Q1(s, a) + λ √( Q2(s, a) − Q1(s, a)² )

Update rules (from the paper):
  Q1 target:  cost + γ · Q1_target(s', a*)
  Q2 target:  cost² + γ² · Q2_target(s', a*) + 2γ · cost · Q1_target(s', a*)
  where  a* = argmin_a F(s', a)   using the ONLINE networks
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .abstract_agent import AbstractHedgingAgent
from ._rl_common import MLP, ReplayBuffer, get_device, hard_update


class DQNHedgingAgent(AbstractHedgingAgent):
    """
    Discrete-action DQN with dual Q-functions (Q1, Q2) to minimise
    Y = E[C] + λ·σ(C)  as in Cao et al. (2021).
    """

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        super().__init__(agent_cfg)
        self.device = get_device()
        self.state_dim = int(agent_cfg.get("state_dim", 4))
        self.hidden_dims = tuple(agent_cfg.get("hidden_dims", [128, 128]))
        self.lr = float(agent_cfg.get("learning_rate", 5e-4))
        self.batch_size = int(agent_cfg.get("learning_batch_size", 128))
        self.buffer_size = int(agent_cfg.get("replay_capacity", 100_000))
        self.min_buffer_size = int(agent_cfg.get("min_buffer_size", self.batch_size))
        self.target_update_freq = int(agent_cfg.get("target_update_freq", 100))
        self.grad_clip = float(agent_cfg.get("grad_clip", 1.0))
        self.risk_lambda = float(agent_cfg.get("risk_lambda", 1.5))

        self.epsilon = float(agent_cfg.get("exploration_rate_start", 1.0))
        self.epsilon_min = float(agent_cfg.get("exploration_rate_end", 0.05))
        self.epsilon_decay = float(agent_cfg.get("exploration_rate_decay", 0.995))

        action_low = float(agent_cfg.get("action_low", 0.0))
        action_high = float(agent_cfg.get("action_high", 1.0))
        action_grid_size = int(agent_cfg.get("action_grid_size", 21))
        self.action_grid = np.linspace(action_low, action_high, action_grid_size, dtype=np.float32)
        n_actions = len(self.action_grid)

        # Two Q-networks + their targets
        self.q1_net = MLP(self.state_dim, n_actions, self.hidden_dims).to(self.device)
        self.q2_net = MLP(self.state_dim, n_actions, self.hidden_dims).to(self.device)
        self.q1_target = MLP(self.state_dim, n_actions, self.hidden_dims).to(self.device)
        self.q2_target = MLP(self.state_dim, n_actions, self.hidden_dims).to(self.device)
        hard_update(self.q1_target, self.q1_net)
        hard_update(self.q2_target, self.q2_net)

        params = list(self.q1_net.parameters()) + list(self.q2_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.state_dim)
        self.train_mode_enabled = True
        self.learn_steps = 0

    # ── helpers ──────────────────────────────────────────────────────

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(state, dtype=np.float32),
                               device=self.device).unsqueeze(0)

    def _action_to_index(self, action: float) -> int:
        return int(np.argmin(np.abs(self.action_grid - float(action))))

    def _index_to_action(self, index: int) -> float:
        return float(self.action_grid[int(index)])

    def _F_values(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Compute F(s, a) = Q1 + λ √(Q2 − Q1²)  for all actions."""
        variance = torch.clamp(q2 - q1.pow(2), min=1e-8)
        return q1 + self.risk_lambda * torch.sqrt(variance)

    def _greedy_action_idx(self, state_t: torch.Tensor) -> int:
        """argmin_a F(s, a)  using online networks."""
        q1 = self.q1_net(state_t)
        q2 = self.q2_net(state_t)
        f_vals = self._F_values(q1, q2)
        return int(torch.argmin(f_vals, dim=1).item())   # argMIN (cost)

    # ── public API ──────────────────────────────────────────────────

    def act(self, state: Any, eval_mode: bool = False) -> float:
        if (not eval_mode) and self.train_mode_enabled and np.random.rand() < self.epsilon:
            return float(np.random.choice(self.action_grid))

        with torch.no_grad():
            idx = self._greedy_action_idx(
                self._state_tensor(np.asarray(state, dtype=np.float32))
            )
        return self._index_to_action(idx)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.replay_buffer) < self.min_buffer_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        cost = -batch.rewards                       # reward = -cost

        action_indices = torch.as_tensor(
            [self._action_to_index(a)
             for a in batch.actions.squeeze(-1).detach().cpu().numpy()],
            dtype=torch.long, device=self.device,
        ).unsqueeze(-1)

        # Current estimates
        q1_all = self.q1_net(batch.states)
        q2_all = self.q2_net(batch.states)
        q1_sa = q1_all.gather(1, action_indices)
        q2_sa = q2_all.gather(1, action_indices)

        with torch.no_grad():
            not_done = 1.0 - batch.dones

            # Greedy next action via TARGET F-values (standard DQN)
            next_q1_tgt = self.q1_target(batch.next_states)
            next_q2_tgt = self.q2_target(batch.next_states)
            next_f = self._F_values(next_q1_tgt, next_q2_tgt)
            next_greedy = next_f.argmin(dim=1, keepdim=True)    # argMIN

            # Evaluate with same TARGET networks
            next_q1_val = next_q1_tgt.gather(1, next_greedy)
            next_q2_val = next_q2_tgt.gather(1, next_greedy)

            # Paper update rules
            target_q1 = cost + self.gamma * not_done * next_q1_val
            target_q2 = (cost.pow(2)
                         + (self.gamma ** 2) * not_done * next_q2_val
                         + 2.0 * self.gamma * not_done * cost * next_q1_val)

        loss_q1 = F.mse_loss(q1_sa, target_q1)
        loss_q2 = F.mse_loss(q2_sa, target_q2)
        loss = loss_q1 + loss_q2

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1_net.parameters()) + list(self.q2_net.parameters()),
            self.grad_clip,
        )
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            hard_update(self.q1_target, self.q1_net)
            hard_update(self.q2_target, self.q2_net)

        if self.train_mode_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    # ── save / load / mode ──────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "q1_net": self.q1_net.state_dict(),
            "q2_net": self.q2_net.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
            "action_grid": self.action_grid,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q1_net.load_state_dict(ckpt["q1_net"])
        self.q2_net.load_state_dict(ckpt["q2_net"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = float(ckpt.get("epsilon", self.epsilon))
        self.learn_steps = int(ckpt.get("learn_steps", 0))
        self.action_grid = np.asarray(ckpt.get("action_grid", self.action_grid),
                                       dtype=np.float32)

    def set_eval_mode(self):
        self.train_mode_enabled = False
        for net in (self.q1_net, self.q2_net, self.q1_target, self.q2_target):
            net.eval()

    def set_train_mode(self):
        self.train_mode_enabled = True
        for net in (self.q1_net, self.q2_net, self.q1_target, self.q2_target):
            net.train()
