from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract_agent import AbstractHedgingAgent
from ._rl_common import CriticMLP, MLP, ReplayBuffer, get_device, hard_update, soft_update


class _Actor(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: tuple[int, ...], action_low: float, action_high: float) -> None:
        super().__init__()
        self.backbone = MLP(state_dim, 1, hidden_dims, output_activation=nn.Tanh())
        self.register_buffer("action_low", torch.tensor([action_low], dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor([action_high], dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(state)
        mid = 0.5 * (self.action_high + self.action_low)
        half_range = 0.5 * (self.action_high - self.action_low)
        return mid + half_range * raw


class DeepDPGHedgingAgent(AbstractHedgingAgent):
    """
    Paper-aligned continuous-action actor-critic.

    Difference versus vanilla DDPG:
    - critic_1 learns the expected future hedging cost E[C_t]
    - critic_2 learns the second non-central moment E[C_t^2]
    - actor minimizes Q1 + lambda * sqrt(Q2 - Q1^2)

    This is much closer to Cao et al. than a standard reward-maximizing DDPG.
    """

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        super().__init__(agent_cfg)
        self.device = get_device()
        self.state_dim = int(agent_cfg.get("state_dim", 3))
        self.action_dim = 1
        self.hidden_dims = tuple(agent_cfg.get("hidden_dims", [128, 128]))
        self.lr_actor = float(agent_cfg.get("actor_learning_rate", agent_cfg.get("learning_rate", 1e-4)))
        self.lr_critic = float(agent_cfg.get("critic_learning_rate", agent_cfg.get("learning_rate", 1e-3)))
        self.batch_size = int(agent_cfg.get("learning_batch_size", agent_cfg.get("learning_bactch_size", 128)))
        self.buffer_size = int(agent_cfg.get("replay_capacity", 100_000))
        self.min_buffer_size = int(agent_cfg.get("min_buffer_size", self.batch_size))
        self.tau = float(agent_cfg.get("tau", 5e-3))
        self.grad_clip = float(agent_cfg.get("grad_clip", 1.0))
        self.risk_lambda = float(agent_cfg.get("risk_lambda", 1.5))

        self.action_low = float(agent_cfg.get("action_low", -1.0))
        self.action_high = float(agent_cfg.get("action_high", 1.0))

        self.noise_std = float(agent_cfg.get("exploration_noise_start", 0.20))
        self.noise_std_min = float(agent_cfg.get("exploration_noise_end", 0.02))
        self.noise_decay = float(agent_cfg.get("exploration_noise_decay", 0.9995))

        self.actor = _Actor(self.state_dim, self.hidden_dims, self.action_low, self.action_high).to(self.device)
        self.actor_target = _Actor(self.state_dim, self.hidden_dims, self.action_low, self.action_high).to(self.device)
        self.critic_1 = CriticMLP(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.critic_1_target = CriticMLP(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.critic_2 = CriticMLP(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.critic_2_target = CriticMLP(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_1_target, self.critic_1)
        hard_update(self.critic_2_target, self.critic_2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.lr_critic)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.lr_critic)

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)
        self.train_mode_enabled = True

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    def act(self, state: Any, eval_mode: bool = False) -> float:
        state_t = self._state_tensor(np.asarray(state, dtype=np.float32))
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()[0]

        if (not eval_mode) and self.train_mode_enabled:
            action += np.random.normal(0.0, self.noise_std)

        action = np.clip(action, self.action_low, self.action_high)
        return float(action)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.replay_buffer) < self.min_buffer_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        cost = -batch.rewards

        with torch.no_grad():
            next_actions = self.actor_target(batch.next_states)
            next_q1 = self.critic_1_target(batch.next_states, next_actions)
            next_q2 = self.critic_2_target(batch.next_states, next_actions)
            not_done = 1.0 - batch.dones

            target_q1 = cost + self.gamma * not_done * next_q1
            target_q2 = (cost * cost) + 2.0 * self.gamma * not_done * cost * next_q1 + (self.gamma ** 2) * not_done * next_q2

        current_q1 = self.critic_1(batch.states, batch.actions)
        current_q2 = self.critic_2(batch.states, batch.actions)

        critic_1_loss = F.mse_loss(current_q1, target_q1)
        critic_2_loss = F.mse_loss(current_q2, target_q2)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip)
        self.critic_2_optimizer.step()

        actor_actions = self.actor(batch.states)
        q1_actor = self.critic_1(batch.states, actor_actions)
        q2_actor = self.critic_2(batch.states, actor_actions)
        variance_actor = torch.clamp(q2_actor - q1_actor.pow(2), min=1e-8)
        risk_objective = q1_actor + self.risk_lambda * torch.sqrt(variance_actor)
        actor_loss = risk_objective.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)

        if self.train_mode_enabled:
            self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay)

        total_loss = critic_1_loss + critic_2_loss + actor_loss.detach()
        return float(total_loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_1_target": self.critic_1_target.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_2_target": self.critic_2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
                "noise_std": self.noise_std,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_1_target.load_state_dict(checkpoint["critic_1_target"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.critic_2_target.load_state_dict(checkpoint["critic_2_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.noise_std = float(checkpoint.get("noise_std", self.noise_std))

    def set_eval_mode(self):
        self.train_mode_enabled = False
        self.actor.eval()
        self.actor_target.eval()
        self.critic_1.eval()
        self.critic_1_target.eval()
        self.critic_2.eval()
        self.critic_2_target.eval()

    def set_train_mode(self):
        self.train_mode_enabled = True
        self.actor.train()
        self.actor_target.train()
        self.critic_1.train()
        self.critic_1_target.train()
        self.critic_2.train()
        self.critic_2_target.train()
