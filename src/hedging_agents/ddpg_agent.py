"""
Deep DPG with dual critics — Cao et al. (2021) Sections 2.6, 3.4, 4.

Paper-aligned design choices:
  - critic_1 → E[C_t],  critic_2 → E[C_t²]
  - actor minimises F = Q1 + λ √(Q2 - Q1²)
  - ε-greedy exploration (Section 2.6):
      "With probability ε, a random action is taken, and with
       probability 1−ε the policy function is followed."
  - Periodic hard copy of target networks (Section 2.5):
      "deep Q-learning keeps a separate copy of the Q-function for
       constructing the update target, and only updates this copy
       periodically."
  - Prioritized Experience Replay (Section 4)
  - IS weights on critic losses
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from cpprb import PrioritizedReplayBuffer

from .abstract_agent import AbstractHedgingAgent
from ._rl_common import CriticMLP, MLP, DEVICE, hard_update


class _Actor(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_low, action_high):
        super().__init__()
        self.backbone = MLP(state_dim, 1, hidden_dims, output_activation=nn.Tanh())
        self.mid = 0.5 * (action_high + action_low)
        self.half = 0.5 * (action_high - action_low)

    def forward(self, s):
        return self.mid + self.half * self.backbone(s)


class DeepDPGHedgingAgent(AbstractHedgingAgent):

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        super().__init__(agent_cfg)
        self.device = DEVICE
        self.state_dim = int(agent_cfg.get("state_dim", 4))
        self.hidden_dims = tuple(agent_cfg.get("hidden_dims", [128, 128]))
        self.lr_actor = float(agent_cfg.get("actor_learning_rate", 1e-4))
        self.lr_critic = float(agent_cfg.get("critic_learning_rate", 1e-3))
        self.batch_size = int(agent_cfg.get("learning_batch_size", 128))
        self.buffer_size = int(agent_cfg.get("replay_capacity", 100_000))
        self.min_buffer = int(agent_cfg.get("min_buffer_size", self.batch_size))
        self.grad_clip = float(agent_cfg.get("grad_clip", 1.0))
        self.risk_lambda = float(agent_cfg.get("risk_lambda", 1.5))
        self.action_low = float(agent_cfg.get("action_low", 0.0))
        self.action_high = float(agent_cfg.get("action_high", 1.0))

        # epsilon-greedy exploration
        self.epsilon = float(agent_cfg.get("exploration_rate_start", 1.0))
        self.epsilon_min = float(agent_cfg.get("exploration_rate_end", 0.05))
        self.epsilon_decay = float(agent_cfg.get("exploration_rate_decay", 0.995))

        # periodic hard target update
        self.target_update_freq = int(agent_cfg.get("target_update_freq", 100))

        # networks
        self.actor = _Actor(self.state_dim, self.hidden_dims, self.action_low, self.action_high).to(self.device)
        self.actor_target = _Actor(self.state_dim, self.hidden_dims, self.action_low, self.action_high).to(self.device)
        self.critic_1 = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_1_target = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_2 = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_2_target = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_1_target, self.critic_1)
        hard_update(self.critic_2_target, self.critic_2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=self.lr_critic)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=self.lr_critic)

        # prioritized replay backend
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

    def _st(self, state):
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    def _replay_size(self) -> int:
        return int(self.replay_buffer.get_stored_size())

    def _update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.replay_buffer.update_priorities(indices, priorities)

    def _sample_batch_tensors(self) -> dict[str, Any]:
        self.per_frame += 1
        beta = min(
            1.0,
            self.per_beta_start + self.per_frame * (1.0 - self.per_beta_start) / self.per_beta_frames,
        )
        batch = self.replay_buffer.sample(self.batch_size, beta=beta)
        rewards = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).reshape(-1)
        dones = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).reshape(-1)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device).reshape(-1)
        indexes = np.asarray(batch["indexes"], dtype=np.int64).reshape(-1)
        return {
            "states": torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device).reshape(-1, 1),
            "rewards": rewards,
            "next_states": torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device),
            "dones": dones,
            "weights": weights,
            "indexes": indexes,
        }

    def act(self, state, eval_mode=False):
        # ── ε-greedy (Paper Section 2.6) ─────────────────────────────
        # "With probability ε, a random action is taken, and with
        #  probability 1−ε the policy function is followed."
        if (not eval_mode) and self.train_mode_enabled:
            if np.random.rand() < self.epsilon:
                return float(np.random.uniform(self.action_low, self.action_high))

        with torch.no_grad():
            a = self.actor(self._st(state)).squeeze(0).cpu().numpy()[0]
        return float(np.clip(a, self.action_low, self.action_high))

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            obs=np.asarray(state, dtype=np.float32),
            act=np.asarray([float(action)], dtype=np.float32),
            rew=float(reward),
            next_obs=np.asarray(next_state, dtype=np.float32),
            done=float(done),
        )

    def learn(self):
        if self._replay_size() < self.min_buffer:
            return None

        batch = self._sample_batch_tensors()
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        w = batch["weights"]

        cost = -rewards

        with torch.no_grad():
            na = self.actor_target(next_states)
            nq1 = self.critic_1_target(next_states, na).squeeze(-1)
            nq2 = self.critic_2_target(next_states, na).squeeze(-1)
            nd = 1.0 - dones
            tgt_q1 = cost + self.gamma * nd * nq1
            tgt_q2 = (cost**2 + 2 * self.gamma * nd * cost * nq1 + (self.gamma**2) * nd * nq2)

        cq1 = self.critic_1(states, actions).squeeze(-1)
        cq2 = self.critic_2(states, actions).squeeze(-1)

        td1 = (cq1 - tgt_q1).pow(2)
        td2 = (cq2 - tgt_q2).pow(2)
        loss_c1 = (w * td1).mean()
        loss_c2 = (w * td2).mean()

        self.critic_1_opt.zero_grad()
        loss_c1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        loss_c2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip)
        self.critic_2_opt.step()

        prios = (td1.detach().sqrt() + td2.detach().sqrt()).cpu().numpy().reshape(-1) + self.per_eps
        self._update_priorities(batch["indexes"], prios)

        for p in self.critic_1.parameters():
            p.requires_grad_(False)
        for p in self.critic_2.parameters():
            p.requires_grad_(False)

        aa = self.actor(states)
        q1a = self.critic_1(states, aa)
        q2a = self.critic_2(states, aa)
        var_a = torch.clamp(q2a - q1a.pow(2), min=1e-8)
        actor_loss = (q1a + self.risk_lambda * torch.sqrt(var_a)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        for p in self.critic_1.parameters():
            p.requires_grad_(True)
        for p in self.critic_2.parameters():
            p.requires_grad_(True)

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_1_target, self.critic_1)
            hard_update(self.critic_2_target, self.critic_2)

        if self.train_mode_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float((loss_c1 + loss_c2 + actor_loss.detach()).item())


    def set_eval_mode(self):
        self.train_mode_enabled = False
        for m in [self.actor, self.actor_target, self.critic_1,
                  self.critic_1_target, self.critic_2, self.critic_2_target]:
            m.eval()

    def set_train_mode(self):
        self.train_mode_enabled = True
        for m in [self.actor, self.actor_target, self.critic_1,
                  self.critic_1_target, self.critic_2, self.critic_2_target]:
            m.train()
