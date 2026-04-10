"""
Deep DPG with dual critics — Cao et al. (2021) Section 3.4 + Section 4.

Key paper-aligned features:
  - critic_1 → E[C_t],  critic_2 → E[C_t²]
  - actor minimises F = Q1 + λ √(Q2 - Q1²)
  - Prioritized Experience Replay (Schaul et al., 2015)
    "we also implement the prioritized experience replay method" (Section 4)
  - IS weights applied to critic losses
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract_agent import AbstractHedgingAgent
from ._rl_common import (CriticMLP, MLP, PrioritizedReplayBuffer,
                          get_device, hard_update, soft_update)


class _Actor(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_low, action_high):
        super().__init__()
        self.backbone = MLP(state_dim, 1, hidden_dims, output_activation=nn.Tanh())
        self.register_buffer("a_lo", torch.tensor([action_low], dtype=torch.float32))
        self.register_buffer("a_hi", torch.tensor([action_high], dtype=torch.float32))

    def forward(self, s):
        mid = 0.5 * (self.a_hi + self.a_lo)
        half = 0.5 * (self.a_hi - self.a_lo)
        return mid + half * self.backbone(s)


class DeepDPGHedgingAgent(AbstractHedgingAgent):

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        super().__init__(agent_cfg)
        self.device = get_device()
        self.state_dim   = int(agent_cfg.get("state_dim", 4))
        self.hidden_dims = tuple(agent_cfg.get("hidden_dims", [128, 128]))
        self.lr_actor    = float(agent_cfg.get("actor_learning_rate", 1e-4))
        self.lr_critic   = float(agent_cfg.get("critic_learning_rate", 1e-3))
        self.batch_size  = int(agent_cfg.get("learning_batch_size", 128))
        self.buffer_size = int(agent_cfg.get("replay_capacity", 100_000))
        self.min_buffer  = int(agent_cfg.get("min_buffer_size", self.batch_size))
        self.tau         = float(agent_cfg.get("tau", 5e-3))
        self.grad_clip   = float(agent_cfg.get("grad_clip", 1.0))
        self.risk_lambda = float(agent_cfg.get("risk_lambda", 1.5))
        self.action_low  = float(agent_cfg.get("action_low", 0.0))
        self.action_high = float(agent_cfg.get("action_high", 1.0))
        self.noise_std     = float(agent_cfg.get("exploration_noise_start", 0.15))
        self.noise_std_min = float(agent_cfg.get("exploration_noise_end", 0.02))
        self.noise_decay   = float(agent_cfg.get("exploration_noise_decay", 0.9995))

        # ── Networks ─────────────────────────────────────────────────
        self.actor          = _Actor(self.state_dim, self.hidden_dims, self.action_low, self.action_high).to(self.device)
        self.actor_target   = _Actor(self.state_dim, self.hidden_dims, self.action_low, self.action_high).to(self.device)
        self.critic_1       = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_1_target= CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_2       = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_2_target= CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_1_target, self.critic_1)
        hard_update(self.critic_2_target, self.critic_2)

        self.actor_opt   = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_1_opt= torch.optim.Adam(self.critic_1.parameters(), lr=self.lr_critic)
        self.critic_2_opt= torch.optim.Adam(self.critic_2.parameters(), lr=self.lr_critic)

        # ── Prioritized Replay (paper Section 4) ────────────────────
        per_alpha = float(agent_cfg.get("per_alpha", 0.6))
        per_beta  = float(agent_cfg.get("per_beta_start", 0.4))
        per_frames= int(agent_cfg.get("per_beta_frames", 100_000))
        self.replay_buffer = PrioritizedReplayBuffer(
            self.buffer_size, self.state_dim, 1,
            alpha=per_alpha, beta_start=per_beta, beta_frames=per_frames)
        self.train_mode_enabled = True

    def _st(self, state):
        return torch.as_tensor(np.asarray(state, dtype=np.float32),
                               device=self.device).unsqueeze(0)

    def act(self, state, eval_mode=False):
        with torch.no_grad():
            a = self.actor(self._st(state)).squeeze(0).cpu().numpy()[0]
        if (not eval_mode) and self.train_mode_enabled:
            a += np.random.normal(0.0, self.noise_std)
        return float(np.clip(a, self.action_low, self.action_high))

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.min_buffer:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        cost = -batch.rewards
        w = batch.weights          # IS weights from PER

        with torch.no_grad():
            na = self.actor_target(batch.next_states)
            nq1 = self.critic_1_target(batch.next_states, na)
            nq2 = self.critic_2_target(batch.next_states, na)
            nd = 1.0 - batch.dones
            tgt_q1 = cost + self.gamma * nd * nq1
            tgt_q2 = cost**2 + 2*self.gamma*nd*cost*nq1 + (self.gamma**2)*nd*nq2

        cq1 = self.critic_1(batch.states, batch.actions)
        cq2 = self.critic_2(batch.states, batch.actions)

        # IS-weighted MSE losses
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

        # Update priorities with TD error
        if batch.indices is not None:
            prios = (td1.detach().sqrt().squeeze(-1).cpu().numpy()
                     + td2.detach().sqrt().squeeze(-1).cpu().numpy())
            self.replay_buffer.update_priorities(batch.indices, prios)

        # Actor: minimise F = Q1 + λ√(Q2 - Q1²)
        aa = self.actor(batch.states)
        q1a = self.critic_1(batch.states, aa)
        q2a = self.critic_2(batch.states, aa)
        var_a = torch.clamp(q2a - q1a.pow(2), min=1e-8)
        actor_loss = (q1a + self.risk_lambda * torch.sqrt(var_a)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)

        if self.train_mode_enabled:
            self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay)
        return float((loss_c1 + loss_c2 + actor_loss.detach()).item())

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_1": self.critic_1.state_dict(), "critic_1_target": self.critic_1_target.state_dict(),
            "critic_2": self.critic_2.state_dict(), "critic_2_target": self.critic_2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_1_opt": self.critic_1_opt.state_dict(),
            "critic_2_opt": self.critic_2_opt.state_dict(),
            "noise_std": self.noise_std,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_1_target.load_state_dict(ckpt["critic_1_target"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.critic_2_target.load_state_dict(ckpt["critic_2_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_1_opt.load_state_dict(ckpt["critic_1_opt"])
        self.critic_2_opt.load_state_dict(ckpt["critic_2_opt"])
        self.noise_std = float(ckpt.get("noise_std", self.noise_std))

    def set_eval_mode(self):
        self.train_mode_enabled = False
        for m in [self.actor, self.actor_target, self.critic_1, self.critic_1_target,
                  self.critic_2, self.critic_2_target]:
            m.eval()

    def set_train_mode(self):
        self.train_mode_enabled = True
        for m in [self.actor, self.actor_target, self.critic_1, self.critic_1_target,
                  self.critic_2, self.critic_2_target]:
            m.train()
