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

from ._rl_common import CriticMLP, PERActorCriticAgent, hard_update


class DeepDPGHedgingAgent(PERActorCriticAgent):
    """Deep DPG hedging agent with mean/variance dual critics and PER (Cao et al. 2021)."""

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        """Build the two critics (for E[C] and E[C^2]) on top of the shared plumbing."""
        super().__init__(agent_cfg)
        self.risk_lambda = float(agent_cfg.get("risk_lambda", 1.5))

        self.critic_1 = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_1_target = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_2 = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_2_target = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)
        hard_update(self.critic_2_target, self.critic_2)

        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=self.lr_critic)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=self.lr_critic)

    def _networks(self) -> list[nn.Module]:
        """Every network driven by the train/eval toggle."""
        return [self.actor, self.actor_target,
                self.critic_1, self.critic_1_target,
                self.critic_2, self.critic_2_target]

    def _critic_nets(self) -> list[nn.Module]:
        """Online critics frozen during the actor update."""
        return [self.critic_1, self.critic_2]

    def _sync_targets(self) -> None:
        """Hard-copy actor and both critics into their target networks."""
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_1_target, self.critic_1)
        hard_update(self.critic_2_target, self.critic_2)

    def learn(self) -> float | None:
        """Run one gradient update on both critics and the mean-variance actor.

        Critics fit ``E[C]`` and ``E[C^2]`` via IS-weighted MSE; the actor
        minimises ``Q1 + risk_lambda * sqrt(Q2 - Q1^2)``. Returns a
        scalar loss, or ``None`` if the buffer is still below
        ``min_buffer``.
        """
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
            tgt_q2 = cost**2 + 2 * self.gamma * nd * cost * nq1 + (self.gamma**2) * nd * nq2

        cq1 = self.critic_1(states, actions).squeeze(-1)
        cq2 = self.critic_2(states, actions).squeeze(-1)

        td1 = (cq1 - tgt_q1).pow(2)
        td2 = (cq2 - tgt_q2).pow(2)
        loss_c1 = (w * td1).mean()
        loss_c2 = (w * td2).mean()

        self.critic_1_opt.zero_grad()
        loss_c1.backward()
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        loss_c2.backward()
        self.critic_2_opt.step()

        prios = (td1.detach().sqrt() + td2.detach().sqrt()).cpu().numpy().reshape(-1)
        prios = np.abs(prios) + self.per_eps
        self._update_priorities(batch["indexes"], prios)

        self._freeze_critics()
        aa = self.actor(states)
        q1a = self.critic_1(states, aa)
        q2a = self.critic_2(states, aa)
        var_a = torch.clamp(q2a - q1a.pow(2), min=1e-8)
        actor_loss = (q1a + self.risk_lambda * torch.sqrt(var_a)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self._unfreeze_critics()

        self._post_learn_step()
        return float((loss_c1 + loss_c2 + actor_loss.detach()).item())
