"""
CVaR-DDPG: Deep DPG with CVaR (Expected Shortfall) objective.

Extension of Cao et al. (2021) — replacing the mean-std objective
Y = E[C] + λσ(C)  with  CVaR_α(C).

CVaR_α(C) = E[C | C ≥ VaR_α(C)]
           = expected cost in the worst (1-α)% of scenarios.

Motivation: Buehler et al. (2019), cited in the paper, use expected
shortfall as their objective. The paper (Section 3.4) states:
  "Other objective functions Y(t) can be accommodated similarly."

Implementation:
  - Critics Q1, Q2 are trained identically (same targets).
  - Actor objective differs: instead of minimising F = Q1 + λ√(Q2-Q1²),
    we sort the batch Q1 values and minimise the mean of the top (1-α)
    quantile, which approximates CVaR_α.
  - This is a simple batch-level CVaR approximation (valid when batch
    size is large enough to represent the tail).

α = 0.95 means we focus on the worst 5% of hedging outcomes.
"""
from __future__ import annotations
from typing import Any
import torch

from .ddpg_agent import DeepDPGHedgingAgent
from ._rl_common import hard_update


class CVaRDeepDPGHedgingAgent(DeepDPGHedgingAgent):
    """
    DDPG where the actor minimises CVaR_α(cost) instead of E[C]+λσ(C).

    Inherits everything from DeepDPGHedgingAgent — only the actor loss
    computation is overridden.
    """

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        super().__init__(agent_cfg)
        self.cvar_alpha = float(agent_cfg.get("cvar_alpha", 0.95))
        if not (0.0 < self.cvar_alpha < 1.0):
            raise ValueError("cvar_alpha must be in (0, 1)")
        # Number of samples in the tail: top (1-α)% of the batch
        self.tail_k = max(1, int(self.batch_size * (1 - self.cvar_alpha)))

    def learn(self):
        if len(self.replay_buffer) < self.min_buffer:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        cost = -batch.rewards
        w = batch.weights

        # ── Critics: identical to standard DDPG ─────────────────────
        with torch.no_grad():
            na  = self.actor_target(batch.next_states)
            nq1 = self.critic_1_target(batch.next_states, na)
            nq2 = self.critic_2_target(batch.next_states, na)
            nd  = 1.0 - batch.dones
            tgt_q1 = cost + self.gamma * nd * nq1
            tgt_q2 = (cost**2
                      + 2 * self.gamma * nd * cost * nq1
                      + (self.gamma**2) * nd * nq2)

        cq1 = self.critic_1(batch.states, batch.actions)
        cq2 = self.critic_2(batch.states, batch.actions)

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

        if batch.indices is not None:
            prios = (td1.detach().sqrt().squeeze(-1).cpu().numpy()
                     + td2.detach().sqrt().squeeze(-1).cpu().numpy())
            self.replay_buffer.update_priorities(batch.indices, prios)

        # ── Actor: minimise CVaR_α instead of E[C]+λσ(C) ────────────
        # CVaR_α(C) ≈ mean of the top (1-α) fraction of Q1 values
        aa  = self.actor(batch.states)
        q1a = self.critic_1(batch.states, aa).squeeze(-1)  # (batch,)

        # Sort by predicted cost (Q1), take the worst (1-α)% = tail_k
        sorted_q1, _ = torch.sort(q1a, descending=True)
        cvar_estimate = sorted_q1[:self.tail_k].mean()
        actor_loss = cvar_estimate

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        # Keep target updates consistent with base DDPG.
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_1_target, self.critic_1)
            hard_update(self.critic_2_target, self.critic_2)

        if self.train_mode_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float((loss_c1 + loss_c2 + actor_loss.detach()).item())
