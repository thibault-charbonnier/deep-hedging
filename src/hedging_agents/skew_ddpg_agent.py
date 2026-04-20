"""
Skew-DDPG: Deep DPG with a 3rd critic Q3 to control skewness of hedging cost.

Q1(s,a) ~= E[C]
Q2(s,a) ~= E[C^2]
Q3(s,a) ~= E[C^3]

Actor objective extends mean-std with a skewness penalty:
    F = E[C] + lambda_std * std(C) + lambda_skew * penalty(skew(C))
"""
from __future__ import annotations

from typing import Any

import torch

from .ddpg_agent import DeepDPGHedgingAgent
from ._rl_common import CriticMLP, hard_update, soft_update


class SkewDeepDPGHedgingAgent(DeepDPGHedgingAgent):
    """DDPG variant using Q1/Q2/Q3 with gradient clipping for stability."""

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        super().__init__(agent_cfg)

        self.skew_lambda = float(agent_cfg.get("skew_lambda", 0.1))
        self.skew_eps = float(agent_cfg.get("skew_eps", 1e-6))
        self.skew_penalty = str(agent_cfg.get("skew_penalty", "positive")).lower()
        self.grad_clip_q3 = float(agent_cfg.get("grad_clip_q3", self.grad_clip))

        # Third critic for E[C^3]
        self.critic_3 = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_3_target = CriticMLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        hard_update(self.critic_3_target, self.critic_3)
        self.critic_3_opt = torch.optim.Adam(self.critic_3.parameters(), lr=self.lr_critic)

    def _apply_skew_penalty(self, skew: torch.Tensor) -> torch.Tensor:
        if self.skew_penalty == "absolute":
            return torch.abs(skew)
        if self.skew_penalty == "signed":
            return skew
        # default: only penalize right-tail (large positive costs)
        return torch.relu(skew)

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
            nq3 = self.critic_3_target(next_states, na).squeeze(-1)
            nd = 1.0 - dones

            tgt_q1 = cost + self.gamma * nd * nq1
            tgt_q2 = (
                cost.pow(2)
                + 2.0 * self.gamma * nd * cost * nq1
                + (self.gamma ** 2) * nd * nq2
            )
            tgt_q3 = (
                cost.pow(3)
                + 3.0 * self.gamma * nd * cost.pow(2) * nq1
                + 3.0 * (self.gamma ** 2) * nd * cost * nq2
                + (self.gamma ** 3) * nd * nq3
            )

        cq1 = self.critic_1(states, actions).squeeze(-1)
        cq2 = self.critic_2(states, actions).squeeze(-1)
        cq3 = self.critic_3(states, actions).squeeze(-1)

        td1 = (cq1 - tgt_q1).pow(2)
        td2 = (cq2 - tgt_q2).pow(2)
        td3 = (cq3 - tgt_q3).pow(2)

        loss_c1 = (w * td1).mean()
        loss_c2 = (w * td2).mean()
        loss_c3 = (w * td3).mean()

        self.critic_1_opt.zero_grad()
        loss_c1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        loss_c2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip)
        self.critic_2_opt.step()

        self.critic_3_opt.zero_grad()
        loss_c3.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_3.parameters(), self.grad_clip_q3)
        self.critic_3_opt.step()

        prios = (td1.detach().sqrt() + td2.detach().sqrt() + td3.detach().sqrt()).cpu().numpy().reshape(-1) + self.per_eps
        self._update_priorities(batch["indexes"], prios)

        for p in self.critic_1.parameters():
            p.requires_grad_(False)
        for p in self.critic_2.parameters():
            p.requires_grad_(False)
        for p in self.critic_3.parameters():
            p.requires_grad_(False)

        aa = self.actor(states)
        q1a = self.critic_1(states, aa)
        q2a = self.critic_2(states, aa)
        q3a = self.critic_3(states, aa)

        var_a = torch.clamp(q2a - q1a.pow(2), min=self.skew_eps)
        std_a = torch.sqrt(var_a)
        central_m3 = q3a - 3.0 * q1a * q2a + 2.0 * q1a.pow(3)
        skew_a = central_m3 / (std_a.pow(3) + self.skew_eps)

        actor_loss = (
            q1a + self.risk_lambda * std_a + self.skew_lambda * self._apply_skew_penalty(skew_a)
        ).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        for p in self.critic_1.parameters():
            p.requires_grad_(True)
        for p in self.critic_2.parameters():
            p.requires_grad_(True)
        for p in self.critic_3.parameters():
            p.requires_grad_(True)

        self.learn_steps += 1
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)
        soft_update(self.critic_3_target, self.critic_3, self.tau)

        if self.train_mode_enabled:
            self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay)

        return float((loss_c1 + loss_c2 + loss_c3 + actor_loss.detach()).item())


    def set_eval_mode(self):
        self.train_mode_enabled = False
        for m in [
            self.actor,
            self.actor_target,
            self.critic_1,
            self.critic_1_target,
            self.critic_2,
            self.critic_2_target,
            self.critic_3,
            self.critic_3_target,
        ]:
            m.eval()

    def set_train_mode(self):
        self.train_mode_enabled = True
        for m in [
            self.actor,
            self.actor_target,
            self.critic_1,
            self.critic_1_target,
            self.critic_2,
            self.critic_2_target,
            self.critic_3,
            self.critic_3_target,
        ]:
            m.train()

