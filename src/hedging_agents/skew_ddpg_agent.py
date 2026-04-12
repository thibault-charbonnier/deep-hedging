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
from ._rl_common import CriticMLP, hard_update


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
        if len(self.replay_buffer) < self.min_buffer:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        cost = -batch.rewards
        w = batch.weights

        with torch.no_grad():
            na = self.actor_target(batch.next_states)
            nq1 = self.critic_1_target(batch.next_states, na)
            nq2 = self.critic_2_target(batch.next_states, na)
            nq3 = self.critic_3_target(batch.next_states, na)
            nd = 1.0 - batch.dones

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

        cq1 = self.critic_1(batch.states, batch.actions)
        cq2 = self.critic_2(batch.states, batch.actions)
        cq3 = self.critic_3(batch.states, batch.actions)

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

        if batch.indices is not None:
            prios = (
                td1.detach().sqrt().squeeze(-1).cpu().numpy()
                + td2.detach().sqrt().squeeze(-1).cpu().numpy()
                + td3.detach().sqrt().squeeze(-1).cpu().numpy()
            )
            self.replay_buffer.update_priorities(batch.indices, prios)

        for p in self.critic_1.parameters():
            p.requires_grad_(False)
        for p in self.critic_2.parameters():
            p.requires_grad_(False)
        for p in self.critic_3.parameters():
            p.requires_grad_(False)

        aa = self.actor(batch.states)
        q1a = self.critic_1(batch.states, aa)
        q2a = self.critic_2(batch.states, aa)
        q3a = self.critic_3(batch.states, aa)

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
        if self.learn_steps % self.target_update_freq == 0:
            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_1_target, self.critic_1)
            hard_update(self.critic_2_target, self.critic_2)
            hard_update(self.critic_3_target, self.critic_3)

        if self.train_mode_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float((loss_c1 + loss_c2 + loss_c3 + actor_loss.detach()).item())

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_1_target": self.critic_1_target.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_2_target": self.critic_2_target.state_dict(),
                "critic_3": self.critic_3.state_dict(),
                "critic_3_target": self.critic_3_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_1_opt": self.critic_1_opt.state_dict(),
                "critic_2_opt": self.critic_2_opt.state_dict(),
                "critic_3_opt": self.critic_3_opt.state_dict(),
                "epsilon": self.epsilon,
                "learn_steps": self.learn_steps,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_1_target.load_state_dict(ckpt["critic_1_target"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.critic_2_target.load_state_dict(ckpt["critic_2_target"])
        self.critic_3.load_state_dict(ckpt["critic_3"])
        self.critic_3_target.load_state_dict(ckpt["critic_3_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_1_opt.load_state_dict(ckpt["critic_1_opt"])
        self.critic_2_opt.load_state_dict(ckpt["critic_2_opt"])
        self.critic_3_opt.load_state_dict(ckpt["critic_3_opt"])
        self.epsilon = float(ckpt.get("epsilon", self.epsilon))
        self.learn_steps = int(ckpt.get("learn_steps", 0))

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

