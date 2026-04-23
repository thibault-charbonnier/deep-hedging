"""
QR-DDPG — Distributional Deep DPG with quantile regression.

Instead of learning the expectation Q(s,a) = E[return | s, a] as a scalar,
the critic predicts the FULL DISTRIBUTION of the return via N quantile
values:
    θ_i(s, a) ≈ F⁻¹_{return | s,a}(τ_i)   with τ_i = (i - 0.5)/N, i=1..N

Training: quantile Huber regression (Dabney et al. 2018, QR-DQN).
    L = Σ_i  E_j [ ρ_κ^{τ_i}(y_j − θ_i) ]
    ρ_κ^τ(u) = |τ − 1_{u<0}| · Huber_κ(u)
    Huber_κ(u) = 0.5·u²            if |u|≤κ
               = κ(|u| − 0.5κ)     else
y_j = r + γ·(1−done)·θ_j(s', π_target(s'))  — target quantiles.

Actor objective: minimise CVaR_α of the predicted cost distribution
(= average of the upper tail).  α is the risk level; α=0.95 means we
average the ~5% worst outcomes.  This replaces the mean−std−skew
decomposition of SkewDDPG with a single, financially interpretable
risk metric.

Keeps ε-greedy exploration and PER from the DDPG baseline so the
infra stays identical.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cpprb import PrioritizedReplayBuffer

from .abstract_agent import AbstractHedgingAgent
from ._rl_common import MLP, QuantileCriticMLP, get_device, hard_update


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


class QRDeepDPGHedgingAgent(AbstractHedgingAgent):
    """Quantile Regression DDPG with CVaR-based actor objective."""

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        super().__init__(agent_cfg)
        self.device = get_device()
        self.state_dim = int(agent_cfg.get("state_dim", 4))
        self.hidden_dims = tuple(agent_cfg.get("hidden_dims", [128, 128]))
        self.lr_actor = float(agent_cfg.get("actor_learning_rate", 1e-4))
        self.lr_critic = float(agent_cfg.get("critic_learning_rate", 1e-3))
        self.batch_size = int(agent_cfg.get("learning_batch_size", 128))
        self.buffer_size = int(agent_cfg.get("replay_capacity", 100_000))
        self.min_buffer = int(agent_cfg.get("min_buffer_size", self.batch_size))
        self.action_low = float(agent_cfg.get("action_low", 0.0))
        self.action_high = float(agent_cfg.get("action_high", 1.0))

        # Distributional-specific hyperparameters.
        self.n_quantiles = int(agent_cfg.get("n_quantiles", 51))
        self.cvar_alpha = float(agent_cfg.get("cvar_alpha", 0.95))
        self.huber_kappa = float(agent_cfg.get("huber_kappa", 1.0))

        # ε-greedy (identical to DDPG).
        self.epsilon = float(agent_cfg.get("exploration_rate_start", 1.0))
        self.epsilon_min = float(agent_cfg.get("exploration_rate_end", 0.05))
        self.epsilon_decay = float(agent_cfg.get("exploration_rate_decay", 0.995))

        self.target_update_freq = int(agent_cfg.get("target_update_freq", 100))

        # Networks.
        self.actor = _Actor(self.state_dim, self.hidden_dims, self.action_low, self.action_high).to(self.device)
        self.actor_target = _Actor(self.state_dim, self.hidden_dims, self.action_low, self.action_high).to(self.device)
        self.critic = QuantileCriticMLP(self.state_dim, 1, self.n_quantiles, self.hidden_dims).to(self.device)
        self.critic_target = QuantileCriticMLP(self.state_dim, 1, self.n_quantiles, self.hidden_dims).to(self.device)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # τ_i = (i − 0.5)/N for i=1..N  — fixed quantile fractions.
        tau = (torch.arange(self.n_quantiles, dtype=torch.float32) + 0.5) / self.n_quantiles
        self.register_tau = tau.to(self.device)

        # Mask of which quantiles belong to the upper (1−α) tail — used for CVaR.
        # For minimisation of cost: upper tail = worst outcomes.
        self.cvar_mask = (tau >= self.cvar_alpha).float().to(self.device)
        if self.cvar_mask.sum() == 0:
            # If α too close to 1 with too few quantiles, fall back to the
            # single worst quantile so the objective is well defined.
            self.cvar_mask = torch.zeros_like(tau, device=self.device)
            self.cvar_mask[-1] = 1.0

        # Prioritized Experience Replay.
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

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

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
        return {
            "states": torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device).reshape(-1, 1),
            "rewards": torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).reshape(-1),
            "next_states": torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).reshape(-1),
            "weights": torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device).reshape(-1),
            "indexes": np.asarray(batch["indexes"], dtype=np.int64).reshape(-1),
        }

    def _huber_quantile_loss(self, td_errors: torch.Tensor) -> torch.Tensor:
        """Quantile Huber loss. td_errors: [B, N_current, N_target]."""
        abs_e = td_errors.abs()
        huber = torch.where(
            abs_e <= self.huber_kappa,
            0.5 * td_errors.pow(2),
            self.huber_kappa * (abs_e - 0.5 * self.huber_kappa),
        )
        # τ_i indexed on the SECOND axis (N_current) → broadcast to [1, N, 1].
        tau = self.register_tau.view(1, -1, 1)
        weight = (tau - (td_errors.detach() < 0).float()).abs()
        return (weight * huber).mean(dim=2).sum(dim=1)  # → [B]

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def act(self, state, eval_mode=False):
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

        # COST convention: we learn the quantiles of the COST distribution
        # (C = −R) so that "upper tail = worst" and CVaR = expected worst loss.
        cost = -rewards

        # ── Target quantiles ─────────────────────────────────────────
        with torch.no_grad():
            next_a = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_a)  # [B, N]
            nd = (1.0 - dones).unsqueeze(-1)
            y = cost.unsqueeze(-1) + self.gamma * nd * next_q  # [B, N]

        # ── Critic update (quantile Huber regression) ────────────────
        current_q = self.critic(states, actions)  # [B, N]
        # td_errors: target_j − current_i → shape [B, N_current, N_target]
        td = y.unsqueeze(1) - current_q.unsqueeze(2)
        loss_per_sample = self._huber_quantile_loss(td)  # [B]
        loss_c = (w * loss_per_sample).mean()

        self.critic_opt.zero_grad()
        loss_c.backward()
        self.critic_opt.step()

        # ── Per-sample TD priorities (mean |td| across quantiles) ────
        prios = td.detach().abs().mean(dim=(1, 2)).cpu().numpy().reshape(-1)
        prios = np.abs(prios) + self.per_eps
        self._update_priorities(batch["indexes"], prios)

        # ── Actor update (CVaR of predicted cost distribution) ───────
        for p in self.critic.parameters():
            p.requires_grad_(False)

        aa = self.actor(states)
        q_pred = self.critic(states, aa)  # [B, N] — predicted cost quantiles
        # CVaR_α = mean of quantiles i where τ_i ≥ α.
        mask = self.cvar_mask  # [N]
        cvar = (q_pred * mask).sum(dim=-1) / mask.sum()
        actor_loss = cvar.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(True)

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_target, self.critic)

        if self.train_mode_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float((loss_c + actor_loss.detach()).item())

    def set_eval_mode(self):
        self.train_mode_enabled = False
        for m in [self.actor, self.actor_target, self.critic, self.critic_target]:
            m.eval()

    def set_train_mode(self):
        self.train_mode_enabled = True
        for m in [self.actor, self.actor_target, self.critic, self.critic_target]:
            m.train()
