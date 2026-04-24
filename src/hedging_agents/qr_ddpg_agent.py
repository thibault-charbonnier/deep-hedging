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

Actor objective — selected via ``hedging_agent.actor_objective``:

  - ``"cvar"`` (default): minimise CVaR_α of the predicted cost
    distribution (= average of the upper-tail quantiles). Single,
    financially interpretable risk metric — standard in Basel /
    Solvency II frameworks.

  - ``"mean_variance"``: minimise ``E[C] + risk_lambda · std[C]``,
    estimated from the quantile grid. Same objective shape as the
    Cao et al. (2021) DDPG baseline — directly comparable.

Keeps ε-greedy exploration and PER from the DDPG baseline so the
infra stays identical.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ._rl_common import PERActorCriticAgent, QuantileCriticMLP, hard_update


class QRDeepDPGHedgingAgent(PERActorCriticAgent):
    """Quantile Regression DDPG with a configurable risk objective (CVaR or mean-variance)."""

    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        """Build the quantile critic and the risk objective on top of the shared plumbing."""
        super().__init__(agent_cfg)

        self.n_quantiles = int(agent_cfg.get("n_quantiles", 51))
        self.cvar_alpha = float(agent_cfg.get("cvar_alpha", 0.95))
        self.huber_kappa = float(agent_cfg.get("huber_kappa", 1.0))
        self.actor_objective = str(agent_cfg.get("actor_objective", "cvar")).lower()
        if self.actor_objective not in ("cvar", "mean_variance"):
            raise ValueError(
                f"actor_objective must be 'cvar' or 'mean_variance', got {self.actor_objective!r}"
            )
        self.risk_lambda = float(agent_cfg.get("risk_lambda", 1.5))

        self.critic = QuantileCriticMLP(
            self.state_dim, 1, self.n_quantiles, self.hidden_dims
        ).to(self.device)
        self.critic_target = QuantileCriticMLP(
            self.state_dim, 1, self.n_quantiles, self.hidden_dims
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # τ_i = (i − 0.5)/N for i=1..N — fixed quantile fractions.
        tau = (torch.arange(self.n_quantiles, dtype=torch.float32) + 0.5) / self.n_quantiles
        self.register_tau = tau.to(self.device)

        # Upper-tail mask for the CVaR objective. If alpha is so high that no
        # quantile lies above it, fall back to the single worst quantile so
        # the objective is still well-defined.
        self.cvar_mask = (tau >= self.cvar_alpha).float().to(self.device)
        if self.cvar_mask.sum() == 0:
            self.cvar_mask = torch.zeros_like(tau, device=self.device)
            self.cvar_mask[-1] = 1.0

    def _networks(self) -> list[nn.Module]:
        """Every network driven by the train/eval toggle."""
        return [self.actor, self.actor_target, self.critic, self.critic_target]

    def _critic_nets(self) -> list[nn.Module]:
        """Online critic frozen during the actor update."""
        return [self.critic]

    def _sync_targets(self) -> None:
        """Hard-copy actor and critic into their target networks."""
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def _actor_loss_from_quantiles(self, q_pred: torch.Tensor) -> torch.Tensor:
        """Scalar actor objective per sample, derived from the predicted cost quantiles ``[B, N]``.

        ``cvar``: average of the quantiles whose fraction lies in the
        upper tail (mask selected at ``cvar_alpha``).
        ``mean_variance``: empirical ``mean + risk_lambda * std`` over
        the quantile grid, using ``ddof=0`` to match the DDPG baseline.
        """
        if self.actor_objective == "cvar":
            return (q_pred * self.cvar_mask).sum(dim=-1) / self.cvar_mask.sum()
        # mean_variance
        mean = q_pred.mean(dim=-1)
        var = q_pred.var(dim=-1, unbiased=False)
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        return mean + self.risk_lambda * std

    def _huber_quantile_loss(self, td_errors: torch.Tensor) -> torch.Tensor:
        """Quantile Huber loss. ``td_errors`` has shape ``[B, N_current, N_target]``."""
        abs_e = td_errors.abs()
        huber = torch.where(
            abs_e <= self.huber_kappa,
            0.5 * td_errors.pow(2),
            self.huber_kappa * (abs_e - 0.5 * self.huber_kappa),
        )
        tau = self.register_tau.view(1, -1, 1)
        weight = (tau - (td_errors.detach() < 0).float()).abs()
        return (weight * huber).mean(dim=2).sum(dim=1)  # → [B]

    def learn(self) -> float | None:
        """Run one gradient update on the quantile critic and the risk actor.

        Critic: quantile Huber regression against the bootstrapped
        target quantiles ``y = r + gamma * (1 - done) * theta(s', pi_target(s'))``.
        Actor: minimises the scalar risk objective selected by
        ``actor_objective`` (``"cvar"`` or ``"mean_variance"``) derived
        from the predicted cost quantiles.
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

        # COST convention: learn the quantiles of the COST distribution
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

        # PER priorities: mean |TD| across the quantile grid.
        prios = td.detach().abs().mean(dim=(1, 2)).cpu().numpy().reshape(-1)
        prios = np.abs(prios) + self.per_eps
        self._update_priorities(batch["indexes"], prios)

        # ── Actor update (risk objective over the predicted cost distribution) ───
        self._freeze_critics()
        aa = self.actor(states)
        q_pred = self.critic(states, aa)  # [B, N] — predicted cost quantiles
        actor_loss = self._actor_loss_from_quantiles(q_pred).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self._unfreeze_critics()

        self._post_learn_step()
        return float((loss_c + actor_loss.detach()).item())
