"""
Double DQN with dual Q-functions + Prioritized Experience Replay.
Online nets SELECT the greedy action, target nets EVALUATE it.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from .abstract_agent import AbstractHedgingAgent
from ._rl_common import MLP, PrioritizedReplayBuffer, get_device, hard_update


class DoubleQDNHedgingAgent(AbstractHedgingAgent):
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
        n_act = len(self.action_grid)

        self.q1_net = MLP(self.state_dim, n_act, self.hidden_dims).to(self.device)
        self.q2_net = MLP(self.state_dim, n_act, self.hidden_dims).to(self.device)
        self.q1_target = MLP(self.state_dim, n_act, self.hidden_dims).to(self.device)
        self.q2_target = MLP(self.state_dim, n_act, self.hidden_dims).to(self.device)
        hard_update(self.q1_target, self.q1_net); hard_update(self.q2_target, self.q2_net)

        params = list(self.q1_net.parameters()) + list(self.q2_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        self.replay_buffer = PrioritizedReplayBuffer(
            self.buffer_size, self.state_dim,
            alpha=float(agent_cfg.get("per_alpha", 0.6)),
            beta_start=float(agent_cfg.get("per_beta_start", 0.4)))
        self.train_mode_enabled = True
        self.learn_steps = 0

    def _state_tensor(self, s):
        return torch.as_tensor(np.asarray(s, dtype=np.float32), device=self.device).unsqueeze(0)
    def _action_to_index(self, a): return int(np.argmin(np.abs(self.action_grid - float(a))))
    def _index_to_action(self, i): return float(self.action_grid[int(i)])
    def _F(self, q1, q2):
        return q1 + self.risk_lambda * torch.sqrt(torch.clamp(q2 - q1.pow(2), min=1e-8))

    def act(self, state, eval_mode=False):
        if (not eval_mode) and self.train_mode_enabled and np.random.rand() < self.epsilon:
            return float(np.random.choice(self.action_grid))
        with torch.no_grad():
            st = self._state_tensor(state)
            return self._index_to_action(int(torch.argmin(self._F(self.q1_net(st), self.q2_net(st)), dim=1).item()))

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.min_buffer_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        cost = -batch.rewards
        aidx = torch.as_tensor(
            [self._action_to_index(a) for a in batch.actions.squeeze(-1).cpu().numpy()],
            dtype=torch.long, device=self.device).unsqueeze(-1)

        q1_sa = self.q1_net(batch.states).gather(1, aidx)
        q2_sa = self.q2_net(batch.states).gather(1, aidx)

        with torch.no_grad():
            not_done = 1.0 - batch.dones
            # Double DQN: ONLINE selects, TARGET evaluates
            nq1_on = self.q1_net(batch.next_states)
            nq2_on = self.q2_net(batch.next_states)
            ng = self._F(nq1_on, nq2_on).argmin(dim=1, keepdim=True)
            nq1v = self.q1_target(batch.next_states).gather(1, ng)
            nq2v = self.q2_target(batch.next_states).gather(1, ng)
            tgt_q1 = cost + self.gamma * not_done * nq1v
            tgt_q2 = cost.pow(2) + (self.gamma**2)*not_done*nq2v + 2*self.gamma*not_done*cost*nq1v

        td1 = (q1_sa - tgt_q1).pow(2)
        td2 = (q2_sa - tgt_q2).pow(2)
        loss = (batch.weights * (td1 + td2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1_net.parameters()) + list(self.q2_net.parameters()), self.grad_clip)
        self.optimizer.step()

        if batch.indices is not None:
            self.replay_buffer.update_priorities(batch.indices, td1.detach().squeeze(-1).cpu().numpy())

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            hard_update(self.q1_target, self.q1_net); hard_update(self.q2_target, self.q2_net)
        if self.train_mode_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path):
        torch.save({"q1": self.q1_net.state_dict(), "q2": self.q2_net.state_dict(),
                     "q1t": self.q1_target.state_dict(), "q2t": self.q2_target.state_dict(),
                     "opt": self.optimizer.state_dict(), "eps": self.epsilon,
                     "steps": self.learn_steps, "grid": self.action_grid}, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.q1_net.load_state_dict(c["q1"]); self.q2_net.load_state_dict(c["q2"])
        self.q1_target.load_state_dict(c["q1t"]); self.q2_target.load_state_dict(c["q2t"])
        self.optimizer.load_state_dict(c["opt"])
        self.epsilon = float(c.get("eps", self.epsilon))
        self.learn_steps = int(c.get("steps", 0))
        self.action_grid = np.asarray(c.get("grid", self.action_grid), dtype=np.float32)

    def set_eval_mode(self):
        self.train_mode_enabled = False
        for n in (self.q1_net, self.q2_net, self.q1_target, self.q2_target): n.eval()
    def set_train_mode(self):
        self.train_mode_enabled = True
        for n in (self.q1_net, self.q2_net, self.q1_target, self.q2_target): n.train()
