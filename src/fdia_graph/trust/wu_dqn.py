"""[WU26]'s Solution 2: the trusted PMUs by a deep Q-network trained with Algorithm 1 (Sec. IV-D2, p. 658)
on trusted-PMU MDPs (`trust.WuDefenseEnv`; docs/plans/WU_DEFENSE_PLAN.md, PR D).

Algorithm 1, as the paper gives it: initialize the Q-network and an empty replay buffer D, and the target
network to the same weights; for each iteration, initialize the system state and generate the attack for
the current measurement matrix; for each step, choose a random action with probability epsilon or else
the greedy one (eq. 34), perform it, and when the defended cost falls below the undefended one break,
else store (o_t, u_t, r_t, o_t+1) in D; then sample a minibatch from D, update the Q-network by the
squared loss to the target r + gamma max Q_target(o', u') (eqs. 35-37), and every d iterations copy it to
the target network. The hyperparameters are Sec. V's and the Appendix's (`WuDqnConfig`). Each iteration
here draws one training environment (one window, "dynamic grid scenarios", p. 658) at random.

Ours, where the paper is silent: the network (two hidden layers of `WuDqnConfig.hidden` units with ReLU,
in torch rather than TensorFlow v2.5), the Q-values of PMUs already trusted masked out, a terminal
transition (every step taken, the attack infeasible, or line 8's break) bootstrapping nothing, and each
state feature divided by its largest magnitude over the training environments' first states, since
readings in MW and per-unit ratios differ by orders of magnitude.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from ..models.config import WuDqnConfig
from ..models.inputs import SameDefense
from .defense import WuDefenseEnv
from .dqn import _Replay, q_network
from .dqn import _torch as _dqn_torch

if TYPE_CHECKING:
    import torch
    from torch import nn


def _torch() -> ModuleType:
    return _dqn_torch("TrustedPMUsDQN")


class TrustedPMUsDQN:
    """Algorithm 1 over `envs` (the same candidate PMUs and steps in each): `fit` trains and fills
    `history` (each iteration's return) and `breaks` (iterations line 8 ended); `order(env)` is the
    trained policy's greedy sequence on an environment, the actions as indices into its PMUs."""

    def __init__(self, envs: Sequence[WuDefenseEnv], config: WuDqnConfig = WuDqnConfig()) -> None:
        SameDefense([env.n_actions for env in envs], [len(env.config.slots) for env in envs])
        self.envs, self.config = list(envs), config
        self.history: list[float] = []
        self.breaks = 0
        first = np.stack([env.reset() for env in self.envs])
        self.scale = np.where(np.abs(first).max(axis=0) > 0, np.abs(first).max(axis=0), 1.0)

    def fit(self) -> TrustedPMUsDQN:
        """Train the Q-network by [WU26 Alg. 1] for `config.episodes` iterations."""
        torch, cfg = _torch(), self.config
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.net, target = self._net(), self._net()
        target.load_state_dict(self.net.state_dict())
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        replay = _Replay(cfg.buffer)
        for ep in range(cfg.episodes):
            env = self.envs[int(rng.integers(len(self.envs)))]
            self.history.append(self._episode(env, math.exp(-cfg.epsilon_decay * ep), rng, replay))
            if len(replay) >= cfg.batch:
                self._learn(target, opt, replay.sample(rng.integers(len(replay), size=cfg.batch)))
            if (ep + 1) % cfg.target_every == 0:
                target.load_state_dict(self.net.state_dict())
        return self

    def order(self, env: WuDefenseEnv) -> list[int]:
        """The trained policy's greedy sequence on `env` (eq. 38), until the episode ends."""
        obs, taken, done = env.reset(), [], False
        while not done and env.valid().any():
            action = self._act(obs, env.valid(), 0.0, np.random.default_rng(0))
            obs, _, done = env.step(action)
            taken.append(action)
        return taken

    # ---- Algorithm 1 --------------------------------------------------------------------------------
    def _episode(self, env: WuDefenseEnv, eps: float, rng: np.random.Generator, replay: _Replay) -> float:
        """One iteration's steps on `env` (lines 4-14), storing the transitions; returns the return."""
        obs, total, breaks = env.reset(), 0.0, env.breaks
        done = False
        while not done and env.valid().any():
            action = self._act(obs, env.valid(), eps, rng)
            nxt, reward, done = env.step(action)
            if env.breaks > breaks:  # line 8: break after recording the state, no transition stored
                self.breaks += 1
                break
            replay.append((obs / self.scale, action, reward, nxt / self.scale, done, env.valid()))
            obs, total = nxt, total + reward
        return total

    def _act(self, obs: np.ndarray, valid: np.ndarray, eps: float, rng: np.random.Generator) -> int:
        """Epsilon-greedy over the PMUs still on offer (line 6, [WU26 eq. 34])."""
        if rng.random() < eps:
            return int(rng.choice(np.flatnonzero(valid)))
        torch = _torch()
        with torch.no_grad():
            q = self._q(self.net, (obs / self.scale)[None], valid[None])[0]
        return int(q.argmax())

    def _learn(self, target: nn.Module, opt: torch.optim.Optimizer, batch: list[tuple]) -> None:
        """One minibatch update of [WU26 eqs. 35-37]: the squared error to r + gamma max Q_target(o', u')."""
        torch = _torch()
        s, a, r, s2, done, valid2 = (np.array(x) for x in zip(*batch))
        q = self._q(self.net, s, np.ones((len(s), self._actions()), bool)).gather(
            1, torch.as_tensor(a)[:, None]
        )
        with torch.no_grad():
            best = self._q(target, s2, valid2).max(dim=1).values
            best = torch.where(torch.as_tensor(valid2.any(axis=1)), best, torch.zeros_like(best))
            y = torch.as_tensor(r, dtype=torch.float32) + self.config.gamma * best * (
                1.0 - torch.as_tensor(done, dtype=torch.float32)
            )
        loss = torch.nn.functional.mse_loss(q.squeeze(1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # ---- the network --------------------------------------------------------------------------------
    def _actions(self) -> int:
        return self.envs[0].n_actions

    def _net(self) -> nn.Module:
        return q_network(len(self.scale), self.config.hidden, self._actions(), "TrustedPMUsDQN")

    def _q(self, net: nn.Module, states: np.ndarray, valid: np.ndarray) -> torch.Tensor:
        """Q-values with the PMUs no longer on offer masked out."""
        torch = _torch()
        q = net(torch.as_tensor(np.asarray(states), dtype=torch.float32))
        return q.masked_fill(~torch.as_tensor(np.asarray(valid, bool)), -1e9)
