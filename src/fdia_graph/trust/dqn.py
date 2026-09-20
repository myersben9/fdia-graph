"""The trusted-meter selection as a Markov decision process solved by a deep Q-network [WU26,
the learned selection].

State: which meters are secured (a binary vector over the m metered channels). Action: secure
one more meter. Reward: the rise in the attack cost that meter buys, the same kernel the greedy
selector maximizes one step at a time; closing the attack subspace ends the episode with a bonus
of m. An episode secures `k` meters. The Q-network is a two-layer MLP over the state, trained
with epsilon-greedy exploration, a replay buffer and a target network; after training the policy
is rolled out greedily to give the order. The point of the learned policy is that a trained
network picks a k-set in one rollout of m forward passes, where the greedy selection re-evaluates
the attack cost on every candidate at every step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..formulas.trust import attack_cost
from .base import TrustSelector


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError("TrustedMetersDQN needs torch: pip install 'fdia-graph[torch]'") from e


class TrustedMetersDQN(TrustSelector):
    """The DQN selector. `episodes` training episodes of `k` steps each; `gamma` the discount,
    `lr` the Adam step, `hidden` the MLP width, `seed` the exploration and initialization seed.
    The greedy rollout after training is the selection; `history` keeps the return per episode."""

    def __init__(
        self,
        k: int,
        episodes: int = 200,
        gamma: float = 0.95,
        lr: float = 1e-3,
        hidden: int = 128,
        seed: int = 0,
        fa_target: float = 0.01,
    ) -> None:
        super().__init__(k, fa_target)
        self.episodes, self.gamma, self.lr, self.hidden, self.seed = episodes, gamma, lr, hidden, seed
        self.history: list[float] = []

    # ---- the MDP --------------------------------------------------------------------------------
    def _cost(self, secured: np.ndarray) -> float:
        c = attack_cost(self.H, np.flatnonzero(secured))[0]
        return float(self.m) if c == float("inf") else c  # a closed subspace is worth every meter

    def _step(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]:
        """Secure `action`; the reward is the attack-cost rise, the episode ends when closed."""
        before = self._cost(state)
        nxt = state.copy()
        nxt[action] = 1
        after = self._cost(nxt)
        return nxt, after - before, after >= self.m

    # ---- the network ----------------------------------------------------------------------------
    def _net(self) -> Any:
        torch = _torch()
        return torch.nn.Sequential(
            torch.nn.Linear(self.m, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, self.m),
        )

    def _q(self, net: Any, states: np.ndarray) -> Any:
        """Q-values with the secured meters masked out (they cannot be secured twice)."""
        torch = _torch()
        s = torch.as_tensor(states, dtype=torch.float32)
        return net(s).masked_fill(s > 0, -1e9)

    def _act(self, net: Any, state: np.ndarray, eps: float, rng: np.random.Generator) -> int:
        if rng.random() < eps:
            return int(rng.choice(np.flatnonzero(state == 0)))
        torch = _torch()
        with torch.no_grad():
            return int(self._q(net, state[None])[0].argmax())

    def _learn(self, net: Any, target: Any, opt: Any, batch: list[tuple]) -> None:
        torch = _torch()
        s, a, r, s2, done = (np.array(x) for x in zip(*batch))
        q = self._q(net, s).gather(1, torch.as_tensor(a)[:, None]).squeeze(1)
        with torch.no_grad():
            q2 = self._q(target, s2).max(dim=1).values
            y = torch.as_tensor(r, dtype=torch.float32) + self.gamma * q2 * (
                1.0 - torch.as_tensor(done, dtype=torch.float32)
            )
        loss = torch.nn.functional.smooth_l1_loss(q, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # ---- training and rollout ---------------------------------------------------------------------
    def _select(self) -> None:
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        net, target = self._net(), self._net()
        target.load_state_dict(net.state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        replay: list[tuple] = []
        for ep in range(self.episodes):
            eps = max(0.05, 1.0 - ep / max(1, 0.6 * self.episodes))  # explore first, exploit late
            state, total = np.zeros(self.m, np.float32), 0.0
            for _ in range(self.k):
                action = self._act(net, state, eps, rng)
                nxt, reward, done = self._step(state, action)
                replay.append((state, action, reward, nxt, done))
                replay = replay[-5000:]
                total += reward
                state = nxt
                if len(replay) >= 64:
                    self._learn(net, target, opt, [replay[i] for i in rng.integers(len(replay), size=64)])
                if done:
                    break
            self.history.append(total)
            if ep % 10 == 9:
                target.load_state_dict(net.state_dict())
        self.net = net
        self.order, self.cost = self._rollout(net)

    def _rollout(self, net: Any) -> tuple[list[int], list[float]]:
        """The greedy policy from the empty set: the selection and the attack cost after each meter."""
        state, order, cost = np.zeros(self.m, np.float32), [], []
        for _ in range(self.k):
            action = self._act(net, state, 0.0, np.random.default_rng(0))
            state, _, done = self._step(state, action)
            order.append(action)
            c = attack_cost(self.H, np.flatnonzero(state))[0]
            cost.append(c)
            if done:
                break
        return order, cost
