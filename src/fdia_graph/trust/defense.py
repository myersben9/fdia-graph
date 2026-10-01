"""[WU26]'s trusted-PMU configuration as a Markov decision process (Sec. IV-D2, Fig. 1, Algorithm 1), on
the overload attack of one window (docs/plans/WU_DEFENSE_PLAN.md, PR B).

    env = WuDefenseEnv(g, states, goal, k, WuDefenseConfig(pmus, slots))
    obs = env.reset()
    obs, reward, done = env.step(env.valid().nonzero()[0][0])

Action: trust one more PMU at the next configuration step ("selects one PMU per step", p. 657). Reward:
the rise in the attacker's minimum cost that the trusted PMU buys [WU26 eq. 33], per step, the cost being
the fewest-tamper search's answer (`MinimizeMixin.min_tamper`) under the schedule so far (eqs. 28-32),
counted in measurements (eq. 33's unit) or devices (`WuDefenseConfig.unit`, [E3]). State: what
Fig. 1 lists, the SCADA/PMU measurements P_i, Q_i, P_ij, Q_ij, V_i and theta_i the operator receives at
the step's snapshot, the load rate S_ij / S_max of the target lines, and the trusted configuration.

The search is costly and a training run revisits the same schedules constantly, so every answer is
cached by the schedule (the window is fixed per environment). Ours, where the paper is silent: the
measurements in the state are the noiseless readings of the attack under the schedule so far (the
operator's scan less its noise); an attack the schedule makes infeasible costs every attackable channel
(or every device), the most the attacker could pay; and Algorithm 1's line 8 (a defended cost below the
undefended one ends the episode without a reward) is kept as written and counted in `breaks`, since it can
only fire through the search's own suboptimality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..engine.attacks.minimize import _Window
from ..formulas.attacks import tampered_devices
from ..models.choices import CostUnit, SupportMethod
from ..models.config import TrustSchedule, WuDefenseConfig
from ..models.frames import FlowGoal, FrameKnobs, MinimizerResult
from ..models.inputs import ChosenAction, TrustablePmus, WindowSlots

if TYPE_CHECKING:
    from ..engine.core import FdiaGenerator

_Schedule = tuple[int, ...]  # the actions taken so far: indices into `WuDefenseConfig.pmus`, in order


class WuDefenseEnv:
    """The MDP of one attack window: `reset`, `step`, `valid`, with the cost of every schedule cached."""

    def __init__(
        self,
        g: FdiaGenerator,
        states: list[np.ndarray],
        goal: FlowGoal,
        k: FrameKnobs,
        config: WuDefenseConfig,
    ) -> None:
        TrustablePmus(config.pmus, frozenset(g.meters.pmu), g.C)  # every candidate is a PMU of the plan
        WindowSlots(config.slots, len(states))  # every step falls in the window
        # the oracle is the fewest-tamper search whatever the knobs asked of the overload attack (the
        # row reduction matched Table II worse), minimizing the unit the reward counts (eq. 33)
        self.k = k._replace(objective=config.unit, support_method=SupportMethod.SEARCH.value)
        self.g, self.states, self.goal, self.config = g, states, goal, config
        self.cache: dict[_Schedule, MinimizerResult] = {}
        self.solves = 0  # searches run (cache misses)
        self.breaks = 0  # episodes Algorithm 1's line 8 ended
        self.window = _Window(g, states, goal, self.k)
        self.closed_cost = self._everything()
        self.trusted: _Schedule = ()
        self.undefended = self.cost_of(())
        self.cost = self.undefended

    @property
    def n_actions(self) -> int:
        """How many PMUs the agent chooses among."""
        return len(self.config.pmus)

    def reset(self) -> np.ndarray:
        """Start an episode with no PMU trusted; returns the first state."""
        self.trusted, self.cost = (), self.undefended
        return self.observe()

    def valid(self) -> np.ndarray:
        """Which actions are left: every PMU not yet trusted, none once every step is taken."""
        mask = np.ones(self.n_actions, bool)
        mask[list(self.trusted)] = False
        return mask if len(self.trusted) < len(self.config.slots) else np.zeros(self.n_actions, bool)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Trust PMU `action` at the next step: (the next state, the reward, whether the episode ended).
        It ends when every step is taken, when the attack is infeasible, or at Algorithm 1's line 8."""
        ChosenAction(action, self.valid())  # a PMU still on offer, refused before anything changes
        trusted = (*self.trusted, int(action))
        result = self.result_of(trusted)
        after = self._cost(result)
        if after < self.undefended:  # Algorithm 1, line 8: break, no reward
            self.breaks += 1
            return self.observe(), 0.0, True
        reward = after - self.cost
        self.trusted, self.cost = trusted, after
        done = len(trusted) == len(self.config.slots) or result.devices < 0
        return self.observe(), reward, done

    def cost_of(self, trusted: _Schedule) -> float:
        """The attacker's minimum cost under the schedule `trusted` (indices into the PMUs, in order)."""
        return self._cost(self.result_of(trusted))

    def result_of(self, trusted: _Schedule) -> MinimizerResult:
        """The search's answer under the schedule `trusted`, from the cache when it was asked before."""
        if trusted not in self.cache:
            self.solves += 1
            found = self.g.min_tamper(self.states, self.goal, self.k, trust=self._schedule(trusted))
            assert found is not None, (
                "the goal's lines have an attack area (the environment was built on one)"
            )
            self.cache[trusted] = found
        return self.cache[trusted]

    def observe(self) -> np.ndarray:
        """The state at the next step's snapshot (the last step's once all are taken): the node readings
        P_i, Q_i, |V|_i and theta_i (`models.grid.NODE` columns) and the flows P_ij, Q_ij the operator
        receives under the attack so far, the target lines' load rate, the trusted mask over the PMUs,
        and the step index over the number of steps."""
        slots = self.config.slots
        t = int(slots[min(len(self.trusted), len(slots) - 1)])
        X = self._reported_state(t)
        flows = self.g.all_flows_from_states(X[None])[0]
        lines = list(self.goal.lines)
        rate = np.abs(flows[lines]) / self.g.line_ratings()[lines]  # from the full flows, metered or not
        node = np.where(self.window.node_m > 0, X, 0.0)  # what the meters read, unmetered channels zero
        edge = np.where(self.window.edge_m > 0, np.stack([flows.real, flows.imag], axis=1), 0.0)
        mask = np.zeros(self.n_actions)
        mask[list(self.trusted)] = 1.0
        return np.concatenate(
            [
                node.ravel(),
                edge.ravel(),
                rate,
                mask,
                [len(self.trusted) / len(slots)],
            ]
        )

    def _reported_state(self, t: int) -> np.ndarray:
        """Snapshot t's false state under the schedule so far (its noiseless readings are what the
        operator receives), or the true state when the schedule leaves no attack."""
        result = self.result_of(self.trusted)
        if result.devices < 0:
            return np.asarray(self.states[t], np.float64)
        trust = self._schedule(self.trusted)
        window = _Window(self.g, self.states, self.goal, self.k, trust=trust)  # the schedule's pins
        plan = result.plan or tuple(result.support for _ in window.segments)
        Xa, _ = self.g.goal_state(self.goal, t, self.states[t], window.support_at(t, plan), self.k)
        return np.asarray(self.states[t] if Xa is None else Xa, np.float64)

    def _schedule(self, trusted: _Schedule) -> Optional[TrustSchedule]:
        """The trust schedule of the actions `trusted`, None before the first."""
        if not trusted:
            return None
        pmus, slots = self.config.pmus, self.config.slots
        return TrustSchedule(
            [int(pmus[i]) for i in trusted],
            [int(slots[j]) for j in range(len(trusted))],
            self.config.per_slot,
        )

    def _everything(self) -> float:
        """The cost of an attack the schedule makes infeasible (ours): every channel the plan meters, or
        every device holding one (`tampered_devices` over the full meter masks), the most the attacker
        could ever tamper."""
        w = self.window
        node, edge = w.node_m > 0, w.edge_m > 0
        current = None if w.i_m is None else w.i_m > 0
        if self.config.unit == CostUnit.CHANNELS.value:
            return float(node.sum() + edge.sum() + (0 if current is None else current.sum()))
        ei = self.g.ei
        return float(len(tampered_devices(node, edge, w.pmu, ei[0], current, ei[1])))

    def _cost(self, result: MinimizerResult) -> float:
        """The attack cost of a search answer in the configured unit; an infeasible attack costs all."""
        if result.devices < 0:
            return self.closed_cost
        return float(result.channels if self.config.unit == CostUnit.CHANNELS.value else result.devices)
