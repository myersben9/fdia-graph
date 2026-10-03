"""[WU26]'s trusted-PMU defense on the faithful attack (`trust.wu26`): the MDP (Sec. IV-D2, Fig. 1,
Algorithm 1) and Solution 1's row reduction (Sec. IV-D1, p. 657).

    env = Wu26DefenseEnv(net, states, area, targets, Wu26Attack(), WuDefenseConfig(pmus, slots, "devices"))
    order = wu26_solution1(net, targets, n=10)        # the PMUs Solution 1 trusts, in order

The MDP is `WuDefenseEnv`'s on the faithful attack: action, trust one more PMU at the next configuration
step; reward, the rise in the attack's cost under the schedule so far [WU26 eq. 33]. The cost is the
devices tampered at any snapshot (Fig. 12's "newly tampered measurement devices"; unit "devices"), or eq.
(33)'s l0 of the window's net attack over channels (unit "channels"). Trust is incremental (E1): each
trusted PMU keeps the offset its bus had in the undefended attack at the snapshot before its step.
Ours, as in `WuDefenseEnv`: an attack the schedule leaves infeasible costs every device (or channel),
Algorithm 1's line 8 is kept as written, and the state's readings are the attacked, noiseless ones.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..formulas.trust import rref
from ..models.choices import CostUnit
from ..models.config import Wu26Attack, WuDefenseConfig
from ..models.frames import WindowAttack
from ..models.inputs import ChosenAction, TrustablePmus, WindowSlots
from .wu26 import Wu26Network, incremental_freeze, solve_window

_Schedule = tuple[int, ...]  # the actions taken so far: indices into `WuDefenseConfig.pmus`, in order


class Wu26DefenseEnv:
    """The MDP of one window of the faithful attack: `reset`, `step`, `valid`, every schedule's attack
    cached. `config.pmus` are bus indices (0-based) of the meter plan's PMUs, `config.slots` 0-based
    snapshots of the window."""

    def __init__(
        self,
        net: Wu26Network,
        states: list[tuple[np.ndarray, np.ndarray]],
        area: Sequence[int],
        targets: Sequence[tuple[int, int]],
        attack: Wu26Attack,
        config: WuDefenseConfig,
    ) -> None:
        TrustablePmus(config.pmus, frozenset(net.pmus), net.n)  # every candidate is a PMU of the plan
        WindowSlots(config.slots, len(states))  # every step falls in the window
        self.net, self.states, self.area, self.targets = net, states, tuple(area), tuple(targets)
        self.attack, self.config = attack, config
        self.solves, self.breaks = 0, 0
        self.cache: dict[_Schedule, WindowAttack] = {}
        self.base = self.result_of(())
        self.undefended = self._cost(self.base)
        self.trusted: _Schedule = ()
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
        """Trust PMU `action` at the next step: (the next state, the reward, whether the episode ended)."""
        ChosenAction(action, self.valid())
        trusted = (*self.trusted, int(action))
        result = self.result_of(trusted)
        after = self._cost(result)
        if after < self.undefended:  # Algorithm 1, line 8: break, no reward
            self.breaks += 1
            return self.observe(), 0.0, True
        reward = after - self.cost
        self.trusted, self.cost = trusted, after
        done = len(trusted) == len(self.config.slots) or not result.feasible
        return self.observe(), reward, done

    def cost_of(self, trusted: _Schedule) -> float:
        """The attack's cost under the schedule `trusted` (indices into the PMUs, in order)."""
        return self._cost(self.result_of(trusted))

    def result_of(self, trusted: _Schedule) -> WindowAttack:
        """The attack under the schedule `trusted`, from the cache when it was asked before."""
        if trusted not in self.cache:
            self.solves += 1
            freeze = None
            if trusted:
                buses = [int(self.config.pmus[i]) for i in trusted]
                freeze = incremental_freeze(buses, list(self.config.slots[: len(trusted)]), self.base)
            self.cache[trusted] = solve_window(
                self.net, self.states, self.area, self.targets, self.attack, freeze
            )
        return self.cache[trusted]

    def observe(self) -> np.ndarray:
        """Fig. 1's state at the next step's snapshot: the readings the operator receives under the attack
        so far, the target lines' load rate against their goal, the trusted mask and the step index."""
        slots = self.config.slots
        t = int(slots[min(len(self.trusted), len(slots) - 1)])
        result = self.result_of(self.trusted)
        Vm, Va = self.states[t]
        readings = self.net.measure(Vm, Va) + result.dz[t]
        rate = []
        for a, b in self.targets:
            k, e = self.net.branch_of(a, b)
            p_row, q_row = self.net.flow_rows(k, e)
            p, q = readings[p_row], readings[q_row]
            rate.append(np.hypot(p, q) / (abs(self.net.flow(Vm, Va, k, e)) * self.attack.rho))
        mask = np.zeros(self.n_actions)
        mask[list(self.trusted)] = 1.0
        return np.concatenate([readings, rate, mask, [len(self.trusted) / len(slots)]])

    def _cost(self, result: WindowAttack) -> float:
        """The attack's cost in the configured unit; an infeasible attack costs every device (channel)."""
        devices = self.config.unit == CostUnit.DEVICES.value
        if not result.feasible:
            return float(
                len(self.net.devices(np.arange(len(self.net.kind)))) if devices else len(self.net.kind)
            )
        return float(len(result.devices_any) if devices else result.channels_net)


def wu26_solution1(
    net: Wu26Network, targets: Sequence[tuple[int, int]], n: int, tol: float = 1e-6
) -> list[int]:
    """Solution 1's trusted PMUs, in order (buses as the paper numbers them), at the case's own state: each
    round row-reduces the transposed Jacobian of the channels the trusted PMUs leave attackable (steps 1-2),
    takes the sparsest attack that moves a target line (steps 3-5; target-aware, ours) and trusts the PMU
    carrying most of its weight (ours, as `TrustedPMUs`); with no such attack, the PMU with most weight
    over every attack."""
    Vm = net.case.res_bus.vm_pu.values.copy()
    Va = np.radians(net.case.res_bus.va_degree.values)
    H = net.jacobian(Vm, Va, np.arange(net.n))
    lines = {net.branch_of(a, b)[0] for a, b in targets}
    trusted: list[int] = []
    for _ in range(min(n, len(net.pmus))):
        open_rows = np.flatnonzero(~((net.dev_type == 1) & np.isin(net.bus, trusted)))
        R, pivots = rref(H[open_rows].T)
        weights = _pmu_weights(net, R[: len(pivots)], open_rows, lines, trusted, tol)
        trusted.append(max(weights, key=lambda b: weights[b]))
    return [b + 1 for b in trusted]


def _pmu_weights(
    net: Wu26Network, rows: np.ndarray, cols: np.ndarray, lines: set[int], trusted: list[int], tol: float
) -> dict[int, float]:
    """Each open PMU's weight in the sparsest attack row that moves a target line, or, with none, its
    weight summed over every row."""
    open_pmus = [b for b in net.pmus if b not in trusted]
    total: dict[int, float] = dict.fromkeys(open_pmus, 0.0)
    for row in sorted(rows, key=lambda r: int((np.abs(r) > tol).sum())):
        supp = np.abs(row) > tol
        w = _row_weights(net, row[supp], cols[supp], open_pmus)
        for b in open_pmus:
            total[b] += w[b]
        if np.isin(net.br[cols[supp]], list(lines)).any() and max(w.values(), default=0.0) > 0:
            return {b: v for b, v in w.items() if v > 0}
    return total


def _row_weights(
    net: Wu26Network, values: np.ndarray, meters: np.ndarray, pmus: list[int]
) -> dict[int, float]:
    """Each PMU's share of an attack row: the absolute entries on its meters."""
    on_pmu = net.dev_type[meters] == 1
    return {b: float(np.abs(values[on_pmu & (net.bus[meters] == b)]).sum()) for b in pmus}
