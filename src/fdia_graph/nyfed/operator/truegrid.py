"""The true grid of a day: what the operator's actions do to the flows the attack never changed. An AC power flow
at the true loads for any unit dispatch (`TrueGrid.state`), the true state once a run has acted (`run_state`, the
closed loop), and the interface flows and the worst single 345 kV outage it gives (`change`). A power flow that fails
returns None and its scan is unknown, never safe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.operator.day import OperatorDay
from fdia_graph.nyfed.operator.sced import (
    outage_factors,
    shift_factors,
)
from fdia_graph.nyfed.scenario.solver import AcSolver

if TYPE_CHECKING:  # the run record, whose module imports this one
    from fdia_graph.nyfed.operator.run import Twin


class TrueGrid:
    """The true grid of a day: AC power flows at the true loads for a given unit dispatch, the true
    interface flows and the worst single outage of a New York 345 kV branch."""

    def __init__(self, day: OperatorDay):
        c = day.case
        self.day, self.net = day, Network(c)
        self.solver = AcSolver(c)
        pt, qt = self.net.injections(day.vm, day.va)
        gen = np.zeros_like(pt)
        np.add.at(gen.T, c.unit_bus, np.nan_to_num(day.unit_p).T)
        has_unit = np.zeros(len(c.bus_id), bool)
        has_unit[c.unit_bus] = True
        self.pd, self.qd = gen - pt, np.where(has_unit, 0.0, -qt)  # loads net of the HVDC links
        ny345 = c.is_ny[c.f] & c.is_ny[c.t] & (c.kv[c.f] >= 345) & (c.kv[c.t] >= 345)
        self.outages = np.flatnonzero(ny345)
        self.lodf = outage_factors(c, shift_factors(c), self.outages)
        self._base: dict[int, tuple[np.ndarray, float]] = {}

    def state(
        self, t: int, units: np.ndarray, shed: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The true AC state `(vm, va)` at scan `t` with the units at `units [U]` (MW) and `shed [N]` (MW) taken off
        the true loads, or None where the power flow fails."""
        d = self.day
        q = self.qd[t]
        if shed is not None:  # a shed load takes its reactive demand with it, at its own power factor
            pf = np.divide(self.qd[t], self.pd[t], out=np.zeros_like(self.qd[t]), where=self.pd[t] > 1e-6)
            q = self.qd[t] - shed * pf
        r = self.solver.power_flow(
            self.pd[t] if shed is None else self.pd[t] - shed,
            q,
            units,
            np.ones(len(units), bool),  # every unit in service for Q, as the pool solves it
            d.vm[t, d.case.unit_bus],
            np.zeros(len(d.case.dc_f)),
            start=(d.vm[t], d.va[t]),
        )
        return None if r is None else (r["vm"], r["va"])

    def run_state(self, run: Twin, t: int, share: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """The true state at scan `t` of an operator's `run`: its units at their basepoint plus its dispatch and
        regulation, the neighbours' supply at their units in proportion to `share` (`entry_share`), and the load it
        shed. What the estimator truly sees once the operator has acted (the closed loop, N76); None where the
        power flow fails."""
        units = self.day.unit_p[t] + run.dev[t] + run.reg[t] + run.uncovered[t] * share
        return self.state(t, units, run.shed[t])

    def _flows(self, t: int, units: np.ndarray, shed: np.ndarray | None = None) -> np.ndarray | None:
        x = self.state(t, units, shed)
        return None if x is None else self.net.flows(*x)[0]

    def _measure(self, pf: np.ndarray) -> tuple[np.ndarray, float]:
        """Every interface's flow, and the largest flow any of them carries after the worst single outage."""
        iface = np.array([it.model_flow(pf) for it in self.day.interfaces])
        worst = max(
            float((iface[i] + it.signs @ (self.lodf[it.branches] * pf[self.outages])).max())
            for i, it in enumerate(self.day.interfaces)
        )
        return iface, worst

    def change(self, t: int, units: np.ndarray, shed: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """The true interface flows' and the worst N-1 flow's change from the basepoint at scan `t`, with
        `shed [N]` (MW) taken off the true loads."""
        if t not in self._base:
            pf = self._flows(t, self.day.unit_p[t])
            self._base[t] = (
                self._measure(pf) if pf is not None else (np.full(len(self.day.interfaces), np.nan), np.nan)
            )
        pf = self._flows(t, units, shed)
        if pf is None:
            return np.full(len(self.day.interfaces), np.nan), np.nan
        iface, worst = self._measure(pf)
        return iface - self._base[t][0], worst - self._base[t][1]
