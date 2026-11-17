"""The package's own OPF engine: one polar AC-OPF model of a Case in Pyomo, solved by Ipopt.

`IpoptSolver.opf` is the plain least-cost solve every engine has. The New York dispatch adds its own
rules on the same model: `project` (the AC point nearest the hourly setpoints) and `dispatch` (least
cost within each frame's limits, with units held near a schedule)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pyomo.environ as pe

from fdia_graph.constants.newyork import HQ_BUS
from fdia_graph.data.ipopt import ipopt
from fdia_graph.data.solver import Solver
from fdia_graph.models.grid import OperatingPoint, OpfPoint
from fdia_graph.models.inputs import UnitCosts
from fdia_graph.systems.case import Case
from fdia_graph.systems.network import Network

# A unit held at a value its last bounds miss by rounding makes Pyomo warn (W1002); the bounds are reset each solve.
logging.getLogger("pyomo.core").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Compensation:
    """Reactive support for a sensitivity of the benign dispatch; the release solves without it. Switched shunt
    banks range over `shunt_lo`, `shunt_hi [N]` (MVAr at 1 p.u., capacitors positive, 0 where a bus has none),
    with output `b vm^2`. The dispatch holds the voltage schedule `v_schedule [N]` (p.u., NaN where a bus does
    not regulate) by a quadratic penalty. `q_lo`, `q_hi [G]` (MVAr) and the band `v_lo`, `v_hi [N]` (p.u., NaN
    keeps the case's limit) replace the case's limits in the dispatch."""

    shunt_lo: np.ndarray
    shunt_hi: np.ndarray
    v_schedule: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray
    v_lo: np.ndarray
    v_hi: np.ndarray
    v_weight: float = 1.0e4  # k$/h per p.u.^2 off the schedule
    shunt_cost: float = 0.005  # k$/h per MVAr switched in, so a bank is not switched in only to cut losses

    def band(self, case: Case) -> tuple[np.ndarray, np.ndarray]:
        """`(lo, hi) [N]` p.u.: the voltage limits the dispatch holds, the case's where the band gives none."""
        lo = np.where(np.isfinite(self.v_lo), self.v_lo, case.vmin)
        return lo, np.where(np.isfinite(self.v_hi), self.v_hi, case.vmax)

    @property
    def shunt_buses(self) -> np.ndarray:
        """Positions of the buses with a bank."""
        return np.flatnonzero((self.shunt_lo != 0) | (self.shunt_hi != 0))


def _opf_model(network: Network, ref: int, comp: Compensation | None = None) -> pe.ConcreteModel:
    """Polar AC-OPF of acproj.m in p.u. and radians, with mutable loads, setpoints and limits so one model
    serves every solve. Two objectives: `deviation`, the squared distance from the setpoints `p0` (MW), and
    `generation_cost`, the linear cost `cost` ($/MWh) in k$/h plus the anchor and `Compensation` penalties."""
    case = network.case
    y = network.ybus.tocoo()
    n, g = len(case.bus_id), len(case.unit_bus)
    lines = [[] for _ in range(n)]
    for i, j, val in zip(y.row, y.col, y.data):
        lines[i].append((int(j), float(val.real), float(val.imag)))
    units = [[] for _ in range(n)]
    for k, b in enumerate(case.unit_bus):
        units[b].append(k)

    m = pe.ConcreteModel()
    m.B, m.G = pe.RangeSet(0, n - 1), pe.RangeSet(0, g - 1)
    m.load_p = pe.Param(m.B, mutable=True, initialize=0.0)
    m.load_q = pe.Param(m.B, mutable=True, initialize=0.0)
    m.p0 = pe.Param(m.G, mutable=True, initialize=0.0)
    m.cost = pe.Param(m.G, mutable=True, initialize=0.0)
    m.anchor = pe.Param(m.G, mutable=True, initialize=0.0)  # MW a unit is held to by a quadratic penalty
    m.anchor_w = pe.Param(m.G, mutable=True, initialize=0.0)  # $/MW^2h, 0 for units free to the cost
    m.vm = pe.Var(m.B, bounds=lambda _, i: (case.vmin[i], case.vmax[i]), initialize=1.0)
    m.va = pe.Var(m.B, initialize=0.0)
    m.va[ref].fix(0.0)
    m.pg = pe.Var(m.G, initialize=0.0)
    m.qg = pe.Var(m.G, initialize=0.0)

    def injection(i, part):
        terms = []
        for j, gij, bij in lines[i]:
            d = m.va[i] - m.va[j]
            if part == "p":
                terms.append(m.vm[j] * (gij * pe.cos(d) + bij * pe.sin(d)))
            else:
                terms.append(m.vm[j] * (gij * pe.sin(d) - bij * pe.cos(d)))
        return m.vm[i] * sum(terms)

    banks = [] if comp is None else comp.shunt_buses.tolist()
    base = case.base_mva
    cap_hi = {b: max(float(comp.shunt_hi[b]), 0.0) / base for b in banks} if comp is not None else {}
    rea_hi = {b: max(-float(comp.shunt_lo[b]), 0.0) / base for b in banks} if comp is not None else {}
    m.S = pe.Set(initialize=banks)
    m.cap = pe.Var(m.S, bounds=lambda _, b: (0.0, cap_hi[b]), initialize=0.0)
    m.rea = pe.Var(m.S, bounds=lambda _, b: (0.0, rea_hi[b]), initialize=0.0)

    def bank(i):
        return (m.cap[i] - m.rea[i]) * m.vm[i] ** 2 if i in cap_hi else 0.0

    m.p_balance = pe.Constraint(
        m.B, rule=lambda _, i: sum(m.pg[k] for k in units[i]) - m.load_p[i] == injection(i, "p")
    )
    m.q_balance = pe.Constraint(
        m.B, rule=lambda _, i: sum(m.qg[k] for k in units[i]) - m.load_q[i] + bank(i) == injection(i, "q")
    )
    m.deviation = pe.Objective(expr=sum((m.pg[k] * case.base_mva - m.p0[k]) ** 2 for k in m.G))
    support = 0.0
    if comp is not None:
        regulated = np.flatnonzero(np.isfinite(comp.v_schedule)).tolist()
        support = comp.v_weight * sum((m.vm[b] - float(comp.v_schedule[b])) ** 2 for b in regulated) + (
            comp.shunt_cost * base * sum(m.cap[b] + m.rea[b] for b in banks)
        )
    m.generation_cost = pe.Objective(
        expr=(
            sum(m.cost[k] * m.pg[k] for k in m.G) * case.base_mva
            + sum(m.anchor_w[k] * (m.pg[k] * case.base_mva - m.anchor[k]) ** 2 for k in m.G)
        )
        / 1000
        + support
    )
    m.generation_cost.deactivate()
    return m


class IpoptSolver(Solver):
    """Solves the OPFs of one Case, one after another. Holds the AC-OPF model and the last solved state,
    which starts the next `project` or `dispatch` (`opf` starts from the case's base state every time).
    `compensation` is None in the release."""

    def __init__(self, case: Case, compensation: Compensation | None = None):
        super().__init__(case)
        self.compensation = compensation
        # the external units, which updateOpCond.m L360 bounds at 1.5 times their setpoint; the HQ import is
        # appended after that line with its own -1100 to 1500 MW (L367-377), so it keeps its case limits
        self._import = (case.fuel == "Import") & (case.bus_id[case.unit_bus] != HQ_BUS)
        self._opf = _opf_model(self.network, self._ref, compensation)
        self._ipopt = ipopt(tol=1e-8, max_iter=300)

    def opf(self, pd: np.ndarray, qd: np.ndarray, costs: UnitCosts) -> OpfPoint | None:
        """The AC point of least cost `costs` for the bus loads `pd`, `qd [N]` (MW, MVAr): every unit in
        service free within the case's P and Q limits, the voltages within theirs, no branch limit, HVDC
        links at zero. It starts from the case's base state, so the point does not depend on the solves
        before it. None when Ipopt finds no optimum."""
        c = self.case
        self.vm, self.va = c.vm.copy(), c.va.copy()
        idle, start = np.zeros(len(c.dc_f)), (c.unit_p, c.unit_q)
        curve = costs.quadratic if costs.quadratic.any() else None
        return self.dispatch(
            pd, qd, costs.linear, c.pmin, c.pmax, c.qmin, c.qmax, c.unit_on, idle, init=start, quadratic=curve
        )

    def _curved_cost(self, quadratic: np.ndarray) -> pe.Objective:
        """The objective of a dispatch whose units have a quadratic cost `quadratic [G]` ($/MW^2h):
        `generation_cost` plus that term, in k$/h. Added to the model by the first solve that has one."""
        m, base = self._opf, self.case.base_mva
        if m.component("curved_cost") is None:
            m.cost2 = pe.Param(m.G, mutable=True, initialize=0.0)
            curve = sum(m.cost2[k] * (m.pg[k] * base) ** 2 for k in m.G) / 1000
            m.curved_cost = pe.Objective(expr=m.generation_cost.expr + curve)
        for k in m.G:
            m.cost2[k] = quadratic[k]
        return m.curved_cost

    def project(
        self, pd: np.ndarray, qd: np.ndarray, unit_p: np.ndarray, unit_on: np.ndarray, hvdc_p: np.ndarray
    ) -> OperatingPoint | None:
        """The AC operating point nearest the setpoints (acproj.m): least squared unit deviation under the AC
        equations and the unit and voltage limits. P limits widen to admit the setpoint; external units range 0
        to 1.5 times it (updateOpCond.m L358-364). Returns vm, va (degrees), unit_p, unit_q, or None (no optimum)."""
        c, m, base = self.case, self._opf, self.case.base_mva
        m.generation_cost.deactivate(), m.deviation.activate()
        for b in m.S:  # the hourly AC point stays NYgrid's, without the sensitivity's banks or band
            m.cap[b].fix(0.0), m.rea[b].fix(0.0)
        if self.compensation is not None:
            for i in m.B:
                m.vm[i].setlb(c.vmin[i]), m.vm[i].setub(c.vmax[i])
        lo = np.minimum(c.pmin, unit_p)
        hi = np.maximum(c.pmax, unit_p)
        imp = self._import & (unit_p > 0)
        lo[imp], hi[imp] = 0.0, 1.5 * unit_p[imp]
        load_p = self._net_load(pd, hvdc_p) / base
        for i in m.B:
            m.load_p[i], m.load_q[i] = load_p[i], qd[i] / base
            m.vm[i].value = min(max(self.vm[i], c.vmin[i]), c.vmax[i])  # Ipopt ends up to 1e-8 past a limit
            if i != self._ref:
                m.va[i].value = np.deg2rad(self.va[i] - self.va[self._ref])
        for k in m.G:
            m.p0[k] = unit_p[k]
            if unit_on[k]:
                m.pg[k].setlb(lo[k] / base), m.pg[k].setub(hi[k] / base), m.pg[k].unfix()
                m.qg[k].setlb(c.qmin[k] / base), m.qg[k].setub(c.qmax[k] / base), m.qg[k].unfix()
                m.pg[k].value = unit_p[k] / base
            else:
                m.pg[k].fix(0.0), m.qg[k].fix(0.0)
        res = self._ipopt.solve(m, load_solutions=False)
        if res.solver.termination_condition != pe.TerminationCondition.optimal:
            return None
        m.solutions.load_from(res)
        self.vm = np.array([pe.value(m.vm[i]) for i in m.B])
        self.va = np.rad2deg([pe.value(m.va[i]) for i in m.B]) + c.va[self._ref]
        return OperatingPoint(
            vm=self.vm,
            va=self.va,
            unit_p=np.array([pe.value(m.pg[k]) for k in m.G]) * base,
            unit_q=np.array([pe.value(m.qg[k]) for k in m.G]) * base,
        )

    def dispatch(
        self,
        pd: np.ndarray,
        qd: np.ndarray,
        cost: np.ndarray,
        p_lo: np.ndarray,
        p_hi: np.ndarray,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
        unit_on: np.ndarray,
        hvdc_p: np.ndarray,
        init: tuple[np.ndarray, np.ndarray] | None = None,
        anchor: np.ndarray | None = None,
        anchor_weight: float = 1000.0,
        quadratic: np.ndarray | None = None,
    ) -> OpfPoint | None:
        """One real-time dispatch: the AC point of least cost `cost [G]` ($/MWh) with unit P in `p_lo`, `p_hi`
        (MW) and Q in `q_lo`, `q_hi` (MVAr). `anchor [G]` (MW, NaN for none) holds a unit near a schedule by a
        penalty of `anchor_weight` $/MW^2h. `quadratic [G]` ($/MW^2h) adds a quadratic cost; None, as the New
        York dispatch passes, leaves the model as it is built. Returns vm, va (degrees), unit_p, unit_q, cost
        ($/h, generation cost only) and shunt_q (MVAr per bus, zero without a `Compensation`), or None."""
        # `init` = (unit_p, unit_q) of the previous dispatch starts Ipopt there: started from an hour's
        # projection far from the frame, it reaches local infeasibility in most frames. At the default
        # anchor weight even a 400 $/MWh price gap moves an anchored unit only 0.2 MW.
        c, m, base = self.case, self._opf, self.case.base_mva
        m.deviation.deactivate(), m.generation_cost.activate()
        if quadratic is not None:
            m.generation_cost.deactivate()
            self._curved_cost(quadratic).activate()
        for b in m.S:
            m.cap[b].unfix(), m.rea[b].unfix()
        v_lo, v_hi = (c.vmin, c.vmax) if self.compensation is None else self.compensation.band(c)
        load_p = self._net_load(pd, hvdc_p) / base
        for i in m.B:
            m.load_p[i], m.load_q[i] = load_p[i], qd[i] / base
            if self.compensation is not None:
                m.vm[i].setlb(v_lo[i]), m.vm[i].setub(v_hi[i])
            m.vm[i].value = min(max(self.vm[i], v_lo[i]), v_hi[i])
            if i != self._ref:
                m.va[i].value = np.deg2rad(self.va[i] - self.va[self._ref])
        p0 = None if init is None else np.asarray(init[0], float) / base
        q0 = None if init is None else np.asarray(init[1], float) / base
        has_anchor = np.zeros(len(cost), bool) if anchor is None else np.isfinite(anchor)
        for k in m.G:
            m.cost[k] = cost[k]
            m.anchor_w[k] = anchor_weight if has_anchor[k] else 0.0
            m.anchor[k] = float(anchor[k]) if has_anchor[k] and anchor is not None else 0.0
            if p0 is not None and q0 is not None:  # both come from `init`
                m.pg[k].value, m.qg[k].value = p0[k], q0[k]
            if not unit_on[k]:
                m.pg[k].fix(0.0), m.qg[k].fix(0.0)
                continue
            m.pg[k].setlb(p_lo[k] / base), m.pg[k].setub(p_hi[k] / base)
            if p_hi[k] - p_lo[k] < 1e-6:
                m.pg[k].fix(p_lo[k] / base)
            else:
                m.pg[k].unfix()
                m.pg[k].value = min(max(m.pg[k].value or 0.0, p_lo[k] / base), p_hi[k] / base)
            m.qg[k].setlb(q_lo[k] / base), m.qg[k].setub(q_hi[k] / base), m.qg[k].unfix()
            m.qg[k].value = min(max(m.qg[k].value or 0.0, q_lo[k] / base), q_hi[k] / base)
        res = self._ipopt.solve(m, load_solutions=False)
        if quadratic is not None:
            m.curved_cost.deactivate()
        if res.solver.termination_condition != pe.TerminationCondition.optimal:
            return None
        m.solutions.load_from(res)
        self.vm = np.array([pe.value(m.vm[i]) for i in m.B])
        self.va = np.rad2deg([pe.value(m.va[i]) for i in m.B]) + c.va[self._ref]
        unit_p = np.array([pe.value(m.pg[k]) for k in m.G]) * base
        curve = 0.0 if quadratic is None else float(quadratic @ unit_p**2)
        shunt_q = np.zeros(len(self.vm))
        for b in m.S:
            shunt_q[b] = (pe.value(m.cap[b]) - pe.value(m.rea[b])) * base * self.vm[b] ** 2
        return OpfPoint(
            vm=self.vm,
            va=self.va,
            unit_p=unit_p,
            unit_q=np.array([pe.value(m.qg[k]) for k in m.G]) * base,
            cost=float(cost @ unit_p) + curve,
            shunt_q=shunt_q,
        )
