"""AC solutions of a Case: the hourly projection (an AC-OPF in Pyomo, solved by Ipopt), the power flows between
hours (pandapower), and the operator's AC dispatch model (`OperatorOpf`), solved scan by scan."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandapower as pp
import pyomo.environ as pe
from pandapower.converter.pypower import from_ppc

from fdia_graph.nyfed.grid.case import REF, Case
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.grid.nygrid import HQ_BUS
from fdia_graph.nyfed.solvers import ipopt

_NO_LIMIT = 9999.0
# A unit held at a value its last bounds miss by rounding makes Pyomo warn (W1002); the bounds are reset each solve.
logging.getLogger("pyomo.core").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Compensation:
    """Reactive support for a sensitivity of the benign dispatch; the release solves without it. Switched shunt
    banks: at each bus a range `shunt_lo`, `shunt_hi [N]` (MVAr at 1 p.u., capacitors positive, reactors negative,
    both 0 where a bus has none) within which every frame's dispatch sets the bank, its output `b vm^2`. A generator
    voltage schedule `v_schedule [N]` (p.u., NaN at a bus that does not regulate), which the dispatch holds by a
    quadratic penalty of `v_weight` k$/h per p.u.^2, as a unit's voltage regulator holds its schedule until its
    reactive limit binds. `shunt_cost` (k$/h per MVAr switched in) keeps a bank out unless the schedule or the
    reactive limits need it, so banks are not switched in only to cut losses. Unit reactive limits `q_lo`, `q_hi
    [G]` (MVAr, `unit_q_limits`) and the operating band `v_lo`, `v_hi [N]` (p.u., NaN keeps the case's limit) replace
    the case's in the dispatch."""

    shunt_lo: np.ndarray
    shunt_hi: np.ndarray
    v_schedule: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray
    v_lo: np.ndarray
    v_hi: np.ndarray
    v_weight: float = 1.0e4
    shunt_cost: float = 0.005

    def band(self, case: Case) -> tuple[np.ndarray, np.ndarray]:
        """`(lo, hi) [N]` p.u.: the voltage limits the dispatch holds, the case's where the band gives none."""
        return np.where(np.isfinite(self.v_lo), self.v_lo, case.vmin), np.where(
            np.isfinite(self.v_hi), self.v_hi, case.vmax
        )

    @property
    def shunt_buses(self) -> np.ndarray:
        """Positions of the buses with a bank."""
        return np.flatnonzero((self.shunt_lo != 0) | (self.shunt_hi != 0))


PLACEHOLDER_Q = 9999.0  # MVAr: NYgrid's reactive limits of a unit with no data (nygrid_units)


def unit_q_limits(case: Case, lag_pf: float = 0.85, lead_pf: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """`(q_lo, q_hi) [G]` MVAr: each unit's reactive capability at power factor `lag_pf` lagging (producing) and
    `lead_pf` leading (absorbing) at its Pmax, `Pmax tan(acos(pf))`, for the units whose case limits are NYgrid's
    placeholder of plus or minus 9999 MVAr; a unit with real limits (the NPCC case's external units) keeps them.
    The HQ import's Pmax is its 1500 MW import limit."""
    pmax = np.abs(np.asarray(case.pmax, float))
    loose = (np.abs(case.qmax) >= PLACEHOLDER_Q) | (np.abs(case.qmin) >= PLACEHOLDER_Q)
    q_hi = np.where(loose, np.tan(np.arccos(lag_pf)) * pmax, case.qmax)
    q_lo = np.where(loose, -np.tan(np.arccos(lead_pf)) * pmax, case.qmin)
    return q_lo.astype(float), q_hi.astype(float)


def _ppc(case: Case) -> dict:
    """The network part of a Case as a MATPOWER case, with one placeholder unit at the reference bus."""
    n, nl = len(case.bus_id), len(case.f)
    bus = np.zeros((n, 13))
    bus[:, 0] = np.arange(1, n + 1)
    bus[:, 1] = np.where(case.bus_type == REF, REF, 1)
    bus[:, 6] = case.area
    bus[:, 7], bus[:, 8], bus[:, 9] = case.vm, case.va, case.kv
    bus[:, 10], bus[:, 11], bus[:, 12] = 1, case.vmax, case.vmin
    br = np.zeros((nl, 13))
    br[:, 0], br[:, 1] = case.f + 1, case.t + 1
    br[:, 2], br[:, 3], br[:, 4] = case.r, case.x, case.b
    br[:, 5] = _NO_LIMIT  # from_ppc fails on zero ratings; ratings do not enter a power flow
    br[:, 8] = np.where(case.tap == 1.0, 0.0, case.tap)
    br[:, 9], br[:, 10], br[:, 11], br[:, 12] = case.shift, 1, -360, 360
    ref = int(np.flatnonzero(case.bus_type == REF)[0])
    gen = np.zeros((1, 21))
    gen[0, [0, 5, 7]] = [ref + 1, case.vm[ref], 1]
    gen[0, 3], gen[0, 4], gen[0, 8], gen[0, 9] = _NO_LIMIT, -_NO_LIMIT, _NO_LIMIT, -_NO_LIMIT
    return {"version": "2", "baseMVA": case.base_mva, "bus": bus, "branch": br, "gen": gen}


def _opf_model(network: Network, ref: int, comp: Compensation | None = None) -> pe.ConcreteModel:
    """Polar AC-OPF of acproj.m in p.u. and radians: unit outputs `pg`, `qg` meet the loads at every bus
    through the AC power flow equations, at minimum sum of squared deviations from the setpoints `p0`
    (in MW). Loads, setpoints and limits are mutable, so one model serves every hour. A second objective,
    inactive until `AcSolver.dispatch` switches to it, is the linear generation cost `cost` ($/MWh) of the
    units' output, scaled to thousands of dollars per hour. With a `Compensation`, each bank is a capacitor part
    `cap` and a reactor part `rea` (p.u. at 1 p.u. voltage) in its bus's reactive balance, and the cost objective
    adds the voltage schedule's penalty and the banks' switching cost; the deviation objective is unchanged and
    `AcSolver.project` holds the banks out."""
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


class AcSolver:
    """Solves one Case hour after hour. Holds its admittances, the AC-OPF model, a pandapower net for
    power flows and the last solved state, which starts the next solve. `compensation` (None in the release) gives
    the dispatch switched shunt banks and a generator voltage schedule (`Compensation`)."""

    def __init__(self, case: Case, compensation: Compensation | None = None):
        self.case = case
        self.compensation = compensation
        self.network = Network(case)
        self.vm, self.va = case.vm.copy(), case.va.copy()
        self._ref = int(np.flatnonzero(case.bus_type == REF)[0])
        # the external units, which updateOpCond.m L360 bounds at 1.5 times their setpoint; the HQ import is
        # appended after that line with its own -1100 to 1500 MW (L367-377), so it keeps its case limits (N45)
        self._import = (case.fuel == "Import") & (case.bus_id[case.unit_bus] != HQ_BUS)
        self._opf = _opf_model(self.network, self._ref, compensation)
        self._ipopt = ipopt(tol=1e-8, max_iter=300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = from_ppc(_ppc(case), f_hz=60, validate_conversion=False)
        net.gen.drop(net.gen.index, inplace=True)
        bus = net.bus.index.to_numpy()  # from_ppc keeps the case's bus order under its own labels
        pp.create_loads(net, buses=bus, p_mw=0.0, q_mvar=0.0)
        pp.create_gens(net, buses=bus[case.unit_bus], p_mw=0.0, vm_pu=1.0)
        self.net = net

    def _net_load(self, pd: np.ndarray, hvdc_p: np.ndarray) -> np.ndarray:
        p = pd.copy()
        np.add.at(
            p, self.case.dc_f, hvdc_p
        )  # a lossless link draws at its from end and injects at its to end
        np.add.at(p, self.case.dc_t, -hvdc_p)
        return p

    def project(
        self, pd: np.ndarray, qd: np.ndarray, unit_p: np.ndarray, unit_on: np.ndarray, hvdc_p: np.ndarray
    ) -> dict | None:
        """The AC operating point nearest the setpoints (acproj.m): minimize the sum of squared unit
        deviations subject to the AC power flow equations, unit P and Q limits and bus voltage limits.
        P limits widen to admit the setpoint, and external units range 0 to 1.5 times it
        (updateOpCond.m L358-364); the HQ import keeps its -1100 to 1500 MW (L367-377, N45). Units that are off are held at zero. Returns vm, va (degrees),
        unit_p, unit_q, or None when Ipopt does not reach an optimum."""
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
        return {
            "vm": self.vm,
            "va": self.va,
            "unit_p": np.array([pe.value(m.pg[k]) for k in m.G]) * base,
            "unit_q": np.array([pe.value(m.qg[k]) for k in m.G]) * base,
        }

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
    ) -> dict | None:
        """One real-time dispatch (N32): the AC operating point of least generation cost `cost [G]` ($/MWh)
        subject to the AC power flow equations, bus voltage limits, unit outputs within `p_lo`, `p_hi [G]`
        (MW, the caller's Pmin/Pmax intersected with the ramp from the previous dispatch) and reactive
        outputs within `q_lo`, `q_hi [G]` (MVAr). Units that are off, or whose bounds meet, are held. Starts
        from the last solved voltages and from `init` = (unit_p, unit_q), the previous dispatch, else from
        the model's last values; Ipopt reaches local infeasibility in most frames when it starts from an
        hour's projection far from the frame. `anchor [G]` (MW, NaN for none) holds a unit near a schedule
        by a quadratic penalty of `anchor_weight` $/MW^2h, so it leaves the schedule only as far as the AC
        solution needs (the min-deviation rule of acproj.m; at 1000 even a 400 $/MWh price gap moves a unit
        0.2 MW). Returns vm, va (degrees), unit_p, unit_q, cost ($/h, generation cost only) and `shunt_q [N]`
        (MVAr each bank injects, `b vm^2`, zero without a `Compensation`), or None when Ipopt does not reach an
        optimum."""
        c, m, base = self.case, self._opf, self.case.base_mva
        m.deviation.deactivate(), m.generation_cost.activate()
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
        if res.solver.termination_condition != pe.TerminationCondition.optimal:
            return None
        m.solutions.load_from(res)
        self.vm = np.array([pe.value(m.vm[i]) for i in m.B])
        self.va = np.rad2deg([pe.value(m.va[i]) for i in m.B]) + c.va[self._ref]
        unit_p = np.array([pe.value(m.pg[k]) for k in m.G]) * base
        shunt_q = np.zeros(len(self.vm))
        for b in m.S:
            shunt_q[b] = (pe.value(m.cap[b]) - pe.value(m.rea[b])) * base * self.vm[b] ** 2
        return {
            "vm": self.vm,
            "va": self.va,
            "unit_p": unit_p,
            "unit_q": np.array([pe.value(m.qg[k]) for k in m.G]) * base,
            "cost": float(cost @ unit_p),
            "shunt_q": shunt_q,
        }

    def power_flow(
        self,
        pd: np.ndarray,
        qd: np.ndarray,
        unit_p: np.ndarray,
        unit_on: np.ndarray,
        unit_vm: np.ndarray,
        hvdc_p: np.ndarray,
        start: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict | None:
        """AC power flow at fixed unit P and voltage setpoints, the reference bus taking the mismatch. `start`
        = (vm, va degrees) starts Newton there, e.g. at the dispatch's own AC solution, so the power flow
        cannot settle on another (low-voltage) solution of the same equations; otherwise it starts from the
        last power flow."""
        g = self.net.gen
        g["p_mw"], g["vm_pu"], g["in_service"] = unit_p, unit_vm, unit_on
        self.net.load["p_mw"], self.net.load["q_mvar"] = self._net_load(pd, hvdc_p), qd
        self.net.ext_grid["vm_pu"] = unit_vm[self.case.unit_bus == self._ref][0]
        if start is not None:
            init = {"init": "auto", "init_vm_pu": start[0], "init_va_degree": start[1]}
        else:
            init = {"init": "results" if self.net.converged else "dc"}
        try:
            pp.runpp(self.net, max_iteration=30, numba=False, **init)
        except Exception:
            return None
        r = self.net.res_bus
        self.vm, self.va = r.vm_pu.to_numpy().copy(), r.va_degree.to_numpy().copy()
        return {
            "vm": self.vm,
            "va": self.va,
            "unit_p": self.net.res_gen.p_mw.to_numpy().copy(),
            "unit_q": self.net.res_gen.q_mvar.to_numpy().copy(),
        }


SEGMENTS = 3  # the priced segments of the flow an `OperatorOpf` leaves over an interface


@dataclass(frozen=True)
class OpfSolution:
    """One solve of `OperatorOpf`: the units' output `unit_p [G]` (MW), the load shed `shed [N]` (MW), the flow left over
    each interface `over [I]` (MW, its segments summed), the balance left at the reference bus `unmatched` (MW, positive
    short), each interface's AC flow in the model `flow [I]` (MW, before its reading correction), the state `vm`, `va [N]`
    (p.u., degrees), and the duals: `price [N]` of each bus's active balance and `congestion [I]` of each interface row,
    per MW in the objective's units ($/MWh under a cost objective), positive where more load or more room raises the
    objective."""

    unit_p: np.ndarray
    shed: np.ndarray
    over: np.ndarray
    unmatched: float
    flow: np.ndarray
    vm: np.ndarray
    va: np.ndarray
    price: np.ndarray
    congestion: np.ndarray


class OperatorOpf:
    """The operator's AC dispatch model of a Case, built once and solved scan after scan with new loads, bounds and
    weights. Unit outputs meet the bus loads through the AC power flow equations, within bus voltage limits, each unit's
    P bounds and each unit bus's reactive range (its units' together, every unit in service for reactive power as the
    pool solves it). Each enforced interface's AC flow, the signed sum of its branches' from-end flows, plus a reading
    correction stays within its limit or leaves the excess in up to `SEGMENTS` priced segments. Load may be shed at the
    buses allowed to, taking its reactive load at the bus's power factor, and the reference bus's balance may be left
    short or over at a price. The objective is `scale` times: `w_cost` times the units' linear cost, `w_move` per MW each
    movable unit moves from its basepoint, a quadratic penalty holding each anchored unit near its anchor (the pool's
    rule for hydro and external units), and the prices of the shed, the segments and the balance slack."""

    def __init__(self, case: Case, interfaces: list[tuple[np.ndarray, np.ndarray]]):
        self.case, self.network = case, Network(case)
        self.ref = int(np.flatnonzero(case.bus_type == REF)[0])
        self.unit_buses = np.unique(case.unit_bus)
        self.n_if = len(interfaces)
        self._ipopt = ipopt(tol=1e-8, max_iter=500)
        self.model = self._build(interfaces)

    def _build(self, interfaces: list[tuple[np.ndarray, np.ndarray]]) -> pe.ConcreteModel:
        case, net, ref, base = self.case, self.network, self.ref, self.case.base_mva
        y = net.ybus.tocoo()
        n, g, k_if = len(case.bus_id), len(case.unit_bus), len(interfaces)
        lines = [[] for _ in range(n)]
        for i, j, val in zip(y.row, y.col, y.data):
            lines[i].append((int(j), float(val.real), float(val.imag)))
        units = [[] for _ in range(n)]
        for k, b in enumerate(case.unit_bus):
            units[b].append(k)
        yf = net.yf.tocsr()
        m = pe.ConcreteModel()
        m.B, m.G = pe.RangeSet(0, n - 1), pe.RangeSet(0, g - 1)
        m.U = pe.Set(initialize=self.unit_buses.tolist())
        m.I, m.K = pe.RangeSet(0, max(k_if, 1) - 1), pe.RangeSet(0, SEGMENTS - 1)
        param = {"mutable": True, "initialize": 0.0}
        m.load_p, m.load_q, m.qratio, m.shed_w = (pe.Param(m.B, **param) for _ in range(4))
        m.p0, m.cost, m.anchor, m.anchor_w, m.move_w = (pe.Param(m.G, **param) for _ in range(5))
        m.corr, m.lim = pe.Param(m.I, **param), pe.Param(m.I, **param)
        m.seg_price = pe.Param(m.K, **param)
        m.w_cost, m.bal_w, m.scale = pe.Param(**param), pe.Param(**param), pe.Param(**param)
        m.vm = pe.Var(m.B, bounds=lambda _, i: (case.vmin[i], case.vmax[i]), initialize=1.0)
        m.va = pe.Var(m.B, initialize=0.0)
        m.va[ref].fix(0.0)
        m.pg = pe.Var(m.G, initialize=0.0)
        m.qb = pe.Var(m.U, initialize=0.0)
        m.up, m.down = pe.Var(m.G, bounds=(0.0, None)), pe.Var(m.G, bounds=(0.0, None))
        m.shed = pe.Var(m.B, bounds=(0.0, 0.0), initialize=0.0)
        m.over = pe.Var(m.I, m.K, bounds=(0.0, None), initialize=0.0)
        m.short, m.surplus = pe.Var(bounds=(0.0, None)), pe.Var(bounds=(0.0, None))

        def injection(i, part):
            terms = []
            for j, gij, bij in lines[i]:
                d = m.va[i] - m.va[j]
                if part == "p":
                    terms.append(m.vm[j] * (gij * pe.cos(d) + bij * pe.sin(d)))
                else:
                    terms.append(m.vm[j] * (gij * pe.sin(d) - bij * pe.cos(d)))
            return m.vm[i] * sum(terms)

        def slack(i):
            return (m.short - m.surplus) / base if i == ref else 0.0

        m.p_balance = pe.Constraint(
            m.B,
            rule=lambda _, i: sum(m.pg[k] for k in units[i]) + m.shed[i] / base + slack(i) - m.load_p[i]
            == injection(i, "p"),
        )
        m.q_balance = pe.Constraint(
            m.B,
            rule=lambda _, i: (m.qb[i] if i in self.unit_buses else 0.0)
            + m.shed[i] * m.qratio[i] / base
            - m.load_q[i]
            == injection(i, "q"),
        )

        def branch_p(l):  # from-end active flow of branch l (p.u.)
            f, t = int(case.f[l]), int(case.t[l])
            gff, gft, bft = float(yf[l, f].real), float(yf[l, t].real), float(yf[l, t].imag)
            d = m.va[f] - m.va[t]
            return m.vm[f] ** 2 * gff + m.vm[f] * m.vm[t] * (gft * pe.cos(d) + bft * pe.sin(d))

        self._flow = [
            sum(float(s) * branch_p(int(l)) for l, s in zip(br, sg)) * base for br, sg in interfaces
        ]
        m.iface = pe.Constraint(
            m.I,
            rule=lambda _, k: (self._flow[k] + m.corr[k] - sum(m.over[k, j] for j in m.K) <= m.lim[k])
            if k_if
            else pe.Constraint.Skip,
        )
        m.link = pe.Constraint(m.G, rule=lambda _, k: m.pg[k] * base - m.p0[k] - m.up[k] + m.down[k] == 0)
        m.objective = pe.Objective(
            expr=m.scale
            * (
                m.w_cost * sum(m.cost[k] * m.pg[k] for k in m.G) * base
                + sum(m.move_w[k] * (m.up[k] + m.down[k]) for k in m.G)
                + sum(m.anchor_w[k] * (m.pg[k] * base - m.anchor[k]) ** 2 for k in m.G)
                + sum(m.shed_w[i] * m.shed[i] for i in m.B)
                + sum(m.seg_price[j] * m.over[k, j] for k in m.I for j in m.K)
                + m.bal_w * (m.short + m.surplus)
            )
        )
        m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        return m

    def solve(
        self,
        load_p: np.ndarray,
        load_q: np.ndarray,
        *,
        p0: np.ndarray,
        p_lo: np.ndarray,
        p_hi: np.ndarray,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
        cost: np.ndarray,
        move_w: np.ndarray,
        anchor: np.ndarray,
        anchor_w: np.ndarray,
        shed_ub: np.ndarray,
        shed_w: float,
        qratio: np.ndarray,
        corr: np.ndarray,
        lim: np.ndarray,
        seg_width: tuple[float, ...],
        seg_price: tuple[float, ...],
        w_cost: float,
        bal_w: float,
        scale: float,
        start: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> OpfSolution | None:
        """Solve one scan. Loads `load_p`, `load_q [N]` (MW, MVAr); each unit's basepoint `p0 [G]` (the anchor of an
        anchored unit, the point a movable unit's movement is counted from) and bounds `p_lo`, `p_hi [G]` (MW, a unit
        whose bounds meet is held); each unit bus's reactive range `q_lo`, `q_hi` (MVAr, over `unit_buses`); the units'
        cost [G] ($/MWh), the movement weight `move_w [G]` (per MW, 0 for a unit that is not counted), the `anchor
        [G]` (MW) each anchored unit is held near and its weight `anchor_w [G]` (per MW^2, 0 for none); the load each bus may shed `shed_ub [N]` (MW) at `shed_w` per MW
        with its reactive load per MW `qratio [N]`; each interface's reading correction `corr [I]` and limit `lim [I]`
        (MW, NaN unenforced), the segments' widths and prices; and the objective's weights. `start` = (vm, va degrees,
        unit_p MW) starts Ipopt. Returns None where Ipopt reaches no optimum."""
        m, case, base, ref = self.model, self.case, self.case.base_mva, self.ref
        vm0, va0, pg0 = start
        for i in m.B:
            m.load_p[i], m.load_q[i] = float(load_p[i]) / base, float(load_q[i]) / base
            m.qratio[i], m.shed_w[i] = float(qratio[i]), float(shed_w)
            m.shed[i].setub(max(float(shed_ub[i]), 0.0))
            m.shed[i].value = 0.0
            m.vm[i].value = min(max(float(vm0[i]), case.vmin[i]), case.vmax[i])
            if i != ref:
                m.va[i].value = float(np.deg2rad(va0[i] - va0[ref]))
        for k in m.G:
            m.p0[k], m.cost[k] = float(p0[k]), float(cost[k])
            m.move_w[k], m.anchor_w[k], m.anchor[k] = float(move_w[k]), float(anchor_w[k]), float(anchor[k])
            lo, hi = float(p_lo[k]), float(p_hi[k])
            if hi - lo < 1e-6:  # the last solve's bounds would make the held value look infeasible
                m.pg[k].setlb(None)
                m.pg[k].setub(None)
                m.pg[k].fix(lo / base)
            else:
                m.pg[k].unfix()
                m.pg[k].setlb(lo / base)
                m.pg[k].setub(hi / base)
                m.pg[k].value = min(max(float(pg0[k]), lo), hi) / base
            d = float(pg0[k]) - float(p0[k]) if hi - lo >= 1e-6 else lo - float(p0[k])
            m.up[k].value, m.down[k].value = max(d, 0.0), max(-d, 0.0)
        for j, b in enumerate(self.unit_buses):
            m.qb[int(b)].setlb(float(q_lo[j]) / base)
            m.qb[int(b)].setub(float(q_hi[j]) / base)
        for k in range(self.n_if):
            on = bool(np.isfinite(lim[k]))
            m.corr[k], m.lim[k] = float(np.nan_to_num(corr[k])), float(np.nan_to_num(lim[k]))
            if on:
                m.iface[k].activate()
            else:
                m.iface[k].deactivate()
            for j in m.K:
                m.over[k, j].setub(None if not np.isfinite(seg_width[j]) else float(seg_width[j]))
                m.over[k, j].value = 0.0
        for j in m.K:
            m.seg_price[j] = float(seg_price[j])
        m.w_cost, m.bal_w, m.scale = float(w_cost), float(bal_w), float(scale)
        m.short.value, m.surplus.value = 0.0, 0.0
        res = self._ipopt.solve(m, load_solutions=False)
        if res.solver.termination_condition != pe.TerminationCondition.optimal:
            return None
        m.solutions.load_from(res)
        n_if = self.n_if
        dual = m.dual
        on = [k for k in range(n_if) if m.iface[k].active]
        congestion = np.full(n_if, np.nan)
        for (
            k
        ) in on:  # Ipopt's multiplier of a binding upper row is negative: the objective falls with more room
            congestion[k] = -float(dual.get(m.iface[k], 0.0))
        return OpfSolution(
            unit_p=np.array([pe.value(m.pg[k]) for k in m.G]) * base,
            shed=np.array([pe.value(m.shed[i]) for i in m.B]),
            over=np.array([sum(pe.value(m.over[k, j]) for j in m.K) for k in range(n_if)]),
            unmatched=float(pe.value(m.short) - pe.value(m.surplus)),
            flow=np.array([pe.value(f) for f in self._flow]),
            vm=np.array([pe.value(m.vm[i]) for i in m.B]),
            va=np.rad2deg([pe.value(m.va[i]) for i in m.B]) + float(va0[ref]),
            # the balance row reads supply less load: one more MW of load is a MW less on its right side
            price=np.array([float(dual.get(m.p_balance[i], 0.0)) for i in m.B]) / base,
            congestion=congestion,
        )
