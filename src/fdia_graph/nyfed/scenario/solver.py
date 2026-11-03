"""AC solutions of a Case: the hourly projection (an AC-OPF in Pyomo, solved by Ipopt) and the power flows between
hours (pandapower)."""

from __future__ import annotations

import logging
import os
import sys
import warnings

import numpy as np
import pandapower as pp
import pyomo.environ as pe
from pandapower.converter.pypower import from_ppc

from fdia_graph.nyfed.grid.case import REF, Case
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.grid.nygrid import HQ_BUS

_NO_LIMIT = 9999.0
# A unit held at a value its last bounds miss by rounding makes Pyomo warn (W1002); the bounds are reset each solve.
logging.getLogger("pyomo.core").setLevel(logging.ERROR)


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


def _ipopt() -> pe.SolverFactory:
    """Ipopt from the environment's conda Library folder, whose DLLs it needs on PATH."""
    lib = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(lib) and lib not in os.environ["PATH"]:
        os.environ["PATH"] = lib + os.pathsep + os.environ["PATH"]
    exe = os.path.join(lib, "ipopt.exe")
    solver = pe.SolverFactory("ipopt", executable=exe) if os.path.exists(exe) else pe.SolverFactory("ipopt")
    solver.options.update({"tol": 1e-8, "max_iter": 300})
    return solver


def _opf_model(network: Network, ref: int) -> pe.ConcreteModel:
    """Polar AC-OPF of acproj.m in p.u. and radians: unit outputs `pg`, `qg` meet the loads at every bus
    through the AC power flow equations, at minimum sum of squared deviations from the setpoints `p0`
    (in MW). Loads, setpoints and limits are mutable, so one model serves every hour. A second objective,
    inactive until `AcSolver.dispatch` switches to it, is the linear generation cost `cost` ($/MWh) of the
    units' output, scaled to thousands of dollars per hour."""
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

    m.p_balance = pe.Constraint(
        m.B, rule=lambda _, i: sum(m.pg[k] for k in units[i]) - m.load_p[i] == injection(i, "p")
    )
    m.q_balance = pe.Constraint(
        m.B, rule=lambda _, i: sum(m.qg[k] for k in units[i]) - m.load_q[i] == injection(i, "q")
    )
    m.deviation = pe.Objective(expr=sum((m.pg[k] * case.base_mva - m.p0[k]) ** 2 for k in m.G))
    m.generation_cost = pe.Objective(
        expr=(
            sum(m.cost[k] * m.pg[k] for k in m.G) * case.base_mva
            + sum(m.anchor_w[k] * (m.pg[k] * case.base_mva - m.anchor[k]) ** 2 for k in m.G)
        )
        / 1000
    )
    m.generation_cost.deactivate()
    return m


class AcSolver:
    """Solves one Case hour after hour. Holds its admittances, the AC-OPF model, a pandapower net for
    power flows and the last solved state, which starts the next solve."""

    def __init__(self, case: Case):
        self.case = case
        self.network = Network(case)
        self.vm, self.va = case.vm.copy(), case.va.copy()
        self._ref = int(np.flatnonzero(case.bus_type == REF)[0])
        # the external units, which updateOpCond.m L360 bounds at 1.5 times their setpoint; the HQ import is
        # appended after that line with its own -1100 to 1500 MW (L367-377), so it keeps its case limits (N45)
        self._import = (case.fuel == "Import") & (case.bus_id[case.unit_bus] != HQ_BUS)
        self._opf = _opf_model(self.network, self._ref)
        self._ipopt = _ipopt()
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
        0.2 MW). Returns vm, va (degrees), unit_p, unit_q and cost ($/h, generation cost only), or None when
        Ipopt does not reach an optimum."""
        c, m, base = self.case, self._opf, self.case.base_mva
        m.deviation.deactivate(), m.generation_cost.activate()
        load_p = self._net_load(pd, hvdc_p) / base
        for i in m.B:
            m.load_p[i], m.load_q[i] = load_p[i], qd[i] / base
            m.vm[i].value = min(max(self.vm[i], c.vmin[i]), c.vmax[i])
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
        return {
            "vm": self.vm,
            "va": self.va,
            "unit_p": unit_p,
            "unit_q": np.array([pe.value(m.qg[k]) for k in m.G]) * base,
            "cost": float(cost @ unit_p),
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
