"""pandapower as a solve engine of a Case: its Newton AC power flow and its interior-point AC OPF.

The net is built from the Case (one load per bus, one generator per unit), so both run on the same
network and limits as the package's own engine (`data/ipopt_solver.py`)."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandapower as pp
from pandapower import LoadflowNotConverged, OPFNotConverged
from pandapower.converter.pypower import from_ppc

from fdia_graph.constants.grid import NO_LIMIT, BusType
from fdia_graph.data.solver import Solver
from fdia_graph.models.grid import OperatingPoint, OpfPoint
from fdia_graph.models.inputs import UnitCosts
from fdia_graph.systems.case import Case

if TYPE_CHECKING:
    from fdia_graph.systems.pandapower_types import PandapowerNet


def _ppc(case: Case) -> dict[str, object]:
    """The network part of a Case as a MATPOWER case, with one placeholder unit at the reference bus."""
    n, nl = len(case.bus_id), len(case.f)
    bus = np.zeros((n, 13))
    bus[:, 0] = np.arange(1, n + 1)
    bus[:, 1] = np.where(case.bus_type == BusType.REF, BusType.REF, 1)
    bus[:, 4], bus[:, 5] = case.gs, case.bs
    # a branch's shunt conductance (a transformer's iron losses) has no MATPOWER column: half at each end
    # as a bus shunt (MW at 1 p.u.), the from end's seen through the tap
    half = 0.5 * case.g * case.base_mva
    np.add.at(bus[:, 4], case.f, half / case.tap**2)
    np.add.at(bus[:, 4], case.t, half)
    bus[:, 6] = case.area
    bus[:, 7], bus[:, 8], bus[:, 9] = case.vm, case.va, case.kv
    bus[:, 10], bus[:, 11], bus[:, 12] = 1, case.vmax, case.vmin
    br = np.zeros((nl, 13))
    br[:, 0], br[:, 1] = case.f + 1, case.t + 1
    br[:, 2], br[:, 3], br[:, 4] = case.r, case.x, case.b
    br[:, 5] = NO_LIMIT  # from_ppc fails on zero ratings; ratings do not enter a solve here
    br[:, 8] = np.where(case.tap == 1.0, 0.0, case.tap)
    br[:, 9], br[:, 10], br[:, 11], br[:, 12] = case.shift, 1, -360, 360
    ref = int(np.flatnonzero(case.bus_type == BusType.REF)[0])
    gen = np.zeros((1, 21))
    gen[0, [0, 5, 7]] = [ref + 1, case.vm[ref], 1]
    gen[0, 3], gen[0, 4], gen[0, 8], gen[0, 9] = NO_LIMIT, -NO_LIMIT, NO_LIMIT, -NO_LIMIT
    return {"version": "2", "baseMVA": case.base_mva, "bus": bus, "branch": br, "gen": gen}


class PandapowerSolver(Solver):
    """Solves one Case in pandapower. `net` is the power-flow net: the case's branches and bus shunts, a
    load at every bus, a generator per unit and the external grid at the reference bus. The OPF runs on
    a copy of it without the external grid."""

    def __init__(self, case: Case) -> None:
        super().__init__(case)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = from_ppc(_ppc(case), f_hz=60, validate_conversion=False)
        net.gen.drop(net.gen.index, inplace=True)
        bus = net.bus.index.to_numpy()  # from_ppc keeps the case's bus order under its own labels
        pp.create_loads(net, buses=bus, p_mw=0.0, q_mvar=0.0)
        pp.create_gens(net, buses=bus[case.unit_bus], p_mw=0.0, vm_pu=1.0)
        self.net = net
        self._opf_net: PandapowerNet | None = None  # built by the first `opf`

    def power_flow(
        self,
        pd: np.ndarray,
        qd: np.ndarray,
        unit_p: np.ndarray,
        unit_on: np.ndarray,
        unit_vm: np.ndarray,
        hvdc_p: np.ndarray,
        start: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> OperatingPoint | None:
        """AC power flow at fixed unit P and voltage setpoints, the reference bus taking the mismatch. `start`
        = (vm, va degrees) starts Newton there (the dispatch's own solution), so it cannot settle on another,
        low-voltage solution of the same equations; otherwise it starts from the last power flow, or flat.
        None when Newton does not converge."""
        g = self.net.gen
        g["p_mw"], g["vm_pu"], g["in_service"] = unit_p, unit_vm, unit_on
        self.net.load["p_mw"], self.net.load["q_mvar"] = self._net_load(pd, hvdc_p), qd
        self.net.ext_grid["vm_pu"] = unit_vm[self.case.unit_bus == self._ref][0]
        if start is not None:
            init = {"init": "auto", "init_vm_pu": start[0], "init_va_degree": start[1]}
        else:
            init = {"init": "results" if self.net.converged else "flat"}
        try:
            pp.runpp(self.net, max_iteration=30, numba=False, **init)
        except LoadflowNotConverged:
            return None
        r = self.net.res_bus
        self.vm, self.va = r.vm_pu.to_numpy().copy(), r.va_degree.to_numpy().copy()
        return OperatingPoint(
            vm=self.vm,
            va=self.va,
            unit_p=self.net.res_gen.p_mw.to_numpy().copy(),
            unit_q=self.net.res_gen.q_mvar.to_numpy().copy(),
        )

    def opf(self, pd: np.ndarray, qd: np.ndarray, costs: UnitCosts) -> OpfPoint | None:
        """The AC point of least cost `costs` for the bus loads `pd`, `qd [N]` (MW, MVAr), by
        `pandapower.runopp` from a flat start: every unit in service free within the case's P and Q limits,
        the voltages within theirs, no branch limit. None when it does not converge."""
        if self._opf_net is None:
            self._opf_net = self._free_units()
        net = self._opf_net
        net.load["p_mw"], net.load["q_mvar"] = pd, qd
        net.poly_cost["cp1_eur_per_mw"], net.poly_cost["cp2_eur_per_mw2"] = costs.linear, costs.quadratic
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runopp(net, numba=False, init="flat")
        except OPFNotConverged:
            return None
        unit_p = np.nan_to_num(net.res_gen.p_mw.to_numpy())  # NaN for a unit out of service
        va = net.res_bus.va_degree.to_numpy()
        self.vm = net.res_bus.vm_pu.to_numpy().copy()
        self.va = va - va[self._ref] + self.case.va[self._ref]
        return OpfPoint(
            vm=self.vm,
            va=self.va,
            unit_p=unit_p,
            unit_q=np.nan_to_num(net.res_gen.q_mvar.to_numpy()),
            cost=costs.of(unit_p),
            shunt_q=np.zeros(len(self.vm)),
        )

    def _free_units(self) -> PandapowerNet:
        """The OPF's net: a copy of `net` with every unit free within the case's limits and a linear cost
        each, the first unit at the reference bus as the angle reference in place of the external grid."""
        c, net = self.case, copy.deepcopy(self.net)
        net.ext_grid.drop(net.ext_grid.index, inplace=True)
        g = net.gen
        g["min_p_mw"], g["max_p_mw"], g["min_q_mvar"], g["max_q_mvar"] = c.pmin, c.pmax, c.qmin, c.qmax
        g["in_service"], g["controllable"] = c.unit_on, True
        g["slack"] = np.arange(len(g)) == np.flatnonzero(c.unit_bus == self._ref)[0]
        pp.create_poly_costs(net, g.index, "gen", cp1_eur_per_mw=0.0)
        return net
