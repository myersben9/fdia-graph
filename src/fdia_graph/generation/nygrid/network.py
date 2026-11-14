"""Admittances of a Case, built once, and the AC quantities they give for any state, from the SDK's network
formulas (`formulas.network`)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from fdia_graph.formulas.network import (
    branch_admittances,
    branch_flows,
    bus_injections,
    complex_voltages,
    edge_adjacency,
    hop_distances,
)
from fdia_graph.generation.nygrid.case import Case
from fdia_graph.models.grid import BranchModel


class Network:
    """Holds a Case with its bus admittance `ybus`, the from-end and to-end branch admittances `yf`
    and `yt`, and the bus adjacency. States are `vm` in p.u. and `va` in degrees, powers come back
    in MW and MVAr, currents in p.u."""

    def __init__(self, case: Case):
        self.case = case
        n = len(case.bus_id)
        self._edges = np.vstack([case.f, case.t])
        branch = BranchModel(case.r, case.x, case.b, np.zeros(len(case.f)), case.tap, case.shift)
        adm = branch_admittances(branch, self._edges, n, case.gs, case.bs, case.base_mva)
        self.ybus, self.yf, self.yt = (sp.csr_matrix(m) for m in adm)
        self.adjacency = edge_adjacency(self._edges, n).astype(int)

    def _v(self, vm: np.ndarray, va: np.ndarray) -> np.ndarray:
        return complex_voltages(vm, va)

    def injections(self, vm: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Net bus injections P, Q (generation minus load), last axis = buses."""
        s = bus_injections(self._v(vm, va), self.ybus, self.case.base_mva)
        return s.real, s.imag

    def flows(self, vm: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Branch flows Pf, Qf, Pt, Qt into each branch at its from and to ends."""
        v = self._v(vm, va)
        sf = branch_flows(v, self.yf, self.case.f, self.case.base_mva)
        st = branch_flows(v, self.yt, self.case.t, self.case.base_mva)
        return sf.real, sf.imag, st.real, st.imag

    def currents(self, vm: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Complex branch currents at the from and to ends."""
        v = self._v(vm, va)
        return v @ self.yf.T, v @ self.yt.T

    def neighbors(self, buses, hops: int) -> np.ndarray:
        """Sorted positions of all buses within `hops` branches of `buses`, the buses included."""
        d = hop_distances(self.adjacency, np.atleast_1d(buses))
        return np.flatnonzero((d >= 0) & (d <= hops))
