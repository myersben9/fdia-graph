"""Admittances of a Case, built once, and the AC quantities they give for any state, from the SDK's network
formulas (`formulas.network`)."""

from __future__ import annotations

from typing import Union

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
from fdia_graph.models.grid import BranchModel
from fdia_graph.systems.case import Case


class Network:
    """Holds a Case with its bus admittance `ybus`, the from-end and to-end branch admittances `yf`
    and `yt`, and the bus adjacency. States are `vm` in p.u. and `va` in degrees, powers come back
    in MW and MVAr, currents in p.u.

    `matpower_order` is for a grid whose stored files were made with MATPOWER's bus admittance (the
    IEEE cases): `ybus` is then summed in MATPOWER's order (`branch_admittances`, `sides_apart`) and
    held by column. Those files depend on its last bit and on the order a product runs over it."""

    def __init__(self, case: Case, matpower_order: bool = False) -> None:
        self.case = case
        n = len(case.bus_id)
        self._edges = np.vstack([case.f, case.t])
        branch = BranchModel(case.r, case.x, case.b, case.g, case.tap, case.shift)
        adm = branch_admittances(branch, self._edges, n, case.gs, case.bs, case.base_mva, matpower_order)
        self.ybus = sp.csc_matrix(adm.ybus) if matpower_order else sp.csr_matrix(adm.ybus)
        self.yf, self.yt = sp.csr_matrix(adm.yf), sp.csr_matrix(adm.yt)
        self.adjacency = edge_adjacency(self._edges, n).astype(int)

    @classmethod
    def from_ieee(cls, system: int) -> Network:
        """The network of the IEEE case of `system` buses, its admittance in MATPOWER's order: the
        stored IEEE files were made with it, to the last bit."""
        return cls(Case.from_ieee(system), matpower_order=True)

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

    def neighbors(self, buses: Union[int, np.ndarray], hops: int) -> np.ndarray:
        """Sorted positions of all buses within `hops` branches of `buses`, the buses included."""
        d = hop_distances(self.adjacency, np.atleast_1d(buses))
        return np.flatnonzero((d >= 0) & (d <= hops))
