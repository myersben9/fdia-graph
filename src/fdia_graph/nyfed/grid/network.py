"""Admittances of a Case, built once, and the AC quantities they give for any state."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from fdia_graph.nyfed.grid.case import Case


class Network:
    """Holds a Case with its bus admittance `ybus`, the from-end and to-end branch admittances `yf`
    and `yt`, and the bus adjacency. States are `vm` in p.u. and `va` in degrees, powers come back
    in MW and MVAr, currents in p.u."""

    def __init__(self, case: Case):
        self.case = case
        n, nl = len(case.bus_id), len(case.f)
        # Branch pi model with an off-nominal tap on the from end (MATPOWER makeYbus).
        ys = 1.0 / (case.r + 1j * case.x)
        a = case.tap * np.exp(1j * np.deg2rad(case.shift))
        ytt = ys + 0.5j * case.b
        yff = ytt / (a * np.conj(a))
        yft = -ys / np.conj(a)
        ytf = -ys / a
        rows = np.arange(nl)
        cf = sp.csr_matrix((np.ones(nl), (rows, case.f)), shape=(nl, n))
        ct = sp.csr_matrix((np.ones(nl), (rows, case.t)), shape=(nl, n))
        self.yf = (sp.diags(yff) @ cf + sp.diags(yft) @ ct).tocsr()
        self.yt = (sp.diags(ytf) @ cf + sp.diags(ytt) @ ct).tocsr()
        ysh = (case.gs + 1j * case.bs) / case.base_mva
        self.ybus = (cf.T @ self.yf + ct.T @ self.yt + sp.diags(ysh)).tocsr()
        adj = (cf.T @ ct).tocsr()
        self.adjacency = ((adj + adj.T) > 0).astype(int).tocsr()

    def _v(self, vm: np.ndarray, va: np.ndarray) -> np.ndarray:
        return vm * np.exp(1j * np.deg2rad(va))

    def injections(self, vm: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Net bus injections P, Q (generation minus load), last axis = buses."""
        v = self._v(vm, va)
        s = v * np.conj(v @ self.ybus.T) * self.case.base_mva
        return s.real, s.imag

    def flows(self, vm: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Branch flows Pf, Qf, Pt, Qt into each branch at its from and to ends."""
        v = self._v(vm, va)
        sf = v[..., self.case.f] * np.conj(v @ self.yf.T) * self.case.base_mva
        st = v[..., self.case.t] * np.conj(v @ self.yt.T) * self.case.base_mva
        return sf.real, sf.imag, st.real, st.imag

    def currents(self, vm: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Complex branch currents at the from and to ends."""
        v = self._v(vm, va)
        return v @ self.yf.T, v @ self.yt.T

    def neighbors(self, buses, hops: int) -> np.ndarray:
        """Sorted positions of all buses within `hops` branches of `buses`, the buses included."""
        reach = np.zeros(self.adjacency.shape[0], dtype=bool)
        reach[np.atleast_1d(buses)] = True
        for _ in range(hops):
            reach = reach | (self.adjacency @ reach.astype(int) > 0)
        return np.flatnonzero(reach)
