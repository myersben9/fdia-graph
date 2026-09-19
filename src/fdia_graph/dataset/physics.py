"""The admittance matrices built from the static graph, and the clean power flow on every branch derived from
the clean state through them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


import h5py
import numpy as np

from ..formulas.network import Admittances, BranchModel, branch_admittances, branch_flows, complex_voltages
from ..models.grid import NODE
from .base import (
    DatasetBase,
    _torch,
)


class AdmittanceMixin(DatasetBase):
    def _admittances(self) -> Admittances:
        """Ybus [N,N], Yf [E,N] and Yt [E,N] from the stored branch physics, built once and cached.

        Same branch model as pandapower's makeYbus: series admittance, charging split half per end,
        tap ratio and phase shift on the from side, in-service status; bus shunts on the Ybus
        diagonal. Rows of Yf/Yt follow edge_index, columns and Ybus follow node_x bus order. So for a
        complex bus voltage vector V, `V[f] * conj(Yf @ V)` is the from-end branch flow and
        `V * conj(Ybus @ V)` the bus injection, both per-unit. Static: one topology per shard.
        """
        cached = getattr(self, "_adm", None)
        if cached is not None:
            return cached
        need = ("edge_r", "edge_x", "edge_b", "edge_g", "edge_tap", "edge_shift")
        missing = [k for k in need if self._phys.get(k) is None]
        if missing:
            raise AttributeError(
                f"ybus/yf/yt need a v0.5.0+ shard with branch physics; missing graph/{', '.join(missing)}"
            )
        p = self._phys
        self._adm = branch_admittances(
            BranchModel(
                p["edge_r"],
                p["edge_x"],
                p["edge_b"],
                p["edge_g"],
                p["edge_tap"],
                p["edge_shift"],
                p["edge_status"],
            ),
            self.edge_index_np,
            self.N,
            bus_shunt_g=p.get("bus_shunt_g"),
            bus_shunt_b=p.get("bus_shunt_b"),
            base_mva=self.baseMVA,
        )
        return self._adm

    def _clean_flows_full(self) -> Optional[np.ndarray]:
        """[Tpool,E,2] = [P_from, Q_from] exact AC from-end flows on EVERY branch, from the clean state.

        `edge_clean` (stored) zeroes unmetered branches to mirror `edge_x`; a graph model that supervises
        every edge needs the flow on the unmetered ones too. This is the same physics the generator
        used for `edge_clean` (`V[from] * conj(Yf @ V) * baseMVA` with the clean voltages), computed
        once from the shard's own Yf and cached, so it equals `edge_clean` wherever a flow meter
        exists. Physical units (MW, MVAr) like the stored layer; None without the clean layer or the
        branch physics.
        """
        if self._eclean_full_np is not None:
            return self._eclean_full_np
        if not self.has_clean_full or self._clean_np is None:
            return None
        # Only |V| and theta enter the flow; the same physics primitive the generator emits with.
        V = complex_voltages(
            self._clean_np[:, :, NODE.v].astype(np.float64),
            self._clean_np[:, :, NODE.theta].astype(np.float64),
        )
        Sf = branch_flows(
            V, self.yf_np, self.edge_index_np[0], self.baseMVA
        )  # [Tpool,E] from-end complex flow
        self._eclean_full_np = np.stack([Sf.real, Sf.imag], axis=2).astype(np.float32)
        return self._eclean_full_np

    @property
    def ybus_np(self) -> np.ndarray:
        """Full nodal admittance matrix Ybus [N,N], complex per-unit, node_x bus order (see _admittances)."""
        return self._admittances().ybus

    @property
    def yf_np(self) -> np.ndarray:
        """From-end branch admittance matrix Yf [E,N]: `V[f] * conj(Yf @ V)` is the from-end flow."""
        return self._admittances().yf

    @property
    def yt_np(self) -> np.ndarray:
        """To-end branch admittance matrix Yt [E,N]: `V[t] * conj(Yt @ V)` is the to-end flow."""
        return self._admittances().yt

    @property
    def ybus(self) -> torch.Tensor:
        """ybus_np as a complex128 torch tensor [N,N]."""
        return _torch().as_tensor(self.ybus_np)

    @property
    def yf(self) -> torch.Tensor:
        """yf_np as a complex128 torch tensor [E,N]."""
        return _torch().as_tensor(self.yf_np)

    @property
    def yt(self) -> torch.Tensor:
        """yt_np as a complex128 torch tensor [E,N]."""
        return _torch().as_tensor(self.yt_np)

    def _h(self) -> h5py.File:
        # Open+cache the h5py handle on first access (not __init__) so each DataLoader worker gets its
        # OWN handle — fork-safe, no shared-handle crash across processes.
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f
