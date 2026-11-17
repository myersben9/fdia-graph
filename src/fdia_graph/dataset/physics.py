"""The admittance matrices built from the static graph, and the clean power flow on every branch derived from
the clean state through them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


import h5py
import numpy as np

from ..constants.channels import NodeChannel
from ..formulas.network import branch_admittances, branch_flows, complex_voltages
from ..models.grid import Admittances, BranchModel
from ..schema import BRANCH_MODEL_PHYSICS, Static
from .base import DatasetBase, _torch


class AdmittanceMixin(DatasetBase):
    def _admittances(self) -> Admittances:
        """Ybus [N,N], Yf [E,N] and Yt [E,N], per unit, from the stored branch physics
        (`branch_admittances`), built once and cached. Rows of Yf and Yt follow edge_index; columns
        and Ybus follow the node_x bus order. Static: one topology per file."""
        cached = getattr(self, "_adm", None)
        if cached is not None:
            return cached
        p, missing = self._physics_arrays(tuple(BRANCH_MODEL_PHYSICS.values()))
        if missing:
            raise AttributeError(
                f"ybus/yf/yt need a v0.5.0+ shard with branch physics; missing graph/{', '.join(missing)}"
            )
        by_field = {field: p[name] for field, name in BRANCH_MODEL_PHYSICS.items()}
        branch = BranchModel(**by_field, status=self._phys.get(Static.EDGE_STATUS))
        self._adm = branch_admittances(
            branch,
            self.edge_index_np,
            self.N,
            bus_shunt_g=self._phys.get(Static.BUS_SHUNT_G),
            bus_shunt_b=self._phys.get(Static.BUS_SHUNT_B),
            base_mva=self.baseMVA,
        )
        return self._adm

    def _clean_flows_full(self) -> Optional[np.ndarray]:
        """[Tpool,E,2] = [P_from, Q_from] (MW, MVAr): the exact AC from-end flows on every branch, from
        the clean state through Yf, cached. The stored `edge_clean` zeroes unmetered branches; this
        equals it wherever a flow meter exists. None without the clean layer or the branch physics."""
        if self._eclean_full_np is not None:
            return self._eclean_full_np
        if not self.has_clean_full or self._clean_np is None:
            return None
        # Only |V| and theta enter the flow.
        V = complex_voltages(
            self._clean_np[:, :, NodeChannel.V].astype(np.float64),
            self._clean_np[:, :, NodeChannel.THETA].astype(np.float64),
        )
        # [Tpool,E] from-end complex flow
        Sf = branch_flows(V, self.yf_np, self.edge_index_np[0], self.baseMVA)
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
        # The h5py handle opens on first access, not in __init__, so each DataLoader worker gets its
        # own handle (a handle shared across processes crashes).
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f
