"""The static graph of a shard: connectivity and the per-branch and per-bus physics, as tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


import numpy as np

from ..formulas.network import series_admittance
from ..schema import Static
from .base import DatasetBase, _torch


class GraphMixin(DatasetBase):
    @property
    def edge_index(self) -> torch.Tensor:
        return _torch().as_tensor(self.edge_index_np)  # [2,E] long connectivity for message passing

    def _p(self, key: str) -> torch.Tensor:
        v = self._phys.get(key)
        if v is None:
            raise AttributeError(f"{key} needs a v0.5.0+ shard; this file predates the physics schema")
        return _torch().as_tensor(v)

    # Per-unit branch physics, ppc order (lines then transformers), aligned with edge_index. The names
    # are branch_*: `edge_x` on every record is the branch flows.
    @property
    def branch_r(self) -> torch.Tensor:
        return self._p(Static.EDGE_R)  # series resistance

    @property
    def branch_x(self) -> torch.Tensor:
        return self._p(Static.EDGE_X)  # series reactance

    @property
    def branch_b(self) -> torch.Tensor:
        return self._p(Static.EDGE_B)  # charging susceptance

    @property
    def branch_g(self) -> torch.Tensor:
        return self._p(Static.EDGE_G)  # charging conductance, transformer iron losses

    def _series_adm(self, imag: bool) -> torch.Tensor:
        # Series admittance 1/(r+jx). Stored on v0.7.0+ shards; derived from r,x for older ones so edge_attr
        # keeps working on every physics shard.
        key = "edge_bs" if imag else "edge_gs"
        v = self._phys.get(key)
        if v is None:
            r, x = self._phys.get("edge_r"), self._phys.get("edge_x")
            if r is None or x is None:
                return self._p(key)  # no physics at all -> standard "needs v0.5.0+" error
            ys = series_admittance(r, x)
            v = np.imag(ys) if imag else np.real(ys)
        return _torch().as_tensor(v)

    @property
    def branch_gs(self) -> torch.Tensor:
        return self._series_adm(False)  # series conductance, Re(1/(r+jx))

    @property
    def branch_bs(self) -> torch.Tensor:
        return self._series_adm(True)  # series susceptance, Im(1/(r+jx))

    @property
    def branch_tap(self) -> torch.Tensor:
        return self._p(Static.EDGE_TAP)  # transformer turns ratio, 1.0 for lines

    @property
    def branch_shift(self) -> torch.Tensor:
        return self._p(Static.EDGE_SHIFT)  # phase shift, degrees

    # Shunts are a BUS property, on the Ybus diagonal, so they were not expressible in an edge schema.
    @property
    def bus_shunt_g(self) -> torch.Tensor:
        return self._p(Static.BUS_SHUNT_G)  # shunt conductance, MW at 1.0 pu

    @property
    def bus_shunt_b(self) -> torch.Tensor:
        return self._p(Static.BUS_SHUNT_B)  # shunt susceptance, MVAr at 1.0 pu

    @property
    def edge_attr(self) -> torch.Tensor:
        """[E,8] stacked per-unit branch features: series impedance (r,x), branch shunt (b,g), series
        admittance (gs,bs = 1/(r+jx)), and transformer tap/shift."""
        columns = [self.branch_r, self.branch_x, self.branch_b, self.branch_g, self.branch_gs, self.branch_bs]
        return _torch().stack([*columns, self.branch_tap, self.branch_shift], dim=1)
