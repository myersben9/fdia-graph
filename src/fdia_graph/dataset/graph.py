"""The static graph of a shard: connectivity and the per-branch and per-bus physics, as tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import warnings

import numpy as np

from .base import (
    DatasetBase,
    _torch,
)


class GraphMixin(DatasetBase):
    @property
    def edge_index(self) -> torch.Tensor:
        return _torch().as_tensor(self.edge_index_np)  # [2,E] long connectivity for message passing

    @property
    def edge_reactance(self) -> torch.Tensor:
        # DEPRECATED: mixes ohms (lines) with vk_percent (transformers), not comparable. Use edge_x (per-unit).
        return _torch().as_tensor(self.edge_reactance_np)

    def _p(self, key: str) -> torch.Tensor:
        v = self._phys.get(key)
        if v is None:
            raise AttributeError(f"{key} needs a v0.5.0+ shard; this file predates the physics schema")
        return _torch().as_tensor(v)

    # Per-unit branch physics, ppc order (lines then transformers), aligned with edge_index. The
    # canonical names are branch_*; the older edge_* spellings still work but warn, because `edge_x`
    # on the dataset (series reactance) collided with `edge_x` on every record (the branch flows).
    @property
    def branch_r(self) -> torch.Tensor:
        return self._p("edge_r")  # series resistance

    @property
    def branch_x(self) -> torch.Tensor:
        return self._p("edge_x")  # series reactance

    @property
    def branch_b(self) -> torch.Tensor:
        return self._p("edge_b")  # charging susceptance

    @property
    def branch_g(self) -> torch.Tensor:
        return self._p("edge_g")  # charging conductance, transformer iron losses

    def _old_name(self, attr: str) -> torch.Tensor:
        # Deprecated edge_* spelling of a branch_* property: same tensor, plus a one-line warning.
        note = " (on records and batches edge_x is the [E,2] branch flows)" if attr == "x" else ""
        warnings.warn(
            f"ds.edge_{attr} is deprecated, use ds.branch_{attr}{note}", DeprecationWarning, stacklevel=3
        )
        return getattr(self, f"branch_{attr}")

    @property
    def edge_r(self) -> torch.Tensor:
        return self._old_name("r")

    @property
    def edge_x(self) -> torch.Tensor:
        return self._old_name("x")

    @property
    def edge_b(self) -> torch.Tensor:
        return self._old_name("b")

    @property
    def edge_g(self) -> torch.Tensor:
        return self._old_name("g")

    def _series_adm(self, imag: bool) -> torch.Tensor:
        # Series admittance 1/(r+jx). Stored on v0.7.0+ shards; derived from r,x for older ones so edge_attr
        # keeps working on every physics shard.
        key = "edge_bs" if imag else "edge_gs"
        v = self._phys.get(key)
        if v is None:
            r, x = self._phys.get("edge_r"), self._phys.get("edge_x")
            if r is None:
                return self._p(key)  # no physics at all -> standard "needs v0.5.0+" error
            z = np.asarray(r) + 1j * np.asarray(x)
            ys = np.zeros_like(z, complex)
            nz = np.abs(z) > 1e-12
            ys[nz] = 1.0 / z[nz]
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
        return self._p("edge_tap")  # transformer turns ratio, 1.0 for lines

    @property
    def branch_shift(self) -> torch.Tensor:
        return self._p("edge_shift")  # phase shift, degrees

    @property
    def edge_gs(self) -> torch.Tensor:
        return self._old_name("gs")

    @property
    def edge_bs(self) -> torch.Tensor:
        return self._old_name("bs")

    @property
    def edge_tap(self) -> torch.Tensor:
        return self._old_name("tap")

    @property
    def edge_shift(self) -> torch.Tensor:
        return self._old_name("shift")

    @property
    def edge_status(self) -> torch.Tensor:
        return self._p("edge_status")  # 1 in service, 0 out (static; see edge_status_per_record)

    @property
    def edge_is_trafo(self) -> torch.Tensor:
        return self._p("edge_is_trafo")

    # Static per-bus attributes, pandapower == ppc bus order (verified identity for case14/118/300).
    @property
    def bus_type(self) -> torch.Tensor:
        return self._p("bus_type")  # 1 PQ, 2 PV, 3 reference

    @property
    def bus_vmin(self) -> torch.Tensor:
        return self._p("bus_vmin")  # voltage limits, pu

    @property
    def bus_vmax(self) -> torch.Tensor:
        return self._p("bus_vmax")

    @property
    def bus_base_kv(self) -> torch.Tensor:
        return self._p("bus_base_kv")  # nominal voltage, kV

    @property
    def bus_is_zero_inj(self) -> torch.Tensor:
        return self._p("bus_is_zero_inj")  # generator's own zero-injection set

    @property
    def bus_has_gen(self) -> torch.Tensor:
        return self._p("bus_has_gen")  # generator or slack on the bus

    @property
    def bus_base_pd(self) -> torch.Tensor:
        return self._p("bus_base_pd")  # base-case load, MW

    @property
    def bus_base_qd(self) -> torch.Tensor:
        return self._p("bus_base_qd")  # base-case load, MVAr

    @property
    def bus_attackable(self) -> torch.Tensor:
        return self._p("bus_attackable")  # carries |p_mw|>0 load (IEEE-300 141/183 rule)

    # Shunts are a BUS property, on the Ybus diagonal, so they were not expressible in an edge schema.
    @property
    def bus_shunt_g(self) -> torch.Tensor:
        return self._p("bus_shunt_g")

    @property
    def bus_shunt_b(self) -> torch.Tensor:
        return self._p("bus_shunt_b")

    @property
    def edge_attr(self) -> torch.Tensor:
        """[E,8] stacked per-unit branch features: series impedance (r,x), branch shunt (b,g), series
        admittance (gs,bs = 1/(r+jx)), and transformer tap/shift. Drop-in replacement for edge_reactance."""
        t = _torch()
        return t.stack(
            [
                self.branch_r,
                self.branch_x,
                self.branch_b,
                self.branch_g,
                self.branch_gs,
                self.branch_bs,
                self.branch_tap,
                self.branch_shift,
            ],
            dim=1,
        )

    @property
    def edge_attr_np(self) -> np.ndarray:
        """`edge_attr` as a float32 numpy array, for the torch-free helpers."""
        return self.edge_attr.numpy().astype(np.float32)
