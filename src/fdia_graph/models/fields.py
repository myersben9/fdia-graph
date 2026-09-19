"""The field groups the data bundles share (docs/plans/FIELD_GROUPS_PLAN.md).

Each group is a frozen dataclass mixin (the only behaviour is `ScanFields.node()` / `edge()`, the
named column views): every field defaults to `None` so any
bundle can inherit any combination, and each field's meaning is written here once. A field's
trailing shape is fixed by the group; the leading axis (none for a record, `B` for a batch, `n`
for a split, `T` for a stream) belongs to the bundle and is stated in its docstring. A bundle
names the fields it cannot do without in `_required` and its dict-key order in `_order` (see
`Bundle`).

Types: the same group serves a numpy split (`to_numpy`), a torch record or batch (`FdiaGraph[i]`,
`collate`) and a numpy stream, so an array field is `Array` (a numpy array or a torch tensor) and
a per-record scalar is `Scalars` (one int on a record, a vector on a batch or a split).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .grid import EdgeColumns, NodeColumns

if TYPE_CHECKING:
    import torch

Array = Union[np.ndarray, "torch.Tensor"]
Scalars = Union[int, np.ndarray, "torch.Tensor"]


@dataclass(frozen=True, eq=False)
class ScanFields:
    """One measurement scan: the bus and branch meters and their masks."""

    node_x: Optional[Array] = None  # [..., N, 4] |V|, P_inj, Q_inj, theta (zero where unmetered)
    node_m: Optional[Array] = None  # [..., N, 4] meter mask, 1 metered
    edge_x: Optional[Array] = None  # [..., E, 2] P_from, Q_from
    edge_m: Optional[Array] = None  # [..., E, 2] flow-meter mask

    def node(self) -> NodeColumns:
        """Named views of `node_x`: `.v`, `.p_inj`, `.q_inj`, `.theta` (no copy)."""
        assert self.node_x is not None, "node_x is not set"
        return NodeColumns.of(self.node_x)

    def edge(self) -> EdgeColumns:
        """Named views of `edge_x`: `.p_from`, `.q_from` (no copy)."""
        assert self.edge_x is not None, "edge_x is not set"
        return EdgeColumns.of(self.edge_x)


@dataclass(frozen=True, eq=False)
class LabelFields:
    """The attack labels of a scan."""

    y: Optional[Array] = None  # [..., N] per-bus attack label, 1 attacked
    family: Optional[Scalars] = None  # [...] 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At, 6 Al, 7 Am (timeline)


@dataclass(frozen=True, eq=False)
class RecordIds:
    """Where a shard record came from."""

    stealthy: Optional[Scalars] = None  # [...] 1 for the re-solve families Aq, At, Al, Am
    seq_id: Optional[Scalars] = None  # [...] ramp sequence id, -1 otherwise
    timestep: Optional[Scalars] = None  # [...] position in the source load profile


@dataclass(frozen=True, eq=False)
class TemporalFields:
    """The two temporal features."""

    temporal_delta: Optional[Array] = None  # [..., N, 2] injection change vs the previous pool scan (v0.3+)
    swing: Optional[Array] = None  # [..., N, 2] that change as a z-score of recent change (v0.4.1+)


@dataclass(frozen=True, eq=False)
class CleanFields:
    """The noiseless attack-free truth (v0.7.2+)."""

    clean: Optional[Array] = None  # [..., N, 4] true state at the record's timestep
    edge_clean: Optional[Array] = None  # [..., E, 2] exact true flows on metered branches
    edge_clean_full: Optional[Array] = None  # [..., E, 2] exact true flows on every branch (v0.15.0+)


@dataclass(frozen=True, eq=False)
class GraphFields:
    """The static graph, the same for every record of a system."""

    edge_index: Optional[Array] = None  # [2, E] from and to bus of every branch
    edge_attr: Optional[Array] = None  # [E, 8] per-unit line physics r, x, b, g, gs, bs, tap, shift (v0.5.0+)


@dataclass(frozen=True, eq=False)
class StreamLayers:
    """A stream's attack-removed layer, next to the observed and the clean ones."""

    benign: Optional[np.ndarray] = None  # [T, N, 4] attack removed, noise kept
    edge_benign: Optional[np.ndarray] = None  # [T, E, 2] attack removed, noise kept


__all__ = [
    "Array",
    "Scalars",
    "ScanFields",
    "LabelFields",
    "RecordIds",
    "TemporalFields",
    "CleanFields",
    "GraphFields",
    "StreamLayers",
]
