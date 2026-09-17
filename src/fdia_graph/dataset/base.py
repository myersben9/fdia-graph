"""The loader's shared state and the constants its concerns read.

`FdiaGraph` is assembled from four mixins over `DatasetBase` (the same pattern as the generator's
`GridBase`): `GraphMixin` (the static graph), `AdmittanceMixin` (the admittance matrices and clean
flows), `RecordsMixin` (indexing, collate, DataLoader) and `ExportMixin` (whole-split arrays and
tables). `DatasetBase` declares every attribute the constructor sets so each mixin type-checks on
its own, and stubs the methods one mixin calls on another.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    pass


import h5py
import numpy as np

# On-disk `data/family` codes -> display name; the SDK speaks in codes.
FAMILIES = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}
STEALTHY_FAMILIES = {1, 5, 6}  # Aq, At, Al — evade classical bad-data detection
_FAMILY_ALIAS = {"Ao": 1, "SLS": 1, "ramp": 5, "LRA": 6}  # backward-compatible family-name aliases
_SPLIT = {"train": 0, "val": 1, "test": 2}  # on-disk `data/split` codes (precomputed)


def check_split(split):
    """Reject an unknown partition name before any file is opened or downloaded."""
    if split is not None and split not in _SPLIT:
        raise ValueError(f"split must be one of {sorted(_SPLIT)} or None, got {split!r}")


def check_units(units):
    """Reject an unknown unit system before any file is opened or downloaded."""
    if units not in ("physical", "pu"):
        raise ValueError(f"units must be 'physical' or 'pu', got {units!r}")


_HELDOUT_TRAIN_EXCLUDE = {
    3,
    4,
}  # As, Ar reserved for test-only in the unseen-attack protocol (Boyaci et al. 2022)


def family_ids(families: Sequence[Union[str, int]]) -> list[int]:
    """Family names (with the legacy aliases) or raw integer codes as integer codes. An unknown
    name or code is an error rather than a silently empty selection."""
    names = {v: k for k, v in FAMILIES.items()}
    names.update(_FAMILY_ALIAS)
    out: list[int] = []
    for f in families:
        if isinstance(f, str):
            code = names.get(f)
        elif isinstance(f, numbers.Integral) and not isinstance(f, bool) and int(f) in FAMILIES:
            code = int(f)
        else:
            code = None  # a float such as 1.9, a bool, or a code outside the table
        if code is None:
            raise ValueError(f"unknown family {f!r}; known: {sorted(names)} or codes {sorted(FAMILIES)}")
        out.append(code)
    return out


def _torch() -> ModuleType:
    # Lazy torch import (+ install hint) so `import fdia_graph` stays light.
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError("PyTorch is required: pip install 'fdia-graph[torch]'") from e


# v0.5.0+ static per-branch physics, bus shunts and per-bus attributes stored under graph/.
_STATIC_PHYSICS = (
    "edge_r",
    "edge_x",
    "edge_b",
    "edge_g",
    "edge_gs",
    "edge_bs",
    "edge_tap",
    "edge_shift",
    "edge_status",
    "edge_is_trafo",
    "bus_shunt_g",
    "bus_shunt_b",
    "bus_type",
    "bus_vmin",
    "bus_vmax",
    "bus_base_kv",
    "bus_is_zero_inj",
    "bus_has_gen",
    "bus_base_pd",
    "bus_base_qd",
    "bus_attackable",
)
# Per-record tensors collate() stacks into a batch (those the record carries), and the scalars it gathers.


_BATCH_STACKED = (
    "node_x",
    "node_m",
    "edge_x",
    "edge_m",
    "y",
    "temporal_delta",
    "swing",
    "clean",
    "edge_clean",
    "edge_clean_full",
)
_BATCH_SCALARS = ("family", "stealthy", "seq_id", "timestep")
# Layers stored once per POOL timestep (not per record), resolved through data/timestep.


_CLEAN_LAYERS = ("clean", "edge_clean", "edge_clean_full")
# Which unit conversion each returned array takes under units="pu" (masks, labels, swing: none).
_UNIT_KIND = {
    "node_x": "node",
    "clean": "node",
    "edge_x": "edge",
    "edge_clean": "edge",
    "edge_clean_full": "edge",
    "temporal_delta": "td",
}


class DatasetBase:
    """Attributes set by `FdiaGraph.__init__` and read by the mixins."""

    path: str
    format: str
    units: str
    system: int
    N: int
    E: int
    baseMVA: float
    slack: Optional[int]
    idx: np.ndarray
    edge_index_np: np.ndarray
    edge_reactance_np: np.ndarray
    has_physics: bool
    has_temporal: bool
    has_swing: bool
    has_clean: bool
    has_clean_full: bool
    edge_status_per_record: Optional[np.ndarray]
    _f: Optional[h5py.File]
    _phys: dict[str, Any]  # graph/* arrays, None where the file predates the schema
    _clean_np: Optional[np.ndarray]
    _eclean_np: Optional[np.ndarray]
    _eclean_full_np: Optional[np.ndarray]
    _mem: Optional[dict[str, np.ndarray]]

    def __len__(self) -> int: ...

    # cross-mixin members (defined in the concern mixins)
    @property
    def edge_index(self) -> torch.Tensor: ...

    @property
    def edge_attr(self) -> torch.Tensor: ...

    def _h(self) -> h5py.File: ...

    def _to_units(self, arr: np.ndarray, kind: str) -> np.ndarray: ...

    def _clean_flows_full(self) -> Optional[np.ndarray]: ...
