"""The loader's shared state and the constants its concerns read.

`FdiaGraph` is assembled from five mixins over `DatasetBase` (the same pattern as the generator's
`GridBase`): `GraphMixin` (the static graph), `AdmittanceMixin` (the admittance matrices and clean
flows), `RecordsMixin` (indexing, collate, DataLoader), `ExportMixin` (whole-split arrays and
tables) and `SequenceMixin` (windows and episodes of a timeline file). `DatasetBase` declares
every attribute the constructor sets so each mixin type-checks on its own, and stubs the methods
one mixin calls on another.
"""

from __future__ import annotations

import numbers
import warnings
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

from .. import schema
from ..models.choices import (  # noqa: F401  re-exported beside the code that reads them
    Capability,
    Order,
    RecordFormat,
    Units,
)
from ..models.config import LoadOptions
from ..models.validation import MissingCapability, present

# On-disk `data/family` codes -> display name; the SDK speaks in codes.
from ..schema import (  # noqa: F401  re-exported: the loader's callers import them from here
    FAMILIES,
    STEALTHY_FAMILIES,
    Split,
)
from ..schema import (
    FAMILY_ALIAS as _FAMILY_ALIAS,
)
from ..schema import (
    SPLIT_CODE as _SPLIT,  # noqa: F401  re-exported: the loader reads the codes from here
)
from ..schema import (
    STATIC_PHYSICS as _STATIC_PHYSICS,  # noqa: F401
)


def check_split(split: Optional[str]) -> None:
    """Deprecated: `models.config.LoadOptions` checks the partition."""
    _deprecated_check("check_split")
    LoadOptions(split=split)


def check_units(units: str) -> None:
    """Deprecated: `models.config.LoadOptions` checks the unit system."""
    _deprecated_check("check_units")
    LoadOptions(units=units)


def check_order(order: str) -> None:
    """Deprecated: `models.config.LoadOptions` checks the record order."""
    _deprecated_check("check_order")
    LoadOptions(order=order)


def _deprecated_check(name: str) -> None:
    warnings.warn(
        f"{name} is deprecated; LoadOptions checks the loader's arguments", DeprecationWarning, stacklevel=3
    )


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
        out.append(present(code, f"unknown family {f!r}; known: {sorted(names)} or codes {sorted(FAMILIES)}"))
    return out


def _torch() -> ModuleType:
    # Lazy torch import (+ install hint) so `import fdia_graph` stays light.
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError("PyTorch is required: pip install 'fdia-graph[torch]'") from e


# v0.5.0+ static per-branch physics, bus shunts and per-bus attributes stored under graph/.
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
    "benign",
    "edge_benign",
)
_BATCH_SCALARS = ("family", "stealthy", "seq_id", "timestep")
# Layers stored once per POOL timestep (not per record), resolved through data/timestep.


_CLEAN_LAYERS = ("clean", "edge_clean", "edge_clean_full")
# The attack-removed layer of a timeline file: record field -> dataset path, one row per frame.
_BENIGN_LAYERS = {k: schema.FIELD_PATH[k] for k in ("benign", "edge_benign")}
# The previous frame's readings on a timeline (the row emitted just before each record, whatever
# split or family it belongs to): observed data an operator holds, offered on request only.
_PREV_FIELDS = {
    "prev_node_x": "node_x",
    "prev_edge_x": "edge_x",
    "prev_timestep": "timestep",
    "prev_swing": "swing",
}
# Which unit conversion each returned array takes under units="pu" (masks, labels, swing: none).
_UNIT_KIND = {
    "node_x": "node",
    "prev_node_x": "node",
    "clean": "node",
    "benign": "node",
    "edge_x": "edge",
    "prev_edge_x": "edge",
    "edge_clean": "edge",
    "edge_clean_full": "edge",
    "edge_benign": "edge",
    "temporal_delta": "td",
}


def _is_timeline(ds: DatasetBase) -> bool:
    return ds.is_timeline


def _has_benign(ds: DatasetBase) -> bool:
    return ds.has_benign


def _has_clean(ds: DatasetBase) -> bool:
    return ds.has_clean


def _physical(ds: DatasetBase) -> bool:
    return ds.units == "physical"


def _time_order(ds: DatasetBase) -> bool:
    return ds._perm is None


def _consecutive(ds: DatasetBase) -> bool:
    return not len(ds.idx) or bool(np.all(np.diff(ds.idx) == 1))


# What a view may lack that a consumer needs: the test and what the consumer needs, in words.
CAPABILITIES = {
    Capability.TIMELINE: (_is_timeline, "a timeline file, not a record shard"),
    Capability.BENIGN_LAYER: (_has_benign, "a timeline view with the benign layer"),
    Capability.CLEAN_LAYER: (
        _has_clean,
        'a clean layer (load a timeline or a v0.7.2 record shard, or fit with calibrate="measured")',
    ),
    Capability.PHYSICAL_UNITS: (
        _physical,
        "units='physical' datasets (the default): it converts the stored units itself, so a per-unit "
        "view would be converted twice",
    ),
    Capability.TIME_ORDER: (_time_order, "order='time'; this view is a random permutation"),
    Capability.CONSECUTIVE: (_consecutive, "consecutive frames; a families= or heldout= view is not"),
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
    has_benign: bool
    is_timeline: bool  # the file attribute kind == "timeline" (one row per frame, in time order)
    edge_status_per_record: Optional[np.ndarray]
    _perm: Optional[np.ndarray]  # order="random": view position -> position in idx
    _episodes: Optional[Any]  # EpisodeTable of the whole file, None on a shard
    _f: Optional[h5py.File]
    _phys: dict[str, Any]  # graph/* arrays, None where the file predates the schema
    _clean_np: Optional[np.ndarray]
    _eclean_np: Optional[np.ndarray]
    _eclean_full_np: Optional[np.ndarray]
    _mem: Optional[dict[str, np.ndarray]]

    def require(self, *capabilities: str, by: str) -> None:
        """Refuse this view when it lacks a capability `by` (the consumer, in words) needs; the
        tests and the messages are the `CAPABILITIES` table."""
        for cap in capabilities:
            test, needs = CAPABILITIES[Capability(cap)]
            if not test(self):
                raise MissingCapability(f"{by} needs {needs}")

    def __len__(self) -> int: ...

    # cross-mixin members (defined in the concern mixins)
    @property
    def edge_index(self) -> torch.Tensor: ...

    @property
    def edge_attr(self) -> torch.Tensor: ...

    def _h(self) -> h5py.File: ...

    def _to_units(self, arr: np.ndarray, kind: str) -> np.ndarray: ...

    def _clean_flows_full(self) -> Optional[np.ndarray]: ...

    def export(self, fields: Optional[Sequence[str]] = None, format: str = "numpy") -> Any: ...
