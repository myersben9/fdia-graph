"""The loader's shared state and the constants its concerns read.

`FdiaGraph` is assembled from five mixins over `DatasetBase` (the same pattern as the generator's
`GridBase`): `GraphMixin` (the static graph), `AdmittanceMixin` (the admittance matrices and clean
flows), `RecordsMixin` (indexing, collate, DataLoader), `ExportMixin` (whole-split arrays and
tables) and `SequenceMixin` (windows and episodes of a timeline file). `DatasetBase` declares
every attribute the constructor sets so each mixin type-checks on its own, and stubs the methods
one mixin calls on another.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import pandas as pd

    from ..models.data import ArraysBundle, EpisodeTable


import h5py
import numpy as np

from .. import schema
from ..models.choices import FAMILY_ALIAS as _FAMILY_ALIAS  # noqa: F401  kept where it was read before
from ..models.choices import (  # noqa: F401  re-exported beside the code that reads them
    FAMILY_CODE,
    HELDOUT_FAMILIES,
    Capability,
    Order,
    RecordFormat,
    Units,
)
from ..models.config import LoadOptions
from ..models.inputs import FamilySelection, Requirement
from ..models.validation import MissingCapability

# On-disk `data/family` codes -> display name; the SDK speaks in codes.
from ..schema import (  # noqa: F401  re-exported: the loader's callers import them from here
    FAMILIES,
    STEALTHY_FAMILIES,
    Split,
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


# As, Ar reserved for test-only in the unseen-attack protocol (Boyaci et al. 2022)
_HELDOUT_TRAIN_EXCLUDE = {FAMILY_CODE[n] for n in HELDOUT_FAMILIES}


def family_ids(families: Sequence[Union[str, int]]) -> list[int]:
    """Family names (with the legacy aliases) or raw integer codes as integer codes. An unknown
    name or code is an error rather than a silently empty selection."""
    return list(FamilySelection(families).codes)


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
    "pmu_i",
    "pmu_i_m",
    "pmu_i_benign",
)
_BATCH_SCALARS = ("family", "stealthy", "seq_id", "timestep")
# Layers stored once per POOL timestep (not per record), resolved through data/timestep.


_CLEAN_LAYERS = ("clean", "edge_clean", "edge_clean_full")
# The attack-removed layer of a timeline file: record field -> dataset path, one row per frame.
_BENIGN_LAYERS = {k: schema.FIELD_PATH[k] for k in ("benign", "edge_benign")}
# The PMU branch-current layers of a hybrid-meter timeline (D10): record field -> dataset path, one row
# per frame, per unit on every view (a current has no MW form to convert from).
_CURRENT_LAYERS = {k: schema.FIELD_PATH[k] for k in ("pmu_i", "pmu_i_m", "pmu_i_benign")}
# The previous frame's readings on a timeline (the row emitted just before each record, whatever
# split or family it belongs to): observed data an operator holds, offered on request only.
_PREV_FIELDS = {
    "prev_node_x": "node_x",
    "prev_edge_x": "edge_x",
    "prev_timestep": "timestep",
    "prev_swing": "swing",
    "prev_pmu_i": "pmu_i",  # hybrid-meter timelines only
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


def _has_split(ds: DatasetBase) -> bool:
    return ds.has_split


def _has_swing(ds: DatasetBase) -> bool:
    return ds.has_swing


def _has_temporal(ds: DatasetBase) -> bool:
    return ds.has_temporal


def _has_currents(ds: DatasetBase) -> bool:
    return ds.has_currents


def _consecutive(ds: DatasetBase) -> bool:
    return not len(ds.idx) or bool(np.all(np.diff(ds.idx) == 1))


# What a view may lack that a consumer needs: the test and what the consumer needs, in words.
CAPABILITIES: dict[str, tuple[Callable[[DatasetBase], bool], str]] = {
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
    Capability.SPLIT: (_has_split, "a file with the split column"),
    Capability.SWING: (_has_swing, "a file with the swing layer"),
    Capability.TEMPORAL: (_has_temporal, "a file with the temporal_delta layer"),
    Capability.PMU_CURRENTS: (
        _has_currents,
        'a hybrid-meter timeline (meter_model="hybrid"), which carries PMU currents',
    ),
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
    has_currents: bool
    has_split: bool
    is_timeline: bool  # the file attribute kind == "timeline" (one row per frame, in time order)
    edge_status_per_record: Optional[np.ndarray]
    _perm: Optional[np.ndarray]  # order="random": view position -> position in idx
    _episodes: Optional[EpisodeTable]  # the whole file's, None on a shard
    _f: Optional[h5py.File]
    _phys: dict[str, Optional[np.ndarray]]  # graph/* arrays, None where the file predates the schema
    _clean_np: Optional[np.ndarray]
    _eclean_np: Optional[np.ndarray]
    _eclean_full_np: Optional[np.ndarray]
    _mem: Optional[dict[str, np.ndarray]]

    def require(self, *capabilities: str, by: str) -> None:
        """Refuse this view when it lacks a capability `by` (the consumer, in words) needs; the
        tests and the messages are the `CAPABILITIES` table."""
        for cap in Requirement(capabilities, by).capabilities:
            test, needs = CAPABILITIES[cap]
            if not test(self):
                raise MissingCapability(f"{by} needs {needs}")

    def __len__(self) -> int: ...

    # cross-mixin members (defined in the concern mixins)
    @property
    def edge_index(self) -> torch.Tensor: ...

    @property
    def edge_attr(self) -> torch.Tensor: ...

    def _physics_arrays(self, need: Sequence[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        """The graph/* arrays among `need` that the file carries, and the names it lacks (a file that
        predates the physics schema)."""
        got = {k: v for k in need if (v := self._phys.get(k)) is not None}
        return got, [k for k in need if k not in got]

    def _h(self) -> h5py.File: ...

    def _to_units(self, arr: np.ndarray, kind: str) -> np.ndarray: ...

    def _clean_flows_full(self) -> Optional[np.ndarray]: ...

    def export(
        self, fields: Optional[Sequence[str]] = None, format: str = "numpy"
    ) -> Union[ArraysBundle, pd.DataFrame]: ...
