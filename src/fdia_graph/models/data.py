"""What a user gets back: one record, a batch, a whole split, a summary, the episode
table, and the per-record truth an estimator is scored against. Each is a Bundle, so it is still
the dict it always was.

The record-shaped bundles share the fields of `fields.RecordFields`, so each field's meaning
is written once; what a bundle adds is its leading axis, which fields it requires, and the order
its dict view keeps (the order the old dict had)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..constants.records import COMMON_PREV_FIELDS, CURRENT_FIELDS, ID_FIELDS, LAYER_FIELDS, SCAN_FIELDS
from .base import Bundle
from .fields import Array, RecordFields


@dataclass(frozen=True, eq=False)
class RecordBundle(RecordFields):
    """One record as `FdiaGraph[i]` returns it (format="torch"): tensors in self.units with no
    leading axis, the static graph shared by every record, the label and provenance, and the
    optional layers the file carries (the benign layer on a timeline file). A dict as well, so
    DataLoaders, `**item` and `item["node_x"]` keep working."""

    _required = ("edge_index", *SCAN_FIELDS, "y", *ID_FIELDS)
    _order = ("edge_index", *SCAN_FIELDS, "y", *ID_FIELDS, "edge_attr", *LAYER_FIELDS, *CURRENT_FIELDS)


@dataclass(frozen=True, eq=False)
class BatchBundle(RecordFields):
    """A batch of records as `FdiaGraph.collate` builds it: per-record tensors stacked along a
    leading batch axis B, the static graph once (the first record's), scalar metadata as long
    tensors [B]."""

    _required = (*SCAN_FIELDS, "y")
    _order = (*SCAN_FIELDS, "y", *LAYER_FIELDS, "edge_index", "edge_attr", *ID_FIELDS, *CURRENT_FIELDS)


@dataclass(frozen=True, eq=False)
class ArraysBundle(RecordFields):
    """A whole split of n records as `export` returns it (arrays, or tensors with format="torch"
    or "tf"), leading axis n: every per-record field that was requested and the file carries, plus
    the static graph. Fields not requested are absent from the dict view."""

    # the readings of the frame emitted just before each record on a timeline (export only)
    prev_node_x: Optional[Array] = None  # [..., N, 4] the previous frame's node readings
    prev_edge_x: Optional[Array] = None  # [..., E, 2] the previous frame's branch-flow readings
    prev_timestep: Optional[Array] = None  # [...] the previous frame's pool timestep
    prev_swing: Optional[Array] = None  # [..., N, 2] the previous frame's swing (dimensionless)
    prev_pmu_i: Optional[Array] = None  # [..., E, 4] the previous frame's PMU branch currents (hybrid meters)

    # edge_attr comes with the shared fields; the exports never fill it, so it is None and absent from the dict
    _order = (
        "edge_index",
        *SCAN_FIELDS,
        "y",
        *LAYER_FIELDS,
        *ID_FIELDS,
        *COMMON_PREV_FIELDS,
        "edge_attr",
        *CURRENT_FIELDS,
        "prev_pmu_i",
    )


@dataclass(frozen=True, eq=False)
class EpisodeTable(Bundle):
    """`FdiaGraph.episodes` on a timeline file: one row per attack episode in the view."""

    onset: np.ndarray  # [K] the file row (frame) the episode starts at
    length: np.ndarray  # [K] frames
    family: np.ndarray  # [K] family code
    buses: list[np.ndarray] = field(default_factory=list)  # K arrays, the buses the episode labelled

    def __len__(self) -> int:
        return len(self.onset)


@dataclass(frozen=True, eq=False)
class Summary(Bundle):
    """`FdiaGraph.summary()`: the system, its size, the number of records in the view, and the
    record count per family present."""

    system: int  # bus count of the system (14, 118, 300, ...)
    N: int  # buses
    E: int  # branches
    n: int  # records in this view
    families: dict[str, int]  # record count per family name present
