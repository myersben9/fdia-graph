"""What a user gets back: one record, a batch, a whole split, a summary, a stream, the stacked
arrays a shard is written from, and the per-record truth an estimator is scored against. Each is
a Bundle, so it is still the dict it always was.

The shard-shaped bundles are built from the field groups in `fields.py`, so each field's meaning
is written once; what a bundle adds is its leading axis, which fields it requires, and the order
its dict view keeps (the order the old dict had)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import Bundle
from .fields import CleanFields, GraphFields, LabelFields, RecordIds, ScanFields, StreamLayers, TemporalFields

_SCAN = ("node_x", "node_m", "edge_x", "edge_m")
_TEMPORAL = ("temporal_delta", "swing")
_CLEAN = ("clean", "edge_clean", "edge_clean_full")
_IDS = ("family", "stealthy", "seq_id", "timestep")


@dataclass(frozen=True, eq=False)
class RecordBundle(GraphFields, CleanFields, TemporalFields, RecordIds, LabelFields, ScanFields, Bundle):
    """One record as `FdiaGraph[i]` returns it (format="torch"): tensors in self.units with no
    leading axis, the static graph shared by every record, the label and provenance, and the
    optional layers the file carries. A dict as well, so DataLoaders, `**item` and `item["node_x"]`
    keep working."""

    _required = ("edge_index", *_SCAN, "y", *_IDS)
    _order = ("edge_index", *_SCAN, "y", *_IDS, "edge_attr", *_TEMPORAL, *_CLEAN)


@dataclass(frozen=True, eq=False)
class BatchBundle(GraphFields, CleanFields, TemporalFields, RecordIds, LabelFields, ScanFields, Bundle):
    """A batch of records as `FdiaGraph.collate` builds it: per-record tensors stacked along a
    leading batch axis B, the static graph once (the first record's), scalar metadata as long
    tensors [B]."""

    _required = (*_SCAN, "y")
    _order = (*_SCAN, "y", *_TEMPORAL, *_CLEAN, "edge_index", "edge_attr", *_IDS)


@dataclass(frozen=True, eq=False)
class ArraysBundle(GraphFields, CleanFields, TemporalFields, RecordIds, LabelFields, ScanFields, Bundle):
    """A whole split of n records as `to_numpy` (arrays), `to_torch` (tensors) or `to_tf` return
    it, leading axis n: every per-record field that was requested and the file carries, plus the
    static graph. Fields not requested are absent from the dict view."""

    edge_reactance: Optional[np.ndarray] = None  # [E], deprecated units, kept for old callers

    # edge_attr comes with GraphFields; the exports never fill it, so it is None and absent from the dict
    _order = ("edge_index", "edge_reactance", *_SCAN, "y", *_TEMPORAL, *_CLEAN, *_IDS, "edge_attr")


@dataclass(frozen=True, eq=False)
class ShardArrays(Bundle):
    """The stacked arrays of a whole shard as the writer receives them, leading axis n: one row per
    record in the order the records were built. Kept explicit rather than built from the field
    groups: every field is required and numpy, and the writer relies on both."""

    node_x: np.ndarray  # [n, N, 4]
    node_m: np.ndarray  # [n, N, 4]
    edge_x: np.ndarray  # [n, E, 2]
    edge_m: np.ndarray  # [n, E, 2]
    y: np.ndarray  # [n, N]
    temporal_delta: np.ndarray  # [n, N, 2]
    swing: np.ndarray  # [n, N, 2]
    family: np.ndarray  # [n]
    seq_id: np.ndarray  # [n]
    timestep: np.ndarray  # [n]
    gap: np.ndarray  # [n] 1 for a gap (skipped scan) record, else 0
    stealthy: np.ndarray  # [n]


@dataclass(frozen=True, eq=False)
class Stream(
    StreamLayers, GraphFields, CleanFields, TemporalFields, RecordIds, LabelFields, ScanFields, Bundle
):
    """A continuous attacked time series as `generate_stream` and `load_stream` return it, leading
    axis T: three aligned measurement layers per frame (observed `node_x`, `benign`, `clean`), the
    same three for branch flows, the static graph and meter masks, labels, the two temporal
    features, and the episode list. `stealthy`, `seq_id` and `edge_clean_full` are not part of a
    stream. A dict as well, so `windows`, `pyg_stream` and every `s["node_x"]` keep working."""

    episodes: Optional[list[dict[str, Any]]] = None  # list of {onset, length, family, buses}
    system: Optional[int] = None  # bus count (generate_stream and load_stream both set it)
    attacked_frac: Optional[float] = None  # fraction of frames with at least one attacked bus (both set it)

    _required = (
        "node_x",
        "benign",
        "clean",
        "edge_x",
        "edge_benign",
        "edge_clean",
        "edge_index",
        "edge_attr",
        "node_m",
        "edge_m",
        "y",
        "family",
        "temporal_delta",
        "swing",
        "timestep",
        "episodes",
    )
    # the three group fields a stream never fills come last: None, so absent from the dict
    _order = (*_required, "system", "attacked_frac", "stealthy", "seq_id", "edge_clean_full")


@dataclass(frozen=True, eq=False)
class Summary(Bundle):
    """`FdiaGraph.summary()`: the system, its size, the number of records in the view, and the
    record count per family present."""

    system: int  # bus count of the system (14, 118, 300, ...)
    N: int  # buses
    E: int  # branches
    n: int  # records in this view
    families: dict[str, int]  # record count per family name present


@dataclass(frozen=True, eq=False)
class TrueState(Bundle):
    """The per-record truth an estimator is scored against."""

    x: np.ndarray  # [n, 2N-1] the true state: non-slack angles (rad), then every voltage magnitude (pu)
    thsl: np.ndarray  # [n] the slack angle reference per record (rad)
