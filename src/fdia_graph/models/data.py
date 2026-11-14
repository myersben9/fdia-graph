"""What a user gets back: one record, a batch, a whole split, a summary, a stream, the episode
table, and the per-record truth an estimator is scored against. Each is a Bundle, so it is still
the dict it always was.

The shard-shaped bundles are built from the field groups in `fields.py`, so each field's meaning
is written once; what a bundle adds is its leading axis, which fields it requires, and the order
its dict view keeps (the order the old dict had)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional, TypedDict, Union

import numpy as np

from .base import Bundle
from .fields import (
    CleanFields,
    GraphFields,
    LabelFields,
    PmuCurrentFields,
    PreviousFrameFields,
    RecordIds,
    ScanFields,
    StreamLayers,
    TemporalFields,
)

_SCAN = ("node_x", "node_m", "edge_x", "edge_m")
_TEMPORAL = ("temporal_delta", "swing")
_CLEAN = ("clean", "edge_clean", "edge_clean_full")
_IDS = ("family", "stealthy", "seq_id", "timestep")
_BENIGN = ("benign", "edge_benign")
_PREV = ("prev_node_x", "prev_edge_x", "prev_timestep", "prev_swing")
_PMU_I = ("pmu_i", "pmu_i_m", "pmu_i_benign")


@dataclass(frozen=True, eq=False)
class RecordBundle(
    PmuCurrentFields,
    StreamLayers,
    GraphFields,
    CleanFields,
    TemporalFields,
    RecordIds,
    LabelFields,
    ScanFields,
    Bundle,
):
    """One record as `FdiaGraph[i]` returns it (format="torch"): tensors in self.units with no
    leading axis, the static graph shared by every record, the label and provenance, and the
    optional layers the file carries (the benign layer on a timeline file). A dict as well, so
    DataLoaders, `**item` and `item["node_x"]` keep working."""

    _required = ("edge_index", *_SCAN, "y", *_IDS)
    _order = ("edge_index", *_SCAN, "y", *_IDS, "edge_attr", *_TEMPORAL, *_CLEAN, *_BENIGN, *_PMU_I)


@dataclass(frozen=True, eq=False)
class BatchBundle(
    PmuCurrentFields,
    StreamLayers,
    GraphFields,
    CleanFields,
    TemporalFields,
    RecordIds,
    LabelFields,
    ScanFields,
    Bundle,
):
    """A batch of records as `FdiaGraph.collate` builds it: per-record tensors stacked along a
    leading batch axis B, the static graph once (the first record's), scalar metadata as long
    tensors [B]."""

    _required = (*_SCAN, "y")
    _order = (*_SCAN, "y", *_TEMPORAL, *_CLEAN, *_BENIGN, "edge_index", "edge_attr", *_IDS, *_PMU_I)


@dataclass(frozen=True, eq=False)
class ArraysBundle(
    PmuCurrentFields,
    PreviousFrameFields,
    StreamLayers,
    GraphFields,
    CleanFields,
    TemporalFields,
    RecordIds,
    LabelFields,
    ScanFields,
    Bundle,
):
    """A whole split of n records as `export` returns it (arrays, or tensors with format="torch"
    or "tf"), leading axis n: every per-record field that was requested and the file carries, plus
    the static graph. Fields not requested are absent from the dict view."""

    edge_reactance: Optional[np.ndarray] = None  # [E], deprecated units, kept for old callers

    # edge_attr comes with GraphFields; the exports never fill it, so it is None and absent from the dict
    _order = (
        "edge_index",
        "edge_reactance",
        *_SCAN,
        "y",
        *_TEMPORAL,
        *_CLEAN,
        *_BENIGN,
        *_IDS,
        *_PREV,
        "edge_attr",
        *_PMU_I,
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


class EpisodeRow(TypedDict):
    """One attack episode of a stream: its onset frame, its length in frames, its family code and
    the buses it attacked."""

    onset: int
    length: int
    family: int
    buses: list[int]


class StreamSummary(TypedDict):
    """The two summary fields of a stream, derived from its arrays."""

    system: int  # the bus count
    attacked_frac: float  # the fraction of frames with at least one attacked bus


@dataclass(frozen=True, eq=False)
class Stream(
    PmuCurrentFields,
    StreamLayers,
    GraphFields,
    CleanFields,
    TemporalFields,
    RecordIds,
    LabelFields,
    ScanFields,
    Bundle,
):
    """A continuous attacked time series as `generate_stream` and `load_stream` return it, leading
    axis T: three aligned measurement layers per frame (observed `node_x`, `benign`, `clean`), the
    same three for branch flows, the static graph and meter masks, labels, the two temporal
    features, and the episode list; on a hybrid-meter file also the PMU branch currents (`pmu_i`,
    `pmu_i_benign` per frame, the static mask `pmu_i_m`). `stealthy`, `seq_id` and `edge_clean_full`
    are not part of a stream. A dict as well, so `windows` and every `s["node_x"]` keep working."""

    episodes: Optional[list[EpisodeRow]] = None  # list of {onset, length, family, buses}
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
    _order = (*_required, "system", "attacked_frac", *_PMU_I, "stealthy", "seq_id", "edge_clean_full")


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


# What a stream consumer accepts: the `Stream` that `load_stream` and `generate_stream` return
# (arrays plus the episode list and scalars), or a plain dict of the stream's arrays.
StreamLike = Union[Stream, Mapping[str, np.ndarray]]
