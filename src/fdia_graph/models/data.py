"""What a user gets back: one record, a batch, a whole split, a summary, a stream, the stacked
arrays a shard is written from, and the per-record truth an estimator is scored against. Each is
a Bundle, so it is still the dict it always was."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .base import Bundle


@dataclass(frozen=True, eq=False)
class RecordBundle(Bundle):
    """One record as `FdiaGraph[i]` returns it (format="torch"): tensors in self.units, the static
    graph shared by every record, the label and provenance, and the optional layers the file carries.
    A dict as well, so DataLoaders, `**item` and `item["node_x"]` keep working."""

    edge_index: Any  # [2, E] long, the same tensor for every record
    node_x: Any  # [N, 4] |V|, P_inj, Q_inj, theta
    node_m: Any  # [N, 4] meter mask
    edge_x: Any  # [E, 2] P_from, Q_from
    edge_m: Any  # [E, 2] flow-meter mask
    y: Any  # [N] per-bus attack label
    family: int  # 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At, 6 Al
    stealthy: int  # 1 for the re-solve families Aq, At, Al
    seq_id: int  # source sequence of the record
    timestep: int  # position in the source load profile
    edge_attr: Any = None  # [E, 8] per-unit line physics (v0.5.0+ shards)
    temporal_delta: Any = None  # [N, 2] injection change vs the previous pool scan (v0.3+)
    swing: Any = None  # [N, 2] that change as a z-score of the bus's typical recent change (v0.4.1+)
    clean: Any = None  # [N, 4] noiseless attack-free truth at the record's timestep (v0.7.2+)
    edge_clean: Any = None  # [E, 2] exact true flows on metered branches
    edge_clean_full: Any = None  # [E, 2] exact true flows on every branch


@dataclass(frozen=True, eq=False)
class BatchBundle(Bundle):
    """A batch of records as `FdiaGraph.collate` builds it: per-record tensors stacked along a
    leading batch axis B, the static graph once, scalar metadata as long tensors."""

    node_x: Any  # [B, N, 4]
    node_m: Any  # [B, N, 4]
    edge_x: Any  # [B, E, 2]
    edge_m: Any  # [B, E, 2]
    y: Any  # [B, N]
    temporal_delta: Any = None  # [B, N, 2]
    swing: Any = None  # [B, N, 2]
    clean: Any = None  # [B, N, 4]
    edge_clean: Any = None  # [B, E, 2]
    edge_clean_full: Any = None  # [B, E, 2]
    edge_index: Any = None  # [2, E], the first record's (the same for every record)
    edge_attr: Any = None  # [E, 8], the first record's
    family: Any = None  # [B] long
    stealthy: Any = None  # [B] long
    seq_id: Any = None  # [B] long
    timestep: Any = None  # [B] long


@dataclass(frozen=True, eq=False)
class ArraysBundle(Bundle):
    """A whole split of n records as `to_numpy` (arrays), `to_torch` (tensors) or `to_tf` return
    it: every per-record field that was requested and the file carries, plus the static graph.
    Fields not requested are absent from the dict view."""

    edge_index: Any = None  # [2, E]
    edge_reactance: Any = None  # [E], deprecated units, kept for old callers
    node_x: Any = None  # [n, N, 4]
    node_m: Any = None  # [n, N, 4]
    edge_x: Any = None  # [n, E, 2]
    edge_m: Any = None  # [n, E, 2]
    y: Any = None  # [n, N]
    temporal_delta: Any = None  # [n, N, 2]
    swing: Any = None  # [n, N, 2]
    clean: Any = None  # [n, N, 4]
    edge_clean: Any = None  # [n, E, 2]
    edge_clean_full: Any = None  # [n, E, 2]
    family: Any = None  # [n]
    stealthy: Any = None  # [n]
    seq_id: Any = None  # [n]
    timestep: Any = None  # [n]


@dataclass(frozen=True, eq=False)
class Summary(Bundle):
    """`FdiaGraph.summary()`: the system, its size, the number of records in the view, and the
    record count per family present."""

    system: int  # bus count of the system (14, 118, 300, ...)
    N: int  # buses
    E: int  # branches
    n: int  # records in this view
    families: Dict[str, int]  # record count per family name present


@dataclass(frozen=True, eq=False)
class ShardArrays(Bundle):
    """The records of a shard stacked into the arrays the file stores, one field per dataset."""

    node_x: np.ndarray  # [T, N, 4]
    node_m: np.ndarray  # [T, N, 4]
    edge_x: np.ndarray  # [T, E, 2]
    edge_m: np.ndarray  # [T, E, 2]
    y: np.ndarray  # [T, N]
    temporal_delta: np.ndarray  # [T, N, 2]
    swing: np.ndarray  # [T, N, 2]
    family: np.ndarray  # [T] int8
    seq_id: np.ndarray  # [T] int32
    timestep: np.ndarray  # [T] int32
    gap: np.ndarray  # [T] uint8
    stealthy: np.ndarray  # [T] uint8


@dataclass(frozen=True, eq=False)
class Stream(Bundle):
    """A continuous attacked time series as `generate_stream` and `load_stream` return it: three
    aligned measurement layers per frame, the same three for branch flows, the static graph and
    meter masks, labels, the two temporal features, and the episode list. A dict as well, so
    `windows`, `pyg_stream` and every `s["node_x"]` keep working."""

    node_x: np.ndarray  # [T, N, 4] observed
    benign: np.ndarray  # [T, N, 4] attack removed, noise kept
    clean: np.ndarray  # [T, N, 4] noiseless truth
    edge_x: np.ndarray  # [T, E, 2] observed flows
    edge_benign: np.ndarray  # [T, E, 2] attack removed, noise kept
    edge_clean: np.ndarray  # [T, E, 2] noiseless true flows
    edge_index: np.ndarray  # [2, E]
    edge_attr: np.ndarray  # [E, 8]
    node_m: np.ndarray  # [N, 4]
    edge_m: np.ndarray  # [E, 2]
    y: np.ndarray  # [T, N]
    family: np.ndarray  # [T]
    temporal_delta: np.ndarray  # [T, N, 2]
    swing: np.ndarray  # [T, N, 2]
    timestep: np.ndarray  # [T]
    episodes: Any  # list of {onset, length, family, buses}
    system: Optional[int] = None  # bus count, set by generate_stream
    attacked_frac: Optional[float] = None  # fraction of frames with an attacked bus, set by generate_stream


@dataclass(frozen=True, eq=False)
class TrueState(Bundle):
    """The true state of a batch of records from the clean layer: x [n, 2N-1] = [theta (rad, non-slack)
    | V (pu, every bus)] and the slack angle reference thsl [n] (rad) the solver pins."""

    x: np.ndarray
    thsl: np.ndarray
