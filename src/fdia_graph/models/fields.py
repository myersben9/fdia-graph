"""The fields the record bundles share, each one's meaning written once.

`RecordFields` is the base of a record, a batch and an exported split (`models/data.py`). Every field
defaults to `None`, since which ones are filled depends on the file and the request. The columns of a
measurement array are named in `constants.channels` (`node_x[..., NodeChannel.THETA]`). A field's
trailing shape is fixed here; the leading axis (none for a record, `B` for a batch, `n` for a
split) belongs to the bundle.

An array field is `Array` (a numpy array or a torch tensor) and a per-record scalar is `Scalars`
(one int on a record, a vector on a batch or a split)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .base import Bundle

if TYPE_CHECKING:
    import torch

Array = Union[np.ndarray, "torch.Tensor"]
Scalars = Union[int, np.ndarray, "torch.Tensor"]


@dataclass(frozen=True, eq=False)
class RecordFields(Bundle):
    """Every field a record, a batch and an exported split can carry, by group."""

    # one measurement scan: the bus and branch meters and their masks
    node_x: Optional[Array] = None  # [..., N, 4] |V|, P_inj, Q_inj, theta (zero where unmetered)
    node_m: Optional[Array] = None  # [..., N, 4] meter mask, 1 metered
    edge_x: Optional[Array] = None  # [..., E, 2] P_from, Q_from
    edge_m: Optional[Array] = None  # [..., E, 2] flow-meter mask
    # the attack labels of a scan
    y: Optional[Array] = None  # [..., N] per-bus attack label, 1 attacked
    family: Optional[Scalars] = None  # [...] 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At, 6 Al, 7 Am (timeline)
    # where a record came from
    stealthy: Optional[Scalars] = None  # [...] 1 for the re-solve families Aq, At, Al, Am
    seq_id: Optional[Scalars] = None  # [...] episode index of the frame's attack (every family), -1 benign
    timestep: Optional[Scalars] = None  # [...] position in the source load profile
    # the two temporal features
    temporal_delta: Optional[Array] = None  # [..., N, 2] injection change vs the previous pool scan (v0.3+)
    swing: Optional[Array] = None  # [..., N, 2] that change as a z-score of recent change (v0.4.1+)
    # the noiseless attack-free truth (v0.7.2+)
    clean: Optional[Array] = None  # [..., N, 4] true state at the record's timestep
    edge_clean: Optional[Array] = None  # [..., E, 2] exact true flows on metered branches
    edge_clean_full: Optional[Array] = None  # [..., E, 2] exact true flows on every branch (v0.15.0+)
    # the static graph, the same for every record of a system
    edge_index: Optional[Array] = None  # [2, E] from and to bus of every branch
    edge_attr: Optional[Array] = None  # [E, 8] per-unit line physics r, x, b, g, gs, bs, tap, shift (v0.5.0+)
    # the attack-removed layer, next to the observed and the clean ones (every record of a timeline file)
    benign: Optional[Array] = None  # [..., N, 4] attack removed, noise kept
    edge_benign: Optional[Array] = None  # [..., E, 2] attack removed, noise kept
    # The PMU branch-current phasors of a hybrid-meter timeline [WU26 eqs. 19-20]: per branch the real
    # and imaginary current at each end (`CurrentChannel` columns), per unit on the base current, zero
    # where no PMU sits at that end. Absent from a v0.8.3-meter file.
    pmu_i: Optional[Array] = None  # [..., E, 4] observed (attacked where attacked), noise kept
    pmu_i_m: Optional[Array] = None  # [..., E, 4] 1 where a PMU reads that end of the branch
    pmu_i_benign: Optional[Array] = None  # [..., E, 4] attack removed, noise kept
