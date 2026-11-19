"""The column of each quantity on the last axis of a measurement array: `node_x[..., NodeChannel.P_INJ]`.

A member is its column number, so it indexes a numpy array or a torch tensor directly. The orders are
the ones every timeline file and operating-point pool stores (docs/reference/DATA_DICTIONARY.md).
"""

from __future__ import annotations

from enum import IntEnum


class NodeChannel(IntEnum):
    """The four columns of the bus layers `node_x`, `node_m`, `benign`, `clean` and `node_tamper`, and of
    a stored pool state [N, 4]. `P_INJ` and `Q_INJ` are load-positive, in a pool state and in the layers
    emitted from it: the load at the bus minus its generation, a shunt's draw left out (MW and MVAr)."""

    V = 0  # voltage magnitude (pu)
    P_INJ = 1  # active injection (MW, or pu on baseMVA); load-positive
    Q_INJ = 2  # reactive injection (MVAr, or pu); load-positive
    THETA = 3  # voltage angle (degrees, or radians on a per-unit view)


class EdgeChannel(IntEnum):
    """The columns of the flow layers `edge_x`, `edge_m`, `edge_benign`, `edge_clean` and `edge_tamper`.

    A flow is metered at the from end of its branch."""

    P_FROM = 0  # active flow leaving the from bus (MW, or pu)
    Q_FROM = 1  # reactive flow leaving the from bus (MVAr, or pu)


class CurrentChannel(IntEnum):
    """The four columns of the PMU branch-current layers `pmu_i`, `pmu_i_m`, `pmu_i_benign` and
    `pmu_i_tamper`: the phasor a PMU reads at each end of a branch [WU26 eqs. 19-20] as
    real and imaginary part, per unit on the base current. At the from end it is the current leaving
    the from bus into the branch (`Yf V`), at the to end the current leaving the to bus (`Yt V`)."""

    RE_FROM = 0
    IM_FROM = 1
    RE_TO = 2
    IM_TO = 3


# The one layout of a timeline file, as a file declares it: its flow legend (`edge_feat`, the names
# of `EdgeChannel`) and the form of a PMU current phasor (`current_form`, the parts of
# `CurrentChannel`). A file with neither attribute has this layout.
EDGE_FEAT_DEFAULT = "P_from,Q_from"
CURRENT_RECTANGULAR = "rectangular"
# Columns between a current at the from end and the same quantity at the to end: RE_TO - RE_FROM.
TO_END_OFFSET = 2
