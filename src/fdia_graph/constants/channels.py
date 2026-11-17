"""The column of each quantity on the last axis of a measurement array: `node_x[..., NodeChannel.P_INJ]`.

A member is its column number, so it indexes a numpy array or a torch tensor directly. The orders are
the ones every timeline file and operating-point pool stores (docs/reference/DATA_DICTIONARY.md).
"""

from __future__ import annotations

from enum import IntEnum


class NodeChannel(IntEnum):
    """The four columns of the bus layers `node_x`, `node_m`, `benign`, `clean` and `node_tamper`."""

    V = 0  # voltage magnitude (pu)
    P_INJ = 1  # active injection (MW, or pu on baseMVA)
    Q_INJ = 2  # reactive injection (MVAr, or pu)
    THETA = 3  # voltage angle (degrees, or radians on a per-unit view)


class EdgeChannel(IntEnum):
    """The columns of the flow layers `edge_x`, `edge_m`, `edge_benign`, `edge_clean` and `edge_tamper`.

    An IEEE file meters the from end of a branch and has the first two columns; a New York file meters
    both ends and has all four (its `edge_feat` attribute lists them)."""

    P_FROM = 0  # active flow leaving the from bus (MW, or pu)
    Q_FROM = 1  # reactive flow leaving the from bus (MVAr, or pu)
    P_TO = 2  # active flow leaving the to bus (MW, or pu): files with both ends only
    Q_TO = 3  # reactive flow leaving the to bus (MVAr, or pu): files with both ends only


class CurrentChannel(IntEnum):
    """The four columns of the PMU branch-current layers `pmu_i`, `pmu_i_m`, `pmu_i_benign` and
    `pmu_i_tamper` of an IEEE file: the phasor a PMU reads at each end of a branch [WU26 eqs. 19-20] as
    real and imaginary part, per unit on the base current. At the from end it is the current leaving
    the from bus into the branch (`Yf V`), at the to end the current leaving the to bus (`Yt V`)."""

    RE_FROM = 0
    IM_FROM = 1
    RE_TO = 2
    IM_TO = 3


class PolarCurrentChannel(IntEnum):
    """The same four columns in a New York file, which stores each phasor as magnitude and angle and
    says so in its `current_form` attribute (`schema.Attr.CURRENT_FORM`, "polar")."""

    MAG_FROM = 0  # current magnitude at the from end (pu on the base current)
    ANGLE_FROM = 1  # its angle (degrees)
    MAG_TO = 2  # current magnitude at the to end (pu)
    ANGLE_TO = 3  # its angle (degrees)


# Columns between a branch quantity at the from end and the same quantity at the to end, in a flow
# layer with both ends and in a current layer: P_TO - P_FROM, RE_TO - RE_FROM, MAG_TO - MAG_FROM.
TO_END_OFFSET = 2
