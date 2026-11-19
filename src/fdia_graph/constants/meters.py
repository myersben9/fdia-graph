"""The meters: their kinds, their accuracy classes and noise floors, and where each kind's reading sits
in a timeline file."""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from .channels import CurrentChannel, EdgeChannel, NodeChannel

# Meter accuracy classes [ASP14], by the meter type's name in that table: the standard deviation of
# the error of every meter type. |V| and the angle are the class-0.2/sqrt(3) instrument-transformer figures, absolute (pu, radians);
# injections ("pi", "qi") and flows ("pf", "qf") are a 1.7% power-measurement figure, relative to
# the reading.
ACCURACY_CLASS = {"pf": 0.017, "qf": 0.017, "v": 0.0012, "pi": 0.017, "qi": 0.017, "va": 0.00168}
POWER_NOISE_FLOOR_MW = 1e-3  # MW or MVAr: the absolute floor under a power meter's relative class
# Share of each class that is per-scan jitter; the rest is a constant per-meter bias, with
# bias^2 + jitter^2 = class^2. Meter error is mostly a fixed calibration offset.
JITTER_FRAC = 0.25
# The PMU branch-current class [C37118]: a total vector error of at most 1% in steady state, taken as
# three standard deviations of each component, relative to the phasor's magnitude.
PMU_CURRENT_CLASS = 0.01 / 3
CURRENT_FLOOR_PU = 1e-5  # pu on the base current: keeps a branch carrying almost no current from zero noise
# The measurement noise of [WU26]'s case studies (pu): 0.03 on SCADA channels, 0.01 on PMU channels.
WU26_NOISE = {"scada": 0.03, "pmu": 0.01}


class MeterKind(IntEnum):
    """What a meter reads: the `kind` code of a row of a meter plan. A PMU's branch current is two
    rows, the real and the imaginary part of the phasor."""

    V = 0  # voltage magnitude at a bus (pu)
    VA = 1  # voltage angle at a PMU bus (degrees)
    P = 2  # active injection at a bus (MW)
    Q = 3  # reactive injection at a bus (MVAr)
    PF = 4  # active flow at one end of a branch (MW)
    QF = 5  # reactive flow at one end of a branch (MVAr)
    IR = 6  # real part of the current phasor at one end of a branch, read by a PMU (pu)
    II = 7  # its imaginary part (pu)


# The kinds read at a bus, in the order of a client's node features.
BUS_KINDS = (MeterKind.V, MeterKind.VA, MeterKind.P, MeterKind.Q)
ANGLE_KINDS = (MeterKind.VA,)  # the kinds read in degrees, which wrap at +-180
FLOW_KINDS = (MeterKind.PF, MeterKind.QF)  # the kinds of a flow meter, in a flow layer's column order
# The two parts of a current phasor stored as real and imaginary part. Their error is relative to
# the phasor's magnitude |I|, not to the part's own size [C37118].
PHASOR_KINDS = (MeterKind.IR, MeterKind.II)

# The accuracy of each kind as (relative, absolute) standard deviations, sigma = rel |z| + abs: the
# classes of `ACCURACY_CLASS` per meter kind, with the angles in degrees like the state. A PMU current
# takes 1/300 of |I| on each of the real and the imaginary part.
KIND_ACCURACY = {
    MeterKind.V: (0.0, ACCURACY_CLASS["v"]),
    MeterKind.VA: (0.0, float(np.degrees(ACCURACY_CLASS["va"]))),
    MeterKind.P: (ACCURACY_CLASS["pi"], POWER_NOISE_FLOOR_MW),
    MeterKind.Q: (ACCURACY_CLASS["qi"], POWER_NOISE_FLOOR_MW),
    MeterKind.PF: (ACCURACY_CLASS["pf"], POWER_NOISE_FLOOR_MW),
    MeterKind.QF: (ACCURACY_CLASS["qf"], POWER_NOISE_FLOOR_MW),
    MeterKind.IR: (PMU_CURRENT_CLASS, CURRENT_FLOOR_PU),
    MeterKind.II: (PMU_CURRENT_CLASS, CURRENT_FLOOR_PU),
}
# Where a kind's readings sit in a timeline file: the layer (0 the bus layer, 1 the flow layer, 2 the
# PMU current layer) and the column. A current's column is its from end's; its to end's is
# `channels.TO_END_OFFSET` further.
METER_SLOT = {
    MeterKind.V: (0, NodeChannel.V),
    MeterKind.VA: (0, NodeChannel.THETA),
    MeterKind.P: (0, NodeChannel.P_INJ),
    MeterKind.Q: (0, NodeChannel.Q_INJ),
    MeterKind.PF: (1, EdgeChannel.P_FROM),
    MeterKind.QF: (1, EdgeChannel.Q_FROM),
    MeterKind.IR: (2, CurrentChannel.RE_FROM),
    MeterKind.II: (2, CurrentChannel.IM_FROM),
}
