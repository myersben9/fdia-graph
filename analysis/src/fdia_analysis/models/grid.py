"""The pseudo-measurement records of the estimator's PMU current channels [WU26 eq. 3]: where a
pseudo voltage phasor can be placed, and the phasors with their propagated noise."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PseudoLinks(NamedTuple):
    """Where [WU26 eq. 3] can place a pseudo voltage phasor: one row per metered branch end whose
    bus has a PMU and whose far bus does not (`formulas.estimation.pmu_pseudo_links`). The
    admittances are the branch's pi-model entries seen from the metered end, taps and shifts
    included: I_near = y_nn V_near + y_nf V_far."""

    near: np.ndarray  # [J] the PMU bus
    far: np.ndarray  # [J] the bus the pseudo-measurement lands on
    edge: np.ndarray  # [J] the branch
    end: np.ndarray  # [J] 0 when the PMU reads the from end, 1 the to end
    y_nn: np.ndarray  # [J] complex, Yf[e, from] or Yt[e, to]
    y_nf: np.ndarray  # [J] complex, Yf[e, to] or Yt[e, from]


class PseudoVoltages(NamedTuple):
    """The pseudo voltage phasors of [WU26 eq. 3] and their first-order propagated noise
    (`formulas.estimation.pmu_pseudo_voltages`), NaN at the buses no link reaches."""

    v: np.ndarray  # [..., N] |V| pu
    theta: np.ndarray  # [..., N] angle rad
    var_v: np.ndarray  # [..., N] variance of |V|, pu^2
    var_theta: np.ndarray  # [..., N] variance of the angle, rad^2
    reached: np.ndarray  # [N] bool, a link lands on the bus
