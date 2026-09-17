"""The static description of a system: the per-branch pi model and the admittance matrices built
from it, which meters exist, the constant meter bias, and the N-1 contingency a generator was
built with. Nothing here changes from one scan to the next."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


class BranchModel(NamedTuple):
    """The per-branch pi model [MP19], one entry per branch, per unit; the shapes a shard stores."""

    r: np.ndarray  # series resistance
    x: np.ndarray  # series reactance
    b: np.ndarray  # charging susceptance
    g: np.ndarray  # charging conductance (transformer iron losses)
    tap: np.ndarray  # turns ratio, 1 (or 0, read as 1) for lines
    shift_deg: np.ndarray  # phase shift in degrees
    status: Optional[np.ndarray] = None  # 1 in service, 0 out; None = all in service


class Admittances(NamedTuple):
    """The nodal and branch admittance matrices of one topology, complex per unit."""

    ybus: np.ndarray  # [N, N]
    yf: np.ndarray  # [E, N] from-end: I_f = Yf @ V
    yt: np.ndarray  # [E, N] to-end: I_t = Yt @ V


class MeterPlan(NamedTuple):
    """The sparse metering plan, sampled once per generator: which buses carry a voltage-magnitude
    meter, which carry a PMU (|V| and angle), which carry P/Q injection meters, and which branches
    carry a flow meter."""

    vbus: set[int]
    pmu: set[int]
    inj: list[int]
    flow: np.ndarray  # [E] bool


class MeterBias(NamedTuple):
    """The per-meter SYSTEMATIC bias drawn once (constant across scans): relative for P/Q
    injections and flows, absolute for |V| and angle (radians)."""

    pi: np.ndarray  # [N]
    qi: np.ndarray  # [N]
    v: np.ndarray  # [N]
    va: np.ndarray  # [N]
    pf: np.ndarray  # [E]
    qf: np.ndarray  # [E]


class Outage(NamedTuple):
    """The N-1 contingency a generator was built with: the pandapower line id (None = intact), its
    branch position, its name, its terminals, and its intact-case active flow (the contingency's
    size)."""

    line: Optional[int]
    pos: int
    name: str
    from_bus: int
    to_bus: int
    base_flow_mw: float


INTACT = Outage(None, -1, "", -1, -1, float("nan"))
