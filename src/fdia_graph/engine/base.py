"""The state FdiaGenerator's mixins share: attribute annotations only.

Set for real in FdiaGenerator.__init__ (core.py); declared here (no runtime effect) so each mixin's
`self.<attr>` type-checks. An attribute only core.py reads needs no line here."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.frames import LayerSet, MeterLayers
from ..sensing.measure import Measurement
from ..sensing.meters import MeterPlan
from ..sensing.noise import NoiseModel
from ..systems.case import Case
from ..systems.network import Network


class GridBase:
    # grid + rng
    case: Case
    # its bus admittance `net.ybus` is the one every local solve reads; it is summed in MATPOWER's order
    # and held by column, because the stored IEEE files depend on its last bit (`Network`)
    net: Network
    C: int
    spec: LayerSet  # the layers a scan is laid out in (the flow layer's ends)
    E: int
    rng: np.random.Generator
    seed: int
    # the timestep whose jitter the next emission draws (a timeline walk sets it), or None: the jitter
    # then comes from `rng` like every other draw
    scan_key: Optional[int]
    # meters and their noise
    plan: MeterPlan
    meas: Measurement
    noise: NoiseModel
    masks: MeterLayers  # 1 where a meter sits: node [N, 4], flow [E, 2], PMU current [E, 4]
    pmu_buses: np.ndarray  # the buses with a PMU, sorted
    flow_metered: np.ndarray  # [E] bool, a flow meter at the branch's from end
    zero_inj: list[int]
    slack_bus: int
    # buses / loads / attackability
    load_bus: np.ndarray
    load_p0: np.ndarray  # [n_loads] base active load per loaded bus (MW)
    load_q0: np.ndarray  # [n_loads] base reactive load per loaded bus (MVAr)
    stealthy_pos: np.ndarray  # attackable loads off generator buses, the stealthy families' targets
    load_base: np.ndarray  # [N, 2] base-case load P, Q per bus
    gen_base: np.ndarray  # [N, 2] base-case generation P per bus (Q column zero)
    p_lim: np.ndarray  # [N, 2] generator active limits per bus, ±inf without one
    q_lim: np.ndarray  # [N, 2] generator reactive limits per bus
    has_gen: np.ndarray  # [N] bool, a generator on the bus
    v_case: np.ndarray  # [N, 2] the case's bus voltage limits
    ei: np.ndarray  # [2, E] the from bus and the to bus of every branch
