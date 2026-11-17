"""The state FdiaGenerator's mixins share: attribute annotations only.

Set for real in FdiaGenerator.__init__ (core.py); declared here (no runtime effect) so each mixin's
`self.<attr>` type-checks. An attribute only core.py reads needs no line here."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..models.grid import BranchModel, MeterBias, MeterPlan

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from .pandapower_types import PandapowerNet


class GridBase:
    # grid + rng
    base: PandapowerNet
    C: int
    E: int
    rng: np.random.Generator
    seed: int
    # the timestep whose jitter the next emission draws (a timeline walk sets it), or None: the jitter
    # then comes from `rng` like every other draw
    scan_key: Optional[int]
    _jitter: np.random.Generator  # the stream the current emission draws its jitter from
    # noise model
    SD: dict[str, float]
    SDj: dict[str, float]
    bias: MeterBias
    _i_jitter: float  # hybrid meters: the per-scan relative jitter of the PMU branch-current channels
    # metering plan
    meters: MeterPlan
    zero_inj: list[int]
    slack_bus: int
    # buses / loads / attackability
    load_bus: np.ndarray
    load_p0: np.ndarray  # [n_loads] base active load per element (MW)
    load_q0: np.ndarray  # [n_loads] base reactive load per element (MVAr)
    stealthy_pos: np.ndarray  # attackable loads off generator buses, the stealthy families' targets
    load_base: np.ndarray  # [N, 2] base-case load P, Q per bus
    gen_base: np.ndarray  # [N, 2] base-case generation P per bus (Q column zero)
    p_lim: np.ndarray  # [N, 2] generator active limits per bus, ±inf without one
    q_lim: np.ndarray  # [N, 2] generator reactive limits per bus
    has_gen: np.ndarray  # [N] bool, a generator on the bus
    v_case: np.ndarray  # [N, 2] the case's bus voltage limits
    # topology + admittance
    ei: np.ndarray
    branch: BranchModel
    _Ybus: csr_matrix
    _Yf: csr_matrix
    _Yt: csr_matrix
    _base_mva: float
    _ppc_row: np.ndarray
    _from_bus_ppc: np.ndarray
    _n_ppc_buses: int
