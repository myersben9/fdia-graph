"""Shared state contract for FdiaGenerator's mixins — attribute + cross-method annotations only.

Set for real in FdiaGenerator.__init__; declared here (no runtime effect) so each mixin's `self.<attr>`
and cross-mixin method calls type-check. See core.py for the actual assignments."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..models.grid import (  # noqa: F401  re-exported: defined here before the models package
    INTACT,
    BranchModel,
    MeterBias,
    MeterPlan,
    Outage,
)

# Meter accuracy classes [ASP14]: the error std of every meter type. |V| and angle are the
# class-0.2/sqrt(3) instrument-transformer figures, absolute (pu, rad); injections and flows a ~1.7%
# power-measurement std, relative to the reading, with POWER_NOISE_FLOOR_MW as an absolute floor.
# The generator's noise model and a measurement-only estimator calibration both read these.
ACCURACY_CLASS = {"pf": 0.017, "qf": 0.017, "v": 0.0012, "pi": 0.017, "qi": 0.017, "va": 0.00168}
POWER_NOISE_FLOOR_MW = 1e-3

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from .pp_types import PandapowerNet
    from .records import Scan


class GridBase:
    # grid + rng
    pp: ModuleType
    NET: Callable[[], PandapowerNet]
    base: PandapowerNet
    C: int
    E: int
    n_lines: int
    rng: np.random.Generator
    # noise model
    SD: dict[str, float]
    SDj: dict[str, float]
    _bias_sd: dict[str, float]
    bias: MeterBias
    _i_jitter: float  # hybrid meters: the per-scan relative jitter of the PMU branch-current channels
    # metering plan
    meters: MeterPlan
    zero_inj: list[int]
    slack_bus: int
    _injection_buses: list[int]
    # buses / loads / attackability
    load_bus: np.ndarray
    load_p0: np.ndarray  # [n_loads] base active load per element (MW)
    load_q0: np.ndarray  # [n_loads] base reactive load per element (MVAr)
    attackable_pos: np.ndarray
    _attackable_mask: np.ndarray
    stealthy_pos: np.ndarray  # attackable loads off generator buses, the stealthy families' targets
    _stealthy_mask: np.ndarray
    max_load_mw: Optional[float]
    load_base: np.ndarray  # [N, 2] base-case load P, Q per bus
    gen_base: np.ndarray  # [N, 2] base-case generation P per bus (Q column zero)
    p_lim: np.ndarray  # [N, 2] generator active limits per bus, ±inf without one
    q_lim: np.ndarray  # [N, 2] generator reactive limits per bus
    has_gen: np.ndarray  # [N] bool, a generator on the bus
    v_case: np.ndarray  # [N, 2] the case's bus voltage limits
    # topology + admittance
    ei: np.ndarray
    branch: BranchModel
    edge_gs: np.ndarray
    edge_bs: np.ndarray
    edge_is_trafo: np.ndarray
    bus_shunt_g: np.ndarray
    bus_shunt_b: np.ndarray
    x_react: np.ndarray
    _Ybus: csr_matrix
    _Yf: csr_matrix
    _Yt: csr_matrix
    _base_mva: float
    _ppc_row: np.ndarray
    _from_bus_ppc: np.ndarray
    _n_ppc_buses: int
    # contingency
    contingency: Outage

    # cross-mixin methods (defined in the concern mixins)
    def _draw_noise(self, s: float) -> float: ...
    def emit_from_state(self, X: np.ndarray) -> Scan: ...
    def meter_masks(self) -> tuple[np.ndarray, np.ndarray]: ...
    def current_mask(self) -> Optional[np.ndarray]: ...
    def clean_flows_from_states(self, X: np.ndarray) -> np.ndarray: ...
    def all_flows_from_states(self, X: np.ndarray) -> np.ndarray: ...
    def currents_from_states(self, X: np.ndarray) -> np.ndarray: ...
    def true_load(self, Xt: np.ndarray) -> np.ndarray: ...
    def true_reactive_load(self, Xt: np.ndarray) -> np.ndarray: ...
    def scan_generation(self, Xt: np.ndarray) -> np.ndarray: ...
    def scan_reactive_generation(self, Xt: np.ndarray) -> np.ndarray: ...
