"""Shared state contract for FdiaGenerator's mixins — attribute + cross-method annotations only.

Set for real in FdiaGenerator.__init__; declared here (no runtime effect) so each mixin's `self.<attr>`
and cross-mixin method calls type-check. See core.py for the actual assignments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..models.grid import (  # noqa: F401  re-exported: defined here before the models package
    INTACT,
    BranchModel,
    MeterBias,
    MeterPlan,
    Outage,
)

if TYPE_CHECKING:
    from .attacks import Redistribution
    from .records import Scan


class GridBase:
    # grid + rng
    pp: Any
    NET: Any
    base: Any
    C: int
    E: int
    nl: int
    rng: np.random.Generator
    # noise model
    SD: dict[str, float]
    SDj: dict[str, float]
    _sd_bias: dict[str, float]
    bias: MeterBias
    # metering plan
    meters: MeterPlan
    zero_inj: list[int]
    slack_bus: int
    _inj_buses: list[int]
    # buses / loads / attackability
    load_bus: np.ndarray
    load_genP: np.ndarray
    attackable_pos: np.ndarray
    _attackable_mask: np.ndarray
    max_load_mw: Optional[float]
    load_base: np.ndarray  # [N, 2] base-case load P, Q per bus
    gen_base: np.ndarray  # [N, 2] base-case generation P per bus (Q column zero)
    p_lim: np.ndarray  # [N, 2] generator active limits per bus, ±inf without one
    q_lim: np.ndarray  # [N, 2] generator reactive limits per bus
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
    _Ybus: Any
    _Yf: Any
    _Yt: Any
    _bMVA: float
    _lut: Any
    _fb: np.ndarray
    _nppc: int
    _ptdf: np.ndarray
    _ptdf_lb: np.ndarray
    _solvenet: Any
    # contingency
    contingency: Outage
    # LRA target pool (set in _pick_lra_target)
    _Lcands: list[int]
    _sgn: dict[int, float]
    _Ltgt: int
    # replay buffer
    benign_buf: list[np.ndarray]

    # cross-mixin methods (defined in the concern mixins)
    def _n(self, s: float) -> float: ...
    def emit_from_state(self, X: np.ndarray) -> Scan: ...
    def clean_flows_from_states(self, X: np.ndarray) -> np.ndarray: ...
    def local_region(self, seeds: np.ndarray, hops: int) -> Optional[np.ndarray]: ...
    def solve_local(
        self, Xt: np.ndarray, interior: np.ndarray, Lp: np.ndarray, Lq: np.ndarray
    ) -> Optional[np.ndarray]: ...
    def state_from_net(self, net: Any) -> np.ndarray: ...

    def solve(
        self,
        Lp: np.ndarray,
        Lq: np.ndarray,
        Xt: Optional[np.ndarray] = ...,
        Lp_true: Optional[np.ndarray] = ...,
    ) -> Optional[Any]: ...

    def _lra_for_line(
        self, L: int, Lp: np.ndarray, rel: float, K: int, rand: bool = ..., floor: float = ...
    ) -> Optional[Redistribution]: ...
