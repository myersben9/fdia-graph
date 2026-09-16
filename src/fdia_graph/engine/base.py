"""Shared state contract for FdiaGenerator's mixins — attribute + cross-method annotations only.

Set for real in FdiaGenerator.__init__; declared here (no runtime effect) so each mixin's `self.<attr>`
and cross-mixin method calls type-check. See core.py for the actual assignments."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from ..models.grid import BranchModel, MeterPlan, MeterBias, Outage, INTACT  # noqa: F401  re-exported: defined here before the models package

if TYPE_CHECKING:
    from .attacks import Redistribution
    from .records import Scan


def _deprecated(old: str, new: str) -> None:
    warnings.warn(f"FdiaGenerator.{old} is deprecated, use {new}", DeprecationWarning, stacklevel=3)


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
    SD: Dict[str, float]
    SDj: Dict[str, float]
    _sd_bias: Dict[str, float]
    bias: MeterBias
    # metering plan
    meters: MeterPlan
    zero_inj: List[int]
    _inj_buses: List[int]
    # buses / loads / attackability
    load_bus: np.ndarray
    load_genP: np.ndarray
    attackable_pos: np.ndarray
    _attackable_mask: np.ndarray
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
    _Lcands: List[int]
    _sgn: Dict[int, float]
    _Ltgt: int
    # replay buffer
    benign_buf: List[np.ndarray]

    # cross-mixin methods (defined in the concern mixins)
    def _n(self, s: float) -> float: ...
    def emit_from_state(self, X: np.ndarray) -> "Scan": ...
    def clean_flows_from_states(self, X: np.ndarray) -> np.ndarray: ...
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
    ) -> Optional["Redistribution"]: ...

    # ---- the attribute names of releases before 0.16, kept for one minor version ----------------------
    # Each reads from the model that replaced it and warns once per call site.
    @property
    def M(self) -> Dict[str, Any]:
        _deprecated("M", "meters (MeterPlan)")
        return {"vbus": self.meters.vbus, "pmu": self.meters.pmu, "inj": self.meters.inj}

    @property
    def flow_meter(self) -> np.ndarray:
        _deprecated("flow_meter", "meters.flow")
        return self.meters.flow

    @property
    def bias_pi(self) -> np.ndarray:
        _deprecated("bias_pi", "bias.pi")
        return self.bias.pi

    @property
    def bias_qi(self) -> np.ndarray:
        _deprecated("bias_qi", "bias.qi")
        return self.bias.qi

    @property
    def bias_v(self) -> np.ndarray:
        _deprecated("bias_v", "bias.v")
        return self.bias.v

    @property
    def bias_va(self) -> np.ndarray:
        _deprecated("bias_va", "bias.va")
        return self.bias.va

    @property
    def bias_pf(self) -> np.ndarray:
        _deprecated("bias_pf", "bias.pf")
        return self.bias.pf

    @property
    def bias_qf(self) -> np.ndarray:
        _deprecated("bias_qf", "bias.qf")
        return self.bias.qf

    @property
    def edge_r(self) -> np.ndarray:
        _deprecated("edge_r", "branch.r")
        return self.branch.r

    @property
    def edge_x(self) -> np.ndarray:
        _deprecated("edge_x", "branch.x")
        return self.branch.x

    @property
    def edge_b(self) -> np.ndarray:
        _deprecated("edge_b", "branch.b")
        return self.branch.b

    @property
    def edge_g(self) -> np.ndarray:
        _deprecated("edge_g", "branch.g")
        return self.branch.g

    @property
    def edge_tap(self) -> np.ndarray:
        _deprecated("edge_tap", "branch.tap")
        return self.branch.tap

    @property
    def edge_shift(self) -> np.ndarray:
        _deprecated("edge_shift", "branch.shift_deg")
        return self.branch.shift_deg

    @property
    def edge_status(self) -> np.ndarray:
        _deprecated("edge_status", "branch.status")
        assert self.branch.status is not None
        return self.branch.status

    @property
    def outage(self) -> Optional[int]:
        _deprecated("outage", "contingency.line")
        return self.contingency.line

    @property
    def outage_pos(self) -> int:
        _deprecated("outage_pos", "contingency.pos")
        return self.contingency.pos

    @property
    def outage_name(self) -> str:
        _deprecated("outage_name", "contingency.name")
        return self.contingency.name

    @property
    def outage_from_bus(self) -> int:
        _deprecated("outage_from_bus", "contingency.from_bus")
        return self.contingency.from_bus

    @property
    def outage_to_bus(self) -> int:
        _deprecated("outage_to_bus", "contingency.to_bus")
        return self.contingency.to_bus

    @property
    def outage_base_flow_mw(self) -> float:
        _deprecated("outage_base_flow_mw", "contingency.base_flow_mw")
        return self.contingency.base_flow_mw
