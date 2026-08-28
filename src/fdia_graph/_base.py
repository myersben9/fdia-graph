"""Shared state contract for FdiaGenerator's mixins — attribute + cross-method annotations only.

Set for real in FdiaGenerator.__init__; declared here (no runtime effect) so each mixin's `self.<attr>`
and cross-mixin method calls type-check. See _core.py for the actual assignments."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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
    bias_pi: np.ndarray; bias_qi: np.ndarray; bias_v: np.ndarray
    bias_va: np.ndarray; bias_pf: np.ndarray; bias_qf: np.ndarray
    # metering plan
    M: Dict[str, Any]
    flow_meter: np.ndarray
    zero_inj: List[int]
    # buses / loads / attackability
    load_bus: np.ndarray
    load_genP: np.ndarray
    attackable_pos: np.ndarray
    _attackable_mask: np.ndarray
    # topology + admittance
    ei: np.ndarray
    _Ybus: Any; _Yf: Any; _Yt: Any
    _bMVA: float; _lut: Any; _fb: np.ndarray; _nppc: int
    _ptdf: np.ndarray; _ptdf_lb: np.ndarray
    _solvenet: Any
    # contingency
    outage: Optional[int]; outage_pos: int
    # LRA target pool (set in _pick_lra_target)
    _Lcands: List[int]; _sgn: Dict[int, float]; _Ltgt: int
    # replay buffer
    benign_buf: List[np.ndarray]

    # cross-mixin methods (defined in the concern mixins)
    def _n(self, s: float) -> float: ...
    def emit_from_state(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def state_from_net(self, net: Any) -> np.ndarray: ...
    def solve(self, Lp: np.ndarray, Lq: np.ndarray, Xt: Optional[np.ndarray] = ...,
              Lp_true: Optional[np.ndarray] = ...) -> Optional[Any]: ...
    def _lra_for_line(self, L: int, Lp: np.ndarray, rel: float, K: int, rand: bool = ...,
                      floor: float = ...) -> Optional[Tuple[np.ndarray, np.ndarray, float]]: ...
