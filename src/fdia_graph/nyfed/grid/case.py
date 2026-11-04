"""The grid as one frozen record of arrays."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

PQ, PV, REF = 1, 2, 3  # MATPOWER bus types


@dataclass(frozen=True)
class Case:
    """Buses, branches, units and HVDC links of one grid, with the base operating state.

    Branch ends `f`, `t` and `unit_bus` are positions into the bus arrays, not bus numbers.
    Powers are in MW and MVAr, impedances in p.u. on `base_mva`, angles in degrees.
    """

    base_mva: float
    # buses [N]
    bus_id: np.ndarray
    name: np.ndarray
    zone: np.ndarray  # NYISO load zone letter A..K, "" outside New York
    area: np.ndarray
    kv: np.ndarray
    vmin: np.ndarray
    vmax: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    is_ny: np.ndarray
    # branches [L]
    f: np.ndarray
    t: np.ndarray
    r: np.ndarray
    x: np.ndarray
    b: np.ndarray
    tap: np.ndarray  # 1.0 for lines
    shift: np.ndarray
    rating: np.ndarray  # MVA, 0 where unknown
    # units [G]
    unit_bus: np.ndarray
    fuel: np.ndarray
    pmin: np.ndarray
    pmax: np.ndarray
    qmin: np.ndarray
    qmax: np.ndarray
    ramp: np.ndarray  # MW/min load-following (AGC) ramp, NaN where the source gives none
    # HVDC links [D]
    dc_f: np.ndarray
    dc_t: np.ndarray
    dc_pmin: np.ndarray
    dc_pmax: np.ndarray
    # base state
    bus_type: np.ndarray
    pd: np.ndarray
    qd: np.ndarray
    gs: np.ndarray
    bs: np.ndarray
    vm: np.ndarray
    va: np.ndarray
    unit_p: np.ndarray
    unit_q: np.ndarray
    unit_vg: np.ndarray
    unit_on: np.ndarray

    def with_voltage_margin(self, margin: float) -> Case:
        """A copy with every bus's voltage limits widened by `margin` (p.u.), this case unchanged. The attacker's
        sensitivity case of N71: the reduced network pins New York City buses at their upper limit."""
        return replace(self, vmin=self.vmin - margin, vmax=self.vmax + margin)
