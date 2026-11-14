"""Hourly unit dispatch, NYgrid's allocation of the NYISO fuel mix to the fixed unit list."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from fdia_graph.nyfed.grid.case import Case
from fdia_graph.nyfed.grid.fixes import externals_at_base
from fdia_graph.nyfed.scenario.inputs import EXT_PRICES, FUELS, INTERFACES, NygridHour

_NIAGARA, _ST_LAWRENCE = 55, 48  # hydro buses with their own rule (updateOpCond.m L143-152)
_NIAGARA_CAP, _ST_LAWRENCE_CAP = 2460.0, 856.0
# updateOpCond.m L485-491 draws nuclear costs 1 + 2 rand and hydro costs 10 rand ($/MWh) after rng("default")
# every hour, so they are fixed numbers: MATLAB's default stream is Mersenne Twister seeded 5489, the first
# 16 draws of numpy's RandomState(5489), in the unit order of nygrid_units.
_NUCLEAR_COST = (2.6294, 2.8116, 1.2540, 2.8268, 2.2647, 1.1951)
_HYDRO_COST = (2.7850, 5.4688, 9.5751, 9.6489, 1.5761, 9.7059, 9.5717, 4.8538, 8.0028, 1.4189)
_AREA_PRICE = {
    1: "NPX",
    4: "O H",
    5: "PJM",
    6: "PJM",
}  # external areas and the proxy bus pricing them (L497-500)


@dataclass(frozen=True)
class Dispatch:
    """Setpoints per hour: `unit_p [H, G]` MW, `unit_on [H, G]`, `hvdc_p [H, D]` MW from `dc_f` to `dc_t`,
    and `pd [H, N]`, `qd [H, N]` the bus loads the setpoints balance."""

    hour: np.ndarray
    unit_p: np.ndarray
    unit_on: np.ndarray
    hvdc_p: np.ndarray
    pd: np.ndarray
    qd: np.ndarray
    unit_cost: np.ndarray


def _unit_cost(case: Case, h: NygridHour) -> np.ndarray:
    """[G] $/MWh: thermal units at the week's fuel price times heat rate, nuclear and hydro at NYgrid's
    fixed draws, external units at their area's proxy-bus LBMP and the HQ import at H Q's."""
    cost = np.zeros(len(case.fuel))
    th = ~np.isin(case.fuel, ("Nuclear", "Hydro", "Import"))
    cost[th] = h.thermal_cost
    cost[case.fuel == "Nuclear"] = _NUCLEAR_COST
    cost[case.fuel == "Hydro"] = _HYDRO_COST
    lbmp: dict[str, float] = dict(zip(EXT_PRICES, h.ext_lbmp))
    imp = np.flatnonzero(case.fuel == "Import")
    for k in imp[:-1]:
        cost[k] = lbmp[_AREA_PRICE[int(case.area[case.unit_bus[k]])]]
    cost[imp[-1]] = lbmp["H Q"]
    return cost


def _iface(h: NygridHour, *names: str) -> float:
    return float(sum(h.iface_flow[INTERFACES.index(n)] for n in names))


def _hour(case: Case, base: Case, h: NygridHour, hold_externals: bool = True) -> Case:
    """updateOpCond.m L60-381 for one hour on the full 140-bus case, then the external fix."""
    fuel = dict(zip(FUELS, h.fuel_mw))
    kind = case.fuel
    bus_id = case.bus_id[case.unit_bus]
    p = np.zeros(len(kind))

    nuc = kind == "Nuclear"
    p[nuc] = h.nuclear_mw

    hyd = np.flatnonzero(kind == "Hydro")
    small_cap = case.pmax[hyd].sum() - _NIAGARA_CAP - _ST_LAWRENCE_CAP
    for i in hyd:
        if bus_id[i] == _NIAGARA:
            p[i] = 0.8 * fuel["Hydro"] - h.stl_cf * _ST_LAWRENCE_CAP
        elif bus_id[i] == _ST_LAWRENCE:
            p[i] = h.stl_cf * _ST_LAWRENCE_CAP
        else:
            p[i] = 0.2 * fuel["Hydro"] * case.pmax[i] / small_cap

    # Thermal units take their RGGI hourly output; the rest of NYISO's thermal need goes to the zone J
    # units that are running, in proportion to their output (L182-210).
    th = np.flatnonzero(~np.isin(kind, ("Nuclear", "Hydro", "Import")))
    p[th] = h.thermal_mw
    need = sum(h.fuel_mw) - p[hyd].sum() - p[nuc].sum() - fuel["Wind"] - fuel["Other Renewables"]
    in_j = th[(case.zone[case.unit_bus[th]] == "J") & (p[th] != 0)]
    p[in_j] += (need - p[th].sum()) * p[in_j] / p[in_j].sum()

    # External areas scale load with NY and generation to cover load plus the scheduled export to NY
    # (L313-364), the HQ import is the unit at bus 48 (L367-378).
    ny_bus = case.is_ny
    pd = base.pd.copy()
    qd = base.qd.copy()
    pd[ny_bus], qd[ny_bus] = h.ny_pd[ny_bus], h.ny_qd[ny_bus]
    ratio = pd[ny_bus].sum() / base.pd[ny_bus].sum()
    flows = {
        (5, 6): _iface(h, "SCH - PJ - NY", "SCH - PJM_HTP", "SCH - PJM_NEPTUNE", "SCH - PJM_VFT"),
        (1,): _iface(h, "SCH - NE - NY", "SCH - NPX_1385", "SCH - NPX_CSC"),
        (4,): _iface(h, "SCH - OH - NY"),
    }
    ext = np.flatnonzero(kind == "Import")[:-1]
    for areas, flow in flows.items():
        bus_in = np.isin(case.area, areas)
        units = ext[np.isin(case.area[case.unit_bus[ext]], areas)]
        load_old = base.pd[bus_in].sum()
        pd[bus_in] *= ratio
        qd[bus_in] *= ratio
        p[units] = base.unit_p[units] * (load_old * ratio + flow) / base.unit_p[units].sum()
    p[-1] = _iface(h, "SCH - HQ - NY", "SCH - HQ_CEDARS")
    pd -= h.renew_mw

    hour = replace(case, pd=pd, qd=qd, unit_p=p, unit_on=np.ones(len(p), dtype=bool))
    return externals_at_base(hour, base) if hold_externals else hour


def hourly_dispatch(case: Case, hours: list[NygridHour], hold_externals: bool = True) -> Dispatch:
    """NYgrid's hourly operating points on `case` (its base state is the NPCC base load and external
    dispatch), with the external areas held at base load; `hold_externals=False` keeps NYgrid's own external
    scaling."""
    rows = [_hour(case, case, h, hold_externals) for h in hours]
    hvdc = np.array(
        [
            [
                _iface(h, "SCH - NPX_1385", "SCH - NPX_CSC"),
                _iface(h, "SCH - PJM_NEPTUNE"),
                _iface(h, "SCH - PJM_HTP"),
                _iface(h, "SCH - PJM_VFT"),
            ]
            for h in hours
        ]
    )
    return Dispatch(
        hour=np.array([h.time for h in hours]),
        unit_p=np.array([r.unit_p for r in rows]),
        unit_on=np.array([r.unit_on for r in rows]),
        hvdc_p=hvdc if len(case.dc_f) else np.zeros((len(hours), 0)),
        pd=np.array([r.pd for r in rows]),
        qd=np.array([r.qd for r in rows]),
        unit_cost=np.array([_unit_cost(case, h) for h in hours]),
    )
