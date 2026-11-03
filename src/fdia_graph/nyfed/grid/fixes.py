"""The changes that make NYgrid's operating points solvable in AC (checked in MATLAB on 8760 hours of 2019)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fdia_graph.nyfed.grid.case import PV, Case


def _pos(case: Case) -> dict[int, int]:
    return {b: i for i, b in enumerate(case.bus_id.tolist())}


def split_zone_j(case: Case) -> Case:
    """NYgrid places every zone J unit at bus 82 and none at bus 81, though allocateLoad.m splits zone J's
    load evenly over the two (neither has base load). Move units to bus 81 in proportion to its half of
    the load, taking every other unit in list order, and make bus 81 a PV bus."""
    pos = _pos(case)
    b81, b82 = pos[81], pos[82]
    at82 = np.flatnonzero(case.unit_bus == b82)
    n = int(round(len(at82) * 0.5))
    unit_bus = case.unit_bus.copy()
    unit_bus[at82[0 : 2 * n : 2]] = b81
    bus_type = case.bus_type.copy()
    bus_type[b81] = PV
    return replace(case, unit_bus=unit_bus, bus_type=bus_type)


def restore_765kv(case: Case) -> Case:
    """Add the 765 kV Massena-Marcy line (47-43) that modifyMPC.m carries commented out at L75."""
    pos = _pos(case)
    return replace(
        case,
        f=np.r_[case.f, pos[47]],
        t=np.r_[case.t, pos[43]],
        r=np.r_[case.r, 0.004],
        x=np.r_[case.x, 0.0085],
        b=np.r_[case.b, 0.0],
        tap=np.r_[case.tap, 1.0],
        shift=np.r_[case.shift, 0.0],
        rating=np.r_[case.rating, 0.0],
    )


def externals_at_base(case: Case, base: Case) -> Case:
    """Hold the external areas at the base case load instead of NYgrid's scaling with NY load, and
    scale the external units so total generation still covers total load at NY's scheduled import."""
    ext_bus = ~case.is_ny
    pd, qd = case.pd.copy(), case.qd.copy()
    pd[ext_bus], qd[ext_bus] = base.pd[ext_bus], base.qd[ext_bus]
    on = case.unit_on
    ext_unit = ~case.is_ny[case.unit_bus] & on
    need = pd.sum() - case.unit_p[~ext_unit & on].sum()
    unit_p = case.unit_p.copy()
    unit_p[ext_unit] *= need / unit_p[ext_unit].sum()
    return replace(case, pd=pd, qd=qd, unit_p=unit_p)


# updateOpCond.m L407-412: the four HVDC links of the reduced case, as from bus, to bus and limit (MW).
_HVDC = [(21, 80, 530.0), (124, 79, 660.0), (125, 81, 660.0), (125, 81, 660.0)]


def hvdc_links(case: Case) -> Case:
    """Add NYgrid's HVDC links (Cross Sound and 1385, Neptune, HTP, VFT) as DC lines, removing the AC
    branch 81-125 they replace (updateOpCond.m L413)."""
    pos = _pos(case)
    lim = np.array([m for *_, m in _HVDC])
    ends = {(pos[81], pos[125]), (pos[125], pos[81])}
    keep = np.array([(f, t) not in ends for f, t in zip(case.f.tolist(), case.t.tolist())])
    return replace(
        case,
        f=case.f[keep],
        t=case.t[keep],
        r=case.r[keep],
        x=case.x[keep],
        b=case.b[keep],
        tap=case.tap[keep],
        shift=case.shift[keep],
        rating=case.rating[keep],
        dc_f=np.array([pos[a] for a, *_ in _HVDC]),
        dc_t=np.array([pos[b] for _, b, _ in _HVDC]),
        dc_pmin=-lim,
        dc_pmax=lim,
    )
