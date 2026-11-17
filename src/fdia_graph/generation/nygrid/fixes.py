"""Two variants of a case, applied when one is asked for: the external areas held at the base load, and NYgrid's
HVDC links in place of the AC branch 81-125. The changes every build has (zone J's split, the 765 kV line) are in
the case file itself (`case.py` reads it)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fdia_graph.constants.newyork import HVDC_LINKS
from fdia_graph.generation.nygrid.case import Case


def _pos(case: Case) -> dict[int, int]:
    return {b: i for i, b in enumerate(case.bus_id.tolist())}


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


def hvdc_links(case: Case) -> Case:
    """Add NYgrid's HVDC links (Cross Sound and 1385, Neptune, HTP, VFT) as DC lines, removing the AC
    branch 81-125 they replace (updateOpCond.m L413)."""
    pos = _pos(case)
    lim = np.array([m for *_, m in HVDC_LINKS])
    ends = {(pos[81], pos[125]), (pos[125], pos[81])}
    keep = np.array([(f, t) not in ends for f, t in zip(case.f.tolist(), case.t.tolist())])
    return replace(
        case,
        f=case.f[keep],
        t=case.t[keep],
        r=case.r[keep],
        x=case.x[keep],
        b=case.b[keep],
        g=case.g[keep],
        tap=case.tap[keep],
        shift=case.shift[keep],
        rating=case.rating[keep],
        dc_f=np.array([pos[a] for a, *_ in HVDC_LINKS]),
        dc_t=np.array([pos[b] for _, b, _ in HVDC_LINKS]),
        dc_pmin=-lim,
        dc_pmax=lim,
    )
