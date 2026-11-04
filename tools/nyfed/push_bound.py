"""A bound on the interchange-consistent attacker (N68, N70): for each onset and interface, the largest push of
the reported interface flow any attacker can make in the DC model under the search's injection bounds. Every
attackable New York load moves within `tau` of its load. A generator bus moves its units within `band` of their
upper limit and inside their operating limits (widened to the true output), and the load it carries (the units'
output less the bus injection, N42) within `tau`. Zero-injection buses and the reference bus hold their true
injection, as in the search. Four reaches:

- `net`: every bus outside New York at its true injection, so New York's net injection and the net interchange
  stay true (individual ties may still move in loops);
- `net_ties`: as `net` and every tie's own flow unchanged, the strictest;
- `net55`: as `net`, but up to the interchange check's threshold may enter New York, drawn from the units outside
  in proportion to their capacity (the attacker who knows the threshold);
- `net_local`: the DC form of the search's attacker, whose false state moves the angles of one subnetwork
  (the New York buses within `HOPS` branches of one end of an interface branch, the slack left out) and holds
  every other angle, so only the subnetwork and its edge change injection, the best anchor taken. It separates
  what the band protects from what the locality of the attacker does.

In the DC model stealth and the step bounds only lower the push; the AC equations can move it either way (on
the reduced network the voltage limits close it where the NYC generator buses sit at vmax, N71). The split of
the push between loads and units shows which bound binds.

    python tools/nyfed/push_bound.py OUT.csv --onsets out/damage_study/damage.json [...] [--per-interface 60]

Run from the folder whose `data/` holds the inputs. The onsets of `--onsets` were drawn for the attacked
interface (Sprain Brook/Dunwoodie South, the first of `NAMES`) and are bounded on it only. `--per-interface N`
adds, for every interface, N onsets drawn among its own near-limit hours (real flow at least 90 percent of the
real limit).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from fdia_graph.nyfed.build import production_builder
from fdia_graph.nyfed.config import Settings, rng
from fdia_graph.nyfed.grid.interfaces import hourly_interfaces
from fdia_graph.nyfed.operator.sced import shift_factors
from fdia_graph.nyfed.scenario.pool import SOLVED

NAMES = ["SPR/DUN-SOUTH", "UPNY CONED", "CENTRAL EAST - VC", "MOSES SOUTH"]
TAUS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.3)
BANDS = (0.02, 0.03, 0.05, 0.1)
FORMS = ("net", "net_ties", "net55", "net_local")
HOPS = 3  # the attacker's subnetwork, as in the search
RISE = 16
CHECK_MW = 55.0  # the interchange check's threshold (tie_check.py)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--onsets", nargs="*", default=[])
    ap.add_argument("--per-interface", type=int, default=0)
    args = ap.parse_args()
    t0 = time.perf_counter()
    b = production_builder("out/pool_2019_v3.h5", ("Am",), workers=1, survey=0)
    att, case, pool = b.att, b.att.net.case, b.pool
    ifs = hourly_interfaces(case, pool.time, "data/NYgrid/Data", NAMES)
    ptdf = shift_factors(case)
    sf = np.array([it.signs @ ptdf[it.branches] for it in ifs])  # [I, N]
    ny = att.owner >= 0
    ties = np.flatnonzero(case.is_ny[case.f] != case.is_ny[case.t])
    pmax_bus = np.zeros(len(case.bus_id))
    np.add.at(pmax_bus, case.unit_bus, np.where(np.isfinite(case.pmax), case.pmax, 0.0))
    ext_w = np.where(~ny, pmax_bus, 0.0)  # where the threshold-aware attacker's interchange enters
    ext_w = ext_w / ext_w.sum()
    # the DC branch-flow and bus matrices, for the local attacker whose state moves one subnetwork only
    bsus = 1.0 / (case.x * np.where(case.tap == 0, 1.0, case.tap))
    inc = np.zeros((len(case.f), len(case.bus_id)))
    inc[np.arange(len(case.f)), case.f] = 1.0
    inc[np.arange(len(case.f)), case.t] = -1.0
    bflow = bsus[:, None] * inc  # branch flow per radian at each bus
    bbus = inc.T @ bflow  # injection per radian
    local = []  # per interface, the interior of each anchor end's subnetwork
    for it in ifs:
        areas = []
        for br in it.branches:
            for end in (int(case.f[br]), int(case.t[br])):
                area = att.area(end, HOPS)
                if area is not None and len(area := area[ny[area]]):
                    areas.append(area)
        local.append(areas)
    jobs = []  # (onset, interface index, source)
    for f in args.onsets:
        for o in json.load(open(f))["onsets"]:
            jobs.append((int(o), 0, "drawn"))
    if args.per_interface:
        g = rng(Settings(), "push_bound")
        solved = pool.solve_kind == SOLVED
        for i, it in enumerate(ifs):
            with np.errstate(invalid="ignore"):
                near = (it.flow >= 0.9 * it.limit) & (it.limit > 0)
            ok = np.flatnonzero(near[RISE - 1 :] & solved[RISE - 1 :])  # onsets whose last rise scan is near
            for o in g.choice(ok, size=min(args.per_interface, len(ok)), replace=False):
                jobs.append((int(o), i, "own"))
    rows = []
    for o, i, source in sorted(set(jobs)):
        r = o + RISE - 1
        vm, va = pool.vm[r], pool.va[r]
        if not np.isfinite(vm).all():
            continue
        load = att.bus_load(vm, va)
        # the load at a generator bus is the units' output less the bus injection (N42); the search lets it move
        # within the load cap on top of the units' band
        v = vm * np.exp(1j * np.deg2rad(va))
        p_inj = (v * np.conj(att.ybus @ v)).real * att.base_mva
        units = np.zeros(len(case.bus_id))
        np.add.at(units, case.unit_bus, np.nan_to_num(pool.unit_p[r]))
        gen_load = np.clip(units - p_inj, 0.0, None)
        free = np.zeros(len(case.bus_id), bool)
        free[att.free_injection_buses(vm, va, np.arange(len(case.bus_id)))] = True
        # the search's interior grows over every edge bus whose injection may not change (a zero-injection bus,
        # a load under the floor): its angle joins the free ones, its injection stays held
        grown = [a_[ny[a_]] for a_ in (att._absorb(vm, va, area) for area in local[i])]
        it = ifs[i]
        H = float(it.limit[r] - it.flow[r])
        for tau in TAUS:
            for band in BANDS:
                # a generator bus moves its units within the band and inside their operating limits (widened to
                # the true output, (7), (9)) and its load within the cap, as the search does; a load bus within the cap
                unit_lo = np.maximum(-band * pmax_bus, np.minimum(att.p_lo, units) - units)
                unit_hi = np.minimum(band * pmax_bus, np.maximum(att.p_hi, units) - units)
                lo_cap = np.where(att.gen_bus, unit_lo - tau * gen_load, -tau * np.clip(load, 0, None))
                hi_cap = np.where(att.gen_bus, unit_hi + tau * gen_load, tau * np.clip(load, 0, None))
                keep = free & ny
                keep[att.slack] = (
                    False  # the reference bus holds its true injection in the search (attacker.py)
                )
                lo_cap, hi_cap = np.where(keep, lo_cap, 0.0), np.where(keep, hi_cap, 0.0)
                cap = hi_cap  # kept for the net55 form's shape
                for form in FORMS:
                    lo, hi = lo_cap.copy(), hi_cap.copy()
                    obj, a_ub, b_ub = -sf[i], None, None
                    a_eq, b_eq = [np.ones(len(cap))], [0.0]  # DC balance over every bus
                    lo[~ny], hi[~ny] = 0.0, 0.0  # every external injection held: the net interchange is true
                    # net_ties: every tie's own flow unchanged
                    if form == "net_ties":
                        a_eq += list(ptdf[ties])
                        b_eq += [0.0] * len(ties)
                    # net55: up to the check's threshold enters New York, drawn from the external units by capacity
                    if form == "net55":
                        obj = -(sf[i] - (sf[i] @ ext_w) * ny)
                        a_eq, b_eq = None, None
                        a_ub = np.vstack([ny.astype(float), -ny.astype(float)])
                        b_ub = [CHECK_MW, CHECK_MW]
                    if form == "net_local":  # angles of the interior free, every injection within its cap
                        best = None
                        for area in grown:
                            r_ = linprog(
                                -(it.signs @ bflow[it.branches][:, area]),
                                A_ub=np.vstack([bbus[:, area], -bbus[:, area]]),
                                b_ub=np.r_[hi_cap, -lo_cap],
                                bounds=[(None, None)] * len(area),
                                method="highs",
                            )
                            if r_.status == 0 and (best is None or r_.fun < best.fun):
                                best = r_
                        res = best if best is not None else r_
                    else:
                        res = linprog(
                            obj,
                            A_ub=a_ub,
                            b_ub=b_ub,
                            A_eq=None if a_eq is None else np.array(a_eq),
                            b_eq=b_eq,
                            bounds=list(zip(lo, hi)),
                            method="highs",
                        )
                    push = -res.fun if res.status == 0 else (np.inf if res.status == 3 else np.nan)
                    from_units = (
                        float(sf[i][att.gen_bus] @ res.x[att.gen_bus])
                        if res.status == 0 and form != "net_local"
                        else np.nan
                    )
                    rows.append(
                        {
                            "onset": o,
                            "time": str(pool.time.astype("datetime64[s]")[o]),
                            "interface": it.name,
                            "source": source,
                            "headroom_mw": H,
                            "tau": tau,
                            "band": band,
                            "form": form,
                            "push_mw": push,
                            "push_from_units_mw": from_units,
                        }
                    )
        print(o, it.name, f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
