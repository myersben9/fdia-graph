"""Read NYgrid's NPCC-140 case and apply NYgrid's own modifications (modifyMPC.m)."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io

from fdia_graph.nyfed.grid.case import PQ, PV, REF, Case

NY_AREAS = (2, 3)


def read_npcc(path: str | Path) -> Case:
    """Case from `Data/npcc.mat`, with coordinates and zones from `Data/npcc.csv` beside it."""
    path = Path(path)
    mpc = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)["mpc"]
    bus, branch, gen = mpc.bus, mpc.branch, mpc.gen
    meta = {int(r["idx"]): r for r in csv.DictReader(open(path.with_name("npcc.csv"), encoding="utf-8"))}

    bus_id = bus[:, 0].astype(int)
    pos = {b: i for i, b in enumerate(bus_id.tolist())}
    area = bus[:, 6].astype(int)
    zone = np.array([meta[b]["zone"] if meta[b]["zone"] != "NA" else "" for b in bus_id])
    tap = branch[:, 8].copy()
    tap[tap == 0] = 1.0  # MATPOWER writes 0 for a line
    on = branch[:, 10] > 0
    n_gen = len(gen)
    empty = np.zeros(0)
    return Case(
        base_mva=float(mpc.baseMVA),
        bus_id=bus_id,
        name=np.array([str(s).strip() for s in mpc.bus_name]),
        zone=zone,
        area=area,
        kv=bus[:, 9].copy(),
        vmin=bus[:, 12].copy(),
        vmax=bus[:, 11].copy(),
        lat=np.array([float(meta[b]["ycoord"]) for b in bus_id]),
        lon=np.array([float(meta[b]["xcoord"]) for b in bus_id]),
        is_ny=np.isin(area, NY_AREAS),
        f=np.array([pos[int(b)] for b in branch[on, 0]]),
        t=np.array([pos[int(b)] for b in branch[on, 1]]),
        r=branch[on, 2].copy(),
        x=branch[on, 3].copy(),
        b=branch[on, 4].copy(),
        tap=tap[on],
        shift=branch[on, 9].copy(),
        rating=branch[on, 5].copy(),
        unit_bus=np.array([pos[int(b)] for b in gen[:, 0]]),
        fuel=np.full(n_gen, ""),
        pmin=gen[:, 9].copy(),
        pmax=gen[:, 8].copy(),
        qmin=gen[:, 4].copy(),
        qmax=gen[:, 3].copy(),
        ramp=np.full(n_gen, np.nan),  # NYgrid's unit list (nygrid_units) carries the ramps
        dc_f=empty.astype(int),
        dc_t=empty.astype(int),
        dc_pmin=empty,
        dc_pmax=empty,
        bus_type=bus[:, 1].astype(int),
        pd=bus[:, 2].copy(),
        qd=bus[:, 3].copy(),
        gs=bus[:, 4].copy(),
        bs=bus[:, 5].copy(),
        vm=bus[:, 7].copy(),
        va=bus[:, 8].copy(),
        unit_p=gen[:, 1].copy(),
        unit_q=gen[:, 2].copy(),
        unit_vg=gen[:, 5].copy(),
        unit_on=gen[:, 7] > 0,
    )


# modifyMPC.m: bus type changes, the three PJM-IESO lines it deletes, the tie ratings it sets
# (RATE_A to RATE_C, MVA) and the two lines it adds for the E-G corridor.
_TYPES = {39: PV, 77: PV, 45: PV, 62: PV, 72: PQ, 53: PQ, 54: PQ, 68: PQ, 78: PQ, 74: REF}
_DELETED = [(84, 116), (87, 115), (90, 114)]
_RATINGS = {
    (66, 134): 315,
    (67, 138): 660,
    (81, 125): 660,
    (75, 124): 2000,
    (60, 140): 550,
    (29, 37): 200,
    (35, 73): 1400,
    (48, 100): 450,
    (54, 102): 650,
    (54, 103): 650,
}
_ADDED = [(38, 77, 0.02, 0.0147), (43, 38, 0.02, 0.0427)]


def _csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# updateOpCond.m L76-90: the six nuclear units by bus, in the order the hourly case lists them.
_NUCLEAR = {53: [581.7], 50: [854.5, 629.0, 1299.0], 74: [1025.9, 1039.9]}
HQ_BUS = 48  # updateOpCond.m L367-377: Hydro-Quebec import as a unit at bus 48, -1100 to 1500 MW


def nygrid_units(case: Case, data_dir: str | Path) -> Case:
    """The fixed unit list NYgrid's hourly case is built from (updateOpCond.m L60-381): thermal units
    from the RGGI table sorted by name and PTID as MATLAB's joins leave them, then nuclear, hydro, the
    external units of the NPCC case and the HQ import. Hours change dispatch and status, never the list."""
    data_dir = Path(data_dir)
    pos = {b: i for i, b in enumerate(case.bus_id.tolist())}
    hub = {(r["NYISOName"], r["PTID"]): int(r["HubName"]) for r in _csv(data_dir / "gen_bus_assignment.csv")}
    thermal = sorted(
        _csv(data_dir / "genParamAll_newramp.csv"), key=lambda r: (r["NYISOName"], int(r["PTID"]))
    )
    renew = _csv(data_dir / "RenewableGen.csv")

    bus, fuel, pmin, pmax, qmin, qmax, ramp = [], [], [], [], [], [], []

    def add(b, kind, lo, hi, qlo=-9999.0, qhi=9999.0, rate=np.nan):
        (
            bus.append(pos[b]),
            fuel.append(kind),
            pmin.append(lo),
            pmax.append(hi),
            qmin.append(qlo),
            qmax.append(qhi),
        )
        ramp.append(rate)

    # ramps: the thermal units' maxRampAgc (MW/min, MATPOWER's RAMP_AGC, the column updateOpCond.m L235
    # copies; maxRamp10 is exactly ten times it); nuclear 0 (it does not follow load); hydro and imports
    # have no ramp data (NaN)
    for r in thermal:
        add(
            hub[(r["NYISOName"], r["PTID"])],
            r["FuelType"],
            float(r["minPower"]),
            float(r["maxPower"]),
            rate=float(r["maxRampAgc"]),
        )
    for r in renew:
        for cap in _NUCLEAR.get(int(r["bus_id"]), []) if r["Pg Nuclear Cap"] else []:
            add(int(r["bus_id"]), "Nuclear", 0.0, cap, rate=0.0)
    for r in renew:
        if r["Pg Hydro Cap"] and float(r["Pg Hydro Cap"]) != 0:
            add(int(r["bus_id"]), "Hydro", 0.0, float(r["Pg Hydro Cap"]))
    n_ny = len(bus)
    ext = ~case.is_ny[case.unit_bus]
    for i in np.flatnonzero(ext):
        add(
            int(case.bus_id[case.unit_bus[i]]),
            "Import",
            0.0,
            float(case.pmax[i]),
            float(case.qmin[i]),
            float(case.qmax[i]),
        )
    add(HQ_BUS, "Import", -1100.0, 1500.0)

    n = len(bus)
    unit_p = np.zeros(n)
    unit_p[n_ny:-1] = case.unit_p[ext]
    unit_vg = np.ones(n)
    unit_vg[n_ny:-1] = case.unit_vg[ext]
    return replace(
        case,
        unit_bus=np.array(bus),
        fuel=np.array(fuel),
        pmin=np.array(pmin),
        pmax=np.array(pmax),
        qmin=np.array(qmin, dtype=float),
        qmax=np.array(qmax, dtype=float),
        ramp=np.array(ramp, dtype=float),
        unit_p=unit_p,
        unit_q=np.zeros(n),
        unit_vg=unit_vg,
        unit_on=np.ones(n, dtype=bool),
    )


def modify_mpc(case: Case) -> Case:
    """Every change `modifyMPC.m` makes to the NPCC-140 case."""
    pos = {b: i for i, b in enumerate(case.bus_id.tolist())}
    ends = list(zip(case.bus_id[case.f].tolist(), case.bus_id[case.t].tolist()))

    bus_type = case.bus_type.copy()
    for b, kind in _TYPES.items():
        bus_type[pos[b]] = kind

    keep = np.array([e not in _DELETED for e in ends])
    rating = case.rating.copy()
    for i, e in enumerate(ends):
        if e in _RATINGS:
            rating[i] = _RATINGS[e]

    add_f = np.array([pos[a] for a, *_ in _ADDED])
    add_t = np.array([pos[b] for _, b, *_ in _ADDED])
    n_add = len(_ADDED)
    return replace(
        case,
        pd=np.abs(case.pd),
        bus_type=bus_type,
        f=np.r_[case.f[keep], add_f],
        t=np.r_[case.t[keep], add_t],
        r=np.r_[case.r[keep], [a[2] for a in _ADDED]],
        x=np.r_[case.x[keep], [a[3] for a in _ADDED]],
        b=np.r_[case.b[keep], np.zeros(n_add)],
        tap=np.r_[case.tap[keep], np.ones(n_add)],
        shift=np.r_[case.shift[keep], np.zeros(n_add)],
        rating=np.r_[rating[keep], np.zeros(n_add)],
    )
