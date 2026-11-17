"""The recipe of the New York case file: `case_nygrid.m` made again from NYgrid's `Data/` folder.

    python pipelines/newyork/make_case.py NYGRID_DATA_DIR [OUT.m]

The package reads the grid from `src/fdia_graph/generation/nygrid/case_nygrid.m` (`nygrid_case()`); this script is
how that file was made, kept so the file can be checked against its source. `recipe` applies, in order:
`read_npcc` (NYgrid's `npcc.mat` with the coordinates and zones of `npcc.csv`), `modify_mpc` (every change of
NYgrid's `modifyMPC.m`), `nygrid_units` (the fixed unit list of NYgrid's hourly case), `split_zone_j` (half of
zone J's units moved to bus 81) and `restore_765kv` (the 765 kV line 47-43). OUT may be left out only in a
source checkout, where it is the package's file.
`write_case` lays the file out as `Case.from_matpower` reads it.
A change to the grid from here on is an edit to the file, not to this script.
"""

from __future__ import annotations

import re
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io

from fdia_graph.constants.grid import BusType
from fdia_graph.constants.newyork import (
    FILE_AREA,
    HQ_BUS,
    NY_AREA,
    NYGRID_COMMIT,
    REGIONS,
    ZONE_LEGEND,
    ZONES,
)
from fdia_graph.errors import DataConditionError
from fdia_graph.generation.nygrid.case import CASE_FILE, Case
from fdia_graph.generation.nygrid.tables import read_rows

# NYgrid's NPCC case has two New York areas, 2 upstate and 3 downstate. The case file does not tell them apart:
# every New York bus is `NY_AREA` in a `Case`, and a neighbor's bus keeps its NPCC area number (the area in `REGIONS`).
NPCC_NY_AREAS = (2, 3)
WIDTH = 97  # of a line of the file's header and of the comment above a matrix, before its `% `
ONE_AREA = textwrap.fill(
    f"One area, on purpose: NYgrid's NPCC case has six areas, and this file has one, so the bus area column "
    f"is {FILE_AREA} on every row. Where a bus is, is its zone (the bus zone column): {ZONE_LEGEND}. The "
    f"neighboring regions are the zone codes {REGIONS[0][0]} to {REGIONS[-1][0]}. The package reads a "
    f"bus's region from the zone column, not from the area column.",
    WIDTH,
)
TITLE = f"""CASE_NYGRID  The New York grid of fdia-graph: 140 buses, 227 branches, 271 units.

The New York and neighbors network (NYgrid's 140-bus NPCC case) that every New York pool and
timeline of the package is built on. The package reads this file with matpowercaseframes, the
reader pandapower uses (fdia_graph.generation.nygrid.case.nygrid_case); a change to the grid is
an edit to this file.

It is the network with the units' limits, NOT a solved operating point: the New York units are at
Pg = 0, and the hourly loads and dispatch come from the pool built on this case.

To change the grid, edit rows in place, or add and delete whole rows (a new bus or unit needs its
row in every field that has one row per bus or per unit). Keep one row and one string per line.
An indexed assignment such as mpc.bus(1, 3) = 500; and a field the package does not know are
refused, because the reader would pass over them without a word. A bus area other than {FILE_AREA} and a
zone outside 1 to {REGIONS[-1][0]} are refused as well. A branch with status 0 is left out. The package does
not use gen mBase or branch rateB, rateC, angmin and angmax.

Derived from NYgrid (https://github.com/AndersonEnergyLab-Cornell/NYgrid, MIT License), commit
{NYGRID_COMMIT}, by pipelines/newyork/make_case.py:
  1. Data/npcc.mat, the NPCC 140-bus case (branches out of service left out), with the
     coordinates and NYISO load zones of Data/npcc.csv;
  2. every change of NYgrid's modifyMPC.m: the bus types of 39, 45, 62, 77 (PV), 53, 54, 68, 72,
     78 (PQ) and 74 (reference), loads made positive, the PJM-IESO lines 84-116, 87-115 and
     90-114 removed, the ten tie ratings (rateA, MVA) and the lines 38-77 and 43-38 added;
  3. the fixed unit list of NYgrid's hourly case (updateOpCond.m L60-381): thermal units from
     genParamAll_newramp.csv at the buses of gen_bus_assignment.csv, then nuclear and hydro
     units from RenewableGen.csv, the units of the neighboring regions (Import) and the
     Hydro-Quebec import at bus 48. Qmax and Qmin of 9999 and -9999 mean the source gives no
     reactive limit;
  4. zone J's units, all at bus 82 in NYgrid, split evenly over buses 81 and 82, and bus 81
     made PV;
  5. the 765 kV Massena-Marcy line 47-43, which modifyMPC.m carries commented out.

Units: MW and MVAr, impedances in p.u. on baseMVA, angles in degrees. A rateA of 0 is no rating.

{ONE_AREA}

Beyond MATPOWER's own fields (mpc.bus_name and mpc.genfuel among them) the file has
mpc.bus_latlon and mpc.gen_ramp, each described above its rows. There is no mpc.gencost: unit
costs are read with the hourly data, not from this file."""


def read_npcc(path: str | Path) -> Case:
    """Case from `Data/npcc.mat`, with coordinates and zones from `Data/npcc.csv` beside it."""
    path = Path(path)
    mpc = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)["mpc"]
    bus, branch, gen = mpc.bus, mpc.branch, mpc.gen
    meta = {int(r["idx"]): r for r in read_rows(path.with_name("npcc.csv"))}

    bus_id = bus[:, 0].astype(int)
    pos = {b: i for i, b in enumerate(bus_id.tolist())}
    npcc_area = bus[:, 6].astype(int)
    is_ny = np.isin(npcc_area, NPCC_NY_AREAS)
    strangers = sorted(set(npcc_area[~is_ny].tolist()) - {area for _, _, area in REGIONS})
    if strangers:
        raise DataConditionError(f"{path}: area {strangers[0]} is not New York's or a neighbor's (REGIONS)")
    area = np.where(is_ny, NY_AREA, npcc_area)
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
        is_ny=is_ny,
        f=np.array([pos[int(b)] for b in branch[on, 0]]),
        t=np.array([pos[int(b)] for b in branch[on, 1]]),
        r=branch[on, 2].copy(),
        x=branch[on, 3].copy(),
        b=branch[on, 4].copy(),
        g=np.zeros(int(on.sum())),
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
_TYPES = {
    39: BusType.PV,
    77: BusType.PV,
    45: BusType.PV,
    62: BusType.PV,
    72: BusType.PQ,
    53: BusType.PQ,
    54: BusType.PQ,
    68: BusType.PQ,
    78: BusType.PQ,
    74: BusType.REF,
}
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
        g=np.r_[case.g[keep], np.zeros(n_add)],
        tap=np.r_[case.tap[keep], np.ones(n_add)],
        shift=np.r_[case.shift[keep], np.zeros(n_add)],
        rating=np.r_[rating[keep], np.zeros(n_add)],
    )


# updateOpCond.m L76-90: the six nuclear units by bus, in the order the hourly case lists them.
_NUCLEAR = {53: [581.7], 50: [854.5, 629.0, 1299.0], 74: [1025.9, 1039.9]}


def nygrid_units(case: Case, data_dir: str | Path) -> Case:
    """The fixed unit list NYgrid's hourly case is built from (updateOpCond.m L60-381): thermal units
    from the RGGI table sorted by name and PTID as MATLAB's joins leave them, then nuclear, hydro, the
    external units of the NPCC case and the HQ import. Hours change dispatch and status, never the list."""
    data_dir = Path(data_dir)
    pos = {b: i for i, b in enumerate(case.bus_id.tolist())}
    assignment = read_rows(data_dir / "gen_bus_assignment.csv", "utf-8-sig")
    hub = {(r["NYISOName"], r["PTID"]): int(r["HubName"]) for r in assignment}
    thermal = sorted(
        read_rows(data_dir / "genParamAll_newramp.csv", "utf-8-sig"),
        key=lambda r: (r["NYISOName"], int(r["PTID"])),
    )
    renew = read_rows(data_dir / "RenewableGen.csv", "utf-8-sig")

    bus, fuel, pmin, pmax, qmin, qmax, ramp = [], [], [], [], [], [], []

    def add(b, kind, lo, hi, qlo=-9999.0, qhi=9999.0, rate=np.nan):
        bus.append(pos[b])
        fuel.append(kind)
        pmin.append(lo)
        pmax.append(hi)
        qmin.append(qlo)
        qmax.append(qhi)
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


def split_zone_j(case: Case) -> Case:
    """NYgrid places every zone J unit at bus 82 and none at bus 81, though allocateLoad.m splits zone J's
    load evenly over the two (neither has base load). Move units to bus 81 in proportion to its half of
    the load, taking every other unit in list order, and make bus 81 a PV bus."""
    pos = {b: i for i, b in enumerate(case.bus_id.tolist())}
    b81, b82 = pos[81], pos[82]
    at82 = np.flatnonzero(case.unit_bus == b82)
    n = int(round(len(at82) * 0.5))
    unit_bus = case.unit_bus.copy()
    unit_bus[at82[0 : 2 * n : 2]] = b81
    bus_type = case.bus_type.copy()
    bus_type[b81] = BusType.PV
    return replace(case, unit_bus=unit_bus, bus_type=bus_type)


def restore_765kv(case: Case) -> Case:
    """Add the 765 kV Massena-Marcy line (47-43) that modifyMPC.m carries commented out at L75."""
    pos = {b: i for i, b in enumerate(case.bus_id.tolist())}
    return replace(
        case,
        f=np.r_[case.f, pos[47]],
        t=np.r_[case.t, pos[43]],
        r=np.r_[case.r, 0.004],
        x=np.r_[case.x, 0.0085],
        b=np.r_[case.b, 0.0],
        g=np.r_[case.g, 0.0],
        tap=np.r_[case.tap, 1.0],
        shift=np.r_[case.shift, 0.0],
        rating=np.r_[case.rating, 0.0],
    )


def recipe(data_dir: str | Path) -> Case:
    """The case of the file, built from NYgrid's `Data/` folder."""
    data_dir = Path(data_dir)
    case = nygrid_units(modify_mpc(read_npcc(data_dir / "npcc.mat")), data_dir)
    return restore_765kv(split_zone_j(case))


def _number(value: float) -> str:
    """A float in the fewest digits that read back to the same bits, spelled as MATLAB does (`100`, `NaN`)."""
    text = repr(float(value)).replace("nan", "NaN").replace("inf", "Inf")
    return text[:-2] if text.endswith(".0") else text


def _matrix(name: str, note: str, heading: str, columns: list) -> list[str]:
    """`mpc.<name> = [ ... ];` with one row per line, under a comment (`note`, wrapped at `WIDTH`) and a line of
    column headings."""
    first, *more = textwrap.wrap(note, WIDTH)
    comment = [f"%% {first}", *(f"% {line}" for line in more)]
    rows = ["\t" + "\t".join(_number(v) for v in row) + ";" for row in zip(*columns)]
    return ["", *comment, "%\t" + heading.replace(" ", "\t"), f"mpc.{name} = [", *rows, "];"]


def _strings(name: str, note: str, values: np.ndarray) -> list[str]:
    """`mpc.<name> = { ... };` with one quoted string per line (MATLAB doubles a quote inside a string)."""
    rows = ["\t'" + str(v).replace("'", "''") + "';" for v in values]
    return ["", f"%% {note}", f"mpc.{name} = {{", *rows, "};"]


def write_case(case: Case, path: str | Path, title: str) -> None:
    """Write `case` as the MATPOWER version-2 case file `Case.from_matpower` reads, under the comment `title`.
    Positions become bus numbers. The bus `zone` column says where each bus is (`ZONE_LEGEND`): a New York bus's
    load zone, and the code of the neighbor (`REGIONS`) whose area a bus outside New York has; a bus outside
    New York with any other area is refused. The bus `area` column is `FILE_AREA` on every row, so the area of a
    New York bus is not written: a case whose New York buses carry another number (3, in a file stored before
    the case had one area) reads back with `NY_AREA` there. A tap of 1 is written as MATPOWER's 0, and
    `mpc.dcline` is written only when the case has HVDC links. A branch with a shunt conductance (`Case.g`,
    which an IEEE case read from pandapower can have) is refused: MATPOWER's branch table has no column for it."""
    path = Path(path)
    if not re.fullmatch(r"[A-Za-z]\w*", path.stem):
        raise DataConditionError(f"{path}: a case file is a MATLAB function, so {path.stem!r} cannot name it")
    if not (case.is_ny == (case.zone != "")).all():
        raise DataConditionError(f"{path}: a bus must be in New York exactly when it has a load zone")
    code = {area: code for code, _, area in REGIONS}
    if not np.isin(case.area[~case.is_ny], list(code)).all():
        raise DataConditionError(f"{path}: a bus outside New York must have the area of a neighbor (REGIONS)")
    if case.g.any():
        raise DataConditionError(
            f"{path}: a branch has a shunt conductance, which a MATPOWER case file cannot hold"
        )
    n, m, g, d = len(case.bus_id), len(case.f), len(case.unit_bus), len(case.dc_f)
    number = case.bus_id
    zone = [ZONES.index(z) + 1 if z else code[int(a)] for z, a in zip(case.zone, case.area)]
    ratio = np.where(case.tap == 1.0, 0.0, case.tap)
    area = np.full(n, FILE_AREA)
    bus = [number, case.bus_type, case.pd, case.qd, case.gs, case.bs, area, case.vm, case.va, case.kv]
    bus += [zone, case.vmax, case.vmin]
    gen = [number[case.unit_bus], case.unit_p, case.unit_q, case.qmax, case.qmin, case.unit_vg]
    gen += [np.full(g, case.base_mva), case.unit_on, case.pmax, case.pmin]
    branch = [number[case.f], number[case.t], case.r, case.x, case.b, case.rating, np.zeros(m), np.zeros(m)]
    branch += [ratio, case.shift, np.ones(m), np.full(m, -360.0), np.full(m, 360.0)]
    zero, one = np.zeros(d), np.ones(d)
    dcline = [number[case.dc_f], number[case.dc_t], one, zero, zero, zero, zero, one, one]
    dcline += [case.dc_pmin, case.dc_pmax, zero, zero, zero, zero, zero, zero]
    out = [
        f"function mpc = {path.stem}",
        *("% " + line if line else "%" for line in title.splitlines()),
        "",
        "%% MATPOWER Case Format : Version 2",
        "mpc.version = '2';",
        "",
        "%% system MVA base",
        f"mpc.baseMVA = {_number(case.base_mva)};",
        *_matrix(
            "bus",
            f"bus data (area: {FILE_AREA} on every row; zone: {ZONE_LEGEND})",
            "bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin",
            bus,
        ),
        *_matrix("gen", "generator data", "bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin", gen),
        *_matrix(
            "branch",
            "branch data (ratio 0 is a line)",
            "fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax",
            branch,
        ),
    ]
    if d:
        heading = "fbus tbus status Pf Pt Qf Qt Vf Vt Pmin Pmax QminF QmaxF QminT QmaxT loss0 loss1"
        out += _matrix(
            "dcline", "HVDC links: from bus, to bus and the limits Pmin, Pmax (MW)", heading, dcline
        )
    out += [
        *_strings("bus_name", "bus names, one per row of mpc.bus", case.name),
        *_matrix("bus_latlon", "where each bus is, in degrees", "latitude longitude", [case.lat, case.lon]),
        *_strings(
            "genfuel", "fuel of each unit, one per row of mpc.gen (Import is a neighbor's unit)", case.fuel
        ),
        *_matrix(
            "gen_ramp",
            "load-following (AGC) ramp of each unit, one per row of mpc.gen, NaN where the source gives none",
            "MW/min",
            [case.ramp],
        ),
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as file:  # LF on every platform: the file is data
        file.write("\n".join(out))


USAGE = (
    "usage: python pipelines/newyork/make_case.py NYGRID_DATA_DIR [OUT.m]  "
    "(OUT.m may be left out only in a source checkout, where it is the package's case_nygrid.m)"
)


def main(arguments: list[str]) -> int:
    """Write the case file from the NYgrid `Data/` folder named first, at the path named second."""
    in_checkout = Path(__file__).resolve().parents[2] in CASE_FILE.resolve().parents
    if not 1 <= len(arguments) <= 2 or (len(arguments) == 1 and not in_checkout):
        print(USAGE, file=sys.stderr)
        return 2
    out = Path(arguments[1]) if len(arguments) == 2 else CASE_FILE
    built = recipe(arguments[0])
    write_case(built, out, TITLE)
    print(
        f"{out}: {len(built.bus_id)} buses, {len(built.f)} branches, {len(built.unit_bus)} units", flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
