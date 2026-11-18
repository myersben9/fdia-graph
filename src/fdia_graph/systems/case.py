"""The grid as one frozen record of arrays, read from a MATPOWER case file or from a pandapower net.

The New York grid is the file `case_nygrid.m` beside this module: `nygrid_case()` reads it, and a change to the grid
is an edit to that file. `Case.from_matpower` reads a case file with matpowercaseframes, the reader pandapower uses,
taking each column by MATPOWER's own name, and `Case.check` refuses a file whose fields disagree.
`Case.from_pandapower` reads a pandapower net, and `Case.from_ieee` the IEEE case pandapower ships for a bus
count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fdia_graph.constants import newyork
from fdia_graph.constants.grid import (
    IEEE_CASES,
    PANDAPOWER_IN_SERVICE,
    PANDAPOWER_NOT_READ,
    PANDAPOWER_SCALED,
    BranchColumn,
    BusColumn,
    BusType,
)
from fdia_graph.errors import DataConditionError
from fdia_graph.formulas.network import series_admittance
from fdia_graph.models.inputs import ModelledNet, UnitCosts
from fdia_graph.schema import Static

if TYPE_CHECKING:
    import pandas as pd
    from matpowercaseframes import CaseFrames

    from fdia_graph.systems.pandapower_types import PandapowerNet


CASE_FILE = Path(__file__).with_name("case_nygrid.m")
STATEMENT = re.compile(r"\s*mpc\.(\w+)\s*(\(?)")  # `mpc.<field>`, and the bracket of an indexed assignment
QUOTED = re.compile(r"'((?:[^']|'')*)'")  # a 'string'; '' is a quote inside it


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
    area: np.ndarray  # `NY_AREA` in New York, the neighbor's area in `REGIONS` outside it
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
    g: np.ndarray  # shunt conductance (a transformer's iron losses), 0 on a MATPOWER branch
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
        """A copy with every bus's voltage limits widened by `margin` (p.u.). It is the attacker's sensitivity
        case: the reduced network pins New York City buses at their upper limit."""
        return replace(self, vmin=self.vmin - margin, vmax=self.vmax + margin)

    def static_graph(self) -> dict[str, np.ndarray]:
        """The static graph every timeline file carries, by `Static` name: the per-unit branch model
        with its series admittance g_s + j b_s = 1 / (r + j x), and the bus shunts (MW and MVAr at
        1 pu). Every branch of a case is in service; a transformer is a branch between two voltage
        levels, or with a tap ratio or a phase shift."""
        series = series_admittance(self.r, self.x)
        transformer = (self.kv[self.f] != self.kv[self.t]) | (self.tap != 1.0) | (self.shift != 0.0)
        return {
            Static.EDGE_R: self.r,
            Static.EDGE_X: self.x,
            Static.EDGE_B: self.b,
            Static.EDGE_G: self.g,
            Static.EDGE_GS: series.real,
            Static.EDGE_BS: series.imag,
            Static.EDGE_TAP: self.tap,
            Static.EDGE_SHIFT: self.shift,
            Static.EDGE_STATUS: np.ones(len(self.f)),
            Static.EDGE_IS_TRAFO: transformer.astype(np.float64),
            Static.BUS_SHUNT_G: self.gs,
            Static.BUS_SHUNT_B: self.bs,
        }

    # ---- reading a MATPOWER case file ----

    @classmethod
    def from_matpower(cls, path: str | Path) -> Case:
        """The checked `Case` of the MATPOWER case file at `path`. A branch out of service is left out and a
        tap ratio of 0 (MATPOWER's mark of a line) is 1. Beyond MATPOWER's fields the file has `mpc.bus_latlon`
        (degrees) and `mpc.gen_ramp` (MW/min, NaN where unknown). A statement the reader would silently skip
        (an indexed assignment, an unknown field) raises `DataConditionError`."""
        from matpowercaseframes import CaseFrames  # here, so that a Case can be used without the [ny] extra

        text = Path(path).read_text(encoding="utf-8")
        cls._check_statements(text, path)
        try:
            case = cls._from_frames(
                CaseFrames(str(path), allow_any_keys=True, update_index=False), text, path
            )
        except DataConditionError:
            raise
        except KeyError as error:  # a column name the reader's table does not have
            raise DataConditionError(f"{path}: the case file has no column {error.args[0]!r}") from error
        except Exception as error:  # a row with a value missing or too many, a word among the numbers
            raise DataConditionError(f"{path}: cannot be read as a MATPOWER case file: {error!r}") from error
        case.check(str(path))
        return case

    @classmethod
    def _from_frames(cls, frames: CaseFrames, text: str, path: str | Path) -> Case:
        """The `Case` of the tables matpowercaseframes read, each column by MATPOWER's name. The two lists of
        strings come from `text`, the file itself."""
        bus, gen = frames.bus, frames.gen
        branch = frames.branch[frames.branch["BR_STATUS"] > 0]
        links = frames.dcline if newyork.DCLINE in frames.attributes else None
        col = cls._column
        position = cls._bus_positions(col(bus, "BUS_I"), path)
        zone, area = cls._zones_and_areas(col(bus, "ZONE"), col(bus, "BUS_AREA"), path)
        ratio = col(branch, "TAP")
        tables = {"bus": bus, "branch": branch, "gen": gen, newyork.DCLINE: links}
        plain = {field: col(tables[table], name) for field, (table, name) in newyork.CASE_COLUMNS.items()}
        return cls(
            base_mva=float(frames.baseMVA),
            bus_id=col(bus, "BUS_I").astype(int),
            name=cls._strings(text, newyork.BUS_NAME),
            zone=zone,
            area=area,
            lat=col(frames.bus_latlon, 0),
            lon=col(frames.bus_latlon, 1),
            is_ny=zone != "",
            f=cls._positions(col(branch, "F_BUS"), position, "mpc.branch", path),
            t=cls._positions(col(branch, "T_BUS"), position, "mpc.branch", path),
            g=np.zeros(len(branch)),
            tap=np.where(ratio == 0, 1.0, ratio),
            unit_bus=cls._positions(col(gen, "GEN_BUS"), position, "mpc.gen", path),
            fuel=cls._strings(text, newyork.GEN_FUEL),
            ramp=col(frames.gen_ramp, 0),
            dc_f=cls._positions(col(links, "F_BUS"), position, "mpc.dcline", path),
            dc_t=cls._positions(col(links, "T_BUS"), position, "mpc.dcline", path),
            bus_type=col(bus, "BUS_TYPE").astype(int),
            unit_on=col(gen, "GEN_STATUS") > 0,
            **plain,
        )

    # ---- reading a pandapower net ----

    @classmethod
    def from_ieee(cls, buses: int) -> Case:
        """The IEEE case of `buses` buses that pandapower ships, with its base power flow solved. The
        net itself is not kept."""
        # pandapower is heavy and optional: imported only when generating
        import pandapower as pp
        import pandapower.networks as pn

        net = getattr(pn, IEEE_CASES[buses])()
        pp.runpp(net)
        return cls.from_pandapower(net)

    @classmethod
    def from_pandapower(cls, net: PandapowerNet) -> Case:
        """The checked `Case` of a pandapower net whose power flow is solved (`pandapower.runpp`, which
        builds the per-unit branch model read here): the buses and branches in the net's own order,
        lines then transformers. Ratings, places and ramps are not read (0 and NaN). A net holding what
        a `Case` does not model (`ModelledNet`) raises `DataConditionError`."""
        cls._modelled(net)
        row = net._pd2ppc_lookups["bus"][net.bus.index.to_numpy()]  # each bus's row in the solver's tables
        position = np.argsort(row)  # and each row's bus
        bus, branch = net._ppc["bus"].real[row], net._ppc["branch"].real
        buses, none = len(net.bus), np.zeros(0)
        case = cls(
            base_mva=float(net.sn_mva),
            bus_id=net.bus["name"].to_numpy().astype(int),  # MATPOWER's bus numbers
            name=net.bus["name"].to_numpy().astype(str),
            zone=np.full(buses, ""),
            area=np.zeros(buses, int),
            kv=net.bus["vn_kv"].to_numpy(float),
            vmin=net.bus["min_vm_pu"].to_numpy(float),
            vmax=net.bus["max_vm_pu"].to_numpy(float),
            lat=np.full(buses, np.nan),
            lon=np.full(buses, np.nan),
            is_ny=np.zeros(buses, bool),
            f=position[branch[:, BranchColumn.F_BUS].astype(int)],
            t=position[branch[:, BranchColumn.T_BUS].astype(int)],
            r=branch[:, BranchColumn.BR_R].copy(),
            x=branch[:, BranchColumn.BR_X].copy(),
            b=branch[:, BranchColumn.BR_B].copy(),
            g=branch[:, BranchColumn.BR_G].copy(),
            tap=branch[:, BranchColumn.TAP].copy(),  # pandapower writes 1 for a line
            shift=branch[:, BranchColumn.SHIFT].copy(),
            rating=np.zeros(len(branch)),
            dc_f=none.astype(int),
            dc_t=none.astype(int),
            dc_pmin=none,
            dc_pmax=none,
            bus_type=cls._pandapower_bus_types(net),
            gs=bus[:, BusColumn.GS].copy(),  # MW at 1 pu
            bs=bus[:, BusColumn.BS].copy(),  # MVAr at 1 pu
            vm=net.res_bus["vm_pu"].to_numpy(float),
            va=net.res_bus["va_degree"].to_numpy(float),
            **cls._pandapower_loads(net),
            **cls._pandapower_units(net),
        )
        case.check("the pandapower net")
        return case

    @staticmethod
    def _modelled(net: PandapowerNet) -> ModelledNet:
        """What `net` holds that a `Case` would drop or misread, as the model that refuses any of it."""
        off = {name: ~net[name]["in_service"].to_numpy(bool) for name in (*PANDAPOWER_IN_SERVICE, "sgen")}
        # an idle static generator out of service changes nothing (IEEE-200 has eleven)
        off["sgen"] &= (net.sgen["p_mw"].to_numpy(float) != 0) | (net.sgen["q_mvar"].to_numpy(float) != 0)
        scaled = {name: net[name]["scaling"].to_numpy(float) != 1 for name in PANDAPOWER_SCALED}
        status = net._ppc["branch"].real[:, BranchColumn.BR_STATUS]
        return ModelledNet(
            out_of_service=tuple(f"{name} {i}" for name, rows in off.items() for i in net[name].index[rows]),
            scaled=tuple(f"{name} {i}" for name, rows in scaled.items() for i in net[name].index[rows]),
            not_read=tuple(name for name in PANDAPOWER_NOT_READ if name in net and len(net[name])),
            open_branches=int((status == 0).sum()),
            buses=len(net.bus),
            solved_buses=len(net._ppc["bus"]),
        )

    @staticmethod
    def _pandapower_bus_types(net: PandapowerNet) -> np.ndarray:
        """Each bus's type: the reference where the external grid connects, PV where a generator
        holds the voltage, PQ elsewhere (a static generator leaves its bus PQ)."""
        kind = np.full(len(net.bus), int(BusType.PQ))
        kind[net.gen["bus"].to_numpy(int)] = BusType.PV
        kind[net.ext_grid["bus"].to_numpy(int)] = BusType.REF
        return kind

    @staticmethod
    def _pandapower_loads(net: PandapowerNet) -> dict[str, np.ndarray]:
        """The load fields: each bus's load (MW, MVAr), summed over the load elements at the bus."""
        at = net.load["bus"].to_numpy(int)
        pd, qd = np.zeros(len(net.bus)), np.zeros(len(net.bus))
        np.add.at(pd, at, net.load["p_mw"].to_numpy(float))
        np.add.at(qd, at, net.load["q_mvar"].to_numpy(float))
        return {"pd": pd, "qd": qd}

    @staticmethod
    def _pandapower_units(net: PandapowerNet) -> dict[str, np.ndarray]:
        """The unit fields: the generators, then the static generators (their output is fixed, so it
        is both of their limits), then the external grid, whose output is the base power flow's."""
        gen, static, ext = net.gen, net.sgen, net.ext_grid
        fixed_p, fixed_q = static["p_mw"].to_numpy(float), static["q_mvar"].to_numpy(float)
        units = len(gen) + len(static) + len(ext)
        return {
            "unit_bus": np.r_[gen["bus"], static["bus"], ext["bus"]].astype(int),
            "fuel": np.full(units, ""),
            "pmin": np.r_[gen["min_p_mw"], fixed_p, ext["min_p_mw"]].astype(float),
            "pmax": np.r_[gen["max_p_mw"], fixed_p, ext["max_p_mw"]].astype(float),
            "qmin": np.r_[gen["min_q_mvar"], fixed_q, ext["min_q_mvar"]].astype(float),
            "qmax": np.r_[gen["max_q_mvar"], fixed_q, ext["max_q_mvar"]].astype(float),
            "ramp": np.full(units, np.nan),
            "unit_p": np.r_[gen["p_mw"], fixed_p, net.res_ext_grid["p_mw"]].astype(float),
            "unit_q": np.r_[net.res_gen["q_mvar"], fixed_q, net.res_ext_grid["q_mvar"]].astype(float),
            "unit_vg": np.r_[gen["vm_pu"], np.full(len(static), np.nan), ext["vm_pu"]].astype(float),
            "unit_on": np.ones(units, bool),
        }

    @staticmethod
    def pandapower_costs(net: PandapowerNet) -> UnitCosts:
        """Each unit's cost in the net's `poly_cost` table, in the unit order of `from_pandapower`: the
        linear term ($/MWh) and the quadratic term ($/MW^2h). The constant term is not read. A static
        generator without a cost costs 0 (its output is fixed); a generator or an external grid without
        one is refused by `UnitCosts`."""
        table = net.poly_cost.set_index(["et", "element"])
        units = [(kind, int(i)) for kind in ("gen", "sgen", "ext_grid") for i in net[kind].index]
        priced = table.reindex(units)
        none = priced["cp1_eur_per_mw"].isna().to_numpy()
        return UnitCosts(
            linear=priced["cp1_eur_per_mw"].fillna(0.0).to_numpy(float),
            quadratic=priced["cp2_eur_per_mw2"].fillna(0.0).to_numpy(float),
            unpriced=tuple(f"{kind} {i}" for (kind, i), no in zip(units, none) if no and kind != "sgen"),
        )

    def check(self, source: str) -> None:
        """Refuse a case whose parts disagree, naming `source` (its file): one name and one place per bus,
        one fuel and one ramp per unit, bus types that are MATPOWER's, one reference bus."""
        buses, units = len(self.bus_id), len(self.unit_bus)
        counts = {
            f"mpc.{newyork.BUS_NAME}": (len(self.name), buses),
            "mpc.bus_latlon": (len(self.lat), buses),
            f"mpc.{newyork.GEN_FUEL}": (len(self.fuel), units),
            "mpc.gen_ramp": (len(self.ramp), units),
        }
        for field, (rows, expected) in counts.items():
            if rows != expected:
                raise DataConditionError(
                    f"{source}: {field} has {rows} rows, the case has {expected} of them"
                )
        if not np.isin(self.bus_type, (BusType.PQ, BusType.PV, BusType.REF)).all():
            raise DataConditionError(
                f"{source}: mpc.bus: a bus type must be {BusType.PQ}, {BusType.PV} or {BusType.REF}"
            )
        references = int((self.bus_type == BusType.REF).sum())
        if references != 1:
            what = f"mpc.bus has {references} reference buses (type {BusType.REF})"
            raise DataConditionError(f"{source}: {what}, and a case has exactly one")

    @staticmethod
    def _check_statements(text: str, source: str | Path) -> None:
        """Refuse what the reader would silently skip (an indexed assignment, an unknown field, a field
        assigned twice) and a file that lacks a field a `Case` needs."""
        seen: list[str] = []
        known = (*newyork.CASE_FIELDS_NEEDED, newyork.DCLINE, *newyork.CASE_FIELDS_PASSED_OVER)
        for number, line in enumerate(text.splitlines(), 1):
            statement = STATEMENT.match(line)
            if statement is None:
                continue
            name, indexed = statement.groups()
            where = f"{source}: line {number}: `{line.strip()}`"
            if indexed:
                raise DataConditionError(f"{where} is an indexed assignment, which is not read: edit the row")
            if name not in known:
                listed = ", ".join(known)
                raise DataConditionError(f"{where}: mpc.{name} is not a field of a case file ({listed})")
            if name in seen:
                raise DataConditionError(f"{where}: mpc.{name} is assigned twice")
            seen.append(name)
        missing = [name for name in newyork.CASE_FIELDS_NEEDED if name not in seen]
        if missing:
            raise DataConditionError(f"{source}: has no mpc.{missing[0]}")

    @staticmethod
    def _column(frame: pd.DataFrame | None, name: str | int) -> np.ndarray:
        """One column of a field as floats; no rows for a field the file does not have."""
        return np.zeros(0) if frame is None else frame[name].to_numpy(dtype=float, copy=True)

    @staticmethod
    def _strings(text: str, field: str) -> np.ndarray:
        """The quoted strings of the `mpc.<field> = { ... };` list of a case file, `%` comments left out."""
        block = re.search(rf"^\s*mpc\.{field}\s*=\s*\{{(.*?)\}}", text, re.S | re.M)
        lines = block.group(1).splitlines() if block else []
        found = [s for line in lines for s in QUOTED.findall(line.split("%")[0])]
        return np.array([s.replace("''", "'") for s in found], dtype=str)

    @staticmethod
    def _bus_positions(numbers: np.ndarray, source: str | Path) -> dict[int, int]:
        """Each bus number's row in `mpc.bus`; the numbers must be whole and each on one row."""
        if not (numbers == np.round(numbers)).all():
            raise DataConditionError(f"{source}: mpc.bus: a bus number (BUS_I) must be a whole number")
        position = {int(number): row for row, number in enumerate(numbers)}
        if len(position) != len(numbers):
            twice = [n for n in position if (numbers == n).sum() > 1]
            raise DataConditionError(f"{source}: mpc.bus: bus number {twice[0]} is on more than one row")
        return position

    @staticmethod
    def _positions(
        numbers: np.ndarray, position: dict[int, int], field: str, source: str | Path
    ) -> np.ndarray:
        """Bus numbers of `field` as positions into the bus arrays; each must be a bus of `mpc.bus`."""
        strangers = [n for n in numbers.tolist() if n not in position]
        if strangers:
            raise DataConditionError(f"{source}: {field}: bus {strangers[0]:g} is not a bus of mpc.bus")
        return np.array([position[n] for n in numbers.tolist()], dtype=int)

    @staticmethod
    def _zones_and_areas(
        codes: np.ndarray, file_area: np.ndarray, source: str | Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """`(zone, area)` of every bus from the file's `zone` column (`ZONE_LEGEND`): the zone letter ("" outside
        New York) and the area (`NY_AREA`, or the neighbor's in `REGIONS`). The file's own `area` column must be
        `FILE_AREA` on every row."""
        if not (file_area == newyork.FILE_AREA).all():
            raise DataConditionError(
                f"{source}: mpc.bus: the area (BUS_AREA) must be {newyork.FILE_AREA} on every row: the case is one "
                "area, and where a bus is, is read from its zone (ZONE)"
            )
        letter = dict(enumerate(newyork.ZONES, 1)) | {code: "" for code, _, _ in newyork.REGIONS}
        area = {code: newyork.NY_AREA for code in letter} | {code: area for code, _, area in newyork.REGIONS}
        if not np.isin(codes, list(letter)).all():
            raise DataConditionError(
                f"{source}: mpc.bus: a zone (ZONE) must be a whole number, {newyork.ZONE_LEGEND}"
            )
        whole = codes.astype(int).tolist()
        return np.array([letter[c] for c in whole]), np.array([area[c] for c in whole], dtype=int)


def nygrid_case() -> Case:
    """The New York grid the release is built on, read from the case file shipped with the package: NYgrid's
    NPCC-140 case with NYgrid's own modifications, its fixed unit list, zone J's units split over buses 81 and 82
    and the 765 kV line 47-43 (140 buses, 227 branches, 271 units, no HVDC link)."""
    return Case.from_matpower(CASE_FILE)
