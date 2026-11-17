"""The grid as one frozen record of arrays, read from a MATPOWER case file.

The New York grid is the file `case_nygrid.m` beside this module: `nygrid_case()` reads it, and a change to the grid
is an edit to that file. `Case.from_matpower` reads a case file with matpowercaseframes, the reader pandapower uses,
taking each column by MATPOWER's own name, and `Case.check` refuses a file whose fields disagree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fdia_graph.constants import newyork
from fdia_graph.constants.grid import BusType
from fdia_graph.errors import DataConditionError

if TYPE_CHECKING:
    import pandas as pd
    from matpowercaseframes import CaseFrames


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

    # ---- reading a MATPOWER case file ----

    @classmethod
    def from_matpower(cls, path: str | Path) -> Case:
        """The `Case` of the MATPOWER case file at `path`, checked (`check`).

        The matrices keep MATPOWER's columns. The bus `zone` column says where each bus is, as a number
        (`ZONE_LEGEND`): `zone` and `area` of the `Case` both come from it (`NY_AREA` in New York, the neighbor's
        area in `REGIONS` outside), and the bus `area` column must be `FILE_AREA` on every row. A branch out of
        service is left out, and a tap ratio of 0 (MATPOWER's mark of a line) is 1. Beyond MATPOWER's own fields the file
        has `mpc.bus_latlon` (degrees) and `mpc.gen_ramp` (MW/min, NaN where unknown). A statement the reader
        would pass over without a word (an indexed assignment, a field it does not know) raises
        `DataConditionError`."""
        from matpowercaseframes import CaseFrames  # here, so that a Case can be used without the [ny] extra

        text = Path(path).read_text(encoding="utf-8")
        cls._check_statements(text, path)
        try:
            case = cls._from_frames(
                CaseFrames(str(path), allow_any_keys=True, update_index=False), text, path
            )
        except DataConditionError:
            raise
        except Exception as error:  # a row with a value missing or too many, a word among the numbers
            raise DataConditionError(f"{path}: cannot be read as a MATPOWER case file: {error!r}") from error
        case.check(str(path))
        return case

    @classmethod
    def _from_frames(cls, frames: CaseFrames, text: str, path: str | Path) -> Case:
        """The `Case` of the tables matpowercaseframes read, each column taken by MATPOWER's name for it. The
        two lists of strings come from `text`, the file itself."""
        bus, gen = frames.bus, frames.gen
        branch = frames.branch[frames.branch["BR_STATUS"] > 0]
        links = frames.dcline if newyork.DCLINE in frames.attributes else None
        col = cls._column
        position = cls._bus_positions(col(bus, "BUS_I"), path)
        zone, area = cls._zones_and_areas(col(bus, "ZONE"), col(bus, "BUS_AREA"), path)
        ratio = col(branch, "TAP")
        return cls(
            base_mva=float(frames.baseMVA),
            bus_id=col(bus, "BUS_I").astype(int),
            name=cls._strings(text, newyork.BUS_NAME),
            zone=zone,
            area=area,
            kv=col(bus, "BASE_KV"),
            vmin=col(bus, "VMIN"),
            vmax=col(bus, "VMAX"),
            lat=col(frames.bus_latlon, 0),
            lon=col(frames.bus_latlon, 1),
            is_ny=zone != "",
            f=cls._positions(col(branch, "F_BUS"), position, "mpc.branch", path),
            t=cls._positions(col(branch, "T_BUS"), position, "mpc.branch", path),
            r=col(branch, "BR_R"),
            x=col(branch, "BR_X"),
            b=col(branch, "BR_B"),
            tap=np.where(ratio == 0, 1.0, ratio),
            shift=col(branch, "SHIFT"),
            rating=col(branch, "RATE_A"),
            unit_bus=cls._positions(col(gen, "GEN_BUS"), position, "mpc.gen", path),
            fuel=cls._strings(text, newyork.GEN_FUEL),
            pmin=col(gen, "PMIN"),
            pmax=col(gen, "PMAX"),
            qmin=col(gen, "QMIN"),
            qmax=col(gen, "QMAX"),
            ramp=col(frames.gen_ramp, 0),
            dc_f=cls._positions(col(links, "F_BUS"), position, "mpc.dcline", path),
            dc_t=cls._positions(col(links, "T_BUS"), position, "mpc.dcline", path),
            dc_pmin=col(links, "PMIN"),
            dc_pmax=col(links, "PMAX"),
            bus_type=col(bus, "BUS_TYPE").astype(int),
            pd=col(bus, "PD"),
            qd=col(bus, "QD"),
            gs=col(bus, "GS"),
            bs=col(bus, "BS"),
            vm=col(bus, "VM"),
            va=col(bus, "VA"),
            unit_p=col(gen, "PG"),
            unit_q=col(gen, "QG"),
            unit_vg=col(gen, "VG"),
            unit_on=col(gen, "GEN_STATUS") > 0,
        )

    def check(self, source: str) -> None:
        """Refuse a case whose parts disagree, naming `source` (the file it came from) and what is wrong: one name
        and one place per bus, one fuel and one ramp per unit, bus types that are MATPOWER's, one reference bus."""
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
        """Refuse what the reader would pass over without a word: an indexed assignment, a field it does not know
        and a field assigned twice. Refuse a file that lacks a field a `Case` needs."""
        seen: list[str] = []
        for number, line in enumerate(text.splitlines(), 1):
            statement = STATEMENT.match(line)
            if statement is None:
                continue
            name, indexed = statement.groups()
            where = f"{source}: line {number}: `{line.strip()}`"
            if indexed:
                raise DataConditionError(f"{where} is an indexed assignment, which is not read: edit the row")
            if name not in (*newyork.CASE_FIELDS_NEEDED, newyork.DCLINE, *newyork.CASE_FIELDS_PASSED_OVER):
                known = ", ".join(
                    (*newyork.CASE_FIELDS_NEEDED, newyork.DCLINE, *newyork.CASE_FIELDS_PASSED_OVER)
                )
                raise DataConditionError(f"{where}: mpc.{name} is not a field of a case file ({known})")
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
        """`(zone, area)` of every bus from the bus `zone` column (`ZONE_LEGEND`): the zone letter, "" outside New
        York, and the area (`NY_AREA`, or the neighbor's area in `REGIONS`). The file's own `area` column is not where
        a bus's region is read from, and must be `FILE_AREA` on every row."""
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
