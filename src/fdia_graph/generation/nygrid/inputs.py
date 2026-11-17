"""Historical inputs: NYgrid's hourly 2019 tables and NYISO's 5-minute zonal load, and `fetch_inputs`, which
downloads everything the build reads at its pinned version."""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from fdia_graph.constants import newyork
from fdia_graph.generation.nygrid.config import Settings
from fdia_graph.profiles import nyiso_zone_load
from fdia_graph.systems.tables import read_rows


def download_once(url: str, path: Path) -> Path:
    """`url` saved to `path` once; a file already there is kept. The download goes to a `.part` file first,
    so an interrupted one is never taken for the whole file."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        with (
            urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=300) as r,
            open(part, "wb") as f,
        ):
            shutil.copyfileobj(r, f)
        part.replace(path)
    return path


@dataclass(frozen=True)
class ZoneLoad:
    """NYISO real-time zonal load, MW, `load [T, 11]` in zone order A..K."""

    time: np.ndarray
    load: np.ndarray


@dataclass(frozen=True)
class NygridHour:
    """One hour of NYgrid's inputs, as readOpCond.m and allocateLoad.m produce them.

    `ny_pd`, `ny_qd`, `renew_mw` are over the 140 NPCC buses in bus-number order: the zone loads split onto NY
    buses (0.98 power factor) and the wind plus other renewables netted off them. `thermal_mw` is in the case
    file's thermal-unit order, `nuclear_mw` its six nuclear units. `thermal_cost` ($/MWh) is the week's
    fuel price times the unit's heat rate (allocateGen.m 'lm'); `ext_lbmp` is the hour's real-time LBMP of
    the proxy buses NPX, O H, PJM and H Q, which price the external units (updateOpCond.m L493-501)."""

    time: np.datetime64
    fuel_mw: np.ndarray  # [7], FUELS order
    iface_flow: np.ndarray  # [17], INTERFACE_NAMES order
    iface_pos: np.ndarray
    iface_neg: np.ndarray
    nuclear_mw: np.ndarray
    stl_cf: float  # St. Lawrence monthly capacity factor
    thermal_mw: np.ndarray
    ny_pd: np.ndarray
    ny_qd: np.ndarray
    renew_mw: np.ndarray
    thermal_cost: np.ndarray  # [227] $/MWh
    ext_lbmp: np.ndarray  # [4] $/MWh, EXT_PRICES order


def _bus_weights(data_dir: Path) -> np.ndarray:
    """[140, 11] share of each zone's load at each bus (allocateLoad.m 'weighted'): in proportion to the
    base load of the zone's load buses, or evenly over all its buses where none has base load (zone J)."""
    rows = read_rows(data_dir / "npcc_new.csv", "utf-8-sig")
    w = np.zeros((len(rows), len(newyork.ZONES)))
    for k, z in enumerate(newyork.ZONES):
        idx = [int(r["idx"]) - 1 for r in rows if r["zone"] == z]
        base = np.array([float(rows[i]["sumLoadP0"]) for i in idx])
        w[idx, k] = base / base.sum() if (base > 0).any() else 1.0 / len(idx)
    return w


def _renewable_caps(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    wind, other = np.zeros(140), np.zeros(140)
    for r in read_rows(data_dir / "RenewableGen.csv", "utf-8-sig"):
        b = int(r["bus_id"]) - 1
        wind[b] = float(r["Pg Wind Cap"] or 0)
        other[b] = float(r["Pg Other Cap"] or 0)
    return wind, other


def _hourly(
    path: Path, column: str, value: str, names: tuple[str, ...], hours: pd.Index | None = None
) -> pd.DataFrame:
    """One of NYgrid's hourly tables: `value` by hour (`hours`, default the file's own) and by the `names`
    of `column`."""
    table = pd.read_csv(path)
    table["t"] = pd.to_datetime(table.TimeStamp, format="%m/%d/%Y %H:%M:%S")
    pivot = table.pivot_table(index="t", columns=column, values=value)
    if hours is None:
        return pivot.reindex(columns=list(names))
    return pivot.reindex(index=hours, columns=list(names))


def read_nygrid(data_dir: str | Path, year: int) -> list[NygridHour]:
    """Every hour of `year` from NYgrid's `Data/` tables."""
    d = Path(data_dir)
    stamp = "%m/%d/%Y %H:%M:%S"

    fuel = _hourly(d / f"fuelmixHourly_{year}.csv", "FuelCategory", "GenMW", newyork.FUELS)
    hours = fuel.index

    flow = pd.read_csv(d / f"interflowHourly_{year}.csv")
    flow["t"] = pd.to_datetime(flow.TimeStamp, format="%m/%d/%Y %H:%M")
    piv = {
        c: flow.pivot_table(index="t", columns="InterfaceName", values=c).reindex(
            index=hours, columns=list(newyork.INTERFACE_NAMES)
        )
        for c in ("FlowMWH", "PositiveLimitMWH", "NegativeLimitMWH")
    }

    nuc = pd.read_csv(d / f"nuclearGenDaily_{year}.csv")
    nuc.index = pd.to_datetime(nuc.TimeStamp, format="%m/%d/%Y")
    nuc = nuc.reindex(hours.normalize())[list(newyork.NUCLEAR_COLS)].to_numpy()

    hyd = pd.read_csv(d / f"hydroGenMonthly_{year}.csv")
    hyd.index = pd.to_datetime(hyd.TimeStamp, format="%m/%d/%Y")
    stl = hyd.stlCF.reindex(hours.to_period("M").to_timestamp()).to_numpy()

    params = sorted(
        read_rows(d / "genParamAll_newramp.csv", "utf-8-sig"), key=lambda r: (r["NYISOName"], int(r["PTID"]))
    )
    units = [(r["NYISOName"], int(r["PTID"])) for r in params]

    # allocateGen.m createGenCost 'lm': the latest weekly fuel price on or before the hour times HeatRateLM_1;
    # gas by zone (J, K, the rest at NG_A2E), oil and coal at the upstate price.
    fp = pd.read_csv(d / f"fuelPriceWeekly_{year}.csv")
    fp.index = pd.to_datetime(fp.TimeStamp, format="%m/%d/%Y")
    fp = fp.drop(columns="TimeStamp").reindex(hours, method="ffill")
    col = [
        {"J": "NG_J", "K": "NG_K"}.get(r["Zone"], "NG_A2E")
        if r["FuelType"] == "Natural Gas"
        else newyork.FUEL_PRICE[r["FuelType"]]
        for r in params
    ]
    thermal_cost = fp[col].to_numpy() * np.array([float(r["HeatRateLM_1"]) for r in params])
    lbmp = _hourly(d / f"priceHourly_{year}.csv", "ZoneName", "LBMP", newyork.EXT_PRICES, hours).to_numpy()
    th = pd.read_csv(d / f"thermalGenHourly_{year}.csv")
    th["t"] = pd.to_datetime(th.TimeStamp, format=stamp)
    th = th.groupby(["t", "NYISOName", "PTID"]).hourlyGen.first().unstack(["NYISOName", "PTID"])
    th = th.reindex(index=hours, columns=pd.MultiIndex.from_tuples(units)).fillna(0.0).to_numpy()

    load = _hourly(d / f"loadHourly_{year}.csv", "ZoneID", "Load", newyork.ZONES, hours)
    ny_pd = load.to_numpy() @ _bus_weights(d).T
    ny_qd = ny_pd * np.tan(np.arccos(0.98))

    # Wind and other renewables enter as negative load at a fleet-wide capacity factor (updateOpCond.m L286-302).
    wind_cap, other_cap = _renewable_caps(d)
    f = fuel.to_numpy()
    renew = np.outer(f[:, newyork.FUELS.index("Wind")] / wind_cap.sum(), wind_cap) + np.outer(
        f[:, newyork.FUELS.index("Other Renewables")] / other_cap.sum(), other_cap
    )

    return [
        NygridHour(
            time=np.datetime64(hours[h]),
            fuel_mw=f[h],
            iface_flow=piv["FlowMWH"].iloc[h].to_numpy(),
            iface_pos=piv["PositiveLimitMWH"].iloc[h].to_numpy(),
            iface_neg=piv["NegativeLimitMWH"].iloc[h].to_numpy(),
            nuclear_mw=nuc[h],
            stl_cf=float(stl[h]),
            thermal_mw=th[h],
            ny_pd=ny_pd[h],
            ny_qd=ny_qd[h],
            renew_mw=renew[h],
            thermal_cost=thermal_cost[h],
            ext_lbmp=lbmp[h],
        )
        for h in range(len(hours))
    ]


def fetch_nyiso(year: int, cache_dir: str | Path = "data/nyiso") -> ZoneLoad:
    """NYISO real-time actual load ('pal' feed, about 5-minute samples) for every zone and month of
    `year`, on a clean 5-minute grid. Monthly archives are cached in `cache_dir`."""
    grid = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="5min", inclusive="left")
    months = [date(year, m, 1) for m in range(1, 13)]
    piv = nyiso_zone_load(months, grid, cache_dir).rename(columns=newyork.ZONE_NAME)[list(newyork.ZONES)]
    return ZoneLoad(time=piv.index.to_numpy(), load=piv.to_numpy())


def _nygrid(data: Path) -> None:
    """NYgrid's `Data/` folder at the pinned commit, under `data/NYgrid/Data`."""
    dest = data / "NYgrid" / "Data"
    if dest.exists():
        return
    archive = download_once(newyork.NYGRID_ZIP_URL, data / f"NYgrid_{newyork.NYGRID_COMMIT[:7]}.zip")
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) > 2 and parts[1] == "Data" and not name.endswith("/"):
                out = dest.joinpath(*parts[2:])
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(out, "wb") as f:
                    shutil.copyfileobj(src, f)


def fetch_inputs(root: str | Path) -> Path:
    """Download the build's inputs under `root/data` and return that folder: NYgrid's `Data/` at its pinned
    commit, NYISO's five-minute zonal load ('pal'), the NYS DPS utility territories and the bus locations.
    A file already present is kept. `LoadModel` fetches the load-profile sources on its first build."""
    data = Path(root) / "data"
    _nygrid(data)
    terr = data / "territories"
    download_once(newyork.TERRITORIES_URL, terr / newyork.TERRITORIES_FILE)
    if not (terr / newyork.BUS_LOCATIONS_FILE).exists():
        shutil.copyfile(Path(__file__).parent / newyork.BUS_LOCATIONS_FILE, terr / newyork.BUS_LOCATIONS_FILE)
    fetch_nyiso(Settings().year, data / "nyiso")
    return data
