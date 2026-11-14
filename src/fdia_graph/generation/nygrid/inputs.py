"""Historical inputs: NYgrid's hourly 2019 tables and NYISO's 5-minute zonal load."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from fdia_graph.generation.nygrid.tables import read_rows

ZONES = tuple("ABCDEFGHIJK")
FUELS = ("Dual Fuel", "Hydro", "Natural Gas", "Nuclear", "Other Fossil Fuels", "Other Renewables", "Wind")
# Interfaces NYgrid reads (updateOpCond.m L248-262 and L445-451), schedules into NY are positive.
INTERFACES = (
    "SCH - PJ - NY",
    "SCH - PJM_HTP",
    "SCH - PJM_NEPTUNE",
    "SCH - PJM_VFT",
    "SCH - HQ - NY",
    "SCH - HQ_CEDARS",
    "SCH - NE - NY",
    "SCH - NPX_1385",
    "SCH - NPX_CSC",
    "SCH - OH - NY",
    "DYSINGER EAST",
    "WEST CENTRAL",
    "TOTAL EAST",
    "MOSES SOUTH",
    "CENTRAL EAST - VC",
    "UPNY CONED",
    "SPR/DUN-SOUTH",
)
EXT_PRICES = ("NPX", "O H", "PJM", "H Q")  # proxy buses pricing the NE, IESO, PJM and HQ units
_FUEL_PRICE = {"Coal": "coal_NY", "Fuel Oil 2": "FO2_UPNY", "Kerosene": "FO2_UPNY", "Fuel Oil 6": "FO6_UPNY"}
_NUCLEAR_COLS = (  # unit order of nygrid_units: Ginna (53), FitzPatrick, Nine Mile 1, 2 (50), Indian Point 2, 3 (74)
    "GinnaGen",
    "FitzPatrickGen",
    "NineMilePoint1Gen",
    "NineMilePoint2Gen",
    "IndianPoint2Gen",
    "IndianPoint3Gen",
)


@dataclass(frozen=True)
class ZoneLoad:
    """NYISO real-time zonal load, MW, `load [T, 11]` in zone order A..K."""

    time: np.ndarray
    load: np.ndarray


@dataclass(frozen=True)
class NygridHour:
    """One hour of NYgrid's inputs, as readOpCond.m and allocateLoad.m produce them.

    `ny_pd`, `ny_qd` and `renew_mw` are over the 140 NPCC buses in bus-number order: NYgrid's split of
    the zone loads onto NY buses (0.98 power factor) and the wind plus other renewables it nets off them.
    `thermal_mw` follows the thermal units of `nygrid_units`, `nuclear_mw` its six nuclear units.
    `thermal_cost` is each thermal unit's linear generation cost, $/MWh: the week's fuel price times its
    heat rate (allocateGen.m 'lm', the cost updateOpCond.m L474-479 puts in gencost); `ext_lbmp` the hour's
    real-time LBMP of NYISO's proxy buses NPX, O H, PJM and H Q, which price the external units (L493-501)."""

    time: np.datetime64
    fuel_mw: np.ndarray  # [7], FUELS order
    iface_flow: np.ndarray  # [17], INTERFACES order
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
    w = np.zeros((len(rows), len(ZONES)))
    for k, z in enumerate(ZONES):
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


def read_nygrid(data_dir: str | Path, year: int) -> list[NygridHour]:
    """Every hour of `year` from NYgrid's `Data/` tables."""
    d = Path(data_dir)
    stamp = "%m/%d/%Y %H:%M:%S"

    fuel = pd.read_csv(d / f"fuelmixHourly_{year}.csv")
    fuel["t"] = pd.to_datetime(fuel.TimeStamp, format=stamp)
    fuel = fuel.pivot_table(index="t", columns="FuelCategory", values="GenMW").reindex(columns=list(FUELS))
    hours = fuel.index

    flow = pd.read_csv(d / f"interflowHourly_{year}.csv")
    flow["t"] = pd.to_datetime(flow.TimeStamp, format="%m/%d/%Y %H:%M")
    piv = {
        c: flow.pivot_table(index="t", columns="InterfaceName", values=c).reindex(
            index=hours, columns=list(INTERFACES)
        )
        for c in ("FlowMWH", "PositiveLimitMWH", "NegativeLimitMWH")
    }

    nuc = pd.read_csv(d / f"nuclearGenDaily_{year}.csv")
    nuc.index = pd.to_datetime(nuc.TimeStamp, format="%m/%d/%Y")
    nuc = nuc.reindex(hours.normalize())[list(_NUCLEAR_COLS)].to_numpy()

    hyd = pd.read_csv(d / f"hydroGenMonthly_{year}.csv")
    hyd.index = pd.to_datetime(hyd.TimeStamp, format="%m/%d/%Y")
    stl = hyd.stlCF.reindex(hours.to_period("M").to_timestamp()).to_numpy()

    params = sorted(
        read_rows(d / "genParamAll_newramp.csv", "utf-8-sig"),
        key=lambda r: (r["NYISOName"], int(r["PTID"])),
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
        else _FUEL_PRICE[r["FuelType"]]
        for r in params
    ]
    thermal_cost = fp[col].to_numpy() * np.array([float(r["HeatRateLM_1"]) for r in params])
    price = pd.read_csv(d / f"priceHourly_{year}.csv")
    price["t"] = pd.to_datetime(price.TimeStamp, format=stamp)
    lbmp = price.pivot_table(index="t", columns="ZoneName", values="LBMP").reindex(
        index=hours, columns=list(EXT_PRICES)
    )
    lbmp = lbmp.to_numpy()
    th = pd.read_csv(d / f"thermalGenHourly_{year}.csv")
    th["t"] = pd.to_datetime(th.TimeStamp, format=stamp)
    th = th.groupby(["t", "NYISOName", "PTID"]).hourlyGen.first().unstack(["NYISOName", "PTID"])
    th = th.reindex(index=hours, columns=pd.MultiIndex.from_tuples(units)).fillna(0.0).to_numpy()

    load = pd.read_csv(d / f"loadHourly_{year}.csv")
    load["t"] = pd.to_datetime(load.TimeStamp, format=stamp)
    load = load.pivot_table(index="t", columns="ZoneID", values="Load").reindex(
        index=hours, columns=list(ZONES)
    )
    ny_pd = load.to_numpy() @ _bus_weights(d).T
    ny_qd = ny_pd * np.tan(np.arccos(0.98))

    # Wind and other renewables enter as negative load at a fleet-wide capacity factor (updateOpCond.m L286-302).
    wind_cap, other_cap = _renewable_caps(d)
    f = fuel.to_numpy()
    renew = np.outer(f[:, FUELS.index("Wind")] / wind_cap.sum(), wind_cap) + np.outer(
        f[:, FUELS.index("Other Renewables")] / other_cap.sum(), other_cap
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
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for m in range(1, 13):
        path = cache / f"{year}{m:02d}01pal_csv.zip"
        if not path.exists():
            url = f"https://mis.nyiso.com/public/csv/pal/{path.name}"
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r:
                path.write_bytes(r.read())
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                df = pd.read_csv(io.BytesIO(z.read(name)))
                frames.append(df[["Time Stamp", "Name", "Load"]])
    full = pd.concat(frames)
    full["t"] = pd.to_datetime(full["Time Stamp"])
    full["zone"] = full.Name.map(_ZONE_NAME)
    piv = full.dropna(subset=["zone"]).pivot_table(index="t", columns="zone", values="Load", aggfunc="mean")
    # Zones report a few seconds apart; put them on one 5-minute grid before using them together.
    grid = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="5min", inclusive="left")
    piv = piv.reindex(piv.index.union(grid)).interpolate(limit=3).reindex(grid)[list(ZONES)]
    return ZoneLoad(time=piv.index.to_numpy(), load=piv.to_numpy())


# NYISO zone names in the pal feed, as NYgrid's loadHourly table maps them to zones.
_ZONE_NAME = {
    "WEST": "A",
    "GENESE": "B",
    "CENTRL": "C",
    "NORTH": "D",
    "MHK VL": "E",
    "CAPITL": "F",
    "HUD VL": "G",
    "MILLWD": "H",
    "DUNWOD": "I",
    "N.Y.C.": "J",
    "LONGIL": "K",
}
