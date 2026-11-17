"""Where the bottom-up load profiles come from ([LI21]), each downloaded once into a cache: the
NREL ResStock and ComStock county profiles, each utility's EIA-861 sector sales, the census county centroids and
the IAC industrial plants, with the clock rules the plants run on. `LoadModel` (`loads.py`) builds the bus
profiles from these."""

from __future__ import annotations

import csv
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from fdia_graph.constants.newyork import (
    EIA_861_URL,
    EIA_UTILITY,
    GAZETTEER_URL,
    IAC_URL,
    OEDI_URL,
    STOCK_KEY_PREFIX,
)
from fdia_graph.constants.time import HOURS_PER_YEAR, QUARTERS_PER_YEAR
from fdia_graph.data.inputs import download_once


def _list_keys(prefix: str) -> list[str]:
    keys, token = [], ""
    while True:
        url = f"{OEDI_URL}/?list-type=2&prefix={prefix}" + (f"&continuation-token={token}" if token else "")
        with urlopen(url, timeout=120) as r:
            body = r.read().decode()
        keys += [k.split("</Key>")[0] for k in body.split("<Key>")[1:]]
        if "<IsTruncated>true</IsTruncated>" not in body:
            return keys
        token = body.split("<NextContinuationToken>")[1].split("<")[0].replace("+", "%2B").replace("/", "%2F")


def _quarter_per_unit(key: str) -> tuple[str, str, str, float, np.ndarray]:
    """One county file as a 15-minute per-unit electricity profile, kW, for 2018 in EST. ResStock is per
    dwelling, ComStock per ft2. The energy rows are period-ending (the first is stamped 00:15), so row r
    covers minutes 15 r to 15 (r + 1) of the year and its mean power is four times its kWh."""
    sector = "res" if "resstock" in key else "com"
    size_col = "units_represented" if sector == "res" else "floor_area_represented"
    df = pd.read_csv(f"{OEDI_URL}/{key}", usecols=[size_col, "out.electricity.total.energy_consumption"])
    kwh = df["out.electricity.total.energy_consumption"].to_numpy()
    size = float(df[size_col].iloc[0])
    county, kind = key.rsplit("/", 1)[1][:-4].split("-", 1)
    return sector, county.upper(), kind, size, 4 * kwh[:QUARTERS_PER_YEAR] / size


def _building_stock(cache: Path) -> dict[str, np.ndarray]:
    """All NY county ResStock and ComStock profiles (both AMY2018 release 1) at their native 15 minutes,
    reduced to total electricity per unit and cached, since the full files are about 11 GB."""
    path = cache / "building_stock_ny_15min.npz"
    if not path.exists():
        keys = [
            k
            for s in ("resstock", "comstock")
            for k in _list_keys(STOCK_KEY_PREFIX.format(s))
            if k.endswith(".csv")
        ]
        with ThreadPoolExecutor(8) as pool:
            rows = list(pool.map(_quarter_per_unit, keys))
        np.savez_compressed(
            path,
            sector=np.array([r[0] for r in rows]),
            county=np.array([r[1] for r in rows]),
            kind=np.array([r[2] for r in rows]),
            size=np.array([r[3] for r in rows]),
            profile=np.stack([r[4] for r in rows]).astype(np.float32),
        )
    return dict(np.load(path))


def _sector_mix(cache: Path) -> dict[str, np.ndarray]:
    """[LI21, II-B]: each utility's residential, commercial and industrial share of its 2019 MWh sales to ultimate
    customers, bundled (Part A) plus delivery (Part C). Transportation sales count as commercial."""
    with zipfile.ZipFile(download_once(EIA_861_URL, cache / "f8612019.zip")) as z:
        s = pd.read_excel(io.BytesIO(z.read("Sales_Ult_Cust_2019.xlsx")), sheet_name="States", header=2)
    s = s[(s.State == "NY") & s.Part.isin(["A", "C"])]
    mwh = s[["Megawatthours", "Megawatthours.1", "Megawatthours.2", "Megawatthours.3"]]
    mwh = mwh.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()  # "." means not reported
    mix = {}
    for name, number in EIA_UTILITY.items():
        r, c, i, t = mwh[(s["Utility Number"] == number).to_numpy()].sum(axis=0)
        mix[name] = np.array([r, c + t, i]) / (r + c + t + i)
    return mix


def _county_centroids(cache: Path) -> dict[str, tuple[float, float]]:
    """NY county internal points (lat, lon) from the 2019 Census gazetteer, keyed by NHGIS code as the OEDI files are."""
    with zipfile.ZipFile(download_once(GAZETTEER_URL, cache / "2019_Gaz_counties_national.zip")) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = csv.DictReader(io.StringIO(text), delimiter="\t")
    out = {}
    for r in rows:
        if r["USPS"] == "NY":
            g = r["GEOID"]
            out[f"G{g[:2]}0{g[2:]}0"] = (
                float(r["INTPTLAT"]),
                float(r[next(k for k in r if k.startswith("INTPTLONG"))]),
            )
    return out


def _plants(cache: Path) -> dict[str, np.ndarray]:
    """[LI21, II-C2]: every IAC plant with usable electricity data, as the parameters of its yearly curve. The
    curve follows ORNL/TM-2013/407 (Sec. 2, App. B): 1 p.u. on weekdays 9 AM to 4 PM, else the flat level that
    gives the SIC's load factor, the mean over its plants of yearly kWh / (yearly kW-months / 12 x 8760).
    Ours: [LI21]'s ORNL curves are figures only, so they are rebuilt from the report's rule; weekends sit at
    the weekday minimum; a plant with a load factor above 1 or more than 8760 hours is a data error."""
    path = cache / "iac_plants.npz"
    if not path.exists():
        with zipfile.ZipFile(download_once(IAC_URL, cache / "ITAC_Database.zip")) as z:
            a = pd.read_excel(io.BytesIO(z.read(z.namelist()[0])), sheet_name="ASSESS")
        a = a[
            (a.EC_plant_usage > 0)
            & (a.ED_plant_usage > 0)
            & (a.PRODHOURS > 0)
            & (a.PRODHOURS <= HOURS_PER_YEAR)
            & a.SIC.notna()
        ].copy()
        a["lf"] = a.EC_plant_usage / (a.ED_plant_usage / 12 * HOURS_PER_YEAR)
        a = a[a.lf <= 1]
        sic_lf = a.groupby("SIC").lf.transform("mean").to_numpy()
        np.savez_compressed(
            path,
            kwh=a.EC_plant_usage.to_numpy(float),
            hours=a.PRODHOURS.to_numpy(float),
            sic=a.SIC.to_numpy(float),
            sic_lf=sic_lf,
        )
    return dict(np.load(path))


def _work_hours() -> np.ndarray:
    """[24] 1 in the ORNL working block, 9 AM to 4 PM."""
    w = np.zeros(24)
    w[9:16] = 1.0
    return w


def _operating_rank(year: int) -> np.ndarray:
    """[365, 365] rank of each day as a plant's operating day when its year starts on day s: weekdays in
    order from s first, then weekend days, so a plant runs its first `n` ranked days."""
    weekday = np.asarray(pd.date_range(f"{year}-01-01", periods=365, freq="D").dayofweek < 5)
    rank = np.empty((365, 365), dtype=np.int16)
    for s in range(365):
        order = np.roll(np.arange(365), -s)
        order = np.concatenate([order[weekday[order]], order[~weekday[order]]])
        rank[s, order] = np.arange(365)
    return rank
