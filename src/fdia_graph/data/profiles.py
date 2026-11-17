"""Load profiles: a real load time series as a load-scaling vector, read from a file or fetched from an
operator's feed.

    load_profile(source) or fetch_profile(iso, start, end)  ->  a load-scaling vector S [T]

`data/ieee_pool.py` solves the pool of operating states an IEEE case takes from it; the New York loads read
NYISO's zone feed here (`nyiso_zone_load`).
"""

from __future__ import annotations

import datetime as _dt
import glob
import io
import os
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Union, runtime_checkable
from urllib.request import Request, urlopen

import numpy as np

from fdia_graph.constants.files import EXPORT_COLUMNS
from fdia_graph.models.choices import Iso
from fdia_graph.models.config import IsoExport, ProfileFetch
from fdia_graph.models.inputs import CsvSpec, DateSpan, LoadValues, ProfileSource


def _pandas():
    """pandas, imported on first use: an optional dependency only load profiles need."""
    try:
        import pandas as pd

        return pd
    except ImportError as e:
        raise ImportError("pandas is required for load profiles: pip install 'fdia-graph[generate]'") from e


if TYPE_CHECKING:
    import pandas as pd


def _standardize(loads: np.ndarray) -> np.ndarray:
    """The load-scaling vector S [T]: zero mean, unit variance, so the K knob means the same for
    sources of any magnitude. A flat series stays at zero."""
    sd = loads.std()
    return (loads - loads.mean()) / (sd if sd > 0 else 1.0)


@runtime_checkable
class LoadSource(Protocol):
    """Anything that reads a load series: `IsoFolder`, `CsvColumn`, `RawSeries`, or a source of your
    own with the same method."""

    def loads(self) -> np.ndarray: ...


class RawSeries:
    """Load values you already have, in any unit."""

    def __init__(self, values: Union[Sequence[float], np.ndarray]) -> None:
        self.values: np.ndarray = LoadValues(values).array

    def loads(self) -> np.ndarray:
        return self.values


class CsvColumn:
    """One column of one CSV file."""

    def __init__(self, path: str, column: str) -> None:
        spec = CsvSpec(path, column)
        self.path, self.column = spec.path, spec.column

    def loads(self) -> np.ndarray:
        return _pandas().read_csv(self.path)[self.column].dropna().to_numpy(dtype=float)


class IsoFolder:
    """Every CSV of one operator's load export in a directory, concatenated in time order. The operator
    fixes the column names (`constants.files.EXPORT_COLUMNS`); an operator with no known export
    format is refused here, not when the folder is read."""

    def __init__(self, iso: str, directory: str = ".") -> None:
        self.spec = IsoExport(iso, directory)

    @property
    def columns(self) -> tuple[str, str]:
        """(timestamp column, load column) of this operator's export."""
        return EXPORT_COLUMNS[self.spec.iso]

    def loads(self) -> np.ndarray:
        pd = _pandas()
        files = sorted(glob.glob(os.path.join(self.spec.directory, "*.csv")))
        if not files:
            raise FileNotFoundError(f"no CSV files found under {self.spec.directory!r} for {self.spec.iso}")
        ts, load = self.columns
        frames = [pd.read_csv(f, parse_dates=[ts], index_col=ts) for f in files]
        return pd.concat(frames).sort_index()[load].dropna().to_numpy(dtype=float)


def load_profile(source: LoadSource) -> np.ndarray:
    """A load-scaling vector S [T] (zero mean, unit variance) from a load series: `IsoFolder("nyiso",
    "data/nyiso")`, `CsvColumn("loads.csv", "load_mw")` or `RawSeries(values)`."""
    ProfileSource(source)  # anything else is refused here
    return _standardize(source.loads())


# ---- ISO load-profile download: NYISO from its public monthly archives (built in, no account);
# ---- CAISO and ERCOT through the `gridstatus` package: pip install 'fdia-graph[iso]'


def _month_firsts(start: _dt.date, end: _dt.date) -> Iterator[_dt.date]:
    """The first day of every month spanned by [start, end]."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield _dt.date(y, m, 1)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def nyiso_pal_month(first: _dt.date, cache_dir: Optional[Union[str, Path]] = None) -> bytes:
    """One month's zip of NYISO's 'pal' feed: real-time actual load by zone, about 5-minute samples, one
    CSV per day. With `cache_dir` the zip is kept there and read from there next time."""
    name = f"{first:%Y%m01}pal_csv.zip"
    path = None if cache_dir is None else Path(cache_dir) / name
    if path is not None and path.exists():
        return path.read_bytes()
    url = f"https://mis.nyiso.com/public/csv/pal/{name}"
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r:
        data = r.read()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def nyiso_zone_load(
    firsts: Iterable[_dt.date],
    grid: Optional[pd.DatetimeIndex] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """NYISO's real-time actual load (MW), one column per zone, on the 5-minute `grid` (None: the data's
    own span), from the monthly archives starting at `firsts`. Zones report a few seconds apart, so each
    is put on the grid first; gaps of up to three samples are interpolated."""
    pd = _pandas()
    frames = []
    for first in firsts:
        with zipfile.ZipFile(io.BytesIO(nyiso_pal_month(first, cache_dir))) as z:
            for nm in z.namelist():
                frames.append(pd.read_csv(io.BytesIO(z.read(nm)))[["Time Stamp", "Name", "Load"]])
    full = pd.concat(frames)
    full["Time Stamp"] = pd.to_datetime(full["Time Stamp"])
    piv = full.pivot_table(index="Time Stamp", columns="Name", values="Load", aggfunc="mean")
    if grid is None:
        grid = pd.date_range(piv.index.min().floor("5min"), piv.index.max().ceil("5min"), freq="5min")
    return piv.reindex(piv.index.union(grid)).interpolate(limit=3).reindex(grid)


class NyisoArchive:
    """NYISO system load (MW) at 5-minute resolution from the public monthly archives: no account and
    no extra dependency. It reads the 'pal' feed (real-time actual; 'palIntegrated' is only hourly)
    and sums the 11 zones per timestamp (`nyiso_zone_load`)."""

    def fetch(self, start: _dt.date, end: _dt.date) -> pd.Series:
        piv = nyiso_zone_load(_month_firsts(start, end))
        system = piv.sum(axis=1, min_count=piv.shape[1]).dropna().sort_index()  # all zones present
        mask = (system.index.date >= start) & (system.index.date <= end)
        return system[mask]


class GridstatusFeed:
    """System load (MW) through the `gridstatus` package; `operator` is its class name. The package
    is optional (pip install 'fdia-graph[iso]') and imported only when the feed fetches."""

    def __init__(self, operator: str) -> None:
        self.operator = operator

    def fetch(self, start: _dt.date, end: _dt.date) -> pd.Series:
        try:
            import gridstatus
        except ImportError as e:
            raise ImportError(
                f"fetching {self.operator} load needs the gridstatus package: pip install 'fdia-graph[iso]' "
                "(NYISO works without it)"
            ) from e
        day_after = end + _dt.timedelta(days=1)
        df = getattr(gridstatus, self.operator)().get_load(start=str(start), end=str(day_after))
        ts = df["Time"] if "Time" in df.columns else df.index
        return _pandas().Series(df["Load"].to_numpy(), index=list(ts)).sort_index()


class LoadFeed(Protocol):
    """An operator's load service: the system load (MW) over [start, end] as a timestamped series."""

    def fetch(self, start: _dt.date, end: _dt.date) -> pd.Series: ...


_FEEDS: dict[str, LoadFeed] = {
    Iso.NYISO: NyisoArchive(),
    Iso.CAISO: GridstatusFeed("CAISO"),
    Iso.ERCOT: GridstatusFeed("Ercot"),
}


def fetch_profile(
    iso: str,
    start: Union[str, _dt.date, _dt.datetime],
    end: Union[str, _dt.date, _dt.datetime],
    out: Optional[str] = None,
    resample_min: Optional[int] = None,
) -> np.ndarray:
    """Download the system load of `iso` ("nyiso", "caiso" or "ercot") over the days [start, end],
    inclusive, and return its scaling vector S [T]. `out` also saves the raw series as a CSV
    (timestamp, load_mw); `resample_min` time-interpolates to that minute cadence before standardizing."""
    req, span = ProfileFetch(iso, resample_min), DateSpan(start, end)
    series = _FEEDS[req.iso].fetch(span.first_day, span.last_day)
    if req.resample_min is not None:
        pd = _pandas()
        series = series.sort_index()
        series.index = pd.to_datetime(series.index)
        series = series.resample(f"{req.resample_min}min").interpolate(method="time").dropna()
    loads = series.to_numpy(dtype=float)
    if out is not None:
        pd = _pandas()
        pd.DataFrame({"timestamp": series.index, "load_mw": loads}).to_csv(out, index=False)
    return _standardize(loads)
