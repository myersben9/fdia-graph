"""Load profiles and operating states: the front of the pipeline.

A real load time series becomes a sequence of AC operating points, the pool `generate()` attacks:

    load_profile(source) or fetch_profile(iso, start, end)  ->  a load-scaling vector S [T]
    generate_states(system, S)                              ->  the pool of states [T, N, 4]

The pool's columns are [|V| (pu), P_inj (MW), Q_inj (MVAr), theta (deg)], in pandapower's
consumer-positive convention.
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

from .constants.files import EXPORT_COLUMNS
from .constants.grid import IEEE_CASES  # bus-count -> pandapower builder; single source of supported systems
from .constants.time import LOAD_JITTER_RHO
from .models.choices import Iso
from .models.config import IsoExport, ProfileFetch
from .models.inputs import CsvSpec, DateSpan, LoadValues, ProfileSource, SupportedSystem


def _pandas():
    """pandas, imported on first use: an optional dependency only load profiles need."""
    try:
        import pandas as pd

        return pd
    except ImportError as e:
        raise ImportError("pandas is required for load profiles: pip install 'fdia-graph[generate]'") from e


if TYPE_CHECKING:
    import pandas as pd

    from .systems.pandapower_types import PandapowerNet
# per-bus scale at timestep t: clip(1 + K*S_t + jitter_t, *CLIP). K is the profile's weight, SIGMA the
# jitter's std, and the clip holds a plausible load band.
K_DEFAULT = 0.1
SIGMA_DEFAULT = 0.03
CLIP_DEFAULT = (0.7, 1.3)


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


def _case_buses(key: int) -> np.ndarray:
    """The buses with a load or a generator, sorted: the column order of a scale-factor row. The
    worker rebuilds the same order."""
    import pandapower.networks as pn

    base = getattr(pn, IEEE_CASES[key])()
    return np.unique(np.concatenate([base.load["bus"].to_numpy(), base.gen["bus"].to_numpy()]))


def _solve_states_chunk(key: int, sf_chunk: np.ndarray) -> list[np.ndarray]:
    """The AC operating state [N, 4] of each scale-factor row of `sf_chunk` [L, nbus] (columns in
    `_case_buses` order). A row whose power flow does not converge is skipped. Module-level, so it
    can run in a worker process."""
    import pandapower as pp
    import pandapower.networks as pn

    base = getattr(pn, IEEE_CASES[key])()
    pp.runpp(base)  # seed a valid base state
    nodelist = sorted(base.bus.index)
    base_load_p = base.load["p_mw"].to_numpy().copy()
    base_load_q = base.load["q_mvar"].to_numpy().copy()
    base_gen_p = base.gen["p_mw"].to_numpy().copy()
    load_buses = base.load["bus"].to_numpy()
    gen_buses = base.gen["bus"].to_numpy()
    all_buses = np.unique(np.concatenate([load_buses, gen_buses]))
    # each load's and generator's column in a scale-factor row (all_buses is sorted and unique)
    at_load, at_gen = np.searchsorted(all_buses, load_buses), np.searchsorted(all_buses, gen_buses)
    pos = {int(b): i for i, b in enumerate(nodelist)}
    out = []
    for sf in sf_chunk:
        sf = np.asarray(sf, np.float64)
        base.load["p_mw"] = base_load_p * sf[at_load]
        base.load["q_mvar"] = base_load_q * sf[at_load]
        base.gen["p_mw"] = base_gen_p * sf[at_gen]
        try:
            pp.runpp(base, init="flat", max_iteration=50, tolerance_mva=1e-6)
        except Exception:
            continue  # no power-flow solution: skip this timestep
        # [N, 4] = [|V|, Pinj, Qinj, theta], the column order the pool, the engine and node_x share
        z = base.res_bus.reindex(nodelist)[["vm_pu", "p_mw", "q_mvar", "va_degree"]].to_numpy().copy()
        _remove_shunt_injections(z, base, pos)
        out.append(z)
    return out


def _remove_shunt_injections(z: np.ndarray, base: PandapowerNet, pos: dict[int, int]) -> None:
    """Subtract each shunt's draw from its bus's P and Q, in place. State estimation models a shunt in
    the admittance matrix, not as an injection, so the stored injection must exclude it."""
    if not len(base.res_shunt):
        return
    keep = np.array([int(b) in pos for b in base.shunt.bus.to_numpy()], bool)
    rows = np.array([pos[int(b)] for b in base.shunt.bus.to_numpy()[keep]], int)
    # unbuffered: two shunts on one bus both subtract
    np.subtract.at(z[:, 1], rows, base.res_shunt.p_mw.to_numpy()[keep])
    np.subtract.at(z[:, 2], rows, base.res_shunt.q_mvar.to_numpy()[keep])


def _ar1_scale(
    S: np.ndarray, nbus: int, k: float, sigma: float, clip: tuple[float, float], rho: float, seed: int
) -> np.ndarray:
    """The per-bus scale factors [T, nbus]: clip(1 + k*S_t + jitter_t), with a per-bus AR(1) jitter
    jitter_t = rho*jitter_{t-1} + sqrt(1-rho^2)*sigma*eps. The AR(1) makes each load drift smoothly
    (~0.5%/min at rho=0.98, against ~4% for independent jitter) while its stationary std stays sigma."""
    T = len(S)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((T, nbus))
    jit = np.empty((T, nbus))
    jit[0] = sigma * eps[0]
    step = sigma * np.sqrt(1.0 - rho * rho)
    for t in range(1, T):
        jit[t] = rho * jit[t - 1] + step * eps[t]
    return np.clip(1.0 + k * S[:, None] + jit, clip[0], clip[1])


def generate_states(
    system: Union[int, str],
    profile: Union[np.ndarray, Sequence[float]],
    k: float = K_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
    clip: tuple[float, float] = CLIP_DEFAULT,
    n: Optional[int] = None,
    seed: int = 123,
    workers: Optional[int] = None,
) -> np.ndarray:
    """A pool of AC states [T', N, 4] = [|V| (pu), P_inj (MW), Q_inj (MVAr), theta (deg)] from a profile of
    T steps: each timestep scales the loads and generator setpoints per bus (`_ar1_scale`) and solves the AC
    power flow. One that does not converge is dropped, so T' <= T and the rows no longer align with the
    profile. `workers` > 1 needs an `if __name__ == "__main__":` guard."""
    key = SupportedSystem(system, frozenset(IEEE_CASES)).number
    S = np.asarray(profile, dtype=float).ravel()
    if n is not None:
        S = S[:n]
    # the whole scale-factor matrix is built here, so the AR(1) jitter is continuous across chunks
    nbus = len(_case_buses(key))
    SF = _ar1_scale(S, nbus, k, sigma, clip, LOAD_JITTER_RHO, seed)  # [T, nbus]
    if workers and workers > 1 and len(S) >= workers:
        import multiprocessing as mp

        sf_chunks = np.array_split(SF, workers)  # contiguous rows: concatenating keeps time order
        args = [(key, sf_chunks[i]) for i in range(len(sf_chunks))]
        with mp.Pool(workers) as pool:
            parts = pool.starmap(_solve_states_chunk, args)
        out = [z for part in parts for z in part]
    else:
        out = _solve_states_chunk(key, SF)
    if not out:
        raise RuntimeError("no operating points converged; check the profile / scaling knobs")
    return np.asarray(out, dtype=np.float64)


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
