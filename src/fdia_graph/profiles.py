"""Load-profile ingestion + operating-state generation — the FRONT of the pipeline.

Take a real load time series from any source, turn it into a sequence of AC power-flow operating points, and
feed those into `generate()` to inject attacks. Profiles are pluggable across ISOs and time periods.

Pipeline:  load profile  ->  per-timestep scaled loads  ->  AC power flow (pandapower)  ->  operating states
The operating states are the [T, N, 4] pool the attack generator injects onto:
    columns = [ |V| (p.u.), P_inj (MW), Q_inj (MVAr), theta (deg) ]   (pandapower consumer-positive convention)

    load_profile(IsoFolder | CsvColumn | RawSeries)  ->  a normalized load-scaling vector S [T]
    fetch_profile(iso, start, end)                     ->  the same, downloaded from the operator
    generate_states(system, S) ->  a pool of operating states [T, N, 4]
"""

from __future__ import annotations

import datetime as _dt
import glob
import io
import os
import warnings
import zipfile
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Optional, Protocol, Union, runtime_checkable

import numpy as np

from .engine.core import _CASE  # bus-count -> pandapower builder; single source of supported systems
from .models.choices import (  # noqa: F401  re-exported beside the code that reads them
    Iso,
)
from .models.config import EXPORT_COLUMNS, IsoExport, ProfileFetch
from .models.inputs import CsvSpec, DateSpan, LoadValues, ProfileSource, SupportedSystem
from .registry import system_id  # noqa: F401  re-exported: callers read it from here before


def _pandas():
    """Lazy pandas import (optional dep, needed only for load-profile ingestion)."""
    try:
        import pandas as pd

        return pd
    except ImportError as e:
        raise ImportError("pandas is required for load profiles: pip install 'fdia-graph[generate]'") from e


if TYPE_CHECKING:
    import pandas as pd

    from .engine.pp_types import PandapowerNet
# per-bus scale at timestep t: clip(1 + K*S_t + N(0,SIGMA), CLIP_LO, CLIP_HI). K drives from the profile,
# SIGMA is per-bus jitter, clip holds a plausible load band.
K_DEFAULT = 0.1
SIGMA_DEFAULT = 0.03
CLIP_DEFAULT = (0.7, 1.3)
# AR(1) coefficient for load jitter: ~0.98 evolves smoothly minute-to-minute (~0.5%/min change vs ~4% white
# noise) at the same stationary bus-to-bus spread.
JITTER_RHO = 0.98


def _standardize(loads: np.ndarray) -> np.ndarray:
    """The load-scaling vector S [T]: zero mean, unit variance. Standardizing makes the K knob mean the
    same thing across sources of different absolute magnitude; a flat series stays at zero."""
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
    fixes the column names (`models.config.EXPORT_COLUMNS`); `IsoExport` refuses an operator without
    a known export format when the folder is described, not when it is read."""

    def __init__(self, iso: str, directory: str = ".") -> None:
        self.spec = IsoExport(iso, directory)

    @property
    def columns(self) -> tuple[str, str]:
        """(timestamp column, load column) of this operator's export."""
        return EXPORT_COLUMNS[self.spec.iso]

    def loads(self) -> np.ndarray:
        pd = _pandas()
        files = sorted(glob.glob(os.path.join(self.spec.directory, "*.csv")))
        if not files:  # the file system, not the input: the directory holds no export
            raise FileNotFoundError(f"no CSV files found under {self.spec.directory!r} for {self.spec.iso}")
        ts, load = self.columns
        frames = [pd.read_csv(f, parse_dates=[ts], index_col=ts) for f in files]
        return pd.concat(frames).sort_index()[load].dropna().to_numpy(dtype=float)


def load_profile(
    source: Union[LoadSource, str, Sequence[float], np.ndarray],
    path: Optional[str] = None,
    column: Optional[str] = None,
) -> np.ndarray:
    """A normalized load-scaling vector S [T] (zero mean, unit variance) from a load time series.

    source : where the loads come from, `IsoFolder("nyiso", "data/nyiso")`,
             `CsvColumn("loads.csv", "load_mw")` or `RawSeries(values)`; each reads itself.

    The older form, a string or an array with `path` / `column`, still works for one minor version and
    warns (`_legacy_source`).
    """
    spec = ProfileSource(source, path, column)
    reader: LoadSource = source if spec.kind == "source" else _legacy_source(spec)  # type: ignore[assignment]
    return _standardize(reader.loads())


def _legacy_source(spec: ProfileSource) -> LoadSource:
    """The pre-0.21 call `load_profile(source, path, column)` as a source: an operator name with a
    directory, a CSV path with its column (`CsvSpec` requires it), or the values themselves."""
    warnings.warn(
        "load_profile(source, path=, column=) is deprecated; pass IsoFolder, CsvColumn or RawSeries",
        DeprecationWarning,
        stacklevel=3,
    )
    if spec.kind == "values":
        return RawSeries(spec.series)
    if spec.kind == "iso":
        return IsoFolder(spec.text, spec.path or ".")
    csv = CsvSpec(spec.text, spec.column)
    return CsvColumn(os.fspath(csv.path), str(csv.column))


def _case_buses(key: int) -> np.ndarray:
    """Return (all_buses ordering) for a case — must match between the central jitter build and the worker."""
    import pandapower.networks as pn

    base = getattr(pn, _CASE[key])()
    return np.unique(np.concatenate([base.load["bus"].to_numpy(), base.gen["bus"].to_numpy()]))


def _solve_states_chunk(key: int, sf_chunk: np.ndarray) -> list[np.ndarray]:
    """Solve the AC operating state for each PRECOMPUTED per-bus scale-factor row in sf_chunk [L, nbus] (aligned
    to _case_buses order). Module-level/picklable so it runs as a multiprocessing worker. Returns a list of
    [N,4] state arrays; non-converging steps are skipped."""
    import pandapower as pp
    import pandapower.networks as pn

    base = getattr(pn, _CASE[key])()
    pp.runpp(base)  # seed a valid base state
    nodelist = sorted(base.bus.index)  # consistent bus order for the [N,4] rows
    base_load_p = base.load["p_mw"].to_numpy().copy()
    base_load_q = base.load["q_mvar"].to_numpy().copy()
    base_gen_p = base.gen["p_mw"].to_numpy().copy()
    load_buses = base.load["bus"].to_numpy()
    gen_buses = base.gen["bus"].to_numpy()
    all_buses = np.unique(np.concatenate([load_buses, gen_buses]))
    pos = {int(b): i for i, b in enumerate(nodelist)}
    out = []
    for sf in sf_chunk:  # sf: [nbus] scale factor per bus for this timestep
        b2s = dict(zip(all_buses.tolist(), sf.tolist()))
        base.load["p_mw"] = base_load_p * np.array([b2s[int(b)] for b in load_buses])
        base.load["q_mvar"] = base_load_q * np.array([b2s[int(b)] for b in load_buses])
        base.gen["p_mw"] = base_gen_p * np.array([b2s[int(b)] for b in gen_buses])
        try:
            pp.runpp(base, init="flat", max_iteration=50, tolerance_mva=1e-6)
        except Exception:
            continue  # infeasible operating point -> skip this timestep
        # [N,4] = [|V|, Pinj, Qinj, theta], the one column order the pool, the engine and node_x share.
        z = base.res_bus.reindex(nodelist)[["vm_pu", "p_mw", "q_mvar", "va_degree"]].to_numpy().copy()
        _remove_shunt_injections(z, base, pos)
        out.append(z)
    return out


def _remove_shunt_injections(z: np.ndarray, base: PandapowerNet, pos: dict[int, int]) -> None:
    """Subtract each shunt's draw from its bus's P and Q in place: state estimation models the shunt in
    the admittance matrix, not as an injection, so the stored injection must exclude it to be
    bad-data-clean."""
    if not len(base.res_shunt):
        return
    for b, ps, qs in zip(
        base.shunt.bus.to_numpy(), base.res_shunt.p_mw.to_numpy(), base.res_shunt.q_mvar.to_numpy()
    ):
        if int(b) in pos:
            z[pos[int(b)], 1] -= ps
            z[pos[int(b)], 2] -= qs


def _ar1_scale(
    S: np.ndarray, nbus: int, k: float, sigma: float, clip: tuple[float, float], rho: float, seed: int
) -> np.ndarray:
    """Build the [T, nbus] per-bus scale-factor matrix clip(1 + k*S_t + jitter_t), where jitter is a per-bus
    AR(1) process jitter_t = rho*jitter_{t-1} + sqrt(1-rho^2)*sigma*eps. AR(1) evolves the load smoothly
    (~0.5%/min at rho=0.98 vs ~4% for independent jitter) at unchanged stationary spread (std = sigma)."""
    T = len(S)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((T, nbus))
    jit = np.empty((T, nbus))
    jit[0] = sigma * eps[0]
    step = sigma * np.sqrt(1.0 - rho * rho)
    for t in range(1, T):  # AR(1) recursion
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
    """Turn a load-scaling profile into a pool of AC operating states [T, N, 4].

    For each timestep t: draw a per-bus scale factor clip(1 + k*S_t + j_t, *clip), with j_t an AR(1) jitter
    of stationary std sigma and coefficient JITTER_RHO, apply it to the
    case's base loads and generator setpoints, solve the AC power flow, and record the clean operating state.
    The recorded injection subtracts res_shunt (SE/BDD excludes the shunt), which is why benign states pass
    BDD. Non-converging timesteps are skipped.

    workers>1 splits timesteps across processes (independent power flows, near-linear scaling); contiguous
    chunks keep time order, each seeded seed+chunk_index. NOTE: spawning processes means a caller passing
    workers>1 must guard its top-level script with `if __name__ == "__main__":` (multiprocessing on Windows).

    Returns states [T, N, 4] = [|V| (p.u.), P_inj (MW), Q_inj (MVAr), theta (deg)], the pool `generate()` and
    `emit_from_state` consume, in the same column order as node_x and clean. (Before 0.12 the pool was
    [P, Q, V, theta]; generate() still accepts such pools and converts them, see generation.as_v_first.)
    """
    key = SupportedSystem(system, frozenset(_CASE)).number
    S = np.asarray(profile, dtype=float).ravel()
    if n is not None:
        S = S[:n]
    # Build the full scale-factor matrix centrally so AR(1) jitter has no discontinuity at chunk boundaries.
    nbus = len(_case_buses(key))
    SF = _ar1_scale(S, nbus, k, sigma, clip, JITTER_RHO, seed)  # [T, nbus]
    if workers and workers > 1 and len(S) >= workers:
        import multiprocessing as mp

        sf_chunks = np.array_split(SF, workers)  # contiguous rows -> concatenation preserves time order
        args = [(key, sf_chunks[i]) for i in range(len(sf_chunks))]
        with mp.Pool(workers) as pool:
            parts = pool.starmap(_solve_states_chunk, args)
        out = [z for part in parts for z in part]  # flatten in chunk (time) order
    else:
        out = _solve_states_chunk(key, SF)
    if not out:
        raise RuntimeError("no operating points converged; check the profile / scaling knobs")
    return np.asarray(out, dtype=np.float64)  # [T, N, 4]


# ---------------------------------------------------------------------------------------------------------
# Automatic ISO load-profile download. NYISO is a zero-dependency built-in (public monthly archives, no
# account); CAISO/ERCOT (and optionally NYISO) go through the `gridstatus` package: pip install 'fdia-graph[iso]'.
# ---------------------------------------------------------------------------------------------------------


def _month_firsts(start: _dt.date, end: _dt.date) -> Iterator[_dt.date]:
    """Yield the first-of-month date for every month spanned by [start, end]."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield _dt.date(y, m, 1)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


class NyisoArchive:
    """NYISO system load (MW) at 5-MINUTE resolution from the public monthly archives: built in, no
    account and no extra dependency.

    Uses the 'pal' feed (real-time actual, ~288 samples/day; 'palIntegrated' is only hourly), one CSV
    per day in a monthly zip. Each row is a control-zone reading; system load sums the 11 zones per
    timestamp."""

    def fetch(self, start: _dt.date, end: _dt.date) -> pd.Series:
        import requests

        pd = _pandas()
        frames = []
        for first in _month_firsts(start, end):
            url = f"https://mis.nyiso.com/public/csv/pal/{first:%Y%m01}pal_csv.zip"
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for nm in z.namelist():
                    df = pd.read_csv(io.BytesIO(z.read(nm)))
                    df["Time Stamp"] = pd.to_datetime(df["Time Stamp"])
                    frames.append(df[["Time Stamp", "Name", "Load"]])
        full = pd.concat(frames)
        # Zones have small clock offsets; pivot onto a clean 5-min grid before summing so no timestamp
        # under-counts.
        piv = full.pivot_table(index="Time Stamp", columns="Name", values="Load", aggfunc="mean")
        grid = pd.date_range(piv.index.min().floor("5min"), piv.index.max().ceil("5min"), freq="5min")
        piv = piv.reindex(piv.index.union(grid)).interpolate(limit=3).reindex(grid)
        system = piv.sum(axis=1, min_count=piv.shape[1]).dropna().sort_index()  # all zones present
        mask = (system.index.date >= start) & (system.index.date <= end)
        return system[mask]


class GridstatusFeed:
    """System load (MW) through the `gridstatus` package, which wraps each operator's data service
    behind one `.get_load(start, end)`. `operator` is the gridstatus class name. The package is an
    optional dependency (pip install 'fdia-graph[iso]'); the feed imports it when it fetches, so only
    a caller of this feed needs it."""

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
        df = getattr(gridstatus, self.operator)().get_load(
            start=str(start), end=str(end + _dt.timedelta(days=1))
        )
        ts = df["Time"] if "Time" in df.columns else df.index
        return _pandas().Series(df["Load"].to_numpy(), index=list(ts)).sort_index()


class LoadFeed(Protocol):
    """An operator's load service: the system load (MW) over [start, end] as a timestamped series."""

    def fetch(self, start: _dt.date, end: _dt.date) -> pd.Series: ...


# Where each operator's load comes from.
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
    """Download an ISO system-load series over [start, end] and return a normalized scaling vector S [T].

    iso          : "caiso" | "nyiso" | "ercot" (case-insensitive).
    start, end   : 'YYYY-MM-DD' strings (or date/datetime), inclusive.
    out          : optional path to also save the raw load series as a CSV (timestamp, load_mw) for provenance.
    resample_min : if set (e.g. 1), time-interpolate onto a uniform grid at that minute cadence before
                   standardizing (upsamples the 5-min feed; intermediate points are interpolated).

    Each operator's feed is in `_FEEDS`: NYISO is the built-in `NyisoArchive` (no dependency or account),
    CAISO and ERCOT a `GridstatusFeed` (pip install 'fdia-graph[iso]'). The returned S feeds
    generate_states() exactly like load_profile()'s output.
    """
    req, span = ProfileFetch(iso, resample_min), DateSpan(start, end)
    series = _FEEDS[req.iso].fetch(span.first_day, span.last_day)
    if req.resample_min is not None:
        # Time-interpolate onto a uniform resample_min-minute grid (as reference resample("1T").interpolate("time")).
        pd = _pandas()
        series = series.sort_index()
        series.index = pd.to_datetime(series.index)
        series = series.resample(f"{req.resample_min}min").interpolate(method="time").dropna()
    loads = series.to_numpy(dtype=float)
    if out is not None:
        pd = _pandas()
        pd.DataFrame({"timestamp": series.index, "load_mw": loads}).to_csv(out, index=False)
    return _standardize(loads)
