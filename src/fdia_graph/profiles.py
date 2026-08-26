"""Load-profile ingestion + operating-state generation — the FRONT of the pipeline.

Take a real load time series from any source, turn it into a sequence of AC power-flow operating points, and
feed those into `generate()` to inject attacks. Profiles are pluggable across ISOs and time periods.

Pipeline:  load profile  ->  per-timestep scaled loads  ->  AC power flow (pandapower)  ->  operating states
The operating states are the [T, N, 4] pool the attack generator injects onto:
    columns = [ P_inj (MW), Q_inj (MVAr), |V| (p.u.), theta (deg) ]   (pandapower consumer-positive convention)

    load_profile(source, ...)  ->  a normalized load-scaling vector S [T]
    generate_states(system, S) ->  a pool of operating states [T, N, 4]
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Tuple, Union

import os
import io
import glob
import zipfile
import datetime as _dt
import numpy as np

from ._core import _CASE   # bus-count -> pandapower builder; single source of supported systems

if TYPE_CHECKING:
    import pandas as pd

# per-bus scale at timestep t: clip(1 + K*S_t + N(0,SIGMA), CLIP_LO, CLIP_HI). K drives from the profile,
# SIGMA is per-bus jitter, clip holds a plausible load band.
K_DEFAULT = 0.1
SIGMA_DEFAULT = 0.03
CLIP_DEFAULT = (0.7, 1.3)
# AR(1) coefficient for load jitter: ~0.98 evolves smoothly minute-to-minute (~0.5%/min change vs ~4% white
# noise) at the same stationary bus-to-bus spread.
JITTER_RHO = 0.98

# how each supported ISO CSV names its timestamp and load columns
_ISO_COLS = {
    "caiso": ("interval_start_local", "load"),
    "nyiso": ("Time Stamp", "Load"),
}


def load_profile(source: Union[str, Sequence[float], np.ndarray], path: Optional[str] = None,
                 column: Optional[str] = None) -> np.ndarray:
    """Return a normalized load-scaling vector S [T] (zero-mean, unit-variance) from a load time series.

    `source` can be:
      - "caiso" or "nyiso": read every CSV under `path` (a directory), concatenate in time order, and
        standardize the ISO's load column.
      - a path to a single CSV plus `column`: standardize that column.
      - an array-like of raw load values: standardize it directly.

    Standardization makes the K knob mean the same thing across sources of different absolute magnitude.
    """
    if isinstance(source, (list, tuple, np.ndarray)):          # bring-your-own raw load series
        loads = np.asarray(source, dtype=float).ravel()
    elif isinstance(source, str) and source.lower() in _ISO_COLS:   # built-in ISO CSV directory
        import pandas as pd
        ts_col, load_col = _ISO_COLS[source.lower()]
        files = sorted(glob.glob(os.path.join(path or ".", "*.csv")))
        if not files:
            raise FileNotFoundError(f"no CSV files found under {path!r} for source {source!r}")
        frames = [pd.read_csv(f, parse_dates=[ts_col], index_col=ts_col) for f in files]
        full = __import__("pandas").concat(frames).sort_index()
        loads = full[load_col].dropna().to_numpy(dtype=float)
    else:                                                      # generic CSV path + column name
        import pandas as pd
        if column is None:
            raise ValueError("for a generic CSV `source`, pass the load `column` name")
        loads = pd.read_csv(source)[column].dropna().to_numpy(dtype=float)
    mu, sd = loads.mean(), loads.std()
    return (loads - mu) / (sd if sd > 0 else 1.0)             # standardized scaling vector S [T]


def _case_buses(key: int) -> np.ndarray:
    """Return (all_buses ordering) for a case — must match between the central jitter build and the worker."""
    import pandapower.networks as pn
    base = getattr(pn, _CASE[key])()
    return np.unique(np.concatenate([base.load["bus"].to_numpy(), base.gen["bus"].to_numpy()]))


def _solve_states_chunk(key: int, sf_chunk: np.ndarray) -> List[np.ndarray]:
    """Solve the AC operating state for each PRECOMPUTED per-bus scale-factor row in sf_chunk [L, nbus] (aligned
    to _case_buses order). Module-level/picklable so it runs as a multiprocessing worker. Returns a list of
    [N,4] state arrays; non-converging steps are skipped."""
    import pandapower as pp
    import pandapower.networks as pn
    base = getattr(pn, _CASE[key])()
    pp.runpp(base)                                             # seed a valid base state
    nodelist = sorted(base.bus.index)                          # consistent bus order for the [N,4] rows
    base_load_p = base.load["p_mw"].to_numpy().copy()
    base_load_q = base.load["q_mvar"].to_numpy().copy()
    base_gen_p = base.gen["p_mw"].to_numpy().copy()
    load_buses = base.load["bus"].to_numpy()
    gen_buses = base.gen["bus"].to_numpy()
    all_buses = np.unique(np.concatenate([load_buses, gen_buses]))
    pos = {int(b): i for i, b in enumerate(nodelist)}
    out = []
    for sf in sf_chunk:                                        # sf: [nbus] scale factor per bus for this timestep
        b2s = dict(zip(all_buses.tolist(), sf.tolist()))
        base.load["p_mw"] = base_load_p * np.array([b2s[int(b)] for b in load_buses])
        base.load["q_mvar"] = base_load_q * np.array([b2s[int(b)] for b in load_buses])
        base.gen["p_mw"] = base_gen_p * np.array([b2s[int(b)] for b in gen_buses])
        try:
            pp.runpp(base, init="flat", max_iteration=50, tolerance_mva=1e-6)
        except Exception:
            continue                                          # infeasible operating point -> skip this timestep
        z = base.res_bus.reindex(nodelist)[["p_mw", "q_mvar", "vm_pu", "va_degree"]].to_numpy().copy()
        if len(base.res_shunt):                               # SE excludes the shunt, so subtract it -> BDD-clean injection
            for b, ps, qs in zip(base.shunt.bus.to_numpy(), base.res_shunt.p_mw.to_numpy(), base.res_shunt.q_mvar.to_numpy()):
                if int(b) in pos:
                    z[pos[int(b)], 0] -= ps
                    z[pos[int(b)], 1] -= qs
        out.append(z)
    return out


def _ar1_scale(S: np.ndarray, nbus: int, k: float, sigma: float, clip: Tuple[float, float],
               rho: float, seed: int) -> np.ndarray:
    """Build the [T, nbus] per-bus scale-factor matrix clip(1 + k*S_t + jitter_t), where jitter is a per-bus
    AR(1) process jitter_t = rho*jitter_{t-1} + sqrt(1-rho^2)*sigma*eps. AR(1) evolves the load smoothly
    (~0.5%/min at rho=0.98 vs ~4% for independent jitter) at unchanged stationary spread (std = sigma)."""
    T = len(S)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((T, nbus))
    jit = np.empty((T, nbus))
    jit[0] = sigma * eps[0]
    step = sigma * np.sqrt(1.0 - rho * rho)
    for t in range(1, T):                                     # AR(1) recursion
        jit[t] = rho * jit[t - 1] + step * eps[t]
    return np.clip(1.0 + k * S[:, None] + jit, clip[0], clip[1])


def generate_states(system: Union[int, str], profile: Union[np.ndarray, Sequence[float]], k: float = K_DEFAULT,
                    sigma: float = SIGMA_DEFAULT, clip: Tuple[float, float] = CLIP_DEFAULT, n: Optional[int] = None,
                    seed: int = 123, workers: Optional[int] = None) -> np.ndarray:
    """Turn a load-scaling profile into a pool of AC operating states [T, N, 4].

    For each timestep t: draw a per-bus scale factor clip(1 + k*S_t + N(0, sigma), *clip), apply it to the
    case's base loads and generator setpoints, solve the AC power flow, and record the clean operating state.
    The recorded injection subtracts res_shunt (SE/BDD excludes the shunt), which is why benign states pass
    BDD. Non-converging timesteps are skipped.

    workers>1 splits timesteps across processes (independent power flows, near-linear scaling); contiguous
    chunks keep time order, each seeded seed+chunk_index. NOTE: spawning processes means a caller passing
    workers>1 must guard its top-level script with `if __name__ == "__main__":` (multiprocessing on Windows).

    Returns states [T, N, 4] = [P_inj (MW), Q_inj (MVAr), |V| (p.u.), theta (deg)], the pool `generate()` and
    `emit_from_state` consume.
    """
    key = int(str(system).lower().replace("ieee", ""))
    if key not in _CASE:
        raise ValueError(f"unknown system {system!r}; supported: {sorted(_CASE)}")
    S = np.asarray(profile, dtype=float).ravel()
    if n is not None:
        S = S[:n]
    # Build the full scale-factor matrix centrally so AR(1) jitter has no discontinuity at chunk boundaries.
    nbus = len(_case_buses(key))
    SF = _ar1_scale(S, nbus, k, sigma, clip, JITTER_RHO, seed)   # [T, nbus]
    if workers and workers > 1 and len(S) >= workers:
        import multiprocessing as mp
        sf_chunks = np.array_split(SF, workers)               # contiguous rows -> concatenation preserves time order
        args = [(key, sf_chunks[i]) for i in range(len(sf_chunks))]
        with mp.Pool(workers) as pool:
            parts = pool.starmap(_solve_states_chunk, args)
        out = [z for part in parts for z in part]             # flatten in chunk (time) order
    else:
        out = _solve_states_chunk(key, SF)
    if not out:
        raise RuntimeError("no operating points converged; check the profile / scaling knobs")
    return np.asarray(out, dtype=np.float64)                  # [T, N, 4]


# ---------------------------------------------------------------------------------------------------------
# Automatic ISO load-profile download. NYISO is a zero-dependency built-in (public monthly archives, no
# account); CAISO/ERCOT (and optionally NYISO) go through the `gridstatus` package: pip install 'fdia-graph[iso]'.
# ---------------------------------------------------------------------------------------------------------
_ISOS = ("caiso", "nyiso", "ercot")


def _as_date(d: Union[str, _dt.date, _dt.datetime]) -> _dt.date:
    """Accept 'YYYY-MM-DD', a date, or a datetime and return a date."""
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, _dt.date):
        return d
    return _dt.datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


def _month_firsts(start: _dt.date, end: _dt.date) -> Iterator[_dt.date]:
    """Yield the first-of-month date for every month spanned by [start, end]."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield _dt.date(y, m, 1)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _fetch_nyiso(start: _dt.date, end: _dt.date) -> "pd.Series":
    """NYISO system load (MW) over [start, end] at 5-MINUTE resolution, built-in and account-free.

    Uses the 'pal' feed (real-time actual, ~288 samples/day; 'palIntegrated' is only hourly), one CSV per day
    in a monthly zip. Each row is a control-zone reading; system load sums the 11 zones per timestamp. Returns
    a pandas Series indexed by timestamp.
    """
    import requests
    import pandas as pd
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
    # Zones have small clock offsets; pivot onto a clean 5-min grid before summing so no timestamp under-counts.
    piv = full.pivot_table(index="Time Stamp", columns="Name", values="Load", aggfunc="mean")
    grid = pd.date_range(piv.index.min().floor("5min"), piv.index.max().ceil("5min"), freq="5min")
    piv = piv.reindex(piv.index.union(grid)).interpolate(limit=3).reindex(grid)
    system = piv.sum(axis=1, min_count=piv.shape[1]).dropna().sort_index()      # all zones present -> valid sum
    mask = (system.index.date >= start) & (system.index.date <= end)
    return system[mask]


def _fetch_gridstatus(iso: str, start: _dt.date, end: _dt.date) -> "pd.Series":
    """System load (MW) over [start, end] via the `gridstatus` package — uniform across CAISO/NYISO/ERCOT.

    gridstatus wraps each operator's data service behind one `.get_load(start, end)`. Returns a pandas Series.
    """
    import gridstatus
    cls = {"caiso": gridstatus.CAISO, "nyiso": gridstatus.NYISO, "ercot": gridstatus.Ercot}[iso]
    df = cls().get_load(start=str(start), end=str(end + _dt.timedelta(days=1)))
    ts = df["Time"] if "Time" in df.columns else df.index
    return __import__("pandas").Series(df["Load"].to_numpy(), index=list(ts)).sort_index()


def fetch_profile(iso: str, start: Union[str, _dt.date, _dt.datetime],
                  end: Union[str, _dt.date, _dt.datetime], out: Optional[str] = None,
                  resample_min: Optional[int] = None) -> np.ndarray:
    """Download an ISO system-load series over [start, end] and return a normalized scaling vector S [T].

    iso          : "caiso" | "nyiso" | "ercot" (case-insensitive).
    start, end   : 'YYYY-MM-DD' strings (or date/datetime), inclusive.
    out          : optional path to also save the raw load series as a CSV (timestamp, load_mw) for provenance.
    resample_min : if set (e.g. 1), time-interpolate onto a uniform grid at that minute cadence before
                   standardizing (upsamples the 5-min feed; intermediate points are interpolated).

    NYISO needs no dependencies or account; CAISO/ERCOT use `gridstatus` (pip install 'fdia-graph[iso]'). The
    returned S feeds generate_states() exactly like load_profile()'s output.
    """
    iso = str(iso).lower()
    if iso not in _ISOS:
        raise ValueError(f"unknown iso {iso!r}; expected one of {_ISOS}")
    start, end = _as_date(start), _as_date(end)
    if iso == "nyiso":
        series = _fetch_nyiso(start, end)                     # zero-dependency built-in, 5-minute
    else:                                                     # caiso / ercot -> gridstatus (5-min native)
        try:
            import gridstatus  # noqa: F401
        except ImportError:
            raise ImportError(
                f"fetching {iso.upper()} load needs the gridstatus package: pip install 'fdia-graph[iso]' "
                f"(NYISO works without it)."
            )
        series = _fetch_gridstatus(iso, start, end)
    if resample_min is not None:
        # Time-interpolate onto a uniform resample_min-minute grid (as reference resample("1T").interpolate("time")).
        import pandas as pd
        series = series.sort_index()
        series.index = pd.to_datetime(series.index)
        series = series.resample(f"{int(resample_min)}min").interpolate(method="time").dropna()
    loads = series.to_numpy(dtype=float)
    if out is not None:
        import pandas as pd
        pd.DataFrame({"timestamp": series.index, "load_mw": loads}).to_csv(out, index=False)
    mu, sd = loads.mean(), loads.std()
    return (loads - mu) / (sd if sd > 0 else 1.0)             # standardized scaling vector S [T]
