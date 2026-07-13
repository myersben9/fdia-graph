"""Load-profile ingestion + operating-state generation — the FRONT of the pipeline.

Until now the SDK consumed pre-computed operating states (a pool shipped with each release). This module
lets the group own the whole pipeline instead: take a real load time series from any source, turn it into a
sequence of AC power-flow operating points, and feed those straight into `generate()` to inject attacks. Load
profiles are pluggable, so you can swap in data from different ISOs and different time periods and regenerate
everything with full control.

Pipeline:  load profile  ->  per-timestep scaled loads  ->  AC power flow (pandapower)  ->  operating states
The operating states are the [T, N, 4] pool the attack generator injects onto:
    columns = [ P_inj (MW), Q_inj (MVAr), |V| (p.u.), theta (deg) ]   (pandapower consumer-positive convention)

Two public functions:
    load_profile(source, ...)  ->  a normalized load-scaling vector S [T]
    generate_states(system, S) ->  a pool of operating states [T, N, 4]
"""
import os
import io
import glob
import zipfile
import datetime as _dt
import numpy as np

# scaling recipe (matches the reference init_dataset pipeline): the per-bus scale at timestep t is
#   clip( 1 + K * S_t + N(0, SIGMA) ,  CLIP_LO, CLIP_HI )
# K sets how hard the (standardized) load profile drives the operating point; SIGMA is per-bus jitter so
# buses do not all move in lockstep; the clip keeps every bus in a plausible load band.
K_DEFAULT = 0.1
SIGMA_DEFAULT = 0.03
CLIP_DEFAULT = (0.7, 1.3)

# how each supported ISO CSV names its timestamp and load columns
_ISO_COLS = {
    "caiso": ("interval_start_local", "load"),
    "nyiso": ("Time Stamp", "Load"),
}


def load_profile(source, path=None, column=None):
    """Return a normalized load-scaling vector S [T] (zero-mean, unit-variance) from a load time series.

    `source` can be:
      - "caiso" or "nyiso": read every CSV under `path` (a directory), concatenate in time order, and
        standardize the ISO's load column. This is the built-in ISO ingestion.
      - a path to a single CSV plus `column`: standardize that column (generic source).
      - an array-like of raw load values: standardize it directly (bring your own profile).

    Standardization (subtract mean, divide by std) matches the reference pipeline and makes the recipe's K
    knob mean the same thing across sources of different absolute magnitude.
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


def generate_states(system, profile, k=K_DEFAULT, sigma=SIGMA_DEFAULT, clip=CLIP_DEFAULT, n=None, seed=123):
    """Turn a load-scaling profile into a pool of AC operating states [T, N, 4].

    For each timestep t: draw a per-bus scale factor clip(1 + k*S_t + N(0, sigma), *clip), apply it to the
    case's base loads and generator setpoints, solve the AC power flow, and record the clean operating state.
    The recorded injection is the SE-CONSISTENT (without-shunt) injection: pandapower's WLS/BDD measurement
    function excludes the shunt element, so we subtract res_shunt from the bus P/Q. Keeping the with-shunt
    value would make shunt buses inconsistent with the estimator and flag them under bad-data detection — this
    subtraction is exactly why the resulting benign states pass BDD. Non-converging timesteps are skipped.

    Returns states [T, N, 4] = [P_inj (MW), Q_inj (MVAr), |V| (p.u.), theta (deg)], the pool `generate()` and
    `emit_from_state` consume.
    """
    import pandapower as pp
    import pandapower.networks as pn
    cases = {14: pn.case14, 118: pn.case118, 300: pn.case300}
    key = int(str(system).lower().replace("ieee", ""))
    if key not in cases:
        raise ValueError(f"unknown system {system!r}; expected 14, 118, or 300")
    base = cases[key]()
    pp.runpp(base)                                             # solve the base case to seed a valid state
    nodelist = sorted(base.bus.index)                          # consistent bus order for the [N,4] rows
    # snapshot the base loads/gens once; every timestep scales THESE, not the previous (drifted) values
    base_load_p = base.load["p_mw"].to_numpy().copy()
    base_load_q = base.load["q_mvar"].to_numpy().copy()
    base_gen_p = base.gen["p_mw"].to_numpy().copy()
    load_buses = base.load["bus"].to_numpy()
    gen_buses = base.gen["bus"].to_numpy()
    all_buses = np.unique(np.concatenate([load_buses, gen_buses]))
    pos = {int(b): i for i, b in enumerate(nodelist)}

    rng = np.random.default_rng(seed)
    S = np.asarray(profile, dtype=float).ravel()
    if n is not None:
        S = S[:n]

    out = []
    for St in S:
        # per-bus scale: the profile drives the mean, the jitter decorrelates buses, the clip keeps it plausible
        sf = np.clip(1.0 + k * St + rng.normal(0.0, sigma, size=len(all_buses)), clip[0], clip[1])
        b2s = dict(zip(all_buses.tolist(), sf.tolist()))
        base.load["p_mw"] = base_load_p * np.array([b2s[int(b)] for b in load_buses])
        base.load["q_mvar"] = base_load_q * np.array([b2s[int(b)] for b in load_buses])
        base.gen["p_mw"] = base_gen_p * np.array([b2s[int(b)] for b in gen_buses])
        try:
            pp.runpp(base, init="flat", max_iteration=50, tolerance_mva=1e-6)
        except Exception:
            continue                                          # infeasible operating point -> skip this timestep
        z = base.res_bus.reindex(nodelist)[["p_mw", "q_mvar", "vm_pu", "va_degree"]].to_numpy().copy()
        if len(base.res_shunt):                               # SE-consistent without-shunt injection (BDD-clean)
            for b, ps, qs in zip(base.shunt.bus.to_numpy(), base.res_shunt.p_mw.to_numpy(), base.res_shunt.q_mvar.to_numpy()):
                if int(b) in pos:
                    z[pos[int(b)], 0] -= ps                    # remove shunt conductance draw from P
                    z[pos[int(b)], 1] -= qs                    # remove shunt susceptance draw from Q
        out.append(z)
    if not out:
        raise RuntimeError("no operating points converged; check the profile / scaling knobs")
    return np.asarray(out, dtype=np.float64)                  # [T, N, 4]


# ---------------------------------------------------------------------------------------------------------
# Automatic ISO load-profile download.
# Fetch a real system-load time series straight from the operator's public data service, so a load profile
# is one call — no manual CSV wrangling. NYISO is a zero-dependency built-in (public monthly archives, no
# account). For CAISO and ERCOT we prefer the maintained `gridstatus` package (uniform, handles every ISO
# quirk and any auth); it also covers NYISO. Install it with:  pip install 'fdia-graph[iso]'.
# ---------------------------------------------------------------------------------------------------------
_ISOS = ("caiso", "nyiso", "ercot")


def _as_date(d):
    """Accept 'YYYY-MM-DD', a date, or a datetime and return a date."""
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, _dt.date):
        return d
    return _dt.datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


def _month_firsts(start, end):
    """Yield the first-of-month date for every month spanned by [start, end]."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield _dt.date(y, m, 1)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _fetch_nyiso(start, end):
    """NYISO system load (MW) over [start, end] at 5-MINUTE resolution, built-in and account-free.

    NYISO publishes 'pal' (real-time actual load) as one CSV per day at 5-minute cadence, bundled into a
    monthly zip. (Their 'palIntegrated' feed is only hourly — we use 'pal' for the finest resolution NYISO
    offers, ~288 samples/day.) Each row is one control-zone reading; the system load is the sum over the 11
    zones at each timestamp. Returns a pandas Series indexed by timestamp.
    """
    import requests
    import pandas as pd
    frames = []
    for first in _month_firsts(start, end):
        url = f"http://mis.nyiso.com/public/csv/pal/{first:%Y%m01}pal_csv.zip"
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for nm in z.namelist():
                df = pd.read_csv(io.BytesIO(z.read(nm)))
                df["Time Stamp"] = pd.to_datetime(df["Time Stamp"])
                frames.append(df[["Time Stamp", "Name", "Load"]])
    full = pd.concat(frames)
    # Zones report on the same 5-min cadence but with small clock offsets; pivot to a zone-per-column table
    # on a clean 5-min grid (interpolating short gaps), then sum across zones -> a proper system total that
    # is not under-counted at unaligned timestamps.
    piv = full.pivot_table(index="Time Stamp", columns="Name", values="Load", aggfunc="mean")
    grid = pd.date_range(piv.index.min().floor("5min"), piv.index.max().ceil("5min"), freq="5min")
    piv = piv.reindex(piv.index.union(grid)).interpolate(limit=3).reindex(grid)
    system = piv.sum(axis=1, min_count=piv.shape[1]).dropna().sort_index()      # all zones present -> valid sum
    mask = (system.index.date >= start) & (system.index.date <= end)
    return system[mask]


def _fetch_gridstatus(iso, start, end):
    """System load (MW) over [start, end] via the `gridstatus` package — uniform across CAISO/NYISO/ERCOT.

    gridstatus wraps each operator's real data service (and any required credentials) behind one
    `.get_load(start, end)` returning a frame with a 'Load' column. Returns a pandas Series.
    """
    import gridstatus
    cls = {"caiso": gridstatus.CAISO, "nyiso": gridstatus.NYISO, "ercot": gridstatus.Ercot}[iso]
    df = cls().get_load(start=str(start), end=str(end + _dt.timedelta(days=1)))
    ts = df["Time"] if "Time" in df.columns else df.index
    return __import__("pandas").Series(df["Load"].to_numpy(), index=list(ts)).sort_index()


def fetch_profile(iso, start, end, out=None):
    """Download an ISO system-load series over [start, end] and return a normalized scaling vector S [T].

    iso        : "caiso" | "nyiso" | "ercot" (case-insensitive).
    start, end : 'YYYY-MM-DD' strings (or date/datetime), inclusive.
    out        : optional path to also save the raw load series as a CSV (timestamp, load_mw) for provenance.

    NYISO works with no dependencies or account. CAISO and ERCOT use the `gridstatus` package when installed
    (pip install 'fdia-graph[iso]'); NYISO also uses it if present. The returned S feeds generate_states()
    exactly like load_profile()'s output, so switching ISOs or time windows is a one-line change.
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
    loads = series.to_numpy(dtype=float)
    if out is not None:
        import pandas as pd
        pd.DataFrame({"timestamp": series.index, "load_mw": loads}).to_csv(out, index=False)
    mu, sd = loads.mean(), loads.std()
    return (loads - mu) / (sd if sd > 0 else 1.0)             # standardized scaling vector S [T]
