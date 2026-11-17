"""generate(system, name, **knobs) — build a custom dataset and register it as `name`.

`generate` writes one timeline file (`fdia_graph.timeline`): every frame of the operating-point
pool scanned in time order with At and Am attack episodes, which `fg.load(name)` reads as a
record table (`order="random"`) or as the timeline (`order="time"`). The knobs are those of
`timeline.generate_timeline` plus `frames` (how many pool timesteps to walk, default the whole
pool). This module also reads the operating-point pool (`_load_states`, any column order, npz or
HDF5).

The one path that generates a timeline is `TimelineSource.generate` (`generation/source.py`); the grids are
its subclasses, `IeeeSource` (`generation/ieee.py`) and the New York grid's (`generation/nygrid/source.py`),
`DesignPool` (`generation/design.py`) is the worker processes both design on, and `TimelineWriter`
(`generation/write.py`) writes the file of both.

  states            source of operating points: path to a pool .npz or .h5 (key 'X' [T,N,4]) or an init dir of
                    X_*.npy; if None, uses $FDIA_GRAPH_INIT or downloads the system's operating-point pool.
  out               output .h5 path (default under the cache dir)
"""

from __future__ import annotations

import glob
import os
from typing import Any, Optional, Union

import h5py
import numpy as np

from ..constants.channels import NodeChannel
from ..errors import UnknownColumnOrder
from ..models.config import ShardRun, TimelineSettings
from ..models.inputs import StatePool, StateSource
from ..registry import CACHE_DIR, register_local
from ..schema import KIND_TIMELINE


def as_v_first(X: np.ndarray) -> np.ndarray:
    """Return a state pool [T,N,4] in the SDK's one column order [|V|, Pinj, Qinj, theta].

    Pools written before 0.12 (the downloadable pool assets, older `states=` files, the original
    pipeline's X_*.npy) are [Pinj, Qinj, |V|, theta]. The order is detected from the data: the voltage
    column is the only one whose every value sits in a per-unit band around 1.0, so a pool is
    converted when column 2 looks like |V| and column 0 does not. Anything else is an error rather
    than a guess.
    """
    X = StatePool(X).X

    def looks_like_v(col: np.ndarray) -> bool:
        return bool(np.all((col > 0.5) & (col < 1.5)))

    v0, v2 = looks_like_v(X[:, :, NodeChannel.V]), looks_like_v(X[:, :, NodeChannel.Q_INJ])
    if v0 and not v2:
        return X
    if v2 and not v0:
        return X[:, :, [2, 0, 1, 3]]  # [P, Q, V, th] -> [V, P, Q, th]
    raise UnknownColumnOrder("cannot tell the pool's column order; expected [|V|, Pinj, Qinj, theta]")


def _load_states(
    system: Union[int, str], states: Optional[Union[str, np.ndarray]], pool_cap: int = 8000
) -> np.ndarray:
    """Return an operating-point pool [T,N,4] = [|V|, Pinj, Qinj, theta] to inject attacks onto.

    Priority: explicit `states` -> $FDIA_GRAPH_INIT -> downloadable pool asset. A local init DIRECTORY can
    hold ~86k per-timestep files, so we stride-sample at most `pool_cap` evenly across the timeline (keeps
    daily/seasonal variety) rather than reading every file, which made a naive load take minutes. Every
    source goes through as_v_first, so pools saved in the older [P, Q, V, theta] order still load.
    """
    return as_v_first(_read_states(system, states, pool_cap))


def _read_states(
    system: Union[int, str], states: Optional[Union[str, np.ndarray]], pool_cap: int
) -> np.ndarray:
    source = StateSource(states)
    if source.array is not None:  # an in-memory pool is used as it is (no disk)
        return source.array
    # Source: caller arg wins, else FDIA_GRAPH_INIT, else None (downloaded below).
    src = source.path or os.environ.get("FDIA_GRAPH_INIT")
    if src and os.path.isdir(src):
        # Init directory: all X_*.npy sorted by integer timestep (name "X_<t>.npy").
        xs = sorted(glob.glob(os.path.join(src, "X_*.npy")), key=lambda p: int(os.path.basename(p)[2:-4]))
        stride = max(1, len(xs) // pool_cap)  # even stride -> diverse sample, bounded cost
        # float64 for the WLS/AC solve downstream.
        return np.stack([np.load(f) for f in xs[::stride][:pool_cap]]).astype(np.float64)
    if src and src.endswith(".npz"):
        return np.load(src)["X"].astype(np.float64)  # precomputed compact pool (the SDK default)
    if src and src.endswith((".h5", ".hdf5")):
        with h5py.File(src, "r") as f:  # the pool as HDF5, dataset "X" [T, N, 4]
            return np.asarray(f["X"], np.float64)
    # fall back to the downloadable operating-point pool for this system, at the pinned data release
    # (which carries all eight ladder pools; a hardcoded old tag here once 404'd newer systems)
    from ..download import ensure_local
    from ..registry import pool_spec

    return _read_states(system, ensure_local(pool_spec(system)), pool_cap)


def generate(
    system: Union[int, str],
    name: str,
    frames: Optional[int] = None,
    states: Optional[Union[str, np.ndarray]] = None,
    out: Optional[str] = None,
    seed: int = 123,
    settings: Optional[TimelineSettings] = None,
    **knobs: Any,
) -> str:
    """Walk one attacked timeline over the operating-point pool of `system`, write it as one HDF5
    file and register it as `name`; returns the path. `frames` caps the pool timesteps walked;
    `settings` and every other knob are `timeline.generate_timeline`'s (a `TimelineSettings`, and the
    flat keywords families, attacked_frac, ramp_rate, ramp_len, am_len, hops, max_load_mw, redundancy,
    split, min_tamper, min_budget, am_attack, stealth_scale, workers). The registry records the
    settings as nested plain values."""
    from ..timeline import generate_timeline

    frames = ShardRun(frames).frames
    s = TimelineSettings.of(settings, **knobs)
    X = _load_states(system, states)
    if frames is not None:
        X = X[:frames]
    out = out or os.path.join(CACHE_DIR, f"{name}.h5")
    path = generate_timeline(system, states=X, settings=s, seed=seed, out=out)
    meta = dict(system=system, kind=KIND_TIMELINE, frames=len(X), seed=seed, settings=s.as_record())
    register_local(name, path, meta=meta)
    return path
