"""generate(system, name, **knobs) — build a custom dataset and register it as `name`.

`generate` writes one timeline file (`fdia_graph.timeline`): every frame of the operating-point
pool scanned in time order with At and Am attack episodes. `fg.load(name)` reads it as a record
table (`order="random"`) or as the timeline (`order="time"`). This module also reads the pool
(`_load_states`): a .npz or .h5 file with key 'X' [T, N, 4], or a directory of X_*.npy files.

The one path that generates a timeline is `TimelineSource.generate` (`generation/source.py`). Its
subclass is `SearchSource` (`generation/search.py`; `IeeeSource` in `generation/ieee.py` is it for
an IEEE case by name).
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
    """A state pool [T, N, 4] in the column order [|V|, Pinj, Qinj, theta]. The downloadable pools are
    [Pinj, Qinj, |V|, theta]; the order is detected from the data, since |V| is the only column whose
    every value sits near 1.0 pu. A pool in neither order raises UnknownColumnOrder."""
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
    """The operating-point pool [T, N, 4] = [|V|, Pinj, Qinj, theta]: `states` if given, else
    $FDIA_GRAPH_INIT, else the downloaded pool. A directory of per-timestep files is sampled at an
    even stride, at most `pool_cap` files, so the load stays fast and keeps the daily variety."""
    return as_v_first(_read_states(system, states, pool_cap))


def _read_states(
    system: Union[int, str], states: Optional[Union[str, np.ndarray]], pool_cap: int
) -> np.ndarray:
    source = StateSource(states)
    if source.array is not None:  # an in-memory pool is used as it is
        return source.array
    src = source.path or os.environ.get("FDIA_GRAPH_INIT")
    if src and os.path.isdir(src):
        # sorted by integer timestep (name "X_<t>.npy")
        xs = sorted(glob.glob(os.path.join(src, "X_*.npy")), key=lambda p: int(os.path.basename(p)[2:-4]))
        stride = max(1, len(xs) // pool_cap)  # even stride -> diverse sample, bounded cost
        # float64 for the AC solve downstream
        return np.stack([np.load(f) for f in xs[::stride][:pool_cap]]).astype(np.float64)
    if src and src.endswith(".npz"):
        return np.load(src)["X"].astype(np.float64)
    if src and src.endswith((".h5", ".hdf5")):
        with h5py.File(src, "r") as f:
            return np.asarray(f["X"], np.float64)
    # fall back to the downloadable pool of this system, at the pinned data release
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
    time: Optional[np.ndarray] = None,
    **knobs: Any,
) -> str:
    """Walk one attacked timeline over the operating-point pool of `system`, write it as one HDF5
    file, register it as `name` and return the path. `frames` caps the pool timesteps walked; `time`
    [T] is each frame's time, for a pool that has it (a split by weeks reads it); the other knobs
    are those of `timeline.generate_timeline` (docs/guides/generation.md lists them)."""
    from ..timeline import generate_timeline

    frames = ShardRun(frames).frames
    s = TimelineSettings.of(settings, **knobs)
    X = _load_states(system, states)
    if frames is not None:
        X, time = X[:frames], None if time is None else time[:frames]
    out = out or os.path.join(CACHE_DIR, f"{name}.h5")
    path = generate_timeline(system, states=X, settings=s, seed=seed, out=out, time=time)
    meta = dict(system=system, kind=KIND_TIMELINE, frames=len(X), seed=seed, settings=s.as_record())
    register_local(name, path, meta=meta)
    return path
