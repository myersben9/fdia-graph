"""The pool file of the build, HDF5. A timeline is written by the writer every grid uses
(`fdia_graph.generation.write`) and read through `fdia_graph.dataset`."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from fdia_graph.generation.nygrid.pool import Pool

_POOL_ARRAYS = ("vm", "va", "p", "q", "solve_kind")


def write_pool(path: str | Path, pool: Pool) -> None:
    """The pool as one HDF5 file: `time` as int64 seconds of NYISO's local clock (naive, the clock of the
    load data) and the state arrays, compressed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("time", data=pool.time.astype("datetime64[s]").astype(np.int64))
        h["time"].attrs["units"] = "seconds since 1970-01-01, NYISO local clock"
        for name in _POOL_ARRAYS:
            h.create_dataset(name, data=getattr(pool, name), compression="gzip", shuffle=True)
        if pool.unit_p is not None:  # each unit's dispatch basepoint (N32), in the case's unit order
            h.create_dataset("unit_p", data=pool.unit_p, compression="gzip", shuffle=True)


def read_pool(path: str | Path) -> Pool:
    """A pool file; `unit_p` is None for a pool written before N32."""
    with h5py.File(path, "r") as h:
        unit_p = h["unit_p"][:] if "unit_p" in h else None
        return Pool(
            time=h["time"][:].astype("datetime64[s]"),
            unit_p=unit_p,
            **{name: h[name][:] for name in _POOL_ARRAYS},
        )
