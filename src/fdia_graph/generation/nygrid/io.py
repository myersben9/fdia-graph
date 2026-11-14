"""HDF5 files of the build. The only module that imports h5py."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from fdia_graph.engine.attacks.family import Episode
from fdia_graph.engine.sensing.meters import KINDS, MeterPlan
from fdia_graph.federated.utilities import Ownership
from fdia_graph.generation.nygrid.case import Case
from fdia_graph.generation.nygrid.pool import Pool

_POOL_ARRAYS = ("vm", "va", "p", "q", "solve_kind")

SPLITS = ("train", "val", "test")  # the codes 0, 1, 2 of `frames/split`
FAMILIES = (
    "benign",
    "Am",
    "At",
    "Ap",
)  # the codes 0..3 of `frames/family` (Ap appended, so Am, At keep theirs)

# where each per-frame array of a block goes, and its stored type
_FRAME_ARRAYS = {
    "time": ("frames", np.int64),
    "split": ("frames", np.int8),
    "family": ("frames", np.int8),
    "episode": ("frames", np.int32),
    "z": ("frames", np.float32),
    "z_benign": ("frames", np.float32),
    "vm": ("frames", np.float64),
    "va": ("frames", np.float64),
    "bus_attacked": ("labels", bool),
    "meter_tampered": ("labels", bool),
}
_CHUNK = 288  # frames per HDF5 chunk, one day


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


def _storable(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return np.char.encode(a, "utf-8") if a.dtype.kind == "U" else a


def _readable(a: np.ndarray) -> np.ndarray:
    return np.char.decode(a, "utf-8") if a.dtype.kind == "S" else a


class TimelineWriter:
    """An open timeline file (the layout in ARCHITECTURE.md): the static case, ownership and meter plan,
    the per-frame arrays written a block at a time, and the episode table."""

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.h = h5py.File(path, "w")

    def write_static(self, case: Case, own: Ownership, plan: MeterPlan) -> None:
        """`case/` every Case array with the ownership (`bus_client`, `client_names`, `tie_lines`), and
        `meters/` the MeterPlan table."""
        g = self.h.create_group("case")
        for f in fields(Case):
            g.create_dataset(f.name, data=_storable(getattr(case, f.name)))
        g.create_dataset("bus_client", data=own.bus_client)
        g.create_dataset("client_names", data=_storable(own.names))
        g.create_dataset("tie_lines", data=own.tie_lines)
        m = self.h.create_group("meters")
        for f in fields(MeterPlan):
            m.create_dataset(f.name, data=getattr(plan, f.name))
        m.attrs["kind_codes"] = ",".join(KINDS)

    def write_frames(self, t0: int, block: dict[str, np.ndarray]) -> None:
        """The per-frame arrays of frames t0.. (each `[B, ...]`, keys of `_FRAME_ARRAYS`), growing each
        dataset as needed. `time` is datetime64, stored as int64 seconds like the pool."""
        for name, arr in block.items():
            group, dtype = _FRAME_ARRAYS[name]
            arr = np.asarray(arr)
            if name == "time":
                arr = arr.astype("datetime64[s]").astype(np.int64)
            key = f"{group}/{name}"
            if key not in self.h:
                self.h.create_dataset(
                    key,
                    shape=(0,) + arr.shape[1:],
                    maxshape=(None,) + arr.shape[1:],
                    dtype=dtype,
                    chunks=(_CHUNK,) + arr.shape[1:],
                    compression="gzip",
                    shuffle=True,
                )
            d = self.h[key]
            t1 = t0 + len(arr)
            if d.shape[0] < t1:
                d.resize(t1, axis=0)
            d[t0:t1] = arr.astype(dtype)
        self.h["frames"].attrs["split_codes"] = ",".join(SPLITS)
        self.h["frames"].attrs["family_codes"] = ",".join(FAMILIES)

    def write_episodes(self, episodes: list[Episode]) -> None:
        """`episodes/`, one row per episode in `frames/episode` order: `family`, `onset`, `length`,
        `target_branch`, and the ragged `buses` and `clients` as flat arrays with `[S + 1]` offsets
        (`buses_ptr`, `clients_ptr`). The attack vectors are not stored: `z - z_benign` is the attack."""
        g = self.h.create_group("episodes")
        g.create_dataset("family", data=_storable(np.array([e.family for e in episodes], dtype="U2")))
        for name in ("onset", "length", "target_branch"):
            g.create_dataset(name, data=np.array([getattr(e, name) for e in episodes], np.int64))
        for name in ("buses", "clients"):
            parts = [np.asarray(getattr(e, name), np.int64) for e in episodes]
            g.create_dataset(name, data=np.concatenate(parts) if parts else np.zeros(0, np.int64))
            g.create_dataset(
                f"{name}_ptr", data=np.r_[0, np.cumsum([len(p) for p in parts])].astype(np.int64)
            )

    def close(self) -> None:
        self.h.close()


def open_timeline(path: str | Path) -> tuple[h5py.File, Case, Ownership, MeterPlan]:
    """A timeline file opened read only, with its static `Case`, `Ownership` and `MeterPlan`. The
    per-frame arrays stay on disk, read by slices through the returned file."""
    h = h5py.File(path, "r")
    g, m = h["case"], h["meters"]
    # a file written before `Case.ramp` existed reads it as unknown (NaN) for every unit
    columns: dict[str, Any] = {
        f.name: _readable(g[f.name][()]) if f.name in g else np.full(len(g["unit_bus"]), np.nan)
        for f in fields(Case)
    }
    case = Case(**columns)
    own = Ownership(
        names=_readable(g["client_names"][:]), bus_client=g["bus_client"][:], tie_lines=g["tie_lines"][:]
    )
    plan = MeterPlan(**{f.name: m[f.name][:] for f in fields(MeterPlan)})
    return h, case, own, plan
