"""An attacked day as the operator meets it: the scans it reads and their benign twin, the true states, the
pool's unit dispatch and costs, the zonal prices, and the interfaces with their real limits (`operator_day`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fdia_graph.errors import DataConditionError
from fdia_graph.nyfed.grid.case import Case
from fdia_graph.nyfed.grid.clients import Ownership
from fdia_graph.nyfed.grid.interfaces import Interface, hourly_interfaces
from fdia_graph.nyfed.io import open_timeline, read_pool
from fdia_graph.nyfed.scenario.dispatch import hourly_dispatch
from fdia_graph.nyfed.scenario.inputs import ZONES, read_nygrid
from fdia_graph.nyfed.scenario.pool import Pool
from fdia_graph.nyfed.sensing.meters import MeterPlan

# NYISO's zone names in priceHourly, by zone letter
LBMP_ZONE = {
    "A": "WEST",
    "B": "GENESE",
    "C": "CENTRL",
    "D": "NORTH",
    "E": "MHK VL",
    "F": "CAPITL",
    "G": "HUD VL",
    "H": "MILLWD",
    "I": "DUNWOD",
    "J": "N.Y.C.",
    "K": "LONGIL",
}


@dataclass(frozen=True)
class OperatorDay:
    """One timeline file's frames with what the operator needs: the true state `vm`, `va [T, N]`, the
    attacked and benign scans `z`, `z_benign [T, M]`, the pool's basepoint `unit_p [T, G]`, the units'
    cost `cost [T, G]` ($/MWh, NYgrid's hourly costs interpolated on the frame clock as the pool build
    does), the real zonal LBMP `lbmp [T, 11]` ($/MWh, zones A to K), the `interfaces` the operator
    enforces, `solved [T]`, and the file's first episode (`onset`, `length`)."""

    case: Case
    own: Ownership
    plan: MeterPlan
    time: np.ndarray
    vm: np.ndarray
    va: np.ndarray
    z: np.ndarray
    z_benign: np.ndarray
    unit_p: np.ndarray
    cost: np.ndarray
    lbmp: np.ndarray
    interfaces: list[Interface]
    solved: np.ndarray
    onset: int
    length: int


# the interfaces the operator enforces, the attacked one first (N64)
INTERFACES = ("SPR/DUN-SOUTH", "UPNY CONED", "CENTRAL EAST - VC", "MOSES SOUTH")
_POOLS: dict[str, Pool] = {}  # pools and NYgrid's hourly tables read once per process: a study runs many days
_HOURS: dict[str, list] = {}


def _hour_weights(time: np.ndarray, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each frame the hour before, the hour after and the weight of the hour after, on the hours'
    midpoints (as `PoolBuilder._brackets`)."""
    mid = (hours.astype("datetime64[s]") + np.timedelta64(1800, "s")).astype(np.int64).astype(float)
    pos = np.interp(time.astype("datetime64[s]").astype(np.int64).astype(float), mid, np.arange(len(mid)))
    lo = np.floor(pos).astype(int)
    return lo, np.minimum(lo + 1, len(mid) - 1), pos - lo


def node_lbmp(data_dir: str | Path, time: np.ndarray, nodes: list[str], year: int = 2019) -> np.ndarray:
    """`[T, len(nodes)]` the real hourly LBMP ($/MWh) of each frame's hour at each of NYISO's price `nodes`, as
    `priceHourly` names them: the zones (`LBMP_ZONE`) and the proxy buses of the neighbors (`PJM`, `NPX` for ISO
    New England, `O H` for Ontario, `H Q`)."""
    price = pd.read_csv(Path(data_dir) / f"priceHourly_{year}.csv")
    price["t"] = pd.to_datetime(price.TimeStamp, format="%m/%d/%Y %H:%M:%S")
    table = price.pivot_table(index="t", columns="ZoneName", values="LBMP")
    hour = pd.DatetimeIndex(time.astype("datetime64[s]")).floor("h")
    return table.reindex(index=hour, columns=nodes).to_numpy(float)


def _zonal_lbmp(data_dir: Path, time: np.ndarray, year: int = 2019) -> np.ndarray:
    """`[T, 11]` the real hourly zonal LBMP ($/MWh) of each frame's hour, zones A to K."""
    return node_lbmp(data_dir, time, [LBMP_ZONE[z] for z in ZONES], year)


def _target_first(interfaces: list[Interface], branch: int) -> list[Interface]:
    """The interfaces with the one holding the attack's target branch first (the summary reads it as the
    attacked interface), every one still enforced."""
    k = next((i for i, it in enumerate(interfaces) if branch in set(it.branches.tolist())), 0)
    return [interfaces[k]] + interfaces[:k] + interfaces[k + 1 :]


def operator_day(
    timeline: str | Path,
    pool: str | Path,
    data_dir: str | Path,
    interfaces: tuple[str, ...] = INTERFACES,
) -> OperatorDay:
    """An `OperatorDay` from a timeline file (a single-episode file, or any timeline), the pool it was
    built from and NYgrid's `Data/` folder."""
    data_dir = Path(data_dir)
    h, case, own, plan = open_timeline(timeline)
    time = h["frames/time"][:]
    stamps = time.astype("datetime64[s]")
    if str(pool) not in _POOLS:
        _POOLS[str(pool)] = read_pool(pool)
    p = _POOLS[str(pool)]
    # a frame after the pool's last one sorts past its end, so clip to compare it with the last frame
    k = np.minimum(np.searchsorted(p.time.astype("datetime64[s]"), stamps), len(p.time) - 1)
    if not (p.time.astype("datetime64[s]")[k] == stamps).all() or p.unit_p is None:
        raise DataConditionError("the pool does not hold every frame of the timeline with its unit dispatch")
    if str(data_dir) not in _HOURS:
        _HOURS[str(data_dir)] = read_nygrid(data_dir, 2019)
    hours = _HOURS[str(data_dir)]
    when = np.array([x.time for x in hours]).astype("datetime64[s]")
    near = np.flatnonzero(
        (when >= stamps[0] - np.timedelta64(3600, "s")) & (when <= stamps[-1] + np.timedelta64(3600, "s"))
    )
    disp = hourly_dispatch(case, [hours[i] for i in near])
    lo, hi, w = _hour_weights(stamps, disp.hour)
    cost = (1 - w)[:, None] * disp.unit_cost[lo] + w[:, None] * disp.unit_cost[hi]
    vm, va = h["frames/vm"][:], h["frames/va"][:]
    target = int(h["episodes/target_branch"][0]) if len(h["episodes/target_branch"]) else -1
    eps = h["episodes"]
    day = OperatorDay(
        case=case,
        own=own,
        plan=plan,
        time=time,
        vm=vm,
        va=va,
        z=h["frames/z"][:].astype(float),
        z_benign=h["frames/z_benign"][:].astype(float),
        unit_p=p.unit_p[k],
        cost=cost,
        lbmp=_zonal_lbmp(data_dir, time),
        interfaces=_target_first(hourly_interfaces(case, time, data_dir, list(interfaces)), target),
        solved=np.isfinite(vm).all(axis=1) & np.isfinite(p.unit_p[k]).all(axis=1),
        onset=int(eps["onset"][0]) if len(eps["onset"]) else -1,
        length=int(eps["length"][0]) if len(eps["length"]) else 0,
    )
    h.close()
    return day
