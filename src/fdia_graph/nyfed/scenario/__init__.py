"""The operating points: a year of true AC states at five-minute steps.

In: the `Case`, NYgrid's hourly data and NYISO's 2019 zonal loads. The loads are split over the buses from the
bottom up (`LoadModel`), NYgrid's hourly dispatch sets each unit's schedule (`hourly_dispatch`), and `PoolBuilder`
solves a ramp-limited least-cost AC dispatch for every five-minute frame (`AcSolver`, Pyomo with Ipopt, then a
pandapower power flow). Out: the `Pool`, one true state per frame with its solve kind.

Not every frame solves: `PoolBuilder.report()` counts `SOLVED`, `FAILED`, `NO_HOUR` and `NO_LOAD`, and only solved
frames are used downstream. A missing or broken Ipopt raises `SolverUnavailable`. Building the pool is slow (hours
for a year); `tools/nyfed/run_pool.py` builds it in parallel chunks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph.nyfed._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from .dispatch import Dispatch, hourly_dispatch
    from .inputs import ZONES, NygridHour, ZoneLoad, fetch_nyiso, read_nygrid
    from .load_fit import fluctuation_fit, switching_fit
    from .loads import LoadModel, LoadSeries
    from .pool import FAILED, NO_HOUR, NO_LOAD, SOLVED, Pool, PoolBuilder, peak_ratings
    from .solver import AcSolver, Compensation, unit_q_limits

# each public name and the module that defines it, imported on first use
_PUBLIC = {
    "switching_fit": ".load_fit",
    "fluctuation_fit": ".load_fit",
    "ZoneLoad": ".inputs",
    "NygridHour": ".inputs",
    "read_nygrid": ".inputs",
    "fetch_nyiso": ".inputs",
    "ZONES": ".inputs",
    "LoadSeries": ".loads",
    "LoadModel": ".loads",
    "Dispatch": ".dispatch",
    "hourly_dispatch": ".dispatch",
    "AcSolver": ".solver",
    "Compensation": ".solver",
    "unit_q_limits": ".solver",
    "Pool": ".pool",
    "PoolBuilder": ".pool",
    "peak_ratings": ".pool",
    "SOLVED": ".pool",
    "FAILED": ".pool",
    "NO_HOUR": ".pool",
    "NO_LOAD": ".pool",
}

__all__ = [
    "switching_fit",
    "fluctuation_fit",
    "AcSolver",
    "Compensation",
    "Dispatch",
    "FAILED",
    "LoadModel",
    "LoadSeries",
    "NO_HOUR",
    "NO_LOAD",
    "NygridHour",
    "Pool",
    "PoolBuilder",
    "SOLVED",
    "ZONES",
    "ZoneLoad",
    "fetch_nyiso",
    "hourly_dispatch",
    "peak_ratings",
    "read_nygrid",
    "unit_q_limits",
]
__getattr__, __dir__ = attach(__name__, _PUBLIC)
