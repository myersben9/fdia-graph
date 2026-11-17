"""The New York system: NYgrid's NPCC-140 case made solvable in AC, its reduced network, the 2019 loads by bus,
the hourly dispatch and the five-minute pool of AC operating states, and the timeline written from them.

It is one of the two grids a timeline is generated on, beside the IEEE cases: `NewYorkSource` is its
`generation.source.TimelineSource`: `source.generate(path)` writes its timeline by the path every timeline
takes (`NewYorkSource.production` builds the source of a release from a pool, `SingleEpisodeSource` the one of a
single episode), and `PoolBuilder` builds the pool from the inputs `fetch_inputs` downloads. The
attack families it uses live in `fdia_graph.engine.attacks`, the meters in `fdia_graph.engine.sensing`, the
alarm rules in `fdia_analysis.se.detect` and the utility clients in `fdia_graph.generation.nygrid.ownership`. Every public
name below loads on first use, so reading a timeline needs neither pandapower nor Pyomo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from fdia_graph.dataset.clients import ClientView, Dataset
    from fdia_graph.engine.solvers import ipopt
    from fdia_graph.generation.nygrid.case import ZONES, Case, nygrid_case
    from fdia_graph.generation.nygrid.config import Settings, rng
    from fdia_graph.generation.nygrid.dispatch import Dispatch, hourly_dispatch
    from fdia_graph.generation.nygrid.fixes import externals_at_base, hvdc_links
    from fdia_graph.generation.nygrid.inputs import (
        NygridHour,
        ZoneLoad,
        fetch_inputs,
        fetch_nyiso,
        read_nygrid,
    )
    from fdia_graph.generation.nygrid.interfaces import NYISO_INTERFACES, Interface, hourly_interfaces
    from fdia_graph.generation.nygrid.io import (
        FAMILIES,
        TimelineWriter,
        open_timeline,
        read_pool,
        write_pool,
    )
    from fdia_graph.generation.nygrid.loads import LoadModel, LoadSeries
    from fdia_graph.generation.nygrid.network import Network
    from fdia_graph.generation.nygrid.ownership import Ownership, assign_clients
    from fdia_graph.generation.nygrid.pool import (
        FAILED,
        NO_HOUR,
        NO_LOAD,
        SOLVED,
        Pool,
        PoolBuilder,
        peak_ratings,
    )
    from fdia_graph.generation.nygrid.solver import AcSolver, Compensation
    from fdia_graph.generation.nygrid.source import NewYorkSource, SingleEpisodeSource, Survey

# each public name and the module that defines it, imported on first use
_PUBLIC = {
    "AcSolver": "fdia_graph.generation.nygrid.solver",
    "Case": "fdia_graph.generation.nygrid.case",
    "ClientView": "fdia_graph.dataset.clients",
    "Compensation": "fdia_graph.generation.nygrid.solver",
    "Dataset": "fdia_graph.dataset.clients",
    "Dispatch": "fdia_graph.generation.nygrid.dispatch",
    "FAILED": "fdia_graph.generation.nygrid.pool",
    "FAMILIES": "fdia_graph.generation.nygrid.io",
    "Interface": "fdia_graph.generation.nygrid.interfaces",
    "LoadModel": "fdia_graph.generation.nygrid.loads",
    "LoadSeries": "fdia_graph.generation.nygrid.loads",
    "NO_HOUR": "fdia_graph.generation.nygrid.pool",
    "NO_LOAD": "fdia_graph.generation.nygrid.pool",
    "NYISO_INTERFACES": "fdia_graph.generation.nygrid.interfaces",
    "NewYorkSource": "fdia_graph.generation.nygrid.source",
    "Network": "fdia_graph.generation.nygrid.network",
    "NygridHour": "fdia_graph.generation.nygrid.inputs",
    "Ownership": "fdia_graph.generation.nygrid.ownership",
    "Pool": "fdia_graph.generation.nygrid.pool",
    "PoolBuilder": "fdia_graph.generation.nygrid.pool",
    "SOLVED": "fdia_graph.generation.nygrid.pool",
    "Settings": "fdia_graph.generation.nygrid.config",
    "SingleEpisodeSource": "fdia_graph.generation.nygrid.source",
    "Survey": "fdia_graph.generation.nygrid.source",
    "TimelineWriter": "fdia_graph.generation.nygrid.io",
    "ZONES": "fdia_graph.generation.nygrid.case",
    "ZoneLoad": "fdia_graph.generation.nygrid.inputs",
    "assign_clients": "fdia_graph.generation.nygrid.ownership",
    "externals_at_base": "fdia_graph.generation.nygrid.fixes",
    "fetch_inputs": "fdia_graph.generation.nygrid.inputs",
    "fetch_nyiso": "fdia_graph.generation.nygrid.inputs",
    "hourly_dispatch": "fdia_graph.generation.nygrid.dispatch",
    "hourly_interfaces": "fdia_graph.generation.nygrid.interfaces",
    "hvdc_links": "fdia_graph.generation.nygrid.fixes",
    "ipopt": "fdia_graph.engine.solvers",
    "nygrid_case": "fdia_graph.generation.nygrid.case",
    "open_timeline": "fdia_graph.generation.nygrid.io",
    "peak_ratings": "fdia_graph.generation.nygrid.pool",
    "read_nygrid": "fdia_graph.generation.nygrid.inputs",
    "read_pool": "fdia_graph.generation.nygrid.io",
    "rng": "fdia_graph.generation.nygrid.config",
    "write_pool": "fdia_graph.generation.nygrid.io",
}

__all__ = [
    "AcSolver",
    "Case",
    "ClientView",
    "Compensation",
    "Dataset",
    "Dispatch",
    "FAILED",
    "FAMILIES",
    "Interface",
    "LoadModel",
    "LoadSeries",
    "NO_HOUR",
    "NO_LOAD",
    "NYISO_INTERFACES",
    "Network",
    "NewYorkSource",
    "NygridHour",
    "Ownership",
    "Pool",
    "PoolBuilder",
    "SOLVED",
    "Settings",
    "SingleEpisodeSource",
    "Survey",
    "TimelineWriter",
    "ZONES",
    "ZoneLoad",
    "assign_clients",
    "externals_at_base",
    "fetch_inputs",
    "fetch_nyiso",
    "hourly_dispatch",
    "hourly_interfaces",
    "hvdc_links",
    "ipopt",
    "nygrid_case",
    "open_timeline",
    "peak_ratings",
    "read_nygrid",
    "read_pool",
    "rng",
    "write_pool",
]

__getattr__, __dir__ = attach(__name__, _PUBLIC)
