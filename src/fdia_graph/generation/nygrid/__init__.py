"""The New York system: NYgrid's NPCC-140 case made solvable in AC, its reduced network, the 2019 loads by bus,
the hourly dispatch and the five-minute pool of AC operating states, and the timeline written from them.

It is one system source of the generator beside the pandapower cases: `production_builder` builds the
`TimelineBuilder` of a release from a pool, `PoolBuilder` the pool from the inputs `fetch_inputs` downloads. The
attack families it uses live in `fdia_graph.engine.attacks`, the meters in `fdia_graph.engine.sensing`, the
operator in `fdia_graph.operator`, the residual test in `fdia_graph.se.detect`, the utility clients in
`fdia_graph.federated.utilities` and the study helpers in `fdia_graph.results.studies`. Every public name below
loads on first use, so reading a timeline needs neither pandapower nor Pyomo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from fdia_graph.dataset.clients import ClientView, Dataset
    from fdia_graph.engine.solvers import ipopt
    from fdia_graph.federated.utilities import Ownership, assign_clients
    from fdia_graph.generation.nygrid.build import fetch_inputs, production_builder
    from fdia_graph.generation.nygrid.case import Case
    from fdia_graph.generation.nygrid.config import Settings, rng
    from fdia_graph.generation.nygrid.dispatch import Dispatch, hourly_dispatch
    from fdia_graph.generation.nygrid.fixes import externals_at_base, hvdc_links, restore_765kv, split_zone_j
    from fdia_graph.generation.nygrid.inputs import ZONES, NygridHour, ZoneLoad, fetch_nyiso, read_nygrid
    from fdia_graph.generation.nygrid.interfaces import NYISO_INTERFACES, Interface, hourly_interfaces
    from fdia_graph.generation.nygrid.io import (
        FAMILIES,
        SPLITS,
        TimelineWriter,
        open_timeline,
        read_pool,
        write_pool,
    )
    from fdia_graph.generation.nygrid.load_fit import fluctuation_fit, switching_fit
    from fdia_graph.generation.nygrid.loads import LoadModel, LoadSeries
    from fdia_graph.generation.nygrid.network import Network
    from fdia_graph.generation.nygrid.npcc import modify_mpc, nygrid_units, read_npcc
    from fdia_graph.generation.nygrid.pool import (
        FAILED,
        NO_HOUR,
        NO_LOAD,
        SOLVED,
        Pool,
        PoolBuilder,
        peak_ratings,
    )
    from fdia_graph.generation.nygrid.ratings import line_ratings
    from fdia_graph.generation.nygrid.schedule import (
        Slot,
        place_episodes,
        split_frames,
        split_weeks,
        week_index,
    )
    from fdia_graph.generation.nygrid.solver import AcSolver, Compensation, unit_q_limits
    from fdia_graph.generation.nygrid.timeline import TimelineBuilder
    from fdia_graph.operator.response import operator_day, respond, summarize

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
    "Network": "fdia_graph.generation.nygrid.network",
    "NygridHour": "fdia_graph.generation.nygrid.inputs",
    "Ownership": "fdia_graph.federated.utilities",
    "Pool": "fdia_graph.generation.nygrid.pool",
    "PoolBuilder": "fdia_graph.generation.nygrid.pool",
    "SOLVED": "fdia_graph.generation.nygrid.pool",
    "SPLITS": "fdia_graph.generation.nygrid.io",
    "Settings": "fdia_graph.generation.nygrid.config",
    "Slot": "fdia_graph.generation.nygrid.schedule",
    "TimelineBuilder": "fdia_graph.generation.nygrid.timeline",
    "TimelineWriter": "fdia_graph.generation.nygrid.io",
    "ZONES": "fdia_graph.generation.nygrid.inputs",
    "ZoneLoad": "fdia_graph.generation.nygrid.inputs",
    "assign_clients": "fdia_graph.federated.utilities",
    "externals_at_base": "fdia_graph.generation.nygrid.fixes",
    "fetch_inputs": "fdia_graph.generation.nygrid.build",
    "fetch_nyiso": "fdia_graph.generation.nygrid.inputs",
    "fluctuation_fit": "fdia_graph.generation.nygrid.load_fit",
    "hourly_dispatch": "fdia_graph.generation.nygrid.dispatch",
    "hourly_interfaces": "fdia_graph.generation.nygrid.interfaces",
    "hvdc_links": "fdia_graph.generation.nygrid.fixes",
    "ipopt": "fdia_graph.engine.solvers",
    "line_ratings": "fdia_graph.generation.nygrid.ratings",
    "modify_mpc": "fdia_graph.generation.nygrid.npcc",
    "nygrid_units": "fdia_graph.generation.nygrid.npcc",
    "open_timeline": "fdia_graph.generation.nygrid.io",
    "operator_day": "fdia_graph.operator.response",
    "peak_ratings": "fdia_graph.generation.nygrid.pool",
    "place_episodes": "fdia_graph.generation.nygrid.schedule",
    "production_builder": "fdia_graph.generation.nygrid.build",
    "read_npcc": "fdia_graph.generation.nygrid.npcc",
    "read_nygrid": "fdia_graph.generation.nygrid.inputs",
    "read_pool": "fdia_graph.generation.nygrid.io",
    "respond": "fdia_graph.operator.response",
    "restore_765kv": "fdia_graph.generation.nygrid.fixes",
    "rng": "fdia_graph.generation.nygrid.config",
    "split_frames": "fdia_graph.generation.nygrid.schedule",
    "split_weeks": "fdia_graph.generation.nygrid.schedule",
    "split_zone_j": "fdia_graph.generation.nygrid.fixes",
    "summarize": "fdia_graph.operator.response",
    "switching_fit": "fdia_graph.generation.nygrid.load_fit",
    "unit_q_limits": "fdia_graph.generation.nygrid.solver",
    "week_index": "fdia_graph.generation.nygrid.schedule",
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
    "NygridHour",
    "Ownership",
    "Pool",
    "PoolBuilder",
    "SOLVED",
    "SPLITS",
    "Settings",
    "Slot",
    "TimelineBuilder",
    "TimelineWriter",
    "ZONES",
    "ZoneLoad",
    "assign_clients",
    "externals_at_base",
    "fetch_inputs",
    "fetch_nyiso",
    "fluctuation_fit",
    "hourly_dispatch",
    "hourly_interfaces",
    "hvdc_links",
    "ipopt",
    "line_ratings",
    "modify_mpc",
    "nygrid_units",
    "open_timeline",
    "operator_day",
    "peak_ratings",
    "place_episodes",
    "production_builder",
    "read_npcc",
    "read_nygrid",
    "read_pool",
    "respond",
    "restore_765kv",
    "rng",
    "split_frames",
    "split_weeks",
    "split_zone_j",
    "summarize",
    "switching_fit",
    "unit_q_limits",
    "week_index",
    "write_pool",
]

__getattr__, __dir__ = attach(__name__, _PUBLIC)
