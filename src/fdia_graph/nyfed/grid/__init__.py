"""The network: the case, its admittances, the NYISO interfaces and the clients.

In: NYgrid's `npcc.mat` and data tables, the utility territories. Out: a `Case` (a frozen record of the buses,
branches and units as arrays), a `Network` built on it (Ybus, branch admittances, adjacency), the `Interface` list
with each hour's real limit and flow, and the `Ownership` that gives every New York bus to one utility (a client).

`read_npcc`, `modify_mpc` and `nygrid_units` reproduce NYgrid's hourly case; `split_zone_j` and `restore_765kv` are
this package's corrections to it (each function's docstring says why). Every step returns a new `Case`; none
changes one in place. A table that disagrees with the case raises a `DataConditionError` naming the bus or branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph.nyfed._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from .case import Case
    from .clients import Ownership, assign_clients
    from .fixes import externals_at_base, hvdc_links, restore_765kv, split_zone_j
    from .interfaces import NYISO_INTERFACES, Interface, hourly_interfaces
    from .network import Network
    from .nygrid import modify_mpc, nygrid_units, read_npcc
    from .ratings import line_ratings

# each public name and the module that defines it, imported on first use
_PUBLIC = {
    "Case": ".case",
    "Network": ".network",
    "read_npcc": ".nygrid",
    "modify_mpc": ".nygrid",
    "nygrid_units": ".nygrid",
    "split_zone_j": ".fixes",
    "restore_765kv": ".fixes",
    "externals_at_base": ".fixes",
    "hvdc_links": ".fixes",
    "line_ratings": ".ratings",
    "Interface": ".interfaces",
    "hourly_interfaces": ".interfaces",
    "NYISO_INTERFACES": ".interfaces",
    "Ownership": ".clients",
    "assign_clients": ".clients",
}

__all__ = [
    "Case",
    "Interface",
    "NYISO_INTERFACES",
    "Network",
    "Ownership",
    "assign_clients",
    "externals_at_base",
    "hourly_interfaces",
    "hvdc_links",
    "line_ratings",
    "modify_mpc",
    "nygrid_units",
    "read_npcc",
    "restore_765kv",
    "split_zone_j",
]
__getattr__, __dir__ = attach(__name__, _PUBLIC)
