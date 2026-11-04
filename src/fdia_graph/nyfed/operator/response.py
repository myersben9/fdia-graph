"""The operator's response, as one module: every name of `day`, `ties`, `truegrid`, `policy`, `run` and
`summary`, re-exported so the imports written before the split keep working. New code imports from those modules
or from `fdia_graph.nyfed.operator`."""

from __future__ import annotations

from fdia_graph._moved import moved
from fdia_graph.nyfed.operator.day import INTERFACES, LBMP_ZONE, OperatorDay, operator_day
from fdia_graph.nyfed.operator.policy import AFTER, ENTRY, MODES, OperatorPolicy, entry_share
from fdia_graph.nyfed.operator.run import (
    QUICK_FUELS,
    QUICK_MAX_MW,
    QUICK_ZONES,
    REG_CAP,
    Response,
    Twin,
    respond,
)
from fdia_graph.nyfed.operator.summary import summarize
from fdia_graph.nyfed.operator.ties import (
    NEIGHBOUR_SEED,
    interchange_alarm,
    neighbour_reads,
    seen_interfaces,
    tie_gap,
    tie_meters,
)
from fdia_graph.nyfed.operator.truegrid import TrueGrid

__all__ = [
    "AFTER",
    "ENTRY",
    "INTERFACES",
    "LBMP_ZONE",
    "MODES",
    "NEIGHBOUR_SEED",
    "QUICK_FUELS",
    "QUICK_MAX_MW",
    "QUICK_ZONES",
    "REG_CAP",
    "OperatorDay",
    "OperatorPolicy",
    "Response",
    "TrueGrid",
    "Twin",
    "entry_share",
    "interchange_alarm",
    "neighbour_reads",
    "operator_day",
    "respond",
    "seen_interfaces",
    "summarize",
    "tie_gap",
    "tie_meters",
]

# Renamed when they became public; the old names work, with a DeprecationWarning, until `_moved.RETIRES`.
_MOVED = {
    "_TrueGrid": ("fdia_graph.nyfed.operator.TrueGrid", TrueGrid),
    "_entry_share": ("fdia_graph.nyfed.operator.entry_share", entry_share),
}


def __getattr__(name: str) -> object:
    return moved(__name__, name, _MOVED)
