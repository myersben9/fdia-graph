"""The grid operator under attack: what it does with false readings, and what that costs on the true grid.

`operator_day` reads an attacked day (a timeline file) with its benign twin. `respond` runs the operator scan by
scan in one of `MODES`: its five-minute security-constrained dispatch on what it reads (`DcSced`), its regulation
and the interchange its neighbors supply, each checked on the true grid by an AC power flow (`TrueGrid`). It can
answer an alarm (`AFTER`) and take the inadvertent interchange where it enters (`ENTRY`). `summarize` gives the
attack's consequence, attacked less benign.

A scan whose AC power flow fails is flagged (`Twin.pf_failed`) and counted, never read as within the limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph.nyfed._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from .response import (
        AFTER,
        ENTRY,
        MODES,
        OperatorDay,
        Response,
        TrueGrid,
        Twin,
        entry_share,
        operator_day,
        respond,
        seen_interfaces,
        summarize,
        tie_gap,
        tie_meters,
    )
    from .sced import DcSced, Step, outage_factors, shift_factors, shortage_usd

# each public name and the module that defines it, imported on first use
_PUBLIC = {
    "DcSced": ".sced",
    "Step": ".sced",
    "shift_factors": ".sced",
    "outage_factors": ".sced",
    "shortage_usd": ".sced",
    "OperatorDay": ".response",
    "Twin": ".response",
    "Response": ".response",
    "TrueGrid": ".response",
    "operator_day": ".response",
    "respond": ".response",
    "summarize": ".response",
    "seen_interfaces": ".response",
    "tie_gap": ".response",
    "tie_meters": ".response",
    "entry_share": ".response",
    "MODES": ".response",
    "AFTER": ".response",
    "ENTRY": ".response",
}

__all__ = [
    "AFTER",
    "DcSced",
    "ENTRY",
    "MODES",
    "OperatorDay",
    "Response",
    "Step",
    "TrueGrid",
    "Twin",
    "entry_share",
    "operator_day",
    "outage_factors",
    "respond",
    "seen_interfaces",
    "shift_factors",
    "shortage_usd",
    "summarize",
    "tie_gap",
    "tie_meters",
]
__getattr__, __dir__ = attach(__name__, _PUBLIC)
