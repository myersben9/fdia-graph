"""The grid operator under attack: what it does with false readings, and what that costs on the true grid.

The work is split by job: `day` reads an attacked day, `ties` holds the interchange check, `truegrid` the AC
power flows on the true grid, `policy` the operator's choices, `run` the scan-by-scan loop and `summary` its
consequence (`response` re-exports them all).

`operator_day` reads an attacked day (a timeline file) with its benign twin. `respond` runs the operator scan by
scan under an `OperatorPolicy` (which operator, how late, how it answers an alarm): its five-minute security-
constrained dispatch on what it reads (`DcSced`), its regulation and the interchange its neighbors supply, each
checked on the true grid by an AC power flow (`TrueGrid`). It can answer an alarm (`AFTER`) and take the inadvertent
interchange where it enters (`ENTRY`). `summarize` gives the attack's consequence, attacked less benign.

A scan whose AC power flow fails is flagged (`Twin.pf_failed`) and counted, never read as within the limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph.nyfed._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from .day import OperatorDay, operator_day
    from .policy import AFTER, ENTRY, MODES, AlarmAnswer, Entry, OperatorMode, OperatorPolicy, entry_share
    from .run import NyisoOperator, Operator, Response, Twin, respond
    from .sced import DcSced, Step, outage_factors, shift_factors, shortage_usd
    from .summary import summarize
    from .ties import interchange_alarm, seen_interfaces, tie_gap, tie_meters
    from .truegrid import TrueGrid

# each public name and the module that defines it, imported on first use
_PUBLIC = {
    "NyisoOperator": ".run",
    "Operator": ".run",
    "Entry": ".policy",
    "AlarmAnswer": ".policy",
    "OperatorMode": ".policy",
    "OperatorPolicy": ".policy",
    "DcSced": ".sced",
    "Step": ".sced",
    "shift_factors": ".sced",
    "outage_factors": ".sced",
    "shortage_usd": ".sced",
    "OperatorDay": ".day",
    "operator_day": ".day",
    "seen_interfaces": ".ties",
    "tie_gap": ".ties",
    "tie_meters": ".ties",
    "interchange_alarm": ".ties",
    "TrueGrid": ".truegrid",
    "MODES": ".policy",
    "AFTER": ".policy",
    "ENTRY": ".policy",
    "entry_share": ".policy",
    "Twin": ".run",
    "Response": ".run",
    "respond": ".run",
    "summarize": ".summary",
}

__all__ = [
    "AFTER",
    "AlarmAnswer",
    "DcSced",
    "ENTRY",
    "Entry",
    "MODES",
    "NyisoOperator",
    "Operator",
    "OperatorDay",
    "OperatorMode",
    "OperatorPolicy",
    "Response",
    "Step",
    "TrueGrid",
    "Twin",
    "entry_share",
    "interchange_alarm",
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
