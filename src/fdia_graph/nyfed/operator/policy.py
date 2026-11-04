"""The operator's choices as one record (`OperatorPolicy`): which operator (`OperatorMode`), how late it acts, where
the inadvertent interchange its neighbours supply enters (`Entry`, `entry_share`), whether it reads the true flows
after its own dispatch, and how it answers an alarm (`AlarmAnswer`). The record checks every choice when it is built,
so a typo is a `ConfigError` before anything runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Optional

import numpy as np

from fdia_graph.errors import ConfigError
from fdia_graph.models.choices import Choice
from fdia_graph.models.validation import AtLeast, Boolean, Finite, Integer, OneOf, Validated
from fdia_graph.nyfed.grid.case import Case
from fdia_graph.nyfed.grid.nygrid import HQ_BUS


class OperatorMode(Choice):
    """Which operator (N66): one that trusts its telemetry, one that keeps its load and control error on the
    common-source tie values, or an oracle that falls back to the benign scans at the interchange alarm."""

    TRUSTING = "trusting"
    ACCOUNTING = "accounting"
    ORACLE = "oracle"


class AlarmAnswer(Choice):
    """How an operator answers an alarm (N78): it reads the scans without the attack, or holds its dispatch with its
    AGC on the tie meters it reads or on the common-source tie values."""

    BENIGN = "benign"
    FREEZE = "freeze"
    FREEZE_CS = "freeze_cs"


class Entry(Choice):
    """Where the inadvertent interchange enters New York (N72), the keys of `ENTRY`."""

    CAPACITY = "capacity"
    PEAK = "peak"
    PJM = "pjm"
    ISONE = "isone"
    WEST = "west"


class Dispatch(Choice):
    """How the operator dispatches (`DcSced` objective): the least generation moved, then the least cost (N53), or
    the least cost a market clears, which also prices each interface's congestion."""

    MOVEMENT = "movement"
    COST = "cost"


MODES = tuple(OperatorMode.values())
AFTER = tuple(AlarmAnswer.values())


# N72: where the inadvertent interchange enters New York, as weights per external area of the source model (1
# ISO-NE, 4 Ontario, 5 MISO, 6 PJM), split within an area by the units' capacity. "capacity" weights every
# synchronous external unit by its capacity alone (N69); "peak" weights each area by its approximate 2019 peak
# load in GW, a proxy for its frequency bias; the others put all of it in one region.
ENTRY: dict[str, dict[int, float] | None] = {
    "capacity": None,
    "peak": {1: 24.4, 4: 21.8, 5: 121.6, 6: 151.3},
    "pjm": {6: 1.0},
    "isone": {1: 1.0},
    "west": {4: 1.0, 5: 1.0, 6: 1.0},
}


def entry_share(c: Case, entry: str) -> np.ndarray:
    """[U] the share of the inadvertent interchange each unit supplies (N69, N72): synchronous units outside
    New York (the HVDC import from Quebec cannot carry it), by capacity within the weights of `ENTRY[entry]`."""
    if entry not in ENTRY:
        raise ConfigError(f"entry must be one of {tuple(ENTRY)}, not {entry!r}")
    cap = np.nan_to_num(c.pmax)
    ext = ~c.is_ny[c.unit_bus] & (c.bus_id[c.unit_bus] != HQ_BUS) & (cap > 0)
    share = np.where(ext, cap, 0.0)
    weights = ENTRY[entry]
    if weights is not None:
        area = c.area[c.unit_bus]
        out = np.zeros_like(share)
        for a, w in weights.items():
            inside = share * (area == a)
            if inside.sum() > 0:
                out += w * inside / inside.sum()
        share = out
    return share / share.sum() if share.sum() > 0 else share


@dataclass(frozen=True)
class OperatorPolicy(Validated):
    """How the operator runs: `mode` which operator, `delay` the scans it acts late (O5), `over` (MW) a what-if push
    of the attacked reading past what the attack reached, `entry` where the inadvertent interchange enters,
    `closed_loop` whether it reads each interface as it truly flows after its own last dispatch (N76), `after`
    how it answers an alarm (None: it does not), and `dispatch` its objective (`Dispatch`)."""

    mode: Annotated[str, OneOf(OperatorMode)] = "trusting"
    delay: Annotated[int, Integer(), AtLeast(0)] = 0
    over: Annotated[float, Finite()] = 0.0
    entry: Annotated[str, OneOf(Entry)] = "capacity"
    closed_loop: Annotated[bool, Boolean()] = False
    after: Annotated[Optional[str], OneOf(AlarmAnswer)] = None
    dispatch: Annotated[str, OneOf(Dispatch)] = "movement"
