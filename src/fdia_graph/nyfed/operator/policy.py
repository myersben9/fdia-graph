"""The operator's choices: which operator (`MODES`), how it answers an alarm (`AFTER`), and where the
inadvertent interchange its neighbours supply enters (`ENTRY`, `entry_share`)."""

from __future__ import annotations

import numpy as np

from fdia_graph.errors import ConfigError
from fdia_graph.nyfed.grid.case import Case
from fdia_graph.nyfed.grid.nygrid import HQ_BUS

MODES = ("trusting", "accounting", "oracle")
# N78: how an operator answers an alarm: it reads the scans without the attack, or holds its dispatch with its
# AGC on the tie meters it reads or on the common-source tie values
AFTER = ("benign", "freeze", "freeze_cs")


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
