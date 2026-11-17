"""One scan of the timeline writer: a benign scan, or an attacked At scan built by `engine.attacks`.

Every attack is built in `engine.attacks` (the `AttackMixin` of FdiaGenerator); this module emits
a benign scan and hands an attacked ramp frame to the mixin.

RNG-order invariant: a file is reproduced bit for bit from its seed (tests/test_split_first.py
writes one twice and compares), so the order of random draws is fixed: one emission per scan, the
true one; a stealthy frame adds its attack vector to it without a draw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..models.choices import BENIGN_CODE
from ..models.frames import AttackDesign, Frame, FrameKnobs

if TYPE_CHECKING:
    from .core import FdiaGenerator


def attack_frame(
    g: FdiaGenerator, Xt: np.ndarray, family: int, design: Optional[AttackDesign], knobs: FrameKnobs
) -> Optional[Frame]:
    """Build the scan of `family` on the stored operating point `Xt` ([N, 4] = |V|, P_inj, Q_inj, theta):
    the benign scan here, an At ramp frame from `engine.attacks` (`AttackMixin.attack_frame`, where
    `design` and the reason a scan is rejected are described)."""
    if family == BENIGN_CODE:
        return _benign_frame(g, Xt)
    assert design is not None, f"family {family} needs an attack design"
    return g.attack_frame(Xt, family, design, knobs)


def _benign_frame(g: FdiaGenerator, Xt: np.ndarray) -> Frame:
    """A benign scan: the stored state emitted through the meter plan."""
    scan = g.emit_from_state(Xt)
    empty = np.zeros(0, int)
    return Frame(
        scan.node_x,
        scan.node_m,
        scan.edge_x,
        scan.edge_m,
        np.zeros(g.C, np.uint8),
        0,
        empty,
        np.zeros(0, float),
        None,
        None,
        i_x=scan.i_x,
        i_m=scan.i_m,
    )
