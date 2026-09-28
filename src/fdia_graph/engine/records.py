"""One scan of the timeline writer: a benign scan, or an attacked one built by `engine.attacks`.

Every attack is built in `engine.attacks` (the `AttackMixin` of FdiaGenerator); this module emits
a benign scan and hands an attacked family to the mixin. Two switches in `FrameKnobs` remain from
the record-shard writer that shared this code until 0.18: `reject_below_floor` (reject a stealthy
scan whose designed change sits inside the noise floor; the timeline keeps every scan and lets the
label say what happened) and `with_benign` (also emit the un-attacked measurement of the same scan,
the timeline's `benign` layer).

RNG-order invariant: every released file is reproduced bit for bit from its seed, so the order of
random draws is fixed: one emission per scan, the true one; a stealthy family adds its attack
vector to it without a draw, and the corrupt-in-place families follow the emission with the
replay-lag draw, then the corruption draws. Changing that order changes every released file.
tests/test_frozen.py holds the line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .._moved import moved
from ..models.choices import BENIGN_CODE
from ..models.frames import (  # noqa: F401  re-exported: defined here before the models package
    AttackDesign,
    Frame,
    FrameKnobs,
    Scan,
)
from .attacks.corrupt import (  # noqa: F401  re-exported: defined here before engine.attacks
    BENIGN_BUFFER,
    CORRUPT_KIND,
    REPLAY_MIN_LAG,
    CorruptMixin,
    replay_frame,
)
from .attacks.false_state import FalseStateMixin, changed_meters
from .attacks.stealthy import (  # noqa: F401  re-exported: defined here before engine.attacks
    AM_FAMILY,
    AQ_FAMILY,
    AQ_HALVINGS,
    LRA_DRAWS,
    LRA_FAMILY,
    RAMP_FAMILY,
    RESOLVE_FAMILIES,
    STEP_HALVINGS,
    StealthyMixin,
)

if TYPE_CHECKING:
    from .core import FdiaGenerator


def attack_frame(
    g: FdiaGenerator, Xt: np.ndarray, family: int, design: Optional[AttackDesign], knobs: FrameKnobs
) -> Optional[Frame]:
    """Build the scan of `family` on the stored operating point `Xt` ([N, 4] = |V|, P_inj, Q_inj, theta):
    the benign scan here, an attacked one from `engine.attacks` (`AttackMixin.attack_frame`, where
    `design` and the reasons a scan is rejected are described)."""
    if family == BENIGN_CODE:
        return _benign_frame(g, Xt)
    return g.attack_frame(Xt, family, design, knobs)


def _benign_frame(g: FdiaGenerator, Xt: np.ndarray) -> Frame:
    """A benign scan: the stored state emitted through the meter plan, and remembered for replay."""
    scan = g.emit_from_state(Xt)
    g.remember_benign(scan.node_x)
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
    )


# the public names this module held before engine.attacks: the mixins' own functions, called with
# the generator first as they were here
remember_benign = CorruptMixin.remember_benign
stealthy_state = FalseStateMixin.stealthy_state
with_region = FalseStateMixin.with_region
is_feasible = FalseStateMixin.is_feasible


# private helpers that moved to engine.attacks: each takes the generator first, as it did here
_MOVED: dict[str, tuple[str, object]] = {
    "_stealthy_frame": (
        "engine.attacks.stealthy.StealthyMixin._stealthy_frame",
        StealthyMixin._stealthy_frame,
    ),
    "_solvable_step": ("engine.attacks.stealthy.StealthyMixin._solvable_step", StealthyMixin._solvable_step),
    "_resolve_frame": ("engine.attacks.stealthy.StealthyMixin._resolve_frame", StealthyMixin._resolve_frame),
    "_lra_frame": ("engine.attacks.stealthy.StealthyMixin._lra_frame", StealthyMixin._lra_frame),
    "_am_frame": ("engine.attacks.stealthy.StealthyMixin._am_frame", StealthyMixin._am_frame),
    "_corrupt_frame": ("engine.attacks.corrupt.CorruptMixin._corrupt_frame", CorruptMixin._corrupt_frame),
    "_within_limits": (
        "engine.attacks.false_state.FalseStateMixin._within_limits",
        FalseStateMixin._within_limits,
    ),
    "_attack_vector": (
        "engine.attacks.false_state.FalseStateMixin._attack_vector",
        FalseStateMixin._attack_vector,
    ),
    "_changed_meters": ("engine.attacks.false_state.changed_meters", changed_meters),
}


def __getattr__(name: str) -> object:
    return moved(__name__, name, _MOVED)
