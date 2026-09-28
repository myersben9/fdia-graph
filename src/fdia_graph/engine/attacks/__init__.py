"""Every attack the generator builds, behind one mixin: `AttackMixin`, mixed into FdiaGenerator.

A reader looking for how an attack is made starts here:

    area.py            where a stealthy attack acts: the interior it moves, the boundary it holds true
    false_state.py     the local false state of a design, its operating limits, the attack vector
                       a = h(x_false) - h(x_true) and the meters it moves
    redistribution.py  the load-conserving redistribution behind Al and Am, and its target lines
    stealthy.py        the frames of Aq, At, Al and Am, and the halving of a step that does not solve
    minimize.py        the fewest-tamper support of an attack window [WU26, eq. 12], behind `min_tamper`
    episodes.py        what an episode attacks, drawn at its onset: the ramp, the single-shot design,
                       the held Am redistribution and the schedule each frame applies
    corrupt.py         Ad, As and Ar, tampered in place, and the benign scans the replay draws from

`attack_frame` is the one entry for an attacked scan; the timeline decides when and where an
episode runs (timeline.py) and `engine.records` emits a benign scan.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...models.frames import (  # noqa: F401  re-exported: engine.attacks named these before the models package
    AttackDesign,
    Band,
    Frame,
    FrameKnobs,
    Redistribution,
    Scan,
    TamperTarget,
)
from .corrupt import CorruptMixin
from .episodes import EpisodeDesignMixin
from .stealthy import AM_FAMILY, LRA_FAMILY, RESOLVE_FAMILIES, StealthyMixin


class AttackMixin(EpisodeDesignMixin, StealthyMixin, CorruptMixin):
    """Build the attacks that tamper measurements or re-solve the grid under false loads."""

    def attack_frame(
        self, Xt: np.ndarray, family: int, design: Optional[AttackDesign], k: FrameKnobs
    ) -> Optional[Frame]:
        """The scan of the attacked `family` on the stored operating point `Xt` ([N, 4] = |V|, P_inj,
        Q_inj, theta). `design` names the attacked loads (and, for the stealthy families, the
        multiplier and the held interior); None for Al, which draws its own redistribution. None when
        the scan is rejected: a non-converging local power flow, no feasible redistribution, or, with
        `k.reject_below_floor`, a designed or realized change inside the noise floor."""
        if family == LRA_FAMILY:
            return self._lra_frame(Xt, k)
        assert design is not None, f"family {family} needs an attack design"
        if family in RESOLVE_FAMILIES:
            return self._resolve_frame(Xt, family, design, k)
        if family == AM_FAMILY:
            return self._am_frame(Xt, design, k)
        return self._corrupt_frame(Xt, family, design.targets, k)
