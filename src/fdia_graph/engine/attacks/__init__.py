"""Every attack the generator builds, behind one mixin: `AttackMixin`, mixed into FdiaGenerator.

A reader looking for how an attack is made starts here:

    area.py            where a stealthy attack acts: the interior it moves, the boundary it holds true
    false_state.py     the local false state of a design, its operating limits, the attack vector
                       a = h(x_false) - h(x_true) and the meters it moves
    stealthy.py        the frames of the ramp At, and the halving of a step that does not solve
    minimize.py        the fewest-tamper support of an attack window [WU26 eq. 12], behind `min_tamper`,
                       and the support strategies (`SearchSupport`, `rref.RrefSupport`) it dispatches to
    rref.py            [WU26]'s row reduction as a support strategy (`RrefSupport`)
    overload.py        Am as the overload attack of [WU26]: the line ratings, the eligible target
                       branches, the flow goal each snapshot must reach and its frames
    episodes.py        what an episode attacks: the family designers (`RampDesigner` for At,
                       `OverloadDesigner` for Am), chosen per family by `EpisodeDesignMixin.designer`

`attack_frame` is the one entry for an attacked At scan (an Am frame comes from `overload_step`);
the timeline decides when and where an episode runs (timeline.py) and `engine.records` emits a
benign scan.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...models.frames import (  # noqa: F401  re-exported: engine.attacks named these before the models package
    AttackDesign,
    Frame,
    FrameKnobs,
    Scan,
)
from .episodes import EpisodeDesignMixin
from .overload import OverloadMixin
from .stealthy import RAMP_FAMILY, StealthyMixin


class AttackMixin(EpisodeDesignMixin, OverloadMixin, StealthyMixin):
    """Build the attacks that re-solve the grid under false loads."""

    def attack_frame(
        self, Xt: np.ndarray, family: int, design: AttackDesign, k: FrameKnobs
    ) -> Optional[Frame]:
        """The scan of the attacked `family` on the stored operating point `Xt` ([N, 4] = |V|, P_inj,
        Q_inj, theta): an At ramp frame, `design` naming the scaled loads, the multiplier and the held
        support. None when its local power flow has no solution at any halving of its step."""
        assert family == RAMP_FAMILY, f"attack_frame builds At frames; family {family} has its own builder"
        return self._ramp_frame(Xt, design, k)
