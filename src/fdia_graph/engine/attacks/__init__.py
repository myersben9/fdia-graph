"""Every attack the generator builds, behind one mixin: `AttackMixin`, mixed into FdiaGenerator.

area.py         where a stealthy attack acts: the interior it moves, the boundary it holds true
false_state.py  the local false state, its operating limits, the attack vector a = h(x_false) - h(x_true)
stealthy.py     the frames of the ramp At
minimize.py     the fewest-tamper support of an attack window [WU26 eq. 12], behind `min_tamper`
rref.py         [WU26]'s row reduction as the choice of a support
overload.py     Am, the overload attack of [WU26]: ratings, target branches, flow goal, frames
episodes.py     what an episode attacks: `ramp_design` for At, `am_overload_design` for Am
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...constants.families import AT_CODE
from ...models.frames import AttackDesign, Frame, FrameKnobs
from .episodes import EpisodeDesignMixin
from .overload import OverloadMixin
from .stealthy import StealthyMixin


class AttackMixin(EpisodeDesignMixin, OverloadMixin, StealthyMixin):
    """Build the attacks that re-solve the grid under false loads."""

    def attack_frame(
        self, Xt: np.ndarray, family: int, design: AttackDesign, k: FrameKnobs
    ) -> Optional[Frame]:
        """The scan of the attacked `family` on the stored operating point `Xt` ([N, 4] = |V|, P_inj,
        Q_inj, theta): an At ramp frame, `design` naming the scaled loads, the multiplier and the held
        support. None when its local power flow has no solution at any halving of its step."""
        assert family == AT_CODE, f"attack_frame builds At frames; family {family} has its own builder"
        return self._ramp_frame(Xt, design, k)
