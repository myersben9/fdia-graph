"""One At frame: the false state of the frame's `AttackDesign`, written as a frame.

`search.RampEpisode.frame` calls `ramp_frame` for each frame of an episode. It calls
`FalseState.with_support` and `stealthy_state` for the false state, then `AttackFrames.frame_from_state`.
The targeted loads are scaled inside the support and the support is re-solved with every other voltage
held true, so only the reach's meters change [WU26]. A ramp is accepted only when every frame of its
window has a false state on its support inside the limits and the stealth bound
(`EpisodeDesigns.ramp_design`), and a frame is the same solve on the same state, so it is built at the
designed step.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...models.frames import AttackDesign, Frame, FrameKnobs
from ..base import Part


class RampFrames(Part):
    """Build a ramp frame: the local false state of its design, added to the true scan."""

    def _stealthy_frame(self, x_true: np.ndarray, design: AttackDesign, knobs: FrameKnobs) -> Optional[Frame]:
        """One local false state [WU26]: the targeted loads scaled by `mult`, the support re-solved with
        the boundary voltages held true, and a = h(x_false) - h(x_true) added to the true scan. Every meter
        keeps its own noise draw, so the scan is a full AC state plus noise; None when no state solves."""
        # Re-emitting the false state would draw each tampered meter's noise from the false reading, which
        # a meter whose true reading is structurally zero (a condenser's P, a zero-injection bus) gives away.
        x_false = self.g.false_state.stealthy_state(x_true, design, knobs)
        if x_false is None:
            return None  # no local solution, or one outside the operating limits
        buses = self.g.load_bus[design.targets]
        dev = np.abs(np.asarray(design.mult, float) - 1.0)
        return self.g.attack_frames.frame_from_state(
            x_true, x_false, buses, np.broadcast_to(dev, buses.shape).astype(float)
        )

    def ramp_frame(self, x_true: np.ndarray, design: AttackDesign, knobs: FrameKnobs) -> Optional[Frame]:
        """An At frame on the stored operating point `x_true` [N, 4]: the targeted loads scaled, and
        the area within `hops` of them (or the episode's fewest-tamper support when the search chose
        one) re-solved locally. None when the targets have no area, or when the local power flow has
        no solution or the state breaks the operating limits."""
        placed = self.g.false_state.with_support(design, knobs)
        if placed is None:
            return None
        return self._stealthy_frame(x_true, placed, knobs)
