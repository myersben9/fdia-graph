"""The stealthy ramp At: one scan at a time, a local false state added to the true scan.

The attacker scales its loads inside a subnetwork and solves that subnetwork's power flow with the
boundary voltages held true, so only the subnetwork's meters change and the measurement vector stays
consistent with an AC state [WU26]. With the fewest-tamper knob the subnetwork is the support that
tampers the fewest devices [WU26 eq. 12].
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...constants.families import AT_CODE
from ...models.frames import AttackDesign, Frame, FrameKnobs
from ..base import Part

# times a ramp frame's step with no local power-flow solution is halved (its step is sub-floor)
STEP_HALVINGS = 6


class RampFrames(Part):
    """Build a ramp frame: the local false state of its design, added to the true scan."""

    def attack_frame(
        self, x_true: np.ndarray, family: int, design: AttackDesign, knobs: FrameKnobs
    ) -> Optional[Frame]:
        """The scan of the attacked `family` on the stored operating point `x_true` ([N, 4] = |V|, P_inj,
        Q_inj, theta): an At ramp frame, `design` naming the scaled loads, the multiplier and the held
        support. None when its local power flow has no solution at any halving of its step."""
        assert family == AT_CODE, f"attack_frame builds At frames; family {family} has its own builder"
        return self._ramp_frame(x_true, design, knobs)

    def _stealthy_frame(self, x_true: np.ndarray, design: AttackDesign, knobs: FrameKnobs) -> Optional[Frame]:
        """One local false state [WU26]: the targeted loads scaled by `mult`, the interior re-solved with
        the boundary voltages held true, and a = h(x_false) - h(x_true) added to the true scan. Every meter
        keeps its own noise draw, so the scan is a full AC state plus noise; None when no state solves."""
        # Re-emitting the false state would draw each tampered meter's noise from the false reading, which
        # a meter whose true reading is structurally zero (a condenser's P, a zero-injection bus) gives away.
        x_false = self.g.false_state.stealthy_state(x_true, design, knobs)
        if x_false is None:
            return None  # no local solution, or one outside the operating limits: the caller halves
        buses = self.g.load_bus[design.targets]
        dev = np.abs(np.asarray(design.mult, float) - 1.0)
        return self.g.attack_frames.frame_from_state(
            x_true, x_false, buses, np.broadcast_to(dev, buses.shape).astype(float)
        )

    def _solvable_step(self, x_true: np.ndarray, design: AttackDesign, knobs: FrameKnobs) -> Optional[Frame]:
        """The stealthy frame of the design, or of the largest halving of its step (at most STEP_HALVINGS)
        that has a local power-flow solution, so a step the region cannot absorb still leaves the frame
        attacked. The frame's magnitudes record the step used."""
        frame = self._stealthy_frame(x_true, design, knobs)
        for _ in range(STEP_HALVINGS):
            if frame is not None:
                break
            design = design._replace(mult=1.0 + (np.asarray(design.mult, float) - 1.0) / 2)
            frame = self._stealthy_frame(x_true, design, knobs)
        return frame

    def _ramp_frame(self, x_true: np.ndarray, design: AttackDesign, knobs: FrameKnobs) -> Optional[Frame]:
        """An At frame: the targeted loads scaled, and the subnetwork within `hops` of them (or the
        episode's fewest-tamper support when the search chose one) re-solved locally."""
        held = design if knobs.min_tamper and design.interior is not None else design._replace(interior=None)
        placed = self.g.false_state.with_region(held, knobs)
        if placed is None:
            return None
        return self._solvable_step(x_true, placed, knobs)
