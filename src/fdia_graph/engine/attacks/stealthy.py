"""The stealthy ramp At: one scan at a time, a local false state added to the true scan.

The attacker scales its loads inside a subnetwork and solves that subnetwork's power flow with the
boundary voltages held true, so only the subnetwork's meters change and the measurement vector stays
consistent with an AC state [WU26] (its measurement model and operating limits); with the
fewest-tamper knob the subnetwork is the support that tampers the fewest devices (eq. 12).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...models.choices import FAMILY_CODE
from ...models.frames import AttackDesign, Frame, FrameKnobs
from .false_state import FalseStateMixin

RAMP_FAMILY = FAMILY_CODE["At"]
AM_FAMILY = FAMILY_CODE["Am"]  # the overload attack (overload.py), multi-snapshot
STEP_HALVINGS = (
    6  # times a ramp frame's step with no local power-flow solution is halved (its step is sub-floor)
)


class StealthyMixin(FalseStateMixin):
    """Build a ramp frame: the local false state of its design, added to the true scan."""

    def _stealthy_frame(self, Xt: np.ndarray, design: AttackDesign, k: FrameKnobs) -> Optional[Frame]:
        """One local false state [WU26]: the targeted loads scaled by `mult`, the interior buses
        re-solved with the boundary voltages held true, and the attack vector a = h(x_false) - h(x_true)
        added to the true scan. Every meter keeps its own noise draw and the tampered ones (the meters
        the false state moves) are shifted by exactly what it moves them, so the measurement is a full
        AC state plus meter noise and the residual test sees noise only. A re-emission of the false
        state would draw each tampered meter's noise from the false reading instead, which a meter
        whose true reading is structurally zero (a condenser's P, a zero-injection bus) gives away."""
        Xa = self.stealthy_state(Xt, design, k)
        if Xa is None:
            return None  # no local solution, or one outside the operating limits: the caller halves
        buses = self.load_bus[design.targets]
        dev = np.abs(np.asarray(design.mult, float) - 1.0)
        return self.frame_from_state(Xt, Xa, buses, np.broadcast_to(dev, buses.shape).astype(float))

    def _solvable_step(self, Xt: np.ndarray, design: AttackDesign, k: FrameKnobs) -> Optional[Frame]:
        """The stealthy frame of the design, or of the largest halving of its step (at most
        STEP_HALVINGS) that has a local power-flow solution. A step the region cannot absorb (a large
        load inside a fixed boundary) keeps the frame attacked at the largest step that solves; the
        frame's magnitudes record the step used."""
        frame = self._stealthy_frame(Xt, design, k)
        for _ in range(STEP_HALVINGS):
            if frame is not None:
                break
            design = design._replace(mult=1.0 + (np.asarray(design.mult, float) - 1.0) / 2)
            frame = self._stealthy_frame(Xt, design, k)
        return frame

    def _ramp_frame(self, Xt: np.ndarray, design: AttackDesign, k: FrameKnobs) -> Optional[Frame]:
        """At: the targeted loads scaled, the subnetwork within `hops` of them (or the episode's
        fewest-tamper support when the search chose one) re-solved locally; a step without a local
        solution is halved at most STEP_HALVINGS times (its design step is sub-floor)."""
        held = design if k.min_tamper and design.interior is not None else design._replace(interior=None)
        placed = self.with_region(held, k)
        if placed is None:
            return None
        return self._solvable_step(Xt, placed, k)
