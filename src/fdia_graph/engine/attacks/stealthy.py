"""The stealthy families Aq, At, Al and Am: one scan each, a local false state added to the true scan.

The attacker scales or redistributes loads inside a subnetwork and solves that subnetwork's power
flow with the boundary voltages held true, so only the subnetwork's meters change and the
measurement vector stays consistent with an AC state. They satisfy the constraints of [WU26] (its
measurement model and operating limits), not its fewest-meter objective.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...models.choices import FAMILY_CODE
from ...models.frames import AttackDesign, Frame, FrameKnobs
from .redistribution import RedistributionMixin

AQ_FAMILY = FAMILY_CODE["Aq"]
RAMP_FAMILY = FAMILY_CODE["At"]
LRA_FAMILY = FAMILY_CODE["Al"]
AM_FAMILY = FAMILY_CODE["Am"]  # multi-snapshot, timelines only
RESOLVE_FAMILIES = (AQ_FAMILY, RAMP_FAMILY)  # the grid is re-solved under a scaled load
LRA_DRAWS = 40  # target lines an Al frame tries before giving up: a region may hold too few loads, or
# every redistribution at this operating point may push a boundary generator past its limits
AQ_HALVINGS = 3  # times an Aq load step with no local power-flow solution is halved before giving up
STEP_HALVINGS = 6  # the same for one frame of a ramp (At, Am), whose step is under the floor anyway


class StealthyMixin(RedistributionMixin):
    """Build the frame of a stealthy family: the local false state of its design, added to the true scan."""

    def _stealthy_frame(self, Xt: np.ndarray, design: AttackDesign, k: FrameKnobs) -> Optional[Frame]:
        """One local false state [WU26]: the targeted loads scaled by `mult`, the interior buses
        re-solved with the boundary voltages held true, and the attack vector a = h(x_false) - h(x_true)
        added to the true scan. Every meter keeps its own noise draw and the tampered ones (the meters
        the false state moves) are shifted by exactly what it moves them, so the measurement is a full
        AC state plus meter noise and the residual test sees noise only. A re-emission of the false
        state would draw each tampered meter's noise from the false reading instead, which a meter
        whose true reading is structurally zero (a condenser's P, a zero-injection bus) gives away.
        Needs `with_benign`: the true scan is the benign twin."""
        assert k.with_benign, "a stealthy frame needs the benign twin (with_benign=True)"
        Xa = self.stealthy_state(Xt, design, k)
        if Xa is None:
            return None  # no local solution, or one outside the operating limits: the caller halves
        buses = self.load_bus[design.targets]
        dev = np.abs(np.asarray(design.mult, float) - 1.0)
        return self.frame_from_state(Xt, Xa, buses, np.broadcast_to(dev, buses.shape).astype(float))

    def _solvable_step(
        self, Xt: np.ndarray, design: AttackDesign, k: FrameKnobs, limit: tuple[int, Optional[float]]
    ) -> Optional[Frame]:
        """The stealthy frame of the design, or of the largest halving of its step that has a local
        power-flow solution: `limit` = (halvings tried, the floor a halved step may not fall under, or
        None). A step the region cannot absorb (a large load inside a fixed boundary) keeps the frame
        attacked at the largest step that solves; the frame's magnitudes record the step used."""
        halvings, floor = limit
        frame = self._stealthy_frame(Xt, design, k)
        for _ in range(halvings):
            if frame is not None:
                break
            mult = 1.0 + (np.asarray(design.mult, float) - 1.0) / 2
            if floor is not None and np.max(np.abs(mult - 1.0)) < floor:
                break
            design = design._replace(mult=mult)
            frame = self._stealthy_frame(Xt, design, k)
        return frame

    def _resolve_frame(
        self, Xt: np.ndarray, family: int, design: AttackDesign, k: FrameKnobs
    ) -> Optional[Frame]:
        """Aq and At: the targeted loads scaled, the subnetwork within `hops` of them re-solved locally.
        A step without a local solution is halved: an Aq step at most AQ_HALVINGS times and never under
        the noise floor, a ramp frame at most STEP_HALVINGS times (its design step is sub-floor)."""
        dev = np.abs(np.asarray(design.mult) - 1.0)  # per-bus designed load-shift fraction
        if k.reject_below_floor and family == AQ_FAMILY and np.max(dev) < k.floor:
            return None  # a within-noise no-op; the ramp is exempt so its per-scan step may stay sub-floor
        # the region around the targets, or the episode's fewest-tamper support when the search chose one
        held = design if k.min_tamper and design.interior is not None else design._replace(interior=None)
        placed = self.with_region(held, k)
        if placed is None:
            return None
        limit = (AQ_HALVINGS, k.floor) if family == AQ_FAMILY else (STEP_HALVINGS, None)
        return self._solvable_step(Xt, placed, k, limit)

    def _lra_frame(self, Xt: np.ndarray, k: FrameKnobs) -> Optional[Frame]:
        """Al: a load-conserving redistribution over up to lra_k buses of the subnetwork around a
        target line, steering that line, re-solved locally [DAT26, WU26]; a line whose redistribution
        has no stealthy state at any halving above the floor is redrawn."""
        Lp = self.true_load(Xt)
        for _ in range(LRA_DRAWS):  # a line with no feasible or no solvable redistribution is redrawn
            red = self.lra_delta(Lp, k.intensity, k.lra_k, floor=k.floor, hops=k.hops)
            a = red.buses
            if len(a) == 0:
                continue
            dev = np.abs(red.delta[a]) / (np.abs(Lp[a]) + 1e-6)  # designed redistribution fraction per bus
            if k.reject_below_floor and np.min(dev) < k.floor:
                return None  # a bus inside the noise floor
            mult = 1.0 + red.delta[a] / np.where(np.abs(Lp[a]) > 1e-9, Lp[a], 1e-9)
            frame = self._solvable_step(Xt, AttackDesign(a, mult, red.interior), k, (AQ_HALVINGS, k.floor))
            if frame is not None:
                return frame
        return None  # no drawn line gave a redistribution with a stealthy state at any halving

    def _am_frame(self, Xt: np.ndarray, design: AttackDesign, k: FrameKnobs) -> Optional[Frame]:
        """Am, a multi-snapshot attack after [WU26], not the paper's fewest-meter overload attack: one
        step of a load redistribution held over an episode, drawn once at onset (`am_design`), whose
        design carries this frame's fraction of it as the multiplier (one per target) and the
        attacker's interior; the local false state of that step."""
        placed = self.with_region(design, k)
        if placed is None:
            return None
        return self._solvable_step(Xt, placed, k, (STEP_HALVINGS, None))
