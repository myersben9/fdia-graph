"""What one episode attacks: its design, drawn at onset (`EpisodeDesigns`).

The timeline decides when an episode starts and how long it lasts; its design decides what it
attacks. `ramp_design` draws the slow ramp At and tests it on every frame it will occupy, since the
operating point drifts and a design feasible at onset can lose its stealthy state later.
`am_overload_design` designs the overload Am of [WU26] (its ratings, goal and frames, overload.py).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...constants.families import AT_CODE
from ...formulas.attacks import ramp_profile
from ...models.frames import AmOverloadDesign, AttackDesign, AttackVector, FrameKnobs, LoadGoal, RampDesign
from ...systems.states import States
from ..base import Part

ONSET_DRAWS = 40  # designs an At episode tries for one with a stealthy state on every frame


def probe_frames(T: int, t: int, length: int) -> range:
    """The frames an episode's design is tested on before it is accepted: every frame it will
    occupy, clipped to the T frames of the pool."""
    return range(t, min(t + length, T))


def draw_ramp(rng: np.random.Generator, apos: np.ndarray, ramp_len: int) -> RampDesign:
    """One ramp design over the loads `apos`: a fixed bus set, a direction, the rise and hold
    lengths (four draws)."""
    a = rng.choice(apos, min(5, len(apos)), replace=False)
    direction = 1.0 if rng.random() < 0.5 else -1.0
    rise = max(1, int(rng.uniform(0.2, 0.45) * ramp_len))
    hold = int(rng.uniform(0.0, 0.25) * ramp_len)
    return RampDesign(a, direction, rise, hold)


class EpisodeDesigns(Part):
    """The design of an episode of each family: the slow ramp At and the overload Am of [WU26]."""

    def target_counts(self) -> dict[int, int]:
        """The targets the case offers the ramp: At needs a load off every generator bus. Am's targets
        are the rated, metered lines of each window, decided per episode."""
        return {AT_CODE: len(self.g.stealthy_pos)}

    def ramp_step(self, design: RampDesign, i: int, rate: float) -> AttackDesign:
        """The design of the ramp's frame i: its load set scaled by the profile's deviation there. The
        deviation is floored at one rate, so every frame labelled At carries an attack."""
        deviation = max(rate, ramp_profile(i, design.rise, design.hold, rate, rate))
        return AttackDesign(design.targets, 1 + design.direction * deviation, design.support)

    def ramp_design(
        self,
        X: States,
        t: int,
        shape: tuple[int, float],
        knobs: FrameKnobs,
        prev: Optional[AttackVector] = None,
    ) -> Optional[RampDesign]:
        """The slow ramp At starting at t whose every frame has a stealthy state; `shape` = (ramp_len,
        ramp_rate). Redrawn up to ONSET_DRAWS times, None when no draw has one. With `knobs.min_tamper` it
        carries the fewest-tamper support [WU26 eq. 12]; `prev` is the attack vector of the frame before t."""
        # the fewest-tamper search spends no random draw
        ramp_len, rate = shape
        for _ in range(ONSET_DRAWS):
            design = draw_ramp(self.g.rng, self.g.stealthy_pos, ramp_len)  # At is stealthy: no generator bus
            frames = probe_frames(len(X), t, ramp_len)
            if all(
                self.g.false_state.is_feasible(X[u], self.ramp_step(design, u - t, rate), knobs)
                for u in frames
            ):
                return (
                    self._fewest_tamper(design, X, (frames, t, rate), knobs, prev)
                    if knobs.min_tamper
                    else design
                )
        return None

    def _fewest_tamper(
        self,
        design: RampDesign,
        X: States,
        window: tuple[range, int, float],
        knobs: FrameKnobs,
        prev: Optional[AttackVector],
    ) -> RampDesign:
        """The accepted ramp with the fewest-tamper support of its window and the search's result;
        unchanged when the search finds none."""
        frames, t, rate = window
        goal = LoadGoal(tuple(self.ramp_step(design, u - t, rate) for u in frames))
        result = self.g.minimizer.min_tamper([X[u] for u in frames], goal, knobs, prev)
        if result is None:
            return design
        held = result.support if result.devices >= 0 else None  # -1: no held support met the constraints
        return design._replace(support=held, tamper=result)

    def am_overload_design(
        self, X: np.ndarray, t: int, length: int, knobs: FrameKnobs, rng: Optional[np.random.Generator] = None
    ) -> Optional[AmOverloadDesign]:
        """One Am episode starting at t: the first target set, in a random order of the eligible branches,
        whose support meets the goal at every snapshot [WU26 eqs. 24-25] inside the limits [WU26 eqs. 21-23]
        and moves a device beyond noise; None when no set does. One draw from `rng` (None: the generator's
        own stream), if a branch exists."""
        window = [X[u] for u in range(t, min(t + length, len(X)))]
        lines = self.g.overload.eligible_lines(window, knobs.hops)
        if len(lines) == 0:
            return None
        order = (self.g.rng if rng is None else rng).permutation(lines)
        for targets in self.g.overload.target_sets(order, knobs.n_lines, knobs.hops):
            goal = self.g.overload.overload_goal(window, *targets)
            result = self.g.minimizer.min_tamper(window, goal, knobs)
            if result is not None and result.devices >= 1:
                # eq. 25: S_max at the end
                ratings = tuple(float(s) for s in goal.targets_at(len(window) - 1))
                return AmOverloadDesign(goal, ratings, result.support, result)
        return None
