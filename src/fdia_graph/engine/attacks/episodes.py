"""What one At episode is: its targets and the design each of its frames applies, drawn at onset.

The timeline decides when an episode starts and how long it lasts; these designs decide what it
attacks. A ramp is tested on every frame it will occupy before it is accepted, since the operating
point drifts along the episode and a design feasible at onset can lose its stealthy state later; a
design that passes cannot fall back to a benign frame. The Am episode is the overload attack
(overload.py).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...formulas.attacks import ramp_profile
from ...models.choices import FAMILY_CODE
from ...models.frames import AttackDesign, AttackVector, FrameKnobs, LoadGoal, RampDesign
from .overload import OverloadMixin

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


def ramp_dev(i: int, rise: int, hold: int, rate: float) -> float:
    """The ramp's deviation at step i, never zero: the profile's first frame and its return leg
    floor at one rate, so every frame labelled At carries an attack."""
    return max(rate, ramp_profile(i, rise, hold, rate, rate))


class EpisodeDesignMixin(OverloadMixin):
    """Draw the design of an attack episode at its onset, and the design each of its frames applies."""

    def target_counts(self) -> dict[int, int]:
        """The targets the case offers the ramp: At needs a load off every generator bus (checked
        against the request by `models.inputs.AdmissibleTargets`; the overload Am's targets are the
        rated, metered lines of each window, decided per episode)."""
        return {FAMILY_CODE["At"]: len(self.stealthy_pos)}

    # ---- At, the slow ramp ------------------------------------------------------------------
    def ramp_step(self, design: RampDesign, i: int, rate: float) -> AttackDesign:
        """The design of the ramp's frame i: its load set scaled by the profile's deviation there, solved
        on the episode's fewest-tamper support when the search chose one."""
        return AttackDesign(
            design.targets, 1 + design.direction * ramp_dev(i, design.rise, design.hold, rate), design.support
        )

    def ramp_design(
        self,
        X: np.ndarray,
        t: int,
        shape: tuple[int, float],
        k: FrameKnobs,
        prev: Optional[AttackVector] = None,
    ) -> Optional[RampDesign]:
        """A ramp starting at t whose every frame has a stealthy state, `shape` = (ramp_len, ramp_rate);
        redrawn up to ONSET_DRAWS times, None when no draw has one (the span then stays benign). With
        `k.min_tamper` the accepted ramp carries the support that tampers the fewest devices over its
        window [WU26, eq. 12], held for every frame, its first step measured from `prev`, the attack
        vector of the frame before t (None: that frame is benign); the search spends no random draw."""
        ramp_len, rate = shape
        for _ in range(ONSET_DRAWS):
            design = draw_ramp(self.rng, self.stealthy_pos, ramp_len)  # At is stealthy: no generator bus
            frames = probe_frames(len(X), t, ramp_len)
            if all(self.is_feasible(X[u], self.ramp_step(design, u - t, rate), k) for u in frames):
                return self._fewest_tamper(design, X, (frames, t, rate), k, prev) if k.min_tamper else design
        return None

    def _fewest_tamper(
        self,
        design: RampDesign,
        X: np.ndarray,
        window: tuple[range, int, float],
        k: FrameKnobs,
        prev: Optional[AttackVector],
    ) -> RampDesign:
        """The accepted ramp with the fewest-tamper support of its window (`frames`) and the search's
        result; unchanged when the search finds none (it always has the region the ramp was accepted on)."""
        frames, t, rate = window
        goal = LoadGoal(tuple(self.ramp_step(design, u - t, rate) for u in frames))
        result = self.min_tamper([X[u] for u in frames], goal, k, prev)
        if result is None:
            return design
        held = result.support if result.devices >= 0 else None  # -1: no held support met the constraints
        return design._replace(support=held, tamper=result)
