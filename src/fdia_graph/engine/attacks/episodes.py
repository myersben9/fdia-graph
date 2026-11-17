"""What one episode attacks: its design, drawn at onset (`EpisodeDesignMixin`).

The timeline decides when an episode starts and how long it lasts; its design decides what it
attacks. `ramp_design` draws the slow ramp At: a ramp is tested on every frame it will occupy
before it is accepted, since the operating point drifts along the episode and a design feasible at
onset can lose its stealthy state later; a design that passes cannot fall back to a benign frame.
`am_overload_design` designs the overload Am of [WU26] (its ratings, goal and frames, overload.py).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...constants.families import AT_CODE
from ...formulas.attacks import ramp_profile
from ...models.frames import AmOverloadDesign, AttackDesign, AttackVector, FrameKnobs, LoadGoal, RampDesign
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


class EpisodeDesignMixin(OverloadMixin):
    """The design of an episode of each family: the slow ramp At and the overload Am of [WU26]."""

    def target_counts(self) -> dict[int, int]:
        """The targets the case offers the ramp: At needs a load off every generator bus (checked
        against the request by `models.inputs.AdmissibleTargets`; the overload Am's targets are the
        rated, metered lines of each window, decided per episode)."""
        return {AT_CODE: len(self.stealthy_pos)}

    def ramp_step(self, design: RampDesign, i: int, rate: float) -> AttackDesign:
        """The design of the ramp's frame i: its load set scaled by the profile's deviation there,
        solved on the episode's fewest-tamper support when the search chose one. The deviation is
        never zero: the profile's first frame and its return leg floor at one rate, so every frame
        labelled At carries an attack."""
        deviation = max(rate, ramp_profile(i, design.rise, design.hold, rate, rate))
        return AttackDesign(design.targets, 1 + design.direction * deviation, design.support)

    def ramp_design(
        self,
        X: np.ndarray,
        t: int,
        shape: tuple[int, float],
        k: FrameKnobs,
        prev: Optional[AttackVector] = None,
    ) -> Optional[RampDesign]:
        """The slow ramp At starting at t whose every frame has a stealthy state. `shape` = (ramp_len,
        ramp_rate); redrawn up to ONSET_DRAWS times, None when no draw has one (the span then stays
        benign). With `k.min_tamper` the accepted ramp carries the support that tampers the fewest
        devices over its window [WU26 eq. 12], held for every frame, its first step measured from
        `prev`, the attack vector of the frame before t (None: that frame is benign); the search
        spends no random draw."""
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

    def am_overload_design(
        self, X: np.ndarray, t: int, length: int, k: FrameKnobs, rng: Optional[np.random.Generator] = None
    ) -> Optional[AmOverloadDesign]:
        """One `Am` episode starting at t: at most `AM_LINE_TRIES` target sets of the eligible branches,
        in a random order (one draw from `rng`, the generator's own when None, only when there is a
        branch), each tried until the fewest-tamper search finds a support that meets the goal at every
        snapshot [WU26 eqs. 24-25], inside the operating limits [WU26 eqs. 21-23], and moves at least one
        device beyond noise. None when no set has one (the design stage then moves the episode,
        `generation.design`). A flow goal has no stealth bound, so nothing here reads the
        frame before t."""
        window = [X[u] for u in range(t, min(t + length, len(X)))]
        lines = self.eligible_lines(window, k.hops)
        if len(lines) == 0:
            return None
        order = (self.rng if rng is None else rng).permutation(lines)
        for targets in self._target_sets(order, k.n_lines, k.hops):
            goal = self.overload_goal(window, *targets)
            result = self.min_tamper(window, goal, k)
            if result is not None and result.devices >= 1:
                # eq. 25: S_max at the end
                ratings = tuple(float(s) for s in goal.targets_at(len(window) - 1))
                return AmOverloadDesign(goal, ratings, result.support, result)
        return None
