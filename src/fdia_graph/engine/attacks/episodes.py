"""What one episode attacks: its design, drawn at onset by the family's designer.

The timeline decides when an episode starts and how long it lasts; a designer decides what it
attacks (decision B4: one strategy object per family, chosen by `EpisodeDesignMixin.designer`, not a
base class). `RampDesigner` draws the slow ramp At: a ramp is tested on every frame it will occupy
before it is accepted, since the operating point drifts along the episode and a design feasible at
onset can lose its stealthy state later; a design that passes cannot fall back to a benign frame.
`OverloadDesigner` designs the overload Am of [WU26] (its ratings, goal and frames, overload.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, cast

import numpy as np

from ...formulas.attacks import ramp_profile
from ...models.choices import FAMILY_CODE
from ...models.frames import (
    AmOverloadDesign,
    AttackDesign,
    AttackVector,
    FrameKnobs,
    LoadGoal,
    RampDesign,
)
from ..base import GridBase

if TYPE_CHECKING:
    from . import AttackMixin
    from .overload import OverloadMixin

ONSET_DRAWS = 40  # designs an At episode tries for one with a stealthy state on every frame
RAMP_CODE, AM_CODE = FAMILY_CODE["At"], FAMILY_CODE["Am"]


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


def ramp_step(design: RampDesign, i: int, rate: float) -> AttackDesign:
    """The design of the ramp's frame i: its load set scaled by the profile's deviation there, solved
    on the episode's fewest-tamper support when the search chose one."""
    return AttackDesign(
        design.targets, 1 + design.direction * ramp_dev(i, design.rise, design.hold, rate), design.support
    )


class RampDesigner:
    """The slow ramp At: a ramp starting at t whose every frame has a stealthy state."""

    def design(
        self,
        g: AttackMixin,
        X: np.ndarray,
        t: int,
        shape: tuple[int, float],
        k: FrameKnobs,
        prev: Optional[AttackVector] = None,
    ) -> Optional[RampDesign]:
        """`shape` = (ramp_len, ramp_rate); redrawn up to ONSET_DRAWS times, None when no draw has one
        (the span then stays benign). With `k.min_tamper` the accepted ramp carries the support that
        tampers the fewest devices over its window [WU26 eq. 12], held for every frame, its first step
        measured from `prev`, the attack vector of the frame before t (None: that frame is benign); the
        search spends no random draw."""
        ramp_len, rate = shape
        for _ in range(ONSET_DRAWS):
            design = draw_ramp(g.rng, g.stealthy_pos, ramp_len)  # At is stealthy: no generator bus
            frames = probe_frames(len(X), t, ramp_len)
            if all(g.is_feasible(X[u], ramp_step(design, u - t, rate), k) for u in frames):
                return (
                    self._fewest_tamper(g, design, X, (frames, t, rate), k, prev) if k.min_tamper else design
                )
        return None

    @staticmethod
    def _fewest_tamper(
        g: AttackMixin,
        design: RampDesign,
        X: np.ndarray,
        window: tuple[range, int, float],
        k: FrameKnobs,
        prev: Optional[AttackVector],
    ) -> RampDesign:
        """The accepted ramp with the fewest-tamper support of its window (`frames`) and the search's
        result; unchanged when the search finds none (it always has the region the ramp was accepted on)."""
        frames, t, rate = window
        goal = LoadGoal(tuple(ramp_step(design, u - t, rate) for u in frames))
        result = g.min_tamper([X[u] for u in frames], goal, k, prev)
        if result is None:
            return design
        held = result.support if result.devices >= 0 else None  # -1: no held support met the constraints
        return design._replace(support=held, tamper=result)


class OverloadDesigner:
    """The overload attack Am of [WU26]: target branches driven to their ratings over the window."""

    def design(
        self,
        g: OverloadMixin,
        X: np.ndarray,
        t: int,
        length: int,
        k: FrameKnobs,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[AmOverloadDesign]:
        """One `Am` episode starting at t: at most `AM_LINE_TRIES` target sets of the eligible branches,
        in a random order (one draw from `rng`, the generator's own when None, only when there is a
        branch), each tried until the fewest-tamper search finds a support that meets the goal at every
        snapshot [WU26 eqs. 24-25], inside the operating limits [WU26 eqs. 21-23], and moves at least one
        device beyond noise. None when no set has one (the design stage then moves the episode,
        `generation.design`). A flow goal has no stealth bound [D11], so nothing here reads the
        frame before t."""
        frames = range(t, min(t + length, len(X)))
        window = [X[u] for u in frames]
        lines = g.eligible_lines(window, k.hops)
        if len(lines) == 0:
            return None
        order = (g.rng if rng is None else rng).permutation(lines)
        for targets in g._target_sets(order, k.n_lines, k.hops):
            goal = g.overload_goal(window, *targets)
            result = g.min_tamper(window, goal, k)
            if result is not None and result.devices >= 1:
                ratings = tuple(
                    float(s) for s in goal.targets_at(len(window) - 1)
                )  # eq. 25: S_max at the end
                return AmOverloadDesign(goal, ratings, result.support, result)
        return None


Designer = Union[RampDesigner, OverloadDesigner]
DESIGNERS: dict[int, Designer] = {RAMP_CODE: RampDesigner(), AM_CODE: OverloadDesigner()}


class EpisodeDesignMixin(GridBase):
    """The family designers behind the generator's episode entry points."""

    def designer(self, family: int) -> Designer:
        """The designer of `family` (a family code): `RampDesigner` for At, `OverloadDesigner` for Am."""
        return DESIGNERS[family]

    def target_counts(self) -> dict[int, int]:
        """The targets the case offers the ramp: At needs a load off every generator bus (checked
        against the request by `models.inputs.AdmissibleTargets`; the overload Am's targets are the
        rated, metered lines of each window, decided per episode)."""
        return {RAMP_CODE: len(self.stealthy_pos)}

    def ramp_step(self, design: RampDesign, i: int, rate: float) -> AttackDesign:
        """`ramp_step`: the design of the ramp's frame i."""
        return ramp_step(design, i, rate)

    def ramp_design(
        self,
        X: np.ndarray,
        t: int,
        shape: tuple[int, float],
        k: FrameKnobs,
        prev: Optional[AttackVector] = None,
    ) -> Optional[RampDesign]:
        """The At designer's design on this generator (`designer`)."""
        ramp = cast(RampDesigner, self.designer(RAMP_CODE))
        return ramp.design(cast("AttackMixin", self), X, t, shape, k, prev)

    def am_overload_design(
        self,
        X: np.ndarray,
        t: int,
        length: int,
        k: FrameKnobs,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[AmOverloadDesign]:
        """The Am designer's design on this generator (`designer`)."""
        overload = cast(OverloadDesigner, self.designer(AM_CODE))
        return overload.design(cast("OverloadMixin", self), X, t, length, k, rng)
