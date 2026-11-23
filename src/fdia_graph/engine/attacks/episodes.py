"""The design of one episode (`EpisodeDesigns`): At's drawn at its onset, Am's from its window.

The timeline decides when an episode starts and how long it lasts; its design decides what it attacks.
`ramp_design` draws the slow ramp At (`draw_ramp`: one neighbourhood of loads) and tests it with
`FalseState.is_feasible` on every frame it will occupy, since the operating point drifts and a design
feasible at onset can lose its false state later. A ramp is kept only on a support that holds its
stealth bound at every frame: the fewest-tamper support with `knobs.min_tamper`
(`Minimizer.min_tamper`), else the area around its targets (`AttackWindow.holds`). When neither does,
other targets are drawn. `am_overload_design` designs the overload Am of [WU26] from `Overload`'s
`eligible_lines`, `target_sets` and `overload_goal`, and calls `Minimizer.min_tamper` for the support.
The ramp's profile is `formulas.attacks.ramp_profile`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...constants.families import AT_CODE
from ...constants.numerics import RAMP_SUPPORT_TRIES
from ...formulas.attacks import ramp_profile
from ...models.frames import AmOverloadDesign, AttackDesign, AttackVector, FrameKnobs, LoadGoal, RampDesign
from ...systems.states import States
from ..base import Part
from .window import AttackWindow

ONSET_DRAWS = 40  # designs an At episode tries for one with a stealthy state on every frame


def probe_frames(T: int, t: int, length: int) -> range:
    """The frames an episode's design is tested on before it is accepted: every frame it will
    occupy, clipped to the T frames of the pool."""
    return range(t, min(t + length, T))


class EpisodeDesigns(Part):
    """The design of an episode of each family: the slow ramp At and the overload Am of [WU26]."""

    def target_counts(self) -> dict[int, int]:
        """The targets the case offers the ramp: At needs a load off every generator bus. Am's targets
        are the rated, metered lines of each window, decided per episode."""
        return {AT_CODE: len(self.g.stealthy_pos)}

    def draw_ramp(self, rng: np.random.Generator, ramp_len: int, knobs: FrameKnobs) -> RampDesign:
        """One ramp design: its loads, a direction, the rise and hold lengths. The loads are one
        neighbourhood, as [WU26]'s attacker works in one connected subnetwork (Sec. III-A): one of the
        ramp's eligible loads drawn uniformly, then up to `knobs.ramp_targets` - 1 others from the
        eligible loads within `knobs.hops` branches of it, fewer when fewer are in reach (so two
        targets are at most twice `hops` apart). Five calls on `rng`; how far they advance it depends
        on how many loads the neighbourhood holds."""
        g, eligible = self.g, self.g.stealthy_pos  # At is stealthy: no load on a generator bus
        first = int(rng.choice(eligible))
        near = g.net.neighbors(g.load_bus[first], knobs.hops)
        others = eligible[np.isin(g.load_bus[eligible], near) & (eligible != first)]
        more = rng.choice(others, min(knobs.ramp_targets - 1, len(others)), replace=False)
        direction = 1.0 if rng.random() < 0.5 else -1.0
        rise = max(1, int(rng.uniform(0.2, 0.45) * ramp_len))
        hold = int(rng.uniform(0.0, 0.25) * ramp_len)
        return RampDesign(np.r_[first, more].astype(eligible.dtype), direction, rise, hold)

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
        """The slow ramp At starting at t whose every frame has a stealthy state inside its stealth
        bound; `shape` = (ramp_len, ramp_rate). Drawn up to ONSET_DRAWS times for a design with a state
        on every frame, and up to RAMP_SUPPORT_TRIES of those are tried for a support that keeps the
        bound (`_held`); None when none does, and the walk then moves the episode. With
        `knobs.min_tamper` the ramp carries its fewest-tamper support [WU26 eq. 12]; `prev` is the
        attack vector of the frame before t."""
        # neither the feasibility test nor the search spends a random draw
        ramp_len, rate = shape
        frames, tried = probe_frames(len(X), t, ramp_len), 0
        for _ in range(ONSET_DRAWS):
            design = self.draw_ramp(self.g.rng, ramp_len, knobs)
            steps = (self.ramp_step(design, u - t, rate) for u in frames)
            if not all(self.g.false_state.is_feasible(X[u], step, knobs) for u, step in zip(frames, steps)):
                continue
            held = self._held(design, X, (frames, t, rate), knobs, prev)
            tried += 1
            if held is not None or tried == RAMP_SUPPORT_TRIES:
                return held
        return None

    def _held(
        self,
        design: RampDesign,
        X: States,
        window: tuple[range, int, float],
        knobs: FrameKnobs,
        prev: Optional[AttackVector],
    ) -> Optional[RampDesign]:
        """The ramp on a support that keeps its stealth bound at every frame of its window: with
        `knobs.min_tamper` the fewest-tamper support and the search's result, else the design as drawn
        when the area its frames re-solve keeps the bound. None when no support does: a ramp is never
        run past its own bound."""
        frames, t, rate = window
        goal = LoadGoal(tuple(self.ramp_step(design, u - t, rate) for u in frames))
        states = [X[u] for u in frames]
        if knobs.min_tamper:
            result = self.g.minimizer.min_tamper(states, goal, knobs, prev)
            return None if result is None else design._replace(support=result.support, tamper=result)
        placed = self.g.false_state.with_support(goal.designs[0], knobs)  # the area the frames run on
        if placed is None or placed.support is None:
            return None
        kept = AttackWindow(self.g, states, goal, knobs, prev=prev).holds(placed.support)
        return design if kept else None

    def am_overload_design(
        self, states: np.ndarray, knobs: FrameKnobs, rng: Optional[np.random.Generator] = None
    ) -> Optional[AmOverloadDesign]:
        """One Am episode over the window `states` [L, N, 4]: the first target set, in a random order of the
        eligible branches, whose support meets the goal at every snapshot [WU26 eqs. 24-25] inside the limits
        [WU26 eqs. 21-23] and moves a device beyond noise; None when no set does. One draw from `rng` (None:
        the generator's own stream), if a branch exists."""
        window = [states[u] for u in range(len(states))]
        lines = self.g.overload.eligible_lines(window, knobs.hops)
        if len(lines) == 0:
            return None
        order = (self.g.rng if rng is None else rng).permutation(lines)
        for targets in self.g.overload.target_sets(order, knobs.n_lines, knobs.hops):
            goal = self.g.overload.overload_goal(window, *targets)
            result = self.g.minimizer.min_tamper(window, goal, knobs)
            # the search gives a flow goal no cost under one device beyond noise (`AttackWindow.least`):
            # a goal that moves no meter beyond its noise shows no overload
            if result is not None and result.devices >= 1:
                # eq. 25: S_max at the end
                ratings = tuple(float(s) for s in goal.targets_at(len(window) - 1))
                return AmOverloadDesign(goal, ratings, result.support, result)
        return None
