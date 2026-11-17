"""The search-based families: the slow ramp At and the overload Am [WU26].

Both re-solve the attacker's local false state on every scan of an episode (`RampEpisode`,
`OverloadEpisode`), where a New York family stores one attack vector for its window
(`family.VectorEpisode`). An overload is designed from its window alone, before its split is walked.
A ramp is designed as the walk reaches its onset (`RampEpisode.begin`): its stealth bound reads the
frame written just before, and its draws come from the generator's own stream at that moment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ...constants.families import AM_CODE, AT_CODE
from ...models.frames import AmOverloadDesign, AttackVector, Frame, FrameKnobs, RampDesign
from ...models.plan import EpisodeRecord
from ...systems.states import States
from .family import DesignedEpisode, EpisodeFamily

if TYPE_CHECKING:
    from ..core import FdiaGenerator


class RampEpisode(DesignedEpisode):
    """One slow-ramp At episode of `length` frames growing by `rate` a frame: a fixed load set scaled
    along a rise, a hold and a return, each scan re-solved locally (`AttackMixin.attack_frame`)."""

    code = AT_CODE
    design: RampDesign  # drawn by `begin`

    def __init__(self, length: int, rate: float, knobs: FrameKnobs) -> None:
        self.length, self.rate, self.knobs = length, rate, knobs

    def begin(self, g: FdiaGenerator, X: States, t: int, prev: AttackVector) -> bool:
        """Design the ramp at frame t (`EpisodeDesignMixin.ramp_design`, on the generator's own
        stream); False when no admissible ramp exists at this onset."""
        design = g.ramp_design(X, t, (self.length, self.rate), self.knobs, prev)
        if design is None:
            return False
        self.design = design
        return True

    def frame(self, g: FdiaGenerator, X: np.ndarray, i: int) -> Optional[Frame]:
        return g.attack_frame(X, AT_CODE, g.ramp_step(self.design, i, self.rate), self.knobs)

    def record(self, onset: int, length: int, buses: np.ndarray) -> EpisodeRecord:
        return EpisodeRecord(self.row(onset, length, buses), self.design.tamper)


class OverloadEpisode(DesignedEpisode):
    """One overload Am episode [WU26 eqs. 24-25]: the target branches' reported flow driven to their
    ratings over the window, each scan re-solved on the held support (`overload_step`). `record`
    reads, per target branch, the noiseless flow the last scan reached and the flow last emitted
    (MVA), so it comes after the scans."""

    code = AM_CODE

    def __init__(self, design: AmOverloadDesign, knobs: FrameKnobs) -> None:
        self.design, self.knobs = design, knobs
        self.reached = self.emitted = np.full(len(design.goal.lines), np.nan)

    def frame(self, g: FdiaGenerator, X: np.ndarray, i: int) -> Optional[Frame]:
        lines = list(self.design.goal.lines)
        frame, self.reached = g.overload_step(self.design, X, i, self.knobs)
        if frame is not None:
            self.emitted = np.hypot(frame.edge_x[lines, 0], frame.edge_x[lines, 1])
        return frame

    def record(self, onset: int, length: int, buses: np.ndarray) -> EpisodeRecord:
        goal, ratings = self.design.goal, self.design.ratings
        last = goal.targets_at(len(goal.targets) - 1)  # the goal at the window's end
        rows = [
            (b, ratings[j], last[j], float(self.reached[j]), float(self.emitted[j]))
            for j, b in enumerate(goal.lines)
        ]
        return EpisodeRecord(self.row(onset, length, buses), self.design.tamper, rows)


class SearchRamp(EpisodeFamily):
    """The slow ramp At. Its episode is designed by the walk, so every window has one."""

    def __init__(self, length: int, rate: float, knobs: FrameKnobs) -> None:
        self.length, self.rate, self.knobs = length, rate, knobs

    def designed(self, onset: int, X: np.ndarray, rng: np.random.Generator) -> RampEpisode:
        return RampEpisode(self.length, self.rate, self.knobs)


class SearchOverload(EpisodeFamily):
    """The overload Am on the generator `g`: the first target set, in a random order of the eligible
    branches, whose support meets the goal over the window (`EpisodeDesignMixin.am_overload_design`)."""

    def __init__(self, g: FdiaGenerator, knobs: FrameKnobs) -> None:
        self.g, self.knobs = g, knobs

    def designed(self, onset: int, X: np.ndarray, rng: np.random.Generator) -> Optional[OverloadEpisode]:
        design = self.g.am_overload_design(X, 0, len(X), self.knobs, rng)
        return None if design is None else OverloadEpisode(design, self.knobs)
