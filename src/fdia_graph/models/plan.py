"""The records of a timeline's plan and design stages: a placed episode (`Slot`), a split and its
frames (`SplitSpan`), an episode not yet written (`Pending`) or with its design (`Designed`), what is
placed (`Schedule`) and what the placement asked for and got (`Placement`), and what a design worker
rebuilds its generator from (`GeneratorSpec`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar, Union

import numpy as np

from .choices import FAMILY_CODE, SPLITS
from .config import MeterSettings

Family = TypeVar(
    "Family", int, str
)  # an episode's family: the timeline's code, or the New York family's name
Design = TypeVar("Design")  # what a design stage finds for one episode (each grid has its own record)


@dataclass(frozen=True, order=True)
class Slot(Generic[Family]):
    """One placed episode: its first frame `onset`, its `family` (the timeline's code, or the New
    York family's name) and its `length` in frames. Slots sort by onset."""

    onset: int
    family: Family
    length: int

    def at(self, onset: int) -> Slot[Family]:
        """The same episode at another onset."""
        return Slot(onset, self.family, self.length)


@dataclass(frozen=True)
class SplitSpan:
    """One split of the timeline: its code (0 train, 1 val, 2 test, the row of `SPLITS`) and its
    frames. A split cut as one block in time order is every frame of [start, stop). A split made of
    whole weeks lists its frames in `frames` (in time order), and [start, stop) is then only the stretch
    of the timeline they lie in. Two spans are equal when their code and stretch are."""

    code: int
    start: int
    stop: int
    frames: Optional[np.ndarray] = field(default=None, compare=False)

    def index(self) -> np.ndarray:
        """The split's frames on the timeline, in time order."""
        return np.arange(self.start, self.stop) if self.frames is None else self.frames

    @property
    def size(self) -> int:
        """How many frames the split holds."""
        return len(self.index())

    def onsets_fit(self, length: int) -> np.ndarray:
        """[size] bool: the positions in the split where an episode of `length` frames may start, its
        frames being consecutive on the timeline (so it never crosses from one week block into the
        next, and never runs past the split's end)."""
        index = self.index()
        fits = np.zeros(len(index), bool)
        n = len(index) - length + 1
        if n > 0:
            fits[:n] = index[length - 1 :] - index[:n] == length - 1
        return fits


@dataclass(frozen=True)
class Pending:
    """An episode not yet written: its index among the split's placed episodes, where it sits now, and
    the moves made so far (the count belongs to its episode). The design stage holds the overload
    episodes still without a design in it; the walk holds every episode it has not emitted yet."""

    index: int
    slot: Slot[int]
    moves: int

    def moved_to(self, slot: Slot[int]) -> Pending:
        """The same episode after one more move, at `slot`."""
        return Pending(self.index, slot, self.moves + 1)


@dataclass(frozen=True)
class Designed(Generic[Family, Design]):
    """A placed episode after the design stage: where it sits and the design found at that onset.
    `design` is None for an episode the walk designs itself as it reaches it (the IEEE ramp At, whose
    stealth bound reads the frame written just before its onset)."""

    slot: Slot[Family]
    design: Optional[Design]


@dataclass
class Schedule:
    """What is placed on the timeline: the families, the episode lengths and the ramp's rate, and
    the attacked fraction the placement fills up to."""

    families: list[int]
    ramp_len: int
    ramp_rate: float
    am_len: int
    attacked_frac: float

    @classmethod
    def build(
        cls, fams: list[int], ramp_len: int, ramp_rate: float, am_len: int, attacked_frac: float
    ) -> Schedule:
        """The schedule of the families `fams` (codes) at these lengths, rate and attacked fraction."""
        return cls(fams, ramp_len, ramp_rate, am_len, attacked_frac)

    def length_of(self, fid: int) -> int:
        """The frames an episode of `fid` takes: the ramp length for At, `am_len` for Am."""
        return self.ramp_len if fid == FAMILY_CODE["At"] else self.am_len

    @property
    def lengths(self) -> dict[int, int]:
        """The episode length of every family, in the schedule's family order."""
        return {f: self.length_of(f) for f in self.families}


@dataclass
class Placement(Generic[Family]):
    """What the placement asked for and got, per split (rows, `SPLITS`) and family (columns, the
    timeline's families): episodes requested, dropped because they would not fit, moved to
    another onset because no design was feasible, and built."""

    families: list[Family]
    requested: np.ndarray
    dropped: np.ndarray
    redraws: np.ndarray
    built: np.ndarray

    @classmethod
    def empty(cls, families: list[Family]) -> Placement[Family]:
        """The record of `families` with every count at zero."""
        shape = (len(SPLITS), len(families))
        return cls(
            families, np.zeros(shape, int), np.zeros(shape, int), np.zeros(shape, int), np.zeros(shape, int)
        )

    def col(self, fid: Family) -> int:
        """The column of family `fid`."""
        return self.families.index(fid)


@dataclass(frozen=True)
class GeneratorSpec:
    """What a worker process needs to rebuild the walk's generator (`generation.design`): the case and
    its meter plan (both fixed by the seed) and the line ratings the parent set over the whole pool,
    or the step of the per-window "delta" ratings (MVA)."""

    system: Union[int, str]
    seed: int
    max_load_mw: Optional[float]
    meters: MeterSettings
    ratings: Optional[np.ndarray]
    rating_delta: Optional[float] = None
