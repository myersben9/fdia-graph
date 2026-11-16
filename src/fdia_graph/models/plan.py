"""The records of a timeline's plan and design stages: a placed episode (`Slot`), a split and its
frames (`SplitSpan`), an episode not yet written (`Pending`) or designed (`Designed`), what is placed
(`Schedule`) and what the placement asked for and got (`Placement`), and what a design worker
rebuilds its generator from (`GeneratorSpec`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Union

import numpy as np

from .choices import FAMILY_CODE, SPLITS
from .config import MeterSettings
from .frames import AmOverloadDesign

Family = TypeVar(
    "Family", int, str
)  # an episode's family: the timeline's code, or the New York family's name


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
    frames [start, stop)."""

    code: int
    start: int
    stop: int


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
class Designed:
    """An Am episode with its design: where it sits and the design found at that onset."""

    slot: Slot[int]
    design: AmOverloadDesign


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


@dataclass
class Placement:
    """What the placement asked for and got, per split (rows, `SPLITS`) and family (columns, the
    schedule's families): episodes requested, dropped because they would not fit, moved to
    another onset because no design was feasible, and built."""

    families: list[int]
    requested: np.ndarray
    dropped: np.ndarray
    redraws: np.ndarray
    built: np.ndarray

    @classmethod
    def empty(cls, families: list[int]) -> Placement:
        """The record of `families` with every count at zero."""
        shape = (len(SPLITS), len(families))
        return cls(
            families, np.zeros(shape, int), np.zeros(shape, int), np.zeros(shape, int), np.zeros(shape, int)
        )

    def col(self, fid: int) -> int:
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
