"""The records of a timeline's plan and design stages: a placed episode (`Slot`), a split and its
frames (`SplitSpan`), a placed episode on its way through the design stage (`Planned`), the attack
as it is written (`Episode`), what is placed (`Schedule`) and what the placement asked for and got
(`Placement`), and what a design worker rebuilds its generator from (`GeneratorSpec`)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Generic, Optional, TypeVar, Union

import numpy as np

from ..constants.families import FAMILY_CODE
from .choices import Split
from .config import MeterSettings

# an episode's family: the timeline's code, or the New York family's name
Family = TypeVar("Family", int, str)
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
    """One split of the timeline: its code (0 train, 1 val, 2 test, the row of `Split`) and its
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
class Planned(Generic[Family, Design]):
    """A placed episode not yet written: where it sits now, the `design` found at that onset (None
    until it has one; the walk designs the IEEE ramp At itself as it reaches it, since its stealth
    bound reads the frame written just before its onset), its `index` among the split's placed
    episodes and the `moves` made so far for want of a feasible design."""

    slot: Slot[Family]
    design: Optional[Design] = None
    index: int = 0
    moves: int = 0

    def moved_to(self, slot: Slot[Family]) -> Planned[Family, Design]:
        """The same episode after one more move, at `slot`."""
        return replace(self, slot=slot, moves=self.moves + 1)


@dataclass(frozen=True)
class Episode:
    """One attack over a window of `length` frames from `onset`: what a New York family designs and
    what every file's episode table holds a row of. `family` is its name ("Am", "At"). `buses` are the
    attacked buses (the pretended loads and generators, the localization label).

    A New York design also carries `clients`, the owners of the tampered meters (one or several: the
    attacker's local subnetwork may span utilities, G5), `target_branch`, the interface branch Am's
    area is anchored on (-1 for At), and `a [length, M]`, the attack vector added to each frame's
    benign scan: h(x_false) - h(x_true) in full, so the operator's one residual test over all meters
    sees noise (global stealth). The tampered meters, those it moves beyond their noise, come from
    `Attacker.tampered`. An IEEE episode has no clients and an empty `a`: its frames carry the attack."""

    family: str
    clients: np.ndarray
    onset: int
    length: int
    buses: np.ndarray
    target_branch: int
    a: np.ndarray


@dataclass
class Schedule:
    """What is placed on the timeline: the families, the episode lengths and the ramp's rate, and
    the attacked fraction the placement fills up to."""

    families: list[int]
    ramp_len: int
    ramp_rate: float
    am_len: int
    attacked_frac: float

    def length_of(self, fid: int) -> int:
        """The frames an episode of `fid` takes: the ramp length for At, `am_len` for Am."""
        return self.ramp_len if fid == FAMILY_CODE["At"] else self.am_len

    @property
    def lengths(self) -> dict[int, int]:
        """The episode length of every family, in the schedule's family order."""
        return {f: self.length_of(f) for f in self.families}


@dataclass
class Placement(Generic[Family]):
    """What the placement asked for and got, per split (rows, `Split`) and family (columns, the
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
        zero = np.zeros((len(Split), len(families)), int)
        return cls(families, requested=zero, dropped=zero.copy(), redraws=zero.copy(), built=zero.copy())

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
