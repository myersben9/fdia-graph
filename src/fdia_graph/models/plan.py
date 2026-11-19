"""The records of a timeline's plan and design stages: a placed episode (`Slot`), a split and its
frames (`SplitSpan`), a placed episode on its way through the design stage (`Planned`), the attack
as it is written (`Episode`, `EpisodeRecord`), what the placement asked for and got (`Placement`),
and what a design worker rebuilds its generator from (`GeneratorSpec`)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

import numpy as np

from .choices import Split
from .config import MeterSettings
from .frames import MinimizerResult

if TYPE_CHECKING:  # a designed episode and a network are a producer's: named for the type checker only
    from ..engine.attacks.family import DesignedEpisode
    from ..engine.physics import LoadSplit
    from ..systems.network import Network

Design = TypeVar("Design")  # what a design stage finds for one episode


@dataclass(frozen=True, order=True)
class Slot:
    """One placed episode: its first frame `onset`, its `family` by name ("At", "Am") and its
    `length` in frames. Slots sort by onset."""

    onset: int
    family: str
    length: int

    def at(self, onset: int) -> Slot:
        """The same episode at another onset."""
        return Slot(onset, self.family, self.length)


@dataclass(frozen=True)
class SplitSpan:
    """One split of the timeline: its code (0 train, 1 val, 2 test, the row of `Split`) and its
    frames. A split cut as one block in time order is every frame of [start, stop). A split made of
    whole weeks lists its frames in `frames` (in time order), and [start, stop) is then only the stretch
    of the timeline they lie in. `runs` [T], when the frames have a time, is the run of regularly
    spaced frames each timeline frame lies in (`generation.plan.time_runs`). Two spans are equal
    when their code and stretch are."""

    code: int
    start: int
    stop: int
    frames: Optional[np.ndarray] = field(default=None, compare=False)
    runs: Optional[np.ndarray] = field(default=None, compare=False)

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
        next, and never runs past the split's end) and, with `runs`, in one run of regularly spaced
        frames (so it never crosses a gap in time)."""
        index = self.index()
        fits = np.zeros(len(index), bool)
        n = len(index) - length + 1
        if n > 0:
            fits[:n] = index[length - 1 :] - index[:n] == length - 1
        if n > 0 and self.runs is not None:
            fits[:n] &= self.runs[index[length - 1 :]] == self.runs[index[:n]]
        return fits


@dataclass(frozen=True)
class Planned(Generic[Design]):
    """A placed episode not yet written: where it sits now, the `design` found at that onset (None
    until it has one), its `index` among the split's placed episodes, the `moves` made so far for
    want of a feasible design and the code of its `split`."""

    slot: Slot
    design: Optional[Design] = None
    index: int = 0
    moves: int = 0
    split: int = 0

    def moved_to(self, slot: Slot) -> Planned[Design]:
        """The same episode after one more move, at `slot`."""
        return replace(self, slot=slot, moves=self.moves + 1)


@dataclass(frozen=True)
class DesignTask:
    """One design to try: the family, the pool frame its window starts at, the true states of the
    window [L, N, 4], and the key of its random stream."""

    family: str
    onset: int
    window: np.ndarray
    key: tuple[int, ...]


@dataclass(frozen=True)
class Tried:
    """What one design attempt came to: the episode or None, the seconds it took, and why there is
    none (a miss reason of `engine.attacks.family`; None with an episode or from a family that gives
    none). `designed` is False for an attempt not made, its window touching an unsolved frame."""

    episode: Optional[DesignedEpisode] = None
    seconds: float = 0.0
    miss: Optional[str] = None
    designed: bool = True


@dataclass(frozen=True)
class Episode:
    """One attack over a window of `length` frames from `onset`: what a New York family designs and
    what every file's episode table holds a row of. An IEEE episode has no clients and an empty `a`:
    its frames carry the attack."""

    family: str  # its name ("Am", "At")
    # the owners of the tampered meters (New York; the attacker's subnetwork may span utilities)
    clients: np.ndarray
    onset: int
    length: int
    buses: np.ndarray  # the attacked buses (the pretended loads and generators, the localization label)
    target_branch: int  # the interface branch Am's area is anchored on (-1 for At)
    # [length, M] the attack vector added to each frame's benign scan: h(x_false) - h(x_true) in full,
    # so the operator's residual test over all meters sees noise. `Attacker.tampered` names the
    # meters it moves beyond their noise.
    a: np.ndarray


@dataclass(frozen=True)
class EpisodeRecord:
    """What a file records of one emitted episode: its row of the episode table, what the fewest-tamper
    search chose for it (None: the search did not run) and, for an overload Am, one row per target
    line: branch, rating, goal, noiseless flow reached, emitted flow (MVA)."""

    episode: Episode
    tamper: Optional[MinimizerResult] = None
    lines: Sequence[tuple[int, float, float, float, float]] = ()


@dataclass
class Placement:
    """What the placement asked for and got, per split (rows, `Split`) and family (columns, the
    timeline's families by name): episodes requested, dropped because they would not fit, moved to
    another onset because no design was feasible, and built."""

    families: list[str]
    requested: np.ndarray
    dropped: np.ndarray
    redraws: np.ndarray
    built: np.ndarray

    @classmethod
    def empty(cls, families: list[str]) -> Placement:
        """The record of `families` with every count at zero."""
        zero = np.zeros((len(Split), len(families)), int)
        return cls(families, requested=zero, dropped=zero.copy(), redraws=zero.copy(), built=zero.copy())

    def col(self, fid: str) -> int:
        """The column of family `fid`."""
        return self.families.index(fid)


@dataclass(frozen=True)
class GeneratorSpec:
    """What a worker process needs to rebuild the walk's generator (`generation.design`): the grid,
    its meter plan (fixed by the seed) and the line ratings the parent set over the whole pool, or
    the step of the per-window "delta" ratings (MVA)."""

    net: Network
    seed: int
    max_load_mw: Optional[float]
    meters: MeterSettings
    ratings: Optional[np.ndarray]
    rating_delta: Optional[float] = None
    split: Optional[LoadSplit] = None  # how the pool's injections divide (None: a scaled pool)
