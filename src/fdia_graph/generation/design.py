"""The design stage of a timeline: every placed episode of a split gets its design before the split
is walked, from its family (`engine.attacks.family.EpisodeFamily.designed`).

A design reads only the operating points of its own window, and its random draws come from a stream
keyed by its own episode. So the episodes of a split are independent and are designed in parallel
(`Designer`, on a `DesignPool`), with the same result for any number of workers. An episode with no
design moves to another free onset of its split, up to REDRAWS times (`SplitDesign`). The search
ramp At is the one family whose episode is designed later, as the walk reaches it
(`engine.attacks.search.RampEpisode`).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from multiprocessing.pool import Pool
from typing import Generic, Optional, TypeVar, cast

import numpy as np

from ..constants.families import AM, AT
from ..constants.streams import DESIGN_STREAM, RELOCATE_STREAM
from ..engine.attacks.family import DesignedEpisode, EpisodeFamily
from ..engine.attacks.search import SearchOverload, SearchRamp
from ..engine.core import FdiaGenerator
from ..errors import ConfigError, NoRoomForEpisode
from ..models.frames import FrameKnobs
from ..models.plan import Design, DesignTask, GeneratorSpec, Placement, Planned, Slot, SplitSpan, Tried
from .plan import REDRAWS, free_onset, give_up

Setup = TypeVar("Setup")  # what a design process is started from (picklable)
State = TypeVar("State")  # what it makes from that once, and designs with
Task = TypeVar("Task")  # what one design reads
Found = TypeVar("Found")  # what one design returns
Families = Mapping[str, EpisodeFamily]  # the families a process designs with, by name


_worker_state: list[object] = []  # what this worker process designs with, made once at its start


class DesignPool(Generic[Setup, State]):
    """The processes a timeline's designs run on, for either grid. Each of the `workers` processes
    starts once and makes what it designs with by `start(setup)`; with one worker the designs run in
    the caller's process. Every design draws from a stream keyed by its own episode, so the result
    does not depend on the number of workers. The processes are spawned, so a script that asks for
    more than one worker must start under `if __name__ == "__main__":`."""

    def __init__(self, workers: int, start: Callable[[Setup], State], setup: Setup) -> None:
        self._pool: Optional[Pool] = None
        self._local: list[State] = []
        if workers > 1:
            self._pool = mp.get_context("spawn").Pool(workers, self._start_worker, (start, setup))
        else:
            self._local = [start(setup)]

    @staticmethod
    def _start_worker(start: Callable[[Setup], State], setup: Setup) -> None:
        """Run in a worker process as it starts: keep what `start` makes for the designs to come."""
        _worker_state[:] = [start(setup)]

    @staticmethod
    def _design_in_worker(design: Callable[[State, Task], Found], task: Task) -> Found:
        """One design in a worker process, with what the process made at its start."""
        return design(cast(State, _worker_state[0]), task)

    def map(self, design: Callable[[State, Task], Found], tasks: Iterable[Task]) -> Iterator[Found]:
        """`design(state, task)` of every task, lazily and in the order of `tasks`. `state` is what
        `start` made in the process that runs the design."""
        if self._pool is None:
            return (design(self._local[0], task) for task in tasks)
        return self._pool.imap(partial(self._design_in_worker, design), tasks)

    def close(self, finished: bool = True) -> None:
        """Stop the worker processes: after the work at hand when the run `finished`, at once when it
        failed."""
        if self._pool is None:
            return
        if finished:
            self._pool.close()
        else:
            self._pool.terminate()
        self._pool.join()


class Designer(ABC, Generic[Setup]):
    """The design stage of one source: its families, made in each of the `workers` processes by
    `start(setup)`, and what every design attempt of the run came to (`log`). `known` holds designs
    made earlier by (family, onset), reused as they are (the New York survey's). A grid says how a
    placed episode becomes a task (`task`)."""

    def __init__(self, workers: int, start: Callable[[Setup], Families], setup: Setup, seed: int) -> None:
        self.workers, self.start, self.setup, self.seed = workers, start, setup, seed
        self.known: dict[tuple[str, int], Tried] = {}
        self.log: list[tuple[Slot, Tried]] = []
        self._pool: Optional[DesignPool[Setup, Families]] = None

    def open(self) -> None:
        """Start a run: an empty log and the design processes."""
        self.log = []
        self._pool = DesignPool(self.workers, self.start, self.setup)

    def close(self, finished: bool) -> None:
        """Stop the design processes (`DesignPool.close`)."""
        if self._pool is not None:
            self._pool.close(finished)
            self._pool = None

    @abstractmethod
    def task(self, p: Planned[DesignedEpisode]) -> Optional[DesignTask]:
        """The design of the placed episode `p` to try at its onset; None when its window cannot be
        designed on."""

    @staticmethod
    def _design(families: Families, task: DesignTask) -> Tried:
        """One design, by its family, from the stream of its key."""
        start, family = time.perf_counter(), families[task.family]
        episode = family.designed(task.onset, task.window, np.random.default_rng(task.key))
        return Tried(episode, time.perf_counter() - start, family.last_miss)

    def tried(self, todo: Sequence[Planned[DesignedEpisode]]) -> list[Tried]:
        """What the design of each placed episode came to at its onset, in order."""
        assert self._pool is not None, "open() starts the design processes"
        out: dict[int, Tried] = {}
        fresh: list[tuple[int, DesignTask]] = []
        for i, task in enumerate(map(self.task, todo)):
            if task is None:
                out[i] = Tried(designed=False)
            elif (task.family, task.onset) in self.known:
                out[i] = self.known[task.family, task.onset]
            else:
                fresh.append((i, task))
        for (i, _), tried in zip(fresh, self._pool.map(self._design, [task for _, task in fresh])):
            out[i] = tried
        return [out[i] for i in range(len(todo))]

    def designs(self, todo: list[Planned[DesignedEpisode]]) -> list[Optional[DesignedEpisode]]:
        """One round of the design stage: the design of every pending episode at its onset, None
        where it has none; every attempt is added to `log`."""
        tried = self.tried(todo)
        self.log += zip((p.slot for p in todo), tried)
        return [t.episode for t in tried]


@dataclass(frozen=True)
class _SearchSetup:
    """What a design process makes the search-based families from: the spec to rebuild the generator,
    or, when the designs run in the walk's own process, the walk's generator itself; the knobs; and
    the ramp's (length, rate)."""

    spec: GeneratorSpec
    g: Optional[FdiaGenerator]
    knobs: FrameKnobs
    ramp: tuple[int, float]

    def families(self) -> Families:
        """The slow ramp At and the overload Am, on this process's generator."""
        g = FdiaGenerator.from_spec(self.spec) if self.g is None else self.g
        return {AT: SearchRamp(*self.ramp, self.knobs), AM: SearchOverload(g, self.knobs)}


class SearchDesigner(Designer[_SearchSetup]):
    """The design stage of the search-based families on the pool `X` [T, N, 4]: in this process on
    the walk's own generator with one worker, else in `workers` processes that each rebuild it from
    `spec`. One designer serves one timeline: the walk uses up the generator's random stream."""

    def __init__(
        self,
        g: FdiaGenerator,
        spec: GeneratorSpec,
        knobs: FrameKnobs,
        X: np.ndarray,
        ramp: tuple[int, float],
        workers: int,
    ) -> None:
        setup = _SearchSetup(spec, None if workers > 1 else g, knobs, ramp)
        super().__init__(workers, _SearchSetup.families, setup, spec.seed)
        self.X, self._used = X, False

    def open(self) -> None:
        """Start the one run; a second is refused."""
        if self._used:
            raise ConfigError("an IeeeSource generates one timeline; build a new source for another")
        self._used = True
        super().open()

    def task(self, p: Planned[DesignedEpisode]) -> DesignTask:
        """The window of `p` and the key of its draws: (seed, split, slot, move)."""
        s, key = p.slot, (self.seed, DESIGN_STREAM, p.split, p.index, p.moves)
        return DesignTask(s.family, s.onset, self.X[s.onset : s.onset + s.length], key)


class SplitDesign(Generic[Design]):
    """The design stage of the split `span`, for either grid. Every placed episode is designed at
    its onset; one with no design moves to another free onset of its split (uniform among those its
    family's mask in `masks` allows, from the stream keyed by seed, split, slot and move), up to
    REDRAWS times, then it is given up. `rec` records the moves."""

    def __init__(
        self,
        seed: int,
        span: SplitSpan,
        rec: Placement,
        masks: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        self.seed, self.span, self.rec, self.masks = seed, span, rec, masks or {}

    def run(
        self, slots: list[Slot], designs: Callable[[list[Planned[Design]]], list[Optional[Design]]]
    ) -> list[Planned[Design]]:
        """The episodes of the split, sorted by onset: each designed by `designs` (one round: the
        design of every pending episode at its onset, None where it has none), moved until it is, or
        given up. The failures of a round are moved in slot order, so the moves do not depend on the
        workers."""
        todo: list[Planned[Design]] = [Planned(s, index=i, split=self.span.code) for i, s in enumerate(slots)]
        done: dict[int, Planned[Design]] = {}
        while todo:
            failed = []
            for p, design in zip(todo, designs(todo)):
                if design is None:
                    failed.append(p)
                else:
                    done[p.index] = replace(p, design=design)
            todo = self._move(failed, done)
        return sorted(done.values(), key=lambda d: d.slot)

    def _move(self, failed: list[Planned[Design]], done: dict[int, Planned[Design]]) -> list[Planned[Design]]:
        """The failed episodes at their new onsets, in slot order, each clear of every other episode
        of the split. An episode past REDRAWS moves, or with no free onset left, is given up."""
        span, rec = self.span, self.rec
        waiting = {p.index: p.slot for p in failed}
        moved: list[Planned[Design]] = []
        for p in failed:
            del waiting[p.index]
            held = [d.slot for d in done.values()] + list(waiting.values()) + [m.slot for m in moved]
            onset = self._new_onset(p, held) if p.moves < REDRAWS else None
            if onset is None:
                give_up(span.code, p.slot.family, p.moves)
                continue
            rec.redraws[span.code, rec.col(p.slot.family)] += 1
            moved.append(p.moved_to(p.slot.at(onset)))
        return moved

    def _new_onset(self, p: Planned[Design], taken: list[Slot]) -> Optional[int]:
        """A free onset of the split for the failed episode `p`, from the stream keyed by its split,
        index and move; None when no onset is left."""
        rng = np.random.default_rng([self.seed, RELOCATE_STREAM, self.span.code, p.index, p.moves])
        try:
            return free_onset(rng, p.slot.length, taken, self.span, self.masks.get(p.slot.family))
        except NoRoomForEpisode:
            return None
