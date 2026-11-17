"""The design stage of a timeline: the overload Am episodes of a split, designed before it is walked.

An Am design reads only the operating points of its own window (a flow goal has no stealth bound, the
plan's D11, so nothing ties it to the frame before its onset), and its one random draw, the order the
target lines are tried in, comes from a stream keyed by (seed, split, slot, move) rather than from the
walk's. So the episodes of a split are independent: they are designed in parallel across processes
(`workers`), and the result is the same whatever the number of workers. An episode with no feasible
design at its onset moves to another onset of its split, uniform among every position where it fits
without touching another placed episode (earlier or later: no frame is written yet), up to REDRAWS
times; past that it is given up, recorded and warned about.

The ramp At is designed by the walk as it reaches it (`generation.emit`): its stealth bound starts
from the attack vector of the frame before its onset, which only the walk knows.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import Pool
from typing import Generic, Optional, TypeVar, cast

import numpy as np

from ..engine import FdiaGenerator
from ..engine.attacks.stealthy import AM_FAMILY
from ..errors import NoRoomForEpisode
from ..models.frames import AmOverloadDesign, FrameKnobs
from ..models.plan import Designed, GeneratorSpec, Pending, Placement, Slot, SplitSpan
from .plan import REDRAWS, free_onset, give_up

DESIGN_STREAM = 0xA3D5  # the key of the Am design draws, beside the jitter's (engine.measurement)
RELOCATE_STREAM = 0xA3D6  # the key of the onsets an infeasible Am moves to
Setup = TypeVar("Setup")  # what a design process is started from (picklable)
State = TypeVar("State")  # what it makes from that once, and designs with
Task = TypeVar("Task")  # what one design reads (a grid's own record)
Found = TypeVar("Found")  # what one design returns


def build_generator(spec: GeneratorSpec) -> FdiaGenerator:
    """The generator of `spec`: its case, meter plan and ratings."""
    g = FdiaGenerator(
        spec.system,
        seed=spec.seed,
        max_load_mw=spec.max_load_mw,
        meter_model=spec.meters.meter_model,
        vbus_frac=spec.meters.vbus_frac,
        pmu_frac=spec.meters.pmu_frac,
        flow_frac=spec.meters.flow_frac,
    )
    g._line_ratings = spec.ratings
    g._rating_delta = spec.rating_delta
    return g


_STATE: list[object] = []  # what this worker process designs with, made once when the process starts


def _start_worker(start: Callable[[Setup], State], setup: Setup) -> None:
    """Run in a worker process as it starts: keep what `start` makes for the designs to come."""
    _STATE[:] = [start(setup)]


def _design_in_worker(design: Callable[[State, Task], Found], task: Task) -> Found:
    """One design in a worker process, with what the process made at its start."""
    return design(cast(State, _STATE[0]), task)


class DesignPool(Generic[Setup, State]):
    """The processes a timeline's designs run on, for either grid. Each of the `workers` processes is
    started once for the timeline and makes what it designs with by `start(setup)` (a generator, the
    attack families). With one worker no process is started: the pool calls `start(setup)` itself and
    the designs run in the caller's process. Every design draws from a stream keyed by its own
    episode, so the designs are the same whatever the number of workers.

    The processes are spawned (not forked) on every platform, so a script that asks for more than one
    worker must start under `if __name__ == "__main__":`."""

    def __init__(self, workers: int, start: Callable[[Setup], State], setup: Setup) -> None:
        self._pool: Optional[Pool] = None
        self._local: list[State] = []
        if workers > 1:
            self._pool = mp.get_context("spawn").Pool(workers, _start_worker, (start, setup))
        else:
            self._local = [start(setup)]

    def map(self, design: Callable[[State, Task], Found], tasks: Iterable[Task]) -> Iterator[Found]:
        """`design(state, task)` of every task, in the order of `tasks`, each as soon as it and those
        before it are done; `state` is what `start` made in the process that runs the design."""
        if self._pool is None:
            return (design(self._local[0], task) for task in tasks)
        return self._pool.imap(partial(_design_in_worker, design), tasks)

    def close(self, finished: bool = True) -> None:
        """Stop the worker processes: after the work at hand when the run `finished`, at once (the
        designs in flight dropped) when it failed."""
        if self._pool is None:
            return
        if finished:
            self._pool.close()
        else:
            self._pool.terminate()
        self._pool.join()


@dataclass(frozen=True)
class _WorkerSetup:
    """What a design process makes its generator from: the spec to rebuild it, or, when the designs
    run in the walk's own process, the walk's generator itself."""

    spec: GeneratorSpec
    g: Optional[FdiaGenerator]


def _worker_generator(setup: _WorkerSetup) -> FdiaGenerator:
    return build_generator(setup.spec) if setup.g is None else setup.g


@dataclass(frozen=True)
class _DesignTask:
    """One Am design to try: the window's operating points [length, N, 4], the knobs, and the key of
    its random draw (seed, split, slot, move)."""

    window: np.ndarray
    knobs: FrameKnobs
    key: tuple[int, int, int, int]


def _design(g: FdiaGenerator, task: _DesignTask) -> Optional[AmOverloadDesign]:
    rng = np.random.default_rng([task.key[0], DESIGN_STREAM, *task.key[1:]])
    return g.am_overload_design(task.window, 0, len(task.window), task.knobs, rng)


AmEpisode = Designed[int, AmOverloadDesign]  # an IEEE episode after the design stage


class AmDesigner:
    """Designs the Am episodes of each split: in this process with `workers` 1 (on the walk's own
    generator), else in a pool of `workers` processes started once for the timeline, each rebuilding
    the generator from `spec` (`close` stops them)."""

    def __init__(self, g: FdiaGenerator, spec: GeneratorSpec, workers: int) -> None:
        self.g, self.seed = g, spec.seed
        self._pool = DesignPool(workers, _worker_generator, _WorkerSetup(spec, None if workers > 1 else g))

    def close(self, finished: bool = True) -> None:
        """Stop the design processes (at once when the run failed)."""
        self._pool.close(finished)

    def _run(self, tasks: list[_DesignTask]) -> list[Optional[AmOverloadDesign]]:
        return list(self._pool.map(_design, tasks))

    def design_split(
        self,
        X: np.ndarray,
        knobs: FrameKnobs,
        span: SplitSpan,
        slots: list[Slot[int]],
        rec: Placement[int],
    ) -> list[AmEpisode]:
        """The placed episodes `slots` of the split `span`, sorted by onset, with every Am either
        designed (its design beside its slot) or moved until it is, or given up: rounds of designs in
        parallel, the failures of a round moved in slot order (so the moves do not depend on the
        workers) and designed again in the next. A ramp At comes back with no design: the walk designs
        it as it reaches it."""
        others = [s for s in slots if s.family != AM_FAMILY]
        todo = [Pending(i, s, 0) for i, s in enumerate(slots) if s.family == AM_FAMILY]
        done: dict[int, AmEpisode] = {}
        while todo:
            failed = self._round(X, knobs, span.code, todo, done)
            todo = self._move(failed, others, done, span, rec)
        ramps: list[AmEpisode] = [Designed(s, None) for s in others]
        return sorted(ramps + list(done.values()), key=lambda d: d.slot)

    def _round(
        self,
        X: np.ndarray,
        knobs: FrameKnobs,
        split: int,
        todo: list[Pending],
        done: dict[int, AmEpisode],
    ) -> list[Pending]:
        """One round: every pending episode designed at its onset (in parallel), the designed ones
        added to `done`; returns those with no design."""
        tasks = [
            _DesignTask(
                X[p.slot.onset : p.slot.onset + p.slot.length], knobs, (self.seed, split, p.index, p.moves)
            )
            for p in todo
        ]
        failed = []
        for p, design in zip(todo, self._run(tasks)):
            if design is None:
                failed.append(p)
            else:
                done[p.index] = Designed(p.slot, design)
        return failed

    def _move(
        self,
        failed: list[Pending],
        others: list[Slot[int]],
        done: dict[int, AmEpisode],
        span: SplitSpan,
        rec: Placement[int],
    ) -> list[Pending]:
        """The failed episodes at their new onsets, in slot order, each placed clear of every episode
        still held (the ramps `others`, the designed Am, the failed ones not yet moved and those moved
        before it); an episode past REDRAWS moves, or with no free onset left, is given up."""
        waiting = {p.index: p.slot for p in failed}
        moved: list[Pending] = []
        for p in failed:
            del waiting[p.index]
            taken = (
                others + [d.slot for d in done.values()] + list(waiting.values()) + [m.slot for m in moved]
            )
            onset = self._new_onset(p, taken, span) if p.moves < REDRAWS else None
            if onset is None:
                give_up(span.code, AM_FAMILY, p.moves)
                continue
            rec.redraws[span.code, rec.col(AM_FAMILY)] += 1
            moved.append(p.moved_to(p.slot.at(onset)))
        return moved

    def _new_onset(self, p: Pending, taken: list[Slot[int]], span: SplitSpan) -> Optional[int]:
        """A free onset of the split for the failed episode `p`, from the stream keyed by its split,
        index and move; None when no onset is left."""
        rng = np.random.default_rng([self.seed, RELOCATE_STREAM, span.code, p.index, p.moves])
        try:
            return free_onset(rng, p.slot.length, taken, span)
        except NoRoomForEpisode:
            return None
