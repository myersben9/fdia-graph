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

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..engine import FdiaGenerator
from ..engine.attacks.stealthy import AM_FAMILY
from ..errors import NoRoomForEpisode
from ..models.frames import AmOverloadDesign, FrameKnobs
from ..models.plan import Designed, GeneratorSpec, Pending, Placement, Slot, SplitSpan
from .plan import REDRAWS, free_onset, give_up

DESIGN_STREAM = 0xA3D5  # the key of the Am design draws, beside the jitter's (engine.measurement)
RELOCATE_STREAM = 0xA3D6  # the key of the onsets an infeasible Am moves to


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


_WORKER: list[FdiaGenerator] = []  # the generator a worker process builds once, at start


def _start_worker(spec: GeneratorSpec) -> None:
    _WORKER[:] = [build_generator(spec)]


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


def _design_in_worker(task: _DesignTask) -> Optional[AmOverloadDesign]:
    return _design(_WORKER[0], task)


class AmDesigner:
    """Designs the Am episodes of each split: in this process with `workers` 1, else in a pool of
    `workers` processes started once for the timeline (`close` stops it)."""

    def __init__(self, g: FdiaGenerator, spec: GeneratorSpec, workers: int) -> None:
        self.g, self.seed = g, spec.seed
        self._pool = (
            ProcessPoolExecutor(workers, initializer=_start_worker, initargs=(spec,)) if workers > 1 else None
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()

    def _run(self, tasks: list[_DesignTask]) -> list[Optional[AmOverloadDesign]]:
        if self._pool is None:
            return [_design(self.g, t) for t in tasks]
        return list(self._pool.map(_design_in_worker, tasks))

    def design_split(
        self,
        X: np.ndarray,
        knobs: FrameKnobs,
        span: SplitSpan,
        slots: list[Slot[int]],
        rec: Placement,
    ) -> tuple[list[Slot[int]], dict[int, AmOverloadDesign]]:
        """The placed episodes `slots` of the split `span` with every Am either designed (its design
        by onset) or moved until it is, or given up: rounds of designs in parallel, the failures of a
        round moved in slot order (so the moves do not depend on the workers) and designed again in
        the next."""
        others = [s for s in slots if s.family != AM_FAMILY]
        todo = [Pending(i, s, 0) for i, s in enumerate(slots) if s.family == AM_FAMILY]
        done: dict[int, Designed] = {}
        while todo:
            failed = self._round(X, knobs, span.code, todo, done)
            todo = self._move(failed, others, done, span, rec)
        built = [d.slot for d in done.values()]
        return sorted(others + built), {d.slot.onset: d.design for d in done.values()}

    def _round(
        self,
        X: np.ndarray,
        knobs: FrameKnobs,
        split: int,
        todo: list[Pending],
        done: dict[int, Designed],
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
        done: dict[int, Designed],
        span: SplitSpan,
        rec: Placement,
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
