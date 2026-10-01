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

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..engine import FdiaGenerator
from ..engine.records import AM_FAMILY
from ..errors import NoRoomForEpisode
from ..models.config import MeterSettings
from ..models.frames import AmOverloadDesign, FrameKnobs
from .plan import REDRAWS, Slot, _Placement, free_onset, give_up

DESIGN_STREAM = 0xA3D5  # the key of the Am design draws, beside the jitter's (engine.measurement)
RELOCATE_STREAM = 0xA3D6  # the key of the onsets an infeasible Am moves to


@dataclass(frozen=True)
class _GeneratorSpec:
    """What a worker process needs to rebuild the walk's generator: the case and its meter plan (both
    fixed by the seed) and the line ratings the parent set over the whole pool."""

    system: Union[int, str]
    seed: int
    max_load_mw: Optional[float]
    meters: MeterSettings
    ratings: Optional[np.ndarray]

    def build(self) -> FdiaGenerator:
        g = FdiaGenerator(
            self.system,
            seed=self.seed,
            max_load_mw=self.max_load_mw,
            meter_model=self.meters.meter_model,
            vbus_frac=self.meters.vbus_frac,
            pmu_frac=self.meters.pmu_frac,
            flow_frac=self.meters.flow_frac,
        )
        g._line_ratings = self.ratings
        return g


_WORKER: list[FdiaGenerator] = []  # the generator a worker process builds once, at start


def _start_worker(spec: _GeneratorSpec) -> None:
    _WORKER[:] = [spec.build()]


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

    def __init__(self, g: FdiaGenerator, spec: _GeneratorSpec, workers: int) -> None:
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
        split: int,
        placed: tuple[list[Slot], tuple[int, int]],
        rec: _Placement,
    ) -> tuple[list[Slot], dict[int, AmOverloadDesign]]:
        """The split's placed episodes with every Am either designed (its design by onset) or moved
        until it is, or given up: rounds of designs in parallel, the failures of a round moved in
        slot order (so the moves do not depend on the workers) and designed again in the next."""
        slots, span = placed
        others = [s for s in slots if s[1] != AM_FAMILY]
        todo = [(i, s, 0) for i, s in enumerate(slots) if s[1] == AM_FAMILY]  # (slot, where, moves)
        done: dict[int, tuple[Slot, AmOverloadDesign]] = {}
        while todo:
            failed = self._round(X, knobs, split, todo, done)
            todo = self._move(failed, (others, done, todo), split, span, rec)
        built = [s for s, _ in done.values()]
        return sorted(others + built), {s[0]: d for s, d in done.values()}

    def _round(
        self,
        X: np.ndarray,
        knobs: FrameKnobs,
        split: int,
        todo: list[tuple[int, Slot, int]],
        done: dict[int, tuple[Slot, AmOverloadDesign]],
    ) -> list[tuple[int, Slot, int]]:
        """One round: every pending episode designed at its onset (in parallel), the designed ones
        added to `done`; returns those with no design."""
        tasks = [_DesignTask(X[s[0] : s[0] + s[2]], knobs, (self.seed, split, i, m)) for i, s, m in todo]
        failed = []
        for (i, s, m), design in zip(todo, self._run(tasks)):
            if design is None:
                failed.append((i, s, m))
            else:
                done[i] = (s, design)
        return failed

    def _move(
        self,
        failed: list[tuple[int, Slot, int]],
        held: tuple[list[Slot], dict[int, tuple[Slot, AmOverloadDesign]], Sequence[tuple[int, Slot, int]]],
        split: int,
        span: tuple[int, int],
        rec: _Placement,
    ) -> list[tuple[int, Slot, int]]:
        """The failed episodes at their new onsets, in slot order, each placed clear of every episode
        still held (the ramps, the designed Am, the failed ones not yet moved and those moved before it);
        an episode past REDRAWS moves, or with no free onset left, is given up."""
        others, done, _ = held
        pending = {i: s for i, s, _ in failed}
        moved: list[tuple[int, Slot, int]] = []
        for i, s, m in failed:
            del pending[i]
            taken = others + [d[0] for d in done.values()] + list(pending.values()) + [x[1] for x in moved]
            onset = self._new_onset(s, taken, (split, i, m), span) if m < REDRAWS else None
            if onset is None:
                give_up(split, AM_FAMILY, m)
                continue
            rec.redraws[split, rec.col(AM_FAMILY)] += 1
            moved.append((i, (onset, s[1], s[2]), m + 1))
        return moved

    def _new_onset(
        self, s: Slot, taken: list[Slot], key: tuple[int, int, int], span: tuple[int, int]
    ) -> Optional[int]:
        rng = np.random.default_rng([self.seed, RELOCATE_STREAM, *key])
        try:
            return free_onset(rng, s[2], taken, span)
        except NoRoomForEpisode:
            return None
