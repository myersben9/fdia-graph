"""The plan stage of a timeline: where its splits are cut and where each split's episodes go.

A timeline is first cut chronologically into train, val and test (`split_bounds`); then each split
gets the whole number of episodes whose frames come closest to its attacked fraction
(`episode_counts`), shared between the families by largest remainder and placed at uniform random
onsets inside the split, never overlapping and never cut (`place_split`). An episode whose design
turns out infeasible is moved to another free onset of its split (`free_onset`). Nothing here
touches the grid: the plan is positions and counts, the design stage decides what each episode
attacks (`generation.design`) and the emit stage writes the frames (`generation.emit`).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..dataset.base import FAMILIES
from ..engine.records import AM_FAMILY, RAMP_FAMILY
from ..errors import NoRoomForEpisode

SPLITS = ("train", "val", "test")  # the order of the per-split attributes and of data/split's codes
PLACE_TRIES = 10  # whole-split placements tried before one episode is dropped (the split-first packing rule)
REDRAWS = 20  # onsets an episode with no feasible design is moved to before it is given up (the split-first redraw rule)

Slot = tuple[int, int, int]  # one placed episode: (onset in timeline frames, family code, length)


@dataclass
class _Schedule:
    """What is placed on the timeline: the families in rotation, weighted by the inverse of their
    episode length so every family gets about the same share of attacked frames, the episode
    lengths and the ramp's rate, and the attacked fraction the placement fills up to."""

    families: list[int]
    weights: np.ndarray
    ramp_len: int
    ramp_rate: float
    am_len: int
    attacked_frac: float

    @classmethod
    def build(
        cls, fams: list[int], ramp_len: int, ramp_rate: float, am_len: int, attacked_frac: float
    ) -> _Schedule:
        length = {RAMP_FAMILY: float(ramp_len), AM_FAMILY: float(am_len)}
        w = np.array([1.0 / length[f] for f in fams], float)
        return cls(fams, w / w.sum() if len(w) else w, ramp_len, ramp_rate, am_len, attacked_frac)

    def length_of(self, fid: int) -> int:
        """The frames an episode of `fid` takes: the ramp length for At, `am_len` for Am."""
        return self.ramp_len if fid == RAMP_FAMILY else self.am_len


@dataclass
class _Placement:
    """What the placement asked for and got, per split (rows, `SPLITS`) and family (columns, the
    schedule's families): episodes requested, dropped because they would not fit, moved to
    another onset because no design was feasible, and built."""

    families: list[int]
    requested: np.ndarray
    dropped: np.ndarray
    redraws: np.ndarray
    built: np.ndarray

    @classmethod
    def empty(cls, families: list[int]) -> _Placement:
        shape = (len(SPLITS), len(families))
        return cls(
            families, np.zeros(shape, int), np.zeros(shape, int), np.zeros(shape, int), np.zeros(shape, int)
        )

    def col(self, fid: int) -> int:
        return self.families.index(fid)


def split_bounds(T: int, frac: Sequence[float]) -> list[tuple[int, int]]:
    """The three chronological splits [a, b) of T frames, cut before anything is placed: train
    round(frac[0] T) frames, val round(frac[1] T), test the rest."""
    n_train = min(T, int(round(frac[0] * T)))
    n_val = min(T - n_train, int(round(frac[1] * T)))
    cuts = (0, n_train, n_train + n_val, T)
    return [(cuts[i], cuts[i + 1]) for i in range(3)]


def largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    """`total` whole units shared in proportion to `weights` by largest remainder: each gets the floor
    of its quota and the units left go to the largest fractional parts (ties to the earlier one)."""
    if total == 0 or len(weights) == 0:
        return np.zeros(len(weights), int)
    quota = total * np.asarray(weights, float) / float(np.sum(weights))
    counts = np.floor(quota).astype(int)
    order = np.argsort(-(quota - counts), kind="stable")
    counts[order[: total - int(counts.sum())]] += 1
    return counts


def episode_counts(plan: _Schedule, n: int) -> np.ndarray:
    """Episodes per family for a split of n frames: the number of whole episodes whose frames come
    closest to `attacked_frac * n`, shared by largest remainder over the schedule's weights (so every
    family gets about the same share of attacked frames), never more frames than the split holds."""
    lengths = np.array([plan.length_of(f) for f in plan.families], int)
    best = np.zeros(len(lengths), int)
    if not len(lengths) or plan.attacked_frac <= 0:
        return best
    target = plan.attacked_frac * n
    best_err = target
    for k in range(1, n // int(lengths.min()) + 1):
        counts = largest_remainder(k, plan.weights)
        frames = int(counts @ lengths)
        if frames > n:
            break
        if abs(frames - target) < best_err:
            best, best_err = counts, abs(frames - target)
    return best


def uniform_onset(occupied: np.ndarray, length: int, rng: np.random.Generator, first: int = 0) -> int:
    """An onset drawn uniformly among every position, at or after `first`, where `length` consecutive
    frames are free."""
    free = np.concatenate([[0], np.cumsum(~occupied)])
    T = len(occupied)
    feasible = np.flatnonzero(free[length : T + 1] - free[: T - length + 1] == length)
    feasible = feasible[feasible >= first]
    if len(feasible) == 0:
        raise NoRoomForEpisode(f"no room for a {length}-frame episode")
    return int(feasible[rng.integers(len(feasible))])


def place_once(rng: np.random.Generator, episodes: list[tuple[int, int]], n: int) -> list[Slot]:
    """One placement of (length, family) episodes in a split of n frames: longest first, each at an
    onset uniform among those where it fits without touching another (NoRoomForEpisode when one
    does not fit). Returns (onset within the split, family, length)."""
    occupied = np.zeros(n, bool)
    placed: list[Slot] = []
    for length, fid in sorted(episodes, key=lambda x: -x[0]):
        onset = uniform_onset(occupied, length, rng)
        occupied[onset : onset + length] = True
        placed.append((onset, fid, length))
    return placed


def place_split(
    rng: np.random.Generator, plan: _Schedule, split: int, bounds: tuple[int, int], rec: _Placement
) -> list[Slot]:
    """The episodes of one split as (onset, family, length) in timeline frames, sorted by onset. A
    placement that jams is retried whole up to PLACE_TRIES times; past that one episode of the family
    with the most is dropped, recorded and warned about, and the placement starts over (bounded: at
    most one drop per episode)."""
    a, b = bounds
    counts = episode_counts(plan, b - a)
    rec.requested[split] = counts
    episodes = [(plan.length_of(f), f) for f, c in zip(plan.families, counts) for _ in range(c)]
    while episodes:
        for _ in range(PLACE_TRIES):
            try:
                return sorted((a + o, f, L) for o, f, L in place_once(rng, episodes, b - a))
            except NoRoomForEpisode:
                pass
        fams = [f for _, f in episodes]
        drop = max(set(fams), key=lambda f: (fams.count(f), -plan.families.index(f)))
        episodes.remove(next(e for e in episodes if e[1] == drop))
        rec.dropped[split, rec.col(drop)] += 1
        warnings.warn(
            f"split {SPLITS[split]}: {PLACE_TRIES} placements jammed, one {FAMILIES[drop]} episode dropped",
            RuntimeWarning,
            stacklevel=2,
        )
    return []


def free_onset(rng: np.random.Generator, length: int, others: Sequence[Slot], span: tuple[int, int]) -> int:
    """An onset in `span` = [a, b), uniform among every position where `length` frames fit without
    touching any of the `others` (NoRoomForEpisode when none is left)."""
    a, b = span
    occupied = np.zeros(b - a, bool)
    for o, _, L in others:
        lo, hi = max(o, a), min(o + L, b)
        if lo < hi:
            occupied[lo - a : hi - a] = True
    return a + uniform_onset(occupied, length, rng)


def give_up(split: int, fid: int, moves: int) -> None:
    """Warn that an episode found no feasible design after `moves` moves (its shortfall is recorded)."""
    warnings.warn(
        f"split {SPLITS[split]}: one {FAMILIES[fid]} episode found no feasible design after {moves} moves",
        RuntimeWarning,
        stacklevel=3,
    )


def split_column(bounds: list[tuple[int, int]], T: int) -> np.ndarray:
    """data/split: train 0, val 1, test 2 by the cut bounds."""
    split = np.zeros(T, np.int8)
    for code, (a, b) in enumerate(bounds):
        split[a:b] = code
    return split
