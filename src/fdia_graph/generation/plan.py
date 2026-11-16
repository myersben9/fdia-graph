"""The plan stage of a timeline: where its splits are cut and where each split's episodes go.

A timeline is first cut into train, val and test: three blocks in time order (`split_bounds`), or whole
weeks drawn at random, stratified on a weight per frame (`split_weeks`). Then each split gets the whole
number of episodes whose frames come closest to its attacked fraction, shared between the families so
each gets about the same share of attacked frames, and placed at uniform random onsets inside the split,
never overlapping and never cut (`place_episodes`). A family may be held to the onsets a mask allows. An
episode whose design turns out infeasible is moved to another free onset of its split (`free_onset`).
Nothing here touches the grid: the plan is positions and counts, the design stage decides what each
episode attacks and the emit stage writes the frames.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, TypeVar, Union

import numpy as np

from ..engine.attacks.stealthy import RAMP_FAMILY
from ..errors import NoRoomForEpisode
from ..models.choices import FAMILIES

SPLITS = ("train", "val", "test")  # the order of the per-split attributes and of data/split's codes
PLACE_TRIES = 10  # whole-split placements tried before one episode is dropped (the split-first packing rule)
WEEK = 7 * 86400  # seconds
STRATUM = 20  # weeks per stratum of `split_weeks`: one 14/3/3 draw at shares of 0.7, 0.15, 0.15
REDRAWS = 20  # onsets an episode with no feasible design is moved to before it is given up (the split-first redraw rule)

Slot = tuple[int, int, int]  # one placed episode: (onset in timeline frames, family code, length)
Family = TypeVar(
    "Family", int, str
)  # an episode's family: the timeline's code, or the New York family's name


@dataclass
class _Schedule:
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
    ) -> _Schedule:
        return cls(fams, ramp_len, ramp_rate, am_len, attacked_frac)

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


def largest_remainder(total: int, weights: Union[Sequence[Fraction], np.ndarray]) -> np.ndarray:
    """`total` whole units shared in proportion to `weights` by largest remainder: each gets the floor
    of its quota and the units left go to the largest fractional parts (ties to the earlier one). With
    Fraction weights the quotas, and so the ties, are exact."""
    shares = list(weights)
    if total == 0 or not shares:
        return np.zeros(len(shares), int)
    whole = sum(shares)
    quota = [total * w / whole for w in shares]
    counts = [math.floor(q) for q in quota]
    by_remainder = sorted(
        range(len(quota)), key=lambda i: counts[i] - quota[i]
    )  # stable: ties to the earlier
    for i in by_remainder[: total - sum(counts)]:
        counts[i] += 1
    return np.array(counts, int)


def whole_episode_counts(lengths: Sequence[int], frac: float, n: int) -> np.ndarray:
    """Episodes per family, of `lengths` frames each, for n frames: the number of whole episodes whose frames
    come closest to `frac * n`, shared by largest remainder over the weights 1 / length (exact fractions, so
    every family gets the same share of attacked frames up to one episode), never more frames than n."""
    sizes = np.array([int(length) for length in lengths], int)
    best = np.zeros(len(sizes), int)
    if not len(sizes) or frac <= 0:
        return best
    weights = [Fraction(1, int(length)) for length in sizes]
    target = frac * n
    best_err = target
    for k in range(1, n // int(sizes.min()) + 1):
        counts = largest_remainder(k, weights)
        frames = int(counts @ sizes)
        if frames > n:
            break
        if abs(frames - target) < best_err:
            best, best_err = counts, abs(frames - target)
    return best


def uniform_onset(
    occupied: np.ndarray,
    length: int,
    rng: np.random.Generator,
    first: int = 0,
    allowed: Optional[np.ndarray] = None,
) -> int:
    """An onset drawn uniformly among every position, at or after `first`, where `length` consecutive
    frames are free (and `allowed` [T] holds, when given)."""
    free = np.concatenate([[0], np.cumsum(~occupied)])
    T = len(occupied)
    fits = free[length : T + 1] - free[: T - length + 1] == length
    if allowed is not None:
        fits &= allowed[: T - length + 1]
    feasible = np.flatnonzero(fits)
    feasible = feasible[feasible >= first]
    if len(feasible) == 0:
        raise NoRoomForEpisode(f"no room for a {length}-frame episode")
    return int(feasible[rng.integers(len(feasible))])


def place_once(
    rng: np.random.Generator,
    episodes: Sequence[tuple[int, Family]],
    n: int,
    allowed: Optional[Mapping[Family, np.ndarray]] = None,
) -> list[tuple[int, Family, int]]:
    """One placement of (length, family) episodes in a split of n frames: longest first, each at an
    onset uniform among those where it fits without touching another (and its family's `allowed` onsets,
    when given). NoRoomForEpisode, naming the family, when one does not fit. Returns (onset within the
    split, family, length)."""
    occupied = np.zeros(n, bool)
    placed: list[tuple[int, Family, int]] = []
    for length, fid in sorted(episodes, key=lambda x: -x[0]):
        try:
            onset = uniform_onset(
                occupied, length, rng, allowed=None if allowed is None else allowed.get(fid)
            )
        except NoRoomForEpisode as e:
            raise NoRoomForEpisode(str(e), family=fid) from None
        occupied[onset : onset + length] = True
        placed.append((onset, fid, length))
    return placed


def capacity(allowed: np.ndarray, length: int) -> int:
    """The most non-overlapping windows of `length` frames with allowed onsets (earliest first), an
    upper bound on what a uniform placement can hold."""
    n, end = 0, -1
    for s in np.flatnonzero(allowed):
        if s > end:
            n, end = n + 1, s + length - 1
    return n


def capped_counts(
    lengths: Mapping[Family, int], frac: float, n: int, allowed: Mapping[Family, np.ndarray]
) -> dict[Family, int]:
    """Episodes per family for a split of n frames (`whole_episode_counts`), a family held to `allowed`
    onsets cut to what those onsets can hold (`capacity`)."""
    fams = list(lengths)
    counts = dict(zip(fams, whole_episode_counts([lengths[f] for f in fams], frac, n).tolist()))
    for f in fams:
        if f in allowed:
            counts[f] = min(counts[f], capacity(allowed[f][: max(n - lengths[f] + 1, 0)], lengths[f]))
    return counts


def place_episodes(
    rng: np.random.Generator,
    lengths: Mapping[Family, int],
    frac: float,
    n: int,
    allowed: Optional[Mapping[Family, np.ndarray]] = None,
) -> tuple[list[tuple[int, Family, int]], dict[Family, int], dict[Family, int]]:
    """The episodes of one split of n frames as (onset within the split, family, length), sorted by onset,
    with the episodes requested and dropped per family.

    `lengths` gives each family's episode length and `frac` the share of the split's frames to attack;
    `allowed` (by family, [n], optional) holds a family to the onsets it may take. The count requested is
    `capped_counts`. A placement that jams is retried whole up to PLACE_TRIES times; past that one episode
    of the family that did not fit is dropped and the placement starts over (bounded: at most one drop per
    episode). The caller reports what was dropped."""
    allowed = {} if allowed is None else allowed
    requested = capped_counts(lengths, frac, n, allowed)
    dropped = dict.fromkeys(lengths, 0)
    episodes = [(lengths[f], f) for f in lengths for _ in range(requested[f])]
    while episodes:
        jammed = episodes[0]
        for _ in range(PLACE_TRIES):
            try:
                return sorted(place_once(rng, episodes, n, allowed)), requested, dropped
            except NoRoomForEpisode as e:  # the error names the family that did not fit
                jammed = next((x for x in episodes if x[1] == e.family), episodes[0])
        episodes.remove(jammed)
        dropped[jammed[1]] += 1
    return [], requested, dropped


def place_split(
    rng: np.random.Generator, plan: _Schedule, split: int, bounds: tuple[int, int], rec: _Placement
) -> list[Slot]:
    """The episodes of one split [a, b) of the schedule, onsets in timeline frames, with what was
    requested and dropped recorded in `rec` (and a warning when an episode was dropped)."""
    a, b = bounds
    lengths = {f: plan.length_of(f) for f in plan.families}
    placed, requested, dropped = place_episodes(rng, lengths, plan.attacked_frac, b - a)
    rec.requested[split] = [requested[f] for f in plan.families]
    rec.dropped[split] = [dropped[f] for f in plan.families]
    for f in plan.families:
        if dropped[f]:
            warnings.warn(
                f"split {SPLITS[split]}: placements jammed {PLACE_TRIES} times, {dropped[f]} "
                f"{FAMILIES[f]} episode(s) dropped",
                RuntimeWarning,
                stacklevel=2,
            )
    return [(a + onset, f, length) for onset, f, length in placed]


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


def week_index(time: np.ndarray) -> np.ndarray:
    """[T] int: each frame's week counted from the first frame's time (`time` [T] in seconds or datetime64)."""
    t = np.asarray(time)
    t = (t.astype("datetime64[s]") if np.issubdtype(t.dtype, np.datetime64) else t).astype(np.int64)
    return ((t - t[0]) // WEEK).astype(int)


def split_weeks(
    time: np.ndarray,
    frac: Sequence[float],
    weight: np.ndarray,
    rng: np.random.Generator,
    stratum: int = STRATUM,
) -> np.ndarray:
    """The split code of each frame [T] (0 train, 1 val, 2 test) by whole weeks from the first frame's
    time (`time` [T] in seconds): the weeks ranked by their summed `weight` [T] (ties in a random order),
    cut into strata of `stratum` consecutive ranks, and each stratum's weeks drawn at random into the
    three splits in the shares `frac` (largest remainder), so every split gets its share of weeks and of
    weight, and no episode crosses from one split into another."""
    week = week_index(time)
    n = int(week.max()) + 1
    w = np.bincount(week, weights=np.asarray(weight, float), minlength=n)
    shuffled = rng.permutation(n)
    ranked = shuffled[np.argsort(-w[shuffled], kind="stable")]
    code = np.zeros(n, np.int8)
    for a in range(0, n, stratum):
        weeks = ranked[a : a + stratum]
        counts = largest_remainder(len(weeks), np.asarray(frac, float))
        code[rng.permutation(weeks)] = np.repeat(np.arange(3, dtype=np.int8), counts)
    return code[week]
