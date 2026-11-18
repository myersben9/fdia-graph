"""The plan stage of a timeline: where its splits are cut and where each split's episodes go.

A timeline is first cut into train, val and test: three blocks in time order (`split_bounds`), or whole
weeks drawn at random, stratified on a weight per frame (`split_weeks`). Each split then gets the whole
number of episodes whose frames come closest to its attacked fraction, shared evenly between the
families and placed at uniform random onsets, never overlapping and never cut (`place_episodes`). An
episode whose design is infeasible moves to another free onset of its split (`free_onset`), on either grid.
Nothing here touches the grid: the plan is positions and counts.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Optional, Union

import numpy as np

from ..constants.time import WEEK_SECONDS
from ..errors import NoRoomForEpisode
from ..models.choices import Split
from ..models.plan import Placement, Slot, SplitSpan

PLACE_TRIES = 10  # whole-split placements tried before one episode is dropped
STRATUM = 20  # weeks per stratum of `split_weeks`: one 14/3/3 draw at shares of 0.7, 0.15, 0.15
REDRAWS = 20  # onsets an episode with no feasible design is moved to before it is given up


def split_bounds(T: int, frac: Sequence[float]) -> list[tuple[int, int]]:
    """The three chronological splits [a, b) of T frames: train round(frac[0] T) frames, val
    round(frac[1] T), test the rest."""
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
    # stable: ties to the earlier
    by_remainder = sorted(range(len(quota)), key=lambda i: counts[i] - quota[i])
    for i in by_remainder[: total - sum(counts)]:
        counts[i] += 1
    return np.array(counts, int)


def whole_episode_counts(lengths: Sequence[int], frac: float, n: int) -> np.ndarray:
    """Episodes per family for n frames: the whole episodes whose frames come closest to `frac * n`,
    shared by largest remainder over the weights 1 / length, so every family gets the same share of
    attacked frames up to one episode. Never more frames than n."""
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
    episodes: Sequence[tuple[int, str]],
    n: int,
    allowed: Optional[Mapping[str, np.ndarray]] = None,
) -> list[Slot]:
    """One placement of (length, family) episodes in a split of n frames: longest first, each at an
    onset uniform among those where it fits and its family's `allowed` mask holds. Onsets are within
    the split. Raises NoRoomForEpisode, naming the family, when one does not fit."""
    occupied = np.zeros(n, bool)
    placed: list[Slot] = []
    for length, fid in sorted(episodes, key=lambda x: -x[0]):
        mask = None if allowed is None else allowed.get(fid)
        try:
            onset = uniform_onset(occupied, length, rng, allowed=mask)
        except NoRoomForEpisode as e:
            raise NoRoomForEpisode(str(e), family=fid) from None
        occupied[onset : onset + length] = True
        placed.append(Slot(onset, fid, length))
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
    lengths: Mapping[str, int], frac: float, n: int, allowed: Mapping[str, np.ndarray]
) -> dict[str, int]:
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
    lengths: Mapping[str, int],
    frac: float,
    n: int,
    allowed: Optional[Mapping[str, np.ndarray]] = None,
) -> tuple[list[Slot], dict[str, int], dict[str, int]]:
    """The episodes of one split of n frames as slots sorted by onset (within the split), then the
    episodes requested and dropped per family. A placement that jams is retried whole PLACE_TRIES
    times; then one episode of the family that did not fit is dropped and the placement starts over."""
    allowed = {} if allowed is None else allowed
    requested = capped_counts(lengths, frac, n, allowed)
    dropped = dict.fromkeys(lengths, 0)
    episodes = [(lengths[f], f) for f in lengths for _ in range(requested[f])]
    while episodes:
        jammed = episodes[0]
        for _ in range(PLACE_TRIES):
            try:
                return sorted(place_once(rng, episodes, n, allowed)), requested, dropped
            except NoRoomForEpisode as e:
                jammed = next((x for x in episodes if x[1] == e.family), episodes[0])
        episodes.remove(jammed)
        dropped[jammed[1]] += 1
    return [], requested, dropped


def place_split(
    rng: np.random.Generator,
    lengths: Mapping[str, int],
    frac: float,
    span: SplitSpan,
    rec: Placement,
    masks: Optional[Mapping[str, np.ndarray]] = None,
) -> list[Slot]:
    """The episodes of the split `span`, onsets in timeline frames; `rec` records what was requested
    and dropped. An episode starts only where its frames are consecutive on the timeline
    (`SplitSpan.onsets_fit`) and where its family's mask in `masks` ([T] bool) allows."""
    index = span.index()
    allowed = {f: span.onsets_fit(length) for f, length in lengths.items()}
    for f, mask in (masks or {}).items():
        allowed[f] &= mask[index]
    placed, requested, dropped = place_episodes(rng, lengths, frac, len(index), allowed)
    rec.requested[span.code] = [requested[f] for f in rec.families]
    rec.dropped[span.code] = [dropped[f] for f in rec.families]
    for f in rec.families:
        if dropped[f]:
            warnings.warn(
                f"split {Split.values()[span.code]}: placements jammed {PLACE_TRIES} times, {dropped[f]} "
                f"{f} episode(s) dropped",
                RuntimeWarning,
                stacklevel=2,
            )
    return [slot.at(int(index[slot.onset])) for slot in placed]


def planned_until(spans: Sequence[SplitSpan], k: int) -> int:
    """The frame the walk may run up to once the splits `spans[0..k]` are planned: the first frame of
    a split still to plan, or the end of the timeline. In block splits that is the end of split k; in
    splits of whole weeks the walk waits until every earlier week is planned."""
    later = [span.start for span in spans[k + 1 :] if span.size]
    return min(later, default=max(span.stop for span in spans))


def free_onset(
    rng: np.random.Generator,
    length: int,
    others: Sequence[Slot],
    span: SplitSpan,
    mask: Optional[np.ndarray] = None,
) -> int:
    """An onset in the split `span`, uniform among every position where `length` consecutive frames
    of the split fit without touching any of the `others` and, with a family's `mask` ([T] bool over
    the timeline), where it allows (NoRoomForEpisode when none is left)."""
    index = span.index()
    occupied = np.zeros(len(index), bool)
    for other in others:
        lo, hi = np.searchsorted(index, (other.onset, other.onset + other.length))
        occupied[lo:hi] = True
    allowed = span.onsets_fit(length)
    if mask is not None:
        allowed &= mask[index]
    return int(index[uniform_onset(occupied, length, rng, allowed=allowed)])


def give_up(split: int, fid: str, moves: int) -> None:
    """Warn that an episode found no feasible design after `moves` moves (its shortfall is recorded)."""
    warnings.warn(
        f"split {Split.values()[split]}: one {fid} episode found no feasible design after {moves} moves",
        RuntimeWarning,
        stacklevel=3,
    )


def split_column(bounds: list[tuple[int, int]], T: int) -> np.ndarray:
    """data/split: train 0, val 1, test 2 by the cut bounds."""
    split = np.zeros(T, np.int8)
    for code, (a, b) in enumerate(bounds):
        split[a:b] = code
    return split


def time_runs(time: np.ndarray) -> np.ndarray:
    """[T] int: the run of regularly spaced frames each frame lies in. The regular step is the median
    step between two frames; a step of any other size is a gap and starts a new run. An episode is
    placed inside one run (`SplitSpan.onsets_fit`)."""
    step = np.diff(time).astype(np.int64)
    if not len(step):
        return np.zeros(len(time), int)
    return np.r_[0, np.cumsum(step != np.median(step))]


def week_index(time: np.ndarray) -> np.ndarray:
    """[T] int: each frame's week counted from the first frame's time (`time` [T] in seconds or datetime64)."""
    t = np.asarray(time)
    t = (t.astype("datetime64[s]") if np.issubdtype(t.dtype, np.datetime64) else t).astype(np.int64)
    return ((t - t[0]) // WEEK_SECONDS).astype(int)


def split_weeks(
    time: np.ndarray,
    frac: Sequence[float],
    weight: np.ndarray,
    rng: np.random.Generator,
    stratum: int = STRATUM,
) -> np.ndarray:
    """The split code of each frame [T] (0 train, 1 val, 2 test), by whole weeks (`time` [T] in seconds).
    Weeks are ranked by their summed `weight` [T] (ties in a random order), cut into strata of `stratum`
    ranks, and each stratum's weeks drawn at random into the splits in the shares `frac`, so each split
    gets its share of weight."""
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
