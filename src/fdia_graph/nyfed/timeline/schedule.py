"""Where the timeline's splits are cut and where each split's episodes go.

The year is cut into whole weeks assigned to train, validation and test by a seeded draw stratified on
each week's loaded frames (`split_weeks`, N41), so every split holds summer, winter and the loaded hours
an overload attacker waits for (N36); no episode crosses a week block, so no window is shared between
two splits. (`split_frames`, three contiguous blocks, was the split before N41.) Each split then gets
the whole number of episodes whose frames come closest to its attacked share, shared between the
families so each gets about the same share of attacked frames, and placed at uniform random onsets
among those its family allows (`place_episodes`), never overlapping and never cut. A family whose
allowed onsets cannot hold its share gets fewer episodes. Whether two episodes touch or leave a gap is
only a statistic of that uniform placement; nothing here spaces them on purpose. Nothing here touches
the grid: the plan is positions and counts, the families decide what each episode attacks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

PLACE_TRIES = 10  # whole placements tried before one episode is dropped
WEEK = 7 * 86400  # seconds
STRATUM = 20  # weeks per stratum of `split_weeks`, one 14/3/3 draw at the default shares
HIT_STRATUM = 5  # weeks per stratum when the weight is the surveyed attacks (N46): they sit in about 9 weeks,
# one of them holding 40%, so a 20-week stratum can leave a split one episode; 5 draws 3/1/1 per stratum


@dataclass(frozen=True)
class Slot:
    """One placed episode: its first frame `onset`, its `family` ("Am", "Ap" or "At") and `length` in frames."""

    onset: int
    family: str
    length: int


def split_frames(T: int, frac: tuple[float, float, float]) -> np.ndarray:
    """The split code of each of T frames [T] (0 train, 1 val, 2 test): three contiguous blocks in time
    order, train round(frac[0] T) frames, val round(frac[1] T), test the rest."""
    n_train = min(T, int(round(frac[0] * T)))
    n_val = min(T - n_train, int(round(frac[1] * T)))
    split = np.full(T, 2, np.int8)
    split[:n_train] = 0
    split[n_train : n_train + n_val] = 1
    return split


def week_index(time: np.ndarray) -> np.ndarray:
    """[T] int: each frame's week counted from the first frame's time (`time` [T] in seconds or datetime64)."""
    t = np.asarray(time)
    t = (t.astype("datetime64[s]") if np.issubdtype(t.dtype, np.datetime64) else t).astype(np.int64)
    return ((t - t[0]) // WEEK).astype(int)


def split_weeks(
    time: np.ndarray,
    frac: tuple[float, float, float],
    weight: np.ndarray,
    rng: np.random.Generator,
    stratum: int = STRATUM,
) -> np.ndarray:
    """The split code of each frame [T] (0 train, 1 val, 2 test) by whole weeks from the first frame's
    time (`time` [T] in seconds): the weeks ranked by their summed `weight` [T] (the loaded frames, N41;
    ties in a random order), cut into strata of `stratum` consecutive ranks, and each stratum's weeks
    drawn at random into the three splits in the shares `frac` (largest remainder), so every split gets
    its share of weeks and of weight."""
    week = week_index(time)
    n = int(week.max()) + 1
    w = np.bincount(week, weights=np.asarray(weight, float), minlength=n)
    shuffled = rng.permutation(n)
    ranked = shuffled[np.argsort(-w[shuffled], kind="stable")]
    code = np.zeros(n, np.int8)
    for a in range(0, n, stratum):
        weeks = ranked[a : a + stratum]
        counts = _largest_remainder(len(weeks), np.asarray(frac, float))
        code[rng.permutation(weeks)] = np.repeat(np.arange(3, dtype=np.int8), counts)
    return code[week]


def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    """`total` whole units shared in proportion to `weights`: each gets the floor of its quota, the units
    left go to the largest fractional parts (ties to the earlier one)."""
    if total == 0 or len(weights) == 0:
        return np.zeros(len(weights), int)
    quota = total * weights / weights.sum()
    counts = np.floor(quota).astype(int)
    order = np.argsort(-(quota - counts), kind="stable")
    counts[order[: total - int(counts.sum())]] += 1
    return counts


def _episode_counts(T: int, lengths: np.ndarray, frac: float) -> np.ndarray:
    """Episodes per family for T frames: the number of whole episodes whose frames come closest to
    `frac * T`, shared by largest remainder with weights 1/length so every family gets about the same
    attacked frames, never more frames than T."""
    best = np.zeros(len(lengths), int)
    if not len(lengths) or frac <= 0:
        return best
    weights = 1.0 / lengths
    target, best_err = frac * T, frac * T
    for k in range(1, T // int(lengths.min()) + 1):
        counts = _largest_remainder(k, weights)
        frames = int(counts @ lengths)
        if frames > T:
            break
        if abs(frames - target) < best_err:
            best, best_err = counts, abs(frames - target)
    return best


def _uniform_onset(
    occupied: np.ndarray, length: int, rng: np.random.Generator, allowed: np.ndarray | None = None
) -> int | None:
    """An onset drawn uniformly among every position where `length` consecutive frames are free (and
    `allowed` [T] holds, when given), or None."""
    free = np.concatenate([[0], np.cumsum(~occupied)])
    T = len(occupied)
    fits = free[length : T + 1] - free[: T - length + 1] == length
    if allowed is not None:
        fits &= allowed[: T - length + 1]
    feasible = np.flatnonzero(fits)
    return None if len(feasible) == 0 else int(feasible[rng.integers(len(feasible))])


def _capacity(allowed: np.ndarray, length: int) -> int:
    """The most non-overlapping windows of `length` frames with allowed onsets (earliest first), an
    upper bound on what a uniform placement can hold."""
    n, end = 0, -1
    for s in np.flatnonzero(allowed):
        if s > end:
            n, end = n + 1, s + length - 1
    return n


def _place_once(
    T: int, episodes: list[tuple[int, str]], rng: np.random.Generator, allowed: dict[str, np.ndarray]
) -> tuple[list[Slot] | None, str | None]:
    """One placement of (length, family) episodes in T frames, longest first, each at an onset uniform
    among those its family allows where it fits without overlapping another: (the slots, None), or
    (None, the family of the episode that did not fit)."""
    occupied = np.zeros(T, bool)
    placed = []
    for length, family in sorted(episodes, key=lambda e: -e[0]):
        onset = _uniform_onset(occupied, length, rng, allowed.get(family))
        if onset is None:
            return None, family
        occupied[onset : onset + length] = True
        placed.append(Slot(onset, family, length))
    return placed, None


def place_episodes(
    T: int,
    lengths: dict[str, int],
    frac: float,
    rng: np.random.Generator,
    allowed: dict[str, np.ndarray] | None = None,
) -> list[Slot]:
    """The episodes of one split of T frames, sorted by onset (onsets within the split). `lengths` gives
    each family's episode length, `frac` the share of the split's frames to attack, and `allowed` (by
    family, [T], optional) the onsets a family may take; a family's count is cut to what its allowed
    onsets can hold (`_capacity`). A placement that jams is retried whole up to PLACE_TRIES times; past
    that one episode of the family that jammed is dropped, with a warning, and the placement starts over."""
    fams = list(lengths)
    allowed = {} if allowed is None else allowed
    counts = _episode_counts(T, np.array([lengths[f] for f in fams], float), frac)
    for i, f in enumerate(fams):
        if f in allowed:
            counts[i] = min(counts[i], _capacity(allowed[f][: max(T - lengths[f] + 1, 0)], lengths[f]))
    episodes = [(lengths[f], f) for f, c in zip(fams, counts) for _ in range(c)]
    while episodes:
        for _ in range(PLACE_TRIES):
            placed, jammed = _place_once(T, episodes, rng, allowed)
            if placed is not None:
                return sorted(placed, key=lambda s: s.onset)
        episodes.remove(next(e for e in episodes if e[1] == jammed))
        warnings.warn(
            f"{PLACE_TRIES} placements jammed, one {jammed} episode dropped", RuntimeWarning, stacklevel=2
        )
    return []
