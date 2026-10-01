"""One continuous attacked timeline per system, written as one HDF5 file.

The timeline is the dataset. It is first cut chronologically into train, val and test (60/20/20
by default); then each split gets its own attack episodes, as many whole episodes as come closest
to the attacked fraction of its frames, shared between the families by largest remainder and placed
at uniform random onsets inside the split, never overlapping and never cut. Whether two episodes
touch or a long quiet stretch separates them is a property of that draw, not of any rule. Every
frame's meter jitter is drawn from its own stream keyed by (seed, timestep), so two timelines on one
seed and pool that place different attacks carry the same benign frames.
The walk then emits every frame in time order, a scan of the grid at one pool timestep with the
attack of its episode applied (or none), and the file carries everything a user reads afterwards:

    attrs         system, N, E, baseMVA, seed, T, families, kind="timeline", the knobs, attacked_frac
    data/         node_x, node_m [T, N, 4]; edge_x, edge_m [T, E, 2]; y [T, N]; family, stealthy,
                  seq_id (episode index, -1 benign), timestep, split [T]; temporal_delta, swing [T, N, 2]
    benign/       node_benign, edge_benign: the same scan with the attack removed and the noise kept
    clean/        node_clean, edge_clean: the noiseless attack-free truth (edge_clean on metered branches)
    graph/        the static topology and per-unit branch physics (as the shards carry them)
    episodes/     onset, length, family [K]; the attacked buses ragged as bus_ptr [K+1], bus_idx
    attack/       the designed magnitude per attacked bus ragged as mag_ptr [T+1], mag_bus, mag, and the
                  tamper masks node_tamper [T, N, 4], edge_tamper [T, E, 2]: the meters the attacker wrote

Temporal features are taken against the previous EMITTED frame, so a stealthy ramp reads as a
small per-step change and a spike as an abrupt jump (the signal the dataset is built on).

`generate_stream` (deprecated) is this writer followed by a read of the file it wrote.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union

import h5py
import numpy as np

from . import schema
from ._moved import moved
from .dataset.base import FAMILIES, STEALTHY_FAMILIES
from .engine import FdiaGenerator
from .engine.attacks.episodes import (
    ONSET_DRAWS,
    EpisodeDesignMixin,
    draw_ramp,
    probe_frames,
    ramp_dev,
)
from .engine.records import (  # noqa: F401  AttackDesign, is_feasible re-exported as before
    AM_FAMILY,
    RAMP_FAMILY,
    AttackDesign,
    Frame,
    FrameKnobs,
    attack_frame,
    is_feasible,
)
from .errors import NoRoomForEpisode
from .formulas.attacks import ramp_profile  # noqa: F401  re-exported as before
from .formulas.temporal import SWING_WINDOW, recent_change_scale, swing_zscore, temporal_delta
from .generation import (
    _CHUNK_ROWS,
    _base_attrs,
    _FrameContext,
    _load_states,
    _write_graph,
)
from .models.choices import (  # noqa: F401  re-exported beside the code that reads them
    BENIGN_CODE,
    FAMILY_CODE,
    GENERATED_FAMILIES,
)
from .models.config import MeterSettings, OverloadSettings, SplitSettings, TimelineKnobs
from .models.data import EpisodeRow
from .models.frames import AmOverloadDesign, AttackVector, MinimizerResult
from .models.inputs import AdmissibleTargets, GeneratedFamilies
from .registry import CACHE_DIR, system_id
from .schema import Attr

KIND = schema.KIND_TIMELINE  # the file attribute that tells a timeline from a shard
# the generator makes the multi-snapshot families [WU26]: the ramp At and the overload Am
DEFAULT_FAMILIES = GENERATED_FAMILIES

# what an episode attacks moved to engine.attacks.episodes (the generator's AttackMixin); the old
# private names keep their old signatures for one minor release, forwarding to the new home
_EPISODES = "engine.attacks.episodes"
_MOVED: dict[str, tuple[str, object]] = {
    "_ONSET_DRAWS": (f"{_EPISODES}.ONSET_DRAWS", ONSET_DRAWS),
    "_ramp_dev": (f"{_EPISODES}.ramp_dev", ramp_dev),
    "_target_counts": (f"{_EPISODES}.EpisodeDesignMixin.target_counts", EpisodeDesignMixin.target_counts),
    "_probe_frames": (
        f"{_EPISODES}.probe_frames",
        lambda ctx, t, length: probe_frames(len(ctx.X), t, length),
    ),
    "_draw_ramp": (
        f"{_EPISODES}.draw_ramp",
        lambda ctx, rng, n: tuple(draw_ramp(rng, ctx.g.stealthy_pos, n))[
            :4
        ],  # its old (targets, direction, rise, hold)
    ),
}


def __getattr__(name: str) -> object:
    return moved(__name__, name, _MOVED)


_BATCH = 256  # frames staged in memory between two flushes to the file
_LAYERS = {  # per-frame datasets: name -> (trailing shape given (C, E), dtype)
    schema.NODE_X: (lambda C, E: (C, 4), np.float32),
    schema.EDGE_X: (lambda C, E: (E, 2), np.float32),
    schema.Y: (lambda C, E: (C,), np.uint8),
    schema.TEMPORAL_DELTA: (lambda C, E: (C, 2), np.float32),
    schema.SWING: (lambda C, E: (C, 2), np.float32),
    schema.NODE_BENIGN: (lambda C, E: (C, 4), np.float32),
    schema.EDGE_BENIGN: (lambda C, E: (E, 2), np.float32),
    schema.NODE_CLEAN: (lambda C, E: (C, 4), np.float32),
    schema.EDGE_CLEAN: (lambda C, E: (E, 2), np.float32),
    schema.NODE_TAMPER: (lambda C, E: (C, 4), np.uint8),
    schema.EDGE_TAMPER: (lambda C, E: (E, 2), np.uint8),
}
# the PMU branch-current layers of a hybrid-meter timeline (the plan's D10), created only when the meter
# plan reads currents, so a v0.8.3-meter file has exactly its old datasets
_CURRENT_LAYERS = {
    schema.PMU_I: (lambda C, E: (E, 4), np.float32),
    schema.PMU_I_BENIGN: (lambda C, E: (E, 4), np.float32),
    schema.PMU_I_TAMPER: (lambda C, E: (E, 4), np.uint8),
}
CleanSlice = Callable[[int, int], tuple[np.ndarray, np.ndarray]]


def _layers(currents: bool) -> dict:
    """The per-frame datasets of a timeline: the v0.8.3 set, plus the current layers when the meter
    plan reads PMU branch currents."""
    return {**_LAYERS, **(_CURRENT_LAYERS if currents else {})}


def _clean_slice(g: FdiaGenerator, X: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """The noiseless truth of frames a..b-1: the states as float32 and the exact flows on metered
    branches (one batched matmul, the engine's physics primitive)."""
    return X[a:b].astype(np.float32), g.clean_flows_from_states(X[a:b])


class _TimelineBuffers:
    """The per-frame layers of one timeline, filled frame by frame in time order.

    node_x / edge_x are the OBSERVED measurements (attacked + noisy where attacked, else benign +
    noisy); benign / edge_benign the same scan with the attack removed and the noise kept;
    edge_clean the noiseless true flows on metered branches. temporal_delta and swing are written
    after the walk from the observed frames (`write_temporal_layers`). seq_id is the episode index of an
    attacked frame, the tamper masks the meters the attacker wrote, and the magnitude lists the
    designed change per attacked bus.

    The `sink` (name -> HDF5 dataset of full length T) receives the layers in batches of _BATCH
    frames, flushed as the walk passes them, so memory is bounded whatever T.
    """

    def __init__(
        self,
        dims: tuple[int, int, int],
        clean: CleanSlice,
        sink: dict[str, h5py.Dataset],
        currents: bool = False,
    ) -> None:
        T, C, E = dims
        n = min(_BATCH, T)
        self.T, self._n, self._base, self._sink = T, n, 0, sink
        self._layers: dict[str, np.ndarray] = {
            name: np.zeros((n, *shape(C, E)), dtype) for name, (shape, dtype) in _layers(currents).items()
        }
        self.currents = currents  # the meter plan reads PMU branch currents (hybrid meters)
        self.i_m: Optional[np.ndarray] = None  # their mask, from the first stored frame
        self.family = np.zeros(T, np.int16)
        self.seq_id = np.full(T, -1, np.int32)
        self.mag_bus: list[np.ndarray] = [np.zeros(0, np.int32)] * T
        self.mag: list[np.ndarray] = [np.zeros(0, np.float32)] * T
        self.node_m: Optional[np.ndarray] = None  # the meter plan, from the first stored frame
        self.edge_m: Optional[np.ndarray] = None
        self.episodes: list[EpisodeRow] = []
        self.min_rows: list[tuple[int, MinimizerResult]] = []  # (episode, fewest-tamper result), knob on
        # (episode, target branch, rating, noiseless flow reached, emitted flow) per overload Am episode
        self.am_rows: list[tuple[int, int, float, float, float, float]] = []
        # the attack vector of the last stored frame, observed minus its benign twin (zero when benign):
        # an adjacent episode's stealth bound starts from it
        self.last_attack = AttackVector(
            np.zeros((C, 4)), np.zeros((E, 2)), np.zeros((E, 4)) if currents else None
        )
        self.attacked = 0  # frames stored so far with at least one attacked bus
        self._clean = clean
        self._clean_batch = clean(0, n)

    def store(self, t: int, fid: int, frame: Frame, sid: int = -1) -> None:
        """Store frame t (frames arrive in order): an attacked frame with its un-attacked twin
        (`with_benign`), or a benign one."""
        if t >= self._base + self._n:
            self.flush()
        r = t - self._base
        L = self._layers
        bnx = frame.node_x if frame.benign_node_x is None else frame.benign_node_x
        bex = frame.edge_x if frame.benign_edge_x is None else frame.benign_edge_x
        L[schema.NODE_X][r] = frame.node_x
        L[schema.NODE_BENIGN][r] = bnx
        L[schema.Y][r] = frame.y
        L[schema.EDGE_X][r] = frame.edge_x
        L[schema.EDGE_BENIGN][r] = bex
        L[schema.NODE_CLEAN][r], L[schema.EDGE_CLEAN][r] = self._clean_batch[0][r], self._clean_batch[1][r]
        self.family[t] = fid
        self.seq_id[t] = sid if fid else -1
        self.attacked += int(frame.y.any())
        if self.node_m is None:
            self.node_m, self.edge_m = frame.node_m, frame.edge_m
        current = self._store_currents(r, fid, frame) if self.currents else None
        self.last_attack = AttackVector(
            np.asarray(frame.node_x, np.float64) - np.asarray(bnx, np.float64),
            np.asarray(frame.edge_x, np.float64) - np.asarray(bex, np.float64),
            current,
        )
        self._store_attack(t, fid, frame, bnx, bex)

    def _store_currents(self, r: int, fid: int, frame: Frame) -> np.ndarray:
        """Stage the PMU branch-current layers of a frame (hybrid meters) and return its attack vector
        on them (observed minus un-attacked, zero on a benign frame)."""
        assert frame.i_x is not None, "a hybrid-meter generator emits the currents on every frame"
        bix = frame.i_x if frame.benign_i_x is None else frame.benign_i_x
        L = self._layers
        L[schema.PMU_I][r], L[schema.PMU_I_BENIGN][r] = frame.i_x, bix
        tamper = frame.i_tamper if fid != BENIGN_CODE and frame.i_tamper is not None else frame.i_x != bix
        L[schema.PMU_I_TAMPER][r] = tamper if fid != BENIGN_CODE else 0
        if self.i_m is None:
            self.i_m = frame.i_m
        return np.asarray(frame.i_x, np.float64) - np.asarray(bix, np.float64)

    def _store_attack(self, t: int, fid: int, frame: Frame, bnx: np.ndarray, bex: np.ndarray) -> None:
        """The attacker's footprint on this frame: the designed magnitudes and the tamper masks
        (zeroed on a benign frame: the staged batch rows are reused between flushes)."""
        r = t - self._base
        if fid == BENIGN_CODE:
            self._layers[schema.NODE_TAMPER][r] = 0
            self._layers[schema.EDGE_TAMPER][r] = 0
            return
        self.mag_bus[t] = np.asarray(frame.mag_bus, np.int32)
        self.mag[t] = np.asarray(frame.mag, np.float32)
        if frame.tamper is not None:  # a stealthy family: the meters its local false state moves
            node, edge = frame.tamper
        else:  # in-place corruption: exactly the meters that changed
            node, edge = frame.node_x != bnx, frame.edge_x != bex
        self._layers[schema.NODE_TAMPER][r] = node
        self._layers[schema.EDGE_TAMPER][r] = edge

    def flush(self) -> None:
        """Write the staged frames to the sink and stage the next batch."""
        m = min(self._n, self.T - self._base)  # frames staged in this batch
        for name, arr in self._layers.items():
            self._sink[name][self._base : self._base + m] = arr[:m]
        self._base += m
        if self._base < self.T:
            self._clean_batch = self._clean(self._base, self._base + self._n)


def _emit_benign(ctx: _FrameContext, t: int) -> Frame:
    """The benign scan of timestep t, its jitter from frame t's own stream."""
    ctx.g.scan_key = t
    frame = attack_frame(ctx.g, ctx.X[t], 0, None, ctx.knobs)
    assert frame is not None  # a benign emission cannot fail
    return frame


@dataclass
class _Walk:
    """What every step of one timeline walk shares: the run context (generator, pool, knobs) and the
    buffers the frames are written to. The random stream is the generator's own, so the order of
    draws is the order the walk makes them."""

    ctx: _FrameContext
    buf: _TimelineBuffers

    @property
    def rng(self) -> np.random.Generator:
        return self.ctx.g.rng

    @property
    def T(self) -> int:
        return len(self.ctx.X)


def _benign_run(w: _Walk, t: int, until: int) -> int:
    """Benign frames from t up to `until` (excluded); returns `until`."""
    for t in range(t, until):
        w.buf.store(t, 0, _emit_benign(w.ctx, t))
    return until


@dataclass
class _Episode:
    """One episode being built: its index, family, and the buses attacked so far."""

    sid: int
    fid: int
    onset: int
    ok: np.ndarray  # [N] uint8, the buses any frame of the episode labelled

    def store(self, w: _Walk, t: int, frame: Optional[Frame]) -> None:
        """Store the attacked frame, or a benign one when the attack could not be built at t."""
        if frame is None:
            w.buf.store(t, 0, _emit_benign(w.ctx, t))
        else:
            w.buf.store(t, self.fid, frame, self.sid)
            self.ok |= frame.y

    def close(self, w: _Walk, t: int) -> int:
        w.buf.episodes.append(
            EpisodeRow(
                onset=self.onset, length=t - self.onset, family=self.fid, buses=np.where(self.ok)[0].tolist()
            )
        )
        return t


def _episode(w: _Walk, fid: int, t: int) -> _Episode:
    return _Episode(len(w.buf.episodes), fid, t, np.zeros(w.ctx.g.C, np.uint8))


def _ramp_episode(w: _Walk, t: int, ramp_len: int, ramp_rate: float) -> Optional[int]:
    """One slow-ramp episode on a fixed bus set (rise, hold, return), its design from the generator
    (`AttackMixin.ramp_design`); returns the next free timestep, or None with no frame emitted when
    no admissible ramp exists at this onset (the walk then moves the episode, `_relocate`)."""
    ctx = w.ctx
    design = ctx.g.ramp_design(ctx.X, t, (ramp_len, ramp_rate), ctx.knobs, w.buf.last_attack)
    if design is None:
        return None
    ep = _episode(w, RAMP_FAMILY, t)
    if design.tamper is not None:  # the fewest-tamper search ran on this episode
        w.buf.min_rows.append((ep.sid, design.tamper))
    for i in range(ramp_len):  # placement keeps every episode inside its split
        step = ctx.g.ramp_step(design, i, ramp_rate)
        ctx.g.scan_key = t
        ep.store(w, t, attack_frame(ctx.g, ctx.X[t], RAMP_FAMILY, step, ctx.knobs))
        t += 1
    return ep.close(w, t)


def _am_overload_episode(w: _Walk, t: int, length: int) -> Optional[int]:
    """One `Am` episode as the overload attack of [WU26] (`AttackMixin.am_overload_design`,
    `overload_step`): a target branch's reported flow driven to its rating over the window, on the
    support that tampers the fewest devices. Returns the next free timestep, or None with no frame
    emitted when no eligible branch reaches its rating stealthily at this onset (the walk then moves
    the episode, `_relocate`)."""
    ctx = w.ctx
    design: Optional[AmOverloadDesign] = ctx.g.am_overload_design(
        ctx.X, t, length, ctx.knobs, w.buf.last_attack
    )
    if design is None:
        return None
    ep = _episode(w, AM_FAMILY, t)
    w.buf.min_rows.append((ep.sid, design.tamper))
    lines = list(design.goal.lines)
    reached = emitted = np.full(len(lines), np.nan)
    for i in range(length):  # placement keeps every episode inside its split
        ctx.g.scan_key = t
        frame, reached = ctx.g.overload_step(design, ctx.X[t], i, ctx.knobs)
        ep.store(w, t, frame)
        if frame is not None:
            emitted = np.hypot(frame.edge_x[lines, 0], frame.edge_x[lines, 1])
        t += 1
    last = design.goal.targets_at(len(design.goal.targets) - 1)  # the goal at the window's end
    for j, line in enumerate(lines):  # one row per target line (D17)
        w.buf.am_rows.append((ep.sid, line, design.ratings[j], last[j], float(reached[j]), float(emitted[j])))
    return ep.close(w, t)


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


SPLITS = ("train", "val", "test")  # the order of the per-split attributes and of data/split's codes
PLACE_TRIES = 10  # whole-split placements tried before one episode is dropped (D7)
REDRAWS = 20  # onsets an episode with no feasible design is moved to before it is given up (D4)


def _split_bounds(T: int, frac: Sequence[float]) -> list[tuple[int, int]]:
    """The three chronological splits [a, b) of T frames, cut before anything is placed: train
    round(frac[0] T) frames, val round(frac[1] T), test the rest."""
    n_train = min(T, int(round(frac[0] * T)))
    n_val = min(T - n_train, int(round(frac[1] * T)))
    cuts = (0, n_train, n_train + n_val, T)
    return [(cuts[i], cuts[i + 1]) for i in range(3)]


def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    """`total` whole units shared in proportion to `weights` by largest remainder: each gets the floor
    of its quota and the units left go to the largest fractional parts (ties to the earlier one)."""
    if total == 0 or len(weights) == 0:
        return np.zeros(len(weights), int)
    quota = total * np.asarray(weights, float) / float(np.sum(weights))
    counts = np.floor(quota).astype(int)
    order = np.argsort(-(quota - counts), kind="stable")
    counts[order[: total - int(counts.sum())]] += 1
    return counts


def _episode_counts(plan: _Schedule, n: int) -> np.ndarray:
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
        counts = _largest_remainder(k, plan.weights)
        frames = int(counts @ lengths)
        if frames > n:
            break
        if abs(frames - target) < best_err:
            best, best_err = counts, abs(frames - target)
    return best


def _uniform_onset(occupied: np.ndarray, length: int, rng: np.random.Generator, first: int = 0) -> int:
    """An onset drawn uniformly among every position, at or after `first`, where `length` consecutive
    frames are free."""
    free = np.concatenate([[0], np.cumsum(~occupied)])
    T = len(occupied)
    feasible = np.flatnonzero(free[length : T + 1] - free[: T - length + 1] == length)
    feasible = feasible[feasible >= first]
    if len(feasible) == 0:
        raise NoRoomForEpisode(f"no room for a {length}-frame episode")
    return int(feasible[rng.integers(len(feasible))])


def _place_once(
    rng: np.random.Generator, episodes: list[tuple[int, int]], n: int
) -> list[tuple[int, int, int]]:
    """One placement of (length, family) episodes in a split of n frames: longest first, each at an
    onset uniform among those where it fits without touching another (NoRoomForEpisode when one
    does not fit). Returns (onset within the split, family, length)."""
    occupied = np.zeros(n, bool)
    placed: list[tuple[int, int, int]] = []
    for length, fid in sorted(episodes, key=lambda x: -x[0]):
        onset = _uniform_onset(occupied, length, rng)
        occupied[onset : onset + length] = True
        placed.append((onset, fid, length))
    return placed


@dataclass
class _Placement:
    """What the placement asked for and got, per split (rows, `SPLITS`) and family (columns, the
    schedule's families): episodes requested, dropped because they would not fit (D7), moved to
    another onset because no design was feasible (D4), and built."""

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


def _place_split(
    rng: np.random.Generator, plan: _Schedule, split: int, bounds: tuple[int, int], rec: _Placement
) -> list[tuple[int, int, int]]:
    """The episodes of one split as (onset, family, length) in timeline frames, sorted by onset. A
    placement that jams is retried whole up to PLACE_TRIES times; past that one episode of the family
    with the most is dropped, recorded and warned about, and the placement starts over (bounded: at
    most one drop per episode)."""
    a, b = bounds
    counts = _episode_counts(plan, b - a)
    rec.requested[split] = counts
    episodes = [(plan.length_of(f), f) for f, c in zip(plan.families, counts) for _ in range(c)]
    while episodes:
        for _ in range(PLACE_TRIES):
            try:
                return sorted((a + o, f, L) for o, f, L in _place_once(rng, episodes, b - a))
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


def _run_episode(w: _Walk, at: tuple[int, int, int], plan: _Schedule) -> Optional[int]:
    """Build one placed episode; returns the next free timestep, or None when its design is
    infeasible at this onset (nothing emitted)."""
    onset, fid, length = at
    if fid == RAMP_FAMILY:
        return _ramp_episode(w, onset, length, plan.ramp_rate)
    return _am_overload_episode(w, onset, length)


def _relocate(
    rng: np.random.Generator, at: tuple[int, int, int], pending: list[tuple[int, int, int]], end: int
) -> Optional[tuple[int, int, int]]:
    """A new onset for an episode whose design failed at `at`: uniform among the onsets after the
    failed one, inside the split (before `end`), where the episode fits without touching a pending
    episode. Frames before the failed onset are already written, so a moved episode can only move
    later. None when no such onset is left."""
    onset, fid, length = at
    occupied = np.zeros(end - onset, bool)
    for o, _, L in pending:
        occupied[o - onset : o - onset + L] = True
    try:
        return (onset + _uniform_onset(occupied, length, rng, first=1), fid, length)
    except NoRoomForEpisode:
        return None


def _give_up(split: int, fid: int, moves: int) -> None:
    warnings.warn(
        f"split {SPLITS[split]}: one {FAMILIES[fid]} episode found no feasible design after {moves} moves",
        RuntimeWarning,
        stacklevel=3,
    )


def _walk_split(w: _Walk, plan: _Schedule, split: int, bounds: tuple[int, int], rec: _Placement) -> None:
    """Emit one split's frames in time order: benign runs between its episodes, each episode where it
    was placed. An episode whose design is infeasible at its onset moves to a later free onset of
    the split, up to REDRAWS times; past that it is given up (the shortfall, `rec`) and warned about."""
    a, b = bounds
    # (onset, family, length, moves so far): the move count belongs to its episode
    pending = [(*at, 0) for at in _place_split(w.rng, plan, split, bounds, rec)]
    t = a
    while pending:
        *head, moves = pending.pop(0)
        at = (head[0], head[1], head[2])
        t = _benign_run(w, t, at[0])
        nxt = _run_episode(w, at, plan)
        if nxt is not None:
            rec.built[split, rec.col(at[1])] += 1
            t = nxt
            continue
        moved = _relocate(w.rng, at, [p[:3] for p in pending], b) if moves < REDRAWS else None
        if moved is None:
            _give_up(split, at[1], moves)
            continue
        rec.redraws[split, rec.col(at[1])] += 1
        pending = sorted([*pending, (*moved, moves + 1)])
    _benign_run(w, t, b)


def _walk(w: _Walk, plan: _Schedule, bounds: list[tuple[int, int]]) -> _Placement:
    """Walk the timeline split by split (the splits are cut first, `_split_bounds`)."""
    rec = _Placement.empty(list(plan.families))
    for split, span in enumerate(bounds):
        _walk_split(w, plan, split, span, rec)
    return rec


def _split_column(bounds: list[tuple[int, int]], T: int) -> np.ndarray:
    """data/split: train 0, val 1, test 2 by the cut bounds."""
    split = np.zeros(T, np.int8)
    for code, (a, b) in enumerate(bounds):
        split[a:b] = code
    return split


def _ragged(rows: Sequence[np.ndarray], dtype) -> tuple[np.ndarray, np.ndarray]:
    """A list of variable-length rows as (ptr [n+1], flat values)."""
    ptr = np.zeros(len(rows) + 1, np.int64)
    ptr[1:] = np.cumsum([len(r) for r in rows])
    flat = np.concatenate([np.asarray(r, dtype) for r in rows]) if rows else np.zeros(0, dtype)
    return ptr, flat


def _create_layers(f: h5py.File, T: int, C: int, E: int, currents: bool = False) -> dict[str, h5py.Dataset]:
    """The per-frame datasets at full length, chunked along the frame axis and gzipped, empty
    until the walk flushes into them (the PMU current layers only when the plan reads currents)."""
    for group in (schema.Group.DATA, schema.Group.BENIGN, schema.Group.CLEAN, schema.Group.ATTACK):
        f.create_group(group)
    sink = {}
    for name, (shape, dtype) in _layers(currents).items():
        trailing = shape(C, E)
        sink[name] = f.create_dataset(
            name,
            shape=(T, *trailing),
            dtype=dtype,
            chunks=(min(_CHUNK_ROWS, T), *trailing),
            compression="gzip",
            compression_opts=4,
        )
    return sink


def _write_masks(f: h5py.File, buf: _TimelineBuffers, T: int) -> None:
    """data/node_m and data/edge_m per frame (the static plan repeated, written in bounded slabs
    so a future N-1 series can switch topology mid-file without a layout change)."""
    assert buf.node_m is not None and buf.edge_m is not None
    masks = [(schema.NODE_M, buf.node_m), (schema.EDGE_M, buf.edge_m)]
    if buf.i_m is not None:  # hybrid meters: the PMU current channels read at each branch end
        masks.append((schema.PMU_I_M, buf.i_m))
    for name, m in masks:
        ds = f.create_dataset(
            name,
            shape=(T, *m.shape),
            dtype=np.uint8,
            chunks=(min(_CHUNK_ROWS, T), *m.shape),
            compression="gzip",
        )
        for a in range(0, T, 4096):
            ds[a : min(a + 4096, T)] = np.broadcast_to(m, (min(a + 4096, T) - a, *m.shape))


def _write_episodes(f: h5py.File, buf: _TimelineBuffers) -> None:
    """episodes/ (one row per episode, buses ragged) and the ragged magnitudes of attack/."""
    eg = f.create_group(schema.Group.EPISODES)
    ep = buf.episodes
    eg.create_dataset(schema.EPISODE_ONSET, data=np.array([e["onset"] for e in ep], np.int32))
    eg.create_dataset(schema.EPISODE_LENGTH, data=np.array([e["length"] for e in ep], np.int32))
    eg.create_dataset(schema.EPISODE_FAMILY, data=np.array([e["family"] for e in ep], np.int8))
    ptr, idx = _ragged([np.asarray(e["buses"]) for e in ep], np.int32)
    eg.create_dataset(schema.EPISODE_BUS_PTR, data=ptr)
    eg.create_dataset(schema.EPISODE_BUS_IDX, data=idx)
    ptr, bus = _ragged(buf.mag_bus, np.int32)
    _, mag = _ragged(buf.mag, np.float32)
    f.create_dataset(schema.MAG_PTR, data=ptr)
    f.create_dataset(schema.MAG_BUS, data=bus)
    f.create_dataset(schema.MAG, data=mag)
    if buf.min_rows:  # the fewest-tamper knob: what the search chose per episode and whether it is proven
        _write_min_rows(eg, buf.min_rows)
    if buf.am_rows:  # the overload attack: its target branch, rating and the flow it reached
        _write_am_rows(eg, buf.am_rows)
    f[schema.Group.ATTACK].attrs[schema.Attr.TAMPER] = (
        "1 where the attacker wrote the meter: the meters the local false state moves for the "
        "stealthy families, the changed channels for Ad/As/Ar"
    )


def _write_min_rows(eg: h5py.Group, min_rows: list[tuple[int, MinimizerResult]]) -> None:
    """episodes/min_*: one row per episode the fewest-tamper search ran on."""
    rows = [r for _, r in min_rows]
    eg.create_dataset(schema.EPISODE_MIN_EPISODE, data=np.array([s for s, _ in min_rows], np.int32))
    eg.create_dataset(schema.EPISODE_MIN_DEVICES, data=np.array([r.devices for r in rows], np.int32))
    eg.create_dataset(schema.EPISODE_MIN_CHANNELS, data=np.array([r.channels for r in rows], np.int32))
    eg.create_dataset(schema.EPISODE_MIN_PROVEN, data=np.array([r.proven for r in rows], np.uint8))
    eg.create_dataset(schema.EPISODE_MIN_EVALUATED, data=np.array([r.evaluated for r in rows], np.int32))
    eg.create_dataset(schema.EPISODE_MIN_LOWER, data=np.array([r.lower_bound for r in rows], np.int32))
    eg.create_dataset(schema.EPISODE_MIN_UNSOLVED, data=np.array([r.unsolved for r in rows], np.int32))
    ptr, idx = _ragged([np.asarray(r.support) for r in rows], np.int32)
    eg.create_dataset(schema.EPISODE_MIN_SUPPORT_PTR, data=ptr)
    eg.create_dataset(schema.EPISODE_MIN_SUPPORT_IDX, data=idx)


def _write_am_rows(eg: h5py.Group, am_rows: list[tuple[int, int, float, float, float, float]]) -> None:
    """episodes/am_*: one row per target line of each overload Am episode (two rows for a two-line
    episode, D17), the episode, the target branch, its rating (MVA), the goal at the window's end,
    the noiseless apparent flow the last frame reached and the emitted (noisy) one."""
    names = (
        (schema.EPISODE_AM_EPISODE, np.int32),
        (schema.EPISODE_AM_LINE, np.int32),
        (schema.EPISODE_AM_RATING, np.float32),
        (schema.EPISODE_AM_TARGET, np.float32),
        (schema.EPISODE_AM_REACHED, np.float32),
        (schema.EPISODE_AM_EMITTED, np.float32),
    )
    for (name, dtype), column in zip(names, zip(*am_rows)):
        eg.create_dataset(name, data=np.array(column, dtype))


def _placement_attrs(
    rec: _Placement, bounds: list[tuple[int, int]], frac: Sequence[float], seq_id: np.ndarray
) -> dict[str, object]:
    """What the split-first placement asked for and got: the fractions and sizes of the splits, the
    attacked fraction each reached, and per split (rows) and family (columns, `placed_families`) the
    episodes requested, built, moved for want of a feasible design, dropped because they did not
    fit, and short of the request (dropped or given up)."""
    sizes = np.array([b - a for a, b in bounds], int)
    attacked = np.array([int((seq_id[a:b] >= 0).sum()) for a, b in bounds], float)
    return {
        Attr.SPLIT_FRAC: np.asarray(frac, float),
        Attr.SPLIT_SIZES: sizes,
        Attr.SPLIT_ATTACKED_FRAC: attacked / np.maximum(1, sizes),
        Attr.PLACED_FAMILIES: ",".join(FAMILIES[f] for f in rec.families),
        Attr.EPISODES_REQUESTED: rec.requested,
        Attr.EPISODES_BUILT: rec.built,
        Attr.EPISODE_REDRAWS: rec.redraws,
        Attr.EPISODES_DROPPED: rec.dropped,
        Attr.EPISODE_SHORTFALL: rec.requested - rec.built,
        Attr.JITTER_KEYED: 1,
    }


def _timeline_attrs(
    g: FdiaGenerator,
    T: int,
    seed: int,
    buf: _TimelineBuffers,
    knobs: dict[str, Any],  # the recorded knobs, each its own type, written as file attributes
) -> dict[str, object]:
    """The attributes every file carries (dims, units, provenance) plus what makes this one a timeline."""
    attrs: dict[str, object] = dict(_base_attrs(g, T, seed))
    in_episodes = sum(e["length"] for e in buf.episodes)
    attrs.update(
        {
            Attr.KIND: KIND,
            Attr.T: T,
            Attr.FAMILIES: ",".join(f"{k}{v}" for k, v in FAMILIES.items()),
            Attr.ATTACKED_FRAC: float(buf.attacked / max(1, T)),
            Attr.N_EPISODES: len(buf.episodes),
            # frames inside a built episode whose own scan could not be built, stored benign
            Attr.FALLBACK_BENIGN: int(in_episodes) - int((buf.seq_id >= 0).sum()),
        }
    )
    attrs.update({k: (-1 if v is None else v) for k, v in knobs.items()})
    return attrs


def _finish_timeline(
    f: h5py.File,
    g: FdiaGenerator,
    buf: _TimelineBuffers,
    placed: tuple[_Placement, list[tuple[int, int]], Sequence[float]],
    seed: int,
    knobs: dict,
) -> None:
    """After the walk: the last batch, the masks, the small per-frame columns (family, stealthy,
    seq_id, timestep, the split), the episodes, the ragged magnitudes and the attributes."""
    buf.flush()
    T = buf.T
    _write_masks(f, buf, T)
    for name, arr in (
        (schema.FAMILY, buf.family.astype(np.int8)),
        (schema.STEALTHY, np.isin(buf.family, sorted(STEALTHY_FAMILIES)).astype(np.uint8)),
        (schema.SEQ_ID, buf.seq_id),
        (schema.TIMESTEP, np.arange(T, dtype=np.int32)),
        (schema.SPLIT, _split_column(placed[1], T)),
    ):
        f.create_dataset(name, data=arr)
    _write_episodes(f, buf)
    f.attrs.update(_timeline_attrs(g, T, seed, buf, knobs))
    f.attrs.update(_placement_attrs(*placed, buf.seq_id))


def write_temporal_layers(f: h5py.File, block: int = 2000) -> None:
    """`temporal_delta` and `swing` of every frame, from the observed injections alone: each frame
    against the previous emitted frame, the swing scale from the observed changes of the frames
    before it (formulas.temporal.recent_change_scale over SWING_WINDOW frames). Nothing but the
    measurements enters, so a detector reading these features at test time sees only what an operator
    sees. Called after the walk and by trust.secured_copy after it pins meters. Runs in blocks of
    frames with bounded memory: each block's scale comes from the kernel over the block and the
    SWING_WINDOW + 1 frames before it, which covers every window the block's frames use."""
    nx = f[schema.NODE_X]
    T, N = nx.shape[0], nx.shape[1]
    every = np.ones(N, bool)
    delta, swing = f[schema.TEMPORAL_DELTA], f[schema.SWING]
    prev = None
    for a in range(0, T, block):
        b = min(a + block, T)
        scale = _block_scale(nx, a, b)
        rows = np.asarray(nx[a:b], np.float32)
        d = np.zeros(rows.shape[:2] + (2,), np.float32)
        s = np.zeros_like(d)
        for j in range(len(rows)):
            before = rows[j] if prev is None else prev
            d[j] = temporal_delta(rows[j], before, every)
            s[j] = swing_zscore(rows[j], before, scale[j], every)
            prev = rows[j]
        delta[a:b], swing[a:b] = d, s


def _block_scale(nx: h5py.Dataset, a: int, b: int) -> np.ndarray:
    """The swing scale of frames a..b-1 [b - a, N, 2]: the kernel run over the frames from
    SWING_WINDOW + 1 before a to b-1, so every frame's window lies inside the slice."""
    g0 = max(0, a - SWING_WINDOW - 1)
    pq = np.zeros((b - g0, nx.shape[1], 4), np.float64)  # the kernel reads columns 1:3
    pq[:, :, 1:3] = nx[g0:b, :, 1:3]
    return recent_change_scale(pq, SWING_WINDOW, nx.shape[1])[a - g0 :]


def _search_attrs(
    tk: TimelineKnobs, overload: Optional[OverloadSettings], meter_model: Optional[str]
) -> dict[str, object]:
    """The attributes of the searches a walk ran and of the meters it read, written only when they
    apply: the fewest-tamper knobs, the Am attack, the
    stealth scale of At's search (Am has no stealth bound, the plan's D11; an Am-only file records
    the scale it was given, which nothing used), and on a hybrid-meter file its meter model and the
    legend of its current layers; the overload attack's rating source and margin (D15)."""
    out: dict[str, object] = {}
    if tk.min_tamper:
        out.update({Attr.MIN_TAMPER: 1, Attr.MIN_BUDGET: tk.min_budget})
    if overload is not None:
        out[Attr.AM_ATTACK] = tk.am_attack
        out.update(
            {
                Attr.RATING_SOURCE: overload.rating_source,
                Attr.RATING_MARGIN: overload.rating_margin,
                Attr.LOAD_CAP: overload.load_cap,
                Attr.N_LINES: overload.n_lines,
                Attr.SUPPORT_METHOD: overload.support_method,
            }
        )
    if tk.min_tamper or overload is not None:
        out[Attr.STEALTH_SCALE] = tk.stealth_scale
    if meter_model is not None:  # a hybrid-meter file, which reads PMU currents
        out.update(
            {
                Attr.METER_MODEL: meter_model,
                Attr.CURRENT_FEAT: "Re_I_from,Im_I_from,Re_I_to,Im_I_to",
                Attr.CURRENT_UNITS: "pu on the base current",
            }
        )
    return out


def _generator(
    system: Union[int, str], seed: int, max_load_mw: Optional[float], redundancy: Optional[dict]
) -> tuple[FdiaGenerator, MeterSettings]:
    """The walk's generator on the meter plan `redundancy` (a dict of `MeterSettings` fields), and
    that plan."""
    meters = MeterSettings(**(redundancy or {}))
    g = FdiaGenerator(
        system,
        seed=seed,
        max_load_mw=max_load_mw,
        meter_model=meters.meter_model,
        vbus_frac=meters.vbus_frac,
        pmu_frac=meters.pmu_frac,
        flow_frac=meters.flow_frac,
    )
    return g, meters


def generate_timeline(
    system: Union[int, str],
    states: Optional[Union[str, np.ndarray]] = None,
    attacked_frac: float = 0.5,
    families: Sequence[str] = DEFAULT_FAMILIES,
    ramp_rate: float = 0.002,
    ramp_len: int = 60,
    am_len: Optional[int] = None,
    hops: int = 2,
    redundancy: Optional[dict] = None,
    split: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 123,
    out: Optional[str] = None,
    max_load_mw: Optional[float] = 2000.0,
    min_tamper: bool = True,
    min_budget: int = 256,
    am_attack: Union[str, dict] = "overload",
    stealth_scale: float = 1.0,
) -> str:
    """Walk one attacked timeline over the operating-point pool of `system` and write it as one
    HDF5 file. Returns the path (default: `timeline_ieee{N}.h5` under the cache directory).

    attacked_frac    fraction of each split's frames under an attack episode (0.5 = balanced): per
                     split, the whole number of episodes whose frames come closest to it (whole
                     episodes, so it can be off by one episode or two), each placed at an onset
                     uniform among those where it fits in the split, never overlapping or cut, so
                     adjacency and gaps are properties of the draw. An episode with no feasible
                     design at its onset moves to a later free onset of its split (up to 20 times);
                     the attributes record per split the requested, built, moved and dropped
                     episodes and the attacked fraction reached. A frame whose local power flow has
                     no solution at any halving of its step stays benign and is counted in
                     `fallback_benign`
    families         the families in rotation, each getting about the same share of attacked frames:
                     the ramp At and the overload Am (the default, both). The single-snapshot
                     families of data releases v0.8.3 and earlier (Aq, Ad, As, Ar, Al) are refused
                     here and stay loadable from those files
    ramp_rate, ramp_len   the At ramp's per-frame growth and episode length
    am_len           Am episode length (default ramp_len)
    hops             the attacker's subnetwork: buses within this many branches of the attacked loads
                     (At) or of the target lines (Am); the boundary voltages are held true and only
                     the subnetwork's meters are written
    max_load_mw      a load above this (MW) is never a target: an area equivalent, not a substation
                     (IEEE-145 lumps regions into 4 to 58 GW loads); None disables the cap
                     Every false state also satisfies the operating limits: each bus voltage
                     within the case's limits (a bus the true state already holds outside a limit
                     may not be made worse) and every generator's implied output within its P and
                     Q limits widened to the range the pool ran it over; a state outside them is
                     halved; v_lo and v_hi record the widest bus limits of the case
    redundancy       the meter plan, a dict of `MeterSettings` fields: coverage {vbus_frac, pmu_frac,
                     flow_frac}, default 0.6/0.2/0.9, and `meter_model`, what the meters measure (the
                     plan's D10): "hybrid", a SCADA voltmeter reads |V| only, the angle
                     is a PMU channel, and every PMU reads the current phasor of each in-service
                     branch at its bus, stored as data/pmu_i with benign/pmu_i_benign and
                     attack/pmu_i_tamper [WU26, eqs. 17-20]
    split            chronological train/val/test fractions (`SplitSettings`), cut before any
                     episode is placed: every episode lies inside one split
    min_tamper       [WU26, eq. 12]: hold each At episode on the support (the buses the false state
                     moves) that tampers the fewest devices over the episode, a change under a
                     meter's noise not counted (default on); off: the region within `hops`. The
                     search's choice per episode is written under episodes/ (min_*)
    min_budget       candidate supports the search solves per episode before it settles on the best
                     found (recorded as not proven)
    am_attack        "overload" (default): Am is the overload attack of [WU26], a metered branch's
                     reported flow driven to its rating over the episode on the fewest-tamper
                     support, its branch, rating and reached flow under episodes/ (am_*); the
                     rating is 1.25 times the branch's peak true flow over the pool (the plan's
                     D15). A dict of `OverloadSettings` fields asks for the overload attack with
                     other ratings: {"rating_margin": 1.5}, or {"rating_source": "pglib"} for the
                     PGLib-OPF ratings (IEEE-14, 118 and 300 only: NoLineRatings elsewhere)
    stealth_scale    a multiplier on At's stealth bound: each channel's attack step between
                     snapshots at most this many times the meters' rated accuracy (the plan's D7); 1
                     by default. Am has no such bound, as in [WU26]: its noise (0.03 pu SCADA, 0.01 pu
                     PMU, D8) only decides which changes its tamper count ignores (D11)
    """
    am_kind, ratings = OverloadSettings.of(am_attack)
    tk = TimelineKnobs(
        attacked_frac,
        hops,
        ramp_len,
        am_len,
        min_tamper,
        min_budget,
        am_kind,
        stealth_scale,
    )
    splits = SplitSettings(split)  # the train/val/test cut, checked before any work
    fams = GeneratedFamilies(families).codes
    overload = ratings if AM_FAMILY in fams else None  # the overload attack's ratings when it runs
    g, meters = _generator(system, seed, max_load_mw, redundancy)
    red = meters.coverage
    currents = g.current_mask() is not None
    X = _load_states(system, states)
    if round(attacked_frac * len(X)) > 0:  # a timeline placing no attacked frame needs no target
        # the overload Am is not checked here: its targets are the rated, metered lines of each
        # window, decided per episode (a window with none stays benign)
        AdmissibleTargets(tuple(f for f in fams if f != AM_FAMILY), g.target_counts())
        if overload is not None:  # the ratings, before any frame is walked (pglib: NoLineRatings early)
            g.use_line_ratings(overload, X)
    T, C = len(X), g.C
    limits = g.operating_limits(X)  # the constraints every false state must satisfy [WU26]
    knobs = FrameKnobs(
        hops,
        limits,
        tk.min_tamper,
        tk.min_budget,
        tk.stealth_scale,
        getattr(overload, "load_cap", None),  # the overload attack's cap; none without it
        getattr(overload, "n_lines", 1),  # the lines an overload episode drives (D17)
        getattr(overload, "support_method", "search"),  # how its support is chosen
    )
    ctx = _FrameContext(g, X, knobs)
    plan = _Schedule.build(list(fams), tk.ramp_len, ramp_rate, tk.am_frames, tk.attacked_frac)
    out = out or os.path.join(CACHE_DIR, f"timeline_ieee{system_id(system)}.h5")
    recorded = {
        Attr.TARGET_ATTACKED_FRAC: attacked_frac,
        Attr.RAMP_RATE: ramp_rate,
        Attr.RAMP_LEN: ramp_len,
        Attr.AM_LEN: tk.am_frames,
        Attr.HOPS: hops,
        Attr.MAX_LOAD_MW: max_load_mw,
        Attr.V_LO: float(limits.v_lo.min()),
        Attr.V_HI: float(limits.v_hi.max()),
        Attr.VBUS_FRAC: red["vbus_frac"],
        Attr.PMU_FRAC: red["pmu_frac"],
        Attr.FLOW_FRAC: red["flow_frac"],
    }
    recorded.update(_search_attrs(tk, overload, meters.meter_model if currents else None))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:  # the file is open for the whole walk: frames flush in batches
        _write_graph(f, g)
        sink = _create_layers(f, T, C, g.E, currents)
        buf = _TimelineBuffers((T, C, g.E), partial(_clean_slice, g, X), sink=sink, currents=currents)
        bounds = _split_bounds(T, splits.fractions)
        rec = _walk(_Walk(ctx, buf), plan, bounds)
        g.scan_key = None
        _finish_timeline(f, g, buf, (rec, bounds, splits.fractions), seed, recorded)
        write_temporal_layers(f)
    return out
