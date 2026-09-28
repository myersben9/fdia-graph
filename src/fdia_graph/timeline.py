"""One continuous attacked timeline per system, written as one HDF5 file (docs/plans/ONE_DATASET_PLAN.md).

The timeline is the dataset. Attack episodes are placed at uniform random onsets over the
operating-point pool, without overlap, their frames summing to exactly the attacked fraction;
whether two episodes touch or a long quiet stretch separates them is a property of that draw, not
of any rule.
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
from .engine.attacks import episodes as _episodes
from .engine.attacks.episodes import (
    AM_DRAWS,
    ONSET_DRAWS,
    EpisodeDesignMixin,
    am_sign,
    draw_ramp,
    pick_targets,
    probe_frames,
    ramp_dev,
)
from .engine.records import (  # noqa: F401  AQ_HALVINGS, AttackDesign, is_feasible re-exported as before
    AM_FAMILY,
    AQ_HALVINGS,
    CORRUPT_KIND,
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
    NOISE_FLOOR,
    _base_attrs,
    _FrameContext,
    _load_states,
    _write_graph,
)
from .models.choices import (  # noqa: F401  re-exported beside the code that reads them
    BENIGN_CODE,
    DEPRECATED_FOR_GENERATION,
    FAMILY_CODE,
    GENERATED_FAMILIES,
    LEGACY_FAMILIES,
    ONE_FRAME_FAMILIES,
    AmDirection,
)
from .models.config import TimelineKnobs
from .models.data import EpisodeRow
from .models.frames import AmOverloadDesign, AttackVector, MinimizerResult
from .models.inputs import AdmissibleTargets, FamilySelection
from .registry import CACHE_DIR, system_id
from .schema import Attr

KIND = schema.KIND_TIMELINE  # the file attribute that tells a timeline from a shard
# new generation makes the multi-snapshot families [WU26]; LEGACY_FAMILIES (with
# am_attack="redistribution", min_tamper=False) reproduces data release v0.8.3 and the frozen timeline
DEFAULT_FAMILIES = GENERATED_FAMILIES

# Per-family episode-length band (frames): Ad, As, Ar (the upper end excluded), used when
# corrupt_len is None. Aq and Al are single-snapshot attacks: every Aq and Al episode is one frame.
_EP_LEN = {2: (5, 25), 3: (5, 25), 4: (5, 25)}
_ONE_FRAME = {FAMILY_CODE[n] for n in ONE_FRAME_FAMILIES}

# what an episode attacks moved to engine.attacks.episodes (the generator's AttackMixin); the old
# private names keep their old signatures for one minor release, forwarding to the new home
_EPISODES = "engine.attacks.episodes"
_MOVED: dict[str, tuple[str, object]] = {
    "_ONSET_DRAWS": (f"{_EPISODES}.ONSET_DRAWS", ONSET_DRAWS),
    "_AM_DRAWS": (f"{_EPISODES}.AM_DRAWS", AM_DRAWS),
    "_AmShape": (f"{_EPISODES}._AmShape", _episodes._AmShape),
    "_am_sign": (f"{_EPISODES}.am_sign", am_sign),
    "_ramp_dev": (f"{_EPISODES}.ramp_dev", ramp_dev),
    "_pick_targets": (f"{_EPISODES}.pick_targets", pick_targets),
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
    "_draw_single_shot": (
        f"{_EPISODES}.EpisodeDesignMixin.single_shot_design",
        lambda ctx, rng, t, fid, n: _old_pair(ctx.g.single_shot_design(ctx.X, t, fid, n, ctx.knobs, rng=rng)),
    ),
    "_am_multipliers": (
        f"{_EPISODES}.EpisodeDesignMixin._am_multipliers",
        lambda ctx, t, a, delta: ctx.g._am_multipliers(ctx.X[t], a, delta),
    ),
    "_am_peak_solves": (
        f"{_EPISODES}.EpisodeDesignMixin._am_peak_solves",
        lambda ctx, t, a, delta, interior: ctx.g._am_peak_solves(ctx.X[t], a, delta, interior, ctx.knobs),
    ),
    "_am_held_delta": (
        f"{_EPISODES}.EpisodeDesignMixin._am_held_delta",
        lambda ctx, t, a, delta, interior, shape: ctx.g._am_held_delta(
            ctx.X, t, a, delta, interior, shape, ctx.knobs
        ),
    ),
}


def _old_pair(design: Optional[AttackDesign]) -> Optional[tuple[np.ndarray, Union[float, np.ndarray]]]:
    """The (targets, multipliers) pair `_draw_single_shot` returned before it moved."""
    return None if design is None else (design.targets, design.mult)


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
CleanSlice = Callable[[int, int], tuple[np.ndarray, np.ndarray]]


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

    def __init__(self, dims: tuple[int, int, int], clean: CleanSlice, sink: dict[str, h5py.Dataset]) -> None:
        T, C, E = dims
        n = min(_BATCH, T)
        self.T, self._n, self._base, self._sink = T, n, 0, sink
        self._layers: dict[str, np.ndarray] = {
            name: np.zeros((n, *shape(C, E)), dtype) for name, (shape, dtype) in _LAYERS.items()
        }
        self.family = np.zeros(T, np.int16)
        self.seq_id = np.full(T, -1, np.int32)
        self.mag_bus: list[np.ndarray] = [np.zeros(0, np.int32)] * T
        self.mag: list[np.ndarray] = [np.zeros(0, np.float32)] * T
        self.node_m: Optional[np.ndarray] = None  # the meter plan, from the first stored frame
        self.edge_m: Optional[np.ndarray] = None
        self.episodes: list[EpisodeRow] = []
        self.min_rows: list[tuple[int, MinimizerResult]] = []  # (episode, fewest-tamper result), knob on
        # (episode, target branch, rating, noiseless flow reached, emitted flow) per overload Am episode
        self.am_rows: list[tuple[int, int, float, float, float]] = []
        # the attack vector of the last stored frame, observed minus its benign twin (zero when benign):
        # an adjacent episode's stealth bound starts from it
        self.last_attack: AttackVector = (np.zeros((C, 4)), np.zeros((E, 2)))
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
        self.last_attack = (
            np.asarray(frame.node_x, np.float64) - np.asarray(bnx, np.float64),
            np.asarray(frame.edge_x, np.float64) - np.asarray(bex, np.float64),
        )
        self._store_attack(t, fid, frame, bnx, bex)

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
    """The benign scan of timestep t (also remembered for the replay families)."""
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


def _ramp_episode(w: _Walk, t: int, ramp_len: int, ramp_rate: float) -> int:
    """One slow-ramp episode on a fixed bus set (rise, hold, return), its design from the generator
    (`AttackMixin.ramp_design`); returns the next free timestep."""
    ctx, T = w.ctx, w.T
    design = ctx.g.ramp_design(ctx.X, t, (ramp_len, ramp_rate), ctx.knobs, w.buf.last_attack)
    if design is None:  # no admissible ramp at this operating point: the placed frames stay benign, counted
        return _benign_run(w, t, min(t + ramp_len, T))
    ep = _episode(w, RAMP_FAMILY, t)
    if design.tamper is not None:  # the fewest-tamper search ran on this episode
        w.buf.min_rows.append((ep.sid, design.tamper))
    for i in range(ramp_len):
        if t >= T:
            break
        step = ctx.g.ramp_step(design, i, ramp_rate)
        ep.store(w, t, attack_frame(ctx.g, ctx.X[t], RAMP_FAMILY, step, ctx.knobs))
        t += 1
    return ep.close(w, t)


def _single_shot_episode(w: _Walk, t: int, fid: int, length: int) -> int:
    """One episode of a single-shot family held for `length` frames, its design from the generator
    (`AttackMixin.single_shot_design`); returns the next free timestep."""
    ctx, T = w.ctx, w.T
    design = ctx.g.single_shot_design(ctx.X, t, fid, length, ctx.knobs)
    if design is None:  # no admissible design at this operating point: the placed frames stay benign
        return _benign_run(w, t, min(t + length, T))
    ep = _episode(w, fid, t)
    for _ in range(length):
        if t >= T:
            break
        ep.store(w, t, attack_frame(ctx.g, ctx.X[t], fid, design, ctx.knobs))
        t += 1
    return ep.close(w, t)


def _am_overload_episode(w: _Walk, t: int, length: int) -> int:
    """One `Am` episode as the overload attack of [WU26] (`AttackMixin.am_overload_design`,
    `overload_step`): a target branch's reported flow driven to its rating over the window, on the
    support that tampers the fewest devices. Returns the next free timestep."""
    ctx, T = w.ctx, w.T
    design: Optional[AmOverloadDesign] = ctx.g.am_overload_design(
        ctx.X, t, length, ctx.knobs, w.buf.last_attack
    )
    if design is None:  # no eligible branch reaches its rating stealthily here: the frames stay benign
        return _benign_run(w, t, min(t + length, T))
    ep = _episode(w, AM_FAMILY, t)
    w.buf.min_rows.append((ep.sid, design.tamper))
    reached = emitted = float("nan")
    line = design.goal.line
    for i in range(length):
        if t >= T:
            break
        frame, reached = ctx.g.overload_step(design, ctx.X[t], i, ctx.knobs)
        ep.store(w, t, frame)
        if frame is not None:
            emitted = float(np.hypot(*frame.edge_x[line]))
        t += 1
    w.buf.am_rows.append((ep.sid, line, design.rating, reached, emitted))
    return ep.close(w, t)


def _am_episode(w: _Walk, t: int, shape: tuple[int, float, str]) -> int:
    """One multi-snapshot episode after [WU26]: a load redistribution drawn once at onset and applied
    frame by frame along a ramp whose per-bus per-frame step stays under the noise floor
    (`AttackMixin.am_design`, `am_step`). `shape` = (length, am_rate, am_direction). Returns the
    next free timestep."""
    ctx, T = w.ctx, w.T
    length = shape[0]
    design = ctx.g.am_design(ctx.X, t, shape, ctx.knobs)
    if design is None:  # no solvable redistribution at this operating point: the placed frames stay benign
        return _benign_run(w, t, min(t + length, T))
    ep = _episode(w, AM_FAMILY, t)
    for i in range(length):
        if t >= T:
            break
        step = ctx.g.am_step(design, ctx.X[t], i)
        ep.store(w, t, attack_frame(ctx.g, ctx.X[t], AM_FAMILY, step, ctx.knobs))
        t += 1
    return ep.close(w, t)


@dataclass
class _Schedule:
    """What is placed on the timeline: the families in rotation, weighted by the inverse of their
    expected episode length so every family gets about the same share of attacked frames, the
    episode shapes, and the attacked fraction the placement fills up to."""

    families: list[int]
    weights: np.ndarray
    ramp_len: int
    ramp_rate: float
    am: tuple[int, float, str]  # (length, am_rate, am_direction)
    corrupt_len: Optional[int]  # Ad/As/Ar episode length; None draws the band
    attacked_frac: float
    am_overload: bool = False  # Am as the overload attack of [WU26]; False: the v0.8.3 redistribution

    @classmethod
    def build(
        cls, fams: list[int], ramp_len: int, ramp_rate: float, am, corrupt_len, attacked_frac
    ) -> _Schedule:
        expected = {f: float(np.mean(_EP_LEN.get(f, (1, 1)))) for f in fams}
        expected.update({RAMP_FAMILY: float(ramp_len), AM_FAMILY: float(am[0])})
        expected.update({f: 1.0 for f in _ONE_FRAME})
        for f in CORRUPT_KIND:
            if corrupt_len is not None:
                expected[f] = float(corrupt_len)
        w = np.array([1.0 / expected[f] for f in fams], float)
        return cls(fams, w / w.sum() if len(w) else w, ramp_len, ramp_rate, am, corrupt_len, attacked_frac)

    def length_of(self, fid: int, rng: np.random.Generator) -> int:
        """The frames an episode of `fid` takes: one for Aq and Al, the ramp lengths for At and Am,
        `corrupt_len` for Ad/As/Ar when set, else a draw from the family's band."""
        if fid in _ONE_FRAME:
            return 1
        if fid == RAMP_FAMILY:
            return self.ramp_len
        if fid == AM_FAMILY:
            return self.am[0]
        if fid in CORRUPT_KIND and self.corrupt_len is not None:
            return self.corrupt_len
        return int(rng.integers(*_EP_LEN.get(fid, (5, 25))))


def _draw_episodes(rng: np.random.Generator, plan: _Schedule, T: int) -> list[tuple[int, int]]:
    """The episodes as (length, family), drawn by the schedule's weights until their frames sum to
    exactly round(attacked_frac * T); the last one is clipped to the remainder."""
    drawn: list[tuple[int, int]] = []
    frames, target = 0, int(round(plan.attacked_frac * T))
    while frames < target:
        fid = int(rng.choice(plan.families, p=plan.weights))
        length = min(plan.length_of(fid, rng), target - frames)
        drawn.append((length, fid))
        frames += length
    return drawn


def _uniform_onset(occupied: np.ndarray, length: int, rng: np.random.Generator) -> int:
    """An onset drawn uniformly among every position where `length` consecutive frames are free."""
    free = np.concatenate([[0], np.cumsum(~occupied)])
    T = len(occupied)
    feasible = np.flatnonzero(free[length : T + 1] - free[: T - length + 1] == length)
    if len(feasible) == 0:
        raise NoRoomForEpisode(
            f"no room for a {length}-frame episode: the attacked fraction cannot be placed"
        )
    return int(feasible[rng.integers(len(feasible))])


def _place_episodes(rng: np.random.Generator, plan: _Schedule, T: int) -> list[tuple[int, int, int]]:
    """Where the episodes go: (onset, family, length), sorted by onset.

    The episodes are drawn first so their frames sum to exactly the attacked fraction, then placed
    longest first, each at an onset drawn uniformly among the onsets where it fits, so every
    episode is placed (longest first so a long episode is never squeezed out by the many one-frame
    ones). The gaps between episodes, and the episodes that touch, are what that draw gives, not
    a rule."""
    if not plan.families or plan.attacked_frac <= 0:
        return []
    occupied = np.zeros(T, bool)
    placed: list[tuple[int, int, int]] = []
    for length, fid in sorted(_draw_episodes(rng, plan, T), key=lambda x: -x[0]):
        onset = _uniform_onset(occupied, length, rng)
        occupied[onset : onset + length] = True
        placed.append((onset, fid, length))
    return sorted(placed)


def _run_episode(w: _Walk, at: tuple[int, int, int], plan: _Schedule) -> int:
    """Build one placed episode; returns the next free timestep."""
    onset, fid, length = at
    if fid == RAMP_FAMILY:
        return _ramp_episode(w, onset, length, plan.ramp_rate)
    if fid == AM_FAMILY and plan.am_overload:
        return _am_overload_episode(w, onset, length)
    if fid == AM_FAMILY:
        return _am_episode(w, onset, (length, plan.am[1], plan.am[2]))
    return _single_shot_episode(w, onset, fid, length)


def _walk(w: _Walk, plan: _Schedule) -> None:
    """Place the episodes, then emit every frame in time order: benign runs between them, the
    episodes where they were placed."""
    t = 0
    for at in _place_episodes(w.rng, plan, w.T):
        t = _benign_run(w, t, at[0])
        t = _run_episode(w, at, plan)
    _benign_run(w, t, w.T)


def _frame_split(T: int, episodes: list[EpisodeRow], frac: Sequence[float]) -> np.ndarray:
    """train(0)/val(1)/test(2) by chronological order, each boundary moved to the end of the episode
    it would cut, so no episode straddles a split. Boundaries are settled in order and a later one
    never falls before an earlier one, so one long episode across both leaves an empty middle
    split rather than a cut episode (episodes are in onset order, so one pass settles a boundary)."""
    bounds: list[int] = []
    for f in (frac[0], frac[0] + frac[1]):
        b = max(int(f * T), bounds[-1] if bounds else 0)
        for e in episodes:
            if e["onset"] < b < e["onset"] + e["length"]:
                b = e["onset"] + e["length"]
        bounds.append(b)
    split = np.zeros(T, np.int8)
    split[bounds[0] :] = 1
    split[bounds[1] :] = 2
    return split


def _ragged(rows: Sequence[np.ndarray], dtype) -> tuple[np.ndarray, np.ndarray]:
    """A list of variable-length rows as (ptr [n+1], flat values)."""
    ptr = np.zeros(len(rows) + 1, np.int64)
    ptr[1:] = np.cumsum([len(r) for r in rows])
    flat = np.concatenate([np.asarray(r, dtype) for r in rows]) if rows else np.zeros(0, dtype)
    return ptr, flat


def _create_layers(f: h5py.File, T: int, C: int, E: int) -> dict[str, h5py.Dataset]:
    """The per-frame datasets at full length, chunked along the frame axis and gzipped, empty
    until the walk flushes into them."""
    for group in (schema.Group.DATA, schema.Group.BENIGN, schema.Group.CLEAN, schema.Group.ATTACK):
        f.create_group(group)
    sink = {}
    for name, (shape, dtype) in _LAYERS.items():
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
    for name, m in ((schema.NODE_M, buf.node_m), (schema.EDGE_M, buf.edge_m)):
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


def _write_am_rows(eg: h5py.Group, am_rows: list[tuple[int, int, float, float, float]]) -> None:
    """episodes/am_*: one row per overload Am episode, the target branch, its rating (MVA), the
    noiseless apparent flow the last frame reached and the emitted (noisy) one."""
    eg.create_dataset(schema.EPISODE_AM_EPISODE, data=np.array([r[0] for r in am_rows], np.int32))
    eg.create_dataset(schema.EPISODE_AM_LINE, data=np.array([r[1] for r in am_rows], np.int32))
    eg.create_dataset(schema.EPISODE_AM_RATING, data=np.array([r[2] for r in am_rows], np.float32))
    eg.create_dataset(schema.EPISODE_AM_REACHED, data=np.array([r[3] for r in am_rows], np.float32))
    eg.create_dataset(schema.EPISODE_AM_EMITTED, data=np.array([r[4] for r in am_rows], np.float32))


def _timeline_attrs(
    g: FdiaGenerator,
    T: int,
    seed: int,
    buf: _TimelineBuffers,
    knobs: dict[str, Any],  # the recorded knobs, each its own type, written as file attributes
) -> dict[str, Union[int, float, str]]:
    """The attributes every file carries (dims, units, provenance) plus what makes this one a timeline."""
    attrs = _base_attrs(g, T, seed)
    attrs.update(
        {
            Attr.KIND: KIND,
            Attr.T: T,
            Attr.FAMILIES: ",".join(f"{k}{v}" for k, v in FAMILIES.items()),
            Attr.ATTACKED_FRAC: float(buf.attacked / max(1, T)),
            Attr.N_EPISODES: len(buf.episodes),
            Attr.FALLBACK_BENIGN: int(round(knobs[Attr.TARGET_ATTACKED_FRAC] * T))
            - int((buf.seq_id >= 0).sum()),
        }
    )
    attrs.update({k: (-1 if v is None else v) for k, v in knobs.items()})
    return attrs


def _finish_timeline(
    f: h5py.File, g: FdiaGenerator, buf: _TimelineBuffers, split: Sequence[float], seed: int, knobs: dict
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
        (schema.SPLIT, _frame_split(T, buf.episodes, split)),
    ):
        f.create_dataset(name, data=arr)
    _write_episodes(f, buf)
    f.attrs.update(_timeline_attrs(g, T, seed, buf, knobs))


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


def _warn_deprecated_families(fams: Sequence[int]) -> None:
    """A DeprecationWarning when a single-snapshot family is asked of new generation (the plan's D6)."""
    old = [FAMILIES[f] for f in fams if FAMILIES[f] in DEPRECATED_FOR_GENERATION]
    if old:
        warnings.warn(
            f"generating {', '.join(old)} is deprecated and refused from 0.22: new generation makes the "
            "multi-snapshot families At and Am [WU26]; released files holding the single-snapshot "
            "families keep loading",
            DeprecationWarning,
            stacklevel=3,
        )


def generate_timeline(
    system: Union[int, str],
    states: Optional[Union[str, np.ndarray]] = None,
    attacked_frac: float = 0.5,
    families: Sequence[str] = DEFAULT_FAMILIES,
    attack_intensity: float = 0.20,
    ramp_rate: float = 0.002,
    ramp_len: int = 60,
    am_len: Optional[int] = None,
    am_rate: float = 0.9,
    am_direction: str = "both",
    hops: int = 2,
    corrupt_len: Optional[int] = 1,
    replay_tau: Optional[int] = None,
    redundancy: Optional[dict] = None,
    split: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 123,
    out: Optional[str] = None,
    max_load_mw: Optional[float] = 2000.0,
    min_tamper: bool = True,
    min_budget: int = 256,
    am_attack: str = "overload",
    stealth_scale: float = 1.0,
) -> str:
    """Walk one attacked timeline over the operating-point pool of `system` and write it as one
    HDF5 file. Returns the path (default: `timeline_ieee{N}.h5` under the cache directory).

    attacked_frac    fraction of frames under an attack episode (0.5 = balanced): the episodes drawn
                     sum to exactly round(attacked_frac * T) frames and every one is placed, at an
                     onset uniform among those where it fits, so adjacency and gaps are properties
                     of the draw; a frame whose local power flow has no solution at any halving
                     of its step stays benign and is counted in the file's `fallback_benign`
                     attribute (zero on every released file)
    families         the families in rotation; each gets about the same share of attacked frames.
                     New generation makes the multi-snapshot families At and Am (the default);
                     Aq, Ad, As, Ar and Al are deprecated for generation (a DeprecationWarning,
                     refused from 0.22) and stay loadable from released files. Data release v0.8.3
                     is reproduced with families=LEGACY_FAMILIES, am_attack="redistribution",
                     min_tamper=False
    attack_intensity per-bus load-shift bound of Aq/Al/Am and the plausibility cap of Ad/As/Ar
    ramp_rate, ramp_len   the At ramp's per-frame growth and episode length
    am_len           Am episode length (default ramp_len)
    am_rate          Am's largest per-bus per-frame load change as a fraction of the noise floor
    hops             the attacker's subnetwork for the stealthy families: buses within this many
                     branches of the attacked loads (Aq, At) or of the target line (Al, Am); the
                     boundary voltages are held true and only the subnetwork's meters are written
    max_load_mw      a load above this (MW) is never a target: an area equivalent, not a substation
                     (IEEE-145 lumps regions into 4 to 58 GW loads); None disables the cap
                     Every false state also satisfies the operating limits: each bus voltage
                     within the case's limits (a bus the true state already holds outside a limit
                     may not be made worse) and every generator's implied output within its P and
                     Q limits widened to the range the pool ran it over; a state outside them is
                     halved; v_lo and v_hi record the widest bus limits of the case
    am_direction     "induce" (the target line reads more loaded than it is), "mask" (it reads
                     lighter, a real overload hidden, the engine's Al sign) or "both" (drawn per
                     episode)
    corrupt_len      episode length of Ad/As/Ar; 1 (default) makes every such frame an independent
                     draw as in the papers, None draws a 5 to 24 frame episode
    replay_tau       Ar replay depth in frames, None = random lag of at least 20
    redundancy       meter coverage {vbus_frac, pmu_frac, flow_frac}, default 0.6/0.2/0.9
    split            chronological train/val/test fractions by frame, episodes never cut
    min_tamper       [WU26, eq. 12]: hold each At episode on the support (the buses the false state
                     moves) that tampers the fewest devices over the episode, a change under a
                     meter's noise not counted (default on); off: the region within `hops`. The
                     search's choice per episode is written under episodes/ (min_*)
    min_budget       candidate supports the search solves per episode before it settles on the best
                     found (recorded as not proven)
    am_attack        "overload" (default): Am is the overload attack of [WU26], a rated, metered
                     branch's reported flow driven to its PGLib-OPF rating over the episode on the
                     fewest-tamper support (IEEE-14, 118 and 300 only: NoLineRatings elsewhere),
                     its branch, rating and reached flow under episodes/ (am_*); "redistribution":
                     the held load redistribution of data release v0.8.3
    stealth_scale    a multiplier on the stealth bound of the multi-snapshot families: each channel's
                     attack step between snapshots at most this many times [WU26]'s case-study noise
                     for Am (0.03 pu SCADA, 0.01 pu PMU, the plan's D8) and the meters' rated accuracy
                     for At (D7); 1 by default
    """
    tk = TimelineKnobs(
        attacked_frac,
        am_rate,
        hops,
        am_direction,
        ramp_len,
        am_len,
        corrupt_len,
        min_tamper,
        min_budget,
        am_attack,
        stealth_scale,
    )
    fams = FamilySelection(families).codes
    _warn_deprecated_families(fams)
    overload = tk.am_attack == "overload" and AM_FAMILY in fams
    red = {"vbus_frac": 0.6, "pmu_frac": 0.2, "flow_frac": 0.9, **(redundancy or {})}
    g = FdiaGenerator(system, seed=seed, max_load_mw=max_load_mw, **red)
    lra_k = min(6, len(g.load_bus))
    g._pick_lra_target(attack_intensity, lra_k, n_targets=15)
    X = _load_states(system, states)
    if round(attacked_frac * len(X)) > 0:  # a timeline placing no attacked frame needs no target
        AdmissibleTargets(fams, g.target_counts())
        if overload:
            g.line_ratings()  # NoLineRatings on a case without ratings, before any frame is walked
    T, C = len(X), g.C
    limits = g.operating_limits(X)  # the constraints every false state must satisfy [WU26]
    knobs = FrameKnobs(
        attack_intensity,
        NOISE_FLOOR,
        lra_k,
        replay_tau,
        False,
        True,
        hops,
        limits,
        tk.min_tamper,
        tk.min_budget,
        tk.stealth_scale,
    )
    ctx = _FrameContext(g, X, knobs, [])
    am = (tk.am_frames, tk.am_rate, tk.am_direction)
    plan = _Schedule.build(list(fams), tk.ramp_len, ramp_rate, am, tk.corrupt_len, tk.attacked_frac)
    plan.am_overload = overload
    out = out or os.path.join(CACHE_DIR, f"timeline_ieee{system_id(system)}.h5")
    recorded = {
        Attr.TARGET_ATTACKED_FRAC: attacked_frac,
        Attr.ATTACK_INTENSITY: attack_intensity,
        Attr.RAMP_RATE: ramp_rate,
        Attr.RAMP_LEN: ramp_len,
        Attr.AM_LEN: am[0],
        Attr.AM_RATE: am_rate,
        Attr.AM_DIRECTION: am_direction,
        Attr.HOPS: hops,
        Attr.MAX_LOAD_MW: max_load_mw,
        Attr.V_LO: float(limits.v_lo.min()),
        Attr.V_HI: float(limits.v_hi.max()),
        Attr.CORRUPT_LEN: corrupt_len,
        Attr.REPLAY_TAU: replay_tau,
        Attr.NOISE_FLOOR: NOISE_FLOOR,
        Attr.VBUS_FRAC: red["vbus_frac"],
        Attr.PMU_FRAC: red["pmu_frac"],
        Attr.FLOW_FRAC: red["flow_frac"],
    }
    if tk.min_tamper:  # recorded only when on, so a v0.8.3 file's attributes are unchanged
        recorded.update({Attr.MIN_TAMPER: 1, Attr.MIN_BUDGET: tk.min_budget})
    if overload:  # the same: recorded only for the overload attack
        recorded[Attr.AM_ATTACK] = tk.am_attack
    if tk.min_tamper or overload:  # the stealth bound of whichever search ran, At's or Am's
        recorded[Attr.STEALTH_SCALE] = tk.stealth_scale
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:  # the file is open for the whole walk: frames flush in batches
        _write_graph(f, g)
        sink = _create_layers(f, T, C, g.E)
        buf = _TimelineBuffers((T, C, g.E), partial(_clean_slice, g, X), sink=sink)
        _walk(_Walk(ctx, buf), plan)
        _finish_timeline(f, g, buf, split, seed, recorded)
        write_temporal_layers(f)
    return out
