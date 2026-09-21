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

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union

import h5py
import numpy as np

from . import schema
from .dataset.base import FAMILIES, STEALTHY_FAMILIES
from .engine import FAM_ID, FdiaGenerator
from .engine.records import (
    AM_FAMILY,
    CORRUPT_KIND,
    RAMP_FAMILY,
    Frame,
    FrameKnobs,
    attack_frame,
    is_feasible,
)
from .formulas.attacks import ramp_profile
from .formulas.temporal import swing_zscore, temporal_delta
from .generation import (
    _CHUNK_ROWS,
    NOISE_FLOOR,
    _base_attrs,
    _FrameContext,
    _load_states,
    _swing_scale,
    _write_graph,
)
from .models.grid import NODE
from .registry import CACHE_DIR, system_id
from .schema import Attr

KIND = schema.KIND_TIMELINE  # the file attribute that tells a timeline from a shard
DEFAULT_FAMILIES = ("Aq", "Ad", "As", "Ar", "At", "Al", "Am")

# Per-family episode-length band (frames): Aq, Ad, As, Ar, Al (the upper end excluded).
_EP_LEN = {1: (15, 45), 2: (5, 25), 3: (5, 25), 4: (5, 25), 6: (10, 30)}
_AM_DRAWS = 10  # redistribution draws an Am episode gets at its onset before its frames stay benign
_ONSET_DRAWS = 10  # designs an Aq or At episode tries at onset for one with a stealthy state there


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
    edge_clean the noiseless true flows on metered branches. temporal_delta and swing are the two
    temporal features against the previous EMITTED frame. seq_id is the episode index of an
    attacked frame, the tamper masks the meters the attacker wrote, and the magnitude lists the
    designed change per attacked bus.

    The `sink` (name -> HDF5 dataset of full length T) receives the layers in batches of _BATCH
    frames, flushed as the walk passes them, so memory is bounded whatever T.
    """

    def __init__(
        self, dims: tuple[int, int, int], scale: np.ndarray, clean: CleanSlice, sink: dict[str, Any]
    ) -> None:
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
        self.episodes: list[dict[str, Any]] = []
        self.attacked = 0  # frames stored so far with at least one attacked bus
        self._scale = scale
        self._clean = clean
        self._clean_batch = clean(0, n)
        self._prev_nx: Optional[np.ndarray] = None
        self._all_buses = np.ones(C, bool)

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
        self._store_attack(t, fid, frame, bnx, bex)
        # The two temporal features against the previous EMITTED frame, through the same kernel the
        # shard uses; computed at every bus (an unmetered bus reads 0 - 0).
        nx = frame.node_x
        prev = self._prev_nx if self._prev_nx is not None else nx
        L[schema.TEMPORAL_DELTA][r] = temporal_delta(nx, prev, self._all_buses)
        L[schema.SWING][r] = swing_zscore(nx, prev, self._scale[t], self._all_buses)
        self._prev_nx = nx

    def _store_attack(self, t: int, fid: int, frame: Frame, bnx: np.ndarray, bex: np.ndarray) -> None:
        """The attacker's footprint on this frame: the designed magnitudes and the tamper masks
        (zeroed on a benign frame: the staged batch rows are reused between flushes)."""
        r = t - self._base
        if fid == 0:
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
    frame = attack_frame(ctx.g, ctx.X[t], 0, None, None, ctx.knobs)
    assert frame is not None  # a benign emission cannot fail
    return frame


def _pick_targets(rng: np.random.Generator, apos: np.ndarray, fid: int) -> np.ndarray:
    """Attacked load-table positions for an episode: 1 to 6 buses for Aq, up to 4 otherwise."""
    nab = len(apos)
    k = int(rng.integers(1, min(6, nab) + 1)) if fid == 1 else min(4, nab)
    return rng.choice(apos, k, replace=False)


def _benign_run(ctx: _FrameContext, buf: _TimelineBuffers, t: int, until: int) -> int:
    """Benign frames from t up to `until` (excluded); returns `until`."""
    for t in range(t, until):
        buf.store(t, 0, _emit_benign(ctx, t))
    return until


@dataclass
class _Episode:
    """One episode being built: its index, family, and the buses attacked so far."""

    sid: int
    fid: int
    onset: int
    ok: np.ndarray  # [N] uint8, the buses any frame of the episode labelled

    def store(self, ctx: _FrameContext, buf: _TimelineBuffers, t: int, frame: Optional[Frame]) -> None:
        """Store the attacked frame, or a benign one when the attack could not be built at t."""
        if frame is None:
            buf.store(t, 0, _emit_benign(ctx, t))
        else:
            buf.store(t, self.fid, frame, self.sid)
            self.ok |= frame.y

    def close(self, buf: _TimelineBuffers, t: int) -> int:
        buf.episodes.append(
            dict(
                onset=self.onset, length=t - self.onset, family=self.fid, buses=np.where(self.ok)[0].tolist()
            )
        )
        return t


def _episode(ctx: _FrameContext, buf: _TimelineBuffers, fid: int, t: int) -> _Episode:
    return _Episode(len(buf.episodes), fid, t, np.zeros(ctx.g.C, np.uint8))


def _ramp_episode(
    ctx: _FrameContext,
    buf: _TimelineBuffers,
    rng: np.random.Generator,
    t: int,
    ramp_len: int,
    ramp_rate: float,
) -> int:
    """One slow-ramp episode on a fixed bus set (rise, hold, return); returns the next free timestep."""
    T = len(ctx.X)
    for _ in range(_ONSET_DRAWS):  # a design whose peak has no stealthy state on its peak frames is redrawn
        a, direction, rise, hold = _draw_ramp(ctx, rng, ramp_len)
        peak = 1 + direction * ramp_rate * rise
        frames = [min(u, T - 1) for u in (t + rise, t + rise + hold)]  # the peak's first and last frame
        if all(is_feasible(ctx.g, ctx.X[u], a, peak, ctx.knobs) for u in frames):
            break
    ep = _episode(ctx, buf, RAMP_FAMILY, t)
    for i in range(ramp_len):
        if t >= T:
            break
        dev = ramp_profile(i, rise, hold, ramp_rate, ramp_rate)
        ep.store(ctx, buf, t, attack_frame(ctx.g, ctx.X[t], RAMP_FAMILY, a, 1 + direction * dev, ctx.knobs))
        t += 1
    return ep.close(buf, t)


def _draw_ramp(
    ctx: _FrameContext, rng: np.random.Generator, ramp_len: int
) -> tuple[np.ndarray, float, int, int]:
    """One ramp design: a fixed bus set, a direction, the rise and hold lengths (four draws)."""
    apos = ctx.g.attackable_pos
    a = rng.choice(apos, min(5, len(apos)), replace=False)
    direction = 1.0 if rng.random() < 0.5 else -1.0
    rise = max(1, int(rng.uniform(0.2, 0.45) * ramp_len))
    hold = int(rng.uniform(0.0, 0.25) * ramp_len)
    return a, direction, rise, hold


def _probe_frames(ctx: _FrameContext, t: int, length: int) -> list[int]:
    """The frames an episode's design is tested on before it is accepted: its first, middle and
    last, since the operating point drifts along the episode and a design feasible at onset can
    lose its stealthy state later (the pool is known ahead, so the walker can look)."""
    last = min(t + length, len(ctx.X)) - 1
    return sorted({t, (t + last) // 2, last})


def _draw_single_shot(
    ctx: _FrameContext, rng: np.random.Generator, t: int, fid: int, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """The targets and load multipliers of an episode (two draws); an Aq design with no stealthy
    state on the episode's first, middle or last frame is redrawn, up to _ONSET_DRAWS."""
    for _ in range(_ONSET_DRAWS):
        a = _pick_targets(rng, ctx.g.attackable_pos, fid)
        mult = 1 + rng.uniform(0.05, ctx.knobs.intensity, size=len(a))
        if fid != 1 or all(
            is_feasible(ctx.g, ctx.X[u], a, mult, ctx.knobs) for u in _probe_frames(ctx, t, length)
        ):
            break
    return a, mult


def _single_shot_episode(
    ctx: _FrameContext,
    buf: _TimelineBuffers,
    rng: np.random.Generator,
    t: int,
    fid: int,
    length: int,
) -> int:
    """One episode of a single-shot family held for `length` frames; returns the next free timestep."""
    T = len(ctx.X)
    a, mult = _draw_single_shot(ctx, rng, t, fid, length)
    ep = _episode(ctx, buf, fid, t)
    for _ in range(length):
        if t >= T:
            break
        ep.store(ctx, buf, t, attack_frame(ctx.g, ctx.X[t], fid, a, mult, ctx.knobs))
        t += 1
    return ep.close(buf, t)


@dataclass
class _AmShape:
    """The schedule of one Am episode: how much of the held redistribution each frame applies."""

    rate: float  # fraction of the full redistribution added per frame on the rise and the fall
    rise: int
    hold: int

    @classmethod
    def under_floor(cls, rel: float, length: int, am_rate: float, floor: float) -> _AmShape:
        """Rise so that the largest per-bus per-frame load change stays under `am_rate` of the noise
        floor: `rel` is the largest per-bus redistribution fraction, so a step of `rate` of the whole
        moves that bus by rate * rel. The rate is never raised: an episode too short to reach the
        full redistribution ramps as far as it gets and returns, its peak below the full delta."""
        rate = min(1.0, am_rate * floor / max(rel, 1e-9))
        rise = min(math.ceil(1.0 / rate), max(1, length // 2))
        return cls(rate, rise, max(0, length - 2 * rise))

    def at(self, i: int) -> float:
        return min(1.0, ramp_profile(i, self.rise, self.hold, self.rate, self.rate))


def _am_multipliers(ctx: _FrameContext, t: int, a: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """The load multipliers that add `delta` (MW, per target) to this frame's true load."""
    Lp = ctx.X[t][ctx.g.load_bus[a], NODE.p_inj] + ctx.g.load_genP[a]
    return 1.0 + delta / np.where(np.abs(Lp) > 1e-9, Lp, 1e-9)


def _am_peak_solves(ctx: _FrameContext, t: int, a: np.ndarray, delta: np.ndarray, interior) -> bool:
    """Whether the held redistribution at its peak (`delta`, MW per target) has a stealthy state
    (a local solution inside the operating limits) on the onset state; no random draw is spent."""
    return is_feasible(ctx.g, ctx.X[t], a, _am_multipliers(ctx, t, a, delta), ctx.knobs, interior)


def _am_sign(direction: str, rng: np.random.Generator) -> float:
    """The sign applied to the engine's redistribution. `lra_delta` orients its delta to raise the
    target line's loading in the false state, so "induce" (a safe line reads as overloaded, the
    [WU26] objective) keeps it (+1) and "mask" (a real overload reads lighter) flips it (-1);
    "both" draws one of the two per episode (one RNG draw)."""
    if direction == "both":
        direction = "induce" if rng.random() < 0.5 else "mask"
    return 1.0 if direction == "induce" else -1.0


def _am_episode(
    ctx: _FrameContext, buf: _TimelineBuffers, rng: np.random.Generator, t: int, shape: tuple[int, float, str]
) -> int:
    """One multi-snapshot episode [WU26]: a load redistribution drawn once at onset (the Al
    construction, PTDF-ranked buses, load-conserving), then applied frame by frame along a ramp
    whose per-bus per-frame step stays under the noise floor, every frame re-solved and made sparse
    by `engine.records._am_frame`. `shape` = (length, am_rate, am_direction), the direction
    resolved by `_am_sign`. Returns the next free timestep."""
    T, k = len(ctx.X), ctx.knobs
    length, am_rate, direction = shape
    Lp0 = ctx.X[t][ctx.g.load_bus, NODE.p_inj] + ctx.g.load_genP
    for _ in range(_AM_DRAWS):  # redrawn when the target-line pool gives no redistribution, or one
        red = ctx.g.lra_delta(Lp0, k.intensity, k.lra_k, floor=k.floor, hops=k.hops)  # without a
        a = red.buses  # local power-flow solution at its peak (a held redistribution must reach it)
        if len(a) == 0:
            continue
        delta = red.delta[a] * _am_sign(direction, rng)
        rel = float(np.max(np.abs(delta) / (np.abs(Lp0[a]) + 1e-6)))
        sh = _AmShape.under_floor(rel, length, am_rate, k.floor)
        if _am_peak_solves(ctx, min(t + sh.rise, T - 1), a, sh.at(sh.rise) * delta, red.interior):
            break
    else:  # no solvable redistribution at this operating point: the placed frames stay benign
        return _benign_run(ctx, buf, t, min(t + length, T))
    ep = _episode(ctx, buf, AM_FAMILY, t)
    for i in range(length):
        if t >= T:
            break
        mult = _am_multipliers(ctx, t, a, sh.at(i) * delta)
        ep.store(ctx, buf, t, attack_frame(ctx.g, ctx.X[t], AM_FAMILY, a, mult, k, red.interior))
        t += 1
    return ep.close(buf, t)


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

    @classmethod
    def build(
        cls, fams: list[int], ramp_len: int, ramp_rate: float, am, corrupt_len, attacked_frac
    ) -> _Schedule:
        expected = {f: float(np.mean(_EP_LEN.get(f, (1, 1)))) for f in fams}
        expected.update({RAMP_FAMILY: float(ramp_len), AM_FAMILY: float(am[0])})
        for f in CORRUPT_KIND:
            if corrupt_len is not None:
                expected[f] = float(corrupt_len)
        w = np.array([1.0 / expected[f] for f in fams], float)
        return cls(fams, w / w.sum() if len(w) else w, ramp_len, ramp_rate, am, corrupt_len, attacked_frac)

    def length_of(self, fid: int, rng: np.random.Generator) -> int:
        """The frames an episode of `fid` takes: the ramp lengths for At and Am, `corrupt_len` for
        Ad/As/Ar when set, else a draw from the family's band."""
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
        raise ValueError(f"no room for a {length}-frame episode: the attacked fraction cannot be placed")
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


def _run_episode(
    ctx: _FrameContext,
    buf: _TimelineBuffers,
    rng: np.random.Generator,
    at: tuple[int, int, int],
    plan: _Schedule,
) -> int:
    """Build one placed episode; returns the next free timestep."""
    onset, fid, length = at
    if fid == RAMP_FAMILY:
        return _ramp_episode(ctx, buf, rng, onset, length, plan.ramp_rate)
    if fid == AM_FAMILY:
        return _am_episode(ctx, buf, rng, onset, (length, plan.am[1], plan.am[2]))
    return _single_shot_episode(ctx, buf, rng, onset, fid, length)


def _walk(ctx: _FrameContext, buf: _TimelineBuffers, rng: np.random.Generator, plan: _Schedule) -> None:
    """Place the episodes, then emit every frame in time order: benign runs between them, the
    episodes where they were placed."""
    T = len(ctx.X)
    t = 0
    for at in _place_episodes(rng, plan, T):
        t = _benign_run(ctx, buf, t, at[0])
        t = _run_episode(ctx, buf, rng, at, plan)
    _benign_run(ctx, buf, t, T)


def _frame_split(T: int, episodes: list[dict[str, Any]], frac: Sequence[float]) -> np.ndarray:
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


def _create_layers(f: h5py.File, T: int, C: int, E: int) -> dict[str, Any]:
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
    f[schema.Group.ATTACK].attrs[schema.Attr.TAMPER] = (
        "1 where the attacker wrote the meter: the meters the local false state moves for the "
        "stealthy families, the changed channels for Ad/As/Ar"
    )


def _timeline_attrs(
    g: FdiaGenerator, T: int, seed: int, buf: _TimelineBuffers, knobs: dict[str, Any]
) -> dict[str, Any]:
    """The attributes every file carries (dims, units, provenance) plus what makes this one a timeline."""
    attrs = _base_attrs(g, T, seed)
    attrs.update(
        {
            Attr.KIND: KIND,
            Attr.T: T,
            Attr.FAMILIES: ",".join(f"{k}{v}" for k, v in FAMILIES.items()),
            Attr.ATTACKED_FRAC: float(buf.attacked / max(1, T)),
            Attr.N_EPISODES: len(buf.episodes),
            Attr.FALLBACK_BENIGN: int(sum(e["length"] for e in buf.episodes) - int((buf.seq_id >= 0).sum())),
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


def _check_knobs(
    attacked_frac: float, am: tuple[float, float, str], lengths: dict[str, Optional[int]]
) -> None:
    """Refuse the knob values that would hang or mislead the walk, before any physics is built.
    `am` = (am_rate, hops, am_direction)."""
    am_rate, hops, am_direction = am
    if am_direction not in ("mask", "induce", "both"):
        raise ValueError(f"am_direction must be 'mask', 'induce' or 'both', got {am_direction!r}")
    if not am_rate > 0:
        raise ValueError(f"am_rate is the per-frame step as a fraction of the noise floor, got {am_rate!r}")
    if not (isinstance(hops, int) and hops >= 1):
        raise ValueError(f"hops is the attacker's reach in branches, at least 1, got {hops!r}")
    if not 0.0 <= attacked_frac <= 1.0:
        raise ValueError(f"attacked_frac is a fraction of frames, got {attacked_frac!r}")
    for knob, value in lengths.items():
        if value is not None and value < 1:  # an empty episode would store no frame and never advance
            raise ValueError(f"{knob} must be at least 1 frame, got {value!r}")


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
    max_load_mw: Optional[float] = 2000.0,
    corrupt_len: Optional[int] = 1,
    replay_tau: Optional[int] = None,
    redundancy: Optional[dict] = None,
    split: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 123,
    out: Optional[str] = None,
) -> str:
    """Walk one attacked timeline over the operating-point pool of `system` and write it as one
    HDF5 file. Returns the path (default: `timeline_ieee{N}.h5` under the cache directory).

    attacked_frac    fraction of frames under an attack episode (0.5 = balanced): the episodes drawn
                     sum to exactly round(attacked_frac * T) frames and every one is placed, at an
                     onset uniform among those where it fits, so adjacency and gaps are properties
                     of the draw; a frame whose local power flow has no solution at any halving
                     of its step stays benign and is counted in the file's `fallback_benign`
                     attribute (zero on every released file)
    families         the families in rotation; each gets about the same share of attacked frames
    attack_intensity per-bus load-shift bound of Aq/Al/Am and the plausibility cap of Ad/As/Ar
    ramp_rate, ramp_len   the At ramp's per-frame growth and episode length
    am_len           Am episode length (default ramp_len)
    am_rate          Am's largest per-bus per-frame load change as a fraction of the noise floor
    hops             the attacker's subnetwork for the stealthy families: buses within this many
                     branches of the attacked loads (Aq, At) or of the target line (Al, Am); the
                     boundary voltages are held true and only the subnetwork's meters are written
    max_load_mw      a load above this (MW) is never a target: an area equivalent, not a substation
                     (IEEE-145 lumps regions into 4 to 58 GW loads); None disables the cap
                     Every false state also satisfies the case's operating limits (each bus
                     voltage within its limits, every generator's implied output within its P and
                     Q limits, and where the true state is already outside a limit the false state
                     may not make it worse); a state outside them is halved; v_lo and v_hi record
                     the widest bus limits of the case
    am_direction     "induce" (the target line reads more loaded than it is, the engine's Al sign),
                     "mask" (it reads lighter, a real overload hidden) or "both" (drawn per episode)
    corrupt_len      episode length of Ad/As/Ar; 1 (default) makes every such frame an independent
                     draw as in the papers, None draws the stream's 5 to 25 frame band
    replay_tau       Ar/As replay depth in frames, None = random lag of at least 20
    redundancy       meter coverage {vbus_frac, pmu_frac, flow_frac}, default 0.6/0.2/0.9
    split            chronological train/val/test fractions by frame, episodes never cut
    """
    am_len = ramp_len if am_len is None else am_len
    _check_knobs(
        attacked_frac,
        (am_rate, hops, am_direction),
        dict(ramp_len=ramp_len, am_len=am_len, corrupt_len=corrupt_len),
    )
    red = {"vbus_frac": 0.6, "pmu_frac": 0.2, "flow_frac": 0.9, **(redundancy or {})}
    g = FdiaGenerator(system, seed=seed, max_load_mw=max_load_mw, **red)
    lra_k = min(6, len(g.load_bus))
    g._pick_lra_target(attack_intensity, lra_k, n_targets=15)
    X = _load_states(system, states)
    T, C = len(X), g.C
    limits = g.operating_limits()  # the constraints every false state must satisfy [WU26]
    knobs = FrameKnobs(attack_intensity, NOISE_FLOOR, lra_k, replay_tau, False, True, hops, limits)
    ctx = _FrameContext(g, X, _swing_scale(X, C), knobs, [])
    am = (am_len, am_rate, am_direction)
    plan = _Schedule.build([FAM_ID[f] for f in families], ramp_len, ramp_rate, am, corrupt_len, attacked_frac)
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
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:  # the file is open for the whole walk: frames flush in batches
        _write_graph(f, g)
        sink = _create_layers(f, T, C, g.E)
        buf = _TimelineBuffers((T, C, g.E), ctx.scale, partial(_clean_slice, g, X), sink=sink)
        _walk(ctx, buf, g.rng, plan)
        _finish_timeline(f, g, buf, split, seed, recorded)
    return out
