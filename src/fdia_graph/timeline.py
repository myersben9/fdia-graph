"""One continuous attacked timeline per system, written as one HDF5 file (docs/plans/ONE_DATASET_PLAN.md).

The timeline is the dataset. The walker alternates benign gaps and attack episodes over the
operating-point pool, so every frame is a scan of the grid at one pool timestep, with the attack of
its episode applied (or none), and the file carries everything a user reads afterwards:

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

The episode primitives here are shared with `generate_stream`, whose scheduler and RNG order stay
what the published streams were built with until the streams retire.
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

from .dataset.base import FAMILIES, STEALTHY_FAMILIES
from .engine import FAM_ID, FdiaGenerator
from .engine.records import AM_FAMILY, CORRUPT_KIND, RAMP_FAMILY, Frame, FrameKnobs, attack_frame
from .formulas.attacks import ramp_profile
from .formulas.temporal import swing_zscore, temporal_delta
from .generation import (
    _CHUNK_ROWS,
    NOISE_FLOOR,
    _FrameContext,
    _load_states,
    _shard_attrs,
    _swing_scale,
    _write_graph,
)
from .models.grid import NODE
from .registry import CACHE_DIR, system_id

KIND = "timeline"  # the file attribute that tells a timeline from a shard
DEFAULT_FAMILIES = ("Aq", "Ad", "As", "Ar", "At", "Al", "Am")

# Per-family episode-length band (frames) of the stream scheduler: Aq, Ad, As, Ar, Al.
_EP_LEN = {1: (15, 45), 2: (5, 25), 3: (5, 25), 4: (5, 25), 6: (10, 30)}
_GAP = (5, 40)  # a benign gap draws its length from this band


_BATCH = 256  # frames staged in memory between two flushes to the file
_LAYERS = {  # per-frame datasets: name -> (trailing shape given (C, E), dtype)
    "data/node_x": (lambda C, E: (C, 4), np.float32),
    "data/edge_x": (lambda C, E: (E, 2), np.float32),
    "data/y": (lambda C, E: (C,), np.uint8),
    "data/temporal_delta": (lambda C, E: (C, 2), np.float32),
    "data/swing": (lambda C, E: (C, 2), np.float32),
    "benign/node_benign": (lambda C, E: (C, 4), np.float32),
    "benign/edge_benign": (lambda C, E: (E, 2), np.float32),
    "clean/node_clean": (lambda C, E: (C, 4), np.float32),
    "clean/edge_clean": (lambda C, E: (E, 2), np.float32),
    "attack/node_tamper": (lambda C, E: (C, 4), np.uint8),
    "attack/edge_tamper": (lambda C, E: (E, 2), np.uint8),
}
_ATTACK_LAYERS = ("attack/node_tamper", "attack/edge_tamper")
CleanSlice = Callable[[int, int], tuple[Optional[np.ndarray], np.ndarray]]


def _clean_slice(
    g: FdiaGenerator, X: np.ndarray, a: int, b: int, states: bool = True
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """The noiseless truth of frames a..b-1: the states as float32 (None when the caller keeps the
    pool itself, the streams) and the exact flows on metered branches (one batched matmul, the
    physics primitive the shards use)."""
    return (X[a:b].astype(np.float32) if states else None), g.clean_flows_from_states(X[a:b])


class _TimelineBuffers:
    """The per-frame layers of one timeline, filled frame by frame in time order.

    node_x / edge_x are the OBSERVED measurements (attacked + noisy where attacked, else benign +
    noisy); benign / edge_benign the same scan with the attack removed and the noise kept;
    edge_clean the noiseless true flows on metered branches. temporal_delta and swing are the two
    temporal features against the previous EMITTED frame. seq_id is the episode index of an
    attacked frame, the tamper masks the meters the attacker wrote, and the magnitude lists the
    designed change per attacked bus.

    With a `sink` (the writer: name -> HDF5 dataset of full length T) the layers are staged in a
    batch of _BATCH frames and flushed as the walk passes them, so memory is bounded whatever T;
    without one (the streams) every layer is held whole and read back by the caller, and the clean
    states are not staged at all (the caller has the pool). `attack=False` skips the tamper masks
    and magnitude lists nothing reads.
    """

    def __init__(
        self,
        dims: tuple[int, int, int],
        scale: np.ndarray,
        clean: CleanSlice,
        attack: bool = True,
        sink: Optional[dict[str, Any]] = None,
    ) -> None:
        T, C, E = dims
        n = T if sink is None else min(_BATCH, T)
        self.T, self._n, self._base, self._sink = T, n, 0, sink
        self.attack = attack
        skip: set[str] = set() if attack else set(_ATTACK_LAYERS)
        if sink is None:
            skip.add("clean/node_clean")  # the streams read the pool itself
        self._layers: dict[str, np.ndarray] = {
            name: np.zeros((n, *shape(C, E)), dtype)
            for name, (shape, dtype) in _LAYERS.items()
            if name not in skip
        }
        self.family = np.zeros(T, np.int16)
        self.seq_id = np.full(T, -1, np.int32)
        self.mag_bus: list[np.ndarray] = [np.zeros(0, np.int32)] * T if attack else []
        self.mag: list[np.ndarray] = [np.zeros(0, np.float32)] * T if attack else []
        self.node_m: Optional[np.ndarray] = None  # the meter plan, from the first stored frame
        self.edge_m: Optional[np.ndarray] = None
        self.episodes: list[dict[str, Any]] = []
        self.attacked = 0  # frames stored so far with at least one attacked bus
        self._scale = scale
        self._clean = clean
        self._clean_batch = clean(0, n)
        self._prev_nx: Optional[np.ndarray] = None
        self._all_buses = np.ones(C, bool)

    # The whole layers, for the streams (no sink): the names the stream result reads.
    node_x = property(lambda self: self._layers["data/node_x"])
    benign = property(lambda self: self._layers["benign/node_benign"])
    edge_x = property(lambda self: self._layers["data/edge_x"])
    edge_benign = property(lambda self: self._layers["benign/edge_benign"])
    edge_clean = property(lambda self: self._layers["clean/edge_clean"])
    y = property(lambda self: self._layers["data/y"])
    temporal_delta = property(lambda self: self._layers["data/temporal_delta"])
    swing = property(lambda self: self._layers["data/swing"])

    def store(self, t: int, fid: int, frame: Frame, sid: int = -1) -> None:
        """Store frame t (frames arrive in order): an attacked frame with its un-attacked twin
        (`with_benign`), or a benign one."""
        if t >= self._base + self._n:
            self.flush()
        r = t - self._base
        L = self._layers
        bnx = frame.node_x if frame.benign_node_x is None else frame.benign_node_x
        bex = frame.edge_x if frame.benign_edge_x is None else frame.benign_edge_x
        L["data/node_x"][r] = frame.node_x
        L["benign/node_benign"][r] = bnx
        L["data/y"][r] = frame.y
        L["data/edge_x"][r] = frame.edge_x
        L["benign/edge_benign"][r] = bex
        if "clean/node_clean" in L:
            assert self._clean_batch[0] is not None
            L["clean/node_clean"][r] = self._clean_batch[0][r]
        L["clean/edge_clean"][r] = self._clean_batch[1][r]
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
        L["data/temporal_delta"][r] = temporal_delta(nx, prev, self._all_buses)
        L["data/swing"][r] = swing_zscore(nx, prev, self._scale[t], self._all_buses)
        self._prev_nx = nx

    def _store_attack(self, t: int, fid: int, frame: Frame, bnx: np.ndarray, bex: np.ndarray) -> None:
        """The attacker's footprint on this frame: the designed magnitudes and the tamper masks
        (zeroed on a benign frame: the staged batch rows are reused between flushes)."""
        if not self.attack:
            return
        r = t - self._base
        if fid == 0:
            self._layers["attack/node_tamper"][r] = 0
            self._layers["attack/edge_tamper"][r] = 0
            return
        self.mag_bus[t] = np.asarray(frame.mag_bus, np.int32)
        self.mag[t] = np.asarray(frame.mag, np.float32)
        if frame.tamper is not None:  # Am decides per meter
            node, edge = frame.tamper
        elif fid in CORRUPT_KIND:  # in-place corruption: exactly the meters that changed
            node, edge = frame.node_x != bnx, frame.edge_x != bex
        else:  # a re-solved state: every metered channel carries the attacker's consistent value
            node, edge = frame.node_m > 0, frame.edge_m > 0
        self._layers["attack/node_tamper"][r] = node
        self._layers["attack/edge_tamper"][r] = edge

    def flush(self) -> None:
        """Write the staged frames to the sink and stage the next batch (no-op without a sink)."""
        if self._sink is None:
            return
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


def _want_attack(buf: _TimelineBuffers, t: int, attacked_frac: float) -> bool:
    """Start an attack episode when the attacked fraction so far is below the target (so a target
    of 0 never attacks, and the walk opens with an episode for any positive target)."""
    return (buf.attacked / max(1, t)) < attacked_frac


def _benign_gap(ctx: _FrameContext, buf: _TimelineBuffers, rng: np.random.Generator, t: int) -> int:
    """A benign gap of 5 to 39 frames; returns the next free timestep."""
    T = len(ctx.X)
    gap = int(rng.integers(*_GAP))
    for _ in range(gap):
        if t >= T:
            break
        buf.store(t, 0, _emit_benign(ctx, t))
        t += 1
    return t


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
    apos = ctx.g.attackable_pos
    a = rng.choice(apos, min(5, len(apos)), replace=False)  # fixed bus set for the ramp
    direction = 1.0 if rng.random() < 0.5 else -1.0
    rise = max(1, int(rng.uniform(0.2, 0.45) * ramp_len))
    hold = int(rng.uniform(0.0, 0.25) * ramp_len)
    ep = _episode(ctx, buf, RAMP_FAMILY, t)
    for i in range(ramp_len):
        if t >= T:
            break
        dev = ramp_profile(i, rise, hold, ramp_rate, ramp_rate)
        ep.store(ctx, buf, t, attack_frame(ctx.g, ctx.X[t], RAMP_FAMILY, a, 1 + direction * dev, ctx.knobs))
        t += 1
    return ep.close(buf, t)


def _single_shot_episode(
    ctx: _FrameContext,
    buf: _TimelineBuffers,
    rng: np.random.Generator,
    t: int,
    fid: int,
    length: Optional[int] = None,
) -> int:
    """One episode of a single-shot family held for `length` frames (None: a random length from the
    family's band); returns the next free timestep."""
    T = len(ctx.X)
    a = _pick_targets(rng, ctx.g.attackable_pos, fid)
    mult = 1 + rng.uniform(0.05, ctx.knobs.intensity, size=len(a))
    L = int(rng.integers(*_EP_LEN.get(fid, (5, 25)))) if length is None else length
    ep = _episode(ctx, buf, fid, t)
    for _ in range(L):
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
    red = ctx.g.lra_delta(Lp0, k.intensity, k.lra_k, floor=k.floor)
    a = red.buses
    if len(a) == 0:  # no feasible redistribution at this operating point: one benign frame, no episode
        buf.store(t, 0, _emit_benign(ctx, t))
        return t + 1
    delta = red.delta[a] * _am_sign(direction, rng)
    rel = float(np.max(np.abs(delta) / (np.abs(Lp0[a]) + 1e-6)))
    sh = _AmShape.under_floor(rel, length, am_rate, k.floor)
    ep = _episode(ctx, buf, AM_FAMILY, t)
    for i in range(length):
        if t >= T:
            break
        mult = _am_multipliers(ctx, t, a, sh.at(i) * delta)
        ep.store(ctx, buf, t, attack_frame(ctx.g, ctx.X[t], AM_FAMILY, a, mult, k))
        t += 1
    return ep.close(buf, t)


@dataclass
class _Schedule:
    """How the timeline is walked: the families in rotation, weighted by the inverse of their
    expected episode length so every family gets about the same share of attacked frames, the
    episode shapes, and the attacked fraction the gaps are sized for."""

    families: list[int]
    weights: np.ndarray
    ramp_len: int
    ramp_rate: float
    am: tuple[int, float, str]  # (length, am_rate, am_direction)
    corrupt_len: Optional[int]  # Ad/As/Ar episode length; None draws the stream band
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


def _advance(
    ctx: _FrameContext, buf: _TimelineBuffers, rng: np.random.Generator, t: int, plan: _Schedule
) -> int:
    """One step of the walk: a benign gap when the attacked fraction is on target (or nothing can
    attack), else one episode of a family drawn by the schedule's weights. Returns the next free timestep."""
    if not plan.families or not _want_attack(buf, t, plan.attacked_frac):
        return _benign_gap(ctx, buf, rng, t)
    fid = int(rng.choice(plan.families, p=plan.weights))
    if fid == RAMP_FAMILY:
        return _ramp_episode(ctx, buf, rng, t, plan.ramp_len, plan.ramp_rate)
    if fid == AM_FAMILY:
        return _am_episode(ctx, buf, rng, t, plan.am)
    length = plan.corrupt_len if fid in CORRUPT_KIND else None
    return _single_shot_episode(ctx, buf, rng, t, fid, length)


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
    for group in ("data", "benign", "clean", "attack"):
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
    for name, m in (("data/node_m", buf.node_m), ("data/edge_m", buf.edge_m)):
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
    eg = f.create_group("episodes")
    ep = buf.episodes
    eg.create_dataset("onset", data=np.array([e["onset"] for e in ep], np.int32))
    eg.create_dataset("length", data=np.array([e["length"] for e in ep], np.int32))
    eg.create_dataset("family", data=np.array([e["family"] for e in ep], np.int8))
    ptr, idx = _ragged([np.asarray(e["buses"]) for e in ep], np.int32)
    eg.create_dataset("bus_ptr", data=ptr)
    eg.create_dataset("bus_idx", data=idx)
    ag = f["attack"]
    assert buf.attack, "buffers built without attack storage"
    ptr, bus = _ragged(buf.mag_bus, np.int32)
    _, mag = _ragged(buf.mag, np.float32)
    ag.create_dataset("mag_ptr", data=ptr)
    ag.create_dataset("mag_bus", data=bus)
    ag.create_dataset("mag", data=mag)
    ag.attrs["tamper"] = (
        "1 where the attacker wrote the meter: every metered channel for the re-solved families, "
        "the changed channels for Ad/As/Ar, the beyond-noise set for Am"
    )


def _timeline_attrs(
    g: FdiaGenerator, T: int, seed: int, buf: _TimelineBuffers, knobs: dict[str, Any]
) -> dict[str, Any]:
    """The shard's attributes (dims, units, provenance) plus what makes this file a timeline."""
    attrs = _shard_attrs(g, T, seed, None)
    attrs.update(
        kind=KIND,
        T=T,
        families=",".join(f"{k}{v}" for k, v in FAMILIES.items()),
        attacked_frac=float(buf.attacked / max(1, T)),
        n_episodes=len(buf.episodes),
        fallback_benign=int(sum(e["length"] for e in buf.episodes) - int((buf.seq_id >= 0).sum())),
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
    d = f["data"]
    for name, arr in (
        ("family", buf.family.astype(np.int8)),
        ("stealthy", np.isin(buf.family, sorted(STEALTHY_FAMILIES)).astype(np.uint8)),
        ("seq_id", buf.seq_id),
        ("timestep", np.arange(T, dtype=np.int32)),
        ("split", _frame_split(T, buf.episodes, split)),
    ):
        d.create_dataset(name, data=arr)
    _write_episodes(f, buf)
    f.attrs.update(_timeline_attrs(g, T, seed, buf, knobs))


def _check_knobs(
    attacked_frac: float, am: tuple[float, float, str], lengths: dict[str, Optional[int]]
) -> None:
    """Refuse the knob values that would hang or mislead the walk, before any physics is built.
    `am` = (am_rate, am_sigma, am_direction)."""
    am_rate, am_sigma, am_direction = am
    if am_direction not in ("mask", "induce", "both"):
        raise ValueError(f"am_direction must be 'mask', 'induce' or 'both', got {am_direction!r}")
    if not am_rate > 0:
        raise ValueError(f"am_rate is the per-frame step as a fraction of the noise floor, got {am_rate!r}")
    if not am_sigma >= 0:
        raise ValueError(f"am_sigma is a number of accuracy-class stds, got {am_sigma!r}")
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
    am_sigma: float = 3.0,
    am_direction: str = "both",
    corrupt_len: Optional[int] = 1,
    replay_tau: Optional[int] = None,
    redundancy: Optional[dict] = None,
    split: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 123,
    out: Optional[str] = None,
) -> str:
    """Walk one attacked timeline over the operating-point pool of `system` and write it as one
    HDF5 file. Returns the path (default: `timeline_ieee{N}.h5` under the cache directory).

    attacked_frac    target fraction of frames under an attack episode (0.5 = balanced)
    families         the families in rotation; each gets about the same share of attacked frames
    attack_intensity per-bus load-shift bound of Aq/Al/Am and the plausibility cap of Ad/As/Ar
    ramp_rate, ramp_len   the At ramp's per-frame growth and episode length
    am_len           Am episode length (default ramp_len)
    am_rate          Am's largest per-bus per-frame load change as a fraction of the noise floor
    am_sigma         Am leaves every meter whose designed change is under this many stds un-attacked
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
        (am_rate, am_sigma, am_direction),
        dict(ramp_len=ramp_len, am_len=am_len, corrupt_len=corrupt_len),
    )
    red = {"vbus_frac": 0.6, "pmu_frac": 0.2, "flow_frac": 0.9, **(redundancy or {})}
    g = FdiaGenerator(system, seed=seed, **red)
    lra_k = min(6, len(g.load_bus))
    g._pick_lra_target(attack_intensity, lra_k, n_targets=15)
    X = _load_states(system, states)
    T, C = len(X), g.C
    knobs = FrameKnobs(attack_intensity, NOISE_FLOOR, lra_k, replay_tau, False, True, am_sigma=am_sigma)
    ctx = _FrameContext(g, X, _swing_scale(X, C), knobs, [])
    am = (am_len, am_rate, am_direction)
    plan = _Schedule.build([FAM_ID[f] for f in families], ramp_len, ramp_rate, am, corrupt_len, attacked_frac)
    out = out or os.path.join(CACHE_DIR, f"timeline_ieee{system_id(system)}.h5")
    recorded = dict(
        target_attacked_frac=attacked_frac,
        attack_intensity=attack_intensity,
        ramp_rate=ramp_rate,
        ramp_len=ramp_len,
        am_len=am[0],
        am_rate=am_rate,
        am_sigma=am_sigma,
        am_direction=am_direction,
        corrupt_len=corrupt_len,
        replay_tau=replay_tau,
        noise_floor=NOISE_FLOOR,
        vbus_frac=red["vbus_frac"],
        pmu_frac=red["pmu_frac"],
        flow_frac=red["flow_frac"],
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:  # the file is open for the whole walk: frames flush in batches
        _write_graph(f, g)
        sink = _create_layers(f, T, C, g.E)
        buf = _TimelineBuffers((T, C, g.E), ctx.scale, partial(_clean_slice, g, X), sink=sink)
        t = 0
        while t < T:
            t = _advance(ctx, buf, g.rng, t, plan)
        _finish_timeline(f, g, buf, split, seed, recorded)
    return out
