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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import h5py
import numpy as np

from .dataset.base import FAMILIES, STEALTHY_FAMILIES
from .engine import FAM_ID, FdiaGenerator
from .engine.records import AM_FAMILY, CORRUPT_KIND, RAMP_FAMILY, Frame, FrameKnobs, attack_frame
from .formulas.attacks import ramp_profile
from .formulas.temporal import swing_zscore, temporal_delta
from .generation import (
    NOISE_FLOOR,
    _chunked,
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


class _TimelineBuffers:
    """The per-frame layers of one timeline, allocated once and filled frame by frame.

    node_x / edge_x are the OBSERVED measurements (attacked + noisy where attacked, else benign +
    noisy); benign / edge_benign the same scan with the attack removed and the noise kept;
    edge_clean the noiseless true flows on metered branches. temporal_delta and swing are the two
    temporal features against the previous EMITTED frame. seq_id is the episode index of an
    attacked frame, the tamper masks the meters the attacker wrote, and the magnitude lists the
    designed change per attacked bus.
    """

    def __init__(
        self, T: int, C: int, E: int, scale: np.ndarray, edge_clean_full: np.ndarray, attack: bool = True
    ) -> None:
        """`attack=False` (the streams) skips the tamper masks and magnitude lists nothing reads."""
        self.node_x = np.zeros((T, C, 4), np.float32)
        self.benign = np.zeros((T, C, 4), np.float32)
        self.edge_x = np.zeros((T, E, 2), np.float32)
        self.edge_benign = np.zeros((T, E, 2), np.float32)
        self.edge_clean = np.zeros((T, E, 2), np.float32)
        self.y = np.zeros((T, C), np.uint8)
        self.family = np.zeros(T, np.int16)
        self.seq_id = np.full(T, -1, np.int32)
        self.temporal_delta = np.zeros((T, C, 2), np.float32)
        self.swing = np.zeros((T, C, 2), np.float32)
        self.attack = attack
        self.node_tamper = np.zeros((T, C, 4), np.uint8) if attack else None
        self.edge_tamper = np.zeros((T, E, 2), np.uint8) if attack else None
        self.mag_bus: list[np.ndarray] = [np.zeros(0, np.int32)] * T if attack else []
        self.mag: list[np.ndarray] = [np.zeros(0, np.float32)] * T if attack else []
        self.node_m: Optional[np.ndarray] = None  # the meter plan, from the first stored frame
        self.edge_m: Optional[np.ndarray] = None
        self.episodes: list[dict[str, Any]] = []
        self.attacked = 0  # frames stored so far with at least one attacked bus
        self._scale = scale
        self._edge_clean_full = edge_clean_full
        self._prev_nx: Optional[np.ndarray] = None
        self._all_buses = np.ones(C, bool)

    def store(self, t: int, fid: int, frame: Frame, sid: int = -1) -> None:
        """Store an attacked frame with its un-attacked twin (`with_benign`), or a benign one."""
        bnx = frame.node_x if frame.benign_node_x is None else frame.benign_node_x
        bex = frame.edge_x if frame.benign_edge_x is None else frame.benign_edge_x
        self.node_x[t] = frame.node_x
        self.benign[t] = bnx
        self.y[t] = frame.y
        self.family[t] = fid
        self.seq_id[t] = sid if fid else -1
        self.edge_x[t] = frame.edge_x
        self.edge_benign[t] = bex
        self.edge_clean[t] = self._edge_clean_full[t]
        self.attacked += int(frame.y.any())
        if self.node_m is None:
            self.node_m, self.edge_m = frame.node_m, frame.edge_m
        self._store_attack(t, fid, frame, bnx, bex)
        # The two temporal features against the previous EMITTED frame, through the same kernel the
        # shard uses; computed at every bus (an unmetered bus reads 0 - 0).
        nx = frame.node_x
        p = self._prev_nx if self._prev_nx is not None else nx
        self.temporal_delta[t] = temporal_delta(nx, p, self._all_buses)
        self.swing[t] = swing_zscore(nx, p, self._scale[t], self._all_buses)
        self._prev_nx = nx

    def _store_attack(self, t: int, fid: int, frame: Frame, bnx: np.ndarray, bex: np.ndarray) -> None:
        """The attacker's footprint on this frame: the designed magnitudes and the tamper masks."""
        if fid == 0 or not self.attack:
            return
        assert self.node_tamper is not None and self.edge_tamper is not None
        self.mag_bus[t] = np.asarray(frame.mag_bus, np.int32)
        self.mag[t] = np.asarray(frame.mag, np.float32)
        if frame.tamper is not None:  # Am decides per meter
            node, edge = frame.tamper
        elif fid in CORRUPT_KIND:  # in-place corruption: exactly the meters that changed
            node, edge = frame.node_x != bnx, frame.edge_x != bex
        else:  # a re-solved state: every metered channel carries the attacker's consistent value
            node, edge = frame.node_m > 0, frame.edge_m > 0
        self.node_tamper[t] = node
        self.edge_tamper[t] = edge


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


def _am_episode(
    ctx: _FrameContext, buf: _TimelineBuffers, rng: np.random.Generator, t: int, shape: tuple[int, float, str]
) -> int:
    """One multi-snapshot episode [WU26]: a load redistribution drawn once at onset (the Al
    construction, PTDF-ranked buses, load-conserving), then applied frame by frame along a ramp
    whose per-bus per-frame step stays under the noise floor, every frame re-solved and made sparse
    by `engine.records._am_frame`. `shape` = (length, am_rate, am_direction). "mask" keeps the
    redistribution's own sign (it hides a real overload, as Al does); "induce" flips it (a safe line
    reads as overloaded); "both" draws one of the two per episode. Returns the next free timestep."""
    T, k = len(ctx.X), ctx.knobs
    length, am_rate, direction = shape
    Lp0 = ctx.X[t][ctx.g.load_bus, NODE.p_inj] + ctx.g.load_genP
    red = ctx.g.lra_delta(Lp0, k.intensity, k.lra_k, floor=k.floor)
    a = red.buses
    if len(a) == 0:  # no feasible redistribution at this operating point: one benign frame, no episode
        buf.store(t, 0, _emit_benign(ctx, t))
        return t + 1
    if direction == "both":
        direction = "mask" if rng.random() < 0.5 else "induce"
    delta = red.delta[a] * (1.0 if direction == "mask" else -1.0)
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


def _write_data(f: h5py.File, X: np.ndarray, buf: _TimelineBuffers, split: np.ndarray) -> None:
    """data/, benign/ and clean/: one row per frame, chunked along the frame axis."""
    T = len(split)
    assert buf.node_m is not None and buf.edge_m is not None
    d = f.create_group("data")
    for name, arr in (
        ("node_x", buf.node_x),
        ("node_m", np.ascontiguousarray(np.broadcast_to(buf.node_m, (T, *buf.node_m.shape)))),
        ("edge_x", buf.edge_x),
        ("edge_m", np.ascontiguousarray(np.broadcast_to(buf.edge_m, (T, *buf.edge_m.shape)))),
        ("y", buf.y),
        ("temporal_delta", buf.temporal_delta),
        ("swing", buf.swing),
    ):
        _chunked(d, name, arr)
    stealthy = np.isin(buf.family, sorted(STEALTHY_FAMILIES)).astype(np.uint8)
    for name, arr in (
        ("family", buf.family.astype(np.int8)),
        ("stealthy", stealthy),
        ("seq_id", buf.seq_id),
        ("timestep", np.arange(T, dtype=np.int32)),
        ("split", split),
    ):
        d.create_dataset(name, data=arr)
    b = f.create_group("benign")
    _chunked(b, "node_benign", buf.benign)
    _chunked(b, "edge_benign", buf.edge_benign)
    c = f.create_group("clean")
    _chunked(c, "node_clean", X[:T].astype(np.float32))
    _chunked(c, "edge_clean", buf.edge_clean)


def _write_episodes(f: h5py.File, buf: _TimelineBuffers) -> None:
    """episodes/ (one row per episode, buses ragged) and attack/ (magnitudes ragged per frame, masks)."""
    eg = f.create_group("episodes")
    ep = buf.episodes
    eg.create_dataset("onset", data=np.array([e["onset"] for e in ep], np.int32))
    eg.create_dataset("length", data=np.array([e["length"] for e in ep], np.int32))
    eg.create_dataset("family", data=np.array([e["family"] for e in ep], np.int8))
    ptr, idx = _ragged([np.asarray(e["buses"]) for e in ep], np.int32)
    eg.create_dataset("bus_ptr", data=ptr)
    eg.create_dataset("bus_idx", data=idx)
    ag = f.create_group("attack")
    assert buf.node_tamper is not None and buf.edge_tamper is not None, "buffers built without attack storage"
    ptr, bus = _ragged(buf.mag_bus, np.int32)
    _, mag = _ragged(buf.mag, np.float32)
    ag.create_dataset("mag_ptr", data=ptr)
    ag.create_dataset("mag_bus", data=bus)
    ag.create_dataset("mag", data=mag)
    _chunked(ag, "node_tamper", buf.node_tamper)
    _chunked(ag, "edge_tamper", buf.edge_tamper)
    ag.attrs["tamper"] = (
        "1 where the attacker wrote the meter: every metered channel for the re-solved families, the changed channels for Ad/As/Ar, the beyond-noise set for Am"
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


def write_timeline(
    out: str,
    g: FdiaGenerator,
    X: np.ndarray,
    buf: _TimelineBuffers,
    split: Sequence[float],
    seed: int,
    knobs: dict,
) -> str:
    """Serialize a walked timeline to one HDF5 file (see the module docstring for the layout)."""
    T = len(buf.family)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs.update(_timeline_attrs(g, T, seed, buf, knobs))
        _write_graph(f, g)
        _write_data(f, X, buf, _frame_split(T, buf.episodes, split))
        _write_episodes(f, buf)
    return out


def _check_knobs(attacked_frac: float, am_direction: str, lengths: dict[str, Optional[int]]) -> None:
    """Refuse the knob values that would hang or mislead the walk, before any physics is built."""
    if am_direction not in ("mask", "induce", "both"):
        raise ValueError(f"am_direction must be 'mask', 'induce' or 'both', got {am_direction!r}")
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
    am_direction     "mask" (hide a real overload), "induce" (a safe line reads overloaded) or "both"
    corrupt_len      episode length of Ad/As/Ar; 1 (default) makes every such frame an independent
                     draw as in the papers, None draws the stream's 5 to 25 frame band
    replay_tau       Ar/As replay depth in frames, None = random lag of at least 20
    redundancy       meter coverage {vbus_frac, pmu_frac, flow_frac}, default 0.6/0.2/0.9
    split            chronological train/val/test fractions by frame, episodes never cut
    """
    am_len = ramp_len if am_len is None else am_len
    _check_knobs(attacked_frac, am_direction, dict(ramp_len=ramp_len, am_len=am_len, corrupt_len=corrupt_len))
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
    buf = _TimelineBuffers(T, C, g.E, ctx.scale, g.clean_flows_from_states(X[:T]))
    t = 0
    while t < T:
        t = _advance(ctx, buf, g.rng, t, plan)
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
    return write_timeline(out, g, X, buf, split, seed, recorded)
