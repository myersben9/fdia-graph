"""The emit stage of a timeline: every frame in time order, benign or under its episode's design.

The walk takes one split at a time: the plan stage places its episodes (`generation.plan`), the
design stage designs its overload Am episodes before any frame is written (`generation.design`),
and the walk then emits the benign runs between the episodes and each episode's frames, the ramp At
designed as the walk reaches it (its stealth bound starts from the attack vector of the frame
before it, which the walk has just written). Each frame's meter jitter comes from its own stream
keyed by the timestep, so the benign frames do not depend on which attacks the timeline places.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np

from .. import schema
from ..engine import FdiaGenerator
from ..engine.records import AM_FAMILY, RAMP_FAMILY, Frame, attack_frame
from ..errors import NoRoomForEpisode
from ..models.choices import BENIGN_CODE
from ..models.data import EpisodeRow
from ..models.frames import AmOverloadDesign, AttackVector, MinimizerResult
from . import _FrameContext
from .plan import REDRAWS, Slot, _Placement, _Schedule, give_up, uniform_onset

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
# the PMU branch-current layers of a hybrid-meter timeline [D10], created only when the meter
# plan reads currents, so a v0.8.3-meter file has exactly its old datasets
_CURRENT_LAYERS = {
    schema.PMU_I: (lambda C, E: (E, 4), np.float32),
    schema.PMU_I_BENIGN: (lambda C, E: (E, 4), np.float32),
    schema.PMU_I_TAMPER: (lambda C, E: (E, 4), np.uint8),
}
CleanSlice = Callable[[int, int], tuple[np.ndarray, np.ndarray]]


def layers(currents: bool) -> dict:
    """The per-frame datasets of a timeline: the v0.8.3 set, plus the current layers when the meter
    plan reads PMU branch currents."""
    return {**_LAYERS, **(_CURRENT_LAYERS if currents else {})}


def clean_slice(g: FdiaGenerator, X: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """The noiseless truth of frames a..b-1: the states as float32 and the exact flows on metered
    branches (one batched matmul, the engine's physics primitive)."""
    return X[a:b].astype(np.float32), g.clean_flows_from_states(X[a:b])


class TimelineBuffers:
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
            name: np.zeros((n, *shape(C, E)), dtype) for name, (shape, dtype) in layers(currents).items()
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


def emit_benign(ctx: _FrameContext, t: int) -> Frame:
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
    buf: TimelineBuffers

    @property
    def rng(self) -> np.random.Generator:
        return self.ctx.g.rng

    @property
    def T(self) -> int:
        return len(self.ctx.X)


def benign_run(w: _Walk, t: int, until: int) -> int:
    """Benign frames from t up to `until` (excluded); returns `until`."""
    for t in range(t, until):
        w.buf.store(t, 0, emit_benign(w.ctx, t))
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
            w.buf.store(t, 0, emit_benign(w.ctx, t))
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


def ramp_episode(w: _Walk, t: int, ramp_len: int, ramp_rate: float) -> Optional[int]:
    """One slow-ramp episode on a fixed bus set (rise, hold, return), its design from the generator's
    ramp designer (`engine.attacks.episodes.RampDesigner`); returns the next free timestep, or None
    with no frame emitted when no admissible ramp exists at this onset (the walk then moves the
    episode, `relocate_later`)."""
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


def overload_episode(w: _Walk, t: int, length: int, design: AmOverloadDesign) -> int:
    """One `Am` episode as the overload attack of [WU26] on the design the design stage found for its
    onset (`generation.design`): a target branch's reported flow driven to its rating over the
    window, on the support that tampers the fewest devices. Returns the next free timestep."""
    ctx = w.ctx
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
    for j, line in enumerate(lines):  # one row per target line [D17]
        w.buf.am_rows.append((ep.sid, line, design.ratings[j], last[j], float(reached[j]), float(emitted[j])))
    return ep.close(w, t)


def relocate_later(rng: np.random.Generator, at: Slot, pending: Sequence[Slot], end: int) -> Optional[Slot]:
    """A new onset for a ramp whose design failed at `at`: uniform among the onsets after the failed
    one, inside the split (before `end`), where the episode fits without touching a pending episode.
    The frames before the failed onset are already written and a ramp's design reads the frame before
    its onset, so a ramp moves only later; an overload Am, whose design reads no earlier frame, is
    moved anywhere in its split before the walk (`generation.design`). None when no onset is left."""
    onset, fid, length = at
    occupied = np.zeros(end - onset, bool)
    for o, _, L in pending:
        occupied[o - onset : o - onset + L] = True
    try:
        return (onset + uniform_onset(occupied, length, rng, first=1), fid, length)
    except NoRoomForEpisode:
        return None


def walk_split(
    w: _Walk,
    plan: _Schedule,
    split: int,
    placed: tuple[list[Slot], dict[int, AmOverloadDesign]],
    span: tuple[tuple[int, int], _Placement],
) -> None:
    """Emit one split's frames in time order: benign runs between its episodes, each episode where it
    was placed. `placed` holds the split's episodes and the designs of its overload Am episodes, by
    onset (`generation.design`, which already moved or gave up the infeasible ones). A ramp whose
    design is infeasible at its onset moves to a later free onset of the split, up to REDRAWS times;
    past that it is given up (the shortfall, `rec`) and warned about."""
    (a, b), rec = span
    slots, designs = placed
    # (onset, family, length, moves so far): the move count belongs to its episode
    pending = [(*at, 0) for at in slots]
    t = a
    while pending:
        *head, moves = pending.pop(0)
        at = (head[0], head[1], head[2])
        t = benign_run(w, t, at[0])
        if at[1] == AM_FAMILY:
            t = overload_episode(w, at[0], at[2], designs[at[0]])
            rec.built[split, rec.col(AM_FAMILY)] += 1
            continue
        nxt = ramp_episode(w, at[0], at[2], plan.ramp_rate)
        if nxt is not None:
            rec.built[split, rec.col(at[1])] += 1
            t = nxt
            continue
        moved = relocate_later(w.rng, at, [p[:3] for p in pending], b) if moves < REDRAWS else None
        if moved is None:
            give_up(split, at[1], moves)
            continue
        rec.redraws[split, rec.col(at[1])] += 1
        pending = sorted([*pending, (*moved, moves + 1)])
    benign_run(w, t, b)
