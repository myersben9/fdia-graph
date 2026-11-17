"""The emit stage of a timeline: every frame in time order, benign or under its episode's design.

The walk takes one split at a time, after the plan stage places its episodes (`generation.plan`)
and the design stage designs its overload Am episodes (`generation.design`). It emits the benign
runs between the episodes and each episode's frames. A ramp At is designed as the walk reaches it,
because its stealth bound starts from the attack vector of the frame just written. Each frame's
meter jitter comes from its own stream keyed by the timestep, so the benign frames do not depend on
which attacks the timeline places.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from ..constants.families import AM_CODE, AT_CODE, FAMILIES
from ..engine.records import attack_frame
from ..errors import NoRoomForEpisode
from ..models.frames import AmOverloadDesign, Frame, FrameContext
from ..models.plan import Episode, Placement, Planned, Schedule, Slot, SplitSpan
from .plan import REDRAWS, give_up, uniform_onset
from .write import AmRows, Tamper, TimelineWriter


class Walk:
    """One timeline walk: the run context `ctx` (generator, pool, knobs) and the `writer` the frames
    are stored through. The random stream is the generator's own (`ctx.g.rng`), so the order of
    draws is the order the walk makes them."""

    def __init__(self, ctx: FrameContext, writer: TimelineWriter) -> None:
        self.ctx, self.writer = ctx, writer

    def emit_benign(self, t: int) -> Frame:
        """The benign scan of timestep t, its jitter from frame t's own stream."""
        self.ctx.g.scan_key = t
        frame = attack_frame(self.ctx.g, self.ctx.X[t], 0, None, self.ctx.knobs)
        assert frame is not None  # a benign emission cannot fail
        return frame

    def store(self, t: int, fid: int, frame: Optional[Frame], sid: int, buses: np.ndarray) -> None:
        """Store frame t of family `fid` in the episode `sid` and add its labels to `buses`. A frame
        whose attack could not be built (None) is stored benign."""
        if frame is None:
            self.writer.store(t, 0, self.emit_benign(t))
        else:
            self.writer.store(t, fid, frame, sid)
            buses |= frame.y

    def record_episode(
        self, fid: int, onset: int, t: int, buses: np.ndarray, tamper: Tamper = None, lines: AmRows = ()
    ) -> int:
        """Record the episode of family `fid` written over the frames onset..t-1 and return t."""
        none = np.zeros(0)  # an IEEE episode has no clients, and its frames carry the attack
        row = Episode(FAMILIES[fid], none, onset, t - onset, np.flatnonzero(buses), -1, none)
        self.writer.add_episode(row, tamper, lines)
        return t

    def benign_run(self, t: int, until: int) -> int:
        """Benign frames from t up to `until` (excluded); returns `until`."""
        for t in range(t, until):
            self.writer.store(t, 0, self.emit_benign(t))
        return until

    def ramp_episode(self, t: int, ramp_len: int, ramp_rate: float) -> Optional[int]:
        """Emit one slow-ramp episode on a fixed bus set (rise, hold, return) and return the next free
        timestep. Returns None, with no frame emitted, when no admissible ramp exists at this onset."""
        ctx = self.ctx
        design = ctx.g.ramp_design(ctx.X, t, (ramp_len, ramp_rate), ctx.knobs, self.writer.last_attack)
        if design is None:
            return None
        onset, sid, buses = t, len(self.writer.episodes), np.zeros(ctx.g.C, np.uint8)
        for i in range(ramp_len):
            step = ctx.g.ramp_step(design, i, ramp_rate)
            ctx.g.scan_key = t
            frame = attack_frame(ctx.g, ctx.X[t], AT_CODE, step, ctx.knobs)
            self.store(t, AT_CODE, frame, sid, buses)
            t += 1
        return self.record_episode(AT_CODE, onset, t, buses, design.tamper)

    def overload_episode(self, t: int, length: int, design: AmOverloadDesign) -> int:
        """Emit one `Am` episode, the overload attack of [WU26], on the design found for its onset: the
        target branches' reported flow driven to their ratings over the window. Returns the next free
        timestep."""
        ctx = self.ctx
        onset, sid, buses = t, len(self.writer.episodes), np.zeros(ctx.g.C, np.uint8)
        lines = list(design.goal.lines)
        reached = emitted = np.full(len(lines), np.nan)
        for i in range(length):
            ctx.g.scan_key = t
            frame, reached = ctx.g.overload_step(design, ctx.X[t], i, ctx.knobs)
            self.store(t, AM_CODE, frame, sid, buses)
            if frame is not None:
                emitted = np.hypot(frame.edge_x[lines, 0], frame.edge_x[lines, 1])
            t += 1
        last = design.goal.targets_at(len(design.goal.targets) - 1)  # the goal at the window's end
        ratings = design.ratings
        rows = [(b, ratings[j], last[j], float(reached[j]), float(emitted[j])) for j, b in enumerate(lines)]
        return self.record_episode(AM_CODE, onset, t, buses, design.tamper, rows)

    @staticmethod
    def relocate_later(
        rng: np.random.Generator, failed: Slot[int], pending: Sequence[Slot[int]], end: int
    ) -> Optional[Slot[int]]:
        """A new slot for a ramp whose design failed at `failed`: an onset uniform among the later ones,
        before `end`, where it fits clear of the `pending` episodes; None when none is left. A ramp moves
        only later, because the earlier frames are written and its design reads the frame before it."""
        occupied = np.zeros(end - failed.onset, bool)
        for other in pending:
            occupied[other.onset - failed.onset : other.onset - failed.onset + other.length] = True
        try:
            return failed.at(failed.onset + uniform_onset(occupied, failed.length, rng, first=1))
        except NoRoomForEpisode:
            return None

    def walk_split(
        self,
        plan: Schedule,
        span: SplitSpan,
        episodes: Sequence[Planned[int, AmOverloadDesign]],
        rec: Placement[int],
    ) -> None:
        """Emit the frames of the split `span` in time order: benign runs between its episodes, each
        episode where it was placed. An overload Am arrives with its design; a ramp At is designed here.
        A ramp with no design at its onset moves to a later free onset, up to REDRAWS times."""
        pending = list(episodes)
        t = span.start
        while pending:
            episode = pending.pop(0)
            slot = episode.slot
            t = self.benign_run(t, slot.onset)
            if episode.design is not None:  # an overload Am, designed before the walk
                t = self.overload_episode(slot.onset, slot.length, episode.design)
                rec.built[span.code, rec.col(AM_CODE)] += 1
                continue
            nxt = self.ramp_episode(slot.onset, slot.length, plan.ramp_rate)
            if nxt is not None:
                rec.built[span.code, rec.col(slot.family)] += 1
                t = nxt
                continue
            moved = None
            if episode.moves < REDRAWS:
                moved = self.relocate_later(self.ctx.g.rng, slot, [p.slot for p in pending], span.stop)
            if moved is None:
                give_up(span.code, slot.family, episode.moves)
                continue
            rec.redraws[span.code, rec.col(slot.family)] += 1
            pending = sorted([*pending, episode.moved_to(moved)], key=lambda p: (p.slot, p.moves))
        self.benign_run(t, span.stop)
