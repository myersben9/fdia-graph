"""The emit stage of a timeline: every frame in time order, benign or under its episode's design.

The walk runs as far as the splits are planned: after the plan stage places a split's episodes
(`generation.plan`) and the design stage designs them (`generation.design`). It emits the benign
runs between the episodes and each episode's frames, the same way whichever family designed the
episode (`engine.attacks.family.DesignedEpisode`). An episode that cannot begin at its onset (a
search ramp is designed there, from the attack vector of the frame just written) moves later. Each
frame's meter jitter comes from its own stream keyed by its pool frame, so the benign frames do not
depend on which attacks the timeline places.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from ..engine.attacks.family import DesignedEpisode
from ..engine.core import FdiaGenerator
from ..errors import NoRoomForEpisode
from ..models.frames import Frame
from ..models.plan import Placement, Planned, Slot, SplitSpan
from ..systems.states import States
from .plan import REDRAWS, give_up, uniform_onset
from .write import TimelineWriter


class Walk:
    """One timeline walk: the generator `g`, the pool `X` and the `writer` the frames are stored
    through. The random stream is the generator's own (`g.rng`), so the order of draws is the order
    the walk makes them. `first` is the pool frame the file starts at: a frame's jitter is keyed by
    its pool frame, so one operating point reads the same in every file that holds it."""

    def __init__(self, g: FdiaGenerator, X: States, writer: TimelineWriter, first: int = 0) -> None:
        self.g, self.X, self.writer, self.first = g, X, writer, first

    def emit_benign(self, t: int) -> Frame:
        """The benign scan of timestep t, its jitter from the stream of its pool frame."""
        self.g.scan_key = self.first + t
        return self.g.emitter.benign_frame(self.X[t])

    def store(self, t: int, fid: int, frame: Optional[Frame], sid: int, buses: np.ndarray) -> None:
        """Store frame t of family `fid` in the episode `sid` and add its labels to `buses`. A frame
        whose attack could not be built (None) is stored benign."""
        if frame is None:
            self.writer.store(t, 0, self.emit_benign(t))
        else:
            self.writer.store(t, fid, frame, sid)
            buses |= frame.y

    def benign_run(self, t: int, until: int) -> int:
        """Benign frames from t up to `until` (excluded); returns `until`."""
        for t in range(t, until):
            self.writer.store(t, 0, self.emit_benign(t))
        return until

    def episode(self, t: int, length: int, designed: DesignedEpisode) -> int:
        """Emit one episode over `length` frames from t, each scan by its design
        (`DesignedEpisode.frame`; a scan it cannot build is stored benign), then record it. Returns
        the next free timestep."""
        g, onset, sid = self.g, t, len(self.writer.episodes)
        buses = np.zeros(g.C, np.uint8)
        for i in range(length):
            g.scan_key = self.first + t
            self.store(t, designed.code, designed.frame(g, self.X[t], i), sid, buses)
            t += 1
        self.writer.add_episode(designed.record(onset, length, buses))
        return t

    @staticmethod
    def relocate_later(
        rng: np.random.Generator, failed: Slot, pending: Sequence[Slot], span: SplitSpan
    ) -> Optional[Slot]:
        """A new slot for an episode that could not begin at `failed`: an onset uniform among the later
        ones of its split `span` where it fits, its frames consecutive, clear of the `pending`
        episodes; None when none is left. It moves only later, because the earlier frames are written
        and its design reads the frame before it."""
        first = int(np.searchsorted(span.index(), failed.onset))
        index, allowed = span.index()[first:], span.onsets_fit(failed.length)[first:]
        occupied = np.zeros(len(index), bool)
        for other in pending:
            lo, hi = np.searchsorted(index, (other.onset, other.onset + other.length))
            occupied[lo:hi] = True
        try:
            return failed.at(int(index[uniform_onset(occupied, failed.length, rng, 1, allowed)]))
        except NoRoomForEpisode:
            return None

    def run(
        self,
        t: int,
        until: int,
        pending: list[Planned[DesignedEpisode]],
        spans: Sequence[SplitSpan],
        rec: Placement,
    ) -> int:
        """Emit the frames from t up to `until` in time order: benign runs between the `pending`
        episodes that start before `until`, each episode where it was placed (it then leaves
        `pending`). One that cannot begin at its onset (`DesignedEpisode.begin`) moves to a later
        free onset of its split (`spans`, by split code), up to REDRAWS times. Returns `until`."""
        while pending and pending[0].slot.onset < until:
            episode = pending.pop(0)
            slot, span, designed = episode.slot, spans[episode.split], episode.design
            t = self.benign_run(t, slot.onset)
            if designed is not None and designed.begin(self.g, self.X, slot.onset, self.writer.last_attack):
                t = self.episode(slot.onset, slot.length, designed)
                rec.built[span.code, rec.col(slot.family)] += 1
                continue
            moved = None
            if episode.moves < REDRAWS:
                moved = self.relocate_later(self.g.rng, slot, [p.slot for p in pending], span)
            if moved is None:
                give_up(span.code, slot.family, episode.moves)
                continue
            rec.redraws[span.code, rec.col(slot.family)] += 1
            pending.append(episode.moved_to(moved))
            pending.sort(key=lambda p: (p.slot, p.moves))
        return self.benign_run(t, until)
