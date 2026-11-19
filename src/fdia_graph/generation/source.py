"""The one source of a timeline: a grid, its pool of operating points, and the path that generates it.

`TimelineSource.generate` makes every timeline in the same order: open the file, prepare, cut the
splits, then split by split place the episodes, design them and walk the frames in time order, and
finish the file. Those steps are written here once, on a generator over the grid, the pool and the
designer of the attack families. A subclass builds them and says what its files record. The
package's one subclass is `generation.search.SearchSource` (any grid, coverage meters, the
search-based families or families of your own; an IEEE case by name is `generation.ieee.IeeeSource`).
docs/guides/generation.md has the steps one by one.
"""

from __future__ import annotations

import os
import time
import zlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Optional, Union

import numpy as np

from .. import schema
from ..constants.files import PART
from ..engine.attacks.family import DesignedEpisode
from ..engine.core import FdiaGenerator
from ..errors import ConfigError
from ..models.choices import Split, SplitBy
from ..models.config import TimelineSettings
from ..models.inputs import SplitRule
from ..models.plan import Episode, Placement, Planned, Slot, SplitSpan
from ..systems.states import States
from .design import Designer, SplitDesign
from .emit import Walk
from .plan import (
    STRATUM,
    place_split,
    planned_until,
    split_bounds,
    split_column,
    split_weeks,
    time_runs,
)
from .write import TimelineWriter

OutPath = Union[str, "os.PathLike[str]"]  # where a timeline file is written


class TimelineSource(ABC):
    """A timeline to generate: the generator `g` over the grid and its meter plan, the pool `X` (one
    frame per operating point), the `designer` of the attack families, the `settings`, the names of
    the families `placed`, the `seed` and, when the pool has it, each frame's `time` [T].

    The splits are cut as the settings say (`split_by`); whole weeks need the frames' time and are
    refused without it (`SplitRule`). The other defaults are the original's: any onset of a split, a
    file that records the case's static graph."""

    legend: Mapping[str, str] = schema.GRAPH_LEGEND
    tamper_rule: str = schema.IEEE_TAMPER_RULE
    first: int = 0  # the pool frame the file starts at (a file of a stretch of its pool starts later)
    stratum: int = STRATUM  # weeks per stratum of a split by weeks
    seconds: float = 0.0  # how long the last run took
    _writer: Optional[TimelineWriter] = None

    def __init__(
        self,
        g: FdiaGenerator,
        X: States,
        designer: Designer,
        settings: TimelineSettings,
        placed: Sequence[str],
        seed: int,
        time: Optional[np.ndarray] = None,
    ) -> None:
        # the frames' times are checked here, and weeks without them refused, before any file is made
        self.split_rule = SplitRule(settings.split_by, len(X), time)
        self.g, self.X, self.designer, self.settings, self.seed = g, X, designer, settings, seed
        self.time = self.split_rule.time
        self.placed, self.fractions = list(placed), settings.split.fractions
        self.episodes: list[Episode] = []  # the episode rows of the last file written

    @property
    def writer(self) -> TimelineWriter:
        """The writer of the open file: every dataset and attribute goes through it."""
        assert self._writer is not None, "open() starts the file"
        return self._writer

    def generate(self, out: Optional[OutPath] = None) -> str:
        """Generate the timeline, write it as one HDF5 file at `out` and return the path. The file is
        written under a `.part` name and renamed only when whole, so a failed run leaves no file and
        a file already at `out` is left as it was."""
        path = self.out_path(out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        finished = False
        try:
            self.open(path + PART)
            self.prepare()
            spans = self.cut_splits()
            for k, span in enumerate(spans):
                slots = self.place(span)
                self.design(span, slots)
                self.walk(planned_until(spans, k))
            self.finish()
            finished = True
        finally:
            self.close(finished)
            self._settle(path, finished)
        return path

    def _settle(self, path: str, finished: bool) -> None:
        """Give the finished file its name, or remove what a failed run wrote."""
        if finished:
            os.replace(path + PART, path)
        elif os.path.exists(path + PART):
            os.remove(path + PART)

    @property
    def frames(self) -> int:
        """The length of the timeline: one frame per operating point of the pool."""
        return len(self.X)

    def out_path(self, out: Optional[OutPath]) -> str:
        """The path of the timeline file: `out`. A source with a default path gives it for None."""
        if out is None:
            raise ConfigError("a timeline needs the path of its file: source.generate(path)")
        return os.fspath(out)

    # ---- open --------------------------------------------------------------------------------------

    def open(self, path: str) -> None:
        """Start a run: the clock, the design processes, the placement stream and record, the file at
        `path` (the graph and the empty per-frame layers) and the walk."""
        self._started = time.perf_counter()
        self.designer.open()
        g = self.g
        self._rng, self.rec = self.placement_rng(), Placement.empty(list(self.lengths()))
        self._pending: list[Planned[DesignedEpisode]] = []  # designed, not yet written
        self._cursor = 0  # the next frame to write
        self._writer = TimelineWriter(path, (self.frames, g.n_bus, g.n_branch), self._clean)
        self._writer.write_graph(g.edge_index, self.graph(), self.legend)
        self._walk = Walk(g, self.X, self._writer, self.first)

    def _clean(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """The noiseless truth of frames a..b-1: the states as float32 and the exact flows at the
        metered branch ends (an unsolved frame keeps its place with NaN)."""
        with np.errstate(invalid="ignore"):
            return self.X[a:b].astype(np.float32), self.g.emitter.clean_flows_from_states(self.X[a:b])

    def graph(self) -> dict[str, np.ndarray]:
        """The static graph the file carries, by `Static` name: the case's (`Case.static_graph`)."""
        return self.g.case.static_graph()

    # ---- prepare, cut, place -----------------------------------------------------------------------

    def prepare(self) -> None:
        """What the source must work out before the splits are cut. Nothing by default; a subclass
        whose families may start only at some onsets finds them here (`onset_masks`)."""

    def cut_splits(self) -> list[SplitSpan]:
        """The train, val and test splits of the timeline, cut before any episode is placed: each
        with the frames `split_codes` gives it."""
        self.split = self.split_codes()
        runs = None if self.time is None else time_runs(self.time)  # no episode crosses a gap in time
        self._spans = [self._span(code, runs) for code in range(len(Split))]
        return self._spans

    def stream(self, name: str) -> np.random.Generator:
        """The random stream of this timeline keyed by its seed and the stream's `name`."""
        return np.random.default_rng((self.seed, zlib.crc32(name.encode())))

    def split_codes(self) -> np.ndarray:
        """[T] the split of every frame (0 train, 1 val, 2 test), cut as the settings' `split_by`
        says: three blocks in time order (`generation.plan.split_bounds`), or whole weeks stratified
        on the frames some family is held to (`onset_masks`), drawn from the seed's "split" stream
        (`generation.plan.split_weeks`)."""
        if self.settings.split_by == SplitBy.TIME:
            return split_column(split_bounds(self.frames, self.fractions), self.frames)
        held = np.zeros(self.frames, bool)
        for mask in self.onset_masks().values():
            held |= mask
        time = self.split_rule.times
        return split_weeks(time, self.fractions, held, self.stream("split"), self.stratum)

    def _span(self, code: int, runs: Optional[np.ndarray]) -> SplitSpan:
        """The split `code` with the frames it holds, in time order, and the timeline's `runs` of
        regularly spaced frames."""
        where = np.flatnonzero(self.split == code)
        start, stop = (int(where[0]), int(where[-1]) + 1) if len(where) else (0, 0)
        return SplitSpan(code, start, stop, where, runs)

    def lengths(self) -> dict[str, int]:
        """The episode length in frames of every family the timeline places, in the order of the
        placement record's columns."""
        return self.settings.lengths_for(self.placed)

    @abstractmethod
    def placement_rng(self) -> np.random.Generator:
        """The random stream the onsets of a run are drawn from."""

    def onset_masks(self) -> Mapping[str, np.ndarray]:
        """The onsets a family is held to, as [T] bool over the timeline's frames; a family with no
        mask may start anywhere in its split (the default: none has one)."""
        return {}

    def place(self, span: SplitSpan) -> list[Slot]:
        """The episodes of the split `span` at uniform random onsets, never overlapping and never cut.
        What was requested and dropped is recorded in `rec`."""
        frac, masks = self.settings.attacked_frac, self.onset_masks()
        return place_split(self._rng, self.lengths(), frac, span, self.rec, masks)

    # ---- design, walk ------------------------------------------------------------------------------

    def design(self, span: SplitSpan, slots: list[Slot]) -> None:
        """Design the placed episodes of `span` and keep them for the walk. One with no design moves
        to another free onset of its split or is given up (`generation.design.SplitDesign`)."""
        stage = SplitDesign[DesignedEpisode](self.seed, span, self.rec, self.onset_masks())
        planned = stage.run(slots, self.designer.designs)
        self._pending = sorted(self._pending + planned, key=lambda p: (p.slot, p.moves))

    def walk(self, until: int) -> None:
        """Emit, in time order, every frame not yet written before the frame `until`, benign or under
        its episode (`generation.emit.Walk.run`). Every frame before `until` lies in a split whose
        episodes are designed by now (`generation.plan.planned_until`)."""
        self._cursor = self._walk.run(self._cursor, until, self._pending, self._spans, self.rec)

    # ---- finish, close -----------------------------------------------------------------------------

    def finish(self) -> None:
        """After the walk, write what the file carries besides its frames: the columns, the episode
        table, the attributes and the temporal features (`TimelineWriter.finish`)."""
        w = self.writer
        self.g.scan_key, self.episodes = None, w.episodes
        header = {**w.header(self.g.case.base_mva, self.seed), **self.header()}
        frac = self.fractions
        w.finish(self.columns(), header, self.recorded(), self.record(), self.rec, frac, self.tamper_rule)

    def columns(self) -> dict[str, np.ndarray]:
        """The per-frame columns the source adds to a file, by dataset path: each frame's split and,
        when the pool has it, its time (int64 seconds)."""
        columns = {schema.SPLIT: self.split}
        if self.time is not None:
            columns[schema.TIME] = self.time.astype("datetime64[s]").astype(np.int64)
        return columns

    def header(self) -> dict[str, object]:
        """What the source adds to a file's header attributes. Nothing by default."""
        return {}

    @abstractmethod
    def recorded(self) -> dict[str, object]:
        """The knobs a file records as attributes, after its header."""

    def record(self) -> dict[str, object]:
        """The settings the timeline is generated with, as nested plain values. The file records them
        as JSON with their hash (`Attr.SETTINGS`, `Attr.SETTINGS_HASH`)."""
        return self.settings.as_record()

    def close(self, finished: bool) -> None:
        """Close the file, stop the design processes (cleanly after a `finished` run, at once after a
        failed one) and the clock."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.designer.close(finished)
        self.seconds = time.perf_counter() - self._started
