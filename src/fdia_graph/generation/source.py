"""The system a timeline is generated on, and the one path that generates it.

`TimelineSource.generate` makes every timeline, on an IEEE case or on the New York grid, in the same
order: the file is started, the source prepares what it must know before the splits, the splits are cut,
then split by split the episodes are placed, designed and the frames walked in time order, and the file
is finished. The steps that depend on the grid are the methods a subclass writes, and the grids are the
subclasses: `generation.ieee.IeeeSource` (a pandapower case and its pool of operating points),
`generation.nygrid.source.NewYorkSource` (the New York grid and its year of five-minute states) and,
under it, `SingleEpisodeSource` (one episode alone, placed by hand).

    step             IEEE case                              New York grid
    prepare          nothing                                the families' onset masks; the onset survey
    cut_splits       three blocks in time order             whole weeks, stratified
    onset_masks      none: any onset of the split           the masks, or the onsets the survey found
    placement_rng    the generator's one stream             the build's "place" stream
    design           Am before the walk; no design: moved   every family; no design: left benign
    walk             a scan per frame, At designed here     the noisy scan plus the attack vector
    open, finish     the one file layout (`schema`), by the one writer (`generation.write.TimelineWriter`)

The placement itself (`place`) is the same code for both: `generation.plan.place_split`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Generic, Optional, Union

import numpy as np

from ..constants.files import PART
from ..models.plan import Design, Family, Placement, Slot, SplitSpan
from .plan import place_split, planned_until
from .write import TimelineWriter

OutPath = Union[str, "os.PathLike[str]"]  # where a timeline file is written


class TimelineSource(ABC, Generic[Family, Design]):
    """A grid a timeline is generated on: its pool of operating points and the steps of `generate`
    that depend on the grid (the table in this module's docstring). `Family` is how the grid names an
    attack family (the IEEE files' integer code, a New York family's name) and `Design` the record its
    design stage finds for one episode.

    A subclass sets `attacked_frac` (the share of each split's frames to place under an episode) and,
    in `open`, a fresh `rec` (what the placement asked for and got, per split and family) and the
    `_writer` of the file."""

    attacked_frac: float
    rec: Placement[Family]
    _writer: Optional[TimelineWriter] = None

    @property
    def writer(self) -> TimelineWriter:
        """The writer of the open file: every dataset and attribute goes through it."""
        assert self._writer is not None, "open() starts the file"
        return self._writer

    def generate(self, out: Optional[OutPath] = None) -> str:
        """Generate the timeline and write it as one HDF5 file at `out`; returns the path.

        The file is written beside `out` under a `.part` name and takes the name `out` only when it
        is whole, so a run that fails leaves no file (and a file already at `out` is left as it was)."""
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
    @abstractmethod
    def frames(self) -> int:
        """The length of the timeline: one frame per operating point of the pool."""

    @abstractmethod
    def out_path(self, out: Optional[OutPath]) -> str:
        """The path of the timeline file: `out`, or the grid's default when it has one."""

    @abstractmethod
    def open(self, path: str) -> None:
        """Start the file at `path` (its static part) and the records of this run."""

    def prepare(self) -> None:
        """What the source must work out before the splits are cut. Nothing by default; the New York
        grid finds here the onsets its families may take (the onset survey)."""

    @abstractmethod
    def cut_splits(self) -> list[SplitSpan]:
        """The train, val and test splits of the timeline, cut before any episode is placed."""

    @abstractmethod
    def lengths(self) -> dict[Family, int]:
        """The episode length in frames of every family, in the families' order."""

    @abstractmethod
    def placement_rng(self) -> np.random.Generator:
        """The random stream the onsets are drawn from."""

    @abstractmethod
    def onset_masks(self) -> Mapping[Family, np.ndarray]:
        """The onsets a family is held to, as [T] bool over the timeline's frames; a family with no
        mask may start anywhere in its split."""

    def place(self, span: SplitSpan) -> list[Slot[Family]]:
        """The episodes of the split `span` at uniform random onsets, never overlapping and never cut
        (`generation.plan.place_split`), with what was requested and dropped recorded in `rec`."""
        return place_split(
            self.placement_rng(), self.lengths(), self.attacked_frac, span, self.rec, self.onset_masks()
        )

    @abstractmethod
    def design(self, span: SplitSpan, slots: list[Slot[Family]]) -> None:
        """Design the placed episodes of `span` and keep them for the walk; decides what happens to an
        episode no design exists for."""

    @abstractmethod
    def walk(self, until: int) -> None:
        """Emit, in time order, every frame not yet written before the frame `until`, benign or under
        its episode. Every frame before `until` lies in a split whose episodes are designed by now
        (`generation.plan.planned_until`)."""

    @abstractmethod
    def finish(self) -> None:
        """After the walk: what the file carries besides its frames (the columns, the episode table
        and the attributes, `TimelineWriter.finish`; on an IEEE file also the temporal features)."""

    @abstractmethod
    def record(self) -> dict[str, object]:
        """What identifies the timeline, as nested plain values: the settings it was generated with. The
        file records them as JSON with their hash (`Attr.SETTINGS`, `Attr.SETTINGS_HASH`)."""

    def close(self, finished: bool) -> None:
        """Close the file. A subclass also stops its design processes: cleanly after a `finished`
        run, at once after a failed one."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
