"""An IEEE case as the source of a timeline: its pool of operating points and its steps of the driver.

`IeeeSource` is one of the two `generation.source.TimelineSource` subclasses. Its splits are three
blocks in time order; any onset of a split may hold an episode; placements, ramp designs and the
moves of an infeasible ramp all draw from the generator's one random stream, in the order the driver
reaches them (so each split is placed after the split before it is walked); an overload Am is designed
before its split is walked and moved when it has no design; and the file is the IEEE layout
(`generation.write`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import h5py
import numpy as np

from ..engine import FdiaGenerator
from ..engine.attacks.stealthy import AM_FAMILY
from ..errors import ConfigError
from ..models.config import TimelineSettings
from ..models.frames import AmOverloadDesign, FrameContext, FrameKnobs
from ..models.inputs import AdmissibleTargets, GeneratedFamilies
from ..models.plan import GeneratorSpec, Placement, Schedule, Slot, SplitSpan
from ..registry import CACHE_DIR, system_id
from ..results.run import config_hash
from ..schema import Attr
from . import _load_states, _write_graph
from .design import AmDesigner, AmEpisode
from .emit import TimelineBuffers, Walk, clean_slice, walk_split
from .plan import split_bounds
from .source import OutPath, TimelineSource
from .write import (
    create_layers,
    finish_timeline,
    placement_attrs,
    search_attrs,
    timeline_attrs,
    write_temporal_layers,
)


@dataclass(frozen=True)
class _Planned:
    """A split after its design stage: the split and its episodes, waiting to be walked."""

    span: SplitSpan
    episodes: list[AmEpisode]


class IeeeSource(TimelineSource[int, AmOverloadDesign]):
    """The IEEE case `system` with its pool of AC operating points (`states`, as `fg.generate` takes
    them), the `settings` of the walk and the `seed` of the generator. Building it checks the settings
    against the case (the families have a target, the lines have ratings) before any file is made.

    One source generates one timeline: the generator's random stream is used up by the walk, and a
    second `generate` is refused."""

    def __init__(
        self,
        system: Union[int, str],
        states: Optional[Union[str, np.ndarray]],
        settings: TimelineSettings,
        seed: int = 123,
    ) -> None:
        s = settings
        self.system, self.seed, self.settings = system, seed, s
        # one column per family: a family named twice adds nothing
        fams = tuple(dict.fromkeys(GeneratedFamilies(s.families).codes))
        self.am_runs = AM_FAMILY in fams
        self.g = self._generator()
        self.X = _load_states(system, states)
        self._check_targets(fams)
        self.limits = self.g.operating_limits(self.X)  # what every false state must satisfy [WU26 eqs. 21-23]
        self.knobs = self._knobs()
        self.schedule = Schedule.build(list(fams), s.ramp.length, s.ramp.rate, s.am_frames, s.attacked_frac)
        self.attacked_frac = s.attacked_frac
        self.recorded = self._recorded()
        self.bounds: list[tuple[int, int]] = []
        self._file: Optional[h5py.File] = None
        self._designer: Optional[AmDesigner] = None
        self._frames: Optional[Walk] = None  # the walk's context and buffers, from `open`
        self._planned: Optional[_Planned] = None  # the split designed last, waiting for its walk

    def _generator(self) -> FdiaGenerator:
        """The walk's generator on the meter plan of the settings."""
        s, m = self.settings, self.settings.meters
        return FdiaGenerator(
            self.system,
            seed=self.seed,
            max_load_mw=s.max_load_mw,
            meter_model=m.meter_model,
            vbus_frac=m.vbus_frac,
            pmu_frac=m.pmu_frac,
            flow_frac=m.flow_frac,
        )

    def _check_targets(self, fams: tuple[int, ...]) -> None:
        """Before any frame is walked: every family has a target on this case, and the overload Am
        its line ratings (pglib: NoLineRatings here, not mid-walk)."""
        if round(self.settings.attacked_frac * len(self.X)) <= 0:
            return  # a timeline placing no attacked frame needs no target
        # the overload Am is not checked here: its targets are the rated, metered lines of each
        # window, decided per episode (a window with none stays benign)
        AdmissibleTargets(tuple(f for f in fams if f != AM_FAMILY), self.g.target_counts())
        if self.am_runs:
            self.g.use_line_ratings(self.settings.overload, self.X)

    def _knobs(self) -> FrameKnobs:
        """The settings every scan shares (`FrameKnobs`): the subnetwork, the operating limits every
        false state must satisfy [WU26 eqs. 21-23], the search, and the overload attack's bounds."""
        s = self.settings
        o = s.overload if self.am_runs else None
        return FrameKnobs(
            s.search.hops,
            self.limits,
            s.search.min_tamper,
            s.search.min_budget,
            s.ramp.stealth_scale,
            getattr(o, "load_cap", None),  # the overload attack's cap; none without it
            getattr(o, "n_lines", 1),  # the lines an overload episode drives
            getattr(o, "support_method", "search"),  # how its support is chosen
            area_rule=s.search.area_rule,  # how the attacker's area is chosen [WU26] Sec. III-A
        )

    def _recorded(self) -> dict[str, object]:
        """The knobs a file records as attributes, and the settings themselves as JSON with their hash."""
        s, limits = self.settings, self.limits
        red = s.meters.coverage
        record = s.as_record()
        out: dict[str, object] = {
            Attr.TARGET_ATTACKED_FRAC: s.attacked_frac,
            Attr.RAMP_RATE: s.ramp.rate,
            Attr.RAMP_LEN: s.ramp.length,
            Attr.AM_LEN: s.am_frames,
            Attr.HOPS: s.search.hops,
            Attr.MAX_LOAD_MW: s.max_load_mw,
            Attr.V_LO: float(limits.v_lo.min()),
            Attr.V_HI: float(limits.v_hi.max()),
            Attr.VBUS_FRAC: red["vbus_frac"],
            Attr.PMU_FRAC: red["pmu_frac"],
            Attr.FLOW_FRAC: red["flow_frac"],
        }
        overload = s.overload if self.am_runs else None
        out.update(search_attrs(s, overload, self.g.current_mask() is not None))
        out.update(
            {Attr.SETTINGS: json.dumps(record, sort_keys=True), Attr.SETTINGS_HASH: config_hash(record)}
        )
        return out

    def _spec(self) -> GeneratorSpec:
        """What a design process rebuilds this generator from."""
        g, s = self.g, self.settings
        ratings = getattr(g, "_line_ratings", None)
        return GeneratorSpec(
            self.system, self.seed, s.max_load_mw, s.meters, ratings, getattr(g, "_rating_delta", None)
        )

    @property
    def frames(self) -> int:
        """The length of the pool: one frame of the timeline per operating point."""
        return len(self.X)

    def out_path(self, out: Optional[OutPath]) -> str:
        """`out`, or `timeline_ieee{N}.h5` under the cache directory."""
        return os.fspath(out) if out else os.path.join(CACHE_DIR, f"timeline_ieee{system_id(self.system)}.h5")

    def open(self, path: str) -> None:
        """Start the file: the graph and the per-frame layers at full length, empty until the walk
        fills them; and the design processes. A source generates once: the generator's random stream
        is used up by the walk, so a second timeline would not be the first."""
        if self._designer is not None:
            raise ConfigError("an IeeeSource generates one timeline; build a new source for another")
        g, T = self.g, self.frames
        self._designer = AmDesigner(g, self._spec(), self.settings.workers if self.am_runs else 1)
        self._file = h5py.File(path, "w")  # open for the whole walk: frames flush in batches
        _write_graph(self._file, g)
        currents = g.current_mask() is not None
        sink = create_layers(self._file, T, g.C, g.E, currents)
        buf = TimelineBuffers((T, g.C, g.E), partial(clean_slice, g, self.X), sink=sink, currents=currents)
        self._frames = Walk(FrameContext(g, self.X, self.knobs), buf)
        self.rec = Placement.empty(list(self.schedule.families))

    def cut_splits(self) -> list[SplitSpan]:
        """Three blocks in time order, by the settings' fractions (`generation.plan.split_bounds`)."""
        self.bounds = split_bounds(self.frames, self.settings.split.fractions)
        return [SplitSpan(code, start, stop) for code, (start, stop) in enumerate(self.bounds)]

    def lengths(self) -> dict[int, int]:
        """The episode length of every family: the ramp's for At, `am_len` for Am."""
        return self.schedule.lengths

    def placement_rng(self) -> np.random.Generator:
        """The generator's one stream: the walk draws from it too, so a split is placed only after
        the split before it is walked."""
        return self.g.rng

    def onset_masks(self) -> dict[int, np.ndarray]:
        """None: an episode may start at any onset of its split."""
        return {}

    def design(self, span: SplitSpan, slots: list[Slot[int]]) -> None:
        """Every overload Am of the split designed before it is walked (`generation.design`), one
        with no design at its onset moved to another free onset of the split until it has one, or
        given up. A ramp At is kept without a design: the walk designs it as it reaches it."""
        assert self._designer is not None, "open() starts the design processes"
        episodes = self._designer.design_split(self.X, self.knobs, span, slots, self.rec)
        self._planned = _Planned(span, episodes)

    def walk(self, until: int) -> None:
        """The frames of the split just designed, each a scan of its operating point, benign or under
        its episode (`generation.emit.walk_split`). The splits are blocks in time order, so the frames
        not yet written before `until` are exactly that split's."""
        assert self._frames is not None and self._planned is not None, "design() comes before walk()"
        span, episodes = self._planned.span, self._planned.episodes
        assert until == span.stop, "a block split ends where the next split to plan begins"
        walk_split(self._frames, self.schedule, span, episodes, self.rec)

    def finish(self) -> None:
        """After the walk: the columns, the episodes, the attributes (the recorded knobs and what the
        placement asked for and got) and the temporal features."""
        f, walk = self._file, self._frames
        assert f is not None and walk is not None, "open() starts the file"
        self.g.scan_key = None
        finish_timeline(f, walk.buf, self.bounds)
        f.attrs.update(timeline_attrs(self.g, self.seed, walk.buf, self.recorded))
        f.attrs.update(placement_attrs(self.rec, self.bounds, self.settings.split.fractions, walk.buf.seq_id))
        write_temporal_layers(f)

    def close(self, finished: bool) -> None:
        """Close the file and stop the design processes."""
        if self._file is not None:
            self._file.close()
        if self._designer is not None:
            self._designer.close(finished)
