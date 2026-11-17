"""An IEEE case as the source of a timeline: its pool of operating points and its steps of the driver.

`IeeeSource` is a `generation.source.TimelineSource`. Its splits are three blocks in time order and
any onset of a split may hold an episode. Placements, ramp designs and the moves of an infeasible
ramp all draw from the generator's one random stream, in the order the driver reaches them, so each
split is placed only after the split before it is walked. An overload Am is designed before its
split is walked and moved when it has no design.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from .. import schema
from ..constants.families import AM_CODE
from ..engine.core import FdiaGenerator
from ..errors import ConfigError
from ..models.config import TimelineSettings
from ..models.frames import AmOverloadDesign, FrameContext, FrameKnobs
from ..models.inputs import AdmissibleTargets, GeneratedFamilies
from ..models.plan import GeneratorSpec, Placement, Schedule, Slot, SplitSpan
from ..registry import CACHE_DIR, system_id
from ..schema import Attr
from . import _load_states
from .design import AmDesigner, AmEpisode
from .emit import Walk
from .plan import split_bounds, split_column
from .source import OutPath, TimelineSource
from .write import TimelineWriter


class IeeeSource(TimelineSource[int, AmOverloadDesign]):
    """The IEEE case `system` with its pool of AC operating points (`states`), the `settings` of the
    walk and the `seed` of the generator. Building it checks the settings against the case (every
    family has a target, the lines have ratings) before any file is made.

    One source generates one timeline: the walk uses up the generator's random stream, so a second
    `generate` is refused."""

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
        self.am_runs = AM_CODE in fams
        # the walk's generator on the meter plan of the settings; `_check_targets` sets the Am ratings
        self.g = FdiaGenerator.from_spec(GeneratorSpec(system, seed, s.max_load_mw, s.meters, None))
        self.X = _load_states(system, states)
        self._check_targets(fams)
        self.limits = self.g.operating_limits(self.X)  # every false state must satisfy [WU26 eqs. 21-23]
        self.knobs = self._knobs()
        self.schedule = Schedule(list(fams), s.ramp.length, s.ramp.rate, s.am_frames, s.attacked_frac)
        self.attacked_frac = s.attacked_frac
        self.recorded = self._recorded()
        self.bounds: list[tuple[int, int]] = []
        self._designer: Optional[AmDesigner] = None
        self._frames: Optional[Walk] = None  # the walk's context and writer, from `open`
        # the split designed last and its episodes, waiting for the walk
        self._planned: Optional[tuple[SplitSpan, list[AmEpisode]]] = None

    def _check_targets(self, fams: tuple[int, ...]) -> None:
        """Check, before any frame is walked, that every family has a target on this case and that
        the overload Am has its line ratings (NoLineRatings is raised here, not mid-walk)."""
        if round(self.settings.attacked_frac * len(self.X)) <= 0:
            return  # a timeline placing no attacked frame needs no target
        # the overload Am is not checked here: its targets are the rated, metered lines of each
        # window, decided per episode (a window with none stays benign)
        AdmissibleTargets(tuple(f for f in fams if f != AM_CODE), self.g.target_counts())
        if self.am_runs:
            self.g.use_line_ratings(self.settings.overload, self.X)

    def _knobs(self) -> FrameKnobs:
        """The settings every scan shares (`FrameKnobs`): the subnetwork, the operating limits every
        false state must satisfy [WU26 eqs. 21-23], the search, and the overload attack's bounds."""
        s, o = self.settings, self.settings.overload
        knobs = FrameKnobs(
            s.search.hops,
            self.limits,
            s.search.min_tamper,
            s.search.min_budget,
            s.ramp.stealth_scale,
            area_rule=s.search.area_rule,  # how the attacker's area is chosen [WU26] Sec. III-A
        )
        if not self.am_runs:  # no overload attack: its knobs keep their defaults (no cap, one line)
            return knobs
        return knobs._replace(load_cap=o.load_cap, n_lines=o.n_lines, support_method=o.support_method)

    def _recorded(self) -> dict[str, object]:
        """The knobs a file records as attributes: the walk's and, only when they apply, those of the
        fewest-tamper search, of the Am attack and of the hybrid meters. An Am-only file records the
        stealth scale it was given, which nothing used (Am has no stealth bound)."""
        s, limits, o = self.settings, self.limits, self.settings.overload
        red = s.meters.coverage
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
            Attr.JITTER_KEYED: 1,
        }
        if s.search.min_tamper:
            out.update({Attr.MIN_TAMPER: 1, Attr.MIN_BUDGET: s.search.min_budget})
        if self.am_runs:
            out[Attr.AM_ATTACK] = s.am_attack
            out[Attr.RATING_SOURCE], out[Attr.RATING_MARGIN] = o.rating_source, o.rating_margin
            out[Attr.LOAD_CAP], out[Attr.N_LINES] = o.load_cap, o.n_lines
            out[Attr.SUPPORT_METHOD] = o.support_method
        if s.search.min_tamper or self.am_runs:
            out[Attr.STEALTH_SCALE] = s.ramp.stealth_scale
        out[Attr.METER_MODEL] = s.meters.meter_model
        out[Attr.CURRENT_FEAT] = schema.CURRENT_FEAT_LEGEND[schema.CURRENT_RECTANGULAR]
        out[Attr.CURRENT_UNITS] = schema.CURRENT_UNITS_LEGEND[schema.CURRENT_RECTANGULAR]
        return out

    def _topology(self) -> dict[str, object]:
        """The topology a file records: "base", the intact case (every branch in service), so the
        attributes of an outaged line are empty."""
        return {
            Attr.TOPOLOGY: "base",
            Attr.OUTAGE_LINE: -1,
            Attr.OUTAGE_BRANCH_POS: -1,
            Attr.OUTAGE_LINE_NAME: "",
            Attr.OUTAGE_FROM_BUS: -1,
            Attr.OUTAGE_TO_BUS: -1,
            Attr.OUTAGE_BASE_FLOW_MW: float("nan"),
        }

    def record(self) -> dict[str, object]:
        """The settings of the walk (`TimelineSettings.as_record`)."""
        return self.settings.as_record()

    def _spec(self) -> GeneratorSpec:
        """What a design process rebuilds this generator from."""
        g, s = self.g, self.settings
        ratings, delta = g._line_ratings, g._rating_delta
        return GeneratorSpec(self.system, self.seed, s.max_load_mw, s.meters, ratings, rating_delta=delta)

    def _clean(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """The noiseless truth of frames a..b-1: the states as float32 and the exact flows on metered
        branches."""
        return self.X[a:b].astype(np.float32), self.g.clean_flows_from_states(self.X[a:b])

    @property
    def frames(self) -> int:
        """The length of the pool: one frame of the timeline per operating point."""
        return len(self.X)

    def out_path(self, out: Optional[OutPath]) -> str:
        """`out`, or `timeline_ieee{N}.h5` under the cache directory."""
        return os.fspath(out) if out else os.path.join(CACHE_DIR, f"timeline_ieee{system_id(self.system)}.h5")

    def open(self, path: str) -> None:
        """Start the file (the graph and the empty per-frame layers) and the design processes. A
        second call is refused: the walk uses up the generator's random stream."""
        if self._designer is not None:
            raise ConfigError("an IeeeSource generates one timeline; build a new source for another")
        g, T = self.g, self.frames
        self._designer = AmDesigner(g, self._spec(), self.settings.workers if self.am_runs else 1)
        self._writer = TimelineWriter(path, (T, g.C, g.E), self._clean)
        self._writer.write_graph(g.ei, g.case.static_graph())
        self._frames = Walk(FrameContext(g, self.X, self.knobs), self._writer)
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
        """Design every overload Am of the split before it is walked (`generation.design`); one with
        no design moves to another free onset or is given up. A ramp At is designed by the walk."""
        assert self._designer is not None, "open() starts the design processes"
        episodes = self._designer.design_split(self.X, self.knobs, span, slots, self.rec)
        self._planned = (span, episodes)

    def walk(self, until: int) -> None:
        """Walk the frames of the split just designed (`generation.emit.Walk.walk_split`). The splits
        are blocks in time order, so the frames not yet written before `until` are exactly that split's."""
        assert self._frames is not None and self._planned is not None, "design() comes before walk()"
        span, episodes = self._planned
        assert until == span.stop, "a block split ends where the next split to plan begins"
        self._frames.walk_split(self.schedule, span, episodes, self.rec)

    def finish(self) -> None:
        """After the walk, write the columns, the episodes, the attributes and the temporal features."""
        w, frac = self.writer, self.settings.split.fractions
        self.g.scan_key = None
        columns = {schema.SPLIT: split_column(self.bounds, self.frames)}
        header = {**w.header(self.g.case.base_mva, self.seed), **self._topology()}
        w.finish(columns, header, self.recorded, self.record(), self.rec, frac)

    def close(self, finished: bool) -> None:
        """Close the file and stop the design processes."""
        super().close(finished)
        if self._designer is not None:
            self._designer.close(finished)
