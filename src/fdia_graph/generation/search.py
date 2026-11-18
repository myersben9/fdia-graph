"""A test system as the source of a timeline, the original generator: coverage meters and the
search-based designs, on any grid.

`SearchSource` is a `generation.source.TimelineSource` on a `Network` and its pool of operating
points. Placements, ramp designs and the moves of an infeasible ramp all draw from the generator's
one random stream, in the order the driver reaches them, so each split is placed only after the
split before it is walked. An overload Am is designed before its split is walked and moved when it
has no design. `generation.ieee.IeeeSource` is this source on an IEEE case by its name.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import schema
from ..constants.families import AM_CODE, FAMILIES
from ..engine.core import FdiaGenerator
from ..models.config import TimelineSettings
from ..models.frames import FrameKnobs
from ..models.inputs import AdmissibleTargets, family_codes
from ..models.plan import GeneratorSpec
from ..schema import Attr
from ..systems.network import Network
from .design import SearchDesigner
from .source import TimelineSource


class SearchSource(TimelineSource):
    """The grid `net` with its pool of AC operating points `X` [T, N, 4], the `settings` of the walk,
    the `seed` of the generator and, when the pool has it, each frame's `time` [T]. Building it
    checks the settings against the case (every family has a target, the lines have ratings) before
    any file is made.

    One source generates one timeline: the walk uses up the generator's random stream, so a second
    `generate` is refused."""

    named: str = "a SearchSource"  # the source in a message
    X: np.ndarray  # the pool in memory [T, N, 4]
    settings: TimelineSettings

    def __init__(
        self,
        net: Network,
        X: np.ndarray,
        settings: TimelineSettings,
        seed: int = 123,
        time: Optional[np.ndarray] = None,
    ) -> None:
        s = settings
        self.seed, self.settings = seed, s
        # one column per family: a family named twice adds nothing
        fams = tuple(dict.fromkeys(family_codes(s.families)))
        self.am_runs = AM_CODE in fams
        # the walk's generator on the meter plan of the settings; `_check_targets` sets the Am ratings
        self.g, self.X = FdiaGenerator(net, seed, s.meters, s.max_load_mw), X
        self._check_targets(fams)
        self.limits = self.g.operating_limits(self.X)  # every false state must satisfy [WU26 eqs. 21-23]
        self.knobs = self._knobs()
        ramp, workers = (s.ramp.length, s.ramp.rate), s.workers if self.am_runs else 1
        # what a design process rebuilds this generator from: the grid, the meters and the ratings set above
        ratings, delta = self.g._line_ratings, self.g._rating_delta
        spec = GeneratorSpec(net, seed, s.max_load_mw, s.meters, ratings, rating_delta=delta)
        designer = SearchDesigner(self.g, spec, self.knobs, self.X, ramp, workers, self.named)
        super().__init__(self.g, self.X, designer, s, [FAMILIES[f] for f in fams], seed, time)

    def _check_targets(self, fams: tuple[int, ...]) -> None:
        """Check, before any frame is walked, that every family has a target on this case and that
        the overload Am has its line ratings (NoLineRatings is raised here, not mid-walk)."""
        if round(self.settings.attacked_frac * len(self.X)) <= 0:
            return  # a timeline placing no attacked frame needs no target
        # the overload Am is not checked here: its targets are the rated, metered lines of each
        # window, decided per episode (an episode on a window with none is moved)
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

    def recorded(self) -> dict[str, object]:
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

    def header(self) -> dict[str, object]:
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

    def placement_rng(self) -> np.random.Generator:
        """The generator's one stream: the walk draws from it too, so a split is placed only after
        the split before it is walked."""
        return self.g.rng
