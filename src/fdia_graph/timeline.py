"""One continuous attacked timeline per system, written as one HDF5 file.

The timeline is the dataset. It is first cut chronologically into train, val and test (60/20/20
by default); then each split gets its own attack episodes, as many whole episodes as come closest
to the attacked fraction of its frames, shared between the families by largest remainder and placed
at uniform random onsets inside the split, never overlapping and never cut. Whether two episodes
touch or a long quiet stretch separates them is a property of that draw, not of any rule. Every
frame's meter jitter is drawn from its own stream keyed by (seed, timestep), so two timelines on one
seed and pool that place different attacks carry the same benign frames.
The walk then emits every frame in time order, a scan of the grid at one pool timestep with the
attack of its episode applied (or none), and the file carries everything a user reads afterwards:

    attrs         system, N, E, baseMVA, seed, T, families, kind="timeline", the knobs, attacked_frac,
                  the settings as JSON and their hash
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

The walk is a pipeline of stages with typed hand-offs, each in its own module of `generation`: plan
(`generation.plan`: the splits, the episode counts and onsets), design (`generation.design`: each
split's overload Am, in parallel across `workers` processes, before the split is walked), emit
(`generation.emit`: the frames in time order, the ramp At designed as the walk reaches it) and write
(`generation.write`: the datasets, episodes and attributes). This module orchestrates them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from functools import partial
from typing import Optional, Union

import h5py
import numpy as np

from .engine import FdiaGenerator
from .engine.attacks.stealthy import AM_FAMILY
from .generation import _load_states, _write_graph
from .generation.design import AmDesigner
from .generation.emit import TimelineBuffers, Walk, clean_slice, walk_split
from .generation.plan import place_split, split_bounds
from .generation.write import (
    create_layers,
    finish_timeline,
    placement_attrs,
    search_attrs,
    timeline_attrs,
    write_temporal_layers,
)
from .models.choices import GENERATED_FAMILIES
from .models.config import TimelineSettings
from .models.frames import FrameContext, FrameKnobs, OperatingLimits
from .models.inputs import AdmissibleTargets, GeneratedFamilies
from .models.plan import GeneratorSpec, Placement, Schedule, SplitSpan
from .registry import CACHE_DIR, system_id
from .results.run import config_hash
from .schema import Attr

# the generator makes the multi-snapshot families [WU26]: the ramp At and the overload Am
DEFAULT_FAMILIES = GENERATED_FAMILIES


def _generator(system: Union[int, str], seed: int, s: TimelineSettings) -> FdiaGenerator:
    """The walk's generator on the meter plan of `s`."""
    m = s.meters
    return FdiaGenerator(
        system,
        seed=seed,
        max_load_mw=s.max_load_mw,
        meter_model=m.meter_model,
        vbus_frac=m.vbus_frac,
        pmu_frac=m.pmu_frac,
        flow_frac=m.flow_frac,
    )


def _recorded(
    s: TimelineSettings, g: FdiaGenerator, limits: OperatingLimits, am_runs: bool
) -> dict[str, object]:
    """The knobs a file records as attributes, and the settings themselves as JSON with their hash."""
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
    out.update(search_attrs(s, s.overload if am_runs else None, g.current_mask() is not None))
    out.update({Attr.SETTINGS: json.dumps(record, sort_keys=True), Attr.SETTINGS_HASH: config_hash(record)})
    return out


def _knobs(s: TimelineSettings, limits: OperatingLimits, am_runs: bool) -> FrameKnobs:
    """The settings every scan shares (`FrameKnobs`): the subnetwork, the operating limits every
    false state must satisfy [WU26 eqs. 21-23], the search, and the overload attack's bounds."""
    o = s.overload if am_runs else None
    return FrameKnobs(
        s.search.hops,
        limits,
        s.search.min_tamper,
        s.search.min_budget,
        s.ramp.stealth_scale,
        getattr(o, "load_cap", None),  # the overload attack's cap; none without it
        getattr(o, "n_lines", 1),  # the lines an overload episode drives
        getattr(o, "support_method", "search"),  # how its support is chosen
        area_rule=s.search.area_rule,  # how the attacker's area is chosen [WU26] Sec. III-A
    )


def _walk(w: Walk, schedule: Schedule, bounds: list[tuple[int, int]], designer: AmDesigner) -> Placement:
    """Walk the timeline split by split (the splits are cut first): place the split's episodes, design
    its overload Am episodes, then emit its frames."""
    rec = Placement.empty(list(schedule.families))
    for code, (start, stop) in enumerate(bounds):
        span = SplitSpan(code, start, stop)
        slots = place_split(w.rng, schedule, span, rec)
        slots, designs = designer.design_split(w.ctx.X, w.ctx.knobs, span, slots, rec)
        walk_split(w, schedule, span, slots, designs, rec)
    return rec


def generate_timeline(
    system: Union[int, str],
    states: Optional[Union[str, np.ndarray]] = None,
    settings: Optional[TimelineSettings] = None,
    seed: int = 123,
    out: Optional[str] = None,
    **knobs: object,
) -> str:
    """Walk one attacked timeline over the operating-point pool of `system` and write it as one
    HDF5 file. Returns the path (default: `timeline_ieee{N}.h5` under the cache directory).

    The walk is set by `settings` (`TimelineSettings`, nested by subject) and any flat keyword over
    it, each name the function has always taken (`TimelineSettings.of`):

    attacked_frac    fraction of each split's frames under an attack episode (0.5 = balanced): per
                     split, the whole number of episodes whose frames come closest to it (whole
                     episodes, so it can be off by one episode or two), each placed at an onset
                     uniform among those where it fits in the split, never overlapping or cut, so
                     adjacency and gaps are properties of the draw. An overload Am with no feasible
                     design at its onset moves to another free onset of its split, earlier or later
                     (up to 20 times); a ramp At, designed as the walk reaches it, moves to a later
                     one. The attributes record per split the requested, built, moved and dropped
                     episodes and the attacked fraction reached. A frame whose local power flow has
                     no solution at any halving of its step stays benign and is counted in
                     `fallback_benign`
    families         the families in rotation, each getting about the same share of attacked frames:
                     the ramp At and the overload Am (the default, both). The single-snapshot
                     families of data releases v0.8.3 and earlier (Aq, Ad, As, Ar, Al) are refused
                     here and stay loadable from those files
    ramp_rate, ramp_len   the At ramp's per-frame growth and episode length (`RampSettings`)
    am_len           Am episode length (default ramp_len)
    hops             the attacker's subnetwork: buses within this many branches of the attacked loads
                     (At) or of the target lines (Am); the boundary voltages are held true and only
                     the subnetwork's meters are written
    max_load_mw      a load above this (MW) is never a target: an area equivalent, not a substation
                     (IEEE-145 lumps regions into 4 to 58 GW loads); None disables the cap
                     Every false state also satisfies the operating limits [WU26 eqs. 21-23]: each bus
                     voltage within the case's limits (a bus the true state already holds outside a
                     limit may not be made worse) and every generator's implied output within its P
                     and Q limits widened to the range the pool ran it over; a state outside them is
                     halved; v_lo and v_hi record the widest bus limits of the case
    redundancy       the meter plan, a `MeterSettings`: coverage {vbus_frac, pmu_frac, flow_frac},
                     default 0.6/0.2/0.9, and `meter_model`, what the meters measure:
                     "hybrid", a SCADA voltmeter reads |V| only, the angle is a PMU channel, and every
                     PMU reads the current phasor of each in-service branch at its bus, stored as
                     data/pmu_i with benign/pmu_i_benign and attack/pmu_i_tamper [WU26 eqs. 17-20]
    split            chronological train/val/test fractions (`SplitSettings`), cut before any
                     episode is placed: every episode lies inside one split
    min_tamper       [WU26 eq. 12]: hold each At episode on the support (the buses the false state
                     moves) that tampers the fewest devices over the episode, a change under a
                     meter's noise not counted (default on); off: the region within `hops`. The
                     search's choice per episode is written under episodes/ (min_*)
    min_budget       candidate supports the search solves per episode before it settles on the best
                     found (recorded as not proven)
    am_attack        "overload" (default): Am is the overload attack of [WU26], a metered branch's
                     reported flow driven to its rating over the episode on the fewest-tamper
                     support, its branch, rating and reached flow under episodes/ (am_*); the
                     rating is 1.25 times the branch's peak true flow over the pool (the plan's
                     D15). An `OverloadSettings` asks for the overload attack with other settings:
                     `OverloadSettings(rating_margin=1.5)`, or `rating_source="pglib"` for the
                     PGLib-OPF ratings (IEEE-14, 118 and 300 only: NoLineRatings elsewhere)
    stealth_scale    a multiplier on At's stealth bound: each channel's attack step between
                     snapshots at most this many times the meters' rated accuracy 1
                     by default. Am has no such bound, as in [WU26]: its noise (0.03 pu SCADA, 0.01 pu
                     PMU, D8) only decides which changes its tamper count ignores
    workers          processes the overload Am designs run on (default 1); the file is the same for
                     any number, since every design draws from its own keyed stream

    The file records the settings as JSON (attribute `settings`) with their hash (`settings_hash`).
    """
    s = TimelineSettings.of(settings, **knobs)
    # one column per family: a family named twice adds nothing
    fams = tuple(dict.fromkeys(GeneratedFamilies(s.families).codes))
    am_runs = AM_FAMILY in fams
    g = _generator(system, seed, s)
    X = _load_states(system, states)
    if round(s.attacked_frac * len(X)) > 0:  # a timeline placing no attacked frame needs no target
        # the overload Am is not checked here: its targets are the rated, metered lines of each
        # window, decided per episode (a window with none stays benign)
        AdmissibleTargets(tuple(f for f in fams if f != AM_FAMILY), g.target_counts())
        if am_runs:  # the ratings, before any frame is walked (pglib: NoLineRatings early)
            g.use_line_ratings(s.overload, X)
    limits = g.operating_limits(X)  # the constraints every false state must satisfy [WU26 eqs. 21-23]
    frame_knobs = _knobs(s, limits, am_runs)
    schedule = Schedule.build(list(fams), s.ramp.length, s.ramp.rate, s.am_frames, s.attacked_frac)
    out = out or os.path.join(CACHE_DIR, f"timeline_ieee{system_id(system)}.h5")
    recorded = _recorded(s, g, limits, am_runs)
    spec = GeneratorSpec(
        system,
        seed,
        s.max_load_mw,
        s.meters,
        getattr(g, "_line_ratings", None),
        getattr(g, "_rating_delta", None),
    )
    designer = AmDesigner(g, spec, s.workers if am_runs else 1)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    try:
        ctx = FrameContext(g, X, frame_knobs)
        _write(out, ctx, schedule, designer, s.split.fractions, seed, recorded)
    finally:
        designer.close()
    return out


def _write(
    out: str,
    ctx: FrameContext,
    schedule: Schedule,
    designer: AmDesigner,
    fractions: Sequence[float],
    seed: int,
    recorded: dict[str, object],
) -> None:
    """The file: the graph, the per-frame layers filled by the walk of `ctx` (the generator, the pool
    and the knobs), then everything after it: the columns, the episodes and the attributes (the
    `recorded` knobs and what the placement of the split `fractions` asked for and got)."""
    g, X = ctx.g, ctx.X
    with h5py.File(out, "w") as f:  # the file is open for the whole walk: frames flush in batches
        _write_graph(f, g)
        currents = g.current_mask() is not None
        sink = create_layers(f, len(X), g.C, g.E, currents)
        buf = TimelineBuffers((len(X), g.C, g.E), partial(clean_slice, g, X), sink=sink, currents=currents)
        bounds = split_bounds(len(X), fractions)
        rec = _walk(Walk(ctx, buf), schedule, bounds, designer)
        g.scan_key = None
        finish_timeline(f, buf, bounds)
        f.attrs.update(timeline_attrs(g, seed, buf, recorded))
        f.attrs.update(placement_attrs(rec, bounds, fractions, buf.seq_id))
        write_temporal_layers(f)
