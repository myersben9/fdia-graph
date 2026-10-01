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

`generate_stream` (deprecated) is this writer followed by a read of the file it wrote.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from functools import partial
from typing import Optional, Union

import h5py
import numpy as np

from ._moved import moved
from .engine import FdiaGenerator
from .engine.attacks.episodes import (
    ONSET_DRAWS,
    EpisodeDesignMixin,
    draw_ramp,
    probe_frames,
    ramp_dev,
)
from .engine.records import (  # noqa: F401  AttackDesign, is_feasible re-exported as before
    AM_FAMILY,
    RAMP_FAMILY,
    AttackDesign,
    FrameKnobs,
    is_feasible,
)
from .formulas.attacks import ramp_profile  # noqa: F401  re-exported as before
from .generation import _FrameContext, _load_states, _write_graph, emit, plan, write
from .generation.design import AmDesigner, _GeneratorSpec
from .generation.emit import TimelineBuffers, clean_slice, walk_split
from .generation.plan import (  # noqa: F401  the plan stage's names the walk uses, and its constants
    PLACE_TRIES,
    REDRAWS,
    SPLITS,
    episode_counts,
    largest_remainder,
    place_split,
    split_bounds,
    split_column,
)
from .generation.write import (  # noqa: F401  KIND and write_temporal_layers are public here, as before
    KIND,
    create_layers,
    finish_timeline,
    search_attrs,
    write_temporal_layers,
)
from .models.choices import (  # noqa: F401  re-exported beside the code that reads them
    BENIGN_CODE,
    FAMILY_CODE,
    GENERATED_FAMILIES,
)
from .models.config import TimelineSettings
from .models.frames import OperatingLimits
from .models.inputs import AdmissibleTargets, GeneratedFamilies
from .registry import CACHE_DIR, system_id
from .results.run import config_hash
from .schema import Attr

# the generator makes the multi-snapshot families [WU26]: the ramp At and the overload Am
DEFAULT_FAMILIES = GENERATED_FAMILIES

# what an episode attacks moved to engine.attacks.episodes (the generator's AttackMixin); the old
# private names keep their old signatures for one minor release, forwarding to the new home
_EPISODES = "engine.attacks.episodes"
_MOVED: dict[str, tuple[str, object]] = {
    "_ONSET_DRAWS": (f"{_EPISODES}.ONSET_DRAWS", ONSET_DRAWS),
    "_ramp_dev": (f"{_EPISODES}.ramp_dev", ramp_dev),
    "_target_counts": (f"{_EPISODES}.EpisodeDesignMixin.target_counts", EpisodeDesignMixin.target_counts),
    "_probe_frames": (
        f"{_EPISODES}.probe_frames",
        lambda ctx, t, length: probe_frames(len(ctx.X), t, length),
    ),
    "_draw_ramp": (
        f"{_EPISODES}.draw_ramp",
        lambda ctx, rng, n: tuple(draw_ramp(rng, ctx.g.stealthy_pos, n))[
            :4
        ],  # its old (targets, direction, rise, hold)
    ),
}
# the stages' names moved to the generation package; those kept with their old signature resolve here
# for one minor release, with a warning
_PLAN, _EMIT, _WRITE = "generation.plan", "generation.emit", "generation.write"
_MOVED.update(
    {
        "_Schedule": (f"{_PLAN}._Schedule", plan._Schedule),
        "_Placement": (f"{_PLAN}._Placement", plan._Placement),
        "_split_bounds": (f"{_PLAN}.split_bounds", plan.split_bounds),
        "_largest_remainder": (f"{_PLAN}.largest_remainder", plan.largest_remainder),
        "_episode_counts": (f"{_PLAN}.episode_counts", plan.episode_counts),
        "_uniform_onset": (f"{_PLAN}.uniform_onset", plan.uniform_onset),
        "_place_once": (f"{_PLAN}.place_once", plan.place_once),
        "_place_split": (f"{_PLAN}.place_split", plan.place_split),
        "_split_column": (f"{_PLAN}.split_column", plan.split_column),
        "_give_up": (f"{_PLAN}.give_up", plan.give_up),
        "_TimelineBuffers": (f"{_EMIT}.TimelineBuffers", emit.TimelineBuffers),
        "_Walk": (f"{_EMIT}._Walk", emit._Walk),
        "_emit_benign": (f"{_EMIT}.emit_benign", emit.emit_benign),
        "_benign_run": (f"{_EMIT}.benign_run", emit.benign_run),
        "_ramp_episode": (f"{_EMIT}.ramp_episode", emit.ramp_episode),
        "_relocate": (f"{_EMIT}.relocate_later", emit.relocate_later),
        "_clean_slice": (f"{_EMIT}.clean_slice", emit.clean_slice),
        "_layers": (f"{_EMIT}.layers", emit.layers),
        "_create_layers": (f"{_WRITE}.create_layers", write.create_layers),
        "_ragged": (f"{_WRITE}.ragged", write.ragged),
        "_write_masks": (f"{_WRITE}._write_masks", write._write_masks),
        "_write_episodes": (f"{_WRITE}._write_episodes", write._write_episodes),
        "_write_min_rows": (f"{_WRITE}._write_min_rows", write._write_min_rows),
        "_write_am_rows": (f"{_WRITE}._write_am_rows", write._write_am_rows),
        "_placement_attrs": (f"{_WRITE}.placement_attrs", write.placement_attrs),
        "_timeline_attrs": (f"{_WRITE}._timeline_attrs", write._timeline_attrs),
        "_block_scale": (f"{_WRITE}._block_scale", write._block_scale),
    }
)


def __getattr__(name: str) -> object:
    return moved(__name__, name, _MOVED)


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
        getattr(o, "n_lines", 1),  # the lines an overload episode drives [D17]
        getattr(o, "support_method", "search"),  # how its support is chosen
    )


def _walk(
    w: emit._Walk, schedule: plan._Schedule, bounds: list[tuple[int, int]], designer: AmDesigner
) -> plan._Placement:
    """Walk the timeline split by split (the splits are cut first): place the split's episodes, design
    its overload Am episodes, then emit its frames."""
    rec = plan._Placement.empty(list(schedule.families))
    for split, span in enumerate(bounds):
        slots = place_split(w.rng, schedule, split, span, rec)
        placed = designer.design_split(w.ctx.X, w.ctx.knobs, split, (slots, span), rec)
        walk_split(w, schedule, split, placed, (span, rec))
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
                     default 0.6/0.2/0.9, and `meter_model`, what the meters measure [D10]:
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
                     snapshots at most this many times the meters' rated accuracy [D7]; 1
                     by default. Am has no such bound, as in [WU26]: its noise (0.03 pu SCADA, 0.01 pu
                     PMU, D8) only decides which changes its tamper count ignores [D11]
    workers          processes the overload Am designs run on (default 1); the file is the same for
                     any number, since every design draws from its own keyed stream

    A dict for `am_attack` or `redundancy` still works and warns (DeprecationWarning): dicts go in 0.22.
    The file records the settings as JSON (attribute `settings`) with their hash (`settings_hash`).
    """
    s = TimelineSettings.of(settings, **knobs)
    # one column per family: an alias of a family already named ("ramp" beside "At") adds nothing
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
    schedule = plan._Schedule.build(list(fams), s.ramp.length, s.ramp.rate, s.am_frames, s.attacked_frac)
    out = out or os.path.join(CACHE_DIR, f"timeline_ieee{system_id(system)}.h5")
    recorded = _recorded(s, g, limits, am_runs)
    spec = _GeneratorSpec(system, seed, s.max_load_mw, s.meters, getattr(g, "_line_ratings", None))
    designer = AmDesigner(g, spec, s.workers if am_runs else 1)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    try:
        _write(out, (g, X, frame_knobs), (schedule, designer), (s.split.fractions, seed, recorded))
    finally:
        designer.close()
    return out


def _write(
    out: str,
    run: tuple[FdiaGenerator, np.ndarray, FrameKnobs],
    stages: tuple[plan._Schedule, AmDesigner],
    meta: tuple[Sequence[float], int, dict[str, object]],
) -> None:
    """The file: the graph, the per-frame layers filled by the walk, then everything after it."""
    g, X, knobs = run
    schedule, designer = stages
    fractions, seed, recorded = meta
    with h5py.File(out, "w") as f:  # the file is open for the whole walk: frames flush in batches
        _write_graph(f, g)
        currents = g.current_mask() is not None
        sink = create_layers(f, len(X), g.C, g.E, currents)
        buf = TimelineBuffers((len(X), g.C, g.E), partial(clean_slice, g, X), sink=sink, currents=currents)
        bounds = split_bounds(len(X), fractions)
        rec = _walk(emit._Walk(_FrameContext(g, X, knobs), buf), schedule, bounds, designer)
        g.scan_key = None
        finish_timeline(f, g, buf, (rec, bounds, fractions), (seed, recorded))
        write_temporal_layers(f)
