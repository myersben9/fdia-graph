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
                  seq_id (episode index, -1 benign), timestep, split [T]; temporal_delta, swing [T, N, 2];
                  on hybrid meters the PMU currents pmu_i, pmu_i_m [T, E, 4] (with their benign and tamper twins)
    benign/       node_benign, edge_benign: the same scan with the attack removed and the noise kept
    clean/        node_clean, edge_clean: the noiseless attack-free truth (edge_clean on metered branches)
    graph/        the static topology and per-unit branch physics (as the shards carry them)
    episodes/     onset, length, family [K]; the attacked buses ragged as bus_ptr [K+1], bus_idx
    attack/       the designed magnitude per attacked bus ragged as mag_ptr [T+1], mag_bus, mag, and the
                  tamper masks node_tamper [T, N, 4], edge_tamper [T, E, 2]: the meters the attacker wrote

A New York file is the same layout with what its meters change, declared in its attributes: the flow layers
are [T, E, 4] (both branch ends, `edge_feat`), pmu_i holds magnitude and angle (`current_form` = "polar"),
and temporal_delta and swing are left out. Only it carries data/time, episodes/target_branch, and in graph/
bus_client (with the attribute `client_names`) and edge_rating.

Temporal features are taken against the previous EMITTED frame, so a stealthy ramp reads as a
small per-step change and a spike as an abrupt jump (the signal the dataset is built on).

The one path that generates a timeline, on an IEEE case or on the New York grid, is
`generation.source.TimelineSource.generate`: the splits are cut, then split by split the episodes are
placed (`generation.plan`), designed and the frames walked in time order, then the file is finished.
The steps that depend on the grid are the methods of the source's subclasses:
`generation.ieee.IeeeSource` (design in `generation.design`, the frames in `generation.emit`, the file
in `generation.write`) and `generation.nygrid.source.NewYorkSource` (whose file layout
docs/concepts/new_york_grid.md gives). `generate_timeline` below builds the IEEE source of a system
and generates it.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .generation.ieee import IeeeSource
from .generation.source import OutPath
from .models.choices import GENERATED_FAMILIES
from .models.config import TimelineSettings

# the generator makes the multi-snapshot families [WU26]: the ramp At and the overload Am
DEFAULT_FAMILIES = GENERATED_FAMILIES


def generate_timeline(
    system: Union[int, str],
    states: Optional[Union[str, np.ndarray]] = None,
    settings: Optional[TimelineSettings] = None,
    seed: int = 123,
    out: Optional[OutPath] = None,
    **knobs: object,
) -> str:
    """Walk one attacked timeline over the operating-point pool of the IEEE `system` and write it as
    one HDF5 file. Returns the path (default: `timeline_ieee{N}.h5` under the cache directory). It is
    `IeeeSource(system, states, settings, seed).generate(out)`: the path every timeline takes
    (`TimelineSource.generate`), on this grid.

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
    return IeeeSource(system, states, TimelineSettings.of(settings, **knobs), seed).generate(out)
