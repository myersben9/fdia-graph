"""The settings every consumer accepts, one model each, checked when built (`models.validation`).

A constructor or function takes keyword arguments, builds its model here and reads the checked
values from it. What a value may be is stated once, on the field.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Annotated, Optional, cast, get_args, get_type_hints

import numpy as np

from ..constants.families import AT, GENERATED_FAMILIES
from ..constants.files import EXPORT_COLUMNS
from ..errors import ConfigError
from .choices import (
    AmAttack,
    AreaRule,
    Format,
    Iso,
    Label,
    Layer,
    LoadRule,
    MeterModel,
    Order,
    RatingSource,
    RecordFormat,
    SolveEngine,
    SolveMethod,
    Split,
    SplitBy,
    SupportMethod,
    Units,
)
from .inputs import GeneratedFamilies
from .validation import (
    AsTuple,
    AtLeast,
    Finite,
    InRange,
    Integer,
    OneOf,
    Parses,
    Positive,
    Validated,
    WhenSet,
)

Count = Annotated[int, Integer(), AtLeast(1)]  # a whole number of rounds, frames, clients
Scale = Annotated[float, Positive(), Finite()]  # a finite positive constant (a Huber c, a rate)


# ---- the dataset --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadOptions(Validated):
    """What `fg.load` and `FdiaGraph` are asked for; checked before any download or file read."""

    split: Annotated[Optional[str], OneOf(Split)] = None
    units: Annotated[str, OneOf(Units)] = "physical"
    order: Annotated[str, OneOf(Order)] = "time"
    format: Annotated[str, OneOf(RecordFormat)] = "torch"


def field_names(fields: object) -> tuple[str, ...]:
    """A list or tuple of field names as a tuple. A lone string is refused rather than split into
    letters, and a mapping or a set is refused rather than read for its keys."""
    if not isinstance(fields, (list, tuple)) or not all(isinstance(f, str) for f in fields):
        raise TypeError(fields)
    return tuple(fields)


@dataclass(frozen=True)
class ExportRequest(Validated):
    """What `export` is asked for: the format ("numpy", "torch", "tf" or "pandas") and the per-record
    fields to read (None: every field). A pandas frame carries every field and refuses `fields`."""

    format: Annotated[str, OneOf(Format)] = "torch"
    fields: Annotated[
        Optional[Sequence[str]], Parses(field_names, "must be a list or tuple of field names")
    ] = None

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            not (self.format == "pandas" and self.fields),
            "a pandas frame carries every field; pass fields with an array format",
        )


@dataclass(frozen=True)
class WindowSpec(Validated):
    """A sliding-window request over a view of T frames: the window length W (1 to T), the stride, how
    a window is labelled and which layer it reads."""

    T: Count
    W: Annotated[int, Integer()]
    stride: Annotated[int, Integer()] = 1
    label: Annotated[str, OneOf(Label)] = "frame"
    layer: Annotated[str, OneOf(Layer)] = "node_x"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            1 <= self.W <= self.T and self.stride >= 1,
            f"need integers 1 <= W <= {self.T} frames and stride >= 1, got W={self.W!r}, stride={self.stride!r}",
        )


# ---- generation --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitSettings(Validated):
    """The chronological train/val/test fractions a timeline is cut into before any episode is placed
    (`generate_timeline(split=...)`): three, each finite in (0, 1), summing to 1."""

    fractions: Annotated[Sequence[float], AsTuple()] = (0.6, 0.2, 0.2)

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield len(self.fractions) == 3, "takes three fractions: train, val and test"
        yield (
            all(math.isfinite(f) and 0.0 < f < 1.0 for f in self.fractions),
            "each fraction must be in (0, 1)",
        )
        yield abs(sum(self.fractions) - 1.0) < 1e-9, "the fractions must sum to 1"


@dataclass(frozen=True)
class OverloadSettings(Validated):
    """The overload attack's settings, as `generate_timeline(am_attack=OverloadSettings(...))` takes them
    (the string "overload" is these defaults). The line rating S_max of each branch is
    `rating_margin` times its peak true apparent flow over the operating pool ("pool", every
    system) or PGLib-OPF's `rate_a` ("pglib", IEEE-14, 118 and 300)."""

    rating_source: Annotated[str, OneOf(RatingSource)] = "pool"
    rating_margin: Annotated[float, Finite(), InRange(1.0, math.inf)] = 1.25  # > 1
    # the load-plausibility cap tau: no load bus the attack moves shows a change beyond tau times its
    # true load [YUA11] (a rule of ours, Yuan's 20% to 50%, the upper end by default)
    load_cap: Annotated[float, Finite(), InRange(0.0, 1.0, hi_closed=True)] = 0.5
    # the lines each episode overloads at once: [WU26]'s case studies always drive two; 1 or 2,
    # since every added line multiplies the target sets an episode tries and the paper uses no more
    n_lines: Annotated[int, Integer(), InRange(1, 2, lo_closed=True, hi_closed=True)] = 2
    # how an episode's support is chosen: the fewest-tamper search, or [WU26]'s row reduction ("rref")
    support_method: Annotated[str, OneOf(SupportMethod)] = "search"
    # with rating_source "delta": each target's S_max is its true flow at the window's end plus this
    # many per unit of the system base (ours, matched to [WU26] Fig. 4's 0.10 to 0.22 pu attacks)
    rating_delta: Annotated[float, Finite(), InRange(0.0, math.inf)] = 0.10


@dataclass(frozen=True)
class TrustSchedule(Validated):
    """[WU26]'s dynamic trusted-PMU configuration (eqs. 26-32) as a constraint on the attack window: the
    PMU at bus `buses[i]` (the generator's bus index) is trusted from snapshot `slots[i]` on. Trust
    accumulates (eqs. 30-31: h^S_t = h^S_t-1 + dh^S_t): at snapshot t every PMU with a slot <= t is
    pinned, and a PMU trusted later can still be tampered with. A trusted PMU pins its own |V| and angle
    rows (eqs. 27, 32: two rows per trusted PMU), so the attack's state deviation
    is zero there (eq. 29); its branch currents stay in the nonsecure set h^S' (eq. 27). One support is
    held for the window; a trusted bus drops out of it from its slot."""

    buses: Annotated[Sequence[int], AsTuple()]
    slots: Annotated[Sequence[int], AsTuple()]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        buses, slots = np.asarray(self.buses), np.asarray(self.slots)
        yield len(buses) == len(slots), "needs one slot per trusted bus"
        yield _indices(buses), "buses must be non-negative bus indices"
        yield len(set(self.buses)) == len(self.buses), "trusts each PMU once"
        yield _indices(slots), "slots must be non-negative snapshot indices"

    def pinned(self, t: int) -> frozenset[int]:
        """The buses whose PMU is trusted at snapshot t (every slot at or before t)."""
        return frozenset(int(b) for b, s in zip(self.buses, self.slots) if s <= t)


@dataclass(frozen=True)
class MeterSettings(Validated):
    """The meter plan a timeline walks, as `generate_timeline(redundancy=...)` takes it: the coverage
    fractions of the voltage meters, the PMUs and the flow meters, and what the meters measure:
    "hybrid" (angles at the PMUs only, PMU branch currents)."""

    vbus_frac: float = 0.6
    pmu_frac: float = 0.2
    flow_frac: float = 0.9
    meter_model: Annotated[str, OneOf(MeterModel)] = "hybrid"

    @property
    def coverage(self) -> dict[str, float]:
        """The coverage fractions as the generator takes them."""
        return {"vbus_frac": self.vbus_frac, "pmu_frac": self.pmu_frac, "flow_frac": self.flow_frac}


@dataclass(frozen=True)
class RampSettings(Validated):
    """The slow ramp At: its per-frame growth `rate`, the episode `length` in frames and `stealth_scale`,
    a multiplier on its stealth bound whose unit is the meters' rated accuracy (the overload Am has
    no such bound)."""

    rate: Scale = 0.002
    length: Annotated[int, Integer(), AtLeast(1)] = 60
    stealth_scale: Scale = 1.0


@dataclass(frozen=True)
class SearchSettings(Validated):
    """The attacker's area and the fewest-tamper search [WU26 eq. 12]: `hops`, the buses within
    this many branches of the attacked loads (At) or target lines (Am); `min_tamper`, hold each At
    episode on the support tampering the fewest devices (off: the whole area); `min_budget`,
    the candidate supports solved per episode before the search settles; `area_rule`, how the
    attacker's area is chosen (`AreaRule`: the buses within `hops`, or [WU26] Sec. III-A's principles)."""

    hops: Count = 2
    min_tamper: bool = True
    min_budget: Count = 256
    area_rule: Annotated[str, OneOf(AreaRule)] = "hops"


# the flat keywords `generate_timeline` takes, by the settings field each one sets (section, field)
_FLAT: dict[str, tuple[str, str]] = {
    "ramp_rate": ("ramp", "rate"),
    "ramp_len": ("ramp", "length"),
    "stealth_scale": ("ramp", "stealth_scale"),
    "hops": ("search", "hops"),
    "min_tamper": ("search", "min_tamper"),
    "min_budget": ("search", "min_budget"),
}
_TOP = ("attacked_frac", "families", "am_len", "max_load_mw", "workers", "split_by", "loads")


@dataclass(frozen=True)
class TimelineSettings(Validated):
    """Everything one timeline walk is built from, nested by subject: `attacked_frac` of each split's
    frames under an episode, the `families` generated (At and Am), the overload Am's length `am_len`
    (None: the ramp's), the load cap `max_load_mw` (None: off) and the processes the Am designs run on
    (`workers`; the result does not depend on it). `split_by` cuts the splits as three blocks in time
    order or as whole weeks (`SplitBy`). `loads` says how the pool's injections divide into load and
    generation (`LoadRule`: "net" for a pool solved by an OPF). `generate_timeline` also takes every field as a flat keyword
    (`TimelineSettings.of`)."""

    attacked_frac: Annotated[float, InRange(0.0, 1.0, lo_closed=True, hi_closed=True)] = 0.5
    families: Annotated[Sequence[str], AsTuple()] = GENERATED_FAMILIES
    ramp: RampSettings = field(default_factory=RampSettings)
    am_len: Annotated[Optional[int], Integer(), AtLeast(1)] = None
    overload: OverloadSettings = field(default_factory=OverloadSettings)
    meters: MeterSettings = field(default_factory=MeterSettings)
    split: SplitSettings = field(default_factory=SplitSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
    max_load_mw: Annotated[Optional[float], Positive()] = 2000.0
    workers: Count = 1
    am_attack: Annotated[str, OneOf(AmAttack)] = "overload"  # [WU26]'s overload attack, the one kind
    split_by: Annotated[str, OneOf(SplitBy), WhenSet(legacy="time")] = "time"
    loads: Annotated[str, OneOf(LoadRule), WhenSet(legacy="scaled")] = "scaled"

    @property
    def am_frames(self) -> int:
        """The Am episode length: `am_len`, or the ramp's length when none is given."""
        return self.ramp.length if self.am_len is None else self.am_len

    def lengths_for(self, families: Sequence[str]) -> dict[str, int]:
        """The episode length in frames of each of `families`: the ramp's for At, `am_frames` for Am."""
        return {f: self.ramp.length if f == AT else self.am_frames for f in families}

    def invariants(self) -> Iterable[tuple[bool, str]]:
        GeneratedFamilies(self.families)  # refuses a family the generator does not make
        return ()

    def as_record(self) -> dict[str, object]:
        """The settings as nested plain values (JSON-ready), what a file and the results store record;
        `workers` is left out, since the timeline does not depend on it."""
        return _record(self, "workers")

    @classmethod
    def of(cls, settings: Optional[TimelineSettings] = None, **knobs: object) -> TimelineSettings:
        """`settings` (default the defaults) with each flat keyword of `generate_timeline` applied over
        it. Three keywords take a model: redundancy (a `MeterSettings`), split (fractions or a
        `SplitSettings`) and am_attack ("overload" or an `OverloadSettings`)."""
        base = settings or cls()
        unknown = sorted(set(knobs) - set(_FLAT) - set(_TOP) - {"redundancy", "split", "am_attack"})
        if unknown:
            raise ConfigError(f"generate_timeline got unknown settings: {', '.join(unknown)}")
        sections = {name: {} for name in ("ramp", "search")}
        for key, (section, name) in _FLAT.items():
            if key in knobs:
                sections[section][name] = knobs[key]
        top = {k: knobs[k] for k in _TOP if k in knobs}
        top.update(_section_knobs(knobs))
        return replace(
            base,
            ramp=replace(base.ramp, **sections["ramp"]),
            search=replace(base.search, **sections["search"]),
            **top,
        )


def _section_knobs(knobs: dict[str, object]) -> dict[str, object]:
    """The settings the flat `redundancy`, `split` and `am_attack` keywords stand for."""
    out: dict[str, object] = {}
    if "redundancy" in knobs:
        out["meters"] = _model(MeterSettings, knobs["redundancy"], "redundancy")
    if "split" in knobs:
        split = knobs["split"]
        out["split"] = (
            split if isinstance(split, SplitSettings) else SplitSettings(cast(Sequence[float], split))
        )
    if "am_attack" in knobs:
        am = knobs["am_attack"]
        if isinstance(am, str):
            out["am_attack"] = am
        else:
            out["overload"] = _model(OverloadSettings, am, "am_attack")
    return out


def _model(model: type, value: object, name: str) -> object:
    """A settings model from its instance or None (the defaults)."""
    if value is None:
        return model()
    if not isinstance(value, model):
        raise ConfigError(f"{name} must be a {model.__name__}")
    return value


def _record(settings: Validated, *left_out: str) -> dict[str, object]:
    """A settings model as nested plain values (JSON-ready), without the fields `left_out` and
    without a `WhenSet` field that holds its legacy value."""
    hints = get_type_hints(type(settings), include_extras=True)
    record: dict[str, object] = {}
    for f in fields(settings):
        value = getattr(settings, f.name)
        unset = any(isinstance(m, WhenSet) and value == m.legacy for m in get_args(hints[f.name]))
        if f.name not in left_out and not unset:
            record[f.name] = _record(value) if isinstance(value, Validated) else _plain(value)
    return record


def _plain(value: object) -> object:
    """Nested settings as JSON-ready values (tuples as lists)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class GeneratorOptions(Validated):
    """The generator's cap on what counts as a single load (MW, None disables it)."""

    max_load_mw: Annotated[Optional[float], Positive()] = 2000.0


@dataclass(frozen=True)
class ShardRun(Validated):
    """How many pool timesteps a shard generation walks; None walks them all."""

    frames: Annotated[Optional[int], Integer(), AtLeast(1)] = None


@dataclass(frozen=True)
class IsoExport(Validated):
    """A directory of one operator's load export; the operator must have a known export format."""

    iso: Annotated[str, OneOf(Iso)]
    directory: str = "."

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            isinstance(self.directory, (str, os.PathLike)),
            f"directory must be a path, got {self.directory!r}",
        )
        yield (
            self.iso in EXPORT_COLUMNS,
            f"no CSV export format is known for {self.iso}; one of {sorted(EXPORT_COLUMNS)}",
        )


@dataclass(frozen=True)
class SolveSettings(Validated):
    """How the pool-building step solves each step, one setting for every test system: the `method`
    (a power flow or an OPF) and the `engine` that runs it. A pool builder's own choice is its default
    (`generate_states`: a power flow in pandapower; `PoolBuilder.from_settings`: an OPF in Ipopt), and
    a builder refuses a choice its test system cannot be solved with."""

    method: Annotated[str, OneOf(SolveMethod)]
    engine: Annotated[str, OneOf(SolveEngine)]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            not (self.method == SolveMethod.POWER_FLOW and self.engine == SolveEngine.IPOPT),
            "a power flow runs in pandapower (engine='pandapower'); the Ipopt engine solves an OPF only",
        )


@dataclass(frozen=True)
class NewYorkSolve(SolveSettings):
    """The solve setting as `PoolBuilder` takes it: an OPF in Ipopt, the builder's own choice and the
    only one it can build the New York pool with. The other choices are refused with the reason."""

    method: Annotated[str, OneOf(SolveMethod)] = "opf"
    engine: Annotated[str, OneOf(SolveEngine)] = "ipopt"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield from super().invariants()
        yield (
            self.method == SolveMethod.OPF,
            "method='power_flow' cannot build the New York pool. PoolBuilder builds by OPF only: the "
            "source model gives hourly generator outputs, at which a power flow leaves voltages far below "
            "the case's limits. A power flow at given outputs is PandapowerSolver(case).power_flow.",
        )
        yield (
            self.engine == SolveEngine.IPOPT,
            "method='opf' with engine='pandapower' cannot build the New York pool. The dispatch limits "
            "each unit's change between frames by its ramp and holds the hydro and external units near "
            "their schedule by a penalty, and pandapower's OPF solves plain least cost only: use "
            "engine='ipopt'. One operating point at plain least cost is PandapowerSolver(case).opf.",
        )


# ---- the New York grid -------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProfileFetch(Validated):
    """A load-profile download: the operator and the optional resampling cadence (minutes)."""

    iso: Annotated[str, OneOf(Iso)]
    resample_min: Annotated[Optional[int], Integer(), AtLeast(1)] = None


def _indices(a: np.ndarray) -> bool:
    """Whether `a` holds only non-negative integers (an empty sequence does)."""
    return a.size == 0 or (np.issubdtype(a.dtype, np.integer) and bool((a >= 0).all()))
