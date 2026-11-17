"""The settings every consumer accepts, one model each, checked when built (`models.validation`).

The public signatures do not change: a constructor or function takes its keyword arguments as
before, builds its model here and reads the checked values from it. What a value may be is stated
once, on the field, and nowhere in the consumer.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Annotated, Optional, Union, cast

import numpy as np

from ..constants.families import GENERATED_FAMILIES, NEW_YORK_FAMILIES
from ..constants.files import EXPORT_COLUMNS
from ..constants.newyork import RELEASE_INTERFACE
from ..errors import ConfigError
from .choices import (
    AmAttack,
    AreaRule,
    Format,
    Iso,
    Label,
    Layer,
    MeterModel,
    Order,
    RatingSource,
    RecordFormat,
    Split,
    SplitBy,
    SupportMethod,
    Units,
)
from .validation import (
    AsTuple,
    AtLeast,
    Boolean,
    Finite,
    InRange,
    Integer,
    OneOf,
    Parses,
    Positive,
    Validated,
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
    """What `export` is asked for."""

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
    """A sliding-window request over a view of T frames."""

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
    (the string "overload" is these defaults): the line ratings, S_max
    of each branch `rating_margin` times its peak true apparent flow over the operating pool
    ("pool", every system) or PGLib-OPF's `rate_a` ("pglib", IEEE-14, 118 and 300); the
    load-plausibility cap `load_cap` on every load bus the attack moves and `n_lines`, the
    lines one episode overloads at once."""

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
    PMU at bus `buses[i]` (the generator's bus index) is trusted from snapshot `slots[i]` of the window
    on. Trust accumulates (eqs. 30-31: h^S_t = h^S_t-1 + dh^S_t), so at snapshot t every PMU trusted at
    a slot <= t is pinned and a PMU trusted later can still be tampered with. A trusted PMU pins its
    own |V| and angle rows (eqs. 27, 32: two rows per trusted PMU, "the rank of hS becomes 2n x 2"),
    so the attack's state deviation is zero there (eq. 29); its branch currents stay in the nonsecure
    set h^S' (eq. 27). One support is held for the window, a trusted bus dropping out of it from its
    slot (E13's support per slot never changed an answer in any run and was removed)."""

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
    fractions of the voltage meters, the PMUs and the flow meters, and what the meters measure (the
    plan's D10, D12): "hybrid" (angles at the PMUs only, PMU branch currents)."""

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
    a multiplier on its stealth bound whose unit is the meters' rated accuracy (the
    overload Am has no such bound, D11)."""

    rate: Scale = 0.002
    length: Annotated[int, Integer(), AtLeast(1)] = 60
    stealth_scale: Scale = 1.0


@dataclass(frozen=True)
class SearchSettings(Validated):
    """The attacker's subnetwork and the fewest-tamper search [WU26 eq. 12]: `hops`, the buses within
    this many branches of the attacked loads (At) or target lines (Am); `min_tamper`, hold each At
    episode on the support tampering the fewest devices (off: the whole subnetwork); `min_budget`,
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
_TOP = ("attacked_frac", "families", "am_len", "max_load_mw", "workers")


@dataclass(frozen=True)
class TimelineSettings(Validated):
    """Everything one timeline walk is built from, nested by subject (the plan's S2 item 4):
    `attacked_frac` of each split's frames under an episode, the `families` generated (At and Am),
    the ramp (`RampSettings`), the overload Am's length `am_len` (None: the ramp's) and settings
    (`OverloadSettings`), the meter plan (`MeterSettings`), the train/val/test cut (`SplitSettings`),
    the search (`SearchSettings`), the load cap `max_load_mw` (None: off) and the processes the Am
    designs run on (`workers`, the result does not depend on it). `generate_timeline` also takes every
    field as a flat keyword (`TimelineSettings.of`)."""

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

    @property
    def am_frames(self) -> int:
        """The Am episode length: `am_len`, or the ramp's length when none is given."""
        return self.ramp.length if self.am_len is None else self.am_len

    def as_record(self) -> dict[str, object]:
        """The settings as nested plain values (JSON-ready), what a file and the results store record;
        `workers` is left out, since the timeline does not depend on it."""
        return _record(self, "workers")

    @classmethod
    def of(cls, settings: Optional[TimelineSettings] = None, **knobs: object) -> TimelineSettings:
        """The settings `generate_timeline` walks: `settings` (default the defaults) with each flat
        keyword applied over it, every name `generate_timeline` has always taken: attacked_frac,
        families, ramp_rate, ramp_len, am_len, hops, redundancy (a `MeterSettings`), split (fractions or
        a `SplitSettings`), max_load_mw, min_tamper, min_budget, am_attack ("overload" or an
        `OverloadSettings`), stealth_scale, workers."""
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
    """A settings model as nested plain values (JSON-ready), without the fields `left_out`."""
    record = {name: value for name, value in asdict(settings).items() if name not in left_out}
    return cast("dict[str, object]", _plain(record))


def _plain(value: object) -> object:
    """Nested settings as JSON-ready values (tuples as lists)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class GeneratorOptions(Validated):
    """The generator's cap on what counts as a single load (MW, None disables it) and the meter model
    its plan follows."""

    max_load_mw: Annotated[Optional[float], Positive()] = 2000.0
    meter_model: Annotated[str, OneOf(MeterModel)] = "hybrid"


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


# ---- the New York grid -------------------------------------------------------------------------------
SurveyGrid = Annotated[int, Integer(), AtLeast(0)]  # frames between two surveyed onsets; 0: no survey
FolderPath = Union[str, os.PathLike[str]]


@dataclass(frozen=True)
class NewYorkSettings(Validated):
    """The settings of a timeline on the New York grid (`NewYorkSource`).

    `length` frames per episode, one number for every family or `{family: frames}` (16 five-minute
    scans by default); `attacked_frac`, the share of each split's frames placed under an episode;
    `split`, the train, val and test shares of the year; `split_by`, whole weeks ("week") or three
    blocks in time order ("time"); `workers`, the processes the designs run on (1 runs them in the
    caller's); `block`, the frames written at a time; `verbose` prints progress; `survey` (frames, 0
    off) designs every waiting family's allowed onsets on that grid first and places its episodes only
    among those with an attack, each chunk of designs saved under `survey_cache` when given, so a
    stopped build resumes."""

    length: Union[int, Mapping[str, int]] = 16
    attacked_frac: Annotated[float, InRange(0.0, 1.0, lo_closed=True, hi_closed=True)] = 0.2
    split: SplitSettings = field(default_factory=lambda: SplitSettings((0.7, 0.15, 0.15)))
    split_by: Annotated[str, OneOf(SplitBy)] = "week"
    workers: Count = 1
    block: Count = 2016
    verbose: Annotated[bool, Boolean()] = False
    survey: SurveyGrid = 0
    survey_cache: Optional[FolderPath] = None

    def invariants(self) -> Iterable[tuple[bool, str]]:
        frames = list(self.length.values()) if isinstance(self.length, Mapping) else [self.length]
        yield (
            all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in frames),
            "length must be a whole number of frames, at least 1, or one per family",
        )
        yield isinstance(self.split, SplitSettings), "split must be a SplitSettings"

    def as_record(self) -> dict[str, object]:
        """The settings as nested plain values (JSON-ready), what a file records. Left out is what the
        timeline does not depend on: `workers`, `block`, `verbose` and `survey_cache`."""
        return _record(self, "workers", "block", "verbose", "survey_cache")

    def lengths_for(self, families: Sequence[str]) -> dict[str, int]:
        """The episode length in frames of each of `families`."""
        if isinstance(self.length, Mapping):
            return {f: int(self.length[f]) for f in families}
        return {f: int(self.length) for f in families}


@dataclass(frozen=True)
class NewYorkRelease(Validated):
    """What the New York release's source is built from (`NewYorkSource.production`).

    `families`, each one of Am and At: Am targets the NYISO `interfaces` (at least one), At is the
    slow ramp. `workers` processes design the episodes; `survey` is the onset survey's grid in frames
    (0 off), saved by chunk under `survey_cache` when given. `root` is the folder whose `data/` holds
    the fetched inputs and that the pool's path is relative to. The attacker's sensitivities:
    `hold_external` keeps every bus outside New York at its true injection (N62); `v_margin` (p.u.)
    widens the voltage limits the attacker must respect, on its own copy of the case (N71);
    `gen_v_tol` (p.u.) holds each generator bus's false voltage that close to its schedule (N75);
    `load_cap` replaces the cap on a pretended load change (None keeps the family's own, 0.1, N28);
    `band` is the share of a unit's upper limit in its basepoint band (None keeps 0.03, N33)."""

    families: Annotated[Sequence[str], AsTuple()] = NEW_YORK_FAMILIES
    interfaces: Annotated[Sequence[str], AsTuple()] = (RELEASE_INTERFACE,)
    workers: Count = 12
    survey: SurveyGrid = 12
    survey_cache: Optional[FolderPath] = None
    root: FolderPath = "."
    hold_external: Annotated[bool, Boolean()] = False
    v_margin: Annotated[float, Finite(), AtLeast(0)] = 0.0
    gen_v_tol: Annotated[Optional[float], Finite(), AtLeast(0)] = None
    load_cap: Annotated[Optional[float], Finite(), Positive()] = None
    band: Annotated[Optional[float], Finite(), Positive()] = None

    def invariants(self) -> Iterable[tuple[bool, str]]:
        unknown = [str(f) for f in self.families if f not in NEW_YORK_FAMILIES]
        yield (
            not unknown,
            f"the release builds the families {', '.join(NEW_YORK_FAMILIES)}; it does not build "
            f"{', '.join(unknown)}",
        )
        yield (
            "Am" not in self.families or len(self.interfaces) > 0,
            "Am attacks a NYISO interface: name at least one in `interfaces`",
        )
        yield isinstance(self.root, (str, os.PathLike)), f"root must be a folder path, got {self.root!r}"

    def as_record(self) -> dict[str, object]:
        """The knobs as nested plain values (JSON-ready), what a file records. Left out is what the
        timeline does not depend on (`workers`, `survey_cache`, `root`) and `survey`, which the settings
        of the timeline record."""
        return _record(self, "workers", "survey", "survey_cache", "root")


@dataclass(frozen=True)
class ProfileFetch(Validated):
    """A load-profile download: the operator and the optional resampling cadence (minutes)."""

    iso: Annotated[str, OneOf(Iso)]
    resample_min: Annotated[Optional[int], Integer(), AtLeast(1)] = None


def _indices(a: np.ndarray) -> bool:
    """Whether `a` holds only non-negative integers (an empty sequence does)."""
    return a.size == 0 or (np.issubdtype(a.dtype, np.integer) and bool((a >= 0).all()))
